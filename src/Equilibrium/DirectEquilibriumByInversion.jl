"""
Contour-tracing flux surface reconstruction for the direct (EFIT) equilibrium path.

Traces level sets of ψ(R,Z) using marching squares (`Contour.jl`), resamples each
closed curve to a uniform geometric-angle grid, constructs an `InverseRunInput`,
and delegates to the existing `equilibrium_solver(::InverseRunInput)`.

This completely eliminates the field-line ODE, `direct_refine`, and the denominator
singularity. Robustness near the separatrix comes from Contour.jl naturally handling
the topological change as ψ → 1: the plasma interior curve remains closed and
well-traced, while open curves (touching the domain boundary) are discarded.

Select via `eq_type = "efit_by_inversion"` in `gpec.toml`.
"""

import Contour as Ctr

"""
    _is_closed_curve(curve)

Returns true if the Contour.jl curve is closed (first ≈ last vertex).
"""
function _is_closed_curve(curve)
    verts = Ctr.vertices(curve)
    length(verts) < 3 && return false
    dx = abs(verts[1][1] - verts[end][1])
    dz = abs(verts[1][2] - verts[end][2])
    return dx < 1e-8 && dz < 1e-8
end

"""
    _point_in_polygon_verts(verts, r_test, z_test)

Ray-casting point-in-polygon test operating directly on the `vertices` vector
returned by `Ctr.vertices`. Avoids allocating the separate R/Z vectors that
`Ctr.coordinates` would produce.
"""
function _point_in_polygon_verts(verts, r_test::Float64, z_test::Float64)::Bool
    n = length(verts)
    inside = false
    j = n
    @inbounds for i in 1:n
        ri, zi = verts[i]
        rj, zj = verts[j]
        if (zi > z_test) != (zj > z_test)
            r_cross = (rj - ri) * (z_test - zi) / (zj - zi) + ri
            if r_test < r_cross
                inside = !inside
            end
        end
        j = i
    end
    return inside
end

"""
    select_plasma_contour(curve_list, ro, zo)

From the list of Contour.jl curves at a given ψ level, return the closed curve
whose interior contains the magnetic axis (ro, zo). This is the plasma flux surface.

Returns `nothing` if no qualifying closed curve is found (psihigh has exceeded
the separatrix, or the target ψ is outside the EFIT domain).

Uses `Ctr.vertices` directly to avoid allocating coordinate vectors for rejected curves.
"""
function select_plasma_contour(curve_list, ro::Float64, zo::Float64)
    for curve in curve_list
        _is_closed_curve(curve) || continue
        verts = Ctr.vertices(curve)
        _point_in_polygon_verts(verts, ro, zo) && return curve
    end
    return nothing
end

"""
    resample_contour_to_theta_grid!(R_out, Z_out, curve, ro, zo, theta_grid)

Resample a Contour.jl curve onto a uniform geometric-angle grid, writing results
directly into the pre-allocated `R_out` and `Z_out` vectors (views into R_table/Z_table).

## Steps:
1. Extract vertices directly from `Ctr.vertices` (avoids `coordinates` allocation)
2. Compute geometric angles: `η_k = mod(atan(Z_k−zo, R_k−ro), 2π)` in a single pass
3. Sort by η (CCW ordering around the magnetic axis)
4. Append first point at η + 2π to close the periodic spline
5. Build cubic splines R(η), Z(η) with LinearBinary search
6. Sample at each `theta_grid` point using shared hints (O(1) searches)
"""
function resample_contour_to_theta_grid!(
    R_out::AbstractVector{Float64}, Z_out::AbstractVector{Float64},
    curve, ro::Float64, zo::Float64, theta_grid::AbstractVector{Float64}
)
    verts = Ctr.vertices(curve)
    nv = length(verts)

    # Drop duplicate closing vertex (Contour.jl closed curves repeat first==last)
    if nv > 1
        r1, z1 = verts[1]
        rn, zn = verts[nv]
        if abs(r1 - rn) < 1e-8 && abs(z1 - zn) < 1e-8
            nv -= 1
        end
    end

    # Compute angles and extract R/Z in a single pass (no broadcast temporaries)
    η_buf = Vector{Float64}(undef, nv)
    R_buf = Vector{Float64}(undef, nv)
    Z_buf = Vector{Float64}(undef, nv)
    @inbounds for i in 1:nv
        r, z = verts[i]
        η_buf[i] = mod(atan(z - zo, r - ro), 2π)
        R_buf[i] = r
        Z_buf[i] = z
    end

    idx = sortperm(η_buf)

    # Build extended (nv+1) sorted arrays with periodic wrap appended
    n_ext = nv + 1
    η_ext = Vector{Float64}(undef, n_ext)
    R_ext = Vector{Float64}(undef, n_ext)
    Z_ext = Vector{Float64}(undef, n_ext)
    @inbounds for i in 1:nv
        k = idx[i]
        η_ext[i] = η_buf[k]
        R_ext[i] = R_buf[k]
        Z_ext[i] = Z_buf[k]
    end
    η_ext[n_ext] = η_ext[1] + 2π
    R_ext[n_ext] = R_ext[1]
    Z_ext[n_ext] = Z_ext[1]

    R_spl = cubic_interp(η_ext, R_ext; bc=CubicFit(), extrap=WrapExtrap(), search=LinearBinary())
    Z_spl = cubic_interp(η_ext, Z_ext; bc=CubicFit(), extrap=WrapExtrap(), search=LinearBinary())

    # theta_grid is in turns [0, 1]; sample the geometric-angle spline at 2π*θ radians.
    # Monotonically increasing → shared hint gives O(1) lookups per pass.
    hint = Ref(1)
    @inbounds for k in eachindex(theta_grid, R_out)
        R_out[k] = R_spl(2π * theta_grid[k]; hint=hint)
    end
    hint[] = 1
    @inbounds for k in eachindex(theta_grid, Z_out)
        Z_out[k] = Z_spl(2π * theta_grid[k]; hint=hint)
    end
end

"""
    equilibrium_solver_by_inversion(raw_profile; refine=4, psilow_contour_threshold=5e-3)

Driver for contour-tracing equilibrium reconstruction.

## Procedure:
1. Find magnetic axis via `direct_position!`
2. Evaluate ψ(R,Z) bicubic spline on a `refine×` fine Cartesian grid (parallelized)
3. For each target ψ_norm surface: trace the closed level-set curve with Contour.jl,
   resample to uniform geometric-angle (R, Z) grid (parallelized)
4. Build an `InverseRunInput` from the R(ψ,θ), Z(ψ,θ) tables
5. Delegate to `equilibrium_solver(::InverseRunInput)` for SFL coordinate construction

## Keyword arguments:
- `refine`: grid refinement factor relative to EFIT resolution (default 4)
- `psilow_contour_threshold`: below this psifac, use circular near-axis fallback
  if Contour.jl cannot resolve the surface (only relevant for psilow < ~1e-4)

Select via `eq_type = "efit_by_inversion"` in `gpec.toml`.
"""
function equilibrium_solver_by_inversion(
    raw_profile::DirectRunInput;
    refine::Int=4,
    psilow_contour_threshold::Float64=5e-3
)
    equil_params = raw_profile.config
    psio = raw_profile.psio
    mpsi = equil_params.mpsi
    mtheta = equil_params.mtheta
    psilow = equil_params.psilow
    psihigh = equil_params.psihigh

    if psihigh >= 1 - 1e-6
        @warn "efit_by_inversion: psihigh = $(psihigh) very close to 1 — separatrix may be reached."
    end

    # Build target psi_norm grid (same ldp scheme as direct solver)
    psi_nodes = [psilow + (psihigh - psilow) * sin((ipsi / mpsi) * (π / 2))^2 for ipsi in 0:mpsi]

    # Find magnetic axis and separatrix
    ro, zo, _rs1, rs2 = direct_position!(raw_profile)

    # Clip fine grid to separatrix bounding box.
    # The EFIT nodal ψ values are free (already stored in the spline); a single coarse
    # Contour.jl call at ψ ≈ psihigh gives the plasma bounding box with no extra spline work.
    # A 5% margin on each side ensures small arc curvature outside the box is covered.
    nw = length(raw_profile.psi_in_xs)
    nh = length(raw_profile.psi_in_ys)
    ψ_coarse = raw_profile.psi_in.nodal_derivs.partials[1, :, :]
    ψ_bbox_target = psio * max(1.0 - psihigh, 1e-3)   # outermost needed surface, ≥ 0.001·psio
    cl_bbox = Ctr.contour(raw_profile.psi_in_xs, raw_profile.psi_in_ys, ψ_coarse, ψ_bbox_target)
    bbox_curve = select_plasma_contour(Ctr.lines(cl_bbox), ro, zo)
    if bbox_curve !== nothing
        bv = Ctr.vertices(bbox_curve)
        r_bbox_lo, r_bbox_hi = extrema(v[1] for v in bv)
        z_bbox_lo, z_bbox_hi = extrema(v[2] for v in bv)
        r_margin = (r_bbox_hi - r_bbox_lo) * 0.05
        z_margin = (z_bbox_hi - z_bbox_lo) * 0.05
        r_lo = max(raw_profile.rmin, r_bbox_lo - r_margin)
        r_hi = min(raw_profile.rmax, r_bbox_hi + r_margin)
        z_lo = max(raw_profile.zmin, z_bbox_lo - z_margin)
        z_hi = min(raw_profile.zmax, z_bbox_hi + z_margin)
    else
        r_lo, r_hi = raw_profile.rmin, raw_profile.rmax
        z_lo, z_hi = raw_profile.zmin, raw_profile.zmax
    end

    # Build fine Cartesian grid for Contour.jl, clipped to the plasma bounding box.
    # Maintain the same grid spacing as the full-domain grid (nr/nz scale with clipped extent).
    # Ranges are passed directly so Contour.jl uses its optimised range path.
    nr_fine = max(4, round(Int, refine * nw * (r_hi - r_lo) / (raw_profile.rmax - raw_profile.rmin)))
    nz_fine = max(4, round(Int, refine * nh * (z_hi - z_lo) / (raw_profile.zmax - raw_profile.zmin)))
    r_fine = range(r_lo, r_hi; length=nr_fine)
    z_fine = range(z_lo, z_hi; length=nz_fine)

    @info "efit_by_inversion: Evaluating ψ on $(nr_fine)×$(nz_fine) fine grid ($(refine)× refinement, clipped to plasma bbox)"

    # Parallel grid evaluation: each column (fixed z) is independent.
    # One hint pair per thread (allocated once), reset at the start of each column.
    # Resetting R-hint is required (R starts over at rmin each column); resetting Z-hint
    # lets it settle on the first lookup (Z is fixed per column, so O(log n) once then O(1)).
    ψ_fine = Matrix{Float64}(undef, nr_fine, nz_fine)
    thread_hints = [(Ref(1), Ref(1)) for _ in 1:Threads.nthreads()]
    Threads.@threads for j in 1:nz_fine
        z = z_fine[j]
        h = thread_hints[Threads.threadid()]
        h[1][] = 1
        h[2][] = 1
        @inbounds for i in 1:nr_fine
            ψ_fine[i, j] = raw_profile.psi_in((r_fine[i], z); hint=h)
        end
    end

    # Theta grid for InverseRunInput: fractional turns in [0, 1].
    # InverseEquilibrium uses theta ∈ [0,1] and formula cos(2π*(theta + deta)),
    # so rz_in_ys must be in turns, not radians.
    theta_grid = collect(range(0.0, 1.0; length=mtheta + 1))

    # Build R(ψ, θ) and Z(ψ, θ) tables
    R_table = Matrix{Float64}(undef, mpsi + 1, mtheta + 1)
    Z_table = Matrix{Float64}(undef, mpsi + 1, mtheta + 1)

    # Parallel contour tracing: each surface is independent.
    # Ctr.contour has no global state (confirmed from source inspection) → thread-safe.
    # surface_status: 0=success, 1=fallback, -1=error (checked serially after the loop).
    surface_status = Vector{Int8}(undef, mpsi + 1)

    Threads.@threads for ipsi in 1:(mpsi+1)
        psifac = psi_nodes[ipsi]
        ψ_target = psio * (1.0 - psifac)

        # Trace level set at ψ_target with Contour.jl (marching squares).
        # r_fine and z_fine are passed as ranges — Contour.jl has an optimised range path.
        cl = Ctr.contour(r_fine, z_fine, ψ_fine, ψ_target)
        curve = select_plasma_contour(Ctr.lines(cl), ro, zo)

        if curve === nothing
            # Near-axis fallback: circular approximation if Contour.jl can't resolve the
            # surface (only expected for psifac < grid spacing², i.e., very small psilow).
            # Accuracy is limited in this regime; psilow > 1e-3 is recommended.
            if psifac < psilow_contour_threshold
                rfac = sqrt(psifac) * (rs2 - ro)
                @inbounds for j in 1:(mtheta+1)
                    θ = theta_grid[j]  # turns ∈ [0, 1]
                    R_table[ipsi, j] = ro + rfac * cos(2π * θ)
                    Z_table[ipsi, j] = zo + rfac * sin(2π * θ)
                end
                surface_status[ipsi] = 1
            else
                surface_status[ipsi] = -1
            end
            continue
        end

        resample_contour_to_theta_grid!(
            @view(R_table[ipsi, :]), @view(Z_table[ipsi, :]),
            curve, ro, zo, theta_grid
        )
        surface_status[ipsi] = 0
    end

    # Serial error check and summary (after all threads complete)
    for ipsi in 1:(mpsi+1)
        if surface_status[ipsi] == -1
            psifac = psi_nodes[ipsi]
            error("efit_by_inversion: No closed flux surface found at psifac = $(@sprintf("%.4f", psifac)) " *
                "— psihigh likely exceeds the separatrix or is outside the EFIT domain.")
        end
    end
    n_contour_success = count(==(0), surface_status)
    n_contour_fallback = count(==(1), surface_status)
    @info "efit_by_inversion: $n_contour_success surfaces traced by Contour.jl, $n_contour_fallback by near-axis fallback"

    # Build InverseRunInput — same type consumed by equilibrium_solver(::InverseRunInput)
    rz_in_xs = psi_nodes
    rz_in_ys = theta_grid
    itp_opts2d = (search=LinearBinary(), bc=(CubicFit(), PeriodicBC()), extrap=(ExtendExtrap(), WrapExtrap()))

    rz_in_R = cubic_interp((rz_in_xs, rz_in_ys), R_table; itp_opts2d...)
    rz_in_Z = cubic_interp((rz_in_xs, rz_in_ys), Z_table; itp_opts2d...)

    inv_input = InverseRunInput(raw_profile.config, raw_profile.sq_in,
        rz_in_xs, rz_in_ys, rz_in_R, rz_in_Z, ro, zo, psio)

    return equilibrium_solver(inv_input)
end
