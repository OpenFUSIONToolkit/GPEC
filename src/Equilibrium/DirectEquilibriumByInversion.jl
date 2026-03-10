"""
Contour-tracing flux surface reconstruction for the direct (EFIT) equilibrium path.

Traces level sets of ψ(R,Z) using marching squares (`Contour.jl`), resamples each
closed curve to a uniform geometric-angle grid, constructs an `InverseRunInput`,
and delegates to the existing `equilibrium_solver(::InverseRunInput)`.

This completely eliminates the field-line ODE, `direct_refine`, and the denominator
singularity. Robustness near the separatrix comes from Contour.jl naturally handling
the topological change as ψ → 1: the plasma interior curve remains closed and
well-traced, while open curves (touching the domain boundary) are discarded.

Near x-points the outermost flux surfaces have sharply curved arms that can exceed
the resolution of a uniform fine grid. A sinh-stretched Z coordinate concentrates
grid points near the x-point(s).
Contour.jl accepts non-uniform coordinate vectors natively and returns physical (R,Z).

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

    R_spl = cubic_interp(η_ext, R_ext; bc=PeriodicBC(), extrap=WrapExtrap(), search=LinearBinary())
    Z_spl = cubic_interp(η_ext, Z_ext; bc=PeriodicBC(), extrap=WrapExtrap(), search=LinearBinary())

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
    classify_topology(raw_profile, psio; xpt_threshold=0.05) → Symbol

Classify the plasma topology as `:limited`, `:sn_lower`, `:sn_upper`, or `:double_null`
by evaluating ψ on the (R, Z) grid and finding the minimum in each Z half-domain.

An x-point is detected when the minimum ψ in a half-domain falls below
`xpt_threshold * psio` (i.e., within 5% of the separatrix by default).

Uses the public `psi_in` interpolant interface with `LinearBinary` hints: within each
Z row, the R hint carries forward (R is monotone), and the Z hint advances naturally
across rows. This avoids coupling to internal spline layout while remaining efficient.
"""
function classify_topology(raw_profile::DirectRunInput, psio::Float64;
                           xpt_threshold::Float64=0.05)
    R_grid = raw_profile.psi_in_xs
    Z_grid = raw_profile.psi_in_ys
    nZ = length(Z_grid)
    mid_Z = nZ ÷ 2
    hint = (Ref(1), Ref(1))

    min_ψ_lower = Inf
    for jZ in 1:mid_Z
        hint[1][] = 1  # reset R hint at start of each row; Z hint advances naturally
        z = Z_grid[jZ]
        for R in R_grid
            ψ = raw_profile.psi_in((R, z); hint=hint)
            min_ψ_lower = min(min_ψ_lower, ψ)
        end
    end

    min_ψ_upper = Inf
    for jZ in (mid_Z+1):nZ
        hint[1][] = 1
        z = Z_grid[jZ]
        for R in R_grid
            ψ = raw_profile.psi_in((R, z); hint=hint)
            min_ψ_upper = min(min_ψ_upper, ψ)
        end
    end

    has_lower = min_ψ_lower / psio < xpt_threshold
    has_upper = min_ψ_upper / psio < xpt_threshold
    if has_lower && has_upper
        return :double_null
    elseif has_lower
        return :sn_lower
    elseif has_upper
        return :sn_upper
    else
        return :limited
    end
end

"""
    make_stretched_z_grid(z_lo, z_hi, nz, topology, β_z) → Vector{Float64}

Build a Z coordinate vector of length `nz` spanning `[z_lo, z_hi]` with sinh-stretching
that concentrates grid points near the x-point end(s) of the domain.

For `:sn_lower`, points are packed toward `z_lo` (lower x-point). For `:sn_upper`,
toward `z_hi`. For `:double_null`, both ends are packed symmetrically. For `:limited`
or `β_z ≤ 0`, a uniform grid is returned.

The transformation is Z(v) = z_lo + Δz·sinh(β·v)/sinh(β) for v ∈ [0,1], which maps
the uniform parameter `v` to a non-uniform Z that clusters near the small-v end.
At β=2.0 the first cell is ~3.8× smaller than the last; at β=3.0 it is ~10× smaller.
Contour.jl accepts non-uniform coordinate vectors natively, so no back-conversion is
needed — the returned Z values are physical coordinates.
"""
function make_stretched_z_grid(z_lo::Float64, z_hi::Float64, nz::Int,
                               topology::Symbol, β_z::Float64)
    if topology == :limited || β_z ≤ 0.0
        return collect(range(z_lo, z_hi; length=nz))
    end
    Δz = z_hi - z_lo
    sinh_β = sinh(β_z)
    z_grid = Vector{Float64}(undef, nz)
    if topology == :sn_lower
        # Pack toward z_lo (lower x-point)
        @inbounds for k in 1:nz
            v = (k - 1) / (nz - 1)
            z_grid[k] = z_lo + Δz * sinh(β_z * v) / sinh_β
        end
    elseif topology == :sn_upper
        # Pack toward z_hi (upper x-point); mirror of sn_lower
        @inbounds for k in 1:nz
            v = (k - 1) / (nz - 1)
            z_grid[k] = z_hi - Δz * sinh(β_z * (1.0 - v)) / sinh_β
        end
    else  # :double_null — pack both ends symmetrically
        z_half = 0.5 * Δz
        @inbounds for k in 1:nz
            v = (k - 1) / (nz - 1)
            if v <= 0.5
                z_grid[k] = z_lo + z_half * sinh(β_z * 2v) / sinh_β
            else
                z_grid[k] = z_hi - z_half * sinh(β_z * 2(1.0 - v)) / sinh_β
            end
        end
    end
    return z_grid
end

"""
    equilibrium_solver_by_inversion(raw_profile; refine=4, β_z=2.0, psilow_contour_threshold=5e-3)

Driver for contour-tracing equilibrium reconstruction.

## Procedure:
1. Find magnetic axis via `direct_position!`
2. Detect plasma topology (limited / SN-lower / SN-upper / double-null) from EFIT nodal ψ
3. Evaluate ψ(R,Z) bicubic spline on a `refine×` fine Cartesian grid (parallelized).
   The Z grid is sinh-stretched toward x-point(s) to concentrate marching-squares resolution
   where outermost flux surface arms are most curved.
4. For each target ψ_norm surface: trace the closed level-set curve with Contour.jl,
   resample to uniform geometric-angle (R, Z) grid (parallelized)
5. Build an `InverseRunInput` from the R(ψ,θ), Z(ψ,θ) tables
6. Delegate to `equilibrium_solver(::InverseRunInput)` for SFL coordinate construction
7. Validate round-trip accuracy; warn if edge error exceeds 2e-3

## Keyword arguments:
- `refine`: grid refinement factor relative to EFIT resolution (default 4)
- `β_z`: sinh-stretching strength for the Z grid toward x-point(s) (default 2.0).
  `β_z=0` disables stretching (uniform grid). Higher values give stronger compression
  near x-points at the cost of coarser sampling away from them.
- `psilow_contour_threshold`: below this psifac, use circular near-axis fallback
  if Contour.jl cannot resolve the surface (only relevant for psilow < ~1e-4)

Select via `eq_type = "efit_by_inversion"` in `gpec.toml`.
"""
function equilibrium_solver_by_inversion(
    raw_profile::DirectRunInput;
    refine::Int=4,
    β_z::Float64=2.0,
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

    # Detect plasma topology for sinh-stretching direction
    topology = classify_topology(raw_profile, psio)
    @info "efit_by_inversion: topology = $topology, β_z = $β_z"

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
    # The R grid is a uniform range (Contour.jl fast path); the Z grid is sinh-stretched
    # toward x-point(s) to concentrate marching-squares resolution where outermost surface
    # arms are most curved. Contour.jl accepts non-uniform Z vectors natively.
    nr_fine = max(4, round(Int, refine * nw * (r_hi - r_lo) / (raw_profile.rmax - raw_profile.rmin)))
    nz_fine = max(4, round(Int, refine * nh * (z_hi - z_lo) / (raw_profile.zmax - raw_profile.zmin)))
    r_fine = range(r_lo, r_hi; length=nr_fine)
    z_grid = make_stretched_z_grid(z_lo, z_hi, nz_fine, topology, β_z)

    @info "efit_by_inversion: Evaluating ψ on $(nr_fine)×$(nz_fine) fine grid ($(refine)× refinement, clipped to plasma bbox)"

    # Parallel grid evaluation: each column (fixed z) is independent.
    # One hint pair per thread (allocated once), reset at the start of each column.
    # Resetting R-hint is required (R starts over at rmin each column); resetting Z-hint
    # lets it settle on the first lookup (Z is fixed per column, so O(log n) once then O(1)).
    # Z-hint cannot be shared across threads for a non-uniform z_grid — each thread resets it.
    ψ_fine = Matrix{Float64}(undef, nr_fine, nz_fine)
    thread_hints = [(Ref(1), Ref(1)) for _ in 1:Threads.nthreads()]
    Threads.@threads for j in 1:nz_fine
        z = z_grid[j]
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
        # r_fine is a range (Contour.jl fast path); z_grid is a non-uniform vector.
        cl = Ctr.contour(r_fine, z_grid, ψ_fine, ψ_target)
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

    pe = equilibrium_solver(inv_input)

    # Round-trip validation: (ψ,θ) → (R,Z) → ψ_spline − ψ_target.
    # Checks 4 angles at the outermost surface and at 75% of the radial grid.
    # A large error indicates that Contour.jl resolution was insufficient near the
    # x-point and the traced surface positions are inaccurate.
    max_rt_err = 0.0
    for ψ_check in (pe.rzphi_xs[end], pe.rzphi_xs[max(1, length(pe.rzphi_xs) * 3 ÷ 4)])
        for θ_check in (0.0, 0.25, 0.5, 0.75)
            r2  = pe.rzphi_rsquared((ψ_check, θ_check))
            off = pe.rzphi_offset((ψ_check, θ_check))
            rfac = sqrt(max(r2, 0.0))
            η    = 2π * (θ_check + off)
            R    = pe.ro + rfac * cos(η)
            Z    = pe.zo + rfac * sin(η)
            ψ_rt = 1.0 - raw_profile.psi_in((R, Z)) / psio
            max_rt_err = max(max_rt_err, abs(ψ_rt - ψ_check))
        end
    end
    if max_rt_err > 2e-3
        @warn "efit_by_inversion: round-trip error at edge = $(@sprintf("%.2e", max_rt_err)) > 2e-3; accuracy near psihigh may be limited. Consider reducing psihigh or increasing β_z."
    else
        @info "efit_by_inversion: round-trip error at edge = $(@sprintf("%.2e", max_rt_err))"
    end

    return pe
end
