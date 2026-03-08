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
    _point_in_polygon(R_pts, Z_pts, r_test, z_test)

Ray-casting point-in-polygon test. Returns true if (r_test, z_test) is inside
the polygon defined by (R_pts, Z_pts).
"""
function _point_in_polygon(R_pts::AbstractVector, Z_pts::AbstractVector, r_test::Float64, z_test::Float64)::Bool
    n = length(R_pts)
    inside = false
    j = n
    for i in 1:n
        zi, zj = Z_pts[i], Z_pts[j]
        ri, rj = R_pts[i], R_pts[j]
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
"""
function select_plasma_contour(curve_list, ro::Float64, zo::Float64)
    for curve in curve_list
        _is_closed_curve(curve) || continue
        R_pts, Z_pts = Ctr.coordinates(curve)
        _point_in_polygon(R_pts, Z_pts, ro, zo) && return curve
    end
    return nothing
end

"""
    resample_contour_to_theta_grid(curve, ro, zo, theta_grid)

Resample a Contour.jl curve onto a uniform geometric-angle grid.

## Steps:
1. Extract (R, Z) vertices from the curve
2. Compute geometric angles: `η_k = mod(atan(Z_k−zo, R_k−ro), 2π)`
3. Sort by η (CCW ordering around the magnetic axis)
4. Append first point at η + 2π to close the periodic spline
5. Build cubic splines R(η), Z(η)
6. Sample at each `theta_grid` point

## Returns:
`(R_resampled, Z_resampled)` — vectors of length `length(theta_grid)`.
"""
function resample_contour_to_theta_grid(
    curve, ro::Float64, zo::Float64, theta_grid::AbstractVector{Float64}
)::Tuple{Vector{Float64},Vector{Float64}}

    R_pts, Z_pts = Ctr.coordinates(curve)

    # Contour.jl closed curves repeat first==last vertex — drop the duplicate to
    # avoid a zero-length interval in the cubic spline (which produces NaN).
    if length(R_pts) > 1 && abs(R_pts[end] - R_pts[1]) < 1e-8 && abs(Z_pts[end] - Z_pts[1]) < 1e-8
        R_pts = R_pts[1:end-1]
        Z_pts = Z_pts[1:end-1]
    end

    # Geometric angle for each contour point, mapped to [0, 2π)
    η = mod.(atan.(Z_pts .- zo, R_pts .- ro), 2π)

    # Sort by angle for monotone CCW ordering
    idx = sortperm(η)
    η_s = η[idx]
    R_s = R_pts[idx]
    Z_s = Z_pts[idx]

    # Extend by one period for the periodic cubic spline
    η_ext = vcat(η_s, η_s[1] + 2π)
    R_ext = vcat(R_s, R_s[1])
    Z_ext = vcat(Z_s, Z_s[1])

    R_spl = cubic_interp(η_ext, R_ext; bc=CubicFit(), extrap=WrapExtrap())
    Z_spl = cubic_interp(η_ext, Z_ext; bc=CubicFit(), extrap=WrapExtrap())

    # theta_grid is in turns [0, 1]; sample the geometric-angle spline at 2π*θ radians
    R_out = [R_spl(2π * θ) for θ in theta_grid]
    Z_out = [Z_spl(2π * θ) for θ in theta_grid]

    return R_out, Z_out
end

"""
    equilibrium_solver_by_inversion(raw_profile; refine=4, psilow_contour_threshold=5e-3)

Driver for contour-tracing equilibrium reconstruction.

## Procedure:
1. Find magnetic axis via `direct_position!`
2. Evaluate ψ(R,Z) bicubic spline on a `refine×` fine Cartesian grid
3. For each target ψ_norm surface: trace the closed level-set curve with Contour.jl,
   resample to uniform geometric-angle (R, Z) grid
4. Build an `InverseRunInput` from the R(ψ,θ), Z(ψ,θ) tables
5. Delegate to `equilibrium_solver(::InverseRunInput)` for SFL coordinate construction

## Keyword arguments:
- `refine`: grid refinement factor relative to EFIT resolution (default 4)
- `psilow_contour_threshold`: below this psifac, use circular near-axis fallback

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

    # Build fine Cartesian grid for Contour.jl
    nw = length(raw_profile.psi_in_xs)
    nh = length(raw_profile.psi_in_ys)
    nr_fine = refine * nw
    nz_fine = refine * nh
    r_fine = range(raw_profile.rmin, raw_profile.rmax; length=nr_fine)
    z_fine = range(raw_profile.zmin, raw_profile.zmax; length=nz_fine)

    @info "efit_by_inversion: Evaluating ψ on $(nr_fine)×$(nz_fine) fine grid ($(refine)× refinement)"
    ψ_fine = [raw_profile.psi_in((r, z)) for r in r_fine, z in z_fine]

    # Theta grid for InverseRunInput: fractional turns in [0, 1].
    # InverseEquilibrium uses theta ∈ [0,1] and formula cos(2π*(theta + deta)),
    # so rz_in_ys must be in turns, not radians.
    theta_grid = collect(range(0.0, 1.0; length=mtheta + 1))

    # Build R(ψ, θ) and Z(ψ, θ) tables
    R_table = Matrix{Float64}(undef, mpsi + 1, mtheta + 1)
    Z_table = Matrix{Float64}(undef, mpsi + 1, mtheta + 1)

    n_contour_success = 0
    n_contour_fallback = 0

    for ipsi in 1:(mpsi+1)
        psifac = psi_nodes[ipsi]
        ψ_target = psio * (1.0 - psifac)

        # Trace level set at ψ_target with Contour.jl (marching squares)
        cl = Ctr.contour(collect(r_fine), collect(z_fine), ψ_fine, ψ_target)
        curve = select_plasma_contour(Ctr.lines(cl), ro, zo)

        if curve === nothing
            # Near-axis fallback: circular approximation if Contour.jl can't resolve the surface.
            # This only occurs for extremely small psifac where the contour is smaller than the
            # fine-grid spacing. The accuracy is limited; psilow should be kept > 1e-3 for
            # best near-axis q accuracy with this method.
            if psifac < psilow_contour_threshold
                rfac = sqrt(psifac) * (rs2 - ro)
                for j in 1:(mtheta+1)
                    θ = theta_grid[j]  # turns ∈ [0, 1]
                    R_table[ipsi, j] = ro + rfac * cos(2π * θ)
                    Z_table[ipsi, j] = zo + rfac * sin(2π * θ)
                end
                n_contour_fallback += 1
                continue
            end
            error("efit_by_inversion: No closed flux surface found at psifac = $(@sprintf("%.4f", psifac)) " *
                "— psihigh likely exceeds the separatrix or is outside the EFIT domain.")
        end

        R_row, Z_row = resample_contour_to_theta_grid(curve, ro, zo, theta_grid)
        R_table[ipsi, :] .= R_row
        Z_table[ipsi, :] .= Z_row
        n_contour_success += 1
    end

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
