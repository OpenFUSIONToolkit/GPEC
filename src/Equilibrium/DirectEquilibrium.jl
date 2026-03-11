"""
    DirectBField

Internal mutable struct to hold B-field components and their derivatives at a point.
It is used as a temporary workspace to avoid allocations in tight loops.
"""
@kwdef mutable struct DirectBField
    psi::Float64 = 0.0
    psir::Float64 = 0.0   # d(psi)/dr
    psiz::Float64 = 0.0   # d(psi)/dz
    psirz::Float64 = 0.0  # d2(psi)/drdz
    psirr::Float64 = 0.0  # d2(psi)/drdr
    psizz::Float64 = 0.0  # d2(psi)/dzdz
    f::Float64 = 0.0      # F = R*Bt
    f1::Float64 = 0.0     # dF/d(psi_norm)
    p::Float64 = 0.0      # mu0*Pressure
    p1::Float64 = 0.0     # dP/d(psi_norm)
    br::Float64 = 0.0     # Br
    bz::Float64 = 0.0     # Bz
    brr::Float64 = 0.0    # d(Br)/dr
    brz::Float64 = 0.0    # d(Br)/dz
    bzr::Float64 = 0.0    # d(Bz)/dr
    bzz::Float64 = 0.0    # d(Bz)/dz
end

"""
    FieldLineDerivParams

A struct to hold constant parameters for the ODE integration, making them
easily accessible within the derivative function `direct_fieldline_der!`.
"""
struct FieldLineDerivParams{I2D<:FastInterpolations.CubicInterpolantND,S<:FastInterpolations.CubicSeriesInterpolant,D}
    ro::Float64
    zo::Float64
    psi_in::I2D
    sq_in::S
    sq_in_deriv::D
    psio::Float64
    power_bp::Int
    power_b::Int
    power_r::Int
    bfield::DirectBField
end

"""
    direct_get_bfield!(bf_out, r, z, psi_in, sq_in, sq_in_deriv, psio; derivs=0)

Calculates the magnetic field and its derivatives at a given (R,Z) point.
The results are stored in-place in the `bf_out` object. This is equivalent
to the `direct_get_bfield` subroutine in the Fortran code, adapted for
the Julia spline implementation.

## Arguments:

  - `bf_out`: A mutable `DirectBField` struct to store the results
  - `r`: R-coordinate to evaluate at
  - `z`: Z-coordinate to evaluate at
  - `psi_in`: 2D cubic interpolant for poloidal flux `ψ(R,Z)`
  - `sq_in`: 1D cubic spline for profiles `F(ψ_norm)` and `P(ψ_norm)`
  - `sq_in_deriv`: Pre-computed derivative view of sq_in
  - `psio`: total toroidal flux
  - `derivs`: An integer specifying number of derivatives to compute (0, 1, or 2)
"""
@with_pool pool function direct_get_bfield!(
    bf_out::DirectBField,
    r::Float64,
    z::Float64,
    psi_in::FastInterpolations.CubicInterpolantND,
    sq_in::FastInterpolations.CubicSeriesInterpolant,
    sq_in_deriv,
    psio::Float64;
    derivs::Int=0
)
    # Evaluate 2D interpolant for psi(r,z) and its derivatives
    if derivs == 0
        bf_out.psi = psi_in((r, z))
    elseif derivs == 1
        bf_out.psi = psi_in((r, z))
        bf_out.psir = psi_in((r, z); deriv=Val((1, 0)))
        bf_out.psiz = psi_in((r, z); deriv=Val((0, 1)))
    else # derivs >= 2
        bf_out.psi = psi_in((r, z))
        bf_out.psir = psi_in((r, z); deriv=Val((1, 0)))
        bf_out.psiz = psi_in((r, z); deriv=Val((0, 1)))
        bf_out.psirr = psi_in((r, z); deriv=Val((2, 0)))
        bf_out.psirz = psi_in((r, z); deriv=Val((1, 1)))
        bf_out.psizz = psi_in((r, z); deriv=Val((0, 2)))
    end

    # Evaluate magnetic fields from equilibrium profiles
    psi_norm = (psio > 1e-12) ? (1.0 - bf_out.psi / psio) : 0.0
    psi_norm = clamp(psi_norm, 0.0, 1.0)

    f_sq = acquire!(pool, eltype(sq_in.y), n_series(sq_in))
    f1_sq = acquire!(pool, eltype(sq_in_deriv.parent.y), n_series(sq_in_deriv.parent))
    sq_in(f_sq, psi_norm)
    sq_in_deriv(f1_sq, psi_norm)
    bf_out.f = f_sq[1]  # F = R*Bt
    bf_out.f1 = f1_sq[1] # dF/dψ
    bf_out.p = f_sq[2]  # μ0*Pressure
    bf_out.p1 = f1_sq[2] # dP/dψ

    (derivs == 0) && return

    # Evaluate B-field derivative components
    bf_out.br = bf_out.psiz / r # Br = (1/R) * ∂ψ/∂Z
    bf_out.bz = -bf_out.psir / r # Bz = -(1/R) * ∂ψ/∂R

    (derivs == 1) && return

    # Evaluate more derivatives of B-field components
    bf_out.brr = (bf_out.psirz - bf_out.br) / r
    bf_out.brz = bf_out.psizz / r
    bf_out.bzr = -(bf_out.psirr + bf_out.bz) / r
    bf_out.bzz = -bf_out.psirz / r
end

"""
    direct_position!(raw_profile)

Finds the key geometric locations of the equilibrium: the magnetic axis (O-point)
and the inboard/outboard separatrix crossings on the midplane. It also updates the
spline representing the poloidal flux `ψ(R,Z)` based on the new magnetic axis location.
This function performs the same overall function as the Fortran `direct_position`
subroutine with better iteration control and error handling. We have also added a
helper function for separatrix finding.

## Arguments:

  - `raw_profile`: A `DirectRunInput` object containing splines and parameters.

## Returns:

  - `ro`: R-coordinate of the magnetic axis [m].
  - `zo`: Z-coordinate of the magnetic axis [m].
  - `rs1`: R-coordinate of the inboard separatrix crossing [m].
  - `rs2`: R-coordinate of the outboard separatrix crossing [m].
  - `psi_in_new` : returns psi_in renormalized by * psio/psi(ro,zo)
"""
function direct_position!(raw_profile::DirectRunInput)

    bfield = DirectBField()
    max_iterations = 200
    sq_in_deriv = deriv1(raw_profile.sq_in)

    # For an axis initial guess, we find zero crossing of Bz on the midplane
    # Note the Fortran had this wrapped in a if ro == 0 block, which we omit here
    # because I don't think it's ever used. If needed, it can be re-added.
    r = (raw_profile.rmax + raw_profile.rmin) / 2.0
    z = (raw_profile.zmax + raw_profile.zmin) / 2.0
    dr = (raw_profile.rmax - raw_profile.rmin) / 20.0

    for _ in 1:max_iterations
        direct_get_bfield!(bfield, r, z, raw_profile.psi_in, raw_profile.sq_in, sq_in_deriv, raw_profile.psio; derivs=1)
        if bfield.bz >= 0.0
            break
        end
        r += dr
    end

    # If we never exited early, the loop failed to find bz = 0
    !(bfield.bz >= 0) && error("Took too many iterations to get bz=0.")

    # Now, use Newton iteration to find the O-point (magnetic axis) where Br=0 and Bz=0
    dr, dz = 0.0, 0.0
    for _ in 1:max_iterations
        direct_get_bfield!(bfield, r, z, raw_profile.psi_in, raw_profile.sq_in, sq_in_deriv, raw_profile.psio; derivs=2)
        det = bfield.brr * bfield.bzz - bfield.brz * bfield.bzr
        if abs(det) < 1e-20
            error("Jacobian matrix is singular near ($r, $z).")
        end
        # Δx = -J⁻¹ F
        dr = (bfield.brz * bfield.bz - bfield.bzz * bfield.br) / det
        dz = (bfield.bzr * bfield.br - bfield.brr * bfield.bz) / det
        r += dr
        z += dz
        if abs(dr) <= 1e-12 * abs(r) && abs(dz) <= 1e-12 * abs(r)
            @info "Magnetic axis found at R = $(@sprintf("%.3f", r)), Z = $(@sprintf("%.3f", z))"
            break
        end
    end

    if !(abs(dr) <= 1e-12 * abs(r) && abs(dz) <= 1e-12 * abs(r))
        error("Failed to find magnetic axis after $max_iterations iterations.")
    end

    ro = r
    zo = z

    # Renormalize psi based on the value at the magnetic axis
    direct_get_bfield!(bfield, ro, zo, raw_profile.psi_in, raw_profile.sq_in, sq_in_deriv, raw_profile.psio; derivs=0)
    x_coords = raw_profile.psi_in_xs
    y_coords = raw_profile.psi_in_ys
    # Access nodal values from psi_in interpolant: partials[1,:,:] = function values
    new_psi_fs = raw_profile.psi_in.nodal_derivs.partials[1, :, :] .* raw_profile.psio / bfield.psi
    # Because DirectRunInput is a mutable struct, we can update the spline here
    raw_profile.psi_in = cubic_interp((x_coords, y_coords), new_psi_fs; search=LinearBinary(),
        bc=CubicFit(), extrap=ExtendExtrap())

    # ψ = 0 at the separatrix (after renormalization), and ψ changes sign between the
    # magnetic axis (ψ > 0) and the region outside the plasma (ψ < 0), so Brent is
    # globally convergent within the bracket (start_r, end_r) and needs no restarts.
    function find_separatrix_crossing(start_r, end_r, label)
        r_sol = find_zero(
            r -> (direct_get_bfield!(bfield, r, zo, raw_profile.psi_in, raw_profile.sq_in, sq_in_deriv, raw_profile.psio; derivs=0); bfield.psi),
            (start_r, end_r), Roots.Brent()
        )
        @info "$label separatrix found at R = $(@sprintf("%.3f", r_sol))"
        return r_sol
    end

    # Find inboard (rs1) and outboard (rs2) separatrix positions
    rs1 = find_separatrix_crossing(ro, raw_profile.rmin, "Inboard")
    rs2 = find_separatrix_crossing(ro, raw_profile.rmax, "Outboard")

    return ro, zo, rs1, rs2
end

"""
    direct_fieldline_int(psifac, raw_profile, ro, zo, rs2)

Performs the field-line integration for a single flux surface. This is a Julia adaptation
of the Fortran `direct_fl_int` subroutine. Note that the array `y_out` is now indexed
from 1:5 rather than 0:4 as in Fortran.

## Arguments:

  - `psifac`: normalized psi value for the surface (ψ_norm).
  - `raw_profile`: `DirectRunInput` object containing splines and parameters.
  - `ro`, `zo`: Coordinates of the magnetic axis [m].
  - `rs2`: R-coordinate of the outboard separatrix [m].

## Returns:

    - `y_out`: A matrix containing the integrated quantities vs. the geometric angle `η`.
        - `y_out[:, 1]`: η (geometric poloidal angle)
        - `y_out[:, 2]`: ∫(dl/Bp)
        - `y_out[:, 3]`: rfac (radial distance from magnetic axis)
        - `y_out[:, 4]`: ∫(dl/(R²Bp))
        - `y_out[:, 5]`: ∫(jac*dl/Bp)

  - `bfield`: A `DirectBField` object with values at the integration start point.
"""
function direct_fieldline_int(psifac::Float64, raw_profile::DirectRunInput, ro::Float64, zo::Float64, rs2::Float64)::Tuple{Matrix{Float64},DirectBField}

    # Find the starting point on the flux surface (outboard midplane)
    psi0_guess = raw_profile.psio * (1.0 - psifac)
    r = ro + sqrt(psifac) * (rs2 - ro)
    z = zo
    bfield = DirectBField()
    sq_in_deriv = deriv1(raw_profile.sq_in)

    # Refine starting R: find r where ψ(r, zo) = ψ₀. The df closure reads bfield.psir
    # which is populated by the preceding f call — Newton guarantees f is evaluated first.
    r = find_zero(
        (r -> (direct_get_bfield!(bfield, r, z, raw_profile.psi_in, raw_profile.sq_in, sq_in_deriv, raw_profile.psio; derivs=1); bfield.psi - psi0_guess),
            _ -> bfield.psir),
        r, Roots.Newton()
    )

    direct_get_bfield!(bfield, r, z, raw_profile.psi_in, raw_profile.sq_in, sq_in_deriv, raw_profile.psio; derivs=2)
    psi0 = bfield.psi

    # Set up and solve the ODE for fieldline following
    # Initial condition
    u0 = zeros(Float64, 4)
    u0[2] = sqrt((r - ro)^2 + (z - zo)^2)

    bfield = DirectBField()
    equil_config = raw_profile.config
    params = FieldLineDerivParams(ro, zo, raw_profile.psi_in, raw_profile.sq_in, sq_in_deriv, raw_profile.psio,
        equil_config.power_bp, equil_config.power_b, equil_config.power_r, bfield)

    # Use a callback to refine the solution at each step to stay on the flux surface
    function refine_affect!(integrator)
        integrator.u[2] = direct_refine(integrator.u[2], integrator.t, psi0, params)
    end

    # We save the solution at each step before refinement (before=true, after=false) to match Fortran
    callback = DiscreteCallback((u, t, i) -> true, refine_affect!; save_positions=(true, false))

    prob = ODEProblem{true}(direct_fieldline_der!, u0, (0.0, 2π), params)
    sol = solve(prob, BS5(); callback=callback, reltol=1e-6, abstol=1e-8, dt=2π / 200, adaptive=true, dense=false)

    sol_matrix = reduce(hcat, sol.u::Vector{Vector{Float64}})'
    return hcat(sol.t::Vector{Float64}, sol_matrix), bfield
end

"""
    direct_fieldline_der!(dy, y, params, eta)

The derivative function for the field-line integration ODE. This is passed to
the `DifferentialEquations.jl` solver. This is a Julia adaptation of the Fortran
`direct_fl_der` subroutine, with an added safeguard against division by zero.

## Arguments:

  - `dy`: The derivative vector (output, modified in-place).
  - `y`: The state vector `[∫(dl/Bp), rfac, ∫(dl/(R²Bp)), ∫(jac*dl/Bp)]`.
  - `params`: A `FieldLineDerivParams` struct with all necessary parameters.
  - `eta`: The independent variable (geometric angle `η`).
"""
function direct_fieldline_der!(dy, y, params::FieldLineDerivParams, eta)

    cos_eta, sin_eta = cos(eta), sin(eta)
    r = params.ro + y[2] * cos_eta
    z = params.zo + y[2] * sin_eta
    direct_get_bfield!(params.bfield, r, z, params.psi_in, params.sq_in, params.sq_in_deriv, params.psio; derivs=1)

    bp = sqrt(params.bfield.br^2 + params.bfield.bz^2)
    bt = params.bfield.f / r
    b = sqrt(bp^2 + bt^2)
    jac = (bp^params.power_bp) * (b^params.power_b) / (r^params.power_r)

    # Denominator for d(l_pol)/d(eta) = rfac |B_pol|/denominator
    denominator = params.bfield.bz * cos_eta - params.bfield.br * sin_eta
    if abs(denominator) < 1e-14
        fill!(dy, 1e20) # Return large derivatives for solver to handle stiffness
        @warn "Denominator in direct_fieldline_der! near zero at eta=$eta."
        return
    end

    # Compute derivatives
    # d/dη [∫(dl/Bp)] = 1/|B_P| dl/d(eta) = rfac/denominator
    dy[1] = y[2] / denominator
    # d(rfac)/d(eta) = rfac/denom *( Br cos(eta) + Bz sin(eta) )
    dy[2] = dy[1] * (params.bfield.br * cos_eta + params.bfield.bz * sin_eta)
    # d/dη [∫(dl/(R²Bp))]
    dy[3] = dy[1] / (r^2)
    # d/dη [∫(jac*dl/Bp)]
    dy[4] = dy[1] * jac
end

"""
    direct_refine(rfac, eta, psi0, params)

Refines the radial distance `rfac` at a given angle `eta` to ensure the
point lies exactly on the target flux surface `psi0`.

## Arguments:

  - `rfac`: The current guess for the radial distance from the magnetic axis.
  - `eta`: The geometric poloidal angle.
  - `psi0`: The target `ψ` value for the flux surface.
  - `params`: A `FieldLineDerivParams` struct.

## Returns:

  - The refined `rfac` value.
"""
function direct_refine(rfac::Float64, eta::Float64, psi0::Float64, params::FieldLineDerivParams)::Float64
    cos_eta, sin_eta = cos(eta), sin(eta)

    function f(rfac_inner)
        r = params.ro + rfac_inner * cos_eta
        z = params.zo + rfac_inner * sin_eta
        direct_get_bfield!(params.bfield, r, z, params.psi_in, params.sq_in,
            params.sq_in_deriv, params.psio; derivs=0)
        return params.bfield.psi - psi0
    end

    function fp(rfac_inner)
        r = params.ro + rfac_inner * cos_eta
        z = params.zo + rfac_inner * sin_eta
        direct_get_bfield!(params.bfield, r, z, params.psi_in, params.sq_in,
            params.sq_in_deriv, params.psio; derivs=1)
        return params.bfield.psir * cos_eta + params.bfield.psiz * sin_eta
    end

    return find_zero((f, fp), rfac, Roots.Newton();
        atol=1e-12*abs(psi0), rtol=1e-12, maxevals=50)
end

"""
    direct_xpoint_find(raw_profile, ro, zo, rs1)

Detect the magnetic x-point (if any) by searching for a null of the poloidal field (Br=0, Bz=0)
near the plasma boundary. Tries the lower null first (below the magnetic axis), then upper null.
Returns `(r_xpoint, z_xpoint, is_diverted, psi_hess_rr, psi_hess_zz, psi_hess_rz)`.

`psi_hess_rr/zz/rz` are ∂²ψ_norm/∂R², ∂²ψ_norm/∂Z², ∂²ψ_norm/∂R∂Z at the X-point, used to
build the asymptotic far-edge geometry in `direct_build_xpoint_asymptotic!`. Returns zeros for
the Hessian when no X-point is found.

After `direct_position!` renormalizes ψ, the separatrix satisfies ψ≈0 and Bp=0.
An x-point is detected when the Newton iteration converges to a point with |ψ| < ψ_threshold.
"""
function direct_xpoint_find(raw_profile::DirectRunInput, ro::Float64, zo::Float64, rs1::Float64)
    bfield = DirectBField()
    sq_in_deriv = deriv1(raw_profile.sq_in)
    max_iterations = 100
    psi_threshold = 1e-3  # convergence criterion: |ψ| < threshold at x-point candidate

    # X-points in diverted plasmas lie on the inboard side, between rs1 and the magnetic axis.
    # A guess of rs1 + 0.35*(ro-rs1) places the starting R in this inboard region; using
    # (rs1+rs2)/2 overshoots to the plasma center and Newton misses the basin of attraction.
    r_xguess = rs1 + 0.35 * (ro - rs1)

    # Try lower null first, then upper null
    for (label, z_guess) in (("lower", raw_profile.zmin + 0.15 * (zo - raw_profile.zmin)),
                              ("upper", raw_profile.zmax - 0.15 * (raw_profile.zmax - zo)))
        r, z = r_xguess, z_guess
        dr, dz = 0.0, 0.0
        converged = false
        for _ in 1:max_iterations
            direct_get_bfield!(bfield, r, z, raw_profile.psi_in, raw_profile.sq_in, sq_in_deriv, raw_profile.psio; derivs=2)
            det = bfield.brr * bfield.bzz - bfield.brz * bfield.bzr
            abs(det) < 1e-20 && break
            dr = (bfield.brz * bfield.bz - bfield.bzz * bfield.br) / det
            dz = (bfield.bzr * bfield.br - bfield.brr * bfield.bz) / det
            r += dr
            z += dz
            if abs(dr) <= 1e-8 * abs(r) && abs(dz) <= 1e-8 * abs(r)
                converged = true
                break
            end
        end
        # Accept candidate if Newton converged and ψ is near zero (≈ separatrix)
        if converged
            direct_get_bfield!(bfield, r, z, raw_profile.psi_in, raw_profile.sq_in, sq_in_deriv, raw_profile.psio; derivs=2)
            if abs(bfield.psi / raw_profile.psio) < psi_threshold
                @printf("   X-point (%s null) found at R = %.5f, Z = %.5f\n", label, r, z)
                return r, z, true, bfield.psirr, bfield.psizz, bfield.psirz
            end
        end
    end

    return NaN, NaN, false, 0.0, 0.0, 0.0
end

"""
    direct_build_xpoint_asymptotic!(profiles, psi_nodes, rzphi_nodes, ro, zo, theta_nodes,
                                    r_xpoint, z_xpoint; edge_layer_width=1e-4)

Extend the rzphi coordinate mapping into the far edge (psin > psihigh) using X-point
asymptotic geometry. Returns extended `(psi_nodes, rzphi_nodes)` arrays covering the
region from psihigh up to the last packed rational surface.

**Physics**: Near the X-point with F'≈0 and P'≈0, the Grad-Shafranov equation reduces to
Δ*ψ = 0. The local solution is a quadratic saddle surface whose level curves are hyperbolas
in the principal eigencoordinates of the Hessian H = ∂²ψ/∂(R,Z)² at the X-point.

In the Hessian approximation, the flux surface at ψ_n > psihigh scales from the X-point
as: R(ψ_n, θ) = R_X + s·(R_high(θ) - R_X), and similarly for Z, where
s = √((1 - ψ_n)/(1 - psihigh)). This preserves the shape of the flux surface while
shrinking it toward the X-point as ψ_n → 1. The coordinate Jacobian J_coord is
approximately constant in ψ in this limit (from the Hessian scaling analysis).

Using this geometry with the EFIT F (held constant by ExtendExtrap), the GS residual is
near zero (since F'≈0, P'≈0 near the separatrix), satisfying the requirement that the
EL equation be evaluated on an approximate GS equilibrium.

Only called for diverted plasmas (when `profiles.q_spline_iota_inverse !== nothing`).
"""
function direct_build_xpoint_asymptotic!(profiles::ProfileSplines,
    psi_nodes::AbstractVector{Float64},
    rzphi_nodes::AbstractArray{Float64,3},
    ro::Float64,
    zo::Float64,
    theta_nodes,
    r_xpoint::Float64,
    z_xpoint::Float64;
    edge_layer_width::Float64=1e-4)

    isnothing(profiles.q_spline_iota_inverse) && return psi_nodes, rzphi_nodes

    iota_inner = profiles.q_spline_iota_inverse.inner
    psihigh = psi_nodes[end]
    ipsi_high = length(psi_nodes)
    mtheta_pts = size(rzphi_nodes, 2)  # mtheta+1 points (0..1 inclusive)

    # Recover R(psihigh, θ) and Z(psihigh, θ) from the rzphi node data
    R_high = zeros(Float64, mtheta_pts)
    Z_high = zeros(Float64, mtheta_pts)
    for itheta in 1:mtheta_pts
        theta = theta_nodes[itheta]
        rfac = sqrt(max(0.0, rzphi_nodes[ipsi_high, itheta, 1]))
        offset = rzphi_nodes[ipsi_high, itheta, 2]
        eta = 2π * (theta + offset)
        R_high[itheta] = ro + rfac * cos(eta)
        Z_high[itheta] = zo + rfac * sin(eta)
    end

    # Scan for n=1 rational surfaces in the far edge to build a grid with ~10 pts per window.
    # Same algorithm as sing_find! so resolution concentrates near the densely-packed surfaces.
    iota_at_psihigh = iota_inner(psihigh)
    edge_surfaces = Float64[]
    hint_scan = Ref(1)
    m_start = trunc(Int, 1.0 / max(iota_at_psihigh, 1e-10)) + 1
    m_end   = min(m_start + 500, 100_000)
    for m in m_start:m_end
        iota_target = 1.0 / m
        iota_target < 1e-10 && break
        iota_target >= iota_at_psihigh && continue
        try
            psi_surf = find_zero(
                psi -> iota_inner(psi; hint=hint_scan) - iota_target,
                (psihigh, 1.0 - 1e-8), Roots.Brent(); xatol=1e-8
            )
            push!(edge_surfaces, psi_surf)
            if length(edge_surfaces) >= 2 && edge_surfaces[end] - edge_surfaces[end-1] < edge_layer_width
                break
            end
        catch
        end
    end

    # Build far-edge ψ grid: 10 interior points per window between consecutive rational surfaces.
    psi_far = Float64[psihigh]
    boundaries = vcat([psihigh], edge_surfaces)
    for i in 1:(length(boundaries)-1)
        win_pts = range(boundaries[i], boundaries[i+1]; length=12)  # 12 pts = 10 interior + 2 endpoints
        append!(psi_far, win_pts[2:end])
    end
    last_surf = isempty(edge_surfaces) ? psihigh : edge_surfaces[end]

    # Extend the dense per-window grid one full rational window beyond psilim
    # (psilim = last_surf + edge_layer_width) so the spline covers the complete
    # window that psilim sits in.  For n=1: m_dense_end = trunc(q(psilim)) + 1,
    # i.e. round((n*q_psilim + 1)/n) from the user's formula.
    psilim_local = min(last_surf, 1.0 - 1e-8)
    iota_at_psilim = iota_inner(psilim_local)
    m_dense_end = trunc(Int, 1.0 / max(iota_at_psilim, 1e-10)) + 1
    iota_dense_end = 1.0 / m_dense_end
    psi_dense_end = last_surf
    if iota_dense_end > 1e-10 && iota_dense_end < iota_inner(psihigh)
        try
            psi_dense_end = find_zero(
                psi -> iota_inner(psi) - iota_dense_end,
                (psilim_local, 1.0 - 1e-8), Roots.Brent(); xatol=1e-8
            )
        catch
        end
    end
    if psi_dense_end > last_surf + 1e-10
        dense_ext = range(last_surf, psi_dense_end; length=12)
        append!(psi_far, dense_ext[2:end])
    end

    # Sparse tail from the end of the dense grid to near the separatrix
    tail_start = max(psi_dense_end, last_surf)
    last_pts = range(tail_start, 1.0 - 1e-6; length=12)
    append!(psi_far, last_pts[2:end])
    unique!(sort!(psi_far))

    npsi_far = length(psi_far)

    # Build far-edge rzphi_nodes using X-point scaling.
    # For each (ψ_n, θ): scale position from X-point by s = √((1-ψ_n)/(1-psihigh)).
    # J_coord stays at the psihigh value (constant in ψ in the Hessian approximation).
    # nu stays at the psihigh value (F ≈ const near separatrix → no toroidal correction change).
    rzphi_far = zeros(Float64, npsi_far, mtheta_pts, 4)
    for (i, psi_n) in enumerate(psi_far)
        s = sqrt(max(0.0, (1.0 - psi_n) / (1.0 - psihigh)))
        for itheta in 1:mtheta_pts
            theta = theta_nodes[itheta]

            # Scaled position from X-point
            R_n = r_xpoint + s * (R_high[itheta] - r_xpoint)
            Z_n = z_xpoint + s * (Z_high[itheta] - z_xpoint)

            # rfac² and offset in JPEC pseudo-circular coordinates
            rfac_sq_n = (R_n - ro)^2 + (Z_n - zo)^2
            eta_n_raw = atan(Z_n - zo, R_n - ro)

            # Adjust eta branch to be consistent with psihigh reference (avoids atan2 discontinuities)
            offset_high = rzphi_nodes[ipsi_high, itheta, 2]
            eta_high = 2π * (theta + offset_high)
            n_wrap = round(Int, (eta_high - eta_n_raw) / (2π))
            eta_n = eta_n_raw + n_wrap * 2π
            offset_n = eta_n / (2π) - theta

            rzphi_far[i, itheta, 1] = rfac_sq_n
            rzphi_far[i, itheta, 2] = offset_n
            rzphi_far[i, itheta, 3] = rzphi_nodes[ipsi_high, itheta, 3]  # nu: constant
            rzphi_far[i, itheta, 4] = rzphi_nodes[ipsi_high, itheta, 4]  # J_coord: constant
        end
    end

    # Extend arrays: skip psi_far[1] = psihigh which is already in psi_nodes
    psi_extended = vcat(psi_nodes, psi_far[2:end])
    rzphi_extended = cat(rzphi_nodes, rzphi_far[2:end, :, :]; dims=1)

    @printf("   X-point asymptotic geometry built over psin ∈ [%.4f, %.6f] (%d far-edge nodes, %d edge rational surfaces)\n",
        psihigh, psi_far[end], npsi_far - 1, length(edge_surfaces))

    return psi_extended, rzphi_extended
end

"""
    equilibrium_solver(raw_profile)

The main driver for the direct equilibrium reconstruction. It orchestrates the entire
process from finding the magnetic axis to integrating along field lines and
constructing the final coordinate and physics quantity splines. This performs the same
overall function as the Fortran `direct_run` subroutine, with better checks for numerical
robustness.

## Arguments:

  - `raw_profile`: A `DirectRunInput` object containing the initial splines (`psi_in`, `sq_in`)
    and run parameters (`equil_input`).

## Returns:

  - A `PlasmaEquilibrium` object containing the final, processed equilibrium data,
    including the profile spline (`sq`), the coordinate mapping spline (`rzphi`), and
    the physics quantity spline (`eqfun`).
"""
@with_pool pool function equilibrium_solver(raw_profile::DirectRunInput)

    # Shorthand
    equil_params = raw_profile.config
    psio = raw_profile.psio
    mtheta = equil_params.mtheta
    mpsi = equil_params.mpsi
    psilow = equil_params.psilow
    psihigh = equil_params.psihigh

    # Warn if psihigh is too close to 1.0
    if psihigh >= 1 - 1e-6
        @warn "Warning: direct equilibrium with psihigh = $(psihigh) could hang on separatrix."
    end

    # TODO: there's some fortran logic for grid_type = original that should be added when needed.

    # Set up radial and poloidal grid
    psi_nodes = Array{Float64}(undef, mpsi + 1)
    if equil_params.grid_type == "ldp"
        psi_nodes .= [psilow + (psihigh - psilow) * sin((ipsi / mpsi) * (π / 2))^2 for ipsi in 0:mpsi]
    else
        # TODO: add additional grid types
        error("Unsupported grid_type: $(equil_params.grid_type)")
    end
    theta_nodes = range(0.0, 1.0; length=mtheta + 1)

    # Find radial position of magnetic axis and separatrix
    ro, zo, rs1, rs2 = direct_position!(raw_profile)

    # Detect x-point (diverted vs limited topology); also retrieve ψ Hessian at x-point
    r_xpoint, z_xpoint, is_diverted, psi_hess_rr, psi_hess_zz, psi_hess_rz = direct_xpoint_find(raw_profile, ro, zo, rs1)
    if is_diverted
        println("   Plasma topology: DIVERTED (x-point detected)")
    else
        println("   Plasma topology: LIMITED (no x-point)")
    end

    # Loop over flux surfaces from outermost to innermost, integrating over field lines
    sq_nodes = zeros!(pool, Float64, mpsi + 1, 4)
    rzphi_nodes = zeros!(pool, Float64, mpsi + 1, mtheta + 1, 4)

    ff_val = zeros!(pool, Float64, 4)
    ff_deriv_val = zeros!(pool, Float64, 4)

    for ipsi in (mpsi+1):-1:1
        # Integrate along the field line for this surface
        y_out, bfield = direct_fieldline_int(psi_nodes[ipsi], raw_profile, ro, zo, rs2)

        # checkpoint pool for Float64 slot
        checkpoint!(pool, Float64)

        # Fit data into temporary straight fieldline poloidal angle splines
        ff_x_nodes = acquire!(pool, Float64, size(y_out, 1))
        @. ff_x_nodes = @view(y_out[:, 5]) / y_out[end, 5]

        ff_fs_nodes = acquire!(pool, Float64, size(y_out, 1), 4)
        @. ff_fs_nodes[:, 1] = @view(y_out[:, 3]) ^ 2
        @. ff_fs_nodes[:, 2] = @view(y_out[:, 1]) / (2π) - ff_x_nodes
        @. ff_fs_nodes[:, 3] = bfield.f * (@view(y_out[:, 4]) - ff_x_nodes * y_out[end, 4])
        @. ff_fs_nodes[:, 4] = @view(y_out[:, 2]) / y_out[end, 2] - ff_x_nodes

        # Enforce exact endpoint matching for periodic data (removes floating-point noise)
        ff_fs_nodes[end, :] .= ff_fs_nodes[1, :]

        # Create series interpolant for all columns
        ff_interp = cubic_interp(ff_x_nodes, ff_fs_nodes; bc=PeriodicBC())
        ff_deriv = deriv1(ff_interp)

        # Interpolate `ff` onto the uniform `theta` grid for `rzphi`
        for itheta in 1:(mtheta+1)
            theta = theta_nodes[itheta]

            # In-place operation to avoid allocations
            ff_interp(ff_val, theta)
            ff_deriv(ff_deriv_val, theta)

            rzphi_nodes[ipsi, itheta, 1] = ff_val[1]
            rzphi_nodes[ipsi, itheta, 2] = ff_val[2]
            rzphi_nodes[ipsi, itheta, 3] = ff_val[3]
            rzphi_nodes[ipsi, itheta, 4] = (1.0 + ff_deriv_val[4]) * y_out[end, 2] * 2π * psio
        end

        # Store surface-averaged quantities for the `sq` spline
        sq_nodes[ipsi, 1] = bfield.f * 2π
        sq_nodes[ipsi, 2] = bfield.p
        sq_nodes[ipsi, 3] = y_out[end, 2] * 2π * psio
        sq_nodes[ipsi, 4] = y_out[end, 4] * bfield.f / (2π)

        # rewind pool for Float64 slot
        rewind!(pool, Float64)
    end

    # Create temporary ProfileSplines for q-profile revision calculation
    profiles = ProfileSplines(
        psi_nodes,
        sq_nodes[:, 1],  # F * 2π
        sq_nodes[:, 2],  # P * μ₀
        sq_nodes[:, 3],  # dV/dψ
        sq_nodes[:, 4]   # q
    )
    # Calculate q0 using linear extrapolation: q(0) = q[1] - q'[1] * psi[1]
    q0 = profiles.q_spline.y[1] - profiles.q_deriv(psi_nodes[1]; hint=Ref(1)) * psi_nodes[1]
    if equil_params.newq0 == -1
        equil_params.newq0 = -q0
    end
    if equil_params.newq0 != 0.0
        @info "Revising q-profile for newq0 = $(@sprintf("%.3f", equil_params.newq0))"
        f0 = profiles.F_spline.y[1] - profiles.F_deriv(psi_nodes[1]; hint=Ref(1)) * psi_nodes[1]
        f0fac = f0^2 * ((equil_params.newq0 / q0)^2 - 1.0)
        for i in 1:(mpsi+1)
            ffac = sqrt(1.0 + f0fac / profiles.F_spline.y[i]^2) * sign(equil_params.newq0)
            sq_nodes[i, 1] *= ffac
            sq_nodes[i, 4] *= ffac
            rzphi_nodes[i, :, 3] .*= ffac
        end
        # Re-create profiles with the revised data
        profiles = ProfileSplines(
            psi_nodes,
            sq_nodes[:, 1],  # F * 2π
            sq_nodes[:, 2],  # P * μ₀
            sq_nodes[:, 3],  # dV/dψ
            sq_nodes[:, 4]   # q
        )
    end
    # For diverted plasmas, build edge inverse splines for q and dV/dψ that are well-behaved
    # toward psin=1. The reciprocals iota=1/q and 1/(dV/dψ) are anchored to 0 at the separatrix.
    # Also extend the rzphi coordinate mapping into the far edge using X-point asymptotic geometry.
    psi_nodes_core = psi_nodes  # save reference to core grid before possible extension
    if is_diverted
        # Edge spline covers the transition region from ~90% of psihigh to psifac=1.0
        psi_edge_start = psihigh * 0.9
        edge_mask = findall(psi -> psi >= psi_edge_start, psi_nodes)

        # Compute derivative-matching phantom knots just above psihigh.
        # Without these, the single anchor at (1.0, 0) pulls the iota spline slope at psihigh
        # away from -q'/q², causing a derivative discontinuity in q = 1/iota at the junction.
        # Adding N phantom knots using the Taylor expansion from the direct spline forces the
        # iota spline to reproduce the correct N-th order behaviour right at psihigh.
        phantom_delta = 1e-4
        q0   = sq_nodes[edge_mask[end], 4]
        q1   = profiles.q_deriv(psihigh)
        q2   = deriv2(profiles.q_spline_direct)(psihigh)
        q3   = deriv3(profiles.q_spline_direct)(psihigh)
        iota1_val  = -q1 / q0^2
        iota2_val  = (2*q1^2 - q0*q2) / q0^3
        iota3_val  = (-6*q1^3 + 6*q0*q1*q2 - q0^2*q3) / q0^4
        phantom_psi_q = [psihigh + j * phantom_delta for j in 1:3]
        phantom_iota  = [1/q0 + iota1_val*(j*phantom_delta) +
                         0.5*iota2_val*(j*phantom_delta)^2 +
                         (1/6)*iota3_val*(j*phantom_delta)^3 for j in 1:3]

        V0   = sq_nodes[edge_mask[end], 3]
        V1   = profiles.dVdpsi_deriv(psihigh)
        V2   = deriv2(profiles.dVdpsi_spline)(psihigh)
        Vinv1_val = -V1 / V0^2
        Vinv2_val = (2*V1^2 - V0*V2) / V0^3
        phantom_Vinv = [1/V0 + Vinv1_val*(j*phantom_delta) +
                        0.5*Vinv2_val*(j*phantom_delta)^2 for j in 1:2]

        # Intermediate points between phantom knots and the psi=1 anchor.
        # CubicFit BC over a long empty interval creates a flat S-curve artifact in iota.
        # Use iota(s) = A*√s + B*s + C*s^(3/2) with s = 1-psi, which enforces C² continuity
        # (value, first, and second derivative) at psi=psihigh and vanishes at psi=1.
        # Solving the 3×3 linear system at s=δ=1-psihigh gives:
        #   A = (3ι₀ + 3ι₁δ + 2ι₂δ²)/√δ
        #   B = -ι₁ - A/√δ - 2ι₂δ
        #   C = (4ι₂√δ)/3 + A/(3δ)
        n_mid = 8
        iota0 = 1.0 / q0
        Vinv0 = 1.0 / V0
        δ = 1.0 - psihigh
        A_iota = (3*iota0 + 3*iota1_val*δ + 2*iota2_val*δ^2) / sqrt(δ)
        B_iota = -iota1_val - A_iota/sqrt(δ) - 2*iota2_val*δ
        C_iota = (4*iota2_val*sqrt(δ))/3 + A_iota/(3*δ)
        A_Vinv = (3*Vinv0 + 3*Vinv1_val*δ + 2*Vinv2_val*δ^2) / sqrt(δ)
        B_Vinv = -Vinv1_val - A_Vinv/sqrt(δ) - 2*Vinv2_val*δ
        C_Vinv = (4*Vinv2_val*sqrt(δ))/3 + A_Vinv/(3*δ)
        psi_mid   = [psihigh + δ * i/(n_mid + 1) for i in 1:n_mid]
        iota_mid  = [A_iota*sqrt(1.0-p) + B_iota*(1.0-p) + C_iota*(1.0-p)^1.5 for p in psi_mid]
        Vinv_mid  = [A_Vinv*sqrt(1.0-p) + B_Vinv*(1.0-p) + C_Vinv*(1.0-p)^1.5 for p in psi_mid]

        edge_psi = vcat(psi_nodes[edge_mask], phantom_psi_q, psi_mid, 1.0)
        sort!(edge_psi)

        # Iota = 1/q; anchor iota(1.0) = 0 (q → ∞ at separatrix)
        q_iota_vals = vcat([1.0 / sq_nodes[i, 4] for i in edge_mask], phantom_iota, iota_mid, 0.0)
        q_iota_vals = q_iota_vals[sortperm(vcat(psi_nodes[edge_mask], phantom_psi_q, psi_mid, 1.0))]
        q_iota_inner = cubic_interp(edge_psi, q_iota_vals; bc=CubicFit(), extrap=ExtendExtrap(), search=LinearBinary())
        profiles.q_spline_iota_inverse = InverseCubicSpline(q_iota_inner)

        # 1/(dV/dψ); anchor (dV/dψ)⁻¹(1.0) = 0 (dV/dψ → ∞ at separatrix)
        # Uses 2 phantom knots (1st + 2nd derivative matching; 3rd not needed for dV/dψ)
        edge_psi_dV = vcat(psi_nodes[edge_mask], phantom_psi_q[1:2], psi_mid, 1.0)
        sort!(edge_psi_dV)
        dVdpsi_inv_vals = vcat([1.0 / sq_nodes[i, 3] for i in edge_mask], phantom_Vinv, Vinv_mid, 0.0)
        dVdpsi_inv_vals = dVdpsi_inv_vals[sortperm(vcat(psi_nodes[edge_mask], phantom_psi_q[1:2], psi_mid, 1.0))]
        dVdpsi_inv_inner = cubic_interp(edge_psi_dV, dVdpsi_inv_vals; bc=CubicFit(), extrap=ExtendExtrap(), search=LinearBinary())
        profiles.dVdpsi_spline_inv = InverseCubicSpline(dVdpsi_inv_inner)

        @printf("   Edge inverse splines built over psin ∈ [%.4f, 1.0] (%d nodes)\n",
            psi_edge_start, length(edge_psi))

        # Extend rzphi nodes into the far edge using X-point asymptotic geometry.
        # This replaces the constant ExtendExtrap behaviour above psihigh with a geometry
        # that shrinks flux surfaces toward the X-point (preserving GS consistency with F'≈0).
        psi_nodes, rzphi_nodes = direct_build_xpoint_asymptotic!(
            profiles, psi_nodes, rzphi_nodes, ro, zo, theta_nodes, r_xpoint, z_xpoint)
    end

    # Create 2D interpolants for geometric quantities (rzphi) with CubicFit/Periodic BCs.
    # theta_nodes includes both 0 and 1 (closed periodic grid).
    # rzphi_xs is the (possibly extended) psi grid; eqfun is built on the core grid only.
    rzphi_xs = psi_nodes
    # rzphi_ys is the materialized Vector stored in PlasmaEquilibrium for indexing/diagnostics.
    # The Range form (theta_nodes) is used for the interpolant: it skips index search during
    # evaluation (O(1) vs binary search) and may differ at machine-precision level from Vector.
    rzphi_ys = collect(theta_nodes)

    grid2d = (rzphi_xs, theta_nodes)

    opts2d = (search=LinearBinary(), bc=(CubicFit(), PeriodicBC()), extrap=(ExtendExtrap(), WrapExtrap()))

    rzphi_rsquared = cubic_interp(grid2d, rzphi_nodes[:, :, 1]; opts2d...)
    rzphi_offset = cubic_interp(grid2d, rzphi_nodes[:, :, 2]; opts2d...)
    rzphi_nu = cubic_interp(grid2d, rzphi_nodes[:, :, 3]; opts2d...)
    rzphi_jac = cubic_interp(grid2d, rzphi_nodes[:, :, 4]; opts2d...)

    # Calculate physics quantities (B-field, metric components, etc.) in 2D spline `eqfun`
    # for use in stability and transport codes. Built on the core psi grid only (ForceFreeStates
    # uses rzphi directly and does not use eqfun, so far-edge extension is not needed here).
    eqfun_fs_nodes = zeros(Float64, mpsi + 1, mtheta + 1, 3)
    v = @MMatrix zeros(Float64, 2, 3)
    for ipsi in 1:(mpsi+1)
        q = profiles.q_spline.y[ipsi]
        f_val = profiles.F_spline.y[ipsi]
        for itheta in 1:(mtheta+1)
            theta_norm = theta_nodes[itheta]
            # Access nodal derivatives from the interpolants (grid points)
            # partials indexing: [1,:,:] = f, [2,:,:] = ∂f/∂x, [3,:,:] = ∂f/∂y, [4,:,:] = ∂²f/∂x∂y
            f = (
                rzphi_rsquared.nodal_derivs.partials[1, ipsi, itheta],
                rzphi_offset.nodal_derivs.partials[1, ipsi, itheta],
                rzphi_nu.nodal_derivs.partials[1, ipsi, itheta],
                rzphi_jac.nodal_derivs.partials[1, ipsi, itheta]
            )
            fx = (
                rzphi_rsquared.nodal_derivs.partials[2, ipsi, itheta],
                rzphi_offset.nodal_derivs.partials[2, ipsi, itheta],
                rzphi_nu.nodal_derivs.partials[2, ipsi, itheta],
                rzphi_jac.nodal_derivs.partials[2, ipsi, itheta]
            )
            fy = (
                rzphi_rsquared.nodal_derivs.partials[3, ipsi, itheta],
                rzphi_offset.nodal_derivs.partials[3, ipsi, itheta],
                rzphi_nu.nodal_derivs.partials[3, ipsi, itheta],
                rzphi_jac.nodal_derivs.partials[3, ipsi, itheta]
            )
            rfac = sqrt(max(0.0, f[1])) # add in protection just in case of small negative due to numerical error
            eta = 2π * (theta_norm + f[2])
            r = ro + rfac * cos(eta)
            jacfac = f[4]

            v[1, 1] = (rfac > 0) ? fx[1] / (2.0 * rfac) : 0.0       # 1/(2rfac) * d(rfac)/d(psi_norm)
            v[1, 2] = fx[2] * 2π * rfac                             # 2π*rfac * d(eta)/d(psi_norm)
            v[1, 3] = fx[3] * r                                     # r * d(phi_s)/d(psi_norm)
            v[2, 1] = (rfac > 0) ? fy[1] / (2.0 * rfac) : 0.0       # 1/(2rfac) d(rfac)/d(theta_new)
            v[2, 2] = (1.0 + fy[2]) * 2π * rfac                     # 2π*rfac * d(eta)/d(theta_new)
            v[2, 3] = fy[3] * r                                     # r * d(phi_s)/d(theta_new)
            v33 = 2π * r
            w11 = (jacfac != 0) ? (1.0 + fy[2]) * (2π)^2 * rfac * r / jacfac : 0.0
            w12 = (jacfac * rfac != 0) ? -fy[1] * π * r / (rfac * jacfac) : 0.0
            delpsi_norm = sqrt(w11^2 + w12^2)
            modB = sqrt(((2π * psio * delpsi_norm)^2 + f_val^2) / (2π * r)^2)

            # Fill in eqfun nodes
            eqfun_fs_nodes[ipsi, itheta, 1] = modB
            denom = jacfac * modB^2
            if abs(denom) > 1e-20
                # Gyrokinetic coefficient C1
                numerator_2 = dot(v[1, :], v[2, :]) + q * v33 * v[1, 3]
                eqfun_fs_nodes[ipsi, itheta, 2] = numerator_2 / denom
                # Gyrokinetic coefficient C2
                numerator_3 = v[2, 3] * v33 + q * v33^2
                eqfun_fs_nodes[ipsi, itheta, 3] = numerator_3 / denom
            else
                eqfun_fs_nodes[ipsi, itheta, 2] = 0.0
                eqfun_fs_nodes[ipsi, itheta, 3] = 0.0
            end
        end
    end
    # Create 2D interpolants for physics quantities (eqfun) on the CORE psi grid
    core_grid2d = (psi_nodes_core, theta_nodes)
    eqfun_B = cubic_interp(core_grid2d, eqfun_fs_nodes[:, :, 1]; opts2d...)
    eqfun_metric1 = cubic_interp(core_grid2d, eqfun_fs_nodes[:, :, 2]; opts2d...)
    eqfun_metric2 = cubic_interp(core_grid2d, eqfun_fs_nodes[:, :, 3]; opts2d...)

    pe = PlasmaEquilibrium(raw_profile.config, EquilibriumParameters(), profiles,
        rzphi_xs, rzphi_ys,
        rzphi_rsquared, rzphi_offset, rzphi_nu, rzphi_jac,
        eqfun_B, eqfun_metric1, eqfun_metric2,
        ro, zo, psio)

    # Store x-point topology and Hessian in equilibrium parameters
    pe.params.r_xpoint = is_diverted ? r_xpoint : nothing
    pe.params.z_xpoint = is_diverted ? z_xpoint : nothing
    pe.params.is_diverted = is_diverted
    pe.params.psi_core_end = psihigh  # last psi in core field-line grid (before far-edge extension)
    if is_diverted
        pe.params.psi_hess_rr = psi_hess_rr
        pe.params.psi_hess_zz = psi_hess_zz
        pe.params.psi_hess_rz = psi_hess_rz
    end

    return pe
end
