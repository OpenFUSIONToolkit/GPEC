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
    power_rc::Int  # minor radius rfac = √((R-R₀)²+(Z-Z₀)²) power exponent
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
        equil_config.power_bp, equil_config.power_b, equil_config.power_r, equil_config.power_rc, bfield)

    # Use a callback to refine the solution at each step to stay on the flux surface
    function refine_affect!(integrator)
        integrator.u[2] = direct_refine(integrator.u[2], integrator.t, psi0, params)
    end

    # We save the solution at each step before refinement (before=true, after=false) to match Fortran
    callback = DiscreteCallback((u, t, i) -> true, refine_affect!; save_positions=(true, false))

    prob = ODEProblem{true}(direct_fieldline_der!, u0, (0.0, 2π), params)
    sol = solve(prob, BS5(); callback=callback, reltol=equil_config.etol, abstol=1e-8, dt=2π / 200, adaptive=true, dense=false)

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
    rfac = y[2]
    jac = (bp^params.power_bp) * (b^params.power_b) / (r^params.power_r * rfac^params.power_rc)

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
    _estimate_log_slope(fieldline_int, raw_profile, ro, zo, rs2, psihigh)

Estimate the logarithmic slope A from two field-line probe integrations near the separatrix,
using the asymptotic form q(ψ) ≃ −A·ln(1−ψ) (Fitzpatrick 2024, eq. 19).

Returns A = |Δq| / ln(2). Falls back to A=2.0 on integration failure.
"""
function _estimate_log_slope(fieldline_int, raw_profile, ro, zo, rs2, psihigh)
    eps_sep = max(1.0 - psihigh, 0.001)
    psi1 = clamp(1.0 - 3 * eps_sep, raw_profile.config.psilow + 0.01, 0.999)
    psi2 = clamp(1.0 - 1.5 * eps_sep, raw_profile.config.psilow + 0.01, 0.999)
    try
        out1 = fieldline_int(psi1, raw_profile, ro, zo, rs2)
        q1 = out1[2].f * out1[1][end, 4] / (2π)
        out2 = fieldline_int(psi2, raw_profile, ro, zo, rs2)
        q2 = out2[2].f * out2[1][end, 4] / (2π)
        A = max(abs(q2 - q1) / log(2), 0.1)
        @info "Estimated separatrix log slope A = $(@sprintf("%.3f", A)) from probe integrations at psi = $(@sprintf("%.4f", psi1)), $(@sprintf("%.4f", psi2))"
        return A
    catch err
        @warn "Failed to estimate log slope from probe integrations, using default A=2.0: $err"
        return 2.0
    end
end

"""
    make_optimal_mpsi(psilow, psihigh, A; tau, psi_split_core, psi_split_edge)

Compute the minimum number of radial knots needed to achieve target accuracy τ in q,
given the separatrix log slope A. Three-region geometric grid: core, pedestal, far edge.
"""
function make_optimal_mpsi(psilow, psihigh, A;
        tau=0.005, psi_split_core=0.15, psi_split_edge=0.95)
    dlog = (13.0 * tau / A)^(1/4)
    N_edge = ceil(Int, log((1.0 - psi_split_edge) / (1.0 - psihigh)) / dlog) + 1
    h_mid = (1.0 - psi_split_edge) * dlog
    N_mid = ceil(Int, (psi_split_edge - psi_split_core) / h_mid)
    N_core = ceil(Int, log(psi_split_core / psilow) / dlog)
    mpsi = N_core + N_mid + N_edge
    @info "Auto-mpsi: N_core=$N_core + N_mid=$N_mid + N_edge=$N_edge = $mpsi (A=$(@sprintf("%.3f",A)), tau=$tau)"
    return mpsi
end

"""
    make_optimal_psi_grid(psilow, psihigh, mpsi; psi_split_core, psi_split_edge)

Build a three-region ψ grid with mpsi+1 knots:
- Core  [psilow, psi_split_core]: geometric in log(ψ)        — handles axis behavior
- Middle [psi_split_core, psi_split_edge]: uniform in ψ       — protects pedestal resolution
- Edge  [psi_split_edge, psihigh]: geometric in log(1−ψ)     — handles logarithmic separatrix

Knot counts are allocated by equal log-weight with N_edge capped at 50% to protect pedestal.
"""
function make_optimal_psi_grid(psilow, psihigh, mpsi;
        psi_split_core=0.15, psi_split_edge=0.95)
    log_core = log(psi_split_core / psilow)
    log_mid  = log(psi_split_edge / psi_split_core)
    log_edge = log((1.0 - psi_split_edge) / (1.0 - psihigh))
    log_total = log_core + log_mid + log_edge

    N_edge = clamp(round(Int, mpsi * log_edge / log_total), 2, mpsi ÷ 2)
    N_core = round(Int, mpsi * log_core / log_total)
    N_mid  = mpsi - N_edge - N_core

    # Core: [psilow, psi_split_core], geometric in log(ψ)
    core_pts = [psilow * (psi_split_core / psilow)^(i / N_core) for i in 0:N_core]
    # Middle: [psi_split_core, psi_split_edge], uniform (skip first to avoid duplicate)
    mid_pts = [psi_split_core + (psi_split_edge - psi_split_core) * i / N_mid for i in 1:N_mid]
    # Edge: [psi_split_edge, psihigh], geometric in log(1−ψ) (skip first to avoid duplicate)
    edge_pts = [1.0 - (1.0 - psi_split_edge) * ((1.0 - psihigh) / (1.0 - psi_split_edge))^(i / N_edge) for i in 1:N_edge]

    return vcat(core_pts, mid_pts, edge_pts)
end

"""
    _build_psi_grid(equil_params, psilow, psihigh, fieldline_int, raw_profile, ro, zo, rs2)

Resolve `mpsi` and build `psi_nodes` for any supported `grid_type`.

For `"log_asymptotic"` with `mpsi=0`, estimates the separatrix log slope from two probe
integrations and computes the minimum knot count for `psi_accuracy`. For `"ldp"`, uses
the sin²-spaced grid. Shared by `direct_fieldline_int` and `efit_by_inversion` solvers.
"""
function _build_psi_grid(equil_params, psilow, psihigh, fieldline_int, raw_profile, ro, zo, rs2)
    mpsi = equil_params.mpsi
    if equil_params.grid_type == "log_asymptotic" && mpsi == 0 && equil_params.psi_accuracy > 0
        A = _estimate_log_slope(fieldline_int, raw_profile, ro, zo, rs2, psihigh)
        mpsi = make_optimal_mpsi(psilow, psihigh, A; tau=equil_params.psi_accuracy)
    elseif mpsi == 0
        mpsi = 128
    end

    psi_nodes = if equil_params.grid_type == "log_asymptotic"
        make_optimal_psi_grid(psilow, psihigh, mpsi)
    elseif equil_params.grid_type == "ldp"
        [psilow + (psihigh - psilow) * sin((ipsi / mpsi) * (π / 2))^2 for ipsi in 0:mpsi]
    else
        error("Unsupported grid_type: $(equil_params.grid_type)")
    end
    return psi_nodes
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
@with_pool pool function equilibrium_solver(raw_profile::DirectRunInput, fieldline_int=direct_fieldline_int)

    equil_params = raw_profile.config
    psio = raw_profile.psio
    mtheta = equil_params.mtheta
    psilow = equil_params.psilow
    psihigh = equil_params.psihigh

    # direct_position! must run before building psi_nodes: probe integrations need ro, zo, rs2
    ro, zo, _, rs2 = direct_position!(raw_profile)

    psi_nodes = _build_psi_grid(equil_params, psilow, psihigh, fieldline_int, raw_profile, ro, zo, rs2)
    mpsi = length(psi_nodes) - 1
    theta_nodes = range(0.0, 1.0; length=mtheta + 1)

    sq_nodes = zeros!(pool, Float64, mpsi + 1, 4)
    rzphi_nodes = zeros!(pool, Float64, mpsi + 1, mtheta + 1, 4)
    ff_val = zeros!(pool, Float64, 4)
    ff_deriv_val = zeros!(pool, Float64, 4)

    for ipsi in (mpsi+1):-1:1  # outermost to innermost
        y_out, bfield = fieldline_int(psi_nodes[ipsi], raw_profile, ro, zo, rs2)
        checkpoint!(pool, Float64)

        # Fit data into temporary straight fieldline poloidal angle splines
        ff_x_nodes = acquire!(pool, Float64, size(y_out, 1))
        @. ff_x_nodes = @view(y_out[:, 5]) / y_out[end, 5]

        ff_fs_nodes = acquire!(pool, Float64, size(y_out, 1), 4)
        @. ff_fs_nodes[:, 1] = @view(y_out[:, 3]) ^ 2
        @. ff_fs_nodes[:, 2] = @view(y_out[:, 1]) / (2π) - ff_x_nodes
        @. ff_fs_nodes[:, 3] = bfield.f * (@view(y_out[:, 4]) - ff_x_nodes * y_out[end, 4])
        @. ff_fs_nodes[:, 4] = @view(y_out[:, 2]) / y_out[end, 2] - ff_x_nodes

        ff_fs_nodes[end, :] .= ff_fs_nodes[1, :]  # enforce periodic endpoint

        ff_interp = cubic_interp(ff_x_nodes, ff_fs_nodes; bc=PeriodicBC())
        ff_deriv = deriv1(ff_interp)

        # Resample ff onto uniform theta grid
        for itheta in 1:(mtheta+1)
            theta = theta_nodes[itheta]
            ff_interp(ff_val, theta)
            ff_deriv(ff_deriv_val, theta)

            rzphi_nodes[ipsi, itheta, 1] = ff_val[1]
            rzphi_nodes[ipsi, itheta, 2] = ff_val[2]
            rzphi_nodes[ipsi, itheta, 3] = ff_val[3]
            rzphi_nodes[ipsi, itheta, 4] = (1.0 + ff_deriv_val[4]) * y_out[end, 2] * 2π * psio
        end

        sq_nodes[ipsi, 1] = bfield.f * 2π
        sq_nodes[ipsi, 2] = bfield.p
        sq_nodes[ipsi, 3] = y_out[end, 2] * 2π * psio
        sq_nodes[ipsi, 4] = y_out[end, 4] * bfield.f / (2π)
        rewind!(pool, Float64)
    end

    # Temporary splines for q0 extrapolation and optional newq0 revision
    profiles = ProfileSplines(
        psi_nodes,
        sq_nodes[:, 1],  # F * 2π
        sq_nodes[:, 2],  # P * μ₀
        sq_nodes[:, 3],  # dV/dψ
        sq_nodes[:, 4]   # q
    )
    # q(0) by linear extrapolation from innermost surface
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
        profiles = ProfileSplines(
            psi_nodes,
            sq_nodes[:, 1],  # F * 2π
            sq_nodes[:, 2],  # P * μ₀
            sq_nodes[:, 3],  # dV/dψ
            sq_nodes[:, 4]   # q
        )
    end

    rzphi_xs = psi_nodes
    # rzphi_ys is a materialized Vector (not the Range) so PlasmaEquilibrium can index it directly
    rzphi_ys = collect(theta_nodes)

    grid2d = (rzphi_xs, theta_nodes)
    opts2d = (search=LinearBinary(), bc=(CubicFit(), PeriodicBC()), extrap=(ExtendExtrap(), WrapExtrap()))

    rzphi_rsquared = cubic_interp(grid2d, rzphi_nodes[:, :, 1]; opts2d...)
    rzphi_offset = cubic_interp(grid2d, rzphi_nodes[:, :, 2]; opts2d...)
    rzphi_nu = cubic_interp(grid2d, rzphi_nodes[:, :, 3]; opts2d...)
    rzphi_jac = cubic_interp(grid2d, rzphi_nodes[:, :, 4]; opts2d...)

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
            rfac = sqrt(max(0.0, f[1]))  # guard against spline overshoot near separatrix
            eta = 2π * (theta_norm + f[2])
            r = ro + rfac * cos(eta)
            jacfac = f[4]

            v[1, 1] = (rfac > 0) ? fx[1] / (2.0 * rfac) : 0.0  # 1/(2rfac) * d(rfac)/d(psi_norm)
            v[1, 2] = fx[2] * 2π * rfac                          # 2π*rfac * d(eta)/d(psi_norm)
            v[1, 3] = fx[3] * r                                   # r * d(phi_s)/d(psi_norm)
            v[2, 1] = (rfac > 0) ? fy[1] / (2.0 * rfac) : 0.0  # 1/(2rfac) d(rfac)/d(theta_new)
            v[2, 2] = (1.0 + fy[2]) * 2π * rfac                  # 2π*rfac * d(eta)/d(theta_new)
            v[2, 3] = fy[3] * r                                   # r * d(phi_s)/d(theta_new)
            v33 = 2π * r
            w11 = (jacfac != 0) ? (1.0 + fy[2]) * (2π)^2 * rfac * r / jacfac : 0.0
            w12 = (jacfac * rfac != 0) ? -fy[1] * π * r / (rfac * jacfac) : 0.0
            delpsi_norm = sqrt(w11^2 + w12^2)
            modB = sqrt(((2π * psio * delpsi_norm)^2 + f_val^2) / (2π * r)^2)

            eqfun_fs_nodes[ipsi, itheta, 1] = modB
            denom = jacfac * modB^2
            if abs(denom) > 1e-20
                numerator_2 = dot(v[1, :], v[2, :]) + q * v33 * v[1, 3]  # gyrokinetic C1
                eqfun_fs_nodes[ipsi, itheta, 2] = numerator_2 / denom
                numerator_3 = v[2, 3] * v33 + q * v33^2                  # gyrokinetic C2
                eqfun_fs_nodes[ipsi, itheta, 3] = numerator_3 / denom
            else
                eqfun_fs_nodes[ipsi, itheta, 2] = 0.0
                eqfun_fs_nodes[ipsi, itheta, 3] = 0.0
            end
        end
    end

    eqfun_B = cubic_interp(grid2d, eqfun_fs_nodes[:, :, 1]; opts2d...)
    eqfun_metric1 = cubic_interp(grid2d, eqfun_fs_nodes[:, :, 2]; opts2d...)
    eqfun_metric2 = cubic_interp(grid2d, eqfun_fs_nodes[:, :, 3]; opts2d...)

    return PlasmaEquilibrium(raw_profile.config, EquilibriumParameters(), profiles,
        rzphi_xs, rzphi_ys,
        rzphi_rsquared, rzphi_offset, rzphi_nu, rzphi_jac,
        eqfun_B, eqfun_metric1, eqfun_metric2,
        ro, zo, psio)
end
