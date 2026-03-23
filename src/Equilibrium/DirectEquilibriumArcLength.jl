"""
Arc-length parameterized field-line integration for the direct (EFIT) equilibrium path.

Provides `arclength_fieldline_int` as a drop-in replacement for `direct_fieldline_int`.
Pass it to `equilibrium_solver` via `eq_type = "efit_arclength"` in `gpec.toml`, which
routes to `equilibrium_solver(raw_profile, arclength_fieldline_int)`.

The arc-length ODE integrates along the tangent to ψ = const, avoiding the
denominator singularity of the geometric-angle ODE near x-points.
"""

"""
    arclength_fieldline_der!(dy, y, params, s)

RHS for the arc-length-parameterized level-set ODE.

State `y = [R, Z, ∫dl/Bp, ∫dl/(R²Bp), ∫jac·dl/Bp]`.
The independent variable `s` is arc length [m].

The tangent direction `(dR/ds, dZ/ds) = (∂ψ/∂Z, −∂ψ/∂R) / |∇ψ|` follows the
ψ = const level set counterclockwise, staying on the flux surface without correction.

Near x-points where Bp → 0, the position integration (dy[1:2]) remains well-behaved
because `grad_norm` never appears in the denominator alone. dy[3:5] use abstol=1e20
so the solver never restricts step size for them.
"""
@with_pool pool function arclength_fieldline_der!(dy, y, params::FieldLineDerivParams, _s)
    R, Z = y[1], y[2]
    direct_get_bfield!(params.bfield, R, Z, params.psi_in, params.sq_in, params.sq_in_deriv, params.psio; derivs=1)

    psir = params.bfield.psir  # ∂ψ/∂R
    psiz = params.bfield.psiz  # ∂ψ/∂Z
    grad_norm = sqrt(psir^2 + psiz^2)

    if grad_norm < 1e-14
        fill!(dy, 0.0)
        @warn "Near-zero |∇ψ| at (R=$R, Z=$Z) in arc-length ODE — likely near O-point."
        return
    end

    # Level-set tangent (CCW orientation)
    dy[1] = psiz / grad_norm   # dR/ds
    dy[2] = -psir / grad_norm  # dZ/ds

    # Bp = |∇ψ| / R (poloidal field [T])
    Bp = grad_norm / R

    # dy[3] and dy[4] use abstol=1e20 so the solver never restricts steps for them.
    Bp_eff = max(Bp, eps(Float64))  # absolute guard against exact Bp=0 only
    dy[3] = 1.0 / Bp_eff           # d/ds [∫dl/Bp]
    dy[4] = 1.0 / (R^2 * Bp_eff)  # d/ds [∫dl/(R²Bp)]

    # d/ds [∫jac·dl/Bp]: integrand = Bp^(power_bp−1) · B^power_b / (R^power_r · rfac^power_rc)
    # (power_bp−1 because arc-length parameterization already carries the 1/Bp from dl/Bp)
    Bt = params.bfield.f / R
    B_val = sqrt(Bp^2 + Bt^2)
    rfac = sqrt((R - params.ro)^2 + (Z - params.zo)^2)
    rfac_eff = max(rfac, eps(Float64))
    jac_core_5 = Bp_eff^(params.power_bp - 1) * B_val^params.power_b / (R^params.power_r * rfac_eff^params.power_rc)
    jac_edge_5 = Bp_eff^(params.power_bp_edge - 1) * B_val^params.power_b_edge / (R^params.power_r_edge * rfac_eff^params.power_rc_edge)
    dy[5] = (1 - params.w) * jac_core_5 + params.w * jac_edge_5
end

"""
    _arclength_fieldline_sol(psifac, raw_profile, ro, zo, rs2) -> (sol, bfield)

Run the arc-length ODE and return the raw ODE solution and the starting-point bfield.
`sol.t[i]` is the arc length at each saved step; `sol.u[i][1:2]` are R,Z.
Called by both `arclength_fieldline_int` (post-processes to y_out) and
`equal_arc_fieldline_output` (uses `sol.t` for the arc-length parameterization).
"""
@with_pool pool function _arclength_fieldline_sol(
    psifac::Float64, raw_profile::DirectRunInput, ro::Float64, zo::Float64, rs2::Float64
)
    psi0_guess = raw_profile.psio * (1.0 - psifac)
    r = ro + sqrt(psifac) * (rs2 - ro)
    z = zo
    bfield = DirectBField()
    sq_in_deriv = deriv1(raw_profile.sq_in)

    # Refine starting point — same Newton solve as direct_fieldline_int
    r = find_zero(
        (r_in -> (direct_get_bfield!(bfield, r_in, z, raw_profile.psi_in, raw_profile.sq_in, sq_in_deriv, raw_profile.psio; derivs=1); bfield.psi - psi0_guess),
            _ -> bfield.psir),
        r, Roots.Newton()
    )

    direct_get_bfield!(bfield, r, z, raw_profile.psi_in, raw_profile.sq_in, sq_in_deriv, raw_profile.psio; derivs=2)

    # Initial state: [R, Z, ∫dl/Bp, ∫dl/(R²Bp), ∫jac·dl/Bp]
    u0 = Float64[r, z, 0.0, 0.0, 0.0]

    equil_config = raw_profile.config
    bfield_ode = DirectBField()
    w = jac_blend_weight(psifac, equil_config.psi_jac_blend_start, equil_config.psi_jac_blend_end)
    params = FieldLineDerivParams(ro, zo, raw_profile.psi_in, raw_profile.sq_in, sq_in_deriv, raw_profile.psio,
        equil_config.power_bp, equil_config.power_b, equil_config.power_r, equil_config.power_rc,
        equil_config.power_bp_edge, equil_config.power_b_edge, equil_config.power_r_edge, equil_config.power_rc_edge,
        w, bfield_ode)

    t_min = π * (r - ro)  # minimum arc before termination (half-circumference lower bound)
    condition(u, _t, integrator) = u[2] - zo
    function affect_upward!(integrator)
        if integrator.t > t_min
            terminate!(integrator)
        end
    end
    callback = ContinuousCallback(condition, affect_upward!, nothing; save_positions=(true, false))

    prob = ODEProblem{true}(arclength_fieldline_der!, u0, (0.0, 1.0e4), params)
    # Tight tolerances on position (y[1:2]); integrals (y[3:5]) effectively unconstrained near x-points
    reltol_vec = [equil_config.etol, equil_config.etol, 1e20, 1e20, 1e20]
    abstol_vec = [1e-8, 1e-8, 1e20, 1e20, 1e20]
    sol = solve(prob, BS5(); callback=callback, reltol=reltol_vec, abstol=abstol_vec,
        dt=2π / 200, adaptive=true, dense=false)

    return sol, bfield
end

"""
    arclength_fieldline_int(psifac, raw_profile, ro, zo, rs2)

Arc-length-parameterized flux surface integration. Drop-in replacement for
`direct_fieldline_int` with identical return format:

- `y_out[:, 1]`: geometric angle η ∈ 0 to 2π (CCW from outboard midplane)
- `y_out[:, 2]`: accumulated ∫dl/Bp
- `y_out[:, 3]`: rfac = √((R−ro)² + (Z−zo)²)
- `y_out[:, 4]`: accumulated ∫dl/(R²Bp)
- `y_out[:, 5]`: accumulated ∫jac·dl/Bp

The ODE is terminated by a `ContinuousCallback` that detects the return to the
outboard midplane (Z = zo, R > ro) after a minimum arc-length guard.
"""
@with_pool pool function arclength_fieldline_int(
    psifac::Float64, raw_profile::DirectRunInput, ro::Float64, zo::Float64, rs2::Float64
)::Tuple{Matrix{Float64},DirectBField}

    sol, bfield = _arclength_fieldline_sol(psifac, raw_profile, ro, zo, rs2)

    n = length(sol.u)
    y_out = Matrix{Float64}(undef, n, 5)

    # Compute geometric angles and unwrap to monotone [0, 2π]
    η_raw = [atan(sol.u[i][2] - zo, sol.u[i][1] - ro) for i in 1:n]
    for i in 2:n
        Δ = η_raw[i] - η_raw[i-1]
        if Δ > π
            η_raw[i:end] .-= 2π
        elseif Δ < -π
            η_raw[i:end] .+= 2π
        end
    end
    η_unwrapped = η_raw .- η_raw[1]  # shift so first point is 0
    η_unwrapped[end] = 2π            # enforce exact 2π at the closed endpoint

    for i in 1:n
        R_i, Z_i = sol.u[i][1], sol.u[i][2]
        y_out[i, 1] = η_unwrapped[i]
        y_out[i, 2] = sol.u[i][3]
        y_out[i, 3] = sqrt((R_i - ro)^2 + (Z_i - zo)^2)
        y_out[i, 4] = sol.u[i][4]
        y_out[i, 5] = sol.u[i][5]
    end

    # bfield at the starting point carries F and P for the surface-averaged quantities
    return y_out, bfield
end

"""
    equal_arc_fieldline_output(equil::PlasmaEquilibrium, psifac::Float64; mtheta_eq::Int)
        -> (R_eq, Z_eq, nu_eq, theta_jac_eq)

Run the arc-length ODE at flux surface `psifac` and interpolate to `mtheta_eq` equally-spaced
arc-length positions σ_k = k/mtheta_eq (k = 0, …, mtheta_eq−1).

Returns:
  - `R_eq`, `Z_eq`: physical coordinates at equal-arc positions
  - `nu_eq`: toroidal angle offset ν(σ_k) = F*(col4(θ_jac) − θ_jac*col4_total), matching `rzphi_nu`
  - `theta_jac_eq`: jac_type SFL angle θ_jac(σ_k) ∈ [0, 1)

Requires `equil.raw_profile` to be a `DirectRunInput` (set for EFIT/CHEASE/analytic paths).
"""
function equal_arc_fieldline_output(equil::PlasmaEquilibrium, psifac::Float64; mtheta_eq::Int)
    raw_profile = equil.raw_profile
    isnothing(raw_profile) && error(
        "equal_arc_fieldline_output requires equil.raw_profile (not available for inverse-method equilibria)")

    # Estimate starting R on the outboard midplane
    rs2_outer = raw_profile.rmax
    rs2 = equil.ro + sqrt(psifac) * (rs2_outer - equil.ro)

    # Run arc-length ODE. sol.t[i] is the arc-length parameter at each saved step — no need
    # to recompute arc from chord lengths. sol.u[i][1:2] are R,Z directly from the ODE state.
    # bfield is evaluated at the starting point (outboard midplane); bfield.f = F(ψ) = R*Bt.
    sol, bfield = _arclength_fieldline_sol(psifac, raw_profile, equil.ro, equil.zo, rs2)

    n = length(sol.u)
    R_raw = [sol.u[i][1] for i in 1:n]
    Z_raw = [sol.u[i][2] for i in 1:n]
    arc_norm = sol.t ./ sol.t[end]  # normalized arc length ∈ [0, 1]

    # Build cubic splines on the ODE grid (arc_norm → quantities)
    # arc_norm ∈ [0,1] inclusive; queries at σ ∈ [0,1) are safely interior.
    opts = (; bc=CubicFit(), extrap=ExtendExtrap(), search=LinearBinary())
    spline_R    = cubic_interp(arc_norm, R_raw;                      opts...)
    spline_Z    = cubic_interp(arc_norm, Z_raw;                      opts...)
    spline_col4 = cubic_interp(arc_norm, [sol.u[i][4] for i in 1:n]; opts...)  # ∫dl/(R²Bp)
    spline_col5 = cubic_interp(arc_norm, [sol.u[i][5] for i in 1:n]; opts...)  # ∫jac·dl/Bp

    col4_total = sol.u[end][4]
    col5_total = sol.u[end][5]

    # F = R*Bt at this flux surface (F depends only on ψ in axisymmetry).
    # ν(ψ, θ_jac) = F * (col4(θ_jac) − θ_jac × col4_total)
    # matching the equilibrium_solver formula: ff_fs_nodes[:,3] = bfield.f * (y4 - θ * y4_total)
    F_psi = bfield.f

    # Interpolate to uniform equal-arc grid σ_k = k/mtheta_eq
    R_eq       = zeros(mtheta_eq)
    Z_eq       = zeros(mtheta_eq)
    nu_eq      = zeros(mtheta_eq)
    theta_jac_eq = zeros(mtheta_eq)

    for k in 1:mtheta_eq
        σ = (k - 1) / mtheta_eq
        R_eq[k] = spline_R(σ)
        Z_eq[k] = spline_Z(σ)

        col4_k = spline_col4(σ)
        col5_k = spline_col5(σ)

        theta_jac_k = (col5_total > 0) ? col5_k / col5_total : σ
        nu_eq[k] = F_psi * (col4_k - theta_jac_k * col4_total)
        theta_jac_eq[k] = theta_jac_k
    end

    return R_eq, Z_eq, nu_eq, theta_jac_eq
end

