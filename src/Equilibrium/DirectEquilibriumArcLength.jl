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

State `y = [R, Z, ∫dl/Bp, ∫dl/(R²Bp), arc_length]`.
The independent variable `s` is arc length [m].

The tangent direction `(dR/ds, dZ/ds) = (∂ψ/∂Z, −∂ψ/∂R) / |∇ψ|` follows the
ψ = const level set counterclockwise, staying on the flux surface without correction.

dy[5] = 1 always (equal_arc: Bp/Bp). Near x-points where Bp → 0, the position
integration (dy[1:2]) remains well-behaved; dy[3:5] use abstol=1e20 so the solver
never restricts step size for them.
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

    # Equal-arc jac: dy[5] = Bp/Bp = 1 always, self-regularizing near x-point.
    # dy[3] and dy[4] use abstol=1e20 so the solver never restricts steps for them.
    Bp_eff = max(Bp, eps(Float64))  # absolute guard against exact Bp=0 only
    dy[3] = 1.0 / Bp_eff           # d/ds [∫dl/Bp]
    dy[4] = 1.0 / (R^2 * Bp_eff)  # d/ds [∫dl/(R²Bp)]
    dy[5] = 1.0                    # d/ds [arc_length] = Bp/Bp_eff → 1 for equal_arc
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
    # Force equal_arc jac internally (power_bp=1,b=0,r=0): dy[5]=1 regardless of Bp near x-point.
    # The user jac SFL angle is recovered in post-processing below.
    params = FieldLineDerivParams(ro, zo, raw_profile.psi_in, raw_profile.sq_in, sq_in_deriv, raw_profile.psio,
        1, 0, 0, 0, bfield_ode, 0.0)

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

    # Replace arc-length in y_out[:,5] with user-jac SFL angle so equilibrium_solver
    # receives the expected SFL abscissa ∫jac·dl/Bp normalized to [0,1].
    # y_out[:,2] = ∫dl/Bp and y_out[:,4] = ∫dl/(R²Bp) are already integrated correctly.
    _sfl_to_user_jac!(y_out, equil_config, sol, ro, zo,
        raw_profile.psi_in, raw_profile.sq_in, sq_in_deriv, raw_profile.psio)

    # bfield at the starting point carries F and P for the surface-averaged quantities
    return y_out, bfield
end

"""
    _sfl_to_user_jac!(y_out, equil_config, sol, ro, zo, psi_in, sq_in, sq_in_deriv, psio)

Replace the arc-length column y_out[:,5] with the user-jac SFL angle ∈ [0,1].

After the equal_arc ODE, y_out[:,5] = arc_length. This converts it to the SFL angle
for the user's jac_type using integrals already accumulated in the ODE (fast paths) or
post-processing bfield calls (general path):

Fast paths (no extra bfield calls):
- Hamada     (power_bp=0, b=0, r=0, rc=0): SFL = ∫dl/Bp,      stored in y_out[:,2]
- PEST       (power_bp=0, b=0, r=2, rc=0): SFL = ∫dl/(R²Bp),  stored in y_out[:,4]
- equal_arc  (power_bp=1, b=0, r=0, rc=0): SFL = arc_length,  in y_out[:,5]

General path (bfield calls at each ODE sample point):
- Boozer     (power_bp=0, b=2, r=0, rc=0): SFL ∝ ∫B²dl/Bp
- Park       (power_bp=0, b=1, r=0, rc=0): SFL ∝ ∫Bdl/Bp
- other: SFL ∝ ∫ Bp^(power_bp−1) ⋅ B^power_b / (R^power_r ⋅ rfac^power_rc) dl
"""
function _sfl_to_user_jac!(y_out::Matrix{Float64}, equil_config::EquilibriumConfig,
    sol, ro::Float64, zo::Float64, psi_in, sq_in, sq_in_deriv, psio::Float64)
    power_bp = equil_config.power_bp
    power_b  = equil_config.power_b
    power_r  = equil_config.power_r
    power_rc = equil_config.power_rc

    if power_bp == 0 && power_b == 0 && power_r == 0 && power_rc == 0
        # Hamada: ∫dl/Bp already in col 2
        @. y_out[:, 5] = y_out[:, 2] / y_out[end, 2]
    elseif power_bp == 0 && power_b == 0 && power_r == 2 && power_rc == 0
        # PEST: ∫dl/(R²Bp) already in col 4
        @. y_out[:, 5] = y_out[:, 4] / y_out[end, 4]
    elseif power_bp == 1 && power_b == 0 && power_r == 0 && power_rc == 0
        # equal_arc: arc_length → normalize in place
        y_out[:, 5] ./= y_out[end, 5]
    else
        # General case: integrand Bp^(power_bp−1) ⋅ B^power_b / (R^power_r ⋅ rfac^power_rc)
        # Requires bfield at each ODE sample point; trapezoid integration over arc length.
        _general_jac_from_trajectory!(y_out, sol, ro, zo,
            power_bp, power_b, power_r, power_rc, psi_in, sq_in, sq_in_deriv, psio)
    end
end

"""
    _general_jac_from_trajectory!(y_out, sol, ro, zo, power_bp, power_b, power_r, power_rc,
                                   psi_in, sq_in, sq_in_deriv, psio)

Compute the general SFL angle by trapezoid integration over the ODE trajectory.

Integrand at each point: `Bp^(power_bp−1) ⋅ B^power_b / (R^power_r ⋅ rfac^power_rc)`
where `B = √(Bp² + Bt²)`, `Bt = F/R`, and `rfac = √((R−R₀)²+(Z−Z₀)²)`.
The result is normalized to [0,1] and stored in `y_out[:,5]`.
"""
function _general_jac_from_trajectory!(y_out::Matrix{Float64}, sol, ro::Float64, zo::Float64,
    power_bp::Int, power_b::Int, power_r::Int, power_rc::Int,
    psi_in, sq_in, sq_in_deriv, psio::Float64)
    n = length(sol.u)
    bfield = DirectBField()

    prev_integrand = _jac_integrand(sol.u[1][1], sol.u[1][2], ro, zo,
        power_bp, power_b, power_r, power_rc, bfield, psi_in, sq_in, sq_in_deriv, psio)
    y_out[1, 5] = 0.0

    for i in 2:n
        R, Z = sol.u[i][1], sol.u[i][2]
        curr_integrand = _jac_integrand(R, Z, ro, zo,
            power_bp, power_b, power_r, power_rc, bfield, psi_in, sq_in, sq_in_deriv, psio)
        ds = sol.t[i] - sol.t[i-1]
        y_out[i, 5] = y_out[i-1, 5] + 0.5 * (prev_integrand + curr_integrand) * ds
        prev_integrand = curr_integrand
    end

    total = y_out[end, 5]
    if total > 0.0
        y_out[:, 5] ./= total
    end
end

"""
    _jac_integrand(R, Z, ro, zo, power_bp, power_b, power_r, power_rc,
                   bfield, psi_in, sq_in, sq_in_deriv, psio)

Evaluate the SFL-angle integrand `Bp^(power_bp−1) ⋅ B^power_b / (R^power_r ⋅ rfac^power_rc)` at (R, Z).
"""
function _jac_integrand(R::Float64, Z::Float64, ro::Float64, zo::Float64,
    power_bp::Int, power_b::Int, power_r::Int, power_rc::Int,
    bfield::DirectBField, psi_in, sq_in, sq_in_deriv, psio::Float64)::Float64
    direct_get_bfield!(bfield, R, Z, psi_in, sq_in, sq_in_deriv, psio; derivs=1)
    Bp = sqrt(bfield.br^2 + bfield.bz^2)
    Bt = bfield.f / R
    B  = sqrt(Bp^2 + Bt^2)
    rfac = sqrt((R - ro)^2 + (Z - zo)^2)
    Bp_eff   = max(Bp,   eps(Float64))
    rfac_eff = max(rfac, eps(Float64))
    return Bp_eff^(power_bp - 1) * B^power_b / (R^power_r * rfac_eff^power_rc)
end
