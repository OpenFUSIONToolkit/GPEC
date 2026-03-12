"""
Arc-length parameterized flux surface tracing for the direct (EFIT) equilibrium path.

Replaces the geometric-angle ODE in `DirectEquilibrium.jl` (`direct_fieldline_int`)
with a level-set ODE that integrates along the tangent to ψ = const, eliminating
both the denominator singularity and the `direct_refine` correction step.

The output format of `arclength_fieldline_int` is identical to `direct_fieldline_int`,
so the surrounding grid-construction logic in `equilibrium_solver_arclength` is a
near-verbatim copy of `equilibrium_solver`, differing only in which integration
function is called.

Select this path via `eq_type = "efit_arclength"` in `gpec.toml`.
"""

"""
    arclength_fieldline_der!(dy, y, params, s)

RHS for the arc-length-parameterized level-set ODE.

State `y = [R, Z, ∫dl/Bp, ∫dl/(R²Bp), ∫jac·dl/Bp]`.
The independent variable `s` is arc length [m].

The tangent direction `(dR/ds, dZ/ds) = (∂ψ/∂Z, −∂ψ/∂R) / |∇ψ|` follows the
ψ = const level set counterclockwise, staying on the flux surface without correction.
This eliminates the `Bz·cos(η) − Br·sin(η)` denominator of the geometric-angle ODE,
which vanishes near the top/bottom of elongated plasmas approaching the separatrix.
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

    # Level-set tangent: CCW orientation confirmed by dZ/ds > 0 at outboard midplane
    # (where psiz = 0, psir < 0, so dZ/ds = -psir/|∇ψ| > 0)
    dy[1] = psiz / grad_norm   # dR/ds
    dy[2] = -psir / grad_norm  # dZ/ds

    # Bp = |∇ψ| / R (poloidal field [T])
    Bp = grad_norm / R
    Bt = params.bfield.f / R
    B  = sqrt(Bp^2 + Bt^2)
    jac = Bp^params.power_bp * B^params.power_b / R^params.power_r

    # Surface integrals (dl = ds since s is arc length).
    # Near x-points Bp → 0 and 1/Bp diverges. Use smooth floor sqrt(Bp²+Bp_floor²)
    # for integral terms only (position dy[1:2] unaffected). This prevents true Inf
    # at exact grad_norm zeros while avoiding the C⁰ kink of max(Bp, floor).
    Bp_eff = sqrt(Bp^2 + params.Bp_floor^2)
    dy[3] = 1.0 / Bp_eff           # d/ds [∫dl/Bp]
    dy[4] = 1.0 / (R^2 * Bp_eff)  # d/ds [∫dl/(R²Bp)]
    dy[5] = jac / Bp_eff           # d/ds [∫jac·dl/Bp]
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
    # Smooth Bp floor: prevents 1/Bp → Inf at exact-zero grad_norm. Using sqrt form
    # (Bp_eff = sqrt(Bp² + Bp_floor²)) avoids the C⁰ kink of max(Bp, Bp_floor) that
    # would create artificial discontinuities in the RHS.
    Bp0 = sqrt(bfield.psir^2 + bfield.psiz^2) / r
    Bp_floor = 1e-4 * Bp0  # 0.01% of outboard-midplane Bp
    params = FieldLineDerivParams(ro, zo, raw_profile.psi_in, raw_profile.sq_in, sq_in_deriv, raw_profile.psio,
        equil_config.power_bp, equil_config.power_b, equil_config.power_r, bfield_ode, Bp_floor)

    # Guard: don't terminate until we've traversed at least half the circumference.
    # t_min is a conservative lower bound on the half-arc-length.
    t_min = π * (r - ro)

    # ContinuousCallback on Z = zo crossings:
    #   affect!     (Z crosses zo upward): terminate after minimum arc (outboard return)
    #   affect_neg! (Z crosses zo downward): do nothing (inboard midplane pass-through)
    condition(u, _t, integrator) = u[2] - zo
    function affect_upward!(integrator)
        if integrator.t > t_min
            terminate!(integrator)
        end
    end
    callback = ContinuousCallback(condition, affect_upward!, nothing; save_positions=(true, false))

    prob = ODEProblem{true}(arclength_fieldline_der!, u0, (0.0, 1.0e4), params)
    # Position components (y[1:2]) need tight tolerances; integral components (y[3:5])
    # have a 1/Bp integrand that spikes near x-points. Loose integral tolerances prevent
    # the adaptive stepper from taking millions of tiny steps through the near-x-point
    # region while keeping position accuracy for the flux surface geometry.
    reltol_vec = [equil_config.etol, equil_config.etol, 1e20, 1e20, 1e20]
    abstol_vec = [1e-8, 1e-8, 1e20, 1e20, 1e20]
    sol = solve(prob, BS5(); callback=callback, reltol=reltol_vec, abstol=abstol_vec,
        dt=2π / 200, adaptive=true, dense=false)

    # Reconstruct y_out from the ODE solution in the same 5-column format as
    # direct_fieldline_int, so the upstream grid-construction loop is unchanged.
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
    equilibrium_solver_arclength(raw_profile)

Driver for the arc-length equilibrium reconstruction. Identical to
`equilibrium_solver(::DirectRunInput)` except it calls `arclength_fieldline_int`
instead of `direct_fieldline_int`.

Select via `eq_type = "efit_arclength"` in `gpec.toml`.
"""
@with_pool pool function equilibrium_solver_arclength(raw_profile::DirectRunInput)

    equil_params = raw_profile.config
    psio = raw_profile.psio
    mtheta = equil_params.mtheta
    mpsi = equil_params.mpsi
    psilow = equil_params.psilow
    psihigh = equil_params.psihigh

    psi_nodes = Array{Float64}(undef, mpsi + 1)
    if equil_params.grid_type == "ldp"
        psi_nodes .= [psilow + (psihigh - psilow) * sin((ipsi / mpsi) * (π / 2))^2 for ipsi in 0:mpsi]
    else
        error("Unsupported grid_type: $(equil_params.grid_type)")
    end
    theta_nodes = range(0.0, 1.0; length=mtheta + 1)

    ro, zo, _rs1, rs2 = direct_position!(raw_profile)

    sq_nodes = zeros!(pool, Float64, mpsi + 1, 4)
    rzphi_nodes = zeros!(pool, Float64, mpsi + 1, mtheta + 1, 4)

    ff_val = zeros!(pool, Float64, 4)
    ff_deriv_val = zeros!(pool, Float64, 4)

    for ipsi in (mpsi+1):-1:1
        # Arc-length ODE replaces the geometric-angle ODE here
        y_out, bfield = arclength_fieldline_int(psi_nodes[ipsi], raw_profile, ro, zo, rs2)

        checkpoint!(pool, Float64)

        ff_x_nodes = acquire!(pool, Float64, size(y_out, 1))
        @. ff_x_nodes = @view(y_out[:, 5]) / y_out[end, 5]

        ff_fs_nodes = acquire!(pool, Float64, size(y_out, 1), 4)
        @. ff_fs_nodes[:, 1] = @view(y_out[:, 3]) ^ 2
        @. ff_fs_nodes[:, 2] = @view(y_out[:, 1]) / (2π) - ff_x_nodes
        @. ff_fs_nodes[:, 3] = bfield.f * (@view(y_out[:, 4]) - ff_x_nodes * y_out[end, 4])
        @. ff_fs_nodes[:, 4] = @view(y_out[:, 2]) / y_out[end, 2] - ff_x_nodes

        ff_fs_nodes[end, :] .= ff_fs_nodes[1, :]

        ff_interp = cubic_interp(ff_x_nodes, ff_fs_nodes; bc=PeriodicBC())
        ff_deriv = deriv1(ff_interp)

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

    profiles = ProfileSplines(
        psi_nodes,
        sq_nodes[:, 1],
        sq_nodes[:, 2],
        sq_nodes[:, 3],
        sq_nodes[:, 4]
    )

    q0 = profiles.q_spline.y[1] - profiles.q_deriv(psi_nodes[1]; hint=Ref(1)) * psi_nodes[1]
    if equil_params.newq0 == -1
        equil_params.newq0 = -q0
    end
    if equil_params.newq0 != 0.0
        @info "efit_arclength: Revising q-profile for newq0 = $(@sprintf("%.3f", equil_params.newq0))"
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
            sq_nodes[:, 1],
            sq_nodes[:, 2],
            sq_nodes[:, 3],
            sq_nodes[:, 4]
        )
    end

    rzphi_xs = psi_nodes
    rzphi_ys = collect(theta_nodes)
    grid2d = (rzphi_xs, theta_nodes)
    opts2d = (search=LinearBinary(), bc=(CubicFit(), PeriodicBC()), extrap=(ExtendExtrap(), WrapExtrap()))

    rzphi_rsquared = cubic_interp(grid2d, rzphi_nodes[:, :, 1]; opts2d...)
    rzphi_offset   = cubic_interp(grid2d, rzphi_nodes[:, :, 2]; opts2d...)
    rzphi_nu       = cubic_interp(grid2d, rzphi_nodes[:, :, 3]; opts2d...)
    rzphi_jac      = cubic_interp(grid2d, rzphi_nodes[:, :, 4]; opts2d...)

    eqfun_fs_nodes = zeros(Float64, mpsi + 1, mtheta + 1, 3)
    v = @MMatrix zeros(Float64, 2, 3)
    for ipsi in 1:(mpsi+1)
        q = profiles.q_spline.y[ipsi]
        f_val = profiles.F_spline.y[ipsi]
        for itheta in 1:(mtheta+1)
            theta_norm = theta_nodes[itheta]
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
            rfac = sqrt(max(0.0, f[1]))
            eta = 2π * (theta_norm + f[2])
            r = ro + rfac * cos(eta)
            jacfac = f[4]

            v[1, 1] = (rfac > 0) ? fx[1] / (2.0 * rfac) : 0.0
            v[1, 2] = fx[2] * 2π * rfac
            v[1, 3] = fx[3] * r
            v[2, 1] = (rfac > 0) ? fy[1] / (2.0 * rfac) : 0.0
            v[2, 2] = (1.0 + fy[2]) * 2π * rfac
            v[2, 3] = fy[3] * r
            v33 = 2π * r
            w11 = (jacfac != 0) ? (1.0 + fy[2]) * (2π)^2 * rfac * r / jacfac : 0.0
            w12 = (jacfac * rfac != 0) ? -fy[1] * π * r / (rfac * jacfac) : 0.0
            delpsi_norm = sqrt(w11^2 + w12^2)
            modB = sqrt(((2π * psio * delpsi_norm)^2 + f_val^2) / (2π * r)^2)

            eqfun_fs_nodes[ipsi, itheta, 1] = modB
            denom = jacfac * modB^2
            if abs(denom) > 1e-20
                numerator_2 = dot(v[1, :], v[2, :]) + q * v33 * v[1, 3]
                eqfun_fs_nodes[ipsi, itheta, 2] = numerator_2 / denom
                numerator_3 = v[2, 3] * v33 + q * v33^2
                eqfun_fs_nodes[ipsi, itheta, 3] = numerator_3 / denom
            else
                eqfun_fs_nodes[ipsi, itheta, 2] = 0.0
                eqfun_fs_nodes[ipsi, itheta, 3] = 0.0
            end
        end
    end

    eqfun_B       = cubic_interp(grid2d, eqfun_fs_nodes[:, :, 1]; opts2d...)
    eqfun_metric1 = cubic_interp(grid2d, eqfun_fs_nodes[:, :, 2]; opts2d...)
    eqfun_metric2 = cubic_interp(grid2d, eqfun_fs_nodes[:, :, 3]; opts2d...)

    return PlasmaEquilibrium(raw_profile.config, EquilibriumParameters(), profiles,
        rzphi_xs, rzphi_ys,
        rzphi_rsquared, rzphi_offset, rzphi_nu, rzphi_jac,
        eqfun_B, eqfun_metric1, eqfun_metric2,
        ro, zo, psio)
end
