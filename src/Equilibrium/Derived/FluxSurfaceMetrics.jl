"""
    flux_surface_metric(equil, psi, theta; hint=(Ref(1), Ref(1))) -> (; r, jac, delpsi)

Evaluate the flux-surface geometry at `(psi, theta)`: major radius `r`, Jacobian `jac`, and the
flux-gradient magnitude `delpsi = |∇ψ|`. The metric components `w11`/`w12` follow GPEC's gpout.f
convention for the rzphi straight-field-line coordinates. This is the single source of truth for the
`|∇ψ|` computation reused across the FFS and PE modules.
"""
function flux_surface_metric(equil::PlasmaEquilibrium, psi::Float64, theta::Float64; hint=(Ref(1), Ref(1)))
    r2 = equil.rzphi_rsquared((psi, theta); hint=hint)
    jac = equil.rzphi_jac((psi, theta); hint=hint)
    deta = equil.rzphi_offset((psi, theta); hint=hint)
    r2_y = equil.rzphi_rsquared((psi, theta); deriv=DerivOp(0, 1), hint=hint)
    deta_y = equil.rzphi_offset((psi, theta); deriv=DerivOp(0, 1), hint=hint)

    rfac = sqrt(abs(r2))
    eta = 2π * (theta + deta)
    r = equil.ro + rfac * cos(eta)

    # Metric components for |∇ψ| in flux coordinates (GPEC gpout.f)
    w11 = (1.0 + deta_y) * (2π)^2 * rfac * r / jac
    w12 = -r2_y * π * r / (rfac * jac)
    delpsi = sqrt(w11^2 + w12^2)

    return (; r, jac, delpsi)
end

"""
    flux_surface_area(equil, psi, mtheta) -> Float64

Compute ∫ J·|∇ψ| dθ over the flux surface at `psi` using `mtheta` equally-spaced θ points and the
trapezoidal rule with periodic end correction.
"""
function flux_surface_area(equil::PlasmaEquilibrium, psi::Float64, mtheta::Int)
    area = 0.0
    last_contrib = 0.0

    hint = (Ref(1), Ref(1))  # Shared 2D hint for hot loop optimization
    for itheta in 0:mtheta
        theta = itheta / mtheta
        m = flux_surface_metric(equil, psi, theta; hint=hint)
        contrib = m.jac * m.delpsi / mtheta
        area += contrib

        if itheta == mtheta
            last_contrib = contrib
        end
    end

    # Trapezoidal rule end correction
    return area - last_contrib
end
