"""
    rzphi_primitives(equil, psi, theta; hint=(Ref(1), Ref(1))) -> (; r2, deta, jac, r2_y, deta_y)

Evaluate the smooth `rzphi` primitives at `(psi, theta)`: `r²`, the `η`-offset, the Jacobian and the
θ-derivatives of `r²` and the offset; `metric_from_primitives` turns them into the metric.
ψ-derivatives are not provided because the bicubic is only C¹ in ψ; differentiate a ψ-resample instead.
"""
function rzphi_primitives(equil::PlasmaEquilibrium, psi::Float64, theta::Float64; hint=(Ref(1), Ref(1)))
    pt = (psi, theta)
    r2 = equil.rzphi_rsquared(pt; hint=hint)
    deta = equil.rzphi_offset(pt; hint=hint)
    jac = equil.rzphi_jac(pt; hint=hint)
    r2_y = equil.rzphi_rsquared(pt; deriv=DerivOp(0, 1), hint=hint)
    deta_y = equil.rzphi_offset(pt; deriv=DerivOp(0, 1), hint=hint)
    return (; r2, deta, jac, r2_y, deta_y)
end

"""
    metric_from_primitives(ro, theta, p) -> (; rfac, r, jac, delpsi)

The gpout.f flux-surface metric at one point from the `rzphi` primitives `p` (the fields of
`rzphi_primitives`): minor radius `rfac = √r²`, major radius `r`, Jacobian `jac` and
`delpsi = |∇ψ|`. The primitives may come from evaluating the bicubic or from its nodal values.
"""
function metric_from_primitives(ro::Float64, theta::Float64, p)
    rfac = sqrt(abs(p.r2))
    r = ro + rfac * cos(2π * (theta + p.deta))
    w11 = (1.0 + p.deta_y) * (2π)^2 * rfac * r / p.jac
    w12 = rfac > 0 ? -p.r2_y * π * r / (rfac * p.jac) : 0.0
    return (; rfac, r, jac=p.jac, delpsi=sqrt(w11^2 + w12^2))
end

"""
    flux_surface_metric(equil, psi, theta; hint=(Ref(1), Ref(1))) -> (; r, jac, delpsi)

Evaluate the flux-surface geometry at `(psi, theta)`: major radius `r`, Jacobian `jac`, and the
flux-gradient magnitude `delpsi = |∇ψ|`, by `metric_from_primitives` on the bicubic at that point.
"""
function flux_surface_metric(equil::PlasmaEquilibrium, psi::Float64, theta::Float64; hint=(Ref(1), Ref(1)))
    (; r, jac, delpsi) = metric_from_primitives(equil.ro, theta, rzphi_primitives(equil, psi, theta; hint=hint))
    return (; r, jac, delpsi)
end

"""
    flux_surface_area(equil, psi, mtheta) -> Float64

Compute ⟨J·|∇ψ|⟩_θ on the flux surface at `psi`: the periodic trapezoidal rule on `mtheta` equally
spaced θ points (θ = 1 duplicates θ = 0).
"""
function flux_surface_area(equil::PlasmaEquilibrium, psi::Float64, mtheta::Int)
    hint = (Ref(1), Ref(1))
    area = 0.0
    for itheta in 0:(mtheta-1)
        m = flux_surface_metric(equil, psi, itheta / mtheta; hint=hint)
        area += m.jac * m.delpsi
    end
    return area / mtheta
end
