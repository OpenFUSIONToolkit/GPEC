# TorqueBalance.jl
#
# `TorqueBalance` solves the torque balance eqution for each rational surface.
# It follows the derivation found in Cole PopP 2006. For simplicity, the script
# uses equation 62 of Cole. This can be improved by incorporating the true
# outer-layer Δ' from the perturbed equilibrium and then solving equation 61.
#
# The viscous torque is:
#
#   T_v = 2 · P (Q_0 - Q) / ((S kappa^hat) · (b_r(r_s)/B_phi))^2
#
# The electromagnetic torque is:
#
#   T_em = Im[Δ_inner(Q)] / |alpha + Δ_inner(Q)|^2
#
# Now solving for critical resonant field, we have:
#
#   (b_r(r_s)/B_phi))^2_crit = max(2 · P (Q_0 - Q) / ((S kappa^hat) · Im[-Δ_inner(Q)^-1])
#
# Normalizations, definitions, and other conventions are taken from the Cole paper.
# The critical resonant field is found at each rational surface. For each, the script
# scans over a range of Q values and finds the maximum value of the right-hand side of
# the equation above.

"""
    TorqueBalance{M<:InnerLayerModel, P}

Per-surface dispersion data: `(model, params, dp_diag, dc, scale, tauk)`.
Calling `sc(Q)` returns the complex residual

```
r(Q) = dp_diag - scale * solve_inner(model, params, Q).tearing - dc
```

A root of `sc` in the complex `Q` plane is a **tearing** eigenvalue at
this surface in the *uncoupled* approximation (only the tearing channel
of the inner-layer response appears — the interchange channel enters the
full 2m×2m dispersion via `MultiSurfaceCoupling`, not this scalar form).
Coupled multi-surface eigenvalues come from `MultiSurfaceCoupling`
evaluating the determinant of the modified Δ' matrix.
"""
struct SurfaceCoupling{M<:InnerLayerModel,P}
    model::M
    params::P
    dp_diag::ComplexF64
    dc::Float64
    scale::Float64
    tauk::Float64
end

function (sc::SurfaceCoupling)(Q::Number)
    Δ = solve_inner(sc.model, sc.params, ComplexF64(Q)).tearing
    return sc.dp_diag - sc.scale * Δ - sc.dc
end

"""
    surface_coupling(model::SLAYERModel, params::SLAYERParameters,
                     dp_diag::Number; dc::Real=0.0) -> SurfaceCoupling

SLAYER convenience constructor. `scale` is set to `params.lu^(1/3)` so that
the dimensionless Δ from `riccati_f` is mapped to outer ψ-units before
subtraction from the Δ' diagonal. `tauk` is taken from `params.tauk` for use
by `MultiSurfaceCoupling` Q rescaling.
"""
function surface_coupling(model::SLAYERModel, params::SLAYERParameters,
    dp_diag::Number; dc::Real=0.0)
    return SurfaceCoupling(model, params, ComplexF64(dp_diag),
        Float64(dc), params.lu^(1 / 3), params.tauk)
end

"""
    surface_coupling(model::GGJModel, params::GGJParameters,
                     dp_diag::Number) -> SurfaceCoupling

GGJ convenience constructor. `scale` is `1.0` because GGJ's `solve_inner`
applies its own `rescale_delta` (S^(2p₁/3)·v1^(2p₁)) internally, so the
returned Δ is already in outer units. `tauk` defaults to `1.0` (GGJ has no
direct analogue of SLAYER's per-surface time normalization, so multi-surface
Q rescaling is a no-op for GGJ surfaces unless overridden).

**No `dc` kwarg**: GGJ's 4m×4m Pletzer-Dewar residual already includes the
interchange channel, which provides Glasser (Mercier) stabilization
natively. A Δ_crit proxy (χ_parallel-matching offset on the diagonal) is
meaningful only for tearing-only slab-layer approximations like SLAYER;
for GGJ it would double-count the interchange physics. The `SurfaceCoupling`
struct's `dc` field is hard-wired to 0 here.
"""
function surface_coupling(model::GGJModel, params::GGJParameters,
    dp_diag::Number)
    return SurfaceCoupling(model, params, ComplexF64(dp_diag),
        0.0, 1.0, 1.0)
end

"""
    surface_coupling(model::InnerLayerModel, params, dp_diag::Number;
                     dc::Real=0.0, scale::Real=1.0, tauk::Real=1.0)
        -> SurfaceCoupling

Generic fallback constructor. Use this when wiring a new inner-layer model
into the dispersion solver — pass the appropriate inner→outer-units `scale`
and per-surface `tauk` explicitly.
"""
function surface_coupling(model::InnerLayerModel, params, dp_diag::Number;
    dc::Real=0.0, scale::Real=1.0, tauk::Real=1.0)
    return SurfaceCoupling(model, params, ComplexF64(dp_diag),
        Float64(dc), Float64(scale), Float64(tauk))
end

function torque_balance_value(tb::TorqueBalance, Q::Number)
    Δ = solve_inner(tb.model, tb.params, ComplexF64(Q)).tearing
    jxb = -imag(1.0 / (Δ + tb.delta_n_p))
    return 2.0 * tb.P * (tb.Q0 - Q) / jxb
end

function scan_torque_balance(tb; Qmin=-10.0, Qmax=10.0, nscan=200)
    qs = range(Qmin, Qmax; length=nscan)
    vals = torque_balance_value.(Ref(tb), qs)
    idx = argmax(vals)
    return qs[idx], vals[idx]
end

br_th = sqrt(maxbal / tb.lu * (tb.sval^2 / 2.0))
