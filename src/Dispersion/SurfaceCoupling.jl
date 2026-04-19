# SurfaceCoupling.jl
#
# `SurfaceCoupling` packages everything the dispersion solver needs at one
# rational surface: the inner-layer model, its parameters, the outer Δ'
# diagonal element, the critical-Δ offset, and the inner→outer-units scale
# factor. The struct is `Q`-callable and returns the complex residual
#
#   r(Q) = Δ'_diag - scale · Δ_inner(Q) - Δ_crit
#
# Constructor convenience: `surface_coupling(model, params, dp_diag; dc=0.0)`
# auto-fills `scale` based on the model type — `S^(1/3)` for SLAYER (mirrors
# the Fortran `dispersion_det` de-normalization at growthrates.f:217-218,260)
# and `1` for GGJ (Δ already in outer units after `rescale_delta`). Use the
# direct constructor with an explicit `scale` keyword for new model types.

"""
    SurfaceCoupling{M<:InnerLayerModel, P}

Per-surface dispersion data: `(model, params, dp_diag, dc, scale)`. Calling
`sc(Q)` returns the complex residual

```
r(Q) = dp_diag - scale * solve_inner(model, params, Q)[1] - dc
```

A root of `sc` in the complex `Q` plane is a tearing eigenvalue at this
surface (uncoupled approximation — true coupled eigenvalues require the
multi-surface determinant in `solve_coupled`).
"""
struct SurfaceCoupling{M<:InnerLayerModel, P}
    model::M
    params::P
    dp_diag::ComplexF64
    dc::Float64
    scale::Float64
end

function (sc::SurfaceCoupling)(Q::Number)
    Δ = solve_inner(sc.model, sc.params, ComplexF64(Q))[1]
    return sc.dp_diag - sc.scale * Δ - sc.dc
end

"""
    surface_coupling(model::SLAYERModel, params::SLAYERParameters,
                     dp_diag::Number; dc::Real=0.0) -> SurfaceCoupling

SLAYER convenience constructor. `scale` is set to `params.lu^(1/3)` so that
the dimensionless Δ from `riccati_f` is mapped to outer ψ-units before
subtraction from the Δ' diagonal. `dc` defaults to `params.dc_tmp` only if
the caller explicitly opts in (see kwargs); otherwise zero, matching the
Fortran convention where `delta_eff` and `dc_tmp` are added separately.
"""
function surface_coupling(model::SLAYERModel, params::SLAYERParameters,
                          dp_diag::Number; dc::Real=0.0)
    return SurfaceCoupling(model, params, ComplexF64(dp_diag),
                           Float64(dc), params.lu^(1/3))
end

"""
    surface_coupling(model::GGJModel, params::GGJParameters,
                     dp_diag::Number; dc::Real=0.0) -> SurfaceCoupling

GGJ convenience constructor. `scale` is `1.0` because GGJ's `solve_inner`
applies its own `rescale_delta` (S^(2p₁/3)·v1^(2p₁)) internally, so the
returned Δ is already in outer units.
"""
function surface_coupling(model::GGJModel, params::GGJParameters,
                          dp_diag::Number; dc::Real=0.0)
    return SurfaceCoupling(model, params, ComplexF64(dp_diag),
                           Float64(dc), 1.0)
end

"""
    surface_coupling(model::InnerLayerModel, params, dp_diag::Number;
                     dc::Real=0.0, scale::Real=1.0) -> SurfaceCoupling

Generic fallback constructor. Use this when wiring a new inner-layer model
into the dispersion solver — pass the appropriate inner→outer-units `scale`
explicitly.
"""
function surface_coupling(model::InnerLayerModel, params, dp_diag::Number;
                          dc::Real=0.0, scale::Real=1.0)
    return SurfaceCoupling(model, params, ComplexF64(dp_diag),
                           Float64(dc), Float64(scale))
end
