# SurfaceCoupling.jl
#
# `SurfaceCoupling` packages everything the dispersion solver needs at one
# rational surface: the inner-layer model, its parameters, the outer Δ'
# diagonal element, the critical-Δ offset, the inner→outer-units scale
# factor, and the per-surface time normalization `tauk`. The struct is
# `Q`-callable and returns the complex residual
#
#   r(Q) = Δ'_diag - scale · Δ_inner(Q) - Δ_crit
#
# `tauk` is unused for single-surface evaluation but is required by the
# multi-surface `MultiSurfaceCoupling` to rescale Q between each surface's
# normalization (Fortran growthrates.f:246).
#
# Constructor convenience: `surface_coupling(model, params, dp_diag; dc=0.0)`
# auto-fills `scale` and `tauk` based on the model type — `scale = S^(1/3)`
# and `tauk = params.tauk` for SLAYER (Fortran de-normalization at
# growthrates.f:217-218,260), `scale = 1` and `tauk = 1` for GGJ (Δ already
# in outer units after `rescale_delta`; no inter-surface Q rescaling).

"""
    SurfaceCoupling{M<:InnerLayerModel, P}

Per-surface dispersion data: `(model, params, dp_diag, dc, scale, tauk)`.
Calling `sc(Q)` returns the complex residual

```
r(Q) = dp_diag - scale * solve_inner(model, params, Q)[1] - dc
```

A root of `sc` in the complex `Q` plane is a tearing eigenvalue at this
surface in the *uncoupled* approximation. Coupled multi-surface
eigenvalues come from `MultiSurfaceCoupling` evaluating the determinant
of the modified Δ' matrix.
"""
struct SurfaceCoupling{M<:InnerLayerModel, P}
    model::M
    params::P
    dp_diag::ComplexF64
    dc::Float64
    scale::Float64
    tauk::Float64
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
subtraction from the Δ' diagonal. `tauk` is taken from `params.tauk` for use
by `MultiSurfaceCoupling` Q rescaling.
"""
function surface_coupling(model::SLAYERModel, params::SLAYERParameters,
                          dp_diag::Number; dc::Real=0.0)
    return SurfaceCoupling(model, params, ComplexF64(dp_diag),
                           Float64(dc), params.lu^(1/3), params.tauk)
end

"""
    surface_coupling(model::GGJModel, params::GGJParameters,
                     dp_diag::Number; dc::Real=0.0) -> SurfaceCoupling

GGJ convenience constructor. `scale` is `1.0` because GGJ's `solve_inner`
applies its own `rescale_delta` (S^(2p₁/3)·v1^(2p₁)) internally, so the
returned Δ is already in outer units. `tauk` defaults to `1.0` (GGJ has no
direct analogue of SLAYER's per-surface time normalization, so multi-surface
Q rescaling is a no-op for GGJ surfaces unless overridden).
"""
function surface_coupling(model::GGJModel, params::GGJParameters,
                          dp_diag::Number; dc::Real=0.0)
    return SurfaceCoupling(model, params, ComplexF64(dp_diag),
                           Float64(dc), 1.0, 1.0)
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
