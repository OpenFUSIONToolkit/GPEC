# SurfaceCoupling.jl
#
# Per-surface dispersion residual `SurfaceCoupling` and its `surface_coupling` constructors.

"""
    SurfaceCoupling{M<:InnerLayerModel, P}

Per-surface dispersion data: `(model, params, dp_diag, dc, scale, tauk, q_shift)`. Calling `sc(Q)` returns the complex residual

```
r(Q) = dp_diag - scale * layer_response(model, params, Q + q_shift).tearing - dc
```

`q_shift` is the surface's E×B Doppler shift, a real frequency offset in the layer's own
frequency variable, zero by default. SLAYER's argument is the normalized Q, so the shift is added
to Q; GGJ's argument is the growth rate s = γ + iω, so the shift is added to ω and enters as
`s + i·q_shift`. The SLAYER and coupled GGJ Tearing runs set it.

A root of `sc` in the complex `Q` plane is an uncoupled tearing eigenvalue at this surface (tearing channel only);
coupled eigenvalues, including the interchange channel, come from `MultiSurfaceCoupling`.
"""
struct SurfaceCoupling{M<:InnerLayerModel,P}
    model::M
    params::P
    dp_diag::ComplexF64
    dc::Float64
    scale::Float64
    tauk::Float64
    q_shift::Float64
end

function (sc::SurfaceCoupling)(Q::Number)
    Δ = layer_response(sc.model, sc.params, layer_argument(sc, Q)).tearing
    return sc.dp_diag - sc.scale * Δ - sc.dc
end

# Doppler-shift a scanned value into the layer's own frequency variable: real for SLAYER's Q, imaginary for GGJ's s.
layer_argument(sc::SurfaceCoupling, Q::Number) = ComplexF64(Q) + sc.q_shift
layer_argument(sc::SurfaceCoupling{<:GGJModel}, s::Number) = ComplexF64(s) + im * sc.q_shift

"""
    surface_coupling(model::SLAYERModel, params::SLAYERParameters,
                     dp_diag::Number; dc::Real=0.0, q_shift::Real=0.0)
        -> SurfaceCoupling

SLAYER convenience constructor. Sets `scale = params.lu^(1/3)` (S^(1/3), mapping the normalized inner-layer Δ to the
r_s-referenced outer convention) and `tauk = params.tauk`. `dp_diag` must already be r_s-referenced (see
`delta_prime_to_rs_reference`).
"""
function surface_coupling(model::SLAYERModel, params::SLAYERParameters,
    dp_diag::Number; dc::Real=0.0, q_shift::Real=0.0)
    return SurfaceCoupling(model, params, ComplexF64(dp_diag),
        Float64(dc), params.lu^(1 / 3), params.tauk, Float64(q_shift))
end

"""
    surface_coupling(model::GGJModel, params::GGJParameters,
                     dp_diag::Number; omega_E::Real=0.0, n::Integer=0) -> SurfaceCoupling

GGJ convenience constructor. `scale = 1` (GGJ's `rescale_delta` already returns Δ in outer units) and `tauk = 1`.

`omega_E` is the surface's E×B rotation per unit toroidal mode number, in rad/s, and `n` the toroidal mode number
(required when `omega_E` is nonzero). GGJ is a plasma-frame layer (Johnson & Greene 1967, Plasma Phys. 9, 611,
Appendix B), so a rotating surface sees the Doppler-shifted growth rate `s + i·n·omega_E`, and `q_shift = n·omega_E`.

There is no `dc` keyword: the GGJ residual already includes the interchange channel, so a Δ_crit offset would
double-count it; `dc` is 0.
"""
function surface_coupling(model::GGJModel, params::GGJParameters, dp_diag::Number; omega_E::Real=0.0, n::Integer=0)
    iszero(omega_E) || n >= 1 ||
        throw(ArgumentError("surface_coupling: E×B rotation omega_E=$(omega_E) rad/s needs the toroidal mode number n ≥ 1; got n=$(n)"))
    return SurfaceCoupling(model, params, ComplexF64(dp_diag),
        0.0, 1.0, 1.0, Float64(n * omega_E))
end

"""
    surface_coupling(model::InnerLayerModel, params, dp_diag::Number;
                     dc::Real=0.0, scale::Real=1.0, tauk::Real=1.0,
                     q_shift::Real=0.0) -> SurfaceCoupling

Generic fallback constructor. Use this when wiring a new inner-layer model
into the dispersion solver — pass the appropriate inner→outer-units `scale`
and per-surface `tauk` explicitly.
"""
function surface_coupling(model::InnerLayerModel, params, dp_diag::Number;
    dc::Real=0.0, scale::Real=1.0, tauk::Real=1.0, q_shift::Real=0.0)
    return SurfaceCoupling(model, params, ComplexF64(dp_diag),
        Float64(dc), Float64(scale), Float64(tauk), Float64(q_shift))
end
