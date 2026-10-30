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
# normalization.
#
# Constructor convenience: `surface_coupling(model, params, dp_diag; dc=0.0)`
# auto-fills `scale` and `tauk` based on the model type — `scale = S^(1/3)`
# and `tauk = params.tauk` for SLAYER (de-normalizes the inner-layer Δ to
# outer units), `scale = 1` and `tauk = 1` for GGJ (Δ already in outer units
# after `rescale_delta`; no inter-surface Q rescaling).

"""
    SurfaceCoupling{M<:InnerLayerModel, P}

Per-surface dispersion data: `(model, params, dp_diag, dc, scale, tauk, q_shift)`. Calling `sc(Q)` returns the complex residual

```
r(Q) = dp_diag - scale * layer_response(model, params, Q + q_shift).tearing - dc
```

`q_shift` is the surface's E×B Doppler shift, a real frequency offset in the layer's own
frequency variable, zero by default. SLAYER's argument is the normalized Q, so the shift is added
to Q; GGJ's argument is the growth rate s = γ + iω, so the shift is added to ω and enters as
`s + i·q_shift`. The Tearing runner sets it for SLAYER surfaces.

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

SLAYER convenience constructor. `scale` is set to `params.lu^(1/3)`, which
maps the dimensionless inner-layer Δ from `riccati_f` to the r_s-referenced
outer convention (S^(1/3) with S = τ_R/τ_H built on r_s; the Lundquist
number, `dc`, and the layer all share the `x̂ = (r−r_s)/r_s` reference
length). `dp_diag` must already be in that same r_s reference — the Tearing
runner converts the ψ_N-referenced BVP Δ' via `delta_prime_to_rs_reference`
before building couplings. `tauk` is taken from `params.tauk` for use by
`MultiSurfaceCoupling` Q rescaling.
"""
function surface_coupling(model::SLAYERModel, params::SLAYERParameters,
    dp_diag::Number; dc::Real=0.0, q_shift::Real=0.0)
    return SurfaceCoupling(model, params, ComplexF64(dp_diag),
        Float64(dc), params.lu^(1 / 3), params.tauk, Float64(q_shift))
end

"""
    surface_coupling(model::GGJModel, params::GGJParameters,
                     dp_diag::Number; omega_E::Real=0.0, n::Integer=0) -> SurfaceCoupling

GGJ convenience constructor. `scale` is `1.0` because GGJ's `solve_inner`
applies its own `rescale_delta` (S^(2p₁/3)·v1^(2p₁)) internally, so the
returned Δ is already in outer units. `tauk` defaults to `1.0` (GGJ has no
direct analogue of SLAYER's per-surface time normalization, so multi-surface
Q rescaling is a no-op for GGJ surfaces unless overridden).

`omega_E` is the surface's E×B rotation per unit toroidal mode number [rad/s] and `n` the
toroidal mode number. GGJ is a plasma-frame layer: rotation enters only beyond the order its
equations retain (Johnson & Greene 1967, Plasma Phys. 9, 611, Appendix B), so a rotating
surface sees the lab-frame growth rate Doppler-shifted, `s_k = s + i·n·omega_E`, and
`q_shift = n·omega_E`. With rotation set on every surface, the scanned `s` is in the lab
frame; differential rotation between coupled surfaces changes the coupled γ, while a common
rotation only shifts ω.

**No `dc` kwarg**: GGJ's 4m×4m Pletzer-Dewar residual already includes the
interchange channel, which provides Glasser (Mercier) stabilization
natively. A Δ_crit proxy (χ_parallel-matching offset on the diagonal) is
meaningful only for tearing-only slab-layer approximations like SLAYER;
for GGJ it would double-count the interchange physics. The `SurfaceCoupling`
struct's `dc` field is hard-wired to 0 here.
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
