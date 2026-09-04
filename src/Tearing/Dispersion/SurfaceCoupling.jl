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
# `q_shift` is a real, additive offset on the layer's Q argument. Re(Q) is the
# lab-frame mode frequency in this surface's normalization (omega = Re(Q)/tauk),
# so `q_shift = tauk * n * Omega_k` imposes the Doppler shift of a surface
# rotating at Omega_k on an otherwise common lab-frame eigenvalue. Zero
# reproduces the static, all-surfaces-corotating case.
#
# Constructor convenience: `surface_coupling(model, params, dp_diag; dc=0.0)`
# auto-fills `scale` and `tauk` based on the model type — `scale = S^(1/3)`
# and `tauk = params.tauk` for SLAYER (de-normalizes the inner-layer Δ to
# outer units), `scale = 1` and `tauk = 1` for GGJ (Δ already in outer units
# after `rescale_delta`; no inter-surface Q rescaling).

"""
    SurfaceCoupling{M<:InnerLayerModel, P}

Per-surface dispersion data: `(model, params, dp_diag, dc, scale, tauk,
q_shift)`. Calling `sc(Q)` returns the complex residual

```
r(Q) = dp_diag - scale * solve_inner(model, params, Q + q_shift).tearing - dc
```

`q_shift` is the real Doppler offset described in the file header; it defaults
to zero, which leaves the residual identical to the pre-rotation form.

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
    Δ = solve_inner(sc.model, sc.params, ComplexF64(Q) + sc.q_shift).tearing
    return sc.dp_diag - sc.scale * Δ - sc.dc
end

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
`MultiSurfaceCoupling` Q rescaling. `q_shift` is the real Doppler offset on the
layer's Q argument (`tauk * n * Omega`); it defaults to zero.
"""
function surface_coupling(model::SLAYERModel, params::SLAYERParameters,
    dp_diag::Number; dc::Real=0.0, q_shift::Real=0.0)
    return SurfaceCoupling(model, params, ComplexF64(dp_diag),
        Float64(dc), params.lu^(1 / 3), params.tauk, Float64(q_shift))
end

"""
    surface_coupling(model::GGJModel, params::GGJParameters,
                     dp_diag::Number; q_shift::Real=0.0) -> SurfaceCoupling

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

`q_shift` is the real Doppler offset on the layer's Q argument. GGJ carries
`tauk = 1`, so the caller must supply the shift already in GGJ's own Q units.
"""
function surface_coupling(model::GGJModel, params::GGJParameters,
    dp_diag::Number; q_shift::Real=0.0)
    return SurfaceCoupling(model, params, ComplexF64(dp_diag),
        0.0, 1.0, 1.0, Float64(q_shift))
end

"""
    surface_coupling(model::InnerLayerModel, params, dp_diag::Number;
                     dc::Real=0.0, scale::Real=1.0, tauk::Real=1.0,
                     q_shift::Real=0.0) -> SurfaceCoupling

Generic fallback constructor. Use this when wiring a new inner-layer model
into the dispersion solver — pass the appropriate inner→outer-units `scale`,
per-surface `tauk`, and (for a rotating surface) `q_shift` explicitly.
"""
function surface_coupling(model::InnerLayerModel, params, dp_diag::Number;
    dc::Real=0.0, scale::Real=1.0, tauk::Real=1.0, q_shift::Real=0.0)
    return SurfaceCoupling(model, params, ComplexF64(dp_diag),
        Float64(dc), Float64(scale), Float64(tauk), Float64(q_shift))
end
