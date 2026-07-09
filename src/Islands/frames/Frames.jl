"""
    Frames

THE frequency/frame conversion module (design `01 §5`, module CLAUDE.md): no
other part of Islands may contain an ω sign convention. The polarization-current
sign disputes in the literature are largely frame disputes; this module owns the
conversions so they cannot be reproduced inconsistently elsewhere.

**Milestone-M2 status: forms only, signs gated.** The frame identities of
`01 §5` are `[CHECKED: Diss19 pp. 46–48]` but not human-cleared (QUESTIONS Q3),
so every sign/normalization here is a `FrameConvention` field with a **NaN
default**: using an un-cleared convention poisons every downstream number
instead of silently committing to a guess. Only the mechanical, sign-free
bookkeeping (`frame_shift`, round-trips, parameter validation) is active.
"""
module Frames

export Level0Parameters, FrameConvention
export frame_shift, omega_dia_form, effective_dlnn_form, is_cleared

"""
    Level0Parameters(; w_hat, omega_E_hat, epsilon, inv_Lq_hat, q_s, tau, nu_star)

The Level-0 input parameter vector `p` (`01 §5`). Species-resolved quantities
(gradients, `η_j`, backgrounds, roles) live on the species list; this struct
carries the scalars.

## Fields

  - `w_hat`       — island **half**-width `w/ρ_θi` (half-width convention pinned
    in the module CLAUDE.md; thresholds are always reported as half-widths).
  - `omega_E_hat` — `ω_E/ω_dia,e` (`≡ −ω₀/ω_dia,e`; a scanned input from day one,
    ordering O4).
  - `epsilon`     — inverse aspect ratio `r_s/R₀`.
  - `inv_Lq_hat`  — `L̂_q⁻¹ = (ψ_s/q) dq/dψ` at `r_s`.
  - `q_s`         — safety factor at the rational surface.
  - `tau`         — `T_e/T_i`.
  - `nu_star`     — per-species banana collisionality `ν_★j`, keyed by species name.
"""
struct Level0Parameters
    w_hat::Float64
    omega_E_hat::Float64
    epsilon::Float64
    inv_Lq_hat::Float64
    q_s::Float64
    tau::Float64
    nu_star::Dict{Symbol,Float64}
    function Level0Parameters(w_hat, omega_E_hat, epsilon, inv_Lq_hat, q_s, tau, nu_star)
        w_hat > 0 || throw(ArgumentError("w_hat must be positive (half-width)"))
        epsilon > 0 || throw(ArgumentError("epsilon must be positive"))
        q_s > 0 || throw(ArgumentError("q_s must be positive"))
        tau > 0 || throw(ArgumentError("tau must be positive"))
        all(v -> v >= 0, values(nu_star)) || throw(ArgumentError("nu_star entries must be nonnegative"))
        return new(w_hat, omega_E_hat, epsilon, inv_Lq_hat, q_s, tau, Dict{Symbol,Float64}(nu_star))
    end
end

function Level0Parameters(; w_hat, omega_E_hat, epsilon, inv_Lq_hat, q_s, tau=1.0, nu_star=Dict{Symbol,Float64}())
    return Level0Parameters(w_hat, omega_E_hat, epsilon, inv_Lq_hat, q_s, tau, nu_star)
end

"""
    FrameConvention(; C_dia=NaN, sign_omega0=NaN, C_gradient_shift=NaN)

The gated sign/normalization set of the frame identities (`01 §5`,
`[CHECKED: Diss19 pp. 46–48]`, awaiting human clearance — QUESTIONS **Q3**).
All fields default to **NaN** so an un-cleared convention poisons results
rather than silently committing to a guess; `is_cleared` tests for this.

## Fields

  - `C_dia`            — prefactor (incl. sign) of the electron diamagnetic
    frequency form `ω_dia,e = C_dia · m T̂_e L̂_n⁻¹ / q_s`.
  - `sign_omega0`      — the island-propagation relation `ω₀ = sign_omega0 · ω_E`.
  - `C_gradient_shift` — prefactor of the frame shift of the effective density
    gradient, `L_n⁻¹ = L_{n0}⁻¹ (1 + C_gradient_shift · Z_j ω_E/ω_dia,e)`.
"""
Base.@kwdef struct FrameConvention
    C_dia::Float64 = NaN
    sign_omega0::Float64 = NaN
    C_gradient_shift::Float64 = NaN
end

"""
    is_cleared(conv::FrameConvention)

`true` only when every gated field has been assigned (no NaN). Downstream code
must check this before producing physics numbers.
"""
is_cleared(conv::FrameConvention) = !(isnan(conv.C_dia) || isnan(conv.sign_omega0) || isnan(conv.C_gradient_shift))

"""
    frame_shift(omega, omega_E)

The frame-invariant combination `ω − ω_E` (`01 §5`): mechanical bookkeeping,
no sign convention (the invariance is what *defines* the shift).
"""
frame_shift(omega, omega_E) = omega - omega_E

"""
    omega_dia_form(m_mode, T_hat, inv_Ln_hat, q_s, conv)

The electron diamagnetic frequency *form* `C_dia · m T̂ L̂_n⁻¹ / q_s`
(structure of `01 §5`; value gated on `conv.C_dia`, QUESTIONS Q3). Returns NaN
until the convention is cleared.
"""
omega_dia_form(m_mode, T_hat, inv_Ln_hat, q_s, conv::FrameConvention) = conv.C_dia * m_mode * T_hat * inv_Ln_hat / q_s

"""
    effective_dlnn_form(inv_Ln0_hat, Z, omega_E_hat, conv)

The frame shift of the effective density gradient *form*
`L̂_n⁻¹ = L̂_{n0}⁻¹ (1 + C_gradient_shift · Z ω̂_E)` (structure of `01 §5`; value
gated on `conv.C_gradient_shift`, QUESTIONS Q3). `omega_E_hat` is already
normalized to `ω_dia,e`. Returns NaN until the convention is cleared.
"""
effective_dlnn_form(inv_Ln0_hat, Z, omega_E_hat, conv::FrameConvention) = inv_Ln0_hat * (1 + conv.C_gradient_shift * Z * omega_E_hat)

end # module Frames
