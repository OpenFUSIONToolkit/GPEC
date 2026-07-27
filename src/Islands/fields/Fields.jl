"""
    Fields

The Level-0 field-equation layer (design `01 §3`, `03 §1`): quasineutrality is
the only field equation at Level 0 (Ampère arrives at Level 3). The residual
term itself lives in `Operators.Quasineutrality`; this module provides the
closure *structure* around it:

  - the island-geometry functions `Q(Ω)`, `h(Ω)` of the flattened-electron
    (WCHH96-class) closure — implemented as **structure with a supplied
    prefactor** (the physics prefactor `w_ψ/2√2` is `[CHECKED]`-uncleared,
    QUESTIONS Q3);
  - the coefficient-free consistency identity `⟨∂²h/∂x²⟩_Ω = 0` (ladder A7 —
    kokuchou's unit set, which caught inherited DK-NTM bugs);
  - [`ElectronClosure`](@ref) — the gated constant set of the closure, NaN-
    poisoned until human-cleared.
"""
module Fields

import QuadGK

export Q_omega, dQ_domega, h_profile, dh_domega, d2h_domega2
export flat_average_d2h_dx2, ElectronClosure, is_cleared

"""
    Q_omega(Ω; rtol=1e-10)

`Q(Ω) = (1/2π) ∮ √(Ω + cos ξ) dξ` over the region where the radicand is
positive (`01 §2.4` structure). For `Ω > 1` the integral covers the full
period; for `|Ω| < 1` it runs between the turning points `cos ξ_b = −Ω`.
"""
function Q_omega(Ω; rtol::Real=1e-10)
    if Ω > 1
        val, _ = QuadGK.quadgk(ξ -> sqrt(Ω + cos(ξ)), 0.0, 2π; rtol=rtol)
        return val / (2π)
    elseif -1 < Ω <= 1
        ξb = acos(-Ω)
        val, _ = QuadGK.quadgk(ξ -> sqrt(max(Ω + cos(ξ), 0.0)), -ξb, ξb; rtol=rtol)
        return val / (2π)
    else
        throw(DomainError(Ω, "Q_omega needs Ω > −1"))
    end
end

"""
    dQ_domega(Ω; rtol=1e-10)

`dQ/dΩ = (1/4π) ∮ (Ω + cos ξ)^{−1/2} dξ` (differentiating `Q_omega` under the
integral; the inside-separatrix endpoint singularity is integrable).
"""
function dQ_domega(Ω; rtol::Real=1e-10)
    if Ω > 1
        val, _ = QuadGK.quadgk(ξ -> 1.0 / sqrt(Ω + cos(ξ)), 0.0, 2π; rtol=rtol)
        return val / (4π)
    elseif -1 < Ω < 1
        ξb = acos(-Ω)
        val, _ = QuadGK.quadgk(ξ -> 1.0 / sqrt(Ω + cos(ξ)), -ξb, ξb; rtol=rtol)
        return val / (4π)
    else
        throw(DomainError(Ω, "dQ_domega needs Ω ∈ (−1, 1) ∪ (1, ∞)"))
    end
end

"""
    h_profile(Ω; prefactor, rtol=1e-10)

The flattened-electron profile function *structure* `h(Ω) = Θ(Ω − 1) · prefactor · ∫₁^Ω dΩ′/Q(Ω′)` (`01 §2.4`): exactly flat inside the separatrix,
`→ x` far outside. The physics prefactor (`w_ψ/2√2` in the sources) is
`[CHECKED]`-uncleared (QUESTIONS Q3) and therefore **supplied**; the A7
identity below is prefactor-independent.
"""
function h_profile(Ω; prefactor, rtol::Real=1e-10)
    Ω <= 1 && return zero(float(Ω))
    val, _ = QuadGK.quadgk(Ωp -> 1.0 / Q_omega(Ωp; rtol=rtol), 1.0, Ω; rtol=rtol)
    return prefactor * val
end

"""
    dh_domega(Ω; prefactor, rtol=1e-10)

`h′(Ω) = prefactor / Q(Ω)` for `Ω > 1`, zero inside.
"""
dh_domega(Ω; prefactor, rtol::Real=1e-10) = Ω > 1 ? prefactor / Q_omega(Ω; rtol=rtol) : zero(float(Ω))

"""
    d2h_domega2(Ω; prefactor, rtol=1e-10)

`h″(Ω) = −prefactor · Q′(Ω)/Q(Ω)²` for `Ω > 1`, zero inside.
"""
function d2h_domega2(Ω; prefactor, rtol::Real=1e-10)
    Ω <= 1 && return zero(float(Ω))
    Q = Q_omega(Ω; rtol=rtol)
    return -prefactor * dQ_domega(Ω; rtol=rtol) / Q^2
end

"""
    flat_average_d2h_dx2(Ω, w; prefactor=1.0, rtol=1e-10)

The ladder-A7 identity integrand: `⟨∂²h/∂x²⟩_Ω` assembled from the chain rule
`∂²h/∂x² = h″(Ω)(∂Ω/∂x)² + h′(Ω) ∂²Ω/∂x²` with `∂Ω/∂x = 4x/w²` on the surface
(`(∂Ω/∂x)² = 8(Ω + cos ξ)/w²`), averaged with the `(Ω + cos ξ)^{−1/2}` weight.
Analytically **exactly zero** for any prefactor — a coefficient-free
consistency check of the `Q`, `h` quadratures and the `⟨·⟩_Ω` machinery
(L23 Eq. 4.1.1-class unit target). Valid for `Ω > 1`.
"""
function flat_average_d2h_dx2(Ω, w; prefactor::Real=1.0, rtol::Real=1e-10)
    Ω > 1 || throw(DomainError(Ω, "the A7 identity applies outside the separatrix (Ω > 1)"))
    hp = dh_domega(Ω; prefactor=prefactor, rtol=rtol)
    hpp = d2h_domega2(Ω; prefactor=prefactor, rtol=rtol)
    num, _ = QuadGK.quadgk(ξ -> (hpp * 8 * (Ω + cos(ξ)) / w^2 + hp * 4 / w^2) / sqrt(Ω + cos(ξ)), 0.0, 2π; rtol=rtol)
    den, _ = QuadGK.quadgk(ξ -> 1.0 / sqrt(Ω + cos(ξ)), 0.0, 2π; rtol=rtol)
    return num / den
end

"""
    ElectronClosure(; k_HS=NaN, f_p=NaN, h_prefactor=NaN, C_phi=NaN)

The gated constant set of the Level-0 flattened-electron closure (`01 §2.4`,
`§3`), all `[CHECKED]`-uncleared (QUESTIONS Q3) and **NaN-defaulted** so an
un-cleared closure poisons results instead of committing to a guess:

## Fields

  - `k_HS`        — the Hirshman–Sigmar flow coefficient (sources: `≃ −1.173`).
  - `f_p`         — the passing fraction (**cleared** 2026-07-11:
    `Coefficients.passing_fraction(ε) = 1 − 1.4624√ε`; may populate this field).
  - `h_prefactor` — the `h(Ω)` amplitude (sources: `w_ψ/2√2`).
  - `C_phi`       — the quasineutrality closure coefficient (sources: `1/2L̂_{n0}`).
"""
Base.@kwdef struct ElectronClosure
    k_HS::Float64 = NaN
    f_p::Float64 = NaN
    h_prefactor::Float64 = NaN
    C_phi::Float64 = NaN
end

"""
    is_cleared(ec::ElectronClosure)

`true` only when every gated closure constant has been assigned (no NaN).
"""
is_cleared(ec::ElectronClosure) = !(isnan(ec.k_HS) || isnan(ec.f_p) || isnan(ec.h_prefactor) || isnan(ec.C_phi))

end # module Fields
