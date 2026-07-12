"""
    Coefficients

Home for the **human-cleared** Level-0 physics coefficient builders (the M2b
derivation-lane fill-ins). Every function here computes a coefficient whose form
was independently re-derived (Decision D7) and signed off by a human, with the
clearance recorded in `docs/01` and the derivation in
`docs/src/islands/derivations/`. Uncleared coefficients stay gated in their home
modules (`Frames`, `Fields`, `Operators`, `Moments`); nothing is promoted here
without that paper trail.

Cleared so far:

  - [`magnetic_drift_frequency`](@ref) — the orbit-averaged drift frequency
    ``\\hat\\omega_D`` and the `:original`/`:improved` ``\\hat L_B^{-1}`` toggle
    (sign-off 2026-07-11; derivation `omega-D-drift-frequency.md`, docs/01 §2.1).
  - [`pitch_diffusivity`](@ref), [`deflection_frequency`](@ref) — the Lorentz
    collision operator's self-adjoint diffusivity ``P(\\lambda)=\\lambda\\sqrt{1-\\lambda B}``
    and the velocity dependence ``\\nu_{jj}(\\hat v)=\\tilde\\nu[\\phi-G]/\\hat v^3``
    (sign-off 2026-07-11; derivation `collision-operator.md`, docs/01 §2.3). The
    ``\\langle\\hat\\nu_{ii}\\rangle_u`` momentum-restoring constant is a deferred
    sub-item and stays gated.
"""
module Coefficients

import QuadGK

export magnetic_drift_frequency, orbit_average_drift_brackets
export pitch_diffusivity, deflection_frequency
export h_amplitude, passing_fraction
export quasineutrality_coefficient
export delta_moment_prefactors

# Model circular equilibrium field modulation (docs/01 §1, I19 p. 6):
# b(θ) = B/B_max = (1 − ε cos θ)/(1 + ε); b ∈ [b_min, 1], b_min = b(0), b(π)=1.
@inline _b(θ, ε) = (1 - ε * cos(θ)) / (1 + ε)

"""
    orbit_average_drift_brackets(; y, epsilon, rtol=1e-8)

The two poloidal-orbit integrals that appear in ``\\hat\\omega_D`` (docs/01 §2.1;
derivation `omega-D-drift-frequency.md` §5), in I19's ``\\langle\\cdot\\rangle_\\theta = \\tfrac{1}{2\\pi}\\oint`` form:

```math
A(y) = \\Big\\langle \\frac{\\sqrt{1-yb}}{b} \\Big\\rangle_\\theta,
\\qquad
G(y) = \\Big\\langle \\frac{2-yb}{b\\sqrt{1-yb}} \\Big\\rangle_\\theta ,
```

with `b(θ) = (1−ε cos θ)/(1+ε)`. Returns `(A, G)`.

**Passing** particles (`y < y_c = 1`): the full poloidal circuit, ``\\tfrac1{2\\pi} \\int_0^{2\\pi}``. **Trapped** particles (`1 < y < (1+ε)/(1−ε)`): the bounce
integral ``\\tfrac1{2\\pi}\\sum_\\sigma\\int_{-\\theta_b}^{\\theta_b}`` between the
turning points ``\\cos\\theta_b = [1-(1+ε)/y]/ε``. `G`'s integrand has an
integrable ``1/\\sqrt{1-yb}`` singularity at the turning points, handled by the
adaptive quadrature. The signed-off derivation covers the passing drift-island
mechanism; the trapped brackets follow I19's stated ``\\langle\\cdot\\rangle_\\theta``.
"""
function orbit_average_drift_brackets(; y::Real, epsilon::Real, rtol::Real=1e-8)
    ε = float(epsilon)
    0 < ε < 1 || throw(ArgumentError("epsilon must be in (0, 1) (got $epsilon)"))
    # y = 0 (deeply-passing endpoint) is physical: A=⟨1/b⟩, G=⟨2/b⟩ are finite; the
    # guard admits the closed pitch domain [0, 1/b_min) the grid samples.
    y >= 0 || throw(ArgumentError("y must be nonnegative"))
    y_forbidden = (1 + ε) / (1 - ε)          # 1/b_min: no particle beyond this
    y < y_forbidden || throw(ArgumentError("y = $y exceeds 1/b_min = $y_forbidden (forbidden region)"))

    fA(θ) = sqrt(max(1 - y * _b(θ, ε), 0.0)) / _b(θ, ε)
    fG(θ) = (2 - y * _b(θ, ε)) / (_b(θ, ε) * sqrt(max(1 - y * _b(θ, ε), 0.0)))

    if y < 1                                  # passing: full circuit
        A, _ = QuadGK.quadgk(fA, 0.0, 2π; rtol=rtol)
        G, _ = QuadGK.quadgk(fG, 0.0, 2π; rtol=rtol)
        return (A / (2π), G / (2π))
    else                                      # trapped: bounce integral between turning points
        cosθb = (1 - (1 + ε) / y) / ε
        θb = acos(clamp(cosθb, -1.0, 1.0))
        # (1/2π) Σ_σ ∫_{-θb}^{θb} = (1/π) ∫_{-θb}^{θb} for σ-even integrands; split at 0
        A, _ = QuadGK.quadgk(fA, -θb, 0.0, θb; rtol=rtol)
        G, _ = QuadGK.quadgk(fG, -θb, 0.0, θb; rtol=rtol)
        return (A / π, G / π)
    end
end

"""
    magnetic_drift_frequency(; y, v_hat, sigma, epsilon, inv_Lq, inv_LB, variant=:original, rtol=1e-8)

The cleared orbit-averaged magnetic drift frequency (docs/01 §2.1; derivation
`omega-D-drift-frequency.md`, sign-off 2026-07-11):

```math
\\hat\\omega_D = \\frac{\\sigma\\hat v}{1+\\varepsilon}
   \\Big[\\; \\hat L_q^{-1}\\,A(y) \\;-\\; \\tfrac12\\,\\hat L_B^{-1}\\,G(y) \\;\\Big],
```

with `A(y)`, `G(y)` the orbit brackets of
[`orbit_average_drift_brackets`](@ref). The **drift-model toggle** is `variant`:

  - `:original` — finite `inv_LB` (the I19/DK-NTM ∇B term is retained);
  - `:improved` — `inv_LB` is forced to `0` (the D21/RDK-NTM proxy: ∂B/∂ψ ∝ cos θ
    orbit-averages to O(ε); derivation §6). This is the ~×6 threshold-width
    toggle (measured as the internal `:original`/`:improved` ratio — the D9 T2
    gate; the absolute 8.73→1.46 ρ_bi pair is T4/audit-gated).

`inv_Lq = L̂_q⁻¹`, `inv_LB = L̂_B⁻¹`, `epsilon = ε`, `v_hat = v/v_thi`,
`sigma = ±1`, `y = λ B_max`.
"""
function magnetic_drift_frequency(; y::Real, v_hat::Real, sigma::Real, epsilon::Real,
    inv_Lq::Real, inv_LB::Real, variant::Symbol=:original, rtol::Real=1e-8)
    variant in (:original, :improved) || throw(ArgumentError("variant must be :original or :improved (got $variant)"))
    LB = variant === :improved ? zero(float(inv_LB)) : float(inv_LB)
    A, G = orbit_average_drift_brackets(; y=y, epsilon=epsilon, rtol=rtol)
    return (sigma * v_hat / (1 + epsilon)) * (inv_Lq * A - 0.5 * LB * G)
end

# ---------------------------------------------------------------------------
# Collision operator (cleared 2026-07-11; derivation collision-operator.md)
# ---------------------------------------------------------------------------
"""
    pitch_diffusivity(λ, B)

The Lorentz pitch-angle operator's self-adjoint **diffusivity**
``P(\\lambda) = \\lambda\\sqrt{1-\\lambda B} \\ge 0`` (docs/01 §2.3; derivation
`collision-operator.md` §2), for the operator ``w^{-1}\\partial_\\lambda(P \\partial_\\lambda)`` with measure ``w = B/\\sqrt{1-\\lambda B}``. `P` vanishes at
`λ=0` and the trapped–passing edge `λ=1/B` (zero-flux endpoints), so it feeds
`conservative_pitch_operator` directly and preserves the A4 conservation gate.
"""
function pitch_diffusivity(λ::Real, B::Real)
    B > 0 || throw(ArgumentError("B must be positive"))
    (0 <= λ <= 1 / B) || throw(ArgumentError("λ must be in [0, 1/B]"))
    return λ * sqrt(max(1 - λ * B, 0.0))
end

# error function φ(X) = (2/√π)∫₀ˣ e^{−t²}dt and its derivative φ'(X) = (2/√π)e^{−X²}
_phi(X; rtol=1e-10) = (2 / sqrt(π)) * first(QuadGK.quadgk(t -> exp(-t^2), 0.0, X; rtol=rtol))
_dphi(X) = (2 / sqrt(π)) * exp(-X^2)
# Chandrasekhar function G(X) = [φ − Xφ']/(2X²)
_chandrasekhar_G(X; rtol=1e-10) = (_phi(X; rtol=rtol) - X * _dphi(X)) / (2 * X^2)

"""
    deflection_frequency(v_hat; nu_tilde=1.0, model=:chandrasekhar, rtol=1e-10)

The velocity dependence of the deflection frequency (docs/01 §2.3; derivation
`collision-operator.md` §3):

```math
\\nu_{jj}(\\hat v) = \\tilde\\nu\\,\\frac{\\phi(\\hat v) - G(\\hat v)}{\\hat v^3}
\\quad(\\texttt{:chandrasekhar}),
\\qquad
\\nu_{jj}(\\hat v) = \\tilde\\nu\\,\\hat v^{-3}\\quad(\\texttt{:vcubed}),
```

with ``\\phi`` the error function and ``G`` the Chandrasekhar function. The
`:chandrasekhar` form ``\\to \\tfrac{4\\tilde\\nu}{3\\sqrt\\pi}\\hat v^{-2}`` at low
speed (the derived linear ``\\phi-G`` limit) and ``\\to \\tilde\\nu\\hat v^{-3}`` at
high speed; the `:vcubed` form is the reduced Diss19/D21 model. The energy-
dependence sub-toggle (ladder E3).
"""
function deflection_frequency(v_hat::Real; nu_tilde::Real=1.0, model::Symbol=:chandrasekhar, rtol::Real=1e-10)
    v_hat > 0 || throw(ArgumentError("v_hat must be positive"))
    if model === :vcubed
        return nu_tilde / v_hat^3
    elseif model === :chandrasekhar
        return nu_tilde * (_phi(v_hat; rtol=rtol) - _chandrasekhar_G(v_hat; rtol=rtol)) / v_hat^3
    else
        throw(ArgumentError("model must be :chandrasekhar or :vcubed (got $model)"))
    end
end

# ---------------------------------------------------------------------------
# Flattened-electron closure (cleared 2026-07-11; derivation electron-closure.md)
# ---------------------------------------------------------------------------
"""
    h_amplitude(w_psi)

The flattened-electron profile amplitude ``C = w_\\psi/(2\\sqrt2)`` (docs/01 §2.4;
derivation `electron-closure.md` §3), i.e. the prefactor of
``h(\\Omega)=\\Theta(\\Omega-1)\\,C\\int_1^\\Omega d\\Omega'/Q(\\Omega')``. Derived
from the flattening constraint ``\\langle\\partial^2 h/\\partial x^2\\rangle_\\Omega=0``
(``\\Rightarrow h'=C/Q``) plus far-field matching ``h\\to x``. Feeds
`Fields.h_profile`'s `prefactor` (and `Fields.ElectronClosure.h_prefactor`); the
closure constants ``k`` and ``f_p`` remain gated.
"""
h_amplitude(w_psi::Real) = w_psi / (2 * sqrt(2))

"""
    passing_fraction(epsilon)

The flattened-electron closure **passing (circulating) particle fraction**
``f_p = 1 - f_t \\simeq 1 - 1.4624\\,\\sqrt\\varepsilon`` (docs/01 §2.4; derivation
`passing-fraction.md`, human sign-off 2026-07-11). The leading coefficient
`1.4624` is the ``\\varepsilon\\to0`` limit of the effective trapped-fraction
integral (Lin-Liu–Miller / Wesson) derived and numerically confirmed in the
derivation; it matches the sources' quoted ``1.46`` (I19 Eq. 22) to three
significant figures. Clears `Fields.ElectronClosure.f_p`; the Hirshman–Sigmar
`k` constant of the same closure remains gated (QUESTIONS Q3/Q5).
"""
function passing_fraction(epsilon::Real)
    epsilon >= 0 || throw(ArgumentError("epsilon must be nonnegative"))
    return 1 - 1.4624 * sqrt(epsilon)
end

# ---------------------------------------------------------------------------
# Quasineutrality closure (cleared 2026-07-11; derivation quasineutrality-closure.md)
# ---------------------------------------------------------------------------
"""
    quasineutrality_coefficient(tau)

The Level-0 quasineutrality closure coefficient ``\\tau/(\\tau+1)`` (docs/01 §3;
derivation `quasineutrality-closure.md`), where ``\\tau=T_e/T_i``:

```math
\\hat\\Phi = \\frac{\\tau}{\\tau+1}\\Big[\\frac{\\delta\\bar n_i}{n_0}
   + \\hat L_{n0}^{-1}\\,(x-\\hat h)\\Big].
```

The ``\\tau/(\\tau+1)`` factor is the sum of the ion and electron adiabatic
shielding responses (``\\to 1/2`` at ``\\tau=1``, the sources' ``T_e=T_i``). The
raw ion-density moment ``\\delta\\bar n_i=M[g_i]`` (`velocity_moment!`) enters
directly — no ``\\delta n_i`` normalization convention — so this coefficient plus
the ``\\hat L_{n0}^{-1}(x-\\hat h)`` drive populate `Operators.Quasineutrality`.
"""
function quasineutrality_coefficient(tau::Real)
    tau > 0 || throw(ArgumentError("tau = T_e/T_i must be positive"))
    return tau / (tau + 1)
end

# ---------------------------------------------------------------------------
# Δ-moment prefactors (cleared 2026-07-11; derivation delta-moment-prefactors.md)
# ---------------------------------------------------------------------------
"""
    delta_moment_prefactors(; mu0_R, w_psi, dq_dpsi, q_s)

The ``\\Delta_{\\cos}``/``\\Delta_{\\sin}`` output-moment prefactors (docs/01 §4;
derivation `delta-moment-prefactors.md`): returns
`(cos=-\\mu_0 R/(2\\tilde\\psi), sin=+\\mu_0 R/(2\\tilde\\psi))` for
`Moments.delta_moments`, with ``\\tilde\\psi`` the cleared island flux
amplitude `Moments.island_flux_amplitude`. ``\\Delta_{\\cos}=\\Delta_{\\rm neo}``
matches Diss19 Eq. 4.12 (stationarity ``\\Delta'+\\Delta_{\\rm neo}=0``); the sin
normalization is the symmetric `[DERIVED]` pin so that
``\\Delta_{\\cos}+i\\Delta_{\\sin}`` maps onto the linear-layer ``\\Delta(Q)``.
"""
function delta_moment_prefactors(; mu0_R::Real, w_psi::Real, dq_dpsi::Real, q_s::Real)
    ψ̃ = (w_psi^2 / 4) * (dq_dpsi / q_s)     # = island_flux_amplitude (docs/01 §1)
    ψ̃ != 0 || throw(ArgumentError("ψ̃ must be nonzero (check w_psi, dq_dpsi)"))
    pref = mu0_R / (2 * ψ̃)
    return (cos=-pref, sin=+pref)
end

end # module Coefficients
