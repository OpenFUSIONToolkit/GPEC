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
"""
module Coefficients

import QuadGK

export magnetic_drift_frequency, orbit_average_drift_brackets

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
    y > 0 || throw(ArgumentError("y must be positive"))
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

end # module Coefficients
