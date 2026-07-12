# Derivation — the passing fraction ``f_p \simeq 1 - 1.46\sqrt\varepsilon``

**Provenance:** `[DERIVED: 2026-07-11]` — independent derivation of the
electron-closure passing-fraction constant (Decision D7), one of the deferred
sub-constants of the flattened-electron closure (QUESTIONS Q3/Q5).
**Status:** ✅ **signed off 2026-07-11** — implemented as
`Coefficients.passing_fraction(ε) = 1 − 1.4624√ε`, which may populate
`Fields.ElectronClosure.f_p`. This chapter derives the constant and cross-checks
it numerically (§3); the reviewer accepted the `1.4624 ≈ 1.46` match (§4). The
companion Hirshman–Sigmar `k` remains gated.

## 1. What `f_p` is

The flattened-electron closure (docs/01 §2.4; `electron-closure.md`) carries the
**passing (circulating) particle fraction** ``f_p`` of a large-aspect-ratio
circular flux surface — the velocity-space fraction of electrons that complete a
poloidal circuit rather than mirror-trap in the outboard well.
The sources quote ``f_p \simeq 1 - 1.46\sqrt\varepsilon`` `[CHECKED: I19
Eq. (22); L23 Eqs. 2.5.5–2.5.8]`, i.e. ``f_p = 1 - f_t`` with the effective
trapped fraction ``f_t \simeq 1.46\sqrt\varepsilon``.
This chapter derives ``f_t`` and hence ``f_p``.

## 2. The effective trapped-fraction integral

Use the model field modulation pinned in `Coefficients.jl` (docs/01 §1),

```math
b(\theta) \;=\; \frac{B(\theta)}{B_{\max}} \;=\; \frac{1-\varepsilon\cos\theta}{1+\varepsilon},
\qquad b\in[b_{\min},\,1],\quad b(\pi)=1 ,
```

\noindent
so ``B_{\max}`` sits on the inboard side ``\theta=\pi``.
A particle of pitch ``\lambda = \mu B_{\max}/E`` is trapped when ``\lambda b(\theta)=1`` somewhere on the surface, i.e. for ``\lambda\in(1,\,1/b_{\min})``; it is passing for ``\lambda<1``.
The neoclassical **effective** trapped fraction is the pitch-space average
(Lin-Liu & Miller; Wesson, *Tokamaks*)

```math
f_t \;=\; 1 - \frac{3}{4}\,\langle b^2\rangle
      \int_0^{1}\frac{\lambda\,d\lambda}{\big\langle\sqrt{1-\lambda b}\,\big\rangle},
\qquad
\langle\cdot\rangle \equiv \frac{1}{2\pi}\oint d\theta ,
```

\noindent
the same flux-surface average ``\langle\cdot\rangle_\theta`` the drift brackets
use (`omega-D-drift-frequency.md`).

## 3. The ``\varepsilon\to0`` limit and the ``\sqrt\varepsilon`` coefficient

At ``\varepsilon=0`` the field is uniform (``b\equiv1``, ``\langle b^2\rangle=1``), and

```math
\int_0^1\frac{\lambda\,d\lambda}{\sqrt{1-\lambda}}
 \;=\; B(2,\tfrac12) \;=\; \frac{\Gamma(2)\,\Gamma(\tfrac12)}{\Gamma(\tfrac52)}
 \;=\; \frac{4}{3},
\qquad\Rightarrow\qquad
f_t(0) = 1-\tfrac34\cdot1\cdot\tfrac43 = 0 ,
```

\noindent
every particle circulates when there is no well — the correct zeroth order.
The leading correction is ``O(\sqrt\varepsilon)``, not ``O(\varepsilon)``:
it comes from the boundary layer near ``\lambda=1``, where ``1-\lambda b(\theta)``
vanishes over part of the circuit (the barely-passing/barely-trapped particles),
so ``\langle\sqrt{1-\lambda b}\rangle`` acquires a ``\sqrt{\varepsilon}``-scale
behaviour.
Writing ``f_t = c_1\sqrt\varepsilon + O(\varepsilon)``, the coefficient ``c_1``
is the ``\varepsilon\to0`` limit of ``f_t/\sqrt\varepsilon``.

Evaluating the integral numerically in this convention (QuadGK, the same
machinery as the cleared brackets) gives a clean limit:

| ``\varepsilon`` | ``f_t`` | ``f_t/\sqrt\varepsilon`` |
|---|---|---|
| ``10^{-2}`` | ``0.145171`` | ``1.451714`` |
| ``10^{-3}`` | ``0.046214`` | ``1.461401`` |
| ``10^{-4}`` | ``0.014623`` | ``1.462324`` |
| ``10^{-5}`` | ``0.0046246`` | ``1.462415`` |

\noindent
so ``c_1 = 1.4624\ldots``, and

```math
\boxed{\;
f_p \;=\; 1 - f_t \;=\; 1 - c_1\sqrt\varepsilon
   \;\simeq\; 1 - 1.46\,\sqrt\varepsilon
\;}
```

\noindent
The derived leading coefficient ``1.4624`` matches the sources' quoted ``1.46``
to three significant figures — the quote is the rounded asymptotic constant, not
a distinct number.

## 4. Cross-check table and the open sign-off item

| Source | Form | Agrees with `[DERIVED]`? |
|---|---|---|
| I19 Eq. (22) / L23 §2.5 (via docs/01 §2.4) | ``f_p \simeq 1-1.46\sqrt\varepsilon`` | ✅ to 3 s.f. (``c_1=1.4624``) |
| ``\varepsilon\to0`` analytic limit | ``f_t(0)=0`` (all-passing), ``B(2,\tfrac12)=\tfrac43`` | ✅ exact (§3) |
| numerical asymptotics (this chapter) | ``c_1 = 1.4624\ldots`` | ✅ converged |

**Open for the reviewer (sign-off gate):** this derivation uses the standard
Lin-Liu–Miller *effective* trapped fraction. Confirm that I19 Eq. (22) /
L23 §2.5 define ``f_p`` by this same effective fraction (and not a bare
pitch-boundary fraction, which carries a slightly different ``O(\sqrt\varepsilon)``
coefficient) before clearing. The ``0.16\%`` gap between ``1.4624`` and the
quoted ``1.46`` is consistent with rounding, but a definition mismatch would show
up here — hence the gate.

## 5. What sign-off would authorize

On sign-off (to be recorded in docs/01 §2.4): `f_p = 1 - 1.4624·√ε` (or the
source's exact constant, if the reviewer pins it) may replace the `NaN`-gated
`Fields.ElectronClosure.f_p`. Until then `f_p` stays gated (QUESTIONS Q3/Q5) and
this chapter remains a **draft**, not a clearance. The companion deferred
constants ``\langle\hat\nu_{ii}\rangle_u`` (collision) and the Hirshman–Sigmar
``k`` (parallel flow) are **not** drafted here — each needs its specific source
integrand (L23 Eq. 4.1.6; the parallel-viscosity moment problem) read in detail,
and are left escalated in QUESTIONS Q3/Q5 rather than derived speculatively.
