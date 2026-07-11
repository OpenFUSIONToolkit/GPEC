# Derivation — the Level-0 pitch-angle collision operator, deflection frequency, and ``\nu_\star`` normalization

**Provenance:** `[DERIVED: 2026-07-11]` — independent re-derivation (Decision D7).
**Clears (on sign-off):** the momentum-conserving pitch-angle (Lorentz)
collision operator structure, the deflection-frequency velocity dependence
``\nu_{jj}(v)`` with the Chandrasekhar/``v^{-3}`` sub-toggle, and the ``\nu_\star``
normalization (`[CHECKED: I19 Eqs. 9–12; Diss19 Eqs. 2.25–2.30; WCHH96 Eq. 62]`,
QUESTIONS Q3).
**Deferred (flagged sub-items, §7):** the analytic velocity average
``\langle\hat\nu_{ii}\rangle_u`` (L23 Eq. 4.1.6 — a separate short derivation),
and the orbit-averaged/discretized diffusivity profile that feeds
`PitchAngleDiffusion` (numerics, ties to the conservation gate A4).

**Status:** ✅ **signed off 2026-07-11** (clearance recorded in docs/01 §2.3) for
the operator structure, deflection frequency, and ``\nu_\star`` normalization —
implemented as `Coefficients.pitch_diffusivity` and
`Coefficients.deflection_frequency`. The ``\langle\hat\nu_{ii}\rangle_u``
constant (§7) and the discretized diffusivity profile remain **deferred /
gated**.

## 1. Starting point and what must be shown

At Level 0 the collision operator is the momentum-conserving **pitch-angle
(Lorentz) model** (orderings O6; docs/01 §2.3). From the drift-kinetic equation
(I19 Eq. 8) the like-species operator is (I19 Eq. 9, verified first-hand,
print p. 4)

```math
C_{jj}(f) = 2\nu_{jj}(v)\left[\frac{\sqrt{1-\lambda B}}{B}\,
   \frac{\partial}{\partial\lambda}\!\Big(\lambda\sqrt{1-\lambda B}\,
   \frac{\partial f}{\partial\lambda}\Big)
   \;+\; \frac{v_\parallel\,\bar u_{\parallel j}}{v_{thj}^2}\,F_{Mj}\right],
```

the ``\lambda``-derivatives taken at fixed ``\psi``, ``\lambda=\mu/\mathcal E`` the
pitch, ``\mathcal E=v^2/2``. This derivation establishes three things: (i) the
first bracket **is** the pitch-angle scattering operator, in self-adjoint
(divergence) form, with diffusivity ``P(\lambda)=\lambda\sqrt{1-\lambda B}``;
(ii) the velocity dependence ``\nu_{jj}(v)``; (iii) the ``\nu_\star``
normalization. The second bracket is the momentum-restoring term (§6).

## 2. The pitch-angle operator is Lorentz scattering in self-adjoint form

The full Fokker–Planck test-particle operator, in the small-mass-ratio /
dominant-deflection limit that defines the Lorentz model, reduces to pure
pitch-angle scattering — diffusion of the velocity-vector *direction* at fixed
speed. In the pitch cosine ``\xi_p=v_\parallel/v`` it is the standard Lorentz
operator ``C=\tfrac{\nu_{jj}}{2}\,\partial_{\xi_p}\!\big[(1-\xi_p^2)\,
\partial_{\xi_p}f\big]``. Change to the constant-of-motion pitch
``\lambda=\mu/\mathcal E`` via ``1-\lambda B=\xi_p^2=v_\parallel^2/v^2`` (at
fixed ``B,v``): then ``\partial_{\xi_p}=-\tfrac{2\sqrt{1-\lambda B}}{B}
\partial_\lambda`` and ``1-\xi_p^2=\lambda B``, so

```math
\partial_{\xi_p}\!\big[(1-\xi_p^2)\partial_{\xi_p}f\big]
 = \frac{4\sqrt{1-\lambda B}}{B}\,\frac{\partial}{\partial\lambda}\!
   \Big(\lambda\sqrt{1-\lambda B}\,\frac{\partial f}{\partial\lambda}\Big),
```

and therefore

```math
C = 2\nu_{jj}\,\frac{\sqrt{1-\lambda B}}{B}\,
   \frac{\partial}{\partial\lambda}\!\Big(\lambda\sqrt{1-\lambda B}\,
   \frac{\partial f}{\partial\lambda}\Big) ,
```

**exactly** the first term of Eq. 9 — the ``2\nu_{jj}`` prefactor is the
change-of-variables Jacobian, so the coefficient is not free. Two structural
facts follow directly:

- **Self-adjoint (divergence) form.** The operator is
  ``w(\lambda)^{-1}\,\partial_\lambda[P(\lambda)\,\partial_\lambda]`` with
  **diffusivity** ``P(\lambda)=\lambda\sqrt{1-\lambda B}\ge 0`` and **measure**
  ``w(\lambda)=B/\sqrt{1-\lambda B}`` (so the prefactor is ``1/w``). This is
  exactly the form the implemented mimetic operator
  `Operators.conservative_pitch_operator` discretizes
  (``K=-W_q^{-1}G^{\mathsf T}\mathrm{diag}(P\,w_q)G``), which is why particle
  conservation and the entropy sign hold *exactly in floating point* (gate A4,
  already green). This derivation identifies its physics profile:
  ``P=\lambda\sqrt{1-\lambda B}``.
- **Zero-flux endpoints.** ``P(\lambda)`` vanishes at ``\lambda=0`` (``v_\parallel
  =v``, ``P=0``) and at ``\lambda=1/B`` (``v_\parallel=0``, ``\sqrt{1-\lambda B}=0``),
  so the scattering flux ``P\,\partial_\lambda f`` vanishes at both ends — no
  pitch boundary conditions are needed, matching the mimetic construction.

## 3. The deflection frequency (velocity dependence)

The speed dependence is carried entirely by ``\nu_{jj}(v)`` (I19 Eq. 11):

```math
\nu_{jj}(v) = \tilde\nu_{jj}\,\frac{\phi(\hat v)-G(\hat v)}{\hat v^{3}},
\qquad \hat v = v/v_{thj},\ \ v_{thj}^2=2T_j/m_j ,
```

with the error function and Chandrasekhar function

```math
\phi(X)=\frac{2}{\sqrt\pi}\int_0^X e^{-t^2}dt,
\qquad
G(X)=\frac{\phi(X)-X\phi'(X)}{2X^2},\quad \phi'=\frac{d\phi}{dX} .
```

This is the standard test-particle **deflection** (perpendicular-diffusion)
rate. Two limits fix the structure and the sub-toggle (docs/01 §2.3, ladder E3).
Expanding with ``\phi(X)=\tfrac{2}{\sqrt\pi}(X-\tfrac{X^3}{3}+\dots)`` and
``\phi'(X)=\tfrac{2}{\sqrt\pi}e^{-X^2}``:

```math
G(X)=\frac{\phi-X\phi'}{2X^2}
    =\frac{2}{\sqrt\pi}\Big(\frac{X}{3}-\frac{X^3}{5}+\dots\Big),
\qquad
\phi-G=\frac{2}{\sqrt\pi}\Big(\frac{2X}{3}-\frac{2X^3}{15}+\dots\Big).
```

- **Low speed** ``\hat v\to 0``: ``\phi-G\to \tfrac{4}{3\sqrt\pi}\hat v``
  (**linear**, not cubic), so
  ``\nu_{jj}\to \tfrac{4\tilde\nu_{jj}}{3\sqrt\pi}\,\hat v^{-2}`` — the deflection
  frequency **diverges as ``\hat v^{-2}``** even in the *full* Chandrasekhar
  form. This is precisely the "``\tilde\nu\propto u^{-2}`` divergence at low
  ``u``" of docs/01 §2.3 that spoils a naive velocity quadrature and motivates
  the analytic ``\langle\hat\nu_{ii}\rangle_u`` (§7).
- **High speed** ``\hat v\to\infty``: ``\phi-G\to 1``, so ``\nu_{jj}\to
  \tilde\nu_{jj}/\hat v^{3}`` — the ``v^{-3}`` tail.
- **Sub-toggle:** I19/L23 use the full ``[\phi-G]/\hat v^3`` (Chandrasekhar,
  ``\sim\hat v^{-2}`` at low ``\hat v``, needed for neoclassical fidelity);
  Diss19/D21 use the pure ``\hat v^{-3}`` (a stronger low-``\hat v`` divergence).
  In both, the divergence is *integrable* in the velocity moments (``\int
  \hat v^4 e^{-\hat v^2}\nu\,d\hat v`` converges), but it makes naive quadrature
  inaccurate — hence L23's analytic value (§7).

## 4. The ``\nu_\star`` normalization

The banana-regime collisionality is (docs/01 §2.3; L23 Eq. 2.3.40)

```math
\nu_\star = \frac{\nu_{jj}\,Rq}{\varepsilon^{3/2} v_{th}},
\qquad
\hat\nu_{jj} = \varepsilon^{3/2}\nu_\star\,\tilde\nu_{jj}(\hat v) ,
```

i.e. ``\nu_\star`` is the ratio of the effective (``\varepsilon^{-1}``
trapped-boundary-enhanced) collision rate to the bounce rate ``v_{th}/Rq``;
``\nu_\star\ll 1`` is the banana regime (O6). The ``\varepsilon^{3/2}`` collects
the trapped fraction ``\sim\sqrt\varepsilon`` and the effective-collisionality
boundary-layer width ``\sim\varepsilon`` (the ``\propto\nu^{1/2}`` layer of
docs/04 §2). The dimensionless drift-kinetic equation carries
``\hat\nu_{jj}=\varepsilon^{3/2}\nu_\star\tilde\nu_{jj}(\hat v)`` as the collision
coefficient.

## 5. Species coupling (electron–ion)

The electron operator ``C_{ei}`` (I19 Eq. 10) has the identical pitch-angle
structure with ``\nu_{ei}`` and a momentum-restoring term
``v_\parallel u_{\parallel i}/v_{the}^2\,F_{Me}`` that drags on the **ion** flow
``u_{\parallel i}`` — so electrons and ions are coupled through momentum
conservation (the flattened-electron closure, docs/01 §2.4, is a separate
derivation). At Level 0 the bulk operator is ``C_{ii}``; multi-species
field-particle coupling is Level 1 (`Collisions{FokkerPlanckMulti}`), plumbing
already present.

## 6. The momentum-restoring term

The second bracket of Eq. 9, ``v_\parallel\bar u_{\parallel j}F_{Mj}/v_{thj}^2``,
restores parallel momentum removed by the pitch-angle scattering (pure Lorentz
scattering is not momentum-conserving on its own). The restoring flow is
(I19 Eq. 12)

```math
\bar u_{\parallel j}(f) = \frac{1}{n\langle\nu_{jj}\rangle_v}\int d^3v\,
   \nu_{jj}\,v_\parallel f,
\qquad
\langle\nu_{jj}\rangle_v = \frac{8}{3\sqrt\pi}\int_0^\infty d\hat v\,
   \hat v^4 e^{-\hat v^2}\,\nu_{jj}(\hat v),
```

with the velocity element ``\int d^3v = \pi B\sum_\sigma\int_0^\infty v^2 dv
\int_0^{B^{-1}} d\lambda/\sqrt{1-\lambda B}`` (I19 Eq. 13). This term is
structurally fixed here; its **magnitude** enters through
``\langle\nu_{jj}\rangle_v``, whose closed form is the deferred item §7.

## 7. Deferred sub-items (flagged, not asserted)

- **Analytic velocity average ``\langle\hat\nu_{ii}\rangle_u``.** The low-``\hat v``
  divergence of ``\tilde\nu`` (``\hat v^{-2}`` Chandrasekhar / ``\hat v^{-3}``
  reduced, §3) makes the momentum-restoring velocity integral of §6 poorly
  behaved under naive quadrature; L23 Eq. 4.1.6 (p. 88) gives the closed form
  ``\langle\hat\nu_{ii}\rangle_u = \tfrac{4\varepsilon^{3/2}\nu_\star}{3\sqrt\pi}
  (\sqrt2-\ln(1+\sqrt2))``. **This specific constant is not derived here** — it
  requires L23's exact reduced integrand and is a self-contained follow-up
  derivation (policy rule 4: not presented as derived until it is). The
  ``\sqrt2`` reflects the ion self-collision reduced mass; ``\ln(1+\sqrt2)=
  \operatorname{arcsinh}(1)`` points to a ``\int d\hat v/\sqrt{1+\hat v^2}``-type
  reduction.
- **Discretized diffusivity profile for `PitchAngleDiffusion`.** The mapping of
  ``P(\lambda)=\lambda\sqrt{1-\lambda B}`` and the measure through the
  orbit-average and onto the ``y=\lambda B_{\max}`` grid (with the ``\theta``-
  and ``\hat v``-dependence) is numerics that ties to the already-green A4
  conservation gate; scoped with the L0 collisional solve.

## 8. Cross-check table

| Source | Form | Agrees with `[DERIVED]`? |
|---|---|---|
| I19 Eq. (9) (first-hand, print p. 4) | ``C_{jj}=2\nu_{jj}[\tfrac{\sqrt{1-\lambda B}}{B}\partial_\lambda(\lambda\sqrt{1-\lambda B}\partial_\lambda f)+\tfrac{v_\parallel\bar u_{\parallel j}}{v_{thj}^2}F_{Mj}]`` | ✅ operator structure + self-adjoint form + ``P=\lambda\sqrt{1-\lambda B}`` |
| I19 Eq. (11) (first-hand) | ``\nu_{jj}=\tilde\nu_{jj}[\phi(\hat v)-G(\hat v)]/\hat v^3`` | ✅ deflection frequency + limits |
| I19 Eqs. (12)–(13) (first-hand) | ``\bar u_{\parallel j}``, ``\langle\nu_{jj}\rangle_v``, ``\int d^3v`` | ✅ structure (constant deferred, §7) |
| L23 Eq. 2.3.40 | ``\nu_\star=\nu_{jj}Rq/(\varepsilon^{3/2}v_{th})`` | ✅ normalization |
| L23 Eq. 4.1.6 | ``\langle\hat\nu_{ii}\rangle_u=\tfrac{4\varepsilon^{3/2}\nu_\star}{3\sqrt\pi}(\sqrt2-\ln(1+\sqrt2))`` | ⏳ deferred (§7) |

**Triage:** operator structure, deflection frequency, and normalization agree
with all first-hand sources — no discrepancy. The one number that carries a
specific closed form (``\langle\hat\nu_{ii}\rangle_u``) is honestly deferred to
its own derivation rather than transcribed.

## 9. What sign-off authorizes

On sign-off (recorded in docs/01 §2.3): the pitch-angle **diffusivity profile**
``P(\lambda)=\lambda\sqrt{1-\lambda B}`` and **measure**
``w=B/\sqrt{1-\lambda B}`` may be used to build the `PitchAngleDiffusion`
operator (via `conservative_pitch_operator`, preserving A4); the deflection
frequency ``\nu_{jj}(\hat v)=\tilde\nu[\phi-G]/\hat v^3`` (with the
``:chandrasekhar``/``:vcubed`` sub-toggle) and the ``\hat\nu=\varepsilon^{3/2}
\nu_\star\tilde\nu`` normalization populate the collision coefficient. The
momentum-restoring magnitude waits on ``\langle\hat\nu_{ii}\rangle_u`` (§7); the
electron closure constants (``k``, ``f_p``) are a separate derivation.
