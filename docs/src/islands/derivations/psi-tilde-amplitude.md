# Derivation — the island flux amplitude ``\tilde\psi``

**Provenance:** `[DERIVED: 2026-07-11]` — independent re-derivation (Decision D7).
**Resolves:** the open `[VERIFY]` on ``\tilde\psi`` (QUESTIONS Q4, docs/01 §1):
is the amplitude ``\tilde\psi = \tfrac{w_\psi^2}{4}\,\tfrac{q_s'}{q_s}`` or
``\tfrac{w_\psi^2}{4}\,\tfrac{q_s}{q_s'}``? **First-hand check of I19 (2026-07-11):
Imada 2019 as printed (print p. 3, text following Eq. 6) shows the second form,
``\tfrac{w_\psi^2}{4}\,\tfrac{q_s}{q_s'}``.** This derivation shows that form is
a typo in the published paper — the physical amplitude is ``q_s'/q_s`` — by three
independent arguments, including I19's *own* internally-inconsistent ``\Omega``
convention (its Eq. 7).

**Status:** awaiting human sign-off. Until signed off, the ``\tilde\psi``
prefactor of the ``\Delta_{\cos}/\Delta_{\sin}`` moments (`Moments.delta_moments`)
stays a supplied, gated argument — this derivation does **not** authorize
writing a value into `src/` by itself.

## 1. Setup and orderings

Level-0 configuration (docs/01 §1, orderings O1–O3): a single-helicity,
constant-``\psi`` magnetic island of helicity ``(m, n)`` at the rational surface
``\psi = \psi_s`` where ``q(\psi_s) = m/n``. Coordinates: ``\psi`` the poloidal
flux, ``\theta`` poloidal angle, ``\phi`` toroidal angle, and the helical angle

```math
\xi = m\theta - n\phi ,
```

equivalent to the docs/01 form ``\xi = m(\theta - \phi/q_s)`` since
``m/q_s = n``. The island is prescribed and fixed (O3): ``\tilde\psi`` is an
input amplitude, not solved. The task is purely the **island geometry** — how
``\tilde\psi`` relates to the island half-width ``w_\psi`` — so no kinetics
enter.

The perturbation is given in vector-potential form (docs/01 §1):

```math
A_\parallel = -\frac{\tilde\psi}{R}\cos\xi ,
```

where ``\tilde\psi`` is the amplitude of the perturbed **helical flux** (the
factor ``1/R`` is the metric relation between the parallel vector potential and
the poloidal-flux-like amplitude; ``A_\parallel`` has dimensions of
flux/length, so ``\tilde\psi`` has dimensions of poloidal flux — used in §5).

## 2. The equilibrium helical flux and its curvature

Define the **helical flux** as the poloidal flux minus the resonant fraction of
the toroidal flux ``\psi_{\rm tor}`` (with ``d\psi_{\rm tor}/d\psi = q``):

```math
\chi_0(\psi) \;=\; \psi \;-\; \frac{1}{q_s}\,\psi_{\rm tor}(\psi),
\qquad
\frac{d\chi_0}{d\psi} \;=\; 1 - \frac{q(\psi)}{q_s}.
```

This is the natural flux for the ``(m, n)`` resonance: its contours are the
equilibrium field lines *projected into the helical frame* (a field line has
``d\xi/d\theta = m - n\,q(\psi)``, which vanishes at ``\psi_s``), and

```math
\left.\frac{d\chi_0}{d\psi}\right|_{\psi_s} = 1 - \frac{q_s}{q_s} = 0 ,
```

so ``\chi_0`` is **stationary at the rational surface** — the defining property
of a resonant flux. Its curvature there is the load-bearing quantity:

```math
\boxed{\;
\chi_0''(\psi_s) \;=\; \frac{d}{d\psi}\!\left(1 - \frac{q}{q_s}\right)_{\psi_s}
              \;=\; -\,\frac{q_s'}{q_s}
\;}
\qquad (q_s' \equiv dq/d\psi|_{\psi_s}).
```

## 3. The constant-``\psi`` island and its half-width

Add the single-helicity perturbation of constant amplitude (the constant-``\psi``
approximation, O3) to the equilibrium helical flux and expand about ``\psi_s``
using ``\chi_0'(\psi_s)=0``, with ``x \equiv \psi - \psi_s``:

```math
\chi(x, \xi) \;=\; \chi_0(\psi_s) \;+\; \tfrac{1}{2}\,\chi_0''(\psi_s)\,x^2
              \;+\; \tilde\psi\,\cos\xi .
```

The contours of ``\chi`` are the perturbed field lines; they form an island.
The two stationary points on ``x = 0`` are ``(x,\xi) = (0, 0)`` and ``(0, \pi)``
— one the O-point (elliptic), one the X-point (hyperbolic), their roles set by
the signs of ``\chi_0''`` and ``\tilde\psi``. The **separatrix** is the contour
through the X-point. Writing ``\chi_X`` for its value and evaluating the
separatrix contour at the O-point's poloidal angle gives the maximum radial
excursion ``x_{\rm sep}``; for either sign assignment,

```math
\tfrac{1}{2}\,|\chi_0''|\,x_{\rm sep}^2 = 2\,|\tilde\psi|
\quad\Longrightarrow\quad
x_{\rm sep} = 2\sqrt{\frac{|\tilde\psi|}{|\chi_0''|}} .
```

By definition ``w_\psi`` is the island **half-width** in ``\psi``-space (the
maximum excursion of the separatrix from ``\psi_s``, docs/01 §1), so
``w_\psi = x_{\rm sep}`` and

```math
w_\psi = 2\sqrt{\frac{|\tilde\psi|}{|\chi_0''|}}
\quad\Longleftrightarrow\quad
|\tilde\psi| = \frac{w_\psi^2}{4}\,|\chi_0''| .
```

## 4. Result

Substituting ``|\chi_0''| = q_s'/q_s`` from §2:

```math
\boxed{\;
\tilde\psi \;=\; \frac{w_\psi^2}{4}\,\frac{q_s'}{q_s}
\;}
```

(with ``\tilde\psi \ge 0``, ``w_\psi`` the half-width, and ``q_s'/q_s`` taken as
its magnitude; the sign is fixed by the pinned ``\Omega`` convention of §5). The
alternative ``\tfrac{w_\psi^2}{4}\,\tfrac{q_s}{q_s'}`` is **excluded** — see the
two independent checks below.

## 5. Cross-checks

**(a) Dimensional necessity (kills the ``q_s/q_s'`` form).** ``\chi_0`` is a
flux, so ``\chi_0'' = d^2\chi_0/d\psi^2`` has dimensions ``[\psi]^{-1}``.
``w_\psi`` is a half-width in ``\psi``-space, so ``w_\psi^2 \sim [\psi]^2``, and
``\tilde\psi`` is a flux (``\sim [\psi]``, from ``A_\parallel = -(\tilde\psi/R)
\cos\xi``, §1). Check each candidate:

| candidate | dimensions | verdict |
|---|---|---|
| ``\tfrac{q_s'}{q_s}`` | ``[\psi]^{-1}`` (``q'/q``, since ``q'\sim[\psi]^{-1}``, ``q\sim 1``) | ``\tfrac{w_\psi^2}{4}\tfrac{q_s'}{q_s}\sim[\psi]^2[\psi]^{-1}=[\psi]`` ✓ = ``[\tilde\psi]`` |
| ``\tfrac{q_s}{q_s'}`` | ``[\psi]`` | ``\tfrac{w_\psi^2}{4}\tfrac{q_s}{q_s'}\sim[\psi]^3`` ✗ |

Only ``q_s'/q_s`` gives ``\tilde\psi`` the dimensions of a flux. The
``q_s/q_s'`` form is dimensionally impossible.

**(b) Consistency with the pinned island label ``\Omega``.** docs/01 §1 pins
(``[CHECKED: I19 Eq. 7; Diss19 Eq. 2.7; L23 Eq. 2.1.8]``)

```math
\Omega(x,\xi) = \frac{2(\psi-\psi_s)^2}{w_\psi^2} - \cos\xi,
\qquad \Omega = -1 \ (\text{O-point}),\ \ \Omega = +1 \ (\text{separatrix}).
```

Take the island-supporting branch ``\chi_0'' > 0`` with the O-point at
``\xi = \pi`` (equivalently, absorb the signs into the orientation of ``\xi`` and
the sign of ``\tilde\psi`` — this is the convention the ``\Omega`` label *fixes*,
not an independent assumption). Normalizing the helical flux of §3 by
``\tilde\psi`` and using the §4 result ``|\chi_0''|/\tilde\psi = 4/w_\psi^2``:

```math
\frac{\chi - \chi_0(\psi_s)}{\tilde\psi}
= \frac{|\chi_0''|}{2\tilde\psi}\,x^2 - \cos\xi
= \frac{2x^2}{w_\psi^2} - \cos\xi
= \Omega .
```

(The bare substitution gives ``(\chi_0''/2\tilde\psi)x^2 + \cos\xi``; the O-point
at ``\Omega=-1`` requires the ``\cos\xi`` term negative, which is exactly the
sign convention just stated — the ``w_\psi^2/4`` magnitude coefficient, the
load-bearing result, is the same on either branch.) The island label ``\Omega``
**is** the ``\tilde\psi``-normalized helical flux,
reproducing the pinned convention exactly (including the factor of 2 and the
O-point/separatrix values). This both fixes the sign convention and confirms the
``w_\psi^2/4`` coefficient self-consistently.

## 6. Cross-check table against the [CHECKED] transcriptions

| Source | Transcribed form (docs/01 §1) | Agrees with `[DERIVED]`? |
|---|---|---|
| Diss19 p. 30 | ``\tilde\psi = \tfrac{w_\psi^2}{4}\,q_s'/q_s`` | ✅ |
| D21 / L23 Eq. (2.1.4) | ``q_s'/q_s`` | ✅ |
| Ω convention (I19 Eq. 7; Diss19 Eq. 2.7; L23 Eq. 2.1.8) | ``\Omega = 2(\psi-\psi_s)^2/w_\psi^2 - \cos\xi`` | ✅ (reproduced in §5b) |
| I19 as printed (p. 3, text after Eq. 6) — **first-hand, 2026-07-11** | ``\tilde\psi = \tfrac{w_\psi^2}{4}\,q_s/q_s'`` | ❌ dimensionally impossible (§5a); typo |
| I19 Eq. (7) — its own ``\Omega`` convention (first-hand) | ``\Omega = 2(\psi-\psi_s)^2/w_\psi^2 - \cos\xi`` | ✅ — and this *requires* ``q_s'/q_s`` (§5b), so I19 is internally inconsistent |

**Triage (docs/05 rule 3) — resolved:** the physics is unambiguous
(``q_s'/q_s``), agrees with Diss19/D21/L23 and with I19's own ``\Omega``
convention (Eq. 7), and the alternative is dimensionally impossible. I19 as
printed shows ``q_s/q_s'`` in the amplitude text, but its Eq. (7) ``\Omega``
requires ``q_s'/q_s`` — so **I19 is internally inconsistent, and the amplitude
text is a published typo** (triage outcome: *their published-equation error*,
the standing docs/05 York-lineage rule; the same class as the L23 §2.6
amendments). Note I19 defines its shear length ``L_q = q\,(dq/dr)^{-1} = q/q'``
(p. 2), consistent with ``q'/q`` structure. **The `[VERIFY]` is closed: use
``q_s'/q_s``.**

## 7. What sign-off authorizes

On human sign-off (recorded in docs/01 with paper/equation/date, policy rule 3),
the ``\tilde\psi`` amplitude
``\tilde\psi = \tfrac{w_\psi^2}{4}\,(q_s'/q_s)`` may be used to construct the
``\Delta_{\cos}/\Delta_{\sin}`` moment prefactors ``\mp\mu_0 R/(2\tilde\psi)``
(`Moments.delta_moments`), replacing the currently-gated supplied argument. The
sin-moment normalization pin (docs/01 §4, a separate `[DERIVED]` item) is **not**
covered here.
