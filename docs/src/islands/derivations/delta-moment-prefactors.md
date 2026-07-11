# Derivation — the ``\Delta_{\cos}``/``\Delta_{\sin}`` moment prefactors

**Provenance:** `[DERIVED: 2026-07-11]` — independent derivation of the structure
plus a documented normalization pin (Decision D7). The sin-moment normalization
is explicitly a `[DERIVED]` choice (docs/01 §4), made here.
**Clears (on sign-off):** the ``\Delta_{\cos}``/``\Delta_{\sin}`` output-moment
prefactors ``\mp\mu_0 R/(2\tilde\psi)`` (`[CHECKED: Diss19 Eq. 4.12 for
\Delta_{\cos}; sin-normalization DERIVED]`, QUESTIONS Q4), building on the
already-cleared ``\tilde\psi`` (`island_flux_amplitude`).

**Status:** awaiting human sign-off. Until signed off, `Moments.delta_moments`
keeps `prefactor_cos`/`prefactor_sin` as required, supplied arguments.

## 1. The two Ampère projections

The Level-0 outputs are the resonant projections of the parallel current
``\bar J_\parallel(x,\xi)=\langle\sum_j e_j n_j u_{\parallel j}\rangle_\theta``
through the island (docs/01 §4; O3: Ampère is a diagnostic, not solved). The
perturbed parallel Ampère law for the resonant helical harmonic, matched to the
outer-region tearing index ``\Delta'``, and the torque (rotation) condition are
(Diss19 Eqs. 2.9–2.10)

```math
\frac{1}{\mu_0 R}\,\Delta'\,\tilde\psi = \int_{\mathbb R} d\psi\oint d\xi\;
   \bar J_\parallel\cos\xi ,
\qquad
0 = \int_{\mathbb R} d\psi\oint d\xi\;\bar J_\parallel\sin\xi .
```

The ``\cos\xi`` projection matches the growth drive to ``\Delta'``; the
``\sin\xi`` projection is the torque-balance (steady-rotation) condition. Both
are the ``m``-harmonic projections of ``\bar J_\parallel`` against the island
perturbation ``A_\parallel=-(\tilde\psi/R)\cos\xi`` (I19 Eq. 5).

## 2. The growth moment ``\Delta_{\cos}\equiv\Delta_{\rm neo}``

The Modified Rutherford Equation is the amplitude equation for the island
half-width; its stationarity (marginal island) is the balance of the outer
drive ``\Delta'`` against the kinetic inner drive. Define the kinetic growth
moment ``\Delta_{\rm neo}`` so that **stationarity reads ``\Delta'+\Delta_{\rm
neo}=0``** (docs/01 §4; Diss19 Eq. 4.12):

```math
\boxed{\;
\Delta_{\cos}\equiv\Delta_{\rm neo}
 = -\frac{\mu_0 R}{2\tilde\psi}\int_{\mathbb R} d\psi\oint d\xi\;
   \bar J_\parallel\cos\xi
\;}
```

The prefactor is fixed piece by piece — each piece sourced (the ``1/2`` adopted
from the Diss19 Eq. 4.12 convention, not re-derived from scratch):

- ``\mu_0 R`` is the geometric factor of parallel Ampère (``\nabla^2 A_\parallel
  = -\mu_0 J_\parallel``; the ``R`` from the helical/toroidal metric of I19
  Eq. 5's ``A_\parallel=-\tilde\psi\cos\xi/R``).
- ``\tilde\psi = \tfrac{w_\psi^2}{4}\,q_s'/q_s`` is the island flux amplitude —
  **already cleared** (`island_flux_amplitude`, docs/01 §1); dividing by it makes
  ``\Delta_{\rm neo}`` the drive *per unit island flux*, with units of ``\Delta'``
  (inverse length), as the MRE requires.
- The ``1/2`` is the Rutherford matching normalization: the ``\cos\xi``
  projection of the constant-``\psi`` perturbation carries the standard factor
  that makes ``\Delta_{\rm neo}`` combine additively with ``\Delta'`` in the
  amplitude equation (Diss19 Eq. 4.12 convention).

## 3. The torque moment ``\Delta_{\sin}`` — the `[DERIVED]` symmetric pin

The ``\sin\xi`` projection has no external ``\Delta'`` to match against (its
outer counterpart vanishes — the torque condition is ``0=\int\!\int\bar
J_\parallel\sin\xi``, §1). Its overall normalization is therefore a **choice**,
flagged `[DERIVED]` in docs/01 §4. **Pin it symmetrically to** ``\Delta_{\cos}``
— same magnitude prefactor, opposite sign so that ``\Delta_{\cos}+i\Delta_{\sin}``
forms the natural complex growth+torque moment that maps onto the linear-layer
``\Delta(Q)`` in the small-amplitude limit (docs/01 §4; ladder D1):

```math
\boxed{\;
\Delta_{\sin}
 = +\frac{\mu_0 R}{2\tilde\psi}\int_{\mathbb R} d\psi\oint d\xi\;
   \bar J_\parallel\sin\xi
\;}
```

so that

```math
\Delta_{\cos}+i\,\Delta_{\sin}
 = \frac{\mu_0 R}{2\tilde\psi}\int d\psi\oint d\xi\;
   \bar J_\parallel\,(-\cos\xi + i\sin\xi) .
```

The symmetric choice is what makes ``\Delta_{\sin}=0`` the clean torque-balance
root (Level-4 ``\omega_E`` closure) and the parity relations of ladder A3
(``\Delta_{\cos}`` even / ``\Delta_{\sin}`` odd under ``\xi\to-\xi``) hold with
matched normalization — both already verified structurally (A3 green). This is a
`[DERIVED: 2026-07-11]` normalization *decision*, recorded as such (policy
rule 4), not a literature transcription.

## 4. Cross-check table

| Source | Form | Agrees with `[DERIVED]`? |
|---|---|---|
| Diss19 Eq. (2.9)–(2.10) (via docs/01 §4) | the ``\cos``/``\sin`` Ampère projections | ✅ structure (§1) |
| Diss19 Eq. (4.12) | ``\Delta_{\rm neo}=-\tfrac{\mu_0 R}{2\tilde\psi}\int\!\int\bar J_\parallel\cos\xi`` | ✅ (§2) |
| docs/01 §4 | ``\Delta_{\sin}`` normalization "chosen symmetric — [DERIVED] pin at implementation" | ✅ — pinned here (§3) |
| A3 gate (green) | ``\Delta_{\cos}`` even / ``\Delta_{\sin}`` odd | ✅ consistent with the matched-normalization pin |

**Triage:** the ``\cos``-moment prefactor is derived structure + the cleared
``\tilde\psi`` + the standard Rutherford ``1/2`` (Diss19 4.12 convention); no
discrepancy. The ``\sin``-moment normalization is a `[DERIVED]` symmetric choice,
now pinned and recorded.

## 5. What sign-off authorizes

On sign-off (recorded in docs/01 §4): the prefactors
``\text{prefactor\_cos}=-\mu_0 R/(2\tilde\psi)`` and
``\text{prefactor\_sin}=+\mu_0 R/(2\tilde\psi)`` may be constructed
(``\tilde\psi`` from the cleared `island_flux_amplitude`, ``\mu_0 R`` from the
equilibrium) and passed to `Moments.delta_moments`, replacing its required gated
arguments. This closes the ``\Delta``-prefactor `[VERIFY]`/`[DERIVED]` items; the
channel decompositions (bootstrap/polarization split, ``\langle\cdot\rangle_\Omega``)
are already implemented structure.
