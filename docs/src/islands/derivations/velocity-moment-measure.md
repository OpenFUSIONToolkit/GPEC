# Derivation — the physical velocity-volume-integral measure (moments, the parallel-flow weight, and the QN density revision)

**Provenance:** `[DERIVED: 2026-07-13]` — independent re-derivation (Decision D7)
of the Level-0 velocity-moment measure, resolving QUESTIONS **Q6** (and unblocking
Q3's parallel-flow weight `W` and the Q5 momentum-restoring term F).
**Clears (on sign-off):** the physical `∫d³v` measure of
`Operators.velocity_moment!`/`weighted_moment!` (the `v̂²` speed Jacobian, the
`1/√(1−yb)` pitch Jacobian + its singular-edge treatment); the parallel-flow
weight `W = v̂_∥` (Q3, `Moments.parallel_current!` → `J̄_∥` → the `Δ` outputs); and
the **re-implementation** of the cleared QN density moment `δn̄_i = {ĝ}_v` with the
physical Jacobians (`[CHECKED: L23 Eqs. 8.4.1–8.4.4]`).
**Status:** ✅ **signed off 2026-07-13** — with the **flux-surface `b`** choice
(§7 option ii): the moment pitch integral uses `b = b_min = (1−ε)/(1+ε)` (outboard
midplane), upper limit `y = 1/b_min = (1+ε)/(1−ε)` (the deeply-trapped edge),
keeping the full trapped range. Revises the signed-off `quasineutrality-closure.md`
moment (user-approved, 2026-07-13).

## 1. Why this is needed (the Q6 finding)

The code's `velocity_moment!`/`weighted_moment!` currently use a **flat** measure
— `Σ_σ ∫dy [Simpson] ∫dE [Gauss–Laguerre, e^{−E}]` — with the Maxwellian carried
by the E-grid (`g = shape`, `e^{−E}` in `wE`; confirmed
`Configure.gradient_far_field`). But the **physical** velocity-volume integral is
(L23 Eq. 8.4.1, first-hand p. 161)

```math
\{\hat f\}_v = \pi B \sum_\sigma\int_0^\infty d\hat v\,\hat v^2
   \int_0^{1/b}\frac{\hat f\,dy}{\sqrt{1-yb}} ,
```

\noindent
which carries a **`v̂²` speed factor** and a **`1/√(1−yb)` pitch Jacobian** that the
flat measure omits. Both are velocity-dependent, so they do **not** factor out of
the moments — they re-weight the density/flow and hence the `Δ` outputs. This
clears the measure to the physical form.

## 2. The `g = shape` convention and the two Jacobians (discretized)

With `g = ĝ/F̂_M` (Maxwellian factored out; the E-grid `wE` supplies `e^{−E}`), a
physical moment of the perturbation `ĝ = e^{−E} g` is

```math
\{\hat g\}_v = \pi B \sum_\sigma\int_0^\infty d\hat v\,\hat v^2
   \int_0^{1/b}\frac{e^{-E} g\,dy}{\sqrt{1-yb}},\qquad E=\hat v^2 .
```

**Speed Jacobian.** `d\hat v\,\hat v^2 = dE\,\tfrac{\sqrt E}{2}` (since `d\hat v =
dE/2\hat v`, `\hat v^2 = E`), and `∫dE\,e^{-E}(\cdot) = \sum_i w_i^{\rm GL}(\cdot)`
(Gauss–Laguerre). So the energy integral discretizes as

```math
\int_0^\infty d\hat v\,\hat v^2\,e^{-E}(\cdot)
 = \sum_{i} w_i^{\rm GL}\,\frac{\sqrt{E_i}}{2}\,(\cdot) ,
```

\noindent
i.e. **fold a `√E/2` factor into the existing Gauss–Laguerre weight**.

**Pitch Jacobian.** `∫_0^{1/b} dy/\sqrt{1-yb}` replaces the flat `∫dy`, i.e. **fold
`1/\sqrt{1-yb}` into the Simpson `y`-weight**. Its integrand is singular as
`y\to 1/b` (`1-yb\to0`); L23 §8.4.1 (Eqs. 8.4.3–8.4.4) splits it at
`y_0 = 1/b - \delta y_0` (`\delta y_0` = the nearest mesh spacing) into a numeric
part on `(0,y_0)` and an **analytic** part on `(y_0,1/b)` assuming `f` linear
there,

```math
\int_{y_0}^{1/b}\frac{Ay+C}{\sqrt{1-yb}}\,dy
 = \frac{2\sqrt{1-y_0 b}}{3b^2}\big(A(y_0 b+2)+3bC\big)
```

\noindent
(L23 Eq. 8.4.4, the `IinvB` function). This is the standard integrable
`1/\sqrt{\cdot}` endpoint handling — the same turning-point structure as the drift
`G`-bracket, here on the **moment** rather than the orbit average.

## 3. The physical moment machinery

Collecting §2, the physical velocity moment of `g` weighted by `W(y,E,σ)` is

```math
\boxed{\;
\{W g\}_v[i_x,i_\xi] = C\sum_\sigma\sum_{i_E} w^{\rm GL}_{i_E}\,\frac{\sqrt{E_{i_E}}}{2}
   \sum_{i_y} w^{\rm phys}_{i_y}\;W\,g,
\qquad w^{\rm phys}_{i_y}=\frac{w_{i_y}}{\sqrt{1-y_{i_y}b}}\ (\text{+ }IinvB\text{ edge}) \;}
```

with `C = \pi b` a global constant (`σ`-, `y`-, `E`-independent) that folds into
the already-cleared `Δ`-prefactor `μ_0R/2\tilde\psi` and the QN normalization.
Setting `W = 1` gives the physical **density** `δn̄_i = {\hat g}_v` (the QN moment,
§5). This is the base measure that `velocity_moment!`/`weighted_moment!` implement.

## 4. The parallel-flow weight `W = v̂_∥` (Q3 cleared)

The parallel current is `J̄_∥ = Σ_j Z_j\{v̂_\parallel\,\hat g_j\}_v` with the local
parallel velocity `v̂_\parallel = σ\hat v\sqrt{1-yb} = σ\sqrt E\sqrt{1-yb}`. In the
physical base measure (§3) its `\sqrt{1-yb}` **cancels** the pitch Jacobian:

```math
\{v̂_\parallel g\}_v = C\sum_\sigma\sigma\sum_{i_E}w^{\rm GL}\frac{\sqrt E}{2}\sqrt E
   \sum_{i_y}\frac{w_{i_y}}{\sqrt{1-yb}}\sqrt{1-yb}\,g
 = C\sum_\sigma\sigma\sum_{i_E}w^{\rm GL}\frac{E}{2}\sum_{i_y}w_{i_y}\,g,
```

\noindent
so `J̄_∥` is **regular** (the singular `IinvB` piece multiplies `√(1−yb)→0` and
drops) and reduces to the flat-`y`, `σ`-odd, `∝E` form. Hence

```math
\boxed{\;W(y,E,\sigma) = \hat v_\parallel = \sigma\sqrt E\,\sqrt{1-yb}\;}
```

is the cleared parallel-flow velocity weight (Q3), consistent with the physical
measure. (Equivalently, in a *flat* base measure `W` would carry the whole `σE/2`;
here the Jacobians live in the base measure and `W` is the bare `v̂_∥`.) This
un-gates `Moments.parallel_current!` → `J̄_∥` → `Δ_cos/Δ_sin`.

## 5. The QN density moment `δn̄_i` — re-implemented physically

The cleared quasineutrality closure (`quasineutrality-closure.md`, signed off
2026-07-11) uses `δn̄_i = M[g]` "directly"; `M` was the **flat** `velocity_moment!`.
With the physical measure (§3, `W=1`) `δn̄_i = \{\hat g\}_v` now carries the
`√E/2` speed Jacobian and the `1/√(1−yb)` pitch Jacobian (`IinvB` edge). The QN
**closure algebra is unchanged** — `R_Φ = M[g] − αΦ̂ + S`, `α=(τ+1)/τ`,
`S = L̂_{n0}^{-1}(x−ĥ)` — only the moment operator `M` becomes physical. The `C=πb`
constant is absorbed into `L̂_{n0}`/the closure normalization (a global factor, not
velocity-dependent). This is the user-approved (2026-07-13) revision of
signed-off work.

## 6. Term F (momentum restoring) unblocked

`Ū_∥ᵢ(f) = (1/√π⟨ν̂_ii⟩_u) Σ_σ σ ∫du u³ν̂_ii ∫dy f` (L23 Eq. 8.3.17) is the same
physical parallel-flow moment with the extra `u³ν̂_ii/⟨ν̂_ii⟩_u` speed weight — i.e.
`Ū = \{(\hat v_\parallel\,\hat\nu_{ii}/\langle\hat\nu_{ii}\rangle_u)\,g\}_v`-type
with the cleared `⟨ν̂_ii⟩_u`. Once the physical measure (§3) and `W = v̂_∥` (§4)
land, term F becomes a bounded velocity moment (no new gated normalization) — so
this clearance also unblocks the Q5 term F.

## 7. Open point for sign-off — the `b` in the moment's pitch integral

The physical `∫_0^{1/b}dy/√(1−yb)` uses the **local** field `b(θ)`, but `g` is
orbit-averaged (no `θ`). Two readings, to confirm at sign-off:

- **(i) `b → 1` (large-aspect proxy)**, L23's own `b²→1` simplification (l. 8669):
  the moment pitch integral uses `b=1`, upper limit `y=1`, `IinvB` edge at `y_c=1`.
  Simple and consistent with the flat-`y` orbit-averaged operator; drops `O(ε)`
  poloidal variation of `b`.
- **(ii) flux-surface `b`** at a reference `θ` (e.g. outboard `b=b_{\min}`, upper
  limit `y=1/b_{\min}`): keeps the full trapped range.

**Decision (user sign-off 2026-07-13): (ii) flux-surface `b`.** The moment uses
`b = b_min = (1−ε)/(1+ε)` (outboard midplane), upper limit `y = 1/b_min =
(1+ε)/(1−ε)`, keeping the **full trapped range**. Then
`w^{\rm phys}_{i_y} = w_{i_y}/\sqrt{1-y\,b_{\min}}` with the `IinvB` singular-edge
split at `y = 1/b_{\min}`, and grid nodes `y > 1/b_{\min}` (forbidden, no
particles) carry zero moment weight. This retains the trapped contribution that
`b→1` would truncate, at the cost of a single reference `b` (the `O(ε)` poloidal
variation of `b` over the orbit is not resolved — consistent with the Level-0
ordering).

## 8. Cross-check table

| Source / check | Statement | Agrees? |
|---|---|---|
| L23 Eq. 8.4.1/8.4.2 (first-hand p. 161) | `{f}_v = πB Σ_σ ∫dv̂ v̂² ∫dy/√(1−yb)` | ✅ measure (§1–§3) |
| L23 Eqs. 8.4.3–8.4.4 (`IinvB`) | numeric+analytic split of the `1/√` pitch edge | ✅ (§2) |
| `g=shape`, `e^{−E}` in `wE` | speed Jacobian `√E/2` via Gauss–Laguerre | ✅ (§2, `gradient_far_field`) |
| `J̄_∥ = Σ Z_j\{v̂_∥ ĝ\}_v` (docs/01 §4) | `W=v̂_∥`, `√(1−yb)` cancels the pitch Jacobian | ✅ (§4) |
| cleared QN closure (`quasineutrality-closure.md`) | closure algebra unchanged; only `M` physical | ✅ (§5) |
| cleared `⟨ν̂_ii⟩_u`, term F (Eq. 8.3.17) | `Ū` is the same moment + `u³ν̂_ii` weight | ✅ unblocks F (§6) |

**Triage:** no discrepancy with L23. The one modelling choice is the `b` in the
moment (§7) — **flux-surface `b_min=(1−ε)/(1+ε)`** per the sign-off. Every factor traces to L23
Eq. 8.4.1–8.4.4; nothing guessed.

## 9. What sign-off authorizes

1. Physical measure in `Operators.velocity_moment!`/`weighted_moment!`: the
   `√E/2` speed Jacobian (folded into the Gauss–Laguerre `E`-weighting) and the
   `1/√(1−yb)` pitch Jacobian (flux-surface `b_min=(1−ε)/(1+ε)` per §7) with the `IinvB` singular-edge split
   (a small helper, L23 Eq. 8.4.4).
2. `Coefficients`/`Moments`: the parallel-flow weight `W = v̂_∥ = σ√E√(1−yb)`
   (Q3), wired into `Moments.parallel_current!` → `J̄_∥`.
3. Re-implement the QN density moment `δn̄_i` with the physical measure (§5); the
   closure algebra and `α`/`S` are unchanged. Re-run the QN regression.
4. This un-gates `J̄_∥`/the `Δ` outputs and unblocks term F (Q5), and resolves Q6.
   The `IinvB`/measure numerics ship with a convergence + a `{1}_v`-normalization
   test (the density of a Maxwellian).
