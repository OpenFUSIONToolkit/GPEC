# `delta_coil` truncation & near-rational sensitivity study

Sensitivity of the STRIDE/Riccati **`delta_coil`** (outer small-solution coefficient at each rational
surface, driven by unit edge/coil modes; `singular/delta_coil_matrix`) to two numerical knobs:

- **`dmlim`** — the outer-domain truncation point, active only when `set_psilim_via_dmlim = true`
  (`psilim = psi(q_last + dmlim)`).
- **`singfac_min`** — the fractional distance from a rational surface at which the ideal jump is
  enforced (how close the integration approaches the rational).

Case: `examples/DIIID-like_gal_resistive_pe_example` (DIII-D-like, n=1, rationals q=2,3,4,5, `qmax=5.426`).

## Two metrics, per singular surface

`delta_coil_matrix` is `(2·msing, ncoil)`: each singular surface is **two rows** (its L/R small
solutions). A surface's response vector `v_s` is that `2×ncoil` block flattened. We report, against a
**nominal** run (the case as-shipped: `set_psilim_via_dmlim=false`, `dmlim=0.2`, `singfac_min=1e-4`):

1. **Norm** `‖v_s‖` — magnitude sensitivity.
2. **Cosine similarity** `|⟨v_nom, v_s⟩| / (‖v_nom‖·‖v_s‖)` — phase-invariant normalized dot product.
   `1.0` = identical pattern (only rescaled); `<1` = the coil-response **pattern rotated** — a change
   the norm cannot see.

All runs use `gal_match_flag = false`: `delta_coil` is produced by the rpec edge loop independently of
the resistive match, so there is no inner-layer `msing`/array guard and `dmlim` is free to change the
surface count. Results are aligned **by q-value**, so a surface appearing/disappearing is handled cleanly.

## Key findings

- **`singfac_min` is inert.** Norm and cosine are identical to 5–6 significant figures across
  `1e-5 … 1e-3` at every surface including the edge — `delta_coil` is independent of the near-rational
  approach distance in both magnitude and pattern.

- **Integration tolerance is inert too.** Sweeping `eulerlagrange_tolerance` over `1e-8 … 1e-12` leaves
  `delta_coil` bit-identical — norms, cosine (= 1.000000), and the full singular-value spectrum unchanged.

- **SVD confirms good conditioning.** The `delta_coil` matrix is well-conditioned (cond ≈ 26, σ₁ ≈ 29,
  σ_min ≈ 1.1). Below the `dmlim` threshold the singular-value spectrum and the leading left singular
  vector are stable (`u₁·u₁_nom` = 0.998–1.000); at `dmlim = 0.5` the matrix drops to rank-6 (q5 gone),
  so its norm and condition number fall (cond → 11) — the discrete surface loss shows cleanly in the SVD.

- **`dmlim` is the dominant sensitivity — and it is a *pattern-level* effect the norm hides.** Under the
  `dmlim` sweep the norms of q2/q3/q4 barely move (~5–16%), which looks robust; but the cosine with the
  nominal drops sharply — e.g. at `dmlim=0.5` the q4 pattern is nearly orthogonal to nominal
  (cosine ≈ 0.17, ~80° rotation) at ~16% norm change. **A norm-only scan would falsely call this robust.**

- **The truncation rule drops the edge rational at a sharp threshold.** `dmlim` keeps the outermost
  rational `q_r` only if `q_r + dmlim < qmax`. Here q=5 survives only for **`dmlim < qmax − 5 = 0.426`**.
  Above that, q5 is dropped entirely (edge becomes q4), and removing it reorganizes the coil response at
  every remaining surface — worst right at `dmlim ≈ 0.5`, partially realigning as `dmlim` grows further.

- **The collapse is a STEP at the threshold, not a smooth rotation** (fine sweep, 0.30–0.50). Just below
  (`dmlim=0.42`) every surface's cosine is `0.9999+` — the truncation sits at `qmax`, i.e. the full
  nominal domain. The instant q5 drops (`dmlim=0.44`) the cosine collapses discontinuously: q4 → 0.104
  (near-orthogonal), q3 → 0.55, q2 → 0.78. Above the threshold it only slowly realigns (q4: 0.10 → 0.17
  by `dmlim=0.5`). So the pattern change is driven by the *discrete* loss of the surface, not a gradual
  boundary shift.

- Practical guidance: for this diverted-style case the shipped default `set_psilim_via_dmlim=true,
  dmlim=0.2` sits comfortably below the 0.426 threshold (all four rationals retained). Choosing
  `dmlim` near/above the threshold both changes which surfaces exist and abruptly rotates the response
  pattern — avoid parking `dmlim` near `qmax − floor(qmax)`.

## Layout

```
scripts/   deltacoil_metrics.jl        norm + cosine-vs-nominal, per surface (dmlim & singfac sweeps)
           deltacoil_svd_tol.jl        SVD (singular values + vectors) + integration-tolerance scan
           deltacoil_sensitivity.jl    per-surface norms only (earlier, simpler version)
           plot_deltacoil_metrics.py   cosine-similarity plots
           plot_deltacoil_dmlim_fine.py  fine dmlim sweep plot
           plot_deltacoil_svd.py       SVD spectrum + tolerance-invariance plot
           plot_deltacoil_sensitivity.py  norm plots
results/   *_result.txt                raw scan tables (incl. deltacoil_svd_tol_result.txt)
figures/   deltacoil_metrics.png       cosine similarity vs dmlim / singfac
           deltacoil_sensitivity.png   |delta_coil| vs dmlim / singfac
           deltacoil_dmlim_fine.png    fine dmlim sweep (0.30–0.50) through the q5-drop threshold
           deltacoil_svd.png           singular-value spectrum vs dmlim + tolerance invariance
```

## Reproduce

```bash
cd <repo-root>
# coarse dmlim + singfac sweeps:
julia --project=. benchmarks/deltacoil_sensitivity/scripts/deltacoil_metrics.jl \
      examples/DIIID-like_gal_resistive_pe_example
# fine dmlim sweep only (skip singfac), custom dmlim points:
SKIP_SINGFAC=1 julia --project=. benchmarks/deltacoil_sensitivity/scripts/deltacoil_metrics.jl \
      examples/DIIID-like_gal_resistive_pe_example 0.30 0.35 0.40 0.42 0.44 0.46 0.48 0.50
# then plot (edit the embedded values to match a new run):
python3 benchmarks/deltacoil_sensitivity/scripts/plot_deltacoil_metrics.py
```

Each sweep point is one GPEC run (~1–2 min). Scans use temp dirs and never overwrite the example's
own HDF5 outputs; the galerkin reference `gpec.h5` is never touched.
