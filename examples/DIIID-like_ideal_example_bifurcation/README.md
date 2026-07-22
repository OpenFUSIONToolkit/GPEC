# DIII-D-like 2/1 → 4/2 magnetic-island bifurcation

A self-contained demonstration that the `FieldLineTracing` module reproduces the RMP
**island bifurcation** seen in DIII-D rotating-RMP experiments: at the q=2 surface a **2/1
island (2 O-points)** gives way to a **4/2 island (4 O-points)** and back, as the competing
`(m,n) = (2,1)` and `(4,2)` resonances trade dominance.

This example is **synthetic** — it uses the DIII-D-like equilibrium (`TkMkr_D3Dlike_Hmode.geqdsk`,
q=2 at ψ_N ≈ 0.518) and **only the I-coils** (`iu`, `il`), with engineered currents. No intrinsic
error field is needed.

## The mechanism

Each 6-coil array is driven with

```
I(β) = A₁·cos(β)·[n=1 pattern] + A₂·sin(β)·[n=2 pattern]
   n=1 pattern = cos(60°·j) = [1, 0.5, −0.5, −1, −0.5, 0.5]   → resonant (2,1) at q=2
   n=2 pattern = cos(120°·j) = [1, −0.5, −0.5, 1, −0.5, −0.5]  → resonant (4,2) at q=2
```

The n=1 and n=2 drives are anti-correlated so each state has a single dominant resonance —
sweeping `β` trades the (2,1) for the (4,2) and back:

| β | I-coil field | q=2 island | O-points |
|---|---|---|---|
| 0 | pure n=1 | 2/1 | **2** |
| π/2 | pure n=2 | 4/2 | **4** |
| π | pure n=1 (flipped) | 2/1 | **2** |

(The real 196073 shot instead holds the n=2 roughly constant and nulls the n=1; here the
drives are anti-correlated so the 2/1 and 4/2 states are cleanly separated.)

This is the same competition that drives the bifurcation in shot 196073 (there the fixed
C-coil + intrinsic n=1 is nulled by the rotating I-coil n=1 while the n=2 persists); here it is
produced deliberately with I-coil currents alone.

The vacuum field-line trace resolves the **multi-n** coil field (`flux_n_max = 3`), so both
resonances appear directly in the Poincaré section — a single-n trace would only ever show the
2/1 island.

## Files

- `gpec.toml` — headline case (β=π/2, pure n=2 → 4/2 island), I-coil only, vacuum flux tracing
  zoomed on the q=2 surface.
- `run_bifurcation.jl` — sweeps β ∈ {0, π/2, π}, runs the pipeline for each, and writes the
  figures below.
- `TkMkr_D3Dlike_Hmode.geqdsk` — the equilibrium (copied from `DIIID-like_ideal_example`).

## Run

```bash
julia --project=. examples/DIIID-like_ideal_example_bifurcation/run_bifurcation.jl
```

Figure outputs:

- `bifurcation_2to4to2.png` — the headline 3-panel Poincaré (θ vs ψ_N) showing **2 → 4 → 2**
  O-points at the q=2 surface.
- `2over1_a/`, `4over2/`, `2over1_b/` — one directory per β with the run's `gpec.h5` and the
  standard `FieldLineTracing` figures: `poincare_RZ_*`, `poincare_flux_*`, `island_widths_*`,
  and the `summary_*` composite.

This is a `TEMP`-style demonstration example; generated outputs are not committed.
