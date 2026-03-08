# Direct Equilibrium Improvements: Implementation Plan

> **Note for PR reviewers**: This file documents the implementation plan for the
> `direct_equilibrium_improvements` branch. It should be **removed before merging**
> into develop, since the PR description and commit history capture the same information.

## Context

The current `equilibrium_solver` for EFIT inputs uses a geometric-angle parameterized
field-line ODE (`direct_fieldline_int`) to trace flux surfaces. This approach has two
known failure modes that limit robustness near the separatrix (psihigh → 1):

1. **Denominator singularity**: The ODE contains a term
   `denominator = Bz·cos(η) − Br·sin(η)` that vanishes when the poloidal field is
   oriented radially — common near the top/bottom of elongated plasmas close to the
   separatrix. When it fires, the code emits a warning and returns `dy = 1e20`, leaving
   the adaptive solver to cope. The current default `psihigh = 0.993` avoids this region.

2. **`direct_refine` drift**: At every ODE step, a Newton iteration corrects the
   radial distance `rfac` to keep the trajectory on the target ψ surface. This correction
   accumulates round-trip errors and is expensive (two extra spline evaluations per step).

## Branch: `direct_equilibrium_improvements`

Branched from `develop`. Contains two new equilibrium solver strategies selectable via
`eq_type` in `gpec.toml`, plus benchmarking scripts to compare all three methods.

## New `eq_type` values

| `eq_type`              | Source file                            | Method                          |
|------------------------|----------------------------------------|---------------------------------|
| `"efit"`               | `DirectEquilibrium.jl` (unchanged)     | Original geometric-angle ODE    |
| `"efit_arclength"`     | `DirectEquilibriumArcLength.jl`        | Strategy A: arc-length ODE      |
| `"efit_by_inversion"`  | `DirectEquilibriumByInversion.jl`      | Strategy B: contour → inverse   |

All other `eq_type` values (`chease`, `chease2`, `lar`, `sol`) are unaffected.

---

## Strategy A: Arc-Length ODE (`efit_arclength`)

### Core idea

Replace the η-parameterized ODE with an arc-length-parameterized level-set ODE.
The RHS follows the direction tangent to the ψ = const level set:

```
dR/ds = +∂ψ/∂Z / |∇ψ|
dZ/ds = −∂ψ/∂R / |∇ψ|
```

The point stays on the flux surface by construction — `direct_refine` is eliminated.
The denominator singularity disappears because arc-length parameterization is always
well-defined wherever |∇ψ| > 0.

### ODE state vector

`u = [R, Z, ∫dl/Bp, ∫dl/(R²Bp), ∫jac·dl/Bp]`

Initial: `u₀ = [R_start, zo, 0, 0, 0]` where `R_start` is found by the same
Newton solve on the outboard midplane as the current code.

### Termination

A `ContinuousCallback` on `Z − zo = 0` (midplane crossing), with:
- `affect!` (upward crossing): terminates if `t > t_min` (guards against spurious t=0 fire)
- `affect_neg!` (downward crossing): does nothing (inboard midplane pass-through)

`t_min = π * (R_start − ro)` — estimated half-circumference as a lower bound guard.

### Output format

After solving, `y_out` is reconstructed in the same 5-column format that the
existing solver loop consumes from `direct_fieldline_int`:

- `y_out[:, 1]` = geometric angle η ∈ [0, 2π] (unwrapped from atan2)
- `y_out[:, 2]` = accumulated `∫dl/Bp`
- `y_out[:, 3]` = rfac = `√((R−ro)² + (Z−zo)²)`
- `y_out[:, 4]` = accumulated `∫dl/(R²Bp)`
- `y_out[:, 5]` = accumulated `∫jac·dl/Bp`

This identical format means the entire grid-construction loop (`equilibrium_solver_arclength`)
can reuse all the spline-fitting and profile-extraction logic from the original solver.

### What is eliminated

- `direct_fieldline_der!` (replaced by `arclength_fieldline_der!`)
- `direct_refine` (gone entirely — no longer needed)
- The denominator `Bz·cos(η) − Br·sin(η)` (replaced by `|∇ψ|` which is non-singular
  away from the O-point)

---

## Strategy B: Contour tracing → InverseRunInput (`efit_by_inversion`)

### Core idea

1. Evaluate the bicubic spline ψ(R,Z) on a fine Cartesian grid (4× EFIT resolution)
2. Use `Contour.jl` (marching squares) to trace closed level-set curves at each target ψ
3. Resample each closed curve to a uniform geometric-angle grid → R(ψ,θ), Z(ψ,θ)
4. Build an `InverseRunInput` from these tables
5. Call the existing `equilibrium_solver(::InverseRunInput)` — zero new physics code

### Why this is robust near the separatrix

As psihigh → 1, Contour.jl naturally handles the topology change: the inner plasma
contour remains a well-traced closed curve, while open curves that touch the domain
boundary are cleanly discarded. The only degradation is when the target ψ surface
is smaller than the fine grid cell size (near axis) or when it intersects the
domain boundary.

### Grid refinement factor

The fine evaluation grid is `refine × nw` × `refine × nh`. Default: `refine = 4`.
The benchmark numerical-parameter scan characterizes accuracy vs. cost for
refine ∈ {2, 4, 8, 16}.

### Near-axis fallback

For surfaces where psifac < `psilow_contour_threshold` (configurable, default 5e-3),
the contour radius is smaller than the fine grid resolution and Contour.jl may fail
to trace a closed curve. These surfaces fall back to a circular approximation:
`R ≈ ro + rfac·cos(θ)`, `Z ≈ zo + rfac·sin(θ)` with `rfac = √psifac * (rs2 − ro)`.
This is also what the direct ODE implicitly assumes near the axis.

### Curve selection

`select_plasma_contour` identifies the correct closed curve among all curves returned
by Contour.jl at a given ψ level:
1. Filter for closed curves: first vertex ≈ last vertex within tolerance
2. Point-in-polygon (ray casting) test to find the curve containing (ro, zo)
3. If no closed curve contains (ro, zo): error — psihigh has exceeded the separatrix

### Angle resampling

`resample_contour_to_theta_grid`:
1. Compute geometric angles: `η_k = mod(atan(Z_k − zo, R_k − ro), 2π)`
2. Sort by η to get CCW-ordered sequence
3. Append first point at η + 2π to enforce periodicity
4. Build cubic spline R(η), Z(η) with `CubicFit` BC
5. Sample at `θ_j = 2π * j / mtheta` for the `InverseRunInput` theta grid

### Note on q-profile

The inverse solver **recomputes q** from the surface integrals of the traced contours.
The q-profile stored in the EFIT file (column 3 of `sq_in`) is not used for q in this
path. This means the by-inversion method derives q from the actual flux surface
geometry rather than from the EFIT polynomial extrapolation — which is more physically
consistent near the separatrix where the EFIT q-profile is unreliable.

---

## Accuracy Benchmarking Strategy

### Ground truth caveats

**For ψ < 0.95** (midrange): The EFIT q-profile is a valid reference — use it for
cross-validation. Similarly for F(ψ) and P(ψ).

**For ψ > 0.95** (far edge): The EFIT q-profile is **not** a valid reference in a
diverted tokamak (DIIID). The EFIT polynomial extrapolates to a finite `q_a`, whereas
the true q → ∞ as ψ → 1 (separatrix). **Do not use the EFIT q-profile as ground truth
in the far edge.** Instead, use:
- Cross-method agreement (inter-method differences as an ensemble)
- Self-consistency metrics (GSE residual, Jacobian consistency, roundtrip error)
- Monotonicity of q(ψ) near the edge (physically required for diverted tokamak)

### Self-consistency metrics (valid everywhere, no external reference needed)

1. **GSE residual**: max and mean of |Δ_GS|/max(|terms|) from `equilibrium_gse!`
2. **Roundtrip error**: (ψ,θ) → (R,Z) → ψ_spline(R,Z); residual = |ψ_spline − ψ_target|
3. **Jacobian consistency**: compare `rzphi_jac` to Jacobian computed from metric
   coefficients of `rzphi_rsquared` and `rzphi_offset` derivatives
4. **q monotonicity near edge**: count non-monotone steps in q(ψ) for ψ > 0.95

---

## Benchmark Scripts

All scripts are standalone Julia scripts, located in `benchmarks/`. They use the
DIIID-like example by default (path hardcoded or passed as command-line argument).
Results are written to CSV files and plots saved as PNG.

### `benchmarks/equil_method_comparison.jl`

Fixed default parameters. Compares all three methods on:
- Runtime (3 warm runs, average last 2)
- q(ψ) deviation from EFIT input (valid for ψ < 0.95)
- Cross-method q differences (for ψ > 0.95)
- GSE residual (full domain and restricted to ψ > 0.90)
- Roundtrip error vs. ψ_norm
- Jacobian consistency

### `benchmarks/equil_psihigh_scan.jl`

```
psihigh_values = [0.980, 0.985, 0.990, 0.993, 0.995, 0.996, 0.997, 0.998, 0.999]
```

For each (method, psihigh):
- Success/failure (try/catch)
- Runtime
- Max GSE residual in outer 10% (ψ > 0.90)
- Max roundtrip error in outer 10%
- q(ψ) panel for ψ ∈ [0.90, psihigh] (all methods, each psihigh)
- q monotonicity violations count

**Key diagnostic**: The q-vs-ψ panel at each psihigh shows whether methods correctly
resolve the divergence of q toward the separatrix (expected: q grows steeply), vs.
the flat EFIT polynomial extrapolation.

### `benchmarks/equil_psilow_scan.jl`

```
psilow_values = [1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4]
```

For each (method, psilow):
- Success/failure
- q0 extrapolation to axis vs. EFIT header value `simag`/`qaxis`
- Max GSE residual for ψ < 0.10
- Roundtrip error for ψ < 0.10
- For `efit_by_inversion`: report smallest psilow where Contour.jl traces closed curve
  (contour fallback threshold)

### `benchmarks/equil_numerical_params.jl`

Parameter sweeps for accuracy vs. cost characterization:

| Parameter         | Values              | Methods affected         |
|-------------------|---------------------|--------------------------|
| `mpsi`            | 64, 128, 256, 512   | all                      |
| `mtheta`          | 128, 256, 512       | all                      |
| ODE `reltol`      | 1e-5, 1e-7, 1e-9   | `efit`, `efit_arclength` |
| Grid refinement   | 2, 4, 8, 16 ×      | `efit_by_inversion`      |

Reports runtime and max-GSE for each combination. Output: Pareto-frontier plot
(runtime vs. max-GSE) for each method.

---

## Files Changed

| File | Status | Notes |
|------|--------|-------|
| `src/Equilibrium/DirectEquilibrium.jl` | **Unchanged** | Original path untouched |
| `src/Equilibrium/InverseEquilibrium.jl` | **Unchanged** | Reused by Strategy B |
| `src/Equilibrium/DirectEquilibriumArcLength.jl` | **New** | Strategy A |
| `src/Equilibrium/DirectEquilibriumByInversion.jl` | **New** | Strategy B |
| `src/Equilibrium/Equilibrium.jl` | Modified | Added includes + dispatch |
| `Project.toml` | Modified | Added `Contour.jl` dependency |
| `benchmarks/equil_method_comparison.jl` | **New** | Accuracy + runtime comparison |
| `benchmarks/equil_psihigh_scan.jl` | **New** | Edge robustness scan |
| `benchmarks/equil_psilow_scan.jl` | **New** | Near-axis scan |
| `benchmarks/equil_numerical_params.jl` | **New** | Parameter sensitivity |
| `docs/direct_equilibrium_improvements_plan.md` | **New** | **Remove before merging** |

---

## Picking Up in a New Session

The branch is `direct_equilibrium_improvements`. All implementation is in the files
listed above. The two new solver functions are:

- `equilibrium_solver_arclength(raw_profile::DirectRunInput)` in `DirectEquilibriumArcLength.jl`
- `equilibrium_solver_by_inversion(raw_profile::DirectRunInput)` in `DirectEquilibriumByInversion.jl`

Both return a `PlasmaEquilibrium` identical in structure to the original.
The dispatch is in `setup_equilibrium` in `Equilibrium.jl`.

If resuming after partial implementation, check the TODO markers in the new files
for what remains incomplete.
