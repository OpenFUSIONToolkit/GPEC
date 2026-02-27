# Branch Status: `perf/remove_callback`

> This file is for Claude Code context when resuming work on this branch.
> Delete it before merging to develop.

---

## Full Plan: ODE Callback Removal and Parallelization

### Context

The Euler-Lagrange (EL) ODE integration in `ForceFreeStates` currently uses a `DiscreteCallback`
that fires at every integrator step to do three jobs: (1) switch tolerances near rational surfaces,
(2) apply Gaussian triangularization to prevent solution divergence, and (3) manually store psi/q/u/ud.

Because JPEC already pre-chunks the integration between singular surfaces (unlike the original Fortran
which handled singularities dynamically), the callback is no longer the right tool for any of these
jobs. This plan removes the callback entirely, leverages OrdinaryDiffEq's built-in storage, and
simplifies the tolerance and regularization logic. A second PR then adds optional parallel integration
via the Riccati reformulation.

### PR 1: Remove Callback, Use Built-in ODE Storage

**Critical files:**
- `src/ForceFreeStates/EulerLagrange.jl` — main changes (solve call, chunk loop, Gaussian reduction)
- `src/ForceFreeStates/ForceFreeStatesStructs.jl` — simplify `OdeState`, `ForceFreeStatesControl`
- `examples/*/jpec.toml` — rename config params
- `test/runtests_eulerlagrange.jl` — update tests

**Step 1: Unify tolerances** ✅ DONE
Remove `tol_r`, `tol_nr`, `crossover` from `ForceFreeStatesControl`. Add single
`eulerlagrange_tolerance`. Remove `compute_tols()` entirely.

**Step 2: Move Gaussian reduction to chunk boundaries** ✅ DONE (but has correctness bug — see below)
Remove Gaussian reduction from the callback. After each `integrate_el_region!` returns, call
`compute_solution_norms!` once as a safety check; apply reduction only if `uratio > ctrl.ucrit`
at the chunk end. At `cross_ideal_singular_surf!`, the forced reduction call is unchanged.

**Step 3: Replace manual storage with ODE built-in** ✅ DONE
Use `saveat` in the solve call instead of the callback storage loop. Removed `q_store`,
removed `resize_storage!` (exact pre-allocation from chunk structure). u_store/ud_store
layout is `(np, np, nsteps, 2)`.

**Step 4: Remove DiscreteCallback** ✅ DONE
Deleted `integrator_callback!`. Removed `cb` from the solve call. Set `save_everystep=false`.

**Step 5: Edge-weighted sub-chunking** ✅ DONE
Each inter-rational region split into `n_subchunks_per_region` sub-chunks with
`[edge_chunk_fraction, middle..., edge_chunk_fraction]` layout. Default N=3, f=0.05 gives
[5%, 90%, 5%] — concentrating norm checks near rational surfaces where solutions grow fastest.

**Verification** ❌ BLOCKED
- All 53 unit tests pass
- DIIID full-run gives `et[1] = -72.15` (target: ~1.707) — bug described below

### PR 2: Optional Riccati Parallelization (future work, not started)

Define the impedance matrix `W(ψ) = U₂ U₁⁻¹`. It satisfies:
```
dW/dψ = C + DW − WA − WBW
```
where A, B, C, D are N×N blocks of the EL coefficient matrices. This is N×N instead of 2N×N.

**Key parallelism**: Forward impedance `W⁺(ψ)` (from axis) and backward admittance `W⁻(ψ)`
(from edge backward) are independent and can run on separate threads. Stability criterion
computed from `W⁺` and `W⁻` at each ψ — following Glasser 2018 §III.

**Critical requirement**: Full U₁, U₂ profiles must be reconstructed for PerturbedEquilibrium
compatibility. Given W⁺(ψ), solve the reduced N×N equation `dU₁/dψ = (A + B·W⁺)·U₁`, then
U₂ = W⁺·U₁. This replaces the 2N×N pass and gives identical u_store output.

**New file**: `src/ForceFreeStates/Riccati.jl`
**Modify**: `EulerLagrange.jl` (parallel dispatch branch), `ForceFreeStatesStructs.jl`
**New tests**: `test/runtests_riccati.jl`
**Reference**: `docs/resources/2018-Glasser-A Riccati solution for the ideal MHD plasma response...pdf`

---

## Current Status

---

## What Is Complete

- Removed `DiscreteCallback` and `DiffEqCallbacks` dependency entirely
- Unified `tol_r`/`tol_nr`/`crossover` → single `eulerlagrange_tolerance`
- Replaced callback storage with ODE built-in `saveat` + exact pre-allocation
- Removed `q_store` (available on-the-fly from `equil.profiles.q_spline`)
- Removed `resize_storage!` (exact size computed from chunk structure upfront)
- Implemented edge-weighted sub-chunking (`n_subchunks_per_region`, `edge_chunk_fraction`):
  - N=3, f=0.05 → [5%, 90%, 5%] of region width per inter-rational region
  - Concentrates norm checks near rational surfaces where solutions grow fastest
- Post-crossing norm baseline: after each `cross_ideal_singular_surf!`, call
  `compute_solution_norms!` to set fresh `unorm0` for the next region
- Updated all `jpec.toml` example files with new parameters
- **53 unit tests in `runtests_eulerlagrange.jl` all pass**

---

## Current Blocker: DIIID `et[1]` Wrong

**Target**: `et[1] ≈ 1.707` (develop branch baseline)
**Actual**: `et[1] = -72.15`

DIIID crossing output (`eulerlagrange_tolerance=1e-7`, `ucrit=1e4`, `n_subchunks_per_region=3`):
```
ψ = 0.551,  q= 2.000,  uratio = 2.81e+00,  steps = 705
ψ = 0.811,  q= 3.000,  uratio = 1.07e+07,  steps = 933   ← problem
ψ = 0.926,  q= 4.000,  uratio = 2.33e+04,  steps = 1075
ψ = 0.986,  q= 5.000,  uratio = 2.72e+03,  steps = 1208
ψ = 0.991,  q= 5.200,  uratio = 1.15e+01,  steps = 1261
```

### Root Cause Hypothesis

`uratio = 1.07e7` at q=3 means `unorm0` is stale. The right-edge chunk just before the
q=3 crossing fires an intermediate Gaussian reduction (uratio > ucrit=1e4), setting
`odet.new=true`. The pre-crossing reset in `eulerlagrange_integration` then clears
`new=false` WITHOUT updating `unorm0`. At the q=3 crossing, `compute_solution_norms!`
computes `unorm /= unorm0` where `unorm0` is still the post-q=2 baseline → ratio = 1.07e7.

The sort order at the crossing may be correct (the resonant m=3 column should have grown
the most from the q=2 baseline), but `et[1] = -72` is catastrophically wrong, suggesting
either:

1. **fixstep boundary bug in `transform_u!`** (primary suspect): When the right-edge chunk
   fires an intermediate reduction AND then the crossing fires another reduction, the
   crossing's stored step (at `odet.step` before incrementing) may fall in the wrong
   transform region. `transform_u!` assigns steps to regions using `fixstep[ifix]` as
   the boundary. If the intermediate reduction's fixstep overlaps with the crossing point,
   the crossing step gets the wrong cumulative transformation matrix.

2. **Wrong column zeroed at crossing**: The Gaussian sort at ucrit=1e4 then again at the
   crossing may cause the wrong column to be eliminated, leaving the resonant solution in
   the matrix.

---

## Recommended Next Steps (in order)

### Step 1: Quick diagnostic — raise ucrit

In `examples/DIIID-like_ideal_example/jpec.toml`, set:
```toml
ucrit = 1e8   # suppress all intermediate reductions
```

Run the DIIID example:
```bash
julia --project=. -e 'using JPEC; JPEC.main(["examples/DIIID-like_ideal_example/jpec.toml"])' 2>&1
```
Then read et[1]:
```julia
using HDF5
h5open("examples/DIIID-like_ideal_example/jpec.h5", "r") do f
    println(read(f["vacuum/et"])[1])
end
```

**If et[1] ≈ 1.707**: the intermediate-reduction `fixstep` boundary is the bug.
The Gaussian sort at crossings is correct; only the `transform_u!` region assignment
for intermediate reductions is broken.

**If et[1] still wrong**: the issue is in the crossing Gaussian sort itself
(stale unorm0 causing wrong column to be sorted to the front).

### Step 2 (if intermediate reductions are the bug): Fix fixstep boundaries

The fix is likely that the crossing step (`odet.step` at crossing time, stored in
`cross_ideal_singular_surf!`) must always fall AFTER `fixstep[ifix]` for any intermediate
reduction that fired in the preceding chunk. Verify that `odet.fixstep[ifix] < odet.step`
holds when the crossing stores its step, and that the next region in `transform_u!` is
correctly bounded.

Add temporary debug output to `transform_u!`:
```julia
println("ifix=$ifix: steps $jfix..$(kfix), fixstep=$(odet.fixstep[ifix]), total step=$(odet.step)")
```

### Step 3 (if crossing sort is the bug): Fix unorm0 at crossing

The pre-crossing logic currently does:
```julia
odet.new = false
cross_ideal_singular_surf!(...)
compute_solution_norms!(odet.u, odet, ctrl, intr, false)  # post-crossing baseline
```

If `new=true` was set by an intermediate reduction, clearing it without updating `unorm0`
means the crossing sort uses stale growth ratios. The fix would be to update `unorm0`
before clearing `new`:
```julia
if odet.new
    odet.unorm .= norm.(eachcol(odet.u[:, :, 1]))
    odet.unorm0 .= odet.unorm  # update baseline before clearing new
end
odet.new = false
cross_ideal_singular_surf!(...)
```
But this was tried before and caused issues — re-examine carefully.

### Step 4: Verify Solovev is correct

Solovev et[1] from last run: `-0.277 + 0.003im` — needs comparison with develop baseline.
```bash
git stash
git checkout develop
julia --project=. -e 'using JPEC; JPEC.main(["examples/Solovev_ideal_example/jpec.toml"])' 2>&1
# read et[1] from jpec.h5 -> vacuum/et
git checkout perf/remove_callback
git stash pop
```

---

## Key Files

- `src/ForceFreeStates/EulerLagrange.jl` — main integration logic
- `src/ForceFreeStates/ForceFreeStatesStructs.jl` — control params (OdeState, ForceFreeStatesControl)
- `examples/DIIID-like_ideal_example/jpec.toml` — test case (compute_response=false while debugging)
- `test/runtests_eulerlagrange.jl` — unit tests (all 53 pass)

## Known Pre-existing Bug (unrelated to this PR)

`ResponseMatrices.jl:61` (`extract_boundary_displacements`): BoundsError when
`compute_response=true` in DIIID toml. This is a pre-existing bug using old `u_store`
indexing convention. Temporarily disabled via `compute_response=false` in DIIID toml.
Fix separately after PR 1 lands.
