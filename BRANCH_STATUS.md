# Branch Status: `perf/remove_callback`

> This file is for Claude Code context when resuming work on this branch.
> Delete it before merging to develop.

## Goal

PR 1 of the ODE Callback Removal and Parallelization plan. Remove the `DiscreteCallback`
from Euler-Lagrange integration and replace it with chunk-boundary Gaussian reduction,
ODE built-in `saveat` storage, and edge-weighted sub-chunking.

Full plan: `~/.claude/plans/binary-whistling-llama.md` (local) — key details reproduced below.

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
