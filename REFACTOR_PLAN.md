# GPEC Julia — ForceFreeStates modularization & `solve(eq, integrator)` API
## Complete multi-PR implementation plan

> **NOTE FOR ALL DEVELOPERS (read this first).**
> This document is the agreed, in-progress plan for a five-PR refactor of the
> ForceFreeStates ↔ PerturbedEquilibrium interface and the top-level driver. It is
> committed directly to `develop` (deliberately, as documentation only — no code
> changes ride with it) so everyone with open PRs can see what is coming and where it
> will touch their work. Key coordination points:
>
> - The PR sequence below assumes **nothing else merges into `develop` before it
>   finishes**. If your PR must land mid-sequence, talk to Matthew first.
> - The HDF5 schema PRs (#363/#364) are treated as guidance on the final HDF5 shape;
>   writers refactored here keep today's dataset paths and will be re-targeted by
>   those PRs afterward. #367 (immutable control structs) merges after this sequence.
> - The regression harness will be re-baselined during this work; a fresh
>   Fortran-agreement comparison is the final validation gate for the whole sequence.
> - The "Decision record" and "Verified code facts" sections are settled — please do
>   not re-litigate them in PR review unless you find a factual error.
> - This file is temporary: it gets checked off as PRs merge and is deleted once the
>   sequence is implemented and vetted.

---

## 1. Context

GPEC's pipeline runs through one ~520-line monolith (`main_from_inputs`,
`src/GeneralizedPerturbedEquilibrium.jl:164`) interleaving equilibrium setup, mode
resolution, singular-surface handling, local stability, matrix assembly, three
integrator code paths, the Δ′ BVP, the Galerkin solve, HDF5 writing, and the
PE → KineticForces → SLAYER stages, communicating through ~8 loose objects.

Target UX:

```julia
eq  = PlasmaEquilibrium("input.geqdsk"; jac_type="hamada")
ffs = solve(eq, Riccati(); nn=1, delta_mlow=8, delta_mhigh=8, vac_flag=true)
rmp = RMPField("coils.dat")
pe  = perturbed_equilibrium(ffs, rmp)
# calculate_quantities(...) is OUT OF SCOPE (later deliverable)
```

### Decision record (settled — do NOT re-litigate)

| # | Decision |
|---|---|
| D1 | Three integrators = three formalisms: **Forward** (serial EL; rename all misuses of "shooting"), **Riccati** (the STRIDE FM-chunk driver currently behind `use_parallel`), **Galerkin** (RDCON; becomes fully standalone). |
| D2 | Riccati uses whatever threads `julia -t` provides. Its ONLY tunable is **number of chunks** (`nchunks`). `parallel_threads` is deleted. Outputs must be independent of thread count ⇒ the auto chunk count derives from problem structure only, never `Threads.nthreads()`. |
| D3 | **No merging of two integration results.** `populate_dense_xi` + `_populate_dense_xi_via_serial_el!` + the standalone serial-Riccati path (`riccati_eulerlagrange_integration`) are deleted FIRST (PR 1). Riccati-fed PE warn-and-skips profile-based outputs until the separate Frobenius-reconstruction work (not in this plan) restores them from `delta_coil` + surface asymptotics. |
| D4 | Kinetic (`kinetic_factor > 0`) is Forward-only. `solve`/driver raises a clear error for Riccati+kinetic and Galerkin+kinetic. |
| D5 | One result struct **`ForceFreeStatesResult`**; optional fields are `Union{Nothing,T}`; consumers use a `require(...)` helper → `@warn` + skip. |
| D6 | Local stability (Ballooning.jl) → new top-level module **`LocalStability`**, depending only on Equilibrium (+ math deps). Only ctrl dependency is `verbose` → kwarg. |
| D7 | Public API via **CommonSolve.jl**: `solve(eq::PlasmaEquilibrium, alg; kwargs...)`. `PlasmaEquilibrium(path; kwargs...)` constructor. Module names unchanged. |
| D8 | Galerkin standalone computes its own vacuum `wv` (no ODE state needed — verified); its result has `free_boundary = nothing`. |
| D9 | TOML: new `integrator = "forward"|"riccati"|"galerkin"` key. Old keys (`use_riccati`, `use_parallel`, `parallel_threads`, `populate_dense_xi`, later `gal_flag`) go to the `_DEPRECATED_FFS_KEYS` warn-and-ignore list AND the `toml-no-deprecated-keys` pre-commit hook pattern. |
| D10 | No back-compat burden; examples/fixtures updated freely; regression re-baselining accepted (HDF5 schema PRs #363/#364 churn it anyway). Final validation = fresh Fortran comparison after the sequence. Assume nothing else merges first. |
| D11 | New structs immutable from day one (eases the later #367 merge). HDF5 writers become functions on result structs (schema itself unchanged here; #363/#364 re-target them later). |
| D12 | Analysis module reads HDF5 files, not live structs — untouched except where dataset names would change (they don't in this plan). |

### Verified code facts the workers must not re-derive

- Dispatch today: `eulerlagrange_integration` (`src/ForceFreeStates/EulerLagrange.jl:151`):
  `use_parallel` → `parallel_eulerlagrange_integration` (Riccati.jl:1647; returns
  `(odet, propagators, chunks, S_at_surface_left)`), `use_riccati` →
  `riccati_eulerlagrange_integration` (Riccati.jl:1312; being deleted), else
  `serial_eulerlagrange_integration` (EulerLagrange.jl:172).
- The Δ′ BVP (`compute_delta_prime_matrix!`, Riccati.jl:274) is called once from
  `src/GeneralizedPerturbedEquilibrium.jl:470-478`, only when propagators exist.
  Active assembly = `_assemble_bvp_S_axis` (Riccati S states); `_assemble_bvp_FM_axis`
  is a never-used fallback. `_solve_bvp_edge_coil` fills `intr.delta_coil` when
  S-axis && `wv !== nothing`.
- Serial-Riccati `u_store` is NOT usable as ξ (renorm right-multiplications never
  recorded/undone); it also leaves `u_store_el_basis == true` (foot-gun; dies with the
  path).
- `populate_dense_xi` = re-run `serial_eulerlagrange_integration` and splice
  (`_populate_dense_xi_via_serial_el!`, Riccati.jl:1930). The "Riccati-gauge ca needed
  by SingularCoupling" comment there is STALE — **PE never reads `ca_l/ca_r`**.
- `balance_integration_chunks` (EulerLagrange.jl:79) sizes chunks with
  `target_n = max(2*msing+3, 4*effective_threads, 8*(msing+1)+msing)` — the middle
  term must go (D2).
- PE reads of OdeState: `u_store` (BOTH components), `du_store` (dense), `xi_s_store`,
  `psi_store`, `q_store`, `step`, `du_store_populated`. NOT `ca_l/ca_r`, `crit_store`,
  `edge_scan`. PE calls `materialize_derivative_stores!` itself
  (`src/PerturbedEquilibrium/PerturbedEquilibrium.jl:88`) and discards the Bool.
- PE reads of `ForceFreeStatesInternal`: `nlow/nhigh/mlow/mhigh/mpert/npert/
  numpert_total`, `psilim`, `qlim`, `msing`, `sing[s].psifac/.q/.q1` only. PE
  recomputes its own Green's functions via `Vacuum.compute_vacuum_response`.
- PE reads `metric.fourier_coeffs` only; `ffit.amats/bmats/cmats`,
  `ffit.fmats_lower/kmats`, `ffit.kinetic_populated`, and calls
  `ForceFreeStates.el_derivatives!`.
- PE sub-calc order/prereqs: `compute_plasma_response!` needs `wt0` + dense stores +
  ffit A/B/C + `metric.fourier_coeffs`; `compute_singular_coupling_metrics!` needs
  `wt0` + `intr.plasma_response` (from the response step) + boundary `u_store` +
  Ξ/Ξ′ near surfaces (+ optional `inner_bpen`, same identity-at-edge basis).
- SLAYER (`src/Tearing/Runner/run_slayer.jl:367`) reads exactly `ffs_intr.sing` and
  `ffs_intr.delta_prime_matrix` (+ `equil`, `dir_path` kwarg).
- Standalone vacuum wv: `VacuumInput(equil, ψ, mthvac, nzvac, mrange, nrange)`
  (`src/Vacuum/DataTypes.jl:64`) → `compute_vacuum_response(inputs, wall).wv` — no
  ODE state. `free_run` applies singfac scaling in place (`Free.jl:86-88`);
  `galerkin_solve` consumes the ALREADY singfac-scaled wv and multiplies by `psio²`
  (`Galerkin/GalerkinSolve.jl:124-129`).
- `EquilibriumConfig` is `@kwdef` (`src/Equilibrium/EquilibriumTypes.jl:43`) —
  keyword construction works today; the Dict constructor is a filter/warn wrapper.
- Writers: FFS `write_outputs_to_HDF5` defined `GeneralizedPerturbedEquilibrium.jl:700`,
  called once at `:490`. PE writer `src/PerturbedEquilibrium/Utils.jl:103`, called once
  at `:618`. `write_imas` (`:1020`) reads `result.free_energies.et/.n_tor_idx` and
  `result.intr.npert/.nlow`; tested in `test/runtests_imas.jl:122,152,175,183,191`.
- Rerun path: `build_inputs_from_h5` (`src/Rerun.jl:199`) returns a 7-tuple funneled
  into `main_from_inputs`; it never reads the FFS flags by name (opaque dict).
- Deprecation machinery: `_drop_deprecated_keys!` + `_DEPRECATED_FFS_KEYS` /
  `_DEPRECATED_EQUIL_KEYS` (`GeneralizedPerturbedEquilibrium.jl:73-86`), applied at
  `:126`, `:184`, and `src/Rerun.jl:269`. Pre-commit hook `toml-no-deprecated-keys`
  mirrors these lists — **update the hook regex whenever the lists change**.
- Tests: `test/runtests.jl:21-50` is a hard-coded include list (no globbing).
  `use_riccati` appears in NO test and NO TOML. `riccati_eulerlagrange_integration`
  called directly at `test/runtests_riccati.jl:115`. `use_parallel` toggles at
  `test/runtests_eulerlagrange.jl:435`, `test/runtests_parallel_integration.jl:234-501`.
  `populate_dense_xi` testsets at `test/runtests_parallel_integration.jl:389-490`.
- Docs: FFS `@autodocs` at `docs/src/stability.md:273-276` (Pages list includes
  `Ballooning.jl`); `docs/src/ballooning.md` has NO autodocs block;
  `docs/make.jl:46` `checkdocs=:exports`; nav at `docs/make.jl:27-45`.
  `docs/development/architecture.md` module list is stale and needs updating anyway.
- Deps: CommonSolve NOT in `[deps]` (indirect in Manifest — add to `[deps]`+`[compat]`).
  `using OrdinaryDiffEq` (`src/ForceFreeStates/ForceFreeStates.jl:8`) already brings
  `solve` (== `CommonSolve.solve`) unqualified into FFS scope — new methods MUST be
  defined via `import CommonSolve: solve` (adding methods to the same generic; the
  existing unqualified `solve(prob, Vern9(); ...)` calls keep working).
- "shooting" rename scope: `EulerLagrange.jl:166` docstring;
  `GeneralizedPerturbedEquilibrium.jl:580,582` comments; `Galerkin/GalerkinMatch.jl:240`
  docstring; benchmarks labels (`benchmarks/compare_jbgradpsi_m2.jl`,
  `scan_resistivity_m2.jl`, `scan_rotation_m2.jl`). **Do NOT rename**: the GGJ
  inner-layer `:shooting` backend (`src/InnerLayer/GGJ/Shooting.jl`, `:ggj_shooting`
  in `src/Tearing/Runner/Control.jl`), the ballooning-doc "shooting boundary"
  (`docs/src/ballooning.md:847`), and the BVP shooting-propagator names
  (`uShootR/uShootL`, `_build_S_axis_shooting_propagators`) — those are correct
  shooting-method/STRIDE terminology.

---

## 2. PR sequence overview

Branch from `develop`, PR back into `develop`. **Every PR requires third-party human
review before merge — non-negotiable.** Run the regression harness once per PR and
report the table (differences are expected and get accepted knowingly; see D10).
All code must be JuliaFormatter-clean per `.JuliaFormatter.toml` before commit.

| PR | Branch | Content |
|----|--------|---------|
| 1 | `refactor/riccati-unification` | Delete serial-Riccati + `populate_dense_xi` + `parallel_threads`; `integrator=` ctrl key; `nchunks` knob; thread-independent chunking; shooting→forward rename |
| 2 | `refactor/local-stability-module` | Extract Ballooning.jl → `LocalStability` module; drop ctrl dependency |
| 3 | `refactor/forcefreestates-result` | `ForceFreeStatesResult` + warn-and-skip consumers (PE, FFS writer, SLAYER, write_imas) |
| 4 | `refactor/staged-main` | Decompose `main_from_inputs` into stage functions; standalone Galerkin; deprecate `gal_flag` |
| 5 | `feature/solve-api` | CommonSolve `solve`, integrator structs public, `PlasmaEquilibrium(path;…)`, `RMPField`, `perturbed_equilibrium` |

---

## 3. PR 1 — `refactor/riccati-unification`

### 3.1 Control struct (`src/ForceFreeStates/ForceFreeStatesStructs.jl`)

- DELETE fields + docstring entries: `use_riccati` (:297), `use_parallel` (:298),
  `parallel_threads` (:290, docstring :258), `populate_dense_xi` (:299, docstring :259).
- ADD fields:
  - `integrator::String = "riccati"` — `"forward" | "riccati" | "galerkin"` is
    validated at dispatch (`"galerkin"` only becomes legal in PR 4; until then it
    errors with "not yet a standalone integrator — use gal_flag").
  - `nchunks::Int = 0` — Riccati chunk-count target; `0` = auto (structure-derived).
- Validation (where `ctrl` is constructed is a splat; add checks at the top of
  `eulerlagrange_integration`): error if `integrator == "riccati" && kinetic_factor > 0`
  ("kinetic runs require integrator=\"forward\""); error on unknown integrator string.

### 3.2 Integration code

- `src/ForceFreeStates/EulerLagrange.jl`:
  - `eulerlagrange_integration` dispatch: `integrator=="riccati"` →
    `riccati_eulerlagrange_integration` (the renamed STRIDE driver), else forward.
  - RENAME `serial_eulerlagrange_integration` → `forward_eulerlagrange_integration`
    (keep the `verbose` kwarg; update the "Serial shooting branch" docstring at :166).
  - `balance_integration_chunks` (:79): remove the `4 * effective_threads` term and
    the `ctrl.parallel_threads` read (:90). New sizing:
    `target_n = ctrl.nchunks > 0 ? max(ctrl.nchunks, 2*intr.msing + 3) : max(2*intr.msing + 3, 8*(intr.msing + 1) + intr.msing)`
    — with `@warn` when an explicit `nchunks` is clamped up. NO `Threads.nthreads()`
    anywhere in chunk sizing.
- `src/ForceFreeStates/Riccati.jl`:
  - DELETE `riccati_eulerlagrange_integration` (:1312-1397) and
    `_populate_dense_xi_via_serial_el!` (:1900-1980).
  - RENAME `parallel_eulerlagrange_integration` → `riccati_eulerlagrange_integration`
    (name is now free; update its docstring: "the Riccati/STRIDE integrator", drop the
    populate_dense_xi paragraph and the `Enable via use_parallel` line). Remove the
    `ctrl.populate_dense_xi && !ctrl.force_termination` block (:1681-1683).
  - Thread pool: replace `bvp_threads = max(1, min(Threads.nthreads(), ctrl.parallel_threads))`
    (:1653) with `Threads.nthreads()` used directly by `_run_parallel_bvp_phase!`;
    per-thread proxies keep sizing by `Threads.maxthreadid()`.
  - After the parallel path, `odet.u_store_el_basis` stays `false` (already set at
    :1795) — this is now the permanent contract: Riccati's odet never claims EL basis.
- `src/PerturbedEquilibrium/SingularCoupling.jl:66-69`: update the hard `error()`
  message (references `populate_dense_xi`) → "dense Ξ′ requires the Forward
  integrator" (message only; the structural gate arrives in PR 3).
- `src/GeneralizedPerturbedEquilibrium.jl`: comments at :580/:582 ("shooting
  solution") → "forward solution". Add the four removed keys to
  `_DEPRECATED_FFS_KEYS` (:73). NOTE: with `use_parallel` warn-ignored, old TOMLs and
  gpec.h5 replays (whose `gpec_toml_raw` embeds old keys) fall through to the default
  `integrator="riccati"` — same physics path as before, so replays stay valid.
- `src/ForceFreeStates/Galerkin/GalerkinMatch.jl:240`: docstring "shooting
  integrator's" → "forward integrator's".

### 3.3 Pre-commit hook + TOML sweep

- Update the `toml-no-deprecated-keys` pygrep pattern in `.pre-commit-config.yaml`
  to include the four new deprecated keys.
- All 12 `examples/*/gpec.toml` + 4 `test/test_data/regression_*/gpec.toml`
  (canonical annotation source = `examples/DIIID-like_ideal_example/gpec.toml` per
  `docs/development/toml-conventions.md`): remove `use_parallel`, `parallel_threads`,
  `populate_dense_xi` lines; add `integrator = "…"` with a convention-conform comment.
  Assignment:
  - `integrator = "forward"` for every deck with a `[PerturbedEquilibrium]` section or
    `kinetic_factor > 0`: `DIIID-like_ideal_example`, `Solovev_ideal_example`,
    `Solovev_kinetic_NTV_example`, `Solovev_kinetic_calculated_example`,
    `a10_kinetic_example`, and the 4 regression fixtures.
  - `integrator = "riccati"` for Δ′/stability-only decks: `Solovev_ideal_example_multi_n`,
    `Solovev_ideal_example_3D`, `LAR_beta_scan`, `LAR_epsilon_scan`,
    `DIIID-like_SLAYER_example` (needs `delta_prime_matrix`).
  - gal decks (`DIIID-like_gal_resistive*`, `LAR_*_match_test`) keep `gal_flag=true`
    and use `integrator = "riccati"` (gal stays additive until PR 4).
  - NEW example `examples/DIIID-like_riccati_deltaprime_example/` (copy of
    DIIID-like_ideal minus `[PerturbedEquilibrium]`/`[ForcingTerms]`, with
    `integrator="riccati"`) so the canonical Δ′-matrix fixture survives the
    DIIID-like_ideal switch to forward. Add a matching regression case
    `regression-harness/cases/diiid_n1_riccati.toml` tracking
    `singular/delta_prime_matrix`-derived quantities (mirror the Δ′ entries of the
    existing `diiid_n1` case; ξ/PE quantities stay on `diiid_n1`).
- `benchmarks/benchmark_threads.jl`, `benchmarks/benchmark_delta_prime_methods.jl`:
  update flag names (`use_riccati`/`parallel_threads` → `integrator`/`nchunks`);
  `benchmarks/compare_jbgradpsi_m2.jl`, `scan_resistivity_m2.jl`,
  `scan_rotation_m2.jl`: label text "shooting" → "forward".

### 3.4 Tests

- `test/runtests_riccati.jl`: replace the direct call at :115 with the renamed driver
  (`FFS.riccati_eulerlagrange_integration(ctrl, equil, ffit, intr)` now returns the
  4-tuple — destructure) or route via `ctrl` with `integrator="riccati"`. Keep the
  energy-agreement assertion vs the forward path (:127). Delete the "(S, I) identity"
  check tied to the deleted serial path (:151) or re-target it to the driver's
  outer-region state.
- `test/runtests_parallel_integration.jl`: `use_parallel` toggles → `integrator=`
  strings; DELETE the `populate_dense_xi` testsets (:389-490); keep/extend the sparse
  u_store control test as "riccati leaves sparse u_store". ADD a unit test that
  `balance_integration_chunks` output is identical for any `Threads.nthreads()`
  (call with same inputs; assert no thread dependence — pure function now) and that
  `nchunks` steering works and clamps with a warning.
- `test/runtests_eulerlagrange.jl:435`: `use_parallel=false` → `integrator="forward"`.
- `test/runtests_rerun_from_h5.jl`: fixture decks pick up new keys automatically; the
  replay of PRE-refactor h5 files exercises the deprecated-key warn path — assert the
  warning fires once (cheap regression for the deprecation mechanism).

### 3.5 Docs

- `docs/src/stability.md`: rewrite the `use_riccati`/`use_parallel` passages
  (:61, :90, :243, :310) around `integrator = "forward"|"riccati"` and `nchunks`.
- `ForceFreeStatesControl` docstring: new entries for `integrator`, `nchunks`.

### 3.6 Verification

1. `julia --project=. test/runtests.jl test/runtests_riccati.jl test/runtests_parallel_integration.jl test/runtests_eulerlagrange.jl test/runtests_rerun_from_h5.jl test/runtests_fullruns.jl`
2. Full suite: `julia --project=. -e 'using Pkg; Pkg.activate("."); include("test/runtests.jl")'`
3. Regression harness: `julia --project=regression-harness regression-harness/regress.jl --cases diiid_n1,solovev_n1 --refs develop,local` — expected: forward-deck quantities unchanged vs develop where the deck previously ran `use_parallel+populate_dense_xi` (ξ was already forward-produced; Δ′ dataset disappears from forward decks — flagged, accepted); riccati decks match develop's parallel path bit-for-bit.
4. Docs build: `julia --project=. build_docs_local.jl`.

---

## 4. PR 2 — `refactor/local-stability-module`

### 4.1 Module extraction

- `git mv src/ForceFreeStates/Ballooning.jl src/LocalStability/Ballooning.jl`; create
  `src/LocalStability/LocalStability.jl`:
  ```julia
  module LocalStability
  using LinearAlgebra, FFTW, OrdinaryDiffEq, FastInterpolations
  using StaticArrays: SVector
  import ..Equilibrium
  include("Ballooning.jl")
  export compute_local_stability, compute_ballooning_stability!,
         ballooning_alpha_boundary, ballooning_alpha_boundaries
  end
  ```
  (Exact `using` set = what Ballooning.jl actually touches; it currently free-rides on
  the FFS module imports — FFTW via `FFTW.fft/ifft`, OrdinaryDiffEq via
  `ODEProblem/solve/DP5/ReturnCode`, FastInterpolations via
  `cubic_interp/Series/PeriodicBC/CubicFit/ExtendExtrap/integrate/cumulative_integrate`.)
- Top module (`src/GeneralizedPerturbedEquilibrium.jl`): `include` + `import .LocalStability`
  + `export LocalStability` after Equilibrium, before Vacuum. Remove the ballooning
  names from the FFS import line (:67).
- Signature changes (drop the `ForceFreeStatesControl` argument everywhere; it only
  supplied `verbose`):
  - `compute_local_stability(plasma_eq; verbose=false)`
  - `compute_ballooning_stability!(locstab_fs, plasma_eq; theta_k=0.0, compute_delta_prime=true, verbose=false)`
  - `ballooning_alpha_boundary(plasma_eq; theta_k=0.0, n_scan=24, verbose=false)`
  - `ballooning_alpha_boundaries`, `ballooning_qprime_boundaries`,
    `ballooning_delta_prime_map`, `ballooning_qprime_delta_prime_map`,
    `scan_delta_prime_map` — same pattern (`ctrl::ForceFreeStatesControl=...` kwarg in
    `scan_delta_prime_map` becomes `verbose::Bool=false`).
- `src/ForceFreeStates/ForceFreeStates.jl`: remove `include("Ballooning.jl")` (:27).
  FFS keeps `local_stability_flag` in its control struct for now (driver reads it);
  a `[LocalStability]` TOML section is future work, out of scope.
- Driver call sites (`GeneralizedPerturbedEquilibrium.jl:338,340`):
  `LocalStability.compute_local_stability(equil; verbose=ctrl.verbose)` /
  `LocalStability.ballooning_alpha_boundary(equil; verbose=ctrl.verbose)`.
- Cross-check test `test/runtests_resist_eval.jl:47`
  (`ForceFreeStates.prepare_ballooning_coefficients`) → `LocalStability.…`.

### 4.2 Docs

- `docs/src/stability.md:273-276`: remove `"Ballooning.jl"` from Pages.
- `docs/src/ballooning.md`: append an `@autodocs` block
  (`Modules = [GeneralizedPerturbedEquilibrium.LocalStability]`) — required because
  `checkdocs=:exports` (`docs/make.jl:46`) now sees the new exports.
- `docs/development/architecture.md`: add LocalStability to the module list and the
  dependency tree (the list is stale anyway; fix minimally — add LocalStability, note
  it depends only on Equilibrium).

### 4.3 Verification

Full test suite; targeted `runtests_resist_eval.jl`, `runtests_fullruns.jl`
(exercises `local_stability_flag=true` decks); docs build (missing-docs gate);
harness `--cases diiid_n1 --refs develop,local` (locstab datasets must be identical).

---

## 5. PR 3 — `refactor/forcefreestates-result`

### 5.1 New file `src/ForceFreeStates/Result.jl` (included from ForceFreeStates.jl)

Reuse existing types wholesale (`SingType`, `OdeState`, `FreeBoundaryResult`,
`GalerkinResult`, `FourFitVars`, `MetricData`, `EdgeScanState`) — minimal-change
discipline; only two new types + helpers:

```julia
"Δ′ outputs of the Riccati STRIDE BVP (moved off ForceFreeStatesInternal at result-build time)."
struct DeltaPrimeData
    matrix::Matrix{ComplexF64}      # msing×msing PEST3 Δ′  (was intr.delta_prime_matrix)
    raw::Matrix{ComplexF64}         # 2msing×2msing side-major D′ (was intr.delta_prime_raw)
    coil::Matrix{ComplexF64}        # 2msing×numpert_total edge coil response (was intr.delta_coil)
end

struct ForceFreeStatesResult
    integrator::Symbol                       # :forward | :riccati | :galerkin
    control::ForceFreeStatesControl          # provenance snapshot (carries mthvac, verbose, …)
    equil::Equilibrium.PlasmaEquilibrium     # possibly re-formed (two-pass)
    # mode space & domain (copied out of intr — plain immutable data)
    mlow::Int; mhigh::Int; mpert::Int
    nlow::Int; nhigh::Int; npert::Int; numpert_total::Int
    psilow::Float64; psilim::Float64; qlim::Float64; q1lim::Float64
    dir_path::String
    wall_settings::Vacuum.WallShapeSettings
    # assembly products (always present)
    metric::MetricData
    ffit::FourFitVars
    surfaces::Vector{SingType}               # alias of intr.sing (ua/restype/α live here)
    kinetic::@NamedTuple{kmsing::Int, kinsing::Vector{SingType}, scan_psi::Vector{Float64}, scan_cond::Vector{Float64}, scan_threshold::Float64}
    # per-integrator products (presence == capability)
    solution::Union{Nothing,OdeState}        # dense stores; see solution_basis
    solution_basis::Symbol                   # :el_axis | :riccati | :gal_native | :none
    free_boundary::Union{Nothing,FreeBoundaryResult}
    delta_prime::Union{Nothing,DeltaPrimeData}
    galerkin::Union{Nothing,GalerkinResult}
end
```

Contract of `solution`/`solution_basis`:
- Forward → dense EL-basis odet, `:el_axis`. PE-usable.
- Riccati → its odet IS carried (psi/q/crit/edge_scan are valid, u_store is chunk
  snapshots), `:riccati`. NOT PE-usable; HDF5 `integration/psi|crit|EdgeScan` still
  written from it, `integration/xi_*` written empty.
- Galerkin with `gal_match_flag` → `gal_matched_odestate(...)`, `:gal_native`
  (PE-usable; `du_store_populated=true` analytic derivatives). Without match →
  `nothing`, `:none`.

Helpers (same file):

```julia
"Warn-and-skip gate: true iff `field` is populated."
function require(result::ForceFreeStatesResult, field::Symbol, calc::AbstractString)
    getfield(result, field) === nothing || return true
    @warn "Skipping $calc: `$field` was not produced by the $(result.integrator) integrator"
    return false
end

"Warn-and-skip gate for ξ-profile consumers: solution present AND in a usable basis."
function require_solution(result, calc; bases=(:el_axis, :gal_native))
    result.solution !== nothing && result.solution_basis in bases && return true
    @warn "Skipping $calc: dense ξ profiles require a Forward (or gal-matched Galerkin) run; " *
          "this result came from the $(result.integrator) integrator (basis=$(result.solution_basis))"
    return false
end

"Assemble the result after integration. Pure data movement — no computation."
build_result(integrator, ctrl, equil, intr, metric, ffit, odet, free_energies, gal_data) -> ForceFreeStatesResult
```

`build_result` sets `delta_prime = isempty(intr.delta_prime_matrix) ? nothing :
DeltaPrimeData(intr.delta_prime_matrix, intr.delta_prime_raw, intr.delta_coil)`,
`free_boundary = free_energies`, `galerkin = gal_data`, and the gal-vs-odet solution
selection currently done inline at `GeneralizedPerturbedEquilibrium.jl:584-592`
(`gal_matched_odestate` + `pe_intr.odet_from_gal`/`inner_bpen` handling moves behind
the result: `inner_bpen` is fetched from `result.galerkin.match.bpen` by the driver).
`ForceFreeStatesInternal` stays as internal scratch during the solve; it no longer
crosses module boundaries after `build_result`.

### 5.2 Consumers

- **PE** (`src/PerturbedEquilibrium/PerturbedEquilibrium.jl`): new signature
  `compute_perturbed_equilibrium(result::ForceFreeStates.ForceFreeStatesResult, ft_ctrl, ctrl, intr)`
  (drop `equil/odet/wt0/mthvac/ffs_intr/metric/ffit` — all read off `result`).
  Internals:
  - `initialize_mode_arrays!` reads mode fields from `result`.
  - Response step: `require(result, :free_boundary, "plasma response") &&
    require_solution(result, "plasma response")` else skip (replaces the wt0-warn at
    :128-130 and the discarded materialize Bool at :88 — `materialize_derivative_stores!`
    is still called, on `result.solution`, only when the gates pass).
  - Coupling step: same two gates + existing internal `plasma_response` gate.
  - All `ffs_intr.X` reads → `result.X`; `wt0` → `result.free_boundary.wt0`;
    `mthvac` → `result.control.mthvac`; odet → `result.solution`.
  - `pe_intr.odet_from_gal` ↔ `result.solution_basis == :gal_native`.
- **FFS HDF5 writer**: re-signature to
  `write_outputs_to_HDF5(result; git_version, inputs, forcing_modes, locstab, ballooning_boundary)`
  — body is today's `:700-991` with `ctrl/equil/intr/odet/free_energies/ffit/gal_data`
  spelled `result.*`; every group that came from an optional field gets the existing
  empty-array fallback (already the pattern for FreeBoundaryStability). Δ′ datasets
  read from `result.delta_prime`. **Dataset names/paths unchanged** (D11).
- **SLAYER**: `Runner.run_slayer(result, control; dir_path)` — reads
  `result.surfaces`, `result.delta_prime === nothing ? empty : result.delta_prime.matrix`,
  `result.equil`. Keep a thin internal method for the old `(equil, sing, dpm)` shape if
  convenient; update `test/runtests_slayer_runner.jl`.
- **`write_imas`** + **`main` return value**: `main`/`main_from_inputs` return
  `(; ffs::ForceFreeStatesResult, pe, slayer)` (pe/slayer possibly `nothing`).
  `write_imas(dd, ret)` reads `ret.ffs.free_boundary` (skip+warn if `nothing`),
  `ret.ffs.npert/.nlow`. Update `test/runtests_imas.jl` call sites.
- Kinetic-forces stage keeps consuming `pe_state` + `result` fields analogously
  (`set_perturbation_data!(kf_intr, pe_state, result, …)` — mode/metric reads only).

### 5.3 Tests

- New `test/runtests_result_struct.jl` (add to `test/runtests.jl` include list):
  build a Solovev case; assert Forward result has `solution_basis == :el_axis`,
  `delta_prime === nothing`; Riccati result has `delta_prime !== nothing`,
  `solution_basis == :riccati`; `require_solution` warns exactly once
  (`@test_logs (:warn,)`) and PE skips without throwing on a Riccati result with a
  `[PerturbedEquilibrium]` deck.
- Update every test that consumed `main`'s old named-tuple return
  (`runtests_fullruns.jl`, `runtests_imas.jl`, `runtests_rerun_from_h5.jl`,
  `runtests_parallel_integration.jl` capture helpers).

### 5.4 Verification

Full suite; `runtests_fullruns.jl` (all decks — forward decks produce identical
HDF5 vs pre-PR, riccati decks now emit empty `integration/xi_*` + PE-skip warnings);
harness `--cases diiid_n1,solovev_n1 --refs develop,local` (forward-deck tracked
quantities unchanged); docs build.

---

## 6. PR 4 — `refactor/staged-main`

### 6.1 Stage functions (all in `src/GeneralizedPerturbedEquilibrium.jl`; `main_from_inputs` becomes ~40 lines of orchestration)

```julia
resolve_mode_space!(intr, ctrl)                    # today's :187-208 n-range block
load_kinetic_context(inputs, intr, ctrl, equil)    # :218-236 kf_ctrl + kinetic_profiles
maybe_reform_equilibrium(equil, eq_config, additional_input, intr, ctrl, kin)  # :241-268 two-pass
snapshot_forcing_modes(inputs, path, ctrl, preloaded)  # :296-313
prepare_force_free_states!(intr, ctrl, equil)      # sing_lim!/sing_find!/filter (:322-360),
                                                   # sing_min! (gal), resist_eval_all!,
                                                   # m-range (:378-396), make_metric/make_matrix/
                                                   # make_kinetic_matrix (+kinsing finder)
run_force_free_states(ctrl, equil, ffit, intr, metric) -> ForceFreeStatesResult
                                                   # integrator dispatch + free_run +
                                                   # compute_delta_prime_matrix! + galerkin
                                                   # + build_result
run_perturbed_equilibrium(result, inputs, forcing_snapshot, preloaded_coils) -> pe_state
run_kinetic_forces(inputs, result, pe_state, kf_ctrl, kinetic_profiles)
run_slayer_stage(result, inputs, pe_file)          # today's closure :512-541, un-closured
```

Rules: rerun (`build_inputs_from_h5` → 7-tuple) and IMAS (`dd` kwarg) entry paths
funnel into the same orchestration untouched; `force_termination` early-exits preserved
(both return the new `(; ffs, pe=nothing, slayer)` shape); the two-pass equilibrium
logic stays a pre-FFS stage but is owned by the FFS-facing function
(`maybe_reform_equilibrium` calls `ForceFreeStates.rational_psi_nodes` +
`Equilibrium.refined_psi_grid`/`setup_equilibrium` exactly as today).

### 6.2 Standalone Galerkin (`integrator = "galerkin"` becomes legal)

- Factor the wv computation out of `free_run` into a shared helper in
  `src/ForceFreeStates/Free.jl`:
  ```julia
  "Raw vacuum response at psilim with the Chance singfac scaling applied (Free.jl:82-88)."
  compute_scaled_wv(ctrl, equil, intr) -> (wv, vac)   # no OdeState involved
  ```
  `free_run` calls it (identical numerics — pure extraction); the galerkin stage calls
  it when `ctrl.vac_flag` to supply `wv` to `galerkin_solve`.
- `run_force_free_states` with `"galerkin"`: skip EL integration and `free_run`
  entirely; `sing_min!` + `galerkin_solve` (+ `gal_match_rpec` via flags as today);
  result: `free_boundary=nothing`, `delta_prime=nothing`, `galerkin=GalerkinResult`,
  `solution` from `gal_matched_odestate` when matched (`:gal_native`) else
  `nothing`/`:none`. Error if `kinetic_factor > 0`. `npert == 1` enforced by
  `galerkin_solve` already.
- Deprecate `gal_flag` (add to `_DEPRECATED_FFS_KEYS` + hook): additive gal is
  REMOVED — `gal_flag=true` decks become `integrator = "galerkin"`. Retoml:
  `DIIID-like_gal_resistive_example`, `DIIID-like_gal_resistive_pe_example`,
  `LAR_ideal_match_test`, `LAR_resistive_match_test` (their FFS-side datasets
  disappear from the HDF5 — accepted per D10; gal datasets identical because
  `galerkin_solve` inputs are unchanged). `gal_*` sub-knobs stay (they become
  `Galerkin(...)` fields in PR 5).
- FFS writer: tolerate `solution === nothing` (write empty `integration/*` datasets —
  extend the existing empty-fallback pattern).

### 6.3 Tests / verification

- Update `runtests_fullruns.jl` gal decks' expectations (gal-only HDF5).
- New testset (in `runtests_fullruns.jl` or the gal tests): `integrator="galerkin"`
  on `LAR_ideal_match_test` produces `galerkin/delta` identical to the PR-3 additive
  run (same `wv` by construction — assert against a stored reference or a paired
  riccati+gal_flag run on the pre-PR commit during development).
- Full suite; harness (gal cases if present, plus diiid/solovev); docs.

---

## 7. PR 5 — `feature/solve-api`

### 7.1 Dependencies

- `Project.toml`: add `CommonSolve` to `[deps]` and `[compat]` (`"0.2"`). It is
  already in the Manifest transitively — no resolver churn expected. Do NOT remove
  anything from Project.toml.

### 7.2 Integrator structs (`src/ForceFreeStates/Integrators.jl`, new file)

```julia
abstract type AbstractIntegrator end
Base.@kwdef struct Forward <: AbstractIntegrator end
Base.@kwdef struct Riccati <: AbstractIntegrator
    nchunks::Int = 0            # 0 = auto (structure-derived); threads come from julia -t
end
Base.@kwdef struct Galerkin <: AbstractIntegrator
    # mirror every gal_* ctrl field with identical defaults, WITHOUT the gal_ prefix:
    solver::String = "LU"; nx::Int = 256; nq::Int = 6; pfac::Float64 = 0.001
    dx0::Float64 = 5e-4; dx1::Float64 = 1e-3; dx2::Float64 = 1e-3; cutoff::Int = 10
    tol::Float64 = 1e-10; gnstep::Int = 20000; dx1dx2_flag::Bool = true
    sing_order::Int = 6; sing_order_ceiling::Bool = true
    rpec_flag::Bool = false; edge_onesided::Bool = false
    match_flag::Bool = false; ideal_flag::Bool = false; inner_solver::String = "ray"
    inner_xfac::Float64 = 10.0; inner_nx::Int = 1280; inner_nq::Int = 5
    inner_cutoff::Int = 5; inner_kmax::Int = 8
    eta::Vector{Float64} = Float64[]; rho::Vector{Float64} = Float64[]
    rotation::Vector{Float64} = Float64[]; gamma::Float64 = 5/3
end
```

Mapping helpers `_integrator_symbol(alg)` and `_apply_alg!(ctrl_kwargs, alg)`
translate an alg struct into the `ForceFreeStatesControl` keyword set (pure
translation — `ForceFreeStatesControl` remains the single source of truth for the
solve; the TOML `integrator=` + flat `gal_*`/`nchunks` keys keep working unchanged).

### 7.3 `solve` + constructors

- In `ForceFreeStates`: `import CommonSolve: solve` (coexists with the
  OrdinaryDiffEq-re-exported `solve`; same generic), then
  ```julia
  function solve(equil::Equilibrium.PlasmaEquilibrium, alg::AbstractIntegrator;
                 nn::Union{Int,UnitRange{Int}}, wall::Vacuum.WallShapeSettings=Vacuum.WallShapeSettings(),
                 dir_path::String=".", kwargs...)   # kwargs = any ForceFreeStatesControl field
      -> ForceFreeStatesResult
  ```
  Body: build `ctrl` from `alg` + kwargs (`nn_low/nn_high` from `nn`), build `intr`,
  then call the PR-4 stages `resolve_mode_space!` → (two-pass reform if the equilibrium
  was built with `grid_type="auto"` and not yet refined — reuse
  `maybe_reform_equilibrium`) → `prepare_force_free_states!` → `run_force_free_states`.
  Top module: `import CommonSolve` and `export solve` (re-export the generic), plus
  `export Forward, Riccati, Galerkin, ForceFreeStatesResult`.
- `Equilibrium`: outer constructor
  `PlasmaEquilibrium(path::AbstractString; eq_type::String="efit", kwargs...) =
   setup_equilibrium(EquilibriumConfig(; eq_type, eq_filename=abspath(path), kwargs...))`
  (the `@kwdef` config makes this a 3-liner; `sol/lar/tj` analytic types keep using
  `setup_equilibrium(config, analytic_config)` directly — documented, not wrapped).
- `RMPField` (in `ForcingTerms`, exported): a lazy forcing description —
  ```julia
  struct RMPField
      ctrl::ForcingTermsControl       # format/file/machine/coil_sets_raw as today
      scale::Float64                  # uniform multiplier applied to loaded amplitudes/currents
  end
  RMPField(path::AbstractString; format=_infer_format(path), scale=1.0, kwargs...)
  RMPField(coil_sets::Vector{Dict{String,Any}}; scale=1.0, kwargs...)   # TOML-shaped coil blocks
  ```
  Constraint (verified): ForcingTerms has no n-keyed amplitude concept — amplitude is
  per-conductor currents (coil format) or per-mode `ForcingMode.amplitude` (file
  formats). `scale` multiplies whichever applies at materialization. A per-n amplitude
  dict is deferred (needs ForcingTerms design work; note in docstring).
- `perturbed_equilibrium(ffs::ForceFreeStatesResult, rmp::RMPField; kwargs...)`
  (top module): builds `PerturbedEquilibriumControl` from kwargs +
  `PerturbedEquilibriumInternal(dir_path=ffs.dir_path)`, materializes forcing modes
  from `rmp` against `ffs.equil` (the logic currently inside
  `compute_perturbed_equilibrium`'s loading block, `PerturbedEquilibrium.jl:92-124`),
  pulls `inner_bpen` from `ffs.galerkin` when `:gal_native`, and calls
  `compute_perturbed_equilibrium(ffs, ft_ctrl, pe_ctrl, pe_intr)`. The TOML driver
  (`run_perturbed_equilibrium`) is rewired through this same function so there is ONE
  forcing-materialization path.

### 7.4 TOML & docs & tests

- Finalize `_DEPRECATED_FFS_KEYS` (now includes `use_riccati, use_parallel,
  parallel_threads, populate_dense_xi, gal_flag`) + pre-commit hook regex.
- Docs: new "Scripting API" page (`docs/src/api.md` or extend `workflow.md`) with the
  four-line UX example; `@autodocs`/`@docs` entries for `solve`, the alg structs,
  `RMPField`, `perturbed_equilibrium`, `ForceFreeStatesResult` (checkdocs=:exports
  will enforce); nav entry in `docs/make.jl`.
- New `test/runtests_solve_api.jl` (added to runtests.jl list): Solovev end-to-end via
  the API only — `PlasmaEquilibrium(...)`; `solve(eq, Forward(); nn=1, ...)` matches a
  TOML-driven `main` run on key numbers (`free_boundary.et[1]`, `nzero`);
  `solve(eq, Riccati(nchunks=40); nn=1)` produces `delta_prime` matching the TOML run;
  `solve(eq, Galerkin(); nn=1)` returns a gal-only result; `perturbed_equilibrium`
  round-trip on the forward result; kwarg validation errors (`Riccati` + kinetic).

### 7.5 Verification

Full suite; harness (all cases, `--refs develop,local`, report table); docs build;
manual smoke: run the 4-line UX from the Context section in a REPL against
`examples/DIIID-like_ideal_example` inputs.

---

## 8. Cross-cutting execution rules (for every PR)

1. **Never merge without third-party human review. State this in every PR body.**
2. Every commit and push requires explicit per-instance maintainer approval.
3. Commit messages: `CODE - TAG - message` (e.g. `FFS - REFACTOR - Unify Riccati integrator`).
4. JuliaFormatter-clean (margin 180, kwargs `f(x; a=1)`, no trailing whitespace, LF,
   single trailing newline). TOML edits follow `docs/development/toml-conventions.md`
   (header block, per-line `# description` copied from the struct docstring,
   descriptions identical across files, no Fortran references).
5. No PR/issue numbers in source comments; no step-numbered comments; struct fields
   documented in the struct docstring.
6. Docstrings are CommonMark — no bare `[x] (y)` bracket-paren sequences.
7. Run the regression harness before requesting review; paste the report into the PR.
8. Test files are registered in `test/runtests.jl`'s hard-coded include list.
9. Keep this `REFACTOR_PLAN.md` updated (check off completed PRs); delete it in a
   final cleanup commit after PR 5 is merged and the Fortran re-comparison is done.

## 9. Sanity map: which capability comes from where (post-refactor)

| Output | Forward | Riccati | Galerkin |
|---|---|---|---|
| dense ξ/Ξ′/Ξ_s profiles (`solution`, PE-usable) | ✅ `:el_axis` | ❌ (until Frobenius work) | matched only (`:gal_native`) |
| STRIDE Δ′ matrix / raw / `delta_coil` (`delta_prime`) | ❌ | ✅ | ❌ (has own `galerkin.delta`/`delta_coil`) |
| free-boundary energies (`free_boundary`) | ✅ | ✅ | ❌ (`nothing`) |
| fixed-boundary crit / nzero / edge scan (on odet) | ✅ | ✅ | ❌ |
| RDCON Δ′ + PEST3 blocks + RPEC match (`galerkin`) | ❌ | ❌ | ✅ |
| kinetic (`kinetic_factor>0`) | ✅ | error | error |
| SLAYER inputs (surfaces + Δ′ matrix) | surfaces only (diag fallback) | ✅ | surfaces only |

## 10. Progress

- [ ] PR 1 — `refactor/riccati-unification`
- [ ] PR 2 — `refactor/local-stability-module`
- [ ] PR 3 — `refactor/forcefreestates-result`
- [ ] PR 4 — `refactor/staged-main`
- [ ] PR 5 — `feature/solve-api`
- [ ] Fortran re-comparison of all important quantities
- [ ] Delete this file
