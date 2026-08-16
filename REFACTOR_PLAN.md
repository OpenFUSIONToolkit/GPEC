# GPEC Julia — ForceFreeStates modularization & `solve(eq, integrator)` API
## Complete multi-PR implementation plan

> **NOTE FOR ALL DEVELOPERS (read this first).**
> This document is the agreed, in-progress plan for a refactor of the
> ForceFreeStates ↔ PerturbedEquilibrium interface and the top-level driver, delivered
> as THREE pull requests: #381 (integrator unification), #387 (LocalStability), and one
> combined "interface PR" whose three commits carry what were originally planned as
> PRs 3-5 (the stack was collapsed once it became clear reviews would batch at the end). It is
> committed directly to `develop` (deliberately, as documentation only — no code
> changes ride with it) so everyone with open PRs can see what is coming and where it
> will touch their work. Key coordination points:
>
> - The PR sequence below assumes **nothing else merges into `develop` mid-sequence**.
>   If your PR must land before it finishes, talk to Matthew first.
> - **Amendment (post-PR-1):** the module-mirroring HDF5 schema (#363) and the vacuum
>   surface-inductance migration (#345) merged into develop before PR 1 branched, so
>   the sequence is built on the NEW schema (`ForceFreeStates/…`, `SingularSurfaces/…`,
>   `LocalStability/…`, `Equilibrium/…`). Writer refactors in PR 3/4 therefore target
>   that schema directly; dataset-path references below have been updated accordingly.
>   Only #364 (self-describing metadata) remains as a later re-target, and #367
>   (immutable control structs) still merges after this sequence. Neither merged PR
>   changes any decision: #345 left `calc_surface_inductance` in PerturbedEquilibrium
>   (only its Vacuum support types moved), so the PE consumer map stands.
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
| D3 | **No merging of two integration results.** `populate_dense_xi` + `_populate_dense_xi_via_serial_el!` + the standalone serial-Riccati path (`riccati_eulerlagrange_integration`) are deleted FIRST (PR 1). Riccati-fed PE warn-and-skips profile-based outputs PERMANENTLY (D14: riccati never produces full profiles); the separate `delta_mn` work (not in this plan) restores the resonant-coupling outputs — not the profile-based ones — from `delta_coil` + surface asymptotics. |
| D4 | Kinetic (`kinetic_factor > 0`) is Forward-only. `solve`/driver raises a clear error for Riccati+kinetic and Galerkin+kinetic. |
| D5 | One result struct **`ForceFreeStatesResult`**; optional fields are `Union{Nothing,T}`; consumers use a `require(...)` helper → `@warn` + skip. |
| D6 | Local stability (Ballooning.jl) → new top-level module **`LocalStability`**, depending only on Equilibrium (+ math deps). Only ctrl dependency is `verbose` → kwarg. |
| D7 | Public API via **CommonSolve.jl**: `solve(eq::PlasmaEquilibrium, alg; kwargs...)`. `PlasmaEquilibrium(path; kwargs...)` constructor. Module names unchanged. |
| D8 | Galerkin standalone computes its own vacuum `wv` (no ODE state needed — verified); its result has `free_boundary = nothing`. |
| D9 | TOML: new `integrator = "forward"|"riccati"|"galerkin"` key. Old keys (`use_riccati`, `use_parallel`, `parallel_threads`, `populate_dense_xi`, later `gal_flag`) go to the `_DEPRECATED_FFS_KEYS` warn-and-ignore list AND the `toml-no-deprecated-keys` pre-commit hook pattern. |
| D10 | No back-compat burden; examples/fixtures updated freely; regression re-baselining accepted. Final validation = fresh Fortran comparison after the sequence. Nothing else merges mid-sequence without coordination (#363/#345 landed before PR 1 and are absorbed — see header amendment). |
| D11 | New structs immutable from day one (eases the later #367 merge). HDF5 writers become functions on result structs, keeping the merged #363 schema paths unchanged; #364 (metadata) re-targets them later. |
| D12 | Analysis module reads HDF5 files, not live structs — untouched except where dataset names would change (they don't in this plan). |
| D13 | Inner-layer matching runs INSIDE `solve` (a `ForceFreeStatesResult` is always a closed basis). `result.solution` holds THE solve's ξ solution product — a thin `SolutionProfiles` interchange type — whenever one exists: forward always; galerkin when matched (built directly from the match — the `gal_matched_odestate` OdeState shim is DELETED); riccati permanently `nothing` — STRIDE matching yields rational-surface data (`bpen`, `delta_mn`), never profiles (D14). Closure is explicit and universal: `result.closure ∈ (:ideal, :matched)` and `result.bpen` (msing × numpert_total; zeros under ideal closure) are always present — the landing pad for any matching implementation. No transitional arbitration API: additive gal is removed in the SAME PR that introduces the result (PR 3), so one run has at most one solution and nothing like `pe_solution` is ever needed. Matching config is integrator-agnostic: a `ResistiveMatch` object (swappable `InnerLayer` model + per-surface `eta/rho/rotation`, `gamma`, `ideal`) passed as a `match=` kwarg to `solve` (PR 5). STRIDE-side matching is a future PR; until then `match` with `Riccati()` errors "not yet implemented". The `gal_*` matching TOML keys are renamed/re-homed by that future PR, not by this stack. |
| D14 | Same physics ⇒ same field, same type, across integrators, organized by the three-class taxonomy in §9 (control surface / full profiles / rational-surface resonant data). Riccati NEVER produces full ξ/ξ′ profiles — `result.solution` is permanently `nothing` for it. The next-cycle work adds `delta_mn` to riccati AND galerkin: a bpen-like matrix encoding the jump in the pitch-resonant derivative of the solution at each rational surface, from outer-solution asymptotics (for riccati: recoverable from `delta_coil`); it yields the perturbed current and shielded resonant flux, and is what PE resonant coupling consumes from a Riccati run (class 2, not class 1). Forward `delta_mn` is NOT planned — no concrete route has been identified and there may be none. There is ONE Δ′/matching data type, unified IN THIS PR: `delta_prime` carries Δ′ matrix, raw D′, `delta_coil`, and the PEST-3 blocks, produced by riccati and galerkin alike — galerkin already computes the same physics content, today under `galerkin.*` fields and different HDF5 names; its Δ′ payload merges into `delta_prime` (fields a formalism doesn't produce stay empty/`nothing`). Control-surface energies (`wp`, `free_boundary`) target all three integrators (galerkin pending its δW implementation). SLAYER consumes the unified `delta_prime`, so riccati- and galerkin-fed SLAYER both work (this PR). SLAYER + GGJ behind one abstract inner-layer interface is a later pass. |

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
| #381 | `refactor/riccati-unification` | Delete serial-Riccati + `populate_dense_xi` + `parallel_threads`; `integrator=` ctrl key; `nchunks` knob; thread-independent chunking; shooting→forward rename |
| #387 | `refactor/local-stability-module` | Extract Ballooning.jl → `LocalStability` module; drop ctrl dependency (stacked on #381) |
| interface PR | `refactor/forcefreestates-result` | ONE PR, three slice-pure commits: **(a)** §5 `ForceFreeStatesResult` + warn-and-skip consumers + standalone Galerkin; **(b)** §6 staged `main`; **(c)** §7 `solve` API (stacked on #387) |

Commit discipline for the interface PR: commit boundaries now do the job PR boundaries
did — keep each commit slice-pure (fixes amend into the right slice before review
starts; ordinary follow-up commits after). Per-slice numerical isolation stays
verifiable via the harness with commit SHAs as refs.

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
    `SingularSurfaces/Delta_prime_matrix`-derived quantities (mirror the Δ′ entries of the
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
harness `--cases diiid_n1 --refs develop,local` (`LocalStability/*` datasets must be identical; note the #363 group name already matches the new module name).

---

## 5. Interface PR, commit (a) — result struct, consumers, standalone Galerkin

### 5.1 New file `src/ForceFreeStates/Result.jl` (included from ForceFreeStates.jl)

Reuse existing types wholesale (`SingType`, `OdeState`, `FreeBoundaryResult`,
`GalerkinResult`, `FourFitVars`, `MetricData`, `EdgeScanState`); new types are
`DeltaPrimeData`, `SolutionProfiles`, and the result itself:

```julia
"Δ′ outputs of the Riccati STRIDE BVP (moved off ForceFreeStatesInternal at result-build time)."
struct DeltaPrimeData
    matrix::Matrix{ComplexF64}      # msing×msing PEST3 Δ′  (was intr.delta_prime_matrix)
    raw::Matrix{ComplexF64}         # 2msing×2msing side-major D′ (was intr.delta_prime_raw)
    coil::Matrix{ComplexF64}        # 2msing×numpert_total edge coil response (was intr.delta_coil)
end

"The solve's ξ solution, in the exact shape PerturbedEquilibrium consumes. Field names
 mirror the OdeState store subset so PE internals change minimally."
struct SolutionProfiles
    basis::Symbol                            # :el_axis (forward) | :gal_native (matched galerkin)
    step::Int                                # number of stored radial nodes
    psi_store::Vector{Float64}
    q_store::Vector{Float64}
    u_store::Array{ComplexF64,4}             # (N, N, 2, step) — Ξ_ψ and conjugate momentum
    du_store::Array{ComplexF64,3}            # (N, N, step) dΞ_ψ/dψ, ALWAYS populated
    xi_s_store::Array{ComplexF64,3}          # (N, N, step) Ξ_s, ALWAYS populated
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
    # closure of the basis at the rationals (D13) — ALWAYS present
    closure::Symbol                          # :ideal (jump condition imposed) | :matched (inner layer)
    bpen::Matrix{ComplexF64}                 # (msing × numpert_total) penetrated resonant field; zeros under :ideal
    # per-integrator products (presence == capability)
    solution::Union{Nothing,SolutionProfiles}   # THE solve's ξ solution; nothing when none exists (riccati; unmatched gal)
    diagnostics::Union{Nothing,OdeState}     # the integrator's raw odet (crit, edge scan, ψ trace, ca); writer-only
    wp::Union{Nothing,Matrix{ComplexF64}}    # fixed-boundary plasma energy W_p at psilim; present for any EL sweep even with vac_flag=false (aliases free_boundary.wp when free_run ran)
    free_boundary::Union{Nothing,FreeBoundaryResult}
    delta_prime::Union{Nothing,DeltaPrimeData}
    galerkin::Union{Nothing,GalerkinResult}
end
```

Contract (D13 — final, no transitional states):
- Forward → `solution` = `SolutionProfiles(:el_axis, …)` aliasing the odet's stores (zero
  copy), `diagnostics` = the same odet, `closure = :ideal`, `bpen` = zeros.
- Riccati → `solution = nothing` PERMANENTLY (chunk-endpoint states are not a ξ solution,
  and no reconstruction is planned; the future STRIDE matching populates
  `closure = :matched`, `bpen`, and `delta_mn` — rational-surface data, never profiles),
  `diagnostics` = its odet (ψ/q/crit/edge scan/ca are valid), `closure = :ideal`.
- Galerkin, matched → `solution` = `SolutionProfiles(:gal_native, …)` built DIRECTLY from
  `GalerkinResult.match`/`solution` (drop `issing` points, analytic Ξ′, `compute_node_xi_s!`
  for Ξ_s — the useful guts of the deleted `gal_matched_odestate`, minus the OdeState
  costume), `diagnostics = nothing`, `closure = :matched` (`:ideal` under `gal_ideal_flag`),
  `bpen = galerkin.match.bpen`.
- Galerkin, unmatched → `solution = nothing` (raw homogeneous gal columns are not a driven
  response basis), `closure = :ideal`.

There is NO `pe_solution` and NO stored-basis arbitration: additive gal is removed in this
PR (§5.3), so a run has at most one solution and PE reads `result.solution` directly.

Helpers (same file):

```julia
"Warn-and-skip gate: true iff the optional `field` is populated."
function require(result::ForceFreeStatesResult, field::Symbol, calc::AbstractString)
    getfield(result, field) === nothing || return true
    @warn "Skipping $calc: `$field` was not produced by the $(result.integrator) integrator"
    return false
end

"Specialized message for the ξ-solution gate."
require_solution(result, calc) = result.solution !== nothing ? true :
    (@warn "Skipping $calc: no ξ solution — dense profiles require a Forward (or matched Galerkin) run; " *
           "this result came from the $(result.integrator) integrator"; false)

"Assemble the published result once the solve is finished."
build_result(integrator, ctrl, equil, intr, metric, ffit, odet, free_energies, gal_data) -> ForceFreeStatesResult
```

`build_result` responsibilities (the ONLY place with assembly logic):
- `delta_prime` from the intr Δ′ fields when non-empty; `free_boundary = free_energies`;
  `galerkin = gal_data`; `diagnostics = odet`.
- Forward: call `materialize_derivative_stores!(odet, …)` HERE (moving the call out of the
  writer and PE — one site, always-populated `du_store`/`xi_s_store`), then wrap the stores
  in `SolutionProfiles(:el_axis, …)`.
- Matched gal: build `SolutionProfiles(:gal_native, …)` from the match (see contract above).
- `closure = (gal_data !== nothing && gal_data.match !== nothing && !ctrl.gal_ideal_flag) ? :matched : :ideal`;
  `bpen` = match bpen or zeros(msing, numpert_total).
`ForceFreeStatesInternal` stays as internal scratch during the solve; it no longer
crosses module boundaries after `build_result`.

### 5.2 Consumers

- **PE** (`src/PerturbedEquilibrium/PerturbedEquilibrium.jl`): new signature
  `compute_perturbed_equilibrium(result::ForceFreeStates.ForceFreeStatesResult, ft_ctrl, ctrl, intr)`
  (drop `equil/odet/wt0/mthvac/ffs_intr/metric/ffit` — all read off `result`).
  Internals:
  - `initialize_mode_arrays!` reads mode fields from `result`.
  - PE's working solution IS `result.solution::SolutionProfiles` (never an OdeState; PE
    internals re-type from `OdeState` to `SolutionProfiles` — field names match, so the
    change is annotations, not logic). No materialize call in PE: `du_store`/`xi_s_store`
    arrive populated.
  - Response step: `require(result, :free_boundary, "plasma response") &&
    require_solution(result, "plasma response")` else skip.
  - Coupling step: same two gates + existing internal `plasma_response` gate.
  - All `ffs_intr.X` reads → `result.X`; `wt0` → `result.free_boundary.wt0`;
    `mthvac` → `result.control.mthvac`.
  - `pe_intr.odet_from_gal` ↔ `result.solution.basis == :gal_native`;
    `pe_intr.inner_bpen = result.bpen` (driver; the gal special-case `if` is deleted).
- **FFS HDF5 writer**: re-signature to
  `write_outputs_to_HDF5(result; git_version, inputs, forcing_modes, locstab, ballooning_boundary)`
  — body is today's `:700-991` with `ctrl/equil/intr/odet/free_energies/ffit/gal_data`
  spelled `result.*`; every group that came from an optional field gets the existing
  empty-array fallback (already the pattern for FreeBoundaryStability). Δ′ datasets
  read from `result.delta_prime`. Solution-adjacent datasets split by source:
  `ForwardIntegration/xi_psi|u2|dxi_psi|xi_s` from `result.solution` when
  `basis == :el_axis` (empty otherwise — the gal-native solution is already persisted
  under the Galerkin group); `psi|q|nstep|nstep_total|crit`, `SingularSurfaces/ca_*`,
  and `EdgeScan/*` from `result.diagnostics` when present (empty otherwise). Output is
  byte-identical for every forward/riccati deck; the four gal decks become gal-only
  files (§5.3). **Dataset names/paths unchanged** (D11).
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

### 5.3 Standalone Galerkin + additive-gal removal (pulled forward from PR 4)

Additive gal is what would force a two-solutions-per-run transitional state; it dies in
this PR so the result contract above is final from day one.

- Factor the wv computation out of `free_run` into a shared helper in
  `src/ForceFreeStates/Free.jl`:
  `compute_scaled_wv(ctrl, equil, intr) -> (wv, vac)` — the `VacuumInput` +
  `compute_vacuum_response` + Chance singfac scaling block (no OdeState involved).
  `free_run` calls it; identical numerics by construction.
- `integrator = "galerkin"` becomes legal: the driver's gal branch skips EL integration
  and `free_run` entirely; runs `sing_min!` + (when `vac_flag`) `compute_scaled_wv` +
  `galerkin_solve` (+ `gal_match_rpec` via the existing flags); `build_result` fills the
  gal fields per the §5.1 contract. Errors if `kinetic_factor > 0`. `npert == 1` enforced
  by `galerkin_solve` already.
- DELETE: `gal_matched_odestate` (GalerkinMatch.jl) and the driver's additive-gal PE
  block (`pe_odet` selection). The additive path (`gal_flag=true` alongside another
  integrator) is REMOVED; `gal_flag` joins `_DEPRECATED_FFS_KEYS` + the pre-commit hook.
- RETAIN `_chord_solution_at` (SingularCoupling.jl) as an uncalled helper: re-typed to
  `SolutionProfiles`, hard-error branch dropped, stub-style docstring. Kept pending the
  `delta_mn` resonant-coupling design (chord-slope derivatives may be useful when PE
  consumes rational-surface data instead of profiles) — do NOT re-delete as dead code.
- Gal → PE this cycle: PerturbedEquilibrium's response step requires the free-boundary
  δW (`wt0`), which the Galerkin formalism does not produce — so PE warn-skips entirely
  on gal results (both gates: `free_boundary` missing kills response, and coupling needs
  the response). The gal-native `solution` consumer path in PE therefore stays dormant
  until the gal-side δW work lands (next cycle, with the STRIDE matching); the contract
  and tests are already in place for it.
- Retoml the four gal decks to `integrator = "galerkin"` (drop `gal_flag`):
  `DIIID-like_gal_resistive_example`, `DIIID-like_gal_resistive_pe_example`,
  `LAR_ideal_match_test`, `LAR_resistive_match_test`. Their HDF5 outputs become gal-only
  (FFS-side integration/energy datasets empty) — accepted per D10; gal datasets identical
  because `galerkin_solve` inputs are unchanged. `gal_*` sub-knobs stay (they become
  `Galerkin(...)` / `ResistiveMatch` fields in PR 5).

### 5.4 Tests

- New `test/runtests_result_struct.jl` (add to `test/runtests.jl` include list):
  build a Solovev case; assert Forward result has `solution.basis == :el_axis`,
  populated `du_store`/`xi_s_store`, `closure == :ideal`, `iszero(bpen)`,
  `delta_prime === nothing`; Riccati result has `delta_prime !== nothing`,
  `solution === nothing`, `diagnostics !== nothing`; `require_solution` warns exactly
  once (`@test_logs (:warn,)`) and PE skips without throwing on a Riccati result with a
  `[PerturbedEquilibrium]` deck; a matched gal deck (LAR_ideal_match_test-class) yields
  `solution.basis == :gal_native` and `bpen == galerkin.match.bpen` (zeros under
  `gal_ideal_flag`, with `closure == :ideal` there).
- Update every test that consumed `main`'s old named-tuple return
  (`runtests_fullruns.jl`, `runtests_imas.jl`, `runtests_rerun_from_h5.jl`,
  `runtests_parallel_integration.jl` capture helpers).

### 5.5 Verification

Full suite; `runtests_fullruns.jl` (forward decks produce byte-identical HDF5 vs the
stack base, riccati decks emit empty `ForwardIntegration/xi_*` + PE-skip warnings, gal
decks become gal-only files); harness vs the stack base
(`--cases diiid_n1,diiid_n1_riccati,solovev_n1 --refs refactor/local-stability-module,local`
— tracked quantities unchanged; gal-flavored cases re-baselined); docs build.

---

## 6. Interface PR, commit (b) — staged `main` (staging ONLY — gal work is in commit (a))

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

### 6.2 Tests / verification

Pure code motion: full suite unchanged; harness vs commit (a) must be identical for ALL
cases (no re-baselining in this slice); docs build. Standalone Galerkin and the
additive-gal removal live in commit (a) (§5.3).

---

## 6A. Interface PR, commit (b2) — unified Δ′/matching payload (D14)

Galerkin computes the same Δ′ physics riccati does (Δ′ matrix, raw D′, `delta_coil`,
PEST-3 blocks), today under separate `galerkin.*` fields and different HDF5 names. This
commit merges the two payloads into the ONE `delta_prime` field so consumers never care
which formalism produced it.

- **Inventory first (mandatory)**: enumerate every Δ′-flavored field in `GalerkinResult`
  and every field in `DeltaPrimeData`, and produce the exact mapping (name, shape,
  normalization, sign/side conventions) BEFORE moving anything. Do not assume the two
  formalisms' arrays are layout-identical — verify shapes/conventions and document any
  genuine mismatch in the type's docstring rather than silently coercing.
- **Type**: extend `DeltaPrimeData` to the union of both payloads (PEST-3 blocks join it).
  Fields a formalism doesn't produce stay empty/`nothing`. `build_result` fills it from
  whichever formalism ran; the Δ′ payload LEAVES the `galerkin` field, which keeps only
  solver internals / FEM diagnostics / RPEC match data (post-inventory list goes in the
  struct docstrings).
- **HDF5**: one set of dataset paths for Δ′ outputs regardless of formalism — the
  riccati/shared paths are canonical; gal's Δ′ datasets move there (clean break per
  `docs/development/hdf5-conventions.md`: update writer, readers, and harness case TOMLs
  together; no legacy-path shim). Coordinate with the pending #364 reconciliation so the
  paths are renamed once, not twice.
- **SLAYER**: `run_slayer` routes through the unified `delta_prime` — gal-fed SLAYER now
  works. Update `runtests_slayer_runner.jl` accordingly.
- **Verification**: gal Δ′ values byte-identical to the pre-unification `galerkin.*`
  datasets (only paths/fields move); riccati decks byte-identical throughout; result-struct
  testsets extended for the unified field on both formalisms; gal harness cases re-baseline
  (h5paths updated).

## 7. Interface PR, commit (c) — `solve` API

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
end

# D13: inner-layer matching config, integrator-agnostic (NOT part of any integrator struct)
Base.@kwdef struct ResistiveMatch
    model = InnerLayer.GGJModel(solver=:ray)   # swappable inner layer; backend knobs
                                               # (xfac/nx/nq/cutoff/kmax ← gal_inner_*) live on the model
    eta::Vector{Float64} = Float64[]           # per-surface, core→edge   (← gal_eta)
    rho::Vector{Float64} = Float64[]           #                          (← gal_rho)
    rotation::Vector{Float64} = Float64[]      # Hz; γ_s = 2πi·n·f_s      (← gal_rotation)
    gamma::Float64 = 5 / 3                     #                          (← gal_gamma)
    ideal::Bool = false                        #                          (← gal_ideal_flag)
end
```

`match !== nothing` replaces `gal_match_flag`. Inside `solve`, matching dispatches per
integrator: Galerkin → `gal_match_rpec`; Riccati → errors "not yet implemented" until
the STRIDE resonant-matching PR lands (that PR also renames/deprecates the `gal_*`
matching TOML keys — until then the TOML keys map onto `ResistiveMatch` internally).

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
                 match::Union{Nothing,ResistiveMatch}=nothing,
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

## 7A. Interface PR, commit (d) — cross-formalism file contract (issue #388 items 1 + 2)

Two #388 follow-ups belong to THIS PR because they complete the D14 promise (same physics ⇒
same layout across integrators) and close a gap this PR itself introduces:

- **ξ axis-order unification (#388 item 1, MINIMAL form only)**: `ForwardIntegration/xi_psi`
  is `(mode, solution, psi)`; `GalerkinIntegration/Solution/xi_psi` is `(mode, psi, solution)` —
  same physics, transposed. Fix: keep the per-formalism groups; `permutedims` at gal write so
  every `Solutions/*/xi_*` dataset shares the Forward axis order; update the gal `dims`
  attributes (and drop the axis-order warning from the gal `long_name`); update the gal
  readers (`benchmarks/verify_gal_{solution,ideal,match}.jl`, `compare_gal_vs_el.jl`, and any
  Analysis readers). The FULL unification (one Solutions layout written from
  `result.solution`) is future work — gal-native grid semantics make that a design job.
- **`Tearing/PerSurface/rational_psi` + `rational_q` (#388 item 2)**: gal-fed SLAYER (new in
  commit (b2)) analyzes gal's in-domain SUBSET of the singular surfaces, so tearing output
  must identify its own surfaces. Write both datasets from the surface list `run_slayer`
  actually used; `rational_q` for schema symmetry; annotate per hdf5-conventions.

Verification: gal ξ values identical under transposed layout (compare vs pre-change with an
axis-aware script); forward decks byte-identical; slayer tests extended for the new
datasets; harness gal cases re-baseline once (with the (b2) renames).

NOT in this commit (stay on #388): item 3 (PE empty-placeholder pattern — align with #368's
zero-extent sentinels after it lands), items 4–5 (schema-owner calls), items 6–8
(comment-audit pass, #354 pattern).

## 7B. Settled design (2026-08-15): source algebra, two-stage PE, deck-as-serialization

Discussion CLOSED with the user; decisions D15/D16 below are binding. Commit (c) is
implemented but UNCOMMITTED, so its concrete `RMPField` is REPLACED in place (no shim).

### D15 — `RMPField` is abstract, with lazy linear algebra (lands in the (c) revision)

- `RMPField` = the user-facing ABSTRACT supertype of every forcing source. File modes,
  coil set + currents, or (future, #377) fields given on ψ=1 / an arbitrary surface via
  equivalent surface currents — "they are all just external fields." Constructors on the
  abstract type return concrete internal subtypes (today: one leaf wrapping
  `ForcingTermsControl`; a surface-field leaf arrives with #377).
- Lazy `+`, `-`, scalar `*`: return a formal linear combination WITHOUT materializing.
  Valid because PE is linear in the forcing — materialization commutes with summation.
  Both current leaf kinds materialize to the same normalized `Vector{ForcingMode}` basis,
  so summation = match (m,n), add amplitudes. Prefer ComplexF64 scale (coil phase
  rotation is physical); scale must apply to the MATERIALIZED modes, format-independent.

### D16 — the deck is the API, serialized (one path)

Every TOML section corresponds 1:1 to an API object/call; the keys ARE the kwargs
(the `@kwdef` splat is the mapping). Consequences, in delivery order:

1. **#393 (this PR)**: (c) revision per D15 + commit (d). Nothing else grows scope.
   ctrl→TOML serialization explicitly deferred to step 3.
2. **Stacked PR: two-stage PE** — `GeneralPE = perturbed_equilibrium(ffs)` builds the
   source-independent response/coupling operators; `force(GeneralPE, fields)` (or
   callable `GeneralPE(fields)`) materializes sources, applies P, computes derived
   quantities. Pairs with the delta_mn resonant-coupling work. Payoff: coil scans and
   optimization reuse one GeneralPE across many cheap force() calls; a TOML deck maps
   onto "GeneralPE + one force()" with no deck-format change.
3. **Follow-on PR: "main = 20 lines"** — kinetic + SLAYER get API entry points;
   `main()` becomes a deck INTERPRETER (parse file → same constructors and calls a
   script would make); `main_from_inputs` and the stage functions dissolve. The writer
   serializes the RESOLVED ctrl structs (defaults included) into every output — same
   blob for TOML and API runs — so every gpec.h5 is replayable and h5→toml regeneration
   is just extracting it. Scripting users get the SAME per-section loaders main uses
   (e.g. `PlasmaEquilibrium("case_dir/")` reads the `[Equilibrium]` section); no second
   config system, ever. Deck completeness is automatic: the deck schema IS the struct
   schema, and TOML array-of-tables (`[[ForcingTerms.source]]` with per-block scale)
   serializes even the source algebra.

Defaults contract (established, keep): both paths splat over the same `@kwdef` struct
defaults — one defaults table. API is deliberately more explicit in two spots (no
default alg; `nn` required, `nn_low/nn_high` kwargs rejected). Deprecated deck keys
warn-and-ignore; unknown API kwargs hard-error (decks are archival, scripts fail fast).

### Reviewer constraints from Nik (Slack, 2026-08-15 — binding on the follow-on PRs)

- **No source-type zoo.** The common currency is the control-surface spectrum per source;
  keep the concrete RMPField kinds minimal. Endpoint: at most ONE more leaf kind, ever — a
  spectrum-literal ("here are control-surface modes, computed elsewhere") — and the #377
  equivalent-surface-currents solve becomes a UTILITY converting fields-on-a-surface into
  that spectrum, NOT a type. External couplings (thincurr/surfmn/ferritic tools) cost GPEC
  zero adapters: they produce spectra, directly or via the utility.
- **`scale` is a linear-combination weight, never a physical amplitude** (amplitudes are
  ambiguous for magnetic materials, coil sets with dropouts, etc.). A degraded coil set is
  `nominal - failed_coil`, not `0.9 * nominal`; material fields are computed at the
  operating point by the code owning their physics, weight meaningful only for small linear
  excursions. Docstrings reworded accordingly (2026-08-15, in the (c) revision).
- Nik explicitly likes the multi-shift/tilt-in-one-run capability (his bookkeeping win) —
  keep it central in the two-stage-PE PR spec.

### North-star usage sketch (user's, verbatim intent; syntax deliberately sloppy —
### requirements catalog for the two-stage-PE PR, NOT #393 scope)

```julia
Source_A = RMPField(coil1)
Source_B = RMPField(ferritic_material_fields_at_psi1)          # needs #377
Total_fields = Source_A + Source_B          # fast: just records both sources

GeneralPE  = perturbed_equilibrium(ffs_result)
SpecificPE = force(GeneralPE, Total_fields) # Biot-Savart for A, Laplace/current-potential
                                            # solve for B, sum on the control surface,
                                            # apply P, derived quantities per output flags

# Error-field sensitivity workflow: per-unit sources built by coil manipulation + algebra
PF1U_nominal = RMPField(pf1u_dat, 1)                # 1 A
PF1U_shifted = shift_coil(PF1U_nominal, 1e-3) - PF1U_nominal   # field per mm of shift

# Named source SETS: force() runs per key, results in per-key (xarray-like) datasets
rmp_set = ("PF1U_shift"=PF1U_shifted, "PF1U_tilt"=PF1U_tilted,
           "ferritic_welds"=surfmn_fields, "REMC"=thincurr_fields)
iter_pe = force(GeneralPE, rmp_set)

# Keyed, labeled linear algebra on operators and results ("@" = xarray-like matmul):
overlaps_per_amp_per_mm = GeneralPE.C_xe @ iter_pe.Phi_sources_root_area_normalized

# Collapse per-unit sources to a physical case: keyed scalar sets with wildcards,
# elementwise multiply, then sum to a single total field
tilts_shifts = ("PF1U_shift"=1.1e-3, "PF1U_tilt"=0.9e-3, "ferritic_welds"=1)
currents     = ("PF1U_*"=14e3,)
total        = sum(tilts_shifts * currents * rmp_set)
real_pe      = force(GeneralPE, total; profile_output=true)
jbgradpsi    = real_pe.Jbgradpsi
```

Requirements this implies for the two-stage-PE PR (catalogue, to be specced there):
named source sets with per-key PE results; coil-geometry manipulation (`shift_coil`,
tilts) composing with source algebra to build per-unit error-field bases; keyed scalar
sets with wildcard matching, elementwise `*` against source sets, `sum` collapsing to
one field; labeled (xarray-style) operator/result access so couplings contract naturally
per key; a `profile_output`-style flag family for derived profile quantities.


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

## 9. Sanity map: capability targets by integrator

This is the TARGET matrix (D14): outputs representing the same physics are unified across
integrators — one field, one data type, regardless of which formalism produced it. Outputs
fall into three physics classes:

- **Control surface**: quantities on the plasma boundary (`wp`, `free_boundary` energies).
  Every integrator can supply these (gal pending its δW implementation).
- **In-plasma class 1 — full profiles**: ξ/ξ′ (or equivalent) across the volume
  (`solution`), used to construct spectral, full-volume perturbed equilibria.
  Forward and matched-Galerkin only; Riccati will NEVER produce these.
- **In-plasma class 2 — rational-surface resonant data**: quantities AT the rational
  surfaces that quantify island-opening drive: `bpen`, and (future) `delta_mn` — the
  matrix encoding the jump in the pitch-resonant derivative of the solution at each
  rational surface, from outer-solution asymptotics (for Riccati: recoverable from
  `delta_coil`). `delta_mn` yields the perturbed current and the shielded resonant flux,
  and is what PE's resonant coupling will consume — no full profiles required.

Legend: ✅ implemented · 🔜 target pending the named follow-on work · ❌ never · — N/A.

| Output | Forward | Riccati | Galerkin |
|---|---|---|---|
| `wp` (control surface) | ✅ | ✅ | 🔜 gal δW work |
| `free_boundary` energies (control surface) | ✅ | ✅ | 🔜 gal δW work |
| `solution` — full ξ/ξ′ profiles (class 1) | ✅ `:el_axis` | ❌ (class 2 covers resonant coupling) | ✅ `:gal_native` |
| `closure` / `bpen` (class 2; always present, zeros under `:ideal`) | ✅ `:ideal` | ✅ `:ideal` (🔜 `:matched` with STRIDE matching) | ✅ `:ideal` or `:matched` |
| `delta_mn` (class 2; resonant-derivative jump) | ❌ not planned (no concrete route identified; may not exist) | 🔜 next-week work, from `delta_coil` | 🔜 next-week work |
| `delta_prime` — ONE unified type: Δ′ matrix, raw D′, `delta_coil`, PEST-3 blocks (THIS PR) | — | ✅ → unify | ✅ → unify (same physics, today under `galerkin.*` fields / different HDF5 names) |
| raw integrator odet (`diagnostics`: crit, nzero, edge scan, ca) | ✅ | ✅ | — (no radial ODE sweep) |
| kinetic (`kinetic_factor>0`) | ✅ | error | error |
| SLAYER inputs (surfaces + Δ′ matrix) | surfaces only (diag fallback) | ✅ | 🔜 via unified `delta_prime` (this PR) |

SLAYER is an inner-layer consumer: SLAYER + GGJ should eventually sit behind one abstract
inner-layer interface (same family as the `ResistiveMatch` models, D13). Later pass, not this one.

## 10. Progress

### Live status (updated 2026-08-15 — read this first when resuming)

- **#381 and #387 MERGED into develop** (a0c270f8, 2026-08-15): riccati unification +
  LocalStability module are in. Branches deleted; #393 auto-retargeted to develop and
  shows MERGEABLE.
- **Interface PR = #393** (`refactor/forcefreestates-result`, worktree `../result-pr3`,
  DRAFT, base = develop):
  - Commit (a) = 8f8e1645, done: result struct + SolutionProfiles + closure/bpen/wp +
    standalone Galerkin + additive-gal removal. Verified: 82/82 result-struct tests,
    357/357 across six files, forward byte-identity (145 datasets), gal-group equivalence
    (LAR_ideal_match_test, 12+16 datasets) — all vs f8996d4f, i.e. PRE-#364 base.
  - Commit (b) committed: staged-main decomposition
    per §6. Verified pure motion — normalized diffs of every stage body vs its old inline
    block are character-identical (only function-boundary lines differ); both
    force_termination early-exits preserved; one inert reorder (local stability hoisted
    ahead of sing_lim!/sing_find!; it reads only equil). Gates: 82/82 result-struct
    tests; fresh byte-identity of the coarsened Solovev fixture vs the commit (a)
    artifact, 143/143 compared datasets identical (145 total incl. git_version + toml
    blob). Review protocol for motion commits: read resulting functions top-down +
    behavioral gates, NOT the raw diff; locally use `git diff --color-moved=dimmed-zebra
    --color-moved-ws=allow-indentation-change --histogram`.
  - Commit (c) implemented, reviewed, and REVISED per D15 (not yet committed): solve API
    per §7, then RMPField reworked in place — now an ABSTRACT type with RMPSource leaf
    (ComplexF64 scale) and RMPFieldSum lazy linear combinations (+, -, scalar *; flattened
    term list); sum materialization evaluates each leaf via a scratch
    PerturbedEquilibriumInternal and merges amplitudes per (n,m), sorted;
    compute_perturbed_equilibrium accepts Union{ForcingTermsControl,RMPField}; algebra
    tests added (type-level testset + one PE call asserting 3A-A == 2A); api.md gained a
    Combining-forcing-sources section. THEN materialization made PURE (user request, fewer
    !-functions for multithreading): materialize_forcing_modes(ffs, forcing; dir_path,
    preloaded_coil_sets, verbose) -> (modes, coil_sets), three dispatch methods, no
    mutation; the preload guard + state writes live ONLY in compute_perturbed_equilibrium
    (double-apply bugs structurally impossible); driver pre-materialize call deleted.
    scale reworded everywhere per Nik: linear-combination WEIGHT, never physical amplitude
    (dropout example: nominal - failed_coil, not 0.9*nominal). Final gates: 70/70 solve
    API + 17/17 fullruns after the refactor; docs build clean.
    THEN problem-type form added (user design call): EulerLagrangeProblem(equil; nn, wall,
    match, dir_path, debug, ctrl kwargs) names WHAT is solved (SciML problem/alg split —
    PlasmaEquilibrium hosts many future problems, so solve(eq, alg) alone was namespace-
    greedy); solve(prob, alg) is canonical, solve(eq, alg; kwargs...) retained as sugar
    forwarding to it; nn_low/nn_high rejection lives in the problem constructor. Name
    chosen over StabilityProblem because kinetic runs make stability an imprecise label.
    Deviations recorded: `solve` lives in the TOP module (prepare_force_free_states!
    needs the KineticForces callback; FFS cannot import KineticForces — same CommonSolve
    generic, so ForceFreeStates.solve still resolves); ResistiveMatch is a plain config
    mapping 1:1 onto gal_* keys (forces gal_rpec_flag=true); solve mirrors TOML side
    effects (HDF5 write, local stability); forcing materialization unified in
    PerturbedEquilibrium.materialize_forcing_modes! and the TOML driver rewired through
    perturbed_equilibrium (ONE forcing path). Verified: 59/59 solve-api + 114/114
    result-struct + 17/17 fullruns (agent + independent rerun), TOML byte-identity
    207/207 datasets after the rewiring, docs build exit 0.
    FOUND pre-existing bug (filed as #396, cross-linked from #377): TOML file-forcing
    never applies convert_forcing_normalization! (snapshot preloads raw modes; the
    isempty guard skips the convert branch) — factor 16.85 on Solovev amplitude-linear
    PE outputs; present since the forcing-snapshot PR; NOT fixed here (needs a design
    decision re: replay double-conversion; fixing moves TOML outputs).
  - Commit (d) planned (§7A, decided 2026-08-15): #388 items 1 (ξ axis-order unification,
    minimal permutedims-at-write form) + 2 (Tearing/PerSurface rational_psi/rational_q).
    Remaining #388 items stay on the issue for follow-ups.
  - Commit (b2) implemented and reviewed (not yet committed; §6A, D14): `DeltaPrimeData`
    (now in ForceFreeStatesStructs.jl for include order) carries matrix/raw/coil + gal-only
    A/B/Gamma; `galerkin_solve` returns `(GalerkinResult, DeltaPrimeData)`; canonical HDF5
    paths `SingularSurfaces/{Delta_prime_matrix,Delta_prime_raw,Delta_coil,pest3_*}` written
    once from `result.delta_prime`; `GalerkinDeltaPrime/` group deleted (per-surface
    identifiers moved to `GalerkinIntegration/`); gal-fed SLAYER works. Convention gate
    verified (PEST-3 combinations term-identical). Found+fixed pre-existing bug: old gal
    `Delta_prime_raw` dataset was (2msing+mpert)×2msing with coil rows duplicated inside.
    Verified: 114/114 result-struct, 71/71 slayer (independently rerun), gal Δ′ values
    byte-identical under new paths (147/147 common), forward deck untouched (138/138),
    benchmarks/ readers repointed. Harness gal_resistive_diiid triage CLOSED: the "3
    changed" rows were the invoking repo's renamed case TOML reading develop's RICCATI
    datasets (the additive deck writes both formalisms, and riccati's datasets sit at
    exactly the new canonical names) against local's GAL datasets — cross-formalism
    apples-to-oranges, not numerical movement. Fresh dual-run proved gal==gal bit-for-bit
    (leading raw block isequal, pest3 diag ratio 1.0, coil isequal). Action: re-baseline
    the case once; harness cross-ref comparisons spanning the rename boundary are
    confounded for this case and should not be repeated.
    Also per D14: riccati will NEVER produce full ξ profiles — next-cycle work is the
    `delta_mn` rational-surface matrix (from `delta_coil` asymptotics) for PE resonant
    coupling, not profile reconstruction.
- **#364 reconciliation DONE** (merge commit b803788e in result-pr3): develop merged
  bottom-up (#381 ← develop, #387 ← #381, result-pr3 ← #381-combined). The FFS-writer
  conflict resolved as our-structure + #364's literature dataset names; two scope bugs
  in auto-merged #364 machinery fixed (`write_root_attrs!` and `apply_main_h5_metadata!`
  referenced the deleted `intr` local); `dVdpsi_spline` kwarg threaded through
  `run_kinetic_forces`; `diiid_n1_riccati.toml` h5paths renamed (10 paths); stale
  `LocalStability/di|dr` docstring in Ballooning.jl fixed (stale on develop too).
  Post-merge smoke: 82/82 result-struct + 66/66 slayer.
  STILL OWED: fresh byte-identity + gal-equivalence re-runs vs the post-merge base, full
  suite, docs build, and one harness re-baseline.
- Standing decisions in force: `_chord_solution_at` retained as uncalled helper (§5.3 —
  do not re-delete); gal→PE warn-skips this cycle (no gal δW yet); matching work lands
  in a new `Matching/` directory (§ follow-on); directory reorg is a separate post-#367
  post-formatter-PR pure-move PR — never folded into feature commits; comment-audit PRs
  follow the #354 pattern, separate from moves.
- Process rules (unchanged): ask before EVERY commit and EVERY push; no formatter ever;
  slice-pure commits; third-party human review before ANY merge — non-negotiable.


- [ ] PR 1 — `refactor/riccati-unification` — **implemented, in review.** Two deltas
  from the §3 spec, both improvements: the new Δ′ example references the DIIID geqdsk
  by relative path instead of copying it, and the TOML sweep covered six regression
  fixtures (two more had landed on develop since the plan was written), all `forward`.
- [ ] PR 2 — `refactor/local-stability-module` — **implemented, in review.** One delta
  from the §4 spec: the signature change also required updating two call-site groups the
  section did not list — `examples/DIIID-like_ideal_example/analyze_example.jl` (five
  ballooning entry points) and two docstring cross-references in
  `src/Analysis/ForceFreeStates.jl`.
- [ ] Interface PR (`refactor/forcefreestates-result`) — three commits: (a) §5, (b) §6, (c) §7.
  Commit (a) — **implemented (re-sliced §5), reviewed.**
  Carries the pivot: no transitional API. `SolutionProfiles` is the one solution slot,
  `closure`/`bpen` are unconditional on the result, standalone Galerkin and additive-gal
  removal are pulled forward from PR 4, and `pe_solution` / `gal_matched_odestate` are
  deleted rather than deferred. Deltas from the §5 spec:
  1. §5 did not say how the ForceFreeStates kernels PE calls keep working once
     `ForceFreeStatesInternal` stops crossing the module boundary. Added an abstract
     `ModeSpace` supertype (`ForceFreeStatesStructs.jl`) that both
     `ForceFreeStatesInternal` and `ForceFreeStatesResult` subtype, and relaxed the
     mode-space-only kernels to it: `el_derivatives!`, `materialize_derivative_stores!`,
     `build_kinetic_metric_matrices`.
  2. `ForceFreeStatesResult` is parameterized on the equilibrium and `FourFitVars` types
     (both are themselves parametric), so `result.equil` / `result.ffit` stay concretely
     typed instead of becoming inference barriers on the PE hot paths.
  3. Two call sites outside `src/` consumed `main`'s old named tuple and are updated:
     `benchmarks/benchmark_diiid_ideal_ntv_torque.jl` and
     `examples/DIIID-like_ideal_example_IMAS/run_imas_example.jl`.
  4. Of the tests §5.4 lists for update, only `runtests_imas.jl` needed it —
     `runtests_fullruns.jl`, `runtests_rerun_from_h5.jl` and
     `runtests_parallel_integration.jl` never read `main`'s return value (the last drives
     the low-level API directly and is unaffected). Coverage was added instead to
     `runtests_slayer_runner.jl` (result-facing `run_slayer` dispatch) and
     `runtests_imas.jl` (the `free_boundary === nothing` warn-and-skip).
  5. `_chord_solution_at` (PerturbedEquilibrium/SingularCoupling.jl) is deleted: with
     `SolutionProfiles.du_store` populated by contract, its `!du_store_populated` branch is
     unreachable. The gal-native / ideal-EL / kinetic branches are unchanged.
  6. The `integrator` TOML description changed in all 21 decks that carry the key (the
     three-way value list), per the identical-descriptions rule in
     `docs/development/toml-conventions.md`.

  Accepted output changes (D10), all spec'd in §5.1/§5.3/§5.4:
  - Riccati decks write `ForceFreeStates/Solutions/ForwardIntegration/xi_psi` and `u2`
    empty instead of the sparse chunk-endpoint snapshots (`dxi_psi`/`xi_s` were already
    empty there). No harness case tracks those datasets.
  - The four gal decks become gal-only files: their Galerkin datasets are unchanged, and
    the FFS-side integration/energy datasets that the removed additive Riccati run used to
    produce are now empty or absent. Verified dataset by dataset (§5.5 gate c).

  Observation for a later PR, not changed here: `result.bpen` has `msing` rows counted
  from `intr.sing` under `:ideal` closure but from the Galerkin surface set under
  `:matched`. The two can differ when `sing_min!` raises `psilow`. This reproduces the
  pre-pivot behavior exactly (the driver previously assigned `gal_data.match.bpen`
  directly, and `SingularCoupling` guards with `s <= size(inner_bpen, 1)`), so it is a
  pre-existing row-alignment wart, not a regression.

  - [ ] Commit (b) — staged `main` (§6)
  - [ ] Commit (c) — `solve` API (§7)
- [ ] Fortran re-comparison of all important quantities
- [ ] Delete this file
