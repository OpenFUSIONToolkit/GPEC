# GPEC Julia — ForceFreeStates modularization & `solve(eq, integrator)` API
## Complete multi-PR implementation plan

> **NOTE FOR ALL DEVELOPERS (read this first).**
>
> **EXECUTION STATUS (2026-08-17) — where to pick up.** Most of this document is
> implemented history kept for reference: §§3-4 are MERGED (#381, #387), §§5-7A are the
> five commits of **#393 (MERGED 2026-08-17)**, and #400 (FFS reorg, pure move) is open,
> now retargeted onto develop. **The live truth is §10 "Live status" — read it first when resuming.**
> The NEXT work, in order: the **§7C matching PR** (commits (0)-(4); commit (0) is
> startable now, the rest stack on Jake's upcoming FourFitVars-split PR), then the
> **§7D interpreter PR**, then the two-stage PE (§7B item 3). Design decisions D1-D18
> are settled — do not re-litigate. Nothing in §§3-7A is left to execute.
>
> This document is the agreed plan for the refactor of the
> ForceFreeStates ↔ PerturbedEquilibrium interface and the top-level driver, originally
> delivered as THREE pull requests: #381 (integrator unification), #387 (LocalStability),
> and one combined "interface PR" (#393) whose commits carry what were originally planned
> as PRs 3-5 (the stack was collapsed once it became clear reviews would batch at the
> end), and since extended with the follow-on stack above. It is
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

| PR | Branch | Status | Content |
|----|--------|--------|---------|
| #381 | `refactor/riccati-unification` | **MERGED** | Delete serial-Riccati + `populate_dense_xi` + `parallel_threads`; `integrator=` ctrl key; `nchunks` knob; thread-independent chunking; shooting→forward rename |
| #387 | `refactor/local-stability-module` | **MERGED** | Extract Ballooning.jl → `LocalStability` module; drop ctrl dependency (stacked on #381) |
| #393 | `refactor/forcefreestates-result` | **MERGED** | ONE PR, five slice-pure commits: **(a)** §5 `ForceFreeStatesResult` + warn-and-skip consumers + standalone Galerkin; **(b)** §6 staged `main`; **(b2)** §6A unified Δ′; **(c)** §7 `solve` API + RMPField algebra; **(d)** §7A ξ unification |
| #400 | `refactor/forcefreestates-reorg` | **MERGED** | Pure-move FFS reorg into subdirectories; seeds `Matching/` (basis-free `resonant_match_rpec` kernel) |
| §7C PR | (post-solve matching) | **NEXT — not started** | Stacked DIRECTLY on #400. Commits (0)-(4): kinetic-on-equilibrium, `InnerLayerModel` + layer-parameter builder, `MatchProblem`, `TearingProblem`, scan benchmarks |
| #383 | `refactor/freeze-fourfitvars` | **MERGED** | FourFitVars → `MatrixSplines`: `ffit` → `mats`, `build_matrix_splines`, `*_spline` fields |
| §7D PR | `refactor/main-deck-interpreter` | **after §7C** | ctrl→TOML serialization; `main` as deck interpreter |

Commit discipline for the interface PR: commit boundaries now do the job PR boundaries
did — keep each commit slice-pure (fixes amend into the right slice before review
starts; ordinary follow-up commits after). Per-slice numerical isolation stays
verifiable via the harness with commit SHAs as refs.

---

## 3. PR 1 — `refactor/riccati-unification` (MERGED as #381)

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

## 4. PR 2 — `refactor/local-stability-module` (MERGED as #387)

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

## 5. Interface PR, commit (a) — result struct, consumers, standalone Galerkin (IMPLEMENTED, in #393)

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

## 6. Interface PR, commit (b) — staged `main` (staging ONLY — gal work is in commit (a)) (IMPLEMENTED, in #393)

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

## 6A. Interface PR, commit (b2) — unified Δ′/matching payload (D14) (IMPLEMENTED, in #393)

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

## 7. Interface PR, commit (c) — `solve` API (IMPLEMENTED, in #393; `match=`/`_apply_match!` are REMOVED again by §7C per D17)

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

## 7A. Interface PR, commit (d) — ξ unification + tearing surface identity (IMPLEMENTED 2026-08-15)

Final scope (converged with the user; supersedes the earlier "minimal transpose" reading):

- **ξ unification (the real one)**: closed axis-to-edge ξ profiles are written from
  `result.solution` into the producing formalism's Solutions group with IDENTICAL names and
  (mode, solution, psi) axis order: `Solutions/ForwardIntegration/*` (unchanged) and new
  `Solutions/GalerkinIntegration/{psi, q, xi_psi, dxi_psidpsi, xi_s}` (the gal grid, issing
  nodes dropped — the same arrays as `result.solution`, which IS `Match/xi` repacked). The
  gal closure (ideal jump or inner-layer Δ) always yields these profiles; a no-closure gal
  run is Δ′-only and writes none. `Match/xi`/`Match/dxidpsi` datasets are REMOVED (they were
  the profiles, mislabeled as matching diagnostics); `Match/` keeps cout/cin/Delta_r/bpen/
  rpec_eig/Inner/ only.
- **Raw outer basis demoted to debug output** (user call: solver internals, like dumping an
  ODE work array): the old `GalerkinIntegration/Solution/` group is now `Basis/`, written
  ONLY under the new `DebugSettings.gal_basis_output` flag ([DEBUG] deck section / `debug=`
  API kwarg), transposed to the shared axis order. `ForceFreeStatesResult` now carries
  `debug_settings` so the writer sees the flag. verify_gal_{solution,ideal}.jl need the flag.
- **Tearing surface identity (#388 item 2)**: `SLAYERResult` gained `rational_psi`/
  `rational_q` (aligned with `params`; empty when built from bare parameters);
  `run_slayer_from_inputs` takes them as kwargs; the loose `run_slayer` fills them from
  `surfaces[p.ising]`; writer emits `Tearing/PerSurface/rational_psi|rational_q` when
  present + annotations. Gal-fed SLAYER output now identifies its surface subset.
- Benchmarks repointed (verify_gal_match/ideal/solution, compare_gal_vs_el,
  scan_{rotation,resistivity}_m2, compare_jbgradpsi_m2 — the filtered psi grid is now
  first-class so several scripts simplified); annotation tables updated (axis-order
  warning dropped); hdf5-conventions.md updated; result-struct testsets assert
  file == result.solution + Basis gating; slayer round-trip asserts surface identity.

NOT done (stays on #388): item 3 (PE empty-placeholder pattern — align with #368), items
4–5 (schema-owner calls), items 6–8 (comment-audit pass). Full shared-Solutions schema for
closed profiles across formalisms (one group, grid-semantics contract) is future work with
the two-stage PE.

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
   ctrl→TOML serialization explicitly deferred to the interpreter PR.
2. **RESEQUENCED 2026-08-17** (design: D17/D18 below; PR specs: §7C/§7D). The delivery
   stack is now: #393 → **#400** (FFS reorg, pure move — already seeds
   `src/ForceFreeStates/Matching/` with the basis-free `resonant_match_rpec` kernel) →
   **§7C matching PR, stacked DIRECTLY on #400** (kinetic-on-equilibrium +
   MatchProblem/TearingProblem; the former "SLAYER API entry point" idea is SUBSUMED by
   TearingProblem — no `tearing_stability` verb). **Jake's FourFitVars-split PR stacks
   on top of OUR work** (Slack, 2026-08-17 evening: he defers to the next morning and
   builds on whatever we have) — so keep `ffit`-facing touches
   (`compute_node_xi_s!` / `compute_sing_asymptotics` consumers) minimal and localized
   to ease his split. Then →
   **§7D interpreter PR** (`main` = deck interpreter; ctrl→TOML serialization; h5→toml).
   The interpreter must come LAST: it interprets the final call sequence
   `solve` → optional MatchProblem solve → `perturbed_equilibrium` → TearingProblem
   solve → NTV. The interpreter-PR content itself (serialization semantics, deck schema
   = struct schema, per-section loaders, no second config system) is unchanged — see
   §7D.
3. **Then: two-stage PE (stacked, AFTER FFS is closed)** — `GeneralPE =
   perturbed_equilibrium(ffs)` builds the source-independent response/coupling
   operators; `force(GeneralPE, fields)` (or callable `GeneralPE(fields)`) materializes
   sources, applies P, computes derived quantities. Pairs with the delta_mn
   resonant-coupling work (same territory, same cycle). Payoff: coil scans and
   optimization reuse one GeneralPE across many cheap force() calls; a TOML deck maps
   onto "GeneralPE + one force()" with no deck-format change.
   **RE-SCOPED 2026-10 (ErrorFields landed)**: the new `ErrorFields` module (merged
   Sept, ~10 PRs) delivers the error-field *workflow* payoff by a different route —
   it linearizes each coil set's control-surface spectrum over its six rigid dofs
   (`compute_coil_sensitivities`, `apply_transforms` ≈ shift_coil of the north-star
   sketch) and projects onto the PE `ResonantCoupling`/`dominant_coupling` (from the
   new PE SVD API), so overlap-type outputs never re-apply P. Faithful to D15 semantics
   (per-unit sources, spectra as currency). The two-stage PE therefore no longer owes
   the tolerance/overlap workflow; it still owes: per-source FULL PE fields (profiles,
   Jbgradpsi, per-source bpen/delta_mn — anything not linear-in-overlap), non-coil
   sources through one interface (spectrum-literal leaf, the #377 surface-current
   utility), ResponseMethod multiplicity + gal-fed PE, and the deck↔API unification.
   It should FEED ErrorFields' linearization (same ForcingMode/spectrum types, shared
   ForcingTerms grid machinery — largely true already), never duplicate it.

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

### Plasma-response methods (Fortran resp_index — binding requirement, 2026-08-15)

Fortran GPEC computes the plasma inductance / permeability P by FIVE selectable methods
(`plas_indmats(0:4)`, `resp_index`): j=0 = ENERGY method (wt0-based when
resp_induct_flag, else eigenmode energies et) — the Fortran default and the ONLY method
ported to Julia (`compute_plasma_response!`, Response.jl); j=1..4 = SURFACE-CURRENT
methods built from the four `kapmats`/`chpmats` variants (surface current κ and scalar
potential χ per identity-at-edge drive, gpresp_eigen → gpeq_surface at psilim) — these
need only the solutions' EDGE VALUES + vacuum Green's functions, NOT δW. Under gal_flag
Fortran computes only j=1 and forces resp_index=1: gal PE worked via surface currents
from the gal eigenfunctions.

Consequences (correcting the earlier "PE requires δW" premise):
- The Julia gal→PE skip is a PORTING GAP artifact, not physics: the one ported method is
  the one method gal cannot feed. Gal's matched solution already provides the
  identity-at-edge columns the surface-current methods consume.
- REQUIREMENT for the two-stage-PE PR: preserve method multiplicity — a ResponseMethod
  selection (energy | surface-current variants, the resp_index analog, as a typed
  argument not a magic integer), with the surface-current port unlocking gal-fed PE
  independently of the gal-δW work. The gal δW work remains scheduled for free-boundary
  stability of gal runs and method-0 parity.
- gal_resistive_pe harness expectations change when either route lands.

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


### Settled design (2026-08-17): matching is a post-solve transformation

#### D17 — MatchProblem / TearingProblem over one InnerLayerModel slot

- **Motivation (user call)**: inner-layer matching currently runs INSIDE the outer solve
  (`gal_match_flag` → `gal_match_rpec`, GalerkinSolve.jl:202), so scanning layer
  quantities (η, ρ, rotation) repeats the expensive FEM assembly + banded solve per scan
  point — the in-repo `scan_rotation_m2.jl`/`scan_resistivity_m2.jl` do exactly this.
  Per-iteration matching is cheap (msing small inner-layer solves + one 4msing×4msing
  linear solve + BLAS recombination of the stored basis), so matching becomes a
  POST-SOLVE transformation.
- **Two problems, one model slot**, both in the established `solve(prob, alg)` grammar:
  - `solve(MatchProblem(ffs; eta=, rho=, rotation=, gamma=, ideal=false), model)` —
    γ PRESCRIBED per surface (γ_s = 2πi·n·f_s): the driven/RPEC match. Returns a NEW
    `ForceFreeStatesResult`: `closure = :matched`, `bpen`/`deltar` filled, `solution`
    replaced by the matched profiles when the producing formalism retained a basis,
    everything else carried over (the result is immutable — a small rebuild helper
    constructs the new one). Scans reuse ONE outer solve across many cheap match solves.
  - `solve(TearingProblem(ffs; coupling_mode=, dc_type=, scan/AMR/pole knobs), model)` —
    γ FREE, root-found where Δ_inner(Q) = Δ′_outer. Returns the tearing result (today
    `SLAYERResult`). Subsumes the "SLAYER API entry point": no `tearing_stability` verb,
    no exported `run_slayer`, and never a bare `match` function (clashes with
    `Base.match`).
  - `InnerLayerModel` structs fill the alg slot: `GGJ(; solver=:ray|:galerkin, inner_*)`
    — MatchProblem today, TearingProblem once the γ-extraction validation flagged in
    `run_slayer.jl` lands — and `SLAYER(; mu_i, zeff, resistivity_model, lnLambda_form,
    χ fallbacks)` — TearingProblem ONLY (slab: single parity, no Δ₂, no reconstructable
    layer profiles; `MatchProblem` + `SLAYER` errors with exactly that physics message).
    Capability gating lives on the model type, same taxonomy as the integrators.
  - **`ResistiveMatch` dissolves**: its physics fields (eta/rho/rotation/gamma/ideal) →
    `MatchProblem` kwargs; its solver fields (inner_solver, inner_*) → the `GGJ` struct.
    The `match=` field on `EulerLagrangeProblem` and `_apply_match!` are REMOVED (#393
    is still in review — evolving them in the stacked PR is fine). The
    gal_match_*/gal_eta/gal_rho/gal_rotation/gal_inner_* deck keys keep working: the
    driver (and later the §7D interpreter) routes them into the MatchProblem call.
- **Match-completeness of the result (verified 2026-08-17)**: everything the match needs
  is already published by #393 — `delta_prime.raw`/`.coil` (gal: rpec_flag; riccati: BVP
  with wv), `surfaces`, `equil`, `ffit`, mode space (result <: ModeSpace; `resist_eval`
  reads only `intr.nlow`; `gal_resonant_surfaces` reads only
  psilow/psilim/sing/mlow/mhigh). The full raw gal basis (`galerkin.solution.xi`/
  `xi_deriv`, all 2·msing+mcoil columns on the full grid) is built unconditionally, and
  the matched outer profiles are pure recombinations of it (GalerkinMatch.jl:167-181);
  the new result's `solution` replaces the old, transparently to every consumer.
- **The cut solution is a diagnostic-only dependency**: `bpen` is read off the inner GGJ
  solution at layer center (pen × cin, GalerkinMatch.jl:152-158) BEFORE the composite
  block, and the cut background's b^ψ contribution carries singfac = m−nq → 0 exactly at
  ψ_s (GalerkinMatch.jl:227-228). Only the composite inner-region ξ/b graft
  (GalerkinMatch.jl:183-231 — the Match/Inner/ plot outputs) needs `xi_cut`, which
  CANNOT be rebuilt post-hoc (needs the FEM workspace + asymptotic series). Policy:
  `xi_cut` stays a solve-time OPT-IN (`cut_solution` knob on `Galerkin`; deck-driven
  matched runs imply it for byte-identity of Match/Inner outputs); `MatchProblem`
  warns-and-skips the composite output when it is absent. bpen/matched-profile scans
  need nothing extra.
- **Riccati matching**: #400's `Matching/ResonantMatch.jl` kernel
  (`resonant_match_rpec(delta_out_raw, delta_coil_raw, sings, equil, intr, ctrl)`, Wang
  et al. 2020 PoP 27, 122509 Eq. 11) is basis-free — a matched riccati result gets
  closure/bpen/deltar with `solution === nothing`; existing warn-and-skip gates handle
  every consumer. The MatchProblem solve unifies `gal_match_rpec` and this kernel onto
  ONE path (kernel = the matching system; the gal branch adds profile reconstruction
  when a basis exists), merges `GalMatchResult`/`ResonantMatchResult` into one type in
  `Matching/`, and strips the kernel's remaining `ctrl.gal_*` reads.
- **resist timing (user call)**: `resist_geometry` → `sing.restype` (η/ρ-free Glasser
  E,F,G,H,K,M; ResistEval.jl:200-211, driver call at :521) STAYS an always-on cheap
  surface diagnostic (ideal-relevant D_R, `SingularSurfaces/` datasets,
  harness-tracked). The η/ρ-dependent `resist_eval` → `GGJParameters` already runs at
  match time and formally becomes the first step of the inner-layer solve —
  ideal-closure runs never pay for it.

#### D18 — layer parameters derive from kinetic profiles; vectors are overrides

One shared per-surface layer-parameter builder:
`(surfaces, ffs.equil.kinetic, mu_i, zeff, resistivity_model, lnLambda_form)` →
per-surface (η_s, ρ_s, f_s), feeding `resist_eval` → `GGJParameters` (MatchProblem) and
`build_slayer_inputs` → `SLAYERParameters` (TearingProblem). SLAYER's existing builders
(neoclassical Sauter-F₃₃/Redl/Spitzer η, ρ from density, rotation from profiles) ARE the
machinery — promoted to shared, not duplicated. Explicit `eta=`/`rho=`/`rotation=`
vectors demote to overrides for artificial scans and to the no-kinetic-data fallback.
DEPENDS on kinetic-on-equilibrium (§7C commit 0). `[SLAYER] profile_file` keeps working
at deck level; the canonical profile home becomes the equilibrium.

## 7C. PR: post-solve matching — MatchProblem / TearingProblem (NEXT — NOT STARTED)

Branch stacked **directly on #400** (→ #393). Slack 2026-08-17: Jake builds his
FourFitVars split ON TOP of our work instead of the reverse — keep `ffit`-facing
touches minimal and localized so his split rebases cleanly over us. Slice-pure commits:

### (0) Kinetic profiles on the equilibrium (promoted — now a D18 dependency)
- `PlasmaEquilibrium` gains `kinetic::Union{Nothing,KineticProfiles}` — the LOADED profile
  data (already in the equilibrium's flux label / ψ₀ normalization), not a file path.
  Species/interpretation knobs (`zi`, `zimp`, `mi`, `mimp`, the *_factor scan knobs) are
  loader kwargs. Attach at construction (`PlasmaEquilibrium(path; kinetic_file=..., zi=...)`)
  or explicitly; the two-pass auto grid consumes `eq.kinetic` at formation.
- `solve` drops its kinetic error: `kinetic_factor > 0` gates on `eq.kinetic !== nothing`
  (clear error otherwise); `prepare_force_free_states!` reads profiles from the equilibrium.
- `load_kinetic_context` shrinks to kf_ctrl construction; the NTV stage reads `eq.kinetic`.
- Gate: kinetic harness cases byte-identical (solovev_kinetic_{ntv,calculated,nuzero}).

### (1) InnerLayerModel structs + shared layer-parameter builder (D18)
- `abstract type InnerLayerModel`; `GGJ`, `SLAYER` structs; the per-surface builder with
  profile-derived η/ρ/rotation and explicit-vector overrides. Unit tests: builder parity
  with `build_slayer_inputs` on a kinetic fixture; override precedence.

### (2) MatchProblem + solve (γ prescribed)
- `galerkin_solve` stops matching in-solve (its match branch is removed; `xi_cut` moves
  behind the `cut_solution` opt-in). `MatchProblem`/`solve` own the unified match path
  per D17 (one kernel; gal profile reconstruction when a basis exists; riccati
  basis-free). `match=`/`_apply_match!` removed from the API. The driver keeps decks
  working pre-interpreter: `run_force_free_states` returns the ideal-closed result and
  `main_from_inputs` immediately applies the post-solve match when `gal_match_flag`.
- Gate: matched outputs byte-identical vs the in-solve path (LAR_ideal_match_test,
  LAR_resistive_match_test, DIIID gal resistive decks incl. Match/Inner composites);
  full suite; harness sweep.

### (3) TearingProblem + solve (γ free)
- `TearingProblem` carries the matching-procedure knobs; `SLAYERControl` remains the
  single source of truth (deck `[SLAYER]` splat unchanged; `inner_model` key ↔ model
  type, exactly like `integrator` ↔ alg struct). `run_slayer_stage` rewires through
  `solve(TearingProblem(ffs; ...), model)`; layer parameters via the D18 builder
  (profile_file path preserved as the deck-level source until eq.kinetic is wired
  through SLAYER decks). Docs + autodocs + tests; SLAYER example byte-identical.

### (4) Scan benchmarks repointed
- `scan_rotation_m2.jl`/`scan_resistivity_m2.jl` collapse to ONE outer solve + a cheap
  MatchProblem loop; assert identical physics numbers vs the per-point re-solve and
  report the speedup in the PR body.

## 7D. PR: "main = 20 lines" — deck interpreter (branch refactor/main-deck-interpreter) (AFTER §7C — NOT STARTED)

Stacked on the §7C PR. Two commits (the former §7C (iii)/(iv), call sequence updated):

### (i) ctrl→TOML serialization (deck is the API, serialized)
- Generic struct→TOML-table serializer for the config structs (all RESOLVED values incl.
  defaults — snapshot semantics; skip nothing; deprecated keys never emitted; coil_sets_raw
  as array-of-tables). Sections from a result: Equilibrium (equil.config), ForceFreeStates
  (ctrl), Wall, DEBUG; PE/ForcingTerms/KineticForces/SLAYER when those stages ran.
- The FFS writer embeds this for API runs (today `Input/gpec_toml_raw` is TOML-path only) —
  every gpec.h5 becomes replayable; `write_deck(h5_path, toml_path)` utility = h5→toml
  regeneration. Round-trip test: deck → run → embedded blob → rerun → byte-identical.

### (ii) main as deck interpreter
- `main_from_inputs` becomes ~20 lines: parse deck → equilibrium (analytic/IMAS/rerun
  `additional_input` dispatch stays at this layer; kinetic attach per §7C(0)) →
  reverse-translate the flat `[ForceFreeStates]` table into (alg struct, MatchProblem
  kwargs, problem kwargs) — the inverse of `_apply_alg!`, with its own unit tests — →
  `EulerLagrangeProblem` → `solve` → optional `solve(MatchProblem, model)` →
  `perturbed_equilibrium` → `solve(TearingProblem, model)` → NTV → ErrorFields. Stage
  functions dissolve into `solve` or become internals; `run_force_free_states` is absorbed.
- **ErrorFields stage (added 2026-10)**: `run_error_fields` is already a thin interpreter
  over library calls (`ResonantCoupling` → `compute_coil_sensitivities` →
  `sensitivity_table`/`dominant_coupling` → `run_monte_carlo` → `locking_risk` →
  `tolerance_scan`/`efc_couplings`), so it dissolves the same way. Two specifics:
  (a) the NESTED-TABLE deck idiom (`[ErrorFields.MonteCarlo]`/`.Risk`/`.scenario`/`.NTV`
  + the separate tolerance TOML) must be handled by the ctrl→TOML serializer in (i);
  (b) the stage currently re-reads `[ForcingTerms]` into a `CoilConfig` and hard-requires
  coil format — under the API it should take the coil-source `RMPField` leaf and pull
  coil sets through the same materialization path PE uses (one-path rule).
- HDF5 write ordering: the writer runs on the FINAL (possibly matched) result, so the
  Match/ groups come off the matched result, never from inside a solve.
- force_termination early-exits, rerun/IMAS funnels, and return shape preserved.
- Gate: byte-identity on EVERY example deck class (forward incl. kinetic, riccati, gal
  ideal + resistive, SLAYER) vs the pre-commit tree; full suite; docs; harness sweep.

Verification discipline unchanged (§8): slice-pure commits, gates per commit, ask before
every commit/push, third-party human review before merge.

## 8. Cross-cutting execution rules (for every PR)

1. **Never merge without third-party human review. State this in every PR body.**
2. Every commit and push requires explicit per-instance maintainer approval.
3. Commit messages: `Area - TAG - message` (e.g. `ForceFreeStates - REFACTOR - Unify Riccati integrator`),
   with closed Area and TAG vocabularies per `docs/development/naming.md`.
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
| `closure` / `bpen` (class 2; always present, zeros under `:ideal`) | ✅ `:ideal` | ✅ `:ideal` (🔜 `:matched` via the §7C MatchProblem — basis-free kernel already on #400) | ✅ `:ideal` or `:matched` (🔜 post-solve per D17) |
| `delta_mn` (class 2; resonant-derivative jump) | ❌ not planned (no concrete route identified; may not exist) | 🔜 next-week work, from `delta_coil` | 🔜 next-week work |
| `delta_prime` — ONE unified type: Δ′ matrix, raw D′, `delta_coil`, PEST-3 blocks | — | ✅ | ✅ (PEST-3 blocks persisted; riccati recovers them via `pest3_decompose`) |
| raw integrator odet (`diagnostics`: crit, nzero, edge scan, ca) | ✅ | ✅ | — (no radial ODE sweep) |
| kinetic (`kinetic_factor>0`) | ✅ | error | error |
| TearingProblem inputs (surfaces + Δ′ matrix) | surfaces only (diag fallback) | ✅ | ✅ via unified `delta_prime` |

SLAYER + GGJ sit behind the D17 `InnerLayerModel` slot (§7C PR): GGJ serves both
MatchProblem and (pending γ-extraction validation) TearingProblem; SLAYER serves
TearingProblem only. `ResistiveMatch` dissolves into MatchProblem kwargs + the GGJ struct.

## 10. Progress

### Live status (updated 2026-10-06 — read this first when resuming)

- **2026-10 pickup**: ~42 PRs merged into develop between 08-26 and 10-05 while this PR
  was parked. Headlines: **#383 MERGED 08-26** (FourFitVars → immutable `MatrixSplines`;
  `result.ffit` → `result.mats`, `build_matrix_splines`/`build_kinetic_matrix_splines`,
  `*_spline` fields — the stacking question below is DEAD, §7C just builds on develop);
  **new 12th module `ErrorFields`** (coil-sensitivity linearization + tolerance Monte
  Carlo + locking risk + NTV-limited correction; see the 2026-10 re-scope under D16
  item 3 and the §7D ErrorFields-stage note — it is D15-faithful and pre-conformant to
  the interpreter vision); PE grew `ResonantCoupling`/`dominant_coupling` SVD API
  (#446) and multi-n PE (#477); Tearing/SLAYER moved (#403 Δ′ → r_s reference length
  before slab matching — audit the dp.raw↔deltar normalization contract during §7C
  commit (2); #431-434 fixes; OPEN: #441 toroidal Δ_crit geometry, #463 coupled
  determinant + Doppler, #415 b_crit — commit (3) should absorb/queue behind these);
  #385 per-stage runtimes in gpec.h5; #397 (draft) golden-values harness; #382 (open)
  TOML variable renames — interacts with D16/§7D, watch it. New repo rules: harness
  comparisons want COMMITTED refs (commit first, then `--refs develop,<branch>`), and
  commit subjects use the closed-vocabulary grammar — validate with
  `python3 ci/conventions/check_subject.py --title "..."`.
- **PR progress (2026-10-06)**: commit (0) COMMITTED as bde0f4c15 (all gates green incl.
  harness vs develop, fully unchanged); commit (1) COMMITTED as 56dff1801
  (`GGJ`/`SLAYER` configs on `InnerLayer.InnerLayerModel`, `closure_capable`,
  `layer_parameters`; 23/23 new tests, docs clean). **Commit (2) IMPLEMENTED in the
  working tree**: match extracted from `galerkin_solve` (cut solution behind
  `gal_cut_solution`/`Galerkin(cut_solution=)`, implied by `gal_match_flag`); ONE
  ctrl-free match path in `Matching/MatchProblem.jl` (`MatchProblem` + `solve` +
  `_compute_match`, the ported rmatch body) with the unified `MatchResult` in
  `Matching/ResonantMatch.jl` (old seed kernel + `GalMatchResult` both subsumed;
  `GalerkinMatch.jl` deleted); `_matched_result` rebuild; deck routing at the tail of
  `run_force_free_states` (both deck and API paths); `ResistiveMatch`/`_apply_match!`/
  `EulerLagrangeProblem.match` REMOVED; `resist_eval` loosened to `ModeSpace`; exports
  `MatchProblem`/`GGJ`/`SLAYER`/`layer_parameters`; api.md matching section rewritten
  with the scan idiom. Tests green (matching 23/23, solve API 71/71 incl. MatchProblem
  gates, result-struct 133/133 incl. matched-gal decks, fullruns 21/21); byte-identity
  runs vs 56dff1801 on LAR_{ideal,resistive}_match_test + DIIID gal resistive in
  flight; docs build pending; harness after commit.
- **Commit (0) third reconciliation DONE 2026-10-06**: branch reset onto develop
  0e68a0553; absorbed the `mats` renames, the runtimes threading, and ONE new
  kinetic-profiles consumer — `run_error_fields`/`efc_couplings` now reads
  `result.equil.kinetic` (efc_couplings keeps its explicit parameter; only the deck
  path sources it from the equilibrium). Develop also added `shift_exb_rotation`
  (kept alongside `attach_kinetic_profiles!` in KineticProfiles.jl). Gates re-running;
  stash `commit0-v2-pre-oct-reconciliation` is the pre-reconciliation backup (the older
  `commit0-kinetic-on-equilibrium` stash is obsolete — both droppable once committed).

### Historical status (2026-08-17/25 — superseded above, kept for context)

- **MERGED into develop**: #381 + #387 (riccati unification, LocalStability), #395 (CI
  pinned manifest), **#367 (input-struct freeze — we resolved its conflicts vs develop,
  applied Nik's 3 review items ourselves in worktree ../pr367, user merged; the worktree
  can be removed)**.
- **#393** (`refactor/forcefreestates-result`): **MERGED into develop 2026-08-17 as
  bb595659** (Jake approved; his review items applied as 3ed2365f; #400 auto-retargeted
  onto develop; worktree ../result-pr3 removable).
  All five commits in ((a) result struct, (b) staged main, (b2) unified Δ′,
  (c) solve API + EulerLagrangeProblem + RMPField algebra + pure materialization,
  (d) ξ unification + Basis debug-gating + Tearing surface identity). All gates green
  (full suite 61 testsets, docs, harness 13 cases: 11 unchanged + 2 accepted deviations
  documented in the PR body). develop (incl. #367/#395) reconciled in at d01ead0d —
  zero code changes needed for the freeze (construct-once already), byte-identical
  except #367's own new `Equilibrium/psihigh_resolved` dataset; PR comment posted for
  Nik. DO NOT MERGE without his re-look at the merge commit.
  **Jake reviewed 2026-08-17 (COMMENTED, "changes look great")**: 4 inline items —
  (1) FFSInternal split proposal → follow-on checklist item below; (2) Free.jl singfac
  FYI, self-recanted; (3) type annotations on `set_perturbation_data!` + (4)
  `ForceFreeStates_results` → `solution` rename — both applied and merged with #393.
- **#400: MERGED into develop 2026-08-25** (squash 6b5e8010) — pure-move FFS reorg
  into subdirectories (Riccati/, Surfaces/, Matching/, Galerkin/); it seeds
  `Matching/ResonantMatch.jl` with the basis-free `resonant_match_rpec` kernel (raw-Δ′
  in, riccati-capable, "bpen empty until Stage 2") and `Matching/DeltaPrime.jl`.
- **Jake's FourFitVars split is OPEN as #383** (`refactor/freeze-fourfitvars`, base
  develop, active 2026-08-25): `result.ffit` → `result.mats`,
  `make_matrix`/`make_kinetic_matrix` → `build_matrix_splines`/`build_kinetic_matrix_splines`,
  matrix fields renamed to `*_spline`. §7C commits (1)+ consume exactly that surface —
  whether to stack on #383 or land commit (0) first and rebase is the USER's call.
- **Develop moved 2026-08-18..25** (all reconciled into the §7C worktree on 08-25):
  multi-species NTV (`resolve_ntv_species`, a `species` object threaded through
  `load_kinetic_context`/`prepare_force_free_states!`/`run_kinetic_forces` — commit (0)
  keeps species resolution in `load_kinetic_context` as NTV-stage config; only the
  PROFILES move onto the equilibrium); #399 SLAYER physical-bt bugfix (tearing
  territory — commit (3) builds on the fixed behavior); #413/#419 FFS bugfixes;
  #404 repo conventions (PR bodies now carry a release-note block — use it when
  opening the §7C PR).
- **CURRENT WORK: the §7C matching PR** (design converged 2026-08-17 → D17/D18):
  matching as a post-solve transformation — `solve(MatchProblem(ffs; ...), GGJ())`
  returns a NEW ForceFreeStatesResult with `:matched` closure;
  `solve(TearingProblem(ffs; ...), SLAYER()|GGJ())` root-finds γ (subsumes the old
  "SLAYER API entry point" — naming question RESOLVED: no verb, solve grammar).
  Worktree `../matching-pr`, branch `refactor/post-solve-matching`, re-based onto
  develop at 6b5e8010. **Commit (0) (kinetic-on-equilibrium) is IMPLEMENTED in the
  working tree, uncommitted**, reconciled with the multi-species threading; ALL gates
  green (2026-08-25): tests (solve API 73/73 incl. new attach testset, KineticForces
  277/277, fullruns 19/19), harness vs 6b5e8010 fully unchanged on
  solovev_kinetic_{ntv,calculated,nuzero} + solovev_n1 (incl. the new 50/50 D-T
  multi-ion configs), local docs build clean. AWAITING the user's commit approval.
  A stash `commit0-kinetic-on-equilibrium` holds the pre-reconciliation version (drop
  once committed). Next: commits (1)-(4). The §7D interpreter PR comes AFTER (worktree
  `../main-interp` is stale at d01ead0d — its plan edits were carried here; it will be
  re-based/re-purposed; no PR opened). **The user wants the COORDINATOR (me)
  implementing directly — NOT background Opus agents.** This plan edit is UNCOMMITTED
  and should ride the first commit.
- Verification practice (unchanged, plus learned traps): gates per commit; byte-identity
  reference chain lives in the session scratchpad (`compare_h5.jl` walk-all-datasets
  script; `h5cmp_e/` = current post-reconciliation Solovev-fixture reference; rebuild
  references from any committed tip via a detached scratch worktree when in doubt).
  runtests.jl args are RELATIVE to test/; never pipe test output through tail (masks
  exit codes); harness `--force` bypasses its cache; harness cross-ref comparisons
  spanning the (b2)/(d) rename boundary are confounded for gal cases (documented in
  §6A/§7A — do not re-triage).
- Fortran GPEC clone for comparisons: `/Users/pharr/Projects/GPEC_dev/GPEC_julia_workspace/GPEC_fortran`.
- Loose ends (non-blocking): a10_kinetic_example + Solovev_ideal_example_3D never
  exercised on the new architecture (smoke runs offered, not run; a10 harness case
  worth proposing); Obsidian progress-log entry for the campaign offered, unanswered;
  issue #396 (forcing normalization skip) and #394 (directory reorg) open; #388 items
  3-8 remain on that issue.
- Process rules (unchanged): ask before EVERY commit and EVERY push; no formatter ever;
  slice-pure commits; third-party human review before ANY merge — non-negotiable.


- [x] PR 1 — `refactor/riccati-unification` — **MERGED as #381.** Two deltas
  from the §3 spec, both improvements: the new Δ′ example references the DIIID geqdsk
  by relative path instead of copying it, and the TOML sweep covered six regression
  fixtures (two more had landed on develop since the plan was written), all `forward`.
- [x] PR 2 — `refactor/local-stability-module` — **MERGED as #387.** One delta
  from the §4 spec: the signature change also required updating two call-site groups the
  section did not list — `examples/DIIID-like_ideal_example/analyze_example.jl` (five
  ballooning entry points) and two docstring cross-references in
  `src/Analysis/ForceFreeStates.jl`.
- [x] Interface PR **#393** (`refactor/forcefreestates-result`) — grew to FIVE commits:
  (a) §5, (b) §6, (b2) §6A, (c) §7, (d) §7A. **MERGED into develop 2026-08-17
  (bb595659, Jake approved).**
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

  - [x] Commit (b) — staged `main` (§6)
  - [x] Commit (b2) — unified Δ′ payload (§6A)
  - [x] Commit (c) — `solve` API + `EulerLagrangeProblem` + RMPField algebra (§7; revised per D15/D16)
  - [x] Commit (d) — ξ unification + Basis debug-gating + Tearing surface identity (§7A)
- [x] #400 — FFS reorg (Nik's, pure move) — **MERGED into develop 2026-08-25 (6b5e8010)**
- [ ] §7C PR — post-solve matching (MatchProblem / TearingProblem, commits (0)-(4)) —
  **NOT STARTED; this is where work picks up.** Stacked DIRECTLY on #400, startable now
  (Slack 2026-08-17: Jake stacks on top of us).
- [x] Jake's FourFitVars-split PR — **MERGED as #383, 2026-08-26** (`ffit` → `mats`)
- [ ] `ForceFreeStatesInternal` split into `ModeGeometry` / `SingularSurfs` /
  `IntegrationLimits` (Jake's #393 review item, agreed) — do AFTER his FourFitVars
  split lands; the `ModeSpace` supertype then dissolves into `ModeGeometry`.
- [ ] §7D PR — deck interpreter (ctrl→TOML serialization + `main` as interpreter)
- [ ] Two-stage PE (§7B item 3: GeneralPE/force, ResponseMethod multiplicity, delta_mn)
- [ ] Fortran re-comparison of all important quantities
- [ ] Delete this file
