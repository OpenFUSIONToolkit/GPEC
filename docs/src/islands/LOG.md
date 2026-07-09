# Islands — session LOG

Cross-session memory spine (design doc `06 §2.5`). Read this at session start
together with `QUESTIONS.md`. Append a short entry before every session end:
**what moved / what's blocked / next action**. Newest entries at the top.
Reference `QUESTIONS.md` IDs (`Q<n>`) and ladder IDs (`A1`, `B5a`, …) where
relevant.

---

## 2026-07-08 — M2 L0 solve machinery: Newton–Krylov + moments + species/frames/fields (structure, gated physics)

- **Contract**: `docs/src/islands/design/M2-launch-prompt.md` (interactive /goal
  run). Branch `feature/islands-m2`, **PR #324** (stacked on
  `feature/islands-m1`/PR #320; retargets to `feature/islands` when #320 merges).
  Full suite green locally.
- **Moved**: the full L0 solve *structure*, every physics coefficient a supplied
  `[VERIFY]`-gated parameter (physics-verifier: **PASS**):
  - `solvers/` — matrix-free Newton–Krylov (Krylov.jl GMRES on a preallocated
    ForwardDiff JVP; Eisenstat–Walker; line search; convergence on norm AND
    max-norm per `04 §5`), `YBlockJacobi` physics-block preconditioner with
    TSVD-regularized pencil solves (the `04 §3` y_c treatment), dense tiny-grid
    debug Jacobian, pseudo-arclength continuation with fold detection (toy fold
    found at step 6 of the test problem).
  - `species/` (D3 plumbing), `frames/` (conversion forms, NaN-gated
    `FrameConvention`), `fields/` (Q(Ω)/h(Ω) structure + NaN-gated
    `ElectronClosure`), `moments/` (J̄_∥, Δ projections with required gated
    prefactors, ⟨·⟩_Ω diagnostics), operators additions (mimetic
    `PitchAngleDiffusion`, `FarFieldConditions` — never bare Neumann,
    `weighted_moment!`).
  - **Structural gates green** (67 new tests in `runtests_islands_solve.jl`):
    A5 (residual exactly 0 at g≡0), solve-MMS at design order (3.98 observed,
    nx 17→33), A4 (conservation ≲1e-11, entropy sign exact), A3 parity, A7
    ⟨∂²h/∂x²⟩_Ω ≈ 1e-16, A8 σ_min monitor + singular detection. Preconditioner
    cuts a stiff collisional solve from 79.5 s/28 Newton to 0.6 s/7 (GMRES
    1700→200-class); all new kernels pass `--check-bounds=yes`.
  - `benchmarks/islands/` created: B2/B4/B5 scripts **skipped**, each naming its
    gating QUESTIONS IDs; `regression-harness` case `islands_l0_structural`
    (solve-MMS err 5.254e-2, 6 Newton/1210 GMRES, A7 8.0e-17, σ_min 0.1139).
  - Paper-I figure contract: `docs/src/islands/papers/paper-1/OUTLINE.md`
    (claims C1–C3 green as CI artifacts; C4–C8 gated on Q2–Q4).
  - **Rendered docs story** (user-flagged gap vs docs/07's M0–M1 intent): new
    `docs/src/islands/numerics.md` — the equations + figures of everything as
    implemented — plus the pinned figure script
    (`benchmarks/islands/figures/make_structural_figures.jl`, five structural
    figures committed as docs assets) and a full "Islands" site-nav section
    (overview, numerics chapter, Paper-I contract, design docs 00–08).
    Remaining docs/07 infra for later milestones: anchor-sync CI check,
    STATE.md dashboard.
- **Physics debugging note**: the first solve-MMS attempt failed to converge —
  the generic `Collisions` (a_y ∂²y) term has no y-BCs, so its BVP
  discretization is unstable under refinement; the *mimetic* divergence form
  (degenerate P → 0 endpoints, zero-flux built in) is the correct structure and
  the far-field x-BCs are what make the advective solve well-posed. Exactly the
  design's point (`01 §3`, `04 §1`).
- **Blocked**: the York gates (B5a/b/c, B2, B4) and Paper-I claims C4–C8 — all
  on the human clearance queue **Q2/Q3/Q4** (unchanged).
- **Next**: human clears Q2–Q4 → thin run fills the L0 coefficients from the
  D7 re-derivation and un-skips the B-ladder; independent M2+ work: kinetic-
  electron toggle (E4), io/ TOML section, trace-species linear pass.

## 2026-07-08 — M1 skeleton: phase-space grids + operator stack + MMS/AD harness

- **PR**: #320 (`feature/islands-m1` → `feature/islands`); full suite green.
- **Moved**: Landed the M1 core (design `03 §1–2`, `04`, ladder `A1/A2`). Three
  `src/Islands/` submodules, all structure-only (no `[VERIFY]` physics numbers):
  - `phasespace/PhaseSpace.jl` — the `(x, ξ, y, E, σ)` grids with layer-clustered
    maps: Fourier spectral `∂ξ`, Fornberg high-order FD `∂x`/`∂y` on `sinh`-stretched
    grids (window sized per-derivative so `D1`/`D2` are both 4th-order incl.
    boundaries), composite-Simpson quadrature weights, Gauss–Laguerre energy nodes.
  - `operators/Operators.jl` — `AbstractTerm` + `apply!` + `residual!`; the term
    structs of `03 §2` (`ParallelStreaming`, `MagneticDrift` with the
    `:original/:improved` toggle, `ExBDrift` as the `(x,ξ)` Poisson bracket,
    `Collisions`, `GradientDrive`, `PerpTransport`/`RadiationSink` L4 stubs,
    `Quasineutrality` field residual). Every physics coefficient is a **supplied
    data field** — no literal in `src/`. Allocation-free, AD-generic.
  - `verify/Verify.jl` — manufactured-solution + AD-vs-FD JVP harness.
  - Tests `test/runtests_islands_{grids,operators}.jl` (wired into `runtests.jl`):
    A1 per-operator MMS → 4th order for `∂x/∂y` terms, machine-precision for the
    `∂ξ` term; assembled kinetic residual → 4th order; A2 JVP-vs-FD agree to ~6e-9;
    **allocation regression = 0 bytes** for every `apply!` and `residual!`. All
    53 Islands tests green. Added `ForwardDiff` to `Project.toml` (design `04 §9`).
- **physics-verifier**: PASS — audited all six new/changed files, no
  `[VERIFY]`-policy violation; the flagged literature numbers (8.73/1.46 ρ_bi,
  k=−1.173, …) appear only in docstring prose, never assigned to a coefficient.
- **Blocked**: nothing. **Q1 RESOLVED**: julia is at
  `/mnt/homes_global/ncl2128/software/julia-1.11.7/bin/julia`; must be run with
  `env -u LD_LIBRARY_PATH` (OMFIT contamination). Used it to run the suite here.
- **Next**: M2 — wire moments (`Δ_cos`, `Δ_sin`), `frames/`, `species/`, and the
  Newton–Krylov solver toward the L0 single-species solve; every physics
  coefficient stays `[VERIFY]`-gated with a skipped benchmark until cleared.

## 2026-07-08 — Harden Stop hook against OMFIT LD_LIBRARY_PATH contamination

- **Moved**: Diagnosed why the Stop hook's package-load check fails on this box.
  A loaded OMFIT module (`module load omfit/unstable`) leaks
  `LD_LIBRARY_PATH=/mnt/codes/atom/mambaforge/envs/omfit/lib:` into the session;
  those conda libs shadow Julia's bundled artifacts, giving `undefined symbol`
  errors in CHOLMOD and the Plots/Cairo/GR native stack (and the ubiquitous
  `libtinfo.so.6` bash warning). Not a code issue — CI is green, and
  `env -u LD_LIBRARY_PATH julia … using GeneralizedPerturbedEquilibrium` loads
  clean (exit 0). Fixed `stop-check.sh` to run the build check with
  `env -u LD_LIBRARY_PATH` (no-op on a clean shell / CI). Repo deps were
  instantiated here; the shared depot (`/mnt/codes/ncl2128/.julia`) is populated.
- **Blocked**: nothing new. This is the concrete shape of **Q1** — the
  automation shell must invoke julia with a clean `LD_LIBRARY_PATH` (unload
  OMFIT, or unset the var) or the overnight loop's *actual* gpec runs fail the
  same way, not just the hook.
- **Next**: (human) launch the loop from a shell without the OMFIT module
  (`module unload omfit`); hook hardening is defense-in-depth on top of that.

## 2026-07-08 — Fix invalid deny rules in `.claude/settings.json`

- **Moved**: `/doctor` flagged two skipped permission-deny rules —
  `Bash(git push:* main)` / `Bash(git push:* develop)` — invalid because `:*`
  (prefix match) is only allowed at the end of a pattern. Rewrote them with a
  mid-pattern wildcard (`Bash(git push* main)` / `Bash(git push* develop)`) so
  they load and again deny pushes to `main`/`develop` for any remote/flags.
- **Blocked**: nothing.
- **Next**: unchanged — pending items are the Phase A bootstrap **Next** below.

## 2026-07-08 — Phase A bootstrap (supervised)

- **Moved**: Created the `Islands` submodule skeleton (`src/Islands/Islands.jl`,
  empty `module Islands`) and wired it into `src/GeneralizedPerturbedEquilibrium.jl`
  (`include` + `import . as` + `export`, last submodule slot before `Rerun.jl`).
  Stood up this `LOG.md` and `QUESTIONS.md`, the `.claude` unattended-run
  guardrails, the `physics-verifier` subagent, and the M1 launch prompt.
- **Landed CI-green** on `feature/islands` (PR #318): both `runtests` jobs pass
  (the wiring is valid — the package loads and the full suite passes) and the
  docs build passes. One fix was needed en route: the exported `Islands` module
  docstring required a manual page under `checkdocs=:exports`, so
  `docs/src/islands.md` (an `@autodocs` block) was added and wired into
  `docs/make.jl` (repo-root CLAUDE.md docs-coverage rule).
- **Blocked**: `julia` is not on the automation shell's PATH (no module, not in
  `$HOME`) → changes could not be run locally; CI is the only Julia validation
  here. See **Q1** — the overnight loop's scratch-clone environment must expose
  `julia` or it cannot run tests / meet M1's definition-of-done.
- **Next**: (human) resolve Q1 + one supervised `dontAsk` dry-run of the hooks,
  then launch the overnight loop on milestone **M1** (design `00 §M1`) —
  phase-space grids + operator-stack skeleton + MMS/AD harness (ladder A1, A2),
  no `[VERIFY]` physics coefficients.
