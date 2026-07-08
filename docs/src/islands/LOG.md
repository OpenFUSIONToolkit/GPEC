# Islands — session LOG

Cross-session memory spine (design doc `06 §2.5`). Read this at session start
together with `QUESTIONS.md`. Append a short entry before every session end:
**what moved / what's blocked / next action**. Newest entries at the top.
Reference `QUESTIONS.md` IDs (`Q<n>`) and ladder IDs (`A1`, `B5a`, …) where
relevant.

---

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
