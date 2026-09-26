# Regression Harness

***This should be used at least once every single pull request before merging into develop. This test harness is what tracks values as they evolve across changes to the code, and must be both kept up to date and used consistently. Do not forget this and make sure to suggest any new regression cases or updates to existing ones as needed. Remind the user of its existence and report back the output regression report you get when modifying the code significantly. This is extremely important, do not forget this tidbit.***

Set up an alias for convenience (optional):
```bash
alias regress='julia --project=regression-harness regression-harness/regress.jl'
```

**List available cases:**
```bash
regress --list-cases
```
```
Available regression cases:
----------------------------------------------------------------
  diiid_n1                 DIII-D-like equilibrium, n=1, ideal + perturbed equilibrium
                           dir: examples/DIIID-like_ideal_example  (24 quantities)
  solovev_multi_n          Solovev analytical equilibrium, multi-n, ideal stability
                           dir: examples/Solovev_ideal_example_multi_n  (12 quantities)
  solovev_n1               Solovev analytical equilibrium, n=1, ideal stability
                           dir: examples/Solovev_ideal_example  (18 quantities)
```

**Compare two branches/commits:**
```bash
regress --cases diiid_n1 --refs develop,feature/kinetic-damping
```

All cases requested in a single invocation share one git worktree (and one `Pkg.instantiate`/precompile) per commit, so `--cases a,b,c` in one command is substantially faster than three separate runs.
```
================================================================
Case: diiid_n1 — DIII-D-like equilibrium, n=1, ideal + perturbed equilibrium
================================================================
[ Info: Cached: diiid_n1 @ 0a905a7d (2026-04-06T23:41:50+09:00)
[ Info: Cached: diiid_n1 @ 44b2494f (2026-04-08T18:30:46+09:00)

Regression Report: diiid_n1
==================================================================================================================
Ref 1: develop  @ 0a905a7d (2026-04-06)
Ref 2: feature/kinetic-damping  @ 44b2494f (2026-04-08)
------------------------------------------------------------------------------------------------------------------
Quantity                     develop                  feature/kinetic-damping  Diff                  Status
------------------------------------------------------------------------------------------------------------------
beta_n                       -1.376214e+00            -1.376214e+00            0.0e+00               OK
beta_t                       1.322850e-02             1.322850e-02             0.0e+00               OK
Chirikov parameter           [4 elements]             [4 elements]             0.0e+00               OK
delta prime                  [4 elements]             [4 elements]             0.0e+00               OK
plasma energy Re(ep[1])      -8.809610e-01            -8.809610e-01            0.0e+00               OK
total energy Im(et[1])       6.175834e-05             6.175834e-05             0.0e+00               OK
total energy Re(et[1])       1.199597e+00             1.199597e+00             0.0e+00               OK
vacuum energy Re(ev[1])      2.080558e+00             2.080558e+00             0.0e+00               OK
island half-widths           [4 elements]             [4 elements]             0.0e+00               OK
mpert                        34                       34                       0.0e+00               OK
# singular surfaces          4                        4                        0.0e+00               OK
npert                        1                        1                        0.0e+00               OK
ODE steps (saved)            740                      740                      0.0e+00               OK
ODE steps (total)            1348                     1348                     0.0e+00               OK
PE plasma energy             0.000000e+00             0.000000e+00             0.0e+00               OK
PE total energy              0.000000e+00             0.000000e+00             0.0e+00               OK
pressure profile (checksum)  657ad2329d7b...          657ad2329d7b...          identical             OK
q0                           1.209710e+00             1.209710e+00             0.0e+00               OK
q95                          4.505007e+00             4.505007e+00             0.0e+00               OK
q profile (checksum)         75912afcc351...          75912afcc351...          identical             OK
||resonant flux||            4.523707e+02             4.523707e+02             0.0e+00               OK
Runtime (s)                  50.9s                    52.0s                                          --
singular psi locations       [4 elements]             [4 elements]             0.0e+00               OK
singular q values            [4 elements]             [4 elements]             0.0e+00               OK
==================================================================================================================
Summary: 23 unchanged, 3 missing/N/A
```

**Compare your uncommitted working tree against develop:**
```bash
regress --cases solovev_n1 --refs develop,local
```

**Track a specific quantity across cached commits:**
```bash
regress --show et_real --case solovev_n1
```
```
History: et_real — solovev_n1
================================================================================
Commit      Date          Value                 Δ from prev           Status
--------------------------------------------------------------------------------
edff6e86    2026-04-02    -4.624928e-01         --                    --
0a905a7d    2026-04-06    -4.624928e-01         0.0e+00               OK
================================================================================
```

**Scan across a range of commits (git-bisect style):**
```bash
regress --cases solovev_n1 --ref-range develop~10..develop
```

**Other useful flags:**
- `--force` — re-run even if cached
- `--verbose` — print GPEC subprocess output
- `--no-instantiate` — skip `Pkg.instantiate()` (faster if deps are already resolved)
- `--no-pin-manifest` — let each ref resolve its own package set (see below)
- `--allow-env-mismatch` — reuse cached results produced in a different environment
- `--fail-on-change` — exit non-zero when any tracked quantity changed

GPEC subprocesses run with `-t auto` (all cores) so GPEC's threaded kernels are active; set `GPEC_REGRESS_THREADS=1` to force single-threaded runs. Tracked quantities are thread-count independent, and the count each run actually used is recorded in its environment fingerprint (shown in the report's `env:` lines). Thread count is deliberately not part of the cache key, so `Runtime (s)` rows cached from single-threaded runs are not comparable to threaded ones — re-baseline with `--force` if runtime tracking matters.

On a machine you share with other people or with your own parallel sessions, `-t auto` is antisocial: set `GPEC_REGRESS_THREADS` to a bounded count, and `JULIA_NUM_PRECOMPILE_TASKS` alongside it, since `Pkg.instantiate()` otherwise fans out to `Sys.CPU_THREADS + 1` precompile workers at the start of every ref. If the machine has a batch scheduler, submit the run through it and pin both variables to the allocation.

## Three things named "regression"

They share the word and nothing else, which is a reliable source of confusion:

- **This harness** (`regression-harness/`) tracks numerical quantities across commits. Its cases
  live in `regression-harness/cases/*.toml` and point at decks under `examples/`. This is what the
  pull-request checklist means by "the regression harness".
- **`test/test_data/regression_*/`** are *input decks for the unit tests* — a `gpec.toml`, and a
  `kinetic.dat` for the kinetic ones. They are not harness cases and the harness never reads them.
- **`test/runtests_*.jl`** are the unit and golden-value tests. Their expected numbers live in the
  test files themselves, so moving one is a hand edit that has to be justified in review.

## Run isolation

Every git ref in a comparison is checked out into its own detached worktree, so a harness run is
already insulated from whatever you do to the working tree while it runs. The `local` ref is the
exception: it runs GPEC in the live checkout.

That matters more than it first appears, because each case runs as a **fresh `julia` subprocess**
that loads `src/` from disk when it starts. A multi-case `--refs develop,local` run therefore reads
the source once per case, and an edit landing between two cases yields a single report whose rows
were produced by different code — with nothing in the output to say so.

For anything you intend to cite — a pull-request report, a bisect, a number you will act on —
commit the work to the feature branch and compare branch refs:

```bash
regress --cases diiid_n1 --refs develop,my-feature-branch
```

`local` stays the right tool for a quick spot check on uncommitted work, provided you leave the tree
alone until it finishes.

Two related hazards outside the harness, for the same reason:

- **`test/runtests.jl` loads GPEC once, in-process, then `include`s its test files in sequence.**
  Editing `src/` mid-run changes nothing the run sees, so it reports green for code no longer on
  disk; editing a `test/runtests_*.jl` file *is* picked up when the run reaches it, so the report
  mixes old and new. Neither failure is loud.
- **Concurrent runs in one checkout collide over output paths.** `runtests_fullruns.jl` writes and
  then deletes `test/test_data/regression_*/gpec.h5`; direct example runs and the harness `local`
  ref write `examples/<case>/gpec.h5`. Two runs in the same clone delete each other's output. Give
  the second one its own worktree (`git worktree add --detach <path> HEAD`, copying `Manifest.toml`
  across so both resolve the same package set).

## Making source code the only variable

`Manifest.toml` is untracked, so a worktree checked out at an old commit used to resolve whatever
package versions were newest at run time. Machine-epsilon differences in library math then get
amplified by the adaptive ODE step controller and by ill-conditioned near-resonant diagnostics
into double-digit-percent "regressions" that no source change caused.

Two mechanisms prevent that:

**The working tree's Manifest is pinned into every worktree** before `Pkg.instantiate()`, so all
refs in a comparison run against one package set. `--no-pin-manifest` opts out (and says so
loudly). If a commit declares a direct dependency the pinned Manifest lacks, `Pkg.instantiate()`
refuses to run: that ref is recorded as a failed run whose error suggests `--no-pin-manifest` to
let it resolve its own package set.

**Every run records the environment that produced it** — Julia version, host, resolved Manifest
hash, Julia and BLAS thread counts. The cache still holds a single result per
`(commit, case)`, so a re-run replaces the stored one rather than keeping a result per
environment; what the fingerprint adds is that a cached result whose environment differs from
the current one is re-run instead of silently reused. `--allow-env-mismatch` skips that check
and reuses whatever is cached, whatever produced it. Every report prints the environment of
each ref:

```
Ref 1: develop  @ a0cad260 (2026-08-12)
       env: julia 1.11.6, arm64-apple-darwin24.0.0, manifest 7e5c34ad (pinned), 1 thread/8 BLAS
```

When two compared runs did not share an environment, the report says so before the table rather
than leaving you to infer it from the numbers.

Results cached before environment fingerprinting existed carry no environment and are therefore
re-run once — those are exactly the entries whose provenance cannot be established.

If the two refs in a comparison ran under different thread counts, the report flags it.

## Exit status

- `0` — every run completed (and, with `--fail-on-change`, nothing changed)
- `1` — a run failed, or a quantity changed under `--fail-on-change`
