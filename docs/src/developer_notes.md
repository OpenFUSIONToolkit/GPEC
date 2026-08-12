# Developer Notes

## Git Workflow

This project uses [GitFlow](http://nvie.com/posts/a-successful-git-branching-model):

- Two permanent branches: `main` and `develop`
- `main` is updated only at release-ready stages
- Feature branches should come off `develop` and merge back with `--no-ff`

## Commit Message Format

```
CODE - TAG - Detailed message
```

Where CODE is the module name (EQUIL, ForceFreeStates, VAC, PERTURBED EQUILIBRIUM, etc.) and TAG describes the type of change (WIP, MINOR, IMPROVEMENT, BUG FIX, NEW FEATURE, REFACTOR, CLEANUP, etc.). This format is used for compiling release notes — tags should be human-readable but are not enforced to a fixed set.

## Documentation standard

Module reference pages follow a shared *Journal of Computational Physics*-style
structure (governing equations → numerical method → validation figures → API),
and every documentation figure is committed together with the script that made
it and a provenance stamp. The full policy — page template, figure
organization under `docs/src/figures/<module>/`, provenance, and the
regenerate-only-when-`depends`-change rule — is in
[`docs/DOC_STANDARD.md`](https://github.com/OpenFUSIONToolkit/GPEC/blob/develop/docs/DOC_STANDARD.md).
The `Ballooning and Mercier Local Stability` and `Inner Layer Module` pages are
the reference exemplars.

## Regression Testing

The regression harness **must be run on every pull request before merging into `develop`**. It is the project's primary safeguard for tracking how numerical results evolve across changes, so it is only useful if every PR exercises it. When you open a PR, paste the regression report into the PR thread so reviewers can see what moved (and what did not). If your change touches a quantity that is not yet tracked, add a new regression case — or extend an existing one — in the same PR.

### Open problem: pinning grid-sensitive Δ′ robustly

The ideal-MHD Δ′ values pinned in `test/runtests_parallel_integration.jl`
(`delta_prime_matrix` diagonal) and tracked by the harness are **single-point
snapshots**: one value at one `psi_accuracy` on one grid. They exist to catch
unintended changes, and are explicitly *not* converged Δ′. The extraction is
intrinsically grid-sensitive — a `psi_accuracy` scan from 2e-3 to 2.5e-4 swings
the DIII-D-like `dpm[1,1]` by roughly 50% (about 6.2 to 9.9), and switching the
grid generator moves it again. Any such pin therefore encodes an arbitrary point
on a varying curve, and re-pinning is required whenever the grid changes, which
weakens it as a regression signal.

A more defensible criterion is to pin the **plateau**: scan Δ′ across
`psi_accuracy` and the integration-truncation controls, and take the value where
the result is stationary — the mode of the resulting distribution rather than any
single sample. Where a plateau exists it is a property of the physics rather than
of the discretization, so it would survive grid changes and would be a genuine
convergence statement.

A `psi_accuracy` scan on the DIII-D-like SLAYER deck (2e-3 down to 3.125e-5, a
64x range) shows what a plateau search is up against:

| psi_accuracy | knots | implied | dpm[1,1] | dpm[2,2] | dpm[3,3] | gamma 2/1 (Hz) |
|---|---|---|---|---|---|---|
| 2.0e-3   | 172 | –    | 6.850 | -5.783 | -16.308 | 165.3 |
| 1.0e-3   | 205 | –    | 6.387 | -5.873 | -16.867 | 154.3 |
| 5.0e-4   | 252 | –    | 8.003 | -6.105 | -16.413 | 192.9 |
| 2.5e-4   | 307 | 522  | 8.567 | -6.235 | -16.529 | 206.2 |
| 1.25e-4  | 372 | 666  | 8.146 | -6.285 | -16.649 | 196.2 |
| 6.25e-5  | 457 | 916  | 8.911 | -6.477 | -16.510 | 214.4 |
| 3.125e-5 | 559 | 1226 | 9.514 | -6.260 | -16.485 | 228.7 |

Three things follow. First, convergence is per-surface: the outer surface
`dpm[3,3]` (q=4) *does* plateau under the auto grid — the last four points agree
to about 1% — while `dpm[2,2]` is marginal and `dpm[1,1]` (q=2) never settles,
still moving ~7% between the two tightest grids. A plateau detector must
therefore report per-surface rather than pass/fail for the whole diagonal.
(As the `ldp` scan below shows, q=2 is not inherently unconvergeable — it is the
auto grid that prevents it from settling.)

Second, the growth rate is linear in Δ′: `gamma/dpm[1,1]` is 24.1 to within 0.5%
across the whole scan, so a converged Δ′ is both necessary and sufficient for a
converged SLAYER growth rate on this surface.

Third — and this is the blocker — the `implied` column (what
`implied_knot_count` requests after the refined solve) grows monotonically
relative to the grid actually used, from 1.7x to 2.2x. The two-pass scheme stops
after pass 2, so tightening `psi_accuracy` moves it *further* from
self-consistency rather than closer, and the warning's advice to "consider
tightening psi_accuracy" is counterproductive in this regime.

The same deck on a deterministic `ldp` grid, which skips the measure-and-re-form
step entirely, converges:

| mpsi | dpm[1,1] | dpm[2,2] | dpm[3,3] | gamma 2/1 (Hz) |
|---|---|---|---|---|
| 128  | 5.208 | -8.666 | -19.157 | 126.2 |
| 256  | 7.097 | -5.187 | -15.729 | 171.2 |
| 512  | 8.489 | -6.217 | -16.517 | 204.4 |
| 1024 | 8.932 | -6.298 | -16.529 | 214.9 |
| 2048 | 8.914 | -6.385 | -16.468 | 214.5 |

Successive `dpm[1,1]` increments are +1.889, +1.392, +0.443, -0.018: the last
doubling moves both Δ′ and the growth rate by 0.2%. So the q=2 Δ′ is **not**
intrinsically ill-conditioned — it converges to ≈8.91, with gamma ≈214.5 Hz, and
`dpm[3,3]` converges to ≈-16.5 on *both* grid families, which is what a physical
value should do. The non-convergence under the auto grid is an artifact of the
generator, not of the Δ′ extraction.

Two consequences. A plateau criterion is implementable today against a fixed
`ldp` grid, without waiting on the auto-grid work. And the auto grid's answers
are biased in both directions relative to the converged value: at its default
`psi_accuracy` it gave 6.39 (28% low), at its tightest 9.51 (7% high). Anything
pinned on the auto grid should be read with that in mind.

This is not implemented. Doing it properly needs:

  - either a fixed `ldp` grid (which already converges, see above) or knot
    refinement iterated to a fixed point (repeat the measure-and-re-form
    step until `implied_knot_count` stops exceeding the grid in use), since
    without it the scan target keeps moving;
  - a scan driver over `psi_accuracy` × truncation (`psihigh`, `dmlim`, `qhigh`)
    that records the Δ′ diagonal per configuration;
  - a plateau/mode detector with an explicit stationarity tolerance, plus a
    defined failure mode for surfaces where no plateau exists (the
    near-separatrix surfaces are expected to fall in this class);
  - a harness case that pins plateau values and their scan width, replacing the
    single-point pins.

Until then, treat the pinned diagonal as an order-of-magnitude and sign
diagnostic only, and expect to re-pin it whenever the equilibrium grid changes.

### Regression Harness: Quick usage guide

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

The full flag list, the environment-pinning behaviour that keeps source code the only variable in
a comparison, and the exit-status contract are documented in
[`docs/development/regression-harness.md`](../../development/regression-harness.md).
