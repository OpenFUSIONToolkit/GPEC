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

## Making source code the only variable

`Manifest.toml` is untracked, so a worktree checked out at an old commit used to resolve whatever
package versions were newest at run time. Machine-epsilon differences in library math then get
amplified by the adaptive ODE step controller and by ill-conditioned near-resonant diagnostics
into double-digit-percent "regressions" that no source change caused.

Two mechanisms prevent that:

**The working tree's Manifest is pinned into every worktree** before `Pkg.instantiate()`, so all
refs in a comparison run against one package set. `--no-pin-manifest` opts out (and says so
loudly). If a pinned package set turns out to be incompatible with an old commit's
`Project.toml`, instantiate re-resolves it and the harness warns that the pin did not hold.

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

Thread counts are recorded but **not** forced: the harness does not silently change how your runs
execute. If the two refs in a comparison ran under different thread counts, the report flags it.

## Exit status

- `0` — every run completed (and, with `--fail-on-change`, nothing changed)
- `1` — a run failed, or a quantity changed under `--fail-on-change`
