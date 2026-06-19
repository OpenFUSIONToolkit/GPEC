# Restored Mercier.jl Equivalence and Core Instability Notes

## Purpose

This note collects the PR #234 evidence needed to answer the review question about
the restored `D_R` / Mercier path:

1. The old `Mercier.jl` calculation was restored locally under `analysis/` and run on the DIII-D-like example.
2. The restored old calculation was compared against the current `Bal.jl` `resistive_interchange()` path.
3. The restored old Mercier calculation was checked in the core region where the local-stability values spike.

## Generated Files

```text
analysis/pr234_mercier_core/restored_Mercier.jl
analysis/pr234_mercier_core/generate_mercier_comparison.jl
analysis/pr234_mercier_core/plot_mercier_comparison.py
analysis/pr234_mercier_core/mercier_comparison.csv
analysis/pr234_mercier_core/mercier_core_1e-4_to_1e-1.png
analysis/pr234_mercier_core/mercier_outer_1e-1_to_0p99.png
```

## Result 1: Restored Mercier.jl and Bal.jl `resistive_interchange()` Are the Same Calculation

The restored old calculation is in:

```text
analysis/pr234_mercier_core/restored_Mercier.jl
```

It is a diagnostic copy of the old `src/ForceFreeStates/Mercier.jl` calculation from `origin/develop`.

The current branch computes the same surface-average Mercier quantities inside:

```text
src/ForceFreeStates/Bal.jl
```

The relevant current function is:

```julia
resistive_interchange(flux_surface_index, plasma_eq)
```

Both paths build the same flux-surface averages:

```julia
ff_fs[itheta, 1] = bsq / dpsisq
ff_fs[itheta, 2] = 1.0 / dpsisq
ff_fs[itheta, 3] = 1.0 / bsq
ff_fs[itheta, 4] = 1.0 / (bsq * dpsisq)
ff_fs[itheta, 5] = bsq
ff_fs[itheta, :] .*= jac / v1
```

Both then use the same Mercier and resistive-interchange formulas:

```julia
term = twopif * p1 * v1 / (q1 * chi1^3) * avg[2]
di = -0.25 + term * (1 - term) +
     p1 * (v1 / (q1 * chi1^2))^2 * avg[1] *
     (p1 * (avg[3] + (twopif / chi1)^2 * avg[4]) - v2 / v1)
h = twopif * p1 * v1 / (q1 * chi1^3) * (avg[2] - avg[1] / avg[5])
dr = di + (h - 0.5)^2
```

The difference is only structural:

- old `Mercier.jl`: looped over all `psi` surfaces and wrote `locstab_fs` directly
- current `Bal.jl`: evaluates one surface and returns `(di, dr, h)`

The numerical comparison confirms this. The maximum relative difference between restored `Mercier.jl` `D_I` and current `Bal.jl` `resistive_interchange().di` is machine precision:

```text
core  psi_N = 1e-4 ... 1e-1:
  max relative difference = 1.18e-16

outer psi_N = 1e-1 ... 0.99:
  max relative difference = 1.97e-16
```

## Result 2: Restored Mercier.jl Is Also Unstable in the Core

The core plot:

```text
analysis/pr234_mercier_core/mercier_core_1e-4_to_1e-1.png
```

shows that the restored old `Mercier.jl` `D_I` has the same core spike as the current `Bal.jl` `resistive_interchange()` calculation.

Representative values:

```text
psi_N       q'         restored Mercier D_I   Bal.jl resistive D_I   det(d0bar) D_I
1.00e-4    -3.970     +2.010                +2.010                +4.685
1.50e-4    -2.434     +5.376                +5.376                +6.620
2.26e-4    -0.426     +96.846               +96.846               +66.236
1.73e-3    -0.178     -164.886              -164.886              -151.517
3.91e-3    +0.146     -134.295              -134.295              -136.387
5.88e-3    +0.108     -150.304              -150.304              -149.781
1.33e-2    +0.156     -30.532               -30.532               -30.293
1.00e-1    +0.382     -1.049                -1.049                -1.049
```

This means the old Mercier route did not avoid the core sensitivity. It already had it.

## Result 3: Outside the Ill-Conditioned Core the Curves Move Together

The outer-region plot:

```text
analysis/pr234_mercier_core/mercier_outer_1e-1_to_0p99.png
```

shows that restored `Mercier.jl`, current `Bal.jl` `resistive_interchange()`, and `det(d0bar)` move together over:

```text
psi_N = 1e-1 ... 0.99
```

In this range:

```text
restored Mercier vs current resistive_interchange:
  max relative difference = 1.97e-16

det(d0bar) vs restored Mercier:
  max relative difference = 1.53e-3
```

So the non-core region supports the intended equivalence between the surface-average Mercier route and the `det(d0bar)` route, while the core exposes the near-axis conditioning problem.

## Why the Core Is Ill-Conditioned

Near the magnetic axis, `q' = dq/dpsi_N` is small and changes sign. The formulas contain explicit `1/q'` and `1/q'^2` dependence.

In the restored/current Mercier formula:

```julia
term = twopif * p1 * v1 / (q1 * chi1^3) * avg[2]
di = -0.25 + term * (1 - term) +
     p1 * (v1 / (q1 * chi1^2))^2 * avg[1] *
     (...)
```

Therefore:

```text
term ~ 1/q'
second contribution ~ 1/q'^2
```

In the `det(d0bar)` route:

```julia
nabla_beta_sq_b_sq_peculiar_2nd = term2_factor * (q_derivative^2)
m0_12 = jac_chiprime ./ nabla_beta_sq_b_sq_peculiar_2nd
```

Therefore:

```text
m0_12 ~ 1/q'^2
```

This is why both calculations become sensitive in the same core region.

The field magnitude is not the trigger in this case. In the same core surfaces:

```text
B^2      remains smooth and safely away from zero
jac      is nearly constant
dpsisq   is small near the axis, as expected, but changes smoothly
q'       is small and sign-changing exactly where D_I spikes
```

## Suggested Reviewer Response

```text
I restored the old Mercier.jl calculation under analysis/pr234_mercier_core and compared it directly against the current Bal.jl resistive_interchange() path. They are the same surface-average Mercier calculation: restored Mercier.jl and current resistive_interchange() agree to machine precision.

The core spikes are not introduced by the Bal.jl rewrite. The restored old Mercier.jl calculation shows the same core behavior. The cause is the near-axis q' / magnetic-shear conditioning: q' is small and changes sign in the psi_N ~ 1e-4 ... 1e-2 region, while both the Mercier formula and the det(d0bar) construction contain 1/q' or 1/q'^2 structures. Outside that ill-conditioned core region, psi_N >= 0.1, the restored Mercier, current resistive_interchange, and det(d0bar) curves move together closely.

So I think the restored D_R path is consistent with the old Mercier route. Separately, the near-axis local-stability diagnostics should probably be documented as unreliable below a small core cutoff, or handled later with a regularized axis/zero-shear limit.
```

