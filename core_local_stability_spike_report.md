# Core Local-Stability Spike Analysis

## Summary

The large local-stability spike in the core region of the DIII-D-like example is most consistent with a near-axis `q' = dq/dpsi_N` conditioning problem, not with the magnetic field magnitude going to zero.

The problematic region is roughly:

```text
psi_N ~ 1e-4 to 1e-2
```

In this range, `q'` becomes small and changes sign several times. Both local-stability routes contain terms that scale like `1/q'` or `1/q'^2`, so small spline/profile errors in `q'` are strongly amplified.

## Files and Code Paths

The old Mercier calculation was removed from `src/ForceFreeStates/Mercier.jl` on the `ballooning` branch. Its single-surface equivalent now lives in:

```text
src/ForceFreeStates/Bal.jl
```

Relevant functions:

- `resistive_interchange(flux_surface_index, plasma_eq)`
- `prepare_ballooning_coefficients(flux_surface_index, plasma_eq; ...)`
- `compute_ballooning_ode!`

The HDF5 outputs are written from `src/GeneralizedPerturbedEquilibrium.jl`:

```julia
out_h5["locstab/di"] = intr.locstab.y[:, 1] ./ locstab_xs
out_h5["locstab/dr"] = intr.locstab.y[:, 2] ./ locstab_xs
out_h5["locstab/ballooning_Delta_prime"] = ctrl.local_stability_flag ? intr.locstab.y[:, 4] : Float64[]
```

## What Was Compared

For `examples/DIIID-like_ideal_example`, I compared:

- `det(d0bar)` from `prepare_ballooning_coefficients()`
- Mercier surface-average `D_I` from `resistive_interchange().di`
- resistive interchange `D_R`
- ballooning `Delta'`
- input/profile quantities: `q`, `q'`, `p'`, alpha, and shear
- core geometry/field quantities: `B^2`, `|grad psi|^2` (`dpsisq`), and Jacobian

Generated diagnostics:

```text
/tmp/gpec_core_diagnostics.csv
/tmp/gpec_core_diagnostics.png
/tmp/gpec_di_compare.csv
/tmp/gpec_di_compare.png
```

## Key Observations

Core comparison table excerpt:

```text
psi_N       q'         det(d0bar) D_I   Mercier D_I     ballooning Delta'
1.00e-4    -3.970     +4.685           +2.010          -4.554
1.50e-4    -2.434     +6.620           +5.376          -3.021
2.26e-4    -0.426     +66.236          +96.846         -2.011
3.39e-4    +1.321     -1.983           -1.444          -1.351
5.10e-4    +1.694     -3.950           -5.218          -0.922
7.67e-4    +0.788     -32.357          -30.946         -0.660
1.15e-3    -0.449     -67.754          -66.434         -0.461
1.73e-3    -0.178     -151.517         -164.886        -0.296
3.91e-3    +0.146     -136.387         -134.295        -0.159
5.88e-3    +0.108     -149.781         -150.304        -0.115
1.33e-2    +0.156     -30.293          -30.532         -0.065
1.00e-1    +0.382     -1.049           -1.049          -0.024
```

The largest spikes line up with small `q'` values and sign changes, especially around:

```text
psi_N = 2.26e-4, 1.73e-3, 3.91e-3, 5.88e-3
```

The magnetic-field and geometry quantities do not show a comparable singular behavior:

```text
B^2      ~ 3.6 to 4.5      smooth, not close to zero
jac      ~ 12.65           nearly constant
dpsisq   grows smoothly from ~8e-4 near the axis
```

`dpsisq` is naturally small near the magnetic axis, but its behavior is smooth and monotonic. It does not match the sign-changing spike pattern. `B^2` is safely away from zero.

## Why `q'` Is the Main Driver

The Mercier surface-average route explicitly divides by `q'`.

In `resistive_interchange()`:

```julia
term = twopif * p1 * v1 / (q1 * chi1^3) * avg[2]
di = -0.25 + term * (1 - term) +
     p1 * (v1 / (q1 * chi1^2))^2 * avg[1] *
     (p1 * (avg[3] + (twopif / chi1)^2 * avg[4]) - v2 / v1)
h = twopif * p1 * v1 / (q1 * chi1^3) * (avg[2] - avg[1] / avg[5])
```

Here:

- `term ~ 1/q'`
- the second contribution to `D_I` contains `1/q'^2`
- `H` also scales like `1/q'`

For `psi_N = 2.26e-4`, the intermediate pieces were:

```text
q'     = -4.265e-1
term   = +4.798e3
second = +2.302e7
D_I    = +9.685e1
```

This means the final `D_I` comes from cancellation between very large intermediate terms. That is numerically ill-conditioned: small differences in `q'`, profile derivatives, or flux-surface averages can produce large changes in the final value.

The `det(d0bar)` route has the same core sensitivity through a different algebraic path:

```julia
nabla_beta_sq_b_sq_peculiar_2nd = term2_factor * (q_derivative^2)
m0_12 = jac_chiprime ./ nabla_beta_sq_b_sq_peculiar_2nd
```

So:

```text
m0_12 ~ 1 / q'^2
```

This feeds `n0_fs`, then `d0bar`, then `det(d0bar)`.

## Interpretation

The spike is not caused by `B^2` going to zero. It is also not primarily caused by a local zero in the 2-D magnetic field. The observed evidence points to a near-axis magnetic-shear problem:

```text
small / sign-changing q' near the magnetic axis
    -> 1/q' and 1/q'^2 amplification
    -> large cancellation in D_I formulas
    -> unstable local D_I and ballooning Delta' behavior
```

This is a known type of numerical hazard for formulas that are singular in the zero-shear limit. The magnetic axis is also a coordinate-singular region, so applying the same finite-radius local formula all the way down to `psi_N = 1e-4` is risky unless the zero-shear/axis limit is handled specially.

## Practical Consequences

For review discussion, the safest statement is:

```text
The restored resistive_interchange() path is the old Mercier surface-average formula.
The core discrepancy is not because that function differs from the old Mercier.jl route.
It comes from evaluating both the Mercier and det(d0bar) local-stability formulas near
the magnetic axis where q' is small/sign-changing. Both routes contain 1/q' or 1/q'^2
structures, so the core surfaces are ill-conditioned.
```

For production output, it may be better to either:

- skip local high-n/Mercier diagnostics below a small core cutoff,
- document that the near-axis local-stability values are unreliable,
- or implement a regularized/analytic near-axis limit for the zero-shear case.

The first option is the simplest and probably the least invasive for this PR.

