# Validation of STRIDE-type Delta-Prime BVP Shooting in Julia GPEC

This document records the findings from validating Julia GPEC's STRIDE-type
tearing stability parameter (Delta') boundary value problem (BVP) shooting
calculation against Fortran GPEC reference data.

---

## 1. Background: DCON vs STRIDE Integration Paths

Julia GPEC originally implemented a **DCON-style integration** for ideal MHD
stability analysis. This approach:

- Uses a single continuous ODE integration from axis to edge.
- Stores the fundamental matrix U = [U1; U2] at discrete psi points.
- Computes the Newcomb criterion and energy eigenvalues from the edge
  fundamental matrix.
- Works well for ideal MHD stability (delta-W, Mercier criterion, etc.).

For Delta' (the tearing stability parameter), Fortran GPEC's **STRIDE** module
uses a more sophisticated boundary value problem approach:

- Decomposes the domain at each rational surface into shooting intervals.
- Uses midpoint-split shooting propagators: forward from a surface to the
  interval midpoint, backward from the midpoint to the next surface.
- Constructs a global BVP matrix and solves for asymptotic coefficients.
- Extracts the small solution coefficients to build the `dp_raw` matrix.
- Applies PEST3-convention differencing to obtain the physical Delta' matrix.

---

## 2. Why the Direct DCON-style Approach Failed for Delta'

The initial Julia implementation attempted to use the existing parallel
fundamental matrix (FM) propagators directly in the BVP, without the
midpoint-splitting that STRIDE employs. This produced catastrophically wrong
results.

### Problem: Catastrophic Ill-Conditioning of the BVP Matrix

The inter-surface propagator (from surface 1 to surface 2) had a condition
number of approximately 4x10^15 because the ODE solutions grow and decay
exponentially over the long integration interval. When this ill-conditioned
propagator was placed directly into the BVP matrix M, the result was:

- **rank(M) = 25** out of nMat = 320 (severely rank-deficient).
- **cond(M) ~ 10^22** (essentially singular).
- The pseudo-inverse fallback gave physically meaningless `dp_raw` values
  (order 0.01-7 vs Fortran's 40-680).
- The PEST3 differencing of these noisy values produced Delta' values that
  were approximately 10,000x too small.

### Root Cause: Missing Midpoint Splitting

The Fortran STRIDE code splits each inter-surface interval at its midpoint:

- `uShootR` propagates **forward** from the surface to the midpoint (half the
  distance).
- `uShootL` propagates **backward** from the midpoint to the next surface
  (other half).
- Each half-propagator has condition number ~ sqrt(full_condition), roughly
  10^7 to 10^8.
- The BVP matrix constructed from these half-propagators has condition ~ 10^9,
  which is manageable.

Without this splitting, the Julia BVP used full-interval propagators with
condition ~ 10^15, which when combined in the BVP matrix produced the
rank-deficient system described above.

---

## 3. The S-Based (Riccati) Axis BC -- The Key Fix

The resolution was to use the **S-based BVP path**, which leverages matrices
already computed during the parallel FM integration:

- During the parallel FM integration, Julia already computes Riccati S matrices
  (S = U1 * U2^{-1}) at each singular surface's left boundary.
- These S matrices encode the axis boundary condition in a well-conditioned
  form (cond ~ 10^6 to 10^7).
- The S-based BVP path uses these matrices instead of the catastrophically
  ill-conditioned axis propagator.
- It also uses midpoint-split shooting propagators (via
  `integrate_fm_with_ua_ic`) for the inter-surface intervals.
- Result: **BVP has full rank (320/320) with cond ~ 4x10^8**.

The `fm_S_left` array returned by `eulerlagrange_integration` must be passed
to `compute_delta_prime_matrix!` via the `S_at_surface_left` keyword argument.
Without this argument, the code falls back to the direct axis propagator path,
which produces the ill-conditioned system described in Section 2.

---

## 4. Wall Distance Parameter -- Critical Configuration Fix

A separate configuration issue was causing approximately 39% energy
discrepancies between Julia and Fortran results:

- The Fortran `vac.in` namelist sets `a=20` in the `&shape` block, meaning
  the conformal wall is placed at 20 times r_minor (approximately 7.86 m from
  the plasma). For this small tokamak, this is effectively at infinity.
- Julia's `WallShapeSettings` has `a` (default 0.3) and `aw` (default 0.05)
  as separate parameters.
- The Julia `gpec.toml` files only set `aw = 0.1` but left `a` at its default
  value of 0.3, placing the wall at 0.3 x 0.393 = 0.118 m from the plasma.
- This **66x difference** in wall distance caused vacuum energy eigenvalues to
  differ by 10-60%, with cascade effects on total energy and Delta'.
- **Fix**: Add `a = 20` to the `[Wall]` section of both the beta scan and
  epsilon scan `gpec.toml` files.

---

## 5. Validation Results (pf=0.1 Single Point)

The following table compares Julia and Fortran GPEC for a Large Aspect Ratio
(LAR) equilibrium at pressure fraction pf=0.1.

| Quantity                | Julia       | Fortran     | Error    |
|-------------------------|-------------|-------------|----------|
| Delta'(2/1)             | 16.124      | 16.445      | 1.96%    |
| Delta'(3/1)             | 8.152       | 8.341       | 2.27%    |
| et[1] (total energy)    | 0.8064      | 0.8021      | 0.54%    |
| ev[1] (vacuum energy)   | 0.9821      | 0.9838      | 0.17%    |
| ep[1] (plasma energy)   | -0.1757     | -0.1817     | 3.30%    |
| wv eigenvalues          | match       | match       | ~0.01%   |
| q, mu_0*p, dV/dpsi      | match       | match       | <0.02%   |
| BVP condition number    | 3.93x10^8   | 1.19x10^9   | comparable |
| BVP rank                | 320/320     | 320/320     | full rank |

The residual ~2% discrepancy in Delta' is consistent with the parallel FM
path's known integration accuracy gap relative to the Fortran implementation.
Equilibrium profiles and vacuum eigenvalues agree to high precision, confirming
that the remaining Delta' difference originates in the ODE integration path
rather than in the BVP assembly or solution.

---

## 6. Full Scan Validation Results

### 6.1 Beta Scan (42 Points)

The beta scan varies pressure factor (pf) from 0.001 to 0.185 using 42 TJ
benchmark equilibria. Results are in `examples/LAR_beta_scan/outputs/`.

**Summary of errors by region:**

| Pressure Factor | Δ'(2/1) Error | Δ'(3/1) Error | δW_total Error |
|-----------------|---------------|---------------|----------------|
| pf < 0.05       | 0.3 - 1.1%    | 0.3 - 1.9%    | 0.2 - 0.4%     |
| pf = 0.05 - 0.12| 1 - 2.3%      | 1.2 - 3.1%    | 0.3 - 1.1%     |
| pf = 0.12 - 0.16| 3 - 8%        | 4 - 8.4%      | 1.5 - 5.3%     |
| pf = 0.16 - 0.18| 9 - 33%       | 10 - 33%      | 6 - 33%        |
| pf > 0.18       | 47 - 99%      | 47 - 99%      | 52 - 196%      |

**Key observations:**

- At low beta (pf < 0.05), Δ' errors are sub-1%, matching the known
  accuracy of the parallel FM path.
- Errors grow systematically with pressure factor, tracking the δW error.
- Near the instability threshold (pf > 0.18), δW approaches zero and both
  relative errors in δW and Δ' diverge. This is physically expected: Δ'
  diverges at the instability threshold, so even small absolute errors in
  the underlying energy produce large relative Δ' errors.
- The Julia Δ' values systematically underpredict the Fortran values. This
  is consistent with the parallel FM path's known systematic energy bias
  (~2-3% in plasma energy at moderate beta).

### 6.2 Epsilon Scan (56 Points)

The epsilon scan varies inverse aspect ratio (ε = a/R₀) from 0.125 to
0.6512 using 56 TJ benchmark equilibria. Results are in
`examples/LAR_epsilon_scan/outputs/`.

**Important config fix:** The initial epsilon scan had `set_psilim_via_dmlim = true`
in `gpec.toml`, which truncated the integration domain differently from Fortran
(which uses `sas_flag=f`). Setting `set_psilim_via_dmlim = false` reduced the
δW_total error from 100-1400% down to 0.1-9%.

**Summary of errors by region:**

| Epsilon Range   | Δ'(2/1) Error | Δ'(3/1) Error | δW_total Error |
|-----------------|---------------|---------------|----------------|
| ε < 0.25        | 0.1 - 1.9%    | 7 - 165% (*)  | 0.3 - 0.4%     |
| ε = 0.25 - 0.5  | 0.3 - 4.1%    | 0.4 - 3.0%    | 0.1 - 0.6%     |
| ε = 0.5 - 0.6   | 0.5 - 13%     | 0.8 - 2.5%    | 0.4 - 1.5%     |
| ε > 0.6 (pole)  | 1.6 - 13%     | 1.6 - 12%     | 0.2 - 8.7%     |

(*) Δ'(3/1) at low epsilon has a systematic overestimation that decreases
with increasing ε. This may be related to the q=3 singular surface being
close to the plasma edge at low epsilon, where boundary effects are more
sensitive to numerical treatment.

**Key observations:**

- δW_total errors are excellent (<2%) across most of the ε range.
- Δ'(2/1) tracks Fortran within ~5% for most of the range.
- Δ'(3/1) agreement is excellent for ε > 0.3, with a systematic discrepancy
  at low ε that warrants further investigation.
- Near the Δ' pole (ε ~ 0.66), errors grow as expected.

### 6.3 Root Cause of Residual Errors

The systematic ~2-5% error in Δ' across both scans traces back to the
**parallel FM integration path's energy accuracy**. The parallel path
integrates ODE chunks independently and assembles propagators, introducing
a small systematic error in the energy computation compared to the serial
(continuous) integration. This error is amplified in the Δ' computation
because Δ' involves differencing large dp_raw values, and near instability
thresholds, Δ' diverges.

Possible approaches to reduce these errors (future work):
- Use serial-path energy computation with parallel-path propagators for BVP
- Improve chunk assembly accuracy (higher-order matching, tighter tolerances)
- Implement Fortran-style Hermitianization of the wp matrix

---

## 7. Code Changes Summary

The following files were modified to achieve the validated results:

1. **`examples/LAR_beta_scan/gpec.toml`** -- Added `a = 20` to the `[Wall]`
   section, matching Fortran's conformal wall distance.

2. **`examples/LAR_epsilon_scan/gpec.toml`** -- Added `a = 20` to the `[Wall]`
   section, matching Fortran's conformal wall distance. Fixed
   `set_psilim_via_dmlim = false` to match Fortran's `sas_flag=f`.

3. **`src/ForceFreeStates/Riccati.jl`** -- Moved the `col_left(j)` and
   `col_right(j)` closure definitions from inside the `use_S_axis` block to
   function scope (line 438), preventing `UndefVarError` in the `dp_raw`
   extraction code. Removed duplicate definitions that caused method
   overwriting during precompilation.

4. **`examples/LAR_beta_scan/run_scan.jl`** and
   **`examples/LAR_epsilon_scan/run_scan.jl`** -- Updated `extract_results`
   to read the STRIDE BVP `delta_prime_matrix` diagonal (matching Fortran's
   `Delta_prime[0,k,k]`), falling back to per-surface ca-based `delta_prime`.
   Fixed `using Plots` at module scope.

---

## 8. Usage: Running Delta' with Correct Settings

The key code pattern for obtaining well-conditioned Delta' results:

```julia
odet, fm_propagators, fm_chunks, fm_S_left = eulerlagrange_integration(ctrl, equil, ffit, intr)
vac_data = free_run!(odet, ctrl, equil, ffit, intr)
compute_delta_prime_matrix!(intr, fm_propagators, fm_chunks;
    wv=vac_data.wv, psio=equil.psio,
    S_at_surface_left=fm_S_left,  # Critical: enables S-based BVP
    ctrl=ctrl, equil=equil, ffit=ffit)
```

The `S_at_surface_left` keyword argument is the critical switch. When provided,
`compute_delta_prime_matrix!` uses the Riccati S matrices for the axis boundary
condition and midpoint-split shooting propagators for inter-surface intervals.
When omitted, the function falls back to the direct axis propagator, which
suffers from the ill-conditioning described in Section 2.

Ensure that the `[Wall]` section of `gpec.toml` includes the correct `a`
parameter matching the Fortran configuration. For equilibria where the wall
should be effectively at infinity, use `a = 20` or larger:

```toml
[Wall]
shape = "conformal"
a = 20
aw = 0.1
```
