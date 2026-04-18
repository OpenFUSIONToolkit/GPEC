# STRIDE Matching Status: Julia GPEC vs Fortran STRIDE

**Last updated**: 2026-04-12

## Executive Summary

Julia GPEC's Re(Δ') agrees with Fortran STRIDE to **1–5% at well-resolved inner surfaces** across both LAR circular and DIII-D shaped equilibria. Three identified root causes explain all remaining discrepancies. All 98 equilibria in the LAR parameter scans now run successfully (0 failures).

## Test Suites

### LAR Circular Equilibria

| Scan | Points | dp21 median | dp31 median | dW_t median | Failures |
|------|--------|-------------|-------------|-------------|----------|
| Epsilon (ε = 0.125–0.66) | 56/56 | 3.3% | 2.5% | 0.9% | 0 |
| Beta (pf = 0.001–0.185) | 42/42 | 8.4%* | 9.8%* | 5.7%* | 0 |
| Beta, pf ≤ 0.15 only | 17/17 | 1.1% | — | 0.4% | 0 |

*Beta scan median dominated by pole proximity (cause C below).

**Scripts**: `examples/LAR_epsilon_scan/run_scan.jl`, `examples/LAR_beta_scan/run_scan.jl`
**Figures**: `examples/scan_comparison_plots/`

### DIII-D Shaped Equilibria

| Shot | Description | n=1 inner dp err | n=2 inner dp err | Notes |
|------|-------------|-------------------|-------------------|-------|
| 147131 | KEFIT H-mode | 2/1: 1.6%, 3/1: 1.4% | 4/2–7/2: 1.8–4.5% | Canonical case |
| 204441 | IDA MSE w/ Er | 2/1–5/1: 0.6–4.1% | 6/2–9/2: 0.6–3.7% | Fortran n=2 broken (dW=-3568) |
| 153833 | IDA, q0<1 | 3/1: 1.6%, 4/1: 1.5% | 5/2–7/2: 0.9–1.5% | Was broken; fixed qlow filter |
| 153072 | Reversed shear | N/A | N/A | Excluded: multi-resonance not supported |

"Inner" = surfaces with ψ < 0.92, away from the integration domain edge.

**Scripts**: `examples/DIIID_stride_comparison/run_fortran_stride.py`, `run_comparison.jl`, `compare_results.py`
**CSV output**: `examples/DIIID_stride_comparison/outputs/diiid_stride_comparison.csv`

## Three Root Causes of Discrepancies

### (A) Equilibrium Profile Offset — 1–5% baseline error

The surface integration ODEs that compute q(ψ) and dV/dψ from the 2D equilibrium geometry use different solvers:
- **Fortran**: LSODE (Livermore Solver, Adams-Moulton multistep method, `mf=10`)
- **Julia**: BS5 (Bogacki-Shampine 5th order Runge-Kutta)

Both use `etol=1e-7` but converge to slightly different values (~0.001% in q, ~0.007% in dV/dψ). This propagates into the Fourfit matrices and ultimately into Δ'.

**Evidence**: Forcing Fortran's exact q values into Julia while keeping Julia's 2D geometry made dp31 WORSE (257% vs 165%), confirming that partial profile replacement creates inconsistency. All quantities must be self-consistent.

**Impact**: ~1–5% Re(Δ') error at inner surfaces. This is the floor — achieving <1% would require matching the ODE solver to LSODE-Adams or tightening tolerances beyond what the solver difference allows.

### (B) PEST3 Cancellation Amplification — affects dp31 at low ε/β

The STRIDE BVP Δ' is computed via the PEST3 convention:
```
deltap[i,j] = dp_raw[2i,2j] - dp_raw[2i,2j-1] - dp_raw[2i-1,2j] + dp_raw[2i-1,2j-1]
```

The dp_raw entries can be ~20,000 while deltap ~ O(1), giving cancellation ratios of 10,000–30,000:1. A 0.01% error in dp_raw translates to 100%+ error in deltap.

**Evidence**: At ε=0.125, dp_raw diagonal entries are ±20,000 and dp31 cancels to 1.04 (Fortran) vs 2.76 (Julia). The absolute error (1.7) is tiny relative to dp_raw (0.009%) but enormous relative to dp31 (165%).

**Impact**: dp31 errors >100% at low ε (< 0.25) and low β (< 0.01). This is inherent to the PEST3 formula and cannot be fixed without matching the equilibrium to machine precision.

### (C) Δ' Pole Proximity — affects high β and outermost surfaces

Δ' diverges as ~1/δW_total at ideal marginal stability (δW → 0). Julia's δW_total is systematically ~0.03 above Fortran's, shifting the pole location:
- **Fortran**: pole at pf ≈ 0.1850 (epsilon scan: ε ≈ 0.663)
- **Julia**: pole at pf ≈ 0.1874 (epsilon scan: ε ≈ 0.665)

**Evidence**: At pf=0.185, Fortran dp21=43,866, Julia dp21=559. Fortran's δW_total=0.015 (near zero), Julia's δW_total=0.043 (farther from zero). The ~0.03 offset is the accumulated effect of cause (A) in the volume-integrated energy.

**Impact**: >90% dp21 error for pf > 0.18. The 5/1 surface in DIII-D shots also suffers (ψ > 0.98, near psilim).

## Im(Δ') Assessment

Large |Im(Δ')| relative to |Re(Δ')| is a diagnostic for numerical artifacts. Both codes produce |Im/Re| < 0.05 for well-resolved inner surfaces. The assessment is mixed:

- **Fortran cleaner** at most inner surfaces (median |Im/Re| = 0.02 vs Julia's 0.11)
- **Julia cleaner** at some outer surfaces where Fortran has |Im/Re| > 5 (e.g., 147131 4/1: F=5.5, J=0.86)
- **204441 n=2**: Fortran produces dW=-3568 (nonsensical), suggesting a Fortran bug. Julia's dW=-0.026 is physical.

Further investigation is needed to determine whether Fortran's generally lower Im(Δ') at inner surfaces reflects genuine numerical superiority or a different (possibly compensating) error pattern in the BVP conditioning.

## Bugs Fixed During This Comparison

1. **`compute_delta_prime_matrix!` assertion** (Riccati.jl:325): Surfaces beyond psilim had no crossing chunks but were counted in msing. Fixed by building `sing_indices` from chunk.ising and creating a local `sing[]` alias for crossed surfaces only. (Fixed 7 epsilon scan failures + edge surfaces in DIII-D.)

2. **`sing_find!` didn't filter by qlow** (GeneralizedPerturbedEquilibrium.jl:178): Surfaces with q < qlow (e.g., 1/1 for 153833 with q0=0.916) were included in `intr.sing`, corrupting the Δ' BVP. Fixed by filtering `intr.sing` after `sing_find!` to exclude surfaces outside [qlow, psilim].

3. **Debug code crash** (Riccati.jl:568): `T_left_mats[1]` accessed without checking `has_ua` when `ctrl.verbose=true`. Fixed by guarding with `if has_ua`.

4. **q_edge reporting** (run_scan.jl): Scan runners reported `equil/qmax` (untruncated) instead of `info/qlim` (truncated at qhigh=3.6). Fixed — now correctly reports qlim.

## How to Re-run

```bash
# LAR scans
julia --project=. examples/LAR_epsilon_scan/run_scan.jl
julia --project=. examples/LAR_beta_scan/run_scan.jl
python3 examples/plot_scan_comparison.py

# DIII-D comparison
python3 examples/DIIID_stride_comparison/run_fortran_stride.py
julia --project=../.. examples/DIIID_stride_comparison/run_comparison.jl
python3 examples/DIIID_stride_comparison/compare_results.py
```

## ODE Solver Change: BS5 → VCABM (2026-04-12)

**Critical finding**: The Newcomb equation ODE solver was the dominant source of dp31 error.

BS5 (Bogacki-Shampine order 5 explicit Runge-Kutta) accumulates per-step errors that are amplified by PEST3 cancellation. Switching to VCABM (Variable-Coefficient Adams-Bashforth-Moulton, the Julia equivalent of Fortran STRIDE's ZVODE Adams-Moulton solver) dramatically improved dp31:

| Solver | dp31 at ε=0.125 | dp31 error vs Fortran | dp21 error |
|--------|----------------|----------------------|------------|
| BS5 (old) | 2.76 | 165% | 1.0% |
| VCABM (Adams-Moulton) | 1.07 | 2.7% | 7-16% oscillatory |
| **Vern9 (new default)** | **0.91** | **13%** | **1.0%** |
| Fortran ZVODE | 1.04 | — | — |

VCABM matched Fortran's dp31 best (2.7%) but produced erratic dp21 oscillations (7-16% at low ε) due to multistep startup transients in short integration chunks near singular surfaces. Vern9 is the best compromise: consistent dp21 (<1%) and dramatically better dp31 than BS5 (13% vs 165%).

**Updated epsilon scan results (Vern9)**:
- dp21: median 3.3%, <1% for 16/56 points — same as BS5
- dp31: max error dropped from 165% to 49%, median from 2.5% to 2.8%
- dp31 <5%: 38/56 points (was 36/56 with BS5)
- No oscillatory artifacts

The equilibrium solver (DirectEquilibrium.jl) retains BS5 since it's a simpler ODE with a refine callback. Vern9 takes fewer total steps (483 vs 1070 for solovev) due to larger steps at order 9.

## Next Steps

- [ ] Investigate the systematic δW offset — may be addressable via equilibrium solver matching
- [ ] Investigate Im(Δ') — is Fortran's lower Im from better conditioning or compensating errors?
- [ ] Support reversed-shear equilibria (multi-resonance surfaces, e.g., 153072)
- [ ] Investigate 204441 n=2 Fortran dW=-3568 — likely a Fortran bug, report upstream
- [ ] Consider Vern9 as alternative to VCABM — better dp21 accuracy, comparable dp31
