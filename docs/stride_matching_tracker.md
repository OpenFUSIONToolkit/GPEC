# STRIDE Matching Tracker

## Goal
Match Fortran STRIDE Δ' values to <1% for ALL equilibria across:
- Epsilon scan (56 points)
- Beta scan (42 points)
- DIII-D shaped equilibria (IDA_run cases)

## Current Status (2026-04-11)

### What Works
| Quantity | ε=0.125 | Notes |
|----------|---------|-------|
| δW_plasma | 1.2% error | Good |
| δW_vacuum | 0.01% error | Fixed by wall parameter correction |
| δW_total | 0.27% error | Good |
| dp21 (m=2) | 1.0% error | Good |
| dp31 (m=3) | **165% error** | **ROOT PROBLEM** |

### What Has Been Tried

| # | Date | Change | Result | Verdict |
|---|------|--------|--------|---------|
| 1 | Pre-2026-04-11 | Wall parameter fix: a=0.1→a=20 in gpec.toml | δW_vacuum 274%→0.01% | **FIXED** (keep) |
| 2 | Pre-2026-04-11 | ν hypothesis: Fortran doesn't use ν in inverse transform | WRONG — both codes use ν | **REVERTED** |
| 3 | 2026-04-11 | Test propagator chaining fallback (ctrl=nothing) | dp21=-15265 (catastrophic) | **BAD** (don't use) |
| 4 | 2026-04-11 | Analysis: BVP cond=8.5e10, PEST3 cancellation=34000:1 | Explains precision loss | Diagnosis only |
| 5 | 2026-04-11 | Compared dp_raw matrices Julia vs Fortran | Non-uniform scaling (0.65× diag, 0.45× offdiag) | Diagnosis only |

### Root Cause Analysis

The dp31 discrepancy is NOT a simple precision problem. The dp_raw matrices differ systematically between codes:
- Julia dp_raw surface 2: [-20751, +1194, +1194, +23142]
- Fortran dp_raw surface 2: [-30450, +2647, +2647, +35745]

These are fundamentally different values, not just round-off differences. The difference must come from:
1. Different ua (asymptotic basis) values
2. Different shooting propagator construction
3. Different interval decomposition
4. Different midpoint positions

### Approach Going Forward

**STOP trying alternative approaches. Match Fortran STRIDE exactly.**

1. Add systematic WRITE statements to Fortran ode.F at every key computation point
2. Add corresponding print statements to Julia Riccati.jl at matching locations
3. Run both codes on the same equilibrium (ε=0.125) and compare step-by-step
4. Identify the FIRST point where values diverge
5. Fix that divergence, then repeat

### Systematic Comparison Points

These are the locations where WRITE/print statements need to be added, in order of execution:

#### Phase 1: Equilibrium/Setup
- [ ] q-profile at rational surfaces (q=2, q=3)
- [ ] Singular surface ψ locations
- [ ] psifac (ψ_singularity) for each surface
- [ ] singfac_min, inner layer width

#### Phase 2: Asymptotic Expansion
- [ ] alpha (Frobenius exponent) at each surface
- [ ] vmatr coefficients (sing_order terms) at each surface
- [ ] vmatl coefficients at each surface
- [ ] ua evaluation point (dpsi from singularity)
- [ ] ua values at each surface (big and small columns, resonant mode)

#### Phase 3: Interval Decomposition
- [ ] Total number of intervals/chunks
- [ ] Interval boundaries (psi_start, psi_end for each)
- [ ] Which intervals are singular boundaries
- [ ] singMidPt / midpoint index for each surface
- [ ] psi values at midpoints

#### Phase 4: Per-Interval FM Integration
- [ ] Per-interval FM (uFM_all) resonant-column norms
- [ ] Integration direction (forward/backward) per interval
- [ ] ZVODE vs BS5 step counts per interval
- [ ] Tolerance settings comparison

#### Phase 5: Shooting Propagator Construction
- [ ] uShootL construction: which intervals are chained
- [ ] uShootR construction: which intervals are chained
- [ ] uShootL resonant-column values (first 5 components)
- [ ] uShootR resonant-column values (first 5 components)
- [ ] uAxis column norms after QR conditioning
- [ ] Condition numbers of all propagators

#### Phase 6: Edge BC
- [ ] wv matrix diagonal entries (resonant modes)
- [ ] psio value
- [ ] uEdge construction (how edge FM is built)

#### Phase 7: BVP Assembly
- [ ] BVP matrix dimension (nMat)
- [ ] BVP matrix condition number
- [ ] Column/row structure verification

#### Phase 8: BVP Solution
- [ ] dp_raw matrix (full, with 12+ digits)
- [ ] PEST3 deltap values
- [ ] BVP residuals (||Mx-b||)
- [ ] Individual x solution components at each surface

### Key Differences Already Identified
1. **Interval count**: Julia 26 vs Fortran 33 (nIntervalsTot=33)
2. **Midpoint position**: Julia ψ_mid=0.737 vs Fortran ψ_mid≈0.725
3. **ua values**: Completely different between codes (need to trace back to asymptotic coefficients)
4. **Shooting propagator construction**: Julia re-integrates from ua ICs; Fortran chains per-interval FMs
5. **ODE solver**: Julia BS5 vs Fortran ZVODE

### Files to Modify
- **Fortran**: `~/Desktop/plasma/GPEC/stride/ode.F` — add WRITE statements
- **Fortran**: `~/Desktop/plasma/GPEC/stride/sing.F` — add WRITE for ua/vmatr
- **Julia**: `src/ForceFreeStates/Riccati.jl` — add @info statements
- **Julia**: `src/ForceFreeStates/Sing.jl` — add @info for ua/vmatr
- **Julia**: `src/ForceFreeStates/EulerLagrange.jl` — add @info for intervals

### Test Equilibria
1. **ε=0.125** — primary debug case (2 rational surfaces, m=2 and m=3)
2. **ε=0.5** — larger aspect ratio, check if dp31 matches better
3. **DIII-D** — from IDA_run, real-world shaped equilibrium
