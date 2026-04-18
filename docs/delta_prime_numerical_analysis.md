# Δ' BVP: Numerical Analysis and Improvement Opportunities

**Purpose**: Identify numerically sensitive aspects of the STRIDE Δ' calculation and catalog opportunities where the Julia implementation could improve upon the Fortran STRIDE.

**Reference**: Glasser & Kolemen, Phys. Plasmas **25**, 082502 (2018) — "A robust solution for the resistive MHD toroidal Δ' matrix in near real-time"

## 1. The Δ' BVP Structure (Paper Sec. II-D, IV)

The Δ' matrix is extracted from a boundary value problem (BVP) built on the toroidal matrix Newcomb equation (Eq. 22 of the paper):

```
(F·ξ' + K·ξ)' - (K†·ξ' + G·ξ) = 0
```

This is recast as a 2M×2M Hamiltonian system (Eq. 24) with q = ξ and p = F·ξ'+K·ξ:

```
u' = L·u,   u = [q; p] ∈ ℂ^{2M}
```

where L is singular at rational surfaces (q(ψ*) = m/n).

### BVP Degrees of Freedom

For N rational surfaces, the BVP has (2N+2)×(2M) unknowns (mode coefficients on each subinterval). After imposing:
- M axis BCs (q(0) = 0)
- M edge BCs (q(1) = 0 or vacuum coupling)
- (2M-2) continuity conditions at each rational surface
- 2M continuity at each interstitial surface

There remain exactly **2N undetermined DOF** — these are the big/small solution coefficients that form the **2N × 2N Δ' matrix**.

### PEST3 Convention

The raw BVP produces a 2N × 2N matrix dp_raw indexed by (L₁, R₁, L₂, R₂, ..., Lₙ, Rₙ). The physical Δ' matrix (N × N) is extracted via the PEST3 formula:

```
Δ'[i,j] = dp_raw[2i,2j] - dp_raw[2i,2j-1] - dp_raw[2i-1,2j] + dp_raw[2i-1,2j-1]
```

This represents Δ' = (A_R - A_L), the difference of small solution coefficients on the right and left of each surface.

## 2. Numerically Sensitive Points

### 2.1. Asymptotic Expansion at Rational Surfaces (Paper Eq. 26-28)

At each rational surface ψ*, the 2M solutions split into:
- **(2M-2) nonresonant modes**: scale as (ψ - ψ*)⁰ → well-behaved
- **2 resonant modes**: scale as (ψ - ψ*)^{1/2 ± √Δ_I}
  - **Big solution** (z^{-α}): diverges as ψ → ψ* — dominates any integrated mode near the surface
  - **Small solution** (z^{+α}): vanishes as ψ → ψ* — gets swamped by big solution during integration

**Numerical challenge**: When integrating TOWARD a rational surface, the big solution component grows exponentially and contaminates all modes. When integrating AWAY from a surface, the small solution component grows and contaminates. This is why STRIDE shoots asymptotic expansions AWAY from surfaces (Paper step 3, Sec. IV).

**Status in Julia**: Julia uses the same shoot-away approach via `integrate_fm_with_ua_ic`. The asymptotic expansion order is controlled by `sing_order` (default 6). Both codes use the same asymptotic basis from Glasser 2016 Sec. IV.

**Improvement opportunity**:
- The asymptotic expansion accuracy depends on ε (distance from the surface where expansions are initialized). Currently `singfac_min = 1e-4` sets ε ~ 1e-4/|n·q'|. Smaller ε gives more accurate asymptotics but requires higher sing_order to avoid truncation error. There may be an optimal ε-vs-sing_order trade-off that differs from Fortran's choice.
- Julia could implement **adaptive sing_order** — automatically increasing the expansion order until the asymptotic basis converges to a specified tolerance, rather than using a fixed order everywhere.

### 2.2. Conditioning of the Shooting Propagators (Paper Eq. 40)

State transition matrices Φ(ψ₂, ψ₁) propagate ODE solutions across intervals. As the interval |ψ₂ - ψ₁| grows, the condition number of Φ grows exponentially (big solutions dominate). The paper notes (Sec. V):

> "each subinterval depicted in Fig. 4 may be further subdivided — as finely as desired — with each subdivision integrated in parallel"

**Numerical challenge**: cond(Φ) can reach 10¹⁵–10²⁵ for full-span propagators. The PEST3 formula subtracts nearly-equal dp_raw entries, amplifying any conditioning errors.

**STRIDE's approach**:
- **Parallel FM**: subdivides into many chunks, multiplies propagators
- **Midpoint shooting**: splits inter-surface gaps at midpoints, giving cond ≈ √(full cond)
- **Asymptotic basis initialization**: shoots from ua ICs for column-by-column accuracy

**Status in Julia**: Julia implements all three techniques. The midpoint splitting and ua-initialized shooting are in `compute_delta_prime_matrix!`.

**Improvement opportunities**:
- **Multiple midpoints**: Instead of a single midpoint per inter-surface gap, Julia could split into 3+ points, further reducing condition numbers. For very wide gaps (e.g., axis to first surface), this could significantly improve conditioning.
- **Riccati-based Δ'**: The Riccati formulation (Paper Sec. V, Ref. 1) maintains bounded state variables by factoring the propagator as S = U₁·U₂⁻¹. Julia already implements Riccati integration for the ODE but uses the FM-based BVP for Δ'. A fully Riccati-based Δ' computation would avoid the exponentially ill-conditioned propagator matrices entirely.
- **S-matrix axis BC**: Julia already uses the Riccati S matrix at the first surface's left boundary as the axis BC, which is well-conditioned (O(1)–O(10⁴)). This is a significant improvement over the raw axis propagator (cond ~ 10²⁴).

### 2.3. PEST3 Cancellation

The PEST3 formula (deltap = dp_raw[2i,2j] - dp_raw[2i,2j-1] - dp_raw[2i-1,2j] + dp_raw[2i-1,2j-1]) involves catastrophic cancellation when the dp_raw diagonal entries are much larger than the Δ' result.

**Observed cancellation ratios**:
- dp21 (2/1 surface): ~600:1 — manageable
- dp31 (3/1 surface): ~15,000–30,000:1 at low ε/β — catastrophic
- Near Δ' poles: ratios can exceed 100,000:1

**Improvement opportunity**:
- **Direct Δ' formulation**: Instead of computing the full 2N×2N dp_raw matrix and taking differences, formulate the BVP directly in terms of (A_R - A_L) — the physical Δ' quantity. This would avoid the PEST3 subtraction entirely.
- **Extended precision**: For the dp_raw solve only, use higher-precision arithmetic (e.g., Double64 from DoubleFloats.jl) to maintain accuracy through the cancellation. This is feasible in Julia but impractical in Fortran.
- **Relative error monitoring**: Compute and report the PEST3 cancellation ratio for each surface, flagging results where the ratio exceeds a threshold (e.g., 1000:1).

### 2.4. Vacuum Coupling at the Edge (Paper Eq. 38)

The plasma edge BC with vacuum response is:

```
U(1, 1) = [0_M; W_V]    (Eq. 38)
```

where W_V is the vacuum response matrix. This couples the edge subinterval to the vacuum calculation.

**Numerical challenge**: The vacuum response matrix W_V is itself computed from a separate Green's function calculation with its own numerical sensitivities. Errors in W_V propagate directly into the Δ' edge BC.

**Status in Julia**: Julia computes W_V via the pure-Julia vacuum module.

**Improvement opportunity**: Investigate whether the Julia vacuum module's W_V differs from Fortran's — this could contribute to the systematic δW offset. The vacuum module uses different quadrature and interpolation methods which could introduce ~0.1% differences in W_V.

### 2.5. Equilibrium Reform (Fortran-specific)

The Fortran STRIDE performs **equilibrium reformation** (`reform_eq_with_psilim`): it re-solves the equilibrium on the truncated domain [psilow, psilim], regenerating all splines on this reduced interval. Julia does NOT do this — it uses the original equilibrium splines evaluated on the truncated domain.

**Impact**: Reformation can change the equilibrium profiles by O(0.01%), particularly near the edges where spline extrapolation behavior differs. This is a likely contributor to the systematic δW_total offset (~0.03) observed in the beta scan.

**Investigation needed**: Compare q and dV/dψ profiles between reformed-Fortran and non-reformed-Julia equilibria. If reformation is significant, consider implementing it in Julia.

### 2.6. ODE Solver Differences

| Feature | Fortran STRIDE | Julia GPEC |
|---------|---------------|------------|
| ODE solver | ZVODE (complex Adams-Moulton) | BS5 (real Bogacki-Shampine 5th order) |
| Tolerance | tol_nr=1e-8, tol_r=1e-8 | eulerlagrange_tolerance=1e-8 |
| Step control | ZVODE internal | DifferentialEquations.jl adaptive |
| Complex arithmetic | Native complex ODE | Real-valued with complex state reshaping |

**Improvement opportunity**: Julia could use LSODE.jl (a Julia wrapper for the same LSODE solver Fortran uses for equilibrium) or implement an Adams-Moulton method to better match Fortran's integration behavior. Alternatively, investigate whether tightening Julia's tolerances beyond 1e-8 converges the Δ' values.

## 3. Opportunities to Outperform Fortran STRIDE

### 3.1. Fully Riccati-Based Δ' (Most Promising)

The current approach computes Δ' via FM propagators + BVP. An alternative:

1. Integrate the Riccati equation dS/dψ = F(S, ψ) from axis to each surface
2. At each surface, the Riccati S matrix directly encodes the ratio of big/small solutions
3. Extract Δ' from S without the ill-conditioned FM matrices

Julia already has the Riccati integration infrastructure (used for δW). Extending it to compute Δ' would:
- Eliminate exponential conditioning issues
- Eliminate PEST3 cancellation (compute Δ' = A_R - A_L directly)
- Potentially be faster (one forward pass instead of parallel FM + BVP solve)

The paper mentions (Sec. V) that "the square-root algorithm for Riccati problems could reduce the computational burden" — this is unexplored territory.

### 3.2. Extended Precision for Critical Computations

Julia's type system makes it trivial to swap Float64 for higher-precision types:
- `Double64` (from DoubleFloats.jl): ~31 decimal digits, ~2× slower than Float64
- `BigFloat`: arbitrary precision, ~100× slower

Strategy: run the equilibrium and bulk ODE integration in Float64, but switch to Double64 for:
- The PEST3 combination of dp_raw
- The asymptotic expansion evaluation near surfaces
- The BVP linear solve

This targeted approach would improve accuracy where it matters most without significant performance impact.

### 3.3. Adaptive Asymptotic Expansion Order

Instead of a fixed `sing_order=6` everywhere, Julia could:
1. Evaluate the expansion at order k and k+2
2. Compare: if the difference exceeds a tolerance, increase k
3. Continue until convergence

This would automatically use higher-order expansions for challenging surfaces (e.g., near the edge where DI approaches -1/4) while keeping the order low for well-behaved inner surfaces.

### 3.4. Reciprocity Relations

The paper notes (Sec. V): "the reciprocity relations of the Δ' matrix discussed in Refs. 13 and 28 could reduce the degrees of freedom of the Δ' BVP."

The self-adjointness of the ideal MHD force operator implies Δ'[i,j] = Δ'[j,i] (the matrix is symmetric). This means only N(N+1)/2 BVP solves are needed instead of 2N. For N=4 surfaces, this reduces from 8 to 10 solves — modest savings, but also provides an independent consistency check.

### 3.5. Parallel-in-ψ Integration

STRIDE already parallelizes by subdividing the ψ interval (Paper Eq. 40, Fig. 7). Julia's implementation uses this. Additional parallelization opportunities:
- **Column-parallel BVP**: The 2N right-hand sides of the BVP can be solved simultaneously
- **Surface-parallel asymptotics**: Each surface's expansion can be computed independently
- **n-parallel**: Different toroidal mode numbers are fully independent

## 4. Key Fortran vs Julia Implementation Differences

From detailed code comparison (stride/ode.F, stride/sing.F vs Riccati.jl):

### 4.1. Equilibrium Reformation

**Fortran** (`stride.F:156-164`): FORCES `reform_eq_with_psilim=.TRUE.` on entry — re-solves and re-splines the equilibrium on the truncated domain [psilow, psilim]. This changes where all profile quantities are evaluated.

**Julia**: No equilibrium reformation. Uses the original equilibrium splines.

**Impact**: This is almost certainly the largest contributor to the systematic δW offset (~0.03). The re-splined Fortran equilibrium has subtly different profiles at all ψ locations.

### 4.2. BVP Architecture

**Fortran**: Dense matrix BVP. Size = (2+2·msing)·mpert. Single-shot shooting from each surface. Solves via LAPACK ZGETRF/ZGETRS (pivoted LU).

**Julia**: Two-path architecture:
- **S-axis path** (default): Uses Riccati S matrix for axis BC (well-conditioned). Size = (2+4·msing)·N with midpoint unknowns.
- **FM-axis fallback**: More similar to Fortran.

Julia's midpoint-splitting for inter-surface segments produces a LARGER BVP matrix but with better-conditioned blocks — fundamentally different from Fortran's single-shot approach.

### 4.3. Asymptotic Basis Handling

**Fortran**: "Bakes" the asymptotic transformation T into shooting propagators via `uFM_sing_init`. Shooters are already in asymptotic basis.

**Julia**: Pre-computes T = [ua[:,:,1]; ua[:,:,2]] separately, then applies T·Φ and T⁻¹·Φ at assembly time. Computes T_inv via `inv()`.

If T is ill-conditioned (possible near Mercier-marginal surfaces where α → 0), the `inv(T)` in Julia could introduce errors that Fortran avoids by baking T directly.

### 4.4. Vacuum Edge BC Sign Convention

**Fortran** (`ode.F:1020`): `uEdge(mpert+1:m2, mpert+1:m2) = -wv * psio²`

**Julia** (`Riccati.jl:691`): `M[..., col_edge] .= wv .* psio²`

The sign difference needs investigation — it may be absorbed by a different convention for the q/p ordering, or it could be an actual bug. Both codes produce similar (not identical) results, suggesting the sign is handled consistently overall but may introduce a subtle phase difference in Im(Δ').

## 5. Investigation Priorities

Ranked by expected impact on Δ' accuracy:

1. **Equilibrium reformation** (Sec. 2.5, 4.1) — Fortran FORCES reformation, Julia doesn't do it. This is almost certainly the dominant source of the systematic δW offset (~0.03) and the 1-5% Δ' baseline error. Implementing or understanding this is the single most impactful improvement.
2. **Vacuum edge BC sign convention** (Sec. 4.4) — Fortran uses -wv·psio², Julia uses +wv·psio². Needs investigation to confirm this isn't causing Im(Δ') discrepancies.
3. **PEST3 cancellation mitigation** (Sec. 2.3) — extended precision or direct Δ' formulation would fix the low-ε/β dp31 issue.
4. **Riccati-based Δ'** (Sec. 3.1) — would fundamentally eliminate conditioning issues and potentially outperform Fortran.
5. **Asymptotic basis conditioning** (Sec. 4.3) — Julia's explicit T⁻¹ may be less stable than Fortran's baked-in approach near Mercier-marginal surfaces.
6. **Adaptive asymptotics** (Sec. 3.3) — would improve edge surface accuracy.
7. **Im(Δ') investigation** — determine whether Julia's larger Im(Δ') at inner surfaces is from the sign convention, T⁻¹ conditioning, or something else.
