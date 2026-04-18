# Riccati-Based Δ' Solver: Working Notes

**Branch**: `experiment/riccati-delta-prime`
**Started**: 2026-04-12
**Goal**: Compute Δ' directly from Riccati S matrices instead of FM propagators + BVP

## Key Insight from Double64 Test (2026-04-12)

Running the BVP solve in Double64 (~31 digits) produced **identical** Δ' values to Float64. This proves the precision bottleneck is NOT in the BVP solve or PEST3 subtraction — it's in the **FM propagators** that feed the BVP. The dp_raw entries themselves carry equilibrium-level errors that no amount of linear algebra precision can fix.

This motivates the Riccati approach: if we can compute Δ' without forming the ill-conditioned FM propagators, we might extract more accurate dp_raw entries from the well-conditioned Riccati S matrices.

## Theory: How Δ' Relates to the Riccati S Matrix

### Single surface case

The Riccati matrix S = U₁·U₂⁻¹ at a rational surface encodes the plasma response:
- **From the left (axis side)**: S_L at ψ*⁻ is bounded, O(1)–O(10⁴)
- **From the right (edge side)**: S_R at ψ*⁺ is bounded similarly
- **The FM propagator**: Φ = [U₁; U₂] has cond ~ 10¹⁵–10²⁵

At the surface, in the asymptotic basis (T = [ua[:,:,1]; ua[:,:,2]]):
- The state vector u = [ξ; η] = T·c where c = [c_big; c_small]
- c_big = coefficient of diverging (z^{-α}) solution
- c_small = coefficient of vanishing (z^{+α}) solution
- Δ' = c_small_right - c_small_left when c_big = 1

The question: given S_L and S_R (bounded), can we extract c_big and c_small without forming Φ?

### The relationship: S and asymptotic coefficients

At the left boundary of a surface:
```
u = [ξ; η] = [S_L·η; η]  (since ξ = S·η in Riccati form)
```

In asymptotic basis:
```
c = T⁻¹·u = T⁻¹·[S_L; I]·η
```

So the big/small coefficients are determined by T⁻¹·[S_L; I]. The c_big component (row ipert of T⁻¹·[S_L; I]) gives the big solution coefficient as a function of η. The c_small component (row ipert+N of T⁻¹·[S_L; I]) gives the small coefficient.

For a driving of c_big = 1, we need η such that (T⁻¹·[S_L; I]·η)_ipert = 1, then Δ'_left = (T⁻¹·[S_L; I]·η)_{ipert+N}.

This is well-conditioned if T and S_L are well-conditioned!

## Attempt Log

### Attempt 1: Minimum-norm pseudoinverse extraction (2026-04-12)

**Approach**: At each surface, compute P = T⁻¹·[S; I], then use the pseudoinverse of the big-solution row P[ipert,:] to drive c_big=1 and read c_small = P[ipert+N,:] · η.

**Result**: FAILED — Riccati Δ' values are ~32,000% off from BVP values. The extracted values are O(dp_raw) ≈ O(10,000) instead of O(deltap) ≈ O(10).

**Root cause**: The minimum-norm pseudoinverse gives η that satisfies the driving constraint but **violates the (2M-2) continuity conditions**. The BVP constrains all nonresonant modes to be continuous across the surface. Without these constraints, the "small solution coefficient" includes contributions from all modes, not just the physical small solution.

**Key insight**: The PEST3 formula Δ' = dp_raw[2i,2j] - dp_raw[2i,2j-1] - dp_raw[2i-1,2j] + dp_raw[2i-1,2j-1] is specifically designed to cancel the contributions of nonresonant modes. When we extract c_small from a single surface without enforcing continuity, we get a "raw" coefficient that still has the nonresonant contamination. The PEST3 subtraction is what removes it.

**Conclusion**: A purely per-surface approach cannot compute the PEST3 Δ'. We need to incorporate the continuity conditions into the Riccati extraction. This means either:
(a) Formulating a small BVP at the surface that uses S_L and S_R as BCs (but this brings back a matrix solve), or
(b) Understanding what the PEST3 subtraction physically means in terms of the Riccati matrices and implementing it directly.

### Attempt 2: Local BVP at each surface with continuity (2026-04-12)

**Approach**: At each surface, formulate a 2N×2N BVP:
- Left: u_L = [S_L; I] · η_L, Right: u_R = [S_R; I] · η_R
- Continuity of nonresonant modes: P_L[k,:]·η_L = P_R[k,:]·η_R for k ≠ ipert, ipert+N
- Driving: P_L[ipert,:]·η_L = 1 and P_R[ipert,:]·η_R = 1

**Result**: FAILED — Still ~32,000% off. Values are O(dp_raw) ≈ O(thousands) instead of O(deltap) ≈ O(10).

**Root cause**: The S matrices encode **all** surfaces between axis/edge and the current surface. S_left at surface j carries the cumulative effect of crossing surfaces 1, 2, ..., j-1. The local BVP at surface j only enforces continuity at surface j, not at all other surfaces simultaneously. The dp_raw values from the per-surface solve include contributions from the other surfaces that the PEST3 formula cancels in the full BVP.

**Key realization**: The PEST3 formula is not just a post-processing step — it's an integral part of extracting physically meaningful Δ' from a multi-surface system. The per-surface Riccati Δ' gives the "coupled" response including all inter-surface effects, while the PEST3 formula extracts the "isolated" per-surface response.

### Next Steps

The correct approach for a Riccati-based Δ' must solve ALL surfaces simultaneously:

**Option A**: Formulate the full multi-surface BVP using S matrices as BCs. This would replace the FM propagators Φ(aᵢ, ψ±ⱼ) with S-matrix-derived quantities. The BVP size is the same as the current approach, but the matrix entries should be better-conditioned since they're derived from bounded S matrices.

**Option B**: Use S matrices to directly derive the Fortran-equivalent "shooting propagators" (uShootR, uShootL) that feed the BVP. This is closer to the current implementation — instead of extracting shooting propagators from FM chunks, derive them from S. The advantage: S at each surface is O(1)–O(10⁴), while the FM propagators can be O(10¹⁵).

**Option C**: Reformulate the Δ' definition itself. Instead of the PEST3 convention (which is tied to the FM-based BVP), define a "Riccati Δ'" that computes the physical tearing stability parameter directly from S matrices and asymptotic coefficients. This would require understanding the relationship between the PEST3 Δ' and the asymptotic matching formulas (Glasser 2016 Sec. VII).

**Assessment**: Options A and B preserve compatibility with the existing PEST3 convention. Option C is more fundamental but risks defining a different quantity that's hard to compare against Fortran. I recommend pursuing Option B first — it's the most incremental change and could give immediate improvements.

### Deeper Investigation: Why the Riccati S matrix can't directly drive big solutions (2026-04-12)

**Diagnostic**: Computed P = T⁻¹·[S_left; I] at surface 1. The big solution row P[ipert,:] has norm 0.007, while the small solution row P[ipert+N,:] has norm 6.15 — a ratio of 0.001. This means the Riccati state at the surface has **nearly zero projection onto the big solution**.

**Root cause**: This is not a bug — it's a fundamental property of the Riccati formulation. The Riccati S = U₁·U₂⁻¹ stays bounded by construction, precisely because both U₁ and U₂ grow together near the surface (the big solution dominates both). In the ratio S, the big solution's absolute amplitude cancels out. S tracks the *relative* mode structure, not the absolute big/small decomposition.

For the FM-based BVP, the propagator Φ = [U₁; U₂] has explicit columns for big and small solutions. The BVP can drive the big solution coefficient to 1 because the column is there. For S, the big solution information has been "divided out" by the renormalization. Trying to recover it from S is numerically ill-conditioned — it's like dividing by the very quantity that was cancelled to make S bounded.

**Implication**: A purely Riccati-based Δ' that tries to extract big/small coefficients from S at the surface is fundamentally limited. The information S discards (the big solution amplitude) is exactly what the Δ' BVP needs.

**However**: The shooting propagators `uShootR`, `uShootL` in the current BVP are NOT full-span propagators — they're initialized with ua ICs (asymptotic basis) at each surface and integrated over short spans. These short-span propagators are well-conditioned (cond ~ 10²-10⁴) because they don't span the full axis-to-edge range. The Riccati S matrix at each surface IS useful for the axis BC (where S encodes the axis regularity condition in bounded form). The ill-conditioning enters only in the full-span propagators that the S-axis BC was designed to replace.

**Revised conclusion**: The existing implementation (S-based axis BC + ua-initialized short-span shooting) is already close to optimal. The remaining conditioning issues are in:
1. The inter-surface segments where the shooting propagators span longer distances
2. The asymptotic basis T itself (cond ~ 10⁶ for these equilibria)

A Riccati-based improvement would target (1) by splitting inter-surface segments more finely or using the Riccati S to construct better-conditioned mid-interval matching conditions. But this is an incremental improvement, not a paradigm shift.

**Status**: The fully Riccati Δ' approach is **analytically valid but numerically ill-conditioned** for the critical step (recovering big solution information from S). The current hybrid approach (Riccati for axis BC + FM for the BVP) is likely near-optimal for the STRIDE formulation. Further improvements should focus on the equilibrium (reformation, ODE solver matching) rather than the BVP linear algebra.

## Equilibrium Reformation Investigation (2026-04-12)

### Key Finding: dp_raw entries differ by 13-23% between Fortran and Julia, but this is EXPECTED

The dp_raw matrices from Fortran (`delta_prime.out`) and Julia (`verbose` output) have very different absolute values. For ε=0.125:

| Entry | Julia | Fortran | Ratio |
|-------|-------|---------|-------|
| dp_raw[1,1] | -9396 | -10820 | 1.15 |
| dp_raw[2,2] | +9730 | +11196 | 1.15 |
| dp_raw[3,3] | -20751 | -26211 | 1.26 |
| dp_raw[4,4] | +23142 | +30028 | 1.30 |

But the PEST3 Δ' (dp21 = 11.36 Julia vs 11.26 Fortran = 0.9%) agrees much better. The dp_raw difference is because **the two codes use different BVP structures**: Fortran uses `nMat = (2+2·msing)·mpert` without midpoint unknowns, Julia uses `nMat = (2+4·msing)·N` with midpoint unknowns for conditioning. Different BVP structures produce different dp_raw bases but should give the same PEST3 Δ'. This is analogous to different coordinate representations of the same physical quantity.

### Reformation impact: small for this test case

For ε=0.125, qmax=3.57 < qhigh=3.6, so psilim=psihigh=0.995 and **Fortran does NOT reform** (condition `psilim /= psihigh` is false). Running Julia with psihigh=psilim changes dp21 by only 0.03%.

### Remaining dp31 discrepancy source

The dp31 PEST3 result (2.76 Julia vs 1.04 Fortran reference) comes from the accumulated effect of:
1. Different ODE solvers (BS5 vs ZVODE) for the Newcomb equation
2. Different equilibrium field-line integrators (BS5 vs LSODE) for q and dV/dψ
3. PEST3 amplification: a 0.008% dp_raw error becomes 165% in dp31 at 15,000:1 cancellation

This is a fundamental limitation that no BVP trick can fix. The only path to matching dp31 is matching the ODE solvers themselves.
