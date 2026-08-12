---
name: galerkin-assembly-map
description: RDCON singular Galerkin Δ′ solver — gal.f ↔ GalerkinAssembly.jl element-level weak-form correspondence and audit notes
metadata:
  type: project
---

# RDCON Galerkin Δ′ assembly: gal.f ↔ GalerkinAssembly.jl

File: `src/ForceFreeStates/Galerkin/GalerkinAssembly.jl` ↔ `GPEC_fortran/rdcon/gal.f`.
On branch `feature/rdcon-galerkin`. Reviewed 2026-06-16.

## Correspondence (verified, not just docstring-trusted)
- gal_hermite (222-262) ↔ gal_hermite: pb/qb EXACT.
- gal_get_fkg (1013-1071) ↔ gal_get_fkg: F=Fbar·sf[i]·sf[j], K=K·sf[i], G=G. singfac=mlow-nn*q+(0..mpert-1) = DIRECT (m-nq), not reciprocal. Correct.
- gal_gauss_quad (1079-1138) ↔ gal_gauss_quad!: F·qq + K·qp + conj(K[jpert,ipert])·pq + G·pp. swap pb(2)/pb(3) (0-based) = Julia pb[3]/pb[4]. Correct.
- gal_extension (1146-1322) ↔ gal_extension!: emat/ediag/rhs/erhs all match incl surface terms 1277-1300 and w1/w2.
- gal_resonant! ↔ gal_lsode_int(780-900)+gal_lsode_der(908-954)+gal_make_arrays signs(1366-1369). QuadGK replaces LSODE.
- gal_assemble_mat (694-772) ↔ gal_assemble_mat!: LU offset=kl+ku+1, chol offset=1, conj(e) Hermitian placement. Correct.
- gal_assemble_rhs (642-686) ↔ gal_assemble_rhs!: isol++ at ext2-left / res-right. Correct.
- gal_set_boundary (1554-1620) ↔ gal_set_boundary!: axis idx, edge 3-way (rpec/vac/fixed). DOF index 3(0-based)=Julia 4.

## Resonant sign chain (the trap — verified correct)
Fortran: LSODE integrates x0→x_lsode giving raw u; then line 882 `IF right: u=-u`; then make_arrays:
erhs=-u_res(1), ediag=+u_res(2), rhs=-u_hermite1, emat=+u_hermite2.
Net with s=+1(left)/-1(right): erhs=-s·res1, ediag=s·res2, rhs=-s·hbig, emat=s·hsmall.
Julia sgn_u = s (=-1 right/+1 left): erhs=-sgn_u·raw1, ediag=sgn_u·raw2, rhs=-sgn_u·hbig, emat=sgn_u·hsmall. MATCHES.
QuadGK limits x0→x_lsode preserve LSODE orientation so raw == pre-negation u. Correct.
du_res conj-weight uses the SMALL solution (isol(2)) in both. Correct.

## Only open item (MINOR, out-of-file)
sing_get_ua_gal/dua_gal two-sided z<0 → sig=-1 series at -z with derivative ×(-1) is a REIMPLEMENTATION
of sing.f's internal singular-surface handling, not a line port. The ×(-1) chain is internally consistent
but worth a dedicated check against sing.f sing_get_ua/dua if Δ′ ever disagrees with Fortran. Lives outside
GalerkinAssembly.jl.

## RPEC matching (GalerkinMatch.jl <-> rmatch/match.f match_rpec) — reviewed 2026-06-20
- match block (mat idx1..idx4, -delta1/delta2 signs) + rmat=-TRANSPOSE(delta coil block) + cout/cin split: EXACT port. PASS.
- matched outer soln Σcout(isol)·sols + sols(:,:,csol): EXACT (match_output_solution L32-41).
- resist.f E/F/H/M/G/K/taua/taur/v1: EXACT (Resist.jl). eta/rho caller-supplied = deliberate documented deviation.
- gal quadrature: jacobi_alloc "gll" = Gauss-Lobatto = Julia gausslobatto. nq1 lower=|nn·q1|, upper=|q1|: match.

## SUSPECT (should-fix) — RPEC forced eigenvalue γ
GalerkinMatch.jl:66 uses γ = 2π·im·n·gal_rotation. Fortran match_rpec passes `initguess` (namelist,
default 0.0) DIRECTLY to deltac_run — NOT i·n·rotation. The `guess+ifac*ntor*rotation` Doppler form
lives only in match_delta (Newton eigenvalue search), NOT the RPEC path. Two issues:
  (1) extra 2π: Fortran eig=ifac*ntor*rotation has NO 2π (rotation is ω[rad/s], eig is 1/time, q0=x0/taua).
  (2) RPEC path in Fortran uses initguess (the user forced eigenvalue), not a per-surface i·n·rotation shift.
Confirm intended convention vs Fortran before trusting resistive Δ(Q); ideal_flag path (γ unused) is unaffected.
RESOLUTION 2026-06-20: author confirms the Hz convention is DELIBERATE (gal_rotation is f[Hz], so γ=2πi·n·f);
the overstated comments were corrected. Δ(Q) now guarded by test/runtests_innerlayer.jl (Julia self-pin on
glasser_wang_2020_eq55). Remaining follow-up: upgrade that guard to a Fortran deltac_run / paper
cross-check to fully close the convention question.

## Verdict: PASS WITH REQUIRED ANNOTATIONS for outer solve; RPEC γ convention is SHOULD-FIX (verify vs match.f).


# Second pass


## PR #266 "Galerkin Integrator" full review (2026-06-25) — PASS
Exhaustive line-by-line vs gal.f / match.f / deltac.f / resist.f / sing.f. All physics-bearing code faithful.
- RESOLVED (was SUSPECT): forced eig γ=2πi·n·rotation IS correct. match.f match_rpec line:
  `rpec_eigenvalues(ising)=2*pi*rotation(ising)*REAL(ntor,r8)*ifac` — the 2π IS in Fortran. Julia matches exactly.
  My earlier "no 2π in Fortran" note was wrong (confused match_delta Newton path with match_rpec). Closed.
- Verified EXACT: gal_get_fkg (F=QF̄Q,K=QK̄,G=Ḡ,singfac=m-nq), gauss_quad weak form (F qq+K qp+K† pq+G pp),
  gal_extension, gal_resonant (mpert-fastest hermite layout; Fortran's apparent (0:np,mpert,2) reshape is a
  no-op relabel — der writes mpert-fastest, caller reads mpert-fastest), assemble_mat/rhs, set_boundary,
  gal_get_solution (restore_uh/us/ul), PEST-3 A/B/Γ/Δ ± combos, Δ′ extraction, match_rpec matrix signs,
  resist_eval E/F/G/H/K/M/taua/taur, mercier_di=E+F+H-1/4, rescale=sfac^(2p1/3)·v1^(2p1) + delta(1)↔(2) swap
  (swap applied in BOTH solve_inner and solve_inner_profile), sing_matvec, sing_get_dua, ξ_s=-A⁻¹(Bξ'+Cξ).
- Documented deviations (all fine): η/ρ caller-supplied (no Spitzer); gal_gamma default 5/3 = resist.f;
  QuadGK replaces LSODE; sing_get_dua omits Fortran `sig` (moved to sing_get_dua_gal left-side ×(-1)).
- InnerAsymptotics.jl + Shooting.jl diffs = PURE JuliaFormatter whitespace, zero numeric change.
- Reference.jl G=8.950e1,H=1.292e-2,K=2.332e2 = corrected GW2020 Eq.55 (prior transcription bug fixed).

## OPEN ITEM (latent, n≥2 only): gal_make_grid resonant-cell width asymmetry
gal.f gal_make_grid genuinely uses nq1=ABS(nn*sing(ising)%q1) on the LOWER bound but nq1=ABS(sing(ising+1)%q1)
(NO nn) on the UPPER bound. Julia GalerkinGrid.jl faithfully reproduces. Natural layer scale is 1/(n q1) on BOTH
sides (singfac'≈-n q1), so the upper/left side of each interior surface gets cells ~n× too wide for n≥2.
No effect at n=1 (RDCON resistive target). galerkin_solve allows single n≥2 (blocks only multi-n npert>1), so it
IS reachable. Recommend: keep faithful but warn, or fix both sides to nn*q1 (deviation, more correct) gated by
validation. Suspected upstream Fortran oversight, not a Julia port error.
