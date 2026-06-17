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

## Verdict: PASS WITH REQUIRED ANNOTATIONS (citation comments only; no physics defects found).
