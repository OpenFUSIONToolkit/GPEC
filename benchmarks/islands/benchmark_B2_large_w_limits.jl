# benchmark_B2_large_w_limits.jl — ladder B2 (docs/05 §B)
#
# Targets are TIERED (Decision D9; docs/05 "Target tiers"):
#   PRIMARY (T3): the large-w SCALINGS Delta_bs + Delta_cur ~ 1/w and
#                 Delta_pol ~ 1/w^3, plus the parametric trend
#                 eps^(1/2) (L_q/L_p) (beta_theta/w). Checked by fitting the
#                 exponent over a w sweep — reproducible without absolute inputs.
#   AUDIT-GATED (T4): the 1/w COEFFICIENT vs WCHH96 Eq. (85) mapped to the
#                 island frame (Diss19 p. 86 frame caveat) [CHECKED: Diss19
#                 pp. 84-86; D21 Fig. 8-class curves]. Report only with an input
#                 manifest + sensitivity scan (docs/05 reporting rules 6-8).
#
# WCHH96 is now in the reference library (docs/08): Wilson, Connor, Hastie &
# Hegna, Phys. Plasmas 3, 248 (1996).
#
# STATUS: SKIPPED — gated per docs/src/islands/QUESTIONS.md:
#   Q2  D7 (re-derived equation set) — clearance in progress (M2b)
#   Q3  the Delta moment prefactors and electron-closure constants uncleared
#   Q4  the psi-tilde amplitude carries an open [VERIFY]

println("SKIPPED: B2 large-w limits — gated on QUESTIONS Q2, Q3, Q4.")
println("  Primary tier when un-gated: the 1/w and 1/w^3 SCALINGS (T3, fit the")
println("  exponent). The WCHH96 Eq. 85 coefficient is T4 (audit-gated).")
