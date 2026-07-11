# benchmark_B5_york_thresholds.jl — ladder B5a/B5b/B5c (docs/05 §B)
#
# Targets are TIERED by reproducibility (Decision D9; docs/05 "Target tiers"):
#
#   PRIMARY (run these first — reproducible without a full input manifest):
#     B5b/E1  T2  the :original -> :improved drift-model TOGGLE DIFFERENTIAL —
#                 a ~x6 reduction in w_c in an otherwise identical Islands
#                 configuration (the robust form of the sources' "8.73 -> 1.46
#                 rho_bi" story; shared input uncertainty cancels in the ratio).
#     B5a     T3  threshold EXISTENCE at w_c ~ O(rho_theta_i), :original model.
#     B5c     T3  the v_star TREND dw_c/dv_star > 0 (roughly linear over
#                 v_star in [5,20]e-3) and w_c proportional to rho_hat_theta_i.
#
#   AUDIT-GATED (T4 — absolute numbers, NOT pass/fail without an input manifest):
#     B5a  w_c ~= 2.76 rho_theta_i (half) = 8.73 rho_bi at eps=0.1 [CHECKED: I19 Fig. 9]
#          NB: I19 is internally inconsistent on its own run collisionality
#              (S4.2 v_star=0.01 vs L23 p.82 v_star=1e-3) — the type specimen of
#              why absolute matches need the input-completeness audit first.
#     B5b  w_c ~= 0.45 rho_theta_i = 1.46 rho_bi half-width [CHECKED: D21/D23a]
#     B5c  w_c ~= 0.440 rho_hat_theta_i + 0.0178 v_star - 7.54e-5 [CHECKED: L23 Eq. 6.3.2]
#          (best T4 candidate — a thesis documents its numerics most fully)
#
# STATUS: SKIPPED — the Level-0 drift-kinetic coefficients this needs are gated
# (QUESTIONS Q2/Q3/Q4; re-derivation-first clearance). When un-skipped: run the
# PRIMARY tier gates; report any T4 absolute comparison only alongside its input
# manifest and a sensitivity scan (docs/05 reporting rules 6-8). Do NOT tune a
# coefficient to hit an absolute number.

println("SKIPPED: B5a/B5b/B5c — gated on QUESTIONS Q2, Q3, Q4.")
println("  Primary tier when un-gated: B5b/E1 toggle differential (T2), threshold")
println("  existence (T3), dw_c/dv_star trend (T3). Absolute w_c values are T4")
println("  (audit-gated). See docs/src/islands/design/05-verification.md 'Target tiers'.")
