# benchmark_B2_large_w_limits.jl — ladder B2 (docs/05 §B)
#
# Target: large-w analytic limits — Δ_bs + Δ_cur ∝ 1/w with coefficient against
# WCHH96 Eq. (85) mapped to the island frame (Diss19 p. 86 frame caveat), and the
# Δ_pol ∝ 1/w³ tail [CHECKED: Diss19 pp. 84–86; D21 Fig. 8-class curves].
#
# STATUS: SKIPPED — gated per docs/src/islands/QUESTIONS.md:
#   Q3  the Δ moment prefactors and electron-closure constants are uncleared
#   Q4  WCHH96 is not yet in the docs/08 reference library (cited only via
#       transcriptions) and the ψ̃ amplitude carries an open [VERIFY]
#   Q2  D7 (re-derived equation set) unratified.

println("SKIPPED: B2 large-w limits — gated on QUESTIONS Q2, Q3, Q4")
println("         (WCHH96 not in the reference library; ψ̃ [VERIFY] open; closure constants uncleared).")
