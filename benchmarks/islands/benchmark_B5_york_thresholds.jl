# benchmark_B5_york_thresholds.jl — ladder B5a/B5b/B5c (docs/05 §B)
#
# Targets (transcribed, awaiting human sign-off — see docs/src/islands/design/05):
#   B5a  DK-NTM threshold w_c ≃ 2.76 ρ_θi (half-width) ≡ 8.73 ρ_bi at ε = 0.1,
#        :original drift model [CHECKED: I19 Fig. 9; PRL p. 4]
#        + open [VERIFY]: run collisionality (I19 §4.2 ν_★ = 0.01 vs L23 p. 82 ν_★ = 1e-3).
#   B5b  RDK-NTM improved-model threshold w_c ≈ 0.45 ρ_θi ≡ 1.46 ρ_bi half-width
#        [CHECKED: D21 abstract + Fig. 8; D23a abstract]
#   B5c  kokuchou finite-ν_★ surface w_c ≈ 0.440 ρ̂_θi + 0.0178 ν_★ − 7.54e-5
#        [CHECKED: L23 Eqs. 6.3.1–6.3.2], L23-amended equation set.
#
# STATUS: SKIPPED — running this requires assigning Level-0 physics
# coefficients that are at most [CHECKED] and not human-cleared:
#   Q2  ratify D7 (re-derived L0 equation set) and D8 (the B5a/b/c triangle)
#   Q3  clear the L0 coefficient set (ω̂_D/L̂_B toggle, collision kernel,
#       electron closure, quasineutrality closure)
#   Q4  resolve the ψ̃ amplitude [VERIFY] and the B5a collisionality [VERIFY]
# per docs/src/islands/QUESTIONS.md. Do not fill in a number to make this run.

println("SKIPPED: B5a/B5b/B5c York thresholds — gated on QUESTIONS Q2, Q3, Q4")
println("         (uncleared [CHECKED] L0 coefficients; D7/D8 unratified).")
println("         See docs/src/islands/QUESTIONS.md and docs/src/islands/design/05-verification.md.")
