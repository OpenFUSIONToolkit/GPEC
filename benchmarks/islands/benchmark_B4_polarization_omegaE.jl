# benchmark_B4_polarization_omegaE.jl — ladder B4 (docs/05 §B)
#
# Target: polarization structure — (i) Wilson–Connor collisionless and Smolyakov
# collisional scalings; (ii) Δ_pol ∝ ω_E² away from zero with the sign reversal
# at ω_E ≈ −0.89 ω_dia,e, reversal point insensitive to w/ρ_θi; (iii)
# torque-balance roots at discrete ω̂_E (Diss19 benchmark root ω₀ = −0.93
# ω_dia,e) [CHECKED: D23b Fig. 8; Diss19 Fig. 4.18].
#
# STATUS: SKIPPED — gated per docs/src/islands/QUESTIONS.md:
#   Q3  the frame-convention signs (src/Islands/frames/) are NaN-gated until
#       cleared — the ω_E-dependence of Δ_pol is exactly the physics the frames
#       module must own before any sign-bearing benchmark runs
#   Q2  D7 unratified (L0 equation set).

println("SKIPPED: B4 polarization ω_E structure — gated on QUESTIONS Q2, Q3")
println("         (frame-convention signs NaN-gated; L0 equation set unratified).")
