# benchmark_B4_polarization_omegaE.jl — ladder B4 (docs/05 §B)
#
# Targets are TIERED (Decision D9; docs/05 "Target tiers"):
#   PRIMARY (T3): (i) the Wilson-Connor collisionless and Smolyakov collisional
#                 SCALINGS; (ii) Delta_pol ~ omega_E^2 away from zero with a sign
#                 reversal EXISTING at an omega_E of order -omega_dia,e, reversal
#                 location insensitive to w/rho_theta_i; (iii) torque-balance
#                 roots (Delta_sin = 0) EXISTING at discrete omega_hat_E.
#   AUDIT-GATED (T4): the reversal LOCATION (sources: ~ -0.89 omega_dia,e) and
#                 the root VALUES (sources: +/-0.93, +/-1.28 omega_dia,e)
#                 [CHECKED: D23b Fig. 8; Diss19 Fig. 4.18].
#
# STATUS: SKIPPED — gated per docs/src/islands/QUESTIONS.md:
#   Q3  the frame-convention signs (src/Islands/frames/) are NaN-gated until
#       cleared — the omega_E-dependence of Delta_pol is exactly the physics the
#       frames module must own before any sign-bearing benchmark runs
#   Q2  D7 (L0 equation set) — clearance in progress (M2b)

println("SKIPPED: B4 polarization omega_E structure — gated on QUESTIONS Q2, Q3.")
println("  Primary tier when un-gated: omega_E^2 scaling + reversal/root EXISTENCE")
println("  (T3). The -0.89 reversal location and root values are T4 (audit-gated).")
