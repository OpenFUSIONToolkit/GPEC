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
# STATUS: SKIPPED. Wired to the M2c assembly (Islands.Configure.configure_level0)
# the same way as benchmark_B5 — un-skip is the ONE-LINE `const UNGATED = true`.
# Gated per docs/src/islands/QUESTIONS.md:
#   Q3  the frame-convention signs (src/Islands/frames/) are NaN-gated until
#       cleared — the omega_E-dependence of Delta_pol is exactly the physics the
#       frames module must own before any sign-bearing benchmark runs
#   Q5  the L0 assembly's gradient-drive (which carries the omega_E frame shift)
#       and the QN (x-h) source are uncleared, so configure_level0 runs only
#       STRUCTURALLY (no physics Delta_pol(omega_E))

using GeneralizedPerturbedEquilibrium
const Isl = GeneralizedPerturbedEquilibrium.Islands

"""
Flip to `true` only when QUESTIONS Q3 (frame signs) and Q5 clear.
"""
const UNGATED = false

# The Delta_pol(omega_E) scan would assemble configure_level0 across an omega_E
# grid (through the gradient-drive frame shift) and check the omega_E^2 scaling +
# reversal/root existence (T3). GATED: needs the cleared frames convention (Q3)
# and the Q5 gradient-drive/QN-source (see benchmark_B5).
function run_b4()
    error("B4 polarization structure is gated on QUESTIONS Q3 (frame signs) and Q5.")
end

if UNGATED
    run_b4()
else
    println("SKIPPED: B4 polarization omega_E structure — gated on QUESTIONS Q3, Q5.")
    println("  Scaffold wired to configure_level0; un-skip = `const UNGATED = true`.")
    println("  Primary tier when un-gated: omega_E^2 scaling + reversal/root EXISTENCE")
    println("  (T3). The -0.89 reversal location and root values are T4 (audit-gated).")
end
