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
# STATUS: SKIPPED. Wired to the M2c assembly (Islands.Configure.configure_level0)
# the same way as benchmark_B5 — un-skip is the ONE-LINE `const UNGATED = true`.
# Gated per docs/src/islands/QUESTIONS.md:
#   Q5  the remaining L0 coefficient families + the QN (x-h) field source are
#       uncleared, so configure_level0 runs only STRUCTURALLY (no physics Δ(w))
#   Q3/Q4  the Δ moment prefactors are now CLEARED (M2b); the electron-closure
#       constants and ψ̃ [VERIFY] are folded into the Q5 lane

using GeneralizedPerturbedEquilibrium
const Isl = GeneralizedPerturbedEquilibrium.Islands

"""
Flip to `true` only when QUESTIONS Q5 clears the remaining L0 coefficient families.
"""
const UNGATED = false

# The large-w Δ(w) sweep would assemble one configure_level0 per island width w
# (varying w_psi in Level0Physics) and fit the 1/w, 1/w^3 exponents (T3). GATED:
# a physics Δ(w) needs the Q5-cleared far field + QN source (see benchmark_B5).
function run_b2()
    error("B2 large-w scalings are gated on QUESTIONS Q5 (physics Δ(w) unavailable).")
end

if UNGATED
    run_b2()
else
    println("SKIPPED: B2 large-w limits — gated on QUESTIONS Q5 (Δ moment prefactors CLEARED).")
    println("  Scaffold wired to configure_level0; un-skip = `const UNGATED = true`.")
    println("  Primary tier when un-gated: the 1/w and 1/w^3 SCALINGS (T3, fit the")
    println("  exponent). The WCHH96 Eq. 85 coefficient is T4 (audit-gated).")
end
