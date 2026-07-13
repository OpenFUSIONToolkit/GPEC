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
#   AUDIT-GATED (T4 — absolute numbers, NOT pass/fail without an input manifest;
#   see docs/src/islands/design/09-input-manifests.md):
#     B5a  w_c ~= 2.76 rho_theta_i (half) = 8.73 rho_bi at eps=0.1 [CHECKED: I19 Fig. 9]
#          NB: I19 is internally inconsistent on its own run collisionality
#              (S4.2 v_star=0.01 vs L23 p.82 v_star=1e-3) — the type specimen of
#              why absolute matches need the input-completeness audit first (docs/09).
#     B5b  w_c ~= 0.45 rho_theta_i = 1.46 rho_bi half-width [CHECKED: D21/D23a]
#     B5c  w_c ~= 0.440 rho_hat_theta_i + 0.0178 v_star - 7.54e-5 [CHECKED: L23 Eq. 6.3.2]
#          (best T4 candidate — a thesis documents its numerics most fully)
#
# STATUS: SKIPPED. The scaffold below is wired to the M2c Level-0 assembly
# (`Islands.Configure.configure_level0`): un-skipping is the ONE-LINE change
# `const UNGATED = true`. It stays gated because the assembly's remaining
# coefficient families are uncleared — parallel streaming, E×B, gradient drive,
# the quasineutrality (x-h) field source, the collision magnitude, the
# orbit-averaged pitch measure, and the neoclassical far field (QUESTIONS Q5).
# Until Q5 clears, `configure_level0` runs only STRUCTURALLY (placeholders), so
# no physics w_c can be extracted; flipping UNGATED before Q5 clears asserts-out.

using GeneralizedPerturbedEquilibrium
const Isl = GeneralizedPerturbedEquilibrium.Islands

"""
Flip to `true` only when QUESTIONS Q5 clears the remaining L0 coefficient families.
"""
const UNGATED = false

# The B5 physics parameter set (eps=0.1, m/n=2/1, tau=1; docs/09 I19/D21 manifest).
_b5_phys(variant) = Isl.Configure.Level0Physics(; epsilon=0.1, inv_Lq=1.0, inv_LB=1.0,
    q_s=2.0, dq_dpsi=0.5, w_psi=0.05, mu0_R=1.0, inv_Ln0=1.0, rho_hat_theta_i=0.05,
    eta_i=1.0, nu_star=0.01, m=2.0, tau=1.0, variant=variant)

# Assemble the :original and :improved configurations that the T2 toggle compares.
# Every Level-0 operator coefficient is now cleared (QUESTIONS Q5); the momentum-
# restoring collision term (F) remains a pending operator addition.
function _assemble_b5(variant)
    grid = Isl.PhaseSpace.IslandGrid(; nx=41, nxi=16, ny=17, nE=6, halfwidth_x=8.0,
        clustering_x=1.2, y_max=1.2, y_c=1.0, clustering_y=0.8, order=4)
    species = [Isl.SpeciesLists.Species(; name=:i, Z=1.0, m=1.0,
        background=Isl.SpeciesLists.Maxwellian(; n=1.0, T=1.0), role=Isl.SpeciesLists.Bulk)]
    return Isl.Configure.configure_level0(grid, _b5_phys(variant), species)
end

# threshold_width(cfg) — the marginal-island w_c from the MRE root dw/dt=0. GATED:
# needs the cleared far field + the (x-h) QN source to produce a physical Δ_neo(w)
# (QUESTIONS Q5). Placeholder assembly cannot yield a physics threshold.
function threshold_width(_cfg)
    error("threshold_width is gated on QUESTIONS Q5 (cleared far field + QN source).")
end

function run_b5()
    # T2 PRIMARY — the :original/:improved toggle differential (E1).
    w_orig = threshold_width(_assemble_b5(:original))
    w_impr = threshold_width(_assemble_b5(:improved))
    ratio = w_orig / w_impr
    println("B5b/E1 (T2) toggle differential: w_c(:original)/w_c(:improved) = ", ratio)
    println("  expect a ~x6 reduction (sources' 8.73 -> 1.46 rho_bi story).")
    # T3 PRIMARY — existence + v_star trend would follow here, over the Q5-cleared solve.
    return ratio
end

if UNGATED
    run_b5()
else
    println("SKIPPED: B5a/B5b/B5c — gated on QUESTIONS Q5 (and Q2/Q3/Q4 clearances).")
    println("  Scaffold wired to Islands.Configure.configure_level0; un-skip = `const UNGATED = true`.")
    println("  Primary tier when un-gated: B5b/E1 toggle differential (T2), threshold")
    println("  existence (T3), dw_c/dv_star trend (T3). Absolute w_c values are T4")
    println("  (audit-gated; docs/09 manifests + docs/05 'Target tiers').")
end
