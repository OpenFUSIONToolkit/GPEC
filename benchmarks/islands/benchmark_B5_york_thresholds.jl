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
# STATUS: SKIPPED. The scaffold is wired to the M2c Level-0 assembly
# (`Islands.Configure.configure_level0`) with a PHYSICAL, DERIVED parameter vector —
# `scenario_from_equilibrium` on examples/DIIID-like_ideal_example at its q=2 surface
# (design 10), no longer hand-set. Un-skipping is the ONE-LINE change
# `const UNGATED = true`. It stays gated because (a) the assembly's remaining
# coefficient families are uncleared — parallel streaming, E×B, gradient drive,
# the quasineutrality (x-h) field source, the collision magnitude, the
# orbit-averaged pitch measure, and the neoclassical far field (QUESTIONS Q5) — and
# (b) the far-field extraction does not localize (QUESTIONS Q7). Until both clear,
# `configure_level0` runs only STRUCTURALLY (placeholders), so no physics w_c can be
# extracted; flipping UNGATED before then asserts-out.

using GeneralizedPerturbedEquilibrium
const Isl = GeneralizedPerturbedEquilibrium.Islands
const Eq = GeneralizedPerturbedEquilibrium.Equilibrium
import TOML

"""
Flip to `true` only when QUESTIONS Q5 clears the remaining L0 coefficient families.
"""
const UNGATED = false

# The pinned physical scenario is DERIVED from a solved equilibrium (design 10),
# not hand-set: examples/DIIID-like_ideal_example at its q=2 H-mode surface yields a
# self-consistent, low-collisionality banana vector (ε≈0.265, ρ̂_θi≈0.075, ν_★≈0.012,
# η_i≈2.16). Loaded once and cached; `w_psi` (island half-width) is the scan variable.
const _B5_EXAMPLE = joinpath(@__DIR__, "..", "..", "examples", "DIIID-like_ideal_example")
const _B5_EQUIL = Ref{Any}(nothing)
const _B5_KP = Ref{Any}(nothing)

function _b5_load_equilibrium()
    if _B5_EQUIL[] === nothing
        inputs = TOML.parsefile(joinpath(_B5_EXAMPLE, "gpec.toml"))
        _B5_EQUIL[] = Eq.setup_equilibrium(Eq.EquilibriumConfig(inputs["Equilibrium"], _B5_EXAMPLE))
        _B5_KP[] = Eq.load_kinetic_profiles(joinpath(_B5_EXAMPLE, "TkMkr_D3Dlike_Hmode_kinetic.h5"); zi=1, mi=2)
    end
    return _B5_EQUIL[], _B5_KP[]
end

# The DERIVED, self-consistent Level-0 vector at the physical q=2 surface (design 10).
function _b5_phys(variant; w_psi::Real=0.05)
    equil, kp = _b5_load_equilibrium()
    return Isl.Configure.scenario_from_equilibrium(equil, kp; m=2, n=1, w_psi=w_psi,
        Z=1.0, mass_amu=2.0, tau=1.0, variant=variant)
end

# Assemble the :original and :improved configurations that the T2 toggle compares.
# Every Level-0 operator coefficient is now cleared (QUESTIONS Q5); the momentum-
# restoring collision term (F) remains a pending operator addition.
function _assemble_b5(variant; w_psi::Real=0.05)
    phys = _b5_phys(variant; w_psi=w_psi)
    # PHYSICAL local domain: x=(ψ−ψ_s)/ψ_s, so the axis is at x=−1. The matching radius
    # is a fixed physical fraction of the surface (design 10's physical_domain), NOT
    # scaled with w — Δ_neo must stay finite as w→0. y_max spans the full trapped+
    # forbidden pitch range (1/b_min=(1+ε)/(1−ε)).
    Lx = Isl.Configure.physical_domain(phys; x_match=0.2)
    y_max = (1 + phys.epsilon) / (1 - phys.epsilon)
    grid = Isl.PhaseSpace.resolved_island_grid(; w=w_psi, nx=41, K=6, Lx_over_w=Lx / w_psi,
        nxi=16, ny=17, nE=6, y_max=y_max, y_c=1.0, clustering_y=0.8, order=4)
    species = [Isl.SpeciesLists.Species(; name=:i, Z=1.0, m=1.0,
        background=Isl.SpeciesLists.Maxwellian(; n=1.0, T=1.0), role=Isl.SpeciesLists.Bulk)]
    return Isl.Configure.configure_level0(grid, phys, species)
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
