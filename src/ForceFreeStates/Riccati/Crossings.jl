# Singular-surface crossing algorithms for the Riccati/fundamental-matrix integration.

"""
    riccati_cross_ideal_singular_surf!(odet, ctrl, equil, mats, intr, ising)

Cross a singular surface for the Riccati formulation. Replaces `cross_ideal_singular_surf!`
for the Riccati integration path with two key differences:

1. **No Gaussian reduction**: `cross_ideal_singular_surf!` calls `compute_solution_norms!`
   which applies Gaussian reduction to (S, I). This divides by pivot elements of S, which
   can be near-zero (S = 0 at axis and grows slowly), producing NaN/Inf in U₂. For Riccati,
   S is bounded so Gaussian reduction is unnecessary.

2. **Direct column zeroing**: Instead of using the GR-sorted `odet.index` to identify the
   column to zero, we use `ipert_res` directly (the resonant mode index). This is valid since
   without GR there is no permutation applied to the columns of S.

**Δ' normalization**: This function expects `odet.u` in the bounded (U₁, U₂) form produced by
`riccati_integrate_chunk!` with `needs_crossing=true` (final renorm skipped). ca_l is computed
from (U₁, U₂) before the crossing, and ca_r from (U₁_new, U₂_new) before `renormalize_riccati!`.
Since column `ipert_res` of [U₁_new; U₂_new] equals the introduced asymptotic solution exactly,
ca_r[ipert_res,ipert_res,2] = 1 regardless of other column normalizations. This gives a
physically meaningful Δ' = ca_r - ca_l with consistent left/right normalization.

After the predictor step and asymptotic introduction, `renormalize_riccati!` is called
to restore the canonical (S_new, I) form before continuing integration.

The u_store entry at the crossing step correctly stores (U₁_new, U₂_new) so that
`evaluate_stability_criterion!` can compute U₁_new / U₂_new = S_new correctly.
"""
function riccati_cross_ideal_singular_surf!(
    odet::OdeState, ctrl::ForceFreeStatesControl, equil::Equilibrium.PlasmaEquilibrium,
    mats::MatrixSplines, intr::ForceFreeStatesInternal, ising::Int
)
    # Skip Gaussian reduction — S is bounded so no large-norm columns exist.
    singp = intr.sing[ising]
    dpsi = singp.psifac - odet.psifac  # ψ_res - ψ_current (positive)
    ipert_res = 1 .+ singp.m .- intr.mlow .+ (singp.n .- intr.nlow) .* intr.mpert

    sing_asymp_left, sing_asymp_right = _two_sided_singular_asymptotics(singp, ctrl, equil, mats, intr)
    _log_riccati_crossing_diagnostics(odet, intr, ising, singp, dpsi, sing_asymp_left, sing_asymp_right)

    _capture_left_crossing_data!(odet, singp, sing_asymp_left, dpsi, intr, ising)
    _predict_across_singular_surface!(odet, ctrl, equil, mats, intr, ising, ipert_res, dpsi, sing_asymp_right)
    _capture_right_crossing_data!(odet, singp, sing_asymp_right, dpsi, intr, ising, ipert_res, ctrl)

    _stash_per_surface_delta_prime_stub!(odet, intr, ising, ipert_res, sing_asymp_right, equil, ctrl)
    _store_crossing_step!(odet)

    # Restore canonical (S_new, I) form before continuing integration.
    renormalize_riccati!(odet, intr)
end

"""
    _two_sided_singular_asymptotics(singp, ctrl, equil, mats, intr) -> (left, right)

Compute left- (`sig=-1`) and right- (`sig=+1`) side singular asymptotics matching
Fortran STRIDE's separate vmatl/vmatr (sing_vmat). Alpha is taken from the right
side and shared with the left.
"""
function _two_sided_singular_asymptotics(singp::SingType, ctrl::ForceFreeStatesControl,
                                         equil::Equilibrium.PlasmaEquilibrium, mats::MatrixSplines,
                                         intr::ForceFreeStatesInternal)
    sing_asymp_right = compute_sing_asymptotics(singp, ctrl, equil, mats, intr; sig=1.0)
    sing_asymp_left  = compute_sing_asymptotics(singp, ctrl, equil, mats, intr; sig=-1.0,
                                                alpha_override=sing_asymp_right.alpha)
    return sing_asymp_left, sing_asymp_right
end

# @debug-only per-crossing diagnostics. Enable via JULIA_DEBUG=GeneralizedPerturbedEquilibrium.
function _log_riccati_crossing_diagnostics(odet, intr, ising, singp, dpsi, sing_asymp_left, sing_asymp_right)
    @debug begin
        ipert_res_diag = 1 .+ singp.m .- intr.mlow .+ (singp.n .- intr.nlow) .* intr.mpert
        msg = "  ising=$ising: psi_sing=$(@sprintf("%.10f", singp.psifac)), psi_eval=$(@sprintf("%.10f", odet.psifac)), dpsi=$(@sprintf("%.10e", dpsi))\n"
        msg *= "  alpha_L = $(sing_asymp_left.alpha), alpha_R = $(sing_asymp_right.alpha)\n"
        for ip in ipert_res_diag
            msg *= "  vmatL[0] big: vmat[$ip,$ip,1,1]=$(@sprintf("%.8e", real(sing_asymp_left.vmat[ip,ip,1,1]))), vmat[$ip,$ip,2,1]=$(@sprintf("%.8e", real(sing_asymp_left.vmat[ip,ip,2,1])))\n"
            msg *= "  vmatR[0] big: vmat[$ip,$ip,1,1]=$(@sprintf("%.8e", real(sing_asymp_right.vmat[ip,ip,1,1]))), vmat[$ip,$ip,2,1]=$(@sprintf("%.8e", real(sing_asymp_right.vmat[ip,ip,2,1])))\n"
        end
        msg
    end
end

# Capture left-side asymptotic data into odet.ca_l and singp.ua_left/psi_ua_left.
function _capture_left_crossing_data!(odet::OdeState, singp::SingType, sing_asymp_left,
                                      dpsi::Float64, intr::ForceFreeStatesInternal, ising::Int)
    ua = sing_get_ua(sing_asymp_left, dpsi)
    singp.ua_left = copy(ua)
    singp.psi_ua_left = odet.psifac
    odet.ca_l[:, :, :, ising] .= sing_get_ca(odet.u, ua, intr)
end

# Trapezoidal predictor across the singular surface: zero the resonant columns,
# evaluate sing_der! on both sides, advance odet by (du1 + du2)·dpsi, and jump
# odet.psifac to the right side. The zeroed columns stay zero through the predictor
# since du[:, ipert_res, :] = 0 when u[:, ipert_res, :] = 0.
function _predict_across_singular_surface!(odet::OdeState, ctrl::ForceFreeStatesControl,
                                           equil::Equilibrium.PlasmaEquilibrium, mats::MatrixSplines,
                                           intr::ForceFreeStatesInternal, ising::Int,
                                           ipert_res, dpsi::Float64, sing_asymp_right)
    if ctrl.kinetic_factor == 0
        for i in eachindex(sing_asymp_right.r1)
            odet.u[:, ipert_res[i], :] .= 0
        end
    end
    params = (ctrl, equil, mats, intr, odet, IntegrationChunk(0.0, 0.0, false, ising, 1))
    du1 = zeros(ComplexF64, intr.numpert_total, intr.numpert_total, 2)
    du2 = zeros(ComplexF64, intr.numpert_total, intr.numpert_total, 2)
    sing_der!(du1, odet.u, params, odet.psifac)
    odet.psifac += 2 * dpsi  # jump to other side of singular surface
    sing_der!(du2, odet.u, params, odet.psifac)
    odet.u .+= (du1 .+ du2) .* dpsi
end

# Inject the right-side small asymptotic into the resonant columns of (U₁_new, U₂_new),
# capture odet.ca_r, and save singp.ua_right / psi_ua_right.
# Column ipert_res of [U₁_new; U₂_new] = ua[:, ipert_res+N, :] (the introduced small asymptotic),
# so ca_r[ipert_res, ipert_res, 2] = 1 regardless of other columns' normalization.
function _capture_right_crossing_data!(odet::OdeState, singp::SingType, sing_asymp_right,
                                       dpsi::Float64, intr::ForceFreeStatesInternal, ising::Int,
                                       ipert_res, ctrl::ForceFreeStatesControl)
    ua = sing_get_ua(sing_asymp_right, dpsi)
    singp.ua_right = copy(ua)
    singp.psi_ua_right = odet.psifac
    if ctrl.kinetic_factor == 0
        for i in eachindex(sing_asymp_right.r1)
            odet.u[ipert_res[i], :, :] .= 0
            odet.u[:, ipert_res[i], :] .= ua[:, ipert_res[i]+intr.numpert_total, :]
        end
    end
    odet.ca_r[:, :, :, ising] .= sing_get_ca(odet.u, ua, intr)
end

# STUB: per-surface ca-based Δ' (not physically valid; see SingType.delta_prime docstring).
# The canonical Δ' is intr.delta_prime_matrix from compute_delta_prime_matrix!.
function _stash_per_surface_delta_prime_stub!(odet::OdeState, intr::ForceFreeStatesInternal,
                                              ising::Int, ipert_res, sing_asymp_right,
                                              equil::Equilibrium.PlasmaEquilibrium,
                                              ctrl::ForceFreeStatesControl)
    ctrl.kinetic_factor == 0 || return
    denom = (2π)^2 * equil.psio
    n_res = length(sing_asymp_right.r1)
    N = intr.numpert_total
    resize!(intr.sing[ising].delta_prime, n_res)
    intr.sing[ising].delta_prime_col = zeros(ComplexF64, N, n_res)
    for i in eachindex(sing_asymp_right.r1)
        Δca_col = (odet.ca_r[:, ipert_res[i], 2, ising] - odet.ca_l[:, ipert_res[i], 2, ising]) / denom
        intr.sing[ising].delta_prime_col[:, i] .= Δca_col
        intr.sing[ising].delta_prime[i] = Δca_col[ipert_res[i]]
    end
end

# Store (U₁_new, U₂_new) into u_store before renormalization so that
# evaluate_stability_criterion! can recover S_new = U₁_new / U₂_new via compute_smallest_eigenvalue.
function _store_crossing_step!(odet::OdeState)
    store_ode_data!(odet, odet.psifac, odet.u)
end
