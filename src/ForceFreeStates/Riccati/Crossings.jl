# Singular-surface crossing algorithms for the Riccati/fundamental-matrix integration.

"""
    riccati_cross_ideal_singular_surf!(odet, ctrl, equil, mats, intr, ising)

Cross a singular surface on the Riccati path: the Riccati counterpart of `cross_ideal_singular_surf!`,
without Gaussian reduction (S is bounded) and zeroing the resonant columns `ipert_res` directly.

Expects `odet.u` in the bounded (U₁, U₂) form from `riccati_integrate_chunk!` with `needs_crossing=true`.
`ca_l`/`ca_r` are captured before `renormalize_riccati!` restores the (S, I) form, so the introduced
small solution has `ca_r[ipert_res, ipert_res, 2] = 1`.
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

Left (`sig=-1`) and right (`sig=+1`) singular asymptotics, each from its own resonant block.
"""
function _two_sided_singular_asymptotics(singp::SingType, ctrl::ForceFreeStatesControl,
    equil::Equilibrium.PlasmaEquilibrium, mats::MatrixSplines,
    intr::ForceFreeStatesInternal)
    sing_asymp_right = compute_sing_asymptotics(singp, ctrl, equil, mats, intr; sig=1.0)
    sing_asymp_left = compute_sing_asymptotics(singp, ctrl, equil, mats, intr; sig=-1.0)
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

# Trapezoidal predictor across the singular surface with the resonant columns zeroed (they stay zero).
function _predict_across_singular_surface!(odet::OdeState, ctrl::ForceFreeStatesControl,
    equil::Equilibrium.PlasmaEquilibrium, mats::MatrixSplines,
    intr::ForceFreeStatesInternal, ising::Int,
    ipert_res, dpsi::Float64, sing_asymp_right)
    if ctrl.kinetic === nothing
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

# Inject the right-side small asymptotic into the resonant columns, then capture odet.ca_r and singp.ua_right.
function _capture_right_crossing_data!(odet::OdeState, singp::SingType, sing_asymp_right,
    dpsi::Float64, intr::ForceFreeStatesInternal, ising::Int,
    ipert_res, ctrl::ForceFreeStatesControl)
    ua = sing_get_ua(sing_asymp_right, dpsi)
    singp.ua_right = copy(ua)
    singp.psi_ua_right = odet.psifac
    if ctrl.kinetic === nothing
        for i in eachindex(sing_asymp_right.r1)
            odet.u[ipert_res[i], :, :] .= 0
            odet.u[:, ipert_res[i], :] .= ua[:, ipert_res[i]+intr.numpert_total, :]
        end
    end
    odet.ca_r[:, :, :, ising] .= sing_get_ca(odet.u, ua, intr)
    odet.ca_populated = true
end

# STUB: per-surface ca-based Δ' (not physically valid; see SingType.delta_prime docstring).
# The canonical Δ' is intr.delta_prime_matrix from compute_delta_prime_matrix!.
function _stash_per_surface_delta_prime_stub!(odet::OdeState, intr::ForceFreeStatesInternal,
    ising::Int, ipert_res, sing_asymp_right,
    equil::Equilibrium.PlasmaEquilibrium,
    ctrl::ForceFreeStatesControl)
    ctrl.kinetic === nothing || return
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
