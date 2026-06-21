"""
    CalculatedKineticMatrices

Bridge from KineticForces' bounce-averaged matrix kernels into the kinetic-MHD
stability path in ForceFreeStates. Used by `make_kinetic_matrix` when
`ctrl.kinetic_source == "calculated"` (via the `calculated_source` callback
injected from `GeneralizedPerturbedEquilibrium.main`).
"""

"""
    compute_calculated_kinetic_matrices(ffs_ctrl, equil, ffs_intr, metric, ffit;
                                        kf_ctrl=KineticForcesControl(),
                                        kinetic_profiles)
        → (kw_flat, kt_flat)

Drive the KineticForces matrix kernel over the ψ grid stored in `metric.xs` and
return `(kw_flat, kt_flat)` arrays of shape `(mpsi, np^2, 6)` matching the
contract that `ForceFreeStates._compute_fkg_matrices!` consumes.

The arrays carry the six bounce-averaged kinetic energy / torque matrices
(Logan 2015 Eqs 7.30–7.35) for every ψ on the equilibrium grid, packed as
block-diagonal matrices over toroidal mode number n ∈ [nlow, nhigh] and
flattened to `(np = mpert·npert)²`.

This routine reads equilibrium-derived profiles (q, dV/dψ, ⟨r⟩, ⟨R⟩) directly
from named splines on `equil.profiles` and `equil.geometry`, and kinetic
profiles (n, T, ω_E, ν) from the `kinetic_profiles` argument, avoiding the
former shadow-copy pattern in KineticForcesInternal. The perturbation-mode
interpolants (`kf_intr.dbob_m`, `kf_intr.divx_m`) remain unwired and are
tracked as follow-up work blocked on PR #196 — see the plan's "Out of scope"
section.

# Arguments
- `ffs_ctrl`: ForceFreeStatesControl (carries `kinetic_factor`, `kinetic_source`)
- `equil`: PlasmaEquilibrium with 2D interpolants and named profile/geometry splines
- `ffs_intr`: ForceFreeStatesInternal (mode indexing)
- `metric`: MetricData (provides ψ grid via `metric.xs`)
- `ffit`: FourFitVars (used only for `numpert_total` cross-check)

# Keyword arguments
- `kf_ctrl`: KineticForcesControl, defaults to `KineticForcesControl()`. Used to
  carry NTV-specific knobs (nl, zi, mi, wdfac, divxfac, electron) that the
  KineticForces kernel needs but ForceFreeStatesControl does not expose.
- `kinetic_profiles::Equilibrium.KineticProfileSplines`: Required. Named kinetic-
  profile splines loaded via `Equilibrium.load_kinetic_profiles`.

# Returns
- `kw_flat::Array{ComplexF64,3}`: Energy matrices, shape `(mpsi, np^2, 6)`
- `kt_flat::Array{ComplexF64,3}`: Torque matrices, shape `(mpsi, np^2, 6)`
"""
function compute_calculated_kinetic_matrices(
    _ffs_ctrl,
    equil,
    ffs_intr,
    metric,
    ffit;
    kf_ctrl::KineticForcesControl = KineticForcesControl(),
    kinetic_profiles::Equilibrium.KineticProfileSplines,
)
    xs = metric.xs
    mpsi = length(xs)
    mpert = ffs_intr.mpert
    npert = ffs_intr.npert
    np = ffs_intr.numpert_total

    @assert ffit.numpert_total == np "FourFitVars and ForceFreeStatesInternal disagree on numpert_total"

    kw_flat = zeros(ComplexF64, mpsi, np^2, 6)
    kt_flat = zeros(ComplexF64, mpsi, np^2, 6)

    # Build the KineticForces internal state from equilibrium + ForceFreeStates mode indexing.
    kf_intr = KineticForcesInternal(equil; verbose=kf_ctrl.verbose)
    kf_intr.mlow = ffs_intr.mlow
    kf_intr.mhigh = ffs_intr.mhigh
    kf_intr.mpert = mpert
    kf_intr.nlow = ffs_intr.nlow
    kf_intr.nhigh = ffs_intr.nhigh
    kf_intr.npert = npert
    kf_intr.numpert_total = np
    kf_intr.mfac = collect(ffs_intr.mlow:ffs_intr.mhigh)

    # Geometric matrices for the kinetic W kernel come from ForceFreeStates' Fourier fit.
    geom_mats = ForceFreeStates.build_kinetic_metric_matrices(equil, ffs_intr, metric)
    kf_intr.smats = geom_mats.smats
    kf_intr.tmats = geom_mats.tmats
    kf_intr.xmats = geom_mats.xmats
    kf_intr.ymats = geom_mats.ymats
    kf_intr.zmats = geom_mats.zmats

    # Loop over flux surfaces and n-blocks, accumulating bounce harmonics into
    # per-n blocks placed on the diagonal of the full np×np block-diagonal matrix.
    # The ψ-loop is embarrassingly parallel: each iteration writes to a unique
    # ipsi row of kw_flat/kt_flat. Per-thread copies of kf_intr provide isolated
    # tpsi_* θ-grid buffers and interpolant hint refs; geometric/profile splines
    # are read-only and safely shared through deepcopy semantics.
    nl = kf_ctrl.nl
    nthreads = Threads.maxthreadid()
    thread_intrs   = [deepcopy(kf_intr) for _ in 1:nthreads]
    thread_full_w  = [zeros(ComplexF64, mpert, mpert, 6) for _ in 1:nthreads]
    thread_full_t  = [zeros(ComplexF64, mpert, mpert, 6) for _ in 1:nthreads]
    thread_block_w = [zeros(ComplexF64, mpert, mpert, 6) for _ in 1:nthreads]
    thread_block_t = [zeros(ComplexF64, mpert, mpert, 6) for _ in 1:nthreads]

    Threads.@threads for ipsi in 1:mpsi
        tid     = Threads.threadid()
        intr_t  = thread_intrs[tid]
        full_w  = thread_full_w[tid]
        full_t  = thread_full_t[tid]
        block_w = thread_block_w[tid]
        block_t = thread_block_t[tid]
        psi     = xs[ipsi]
        for in_idx in 1:npert
            n = ffs_intr.nlow + in_idx - 1
            fill!(full_w, 0)
            fill!(full_t, 0)
            for ell in -nl:nl
                fill!(block_w, 0)
                fill!(block_t, 0)
                compute_kinetic_matrices_at_psi!(
                    block_w, block_t, psi, n, ell,
                    kf_ctrl.zi, kf_ctrl.mi, kf_ctrl.wdfac, kf_ctrl.divxfac,
                    kf_ctrl.electron, equil, intr_t, kinetic_profiles;
                    nutype=kf_ctrl.nutype, f0type=kf_ctrl.f0type, nufac=kf_ctrl.nufac,
                    atol_xlmda=kf_ctrl.atol_xlmda, rtol_xlmda=kf_ctrl.rtol_xlmda
                )
                full_w .+= block_w
                full_t .+= block_t
            end

            # Place the n-block on the diagonal of the full np×np matrix and flatten.
            row_offset = (in_idx - 1) * mpert
            for k in 1:6, j in 1:mpert, i in 1:mpert
                idx = (row_offset + j - 1) * np + (row_offset + i)
                kw_flat[ipsi, idx, k] = full_w[i, j, k]
                kt_flat[ipsi, idx, k] = full_t[i, j, k]
            end
        end
    end

    return kw_flat, kt_flat
end
