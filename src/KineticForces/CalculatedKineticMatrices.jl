"""
    CalculatedKineticMatrices

Bridge from KineticForces' bounce-averaged matrix kernels into the kinetic-MHD
stability path in ForceFreeStates. Used by `make_kinetic_matrix` when
`ctrl.kinetic_source == "calculated"` (via the `calculated_source` callback
injected from `GeneralizedPerturbedEquilibrium.main`).
"""

"""
    compute_calculated_kinetic_matrices(ffs_ctrl, equil, ffs_intr, metric, ffit;
                                        kf_ctrl=KineticForcesControl())
        → (kw_flat, kt_flat)

Drive the KineticForces matrix kernel over the ψ grid stored in `metric.xs` and
return `(kw_flat, kt_flat)` arrays of shape `(mpsi, np^2, 6)` matching the
contract that `ForceFreeStates._compute_fkg_matrices!` consumes.

The arrays carry the six bounce-averaged kinetic energy / torque matrices
(Logan 2015 Eqs 7.30–7.35) for every ψ on the equilibrium grid, packed as
block-diagonal matrices over toroidal mode number n ∈ [nlow, nhigh] and
flattened to `(np = mpert·npert)²`.

The current implementation returns zero matrices and emits a warning: the
plumbing for `kf_intr.sq / kin / geom` (kinetic profile interpolants pulled
from the equilibrium and `kinetic.dat`) and the `dbob_m / divx_m` perturbation
modes is tracked as follow-up work — see plan
`.claude/plans/validated-exploring-clarke.md` "Out of scope" section. Even
though the returned matrices are zero, this routine still wires the dispatch
path end-to-end so that `sing_der!`, `_compute_fkg_matrices!`, and the
single-chunk integration in `EulerLagrange.jl` exercise the kinetic code path.

# Arguments
- `ffs_ctrl`: ForceFreeStatesControl (carries `kinetic_factor`, `kinetic_source`)
- `equil`: PlasmaEquilibrium with 2D interpolants
- `ffs_intr`: ForceFreeStatesInternal (mode indexing)
- `metric`: MetricData (provides ψ grid via `metric.xs`)
- `ffit`: FourFitVars (used only for `numpert_total` cross-check)

# Keyword arguments
- `kf_ctrl`: KineticForcesControl, defaults to `KineticForcesControl()`. Used to
  carry NTV-specific knobs (nl, zi, mi, wdfac, divxfac, electron) that the
  KineticForces kernel needs but ForceFreeStatesControl does not expose.

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

    # Population of kf_intr.sq / kf_intr.kin / kf_intr.geom (q-profile, kinetic
    # profiles, geometric profiles) and the perturbation modes kf_intr.dbob_m /
    # kf_intr.divx_m is not yet wired (see plan "Out of scope"). Without these,
    # `tpsi!` cannot evaluate bounce frequencies or the resonance kernel, so we
    # short-circuit to zeros and warn loudly. The dispatch path through
    # `make_kinetic_matrix → _compute_fkg_matrices! → sing_der!` is still
    # exercised end-to-end with these zero-valued kinetic matrices.
    if isnothing(kf_intr.sq) || isnothing(kf_intr.kin) || isnothing(kf_intr.geom)
        @warn "compute_calculated_kinetic_matrices: kinetic profile interpolants " *
              "(sq/kin/geom) are not populated yet — returning zero kinetic matrices. " *
              "The dispatcher and FKG/sing_der! plumbing still runs, but the kinetic " *
              "physics contribution is null until profile wiring is implemented." maxlog=1
        return kw_flat, kt_flat
    end

    # Once profiles are wired in, the loop below will populate kw_flat / kt_flat
    # for each (ψ, n, ℓ) by accumulating bounce harmonics into per-n blocks.
    nl = kf_ctrl.nl
    full_w = zeros(ComplexF64, mpert, mpert, 6)
    full_t = zeros(ComplexF64, mpert, mpert, 6)
    block_w = zeros(ComplexF64, mpert, mpert, 6)
    block_t = zeros(ComplexF64, mpert, mpert, 6)

    for ipsi in 1:mpsi
        psi = xs[ipsi]
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
                    kf_ctrl.electron, equil, kf_intr,
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
