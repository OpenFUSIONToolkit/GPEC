"""
    CalculatedKineticMatrices

Bridge from KineticForces' bounce-averaged matrix kernels into the kinetic-MHD
stability path in ForceFreeStates. Used by `build_kinetic_matrix_splines` when
`ctrl.kinetic_source == "calculated"` (via the `calculated_source` callback
injected from `GeneralizedPerturbedEquilibrium.main`).
"""

"""
    compute_calculated_kinetic_matrices(ffs_ctrl, equil, ffs_intr, metric, mats;
                                        kf_ctrl=KineticForcesControl(),
                                        kinetic_profiles)
        → (kw_flat, kt_flat)

Drive the KineticForces matrix kernel over the ψ grid stored in `metric.xs` and
return `(kw_flat, kt_flat)` arrays of shape `(mpsi, np^2, 6)` matching the
contract that `ForceFreeStates._compute_fkg_matrices` consumes.

The arrays carry the six bounce-averaged kinetic energy / torque matrices
(Logan 2015 Eqs 7.30–7.35) for every ψ on the equilibrium grid, packed as
block-diagonal matrices over toroidal mode number n ∈ [nlow, nhigh] and
flattened to `(np = mpert·npert)²`.

This routine reads equilibrium-derived profiles (q, dV/dψ, ⟨r⟩, ⟨R⟩) directly
from named splines on `equil.profiles` and `equil.geometry`, and kinetic
profiles (n, T, ω_E, ν) from the `kinetic_profiles` argument, avoiding the
former shadow-copy pattern in KineticForcesInternal. The perturbation-mode
interpolants (`kf_intr.dbob_m`, `kf_intr.divx_m`) are not wired into this routine.

# Arguments

  - `ffs_ctrl`: ForceFreeStatesControl (carries `kinetic_factor`, `kinetic_source`)
  - `equil`: PlasmaEquilibrium with 2D interpolants and named profile/geometry splines
  - `ffs_intr`: ForceFreeStatesInternal (mode indexing)
  - `metric`: MetricData (provides ψ grid via `metric.xs`)
  - `mats`: MatrixSplines (used only for `numpert_total` cross-check)
  - `psis`: ψ grid for the kernel; empty (default) evaluates on `metric.xs`

# Keyword arguments

  - `kf_ctrl`: KineticForcesControl, defaults to `KineticForcesControl()`. Used to
    carry NTV-specific knobs (nl, zi, mi, wdfac, electron) that the
    KineticForces kernel needs but ForceFreeStatesControl does not expose.
  - `kinetic_profiles::Equilibrium.KineticProfileSplines`: Required. Named kinetic-
    profile splines built by `Equilibrium.kinetic_profile_splines`.

# Returns

  - `kw_flat::Array{ComplexF64,3}`: Energy matrices, shape `(mpsi, np^2, 6)`
  - `kt_flat::Array{ComplexF64,3}`: Torque matrices, shape `(mpsi, np^2, 6)`
"""
function compute_calculated_kinetic_matrices(
    _ffs_ctrl,
    equil,
    ffs_intr,
    metric,
    mats;
    kf_ctrl::KineticForcesControl=KineticForcesControl(),
    kinetic_profiles::Equilibrium.KineticProfileSplines,
    species::Union{Nothing,AbstractVector{<:Equilibrium.ResolvedNTVSpecies}}=nothing,
    psis::Vector{Float64}=Float64[],
    axis_psi_c::Float64=0.0
)
    # The kernel is a pure function of psi (it evaluates equilibrium splines), so it can be
    # driven over any knot list; default is the full equilibrium grid.
    xs = isempty(psis) ? metric.xs : psis
    mpsi = length(xs)
    mpert = ffs_intr.mpert
    npert = ffs_intr.npert
    np = ffs_intr.numpert_total

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
    # The ψ-loop is embarrassingly parallel: each iteration writes its own ipsi row of
    # kw_flat/kt_flat, and the kernel only reads kf_intr, so it is shared across threads.
    nl = kf_ctrl.bounce_harmonic_max
    rtol_pitch = matrix_pitch_rtol(kf_ctrl)
    nthreads = Threads.maxthreadid()
    thread_full_w = [zeros(ComplexF64, mpert, mpert, 6) for _ in 1:nthreads]
    thread_full_t = [zeros(ComplexF64, mpert, mpert, 6) for _ in 1:nthreads]
    thread_block_w = [zeros(ComplexF64, mpert, mpert, 6) for _ in 1:nthreads]
    thread_block_t = [zeros(ComplexF64, mpert, mpert, 6) for _ in 1:nthreads]

    # Species set summed into the kinetic matrices: the resolved multi-species set
    # (main ions + neutrality impurity + electrons) when provided, else a single species
    # from the equilibrium's main ion. The kinetic W/torque matrices are additive over species, so we
    # accumulate (+=) each species' block-diagonal contribution.
    splist = species === nothing ?
             [Equilibrium.ResolvedNTVSpecies(main_ion(equil)..., kf_ctrl.electron, "single", kinetic_profiles)] : species


    # Near-axis validity envelope: suppresses the drift-kinetic increments where the zero-orbit-width
    # ordering fails (kernel evaluation skipped where it is 0). The same real scalar must scale kw and
    # kt at the same ψ — that is what keeps the blocks Hermitian and the energy/torque relation intact.
    env = ones(Float64, mpsi)
    if kf_ctrl.axis_validity_suppression
        psi_c = axis_psi_c
        if psi_c > 0
            env .= kinetic_axis_validity_envelope.(xs, psi_c)
            @info "Kinetic axis-validity suppression: psi_c=$(round(psi_c; sigdigits=3)), envelope reaches 1 at " *
                  "psi=$(round(2 * psi_c; sigdigits=3)) (kernel evaluation skipped below psi_c)" maxlog = 1
        end
    end

    for sp in splist
        # Hoisted per-species scalars: the closure then captures concrete values, not a
        # union-typed `sp` (the ternary above mixes the resolved vector with the fallback).
        z_s, m_s, el_s, prof_s = sp.z, sp.m, sp.electron, sp.profiles
        # :static pins one task per thread so the threadid() buffer indexing below is sound
        # (dynamic scheduling may migrate tasks at yield points, silently sharing buffers).
        Threads.@threads :static for ipsi in 1:mpsi
            tid = Threads.threadid()
            full_w = thread_full_w[tid]
            full_t = thread_full_t[tid]
            block_w = thread_block_w[tid]
            block_t = thread_block_t[tid]
            psi = xs[ipsi]
            env[ipsi] == 0.0 && continue   # inside the near-axis validity band: kernel skipped
            # (n, ℓ)-independent surface setup, built once per ψ on this thread's `intr` copy.
            state = _setup_surface_state(psi, z_s, m_s, el_s, equil, kf_intr, prof_s)
            for in_idx in 1:npert
                n = ffs_intr.nlow + in_idx - 1
                fill!(full_w, 0)
                fill!(full_t, 0)
                # The bounce geometry depends on (ψ, n) only; the harmonics share it.
                geometry = _matrix_bounce_geometry(state, psi, n, kf_ctrl.magnetic_drift_factor, kf_intr)
                for ell in -nl:nl
                    fill!(block_w, 0)
                    fill!(block_t, 0)
                    compute_kinetic_matrices_at_psi!(
                        block_w, block_t, psi, n, ell,
                        z_s, m_s, kf_ctrl.magnetic_drift_factor,
                        el_s, equil, kf_intr, prof_s;
                        nutype=kf_ctrl.nutype, f0type=kf_ctrl.f0type, nufac=kf_ctrl.collisionality_factor,
                        atol_xlmda=kf_ctrl.atol_pitch, rtol_xlmda=rtol_pitch,
                        atol_x=kf_ctrl.atol_x, rtol_x=kf_ctrl.rtol_x,
                        nested_tolerance_margin=kf_ctrl.nested_tolerance_margin,
                        state, geometry
                    )
                    full_w .+= block_w
                    full_t .+= block_t
                end

                # Place the n-block on the diagonal of the full np×np matrix and accumulate.
                row_offset = (in_idx - 1) * mpert
                for k in 1:6, j in 1:mpert, i in 1:mpert
                    idx = (row_offset + j - 1) * np + (row_offset + i)
                    kw_flat[ipsi, idx, k] += env[ipsi] * full_w[i, j, k]
                    kt_flat[ipsi, idx, k] += env[ipsi] * full_t[i, j, k]
                end
            end
        end
    end

    return kw_flat, kt_flat
end
