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
interpolants (`kf_intr.dbob_m`, `kf_intr.divx_m`) remain unwired and are
tracked as follow-up work blocked on PR #196 — see the plan's "Out of scope"
section.

# Arguments

  - `ffs_ctrl`: ForceFreeStatesControl (carries `kinetic_factor`, `kinetic_source`)
  - `equil`: PlasmaEquilibrium with 2D interpolants and named profile/geometry splines
  - `ffs_intr`: ForceFreeStatesInternal (mode indexing)
  - `metric`: MetricData (provides ψ grid via `metric.xs`)
  - `mats`: MatrixSplines (used only for `numpert_total` cross-check)
  - `psis`: ψ grid for the kernel; empty (default) evaluates on `metric.xs`

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
    # Surfaces are taken `chunk` at a time to bound the bounce geometry held in memory. Each
    # chunk runs two greedy-scheduled passes — the (n, ℓ)-free surface setup and the (ψ, n)
    # bounce geometry, then every (ψ, n, ℓ) harmonic as its own work item — so expensive
    # surfaces (edge, resonance-dense) no longer pin one thread while the others idle.
    # Per-harmonic blocks are stored by index and reduced in fixed ℓ order, so the sums are
    # independent of thread count and schedule; the kernel only reads kf_intr.
    nl = kf_ctrl.bounce_harmonic_max
    nharm = 2 * nl + 1
    rtol_pitch = matrix_pitch_rtol(kf_ctrl)
    chunk = Threads.maxthreadid()

    # Species set summed into the kinetic matrices: the resolved multi-species set
    # (main ions + neutrality impurity + electrons) when provided, else a single species
    # from kf_ctrl. The kinetic W/torque matrices are additive over species, so we
    # accumulate (+=) each species' block-diagonal contribution.
    splist = species === nothing ?
             [Equilibrium.ResolvedNTVSpecies(kf_ctrl.ion_charge, kf_ctrl.ion_mass, kf_ctrl.electron, "single", kinetic_profiles)] : species


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

    # Surfaces inside the near-axis validity band are skipped (envelope 0).
    active = [ipsi for ipsi in 1:mpsi if env[ipsi] != 0.0]

    for sp in splist
        # Hoisted per-species scalars: the closures then capture concrete values, not a
        # union-typed `sp` (the ternary above mixes the resolved vector with the fallback).
        z_s, m_s, el_s, prof_s = sp.z, sp.m, sp.electron, sp.profiles
        for c0 in 1:chunk:length(active)
            idx = active[c0:min(c0 + chunk - 1, length(active))]
            nc = length(idx)

            states = Vector{Any}(undef, nc)
            geoms = Matrix{Any}(undef, npert, nc)
            Threads.@threads :greedy for j in 1:nc
                psi = xs[idx[j]]
                st = _setup_surface_state(psi, z_s, m_s, el_s, equil, kf_intr, prof_s)
                states[j] = st
                for in_idx in 1:npert
                    n = ffs_intr.nlow + in_idx - 1
                    geoms[in_idx, j] = _matrix_bounce_geometry(st, psi, n, kf_ctrl.magnetic_drift_factor, kf_intr)
                end
            end

            blocks_w = Array{Array{ComplexF64,3}}(undef, nharm, npert, nc)
            blocks_t = Array{Array{ComplexF64,3}}(undef, nharm, npert, nc)
            Threads.@threads :greedy for item in 1:(nc*npert*nharm)
                r, ell_idx = divrem(item - 1, nharm)
                j, in_idx = divrem(r, npert) .+ 1
                ell_idx += 1
                psi = xs[idx[j]]
                n = ffs_intr.nlow + in_idx - 1
                ell = ell_idx - 1 - nl
                block_w = zeros(ComplexF64, mpert, mpert, 6)
                block_t = zeros(ComplexF64, mpert, mpert, 6)
                compute_kinetic_matrices_at_psi!(
                    block_w, block_t, psi, n, ell,
                    z_s, m_s, kf_ctrl.magnetic_drift_factor, kf_ctrl.divxi_factor,
                    el_s, equil, kf_intr, prof_s;
                    nutype=kf_ctrl.nutype, f0type=kf_ctrl.f0type, nufac=kf_ctrl.collisionality_factor,
                    atol_xlmda=kf_ctrl.atol_pitch, rtol_xlmda=rtol_pitch,
                    atol_x=kf_ctrl.atol_x, rtol_x=kf_ctrl.rtol_x,
                    nested_tolerance_margin=kf_ctrl.nested_tolerance_margin,
                    state=states[j], geometry=geoms[in_idx, j]
                )
                blocks_w[ell_idx, in_idx, j] = block_w
                blocks_t[ell_idx, in_idx, j] = block_t
            end

            full_w = zeros(ComplexF64, mpert, mpert, 6)
            full_t = zeros(ComplexF64, mpert, mpert, 6)
            for j in 1:nc, in_idx in 1:npert
                ipsi = idx[j]
                # Reduce in fixed ℓ order (bit-identical to the serial accumulation).
                fill!(full_w, 0)
                fill!(full_t, 0)
                for ell_idx in 1:nharm
                    full_w .+= blocks_w[ell_idx, in_idx, j]
                    full_t .+= blocks_t[ell_idx, in_idx, j]
                end

                # Place the n-block on the diagonal of the full np×np matrix and accumulate.
                row_offset = (in_idx - 1) * mpert
                for k in 1:6, jj in 1:mpert, i in 1:mpert
                    col = (row_offset + jj - 1) * np + (row_offset + i)
                    kw_flat[ipsi, col, k] += env[ipsi] * full_w[i, jj, k]
                    kt_flat[ipsi, col, k] += env[ipsi] * full_t[i, jj, k]
                end
            end
        end
    end

    return kw_flat, kt_flat
end
