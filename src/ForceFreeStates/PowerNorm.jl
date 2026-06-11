"""
Power-normalized flux eigenvalue computation for the FFS edge scan.

Transforms the energy matrix W from ξ-space into power-normalized flux Φ-space and
returns the eigenvalues of `W_Φ = M†·W·M`. The **eigenvalues** (energies) are
coordinate-invariant: energy is a physically meaningful scalar and does not depend
on the choice of straight-field-line coordinate.

    dW = ξ†·W·ξ = Φ†·M†·W·M·Φ

Transformation chain:
  - T = diag(i·2π·χ₁·singfac) converts ξ → flux (χ₁ = 2π·ψ₀, singfac = m − n·q).
  - ptof = sqrtamat·√jarea converts the power-normalized field → flux.
  - M = T⁻¹·ptof maps Φ → ξ.

What is **NOT** invariant across Jacobian choices (see
`scripts/test_power_norm_invariance.jl` for numerical proof):
  - ξ itself (W is Jacobian-dependent).
  - Φ itself — T depends on the m-labelling of the chosen Jacobian, so
    ‖Φ‖ = ‖M⁻¹ξ‖ drifts with Jacobian.
  - The θ-space field reconstructed from Φ.

What **IS** invariant (verified numerically in the test script, within the
area-integral precision floor ~1e-6):
  - Flux-surface area A = ∫ J|∇ψ| dθ.
  - The √weight operator identity ‖sqrtamat·b_fft‖² = N²·∫|b|²·J|∇ψ| dθ.
  - The angle-map convmat (b_jac2 = convmat·b_jac1) to machine precision.
  - The eigenspectrum of `W_Φ`.
"""

# The √weight building blocks `compute_sqrt_jac_delpsi`, `compute_sqrtamat`, and the
# `control_surface_ptof` operator now live in `Equilibrium/CoordinateInvariant.jl` so the
# ForceFreeStates and PerturbedEquilibrium modules share one coordinate-invariant definition.

"""
    compute_power_norm_eigenvalues(wt, wp, wv, sqrtamat, jarea, equil, psi, intr; all_eigenvalues=false)

Transform W from ξ-space to power-normalized flux Φ-space and eigendecompose.

The transformation chain:
  1. T = diag(i·2π·χ₁·singfac) converts ξ → flux (singfac = m − n·q).
  2. ptof = sqrtamat·√jarea converts the power-normalized field → flux.
  3. M = T⁻¹·ptof maps Φ → ξ (power-norm to displacement).
  4. W_Φ = M†·W·M = ptof†·T⁻†·W·T⁻¹·ptof.

Only the eigenspectrum of W_Φ is physically invariant across Jacobian choices —
it is the coordinate-independent energy. The eigenvectors in this basis are
normalized under Julia's `eigen` (‖v‖₂ = 1) rather than the ∫|b|²dA = 1
convention, so `v` should not be reused as a physical Φ-space mode shape — here
it is only used to split λ = v†Wt v into ⟨v,Wp v⟩ + ⟨v,Wv v⟩, both of which are
spectrally invariant because Wp + Wv = Wt.

Returns the standard eigenvalues of W_Φ sorted ascending (most negative/unstable
first), matching the ξ-space convention in `free_run!` and `free_compute_total`.
Also returns `pn_vacuum_eigenvalue = max(0, λ_min(Hermitian(wv_Φ)))` — the Φ-space
counterpart of the ξ-space `vacuum_eigenvalue` diagnostic.
Returns NaN when any singfac ≈ 0 (rational surface crossing makes T singular).
"""
function compute_power_norm_eigenvalues(
    wt::AbstractMatrix{ComplexF64},
    wp::AbstractMatrix{ComplexF64},
    wv::AbstractMatrix{ComplexF64},
    sqrtamat::AbstractMatrix{ComplexF64},
    jarea::Float64,
    equil::Equilibrium.PlasmaEquilibrium,
    psi::Float64,
    intr::ForceFreeStatesInternal;
    all_eigenvalues::Bool=false
)
    Npert = intr.numpert_total
    mpert = intr.mpert
    npert = intr.npert
    chi1 = 2π * equil.psio

    # Compute singfac = m - n·q for each mode
    q_at_psi = equil.profiles.q_spline(psi)
    singfac = vec((intr.mlow:intr.mhigh) .- q_at_psi .* (intr.nlow:intr.nhigh)')

    # Check for rational surface proximity — T is singular there
    if any(abs.(singfac) .< 1e-6)
        if all_eigenvalues
            nan_vec = fill(complex(NaN), Npert)
            nan_mat = fill(complex(NaN), Npert, Npert)
            return (pn_total_eigenvalue=complex(NaN), pn_plasma_energy=complex(NaN), pn_vacuum_energy=complex(NaN),
                pn_vacuum_eigenvalue=NaN, pn_et_all=nan_vec, pn_ep_all=nan_vec, pn_ev_all=nan_vec,
                wt_pn=nan_mat, wp_pn=nan_mat, wv_pn=nan_mat, pn_eigenvectors=nan_mat)
        else
            return (pn_total_eigenvalue=complex(NaN), pn_plasma_energy=complex(NaN), pn_vacuum_energy=complex(NaN),
                pn_vacuum_eigenvalue=NaN)
        end
    end

    # T = diag(i·2π·chi1·singfac): ξ → flux
    T_diag_inv = Vector{ComplexF64}(undef, Npert)
    for i in 1:Npert
        T_diag_inv[i] = 1.0 / (im * 2π * chi1 * singfac[i])
    end

    # ptof = sqrtamat * sqrt(jarea): power-norm field → flux
    # For multi-n, expand mpert×mpert sqrtamat to Npert×Npert block-diagonal
    ptof_block = sqrtamat .* sqrt(jarea)
    if npert == 1
        ptof_full = ptof_block
    else
        ptof_full = zeros(ComplexF64, Npert, Npert)
        for in in 1:npert
            r = ((in - 1) * mpert + 1):(in * mpert)
            ptof_full[r, r] .= ptof_block
        end
    end

    # M = diag(T_inv) · ptof: row-scale ptof by T_inv (maps Φ → ξ)
    M = similar(ptof_full)
    for i in 1:Npert
        M[i, :] .= T_diag_inv[i] .* ptof_full[i, :]
    end

    # Transform energy matrices: W_Φ = M† · W · M
    wt_pn = M' * wt * M
    wp_pn = M' * wp * M
    wv_pn = M' * wv * M

    # Smallest eigenvalue of the vacuum matrix alone in Φ-space, clamped to zero.
    # wv_pn should be PSD by congruence of the PSD ξ-space wv; numerical noise
    # can make the smallest eigenvalue slightly negative.
    pn_vacuum_eigenvalue = real(max(0.0, minimum(real.(eigvals(Hermitian(wv_pn))))))

    # Eigendecompose the total energy in power-norm space. Only the eigenspectrum
    # is Jacobian-invariant; the eigenvectors are coordinate-dependent.
    Ev = eigen(wt_pn)
    eigenvalues = Ev.values
    vectors = Ev.vectors

    # Sort ascending by real part (most negative/most unstable first) — matches ξ-space convention in free_run!/free_compute_total
    eindex = sortperm(real.(eigenvalues); rev=true)

    # λ = v†Wt v = v†Wp v + v†Wv v ; the plasma/vacuum split is the projection
    # onto the same eigenvector and is therefore also Jacobian-invariant.
    if all_eigenvalues
        pn_et_all = Vector{ComplexF64}(undef, Npert)
        pn_ep_all = Vector{ComplexF64}(undef, Npert)
        pn_ev_all = Vector{ComplexF64}(undef, Npert)
        pn_eigenvectors = Matrix{ComplexF64}(undef, Npert, Npert)

        for i in 1:Npert
            v = vectors[:, eindex[Npert+1-i]]
            pn_eigenvectors[:, i] .= v
            pn_et_all[i] = eigenvalues[eindex[Npert+1-i]]
            pn_ep_all[i] = dot(v, wp_pn * v)
            pn_ev_all[i] = dot(v, wv_pn * v)
        end

        # Phase convention: rotate each column so its largest-magnitude entry is real-positive
        # (matches the ξ-space convention in free_run!). Magnitudes are preserved since the
        # eigenvectors come from Julia's `eigen` with ‖v‖₂ = 1, the natural Φ-space norm.
        for isol in 1:Npert
            imax = argmax(abs.(@view pn_eigenvectors[:, isol]))
            phase = abs(pn_eigenvectors[imax, isol]) / pn_eigenvectors[imax, isol]
            @view(pn_eigenvectors[:, isol]) .*= phase
        end

        return (pn_total_eigenvalue=pn_et_all[1], pn_plasma_energy=pn_ep_all[1], pn_vacuum_energy=pn_ev_all[1],
            pn_vacuum_eigenvalue=pn_vacuum_eigenvalue,
            pn_et_all=pn_et_all, pn_ep_all=pn_ep_all, pn_ev_all=pn_ev_all,
            wt_pn=wt_pn, wp_pn=wp_pn, wv_pn=wv_pn, pn_eigenvectors=pn_eigenvectors)
    else
        idx = eindex[Npert]
        v = vectors[:, idx]
        pn_total = eigenvalues[idx]
        pn_plasma = ComplexF64(dot(v, wp_pn * v))
        pn_vacuum = ComplexF64(dot(v, wv_pn * v))

        return (pn_total_eigenvalue=pn_total, pn_plasma_energy=pn_plasma, pn_vacuum_energy=pn_vacuum,
            pn_vacuum_eigenvalue=pn_vacuum_eigenvalue)
    end
end

"""
    free_compute_sqrtamat_spline(ctrl, equil, intr) -> CubicSeriesInterpolant

Pre-compute sqrtamat and jarea over the edge scan ψ range and return a cubic series
interpolant. Uses the same q-evenly-spaced grid as `free_compute_wv_spline`.

The interpolant stores `mpert^2 + 1` complex series per grid point:
  - First mpert^2 values: flattened sqrtamat matrix
  - Last value: jarea (stored as complex with zero imaginary part)
"""
function free_compute_sqrtamat_spline(ctrl::ForceFreeStatesControl, equil::Equilibrium.PlasmaEquilibrium, intr::ForceFreeStatesInternal)
    profiles = equil.profiles
    mtheta_eq = length(equil.rzphi_ys)
    mpert = intr.mpert

    # Same grid as wv spline: evenly spaced in q
    qedge = profiles.q_spline(ctrl.psiedge)
    npsi = max(4, ceil(Int, (intr.qlim - qedge) * intr.nhigh * 4))
    psi_array = zeros(Float64, npsi + 1)

    n_series = mpert^2 + 1  # sqrtamat (flattened) + jarea
    data_array = zeros(ComplexF64, npsi + 1, n_series)

    # Create FourierTransform once (same for all psi since mtheta/mpert/mlow don't change)
    ft = Utilities.FourierTransforms.FourierTransform(mtheta_eq, mpert, intr.mlow)

    for i in 1:(npsi+1)
        qi = qedge + (intr.qlim - qedge) * ((i - 1) / npsi)
        psii = ctrl.psiedge + (intr.psilim - ctrl.psiedge) * ((i - 1) / npsi)
        psi_array[i] = find_zero(
            (psi -> profiles.q_spline(psi) - qi,
                psi -> profiles.q_deriv(psi)),
            psii, Roots.Newton()
        )

        sqrtamat = Equilibrium.compute_sqrtamat(equil, psi_array[i], ft)
        jarea = Equilibrium.flux_surface_area(equil, psi_array[i], mtheta_eq)

        # Flatten sqrtamat into first mpert^2 entries, jarea as last entry
        data_array[i, 1:mpert^2] .= vec(sqrtamat)
        data_array[i, end] = complex(jarea, 0.0)
    end

    return cubic_interp(psi_array, Series(data_array); extrap=ExtendExtrap())
end
