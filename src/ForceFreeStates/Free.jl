"""
    power_norm_matrix!(Nmat, jmat, mpert, npert, dV_dpsi) -> Nmat

Assemble the power-normalization (surface-norm) matrix N from the conjugate-symmetric Jacobian
Fourier band `jmat` (length 2·mpert−1, evaluated from the `ffit.jmats` spline), such that

    ξ†·N·ξ = ∮ J |ξ(θ)|² dθ / (dV/dψ) = ⟨|ξ|²⟩

is the flux-surface average of the squared boundary displacement — the DCON power normalization
as a quadratic form. N is Hermitian Toeplitz within each n-block (block-diagonal over n since the
Jacobian is axisymmetric) and positive definite (J > 0). It is the metric of the generalized
eigenproblem W·v = λ·N·v solved in `free_run!` and `free_compute_total`: because W and N
transform by the same congruence under a change of working (Jacobian) coordinate, the pencil
eigenvalues are power-normalized mode energies that are invariant to that coordinate choice.
"""
function power_norm_matrix!(Nmat::AbstractMatrix{ComplexF64}, jmat::AbstractVector{ComplexF64}, mpert::Int, npert::Int, dV_dpsi::Float64)
    fill!(Nmat, 0.0 + 0.0im)
    for ipert_n in 1:npert
        off = (ipert_n - 1) * mpert
        # Toeplitz band index: harmonic difference (m'−m) ∈ [−(mpert−1), mpert−1] maps to 1…2·mpert−1, midpoint mpert = the m=0 Jacobian coefficient
        for ipert_m in 1:mpert, jpert_m in 1:mpert
            Nmat[off+jpert_m, off+ipert_m] = jmat[jpert_m-ipert_m+mpert] / dV_dpsi
        end
    end
    return Nmat
end

"""
    free_run!(odet::OdeState, ctrl::ForceFreeStatesControl, equil::Equilibrium.PlasmaEquilibrium, ffit::FourFitVars, intr::ForceFreeStatesInternal) -> VacuumData

Compute the free boundary energies using the Julia port of the VACUUM code. Performs the same function as `free_run`
in the Fortran code, except now all data is passed in memory instead of via files. This
modifies `odet` in place to normalize the eigenfunctions stored in `u_store` and `ud_store`,
and returns a `VacuumData` struct containing the data needed for perturbed equilibrium calculations
and data dumping.
"""
@with_pool pool function free_run!(odet::OdeState, ctrl::ForceFreeStatesControl, equil::Equilibrium.PlasmaEquilibrium, ffit::FourFitVars, intr::ForceFreeStatesInternal)

    # Initializations and allocations
    (; mpert, mlow, mhigh, numpert_total, psilim, qlim, npert, nlow, nhigh, wall_settings) = intr
    vac_data = VacuumData(ctrl.mthvac * ctrl.nzvac, intr.numpert_total, ctrl.mthvac)
    etemp = zeros!(pool, ComplexF64, numpert_total)
    wp = zeros!(pool, ComplexF64, numpert_total, numpert_total)
    wpt = zeros!(pool, ComplexF64, numpert_total, numpert_total)
    wvt = zeros!(pool, ComplexF64, numpert_total, numpert_total)
    tmp_mat = zeros!(pool, ComplexF64, numpert_total, numpert_total)

    # Evaluate dV/dpsi at the plasma edge
    dV_dpsi = equil.profiles.dVdpsi_spline(psilim)

    # Compute plasma response matrix W = U₂ * U₁⁻¹
    if ctrl.ode_flag
        @views wp .= (odet.u[:, :, 2] / odet.u[:, :, 1]) ./ equil.psio^2
    end

    # Compute vacuum response matrix in-place (handles 2D single-n, 2D multi-n block-diagonal, and 3D)
    vac_inputs = Vacuum.VacuumInput(equil, psilim, ctrl.mthvac, ctrl.nzvac, mlow:mhigh, nlow:nhigh)
    Vacuum.compute_vacuum_response!(vac_data, vac_inputs, wall_settings)

    # Scale by (m - n*q)(m' - n'*q) [Chance Phys. Plasmas 1997 2161 eq. 126]
    singfac = vec((mlow:mhigh) .- qlim .* (nlow:nhigh)')
    vac_data.wv .*= singfac .* singfac'

    # Power-normalization matrix N at the plasma edge: ξ†·N·ξ = ⟨|ξ|²⟩ (see power_norm_matrix!).
    # The Jacobian band is evaluated at psilim (same surface as W), not at the last grid surface.
    Nmat = zeros!(pool, ComplexF64, numpert_total, numpert_total)
    jmat_edge = zeros!(pool, ComplexF64, 2 * mpert - 1)
    ffit.jmats(jmat_edge, psilim; hint=ffit._hint)
    power_norm_matrix!(Nmat, jmat_edge, mpert, npert, dV_dpsi)

    # Least stable eigenvalue of the vacuum matrix alone, power-normalized via the pencil
    # (wv, N) so it shares the units of the mode energies (should be PSD; clamp noise to zero)
    vac_data.vacuum_eigenvalue = max(0.0, minimum(real.(eigvals(Hermitian(vac_data.wv), Hermitian(Nmat)))))

    # Preserve wp in vac_data so it can be written to HDF5 as W_plasma
    vac_data.wp .= wp

    # Complex energy eigenvalues and vectors of the generalized eigenproblem W·v = λ·N·v.
    # The eigenvalues are stationary values of the power quotient ξ†Wξ/ξ†Nξ — power-normalized
    # mode energies invariant to the working (Jacobian) coordinate, since W and N transform by
    # the same congruence under a poloidal coordinate change.
    vac_data.wt .= wp .+ vac_data.wv
    vac_data.wt0 .= vac_data.wt
    Ev = eigen(vac_data.wt, Nmat)
    vac_data.et .= Ev.values
    eindex = sortperm(real.(vac_data.et); rev=true)

    etemp .= vac_data.et
    # Rearrange wt columns for ascending real eigenvalues (most unstable first)
    for ipert in 1:numpert_total
        orig = eindex[numpert_total+1-ipert]
        vac_data.wt[:, ipert] .= Ev.vectors[:, orig]
        vac_data.et[ipert] = etemp[orig]
        # Store which n this eigenvector corresponds to (needed to write IMAS data)
        # This relies on the block diagonal matrix structure due to n decoupling in tokamaks
        imax = argmax(abs.(Ev.vectors[:, orig]))
        vac_data.n_tor_idx[ipert] = (imax - 1) ÷ mpert
    end

    # Normalize eigenvectors to unit power norm v†·N·v = 1. The generalized eigenvalues are
    # already the power-normalized energies, so et is not rescaled here.
    for isol in 1:numpert_total
        v = @view vac_data.wt[:, isol]
        v ./= sqrt(real(dot(v, Nmat, v)))
    end

    # Normalize phase
    imax = 0
    for isol in 1:numpert_total
        imax = argmax(abs.(vac_data.wt[:, isol]))
        phase = abs(vac_data.wt[imax, isol]) / vac_data.wt[imax, isol]
        vac_data.wt[:, isol] .*= phase
    end

    # Project W_p and W_v into the eigenmode basis. Diagonal entries give
    # the plasma/vacuum energy split for each mode: et[i] = ep[i] + ev[i]
    mul!(tmp_mat, wp, vac_data.wt)
    mul!(wpt, vac_data.wt', tmp_mat)
    mul!(tmp_mat, vac_data.wv, vac_data.wt)
    mul!(wvt, vac_data.wt', tmp_mat)
    vac_data.ep .= diag(wpt)
    vac_data.ev .= diag(wvt)

    # Normalize eigenvectors based on scaled wt
    coeffs = odet.u[:, :, 1, end] \ (vac_data.wt .* (2π * equil.psio * 1e-3))
    @views for istep in 1:odet.step
        mul!(tmp_mat, odet.u_store[:, :, 1, istep], coeffs)
        odet.u_store[:, :, 1, istep] .= tmp_mat
        mul!(tmp_mat, odet.u_store[:, :, 2, istep], coeffs)
        odet.u_store[:, :, 2, istep] .= tmp_mat
        mul!(tmp_mat, odet.ud_store[:, :, 1, istep], coeffs)
        odet.ud_store[:, :, 1, istep] .= tmp_mat
        mul!(tmp_mat, odet.ud_store[:, :, 2, istep], coeffs)
        odet.ud_store[:, :, 2, istep] .= tmp_mat
    end

    # Write energies to screen
    if ctrl.verbose
        @info "Least Stable Eigenmode Energies:\n" *
              "  Plasma = $((@sprintf "%+.3e %+.3ei" real(vac_data.ep[1]) imag(vac_data.ep[1])))\n" *
              "  Vacuum = $((@sprintf "%+.3e %+.3ei" real(vac_data.ev[1]) imag(vac_data.ev[1])))\n" *
              "  Total  = $((@sprintf "%+.3e %+.3ei" real(vac_data.et[1]) imag(vac_data.et[1])))"
    end

    return vac_data
end

"""
    free_compute_wv_spline(ctrl::ForceFreeStatesControl, equil::Equilibrium.PlasmaEquilibrium, intr::ForceFreeStatesInternal)

Compute a spline of vacuum response matrices over the range of psi from 'ctrl.psi_edge' to
`intr.qlim`. This is used for fast evaluation of wt during `ode_record_edge`. Performs the
same function as `free_wvmats` in the Fortran code. Currently defaults to 4 spline points per
q-window minimum.
"""
@with_pool pool function free_compute_wv_spline(ctrl::ForceFreeStatesControl, equil::Equilibrium.PlasmaEquilibrium, intr::ForceFreeStatesInternal)

    profiles = equil.profiles

    # Number of psi grid points for the spline: 4 per q-window minimum
    # TODO: 4 spline points is arbitrary - is there a better way?
    qedge = profiles.q_spline(ctrl.psiedge)
    npsi = max(4, ceil(Int, (intr.qlim - qedge) * intr.nhigh * 4))
    psi_array = zeros!(pool, Float64, npsi + 1)
    wv_array = zeros!(pool, ComplexF64, npsi + 1, intr.numpert_total, intr.numpert_total)

    for i in 1:(npsi+1)
        # Space points evenly in q over [qedge, qlim] (i=1 → qedge, i=npsi+1 → qlim)
        qi = qedge + (intr.qlim - qedge) * ((i - 1) / npsi)

        psii = ctrl.psiedge + (intr.psilim - ctrl.psiedge) * ((i - 1) / npsi)
        psi_array[i] = find_zero(
            (psi -> profiles.q_spline(psi) - qi,
                psi -> profiles.q_deriv(psi)),
            psii, Roots.Newton()
        )

        # Compute raw vacuum matrix at the actual scan psi (singfac NOT applied; free_compute_total applies it analytically)
        vac_inputs = Vacuum.VacuumInput(equil, psi_array[i], ctrl.mthvac, ctrl.nzvac, intr.mlow:intr.mhigh, intr.nlow:intr.nhigh)
        wv, _, _, _, _ = Vacuum.compute_vacuum_response(vac_inputs, intr.wall_settings)
        @views wv_array[i, :, :] .= wv
    end

    # Flatten 3D array to (npsi+1 × numpert_total^2) for series interpolant
    wv_flat = reshape(wv_array, npsi + 1, intr.numpert_total^2)

    # FastInterpolations now natively supports complex values - create complex series interpolant directly
    # Use CubicFit() (default) for native endpoint handling
    wvmat = cubic_interp(psi_array, Series(wv_flat); extrap=ExtendExtrap())

    return wvmat
end

"""
    free_compute_total(equil::Equilibrium.PlasmaEquilibrium, ffit::FourFitVars, intr::ForceFreeStatesInternal, odet::OdeState) -> ComplexF64

Compute total complex energy eigenvalue (total1). This is a trimmed down version of `free_run`
that only computes the total energy eigenvalue for the mode unstable mode, used in `findmax_dW_edge!`
which calls this function at each step in the psiedge -> psilim region of integration. This performs
the same function as `free_test` in the Fortran code, except we have moved the creation of the
wv matrix spline to `free_compute_wv_spline` and pass it in `odet.edge_scan.wvmat` (a complex-valued spline).
"""
@with_pool pool function free_compute_total(equil::Equilibrium.PlasmaEquilibrium, ffit::FourFitVars, intr::ForceFreeStatesInternal, odet::OdeState)

    Npert = intr.numpert_total
    wp = zeros!(pool, ComplexF64, Npert, Npert)
    eigenvalues = zeros!(pool, ComplexF64, Npert)
    wt = zeros!(pool, ComplexF64, Npert, Npert)
    wv = zeros!(pool, ComplexF64, Npert, Npert)
    Nmat = zeros!(pool, ComplexF64, Npert, Npert)
    jmat_local = zeros!(pool, ComplexF64, 2 * intr.mpert - 1)
    eindex = zeros!(pool, Int, Npert)
    evals_real = zeros!(pool, Float64, Npert)
    tmp_v = zeros!(pool, ComplexF64, Npert)

    dV_dpsi = equil.profiles.dVdpsi_spline(odet.psifac)

    # Compute plasma response matrix
    @views wp .= (odet.u[:, :, 2] / odet.u[:, :, 1]) ./ equil.psio^2

    # Retrieve raw vacuum matrix from spline and apply singfac analytically at the local q.
    # Singfac is not pre-applied in the spline (see free_compute_wv_spline) to avoid interpolating
    # a zero-crossing function near rational surfaces, which would distort the peaks.
    es = odet.edge_scan
    es.wvmat(vec(wv), odet.psifac; hint=es.wv_hint)
    q_at_psifac = equil.profiles.q_spline(odet.psifac)
    # Scale by (m - n*q)(m' - n'*q) [Chance Phys. Plasmas 1997 2161 eq. 126]
    singfac = vec((intr.mlow:intr.mhigh) .- q_at_psifac .* (intr.nlow:intr.nhigh)')
    wv .*= singfac .* singfac'

    # Local power-normalization matrix N(ψ) from the Jacobian Fourier band spline, so the
    # power quotient uses the same surface as W (see power_norm_matrix!)
    ffit.jmats(jmat_local, odet.psifac; hint=ffit._hint)
    power_norm_matrix!(Nmat, jmat_local, intr.mpert, intr.npert, dV_dpsi)

    # Total energy matrix and generalized eigen-decomposition of the pencil (W, N) — the
    # eigenvalues are power-normalized, Jacobian-invariant mode energies (see free_run!)
    wt .= wp .+ wv
    Ev = eigen(wt, Nmat)

    # Sort eigenvalues by descending real part
    @inbounds for i in 1:Npert
        evals_real[i] = real(Ev.values[i])
    end
    sortperm!(eindex, evals_real; rev=true)
    for ipert in 1:Npert
        wt[:, ipert] .= Ev.vectors[:, eindex[Npert+1-ipert]]
        eigenvalues[ipert] = Ev.values[eindex[Npert+1-ipert]]
    end

    # Plasma and vacuum energy components for the leading eigenvector via power quotients:
    # plasma_energy + vacuum_energy = total_eigenvalue (wt = wp + wv; λ = v'*wt*v / v'*N*v).
    v = @view wt[:, 1]
    vnorm = real(dot(v, Nmat, v))
    mul!(tmp_v, wp, v)
    plasma_energy = ComplexF64(dot(v, tmp_v)) / vnorm
    mul!(tmp_v, wv, v)
    vacuum_energy = ComplexF64(dot(v, tmp_v)) / vnorm

    # Smallest eigenvalue of the vacuum matrix alone via the pencil (wv, N), so all four outputs
    # share the power-normalized units. The singfac-scaled wv should be PSD by construction
    # (congruence of PSD wv_raw); clamp numerical noise to zero.
    vacuum_eigenvalue = max(0.0, minimum(real.(eigvals(Hermitian(wv), Hermitian(Nmat)))))

    return (total_eigenvalue=eigenvalues[1], plasma_energy=plasma_energy, vacuum_energy=vacuum_energy, vacuum_eigenvalue=vacuum_eigenvalue)
end
