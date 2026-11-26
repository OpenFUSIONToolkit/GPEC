"""
    evaluate_stability_criterion!(odet, profiles) -> nzero

Fill `odet.crit_store` over all saved steps and return the number of zero crossings of the
critical eigenvalue (each signals a fixed-boundary instability). Warns with a count if W⁻¹ was
non-Hermitian beyond tolerance at any step.
"""
function evaluate_stability_criterion!(odet::OdeState, profiles::Equilibrium.ProfileSplines)

    resize!(odet.crit_store, odet.step)
    nonherm = fill(false, odet.step)
    # The criterion of each stored state is independent of the others.
    nblas = BLAS.get_num_threads()
    BLAS.set_num_threads(1)
    try
        Threads.@threads for istep in 1:odet.step
            crit, nonherm[istep] = compute_smallest_eigenvalue(@view odet.u_store[:, :, :, istep])
            odet.crit_store[istep] = crit * profiles.dVdpsi_spline(odet.psi_store[istep])^2
        end
    finally
        BLAS.set_num_threads(nblas)
    end
    nzero = count(istep -> is_zero_crossing(odet, profiles, istep), 2:odet.step)

    nonherm_count = count(nonherm)
    if nonherm_count > 0
        @warn "W inverse matrix was non-Hermitian beyond tolerance at $nonherm_count integration step(s)"
    end

    return nzero
end

"""
    is_zero_crossing(odet, profiles, istep) -> Bool

Whether `odet.crit_store` (the smallest-magnitude eigenvalue of W⁻¹, scaled by (dV/dψ)²) changes sign between steps
`istep - 1` and `istep` at a genuine crossing rather than noise: its value at the interpolated crossing must lie
between its two neighbors and well below both.
"""
function is_zero_crossing(odet::OdeState, profiles::Equilibrium.ProfileSplines, istep::Int)
    crit, crit_prev = odet.crit_store[istep], odet.crit_store[istep-1]
    crit * crit_prev < 0 || return false
    psi = odet.psi_store[istep]
    fac = crit / (crit - crit_prev)
    psi_mid = psi - fac * (psi - odet.psi_store[istep-1])
    u_mid = @views odet.u_store[:, :, :, istep] .- fac .* (odet.u_store[:, :, :, istep] .- odet.u_store[:, :, :, istep-1])
    crit_mid = first(compute_smallest_eigenvalue(u_mid)) * profiles.dVdpsi_spline(psi_mid; hint=odet.spline_hint)^2
    confirmed = (crit_mid - crit) * (crit_mid - crit_prev) < 0 && abs(crit_mid) < 0.5 * min(abs(crit), abs(crit_prev))
    confirmed && @info "Zero crossing detected at ψ = $(@sprintf("%.3f", psi_mid))"
    return confirmed
end

"""
    compute_smallest_eigenvalue(u) -> crit, nonherm

Form W⁻¹ = U₁U₂⁻¹ from the solution `u`, symmetrize it, and return its signed eigenvalue of
smallest magnitude. Also flags whether W⁻¹ was non-Hermitian beyond tolerance before symmetrizing.

### Returns

  - `crit::Float64`: the computed scaled critical eigenvalue
  - `nonherm::Bool`: true if W⁻¹ was non-Hermitian beyond tolerance (> 1e-3)
"""
@with_pool pool function compute_smallest_eigenvalue(u::AbstractArray{ComplexF64,3})

    # In-place wp_inverse = u[:, :, 1] / u[:, :, 2]
    wp_inverse = acquire!(pool, ComplexF64, size(u, 1), size(u, 2))
    U2_tmp = similar!(pool, wp_inverse)
    wp_inverse .= @view u[:, :, 1]
    U2_tmp .= @view u[:, :, 2]
    rdiv!(wp_inverse, lu!(U2_tmp))

    # Measure non-Hermiticity; taking the Hermitian part differs from Hermitian(), which reads one triangle only
    adjoint_wp_inverse = similar!(pool, wp_inverse)
    tmp_mat1 = similar!(pool, wp_inverse)
    tmp_mat2 = similar!(pool, wp_inverse)
    adjoint!(adjoint_wp_inverse, wp_inverse)
    @. tmp_mat1 = 0.5 * (wp_inverse + adjoint_wp_inverse)
    @. tmp_mat2 = 0.5 * (wp_inverse - adjoint_wp_inverse)

    nonherm_error = norm(tmp_mat2) / norm(tmp_mat1)
    nonherm = nonherm_error > 1e-3

    hermitianpart!(wp_inverse)

    # Return the eigenvalue of smallest magnitude, keeping its sign so the caller can detect crossings
    evals = eigvals!(Hermitian(wp_inverse))
    crit = evals[findmin(abs, evals)[2]]
    return crit, nonherm
end
