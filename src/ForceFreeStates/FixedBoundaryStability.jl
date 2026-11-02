"""
    evaluate_stability_criterion!(odet, profiles) -> nzero

Fill `odet.crit_store` over all saved steps and return the number of zero crossings of the
critical eigenvalue (each signals a fixed-boundary instability). Warns with a count if W⁻¹ was
non-Hermitian beyond tolerance at any step.
"""
function evaluate_stability_criterion!(odet::OdeState, profiles::Equilibrium.ProfileSplines)

    resize!(odet.crit_store, odet.step)
    nzero = 0
    nonherm_count = 0

    for istep in 1:odet.step
        zero_cross, nonherm = check_for_zero_crossings!(odet, profiles, istep)
        if zero_cross
            nzero += 1
        end
        if nonherm
            nonherm_count += 1
        end
    end

    if nonherm_count > 0
        @warn "W inverse matrix was non-Hermitian beyond tolerance at $nonherm_count integration step(s)"
    end

    return nzero
end

"""
    check_for_zero_crossings!(odet, profiles, istep) -> zero_cross, nonherm

Compute `crit` (smallest-magnitude eigenvalue of W⁻¹, scaled by (dV/dψ)²) at step `istep` into
`odet.crit_store`, and report whether it changed sign since the previous step. A sign change counts
only if the interpolated midpoint confirms it is a genuine crossing rather than noise.

### Returns

  - `zero_cross::Bool`: True if a physical zero crossing was detected
  - `nonherm::Bool`: True if W⁻¹ was non-Hermitian beyond tolerance
"""
@with_pool pool function check_for_zero_crossings!(odet::OdeState, profiles::Equilibrium.ProfileSplines, istep::Int)

    u = acquire!(pool, eltype(odet.u_store), size(odet.u_store)[1:3])
    psi = odet.psi_store[istep]
    u .= odet.u_store[:, :, :, istep]
    dVdpsi = profiles.dVdpsi_spline(psi; hint=odet.spline_hint)
    crit_val, nonherm = compute_smallest_eigenvalue(u)
    odet.crit_store[istep] = crit_val * dVdpsi^2

    zero_cross = false
    if istep > 1 && odet.crit_store[istep] * odet.crit_store[istep-1] < 0
        crit = odet.crit_store[istep]
        crit_prev = odet.crit_store[istep-1]
        # Reject noise: crit at the interpolated crossing must lie between and well below both neighbors
        fac = crit / (crit - crit_prev)
        psi_mid = psi - fac * (psi - odet.psi_store[istep-1])
        u_mid = u .- fac .* (u .- @view(odet.u_store[:, :, :, istep-1]))
        dVdpsi = profiles.dVdpsi_spline(psi_mid; hint=odet.spline_hint)
        crit_mid_val, _ = compute_smallest_eigenvalue(u_mid)
        crit_mid = crit_mid_val * dVdpsi^2
        if (crit_mid - crit) * (crit_mid - crit_prev) < 0 && abs(crit_mid) < 0.5 * min(abs(crit), abs(crit_prev))
            zero_cross = true
            @info "Zero crossing detected at ψ = $(@sprintf("%.3f", psi_mid))"
        end
    end
    return zero_cross, nonherm
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
