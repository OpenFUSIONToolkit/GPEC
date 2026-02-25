"""
    evaluate_stability_criterion!(odet, profiles) -> nzero

Evaluate the stability criterion over the entire integration, counting the number of
zero crossings of the critical eigenvalue which indicate instability. This acts as an
outer wrapper for `check_for_zero_crossings!` since our in-memory integration allow
this to be done post-integration rather than during like the Fortran. We update
the `crit_store` in `odet` in place, and return the total number of zero crossings found.
If the W inverse matrix was non-Hermitian beyond tolerance at any integration steps,
a warning is printed with the total count.
"""
function evaluate_stability_criterion!(odet::OdeState, profiles::Equilibrium.ProfileSplines)

    # Initialization
    resize!(odet.crit_store, odet.step)
    nzero = 0
    nonherm_count = 0

    # Loop over integration steps, computing crit/checking for zero crossings
    for istep in 1:odet.step
        zero_cross, nonherm = check_for_zero_crossings!(odet, profiles, istep)
        if zero_cross
            nzero += 1
        end
        if nonherm
            nonherm_count += 1
        end
    end

    # Print warning if W⁻¹ was non-Hermitian at any steps
    if nonherm_count > 0
        @warn "W inverse matrix was non-Hermitian beyond tolerance at $nonherm_count integration step(s)"
    end

    return nzero
end

"""
    check_for_zero_crossings!(odet, profiles, istep) -> zero_cross, nonherm

Check if the critical eigenvalue (`crit`), i.e. the smallest eigenvalue of W⁻¹, changed
signs between a given integration step `istep` and the previous step. We first compute
`crit` at the current step, writing the result to the `crit_store` array, then compare
with the previous step to see if it changed signs. If so, we perform additional checks
to ensure the crossing is physical and not just numerical noise, and if so, we return
that a zero_crossing occurred, indicating the presence of a fixed-boundary instability.
This performs the same function as `ode_output_monitor` in the Fortran code, except we
can do it post-integration rather than during and don't directly handle file outputs here.

### Arguments

  - `profiles::ProfileSplines`: Profile splines containing equilibrium profiles
  - `istep::Int`: Current integration step index

### Returns

  - `zero_cross::Bool`: True if a physical zero crossing was detected
  - `nonherm::Bool`: True if W⁻¹ was non-Hermitian beyond tolerance
"""
@with_pool pool function check_for_zero_crossings!(odet::OdeState, profiles::Equilibrium.ProfileSplines, istep::Int)

    # Compute smallest eigenvalue (crit) at current step
    # Use shared hint with LinearBinary() search for O(1) interval lookup during sequential stability evaluation
    u = acquire!(pool, eltype(odet.u_store), size(odet.u_store)[1:3])
    psi = odet.psi_store[istep]
    u .= odet.u_store[:, :, :, istep]
    dVdpsi = profiles.dVdpsi_spline(psi; hint=odet.spline_hint)
    crit_val, nonherm = compute_smallest_eigenvalue(u)
    odet.crit_store[istep] = crit_val * dVdpsi^2

    # Check for zero crossing via change in sign of crit between current and previous step
    zero_cross = false
    if istep > 1 && odet.crit_store[istep] * odet.crit_store[istep-1] < 0
        crit = odet.crit_store[istep]
        crit_prev = odet.crit_store[istep-1]
        # Ensure the zero crossing is physical and not just numerical noise
        fac = crit / (crit - crit_prev)
        psi_mid = psi - fac * (psi - odet.psi_store[istep-1])
        u_mid = u .- fac .* (u .- @view(odet.u_store[:, :, :, istep-1]))
        dVdpsi = profiles.dVdpsi_spline(psi_mid; hint=odet.spline_hint)
        crit_mid_val, _ = compute_smallest_eigenvalue(u_mid)
        crit_mid = crit_mid_val * dVdpsi^2
        if (crit_mid - crit) * (crit_mid - crit_prev) < 0 && abs(crit_mid) < 0.5 * min(abs(crit), abs(crit_prev))
            zero_cross = true
            println("Zero crossing detected at psi = $psi_mid")
        end
    end
    return zero_cross, nonherm
end

"""
    compute_smallest_eigenvalue(u) -> crit, nonherm

Form the inverse plasma response matrix W⁻¹ using the solution matrix `u` and
returns its minimum eigenvalue by magnitude. Performs the same function as
`ode_output_get_crit` in the Fortran code, except we explicitly form W⁻¹ here
from U₁ * U₂⁻¹ using Julia's right division operator `/` instead of
adj(adj(U₂)⁻¹ * adj(U₁)) as done in Fortran. We have also added a check to
ensure W is Hermitian within tolerance, as the matrix should be Hermitian by
construction but may accumulate numerical noise during integration.

### Arguments

  - `u::AbstractArray{ComplexF64, 3}`: Solution matrix at `psi`

### Returns

  - `crit::Float64`: the computed scaled critical eigenvalue
  - `nonherm::Bool`: true if W⁻¹ was non-Hermitian beyond tolerance (> 1e-3)
"""
@with_pool pool function compute_smallest_eigenvalue(u::AbstractArray{ComplexF64,3})

    # Compute inverse plasma response matrix W⁻¹ = U₁ * U₂⁻¹
    # The following is in-place operation equivalent to wp_inverse = u[:, :, 1] / u[:, :, 2]
    wp_inverse = acquire!(pool, ComplexF64, size(u, 1), size(u, 2))
    U2_tmp = similar!(pool, wp_inverse)
    wp_inverse .= @view u[:, :, 1]
    U2_tmp .= @view u[:, :, 2]
    rdiv!(wp_inverse, lu!(U2_tmp))

    # TODO: This section not be necessary since W should be Hermitian by construction.
    # This likely just removes any numerical noise during integration
    # However, if we do remove it, note that Hermitian(wp_inverse) in the eigval solve enforces Hermiticity
    # by ignoring the lower triangle (by default, can ignore upper using `uplo=:L`), which will
    # NOT produce identical results to taking the Hermitian part as done here unless is exactly Hermitian.
    # Check to make sure W is at least close to Hermitian before enforcing it

    # Compute adjoint in-place to avoid allocations
    adjoint_wp_inverse = similar!(pool, wp_inverse)
    tmp_mat1 = similar!(pool, wp_inverse)
    tmp_mat2 = similar!(pool, wp_inverse)
    adjoint!(adjoint_wp_inverse, wp_inverse)
    @. tmp_mat1 = 0.5 * (wp_inverse + adjoint_wp_inverse)
    @. tmp_mat2 = 0.5 * (wp_inverse - adjoint_wp_inverse)

    nonherm_error = norm(tmp_mat2) / norm(tmp_mat1)
    nonherm = nonherm_error > 1e-3

    # Enforce that W is Hermitian
    hermitianpart!(wp_inverse) # Overwrites W⁻¹ with (W⁻¹ + (W⁻¹)') / 2

    # Compute eigenvalues and return the smallest
    crit = findmin(abs, eigvals!(Hermitian(wp_inverse)))[1]
    return crit, nonherm
end
