"""
    evaluate_stability_criterion(ctrl, equil, intr, odet, outp) -> nzero

Evaluate the stability criterion over the entire integration, counting the number of
zero crossings of the critical eigenvalue which indicate instability. This acts as an
outer wrapper for `check_for_zero_crossings!` since our in-memory integration allow
this to be done post-integration rather than during like the Fortran. Optionally, we
also write the `crit.h5` file containing the crit values over the integration.

"""
function evaluate_stability_criterion(ctrl::DconControl, equil::Equilibrium.PlasmaEquilibrium, intr::DconInternal, odet::OdeState, outp::DconOutput)

    # Initialization
    if ctrl.verbose
        println("Evaluating stability criterion over entire integration")
    end
    nzero = 0
    crit_store = zeros(Float64, odet.step)

    # Loop over integration steps, computing crit/checking for zero crossings
    for istep in 1:odet.step
        zero_cross = check_for_zero_crossings!(crit_store, odet, equil.sq, istep)
        if zero_cross
            nzero += 1
        end
    end

    # Write crit values to HDF5 file
    if outp.write_crit_h5
        if ctrl.verbose
            println("   Writing crit.h5 file")
        end
        h5open(joinpath(intr.dir_path, outp.fname_crit_h5), "w") do crit_h5
            crit_h5["psi"] = odet.psi_store
            crit_h5["q"] = odet.q_store
            crit_h5["crit"] = crit_store
        end
    end
    return nzero
end

"""
    check_for_zero_crossings!(crit_store, odet, sq, istep) -> zero_cross

Check if the critical eigenvalue (`crit`), i.e. the smallest eigenvalue of W⁻¹, changed
signs between a given integration step `istep` and the previous step. We first compute
`crit` at the current step, writing the result to the `crit_store` array, then compare
with the previous step to see if it changed signs. If so, we perform additional checks
to ensure the crossing is physical and not just numerical noise, and if so, we return
that a zero_crossing occurred, indicating the presence of a fixed-boundary instability.
This performs the same function as `ode_output_monitor` in the Fortran code, except we
can do it post-integration rather than during and don't directly handle file outputs here.

### Arguments

  - `crit_store::Vector{Float64}`: Stores computed crit values, updated in place at index `istep`
  - `sq::Spl.CubicSpline`: Spline object containing equilibrium profiles
  - `istep::Int`: Current integration step index

"""
function check_for_zero_crossings!(crit_store::Vector{Float64}, odet::OdeState, sq::Spl.CubicSpline{Float64}, istep::Int)

    # Compute smallest eigenvalue (crit) at current step
    psi = odet.psi_store[istep]
    u = odet.u_store[:, :, :, istep]
    crit_store[istep] = compute_smallest_eigenvalue(psi, u, sq)

    # Check for zero crossing via change in sign of crit between current and previous step
    zero_cross = false
    if istep > 1 && crit_store[istep] * crit_store[istep - 1] < 0
        crit = crit_store[istep]
        crit_prev = crit_store[istep - 1]
        # Ensure the zero crossing is physical and not just numerical noise
        fac = crit / (crit - crit_prev)
        psi_mid = psi - fac * (psi - odet.psi_store[istep - 1])
        u_mid = u .- fac .* (u .- @view(odet.u_store[:, :, :, istep - 1]))
        crit_mid = compute_smallest_eigenvalue(psi_mid, u_mid, sq)
        if (crit_mid - crit) * (crit_mid - crit_prev) < 0 && abs(crit_mid) < 0.5 * min(abs(crit), abs(crit_prev))
            zero_cross = true
            println("Zero crossing detected at psi = $psi_mid, q = $q_mid")
        end
    end
    return zero_cross
end

"""
    compute_smallest_eigenvalue(psi, u, sq) -> crit

Form the inverse plasma response matrix W⁻¹ at flux surface `psi` using the
solution matrix `u`,and compute its eigenvalues. Return the smallest eigenvalue
(in magnitude) scaled by dV/dψ². Performs the same function as `ode_output_get_crit`
in the Fortran code.

### Arguments

  - `psi::Float64`: The flux surface at which to evaluate
  - `u::Array{ComplexF64, 3}`: Solution matrix at `psi`
  - `sq::Spl.CubicSpline`: Spline object containing equilibrium profiles

### Returns

  - `crit::Float64`: the computed scaled critical eigenvalue

"""
function compute_smallest_eigenvalue(psi::Float64, u::Array{ComplexF64,3}, sq::Spl.CubicSpline{Float64})

    # Compute inverse plasma response matrix W = U₁ * U₂⁻¹ = adj(adj(U₂)⁻¹ * adj(U₁))
    wp_inverse = adjoint(u[:, :, 1])
    temp = adjoint(u[:, :, 2])
    wp_inverse = temp \ wp_inverse

    # Enforce that W is Hermitian
    # TODO: Is this necessary? The DCON paper says W is Hermitian by construction.
    # Couldn't this mask numerical issues we should be addressing directly?
    # i.e. we should error out in the Hermitian eigenvalue solver if W is not Hermitian?
    wp_inverse .+= adjoint(wp_inverse)
    wp_inverse .*= 0.5

    # Compute eigenvalues, sort, and return the smallest (with a scale factor)
    eig_vals = eigvals!(Hermitian(wp_inverse))
    index = sortperm(eig_vals; by=abs)
    dVdpsi = Spl.spline_eval!(sq, psi)[3]
    return eig_vals[index[1]] * dVdpsi^2
end