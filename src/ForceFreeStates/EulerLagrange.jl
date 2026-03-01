"""
    eulerlagrange_integration(ctrl::ForceFreeStatesControl, equil::Equilibrium.PlasmaEquilibrium, ffit::FourFitVars, intr::ForceFreeStatesInternal)

Main driver for integrating the Euler-Lagrange equations across the plasma and detecting singular surfaces.
Formerly `ode_run`. Has the same functionality as `ode_run` in the Fortran code, with the addition of
a single dump to the `euler.h5` file at the end of integration instead of multiple dumps
to `euler.bin` throughout the integration. We have made the control logic more clear
by pre-computing all integration chunks upfront and using a for loop to iterate through them,
eliminating the while-loop logic and making integration bounds explicit at each step.
We now perform significant post-processing after integration including finding the peak dW
in the edge region and evaluating the stability criterion over the entire integration,
which were previously done during integration in the Fortran code.

### TODOs

Support for `kin_flag`
restype functionality if we decide to do this

### Returns

An OdeState struct containing the final state of the ODE solver after integration is complete.
"""
function eulerlagrange_integration(ctrl::ForceFreeStatesControl, equil::Equilibrium.PlasmaEquilibrium, ffit::FourFitVars, intr::ForceFreeStatesInternal)

    # Initialization
    odet = OdeState(intr.numpert_total, ctrl.numsteps_init, ctrl.numunorms_init, intr.msing)
    if ctrl.sing_start <= 0
        initialize_el_at_axis!(odet, ctrl, equil.profiles, intr)
    elseif ctrl.sing_start <= intr.msing
        error("sing_start > 0 not implemented yet!")
        # initialize_el_at_singular_surf!(ctrl, equil, intr, odet)
    else
        error("Invalid value for sing_start: $(ctrl.sing_start) > msing = $(intr.msing)")
    end

    # Pre-compute all integration chunks
    chunks = chunk_el_integration_bounds(odet, ctrl, intr)

    # Print initial integration condition
    if ctrl.verbose
        @info "   ψ = $((@sprintf "%.3f" odet.psifac)),  q = $((@sprintf "%.3f" equil.profiles.q_spline(odet.psifac)))"
    end

    # Iterate through each integration chunk
    for chunk in chunks
        # Integrate this region and display progress
        integrate_el_region!(odet, ctrl, equil, ffit, intr, chunk)
        if ctrl.verbose
            @info "   ψ = $((@sprintf "%.3f" odet.psifac)),  q = $((@sprintf "%.3f" odet.q)),  steps = $(odet.total_steps)"
        end

        # Cross a rational surface after integration if this chunk requires it
        if chunk.needs_crossing
            if ctrl.kin_flag
                error("kin_flag = true not implemented yet!")
            else
                cross_ideal_singular_surf!(odet, ctrl, equil, ffit, intr, chunk.ising)
            end
        end
    end

    # Deallocate unused storage of integration data
    if ctrl.psiedge < intr.psilim
        # Find the peak dW in the edge region and truncate integration data there
        odet.step = findmax_dW_edge!(odet, ctrl, equil, ffit, intr)
        trim_storage!(odet)
        if ctrl.verbose
            @info "Truncating integration at peak edge dW: ψ = $((@sprintf "%.3f" odet.psi_store[odet.step])),  q = $((@sprintf "%.3f" odet.q_store[odet.step]))"
        end

        # Update u, psilim, and qlim for usage in determining wp and wt
        intr.psilim = odet.psi_store[end]
        intr.qlim = odet.q_store[end]
        odet.u .= odet.u_store[:, :, :, end]
    else
        odet.step -= 1 # step was incremented one extra time in integrate_el_region!
        trim_storage!(odet)
    end

    # Evaluate stability criterion (critical determinant) of saved solutions
    if ctrl.verbose
        @info "Evaluating fixed-boundary stability criterion"
    end
    odet.nzero = evaluate_stability_criterion!(odet, equil.profiles)

    # Form the true solution vectors, undoing the Gaussian reduction applied in `ode_unorm!` during integration
    transform_u!(odet, intr)

    return odet
end

"""
    initialize_el_at_axis!(odet::OdeState, ctrl::ForceFreeStatesControl, profiles::Equilibrium.ProfileSplines, intr::ForceFreeStatesInternal)

Initialize the OdeState struct for the case of sing_start = 0 (axis initialization).
Formerly `ode_axis_init!`. This now only initializes `psifac`, `ising_start`, and `u`.

### TODOs

Support for `kin_flag`
Move ising_start logic to chunk_el_integration_bounds?
"""
function initialize_el_at_axis!(odet::OdeState, ctrl::ForceFreeStatesControl, profiles::Equilibrium.ProfileSplines, intr::ForceFreeStatesInternal)

    # Default psifac to minimum equilibrium psi value
    odet.psifac = profiles.xs[1]

    # Use Newton iteration to find starting psi if qlow is above q0
    if ctrl.qlow > profiles.q_spline.y[1]
        # Find last index where q < qlow
        idx = findlast(jpsi -> profiles.q_spline.y[jpsi-1] < ctrl.qlow, 2:profiles.npts)
        if idx !== nothing
            odet.psifac = profiles.xs[idx]
        end
        odet.psifac = find_zero(
            (psi -> profiles.q_spline(psi) - ctrl.qlow,
             psi -> profiles.q_deriv(psi)),
            odet.psifac, Roots.Newton()
        )
    end

    # Find starting singular surface (where sing.psifac > psi(qlow/q0))
    # Note: This logic is kept in initialize_el_at_axis! rather than chunk_el_integration_bounds
    # because it depends on the starting psifac which is set here. The logic for sing_start != 0
    # and kin_flag = true would also live here when implemented.
    if false #(TODO: kin_flag)
    # for ising = 1:kmsing
    #     if kinsing[ising].psifac > psifac
    #         break
    #     end
    # end
    else
        odet.ising_start = searchsortedfirst(getfield.(intr.sing, :psifac), odet.psifac) - 1
    end

    # Initialize solutions with the identity matrix for U_22 [Glasser Phys. Plasmas 2016 112506 Section VI]
    for ipert in 1:intr.numpert_total
        odet.u[ipert, ipert, 2] = 1
    end
end

# TODO: NOT IMPLEMENTED YET! (low priority, just make sure sing_start = 0 in toml)
function initialize_el_at_singular_surf()
    return
end

"""
    chunk_el_integration_bounds(odet::OdeState, ctrl::ForceFreeStatesControl, intr::ForceFreeStatesInternal)

Pre-compute all integration chunks from the current position to the edge.
Returns a vector of `IntegrationChunk` objects, each representing a region to integrate
and whether it needs a rational surface crossing beforehand.

This function replaces the iterative while-loop logic with a single upfront computation,
making the integration flow more predictable and easier to parallelize (e.g., for STRIDE).

### Arguments

  - `odet::OdeState` - ODE state struct (starting position and singular surface index)
  - `ctrl::ForceFreeStatesControl` - Control parameters
  - `intr::ForceFreeStatesInternal` - Internal data (singular surfaces, limits)

### Returns

  - `Vector{IntegrationChunk}` - Array of integration chunks to process

### TODOs

Support for `kin_flag`
"""
function chunk_el_integration_bounds(odet::OdeState, ctrl::ForceFreeStatesControl, intr::ForceFreeStatesInternal)
    chunks = IntegrationChunk[]

    # Start from current position
    psi_current = odet.psifac
    ising_current = odet.ising_start

    # Wrapper to find next singular surface to integrate toward that is resonant within integration limits
    function find_next_resonant_surface!(ising::Int, intr::ForceFreeStatesInternal)
        ising += 1
        while ising <= intr.msing
            if intr.psilim < intr.sing[ising].psifac ||
               any(m -> intr.mlow <= m <= intr.mhigh, intr.sing[ising].m)
                break
            end
            ising += 1
        end
        return ising
    end

    # -------------------- Create chunks ------------------------
    if false  # TODO: kin_flag
    # Kinetic not implemented yet, some of the below code might be able to be reused?
    else
        # Loop through singular surfaces to cross until edge is reached
        ising_current = find_next_resonant_surface!(ising_current, intr)
        while ising_current <= intr.msing && intr.psilim >= intr.sing[ising_current].psifac && ctrl.singfac_min != 0
            # Set integration limit to just before the next singular surface
            psi_end = intr.sing[ising_current].psifac - ctrl.singfac_min /
                                                        abs(minimum(intr.sing[ising_current].n) * intr.sing[ising_current].q1)

            # Validate chunk bounds
            @assert psi_current < psi_end "Invalid chunk bounds: psi_start=$psi_current >= psi_end=$psi_end"
            @assert isempty(chunks) || psi_current >= chunks[end].psi_end "Overlapping chunks detected"

            push!(chunks, IntegrationChunk(;
                psi_start=psi_current,
                psi_end=psi_end,
                needs_crossing=true,
                ising=ising_current
            ))

            # After crossing, we jump to the other side of the singular surface
            dpsi = intr.sing[ising_current].psifac - psi_end
            psi_current = psi_end + 2 * dpsi

            # Move to next singular surface that is either resonant or beyond integration limits
            ising_current = find_next_resonant_surface!(ising_current, intr)
        end

        # No more singular surfaces to cross, set integration limit to edge
        @assert psi_current < intr.psilim * (1 - eps) "Final chunk has invalid bounds"
        @assert isempty(chunks) || psi_current >= chunks[end].psi_end "Final chunk overlaps with previous chunk"

        push!(chunks, IntegrationChunk(;
            psi_start=psi_current,
            psi_end=(intr.psilim * (1 - eps)),
            needs_crossing=false,
            ising=0
        ))
    end

    return chunks
end

"""
    cross_ideal_singular_surf!(odet::OdeState, ctrl::ForceFreeStatesControl, equil::Equilibrium.PlasmaEquilibrium, ffit::FourFitVars, intr::ForceFreeStatesInternal)

Handle the crossing of a rational surface during integration if `kin_flag` is false.
Formerly `ode_ideal_cross!`. Performs the same function as `ode_ideal_cross` in the Fortran code.
Differences mainly in integration data storage logic, but otherwise identical. It normalizes and
reinitializes the solution vector at the singularity, and updates relevant state variables.

Asymptotics are now computed on-demand here instead of being pre-computed, making it clear
that asymptotic calculations are specific to ideal ForceFreeStates and not inherent to the singular surface.

### Arguments

  - `ising::Int` - Index of the singular surface being crossed
"""
function cross_ideal_singular_surf!(odet::OdeState, ctrl::ForceFreeStatesControl, equil::Equilibrium.PlasmaEquilibrium, ffit::FourFitVars, intr::ForceFreeStatesInternal, ising::Int)

    # Fixup solution at singular surface
    compute_solution_norms!(odet.u, odet, ctrl, intr, true)

    # Compute asymptotic power series for this singular surface
    singp = intr.sing[ising]
    sing_asymp = compute_sing_asymptotics(singp, ctrl, equil, ffit, intr)
    dpsi = singp.psifac - odet.psifac # ψ_res - ψ

    # Get asymptotic coefficients before crossing rational surface
    ua = sing_get_ua(sing_asymp, -dpsi)
    odet.ca_l[:, :, :, ising] .= sing_get_ca(odet.u, ua, intr)

    # Single n: remove largest solution and sub in asymptotics on the other side
    # Multi-n: if we remove the N largest modes in arbitrary order, we can mess up the
    # diagonal structure of the matrix and later calculations. zeroed_idx let's us make sure
    # the solution vector we're zeroing corresponds to the same block as the resonant mode we
    # introduce. It is also needed when transforming u back to the full solution after integration.
    ipert_res = 1 .+ singp.m .- intr.mlow .+ (singp.n .- intr.nlow) .* intr.mpert
    if !ctrl.con_flag
        # Eliminate the solution with the largest norm (in the same block) for each resonance
        odet.zeroed_idx[odet.ifix] = Int[]
        for i in eachindex(sing_asymp.r1)
            push!(odet.zeroed_idx[odet.ifix], findfirst(j -> (ipert_res[i] - 1) ÷ intr.mpert == (odet.index[j, odet.ifix] - 1) ÷ intr.mpert, 1:intr.numpert_total))
            odet.u[:, odet.index[odet.zeroed_idx[odet.ifix][i], odet.ifix], :] .= 0
        end
    end

    # Re-initialize on opposite side of rational surface by approximating solution
    params = (ctrl, equil, ffit, intr, odet, IntegrationChunk(0.0, 0.0, false, ising))
    du1 = zeros(ComplexF64, intr.numpert_total, intr.numpert_total, 2)
    du2 = zeros(ComplexF64, intr.numpert_total, intr.numpert_total, 2)
    sing_der!(du1, odet.u, params, odet.psifac)
    odet.psifac += 2 * dpsi # jump to other side of singular surface
    sing_der!(du2, odet.u, params, odet.psifac)
    odet.u .+= (du1 .+ du2) .* dpsi

    # Apply asymptotic solution on other side of singular surface
    ua = sing_get_ua(sing_asymp, dpsi)
    if !ctrl.con_flag
        for i in eachindex(sing_asymp.r1)
            # Zero out the resonant components
            odet.u[ipert_res[i], :, :] .= 0
            # Introduce the small asymptotic resonant solution on the other side of the singular surface
            odet.u[:, odet.index[odet.zeroed_idx[odet.ifix][i], odet.ifix], :] .= ua[:, ipert_res[i]+intr.numpert_total, :]
        end
    end
    # Get asymptotic coefficients after crossing rational surface
    odet.ca_r[:, :, :, ising] .= sing_get_ca(odet.u, ua, intr)

    # Store values after crossing step and advance
    odet.psi_store[odet.step] = odet.psifac
    odet.q_store[odet.step] = odet.q
    odet.u_store[:, :, :, odet.step] = odet.u
    odet.ud_store[:, :, :, odet.step] = odet.ud
    odet.step += 1
end

# Example stub for kinetic crossing
function cross_kinetic_singular_surf()
    # Implement kinetic crossing logic here
    return
end

"""
    integrate_el_region!(odet::OdeState, ctrl::ForceFreeStatesControl, equil::Equilibrium.PlasmaEquilibrium, ffit::FourFitVars, intr::ForceFreeStatesInternal, chunk::IntegrationChunk)

Integrate the Euler-Lagrange equations from `psi_start` to `psi_end`.
Formerly `ode_step!`. Performs the same function as `ode_step` in the Fortran code, with the addition of
a callback function to handle tolerances, normalization, and storage at each
step of the integration. In Fortran, this was performed by running LSODE in one-step
mode (so ode_step was called hundreds of times) and calling the relevant functions in
a DO loop. Here, we use the DifferentialEquations.jl interface to achieve the same
functionality in a more Julian way. The integration bounds are now explicit arguments,
making it clear what region is being integrated.

### Arguments

  - `odet::OdeState` - ODE state struct (modified in-place)
  - `ctrl::ForceFreeStatesControl` - Control parameters
  - `equil::Equilibrium.PlasmaEquilibrium` - Plasma equilibrium
  - `ffit::FourFitVars` - Fourier fit variables
  - `intr::ForceFreeStatesInternal` - Internal data
  - `chunk::IntegrationChunk` - Integration chunk containing start and end ψ for integration

### TODOs

Check sensitivity of results to tolerances, currently using same logic as Fortran
Check absolute tolerances, currently only relative tolerances are updated
"""
function integrate_el_region!(odet::OdeState, ctrl::ForceFreeStatesControl, equil::Equilibrium.PlasmaEquilibrium, ffit::FourFitVars, intr::ForceFreeStatesInternal, chunk::IntegrationChunk)

    # For integration chunks that extend beyond psihigh in diverted plasmas, switch the
    # q_spline pointer from the direct spline to the iota inverse spline. This allows
    # sing_der! (which evaluates equil.profiles.q_spline at each ODE step) to use the
    # well-behaved iota = 1/q spline near the separatrix without any per-call conditionals.
    profiles = equil.profiles
    using_edge_spline = chunk.psi_end > profiles.xs[end] && !isnothing(profiles.q_spline_iota_inverse)
    if using_edge_spline
        profiles.q_spline = profiles.q_spline_iota_inverse
    end

    # Callback to be run at every step, handles fixups, tolerances, and data storage
    cb = DiscreteCallback((u, t, integrator) -> true, integrator_callback!)

    # Advance differential equation from psi_start to psi_end
    rtol = compute_tols(ctrl, intr, odet, chunk.ising) # initial tolerances
    prob = ODEProblem(sing_der!, odet.u, (chunk.psi_start, chunk.psi_end), (ctrl, equil, ffit, intr, odet, chunk))
    sol = solve(prob, BS5(); reltol=rtol, callback=cb, save_everystep=false, save_end=true)
    # TODO: check absolute tolerances, check how sensitive outputs are to tolerances

    # Restore the direct q_spline after the edge chunk completes
    if using_edge_spline
        profiles.q_spline = profiles.q_spline_direct
    end

    # Update u and psifac with the solution at the end of the interval
    odet.u .= sol.u[end]
    odet.psifac = sol.t[end]
end

"""
    integrator_callback!(integrator)

Callback function for ODE integrator to handle normalization, output, and storage
at each step. This handles the solution normalization logic that was previously
in a DO loop within eulerlagrange_integration and called every step by running LSODE in one step mode
in the Fortran code. However, we now perform the equivalent of `ode_output_step`
and `ode_record_edge` post-integration using the saved data.

With save_interval > 1, this only saves every Nth step to reduce array copying overhead,
but always saves steps near rational surfaces (beginning and end of each integration segment).
"""
function integrator_callback!(integrator)

    # unpack parameters. Note the 2 unused items are needed to match the signature in the integrand sing_der!
    ctrl, _, _, intr, odet, chunk = integrator.p

    # Count every ODE step taken (not just saved ones)
    odet.total_steps += 1

    # Update integration tolerances
    integrator.opts.reltol = compute_tols(ctrl, intr, odet, chunk.ising)

    # Check if the solution norms are above a threshold, if so apply Gaussian reduction
    compute_solution_norms!(integrator.u, odet, ctrl, intr, false)

    # Determine if we should save this step
    # Always save if:
    # 1. First few steps of integration (ensures we capture point right after rational/axis)
    # 2. Every Nth step (save_interval)
    # 3. Near the end of integration segment (ensures we capture point right before next rational)

    # Check if we're near the end of this integration segment
    psi_range = abs(integrator.sol.prob.tspan[2] - integrator.sol.prob.tspan[1])
    psi_remaining = abs(integrator.sol.prob.tspan[2] - integrator.t)
    near_end = psi_remaining < 0.05 * psi_range || psi_remaining < 1e-4

    # Check if we're at the beginning (first 2 steps capture the point right after rational)
    # Count steps within this segment (not global step count)
    steps_in_segment = length(integrator.sol.t)
    near_start = steps_in_segment <= 2

    # Save if interval condition met, or near start/end
    should_save = near_start || near_end || (odet.step % ctrl.save_interval == 0)

    if should_save
        # Grow arrays if out of storage space
        if odet.step >= size(odet.u_store, 4)
            resize_storage!(odet)
        end
        odet.psi_store[odet.step] = integrator.t
        @views odet.u_store[:, :, :, odet.step] .= integrator.u
        odet.q_store[odet.step] = odet.q # these two were set in sing_der!
        @views odet.ud_store[:, :, :, odet.step] .= odet.ud

        # Advance stepper (just like in Fortran, a "step" starts with integration, does callback functions, then stores)
        odet.step += 1
    end
end

"""
    compute_tols(ctrl::ForceFreeStatesControl, intr::ForceFreeStatesInternal, odet::OdeState)

Compute relative and absolute tolerances for the ODE solver based on proximity
to singular surfaces and magnitude of the solution vectors. In Fortran, this was
previously a part of ode_step, and called every integration step due to LSODE's
one-step mode. Here, we call it within the integrator callback to achieve the same
functionality.

### Arguments

  - ising: Int index of the singular surface to process

### TODOs

Support for `kin_flag`
Check sensitivity of results to tolerances, currently using same logic as Fortran
Add back absolute tolerance calculation if needed

### Returns

  - rtol: Relative tolerance
"""
function compute_tols(ctrl::ForceFreeStatesControl, intr::ForceFreeStatesInternal, odet::OdeState, ising::Int)
    singfac_local = Inf
    # Relative tolerance
    if false  # kin_flag (not implemented)
    # Insert kin_flag branch if needed
    else
        # singfac = m - nq = n(m/n - q) = n (q_res - q), use smallest n to be conservative
        # Note: odet.q is updated within the derivative calculation
        if ising > 0 && ising <= intr.msing
            singfac_local = abs(minimum(intr.sing[ising].n) * (intr.sing[ising].q - odet.q))
        end
        # If in between singular surfaces, check distance to both
        if ising > 1
            singfac_local = min(singfac_local, abs(minimum(intr.sing[ising-1].n) * (intr.sing[ising-1].q - odet.q)))
        end
    end
    rtol = tol = singfac_local < ctrl.crossover ? ctrl.tol_r : ctrl.tol_nr
    # Absolute tolerances (not used for now, if so, will need to pass in integrator.u)
    # atol = similar(odet.u, Float64)
    # for ieq in 1:size(odet.u, 3), isol in 1:size(odet.u, 2)
    #     @views atol0 = maximum(abs, odet.u[:, isol, ieq]) * tol
    #     atol0 == 0 && (atol0 = Inf)
    #     atol[:, isol, ieq] .= atol0
    # end
    return rtol
end

"""
    compute_solution_norms!(u::Array{ComplexF64,3}, odet::OdeState, ctrl::ForceFreeStatesControl, intr::ForceFreeStatesInternal, sing_flag::Bool)

Computes norms of the solution vectors of the array `u` and normalizes them
if this is not the first call after a fixup. Formerly `ode_unorm!`.
Throws an error if any vector norm is zero. It then compares the variation in norms
relative to initial values after a fixup, and applies the Gaussian reduction via
`apply_gaussian_reduction!` if the variation exceeds `ctrl.ucrit` or if `sing_flag` is true.
Performs the same function as `ode_unorm` in the Fortran code, with minor differences in indexing
and array handling.

Note that we pass `u` as an argument to reduce the number of copies that are
necessary in the integration callback, since otherwise we'd need to copy from
the integrator state to `odet.u` and then back again. This way, we can just
operate on `u` directly without extra copies.

### Arguments

  - u: Current solution vector array, updated in-place if fixfac is called
  - sing_flag: Indicates if normalization is occuring at a singular surface or not

### TODOs

Add resizing logic for unorm arrays when ifix exceeds allocated size
"""
function compute_solution_norms!(u::Array{ComplexF64,3}, odet::OdeState, ctrl::ForceFreeStatesControl, intr::ForceFreeStatesInternal, sing_flag::Bool)

    # Compute norms of first solution vectors, abort if any are zero
    odet.unorm .= norm.(eachcol(u[:, :, 1]))
    if minimum(odet.unorm) == 0
        jmax = argmin(odet.unorm)
        error("One of the first solution vector norms unorm(1,$jmax) = 0")
    end

    # Normalize unorm and perform Gaussian reduction if required
    if odet.new
        odet.new = false
        odet.unorm0 .= odet.unorm
    else
        odet.unorm ./= odet.unorm0
        uratio = maximum(odet.unorm) / minimum(odet.unorm)
        if uratio > ctrl.ucrit || sing_flag
            # TODO: add resizing logic here as well
            if odet.ifix < ctrl.numunorms_init
                odet.ifix += 1
            else
                @warn "unorm storage reached, no longer saving fixfac data. Stability outputs and unorming will be correct, but cannot reconstruct `u`. \n
                Increase `numunorms_init` if needed. Automatic resizing will be added in a future version."
            end
            apply_gaussian_reduction!(u, odet, intr, sing_flag)
            odet.new = true
        end
    end
end

"""
    apply_gaussian_reduction!(u::Array{ComplexF64,3}, odet::OdeState, intr::ForceFreeStatesInternal, sing_flag::Bool)

Applies Gaussian reduction to orthogonalize solution vectors in `u`.
Formerly `ode_fixup!`. Performs the same function as `ode_fixup` in the Fortran code,
except now relevant `fixfac` data are stored in memory instead of dumped to `euler.bin`.
Used when the spread in norms exceeds a threshold or when a rational surface is reached.
This will update both `u` and relevant fields in `odet` in-place. See the
description of `compute_solution_norms!` for more details on the benefits of in-place `u`
updates.
"""
function apply_gaussian_reduction!(u::Array{ComplexF64,3}, odet::OdeState, intr::ForceFreeStatesInternal, sing_flag::Bool)

    # Store data for the current fixup
    ifix = odet.ifix
    odet.sing_flag[ifix] = sing_flag
    # Since the current step has been fixed-up, we denote the end of the previous
    # fixup region as the previous step (this just avoids a -1 index later)
    odet.fixstep[ifix] = odet.step - 1

    # Initialize fixfac
    for isol in 1:intr.numpert_total
        odet.fixfac[isol, isol, ifix] = 1
    end
    # Sort unorm in descending order (since we triangularize from largest to smallest)
    odet.index[:, ifix] = sortperm(odet.unorm; rev=true)

    # Triangularize primary solutions
    mask = trues(2, intr.numpert_total)
    for isol in 1:intr.numpert_total
        ksol = odet.index[isol, ifix]
        mask[2, ksol] = false
        # Set pivot row based on max location
        @views kpert = argmax(abs.(u[:, ksol, 1]) .* mask[1, :])
        mask[1, kpert] = false
        # Eliminate other solution vectors below the pivot
        for jsol in 1:intr.numpert_total
            if mask[2, jsol]
                odet.fixfac[ksol, jsol, ifix] = -u[kpert, jsol, 1] / u[kpert, ksol, 1]
                @. @views u[:, jsol, :] .= u[:, jsol, :] .+ u[:, ksol, :] .* odet.fixfac[ksol, jsol, ifix]
                u[kpert, jsol, 1] = 0
            end
        end
    end
end

"""
    findmax_dW_edge!(odet::OdeState, ctrl::ForceFreeStatesControl, equil::Equilibrium.PlasmaEquilibrium, ffit::FourFitVars, intr::ForceFreeStatesInternal)

Records the total dW in the integration region between `ctrl.psiedge` and
`ctrl.psilim`. This performs the same function as `ode_record_edge` in the
Fortran, but everything is now done post-integration which cleans up the logic,
i.e. no "_edge" arrays.

The dW is stored at that step index in `odet.dW_edge`; because we initialize
dW_edge to -Inf, we can just take the max value after integration to get the
total dW at the edge and avoid unphysical kink modes that might occur just
inside rational surfaces.

We have also separated the computation of the wv matrix spline and the total dW
calculation into `free_compute_wv_spline` and `free_compute_total` respectively
for clarity. We create the wv matrix spline once prior to the loop.
"""
function findmax_dW_edge!(odet::OdeState, ctrl::ForceFreeStatesControl, equil::Equilibrium.PlasmaEquilibrium, ffit::FourFitVars, intr::ForceFreeStatesInternal)

    # Since we search for the maximum dW, initialize to -Infinity.
    # Resize to match current step count (integration may have grown beyond numsteps_init).
    resize!(odet.dW_edge, odet.step)
    fill!(odet.dW_edge, -Inf * (1 + im))

    # Create a rough spline for wv matrix between psiedge -> psilim so we can approximate dW
    odet.wvmat = free_compute_wv_spline(ctrl, equil, intr)

    # Loop through integration, compute dW at steps where psifac >= psiedge
    for istep in 1:odet.step
        odet.psifac = odet.psi_store[istep]
        if odet.psifac >= ctrl.psiedge
            odet.u .= odet.u_store[:, :, :, istep]
            try
                odet.dW_edge[istep] = free_compute_total(equil, ffit, intr, odet)
            catch e
                e isa LinearAlgebra.SingularException || rethrow(e)
                # U₁ is singular at this step (degenerate near-surface solution) — skip it.
                # dW_edge remains -Inf so this step is ignored in the max search.
            end
        end
    end

    # Extract psi and et[1] for all edge-zone steps (where dW_edge was computed)
    # before storage is trimmed to the peak-dW step.  psi_store still has the full range here.
    edge_mask = .!isinf.(real.(odet.dW_edge))
    odet.psi_edge_scan = odet.psi_store[edge_mask]
    odet.et_edge_scan  = odet.dW_edge[edge_mask]

    # Return the index that maximizes dW_edge to identify truncation point
    return findmax(real.(odet.dW_edge))[2]
end

"""
    transform_u!(odet::OdeState, intr::ForceFreeStatesInternal)

Constructs the transformation matrices to form the true solution vectors. Effectively
"undoes" the Gaussian reduction applied during fixups throughout the integration, such that
we have the true solution vectors for use in GPEC. Modifies the store arrays in `odet` in-place.
Performs a similar function as `idcon_transform` + `idcon_build` in the Fortran code,
except we separate the building of the transformation matrices and determining the coefficients
for a chosen force-free solution, which can be done in postprocessing.
"""
function transform_u!(odet::OdeState, intr::ForceFreeStatesInternal)

    # Gaussian reduction matrices for each fixup
    gauss = Array{ComplexF64,3}(undef, intr.numpert_total, intr.numpert_total, odet.ifix)
    # Transformation matrices for each region between fixups (ifix + 1 regions)
    transforms = Array{ComplexF64,3}(undef, intr.numpert_total, intr.numpert_total, odet.ifix + 1)
    # Temporary workspace matrices
    gauss_buffer = Matrix{ComplexF64}(undef, intr.numpert_total, intr.numpert_total)
    identity = Matrix{ComplexF64}(I, intr.numpert_total, intr.numpert_total)
    temp = Matrix{ComplexF64}(undef, intr.numpert_total, intr.numpert_total)
    mask = trues(intr.numpert_total)

    # Construct gaussian reduction matrices for each fixup
    for ifix in 1:odet.ifix
        gauss[:, :, ifix] .= identity
        mask .= true
        for isol in 1:intr.numpert_total
            ksol = odet.index[isol, ifix]
            mask[ksol] = false
            temp .= identity
            for jsol in 1:intr.numpert_total
                if mask[jsol]
                    temp[ksol, jsol] = odet.fixfac[ksol, jsol, ifix]
                end
            end
            mul!(gauss_buffer, view(gauss,:,:,ifix), temp)
            gauss[:, :, ifix] .= gauss_buffer
        end
        # Account for zeroed indices at singular surfaces in `ode_ideal_cross`
        if odet.sing_flag[ifix]
            for zeroed_idx in odet.zeroed_idx[ifix]
                gauss[:, odet.index[zeroed_idx, ifix], ifix] .= 0.0
            end
        end
    end

    # Concatenate gaussian reduction matrix to form transform matrix for each region
    # Here, the i'th region is between the (i-1)'th and i'th fixup e.g. transforms[:, :, 1]
    # is the transform matrix for the region between init and first fixup
    # and mfix + 1 is the for the region after the last fixup and before the edge
    transforms[:, :, end] .= identity
    for ifix in odet.ifix:-1:1
        mul!(view(transforms,:,:,ifix), view(gauss,:,:,ifix), view(transforms,:,:,(ifix+1)))
    end

    # Now that we have the transform matrices, we can apply them to the solution vectors
    # "undoing" the Gaussian reductions to get the true solution vectors
    jfix = 1
    for ifix in 1:(odet.ifix+1)
        jfix > odet.step && break
        # If after the last fixup (or the fixup is beyond trimmed storage), go to stored end
        kfix = ifix != odet.ifix + 1 ? min(odet.fixstep[ifix], odet.step) : odet.step
        @views for istep in jfix:kfix
            # This is u1->u4 in Fortran
            mul!(gauss_buffer, odet.u_store[:, :, 1, istep], transforms[:, :, ifix])
            odet.u_store[:, :, 1, istep] .= gauss_buffer
            mul!(gauss_buffer, odet.u_store[:, :, 2, istep], transforms[:, :, ifix])
            odet.u_store[:, :, 2, istep] .= gauss_buffer
            mul!(gauss_buffer, odet.ud_store[:, :, 1, istep], transforms[:, :, ifix])
            odet.ud_store[:, :, 1, istep] .= gauss_buffer
            mul!(gauss_buffer, odet.ud_store[:, :, 2, istep], transforms[:, :, ifix])
            odet.ud_store[:, :, 2, istep] .= gauss_buffer
        end
        jfix = kfix + 1
    end
end
