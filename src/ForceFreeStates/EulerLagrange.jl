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

    # Initialization: create minimal OdeState just to compute starting position and chunks
    odet = OdeState(intr.numpert_total, 1, ctrl.numunorms_init, intr.msing)
    if ctrl.sing_start <= 0
        initialize_el_at_axis!(odet, ctrl, equil.profiles, intr)
    elseif ctrl.sing_start <= intr.msing
        error("sing_start > 0 not implemented yet!")
        # initialize_el_at_singular_surf!(ctrl, equil, intr, odet)
    else
        error("Invalid value for sing_start: $(ctrl.sing_start) > msing = $(intr.msing)")
    end

    # Pre-compute all integration chunks from the initial position
    chunks = chunk_el_integration_bounds(odet.psifac, odet.ising_start, ctrl, intr)

    # Pre-allocate storage arrays with exact size: each chunk contributes save_npoints_per_chunk
    # solution points, plus one point per singular surface crossing
    n_per_chunk = max(2, ctrl.save_npoints_per_chunk)
    total_steps = length(chunks) * n_per_chunk + count(c -> c.needs_crossing, chunks)
    np = intr.numpert_total
    odet.psi_store = Vector{Float64}(undef, total_steps)
    odet.u_store = Array{ComplexF64}(undef, np, np, total_steps, 2)
    odet.ud_store = Array{ComplexF64}(undef, np, np, total_steps, 2)
    odet.crit_store = Vector{Float64}(undef, total_steps)
    odet.dW_edge = Array{ComplexF64}(undef, total_steps)

    # Print initial integration condition
    if ctrl.verbose
        println("   ψ = $((@sprintf "%.3f" odet.psifac)),  q = $((@sprintf "%.3f" equil.profiles.q_spline(odet.psifac)))")
    end

    # Iterate through each integration chunk
    for chunk in chunks
        integrate_el_region!(odet, ctrl, equil, ffit, intr, chunk)

        # Cross a rational surface after integration if this chunk requires it
        if chunk.needs_crossing
            if ctrl.verbose
                println("   ψ = $((@sprintf "%.3f" odet.psifac)),  q= $((@sprintf "%.3f" odet.q)),  steps = $(odet.solver_steps)")
            end
            if ctrl.kin_flag
                error("kin_flag = true not implemented yet!")
            else
                cross_ideal_singular_surf!(odet, ctrl, equil, ffit, intr, chunk.ising)
            end
        end
    end
    if ctrl.verbose
        println("   ψ = $((@sprintf "%.3f" odet.psifac)),  q= $((@sprintf "%.3f" equil.profiles.q_spline(odet.psifac))),  steps = $(odet.solver_steps)")
    end

    # Deallocate unused storage of integration data
    if ctrl.psiedge < intr.psilim
        # Find the peak dW in the edge region and truncate integration data there
        odet.step = findmax_dW_edge!(odet, ctrl, equil, ffit, intr)
        trim_storage!(odet)
        if ctrl.verbose
            q_at_step = equil.profiles.q_spline(odet.psi_store[odet.step])
            println("Truncating integration at peak edge dW: ψ = $((@sprintf "%.2f" odet.psi_store[odet.step])),  q = $((@sprintf "%.2f" q_at_step))")
        end

        # Update u, psilim, and qlim for usage in determining wp and wt
        intr.psilim = odet.psi_store[end]
        intr.qlim = equil.profiles.q_spline(odet.psi_store[end])
        odet.u .= odet.u_store[:, :, end, :]
    else
        odet.step -= 1 # step was incremented one extra time in integrate_el_region!
        trim_storage!(odet)
    end

    # Evaluate stability criterion (critical determinant) of saved solutions
    if ctrl.verbose
        println("Evaluating fixed-boundary stability criterion")
    end
    odet.nzero = evaluate_stability_criterion!(odet, equil.profiles)

    # Form the true solution vectors, undoing the Gaussian reduction applied during integration
    transform_u!(odet, intr)

    # Compute ud_store post-integration from the transformed solution arrays.
    # ud_store (ξ'_ψ and ξ_s) is computed by evaluating sing_der! on each stored u, so it is
    # consistent with the transformed u_store and can be used directly by downstream code.
    dummy_chunk = IntegrationChunk(0.0, 0.0, false, 0)
    params = (ctrl, equil, ffit, intr, odet, dummy_chunk)
    u_tmp = zeros(ComplexF64, intr.numpert_total, intr.numpert_total, 2)
    du_tmp = zeros(ComplexF64, intr.numpert_total, intr.numpert_total, 2)
    odet.spline_hint[] = 1
    ffit._hint[] = 1
    for istep in 1:odet.step
        psi_val = odet.psi_store[istep]
        u_tmp .= odet.u_store[:, :, istep, :]
        sing_der!(du_tmp, u_tmp, params, psi_val)
        odet.ud_store[:, :, istep, :] .= odet.ud
    end

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
        # Refine psifac using Newton iteration
        converged = false
        for _ in 1:itmax
            dpsi = (ctrl.qlow - profiles.q_spline(odet.psifac)) / profiles.q_deriv(odet.psifac)
            odet.psifac += dpsi
            abs(dpsi) < eps * abs(odet.psifac) && (converged=true; break)
        end
        if !converged
            error("Newton iteration for psifac did not converge after $itmax iterations.")
        end
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
    chunk_el_integration_bounds(psifac, ising_start, ctrl, intr)

Pre-compute all integration chunks from the current position to the edge.
Returns a vector of `IntegrationChunk` objects, one per inter-rational region plus a
final chunk to the edge. Each chunk runs from just after the previous crossing to just
before the next one. Adaptive Gaussian reduction fires within each chunk via the
`gaussian_reduction_callback!` DiscreteCallback rather than sub-chunk boundaries.

### Arguments

  - `psifac::Float64` - Starting psi position
  - `ising_start::Int` - Index of singular surface to start from
  - `ctrl::ForceFreeStatesControl` - Control parameters
  - `intr::ForceFreeStatesInternal` - Internal data (singular surfaces, limits)

### Returns

  - `Vector{IntegrationChunk}` - Array of integration chunks to process

### TODOs

Support for `kin_flag`
"""
function chunk_el_integration_bounds(psifac::Float64, ising_start::Int, ctrl::ForceFreeStatesControl, intr::ForceFreeStatesInternal)
    chunks = IntegrationChunk[]

    # Start from current position
    psi_current = psifac
    ising_current = ising_start

    # Find next singular surface that is resonant and within integration limits
    function find_next_resonant_surface(ising::Int)
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
        ising_current = find_next_resonant_surface(ising_current)
        while ising_current <= intr.msing && intr.psilim >= intr.sing[ising_current].psifac && ctrl.singfac_min != 0
            # Set integration limit to just before the next singular surface
            psi_end = intr.sing[ising_current].psifac - ctrl.singfac_min /
                                                        abs(minimum(intr.sing[ising_current].n) * intr.sing[ising_current].q1)

            # Validate chunk bounds
            @assert psi_current < psi_end "Invalid chunk bounds: psi_start=$psi_current >= psi_end=$psi_end"
            @assert isempty(chunks) || psi_current >= chunks[end].psi_end "Overlapping chunks detected"

            push!(chunks, IntegrationChunk(;
                psi_start      = psi_current,
                psi_end        = psi_end,
                needs_crossing = true,
                ising          = ising_current
            ))

            # After crossing, we jump to the other side of the singular surface
            dpsi = intr.sing[ising_current].psifac - psi_end
            psi_current = psi_end + 2 * dpsi

            # Move to next singular surface that is either resonant or beyond integration limits
            ising_current = find_next_resonant_surface(ising_current)
        end

        # No more singular surfaces to cross, add final region to edge
        final_end = intr.psilim * (1 - eps)
        @assert psi_current < final_end "Final chunk has invalid bounds"
        @assert isempty(chunks) || psi_current >= chunks[end].psi_end "Final chunk overlaps with previous chunk"

        push!(chunks, IntegrationChunk(;
            psi_start      = psi_current,
            psi_end        = final_end,
            needs_crossing = false,
            ising          = 0
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

    # Sort by growth ratio from the post-previous-crossing baseline and apply Gaussian reduction.
    # The column with the largest growth ratio is the resonant solution to zero and replace
    # with the asymptotic solution on the other side of the surface.
    apply_gaussian_reduction!(odet.u, odet, intr, true, odet.step - 1)

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

    # Record post-crossing norms as baseline for the next crossing's growth-ratio sort.
    # After reinit, the asymptotic column is small and the other columns are at their
    # current (post-reduction) magnitudes. Using these as the denominator ensures that
    # the next crossing sorts by how much each solution has grown since this point.
    odet.unorm0 .= norm.(eachcol(odet.u[:, :, 1]))

    # Store psi and u at the crossing point (ud is computed post-integration)
    odet.psi_store[odet.step] = odet.psifac
    odet.u_store[:, :, odet.step, :] = odet.u
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
Formerly `ode_step!`. Uses OrdinaryDiffEq's built-in `saveat` with ldp-spacing
(`sin²(i/(N-1)·π/2)`) to cluster save points near both ends of each chunk, where
solutions grow fastest near rational surfaces.

A `DiscreteCallback` fires `gaussian_reduction_callback!` at each accepted ODE step,
applying adaptive Gaussian reduction when the column growth ratio exceeds `ctrl.ucrit`.
This replaces the per-chunk end-of-integration reduction from the develop branch.

### Arguments

  - `odet::OdeState` - ODE state struct (modified in-place)
  - `ctrl::ForceFreeStatesControl` - Control parameters
  - `equil::Equilibrium.PlasmaEquilibrium` - Plasma equilibrium
  - `ffit::FourFitVars` - Fourier fit variables
  - `intr::ForceFreeStatesInternal` - Internal data
  - `chunk::IntegrationChunk` - Integration chunk containing start and end ψ for integration
"""
function integrate_el_region!(odet::OdeState, ctrl::ForceFreeStatesControl, equil::Equilibrium.PlasmaEquilibrium, ffit::FourFitVars, intr::ForceFreeStatesInternal, chunk::IntegrationChunk)

    # ldp-spaced saveat: sin²(i/(N-1)·π/2) clusters points near both ends of the chunk,
    # where solution growth is fastest (near rational surfaces)
    N = max(2, ctrl.save_npoints_per_chunk)
    saveat_psi = [chunk.psi_start + (chunk.psi_end - chunk.psi_start) * sin(i / (N - 1) * π / 2)^2
                  for i in 0:(N - 1)]

    # Record global step offset so the callback can compute fixstep correctly
    odet.chunk_step_offset = odet.step - 1

    prob = ODEProblem(sing_der!, odet.u, (chunk.psi_start, chunk.psi_end), (ctrl, equil, ffit, intr, odet, chunk))
    # save_positions=(false, false): callback must not add extra points to sol.t;
    # only the saveat grid controls what gets saved.
    cb = DiscreteCallback((_, _, _) -> true, gaussian_reduction_callback!; save_positions=(false, false))
    sol = solve(prob, BS5(); reltol=ctrl.eulerlagrange_tolerance, saveat=saveat_psi,
                save_everystep=false, callback=cb)
    odet.solver_steps += sol.stats.naccept

    # Append saved solution points to storage arrays (only u and psi; ud computed post-integration)
    for i in eachindex(sol.t)
        odet.psi_store[odet.step] = sol.t[i]
        @views odet.u_store[:, :, odet.step, :] .= sol.u[i]
        odet.step += 1
    end

    # Update current state to end of chunk
    odet.u .= sol.u[end]
    odet.psifac = sol.t[end]
end

"""
    gaussian_reduction_callback!(integrator)

`DiscreteCallback` function fired at each accepted ODE step. Checks if solution columns
have grown disproportionately relative to the post-crossing or post-reduction baseline
stored in `odet.unorm0`, and fires `apply_gaussian_reduction!` when the growth ratio
exceeds `ctrl.ucrit`. Does not switch tolerances or save solution data.

The `fixstep` for the reduction is `odet.chunk_step_offset + length(integrator.sol.t)`,
which is the count of saveat points already appended to `sol.t` before the current step
(DiscreteCallback fires after step acceptance but before the current saveat point is added).
"""
function gaussian_reduction_callback!(integrator)
    ctrl, _, _, intr, odet, _ = integrator.p
    u = integrator.u

    current_norms = norm.(eachcol(u[:, :, 1]))

    # u[:,:,1] starts at zero (axis init); skip until it has grown
    minimum(current_norms) == 0.0 && return

    uratio = maximum(current_norms ./ odet.unorm0) / minimum(current_norms ./ odet.unorm0)

    if uratio > ctrl.ucrit
        # fixstep = count of saveat points saved before the current step
        n_saved = length(integrator.sol.t)
        fixstep_val = odet.chunk_step_offset + n_saved
        if odet.ifix < ctrl.numunorms_init
            apply_gaussian_reduction!(u, odet, intr, false, fixstep_val)
            # Reset baseline so next check measures growth since this reduction
            odet.unorm0 .= norm.(eachcol(u[:, :, 1]))
        else
            @warn "unorm storage reached, Gaussian reduction skipped. Increase numunorms_init."
        end
    end
end

"""
    apply_gaussian_reduction!(u::Array{ComplexF64,3}, odet::OdeState, intr::ForceFreeStatesInternal, sing_flag::Bool, fixstep_val::Int)

Applies Gaussian reduction to orthogonalize solution vectors in `u` at a singular
surface crossing. Formerly `ode_fixup!`. Performs the same function as `ode_fixup`
in the Fortran code, except relevant `fixfac` data are stored in memory instead of
dumped to `euler.bin`. Columns are sorted by their current norm (largest first) to
identify the resonant solution that has grown fastest since the previous crossing;
that column is eliminated and replaced by the asymptotic solution on the other side.
This updates both `u` and relevant fields in `odet` in-place.

### Arguments

  - `fixstep_val::Int` - Global step index to record as the boundary of this fixup region.
    For crossings, pass `odet.step - 1`. For callback-triggered reductions, pass
    `odet.chunk_step_offset + length(integrator.sol.t)`.
"""
function apply_gaussian_reduction!(u::Array{ComplexF64,3}, odet::OdeState, intr::ForceFreeStatesInternal, sing_flag::Bool, fixstep_val::Int)

    # Increment and store fixup data
    odet.ifix += 1
    if odet.ifix > length(odet.sing_flag)
        error("Gaussian reduction storage exceeded. Increase numunorms_init (currently $(length(odet.sing_flag))).")
    end
    ifix = odet.ifix
    odet.sing_flag[ifix] = sing_flag
    odet.fixstep[ifix] = fixstep_val

    # Initialize fixfac to identity
    for isol in 1:intr.numpert_total
        odet.fixfac[isol, isol, ifix] = 1
    end

    # Sort columns by growth ratio (current norm / post-crossing baseline norm), largest first.
    # The column with the largest ratio has grown the most since the last crossing and is the
    # resonant solution to zero. Using ratios rather than absolute norms correctly handles the
    # case where the asymptotic solution introduced at the previous crossing starts small and
    # grows to dominance — it may not have the largest absolute norm yet, but it has the
    # largest growth ratio relative to the post-crossing baseline stored in odet.unorm0.
    current_norms = norm.(eachcol(u[:, :, 1]))
    odet.index[:, ifix] = sortperm(current_norms ./ odet.unorm0; rev=true)

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

    # Since we search for the maximum dW, initialize to -Infinity
    fill!(odet.dW_edge, -Inf * (1 + im))

    # Create a rough spline for wv matrix between psiedge -> psilim so we can approximate dW
    odet.wvmat = free_compute_wv_spline(ctrl, equil, intr)

    # Loop through integration, compute dW at steps where psifac >= psiedge
    for istep in 1:odet.step
        odet.psifac = odet.psi_store[istep]
        if odet.psifac >= ctrl.psiedge
            odet.u .= odet.u_store[:, :, istep, :]
            odet.dW_edge[istep] = free_compute_total(equil, ffit, intr, odet)
        end
    end

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
    # "undoing" the Gaussian reductions to get the true solution vectors.
    # ud_store is not transformed here; it is computed post-transformation from the
    # transformed u_store via a sing_der! pass in eulerlagrange_integration.
    jfix = 1
    for ifix in 1:(odet.ifix+1)
        # If after the last fixup, go to the end of integration
        kfix = ifix != odet.ifix + 1 ? odet.fixstep[ifix] : odet.step
        @views for istep in jfix:kfix
            # This is u1->u4 in Fortran
            mul!(gauss_buffer, odet.u_store[:, :, istep, 1], transforms[:, :, ifix])
            odet.u_store[:, :, istep, 1] .= gauss_buffer
            mul!(gauss_buffer, odet.u_store[:, :, istep, 2], transforms[:, :, ifix])
            odet.u_store[:, :, istep, 2] .= gauss_buffer
        end
        jfix = kfix + 1
    end
end
