"""
    `ode_run(ctrl::DconControl, equil::Equilibrium.PlasmaEquilibrium, intr::DconInternal)`

Main driver for integrating plasma equilibrium and detecting singular surfaces.
Has the same functionality as `ode_run` in the Fortran code, with the addition of
a single dump to the `euler.h5` file at the end of integration instead of multiple dumps
to `euler.bin` throughout the integration. We have made the control logic more clear
including making a clear end condition to the while loop and implementing the unorming
and logic within a callback in the integration. We now perform significant post-processing
after integration including finding the peak dW in the edge region and evaluating the
stability criterion over the entire integration, which were previously done during
integration in the Fortran code.

### TODOs

Support for `kin_flag`
restype functionality if we decide to do this

### Returns

An OdeState struct containing the final state of the ODE solver after integration is complete.
"""
function ode_run(ctrl::DconControl, equil::Equilibrium.PlasmaEquilibrium, ffit::FourFitVars, intr::DconInternal)

    # Initialization
    odet = OdeState(intr.numpert_total, ctrl.numsteps_init, ctrl.numunorms_init, intr.msing)
    if ctrl.sing_start <= 0
        ode_axis_init!(odet, ctrl, equil, intr)
    elseif ctrl.sing_start <= intr.msing
        error("sing_start > 0 not implemented yet!")
        # ode_sing_init!(ctrl, equil, intr, odet)
    else
        error("Invalid value for sing_start: $(ctrl.sing_start) > msing = $(intr.msing)")
    end

    if ctrl.verbose # mimicing output from ode_output_open
        println("   ψ = $((@sprintf "%.3f" odet.psifac)),  q = $((@sprintf "%.3f" Spl.spline_eval!(equil.sq, odet.psifac)[4]))")
    end

    # Always integrate once, even if no rational surfaces are crossed
    ode_step!(odet, ctrl, equil, ffit, intr)

    # If at a rational surface, do the appropriate crossing routine, then integrate again
    while odet.ising != ctrl.ksing && odet.next == "cross"

        if ctrl.kin_flag
            error("kin_flag = true not implemented yet!")
        else
            ode_ideal_cross!(odet, ctrl, equil, ffit, intr)
        end

        ode_step!(odet, ctrl, equil, ffit, intr)
    end

    # Deallocate unused storage of integration data
    if ctrl.psiedge < intr.psilim
        # Find the peak dW in the edge region and truncate integration data there
        odet.step = findmax_dW_edge!(odet, ctrl, equil, ffit, intr)
        trim_storage!(odet)
        if ctrl.verbose
            println("Truncating integration at peak edge dW: ψ = $((@sprintf "%.2f" odet.psi_store[odet.step])),  q = $((@sprintf "%.2f" odet.q_store[odet.step]))")
        end

        # Update u, psilim, and qlim for usage in determining wp and wt
        intr.psilim = odet.psi_store[end]
        intr.qlim = odet.q_store[end]
        odet.u .= odet.u_store[:, :, :, end]
    else
        odet.step -= 1 # step was incremented one extra time in ode_step!
        trim_storage!(odet)
    end

    # Evaluate stability criterion (critical determinant) of saved solutions
    if ctrl.verbose
        println("Evaluating fixed-boundary stability criterion")
    end
    odet.nzero = evaluate_stability_criterion!(odet, equil)

    # Form the true solution vectors, undoing the Gaussian reduction applied in `ode_unorm!` during integration
    transform_u!(odet, intr)

    return odet
end

"""
    ode_axis_init!(odet::OdeState, ctrl::DconControl, equil::Equilibrium.PlasmaEquilibrium, intr::DconInternal)

Initialize the OdeState struct for the case of sing_start = 0 (axis initialization). This includes
determining `psifac`, `psimax`, `ising`, `singfac`, and initializing `u`.

### TODOs

Support for `kin_flag`
"""
function ode_axis_init!(odet::OdeState, ctrl::DconControl, equil::Equilibrium.PlasmaEquilibrium, intr::DconInternal)

    # Shorthand to evaluate q/q1 inside newton iteration
    qval = psi -> Spl.spline_eval!(equil.sq, psi)[4]
    q1val = psi -> Spl.spline_deriv1!(equil.sq, psi)[2][4]

    # Preliminary computations
    odet.psifac = equil.sq.xs[1]

    # Use Newton iteration to find starting psi if qlow is above q0
    if ctrl.qlow > equil.sq.fs[1, 4]
        # Find last index where q < qlow 
        idx = findlast(jpsi -> equil.sq.fs[jpsi-1, 4] < ctrl.qlow, 2:equil.sq.mx)
        if idx !== nothing
            odet.psifac = equil.sq.xs[idx]
        end
        # Refine psifac using Newton iteration
        converged = false
        for _ in 1:itmax
            dpsi = (ctrl.qlow - qval(odet.psifac)) / q1val(odet.psifac)
            odet.psifac += dpsi
            abs(dpsi) < eps * abs(odet.psifac) && (converged = true; break)
        end
        if !converged
            error("Newton iteration for psifac did not converge after $itmax iterations.")
        end
    end

    # Find inner singular surface (where sing.psifac > psi(qlow/q0))
    if false #(TODO: kin_flag)
    # for ising = 1:kmsing
    #     if kinsing[ising].psifac > psifac
    #         break
    #     end
    # end
    else
        odet.ising = searchsortedfirst(getfield.(intr.sing, :psifac), odet.psifac) - 1  
    end

    # Find next singular surface
    if false
        # TODO: (kin_flag)
    else
        # Find next singular surface (either next one in the list or outside integration limits)
        # TODO: clean this up in integration bounds PR, this exact block appears several times
        # TODO: Based on the existing logic, I don't think checking if mlow <= m <= mhigh is necessary,
        # since DCON sets the poloidal mode numbers to include the resonant modes anyway. However, based
        # on discussion with Nik, eventually we might just want to allow the user to set mlow/mhigh, in
        # which case this check would be necessary. In 3D, this might be even more applicable since rational
        # surfaces will be more dense so we might not set out mode spectrum to include all resonances.
        while true
            odet.ising += 1
            if odet.ising > intr.msing || intr.psilim < intr.sing[min(odet.ising, intr.msing)].psifac
                break
            end
            if any(m -> intr.mlow <= m <= intr.mhigh, intr.sing[odet.ising].m)
                break
            end
        end
        # Determine psimax and classify next integration limit type
        if odet.ising > intr.msing || intr.psilim < intr.sing[odet.ising].psifac || ctrl.singfac_min == 0
            odet.psimax = intr.psilim * (1 - eps)
            odet.next = "finish"
        else
            # TODO: Nik: where does singfac_min / n * q' come from? Unclear how to generalize to multi-n
            # Safest choice for now is to use the smallest resonant n for maximum separation
            odet.psimax = intr.sing[odet.ising].psifac - ctrl.singfac_min / abs(minimum(intr.sing[odet.ising].n) * intr.sing[odet.ising].q1)
            odet.next = "cross"
        end
    end

    # Initialize solutions with the identity matrix for U_22 as described in [Glasser PoP 2016] Section VI
    for ipert in 1:intr.numpert_total
        odet.u[ipert, ipert, 2] = 1
    end
end

# TODO: NOT IMPLEMENTED YET! (low priority, just make sure sing_start = 0 in dcon.toml)
function ode_sing_init()
    return
end

"""
    ode_ideal_cross!(odet::OdeState, ctrl::DconControl, equil::Equilibrium.PlasmaEquilibrium, ffit::FourFitVars, intr::DconInternal)

Handle the crossing of a rational surface during ODE integration if `kin_flag` is false.
Performs the same function as `ode_ideal_cross` in the Fortran code. Differences mainly in integration data
storage logic, but otherwise identical. It normalizes and reinitializes the solution vector at the singularity,
and updates relevant state variables and updates `odet` for continued integration. It also determines the
location and parameters of the next singular surface and writes outputs as desired.

### TODOs

Remove while true logic
"""
function ode_ideal_cross!(odet::OdeState, ctrl::DconControl, equil::Equilibrium.PlasmaEquilibrium, ffit::FourFitVars, intr::DconInternal)

    # Fixup solution at singular surface
    ode_unorm!(odet.u, odet, ctrl, intr, true)

    # Get asymptotic coefficients before crossing rational surface
    odet.ca_l[:, :, :, odet.ising] .= sing_get_ca(ctrl, intr, odet)

    # Re-initialize on opposite side of rational surface
    psi_old = odet.psifac
    singp = intr.sing[odet.ising]
    dpsi = singp.psifac - odet.psifac
    odet.psifac += 2 * dpsi # jump to other side of singular surface
    ua = sing_get_ua(ctrl, intr, odet)
    ipert_res = 1 .+ singp.m .- intr.mlow .+ (singp.n .- intr.nlow) .* intr.mpert
    
    # TODO: make this comment shorter?
    # Single n: remove largest solution and sub in asymptotics on the other side
    # Multi-n: if we remove the N largest modes in arbitrary order, we can mess up the
    # diagonal structure of the matrix and later calculations. zeroed_idx let's us make sure
    # the solution vector we're zeroing corresponds to the same block as the resonant mode we
    # introduce. It is also needed when transforming u back to the full solution after integration.
    if !ctrl.con_flag
        # Eliminate the solution with the largest norm (in the same block) for each resonance
        odet.zeroed_idx[odet.ifix] = Int[]
        for i in eachindex(singp.r1)
            push!(odet.zeroed_idx[odet.ifix], findfirst(j -> (ipert_res[i] - 1) ÷ intr.mpert == (odet.index[j, odet.ifix] - 1) ÷ intr.mpert, 1:intr.numpert_total))
            odet.u[:, odet.index[odet.zeroed_idx[odet.ifix][i], odet.ifix], :] .= 0
        end
    end

    # Approximate solution vectors across singular surface
    du1 = zeros(ComplexF64, intr.numpert_total, intr.numpert_total, 2)
    du2 = zeros(ComplexF64, intr.numpert_total, intr.numpert_total, 2)
    params = (ctrl, equil, ffit, intr, odet)
    sing_der!(du1, odet.u, params, psi_old)
    sing_der!(du2, odet.u, params, odet.psifac)
    odet.u .+= (du1 .+ du2) .* dpsi

    # Apply asymptotic solution on other side of singular surface
    if !ctrl.con_flag
        for i in eachindex(singp.r1)
            # Zero out the resonant components
            odet.u[ipert_res[i], :, :] .= 0
            # Introduce the small asymptotic resonant solution on the other side of the singular surface
            odet.u[:, odet.index[odet.zeroed_idx[odet.ifix][i], odet.ifix], :] .= ua[:, ipert_res[i] + intr.numpert_total, :]
        end
    end
    # Get asymptotic coefficients after crossing rational surface
    odet.ca_r[:, :, :, odet.ising] .= sing_get_ca(ctrl, intr, odet)

    # Find next singular surface (either the next in the list or outside integration limits)
    while true
        odet.ising += 1
        if odet.ising > intr.msing || intr.psilim < intr.sing[min(odet.ising, intr.msing)].psifac
            break
        end
        if any(m -> intr.mlow <= m <= intr.mhigh, intr.sing[odet.ising].m)
            break
        end
    end
    
    # Determine psimax and classify next integration limit type
    if odet.ising > intr.msing || intr.psilim < intr.sing[odet.ising].psifac
        odet.psimax = intr.psilim * (1 - eps)
        odet.next = "finish"
    else
        odet.psimax = intr.sing[odet.ising].psifac - ctrl.singfac_min / abs(minimum(intr.sing[odet.ising].n) * intr.sing[odet.ising].q1)
    end

    # Store values after crossing step and advance
    odet.psi_store[odet.step] = odet.psifac
    odet.q_store[odet.step] = odet.q
    odet.u_store[:, :, :, odet.step] = odet.u
    odet.ud_store[:, :, :, odet.step] = odet.ud
    odet.step += 1
end

# Example stub for kinetic crossing
function ode_kin_cross()
    # Implement kinetic crossing logic here
    return
end

"""
    ode_step!(odet::OdeState, ctrl::DconControl, equil::Equilibrium.PlasmaEquilibrium, ffit::FourFitVars, intr::DconInternal)

Integrate the Euler-Lagrange equations to the next rational surface or edge.
Performs the same function as `ode_step` in the Fortran code, with the addition of
a callback function to handle tolerances, normalization, and storage at each
step of the integration. In Fortran, this was performed by running LSODE in one-step
mode (so ode_step was called hundreds of times) and calling the relevant functions in
a DO loop. Here, we use the DifferentialEquations.jl interface to achieve the same
functionality in a more Julian way. In addition to the callback logic, this function
computes and sets the next integration endpoint, and advances the solution using
an adaptive ODE solver. The state in `odet` is then updated in-place with
the solution at the new point.

### TODOs

Check sensitivity of results to tolerances, currently using same logic as Fortran
Check absolute tolerances, currently only relative tolerances are updated
"""
function ode_step!(odet::OdeState, ctrl::DconControl, equil::Equilibrium.PlasmaEquilibrium, ffit::FourFitVars, intr::DconInternal)

    # Callback to be run at every step, handles fixups, tolerances, and data storage
    cb = DiscreteCallback((u, t, integrator) -> true, integrator_callback!)

    # Advance differential equation to next singular surface or edge
    rtol = compute_tols(ctrl, intr, odet) # initial tolerances
    prob = ODEProblem(sing_der!, odet.u, (odet.psifac, odet.psimax), (ctrl, equil, ffit, intr, odet))
    sol = solve(prob, Tsit5(); reltol=rtol, callback=cb)
    # TODO: check absolute tolerances, check how sensitive outputs are to tolerances

    # Update u and psifac with the solution at the end of the interval
    odet.u .= sol.u[end]
    odet.psifac = sol.t[end]

    println("   ψ = $((@sprintf "%.3f" odet.psifac)),  q= $((@sprintf "%.3f" odet.q)),  max(u) = $((@sprintf "%.2e" maximum(abs, odet.u))),  steps = $(odet.step-1)")
end

"""
    integrator_callback!(integrator)

Callback function for ODE integrator to handle normalization, output, and storage
at each step. This handles the solution normalization logic that was previously
in a DO loop within ode_run and called every step by running LSODE in one step mode
in the Fortran code. However, we now perform the equivalent of `ode_output_step`
and `ode_record_edge` post-integration using the saved data.

With save_interval > 1, this only saves every Nth step to reduce array copying overhead,
but always saves steps near rational surfaces (beginning and end of each integration segment).
"""
function integrator_callback!(integrator)

    ctrl, equil, _, intr, odet = integrator.p

    # Update integration tolerances
    integrator.opts.reltol = compute_tols(ctrl, intr, odet)
    # integrator.opts.abstol = atol

    # Check if the solution norms are above a threshold, if so apply Gaussian reduction
    ode_unorm!(integrator.u, odet, ctrl, intr, false)

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
        # Save values
        odet.psi_store[odet.step] = integrator.t
        odet.u_store[:, :, :, odet.step] .= integrator.u
        odet.q_store[odet.step] = odet.q # these two were set in sing_der!
        odet.ud_store[:, :, :, odet.step] .= odet.ud
        # Advance stepper (just like in Fortran, a "step" starts with integration, does callback functions, then stores)
        odet.step += 1
    end
end

"""
    compute_tols(ctrl::DconControl, intr::DconInternal, odet::OdeState)

Compute relative and absolute tolerances for the ODE solver based on proximity
to singular surfaces and magnitude of the solution vectors. In Fortran, this was
previously a part of ode_step, and called every integration step due to LSODE's
one-step mode. Here, we call it within the integrator callback to achieve the same
functionality.

### TODOs

Support for `kin_flag`
Check sensitivity of results to tolerances, currently using same logic as Fortran
Add back absolute tolerance calculation if needed

### Returns

  - rtol: Relative tolerance

"""
function compute_tols(ctrl, intr, odet)
    singfac_local = Inf
    # Relative tolerance
    if false  # kin_flag (not implemented)
    # Insert kin_flag branch if needed
    else
        # singfac = m - nq = n(m/n - q) = n (q_res - q), use smallest n to be conservative
        # Note: odet.q is updated within the derivative calculation
        if odet.ising <= intr.msing
            singfac_local = abs(minimum(intr.sing[odet.ising].n) * (intr.sing[odet.ising].q - odet.q))
        end
        # If in between singular surfaces, check distance to both
        if odet.ising > 1
            singfac_local = min(singfac_local, abs(minimum(intr.sing[odet.ising-1].n) * (intr.sing[odet.ising-1].q - odet.q)))
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
    ode_unorm!(u::Array{ComplexF64,3}, odet::OdeState, ctrl::DconControl, intr::DconInternal, sing_flag::Bool)

Computes norms of the solution vectors of the array `u` and normalizes them
if this is not the first call after a fixup. Throws an error if any vector
norm is zero. It then compares the variation in norms relative to initial values
after a fixup, and applies the Gaussian reduction via `ode_fixup!` if the
variation exceeds `ctrl.ucrit` or if `sing_flag` is true. Performs the same
function as `ode_unorm` in the Fortran code, with minor differences in indexing
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
function ode_unorm!(u::Array{ComplexF64,3}, odet::OdeState, ctrl::DconControl, intr::DconInternal, sing_flag::Bool)
    
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
                Increase `numunorms_init` in dcon.toml if needed. Automatic resizing will be added in a future version."
            end
            ode_fixup!(u, odet, intr, sing_flag)
            odet.new = true
        end
    end
end

"""
    ode_fixup!(u::Array{ComplexF64,3}, odet::OdeState, intr::DconInternal, sing_flag::Bool)

Applies Gaussian reduction to orthogonalize solution vectors in `u`. Performs
the same function as `ode_fixup` in the Fortran code, except now relevant
`fixfac` data are stored in memory instead of dumped to `euler.bin`. Used when
the spread in norms exceeds a threshold or when a rational surface is reached.
This will update both `u` and relevant fields in `odet` in-place. See the
description of `ode_unorm!` for more details on the benefits of in-place `u`
updates.

"""
function ode_fixup!(u::Array{ComplexF64,3}, odet::OdeState, intr::DconInternal, sing_flag::Bool)

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
    findmax_dW_edge!(odet::OdeState, ctrl::DconControl, equil::Equilibrium.PlasmaEquilibrium, ffit::FourFitVars, intr::DconInternal)

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
function findmax_dW_edge!(odet::OdeState, ctrl::DconControl, equil::Equilibrium.PlasmaEquilibrium, ffit::FourFitVars, intr::DconInternal)

    # Since we search for the maximum dW, initialize to -Infinity
    fill!(odet.dW_edge, -Inf * (1 + im))

    # Create a rough spline for wv matrix between psiedge -> psilim so we can approximate dW
    odet.wvmat_spline = free_compute_wv_spline(ctrl, equil, intr)

    # Loop through integration, compute dW at steps where psifac >= psiedge
    for istep in 1:odet.step
        odet.psifac = odet.psi_store[istep]
        if odet.psifac >= ctrl.psiedge
            odet.u .= odet.u_store[:, :, :, istep]
            odet.dW_edge[istep] = free_compute_total(equil, ffit, intr, odet)
        end
    end

    # Return the index that maximizes dW_edge to identify truncation point
    return findmax(real.(odet.dW_edge))[2]
end

"""
    transform_u!(odet::OdeState, intr::DconInternal)

Constructs the transformation matrices to form the true solution vectors. Effectively
"undoes" the Gaussian reduction applied during fixups throughout the integration, such that
we have the true solution vectors for use in GPEC. Modifies the store arrays in `odet` in-place.
Performs a similar function as `idcon_transform` + `idcon_build` in the Fortran code,
except we separate the building of the transformation matrices and determining the coefficients
for a chosen force-free solution, which can be done in postprocessing.
"""
function transform_u!(odet::OdeState, intr::DconInternal)

    # Gaussian reduction matrices for each fixup
    gauss = Array{ComplexF64,3}(undef, intr.numpert_total, intr.numpert_total, odet.ifix)
    # Transformation matrices for each region between fixups (ifix + 1 regions)
    transforms = Array{ComplexF64,3}(undef, intr.numpert_total, intr.numpert_total, odet.ifix + 1)

    # Construct gaussian reduction matrices for each fixup
    identity = Matrix{ComplexF64}(I, intr.numpert_total, intr.numpert_total)
    mask = trues(intr.numpert_total)
    for ifix in 1:odet.ifix
        gauss[:, :, ifix] = copy(identity)
        mask .= true
        for isol in 1:intr.numpert_total
            ksol = odet.index[isol, ifix]
            mask[ksol] = false
            temp = copy(identity)
            for jsol in 1:intr.numpert_total
                if mask[jsol]
                    temp[ksol, jsol] = odet.fixfac[ksol, jsol, ifix]
                end
            end
            # Matrix multiplication gauss = gauss * temp
            gauss[:, :, ifix] .= gauss[:, :, ifix] * temp
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
    transforms[:, :, end] = copy(identity)
    for ifix in odet.ifix:-1:1
        transforms[:, :, ifix] = gauss[:, :, ifix] * transforms[:, :, ifix+1]
    end

    # Now that we have the transform matrices, we can apply them to the solution vectors
    # "undoing" the Gaussian reductions to get the true solution vectors
    jfix = 1
    for ifix in 1:odet.ifix+1
        # If after the last fixup, go to the end of integration
        kfix = ifix != odet.ifix + 1 ? odet.fixstep[ifix] : odet.step
        for istep in jfix:kfix
            # This is u1->u4 in Fortran
            odet.u_store[:, :, 1, istep] .= odet.u_store[:, :, 1, istep] * transforms[:, :, ifix]
            odet.u_store[:, :, 2, istep] .= odet.u_store[:, :, 2, istep] * transforms[:, :, ifix]
            odet.ud_store[:, :, 1, istep] .= odet.ud_store[:, :, 1, istep] * transforms[:, :, ifix]
            odet.ud_store[:, :, 2, istep] .= odet.ud_store[:, :, 2, istep] * transforms[:, :, ifix]
        end
        jfix = kfix + 1
    end
end