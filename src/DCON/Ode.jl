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
    odet = OdeState(intr.mpert, ctrl.numsteps_init, ctrl.numunorms_init, intr.msing)

    if ctrl.sing_start <= 0
        ode_axis_init!(odet, ctrl, equil, intr)
    elseif ctrl.sing_start <= intr.msing
        error("sing_start > 0 not implemented yet!")
        # ode_sing_init!(ctrl, equil, intr, odet)
    else
        error("Invalid value for sing_start: $(ctrl.sing_start) > msing = $(intr.msing)")
    end

    if ctrl.verbose # mimicing output from ode_output_open
        println("   ψ = $(odet.psifac), q = $(Spl.spline_eval!(equil.sq, odet.psifac)[4])")
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
        println("max dw = $(odet.dW_edge[odet.step]) at ψ = $(odet.psi_store[odet.step])")
        trim_storage!(odet)
        if ctrl.verbose
            println("Truncating integration at peak edge dW: ψ = $(odet.psi_store[odet.step]), q = $(odet.q_store[odet.step])")
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
determining `psifac`, `psimax`, `ising`, `m1`, `singfac`, and initializing `u`.

### TODOs

Support for `kin_flag`
Remove while true logic
"""
function ode_axis_init!(odet::OdeState, ctrl::DconControl, equil::Equilibrium.PlasmaEquilibrium, intr::DconInternal)

    # Shorthand to evaluate q/q1 inside newton iteration
    qval = psi -> Spl.spline_eval!(equil.sq, psi)[4]
    q1val = psi -> Spl.spline_deriv1!(equil.sq, psi)[2][4]

    # Preliminary computations
    odet.psifac = equil.sq.xs[1]

    # Use Newton iteration to find starting psi if qlow is above q0
    if ctrl.qlow > equil.sq.fs[1, 4]
        # Start check from the edge for robustness in reverse shear cores
        for jpsi in equil.sq.mx:-1:2  # avoid starting iteration on endpoints
            if equil.sq.fs[jpsi-1, 4] < ctrl.qlow
                odet.psifac = equil.sq.xs[jpsi]
                break
            end
        end
        it = 0
        while it ≤ itmax
            it += 1
            dpsi = (ctrl.qlow - qval(odet.psifac)) / q1val(odet.psifac)
            odet.psifac += dpsi
            if abs(dpsi) < eps * abs(odet.psifac)
                break
            end
        end
    end

    # Find inner singular surface
    if false #(TODO: kin_flag)
    # for ising = 1:kmsing
    #     if kinsing[ising].psifac > psifac
    #         break
    #     end
    # end
    else
        odet.ising = 0
        for i in 1:intr.msing
            if intr.sing[i].psifac > odet.psifac
                odet.ising = max(0, i - 1)
                break
            end
        end
    end

    # Find next singular surface
    if false # TODO: (kin_flag)
    # while true
    #     ising += 1
    #     if ising > kmsing
    #         break
    #     end
    #     if psilim < kinsing[ising].psifac
    #         break
    #     end
    #     odet.q = kinsing[ising].q
    #     if mlow <= nn * odet.q && mhigh >= nn * odet.q
    #         break
    #     end
    # end
    # if ising > kmsing || singfac_min == 0
    #     psimax = psilim * (1 - eps)
    #     next = "finish"
    # elseif psilim < kinsing[ising].psifac
    #     psimax = psilim * (1 - eps)
    #     next = "finish"
    # else
    #     psimax = kinsing[ising].psifac - singfac_min / abs(nn * kinsing[ising].q1)
    #     next = "cross"
    # end
    else
        while true
            odet.ising += 1
            if odet.ising > intr.msing || intr.psilim < intr.sing[min(odet.ising, intr.msing)].psifac
                break
            end
            if intr.mlow <= ctrl.nn * intr.sing[odet.ising].q && intr.mhigh >= ctrl.nn * intr.sing[odet.ising].q
                break
            end
        end
        if odet.ising > intr.msing || intr.psilim < intr.sing[min(odet.ising, intr.msing)].psifac || ctrl.singfac_min == 0
            odet.psimax = intr.psilim * (1 - eps)
            odet.next = "finish"
        else
            odet.psimax = intr.sing[odet.ising].psifac - ctrl.singfac_min / abs(ctrl.nn * intr.sing[odet.ising].q1)
            odet.next = "cross"
        end
    end

    # Initialize solutions with the identity matrix for U_22 as described in [Glasser PoP 2016] Section VI
    for ipert in 1:intr.mpert
        odet.u[ipert, ipert, 2] = 1
    end

    # Compute conditions at next singular surface
    if false #TODO: (kin_flag)
    # if kmsing > 0
    #     m1 = round(Int, nn * kinsing[ising].q)
    # else
    #     m1 = round(Int, nn * qlim) + sign(one, nn * sq.fs1[mpsi, 5])
    # end
    else
        # note: Julia's default round does Banker's rounding, to match NINT in fortran we need to specify RoundFromZero
        if intr.msing > 0
            odet.m1 = round(Int, ctrl.nn * intr.sing[odet.ising].q, RoundFromZero)
        else
            odet.m1 = round(Int, ctrl.nn * intr.qlim, RoundFromZero) + sign(ctrl.nn * equil.sq.fs1[end, 4])
        end
    end
    odet.singfac = abs(odet.m1 - ctrl.nn * equil.sq.fs[1, 4]) # Fortran: q=sq%fs(0,4)
end

# TODO: NOT IMPLEMENTED YET! (low priority, just make sure sing_start = 0 in dcon.toml)
function ode_sing_init()
    return
end
#     # Declare and initialize local variables
#     star = fill(' ', mpert)
#     # local variables
#     ua = Array{Complex{r8}}(undef, mpert, 2*mpert, 2)
#     dpsi = 0.0
#     new = true

#     ising = sing_start
#     dpsi = singfac_min/abs(nn*sing[ising].q1)*10
#     odet.psifac = sing[ising].psifac + dpsi
#     odet.q = sing[ising].q + dpsi*sing[ising].q1

#     # Allocate and initialize solution arrays
#     u      = zeros(Complex{r8}, mpert, mpert, 2)
#     du     = zeros(Complex{r8}, mpert, mpert, 2)
#     unorm0 = zeros(r8, 2*mpert)
#     unorm  = zeros(r8, 2*mpert)
#     index  = zeros(Int, 2*mpert)

#     if old_init
#         u .= 0.0
#         for ipert ∈ 1:mpert
#             u[ipert, ipert, 2] = 1.0
#         end
#     else
#         sing_get_ua(ising, odet.psifac, ua)
#         # Big slice: u = ua[:, mpert+1:2*mpert,:]
#         for i = 1:mpert, j = 1:mpert, k = 1:2
#             u[i, j, k] = ua[i, mpert+j, k]
#         end
#     end
#     msol = mpert
#     neq = 4 * mpert * msol

#     # Compute conditions at next singular surface
#     while true
#         ising += 1
#         if ising > msing || psilim < sing[ising ≤ msing ? ising : msing].psifac
#             break
#         end
#         odet.q = sing[ising].q
#         if mlow ≤ nn*q && mhigh ≥ nn*q
#             break
#         end
#     end
# This needs to be fixed up
#     if ising > msing || psilim < sing[ising ≤ msing ? ising : msing].psifac
#         m1 = round(Int, ctrl.nn*intr.qlim) + round(Int, sign(one, ctrl.nn*equil.sq.fs1[mpsi, 4]))
#         psimax = psilim * (1-eps)
#         next_ = "finish"
#     else
#         m1 = round(Int, nn*sing[ising].q)
#         psimax = sing[ising].psifac - singfac_min/abs(nn*sing[ising].q1)
#         next_ = "cross"
#     end
#     # Terminate, or in Julia just return (no need for RETURN)
#     return nothing  # or could return a struct with all these values, for a more Julian approach
# end

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
    if ctrl.verbose
        println("   ψ = $(intr.sing[odet.ising].psifac), q = $(intr.sing[odet.ising].q)")
    end
    ode_unorm!(odet.u, odet, ctrl, intr, true)

    # Get asymptotic coefficients before crossing rational surface
    ca = sing_get_ca(ctrl, intr, odet)
    odet.ca_l[:, :, :, odet.ising] .= ca

    # Re-initialize on opposite side of rational surface
    psi_old = odet.psifac
    singp = intr.sing[odet.ising]
    ipert0 = round(Int, ctrl.nn * singp.q, RoundFromZero) - intr.mlow + 1
    dpsi = singp.psifac - odet.psifac
    odet.psifac = singp.psifac + dpsi
    ua = sing_get_ua(ctrl, intr, odet)
    if !ctrl.con_flag
        odet.u[:, odet.index[1, odet.ifix], :] .= 0
    end

    # Update solution vectors
    du1 = zeros(ComplexF64, intr.mpert, intr.mpert, 2)
    du2 = zeros(ComplexF64, intr.mpert, intr.mpert, 2)
    params = (ctrl, equil, ffit, intr, odet)
    sing_der!(du1, odet.u, params, psi_old)
    sing_der!(du2, odet.u, params, odet.psifac)
    odet.u .+= (du1 .+ du2) .* dpsi
    if !ctrl.con_flag
        odet.u[ipert0, :, :] .= 0
        odet.u[:, odet.index[1, odet.ifix], :] .= ua[:, ipert0+intr.mpert, :]
    end

    # Get asymptotic coefficients after crossing rational surface
    ca = sing_get_ca(ctrl, intr, odet)
    odet.ca_r[:, :, :, odet.ising] .= ca

    # Find next ising
    while true
        odet.ising += 1
        if odet.ising > intr.msing || intr.psilim < intr.sing[min(odet.ising, intr.msing)].psifac
            break
        end
        if intr.mlow <= ctrl.nn * intr.sing[odet.ising].q && intr.mhigh >= ctrl.nn * intr.sing[odet.ising].q
            break
        end
    end

    # Compute conditions at next singular surface
    if odet.ising > intr.msing || intr.psilim < intr.sing[min(odet.ising, intr.msing)].psifac
        odet.psimax = intr.psilim * (1 - eps)
        odet.m1 = round(Int, ctrl.nn * intr.qlim, RoundFromZero) + sign(ctrl.nn * equil.sq.fs1[end, 4])
        odet.next = "finish"
    else
        singp = intr.sing[odet.ising] # Update singp
        odet.psimax = singp.psifac - ctrl.singfac_min / abs(ctrl.nn * singp.q1)
        odet.m1 = round(Int, ctrl.nn * singp.q, RoundFromZero)
    end

    # Restart ode solver
    odet.new = true
    odet.psi_store[odet.step] = odet.psifac # Store current psi
    odet.q_store[odet.step] = odet.q # Store current q
    odet.u_store[:, :, :, odet.step] = odet.u   # Store current u
    odet.step += 1 # Advance step to account for crossing step
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

    println("   ψ = $(odet.psifac), max u = $(maximum(abs, odet.u)), steps = $(odet.step-1)")
end

"""
    integrator_callback!(integrator)

Callback function for ODE integrator to handle normalization, output, and storage
at each step. This handles the solution normalization logic that was previously
in a DO loop within ode_run and called every step by running LSODE in one step mode
in the Fortran code. However, we now perform the equivalent of `ode_output_step`
and `ode_record_edge` post-integration using the saved data.

"""
function integrator_callback!(integrator)
    
    ctrl, equil, ffit, intr, odet = integrator.p

    # Update integration tolerances
    integrator.opts.reltol = compute_tols(ctrl, intr, odet)
    # integrator.opts.abstol = atol

    # Check if the solution norms are above a threshold, if so apply Gaussian reduction
    ode_unorm!(integrator.u, odet, ctrl, intr, false)

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
        # Note: odet.q is updated within the derivative calculation
        if odet.ising <= intr.msing
            singfac_local = abs(intr.sing[odet.ising].m - ctrl.nn * odet.q)
        end
        # If in between singular surfaces, check distance to both
        if odet.ising > 1
            singfac_local = min(singfac_local, abs(intr.sing[odet.ising-1].m - ctrl.nn * odet.q))
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
    for isol in 1:intr.mpert
        odet.fixfac[isol, isol, ifix] = 1
    end
    # Sort unorm in descending order (since we triangularize from largest to smallest)
    odet.index[:, ifix] = sortperm(odet.unorm; rev=true)

    # Triangularize primary solutions
    mask = trues(2, intr.mpert)
    for isol in 1:intr.mpert
        ksol = odet.index[isol, ifix]
        mask[2, ksol] = false
        # Set pivot row based on max location
        @views kpert = argmax(abs.(u[:, ksol, 1]) .* mask[1, :])
        mask[1, kpert] = false
        # Eliminate other solution vectors below the pivot
        for jsol in 1:intr.mpert
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
    gauss = Array{ComplexF64,3}(undef, intr.mpert, intr.mpert, odet.ifix)
    # Transformation matrices for each region between fixups (ifix + 1 regions)
    transforms = Array{ComplexF64,3}(undef, intr.mpert, intr.mpert, odet.ifix + 1)

    # Construct gaussian reduction matrices for each fixup
    identity = Matrix{ComplexF64}(I, intr.mpert, intr.mpert)
    mask = trues(intr.mpert)
    for ifix in 1:odet.ifix
        gauss[:, :, ifix] = copy(identity)
        mask .= true
        for isol in 1:intr.mpert
            ksol = odet.index[isol, ifix]
            mask[ksol] = false
            temp = copy(identity)
            for jsol in 1:intr.mpert
                if mask[jsol]
                    temp[ksol, jsol] = odet.fixfac[ksol, jsol, ifix]
                end
            end
            # Matrix multiplication gauss = gauss * temp
            gauss[:, :, ifix] .= gauss[:, :, ifix] * temp
        end
        if odet.sing_flag[ifix]
            gauss[:, odet.index[1, ifix], ifix] .= 0.0
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