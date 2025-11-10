"""
OdeState

A mutable struct to hold the state of the ODE solver for DCON.
This struct contains all necessary fields to manage the ODE integration process,
including solution vectors, tolerances, and flags for the integration process.
"""
@kwdef mutable struct OdeState
    # Initialization parameters
    numpert_total::Int                  # total number of modes
    numunorms_init::Int             # initial storage size for unorm data
    msing::Int                   # number of singular surfaces
    numsteps_init::Int             # initial size of data store

    # Saved data throughout integration
    step::Int = 1                    # current step of integration (this is like istep in Fortran)
    psi_store::Vector{Float64} = Vector{Float64}(undef, numsteps_init)  # psi at each step of integration
    q_store::Vector{Float64} = Vector{Float64}(undef, numsteps_init)    # q at each step of integration
    u_store::Array{ComplexF64,4} = Array{ComplexF64}(undef, numpert_total, numpert_total, 2, numsteps_init) # store of u at each step of integration
    ud_store::Array{ComplexF64,4} = Array{ComplexF64}(undef, numpert_total, numpert_total, 2, numsteps_init) # store of ud at each step of integration
    ca_r::Array{ComplexF64,4} = Array{ComplexF64}(undef, numpert_total, numpert_total, 2, msing) # asymptotic coefficients just right of singular surface
    ca_l::Array{ComplexF64,4} = Array{ComplexF64}(undef, numpert_total, numpert_total, 2, msing) # asymptotic coefficients just left of singular surface

    # Used for to find peak dW in the edge
    dW_edge::Vector{ComplexF64} = Array{ComplexF64}(undef, numsteps_init)  # dW at each step in the edge
    wvmat_spline::Union{Nothing,Spl.CubicSpline{ComplexF64}} = nothing  # spline of wv matrices for free_test # TODO: how to initialize a spline?

    # Data for integrator
    psifac::Float64 = 0.0       # normalized flux coordinate
    q::Float64 = 0.0            # q value at psifac
    u::Array{ComplexF64,3} = zeros(ComplexF64, numpert_total, numpert_total, 2)            # solution vectors
    ud::Array{ComplexF64,3} = zeros(ComplexF64, numpert_total, numpert_total, 2)           # derivative of solution vectors used in GPEC
    ising::Int = 0               # index of next singular surface
    psimax::Float64 = 0.0         # maximum psi value for the integrator
    next::String = ""           # next integration action ("cross" or "finish")
    nzero::Int = 0              # count of zero crossings detected

    # Used for Gaussian reduction
    new::Bool = true            # flag for computing new unorm0 after a fixup
    unorm::Vector{Float64} = zeros(Float64, numpert_total)                        # norms of solution vectors
    unorm0::Vector{Float64} = zeros(Float64, numpert_total)                       # initial norms of solution vectors
    ifix::Int = 0                # index for number of unorms performed
    index::Array{Int,2} = zeros(Int, numpert_total, numunorms_init)                                   # indices for sorting solutions
    sing_flag::Vector{Bool} = falses(numunorms_init)                     # flags for singular solutions
    zeroed_idx::Vector{Vector{Int}} = Vector{Vector{Int}}(undef, numunorms_init)  # indices of zeroed solutions at each unorm
    fixfac::Array{ComplexF64,3} = zeros(ComplexF64, numpert_total, numpert_total, numunorms_init)             # fixup factors for Gaussian reduction
    fixstep::Vector{Int64} = zeros(Int64, numunorms_init)               # psi values at which unorms were performed

    # Temporary matrices for sing_der calculations
    amat::Vector{ComplexF64} = Vector{ComplexF64}(undef, numpert_total^2)
    bmat::Vector{ComplexF64} = Vector{ComplexF64}(undef, numpert_total^2)
    cmat::Vector{ComplexF64} = Vector{ComplexF64}(undef, numpert_total^2)
    fmat_lower::Vector{ComplexF64} = Vector{ComplexF64}(undef, numpert_total^2)
    kmat::Vector{ComplexF64} = Vector{ComplexF64}(undef, numpert_total^2)
    gmat::Vector{ComplexF64} = Vector{ComplexF64}(undef, numpert_total^2)
    tmp::Matrix{ComplexF64} = Matrix{ComplexF64}(undef, numpert_total, numpert_total)
    Afact::Union{Cholesky{ComplexF64, Matrix{ComplexF64}}, Nothing} = nothing
    singfac_vec::Vector{Float64} = Vector{Float64}(undef, numpert_total)
end

# Initialize function for OdeState with relevant parameters for array initialization
OdeState(numpert_total::Int, numsteps_init::Int, numunorms_init::Int, msing::Int) = OdeState(; numpert_total, numsteps_init, numunorms_init, msing)

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
function ode_run(ctrl::DconControl, equil::Equilibrium.PlasmaEquilibrium, ffit::FourFitVars, intr::DconInternal, outp::DconOutput)

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
        println("   ψ = $(odet.psifac), q = $(Spl.spline_eval!(equil.sq, odet.psifac)[4])")
    end

    # Always integrate once, even if no rational surfaces are crossed
    ode_step!(odet, ctrl, equil, ffit, intr)

    # If at a rational surface, do the appropriate crossing routine, then integrate again
    while odet.ising != ctrl.ksing && odet.next == "cross"
        if ctrl.verbose
            println("   ψ = $(intr.sing[odet.ising].psifac), q = $(intr.sing[odet.ising].q)")
        end

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
    odet.nzero = evaluate_stability_criterion(ctrl, equil, intr, odet, outp)

    # Form the true solution vectors, undoing the Gaussian reduction applied in `ode_unorm!` during integration
    transform_u!(odet, intr)

    if outp.write_euler_h5
        if ctrl.verbose
            println("Writing saved integration data to $(outp.fname_euler_h5)")
        end
        write_euler_h5(ctrl, equil, intr, odet, outp)
    end

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
        # Find next singular surface (either next one in the list or outside integration limits)
        # TODO: (ask Nik) this is tricky, did I generalize the logic correctly? (shorten this comment later)
        # Basically, want to choose the next singular surface based on being within our integration limits
        # and also being resonant with our included poloidal mode numbers. When there are mulitple toroidal
        # mode numbers, we need to check if any of them are resonant with the singular surface. This is equivalent
        # to checking if any of the m values in that singular surface are within the range of mlow and mhigh,
        # Will this ever not be true? I thought we set mlow and mhigh to be wide enough to include all resonant surfaces?
        # single n: in_q_range = intr.mlow <= ctrl.nn * sing.q <= intr.mhigh
        # multi n: in_q_range = check if any n*q is resonant, equivalent to checking singp m-vector
        for i in (odet.ising + 1):intr.msing
            odet.ising = i
            singp = intr.sing[odet.ising]
            (intr.psilim < singp.psifac || any(m -> intr.mlow <= m <= intr.mhigh, singp.m)) && break
        end
        singp = intr.sing[odet.ising] # singp loses scope outside of loop
        # Determine psimax and classify next integration limit type
        if odet.ising > intr.msing || intr.psilim < singp.psifac || ctrl.singfac_min == 0
            odet.psimax = intr.psilim * (1 - eps)
            odet.next = "finish"
        else
            # TODO: (ask Nik) where does singfac_min / n * q' come from? How does it generalize to multi-n?
            # Safest choice for now is to use the smallest resonant n for maximum separation
            odet.psimax = singp.psifac - ctrl.singfac_min / abs(minimum(singp.n) * singp.q1)
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
    
    # TODO: single n, we remove the largest solution and sub in the asymptotic on the other side
    # However, if we do remove the N largest modes, we can zero out one that lives in the upper block
    # and then sub in the asymptotic solution which lives in the lower block, messing up the block
    # diagonal structure of the matrix and later calculations
    # I don't have a good reason for doing this, but for now we will make sure we only remove the largest mode
    # in the same block as the resonant mode. Does this change in 3D?
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
    # TODO: CLEAN UP THIS MESS
    if odet.ising == intr.msing
        odet.ising += 1
    else
        for i in (odet.ising + 1):intr.msing
            odet.ising = i
            singp = intr.sing[odet.ising]
            (intr.psilim < singp.psifac || any(m -> intr.mlow <= m <= intr.mhigh, singp.m)) && break
        end
    end
    
    # Determine psimax and classify next integration limit type
    if odet.ising > intr.msing || intr.psilim < intr.sing[odet.ising].psifac || ctrl.singfac_min == 0
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
    ode_step!(odet::OdeState, ctrl::DconControl, equil::Equilibrium.PlasmaEquilibrium, ffit::FourFitVars, intr::DconInternal, outp::DconOutput)

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
    
    ctrl, _, _, intr, odet = integrator.p

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
        # TODO: some of this tolerance logic might not even be needed, but for now
        # I'll generalize it to multi-n, preserving as much of the original logic as possible
        # singfac = m-nq = n(m/n - q) = n (q_res - q), use smallest n to be conservative
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

"""
    write_euler_h5(ctrl::DconControl, equil::Equilibrium.PlasmaEquilibrium, intr::DconInternal, odet::OdeState, outp::DconOutput)

Helper function to write the euler.h5 file with relevant run and equilibrium parameters.
This combines the functionality of several pieces of the Fortran code in `ode_output.f`,
primarily `ode_output_open` and the various `bin_euler` writes that occur throughout the
integration. Note that if `vac_flag` is true, additional outputs related to vacuum solutions
are dumped in `free_run`, with this file opened in append mode.

### TODOs

Remove deprecated outputs
Combine spline unpacking if possible, too many extra lines

"""
function write_euler_h5(ctrl::DconControl, equil::Equilibrium.PlasmaEquilibrium, intr::DconInternal, odet::OdeState, outp::DconOutput)

    h5open(joinpath(intr.dir_path, outp.fname_euler_h5), "w") do euler_h5
        # Write run parameters
        euler_h5["info/mpert"] = intr.mpert
        euler_h5["info/mband"] = intr.mband
        euler_h5["info/mlow"] = intr.mlow
        euler_h5["info/mhigh"] = intr.mhigh
        euler_h5["info/npert"] = intr.npert
        euler_h5["info/nlow"] = intr.nlow
        euler_h5["info/nhigh"] = intr.nhigh
        m = [(i - 1) % intr.mpert + intr.mlow for i in 1:(intr.numpert_total)]
        n = [(i - 1) ÷ intr.mpert + intr.nlow for i in 1:(intr.numpert_total)]
        euler_h5["info/mn_index"] = hcat(m, n)   # (N, 2) matrix
        euler_h5["info/singfac_min"] = ctrl.singfac_min
        euler_h5["info/kin_flag"] = ctrl.kin_flag
        euler_h5["info/con_flag"] = ctrl.con_flag
        euler_h5["info/mthvac"] = ctrl.mthvac
        euler_h5["info/mthsurf0"] = outp.mthsurf0 #TODO: mthsurf0 is deprecated

        # Write equilibrium parameters
        euler_h5["equil/nr"] = length(equil.rzphi.xs) # TODO: equil save mpsi as really mpsi - 1, fix this
        euler_h5["equil/nz"] = length(equil.rzphi.ys)
        euler_h5["equil/ro"] = equil.ro
        euler_h5["equil/zo"] = equil.zo
        euler_h5["equil/amean"] = equil.params.amean
        euler_h5["equil/rmean"] = equil.params.rmean
        euler_h5["equil/aratio"] = equil.params.aratio
        euler_h5["equil/kappa"] = equil.params.kappa
        euler_h5["equil/delta1"] = equil.params.delta1
        euler_h5["equil/delta2"] = equil.params.delta2
        euler_h5["equil/li1"] = equil.params.li1
        euler_h5["equil/li2"] = equil.params.li2
        euler_h5["equil/li3"] = equil.params.li3
        euler_h5["equil/betap1"] = equil.params.betap1
        euler_h5["equil/betap2"] = equil.params.betap2
        euler_h5["equil/betap3"] = equil.params.betap3
        euler_h5["equil/betat"] = equil.params.betat
        euler_h5["equil/betan"] = equil.params.betan
        euler_h5["equil/bt0"] = equil.params.bt0
        euler_h5["equil/q0"] = equil.params.q0
        euler_h5["equil/q95"] = equil.params.q95
        euler_h5["equil/qmin"] = equil.params.qmin
        euler_h5["equil/qmax"] = equil.params.qmax
        euler_h5["equil/qa"] = equil.params.qa
        euler_h5["equil/crnt"] = equil.params.crnt
        euler_h5["equil/psio"] = equil.psio
        euler_h5["equil/psilow"] = equil.config.control.psilow
        euler_h5["equil/power_b"] = equil.config.control.power_b
        euler_h5["equil/power_r"] = equil.config.control.power_r
        euler_h5["equil/power_bp"] = equil.config.control.power_bp
        euler_h5["equil/shotnum"] = 0 # TODO: equil.params.shotnum
        euler_h5["equil/shottime"] = 0 # TODO: equil.params.shottime

        # Write spline arrays
        euler_h5["splines/sq/xs"] = Vector(equil.sq.xs)
        # TODO: getting errors when trying to dump just fs, so splitting for now, which adds so many lines
        # This should be fixed if we separate these like Nik mentioned
        euler_h5["splines/sq/fs/2piF"] = equil.sq.fs[:, 1]
        euler_h5["splines/sq/fs/mu0p"] = equil.sq.fs[:, 2]
        euler_h5["splines/sq/fs/dVdpsi"] = equil.sq.fs[:, 3]
        euler_h5["splines/sq/fs/q"] = equil.sq.fs[:, 4]
        euler_h5["splines/sq/fs1/2piF"] = equil.sq.fs1[:, 1]
        euler_h5["splines/sq/fs1/mu0p"] = equil.sq.fs1[:, 2]
        euler_h5["splines/sq/fs1/dVdpsi"] = equil.sq.fs1[:, 3]
        euler_h5["splines/sq/fs1/q"] = equil.sq.fs1[:, 4]
        euler_h5["splines/sq/xpower"] = 0 # TODO: equil.sq.xpower
        euler_h5["splines/rzphi/xs"] = Vector(equil.rzphi.xs)
        euler_h5["splines/rzphi/ys"] = Vector(equil.rzphi.ys)
        euler_h5["splines/rzphi/fs/rcoords"] = equil.rzphi.fs[:, 1]
        euler_h5["splines/rzphi/fs/offset"] = equil.rzphi.fs[:, 2]
        euler_h5["splines/rzphi/fs/nu"] = equil.rzphi.fs[:, 3]
        euler_h5["splines/rzphi/fs/jac"] = equil.rzphi.fs[:, 4]
        euler_h5["splines/rzphi/fsx/rcoords"] = equil.rzphi.fsx[:, 1]
        euler_h5["splines/rzphi/fsx/offset"] = equil.rzphi.fsx[:, 2]
        euler_h5["splines/rzphi/fsx/nu"] = equil.rzphi.fsx[:, 3]
        euler_h5["splines/rzphi/fsx/jac"] = equil.rzphi.fsx[:, 4]
        euler_h5["splines/rzphi/fsy/rcoords"] = equil.rzphi.fsy[:, 1]
        euler_h5["splines/rzphi/fsy/offset"] = equil.rzphi.fsy[:, 2]
        euler_h5["splines/rzphi/fsy/nu"] = equil.rzphi.fsy[:, 3]
        euler_h5["splines/rzphi/fsy/jac"] = equil.rzphi.fsy[:, 4]
        euler_h5["splines/rzphi/fsxy/rcoords"] = equil.rzphi.fsxy[:, 1]
        euler_h5["splines/rzphi/fsxy/offset"] = equil.rzphi.fsxy[:, 2]
        euler_h5["splines/rzphi/fsxy/nu"] = equil.rzphi.fsxy[:, 3]
        euler_h5["splines/rzphi/fsxy/jac"] = equil.rzphi.fsxy[:, 4]
        euler_h5["splines/rzphi/x0"] = 0 # TODO: equil.rzphi.x0
        euler_h5["splines/rzphi/y0"] = 0 # TODO: equil.rzphi.y0
        euler_h5["splines/rzphi/xpower"] = 0 # TODO: equil.rzphi.xpower
        euler_h5["splines/rzphi/fpower"] = 0 # TODO: equil.rzphi.fpower

        # Write integration data
        euler_h5["info/psilim"] = intr.psilim
        euler_h5["info/qlim"] = intr.qlim
        euler_h5["integration/nstep"] = odet.step
        euler_h5["integration/psi"] = odet.psi_store
        euler_h5["integration/q"] = odet.q_store
        if !ctrl.vac_flag # we normalize by wt before dumping if calling free_run
            euler_h5["integration/xi_psi"] = odet.u_store[:, :, 1, :]
            euler_h5["integration/u2"] = odet.u_store[:, :, 2, :] # TODO: what to name this? These are the "conjugate momenta" of u1
            euler_h5["integration/dxi_psi"] = odet.ud_store[:, :, 1, :]
            euler_h5["integration/xi_s"] = odet.ud_store[:, :, 2, :]
        end

        # Write singular surface data
        euler_h5["singular/msing"] = intr.msing
        euler_h5["singular/psi"] = [intr.sing[ising].psifac for ising in 1:intr.msing]
        euler_h5["singular/q"] = [intr.sing[ising].q for ising in 1:intr.msing]
        euler_h5["singular/q1"] = [intr.sing[ising].q1 for ising in 1:intr.msing]
        euler_h5["singular/ca_left"] = odet.ca_l
        euler_h5["singular/ca_right"] = odet.ca_r
    end
end