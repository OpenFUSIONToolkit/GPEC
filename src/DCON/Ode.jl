"""
OdeState

A mutable struct to hold the state of the ODE solver for DCON.
This struct contains all necessary fields to manage the ODE integration process,
including solution vectors, tolerances, and flags for the integration process.
"""
@kwdef mutable struct OdeState
    # Initialization parameters
    mpert::Int                  # poloidal mode number count
    numunorms_init::Int             # initial storage size for unorm data
    msing::Int                   # number of singular surfaces
    numsteps_init::Int             # initial size of data store

    # Saved data throughout integration
    step::Int = 1                    # current step of integration (this is like istep in Fortran)
    psi_store::Vector{Float64} = Vector{Float64}(undef, numsteps_init)  # psi at each step of integration
    u_store::Array{ComplexF64,4} = Array{ComplexF64}(undef, mpert, mpert, 2, numsteps_init) # store of u at each step of integration
    ud_store::Array{ComplexF64,4} = Array{ComplexF64}(undef, mpert, mpert, 2, numsteps_init) # store of ud at each step of integration
    ca_r::Array{ComplexF64,4} = Array{ComplexF64}(undef, mpert, mpert, 2, msing) # asymptotic coefficients just right of singular surface
    ca_l::Array{ComplexF64,4} = Array{ComplexF64}(undef, mpert, mpert, 2, msing) # asymptotic coefficients just left of singular surface
    index::Array{Int,2} = zeros(Int, mpert, numunorms_init)                                   # indices for sorting solutions
    sing_flag::Vector{Bool} = falses(numunorms_init)                     # flags for singular solutions
    fixfac::Array{ComplexF64,3} = zeros(ComplexF64, mpert, mpert, numunorms_init)             # fixup factors for Gaussian reduction
    fixstep::Vector{Int64} = zeros(Int64, numunorms_init)               # psi values at which unorms were performed

    # Used for to find peak dW in the edge
    dW_edge::Vector{ComplexF64} = Array{ComplexF64}(undef, numsteps_init)  # dW at each step in the edge
    wvmat_spline::Union{Missing,Spl.CubicSpline{ComplexF64}} = missing  # spline of wv matrices for free_test # TODO: how to initialize a spline?
    wvmat_spline_created::Bool = false  # flag for if wv matrix spline has been computed

    # Data for integrator
    psifac::Float64 = 0.0       # normalized flux coordinate
    q::Float64 = 0.0            # q value at psifac
    u::Array{ComplexF64,3} = zeros(ComplexF64, mpert, mpert, 2)            # solution vectors
    ud::Array{ComplexF64,3} = zeros(ComplexF64, mpert, mpert, 2)           # derivative of solution vectors used in GPEC
    ising::Int = 0               # index of next singular surface
    m1::Int = 0                 # poloidal mode number for the next singular surface (?)
    psimax::Float64 = 0.0         # maximum psi value for the integrator
    singfac::Float64 = 0.0      # separation from singular surface in terms of m - nq
    next::String = ""           # next integration action ("cross" or "finish")
    psi_prev::Float64 = 0.0     # previous psi value
    crit_prev::Float64 = 0.0    # previous crit value
    u_prev::Array{ComplexF64,3} = zeros(ComplexF64, mpert, mpert, 2) # previous solution array
    nzero::Int = 0              # count of zero crossings detected

    # Used for Gaussian reduction
    new::Bool = true            # flag for computing new unorm0 after a fixup
    unorm::Vector{Float64} = zeros(Float64, mpert)                        # norms of solution vectors
    unorm0::Vector{Float64} = zeros(Float64, mpert)                       # initial norms of solution vectors
    ifix::Int = 0                # index for number of unorms performed

    # Temporary matrices for sing_der calculations
    amat::Vector{ComplexF64} = Vector{ComplexF64}(undef, mpert^2)
    bmat::Vector{ComplexF64} = Vector{ComplexF64}(undef, mpert^2)
    cmat::Vector{ComplexF64} = Vector{ComplexF64}(undef, mpert^2)
    fmat_lower::Vector{ComplexF64} = Vector{ComplexF64}(undef, mpert^2)
    kmat::Vector{ComplexF64} = Vector{ComplexF64}(undef, mpert^2)
    gmat::Vector{ComplexF64} = Vector{ComplexF64}(undef, mpert^2)
    tmp::Matrix{ComplexF64} = Matrix{ComplexF64}(undef, mpert, mpert)
    Afact::Union{Cholesky{ComplexF64,Matrix{ComplexF64}},Nothing} = nothing
    singfac_vec::Vector{Float64} = Vector{Float64}(undef, mpert)
end

# Base struct that handles storage of the ODE solution throughout integration
mutable struct OdeDataStore
    psi::Vector{Float64}
    u::Vector{Array{ComplexF64,3}}
    ud::Vector{Array{ComplexF64,3}}
    step::Int
end

# Initialize function for data storage object of a given size, will resize as needed
OdeDataStore(numsteps_init::Int) =
    OdeDataStore(Vector{Float64}(undef, numsteps_init), Vector{Array{ComplexF64,3}}(undef, numsteps_init), Vector{Array{ComplexF64,3}}(undef, numsteps_init), 0)

# Initialize function for OdeState with relevant parameters for array initialization
OdeState(mpert::Int, numsteps_init::Int, numunorms_init::Int, msing::Int) = OdeState(; mpert, numsteps_init, numunorms_init, msing)

"""
    `ode_run(ctrl::DconControl, equil::Equilibrium.PlasmaEquilibrium, intr::DconInternal)`

Main driver for integrating plasma equilibrium and detecting singular surfaces.
Has the same functionality as `ode_run` in the Fortran code, with the addition of
a single dump to the `euler.h5` file at the end of integration instead of multiple dumps
to `euler.bin` throughout the integration. We have made the control logic more clear
including making a clear end condition to the while loop and implementing the unorming
and output logic within a callback in the integration.

### TODOs

Support for `kin_flag`
restype functionality if we decide to do this

### Returns

An OdeState struct containing the final state of the ODE solver after integration is complete.
"""
function ode_run(ctrl::DconControl, equil::Equilibrium.PlasmaEquilibrium, ffit::FourFitVars, intr::DconInternal, outp::DconOutput)

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

    # Since we search for the peak dW in this region, initialize to -Infinity
    if ctrl.psiedge < intr.psilim
        fill!(odet.dW_edge, -Inf * (1 + im))
    end

    if ctrl.verbose # mimicing output from ode_output_open
        println("   ψ = $(odet.psifac), q = $(Spl.spline_eval!(equil.sq, odet.psifac)[4])")
    end

    # Write header data to files
    ode_output_init(ctrl, equil, intr, odet, outp)

    # Always integrate once, even if no rational surfaces are crossed
    ode_step!(odet, ctrl, equil, ffit, intr, outp)

    # If at a rational surface, do the appropriate crossing routine, then integrate again
    while odet.ising != ctrl.ksing && odet.next == "cross"
        if ctrl.kin_flag
            error("kin_flag = true not implemented yet!")
        else
            ode_ideal_cross!(odet, ctrl, equil, ffit, intr, outp)
        end

        ode_step!(odet, ctrl, equil, ffit, intr, outp)
    end

    # Deallocate unused storage of integration data
    if ctrl.psiedge < intr.psilim
        # Find the peak dW in the edge region and truncate integration data there
        odet.step = findmax(real.(odet.dW_edge))[2]
        trim_storage!(odet)
        if ctrl.verbose
            println("Truncating integration at peak edge dW: ψ = $(odet.psi_store[odet.step]), q = $(Spl.spline_eval!(equil.sq, odet.psi_store[odet.step])[4])")
        end

        # Update u, psilim, and qlim for usage in determining wp and wt
        intr.psilim = odet.psi_store[end]
        intr.qlim = Spl.spline_eval!(equil.sq, intr.psilim)[4]
        odet.u .= odet.u_store[:, :, :, end]
    else
        odet.step -= 1 # step was incremented one extra time in ode_step!
        trim_storage!(odet)
    end

    # form the true solution vectors, undoing the Gaussian reduction applied during fixups throughout the integration
    transform_u!(odet, intr)

    if outp.write_euler_h5
        if ctrl.verbose
            println("Writing saved integration data to euler.h5...")
        end
        # We open in r+ mode to add to the existing file from ode_output_init instead of overwriting it
        h5open(joinpath(intr.dir_path, outp.fname_euler_h5), "r+") do euler_h5
            # Output psilim and qlim here in case they were changed due to peak edge dW search
            euler_h5["info/psilim"] = intr.psilim
            euler_h5["info/qlim"] = intr.qlim
            euler_h5["integration/nstep"] = odet.step
            euler_h5["integration/psi"] = odet.psi_store
            euler_h5["integration/q"] = Spl.spline_eval(equil.sq, odet.psi_store, 0)[4]
            if !ctrl.vac_flag # we normalize by wt before dumping if calling free_run
                euler_h5["integration/xi_psi"] = odet.u_store[:, :, 1, :]
                euler_h5["integration/u2"] = odet.u_store[:, :, 2, :] # TODO: what to name this? These are the "conjugate momenta" of u1
                euler_h5["integration/dxi_psi"] = odet.ud_store[:, :, 1, :]
                euler_h5["integration/xi_s"] = odet.ud_store[:, :, 2, :]
            end
            euler_h5["singular/msing"] = intr.msing
            euler_h5["singular/psi"] = [intr.sing[ising].psifac for ising in 1:intr.msing]
            euler_h5["singular/q"] = [intr.sing[ising].q for ising in 1:intr.msing]
            euler_h5["singular/q1"] = [intr.sing[ising].q1 for ising in 1:intr.msing]
            euler_h5["singular/ca_left"] = odet.ca_l
            euler_h5["singular/ca_right"] = odet.ca_r
            # TODO: restype writing would also be added here, Matt says not needed for now (or maybe ever)
        end
    end
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
    odet.u_prev .= odet.u
    odet.psi_prev = odet.psifac

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
#     u_prev = zeros(Complex{r8}, mpert, mpert, 2)
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
#     u_prev .= u
#     psi_prev = psifac
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
    ode_ideal_cross!(odet::OdeState, ctrl::DconControl, equil::Equilibrium.PlasmaEquilibrium, ffit::FourFitVars, intr::DconInternal, outp::DconOutput)

Handle the crossing of a rational surface during ODE integration if `kin_flag` is false.
Performs the same function as `ode_ideal_cross` in the Fortran code. Differences mainly in integration data
storage logic, but otherwise identical. It normalizes and reinitializes the solution vector at the singularity,
and updates relevant state variables and updates `odet` for continued integration. It also determines the
location and parameters of the next singular surface and writes outputs as desired.

### TODOs

Remove while true logic
"""
function ode_ideal_cross!(odet::OdeState, ctrl::DconControl, equil::Equilibrium.PlasmaEquilibrium, ffit::FourFitVars, intr::DconInternal, outp::DconOutput)

    # Fixup solution at singular surface
    if ctrl.verbose
        println("   ψ = $(intr.sing[odet.ising].psifac), q = $(intr.sing[odet.ising].q)")
    end
    ode_unorm!(odet, ctrl, intr, outp, true)

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
    params = (ctrl, equil, ffit, intr, odet, outp)
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
        if outp.write_crit_out
            write_output(outp, :crit_out, @sprintf("   ising   psi         q          di      re alpha   im alpha\n"))
            write_output(outp, :crit_out, @sprintf("%6d%11.3e%11.3e%11.3e%11.3e%11.3e\n", odet.ising, singp.psifac, singp.q, singp.di, real(singp.alpha), imag(singp.alpha)))
        end
    end

    # Restart ode solver
    odet.new = true
    odet.psi_store[odet.step] = odet.psifac # Store current psi
    odet.u_store[:, :, :, odet.step] = odet.u   # Store current u
    odet.step += 1 # Advance step to account for crossing step
    odet.u_prev .= odet.u
    odet.psi_prev = odet.psifac

    # Write next header before continuing integration
    if outp.write_crit_out
        write_output(outp, :crit_out, "    psifac      dpsi        q       singfac     eval1")
    end
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
a callback function to handle tolerances, normalization, output, and storage at each
step of the integration. In Fortran, this was performed by running LSODE in one-step
mode (so ode_step was called hundreds of times) and calling the relevant functions in
a DO loop. Here, we use the DifferentialEquations.jl interface to achieve the same
functionality in a more Julian way. In addition to the callback logic, this function
computes and sets the next integration endpoint, and advances the solution using
an adaptive ODE solver. The state in `odet` is then updated in-place with
the solution at the new point.

### TODOs

Check if additional output is needed at the start and end of integration
Check sensitivity of results to tolerances, currently using same logic as Fortran
Check absolute tolerances, currently only relative tolerances are updated
"""
function ode_step!(odet::OdeState, ctrl::DconControl, equil::Equilibrium.PlasmaEquilibrium, ffit::FourFitVars, intr::DconInternal, outp::DconOutput)

    # Set up the callback to be run at every step
    cb = DiscreteCallback((u, t, integrator) -> true, integrator_callback!)

    # Choose integration bounds
    if ctrl.node_flag # TODO: I can't find any mention of what this flag is for, can we get rid of it?
        while odet.psifac >= equil.sq.xs[ix]
            ix += 1
        end
        psiout = equil.sq.xs[ix]
        psiout = min(psiout, odet.psimax)
    else
        psiout = odet.psimax
    end

    # TODO: need input here! Fortran emphasized that it was important to record data at the first and last step
    # However, the recording for u, ud, psi is now done after integration. Is there anything still within
    # ode_output_step that needs to be recorded with this precision, or can it just be called within the
    # callback? Currently, ode_output_step does: ode_output_monitor (crit values), and in Fortran also does
    # ode_output_get_evals, and ode_output_sol
    # We don't use the `first` variable, and instead call the functions directly here before/after integration
    ode_output_step(ctrl, equil, intr, odet, outp)

    # Advance differential equation to next singular surface or edge
    rtol, atol = compute_tols(ctrl, intr, odet) # initial tolerances
    prob = ODEProblem(sing_der!, odet.u, (odet.psifac, psiout), (ctrl, equil, ffit, intr, odet, outp))
    sol = solve(prob, Tsit5(); reltol=rtol, abstol=atol, callback=cb)
    # TODO: check absolute tolerances, check how sensitive outputs are to tolerances

    # Update u and psifac with the solution at the end of the interval
    odet.u .= sol.u[end]
    odet.psifac = sol.t[end]

    # TODO: same as above, this might not be needed
    ode_output_step(ctrl, equil, intr, odet, outp) # always output final condition

    println("   ψ = $(odet.psifac), max u = $(maximum(abs, odet.u)), steps = $(odet.step-1)")
end

"""
    integrator_callback!(integrator)

Callback function for ODE integrator to handle normalization, output, and storage at each step.
This handles the logic that was previously in a DO loop within ode_run and called every step
by running LSODE in one step mode in the Fortran code.

### TODOs

Check if additional output is needed at the start and end of integration
"""
function integrator_callback!(integrator)
    ctrl, equil, ffit, intr, odet, outp = integrator.p
    odet.u .= integrator.u
    odet.psifac = integrator.t

    # Update integration tolerances
    rtol, atol = compute_tols(ctrl, intr, odet)
    integrator.opts.reltol = rtol
    # integrator.opts.abstol = atol

    # Note: no need for istep > 0 condition, since this is called after the first integration step always
    ode_unorm!(odet, ctrl, intr, outp, false)
    integrator.u .= odet.u # Update integrator.u with normalized odet.u

    # Compute eigenvalues and check for zero crossing
    # TODO: this logic can be updated - technically, if we save crit to memory,
    # we could do all these calculations in post processing if a stability calculation
    # is desired. If we keep it here, I'd rename to `check_for_zero_crossing` (what is
    # current `ode_output_monitor`). A lot of `ode_output.f` has been deprecated due to
    # in-memory integration
    ode_output_step(ctrl, equil, intr, odet, outp)
    # Compute total dW in edge region (if `psiedge < psilim`)
    # Same as above, this could technically be done in post processing if desired
    ode_record_edge_dW!(odet, ctrl, equil, ffit, intr)

    # Grow arrays if needed
    if odet.step >= size(odet.u_store, 4)
        resize_storage!(odet)
    end
    # Save values
    odet.psi_store[odet.step] = odet.psifac
    odet.u_store[:, :, :, odet.step] .= odet.u
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

### Returns

  - rtol: Relative tolerance
  - atol: Absolute tolerance array matching the shape of `odet.u`
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
    # Absolute tolerances
    atol = similar(odet.u, Float64)
    for ieq in 1:size(odet.u, 3), isol in 1:size(odet.u, 2)
        @views atol0 = maximum(abs, odet.u[:, isol, ieq]) * tol
        atol0 == 0 && (atol0 = Inf)
        atol[:, isol, ieq] .= atol0
    end
    return rtol, atol
end

"""
    resize_storage!(odet::OdeState)

Resize storage arrays in `odet` when the current step exceeds allocated size.
Doubles the size of the storage arrays for `u_store`, `ud_store`, and `psi_store`,
and copies over existing data to the new arrays.
"""
function resize_storage!(odet::OdeState)
    oldlen = size(odet.u_store, 4)
    newlen = 2 * oldlen

    # Allocate new arrays
    u_new = Array{ComplexF64,4}(undef, odet.mpert, odet.mpert, 2, newlen)
    ud_new = Array{ComplexF64,4}(undef, odet.mpert, odet.mpert, 2, newlen)
    psi_new = Vector{Float64}(undef, newlen)

    # Copy old data
    u_new[:, :, :, 1:odet.step] = odet.u_store[:, :, :, 1:odet.step]
    ud_new[:, :, :, 1:odet.step] = odet.ud_store[:, :, :, 1:odet.step]
    psi_new[1:odet.step] = odet.psi_store[1:odet.step]

    # Replace old arrays
    odet.u_store = u_new
    odet.ud_store = ud_new
    odet.psi_store = psi_new
end

"""
    trim_storage!(odet::OdeState)

Trim storage arrays in `odet` to the actual number of steps taken.
Resizes `u_store`, `ud_store`, and `psi_store` to the current step count,
removing any unused allocated space.
"""
function trim_storage!(odet::OdeState)
    resize!(odet.psi_store, odet.step)
    odet.u_store = odet.u_store[:, :, :, 1:odet.step]
    odet.ud_store = odet.ud_store[:, :, :, 1:odet.step]
end

"""
    ode_unorm!(odet::OdeState, ctrl::DconControl, intr::DconInternal, outp::DconOutput, sing_flag::Bool)

Computes norms of the solution vectors, normalizes them
relative to initial values, and applies Gaussian reduction via `ode_fixup!`
if the variation exceeds a threshold or if `sing_flag` is true.
Throws an error if any vector norm is zero. Performs the same function as `ode_unorm`
in the Fortran code, with minor differences in indexing and array handling.

### Arguments

  - sing_flag: Indicates if normalization is occuring at a singular surface or not

### TODOs

Add resizing logic for unorm arrays when ifix exceeds allocated size
"""
function ode_unorm!(odet::OdeState, ctrl::DconControl, intr::DconInternal, outp::DconOutput, sing_flag::Bool)
    # Compute norms of first solution vectors, abort if any are zero
    odet.unorm .= norm.(eachcol(odet.u[:, :, 1]))
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
            ode_fixup!(odet, intr, outp, sing_flag)
            odet.new = true
        end
    end
end

"""
    ode_fixup!(odet::OdeState, intr::DconInternal, sing_flag::Bool)

Applies Gaussian reduction to orthogonalize solution vectors in `odet.u`. Performs
the same function as `ode_fixup` in the Fortran code, except now `fixfac` and other
relevant data are stored in memory instead of dumped to `euler.bin`. Used when
the spread in norms exceeds a threshold or when a rational surface is reached.
"""
function ode_fixup!(odet::OdeState, intr::DconInternal, outp::DconOutput, sing_flag::Bool)

    # Write to output
    if outp.write_crit_out
        write_output(outp, :crit_out, "\n   psifac      dpsi        q       singfac     eval1")
        write_output(outp, :crit_out, @sprintf("\nGaussian Reduction at psi = %10.3e, q = %6.3f\n", odet.psifac, odet.q))
        if !sing_flag
            write_output(outp, :crit_out, "   psifac      dpsi        q       singfac     eval1\n")
        end
    end

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
    masked = zeros(typeof(abs(odet.u[1, 1, 1])), intr.mpert)
    for isol in 1:intr.mpert
        ksol = odet.index[isol, ifix]
        mask[2, ksol] = false
        # Find max location
        @. @views masked = abs(odet.u[:, ksol, 1]) * mask[1, 1:intr.mpert]
        @views kpert = argmax(masked)
        mask[1, kpert] = false
        for jsol in 1:intr.mpert
            if mask[2, jsol]
                odet.fixfac[ksol, jsol, ifix] = -odet.u[kpert, jsol, 1] / odet.u[kpert, ksol, 1]
                @. @views odet.u[:, jsol, :] .= odet.u[:, jsol, :] .+ odet.u[:, ksol, :] .* odet.fixfac[ksol, jsol, ifix]
                odet.u[kpert, jsol, 1] = 0
            end
        end
    end
end

"""
    ode_record_edge_dW!(odet::OdeState, ctrl::DconControl, equil::Equilibrium.PlasmaEquilibrium, ffit::FourFitVars, intr::DconInternal)

Records the total dW if the integration passes `ctrl.psiedge`, which occurs
if psiedge is less than the psilim determined in sing_lim!. This performs the same
function as `ode_record_edge` in the Fortran code, but the logic is different since
we store the integration in memory so we dont need the "_edge" arrays.

The dW is stored at that step index in `odet.dW_edge`; because we initialize
dW_edge to -Inf, we can just take the max value after integration to get the
total dW at the edge and avoid unphysical kink modes that might occur just
inside rational surfaces.

The first time this function is called, we create a rough spline for the wv matrix
in between psiedge and psilim using `free_compute_wv_spline`, which is then used in
`free_compute_total` to compute the total dW at the edge.
"""
function ode_record_edge_dW!(odet::OdeState, ctrl::DconControl, equil::Equilibrium.PlasmaEquilibrium, ffit::FourFitVars, intr::DconInternal)

    if odet.psifac >= ctrl.psiedge
        # Only create wv matrix spline once
        if !odet.wvmat_spline_created
            odet.wvmat_spline = free_compute_wv_spline(ctrl, equil, intr)
            odet.wvmat_spline_created = true
        end
        odet.dW_edge[odet.step] = free_compute_total(equil, ffit, intr, odet)
    end
end

"""
    get_psi_saves!(odet::OdeState, ctrl::DconControl, intr::DconInternal)

Calculates the psi values at which to save the solution during ODE integration based on specified spacing type,
number of saves per region, and number of rational surfaces, and saves into `odet.psi_save_nodes`. Note that
`odet.psi_save_nodes` does not store the start/endpoints, since those are saved automatically.
"""
# TODO: this is no longer used, but might be useful code? Leaving for now, but likely can be removed
function get_psi_saves!(odet::OdeState, ctrl::DconControl, intr::DconInternal)

    # This function is called after the ode init functions, so the starting psi is stored in odet.psifac
    # This repeats some logic used in the init and cross functions

    # Loop through interrational intervals (just one interval if no rational surfaces)
    for ising in 1:intr.msing+1
        # Determine bottom of interrational interval
        if ising == 1
            psibot = odet.psifac
        else
            psibot = intr.sing[ising-1].psifac + ctrl.singfac_min / abs(ctrl.nn * intr.sing[ising-1].q1)
        end
        # Determine top of interrational interval
        if ising <= intr.msing
            psitop = intr.sing[ising].psifac - ctrl.singfac_min / abs(ctrl.nn * intr.sing[ising].q1)
        else
            psitop = intr.psilim * (1 - eps)
        end
        # Calculate save nodes in interval
        if ctrl.save_spacing == "Chebyshev"
            odet.psi_saves[(ising-1)*ctrl.saves_per_region+1:ising*ctrl.saves_per_region] = chebyshev_nodes(psibot, psitop, ctrl.saves_per_region)
        else
            error("Cannot recognize save_spacing = $(ctrl.save_spacing)")
        end
    end
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