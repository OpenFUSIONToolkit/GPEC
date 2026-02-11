# JPEC.jl
module JPEC

include("Splines/Splines.jl")
import .SplinesMod as Spl
export SplinesMod, Spl

include("Utilities/Utilities.jl")
import .Utilities as Utilities
export Utilities

include("Equilibrium/Equilibrium.jl")
import .Equilibrium as Equilibrium
export Equilibrium

include("Vacuum/Vacuum.jl")
import .Vacuum as Vacuum
export Vacuum

include("ForceFreeStates/ForceFreeStates.jl")
import .ForceFreeStates as ForceFreeStates
export ForceFreeStates

include("ForcingTerms/ForcingTerms.jl")
import .ForcingTerms as ForcingTerms
export ForcingTerms

include("PerturbedEquilibrium/PerturbedEquilibrium.jl")
import .PerturbedEquilibrium as PerturbedEquilibrium
export PerturbedEquilibrium

include(joinpath(@__DIR__, "..", "deps", "build_helpers.jl"))
export build_fortran, build_spline_fortran, build_vacuum_fortran

# Additional imports for main function
using TOML
using Printf
using HDF5

# Import ForceFreeStates types and functions needed for main
using .ForceFreeStates: ForceFreeStatesInternal, ForceFreeStatesControl, DebugSettings, VacuumData, OdeState
using .ForceFreeStates: sing_lim!, sing_find!, mercier_scan!, sing_scan!, ode_run, free_run!
using .ForceFreeStates: make_metric, make_matrix

function main(args::Vector{String}=String[])
    # Parse command line arguments
    path = length(args) >= 1 ? args[1] : "./"

    println("\n" * "="^60)
    println("  JPEC - Julia Perturbed Equilibrium Code")
    println("="^60 * "\n")

    start_time = time()

    # Read input data and set up data structures
    intr = ForceFreeStatesInternal(; dir_path=path)
    inputs = TOML.parsefile(joinpath(intr.dir_path, "jpec.toml"))
    ctrl = ForceFreeStatesControl(; (Symbol(k) => v for (k, v) in inputs["ForceFreeStates"])...)

    # Set up equilibrium from jpec.toml or fallback to equil.toml if it exists
    if "Equilibrium" in keys(inputs)
        eq_config = Equilibrium.EquilibriumConfig(inputs["Equilibrium"], intr.dir_path)
        equil = Equilibrium.setup_equilibrium(eq_config)
    elseif isfile(joinpath(intr.dir_path, "equil.toml"))
        @warn "Reading from equil.toml is deprecated. Please move [EQUIL_CONTROL] and [EQUIL_OUTPUT] sections to [Equilibrium] in jpec.toml"
        equil = Equilibrium.setup_equilibrium(joinpath(intr.dir_path, "equil.toml"))
    else
        error("No equilibrium configuration found. Add [Equilibrium] section to jpec.toml")
    end
    # Early exit if user only requested equilibrium setup
    if equil.config.force_termination
        end_time = time() - start_time
        println("\n" * "="^60)
        println("Equilibrium setup complete (force_termination = true).")
        println("Run time: $(@sprintf("%.3e", end_time)) seconds")
        println("Normal termination.")
        println("="^60)
        return
    end


    if "Wall" in keys(inputs)
        intr.wall_settings = Vacuum.WallShapeSettings(; (Symbol(k) => v for (k, v) in inputs["Wall"])...)
    else
        intr.wall_settings = Vacuum.WallShapeSettings()
    end

    if "DEBUG" in keys(inputs)
        intr.debug_settings = DebugSettings(; (Symbol(k) => v for (k, v) in inputs["DEBUG"])...)
    else
        intr.debug_settings = DebugSettings()
    end
    
    # Set up variables
    # TODO: dcon_kin_threads logic?
    ctrl.delta_mhigh *= 2 # for consistency with Fortran DCON TODO: why is this present in the Fortran?

    # Determine psilim and qlim (where we will integrate to)
    sing_lim!(intr, ctrl, equil)
    if ctrl.set_psilim_via_dmlim && ctrl.psiedge < intr.psilim
        @warn "Only one of set_psilim_via_dmlim and psiedge < psilim can be used at a time.
            Setting psiedge = 1.0 and determining dW from psilim = $(intr.psilim) determined from dmlim = $(ctrl.dmlim)."
        ctrl.psiedge = 1.0
    end

    # If truncating before psihigh, reform equilibrium if desired
    if intr.psilim != equil.config.psihigh && ctrl.reform_eq_with_psilim
        @warn "Reforming equilibrium splines from psihigh to psilim not implemented yet. Proceeding with psihigh = $(equil.config.psihigh)."
        # JMH - Nik please put the logic we discussed here
        # something like ?
        # equil.config.psihigh = intr.psilim
        # equil = set_up_equilibrium(equil.config)
    end

    # Compute Mercier and Ballooning stability (if desired)
    # This holds di, dr, h (calculated in mercier_scan), ca1, and ca2 (calculated in ballooning scan)
    # Compute Mercier and Ballooning stability (if desired)
    # This holds di, dr, h (calculated in mercier_scan), ca1, and ca2 (calculated in ballooning scan)
    profiles_xs = equil.profiles.xs
    locstab_fs = zeros(Float64, length(profiles_xs), 5)
    if ctrl.mer_flag
        if ctrl.verbose
            println("Evaluating Mercier criterion")
        end
        mercier_scan!(locstab_fs, equil)
    end
    if ctrl.bal_flag
        compute_ballooning_stability!(ctrl, locstab_fs, equil)
    end
    # Fit data to splines
    intr.locstab = cubic_interp(profiles_xs, locstab_fs; bc=CubicFit(), extrap=:extension)

    # Determine toroidal mode numbers
    if ctrl.nn_low == 0 && ctrl.nn_high == 0
        error("Either nn_low or nn_high must be set in DCON_CONTROL (both are 0)")
    elseif ctrl.nn_low == 0
        ctrl.nn_low = ctrl.nn_high
    elseif ctrl.nn_high == 0
        ctrl.nn_high = ctrl.nn_low
    end
    if ctrl.nn_low > ctrl.nn_high
        error("nn_low cannot be greater than nn_high")
    end
    intr.nlow = ctrl.nn_low
    intr.nhigh = ctrl.nn_high
    intr.npert = intr.nhigh - intr.nlow + 1
    nstring = intr.npert == 1 ? "$(intr.nlow)" : "$(intr.nlow):$(intr.nhigh)"

    # Find all singular surfaces in the equilibrium
    sing_find!(intr, equil)

    # Determine poloidal mode numbers
    if ctrl.delta_mlow < 0 || ctrl.delta_mhigh < 0
        error("Negative delta_mlow or delta_mhigh not allowed")
    end
    if ctrl.cyl_flag
        intr.mlow = ctrl.delta_mlow
        intr.mhigh = ctrl.delta_mhigh
    elseif ctrl.sing_start == 0
        intr.mlow = min(intr.nlow * equil.params.qmin, 0) - 4 - ctrl.delta_mlow
        intr.mhigh = trunc(Int, intr.nhigh * equil.params.qmax) + ctrl.delta_mhigh
    else
        intr.mmin = Inf # HUGE in Fortran
        for ising in Int(ctrl.sing_start):intr.msing
            intr.mmin = min(intr.mmin, sing[ising].m)
        end
        intr.mlow = intr.mmin - ctrl.delta_mlow
        intr.mhigh = trunc(Int, intr.nhigh * equil.params.qmax) + ctrl.delta_mhigh
    end
    intr.mpert = intr.mhigh - intr.mlow + 1
    if ctrl.delta_mband >= intr.mpert
        @warn "Banded matrices not implemented yet, setting delta_mband to 0"
        ctrl.delta_mband = 0
    end
    intr.mband = intr.mpert - 1 - ctrl.delta_mband
    intr.mband = min(max(intr.mband, 0), intr.mpert - 1)
    intr.numpert_total = intr.mpert * intr.npert

    # Fit equilibrium quantities to Fourier-spline functions.
    if ctrl.mat_flag || ctrl.ode_flag
        if ctrl.verbose
            println("Run parameters:")
            println("   q0 = $(@sprintf("%.3f", equil.params.q0)), qmin = $(@sprintf("%.3f", equil.params.qmin)), qmax = $(@sprintf("%.3f", equil.params.qmax)), q95 = $(@sprintf("%.3f", equil.params.q95))")
            println("   qlim = $(@sprintf("%.5f", intr.qlim)), psilim = $(@sprintf("%.9f", intr.psilim))")
            println("   betat = $(@sprintf("%.3f", equil.params.betat)), betan = $(@sprintf("%.3f", equil.params.betan)), betap1 = $(@sprintf("%.3f", equil.params.betap1))")
            println("   mlow = $(@sprintf("%4i", intr.mlow)), mhigh = $(@sprintf("%4i", intr.mhigh)), mpert = $(@sprintf("%4i", intr.mpert)), mband = $(@sprintf("%4i", intr.mband))")
            println("   nlow = $(@sprintf("%4i", intr.nlow)), nhigh = $(@sprintf("%4i", intr.nhigh)), npert = $(@sprintf("%4i", intr.npert))")
        end

        # Compute metric tensor
        metric = make_metric(equil; mband=intr.mband, fft_flag=ctrl.fft_flag)

        if ctrl.verbose
            println("   Computing F, G, and K Matrices")
        end

        # Compute matrices and populate FourFitVars struct
        ffit = make_matrix(equil, intr, metric)

        if ctrl.kin_flag
            error("kin_flag not implemented yet")
        end

        # NOTE: Asymptotic calculations for ideal DCON are now computed on-demand during
        # singular surface crossings in cross_ideal_singular_surf!. This makes it clear that
        # asymptotics are only needed for ideal DCON and are not inherent properties of
        # the singular surface.

    end

    # Integrate Euler-Lagrange Equation
    if ctrl.ode_flag
        if ctrl.verbose
            println("Integrating Euler-Lagrange equation")
        end
        odet = eulerlagrange_integration(ctrl, equil, ffit, intr)
        if odet.nzero > 0 && ctrl.verbose
            println("Fixed-boundary mode unstable for n = $nstring.")
        end
    end

    # Compute free boundary energies
    if ctrl.vac_flag && !(ctrl.ksing > 0 && ctrl.ksing <= intr.msing + 1)
        if ctrl.verbose
            println("Computing free boundary energies")
        end
        vac_data = free_run!(odet, ctrl, equil, ffit, intr)
        if real(vac_data.et[1]) < 0
            if ctrl.verbose
                println("Free-boundary mode unstable for n = $nstring.")
            end
        else
            if ctrl.verbose
                println("All free-boundary modes stable for n = $nstring.")
            end
        end
    end

    if ctrl.write_outputs_to_HDF5
        if ctrl.verbose
            println("Writing saved data to $(ctrl.HDF5_filename)")
        end
        write_outputs_to_HDF5(ctrl, equil, intr, odet, ctrl.vac_flag ? vac_data : nothing)
    end

    # Early exit if user only requested force-free states
    if ctrl.force_termination
        end_time = time() - start_time
        println("\n" * "="^60)
        println("Force-free states complete (force_termination = true).")
        println("Run time: $(@sprintf("%.3e", end_time)) seconds")
        println("Normal termination.")
        println("="^60)
        return
    end

    # Check for PerturbedEquilibrium section and run if present
    if "PerturbedEquilibrium" in keys(inputs)
        # Read ForcingTerms control parameters
        if "ForcingTerms" in keys(inputs)
            ft_ctrl = ForcingTerms.ForcingTermsControl(;
                (Symbol(k) => v for (k, v) in inputs["ForcingTerms"])...
            )
        else
            ft_ctrl = ForcingTerms.ForcingTermsControl()  # Use defaults
        end

        pe_ctrl = PerturbedEquilibrium.PerturbedEquilibriumControl(;
            (Symbol(k) => v for (k, v) in inputs["PerturbedEquilibrium"])...
        )
        pe_intr = PerturbedEquilibrium.PerturbedEquilibriumInternal(; dir_path=intr.dir_path)

        # Run perturbed equilibrium calculations
        # Pass vac_data and intr for response matrix calculations
        pe_state = PerturbedEquilibrium.compute_perturbed_equilibrium(
            equil, odet, ctrl.vac_flag ? vac_data : nothing, intr, ft_ctrl, pe_ctrl, pe_intr
        )

        # Write perturbed equilibrium outputs to same HDF5 file
        if pe_ctrl.write_outputs_to_HDF5
            output_file = isempty(pe_ctrl.output_filename) ? ctrl.HDF5_filename : pe_ctrl.output_filename
            PerturbedEquilibrium.write_outputs_to_HDF5(
                pe_state, pe_intr, pe_ctrl, joinpath(intr.dir_path, output_file)
            )
        end
    end

    end_time = time() - start_time
    println("\n" * "="^60)
    println("Run time: $(@sprintf("%.3e", end_time)) seconds")
    println("Normal termination.")
    println("="^60)

    # TODO: Do not allow perturbed equilibrium calculations if zero crossings are found

    return (ctrl=ctrl, equil=equil, intr=intr, ffit=ffit, odet=odet, vac_data=ctrl.vac_flag ? vac_data : nothing)

end

"""
    write_outputs_to_HDF5(ctrl::ForceFreeStatesControl, equil::Equilibrium.PlasmaEquilibrium, intr::ForceFreeStatesInternal, odet::OdeState)

Helper function to write the HDF5 output file with relevant run and equilibrium parameters.
This combines the functionality of several pieces of the Fortran code in `ode_output.f`,
primarily `ode_output_open` and the various `bin_euler` writes that occur throughout the
integration. Some parameters are only dumped in their respective flags are true, e.g.
vacuum data if `vac_flag` is true.

### TODOs

Combine spline unpacking if possible, too many extra lines

"""
function write_outputs_to_HDF5(ctrl::ForceFreeStatesControl, equil::Equilibrium.PlasmaEquilibrium, intr::ForceFreeStatesInternal, odet::OdeState, vac::Union{VacuumData, Nothing})

    h5open(joinpath(intr.dir_path, ctrl.HDF5_filename), "w") do out_h5

        # Store input parameters
        for (key, val) in zip(fieldnames(ForceFreeStatesControl), getfield.(Ref(ctrl), fieldnames(ForceFreeStatesControl)))
            out_h5["input/ForceFreeStates/$key"] = val
        end
        for (key, val) in zip(fieldnames(Equilibrium.EquilibriumConfig), getfield.(Ref(equil.config), fieldnames(Equilibrium.EquilibriumConfig)))
            out_h5["input/EQUIL_CONTROL/$key"] = val
        end
        # TODO: assuming EQUIL_OUTPUT is going to be deprecated
        # TODO: should we store the equilibrium? difficult since it could be a gfile, sol.in, etc.
        # TODO: if we do one input file, can just pass that in instead and loop easily since its parsed
        # as a dict already (for (k, v) in inputs["DCON_CONTROL"]...). We have to do this since custom structs
        # don't inherently have an iterator by default

        # Write derived run parameters
        out_h5["info/mpert"] = intr.mpert
        out_h5["info/mband"] = intr.mband
        out_h5["info/mlow"] = intr.mlow
        out_h5["info/mhigh"] = intr.mhigh
        out_h5["info/npert"] = intr.npert
        out_h5["info/nlow"] = intr.nlow
        out_h5["info/nhigh"] = intr.nhigh
        m = [(i - 1) % intr.mpert + intr.mlow for i in 1:(intr.numpert_total)]
        n = [(i - 1) ÷ intr.mpert + intr.nlow for i in 1:(intr.numpert_total)]
        out_h5["info/mn_index"] = hcat(m, n)   # (N, 2) matrix
        out_h5["info/psilim"] = intr.psilim
        out_h5["info/qlim"] = intr.qlim
        out_h5["info/q1lim"] = intr.q1lim

        # Write derived equilibrium parameters
        for (key, val) in zip(fieldnames(Equilibrium.EquilibriumParameters), getfield.(Ref(equil.params), fieldnames(Equilibrium.EquilibriumParameters)))
            if val !== nothing # TODO: looks like ro, zo, psio, and b_norm are not set, so skipping those for now but should fix eventually
                out_h5["equil/$key"] = val
            end
        end
        out_h5["equil/psio"] = equil.psio
        out_h5["equil/ro"] = equil.ro
        out_h5["equil/zo"] = equil.zo

        # Write spline arrays (using profiles with named splines)
        profiles = equil.profiles
        out_h5["splines/profiles/xs"] = profiles.xs
        out_h5["splines/profiles/2piF"] = profiles.F_spline.y
        out_h5["splines/profiles/mu0p"] = profiles.P_spline.y
        out_h5["splines/profiles/dVdpsi"] = profiles.dVdpsi_spline.y
        out_h5["splines/profiles/q"] = profiles.q_spline.y
        out_h5["splines/rzphi/xs"] = equil.rzphi_xs
        out_h5["splines/rzphi/ys"] = equil.rzphi_ys
        # Extract grid point values from interpolants for HDF5 output
        out_h5["splines/rzphi/rcoords"] = equil.rzphi_rsquared.nodal_derivs.partials[1, :, :]
        out_h5["splines/rzphi/offset"] = equil.rzphi_offset.nodal_derivs.partials[1, :, :]
        out_h5["splines/rzphi/nu"] = equil.rzphi_nu.nodal_derivs.partials[1, :, :]
        out_h5["splines/rzphi/jac"] = equil.rzphi_jac.nodal_derivs.partials[1, :, :]

        # Write local stability data
        if ctrl.mer_flag
            locstab_xs = intr.locstab.cache.x
            out_h5["locstab/di"] = intr.locstab.y[:, 1] ./ locstab_xs
            out_h5["locstab/dr"] = intr.locstab.y[:, 2] ./ locstab_xs
            out_h5["singular/di0"] = [intr.locstab(sing.psifac)[1] / sing.psifac for sing in intr.sing]
        end
        if ctrl.bal_flag
            out_h5["locstab/ca1"] = intr.locstab.y[:, 4]
        end

        # Write integration data
        # TODO: technically this should only be written if ode_flag is true, but that's going to get deprecated eventually
        out_h5["integration/nstep"] = odet.step
        out_h5["integration/psi"] = odet.psi_store
        out_h5["integration/q"] = odet.q_store
        out_h5["integration/xi_psi"] = odet.u_store[:, :, 1, :]
        out_h5["integration/u2"] = odet.u_store[:, :, 2, :] # TODO: what to name this? These are the "conjugate momenta" of u1
        out_h5["integration/dxi_psi"] = odet.ud_store[:, :, 1, :]
        out_h5["integration/xi_s"] = odet.ud_store[:, :, 2, :]
        out_h5["integration/crit"] = odet.crit_store

        # Write singular surface data
        out_h5["singular/msing"] = intr.msing
        out_h5["singular/psi"] = [sing.psifac for sing in intr.sing]
        out_h5["singular/q"] = [sing.q for sing in intr.sing]
        out_h5["singular/q1"] = [sing.q1 for sing in intr.sing]
        out_h5["singular/ca_left"] = odet.ca_l
        out_h5["singular/ca_right"] = odet.ca_r

        # Write vacuum Data
        if ctrl.vac_flag
            out_h5["vacuum/wt"] = vac.wt
            out_h5["vacuum/wt0"] = vac.wt0
            out_h5["vacuum/ep"] = vac.ep
            out_h5["vacuum/ev"] = vac.ev
            out_h5["vacuum/et"] = vac.et
            out_h5["vacuum/x_plasma"] = vac.xzpts[:, 1]
            out_h5["vacuum/z_plasma"] = vac.xzpts[:, 2]
            out_h5["vacuum/x_wall"] = vac.xzpts[:, 3]
            out_h5["vacuum/z_wall"] = vac.xzpts[:, 4]
        end
    end
end

export main

end # module JPEC
