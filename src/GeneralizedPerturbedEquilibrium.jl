# GeneralizedPerturbedEquilibrium.jl
module GeneralizedPerturbedEquilibrium

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

include("InnerLayer/InnerLayer.jl")
import .InnerLayer as InnerLayer
export InnerLayer

include("ForcingTerms/ForcingTerms.jl")
import .ForcingTerms as ForcingTerms
export ForcingTerms

include("PerturbedEquilibrium/PerturbedEquilibrium.jl")
import .PerturbedEquilibrium as PerturbedEquilibrium
export PerturbedEquilibrium

include("Analysis/Analysis.jl")
import .Analysis as Analysis
export Analysis

# Additional imports for main function
using TOML
using Printf
using HDF5

using FastInterpolations

import AdaptiveArrayPools: @with_pool

# Import ForceFreeStates types and functions needed for main
using .ForceFreeStates: ForceFreeStatesInternal, ForceFreeStatesControl, DebugSettings, VacuumData, OdeState, FourFitVars
using .ForceFreeStates: sing_lim!, sing_find!
using .ForceFreeStates: mercier_scan!, compute_ballooning_stability!
using .ForceFreeStates: make_metric, make_matrix, make_kinetic_matrix
using .ForceFreeStates: eulerlagrange_integration, free_run!

const _BANNER = "="^60
const _SECTION = "-"^40

function main(args::Vector{String}=String[])
    # Parse command line arguments
    path = length(args) >= 1 ? args[1] : "./"

    # Capture git version for reproducibility
    git_version = try
        String(readchomp(`git -C $(@__DIR__) describe --tags --always`))
    catch
        "unknown"
    end

    @info "\n$_BANNER\n  GPEC - Generalized Perturbed Equilibrium Code  [$git_version]\n$_BANNER"
    total_start = time()

    # ----------------------------------------------------------------
    # Equilibrium
    # ----------------------------------------------------------------
    @info "\n  Equilibrium\n$_SECTION"
    equil_start = time()

    # Read input data and set up data structures
    intr = ForceFreeStatesInternal(; dir_path=path)
    inputs = TOML.parsefile(joinpath(intr.dir_path, "gpec.toml"))

    ctrl = ForceFreeStatesControl(; (Symbol(k) => v for (k, v) in inputs["ForceFreeStates"])...)

    # Set up equilibrium from gpec.toml or fallback to equil.toml if it exists.
    # Analytic equilibria ("tj_analytic", "tj_analytic_direct", "sol", "lar") can
    # EITHER point `eq_filename` at a side-car TOML (legacy) OR embed their
    # parameters directly in gpec.toml under a top-level section:
    # [TJ_ANALYTIC_INPUT], [SOL_INPUT], [LAR_INPUT].  When the embedded section
    # is present it takes precedence and the side-car file is not consulted,
    # so a run is fully described by a single gpec.toml.
    #
    # The TJ-analytic equilibrium follows the profile family of
    # R. Fitzpatrick's TJ code (https://github.com/rfitzp/TJ); see
    # `Equilibrium.TJAnalyticConfig`.
    if "Equilibrium" in keys(inputs)
        eq_config = Equilibrium.EquilibriumConfig(inputs["Equilibrium"], intr.dir_path)
        additional_input = nothing
        if eq_config.eq_type in ("tj_analytic", "tj_analytic_direct") && haskey(inputs, "TJ_ANALYTIC_INPUT")
            additional_input = Equilibrium.TJAnalyticConfig(inputs["TJ_ANALYTIC_INPUT"])
        elseif eq_config.eq_type == "sol" && haskey(inputs, "SOL_INPUT")
            additional_input = Equilibrium.SolovevConfig(inputs["SOL_INPUT"])
        elseif eq_config.eq_type == "lar" && haskey(inputs, "LAR_INPUT")
            additional_input = Equilibrium.LargeAspectRatioConfig(inputs["LAR_INPUT"])
        end
        equil = Equilibrium.setup_equilibrium(eq_config, additional_input)
    elseif isfile(joinpath(intr.dir_path, "equil.toml"))
        @warn "Reading from equil.toml is deprecated. Please move [EQUIL_CONTROL] and [EQUIL_OUTPUT] sections to [Equilibrium] in gpec.toml"
        equil = Equilibrium.setup_equilibrium(joinpath(intr.dir_path, "equil.toml"))
    else
        error("No equilibrium configuration found. Add [Equilibrium] section to gpec.toml")
    end

    @info "Equilibrium construction completed in $(@sprintf("%.3f", time() - equil_start)) s"

    # Early exit if user only requested equilibrium setup
    if equil.config.force_termination
        @info "\n$_BANNER\n  GPEC completed successfully in $(@sprintf("%.3f", time() - total_start)) s\n$_BANNER"
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

    # ----------------------------------------------------------------
    # Force-Free States
    # ----------------------------------------------------------------
    @info "\n  Force-Free States\n$_SECTION"
    ffs_start = time()

    # Set up variables
    # TODO: parallel threads logic
    ctrl.delta_mhigh *= 2 # for consistency with Fortran DCON TODO: why is this present in the Fortran?

    # Determine psilim and qlim (where we will integrate to)
    sing_lim!(intr, ctrl, equil)

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
    profiles_xs = equil.profiles.xs
    locstab_fs = zeros(Float64, length(profiles_xs), 5)
    if ctrl.mer_flag
        if ctrl.verbose
            @info "Evaluating Mercier criterion"
        end
        mercier_scan!(locstab_fs, equil)
    end
    if ctrl.bal_flag
        compute_ballooning_stability!(ctrl, locstab_fs, equil)
    end
    # Fit data to splines
    intr.locstab = cubic_interp(profiles_xs, Series(locstab_fs); extrap=ExtendExtrap())

    # Determine toroidal mode numbers (n >= 1 required; 0 means "not specified")
    if ctrl.nn_low == 0 && ctrl.nn_high == 0
        error("Either nn_low or nn_high must be set in [ForceFreeStates] (both are 0)")
    elseif ctrl.nn_low == 0
        ctrl.nn_low = ctrl.nn_high
    elseif ctrl.nn_high == 0
        ctrl.nn_high = ctrl.nn_low
    end
    if ctrl.nn_low > ctrl.nn_high
        error("nn_low=$(ctrl.nn_low) cannot be greater than nn_high=$(ctrl.nn_high)")
    end
    # checks for negative n
    # note that negative n in fortran had code adding the identitiy matrix to grad Green for n=0
    # and some n, nu sign switching in vacuum but was not actually supported by DCON sing_find, etc.
    if ctrl.nn_high < 1
        error("All requested toroidal modes (n=$(ctrl.nn_low):$(ctrl.nn_high)) are below 1; " *
              "n < 1 modes are not supported")
    end
    if ctrl.nn_low < 1
        @warn "Clamping nn_low from $(ctrl.nn_low) to 1; n < 1 modes are not supported"
        ctrl.nn_low = 1
    end
    intr.nlow = ctrl.nn_low
    intr.nhigh = ctrl.nn_high
    intr.npert = intr.nhigh - intr.nlow + 1
    nstring = intr.npert == 1 ? "$(intr.nlow)" : "$(intr.nlow):$(intr.nhigh)"

    # Find all singular surfaces in the equilibrium
    sing_find!(intr, equil)

    # Filter out surfaces outside the integration domain [qlow, qlim].
    # Fortran STRIDE excludes these at the integration level; we remove them
    # from intr.sing so the Δ' BVP sees only crossable surfaces.
    if intr.msing > 0
        qmin_integration = max(ctrl.qlow, equil.params.qmin)
        n_before = intr.msing
        keep = [j for j in 1:intr.msing if intr.sing[j].q >= qmin_integration && intr.sing[j].psifac <= intr.psilim]
        if length(keep) < n_before
            excluded = setdiff(1:n_before, keep)
            excluded_mq = [(intr.sing[j].m, intr.sing[j].q) for j in excluded]
            @info "Filtered $(n_before - length(keep)) singular surface(s) outside integration domain: $(excluded_mq)"
            intr.sing = intr.sing[keep]
            intr.msing = length(keep)
        end
    end

    # Determine poloidal mode numbers
    if ctrl.delta_mlow < 0 || ctrl.delta_mhigh < 0
        error("Negative delta_mlow or delta_mhigh not allowed")
    end
    if ctrl.cyl_flag
        intr.mlow = ctrl.delta_mlow
        intr.mhigh = ctrl.delta_mhigh
    elseif ctrl.sing_start == 0
        intr.mlow = trunc(Int, min(intr.nlow * equil.params.qmin, 0)) - 4 - ctrl.delta_mlow
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
            @info "Run parameters:\n" *
                  "   q0 = $(@sprintf("%.3f", equil.params.q0)), qmin = $(@sprintf("%.3f", equil.params.qmin)), qmax = $(@sprintf("%.3f", equil.params.qmax)), q95 = $(@sprintf("%.3f", equil.params.q95))\n" *
                  "   qlim = $(@sprintf("%.3f", intr.qlim)), psilim = $(@sprintf("%.3f", intr.psilim))\n" *
                  "   betat = $(@sprintf("%.3f", equil.params.betat)), betan = $(@sprintf("%.3f", equil.params.betan)), betap1 = $(@sprintf("%.3f", equil.params.betap1))\n" *
                  "   mlow = $(@sprintf("%4i", intr.mlow)), mhigh = $(@sprintf("%4i", intr.mhigh)), mpert = $(@sprintf("%4i", intr.mpert)), mband = $(@sprintf("%4i", intr.mband))\n" *
                  "   nlow = $(@sprintf("%4i", intr.nlow)), nhigh = $(@sprintf("%4i", intr.nhigh)), npert = $(@sprintf("%4i", intr.npert))"
        end

        # Compute metric tensor
        metric = make_metric(equil; mband=intr.mband, fft_flag=ctrl.fft_flag)

        if ctrl.verbose
            @info "Computing F, G, and K matrices"
        end

        # Compute matrices and populate FourFitVars struct
        ffit = make_matrix(equil, intr, metric)

        if ctrl.kinetic_factor > 0
            if ctrl.verbose
                @info "Computing kinetic matrices (source: $(ctrl.kinetic_source), factor: $(ctrl.kinetic_factor))"
            end
            make_kinetic_matrix(ctrl, equil, ffit, intr, metric)
        end

        # NOTE: Asymptotic calculations for ideal ForceFreeStates are now computed on-demand during
        # singular surface crossings in cross_ideal_singular_surf!. This makes it clear that
        # asymptotics are only needed for ideal ForceFreeStates and are not inherent properties of
        # the singular surface.

    end

    # Integrate Euler-Lagrange Equation
    if ctrl.ode_flag
        if ctrl.verbose
            @info "Integrating Euler-Lagrange equation"
        end
        odet, fm_propagators, fm_chunks, fm_S_left = eulerlagrange_integration(ctrl, equil, ffit, intr)
        if odet.nzero > 0 && ctrl.verbose
            @warn "Fixed-boundary mode unstable for n = $nstring"
        end
    end

    # Compute free boundary energies
    if ctrl.vac_flag && !(ctrl.ksing > 0 && ctrl.ksing <= intr.msing + 1)
        if ctrl.verbose
            wall_desc = intr.wall_settings.shape == "nowall" ? "no wall" : intr.wall_settings.shape
            @info "Computing free boundary energies ($wall_desc)"
        end
        vac_data = free_run!(odet, ctrl, equil, ffit, intr)
        if real(vac_data.et[1]) < 0
            if ctrl.verbose
                @warn "Free-boundary mode unstable for n = $nstring"
            end
        else
            if ctrl.verbose
                @info "All free-boundary modes stable for n = $nstring"
            end
        end

        # Compute inter-surface Δ' matrix (STRIDE BVP) using vacuum edge BC.
        # Requires propagators from parallel FM path and wv from free_run!.
        if ctrl.kinetic_factor == 0 && intr.msing > 0 && fm_propagators !== nothing
            if ctrl.verbose
                @info "Computing Δ' matrix (STRIDE BVP with vacuum coupling)"
            end
            ForceFreeStates.compute_delta_prime_matrix!(intr, fm_propagators, fm_chunks;
                wv=vac_data.wv, psio=equil.psio, debug=ctrl.verbose,
                S_at_surface_left=fm_S_left,
                ctrl=ctrl, equil=equil, ffit=ffit)
        end
    end

    if ctrl.write_outputs_to_HDF5
        write_outputs_to_HDF5(ctrl, equil, intr, odet, ctrl.vac_flag ? vac_data : nothing, ffit, git_version)
        @info "Results written to $(ctrl.HDF5_filename)"
    end

    @info "Force-Free States completed in $(@sprintf("%.3f", time() - ffs_start)) s"

    # Early exit if user only requested force-free states
    if ctrl.force_termination
        @info "\n$_BANNER\n  GPEC completed successfully in $(@sprintf("%.3f", time() - total_start)) s\n$_BANNER"
        return
    end

    # ----------------------------------------------------------------
    # Perturbed Equilibrium
    # ----------------------------------------------------------------
    @info "\n  Perturbed Equilibrium\n$_SECTION"
    pe_start = time()

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
            @info "Results written to $output_file"
        end
    end

    @info "Perturbed Equilibrium completed in $(@sprintf("%.3f", time() - pe_start)) s"

    # ----------------------------------------------------------------
    # Done
    # ----------------------------------------------------------------
    @info "\n$_BANNER\n  GPEC completed successfully in $(@sprintf("%.3f", time() - total_start)) s\n$_BANNER"

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
function write_outputs_to_HDF5(
    ctrl::ForceFreeStatesControl,
    equil::Equilibrium.PlasmaEquilibrium,
    intr::ForceFreeStatesInternal,
    odet::OdeState,
    vac_data::Union{VacuumData,Nothing},
    ffit::Union{FourFitVars,Nothing}=nothing,
    git_version::String="unknown"
)

    h5open(joinpath(intr.dir_path, ctrl.HDF5_filename), "w") do out_h5

        # Store git version for reproducibility
        out_h5["info/git_version"] = git_version

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
        # as a dict already (for (k, v) in inputs["ForceFreeStates"]...). We have to do this since custom structs
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

        # Write local stability data; always write all entries, using empty arrays when not computed
        if ctrl.mer_flag
            locstab_xs = intr.locstab.cache.x
            out_h5["locstab/di"] = intr.locstab.y[:, 1] ./ locstab_xs
            out_h5["locstab/dr"] = intr.locstab.y[:, 2] ./ locstab_xs
        else
            out_h5["locstab/di"] = Float64[]
            out_h5["locstab/dr"] = Float64[]
        end
        out_h5["singular/di0"] = (ctrl.mer_flag && !isempty(intr.sing)) ?
                                 [intr.locstab(sing.psifac)[1] / sing.psifac for sing in intr.sing] : Float64[]
        out_h5["locstab/ca1"] = ctrl.bal_flag ? intr.locstab.y[:, 4] : Float64[]

        # Write integration data
        # TODO: technically this should only be written if ode_flag is true, but that's going to get deprecated eventually
        out_h5["integration/nstep"] = odet.step            # Number of saved solution snapshots
        out_h5["integration/nstep_total"] = odet.total_steps  # Total ODE solver steps taken
        out_h5["integration/psi"] = odet.psi_store
        out_h5["integration/q"] = odet.q_store
        out_h5["integration/xi_psi"] = odet.u_store[:, :, 1, :]
        out_h5["integration/u2"] = odet.u_store[:, :, 2, :] # TODO: what to name this? These are the "conjugate momenta" of u1
        out_h5["integration/dxi_psi"] = odet.ud_store[:, :, 1, :]
        out_h5["integration/xi_s"] = odet.ud_store[:, :, 2, :]
        out_h5["integration/crit"] = odet.crit_store

        # Write edge stability scan data (only present when psiedge < psilim)
        if !isempty(odet.edge_scan.psi)
            es = odet.edge_scan
            out_h5["edge_scan/psi"] = es.psi
            out_h5["edge_scan/q"] = es.q
            out_h5["edge_scan/total_energy"] = es.total_eigenvalue
            out_h5["edge_scan/plasma_energy"] = es.plasma_energy
            out_h5["edge_scan/vacuum_energy"] = es.vacuum_energy
            out_h5["edge_scan/vacuum_eigenvalue"] = es.vacuum_eigenvalue
        end

        # Write singular surface data
        out_h5["singular/msing"] = intr.msing
        out_h5["singular/psi"] = [sing.psifac for sing in intr.sing]
        out_h5["singular/q"] = [sing.q for sing in intr.sing]
        out_h5["singular/q1"] = [sing.q1 for sing in intr.sing]
        out_h5["singular/ca_left"] = odet.ca_l
        out_h5["singular/ca_right"] = odet.ca_r

        if intr.msing > 0
            # Mode numbers at each surface (jagged — pad with 0 to max_modes width)
            max_modes = maximum(s -> length(s.m), intr.sing)
            m_matrix = zeros(Int, intr.msing, max_modes)
            n_matrix = zeros(Int, intr.msing, max_modes)
            for (s, sing) in enumerate(intr.sing)
                for i in 1:length(sing.m)
                    m_matrix[s, i] = sing.m[i]
                    n_matrix[s, i] = sing.n[i]
                end
            end
            out_h5["singular/m"] = m_matrix
            out_h5["singular/n"] = n_matrix
        end

        # Write Δ' if computed (one complex value per resonant mode per singular surface)
        if intr.msing > 0 && all(s -> !isempty(s.delta_prime), intr.sing)
            max_modes = maximum(s -> length(s.delta_prime), intr.sing)
            dp_matrix = zeros(ComplexF64, intr.msing, max_modes)
            for (s, sing) in enumerate(intr.sing)
                for i in 1:length(sing.delta_prime)
                    dp_matrix[s, i] = sing.delta_prime[i]
                end
            end
            out_h5["singular/delta_prime"] = dp_matrix
        end

        # Write full off-diagonal Δ' column if computed (Riccati/parallel FM paths only).
        # Shape: [numpert_total × max_modes × msing], where delta_prime_col[:, i, s] is
        # the coupling of all N modes to resonant mode i at surface s.
        if intr.msing > 0 && all(s -> !isempty(s.delta_prime_col), intr.sing)
            N = size(intr.sing[1].delta_prime_col, 1)
            max_modes = maximum(s -> size(s.delta_prime_col, 2), intr.sing)
            dp_col_tensor = zeros(ComplexF64, N, max_modes, intr.msing)
            for (s, sing) in enumerate(intr.sing)
                n_res = size(sing.delta_prime_col, 2)
                dp_col_tensor[:, 1:n_res, s] = sing.delta_prime_col
            end
            out_h5["singular/delta_prime_col"] = dp_col_tensor
        end

        # Write inter-surface Δ' matrix if computed (parallel FM path only).
        # Shape: [msing × msing] — PEST3-convention deltap (STRIDE BVP with vacuum coupling).
        if intr.msing > 0 && !isempty(intr.delta_prime_matrix)
            out_h5["singular/delta_prime_matrix"] = intr.delta_prime_matrix
        end

        # Write vacuum data; always write all entries, using empty arrays when not computed
        out_h5["vacuum/wt"] = ctrl.vac_flag ? vac_data.wt : ComplexF64[]
        out_h5["vacuum/wt0"] = ctrl.vac_flag ? vac_data.wt0 : ComplexF64[]
        out_h5["vacuum/ep"] = ctrl.vac_flag ? vac_data.ep : ComplexF64[]
        out_h5["vacuum/ev"] = ctrl.vac_flag ? vac_data.ev : ComplexF64[]
        out_h5["vacuum/et"] = ctrl.vac_flag ? vac_data.et : ComplexF64[]
        out_h5["vacuum/vacuum_eigenvalue"] = ctrl.vac_flag ? vac_data.vacuum_eigenvalue : NaN
        out_h5["vacuum/x_plasma"] = ctrl.vac_flag ? vac_data.plasma_pts[:, 1] : Float64[]
        out_h5["vacuum/y_plasma"] = ctrl.vac_flag ? vac_data.plasma_pts[:, 2] : Float64[]
        out_h5["vacuum/z_plasma"] = ctrl.vac_flag ? vac_data.plasma_pts[:, 3] : Float64[]
        out_h5["vacuum/x_wall"] = ctrl.vac_flag ? vac_data.wall_pts[:, 1] : Float64[]
        out_h5["vacuum/y_wall"] = ctrl.vac_flag ? vac_data.wall_pts[:, 2] : Float64[]
        out_h5["vacuum/z_wall"] = ctrl.vac_flag ? vac_data.wall_pts[:, 3] : Float64[]

        # Write kinetic parameters when kinetic mode is enabled
        if ctrl.kinetic_factor > 0
            out_h5["kinetic/kinetic_source"] = ctrl.kinetic_source
            out_h5["kinetic/kinetic_factor"] = ctrl.kinetic_factor
        end

        # Write fundamental matrices on the ψ grid when mat_flag is enabled
        if ctrl.mat_flag && ffit !== nothing
            xs = equil.rzphi_xs
            npsi = length(xs)
            np = intr.numpert_total

            # Helper: evaluate a matrix spline on the psi grid → (npsi, np, np) array
            function _eval_mat_spline(spline)
                arr = zeros(ComplexF64, npsi, np, np)
                hint = Ref(1)
                for i in 1:npsi
                    arr[i, :, :] .= reshape(spline(xs[i]; hint=hint), np, np)
                end
                return arr
            end

            out_h5["matrices/psi"] = xs
            # Ideal primitive matrices (A, B, C, D, E, H)
            # When kinetic mode is on, amats/bmats/cmats hold kinetic-modified values,
            # so we write those as the "effective" matrices and save raw kinetic
            # components separately below.
            # Ideal primitive matrices (A, B, C, D, E, H)
            if ctrl.kinetic_factor > 0
                # Use preserved ideal copies (before kinetic overwrite)
                out_h5["matrices/ideal/A"] = _eval_mat_spline(ffit.amats_ideal)
                out_h5["matrices/ideal/B"] = _eval_mat_spline(ffit.bmats_ideal)
                out_h5["matrices/ideal/C"] = _eval_mat_spline(ffit.cmats_ideal)
            else
                out_h5["matrices/ideal/A"] = _eval_mat_spline(ffit.amats)
                out_h5["matrices/ideal/B"] = _eval_mat_spline(ffit.bmats)
                out_h5["matrices/ideal/C"] = _eval_mat_spline(ffit.cmats)
            end
            out_h5["matrices/ideal/D"] = _eval_mat_spline(ffit.dmats)
            out_h5["matrices/ideal/E"] = _eval_mat_spline(ffit.emats)
            out_h5["matrices/ideal/H"] = _eval_mat_spline(ffit.hmats)

            # Ideal derived matrices (F, K, G)
            out_h5["matrices/ideal/F"] = _eval_mat_spline(ffit.fmats_lower)
            out_h5["matrices/ideal/K"] = _eval_mat_spline(ffit.kmats)
            out_h5["matrices/ideal/G"] = _eval_mat_spline(ffit.gmats)

            # Kinetic-modified matrices
            if ctrl.kinetic_factor > 0
                out_h5["matrices/kinetic/A"] = _eval_mat_spline(ffit.amats)
                out_h5["matrices/kinetic/B"] = _eval_mat_spline(ffit.bmats)
                out_h5["matrices/kinetic/C"] = _eval_mat_spline(ffit.cmats)
                out_h5["matrices/kinetic/f0"] = _eval_mat_spline(ffit.f0mats)
                out_h5["matrices/kinetic/K"] = _eval_mat_spline(ffit.kkmats)
                out_h5["matrices/kinetic/G"] = _eval_mat_spline(ffit.gaats)
            end
        end
    end
end

export main

end # module GeneralizedPerturbedEquilibrium
