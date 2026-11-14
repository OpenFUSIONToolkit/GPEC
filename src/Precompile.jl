# Precompile workload: run small cases through each stage at build time so a fresh process skips most
# first-call compilation. A Solovev case covers the Riccati, forward and Galerkin formalisms; a TJ-analytic
# case with kinetic profiles and coil forcing covers the perturbed equilibrium and the kinetic torque.
# Disable with the PrecompileTools preference `precompile_workload = false` (see docs/src/set_up.md).
using PrecompileTools: @setup_workload, @compile_workload
import Logging

@setup_workload begin
    # Run one deck silently in a scratch directory; `files` are written beside it first.
    function run_workload(inputs; files=())
        mktempdir() do workdir
            foreach(((name, text),) -> write(joinpath(workdir, name), text), files)
            open(io -> TOML.print(io, inputs), joinpath(workdir, "gpec.toml"), "w")
            Logging.with_logger(Logging.NullLogger()) do
                redirect_stdio(; stdout=devnull) do
                    return main([workdir])
                end
            end
        end
    end

    # examples/Solovev_ideal_example, coarsened in mpsi and mtheta_vacuum to keep the build short.
    stability = Dict{String,Any}(
        "Equilibrium" => Dict{String,Any}(
            "jac_type" => "pest", "grid_type" => "core_edge_packed", "psilow" => 1e-4, "psihigh" => 0.9995,
            "mpsi" => 16, "mtheta" => 256, "newq0" => 0, "flux_surface_rtol" => 1e-7
        ),
        "Wall" => Dict{String,Any}("shape" => "conformal", "standoff_fraction" => 0.2415, "equal_arc_wall" => true),
        "ForceFreeStates" => Dict{String,Any}(
            "vac_flag" => true, "dW_edge_scan_start" => 0.99, "qlow" => 1.02, "qhigh" => 1e3, "nn" => 1,
            "mtheta_vacuum" => 64, "singfac_min" => 1e-4
        ),
        "Solovev" => Dict{String,Any}("mr" => 128, "mz" => 128, "ma" => 128, "elongation" => 1.6, "a" => 0.33, "r0" => 1.0, "q0" => 1.9)
    )

    # examples/TJ_kinetic_NTV_example, coarsened, at a DIII-D-like size so the DIII-D C-coil forcing applies, and on the
    # direct solver, so the inverse-solver path stays compiled on first use as before.
    kinetic_table = join(("$(x) 1.2e19 1.2e19 $(20 + 180 * (1 - x)) $(20 + 180 * (1 - x)) $(2e4 * (1 - x))" for x in range(0, 1; length=12)), "\n")
    response = Dict{String,Any}(
        "Equilibrium" => Dict{String,Any}(
            "kinetic_file" => "kinetic.dat", "jac_type" => "hamada", "grid_type" => "core_edge_packed",
            "psilow" => 0.01, "psihigh" => 0.99, "mpsi" => 16, "mtheta" => 64
        ),
        "TJAnalytic" => Dict{String,Any}(
            "r0" => 1.7, "a" => 0.17, "q0" => 1.6, "qa" => 3.2, "p0" => 0.1, "pressure_peaking" => 2.0, "B0" => 1.0,
            "ma" => 32, "mtheta" => 64, "solver" => "direct"
        ),
        "ForceFreeStates" => Dict{String,Any}(
            "vac_flag" => true, "qlow" => 1.02, "qhigh" => 1e3, "nn" => 1, "delta_mlow" => 4, "delta_mhigh" => 4,
            "mtheta_vacuum" => 64, "singfac_min" => 1e-4, "integrator" => "Forward", "truncate_at_rational_offset" => false
        ),
        "Forward" => Dict{String,Any}("eulerlagrange_tolerance" => 1e-6),
        "ForcingTerms" => Dict{String,Any}(
            "forcing_data_format" => "coil", "machine" => "d3d", "mtheta_coil" => 32, "nzeta_coil" => 8,
            "coil_set" => [Dict{String,Any}("name" => "c", "currents" => [1000.0, 500.0, -500.0, -1000.0, -500.0, 500.0])]
        ),
        "PerturbedEquilibrium" => Dict{String,Any}("compute_singular_coupling" => true, "regularization_width" => 0.05),
        "KineticForces" => Dict{String,Any}(
            "bounce_harmonic_max" => 1, "electron" => false, "nutype" => "harmonic", "f0type" => "maxwellian",
            "atol_pitch" => 1e-4, "axis_validity_suppression" => true
        )
    )

    @compile_workload begin
        for formalism in ("Riccati", "Forward", "Galerkin")
            stability["ForceFreeStates"]["integrator"] = formalism
            foreach(name -> delete!(stability, name), ("Riccati", "Forward", "Galerkin"))
            stability[formalism] =
                formalism == "Galerkin" ? Dict{String,Any}("nx" => 32, "rpec_flag" => true) :
                Dict{String,Any}("renorm_threshold" => 1e3, "eulerlagrange_tolerance" => 1e-7)
            run_workload(stability)
        end
        run_workload(response; files=("kinetic.dat" => kinetic_table,))
    end
    # Start each session with empty caches, as a JIT build does, instead of the workload's run state.
    Vacuum.reset_caches!()
    KineticForces.reset_caches!()
end
