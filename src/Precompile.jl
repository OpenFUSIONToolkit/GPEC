# Precompile workload: run a small Solovev case through the Riccati and forward formalisms at
# build time so a fresh process skips most first-call compilation. Disable with the
# PrecompileTools preference `precompile_workload = false` (see docs/src/set_up.md).
using PrecompileTools: @setup_workload, @compile_workload
import Logging

@setup_workload begin
    # examples/Solovev_ideal_example, coarsened in mpsi and mtheta_vacuum to keep the build short.
    inputs = Dict{String,Any}(
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
    @compile_workload begin
        for formalism in ("Riccati", "Forward")
            inputs["ForceFreeStates"]["integrator"] = formalism
            foreach(name -> delete!(inputs, name), ("Riccati", "Forward"))
            inputs[formalism] = Dict{String,Any}("renorm_threshold" => 1e3, "eulerlagrange_tolerance" => 1e-7)
            mktempdir() do workdir
                open(io -> TOML.print(io, inputs), joinpath(workdir, "gpec.toml"), "w")
                Logging.with_logger(Logging.NullLogger()) do
                    redirect_stdio(; stdout=devnull) do
                        return main([workdir])
                    end
                end
            end
        end
    end
    # Start each session with empty caches, as a JIT build does, instead of the workload's run state.
    Vacuum.reset_caches!()
    KineticForces.reset_caches!()
end
