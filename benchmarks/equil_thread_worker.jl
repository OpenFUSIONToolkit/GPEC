#!/usr/bin/env julia
"""
One full `setup_equilibrium` run; prints a single wall-time (seconds) to stdout
for use by thread-sweep drivers. Logging from the solver may go to stderr.

Usage:
    julia -t N --project=. benchmarks/equil_thread_worker.jl [path_to_example_dir]

Default example: examples/DIIID-like_ideal_example (relative to JPEC root).
Expects gpec.toml inside the example directory.
"""
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using TOML
using GeneralizedPerturbedEquilibrium
using GeneralizedPerturbedEquilibrium.Equilibrium

function main()
    base = get(ARGS, 1, joinpath(@__DIR__, "..", "examples", "DIIID-like_ideal_example"))
    config_path = joinpath(base, "gpec.toml")
    raw = TOML.parsefile(config_path)
    cfg = EquilibriumConfig(raw["Equilibrium"], dirname(config_path))

    setup_equilibrium(cfg)  # JIT / cache warmup
    t = @elapsed setup_equilibrium(cfg)
    println(t)
end

main()
