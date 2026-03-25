#!/usr/bin/env julia
#=
One full setup_equilibrium run; prints a single wall-time (seconds) to stdout
for use by thread-sweep drivers. Logging from the solver may go to stderr.

Usage:
    julia -t N --project=. benchmarks/equil_thread_worker.jl [path_to_example_dir]

Default example: examples/DIIID-like_ideal_example (relative to JPEC root).
Expects gpec.toml inside the example directory.

WSL: use Linux paths (e.g. /mnt/c/Users/.../JPEC or a clone under $HOME for faster I/O).
The example path is resolved with abspath relative to your current working directory if relative.
=#
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using TOML
using GeneralizedPerturbedEquilibrium
using GeneralizedPerturbedEquilibrium.Equilibrium

function main()
    base = abspath(get(ARGS, 1, joinpath(@__DIR__, "..", "examples", "DIIID-like_ideal_example")))
    config_path = joinpath(base, "gpec.toml")
    isfile(config_path) || error("Missing gpec.toml at $config_path (use a Linux path on WSL, e.g. /mnt/c/Users/... or ~/PlasmaLab/...)")
    raw = TOML.parsefile(config_path)
    cfg = Equilibrium.EquilibriumConfig(raw["Equilibrium"], dirname(config_path))

    setup_equilibrium(cfg)  # JIT / cache warmup
    t = @elapsed setup_equilibrium(cfg)
    println(t)
end

main()
