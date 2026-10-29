module GPECTests

# Test orchestration needs little optimization; imported package kernels retain their defaults.
Base.Experimental.@optlevel 0

using Test
using Pkg
using GeneralizedPerturbedEquilibrium.Vacuum
using GeneralizedPerturbedEquilibrium.Equilibrium
using GeneralizedPerturbedEquilibrium.ForceFreeStates
using GeneralizedPerturbedEquilibrium.LocalStability
using GeneralizedPerturbedEquilibrium.ForcingTerms
using GeneralizedPerturbedEquilibrium.PerturbedEquilibrium
using GeneralizedPerturbedEquilibrium.Utilities
using FastInterpolations
using LinearAlgebra

# Activate the project environment one level up
Pkg.activate(joinpath(@__DIR__, ".."))
using GeneralizedPerturbedEquilibrium

function run_test_file(testfile)
    @info "Running test file" testfile
    measurement = @timed include(testfile)
    @info "Completed test file" testfile elapsed_seconds=measurement.time compile_seconds=measurement.compile_time recompile_seconds=measurement.recompile_time
    return nothing
end

include("suites.jl")

# Fail if a new test file is omitted or accidentally assigned to multiple groups.
registered = first.(TEST_SUITES)
discovered = filter(f -> startswith(f, "runtests_") && endswith(f, ".jl"), readdir(@__DIR__))
length(unique(registered)) == length(registered) || error("Duplicate test file in suites.jl")
Set(registered) == Set(discovered) || error("Update test/suites.jl to cover every runtests_*.jl file")

group = get(ENV, "GPEC_TEST_GROUP", "")
isempty(group) || group in last.(TEST_SUITES) || error("Unknown GPEC_TEST_GROUP: $group")
selected = isempty(ARGS) ? [file for (file, suite) in TEST_SUITES if isempty(group) || suite == group] : ARGS
for testfile in selected
    run_test_file(testfile)
end

end
