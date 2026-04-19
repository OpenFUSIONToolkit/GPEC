using Test
using Pkg
using GeneralizedPerturbedEquilibrium.Vacuum
using GeneralizedPerturbedEquilibrium.Equilibrium
using GeneralizedPerturbedEquilibrium.ForceFreeStates
using GeneralizedPerturbedEquilibrium.ForcingTerms
using GeneralizedPerturbedEquilibrium.PerturbedEquilibrium
using GeneralizedPerturbedEquilibrium.Utilities
using FastInterpolations
using LinearAlgebra

# Activate the project environment one level up
Pkg.activate(joinpath(@__DIR__, ".."))
using GeneralizedPerturbedEquilibrium

# Check if specific test files are requested via ARGS
if !isempty(ARGS)
    for testfile in ARGS
        @info "Running test file: $testfile"
        include(testfile)
    end
else
    include("./runtests_utilities.jl")
    include("./runtests_vacuum.jl")
    include("./runtests_equil.jl")
    include("./runtests_eulerlagrange.jl")
    include("./runtests_riccati.jl")
    include("./runtests_parallel_integration.jl")
    include("./runtests_sing.jl")
    include("./runtests_tj_analytic.jl")
    include("./runtests_kinetic_profiles.jl")
    include("./runtests_slayer_params.jl")
    include("./runtests_slayer_riccati.jl")
    include("./runtests_slayer_inputs.jl")
    include("./runtests_dispersion_residual.jl")
    include("./runtests_dispersion_coupled.jl")
    include("./runtests_dispersion_scan.jl")
    include("./runtests_dispersion_amr.jl")
    include("./runtests_fullruns.jl")
end
