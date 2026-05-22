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
    include("./runtests_sing.jl")
    include("./runtests_fullruns.jl")
    include("./runtests_coils.jl")
    include("./runtests_imas.jl")
end
