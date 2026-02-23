using Test
using Pkg
using JPEC.Vacuum
using JPEC.Equilibrium
using JPEC.ForceFreeStates
using JPEC.ForcingTerms
using JPEC.PerturbedEquilibrium
using JPEC.Utilities
using FastInterpolations
using LinearAlgebra

# Activate the project environment one level up
Pkg.activate(joinpath(@__DIR__, ".."))
using JPEC

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
    include("./runtests_sing.jl")
    include("./runtests_fullruns.jl")
end
