using Test
using Pkg

# Activate the project environment one level up
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()
using JPEC
using LinearAlgebra, Printf

# Check if specific test files are requested via ARGS
if !isempty(ARGS)
    for testfile in ARGS
        @info "Running test file: $testfile"
        include(testfile)
    end
else
    #=include("./runtests_build.jl")
    include("./runtests_spline.jl")
    #include("./runtests_vacuum_fortran.jl")
    include("./runtests_vacuum_julia.jl")
    include("./runtests_solovev.jl") =#
    include("./runtests_fourfit.jl")
    #=include("./runtests_ode.jl")
    include("./runtests_sing.jl")
    include("./runtests_fullruns.jl")=#
end
