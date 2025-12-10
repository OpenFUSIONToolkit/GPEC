

using Test
using Pkg

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
    include("./runtests_build.jl")
    include("./runtests_spline.jl")
    include("./runtests_vacuum_fortran.jl")
    include("./runtests_solovev.jl")
    include("./runtests_ode.jl")
    include("./runtests_sing.jl")
end