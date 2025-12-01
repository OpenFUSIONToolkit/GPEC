using Test

# Load Utilities first so it defines a `Main.Utilities` module with
# readtable/nunique/iscdft* functions available for Inputs to call.
Core.eval(Main, :(include($(joinpath(@__DIR__, "..", "src", "pentrc", "utilities.jl")))))

# Load DCON stub and Inputs module (use paths relative to this test file)
include(joinpath(@__DIR__, "..", "src", "pentrc", "dcon_interface.jl"))
include(joinpath(@__DIR__, "..", "src", "pentrc", "inputs.jl"))
using .Inputs

# Configure a minimal DCON environment so read_peq can run
DconInterface.mpert[] = 4
DconInterface.mthsurf[] = 3
DconInterface.jac_type = "hamada"
DconInterface.mfac = 1.0
DconInterface.eqfun = nothing
DconInterface.theta = [0.0, pi/2, pi, 3pi/2]

file = joinpath(@__DIR__, "data", "peq_synth.txt")

# Ensure Main.Utilities namespace provides wrappers to functions included
# by the repository's utilities.jl (which may inject helpers at top-level).
if !isdefined(Main, :Utilities)
    Core.eval(Main, quote
        module Utilities
        end
    end)
end
if isdefined(Main, :readtable)
    Core.eval(Main.Utilities, :(function readtable(args...; kwargs...)
            return Main.readtable(args...; kwargs...)
        end))
end
if isdefined(Main, :nunique)
    Core.eval(Main.Utilities, :(function nunique(x)
            return Main.nunique(x)
        end))
end
if isdefined(Main, :iscdftb)
    Core.eval(Main.Utilities, :(function iscdftb(m, arr, nth)
            return Main.iscdftb(m, arr, nth)
        end))
end
if isdefined(Main, :iscdftf)
    Core.eval(Main.Utilities, :(function iscdftf(m, arr)
            return Main.iscdftf(m, arr)
        end))
end

@testset "read_peq synthetic" begin
    # call the reader; should populate Inputs.xs_m[1..3]
    Inputs.InputsExtras.read_peq(file, "", 1, 1, false, false)

    @test Inputs.xs_m[1][] !== nothing
    @test Inputs.xs_m[2][] !== nothing
    @test Inputs.xs_m[3][] !== nothing
end
