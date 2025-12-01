using Test

# Load Utilities first so Inputs can find Utilities.readtable
Core.eval(Main, :(include($(joinpath(@__DIR__, "..", "src", "pentrc", "utilities.jl")))))

# Load DCON stub and Inputs
include(joinpath(@__DIR__, "..", "src", "pentrc", "dcon_interface.jl"))
include(joinpath(@__DIR__, "..", "src", "pentrc", "inputs.jl"))
using .Inputs

# Minimal DCON setup
DconInterface.mpert[] = 4
DconInterface.mthsurf[] = 3
DconInterface.jac_type = "hamada"
DconInterface.mfac = 1.0
DconInterface.eqfun = nothing
DconInterface.theta = [0.0, pi/2, pi, 3pi/2]

file = joinpath(@__DIR__, "data", "pmodb_mouter.txt")
## Ensure Utilities module has wrappers for helpers that may be defined at
## top-level by the included utilities.jl script (e.g., readtable, nunique,
## iscdftb/iscdftf). This mirrors the shims used in other tests.
if isdefined(Main, :Utilities)
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
end

@testset "read_pmodb m-outer synthetic" begin
    dbob, divx = Inputs.InputsExtras.read_pmodb(file, "", 1, 1, false)
    @test Inputs.dbob_m[] !== nothing
    @test Inputs.divx_m[] !== nothing
    @test dbob !== nothing && divx !== nothing
end
