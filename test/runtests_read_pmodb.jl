using Test

# Load modules (paths resolved relative to repo root)
include(joinpath(@__DIR__, "..", "src", "pentrc", "dcon_interface.jl"))
include(joinpath(@__DIR__, "..", "src", "pentrc", "inputs.jl"))
using .Inputs

const IE = Inputs.InputsExtras

@testset "read_pmodb synthetic" begin
    # configure DCON stub to match nm in synthetic file
    DconInterface.mpert[] = 2
    DconInterface.mthsurf[] = 3
    DconInterface.theta = [0.0, π/2, π, 3π/2]
    DconInterface.mfac = [1,2]
    DconInterface.jac_type = "hamada"

    # Ensure Utilities.readtable is callable from Inputs (utilities.jl defines
    # many helpers at top-level; create wrappers if necessary)
    if isdefined(Main, :readtable) && !isdefined(Main, :Utilities)
        Core.eval(Main, :(Utilities = Main))
    elseif isdefined(Main, :readtable) && isdefined(Main, :Utilities)
        if !isdefined(Main.Utilities, :readtable)
            @eval Main.Utilities begin
                readtable(file; kwargs...) = Main.readtable(file; kwargs...)
            end
        end
    end
    # Also ensure common helpers exist on Main.Utilities (nunique, iscdftb/iscdftf)
    if isdefined(Main, :nunique) && isdefined(Main, :Utilities) && !isdefined(Main.Utilities, :nunique)
        @eval Main.Utilities begin
            nunique(x) = Main.nunique(collect(x))
        end
    end
    if isdefined(Main, :iscdftb) && isdefined(Main, :Utilities) && !isdefined(Main.Utilities, :iscdftb)
        @eval Main.Utilities begin
            iscdftb(m, funcm, fs) = Main.iscdftb(m, funcm, fs)
            iscdftf(m, func) = Main.iscdftf(m, func)
        end
    end

    filepath = joinpath(@__DIR__, "data", "pmodb_synth.txt")

    dbob, divx = IE.read_pmodb(filepath, "default", 1, 1, false)

    @test Inputs.dbob_m[] !== nothing
    @test Inputs.divx_m[] !== nothing
    using Main.SplinesMod
    @test isa(Inputs.dbob_m[], Main.SplinesMod.BicubicSpline)
    @test isa(Inputs.divx_m[], Main.SplinesMod.BicubicSpline)
end

println("runtests_read_pmodb.jl completed")
