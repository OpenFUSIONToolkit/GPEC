using Test

# Advanced smoke tests for Inputs: exercise read_pmodb, read_peq, set_peq
include(joinpath(@__DIR__, "..", "src", "pentrc", "dcon_interface.jl"))
include(joinpath(@__DIR__, "..", "src", "pentrc", "inputs.jl"))
using .Inputs

# Configure DconInterface stub for test
DconInterface.mpert = 4
DconInterface.mfac = 1.0
DconInterface.mthsurf = 3
DconInterface.theta = [0.0, 1.0, 2.0, 3.0]
# Provide a simple eqfun-like object with a field `f` so normalization will use it
DconInterface.eqfun = (f = [1.0])
DconInterface.jac_type = "hamada"
DconInterface.verbose = false
DconInterface.sq = (xs = [0.0, 1.0])

# If Utilities module in Main does not expose readtable (many utilities
# functions exist at top-level in the file), alias Main as Utilities so
# calls like Utilities.readtable resolve to the top-level functions.
if isdefined(Main, :readtable) && (!isdefined(Main, :Utilities) || !isdefined(Main.Utilities, :readtable))
    # Forward readtable into Main.Utilities so calls like Utilities.readtable
    # resolve to the top-level function defined by the utilities file.
    @eval Main.Utilities begin
        function readtable(file::String; verbose::Bool=false, debug::Bool=false)
            return Main.readtable(file; verbose=verbose, debug=debug)
        end
    end
end

# Forward other utilities functions if they are defined at top-level
if isdefined(Main, :nunique) && (!isdefined(Main, :Utilities) || !isdefined(Main.Utilities, :nunique))
    @eval Main.Utilities begin
        function nunique(array; op_sorted::Bool=false)
            return Main.nunique(Vector(array); op_sorted=op_sorted)
        end
    end
end
if isdefined(Main, :iscdftb) && (!isdefined(Main, :Utilities) || !isdefined(Main.Utilities, :iscdftb))
    @eval Main.Utilities begin
        function iscdftb(m::Vector{Int}, funcm::Vector{ComplexF64}, fs::Int)
            return Main.iscdftb(m, funcm, fs)
        end
    end
end
if isdefined(Main, :iscdftf) && (!isdefined(Main, :Utilities) || !isdefined(Main.Utilities, :iscdftf))
    @eval Main.Utilities begin
        function iscdftf(m::Vector{Int}, func::Vector{ComplexF64})
            return Main.iscdftf(m, func)
        end
    end
end
if isdefined(Main, :get_free_file_unit) && (!isdefined(Main, :Utilities) || !isdefined(Main.Utilities, :get_free_file_unit))
    @eval Main.Utilities begin
        function get_free_file_unit(lu_max::Int=-1)
            return Main.get_free_file_unit(lu_max)
        end
    end
end

@testset "Inputs pmodb/peq smoke" begin
    # run read_pmodb on synthetic table
    dbob, divx = Inputs.InputsExtras.read_pmodb(joinpath(@__DIR__, "data", "pmodb_test.txt"), "default", 1, 1, false)
    @test Inputs.dbob_m[] !== nothing
    @test Inputs.divx_m[] !== nothing

    # run read_peq on synthetic table
    Inputs.InputsExtras.read_peq(joinpath(@__DIR__, "data", "peq_test.txt"), "default", 1, 1, false, false)
    @test Inputs.xs_m[1][] !== nothing
    @test Inputs.xs_m[2][] !== nothing
    @test Inputs.xs_m[3][] !== nothing
end
