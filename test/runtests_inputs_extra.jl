using Test

# Load the lightweight DCON stub and the Inputs port (use paths relative to this test file)
include(joinpath(@__DIR__, "..", "src", "pentrc", "dcon_interface.jl"))
include(joinpath(@__DIR__, "..", "src", "pentrc", "inputs.jl"))
using .Inputs

const IE = Inputs.InputsExtras

@testset "Inputs.extra tests" begin
    @testset "newm behavior" begin
        m_i = [1,2,3]
        fm_i = ComplexF64[1+1im, 2+0im, 3-1im]
        m_o = [1,2,3,4]
        out = Inputs.newm(m_i, fm_i, m_o)
        @test out[1:3] == fm_i
        @test out[4] == 0+0im

        # misaligned mode spaces should raise an error
        @test_throws ErrorException Inputs.newm([1,2,4], fm_i, [1,2,3])
    end

    @testset "set_peq synthetic" begin
        # configure DCON stub to expose vector theta and multiple m-points
        DconInterface.mpert = 4
        DconInterface.mthsurf = 3
        DconInterface.theta = [0.0, π/2, π, 3π/2]
        DconInterface.mfac = 1.0

        # Create small synthetic psi grid and m-pert spectra
        psi = [0.1, 0.5, 0.9]
        ms = [1,2,3,4]
        npsi = length(psi)
        mpert = DconInterface.mpert

        xmp1mns = Array{ComplexF64}(undef, npsi, mpert)
        xspmns  = Array{ComplexF64}(undef, npsi, mpert)
        xmsmns  = Array{ComplexF64}(undef, npsi, mpert)
        for i in 1:npsi
            for j in 1:mpert
                xmp1mns[i,j] = ComplexF64(sin(psi[i]) * j, cos(psi[i]) * j)
                xspmns[i,j]  = ComplexF64(cos(psi[i]) * j, sin(psi[i]) * j)
                xmsmns[i,j]  = ComplexF64((i+j)/2, (i-j)/3)
            end
        end

        # Ensure Utilities exposes the DFT helpers under Main.Utilities
        if isdefined(Main, :Utilities)
            if !isdefined(Main.Utilities, :iscdftb)
                @eval Main.Utilities begin
                    function iscdftb(m, funcm, fs)
                        return Main.iscdftb(m, funcm, fs)
                    end
                    function iscdftf(m, func)
                        return Main.iscdftf(m, func)
                    end
                end
            end
        end

        # Call set_peq (uses Utilities DFT helpers and Splines)
        IE.set_peq(psi, ms, xmp1mns, xspmns, xmsmns, true, false)

        # After set_peq, Inputs.xs_m entries should be populated with BicubicSpline
        @test Inputs.xs_m[1][] !== nothing
        using Main.SplinesMod
        @test isa(Inputs.xs_m[1][], Main.SplinesMod.BicubicSpline)
        @test isa(Inputs.xs_m[2][], Main.SplinesMod.BicubicSpline)
        @test isa(Inputs.xs_m[3][], Main.SplinesMod.BicubicSpline)
    end
end

println("runtests_inputs_extra.jl completed")
