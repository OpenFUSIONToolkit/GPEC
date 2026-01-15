@testset "Test Vacuum Fortran" begin
    try
        mtheta_in, lmin, lmax, nnin = Int32(4), Int32(1), Int32(4), Int32(2)
        qa1in = 1.23
        xin = rand(Float64, lmax - lmin + 1)
        zin = rand(Float64, lmax - lmin + 1)
        deltain = rand(Float64, lmax - lmin + 1)

        @info "Testing set_dcon_params…"
        JPEC.Vacuum.set_dcon_params(mtheta_in, lmin, lmax, nnin, qa1in, xin, zin, deltain)
        @info "set_dcon_params OK!"

        @info "Testing mscvac…"
        mpert = 5
        mtheta = 256
        mtheta_vacuum = 256
        wv = zeros(ComplexF64, mpert, mpert)
        complex_flag = true
        kernelsignin = -1.0
        wall_flag = false
        farwall_flag = true
        grrio = rand(Float64, 2 * (mtheta_vacuum + 5), mpert * 2)
        xzptso = rand(Float64, mtheta_vacuum + 5, 4)
        op_ahgfile = "aaaa"

        # print wall_flag value
        @info "wall_flag value: $wall_flag"
        JPEC.Vacuum.mscvac(
            wv, mpert, mtheta, mtheta_vacuum,
            complex_flag, kernelsignin,
            wall_flag, farwall_flag,
            grrio, xzptso, op_ahgfile, joinpath(@__DIR__, ".")
        )
        @info "mscvac OK!"
    catch e
        @test false
        @error "mscvac failed: $e"
    end
end
