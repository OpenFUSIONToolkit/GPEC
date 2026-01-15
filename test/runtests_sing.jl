@testset "Test Sing Functions" begin
    @testset " Test sing_der " begin
        # du = zeros(ComplexF64, intr.mpert, odet.msol, 2)
        # Initialize u with some value, e.g., ones
        # odet.u .= ones(ComplexF64, size(odet.u))
        # Initialize structs with relevant data, can do something like
        # ffit = FourfitVars()
        # amat = load in from test_data folder
        # ffit.amats = SplinesMod.CubicSpline("some psi", reshape(amats, mpsi+1, :); bctype="extrap")
        # if using one psi values causes issues, can make a psi array and just fill it all with amat
        # repeat for other matrices, equil data, other relevant constants, etc.

        # params = (ctrl, equil, intr, odet, ffit)
        # sing_der!(du, odet.u, params, odet.psifac)

        # load in du from fortran
        # @test du == expected_du (to within some error)
    end

    @testset " Test sing_find " begin # continue with other functions
    end
end
