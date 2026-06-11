@testset "Utilities Unit Tests" begin
    @testset "FourierCoefficients" begin
        @info "Testing FourierCoefficients from Utilities module"

        # Create 2D function with known Fourier content: cos(2*y)
        npsi, ntheta = 20, 64
        xs = collect(range(0.0; stop=1.0, length=npsi))
        ys = collect(range(0.0; stop=1.0, length=ntheta+1)[1:(end-1)])  # Periodic domain

        fs = zeros(Float64, npsi, ntheta, 1)
        for (ix, x) in enumerate(xs), (iy, y) in enumerate(ys)
            # f(psi, theta) = psi * cos(2 * 2π * theta)
            fs[ix, iy, 1] = x * cos(2 * 2π * y)
        end

        fc = GeneralizedPerturbedEquilibrium.Utilities.FourierCoefficients(xs, ys, fs, 4)

        # Check structure
        @test fc.mband == 4
        @test fc.nqty == 1
        @test length(fc.xs) == npsi

        # Mode 2 should have significant content at ipsi=10 (x=0.5)
        c2 = GeneralizedPerturbedEquilibrium.Utilities.get_complex_coeff(fc, 10, 2, 1)
        @test abs(real(c2)) > 0.1  # Should have cosine content

        # Get all coefficients
        out = zeros(ComplexF64, 5)
        GeneralizedPerturbedEquilibrium.Utilities.get_complex_coeffs!(out, fc, 10, 1)
        @test out[3] == c2  # Mode 2 is at index 3 (0-indexed mode)
    end

    @testset "FourierTransforms" begin
        @info "Testing fourier_transform! and fourier_inverse_transform! from Utilities module"
        using GeneralizedPerturbedEquilibrium.Utilities: fourier_transform!, fourier_inverse_transform!

        @testset "fourier_transform!" begin
            mtheta, mpert = 4, 2
            gil = zeros(mtheta, mpert)
            gij = [1.0 2.0 3.0 4.0; 5.0 6.0 7.0 8.0; 9.0 10.0 11.0 12.0; 13.0 14.0 15.0 16.0]
            cs = zeros(mtheta, mpert)
            cs[:, 1] .= 1.0  # First mode: all ones → output is column sum of gij

            fourier_transform!(gil, gij, cs; row_offset=0, col_offset=0)

            # gil[:, 1] = gij * cs[:, 1] = sum over rows of gij per column → columns of gij summed
            # Column 1 of gij: [1,5,9,13] → sum 28; column 2: [2,6,10,14] → 32; etc. No, mul! does (gil block) = gij * cs.
            # So gil[i, l] = sum_j gij[i,j] * cs[j,l]. For cs[:,1]=1: gil[:,1] = gij * [1,1,1,1]' = row sums of gij.
            # Rows of gij: [1,2,3,4] sum 10, [5,6,7,8] sum 26, [9,10,11,12] sum 42, [13,14,15,16] sum 58.
            @test gil[:, 1] == [10.0, 26.0, 42.0, 58.0]
            @test gil[:, 2] == [0.0, 0.0, 0.0, 0.0]

            # Test with column offset: fill second block
            gil2 = zeros(mtheta, 4)
            cs2 = zeros(mtheta, 2)
            cs2[:, 1] .= 1.0
            fourier_transform!(gil2, gij, cs2; col_offset=2)
            @test gil2[:, 3] == [10.0, 26.0, 42.0, 58.0]
            @test all(gil2[:, 1:2] .== 0) && all(gil2[:, 4] .== 0)
        end

        @testset "fourier_inverse_transform!" begin
            mtheta, mpert = 4, 2
            gil = zeros(mtheta, mpert)
            gil[:, 1] = [1.0, 2.0, 3.0, 4.0]
            cs = zeros(mtheta, mpert)
            cs[:, 1] .= 1.0
            gll = zeros(mpert, mpert)

            fourier_inverse_transform!(gll, gil, cs)

            # gll = (1/mtheta) * cs' * gil  →  gll[1,1] = (1/4) * sum_i cs[i,1]*gil[i,1] = 10/4
            expected = 10.0 / mtheta
            @test isapprox(gll[1, 1], expected)
            @test isapprox(gll[2, 1], 0.0)
        end
    end
end
