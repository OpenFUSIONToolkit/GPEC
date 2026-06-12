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

    @testset "FourierCoefficients spline quadrature (fft_flag=false)" begin
        @info "Testing alias-free spline-quadrature Fourier path (port of Fortran fspline_fit_1)"
        U = GeneralizedPerturbedEquilibrium.Utilities

        # Closed θ grid [0, 2π] with ntheta = N+1 points (last duplicates θ=0), as make_metric supplies.
        Nlo = 16
        ys = collect(range(0.0, 2π; length=Nlo + 1))

        # (a) Band-limited f = 1.7 + 0.4 sin(2θ) + cos(3θ): spline recovers c0=1.7, c2=-0.2i, c3=0.5.
        fb = reshape(1.7 .+ 0.4 .* sin.(2 .* ys) .+ cos.(3 .* ys), 1, Nlo + 1, 1)
        fcb = U.FourierCoefficients([0.0], ys, fb, 5; fft_flag=false)
        @test isapprox(real(U.get_complex_coeff(fcb, 1, 0, 1)), 1.7; atol=1e-3)
        @test isapprox(U.get_complex_coeff(fcb, 1, 2, 1), -0.2im; atol=2e-3)
        @test isapprox(U.get_complex_coeff(fcb, 1, 3, 1), 0.5 + 0im; atol=2e-3)

        # (b) Non-band-limited f = 1/(a - cosθ) has analytic c_m = r^m / sqrt(a^2-1),
        #     r = a - sqrt(a^2-1). The spline is alias-free; the bare FFT aliases badly at coarse N.
        a = 1.2
        r = a - sqrt(a^2 - 1); s = 1 / sqrt(a^2 - 1)
        fnb = reshape(1.0 ./ (a .- cos.(ys)), 1, Nlo + 1, 1)
        fc_spl = U.FourierCoefficients([0.0], ys, fnb, 6; fft_flag=false)
        fc_fft = U.FourierCoefficients([0.0], ys, fnb, 6; fft_flag=true)
        spline_err = maximum(abs(real(U.get_complex_coeff(fc_spl, 1, m, 1)) - s * r^m) for m in 0:6)
        fft_err = maximum(abs(real(U.get_complex_coeff(fc_fft, 1, m, 1)) - s * r^m) for m in 0:6)
        @test spline_err < 3e-3            # accurate even at N=16
        @test spline_err < fft_err / 20    # and dramatically better than the aliased FFT

        # (c) Convergence: refining the grid drives the spline error toward zero.
        ys2 = collect(range(0.0, 2π; length=64 + 1))
        fnb2 = reshape(1.0 ./ (a .- cos.(ys2)), 1, 65, 1)
        fc_spl2 = U.FourierCoefficients([0.0], ys2, fnb2, 6; fft_flag=false)
        spline_err2 = maximum(abs(real(U.get_complex_coeff(fc_spl2, 1, m, 1)) - s * r^m) for m in 0:6)
        @test spline_err2 < 1e-5
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
