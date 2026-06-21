# Unit tests for the high-level FourierTransform functor interface in
# src/Utilities/FourierTransforms.jl. The low-level fourier_transform! /
# fourier_inverse_transform! matrix kernels are already covered in
# runtests_utilities.jl; this file targets the functor, inverse, in-place,
# and compute_fourier_coefficients paths.
#
# Expected mode amplitudes follow from grid orthogonality under the module's
# exp(-imθ), 1/N forward convention (Fortran iscdftf): on the uniform grid
# θ_i = 2π(i-1)/N, cos(m0θ) → 0.5 at ±m0, sin(m0θ) → ∓0.5im at ±m0.

using GeneralizedPerturbedEquilibrium.Utilities: FourierTransform, inverse, transform!, inverse_transform!, compute_fourier_coefficients
using GeneralizedPerturbedEquilibrium.Utilities: empty_FourierCoefficients

const ATOL = 1e-10        # analytic-value comparisons
const ATOL_TIGHT = 1e-12  # in-place vs allocating / formula-vs-formula checks

@testset "FourierTransform functor" begin
    atol = ATOL

    # Mode index l for mode number m (m = mlow + l - 1).
    midx(m, mlow) = m - mlow + 1

    @testset "single real mode → ±m amplitudes" begin
        N, mlow, mpert = 64, -10, 21  # modes -10..10
        ft = FourierTransform(N, mpert, mlow)
        θ = range(; start=0, length=N, step=2π/N)

        # cos(m0 θ): real 0.5 at m = ±m0, zero elsewhere; imag ≈ 0 everywhere.
        m0 = 3
        modes = ft(cos.(m0 .* θ))
        @test isapprox(modes[midx(m0, mlow)], 0.5 + 0im; atol)
        @test isapprox(modes[midx(-m0, mlow)], 0.5 + 0im; atol)
        others = [l for l in 1:mpert if l != midx(m0, mlow) && l != midx(-m0, mlow)]
        @test all(isapprox.(modes[others], 0; atol))

        # sin(m0 θ): -0.5im at +m0, +0.5im at -m0.
        modes = ft(sin.(m0 .* θ))
        @test isapprox(modes[midx(m0, mlow)], -0.5im; atol)
        @test isapprox(modes[midx(-m0, mlow)], 0.5im; atol)
        @test all(isapprox.(modes[others], 0; atol))
    end

    @testset "single complex mode → Kronecker delta" begin
        N, mlow, mpert = 64, -10, 21
        ft = FourierTransform(N, mpert, mlow)
        θ = collect(range(; start=0, length=N, step=2π/N))

        for m0 in (3, -2, 0)
            modes = ft(exp.(im .* m0 .* θ))
            @test isapprox(modes[midx(m0, mlow)], 1.0 + 0im; atol)
            others = [l for l in 1:mpert if l != midx(m0, mlow)]
            @test all(isapprox.(modes[others], 0; atol))
        end
    end

    @testset "round-trip recovery on a complete basis" begin
        # mpert == mtheta with mlow = -N/2 spans a complete DFT basis, so
        # inverse(ft, ft(data)) recovers arbitrary data exactly.
        N = 16
        ft = FourierTransform(N, N, -N ÷ 2)

        x_real = [Float64(2k - 7) + cospi(0.3k) for k in 1:N]  # deterministic, non-band-limited
        rec = inverse(ft, ft(x_real))
        @test all(isapprox.(real.(rec), x_real; atol))
        @test all(isapprox.(imag.(rec), 0; atol))

        x_complex = [Float64(k) - im * cospi(0.2k) for k in 1:N]
        rec = inverse(ft, ft(x_complex))
        @test all(isapprox.(rec, x_complex; atol))
    end

    @testset "matrix input transforms columns independently" begin
        N, mlow, mpert = 48, -8, 17  # modes -8..8
        ft = FourierTransform(N, mpert, mlow)
        θ = collect(range(; start=0, length=N, step=2π/N))

        data = hcat(cos.(3 .* θ), sin.(5 .* θ))
        modes = ft(data)
        @test size(modes) == (mpert, 2)
        @test isapprox(modes[midx(3, mlow), 1], 0.5 + 0im; atol)
        @test isapprox(modes[midx(-3, mlow), 1], 0.5 + 0im; atol)
        @test isapprox(modes[midx(5, mlow), 2], -0.5im; atol)
        @test isapprox(modes[midx(-5, mlow), 2], 0.5im; atol)
    end

    @testset "real-mode inverse is a pure cosine expansion" begin
        N, mlow, mpert = 32, -4, 9
        ft = FourierTransform(N, mpert, mlow)
        θ = collect(range(; start=0, length=N, step=2π/N))

        # A single real mode at m reconstructs (2π/N)·cos(mθ).
        modes = zeros(Float64, mpert)
        modes[midx(2, mlow)] = 1.0
        recon = inverse(ft, modes)
        @test all(isapprox.(recon, (2π / N) .* cos.(2 .* θ); atol))
    end
end

@testset "FourierTransform in-place == allocating" begin
    atol = ATOL_TIGHT
    N, mlow, mpert = 32, -6, 13
    ft = FourierTransform(N, mpert, mlow)
    θ = collect(range(; start=0, length=N, step=2π/N))

    @testset "transform! (vector)" begin
        data_r = cos.(2 .* θ) .+ 0.3 .* sin.(4 .* θ)
        out = zeros(ComplexF64, mpert)
        transform!(out, ft, data_r)
        @test all(isapprox.(out, ft(data_r); atol))

        data_c = data_r .+ im .* sin.(3 .* θ)
        transform!(out, ft, data_c)
        @test all(isapprox.(out, ft(data_c); atol))
    end

    @testset "transform! (matrix)" begin
        data = hcat(cos.(2 .* θ), sin.(3 .* θ), cos.(5 .* θ))
        out = zeros(ComplexF64, mpert, size(data, 2))
        transform!(out, ft, data)
        @test all(isapprox.(out, ft(data); atol))

        datac = ComplexF64.(data) .+ im
        transform!(out, ft, datac)
        @test all(isapprox.(out, ft(datac); atol))
    end

    @testset "inverse_transform!" begin
        modes_v = ComplexF64[Float64(l) - im * l for l in 1:mpert]
        out_v = zeros(ComplexF64, N)
        inverse_transform!(out_v, ft, modes_v)
        @test all(isapprox.(out_v, inverse(ft, modes_v); atol))

        modes_m = reshape(modes_v, mpert, 1) .* [1.0 -2.0im]
        out_m = zeros(ComplexF64, N, 2)
        inverse_transform!(out_m, ft, modes_m)
        @test all(isapprox.(out_m, inverse(ft, modes_m); atol))
    end
end

@testset "compute_fourier_coefficients" begin
    atol = ATOL_TIGHT

    @testset "2D basis is cos(mθ - nν)/sin(mθ - nν)" begin
        # Nonzero n and ν exercise the general phase-shifted form and lock both the
        # grid convention (start=0, step=2π/N) and the -n·ν sign.
        N, mlow, mpert = 32, -3, 7
        n = 2
        ν = collect(range(; start=0.0, length=N, step=0.05))
        cosb, sinb = compute_fourier_coefficients(N, mpert, mlow, 1, 1, 1; n_2D=n, ν=ν)
        @test size(cosb) == (N, mpert)
        @test size(sinb) == (N, mpert)

        θ = collect(range(; start=0, length=N, step=2π/N))
        for (l, m) in enumerate(mlow:(mlow+mpert-1))
            @test all(isapprox.(cosb[:, l], cos.(m .* θ .- n .* ν); atol))
            @test all(isapprox.(sinb[:, l], sin.(m .* θ .- n .* ν); atol))
        end

        # The FourierTransform constructor stores exactly the n=0, ν=0 basis.
        ft = FourierTransform(N, mpert, mlow)
        @test ft.cslth == compute_fourier_coefficients(N, mpert, mlow, 1, 1, 1; n_2D=0, ν=zeros(N))[1]
        @test ft.snlth == compute_fourier_coefficients(N, mpert, mlow, 1, 1, 1; n_2D=0, ν=zeros(N))[2]
    end

    @testset "3D basis shapes" begin
        mtheta, mpert, mlow = 8, 5, -2
        nzeta, npert, nlow = 6, 3, -1
        cosb, sinb = compute_fourier_coefficients(mtheta, mpert, mlow, nzeta, npert, nlow)
        @test size(cosb) == (mtheta * nzeta, mpert * npert)
        @test size(sinb) == (mtheta * nzeta, mpert * npert)
    end
end

@testset "empty_FourierCoefficients" begin
    fc = empty_FourierCoefficients()
    @test isempty(fc.xs)
    @test fc.mmax == 0
    @test fc.nqty == 0
    @test size(fc.cos_coeffs) == (0, 1, 0)
    @test size(fc.sin_coeffs) == (0, 1, 0)
end
