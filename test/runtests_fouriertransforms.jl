# Unit tests for the high-level FourierTransform functor interface in
# src/Utilities/FourierTransforms.jl.

using GeneralizedPerturbedEquilibrium.Utilities: FourierTransform, inverse, compute_fourier_coefficients
using GeneralizedPerturbedEquilibrium.Utilities: empty_FourierCoefficients

const ATOL = 1e-10        # analytic-value comparisons
const ATOL_TIGHT = 1e-12  # in-place vs allocating / formula-vs-formula checks

@testset "FourierTransform" begin
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
end

@testset "compute_fourier_coefficients" begin
    atol = ATOL_TIGHT

    @testset "2D basis is exp(-i(mθ - nν))" begin
        N, mlow, mpert = 32, -3, 7
        m_modes = mlow:(mlow+mpert-1)
        n = 2
        ν = collect(range(; start=0.0, length=N, step=0.05))
        basis = compute_fourier_coefficients(N, m_modes, n, ν)
        @test size(basis) == (N, mpert)

        θ = collect(range(; start=0, length=N, step=2π/N))
        for (l, m) in enumerate(m_modes)
            expected = exp.(-im .* (m .* θ .- n .* ν))
            @test all(isapprox.(basis[:, l], expected; atol))
        end

        ft = FourierTransform(N, mpert, mlow)
        @test ft.basis == compute_fourier_coefficients(N, m_modes, 0, zeros(N))
    end

    @testset "3D basis shapes" begin
        mtheta, mpert, mlow = 8, 5, -2
        nzeta, npert, nlow = 6, 3, -1
        basis = compute_fourier_coefficients(mtheta, mlow:(mlow+mpert-1), nzeta, nlow:(nlow+npert-1))
        @test size(basis) == (mtheta * nzeta, mpert * npert)
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
