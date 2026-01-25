@testset "FastInterpolations-based Splines" begin

    @testset "BicubicSpline - Basic Evaluation" begin
        @info "Testing BicubicSpline basic evaluation"

        # Create 2D function: f(x,y) = sin(x) * cos(y)
        xs = collect(range(0.0; stop=2π, length=50))
        ys = collect(range(0.0; stop=2π, length=50))

        fs = zeros(Float64, length(xs), length(ys), 1)
        for (ix, x) in enumerate(xs), (iy, y) in enumerate(ys)
            fs[ix, iy, 1] = sin(x) * cos(y)
        end

        bcs = JPEC.Spl.BicubicSpline(xs, ys, fs, :extrap, :extrap)

        # Evaluate at interior point
        x_test, y_test = π/3, π/4
        f = JPEC.Spl.evaluate!(bcs, x_test, y_test)

        expected = sin(x_test) * cos(y_test)
        @test abs(f[1] - expected) < 1e-5
    end

    @testset "BicubicSpline - First Derivatives" begin
        @info "Testing BicubicSpline first derivatives"

        xs = collect(range(0.0; stop=2π, length=60))
        ys = collect(range(0.0; stop=2π, length=60))

        fs = zeros(Float64, length(xs), length(ys), 1)
        for (ix, x) in enumerate(xs), (iy, y) in enumerate(ys)
            fs[ix, iy, 1] = sin(x) * cos(y)
        end

        bcs = JPEC.Spl.BicubicSpline(xs, ys, fs, :extrap, :extrap)

        x_test, y_test = π/3, π/4
        f, fx, fy = JPEC.Spl.deriv1!(bcs, x_test, y_test)

        # Analytical derivatives
        expected_fx = cos(x_test) * cos(y_test)
        expected_fy = -sin(x_test) * sin(y_test)

        @test abs(fx[1] - expected_fx) < 1e-3
        @test abs(fy[1] - expected_fy) < 1e-3
    end

    @testset "BicubicSpline - Second Derivatives and Cross-Derivative" begin
        @info "Testing BicubicSpline second derivatives including fxy"

        xs = collect(range(0.0; stop=2π, length=80))
        ys = collect(range(0.0; stop=2π, length=80))

        fs = zeros(Float64, length(xs), length(ys), 1)
        for (ix, x) in enumerate(xs), (iy, y) in enumerate(ys)
            fs[ix, iy, 1] = sin(x) * cos(y)
        end

        bcs = JPEC.Spl.BicubicSpline(xs, ys, fs, :extrap, :extrap)

        x_test, y_test = π/3, π/4
        f, fx, fy, fxx, fxy, fyy = JPEC.Spl.deriv2!(bcs, x_test, y_test)

        # Analytical second derivatives
        expected_fxx = -sin(x_test) * cos(y_test)
        expected_fyy = -sin(x_test) * cos(y_test)
        expected_fxy = -cos(x_test) * sin(y_test)

        @test abs(fxx[1] - expected_fxx) < 1e-2
        @test abs(fyy[1] - expected_fyy) < 1e-2
        @test abs(fxy[1] - expected_fxy) < 1e-2
    end

    @testset "BicubicSpline - Empty Constructor" begin
        @info "Testing BicubicSpline empty constructor"

        empty_bcs = JPEC.Spl.empty_BicubicSpline()
        @test length(empty_bcs.xs) == 4  # Needs 4 points for extrap BC
        @test length(empty_bcs.ys) == 4
        @test empty_bcs.nqty == 1
    end

end

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

    fc = JPEC.Util.FourierCoefficients(xs, ys, fs, 4)

    # Check structure
    @test fc.mband == 4
    @test fc.nqty == 1
    @test length(fc.xs) == npsi

    # Mode 2 should have significant content at ipsi=10 (x=0.5)
    c2 = JPEC.Util.get_complex_coeff(fc, 10, 2, 1)
    @test abs(real(c2)) > 0.1  # Should have cosine content

    # Get all coefficients
    out = zeros(ComplexF64, 5)
    JPEC.Util.get_complex_coeffs!(out, fc, 10, 1)
    @test out[3] == c2  # Mode 2 is at index 3 (0-indexed mode)
end
