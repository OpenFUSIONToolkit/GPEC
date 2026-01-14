@testset "Test Spline Module" begin
    @info "Testing pure Julia spline implementations"

    @testset "CubicSpline1D - Polynomial" begin
        # Make x^3 spline - cubic splines can exactly represent cubics
        xs = collect(range(1.0; stop=2.0, length=21))
        fx3 = xs .^ 3
        spline = JPEC.Spl.CubicSpline1D(xs, fx3; bctype="extrap")

        # Test value interpolation accuracy (primary use case)
        xs_fine = collect(range(1.1; stop=1.9, length=20))
        for x in xs_fine
            f = JPEC.Spl.evaluate!(spline, x)
            @test abs(f[1] - x^3) < 1e-4  # Value interpolation
        end

        # Test first derivative (commonly used)
        f1 = JPEC.Spl.deriv1!(spline, 1.5)
        f = JPEC.Spl.evaluate!(spline, 1.5)
        @test abs(f[1] - 1.5^3) < 1e-4
        @test abs(f1[1] - 3 * 1.5^2) < 0.1
    end

    @testset "CubicSpline1D - Sine" begin
        # Make sine spline
        xs = collect(range(Float64(pi) / 8; stop=3 * Float64(pi) / 8, length=50))
        fs = sin.(xs)
        spline = JPEC.Spl.CubicSpline1D(xs, fs; bctype="extrap")

        # Test value interpolation
        xs_fine = collect(range(Float64(pi) / 6; stop=Float64(pi) / 3, length=20))
        for x in xs_fine
            f = JPEC.Spl.evaluate!(spline, x)
            @test abs(f[1] - sin(x)) < 1e-6
        end

        # Test first derivative
        f1 = JPEC.Spl.deriv1!(spline, Float64(pi) / 4)
        f = JPEC.Spl.evaluate!(spline, Float64(pi) / 4)
        @test abs(f[1] - sin(Float64(pi) / 4)) < 1e-6
        @test abs(f1[1] - cos(Float64(pi) / 4)) < 1e-4
    end

    @testset "CubicSpline1D - Complex" begin
        # Make e^-ix and e^ix complex valued spline
        xs = collect(range(0.0; stop=6.0, length=100))
        fm = exp.(-im .* xs)
        fp = exp.(im .* xs)
        fs_matrix = hcat(fm, fp)
        spline = JPEC.Spl.CubicSpline1D(xs, fs_matrix; bctype="extrap")

        # Test value interpolation
        xs_fine = collect(range(2.0; stop=2.5, length=10))
        for x in xs_fine
            f = JPEC.Spl.evaluate!(spline, x)
            @test abs(f[1] - exp(-im * x)) < 1e-6
            @test abs(f[2] - exp(im * x)) < 1e-6
        end

        # Test first derivative
        f1 = JPEC.Spl.deriv1!(spline, 2.25)
        @test abs(f1[1] + im * exp(-im * 2.25)) < 1e-4
        @test abs(f1[2] - im * exp(im * 2.25)) < 1e-4
    end

    @testset "CubicSpline1D - Periodic" begin
        # Test periodic boundary conditions
        xs = collect(range(0.0; stop=2 * Float64(pi), length=101))
        fs = sin.(xs)
        spline = JPEC.Spl.CubicSpline1D(xs, fs; bctype="periodic")

        # Check continuity at boundary
        f_start = JPEC.Spl.evaluate!(spline, 0.0)
        f_end = JPEC.Spl.evaluate!(spline, 2 * Float64(pi))
        @test abs(f_start[1] - f_end[1]) < 1e-10

        # Check derivative continuity
        f1_start = JPEC.Spl.deriv1!(spline, 0.0)
        f1_end = JPEC.Spl.deriv1!(spline, 2 * Float64(pi))
        @test abs(f1_start[1] - f1_end[1]) < 1e-10
    end

    @testset "BicubicSpline" begin
        # Test bicubic spline setup and evaluation for a 2D function
        xs = collect(range(0.0; stop=2 * Float64(pi), length=50))
        ys = collect(range(0.0; stop=2 * Float64(pi), length=50))

        f1(x, y) = sin(x) * cos(y) + 1
        f2(x, y) = cos(x) * sin(y) + 1
        fvals = Array{Float64}(undef, length(xs), length(ys), 2)
        for (ix, x) in enumerate(xs), (iy, y) in enumerate(ys)
            fvals[ix, iy, 1] = f1(x, y)
            fvals[ix, iy, 2] = f2(x, y)
        end

        bcspline = JPEC.Spl.BicubicSpline(xs, ys, fvals; bctypex="extrap", bctypey="extrap")

        # Test evaluation at interior points
        x_test, y_test = Float64(pi) / 2, Float64(pi) / 4
        f = JPEC.Spl.evaluate!(bcspline, x_test, y_test)
        @test abs(f[1] - f1(x_test, y_test)) < 1e-4
        @test abs(f[2] - f2(x_test, y_test)) < 1e-4

        # Test first derivatives
        f, fx, fy = JPEC.Spl.deriv1!(bcspline, x_test, y_test)
        @test abs(fx[1] - cos(x_test) * cos(y_test)) < 1e-3
        @test abs(fy[1] + sin(x_test) * sin(y_test)) < 1e-3
    end

    @testset "FourierModeSplines" begin
        # Test Fourier mode splines for periodic theta data
        xs = collect(range(0.0; stop=1.0, length=21))
        ys = collect(range(0.0; stop=2 * Float64(pi), length=65))  # endpoint-inclusive
        mband = 8

        # Create test data: f(x, y) = x * cos(y) + x^2 * sin(2y)
        fvals = Array{Float64}(undef, length(xs), length(ys), 1)
        for (ix, x) in enumerate(xs), (iy, y) in enumerate(ys)
            fvals[ix, iy, 1] = x * cos(y) + x^2 * sin(2 * y)
        end

        fspline = JPEC.Spl.FourierModeSplines(xs, ys, fvals, mband)

        # Test evaluation (Fourier decomposition has inherent truncation error)
        x_test, y_test = 0.5, Float64(pi) / 3
        f = JPEC.Spl.evaluate!(fspline, x_test, y_test)
        expected = x_test * cos(y_test) + x_test^2 * sin(2 * y_test)
        @test abs(f[1] - expected) < 0.05  # Looser tolerance for Fourier truncation
    end
end

@testset "Empty Spline Constructors" begin
    @info "Testing empty spline constructors for type stability"

    # Test Float64 empty CubicSpline1D
    empty_cs_f64 = JPEC.Spl.empty_CubicSpline1D(Float64)
    @test length(empty_cs_f64.xs) >= 2  # Minimal placeholder
    @test typeof(empty_cs_f64) <: JPEC.Spl.CubicSpline1D

    # Test ComplexF64 empty CubicSpline1D
    empty_cs_c64 = JPEC.Spl.empty_CubicSpline1D(ComplexF64)
    @test typeof(empty_cs_c64) <: JPEC.Spl.CubicSpline1D

    # Test empty BicubicSpline
    empty_bcs = JPEC.Spl.empty_BicubicSpline()
    @test length(empty_bcs.xs) >= 2
    @test length(empty_bcs.ys) >= 2
    @test typeof(empty_bcs) <: JPEC.Spl.BicubicSpline

    # Test empty FourierModeSplines
    empty_fms = JPEC.Spl.empty_FourierModeSplines()
    @test typeof(empty_fms) <: JPEC.Spl.FourierModeSplines

    # Test empty EqfunSplines
    empty_eqfun = JPEC.Spl.empty_EqfunSplines()
    @test typeof(empty_eqfun) == JPEC.Spl.EqfunSplines
end
