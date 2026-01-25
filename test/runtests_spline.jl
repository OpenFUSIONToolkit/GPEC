@testset "Test Spline Module" begin
    @info "Testing pure Julia spline implementations"

    @testset "MultiQuantityProfile - Polynomial" begin
        # Make x^3 spline - cubic splines can exactly represent cubics
        xs = collect(range(1.0; stop=2.0, length=21))
        fx3 = xs .^ 3
        fs = reshape(fx3, :, 1)  # Make it a matrix
        spline = JPEC.Spl.MultiQuantityProfile(xs, fs)

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

    @testset "MultiQuantityProfile - Sine" begin
        # Make sine spline
        xs = collect(range(Float64(pi) / 8; stop=3 * Float64(pi) / 8, length=50))
        fs = reshape(sin.(xs), :, 1)
        spline = JPEC.Spl.MultiQuantityProfile(xs, fs)

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

        bcspline = JPEC.Spl.BicubicSpline(xs, ys, fvals, :extrap, :extrap)

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
end

@testset "Empty Spline Constructors" begin
    @info "Testing empty spline constructors for type stability"

    # Test empty MultiQuantityProfile
    empty_mqp = JPEC.Spl.empty_MultiQuantityProfile()
    @test length(empty_mqp.xs) >= 4
    @test typeof(empty_mqp) <: JPEC.Spl.MultiQuantityProfile

    # Test empty BicubicSpline
    empty_bcs = JPEC.Spl.empty_BicubicSpline()
    @test length(empty_bcs.xs) >= 2
    @test length(empty_bcs.ys) >= 2
    @test typeof(empty_bcs) <: JPEC.Spl.BicubicSpline
end
