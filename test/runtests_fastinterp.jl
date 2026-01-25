@testset "FastInterpolations-based Splines" begin

    @testset "FastCubicSpline1D - Basic Evaluation" begin
        @info "Testing FastCubicSpline1D basic evaluation"

        # Test with sin(x) which is smoother and works well with extrap BC
        xs = collect(range(0.0; stop=2π, length=51))
        fs = sin.(xs)
        spline = JPEC.Spl.FastCubicSpline1D(xs, fs; bc=JPEC.Spl.extrap_bc(xs, fs))

        # Evaluate within the grid range
        xs_fine = collect(range(0.2; stop=6.0, length=50))
        for x in xs_fine
            f = JPEC.Spl.evaluate!(spline, x)
            @test abs(f - sin(x)) < 2e-5
        end
    end

    @testset "FastCubicSpline1D - First Derivative" begin
        @info "Testing FastCubicSpline1D first derivative"

        # Test with sin(x): derivative is cos(x)
        xs = collect(range(0.0; stop=2π, length=51))
        fs = sin.(xs)
        spline = JPEC.Spl.FastCubicSpline1D(xs, fs; bc=JPEC.Spl.extrap_bc(xs, fs))

        xs_fine = collect(range(0.3; stop=5.8, length=30))
        for x in xs_fine
            f = JPEC.Spl.evaluate!(spline, x)
            f1 = JPEC.Spl.deriv1!(spline, x)
            @test abs(f - sin(x)) < 1e-5
            @test abs(f1 - cos(x)) < 1e-3
        end
    end

    @testset "FastCubicSpline1D - Second Derivative" begin
        @info "Testing FastCubicSpline1D second derivative"

        # Test with sin(x): second derivative is -sin(x)
        xs = collect(range(0.0; stop=2π, length=51))
        fs = sin.(xs)
        spline = JPEC.Spl.FastCubicSpline1D(xs, fs; bc=JPEC.Spl.extrap_bc(xs, fs))

        xs_fine = collect(range(0.5; stop=5.5, length=30))
        for x in xs_fine
            f = JPEC.Spl.evaluate!(spline, x)
            f1 = JPEC.Spl.deriv1!(spline, x)
            f2 = JPEC.Spl.deriv2!(spline, x)
            @test abs(f - sin(x)) < 1e-5
            @test abs(f1 - cos(x)) < 1e-3
            @test abs(f2 + sin(x)) < 1e-2
        end
    end

    @testset "FastCubicSpline1D - Third Derivative" begin
        @info "Testing FastCubicSpline1D third derivative"

        # Test with sin(x): third derivative is -cos(x)
        xs = collect(range(0.0; stop=2π, length=51))
        fs = sin.(xs)
        spline = JPEC.Spl.FastCubicSpline1D(xs, fs; bc=JPEC.Spl.extrap_bc(xs, fs))

        xs_fine = collect(range(0.5; stop=5.5, length=30))
        for x in xs_fine
            f = JPEC.Spl.evaluate!(spline, x)
            f1 = JPEC.Spl.deriv1!(spline, x)
            f2 = JPEC.Spl.deriv2!(spline, x)
            f3 = JPEC.Spl.deriv3!(spline, x)
            @test abs(f - sin(x)) < 1e-5
            @test abs(f1 - cos(x)) < 1e-3
            @test abs(f2 + sin(x)) < 1e-2
            # Third derivative is piecewise constant, less accurate
            @test abs(f3 + cos(x)) < 0.5
        end
    end

    @testset "FastCubicSpline1D - Polynomial Test" begin
        @info "Testing FastCubicSpline1D with x^3 polynomial"

        xs = collect(range(1.0; stop=2.0, length=51))
        fs = xs .^ 3
        spline = JPEC.Spl.FastCubicSpline1D(xs, fs; bc=JPEC.Spl.extrap_bc(xs, fs))

        # Interior points should be more accurate
        xs_fine = collect(range(1.2; stop=1.8, length=30))
        for x in xs_fine
            f = JPEC.Spl.evaluate!(spline, x)
            f1 = JPEC.Spl.deriv1!(spline, x)
            f2 = JPEC.Spl.deriv2!(spline, x)
            f3 = JPEC.Spl.deriv3!(spline, x)
            @test abs(f - x^3) < 1e-4
            @test abs(f1 - 3*x^2) < 1e-3
            @test abs(f2 - 6*x) < 1e-2
            # Third derivative should be close to 6
            @test abs(f3 - 6.0) < 0.5
        end
    end

    @testset "FastCubicSpline1D - Integration" begin
        @info "Testing FastCubicSpline1D cumulative integration"

        # Test with x^2: integral from 0 to x is x^3/3
        xs = collect(range(0.0; stop=2.0, length=101))
        fs = xs .^ 2
        spline = JPEC.Spl.FastCubicSpline1D(xs, fs; bc=JPEC.Spl.extrap_bc(xs, fs))
        fsi = JPEC.Spl.integrate!(spline)

        # Check cumulative integral at grid points
        for (i, x) in enumerate(xs)
            expected_integral = x^3 / 3
            @test abs(fsi[i] - expected_integral) < 2e-4  # Trapezoidal integration has some error
        end
    end

    @testset "FastCubicSpline1DMulti - Multiple Quantities" begin
        @info "Testing FastCubicSpline1DMulti with multiple quantities"

        xs = collect(range(0.0; stop=2π, length=50))
        fs = hcat(sin.(xs), cos.(xs), xs .^ 2)
        spline = JPEC.Spl.FastCubicSpline1DMulti(xs, fs; bc=:extrap)

        x_test = π/4
        f = JPEC.Spl.evaluate!(spline, x_test)
        f1 = JPEC.Spl.deriv1!(spline, x_test)

        @test abs(f[1] - sin(x_test)) < 1e-5
        @test abs(f[2] - cos(x_test)) < 1e-5
        @test abs(f[3] - x_test^2) < 1e-5

        @test abs(f1[1] - cos(x_test)) < 1e-3
        @test abs(f1[2] + sin(x_test)) < 1e-3
        @test abs(f1[3] - 2*x_test) < 1e-3
    end

    @testset "FastCubicSpline1D - Empty Constructor" begin
        @info "Testing FastCubicSpline1D empty constructor"

        empty_spline = JPEC.Spl.empty_FastCubicSpline1D(Float64)
        @test length(empty_spline.xs) >= 4

        empty_spline_complex = JPEC.Spl.empty_FastCubicSpline1D(ComplexF64)
        @test eltype(empty_spline_complex.fs) == ComplexF64
    end

    @testset "FastCubicSpline1DMulti - Empty Constructor" begin
        @info "Testing FastCubicSpline1DMulti empty constructor"

        empty_spline = JPEC.Spl.empty_FastCubicSpline1DMulti(Float64)
        @test length(empty_spline.xs) >= 4
    end

    @testset "ComplexMatrixSpline - Evaluation" begin
        @info "Testing ComplexMatrixSpline basic evaluation"

        # Create complex-valued matrix data
        npsi = 20
        n1, n2 = 3, 3
        xs = collect(range(0.0; stop=1.0, length=npsi))

        # Create data: each element is exp(i*psi * (i+j))
        data = zeros(ComplexF64, npsi, n1, n2)
        for ipsi in 1:npsi, i in 1:n1, j in 1:n2
            data[ipsi, i, j] = exp(1im * xs[ipsi] * (i + j))
        end

        cms = JPEC.Spl.ComplexMatrixSpline(xs, data; bc=:extrap)

        # Evaluate at interior point
        x_test = 0.5
        result = JPEC.Spl.evaluate!(cms, x_test)

        # Check that result has correct shape
        @test size(result) == (n1, n2)

        # Check values (should be close to analytic)
        for i in 1:n1, j in 1:n2
            expected = exp(1im * x_test * (i + j))
            @test abs(result[i, j] - expected) < 1e-4
        end
    end

    @testset "ComplexMatrixSpline - Derivatives" begin
        @info "Testing ComplexMatrixSpline derivatives"

        npsi = 30
        n1, n2 = 2, 2
        xs = collect(range(0.0; stop=1.0, length=npsi))

        # Linear function: f(psi) = (1 + 2im) * psi
        slope = 1.0 + 2.0im
        data = zeros(ComplexF64, npsi, n1, n2)
        for ipsi in 1:npsi
            data[ipsi, :, :] .= slope * xs[ipsi]
        end

        cms = JPEC.Spl.ComplexMatrixSpline(xs, data; bc=:extrap)

        x_test = 0.5
        f = JPEC.Spl.evaluate!(cms, x_test)
        f1 = JPEC.Spl.deriv1!(cms, x_test)

        # First derivative of linear function is the slope
        for i in 1:n1, j in 1:n2
            @test abs(f[i, j] - slope * x_test) < 1e-8
            @test abs(f1[i, j] - slope) < 1e-5
        end

        # Test third derivative (should be zero for linear)
        f3 = JPEC.Spl.deriv3!(cms, x_test)
        for i in 1:n1, j in 1:n2
            @test abs(f3[i, j]) < 1e-5
        end
    end

    @testset "ComplexMatrixSpline - Empty Constructor" begin
        @info "Testing ComplexMatrixSpline empty constructor"

        empty_cms = JPEC.Spl.empty_ComplexMatrixSpline(2, 3)
        @test empty_cms.n1 == 2
        @test empty_cms.n2 == 3
        @test size(empty_cms._out) == (2, 3)
    end

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

    @testset "FourierModeSplines - Basic Evaluation" begin
        @info "Testing FourierModeSplines basic evaluation"

        # Create 2D function with known Fourier content: cos(2*y)
        npsi, ntheta = 20, 64
        xs = collect(range(0.0; stop=1.0, length=npsi))
        ys = collect(range(0.0; stop=1.0, length=ntheta+1)[1:(end-1)])  # Periodic domain

        fs = zeros(Float64, npsi, ntheta, 1)
        for (ix, x) in enumerate(xs), (iy, y) in enumerate(ys)
            # f(psi, theta) = psi * cos(2 * 2π * theta)
            fs[ix, iy, 1] = x * cos(2 * 2π * y)
        end

        fms = JPEC.Spl.FourierModeSplines(xs, ys, fs, 4)

        # Evaluate at interior point
        x_test, y_test = 0.5, 0.25
        f = JPEC.Spl.evaluate!(fms, x_test, y_test)

        expected = x_test * cos(2 * 2π * y_test)
        @test abs(f[1] - expected) < 1e-2  # FFT + spline has some error
    end

    @testset "FourierModeSplines - First Derivatives" begin
        @info "Testing FourierModeSplines first derivatives"

        # Simple function: f(psi, theta) = psi
        npsi, ntheta = 30, 32
        xs = collect(range(0.0; stop=1.0, length=npsi))
        ys = collect(range(0.0; stop=1.0, length=ntheta+1)[1:(end-1)])

        fs = zeros(Float64, npsi, ntheta, 1)
        for (ix, x) in enumerate(xs), (iy, y) in enumerate(ys)
            fs[ix, iy, 1] = x  # Constant in theta
        end

        fms = JPEC.Spl.FourierModeSplines(xs, ys, fs, 4)

        x_test, y_test = 0.5, 0.3
        f, fx, fy = JPEC.Spl.deriv1!(fms, x_test, y_test)

        # Derivative in x should be 1, in y should be 0
        @test abs(f[1] - x_test) < 1e-4
        @test abs(fx[1] - 1.0) < 1e-3
        @test abs(fy[1]) < 1e-6
    end

    @testset "FourierModeSplines - Empty Constructor" begin
        @info "Testing FourierModeSplines empty constructor"

        empty_fms = JPEC.Spl.empty_FourierModeSplines()
        @test length(empty_fms.xs) == 5  # Requires 5 points for FastCubicSpline1D (4 minimum)
        @test length(empty_fms.ys) == 2
        @test empty_fms.mband == 1
        @test empty_fms.nqty == 1
    end

end
