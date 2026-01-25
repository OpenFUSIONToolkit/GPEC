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

@testset "BicubicSpline - PeriodicBC" begin
    # Test periodic boundary conditions for BicubicSpline
    # This validates the fix for closed grid handling in ReadEquilibrium.jl

    # Create a periodic test function: f(x, θ) where θ is periodic in [0, 2π]
    nx = 20
    ntheta = 32  # Number of poloidal points (open grid size)

    xs = collect(range(0.0; stop=1.0, length=nx))
    # Create CLOSED grid: ntheta+1 points from 0 to 2π (last point = first point position)
    ys = collect(range(0.0; stop=2π, length=ntheta + 1))

    # Periodic test functions: R(ψ, θ) and Z(ψ, θ) resembling tokamak geometry
    R(x, θ) = 1.0 + 0.3 * x + 0.1 * (1 - x) * cos(θ)
    Z(x, θ) = 0.15 * (1 - x) * sin(θ)

    # Allocate with closed grid - fill open grid portion first
    fs = zeros(nx, ntheta + 1, 2)
    for (ix, x) in enumerate(xs), iy in 1:ntheta
        θ = ys[iy]
        fs[ix, iy, 1] = R(x, θ)
        fs[ix, iy, 2] = Z(x, θ)
    end
    # Explicitly close the grid: copy first column to last (as done in ReadEquilibrium.jl)
    fs[:, end, :] .= fs[:, 1, :]

    # Verify closed grid property: data at θ=0 equals data at θ=2π
    @test all(fs[:, 1, :] .== fs[:, end, :])

    # Create BicubicSpline with PeriodicBC in y (poloidal) direction
    bcspline = JPEC.Spl.BicubicSpline(xs, ys, fs, :extrap, JPEC.Spl.PeriodicBC())

    # Test value periodicity: f(x, 0) ≈ f(x, 2π) for interior x values
    for x in [0.25, 0.5, 0.75]
        f_at_0 = JPEC.Spl.evaluate!(bcspline, x, 0.0)
        f_at_2π = JPEC.Spl.evaluate!(bcspline, x, 2π)
        @test f_at_0[1] ≈ f_at_2π[1] atol = 1e-10
        @test f_at_0[2] ≈ f_at_2π[2] atol = 1e-10
    end

    # Test derivative continuity across periodic boundary
    for x in [0.25, 0.5, 0.75]
        _, fx_0, fy_0 = JPEC.Spl.deriv1!(bcspline, x, 0.0)
        _, fx_2π, fy_2π = JPEC.Spl.deriv1!(bcspline, x, 2π)
        # Derivatives should match at periodic boundary
        @test fx_0[1] ≈ fx_2π[1] atol = 1e-8
        @test fy_0[1] ≈ fy_2π[1] atol = 1e-8
        @test fx_0[2] ≈ fx_2π[2] atol = 1e-8
        @test fy_0[2] ≈ fy_2π[2] atol = 1e-8
    end

    # Test interpolation accuracy at interior points
    x_test, θ_test = 0.5, π / 3
    f = JPEC.Spl.evaluate!(bcspline, x_test, θ_test)
    @test abs(f[1] - R(x_test, θ_test)) < 1e-4
    @test abs(f[2] - Z(x_test, θ_test)) < 1e-4
end

@testset "Exact Spline Integration" begin
    # Test that cumulative_integral uses exact spline integration formula
    # For periodic functions with PeriodicBC, the spline can match well

    # Test with periodic function (sin) and PeriodicBC - spline fits well
    xs_periodic = collect(range(0.0, 2π; length=65))  # Closed grid
    fs_periodic = sin.(xs_periodic)
    fs_periodic[end] = fs_periodic[1]  # Ensure closed

    fsi_periodic = JPEC.Spl.cumulative_integral(xs_periodic, fs_periodic; bc=JPEC.Spl.PeriodicBC())

    # ∫sin(x)dx = -cos(x), so cumulative from 0 should be -cos(x) + cos(0) = 1 - cos(x)
    exact_periodic = 1.0 .- cos.(xs_periodic)

    # Spline integration should be very accurate for smooth periodic functions
    @test maximum(abs.(fsi_periodic .- exact_periodic)) < 1e-6

    # Compare accuracy: spline vs trapezoidal for periodic case
    fsi_trap = zeros(length(xs_periodic))
    for i in 1:(length(xs_periodic)-1)
        h = xs_periodic[i+1] - xs_periodic[i]
        fsi_trap[i+1] = fsi_trap[i] + h * (fs_periodic[i] + fs_periodic[i+1]) / 2
    end
    trap_error = maximum(abs.(fsi_trap .- exact_periodic))
    spline_error = maximum(abs.(fsi_periodic .- exact_periodic))

    # Exact spline integration should be more accurate than trapezoidal
    @test spline_error < trap_error

    # Test integrate_spline function for total integral
    using FastInterpolations: cubic_interp
    itp = cubic_interp(xs_periodic, fs_periodic; bc=JPEC.Spl.PeriodicBC())
    total_integral = JPEC.Spl.integrate_spline(itp)

    # ∫₀^{2π} sin(x)dx = 0
    @test abs(total_integral) < 1e-10
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
