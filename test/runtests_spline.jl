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

    # NOTE: BicubicSpline has been removed - use native FastInterpolations.cubic_interp instead
    # @testset "BicubicSpline" begin
    #     ... (tests removed)
    # end
end

# NOTE: BicubicSpline has been removed - use native FastInterpolations.cubic_interp instead
# @testset "BicubicSpline - PeriodicBC" begin
#     ... (tests removed)
# end

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
