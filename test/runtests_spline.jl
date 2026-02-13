# NOTE: Splines module has been removed - all spline functionality now uses FastInterpolations directly
# Legacy tests for MultiQuantityProfile and BicubicSpline have been removed

@testset "Exact Spline Integration" begin
    # Test native FastInterpolations integration functions
    using FastInterpolations: cubic_interp, cumulative_integrate, integrate, PeriodicBC

    # Test with periodic function (sin) and PeriodicBC
    xs_periodic = collect(range(0.0, 2π; length=65))  # Closed grid
    fs_periodic = sin.(xs_periodic)
    fs_periodic[end] = fs_periodic[1]  # Ensure closed

    itp_periodic = cubic_interp(xs_periodic, fs_periodic; bc=PeriodicBC())
    fsi_periodic = cumulative_integrate(itp_periodic)

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

    # Test integrate function for total integral
    total_integral = integrate(itp_periodic)

    # ∫₀^{2π} sin(x)dx = 0
    @test abs(total_integral) < 1e-10
end

# NOTE: Empty spline constructor tests removed with Splines module
