@testset "FastInterpolations-based Splines" begin
    # NOTE: BicubicSpline wrapper has been removed in favor of native FastInterpolations API.
    # The codebase now uses FastInterpolations.cubic_interp((xs, ys), fs) directly.
    # These tests are commented out as BicubicSpline no longer exists.
    # See PlasmaEquilibrium struct for examples of native FastInterpolations usage.

    # @testset "BicubicSpline - Basic Evaluation" begin
    #     ... (tests removed - use native cubic_interp instead)
    # end

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

    fc = JPEC.Utilities.FourierCoefficients(xs, ys, fs, 4)

    # Check structure
    @test fc.mband == 4
    @test fc.nqty == 1
    @test length(fc.xs) == npsi

    # Mode 2 should have significant content at ipsi=10 (x=0.5)
    c2 = JPEC.Utilities.get_complex_coeff(fc, 10, 2, 1)
    @test abs(real(c2)) > 0.1  # Should have cosine content

    # Get all coefficients
    out = zeros(ComplexF64, 5)
    JPEC.Utilities.get_complex_coeffs!(out, fc, 10, 1)
    @test out[3] == c2  # Mode 2 is at index 3 (0-indexed mode)
end
