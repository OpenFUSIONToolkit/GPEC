@testset "Utilities Unit Tests" begin
    @testset "adaptive_sample resolves a narrow resonance and brackets a zero crossing" begin
        adaptive_sample = GeneralizedPerturbedEquilibrium.Utilities.adaptive_sample
        # A Lorentzian narrower than the initial spacing, plus a linear component crossing zero.
        calls = Ref(0)
        f(x) = (calls[] += 1; [1 / (1 + ((x - 0.37) / 0.03)^2), x - 0.2])
        x, y = adaptive_sample(f, range(-1, 1; length=9); max_points=41, rtol=0.02)
        @test issorted(x) && allunique(x) && size(y) == (length(x), 2) && calls[] == length(x)
        @test 0.0 in x                                                        # the initial grid survives
        @test length(x) <= 41
        # The peak is resolved: the sampled maximum is within 5 % of the true peak value 1.
        @test maximum(y[:, 1]) > 0.95
        # The zero crossing of the second component is bracketed by neighbours closer than the initial spacing.
        i = findfirst(k -> y[k, 2] * y[k+1, 2] <= 0, 1:(length(x)-1))
        @test i !== nothing && x[i+1] - x[i] < 0.25
        # The same peak riding on a background ten times its height, as one component: the tails are
        # below rtol of the range, so only the scale-free feature trigger can find it.
        g(x) = [1 / (1 + ((x - 0.37) / 0.03)^2) + 5 * (x - 0.2)]
        xg, yg = adaptive_sample(g, range(-1, 1; length=9); max_points=41, rtol=0.02)
        @test maximum(yg[:, 1] .- 5 .* (xg .- 0.2)) > 0.9
        # A smooth function on a fine grid stops immediately.
        x2, y2 = adaptive_sample(x -> [x^2], range(0, 1; length=11); max_points=41, rtol=0.05)
        @test length(x2) == 11
        # Determinism and the point cap.
        x3, _ = adaptive_sample(f, range(-1, 1; length=9); max_points=15, rtol=0.0)
        x4, _ = adaptive_sample(f, range(-1, 1; length=9); max_points=15, rtol=0.0)
        @test x3 == x4 && length(x3) == 15
        @test_throws ArgumentError adaptive_sample(f, [0.0]; max_points=5)
        @test_throws ArgumentError adaptive_sample(f, [0.0, 0.0, 1.0]; max_points=5)
    end

    @testset "shift_exb_rotation moves only the E×B profile" begin
        Eq = GeneralizedPerturbedEquilibrium.Equilibrium
        xs = collect(range(0.0, 1.0; length=21))
        kp = Eq.KineticProfileSplines(xs, fill(1e19, 21), fill(1e19, 21), 1e3 .* (1 .- xs) .* 1.602e-19 .+ 1e-17, 1e3 .* (1 .- xs) .* 1.602e-19 .+ 1e-17,
            1e4 .* (1 .- xs .^ 2), fill(17.0, 21), fill(1e3, 21), fill(1e4, 21), fill(1.5, 21))
        kp2 = Eq.shift_exb_rotation(kp, -2.5e4)
        for ψ in (0.0, 0.31, 0.77, 1.0)
            @test kp2.omegaE_spline(ψ) ≈ kp.omegaE_spline(ψ) - 2.5e4
            @test kp2.Ti_spline(ψ) ≈ kp.Ti_spline(ψ) && kp2.ni_spline(ψ) ≈ kp.ni_spline(ψ) && kp2.nui_spline(ψ) ≈ kp.nui_spline(ψ)
            @test kp2.Ti_deriv(ψ) ≈ kp.Ti_deriv(ψ)
        end
        @test kp2.xs == kp.xs
    end
    @testset "FourierCoefficients" begin
        @info "Testing FourierCoefficients from Utilities module"

        # Create 2D function with known Fourier content: cos(2*y)
        npsi, ntheta = 20, 64
        xs = collect(range(0.0; stop=1.0, length=npsi))
        ys = collect(range(0.0; stop=1.0, length=ntheta + 1)[1:(end-1)])  # Periodic domain

        fs = zeros(Float64, npsi, ntheta, 1)
        for (ix, x) in enumerate(xs), (iy, y) in enumerate(ys)
            # f(psi, theta) = psi * cos(2 * 2π * theta)
            fs[ix, iy, 1] = x * cos(2 * 2π * y)
        end

        fc = GeneralizedPerturbedEquilibrium.Utilities.FourierCoefficients(xs, ys, fs, 4)

        # Check structure
        @test fc.mmax == 4
        @test fc.nqty == 1
        @test length(fc.xs) == npsi

        # Mode 2 should have significant content at ipsi=10 (x=0.5)
        c2 = GeneralizedPerturbedEquilibrium.Utilities.get_complex_coeff(fc, 10, 2, 1)
        @test abs(real(c2)) > 0.1  # Should have cosine content

        # Get all coefficients
        out = zeros(ComplexF64, 5)
        GeneralizedPerturbedEquilibrium.Utilities.get_complex_coeffs!(out, fc, 10, 1)
        @test out[3] == c2  # Mode 2 is at index 3 (0-indexed mode)
    end
end
