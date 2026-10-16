using Random
using Statistics

# The tolerance samplers: closed-form radial distributions at a few quantiles, the exact limits
# of the cylinder axis-line model, reproducibility under a seed, the injectable shared direction
# coherent groups rely on, and the sign convention of the cylinder tilt against apply_transforms.
@testset "tolerance sampling" begin
    GPEC = GeneralizedPerturbedEquilibrium
    EF = GPEC.ErrorFields
    FT = GPEC.ForcingTerms

    @testset "radial distributions" begin
        @test EF.randpow(EF.Flat()) == 1.0
        @test EF.randpow(EF.UniformArea()) == 0.5
        @test EF.randpow(EF.Hollow()) ≈ 1 / 3
        @test EF.randpow(EF.Ring()) == 0.0
        @test EF.randpow(EF.PowerLaw(0.25)) == 0.25
        @test_throws ArgumentError EF.PowerLaw(-1)
        for name in EF.RADIAL_SHAPES
            @test EF.radial_distribution(name) isa EF.RadialDistribution
        end
        @test EF.radial_distribution("Hollow") isa EF.Hollow
        @test_throws ArgumentError EF.radial_distribution("gaussian")

        # Radial CDF (r/R)^(1/p) at three quantiles, N draws, tolerance a few standard errors.
        N = 200_000
        R = 2.0
        tol = 4 / sqrt(N)
        for (d, cdf) in ((EF.Flat(), x -> x), (EF.UniformArea(), x -> x^2), (EF.Hollow(), x -> x^3), (EF.PowerLaw(0.25), x -> x^4))
            rng = Xoshiro(7)
            r = [EF.disk_radius(rng, R, EF.randpow(d)) for _ in 1:N]
            @test all(0 .<= r .<= R)
            for x in (0.3, 0.6, 0.9)
                @test abs(count(<=(x * R), r) / N - cdf(x)) < tol
            end
        end
        rng = Xoshiro(7)
        @test all(EF.disk_radius(rng, R, EF.randpow(EF.Ring())) == R for _ in 1:1000)

        # Disk points: the direction is uniform and independent of the radius.
        rng = Xoshiro(3)
        pts = [EF.sample_disk(rng, R, EF.UniformArea()) for _ in 1:N]
        @test all(abs.(pts) .<= R)
        @test abs(mean(pts)) < 3 * R / sqrt(N)
        @test abs(mean(real.(pts) .^ 2) - R^2 / 4) < 6 * R^2 / sqrt(N)   # ⟨x²⟩ = R²/4 uniform over a disk
        @test abs(count(p -> angle(p) > 0, pts) / N - 0.5) < tol
    end

    @testset "uncertainty" begin
        rng = Xoshiro(11)
        @test EF.sample_uncertainty(rng, 0.0) == 0
        u = [EF.sample_uncertainty(rng, 0.5) for _ in 1:100_000]
        @test abs(mean(abs2, u) - 0.25) < 0.01            # E|σ·n·e^{iφ}|² = σ²
        @test abs(mean(abs, u) - 0.5 * sqrt(2 / π)) < 0.01 # half-normal magnitude, not Rayleigh
        @test EF.sample_uncertainty(Xoshiro(1), 0.5; phase=0.0) |> imag == 0
    end

    @testset "seed reproducibility and shared directions" begin
        a = [EF.sample_additive(Xoshiro(5), 1e-3, 1.0, 0.1, 1 / 3) for _ in 1:3]
        @test all(==(a[1]), a)
        c1 = EF.sample_cylinder(Xoshiro(9), 1e-3, 1.5, 1.0)
        c2 = EF.sample_cylinder(Xoshiro(9), 1e-3, 1.5, 1.0)
        @test c1 == c2
        # Handing the sampler the phase it would have drawn reproduces the default path, and a
        # shared phase gives every member the same direction with its own radius.
        rng = Xoshiro(21)
        φ = 2π * rand(rng)
        s_default = EF.sample_disk(Xoshiro(21), 1.0, 1.0)
        s_given = EF.sample_disk(rng, 1.0, 1.0; phase=φ)
        @test s_default == s_given
        rng = Xoshiro(22)
        members = [EF.sample_disk(rng, 1.0, EF.Hollow(); phase=1.2) for _ in 1:5]
        @test all(m -> angle(m) ≈ 1.2, members)
        @test length(unique(abs.(members))) == 5
    end

    @testset "cylinder model limits" begin
        R, z_top = 1e-3, 1.5
        rng = Xoshiro(4)
        draws = [EF.sample_cylinder(rng, R, z_top, 1.0) for _ in 1:50_000]
        shifts = first.(draws)
        tilts = last.(draws)
        @test all(abs.(shifts) .<= R)
        @test all(abs.(tilts) .<= rad2deg(atan(R / z_top)) + 1e-12)
        # The midplane point is the mean of two independent disk draws: for Flat, ⟨|Δ|²⟩ is half
        # a single draw's R²/3.
        @test abs(mean(abs2, shifts) - R^2 / 6) < 5 * (R^2 / 6) / sqrt(length(shifts))
        # A tall cylinder leaves the tilt at zero; a zero radius leaves everything at zero.
        _, t_tall = EF.sample_cylinder(Xoshiro(4), R, 1e9, 1.0)
        @test abs(t_tall) < 1e-10
        s0, t0 = EF.sample_cylinder(Xoshiro(4), 0.0, z_top, 1.0)
        @test s0 == 0 && t0 == 0
        # Swapping the endpoints flips the lean and keeps the midplane point.
        s_a, t_a = EF.sample_cylinder(Xoshiro(8), R, z_top, 1.0; phase_top=0.3, phase_bot=2.0)
        rng_b = Xoshiro(8)
        r_top = EF.disk_radius(rng_b, R, 1.0)
        r_bot = EF.disk_radius(rng_b, R, 1.0)
        p_top, p_bot = r_top * cis(0.3), r_bot * cis(2.0)
        @test s_a ≈ (p_top + p_bot) / 2
        @test abs(t_a) ≈ rad2deg(atan(abs(p_top - p_bot), 2z_top))
        d = p_top - p_bot
        @test t_a ≈ -abs(t_a) * complex(sin(angle(d)), cos(angle(d)))
        @test_throws ArgumentError EF.sample_cylinder(Xoshiro(1), R, 0.0, 1.0)
    end

    @testset "cylinder tilt sign matches apply_transforms" begin
        # An axis leaning its top toward +x means the coil plane rises on the −x side. Tilt a
        # horizontal hoop by the sampler's (θx, θy) for an endpoint separation along +x and
        # check where it rises.
        hoop = FT.make_pf_hoop(; radius=1.0, height=0.0, name="hoop")
        R, z_top = 0.01, 0.5
        _, tilt = EF.sample_cylinder(Xoshiro(1), R, z_top, 0.0; phase_top=0.0, phase_bot=float(π))  # p_top = +R, p_bot = −R
        @test real(tilt) ≈ 0 atol = 1e-12
        @test imag(tilt) ≈ -rad2deg(atan(2R, 2z_top))
        tilted = FT.apply_transforms(hoop, FT.CoilSetConfig(; tiltx=[real(tilt)], tilty=[imag(tilt)]); n_tilt=1)
        i_minus_x = argmin(vec(tilted.x))
        i_plus_x = argmax(vec(tilted.x))
        @test tilted.z[i_minus_x] > 0 > tilted.z[i_plus_x]
        # And the analogous lean toward +y rises the −y side.
        _, tilt_y = EF.sample_cylinder(Xoshiro(1), R, z_top, 0.0; phase_top=float(π / 2), phase_bot=float(-π / 2))
        tilted_y = FT.apply_transforms(hoop, FT.CoilSetConfig(; tiltx=[real(tilt_y)], tilty=[imag(tilt_y)]); n_tilt=1)
        @test tilted_y.z[argmin(vec(tilted_y.y))] > 0 > tilted_y.z[argmax(vec(tilted_y.y))]
    end
end
