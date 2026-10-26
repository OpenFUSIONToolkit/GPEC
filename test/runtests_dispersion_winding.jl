@testset "Dispersion winding-number root classification" begin
    using GeneralizedPerturbedEquilibrium.Dispersion
    D = GeneralizedPerturbedEquilibrium.Dispersion

    # ----------------------------------------------------------------
    # _winding_number: the argument principle on synthetic functions
    # ----------------------------------------------------------------
    @testset "_winding_number: zeros, poles, both orientations" begin
        z = 0.3 - 0.2im
        @test D._winding_number(Q -> Q - z, z, 0.1)[1] ≈ 1 atol = 1e-10
        @test D._winding_number(Q -> 1 / (Q - z), z, 0.1)[1] ≈ -1 atol = 1e-10
        # a function of conj(Q): its zero winds -1, its pole +1
        @test D._winding_number(Q -> conj(Q - z), z, 0.1)[1] ≈ -1 atol = 1e-10
        @test D._winding_number(Q -> 1 / conj(Q - z), z, 0.1)[1] ≈ 1 atol = 1e-10
        # nothing inside the circle
        @test D._winding_number(Q -> Q - z, z + 1.0, 0.1)[1] ≈ 0 atol = 1e-10
        # two zeros inside, and a zero-pole pair inside
        @test D._winding_number(Q -> (Q - z) * (Q - z - 0.02), z, 0.1)[1] ≈ 2 atol = 1e-10
        @test D._winding_number(Q -> (Q - z) / (Q - z - 0.02), z, 0.1)[1] ≈ 0 atol = 1e-10
    end

    @testset "_winding_number: refines the sampling near the circle" begin
        # zero just inside the circle: the phase turns fast where the circle
        # passes it, so the default 64 samples are not enough
        z = 0.0 + 0.0im
        W, step = D._winding_number(Q -> Q - z, z + 0.0998, 0.1)
        @test W ≈ 1 atol = 1e-10
        @test step < π / 2
        # zero exactly on the circle → unresolved
        W, _ = D._winding_number(Q -> Q - z, z + 0.1, 0.1)
        @test isnan(W)
        # non-finite residual → unresolved
        W, _ = D._winding_number(Q -> ComplexF64(NaN, NaN), z, 0.1)
        @test isnan(W)
    end

    # ----------------------------------------------------------------
    # find_growth_rates(...; classification=:winding)
    # ----------------------------------------------------------------
    # Two zeros and a pole between them, off the scan nodes; the upper zero lies
    # ABOVE the pole.
    z_lo, z_hi, p = -0.4137 - 0.3021im, 0.3519 + 0.4473im, 0.0531 + 0.1077im
    g(Q) = (ComplexF64(Q) - z_lo) * (ComplexF64(Q) - z_hi) / (ComplexF64(Q) - p)
    fanti(Q) = conj(g(Q))      # SLAYER-like: a function of conj(Q)

    @testset "brute-force scan: roots and pole by winding number (anti-holomorphic)" begin
        scan = brute_force_scan(fanti, (-1.0, 1.0), (-1.0, 1.0); nre=81, nim=81, threaded=false)
        r = find_growth_rates(scan, 1.0; residual=fanti, classification=:winding, zero_winding=-1)
        @test r.classification === :winding
        @test length(r.valid_roots) == 2
        @test length(r.poles) == 1
        @test all(w -> w ≈ -1, r.valid_roots_winding)
        @test all(w -> w ≈ 1, r.poles_winding)
        @test length(r.valid_roots_winding) == length(r.valid_roots)
        @test length(r.poles_winding) == length(r.poles)
        @test minimum(abs.(r.valid_roots .- z_hi)) < 1e-8      # polished
        @test minimum(abs.(r.valid_roots .- z_lo)) < 1e-8
        @test abs(r.poles[1] - p) < 0.05
        # the least stable zero is reported even though it lies above the pole
        @test abs(r.Q_root - z_hi) < 1e-8
        @test r.gamma_Hz ≈ imag(z_hi) atol = 1e-8
        @test isnan(real(r.Q_root_secondary))
        @test !(:no_root in r.warning_flags)
    end

    @testset "holomorphic residual: zero_winding = +1" begin
        scan = brute_force_scan(g, (-1.0, 1.0), (-1.0, 1.0); nre=81, nim=81, threaded=false)
        r = find_growth_rates(scan, 1.0; residual=g, classification=:winding, zero_winding=1)
        @test length(r.valid_roots) == 2 && length(r.poles) == 1
        @test abs(r.Q_root - z_hi) < 1e-8
        # the wrong orientation swaps roots and poles: the convention matters
        w = find_growth_rates(scan, 1.0; residual=g, classification=:winding, zero_winding=-1)
        @test length(w.valid_roots) == 1 && length(w.poles) == 2
    end

    @testset "polish = false keeps the contour estimate" begin
        scan = brute_force_scan(fanti, (-1.0, 1.0), (-1.0, 1.0); nre=41, nim=41, threaded=false)
        r = find_growth_rates(scan, 1.0; residual=fanti, classification=:winding, polish=false)
        @test length(r.valid_roots) == 2
        @test 0 < abs(r.Q_root - z_hi) < 0.05
    end

    @testset "AMR scan" begin
        amr = amr_scan(fanti, (-1.0, 1.0), (-1.0, 1.0); nre0=12, nim0=12, passes=3)
        r = find_growth_rates(amr, 1.0; residual=fanti, classification=:winding)
        @test r.classification === :winding
        @test length(r.valid_roots) == 2
        @test length(r.poles) == 1
        @test abs(r.Q_root - z_hi) < 1e-8
    end

    @testset "an intersection with no zero inside is filtered as spurious" begin
        re_paths = [ComplexF64[-1.0, 1.0]]
        im_paths = [ComplexF64[-1.0im, 1.0im]]
        r = D._run_analysis(re_paths, im_paths, [Float64[0.0, 0.0]], 1.0;
            pole_threshold=10.0, filter_above_poles=true, filter_outside_re=true,
            residual=Q -> ComplexF64(1.0), classification=:winding, zero_winding=-1)
        @test isempty(r.valid_roots) && isempty(r.poles)
        @test r.filtered_roots == [0.0 + 0.0im]
        @test r.filtered_roots_winding ≈ [0.0]
        @test :spurious in r.warning_flags
        @test :no_root in r.warning_flags
        @test r.gamma_Hz == 0.0
    end

    @testset "threshold mode is unchanged and carries no winding numbers" begin
        scan = brute_force_scan(fanti, (-1.0, 1.0), (-1.0, 1.0); nre=41, nim=41, threaded=false)
        r = find_growth_rates(scan, 1.0)
        @test r.classification === :threshold
        @test isempty(r.valid_roots_winding) && isempty(r.poles_winding) && isempty(r.filtered_roots_winding)
    end

    @testset "argument validation" begin
        scan = brute_force_scan(fanti, (-1.0, 1.0), (-1.0, 1.0); nre=11, nim=11, threaded=false)
        @test_throws ArgumentError find_growth_rates(scan, 1.0; classification=:winding)
        @test_throws ArgumentError find_growth_rates(scan, 1.0; residual=fanti, classification=:winding, zero_winding=0)
        @test_throws ArgumentError find_growth_rates(scan, 1.0; classification=:bogus)
    end
end
