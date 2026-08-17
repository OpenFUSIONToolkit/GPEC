@testset "Dispersion AMR scan + triangulation extraction" begin
    using GeneralizedPerturbedEquilibrium.InnerLayer
    using GeneralizedPerturbedEquilibrium.InnerLayer: InnerLayerModel, solve_inner
    using GeneralizedPerturbedEquilibrium.Dispersion
    using StaticArrays

    @testset "amr_scan: basic structure and hash-caching" begin
        eval_count = Ref(0)
        function counting_f(Q)
            eval_count[] += 1
            return ComplexF64(Q)^2 - 1
        end

        # Small 2×2 initial grid → 9 unique corners
        amr = amr_scan(counting_f, (-1.0, 1.0), (-1.0, 1.0);
            nre0=2, nim0=2, passes=0)
        @test amr isa AMRResult
        @test length(amr.cells) == 4       # 2×2 cells
        # Dedup: 9 unique corners (3×3)
        @test length(amr.Q) == 9
        @test length(amr.Δ) == 9
        @test eval_count[] == 9            # exactly one call per unique Q
    end

    @testset "amr_scan: refinement concentrates cells near zero crossings" begin
        f(Q) = ComplexF64(Q) - (0.3 + 0.4im)       # single zero
        amr0 = amr_scan(f, (-1.0, 1.0), (-1.0, 1.0); nre0=4, nim0=4, passes=0)
        amr3 = amr_scan(f, (-1.0, 1.0), (-1.0, 1.0); nre0=4, nim0=4, passes=3)
        @test length(amr3.cells) > length(amr0.cells)
        @test length(amr3.Q) > length(amr0.Q)
        # A 4×4 coarse grid is 16 cells; adding 3 refinement passes must
        # leave the total bounded by exponential growth of only the cells
        # bracketing the root (roughly linear in the path length).
        @test length(amr3.cells) < 1000    # not exponential in passes
    end

    @testset "amr_scan: argument validation" begin
        @test_throws ArgumentError amr_scan(identity, (0.0, 1.0), (0.0, 1.0);
            nre0=0, nim0=2, passes=1)
        @test_throws ArgumentError amr_scan(identity, (0.0, 1.0), (0.0, 1.0);
            nre0=2, nim0=0, passes=1)
        @test_throws ArgumentError amr_scan(identity, (0.0, 1.0), (0.0, 1.0);
            nre0=2, nim0=2, passes=-1)
    end

    @testset "amr_scan: max_cells safety cap fires" begin
        # A pathological f that forces every cell to subdivide every pass
        f(Q) = 0.0 + 0.0im        # identically zero → every cell crosses
        @test_throws ErrorException amr_scan(f, (-1.0, 1.0), (-1.0, 1.0);
            nre0=4, nim0=4, passes=10,
            max_cells=100)
    end

    @testset "find_growth_rates(AMR): single isolated root" begin
        Q_root = 0.42 + 0.27im
        f(Q) = ComplexF64(Q) - Q_root
        amr = amr_scan(f, (-1.0, 1.5), (-0.5, 1.0);
            nre0=8, nim0=6, passes=4)
        result = find_growth_rates(amr, 1.0)
        @test result isa GrowthRateResult
        @test abs(result.Q_root - Q_root) < 1e-3     # AMR-resolution limited
        @test isempty(result.poles)
        @test length(result.valid_roots) == 1
    end

    @testset "find_growth_rates(AMR): higher-γ root selected" begin
        Q1 = 0.3 + 0.5im      # higher γ
        Q2 = -0.4 + 0.1im
        f(Q) = (ComplexF64(Q) - Q1) * (ComplexF64(Q) - Q2)
        amr = amr_scan(f, (-1.0, 1.0), (-0.3, 0.8);
            nre0=10, nim0=8, passes=4)
        result = find_growth_rates(amr, 1.0)
        @test length(result.valid_roots) == 2
        @test abs(result.Q_root - Q1) < 1e-2
    end

    @testset "find_growth_rates(AMR): pole detection" begin
        Q_r = 0.4 + 0.2im
        Q_p = -0.5 + 0.6im
        f(Q) = (ComplexF64(Q) - Q_r) / (ComplexF64(Q) - Q_p)
        amr = amr_scan(f, (-1.5, 1.5), (-0.5, 1.5);
            nre0=10, nim0=8, passes=5)
        result = find_growth_rates(amr, 1.0; pole_threshold=10.0)
        @test length(result.poles) >= 1
        @test any(p -> abs(p - Q_p) < 0.05, result.poles)
        @test abs(result.Q_root - Q_r) < 1e-3
    end

    @testset "find_growth_rates(AMR): tauk normalization" begin
        Q_root = 1.0 + 2.0im
        f(Q) = ComplexF64(Q) - Q_root
        amr = amr_scan(f, (-2.0, 3.0), (-1.0, 4.0);
            nre0=8, nim0=8, passes=4)
        tauk = 5e-5
        result = find_growth_rates(amr, tauk)
        @test result.omega_Hz ≈ real(result.Q_root) / tauk
        @test result.gamma_Hz ≈ imag(result.Q_root) / tauk
    end

    @testset "find_growth_rates(AMR): argument validation" begin
        # Too few points to triangulate
        GRE = GeneralizedPerturbedEquilibrium.Dispersion
        @test_throws ArgumentError GRE._extract_growth_rates_amr(
            ComplexF64[0.0+0im, 1.0+0im], ComplexF64[1.0+0im, 2.0+0im], 1.0;
            re_target=0.0, im_target=0.0, pole_threshold=10.0,
            filter_above_poles=true, filter_outside_re=true)
        # Length mismatch
        @test_throws ArgumentError GRE._extract_growth_rates_amr(
            ComplexF64[0.0+0im, 1.0+0im, 1.0+1im],
            ComplexF64[1.0+0im, 2.0+0im], 1.0;
            re_target=0.0, im_target=0.0, pole_threshold=10.0,
            filter_above_poles=true, filter_outside_re=true)
    end

    @testset "AMR vs brute-force: same root to within AMR refinement precision" begin
        # Sanity: the AMR and brute-force paths should find the same root
        # (to roughly the AMR resolution — the AMR typically resolves
        # better per-evaluation than a uniform grid).
        Q_root = 0.5 + 0.3im
        f(Q) = ComplexF64(Q) - Q_root
        scan = brute_force_scan(f, (-1.0, 1.0), (-0.5, 1.0);
            nre=80, nim=60, threaded=false)
        amr = amr_scan(f, (-1.0, 1.0), (-0.5, 1.0);
            nre0=8, nim0=6, passes=4)
        r_grid = find_growth_rates(scan, 1.0)
        r_amr = find_growth_rates(amr, 1.0)
        @test abs(r_grid.Q_root - Q_root) < 1e-3
        @test abs(r_amr.Q_root - Q_root) < 1e-3
        @test abs(r_grid.Q_root - r_amr.Q_root) < 5e-3
    end

    @testset "API: SurfaceCoupling and MultiSurfaceCoupling through amr_scan" begin
        struct LinModel <: InnerLayerModel
            a::ComplexF64
            b::ComplexF64
        end
        GeneralizedPerturbedEquilibrium.InnerLayer.solve_inner(
            m::LinModel, params, Q::Number) =
            InnerLayerResponse(m.a + m.b * ComplexF64(Q), zero(ComplexF64))

        Q_pin = 0.7 - 0.3im
        sc = surface_coupling(LinModel(0.0im, 1.0+0im), nothing,
            Q_pin; scale=1.0, tauk=1.0)
        amr = amr_scan(sc, (-0.5, 1.5), (-1.0, 0.5);
            nre0=8, nim0=6, passes=4)
        r = find_growth_rates(amr, sc.tauk)
        @test abs(r.Q_root - Q_pin) < 1e-2

        # Multi-surface coupled scan through AMR
        Q_a, Q_b = 0.7 - 0.3im, -0.4 + 0.5im
        sc1 = surface_coupling(LinModel(0.0im, 1.0+0im), nothing,
            ComplexF64(0); scale=1.0, tauk=1.0)
        sc2 = surface_coupling(LinModel(0.0im, 1.0+0im), nothing,
            ComplexF64(0); scale=1.0, tauk=1.0)
        dp = ComplexF64[Q_a 0.0; 0.0 Q_b]
        mc = multi_surface_coupling([sc1, sc2], dp)
        amr_c = amr_scan(mc, (-1.0, 1.5), (-1.0, 1.0);
            nre0=10, nim0=8, passes=4)
        r_c = find_growth_rates(amr_c, mc.surfaces[mc.ref_idx].tauk)
        @test abs(r_c.Q_root - Q_b) < 1e-2     # higher-γ root
    end

    # =========================================================================
    # multi_box_amr_scan
    # =========================================================================
    using GeneralizedPerturbedEquilibrium.Dispersion: BoxActivity, NoActivity,
        ReZeroCrossing, ImZeroCrossing, PoleMagnitude, MultiBoxAMRResult,
        multi_box_amr_scan, as_amr_result

    @testset "multi_box_amr_scan: 3-box stripe with zero, pole, and inactive box" begin
        # Synthetic residual: zero at Q=0 (centre stripe), pole at Q=-50
        # (left stripe), nothing in right stripe. Complex offset 1+1im keeps
        # Im(f) above zero in the right stripe so its sign-change tests don't
        # fire spuriously on rational-function residuals (Im=0 contour
        # otherwise crosses the entire real axis).
        f(Q) = (ComplexF64(Q) - 0.0) / (ComplexF64(Q) - (-50.0)) + (1.0 + 1.0im)
        boxes = [((-75.0, -25.0), (-25.0, 25.0)),
            ((-25.0, 25.0), (-25.0, 25.0)),
            ((25.0, 75.0), (-25.0, 25.0))]
        result = multi_box_amr_scan(f, boxes;
            pole_magnitude_threshold=10.0,
            prescreen_nre=25, prescreen_nim=25,
            nre0=25, nim0=25, passes=2,
            max_cells=100_000,
            max_cells_action=:warn_truncate,
            parallel=false)
        @test result isa MultiBoxAMRResult
        @test length(result.box_results) == 3
        @test length(result.box_activity) == 3
        @test result.box_activity[1] != NoActivity   # contains pole
        @test result.box_activity[2] != NoActivity   # contains zero
        @test result.box_activity[3] == NoActivity   # empty stripe
        @test result.box_results[3] === nothing
        @test result.box_results[1] !== nothing
        @test result.box_results[2] !== nothing
        # prescreen_evals is bounded by 3 boxes × 26×26 = 2028 (some shared
        # boundary corners are deduplicated within each box's local cache, so
        # the count is ≤ 2028).
        @test result.prescreen_evals ≤ 3 * 26 * 26

        # as_amr_result wraps cleanly
        amr = as_amr_result(result)
        @test amr isa AMRResult
        @test length(amr.cells) == length(result.cells)
        @test length(amr.Q) == length(result.Q)
    end

    @testset "multi_box_amr_scan: pole-only path" begin
        # Sharp pole at Q=-50+0i with complex offset that keeps Re(f),Im(f) one-
        # signed across the prescreen grid except in the cell containing the
        # pole. Confirms the |Δ| ≥ pole_magnitude_threshold criterion fires
        # independent of sign-change tests.
        g(Q) = 1000.0 / (ComplexF64(Q) - (-50.0))^2 + (5.0 + 5.0im)
        boxes = [((-75.0, -25.0), (-25.0, 25.0)),
            ((-25.0, 25.0), (-25.0, 25.0)),
            ((25.0, 75.0), (-25.0, 25.0))]
        result = multi_box_amr_scan(g, boxes;
            pole_magnitude_threshold=50.0,
            prescreen_nre=25, prescreen_nim=25,
            nre0=25, nim0=25, passes=1,
            max_cells=100_000,
            max_cells_action=:warn_truncate,
            parallel=false)
        @test result.box_activity[1] != NoActivity
        @test result.box_activity[2] == NoActivity
        @test result.box_activity[3] == NoActivity
    end

    @testset "multi_box_amr_scan: argument validation" begin
        f(Q) = ComplexF64(Q)
        boxes = [((-1.0, 1.0), (-1.0, 1.0))]
        @test_throws ArgumentError multi_box_amr_scan(f, boxes;
            pole_magnitude_threshold=1.0, prescreen_nre=0)
        @test_throws ArgumentError multi_box_amr_scan(f, boxes;
            pole_magnitude_threshold=1.0, prescreen_nim=0)
        @test_throws ArgumentError multi_box_amr_scan(f, boxes;
            pole_magnitude_threshold=-1.0)
    end
end
