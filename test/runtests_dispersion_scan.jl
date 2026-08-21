@testset "Dispersion brute-force scan + growth-rate extraction" begin
    using GeneralizedPerturbedEquilibrium.InnerLayer
    using GeneralizedPerturbedEquilibrium.InnerLayer: InnerLayerModel, solve_inner
    using GeneralizedPerturbedEquilibrium.Dispersion
    using StaticArrays

    @testset "brute_force_scan: regular grid evaluation" begin
        f(Q) = ComplexF64(Q)^2 - 1
        scan = brute_force_scan(f, (-2.0, 2.0), (-1.0, 1.0);
            nre=21, nim=11, threaded=false)
        @test scan isa ScanResult
        @test size(scan.Q) == (21, 11)
        @test size(scan.Δ) == (21, 11)
        @test length(scan.re_axis) == 21
        @test length(scan.im_axis) == 11
        @test scan.re_axis[1] == -2.0
        @test scan.re_axis[end] == 2.0
        @test scan.im_axis[1] == -1.0
        @test scan.im_axis[end] == 1.0
        # Spot-check a grid value
        i, j = 11, 6
        @test scan.Q[i, j] ≈ scan.re_axis[i] + scan.im_axis[j]*im
        @test scan.Δ[i, j] ≈ scan.Q[i, j]^2 - 1
    end

    @testset "brute_force_scan: threaded vs non-threaded agree" begin
        f(Q) = sin(ComplexF64(Q))
        s_t = brute_force_scan(f, (-1.0, 1.0), (-0.5, 0.5);
            nre=15, nim=10, threaded=true)
        s_n = brute_force_scan(f, (-1.0, 1.0), (-0.5, 0.5);
            nre=15, nim=10, threaded=false)
        @test s_t.Δ == s_n.Δ
    end

    @testset "brute_force_scan: argument validation" begin
        @test_throws ArgumentError brute_force_scan(identity, (0.0, 1.0),
            (0.0, 1.0); nre=1, nim=10)
        @test_throws ArgumentError brute_force_scan(identity, (0.0, 1.0),
            (0.0, 1.0); nre=10, nim=1)
    end

    @testset "find_growth_rates: single isolated root" begin
        # Δ(Q) = Q - Q_root → unique zero at Q_root
        Q_root = 0.42 + 0.27im
        f(Q) = ComplexF64(Q) - Q_root
        scan = brute_force_scan(f, (-1.0, 1.5), (-0.5, 1.0);
            nre=80, nim=60, threaded=false)
        result = find_growth_rates(scan, 1.0)
        @test result isa GrowthRateResult
        @test isempty(result.poles)
        @test length(result.valid_roots) == 1
        @test abs(result.Q_root - Q_root) < 1e-3      # grid-resolution limited
        @test result.omega_Hz ≈ real(result.Q_root)
        @test result.gamma_Hz ≈ imag(result.Q_root)
    end

    @testset "find_growth_rates: multiple roots — picks highest γ" begin
        # Two roots; the higher-γ one must be reported
        Q1 = 0.3 + 0.5im       # higher γ
        Q2 = -0.4 + 0.1im      # lower γ
        f(Q) = (ComplexF64(Q) - Q1) * (ComplexF64(Q) - Q2)
        scan = brute_force_scan(f, (-1.0, 1.0), (-0.3, 0.8);
            nre=100, nim=80, threaded=false)
        result = find_growth_rates(scan, 1.0)
        @test length(result.valid_roots) == 2
        @test abs(result.Q_root - Q1) < 1e-3        # higher-γ root chosen
        @test imag(result.Q_root) > imag(Q2)
    end

    @testset "find_growth_rates: pole detection" begin
        # Δ(Q) = (Q - Q_root)/(Q - Q_pole) → 1 zero, 1 pole
        Q_r = 0.4 + 0.2im
        Q_p = -0.5 + 0.6im     # pole at higher γ
        f(Q) = (ComplexF64(Q) - Q_r) / (ComplexF64(Q) - Q_p)
        scan = brute_force_scan(f, (-1.5, 1.5), (-0.5, 1.5);
            nre=120, nim=100, threaded=false)
        result = find_growth_rates(scan, 1.0; pole_threshold=10.0)
        # Pole correctly classified — but the root is at lower γ than the
        # pole, so even with filter_above_poles=true the root must survive.
        @test length(result.poles) >= 1
        @test any(p -> abs(p - Q_p) < 0.05, result.poles)
        @test abs(result.Q_root - Q_r) < 1e-3
    end

    @testset "find_growth_rates: tauk normalization to physical Hz" begin
        Q_root = 1.0 + 2.0im
        f(Q) = ComplexF64(Q) - Q_root
        scan = brute_force_scan(f, (-2.0, 3.0), (-1.0, 4.0);
            nre=80, nim=80, threaded=false)
        tauk = 5.0e-5
        result = find_growth_rates(scan, tauk)
        @test result.omega_Hz ≈ real(result.Q_root) / tauk
        @test result.gamma_Hz ≈ imag(result.Q_root) / tauk
        # Check sensible orders of magnitude (Q_root ≈ 1+2im, tauk ≈ 5e-5)
        @test result.omega_Hz ≈ 1 / tauk atol = 1 / tauk * 5e-3
        @test result.gamma_Hz ≈ 2 / tauk atol = 2 / tauk * 5e-3
    end

    @testset "find_growth_rates: empty result when no contour intersections" begin
        # Δ(Q) = 1 + Q (only a single zero at Q=-1; if scanned over a box
        # away from -1 there will be no Im(Δ)=0 contour intersecting Re=0).
        f(Q) = 1.0 + ComplexF64(Q)
        # Choose a box where Δ has no zeros — far above the real axis
        scan = brute_force_scan(f, (1.0, 2.0), (1.0, 2.0);
            nre=30, nim=30, threaded=false)
        result = find_growth_rates(scan, 1.0)
        # Either no valid roots, or a NaN Q_root
        @test isempty(result.valid_roots) || isnan(real(result.Q_root))
    end

    @testset "API: SurfaceCoupling and MultiSurfaceCoupling are scannable" begin
        # Synthetic linear inner-layer model — verifies the Dispersion API
        # accepts the actual residual containers, not just plain functions.
        struct LinModel <: InnerLayerModel
            a::ComplexF64
            b::ComplexF64
        end
        GeneralizedPerturbedEquilibrium.InnerLayer.solve_inner(
            m::LinModel, params, Q::Number) =
            InnerLayerResponse(m.a + m.b * ComplexF64(Q), zero(ComplexF64))

        # Single-surface scan via SurfaceCoupling (Q_root by construction = 0.7-0.3im)
        Q_pin = 0.7 - 0.3im
        sc = surface_coupling(LinModel(0.0im, 1.0+0im), nothing,
            Q_pin; scale=1.0, tauk=1.0)
        scan = brute_force_scan(sc, (-0.5, 1.5), (-1.0, 0.5);
            nre=80, nim=80, threaded=false)
        res = find_growth_rates(scan, sc.tauk)
        @test abs(res.Q_root - Q_pin) < 1e-3

        # Coupled scan via MultiSurfaceCoupling — pair two surfaces with
        # *different* Q_pin values so the resulting determinant has simple
        # (non-degenerate) roots that contour intersection can localize.
        # Note: MultiSurfaceCoupling builds M[k,k] = dp[k,k] - Δ_inner_k(Q),
        # so to put a root at Q = Q_pin_k we need dp[k,k] = Q_pin_k (the
        # full complex value, not just its real part).
        Q_a, Q_b = 0.7 - 0.3im, -0.4 + 0.5im
        sc1 = surface_coupling(LinModel(0.0im, 1.0+0im), nothing,
            ComplexF64(0); scale=1.0, tauk=1.0)
        sc2 = surface_coupling(LinModel(0.0im, 1.0+0im), nothing,
            ComplexF64(0); scale=1.0, tauk=1.0)
        dp = ComplexF64[Q_a 0.0; 0.0 Q_b]               # diagonal Δ'
        mc = multi_surface_coupling([sc1, sc2], dp)
        scan_c = brute_force_scan(mc, (-1.0, 1.5), (-1.0, 1.0);
            nre=120, nim=100, threaded=false)
        res_c = find_growth_rates(scan_c, mc.surfaces[mc.ref_idx].tauk)
        # With diagonal Δ', det = (Q_a - Q)·(Q_b - Q) → roots at Q_a, Q_b.
        # The higher-γ root is Q_b (γ = 0.5).
        @test abs(res_c.Q_root - Q_b) < 1e-2
    end
end
