@testset "Dispersion root polishing" begin
    using GeneralizedPerturbedEquilibrium.Dispersion
    using LinearAlgebra
    D = GeneralizedPerturbedEquilibrium.Dispersion

    # ----------------------------------------------------------------
    # _polish_root: core local-solve behaviour on synthetic residuals
    # ----------------------------------------------------------------
    @testset "_polish_root: recovers a known simple root, |f| → 0" begin
        r = 0.42 + 0.27im
        f(Q) = ComplexF64(Q) - r          # zero exactly at r
        Q0 = r + (0.013 - 0.009im)         # off-root start
        Qp, nev, improved = D._polish_root(f, Q0, 0.1)
        @test improved
        @test abs(Qp - r) < 1e-10
        @test abs(f(Qp)) < 1e-10
        @test nev <= 3 * 8 + 2             # bounded eval budget
    end

    @testset "_polish_root: nonlinear residual converges" begin
        r = -0.3 + 0.5im
        f(Q) = (ComplexF64(Q) - r) * (1 + 0.4 * (ComplexF64(Q) - r))  # simple root at r
        Qp, _, improved = D._polish_root(f, r + 0.02 - 0.015im, 0.15)
        @test improved
        @test abs(Qp - r) < 1e-7
    end

    @testset "_polish_root: no-op-or-better invariant (never worse)" begin
        # The polish must never return a point with larger |f| than the start.
        for r in (0.1 + 0.2im, -0.5 + 0.0im, 0.3 - 0.4im)
            f(Q) = (ComplexF64(Q) - r) * (ComplexF64(Q) + 1.7)
            for off in (0.01 + 0.0im, 0.05 + 0.0im, 0.0 + 0.08im, -0.03 + 0.02im)
                Q0 = r + off
                Qp, _, _ = D._polish_root(f, Q0, 0.2)
                @test abs(f(Qp)) <= abs(f(Q0)) + 1e-12
            end
        end
    end

    @testset "_polish_root: trust region bounds the step (no jump to far root)" begin
        f(Q) = ComplexF64(Q) - (10.0 + 0.0im)   # only root is far outside R
        Q0 = 0.0 + 0.0im
        R = 0.1
        Qp, _, _ = D._polish_root(f, Q0, R)
        @test abs(Qp - Q0) <= R + 1e-9          # never leaves the trust disk
        @test abs(Qp - (10.0 + 0.0im)) > 5.0    # did NOT migrate to the far root
        @test abs(f(Qp)) <= abs(f(Q0))          # still moved downhill within the disk
    end

    @testset "_polish_root: non-finite residual falls back, no crash" begin
        fnan(Q) = ComplexF64(NaN, NaN)
        Qp, nev, improved = D._polish_root(fnan, 0.3 + 0.1im, 0.1)
        @test !improved
        @test Qp == 0.3 + 0.1im                 # hard fallback to the start
        @test nev >= 1
    end

    @testset "_polish_root: zero radius is a no-op" begin
        f(Q) = ComplexF64(Q) - (0.2 + 0.1im)
        Qp, _, improved = D._polish_root(f, 0.25 + 0.05im, 0.0)
        @test !improved
        @test Qp == 0.25 + 0.05im
    end

    @testset "_polish_root: deterministic (same input → same output)" begin
        f(Q) = (ComplexF64(Q) - (0.5 + 0.3im)) * (ComplexF64(Q) - 2)
        a = D._polish_root(f, 0.52 + 0.31im, 0.1)
        b = D._polish_root(f, 0.52 + 0.31im, 0.1)
        @test a[1] == b[1]
        @test a[2] == b[2]
    end

    @testset "_polish_root: BLAS-thread-independent for a det residual" begin
        # det() routes through LAPACK; polishing must be invariant to BLAS threads
        # (this is the property the parallel-scan BLAS pin + polish together give).
        Qstar = 0.3 + 0.2im
        fdet(Q) = det(ComplexF64[1 0 0; 0 1 0; 0 0 (ComplexF64(Q)-Qstar)])
        b0 = BLAS.get_num_threads()
        try
            BLAS.set_num_threads(1)
            r1 = D._polish_root(fdet, 0.31 + 0.21im, 0.1)
            BLAS.set_num_threads(4)
            r4 = D._polish_root(fdet, 0.31 + 0.21im, 0.1)
            @test isapprox(r1[1], r4[1]; atol=1e-10)
            @test abs(fdet(r1[1])) < 1e-8
        finally
            BLAS.set_num_threads(b0)
        end
    end

    # ----------------------------------------------------------------
    # _polish_trust_radius: neighbour-aware bound keeps roots coherent
    # ----------------------------------------------------------------
    @testset "_polish_trust_radius: capped by nearest neighbour spacing" begin
        # Two close intersections + one far: radius of #1 is bound by the close one
        pts = ComplexF64[0+0im, 0.001+0im, 5+0im]
        R1 = D._polish_trust_radius(pts, 1, 0.01)
        @test R1 <= 0.45 * 0.001 + 1e-15        # < half the close-pair spacing
        # Isolated pair: radius falls back to the grid-cell cap (k_cell·h)
        R2 = D._polish_trust_radius(ComplexF64[0+0im, 5+0im], 1, 0.01)
        @test R2 <= 3.0 * 0.01 + 1e-12
        @test R2 > 0
    end

    @testset "_polish_trust_radius + polish: close roots stay coherent (no swap/merge)" begin
        r1 = 0.30 + 0.20im
        r2 = r1 + 0.002                          # second root 0.002 away
        f(Q) = (ComplexF64(Q) - r1) * (ComplexF64(Q) - r2)
        pts = ComplexF64[r1 + 0.0003, r2 - 0.0003]   # coarse contour estimates
        R1 = D._polish_trust_radius(pts, 1, 0.001)
        R2 = D._polish_trust_radius(pts, 2, 0.001)
        p1 = D._polish_root(f, pts[1], R1)[1]
        p2 = D._polish_root(f, pts[2], R2)[1]
        @test abs(p1 - r1) < abs(p1 - r2)        # #1 refined toward its OWN root
        @test abs(p2 - r2) < abs(p2 - r1)        # #2 refined toward its OWN root
        @test p1 != p2                           # did not collapse to one point
    end

    # ----------------------------------------------------------------
    # _median_segment_length: grid-resolution proxy
    # ----------------------------------------------------------------
    @testset "_median_segment_length" begin
        re_paths = [ComplexF64[0+0im, 0.1+0im, 0.3+0im]]   # segment lengths 0.1, 0.2
        im_paths = [ComplexF64[0+0im, 0+0.1im]]            # segment length 0.1
        @test D._median_segment_length(re_paths, im_paths) ≈ 0.1   # median of [0.1,0.2,0.1]
        @test D._median_segment_length(Vector{Vector{ComplexF64}}(),
                                       Vector{Vector{ComplexF64}}()) == 0.0
    end

    # ----------------------------------------------------------------
    # Integration: find_growth_rates with `residual` polishes the root
    # ----------------------------------------------------------------
    @testset "find_growth_rates: polish recovers the true root from a coarse scan" begin
        r = 0.42 + 0.27im
        f(Q) = (ComplexF64(Q) - r) * (1 + 0.4 * (ComplexF64(Q) - r))  # nonlinear, root at r
        scan = brute_force_scan(f, (0.0, 1.0), (0.0, 0.6); nre=15, nim=12, threaded=false)
        gr_raw = find_growth_rates(scan, 1.0; residual=nothing)
        gr_pol = find_growth_rates(scan, 1.0; residual=f)
        @test !isnan(real(gr_raw.Q_root))                 # coarse scan still isolates it
        @test abs(f(gr_pol.Q_root)) <= abs(f(gr_raw.Q_root))   # polish never worsens |f|
        @test abs(gr_pol.Q_root - r) < 1e-4               # and converges to the true root
        # polishing must not invent or drop roots vs the raw pass
        @test length(gr_pol.valid_roots) == length(gr_raw.valid_roots)
    end

    @testset "find_growth_rates: residual=nothing is unchanged (backward compatible)" begin
        r = 0.42 + 0.27im
        f(Q) = ComplexF64(Q) - r
        scan = brute_force_scan(f, (0.0, 1.0), (0.0, 0.6); nre=21, nim=15, threaded=false)
        a = find_growth_rates(scan, 1.0)                  # default: no residual, no polish
        b = find_growth_rates(scan, 1.0; residual=nothing)
        @test a.Q_root == b.Q_root
    end

    # ----------------------------------------------------------------
    # Validity gate: drop intersections that don't polish to a true zero
    # ----------------------------------------------------------------
    @testset "find_growth_rates: validity gate keeps real roots, drops spurious" begin
        r = 0.42 + 0.27im
        f(Q) = (ComplexF64(Q) - r) * (1 + 0.4 * (ComplexF64(Q) - r))  # real root at r
        scan = brute_force_scan(f, (0.0, 1.0), (0.0, 0.6); nre=15, nim=12, threaded=false)

        # Normal tolerance: the genuine root polishes to |f|≈0 ⇒ kept, not flagged
        gr = find_growth_rates(scan, 1.0; residual=f, validity_rtol=1e-3)
        @test !(:spurious in gr.warning_flags)
        @test !isnan(real(gr.Q_root))
        @test abs(gr.Q_root - r) < 1e-4

        # Absurdly tight tolerance: even the true root's residual (~machine eps)
        # exceeds validity_rtol·median(|Δ|) ⇒ every candidate dropped. Exercises
        # the gate firing + the :spurious / :no_root / filtered_roots plumbing.
        gr2 = find_growth_rates(scan, 1.0; residual=f, validity_rtol=1e-300)
        @test :spurious in gr2.warning_flags
        @test :no_root in gr2.warning_flags          # all candidates were spurious
        @test isnan(real(gr2.Q_root))                # ⇒ no root reported
        @test length(gr2.filtered_roots) >= 1        # dropped candidate parked here
        @test isempty(gr2.valid_roots)

        # Gate is inert without a residual (no polishing ⇒ nothing to test against)
        gr3 = find_growth_rates(scan, 1.0; residual=nothing, validity_rtol=1e-300)
        @test !(:spurious in gr3.warning_flags)
        @test !isnan(real(gr3.Q_root))
    end
end
