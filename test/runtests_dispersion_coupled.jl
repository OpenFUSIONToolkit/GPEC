@testset "Dispersion coupled determinant" begin
    using GeneralizedPerturbedEquilibrium.InnerLayer
    using GeneralizedPerturbedEquilibrium.InnerLayer: InnerLayerModel, solve_inner
    using GeneralizedPerturbedEquilibrium.Dispersion
    using LinearAlgebra
    using StaticArrays

    # ---------------------------------------------------------------
    # Synthetic linear inner-layer model with adjustable per-surface
    # tauk for testing the Q rescaling logic.
    #   Δ_inner(Q) = a + b·Q
    # ---------------------------------------------------------------
    struct LinTestModel <: InnerLayerModel
        a::ComplexF64
        b::ComplexF64
    end
    GeneralizedPerturbedEquilibrium.InnerLayer.solve_inner(
        m::LinTestModel, params, Q::Number) =
        InnerLayerResponse(m.a + m.b * ComplexF64(Q), zero(ComplexF64))

    function _slayer_ref()
        return slayer_parameters(
            n_e=5.0e19, t_e=1000.0, t_i=1000.0,
            omega=0.0, omega_e=1.0e4, omega_i=5.0e3,
            qval=2.0, sval_r=1.0, bt=2.0,
            rs=0.5, R0=1.7, mu_i=2.0, zeff=1.0,
            chi_perp=1.0, chi_tor=1.0, m=2, n=1)
    end

    @testset "Constructor validation" begin
        sc1 = surface_coupling(LinTestModel(0.0im, 1.0+0im), nothing,
                               1.0+0im; scale=1.0, tauk=1.0)
        sc2 = surface_coupling(LinTestModel(0.0im, 1.0+0im), nothing,
                               2.0+0im; scale=1.0, tauk=1.0)
        good_dp = ComplexF64[1.0 0.1; 0.1 2.0]

        mc = multi_surface_coupling([sc1, sc2], good_dp)
        @test mc.ref_idx == 1
        @test mc.msing_max == 2          # min(3, 2) = 2
        @test size(mc.dp_matrix) == (2, 2)

        # 3-surface default also caps at 3 (min(3, 3) = 3)
        sc3 = surface_coupling(LinTestModel(0.0im, 1.0+0im), nothing,
                               3.0+0im; scale=1.0, tauk=1.0)
        good_dp3 = ComplexF64[1.0 0.1 0.0; 0.1 2.0 0.0; 0.0 0.0 3.0]
        mc3 = multi_surface_coupling([sc1, sc2, sc3], good_dp3)
        @test mc3.msing_max == 3

        # 4-surface case caps at 3 (the design default — Δ' beyond 3 surfaces
        # tends to be erratic in practice)
        sc4 = surface_coupling(LinTestModel(0.0im, 1.0+0im), nothing,
                               4.0+0im; scale=1.0, tauk=1.0)
        good_dp4 = ComplexF64[1.0 0.0 0.0 0.0;
                               0.0 2.0 0.0 0.0;
                               0.0 0.0 3.0 0.0;
                               0.0 0.0 0.0 4.0]
        mc4 = multi_surface_coupling([sc1, sc2, sc3, sc4], good_dp4)
        @test mc4.msing_max == 3         # default capped at 3
        # Caller can opt in to all 4
        mc4_full = multi_surface_coupling([sc1, sc2, sc3, sc4], good_dp4;
                                           msing_max=4)
        @test mc4_full.msing_max == 4

        # Mismatched dp size
        @test_throws ArgumentError multi_surface_coupling(
            [sc1, sc2], ComplexF64[1.0 0.0 0.0; 0.0 2.0 0.0; 0.0 0.0 3.0])
        @test_throws ArgumentError multi_surface_coupling(
            [sc1, sc2], ComplexF64[1.0 0.0])

        # Out-of-range ref_idx
        @test_throws ArgumentError multi_surface_coupling([sc1, sc2], good_dp;
                                                           ref_idx=3)
        @test_throws ArgumentError multi_surface_coupling([sc1, sc2], good_dp;
                                                           ref_idx=0)

        # Out-of-range msing_max
        @test_throws ArgumentError multi_surface_coupling([sc1, sc2], good_dp;
                                                           msing_max=3)
        @test_throws ArgumentError multi_surface_coupling([sc1, sc2], good_dp;
                                                           msing_max=0)
    end

    @testset "Diagonal Δ' factorizes (det = ∏ per-surface residuals)" begin
        # When dp_matrix is diagonal, no off-diagonal coupling exists and
        # the coupled determinant should reduce exactly to the product of
        # per-surface residuals.
        sc1 = surface_coupling(LinTestModel(1.0+0im, 1.0+0im), nothing,
                               5.0+0im; scale=1.0, tauk=1.0)
        sc2 = surface_coupling(LinTestModel(2.0+0im, 1.0+0im), nothing,
                               7.0+0im; scale=1.0, tauk=1.0)
        sc3 = surface_coupling(LinTestModel(0.5+0im, 0.5+0im), nothing,
                               3.0+0im; scale=1.0, tauk=1.0)
        dp = ComplexF64[5.0 0.0 0.0;
                         0.0 7.0 0.0;
                         0.0 0.0 3.0]
        mc = multi_surface_coupling([sc1, sc2, sc3], dp)
        for Q in (0.5+0im, 2.0+0.3im, -1.0-0.5im, 4.5+1.0im)
            @test mc(Q) ≈ sc1(Q) * sc2(Q) * sc3(Q) rtol = 1e-12
        end
    end

    @testset "Diagonal Δ' roots = single-surface roots" begin
        # With Δ_inner(Q) = b·Q and dp_diag = b·Q_root for each surface,
        # the coupled determinant has its roots exactly at the union of
        # single-surface roots.
        Q1, Q2 = 0.5+0.0im, 2.0+0.0im
        sc1 = surface_coupling(LinTestModel(0.0im, 1.0+0im), nothing,
                               Q1; scale=1.0, tauk=1.0)
        sc2 = surface_coupling(LinTestModel(0.0im, 1.0+0im), nothing,
                               Q2; scale=1.0, tauk=1.0)
        dp = ComplexF64[real(Q1) 0.0; 0.0 real(Q2)]
        mc = multi_surface_coupling([sc1, sc2], dp)
        @test abs(mc(Q1)) < 1e-12
        @test abs(mc(Q2)) < 1e-12
        @test abs(mc(0.0+0.0im)) > 0
    end

    @testset "Off-diagonal coupling shifts the roots away from the diagonal" begin
        sc1 = surface_coupling(LinTestModel(0.0im, 1.0+0im), nothing,
                               0.5+0im; scale=1.0, tauk=1.0)
        sc2 = surface_coupling(LinTestModel(0.0im, 1.0+0im), nothing,
                               2.0+0im; scale=1.0, tauk=1.0)
        # Coupling-free baseline
        dp_diag = ComplexF64[0.5 0.0; 0.0 2.0]
        mc_diag = multi_surface_coupling([sc1, sc2], dp_diag)
        # With off-diagonal coupling
        dp_offd = ComplexF64[0.5 0.3; 0.3 2.0]
        mc_offd = multi_surface_coupling([sc1, sc2], dp_offd)

        # Single-surface roots are no longer roots of the coupled det
        Q1 = 0.5 + 0.0im
        @test abs(mc_diag(Q1)) < 1e-12       # diagonal: still a root
        @test abs(mc_offd(Q1)) > 0           # coupled: no longer a root
        # The shift size matches the off-diagonal magnitude squared
        # det = (0.5-Q)(2-Q) - 0.3² ⇒ at Q=0.5 the det = -0.09
        @test mc_offd(Q1) ≈ -0.09 rtol = 1e-12
    end

    @testset "msing_max truncation uses upper-left submatrix" begin
        sc1 = surface_coupling(LinTestModel(0.0im, 1.0+0im), nothing,
                               1.0+0im; scale=1.0, tauk=1.0)
        sc2 = surface_coupling(LinTestModel(0.0im, 1.0+0im), nothing,
                               2.0+0im; scale=1.0, tauk=1.0)
        sc3 = surface_coupling(LinTestModel(0.0im, 1.0+0im), nothing,
                               3.0+0im; scale=1.0, tauk=1.0)
        dp = ComplexF64[1.0 0.0 0.0;
                         0.0 2.0 0.0;
                         0.0 0.0 3.0]

        # msing_max = 1 reduces to sc1(Q) alone
        mc1 = multi_surface_coupling([sc1, sc2, sc3], dp; msing_max=1)
        for Q in (0.0+0im, 1.0+0im, 2.0+0im)
            @test mc1(Q) ≈ sc1(Q)
        end

        # msing_max = 2 uses the upper-left 2×2 → sc1·sc2
        mc2 = multi_surface_coupling([sc1, sc2, sc3], dp; msing_max=2)
        for Q in (0.0+0im, 0.5+0.5im)
            @test mc2(Q) ≈ sc1(Q) * sc2(Q)
        end

        # msing_max = 3 (default for ≥3 surfaces) uses the full 3×3 → sc1·sc2·sc3
        mc3 = multi_surface_coupling([sc1, sc2, sc3], dp)
        @test mc3.msing_max == 3         # min(3, 3) = 3
        for Q in (0.5+0.5im, 1.5-0.5im)
            @test mc3(Q) ≈ sc1(Q) * sc2(Q) * sc3(Q)
        end
    end

    @testset "Per-surface Q rescaling via tauk_ref / tauk_k" begin
        # Each surface evaluates its inner Δ at Q_k = Q · (tauk_ref/tauk_k).
        # With Δ(Q) = Q (b=1, a=0), the diagonal modification is
        #   M[k,k] = dp_diag_k - scale·Q·(tauk_ref/tauk_k)
        # Verify against an explicit closed form with mismatched tauks.
        sc1 = surface_coupling(LinTestModel(0.0im, 1.0+0im), nothing,
                               0.0+0im; scale=1.0, tauk=2.0)   # ref tauk
        sc2 = surface_coupling(LinTestModel(0.0im, 1.0+0im), nothing,
                               0.0+0im; scale=1.0, tauk=4.0)   # half rate
        dp = ComplexF64[0.0 0.0; 0.0 0.0]
        mc = multi_surface_coupling([sc1, sc2], dp; ref_idx=1)
        for Q in (1.0+0im, 0.5+0.3im)
            # M[1,1] = 0 - Q · (2/2) = -Q
            # M[2,2] = 0 - Q · (2/4) = -Q/2
            # det = M[1,1] · M[2,2] = Q·Q/2 = Q²/2
            @test mc(Q) ≈ Q^2 / 2 rtol = 1e-12
        end

        # Switch ref_idx to surface 2
        mc2 = multi_surface_coupling([sc1, sc2], dp; ref_idx=2)
        for Q in (1.0+0im, 0.5+0.3im)
            # M[1,1] = -Q · (4/2) = -2Q
            # M[2,2] = -Q · (4/4) = -Q
            # det = 2Q · Q = 2Q²
            @test mc2(Q) ≈ 2 * Q^2 rtol = 1e-12
        end
    end

    @testset "SLAYER self-consistency: known coupled root" begin
        # Build a 2-surface SLAYER MultiSurfaceCoupling, evaluate at
        # Q_pin, and back-fill dp_matrix so that det(M(Q_pin)) = 0
        # exactly.
        p_a = _slayer_ref()
        p_b = _slayer_ref()
        m = SLAYERModel()
        sc1 = surface_coupling(m, p_a, 0.0+0im)
        sc2 = surface_coupling(m, p_b, 0.0+0im)

        Q_pin = 0.3 + 0.4im
        ref_tauk = sc1.tauk

        # Compute the diagonal modifications at Q_pin
        Δ1 = solve_inner(m, p_a, Q_pin * (ref_tauk/sc1.tauk)).tearing * sc1.scale
        Δ2 = solve_inner(m, p_b, Q_pin * (ref_tauk/sc2.tauk)).tearing * sc2.scale

        # Build dp such that M(Q_pin) is exactly singular.
        # Choose off-diagonal couplings, then set diagonals so M[k,k]=Δ_k
        # makes the matrix singular by setting M[1,1]·M[2,2] = M[1,2]·M[2,1].
        c12, c21 = 0.05+0im, 0.05+0im
        # Pick M[1,1] arbitrarily, solve for M[2,2]:
        M11 = 0.7 + 0.0im
        M22 = (c12 * c21) / M11
        dp = ComplexF64[M11+Δ1  c12;
                         c21    M22+Δ2]

        mc = multi_surface_coupling([sc1, sc2], dp)
        # The constructed M(Q_pin) is exactly singular by construction
        @test abs(mc(Q_pin)) < 1e-10

        # Off-pin Q gives a non-trivial determinant
        @test abs(mc(Q_pin + 0.05)) > 1e-3
    end

    @testset "GGJ surfaces flow through the coupled API" begin
        p = glasser_wang_2020_eq55()
        sc1 = surface_coupling(GGJModel(solver=:shooting), p, -1.0+0im)
        sc2 = surface_coupling(GGJModel(solver=:shooting), p, -2.0+0im)
        dp = ComplexF64[-1.0 0.1; 0.1 -2.0]
        mc = multi_surface_coupling([sc1, sc2], dp)
        @test mc isa MultiSurfaceCoupling
        @test mc.surfaces[1].tauk == 1.0      # GGJ default
        @test mc(1e-3 + 0.0im) isa ComplexF64
    end

    @testset "Broadcast over a 2D Q grid" begin
        # Coupled residual must be broadcast-compatible for PR 5/6 scans.
        sc1 = surface_coupling(LinTestModel(0.0im, 1.0+0im), nothing,
                               0.0+0im; scale=1.0, tauk=1.0)
        sc2 = surface_coupling(LinTestModel(0.0im, 1.0+0im), nothing,
                               0.0+0im; scale=1.0, tauk=1.0)
        dp = ComplexF64[0.0 0.0; 0.0 0.0]
        mc = multi_surface_coupling([sc1, sc2], dp)

        Q_grid = [(qr + qi*im) for qr in -1.0:0.5:1.0, qi in -1.0:0.5:1.0]
        det_grid = mc.(Q_grid)
        @test size(det_grid) == size(Q_grid)
        @test all(d -> d isa ComplexF64, det_grid)
        # det = Q² with these params; one interior cross-check
        @test det_grid[3, 3] ≈ Q_grid[3, 3]^2
    end
end
