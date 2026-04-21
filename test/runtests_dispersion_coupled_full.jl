@testset "Dispersion full 2m×2m coupled determinant (CoupledFull)" begin
    using GeneralizedPerturbedEquilibrium.InnerLayer
    using GeneralizedPerturbedEquilibrium.InnerLayer: InnerLayerModel, InnerLayerResponse, solve_inner
    using GeneralizedPerturbedEquilibrium.Dispersion
    using GeneralizedPerturbedEquilibrium.ForceFreeStates: pest3_decompose, dprime_outer_matrix
    using LinearAlgebra

    # Synthetic inner-layer model with explicit (tearing, interchange)
    # pair — lets us probe both channels independently.
    struct _LinearInner <: InnerLayerModel
        a_t::ComplexF64; b_t::ComplexF64        # tearing:     Δ_t(Q) = a_t + b_t·Q
        a_i::ComplexF64; b_i::ComplexF64        # interchange: Δ_i(Q) = a_i + b_i·Q
    end
    GeneralizedPerturbedEquilibrium.InnerLayer.solve_inner(
        m::_LinearInner, params, Q::Number) =
        InnerLayerResponse(m.a_t + m.b_t*ComplexF64(Q),
                           m.a_i + m.b_i*ComplexF64(Q))

    # --- Synthetic parity-major 2m × 2m outer matrix -----------------
    # Pletzer-Dewar layout: [[A' B'] [Γ' Δ']] with m=2. Values chosen
    # non-Hermitian to confirm CoupledFull doesn't secretly require it.
    A = ComplexF64[ 1.0+0.0im   0.2+0.1im;  0.15-0.05im   1.5+0.0im]
    B = ComplexF64[ 0.10+0.0im  0.05+0.02im; 0.05+0.01im  0.10+0.0im]
    Γ = ComplexF64[ 0.10+0.0im  0.05+0.01im; 0.05+0.02im  0.10+0.0im]
    Δ = ComplexF64[-5.0+0.0im   0.3+0.0im;   0.3+0.0im   -4.0+0.0im]
    dp_full = [A B; Γ Δ]

    @testset "Constructor + dimension validation" begin
        # Pressureless SLAYER-like: interchange channel zero.
        sc1 = surface_coupling(_LinearInner(-1.0+0im, 0+0im, 0+0im, 0+0im),
                                nothing, 0+0im; scale=1.0, tauk=1.0)
        sc2 = surface_coupling(_LinearInner(-0.5+0im, 0+0im, 0+0im, 0+0im),
                                nothing, 0+0im; scale=1.0, tauk=1.0)
        mcf = multi_surface_coupling_full([sc1, sc2], dp_full)
        @test mcf.dp_full === mcf.dp_full    # holds a Matrix copy
        @test size(mcf.dp_full) == (4, 4)
        @test mcf.msing_max == 2
        @test mcf.ref_idx == 1

        # Wrong outer dimension
        @test_throws ArgumentError multi_surface_coupling_full([sc1, sc2], A)   # 2×2 ≠ 4×4
        # Out-of-range ref_idx
        @test_throws ArgumentError multi_surface_coupling_full([sc1, sc2], dp_full; ref_idx=0)
        @test_throws ArgumentError multi_surface_coupling_full([sc1, sc2], dp_full; ref_idx=3)
        # Out-of-range msing_max
        @test_throws ArgumentError multi_surface_coupling_full([sc1, sc2], dp_full; msing_max=0)
        @test_throws ArgumentError multi_surface_coupling_full([sc1, sc2], dp_full; msing_max=3)
    end

    @testset "Pressureless (SLAYER-like) equivalence to m×m MultiSurfaceCoupling" begin
        # When Δ_interchange ≡ 0 on every surface, the 2m×2m determinant
        # factorizes via Schur complement as
        #
        #   det(D' − D_γ) = det(A') · det( (Δ' − Δ_t·I) − Γ'·A'⁻¹·B' )
        #
        # The m×m MultiSurfaceCoupling computes
        #   det( Δ' − Δ_t·I )
        # which is not quite the Schur-complemented form (it ignores the
        # A'/B'/Γ' couplings). But when B'=Γ'=0 (block-diagonal outer),
        # the two must agree up to the det(A') prefactor.
        A_bd = ComplexF64[1.0 0; 0 1.5]        # block-diag outer
        B_bd = zeros(ComplexF64, 2, 2)
        Γ_bd = zeros(ComplexF64, 2, 2)
        Δ_bd = ComplexF64[-5.0 0.3; 0.3 -4.0]
        dp_bd = [A_bd B_bd; Γ_bd Δ_bd]

        # Populate only the tearing channel
        Δ_t_val = -1.2 + 0.1im
        sc1 = surface_coupling(_LinearInner(Δ_t_val, 0+0im, 0+0im, 0+0im),
                                nothing, 0+0im; scale=1.0, tauk=1.0)
        sc2 = surface_coupling(_LinearInner(Δ_t_val, 0+0im, 0+0im, 0+0im),
                                nothing, 0+0im; scale=1.0, tauk=1.0)

        # m×m path
        mc_red  = multi_surface_coupling([sc1, sc2], Δ_bd; msing_max=2)
        det_red = mc_red(0.5 + 0.0im)         # value at some Q

        # 2m×2m path
        mc_full = multi_surface_coupling_full([sc1, sc2], dp_bd)
        det_full = mc_full(0.5 + 0.0im)

        # det_full should equal det(A_bd) · det_red when B=Γ=0.
        det_expected = det(A_bd) * det_red
        @test abs(det_full - det_expected) / abs(det_expected) < 1e-12
    end

    @testset "Full coupling: Schur-complement identity" begin
        # For general (A,B,Γ,Δ) and arbitrary (Δ_t, Δ_i), the CoupledFull
        # determinant must match the Schur formula
        #   det(D' − D_γ) = det(X) · det(Y − Γ·X⁻¹·B)
        # with X = A' − Δ_i·I, Y = Δ' − Δ_t·I.
        Δ_t_val = -1.2 + 0.1im
        Δ_i_val =  0.5 - 0.2im
        sc1 = surface_coupling(_LinearInner(Δ_t_val, 0+0im, Δ_i_val, 0+0im),
                                nothing, 0+0im; scale=1.0, tauk=1.0)
        sc2 = surface_coupling(_LinearInner(Δ_t_val, 0+0im, Δ_i_val, 0+0im),
                                nothing, 0+0im; scale=1.0, tauk=1.0)
        mcf = multi_surface_coupling_full([sc1, sc2], dp_full)
        det_full = mcf(0.0 + 0.0im)

        X = A - Δ_i_val * I(2)
        Y = Δ - Δ_t_val * I(2)
        det_expected = det(X) * det(Y - Γ * inv(X) * B)
        @test abs(det_full - det_expected) / abs(det_expected) < 1e-12
    end

    @testset "Q rescaling via tauk_ref / tauk_k" begin
        # Independent tauks on the two surfaces should rescale the inner
        # Δ arguments by tauk_ref / tauk_k.
        Δ_t_val = -2.0 + 0.0im
        sc1 = surface_coupling(_LinearInner(0+0im, 1+0im, 0+0im, 0+0im),
                                nothing, 0+0im; scale=1.0, tauk=1.0)     # Δ_t(Q) = Q
        sc2 = surface_coupling(_LinearInner(0+0im, 1+0im, 0+0im, 0+0im),
                                nothing, 0+0im; scale=1.0, tauk=2.0)     # Δ_t(Q') = Q' = Q·(1/2)

        # At Q_pin = 2.0, surface 1 sees Δ_t = 2, surface 2 sees Δ_t = 1.
        Q_pin = 2.0 + 0.0im
        mcf = multi_surface_coupling_full([sc1, sc2], dp_full)
        det_mcf = mcf(Q_pin)

        # Hand-computed expected: D_γ = diag(0, 0, 2, 1) (interchange=0, tearing=2 at s1 and 1 at s2)
        Δ_γ = ComplexF64[0 0 0 0; 0 0 0 0; 0 0 2 0; 0 0 0 1]
        det_expected = det(dp_full - Δ_γ)
        @test abs(det_mcf - det_expected) / abs(det_expected) < 1e-12
    end

    @testset "Interchange channel is physically active" begin
        # Confirm the upper-left block actually gets Δ_interchange subtracted
        # by seeing that det changes when Δ_i goes from 0 to nonzero.
        sc_no_i  = surface_coupling(_LinearInner(-1.2+0.1im, 0+0im, 0+0im, 0+0im),
                                     nothing, 0+0im; scale=1.0, tauk=1.0)
        sc_with_i = surface_coupling(_LinearInner(-1.2+0.1im, 0+0im, 0.5-0.2im, 0+0im),
                                     nothing, 0+0im; scale=1.0, tauk=1.0)
        mc0 = multi_surface_coupling_full([sc_no_i, sc_no_i], dp_full)
        mc1 = multi_surface_coupling_full([sc_with_i, sc_with_i], dp_full)
        @test mc0(0+0im) ≠ mc1(0+0im)
    end

    @testset "dprime_outer_matrix round-trip: CoupledFull ↔ pest3_decompose" begin
        # Build a random-ish side-major dp_raw, rotate to parity-major via
        # dprime_outer_matrix, and confirm CoupledFull consumes it correctly.
        # Reusing the Fortran-matched RR−RL−LR+LL identities this exercises
        # the full end-to-end plumbing from Riccati.jl output → Dispersion.
        # Use a distinct local name (dp_rot) to avoid rebinding the outer
        # @testset's dp_full (Julia @testset does not isolate variable
        # bindings from the enclosing scope).
        dp_raw = ComplexF64[
            1.0   0.5   0.3   0.1 ;
            0.2   3.0   0.1   0.2 ;
            0.1   0.2  -2.0   0.4 ;
            0.05  0.15  0.3   1.0]
        dp_rot = dprime_outer_matrix(dp_raw)

        # The (A,B,Γ,Δ) blocks recovered from pest3_decompose must satisfy
        # dprime_outer_matrix == [A B; Γ Δ].
        blocks = pest3_decompose(dp_raw)
        @test dp_rot[1:2, 1:2] == blocks.A
        @test dp_rot[1:2, 3:4] == blocks.B
        @test dp_rot[3:4, 1:2] == blocks.Γ
        @test dp_rot[3:4, 3:4] == blocks.Δ

        # Build a CoupledFull on it and confirm it evaluates finite.
        sc1 = surface_coupling(_LinearInner(-0.5+0im, 0+0im, 0.1+0im, 0+0im),
                                nothing, 0+0im; scale=1.0, tauk=1.0)
        sc2 = surface_coupling(_LinearInner(-0.5+0im, 0+0im, 0.1+0im, 0+0im),
                                nothing, 0+0im; scale=1.0, tauk=1.0)
        mcf = multi_surface_coupling_full([sc1, sc2], dp_rot)
        @test isfinite(real(mcf(0.3+0.1im)))
        @test isfinite(imag(mcf(0.3+0.1im)))
    end

    @testset "msing_max truncation preserves parity-block structure" begin
        # With msing_max=1, CoupledFull must use the 2×2 parity-symmetric
        # sub-matrix [[A[1,1] B[1,1]] [Γ[1,1] Δ[1,1]]] — not just the
        # upper-left 2×2 of the original 4×4 dp_full.
        sc1 = surface_coupling(_LinearInner(0+0im, 0+0im, 0+0im, 0+0im),
                                nothing, 0+0im; scale=1.0, tauk=1.0)     # Δ ≡ 0
        sc2 = surface_coupling(_LinearInner(0+0im, 0+0im, 0+0im, 0+0im),
                                nothing, 0+0im; scale=1.0, tauk=1.0)
        mcf = multi_surface_coupling_full([sc1, sc2], dp_full; msing_max=1)
        expected = det(ComplexF64[A[1,1] B[1,1]; Γ[1,1] Δ[1,1]])
        @test abs(mcf(0+0im) - expected) < 1e-12
    end
end
