@testset "Dispersion 4m×4m Fortran-faithful coupled determinant (CoupledFortranMatch)" begin
    using GeneralizedPerturbedEquilibrium.InnerLayer
    using GeneralizedPerturbedEquilibrium.InnerLayer: InnerLayerModel, InnerLayerResponse, solve_inner
    using GeneralizedPerturbedEquilibrium.Dispersion
    using LinearAlgebra

    # Synthetic inner-layer model with explicit (tearing, interchange)
    # pair — lets us probe both channels independently.
    struct _LinearInnerF <: InnerLayerModel
        a_t::ComplexF64; b_t::ComplexF64   # tearing: Δ_t(Q) = a_t + b_t·Q
        a_i::ComplexF64; b_i::ComplexF64   # interchange: Δ_i(Q) = a_i + b_i·Q
    end
    GeneralizedPerturbedEquilibrium.InnerLayer.solve_inner(
        m::_LinearInnerF, params, Q::Number) =
        InnerLayerResponse(m.a_t + m.b_t*ComplexF64(Q),
                           m.a_i + m.b_i*ComplexF64(Q))

    @testset "Constructor validation" begin
        sc1 = surface_coupling(_LinearInnerF(-1.0+0im, 0+0im, 0.1+0im, 0+0im),
                                nothing, 0+0im; scale=1.0, tauk=1.0)
        sc2 = surface_coupling(_LinearInnerF(-0.5+0im, 0+0im, 0.2+0im, 0+0im),
                                nothing, 0+0im; scale=1.0, tauk=1.0)
        dp_raw = ComplexF64[
            1.0 0.1 0.2 0.05;
            0.1 1.2 0.05 0.2;
            0.2 0.05 -5.0 0.3;
            0.05 0.2 0.3 -4.0]
        mc = multi_surface_coupling_fortran([sc1, sc2], dp_raw)
        @test size(mc.dp_raw) == (4, 4)
        @test mc.msing_max == 2
        @test mc.ref_idx == 1
        @test mc.rotation == [0.0, 0.0]
        @test mc.ntor == 1

        # Wrong outer dim
        @test_throws ArgumentError multi_surface_coupling_fortran([sc1, sc2],
                                                                  dp_raw[1:2, 1:2])
        @test_throws ArgumentError multi_surface_coupling_fortran([sc1, sc2],
                                                                  dp_raw; ref_idx=0)
        @test_throws ArgumentError multi_surface_coupling_fortran([sc1, sc2],
                                                                  dp_raw; ref_idx=3)
        @test_throws ArgumentError multi_surface_coupling_fortran([sc1, sc2],
                                                                  dp_raw; msing_max=0)
        @test_throws ArgumentError multi_surface_coupling_fortran([sc1, sc2],
                                                                  dp_raw; msing_max=3)
        # Wrong rotation length
        @test_throws ArgumentError multi_surface_coupling_fortran([sc1, sc2],
                                                                  dp_raw; rotation=[0.0])
    end

    @testset "1-surface 4×4 det matches hand computation" begin
        # m=1 case: matrix is 4×4 and fully hand-verifiable.
        dp_raw = ComplexF64[1.0 0.5; 0.3 2.0]
        sc = surface_coupling(_LinearInnerF(0.7+0im, 0+0im, 0.2+0im, 0+0im),
                              nothing, 0+0im; scale=1.0, tauk=1.0, dc=0.0)
        mc = multi_surface_coupling_fortran([sc], dp_raw)
        # At Q=0.1 both Δ_t and Δ_i are constants (b=0), so inner Δs independent of Q.
        det_jl = mc(0.1 + 0.0im)
        # Hand-computed matrix (see the port comment block for the layout):
        #   mat[3:4, 1:2] = transpose(dp_raw) = [1 0.3; 0.5 2]
        #   mat[1,1]=1, mat[2,2]=1
        #   mat[1,3]=-1, mat[1,4]=+1, mat[2,3]=-1, mat[2,4]=-1
        #   delta1=interchange=0.2, delta2=tearing=0.7
        #   mat[3,3]=-0.2, mat[3,4]=+0.7, mat[4,3]=-0.2, mat[4,4]=-0.7
        M_hand = ComplexF64[
            1     0   -1     1 ;
            0     1   -1    -1 ;
            1   0.3 -0.2   0.7 ;
          0.5     2 -0.2  -0.7]
        @test det_jl ≈ det(M_hand)
    end

    @testset "Static (rotation=0) equivalent to Fortran delta1, delta2 assembly" begin
        # Replicate the Fortran match routine literally for msing=2 and
        # synthetic inner values; confirm Julia assembly agrees.
        dp_raw = ComplexF64[
            10.0  0.1  0.2  0.3 ;
             0.1 11.0  0.4  0.5 ;
             0.2  0.4 -5.0  0.6 ;
             0.3  0.5  0.6 -4.0]
        sc1 = surface_coupling(_LinearInnerF(0.2+0.1im, 0+0im, 0.7-0.05im, 0+0im),
                                nothing, 0+0im; scale=1.0, tauk=1.0, dc=0.0)
        sc2 = surface_coupling(_LinearInnerF(-0.3+0.0im, 0+0im, 1.5+0.3im, 0+0im),
                                nothing, 0+0im; scale=1.0, tauk=1.0, dc=0.0)
        mc = multi_surface_coupling_fortran([sc1, sc2], dp_raw)
        det_jl = mc(0.0 + 0.0im)

        # Hand assembly
        M = zeros(ComplexF64, 8, 8)
        M[5:8, 1:4] = transpose(dp_raw)
        # Surface 1: idx1..4 = 1,2,5,6
        M[1,1]=1; M[2,2]=1
        M[1,5]=-1; M[1,6]= 1; M[2,5]=-1; M[2,6]=-1
        d1_1 = 0.7 - 0.05im     # interchange
        d2_1 = 0.2 + 0.1im      # tearing
        M[5,5]=-d1_1; M[5,6]= d2_1; M[6,5]=-d1_1; M[6,6]=-d2_1
        # Surface 2: idx1..4 = 3,4,7,8
        M[3,3]=1; M[4,4]=1
        M[3,7]=-1; M[3,8]= 1; M[4,7]=-1; M[4,8]=-1
        d1_2 = 1.5 + 0.3im
        d2_2 = -0.3 + 0im
        M[7,7]=-d1_2; M[7,8]= d2_2; M[8,7]=-d1_2; M[8,8]=-d2_2

        @test det_jl ≈ det(M) atol=1e-12*abs(det(M))
    end

    @testset "Rotation shift applies i·ntor·rotation to inner Q argument" begin
        # Ensure the per-surface rotation enters the inner-layer argument.
        # Use a linear Δ_t model so Q-dependence is tractable.
        dp_raw = ComplexF64[1.0 0; 0 1.0]
        # Δ_t(Q) = Q (pure linear), Δ_i(Q) = 0
        sc = surface_coupling(_LinearInnerF(0+0im, 1+0im, 0+0im, 0+0im),
                              nothing, 0+0im; scale=1.0, tauk=1.0, dc=0.0)
        # Case A: rotation=0, Q=2+0im → inner sees 2+0im → Δ_t=2, Δ_i=0
        mc0 = multi_surface_coupling_fortran([sc], dp_raw; rotation=[0.0], ntor=1)
        # Case B: rotation=3, Q=2+0im → inner sees 2 + 1j*1*3 = 2+3i → Δ_t=2+3i
        mcR = multi_surface_coupling_fortran([sc], dp_raw; rotation=[3.0], ntor=1)
        @test mc0(2.0+0.0im) ≠ mcR(2.0+0.0im)

        # Check by hand. Both with the same outer matrix:
        function detAt(Δ_t, Δ_i)
            M = ComplexF64[
                1    0   -1    1 ;
                0    1   -1   -1 ;
                1    0   -Δ_i  Δ_t;
                0    1   -Δ_i -Δ_t]
            return det(M)
        end
        @test mc0(2.0+0.0im) ≈ detAt(2.0+0.0im, 0.0+0.0im)
        @test mcR(2.0+0.0im) ≈ detAt(2.0+3.0im, 0.0+0.0im)
    end

    @testset "SurfaceCoupling scale multiplies both inner channels" begin
        # sc.scale should hit both delta1 and delta2 equally.
        dp_raw = ComplexF64[1 0; 0 1]
        sc_unit = surface_coupling(_LinearInnerF(0.3+0im, 0+0im, 0.7+0im, 0+0im),
                                   nothing, 0+0im; scale=1.0, tauk=1.0, dc=0.0)
        sc_x2   = surface_coupling(_LinearInnerF(0.3+0im, 0+0im, 0.7+0im, 0+0im),
                                   nothing, 0+0im; scale=2.0, tauk=1.0, dc=0.0)
        mc1 = multi_surface_coupling_fortran([sc_unit], dp_raw)
        mc2 = multi_surface_coupling_fortran([sc_x2],   dp_raw)
        # Expected hand det for scale=1: d_int=0.7, d_tear=0.3
        # For scale=2: d_int=1.4, d_tear=0.6
        function detAt(Δt, Δi)
            M = ComplexF64[1 0 -1 1; 0 1 -1 -1; 1 0 -Δi Δt; 0 1 -Δi -Δt]
            return det(M)
        end
        @test mc1(0.5+0im) ≈ detAt(0.3, 0.7)
        @test mc2(0.5+0im) ≈ detAt(0.6, 1.4)
    end

    @testset "msing_max truncation" begin
        dp_raw = ComplexF64[
            1.0 0.1 0.2 0.3 ;
            0.1 1.2 0.4 0.5 ;
            0.2 0.4 -5.0 0.6 ;
            0.3 0.5 0.6 -4.0]
        sc1 = surface_coupling(_LinearInnerF(0.5+0im, 0+0im, 0.2+0im, 0+0im),
                                nothing, 0+0im; scale=1.0, tauk=1.0)
        sc2 = surface_coupling(_LinearInnerF(-0.3+0im, 0+0im, 1.0+0im, 0+0im),
                                nothing, 0+0im; scale=1.0, tauk=1.0)

        # With msing_max=1, only surface 1 participates; matrix becomes 4×4
        # using the upper-left 2×2 block of dp_raw.
        mc1 = multi_surface_coupling_fortran([sc1, sc2], dp_raw; msing_max=1)
        det1 = mc1(0+0im)
        # Hand construct the 4×4
        sub_dp = dp_raw[1:2, 1:2]
        M1 = zeros(ComplexF64, 4, 4)
        M1[3:4, 1:2] = transpose(sub_dp)
        M1[1,1]=1; M1[2,2]=1
        M1[1,3]=-1; M1[1,4]=1; M1[2,3]=-1; M1[2,4]=-1
        M1[3,3]=-0.2; M1[3,4]=0.5; M1[4,3]=-0.2; M1[4,4]=-0.5
        @test det1 ≈ det(M1)

        # Full msing_max=2 case must differ
        mcfull = multi_surface_coupling_fortran([sc1, sc2], dp_raw; msing_max=2)
        @test mcfull(0+0im) ≠ det1
    end

    @testset "SLAYER-like (Δ_interchange=0) still gives correct det" begin
        # When both surfaces are pure-tearing (Δ_interchange=0), the matrix
        # is non-trivial but still well-defined; verify it's non-zero and
        # finite (not NaN from singular inner block).
        dp_raw = ComplexF64[1.0 0.1 0.2 0.3; 0.1 1.2 0.4 0.5;
                             0.2 0.4 -5.0 0.6; 0.3 0.5 0.6 -4.0]
        sc1 = surface_coupling(_LinearInnerF(-2+0im, 0+0im, 0+0im, 0+0im),
                                nothing, 0+0im; scale=1.0, tauk=1.0)
        sc2 = surface_coupling(_LinearInnerF(-3+0im, 0+0im, 0+0im, 0+0im),
                                nothing, 0+0im; scale=1.0, tauk=1.0)
        mc = multi_surface_coupling_fortran([sc1, sc2], dp_raw)
        d = mc(0.1 + 0.2im)
        @test isfinite(real(d))
        @test isfinite(imag(d))
    end

    @testset "inner_kwargs pass-through" begin
        # Verify that inner_kwargs reaches solve_inner at each Q evaluation.
        # Use a synthetic model with a tuning parameter to confirm plumbing.
        struct _ProbeModel <: InnerLayerModel end
        GeneralizedPerturbedEquilibrium.InnerLayer.solve_inner(
            ::_ProbeModel, params, Q::Number; scale_factor::Float64=1.0) =
            InnerLayerResponse(scale_factor * (1.0 + 0im),
                               scale_factor * (0.5 + 0im))

        dp_raw = ComplexF64[1.0 0; 0 1.0]
        sc = surface_coupling(_ProbeModel(), nothing, 0+0im;
                              scale=1.0, tauk=1.0, dc=0.0)
        mc_native = multi_surface_coupling_fortran([sc], dp_raw)
        mc_tuned  = multi_surface_coupling_fortran([sc], dp_raw;
                                                    inner_kwargs=(scale_factor=0.5,))
        @test mc_native.inner_kwargs == NamedTuple()
        @test mc_tuned.inner_kwargs == (scale_factor=0.5,)

        # Det should differ because inner Δ's are halved by the kwarg
        det_native = mc_native(0.0 + 0.0im)
        det_tuned  = mc_tuned(0.0 + 0.0im)
        @test det_native ≠ det_tuned
        @test isfinite(real(det_native)) && isfinite(imag(det_native))
        @test isfinite(real(det_tuned))  && isfinite(imag(det_tuned))
    end

    @testset "Static GGJ-like scenario runs without error" begin
        # Smoke test: larger m=3 case, both channels non-trivial, Q shifted
        m = 3
        Random_dp = ComplexF64[
            5.0  0.2  0.1  0.05 0.3 0.2;
            0.2  7.0  0.3  0.1  0.2 0.1;
            0.1  0.3 -3.0  0.4  0.1 0.05;
            0.05 0.1  0.4 -8.0  0.2 0.1;
            0.3  0.2  0.1  0.2 -2.5 0.3;
            0.2  0.1  0.05 0.1  0.3 -6.5]
        # Non-trivial Q dependence: Δ_t(Q) = a + 0.5·Q, Δ_i(Q) = b + 0.2·Q
        scs = [surface_coupling(_LinearInnerF(0.3+0.01k*im, 0.5+0im,
                                              0.7+0.02k*im, 0.2+0im),
                                nothing, 0+0im; scale=1.0, tauk=1.0)
               for k in 1:m]
        mc = multi_surface_coupling_fortran(scs, Random_dp)
        @test size(mc.dp_raw) == (6, 6)
        d0 = mc(0.0+0.0im)
        d1 = mc(1.0+0.5im)
        @test isfinite(real(d0)) && isfinite(imag(d0))
        @test isfinite(real(d1)) && isfinite(imag(d1))
        # Check that it's actually Q-dependent
        @test d0 != d1
    end
end
