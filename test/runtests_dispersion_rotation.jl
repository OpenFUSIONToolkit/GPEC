@testset "Dispersion per-surface rotation shift" begin
    using GeneralizedPerturbedEquilibrium.InnerLayer
    using GeneralizedPerturbedEquilibrium.InnerLayer: InnerLayerModel, solve_inner
    using GeneralizedPerturbedEquilibrium.Dispersion
    using GeneralizedPerturbedEquilibrium.Tearing.Runner: _q_shift, _build_surface_coupling
    using LinearAlgebra

    # Linear inner layer Δ(Q) = a + b·Q makes the applied Q offset readable
    # straight off the residual.
    struct RotTestModel <: InnerLayerModel
        a::ComplexF64
        b::ComplexF64
    end
    GeneralizedPerturbedEquilibrium.InnerLayer.solve_inner(
        m::RotTestModel, params, Q::Number) =
        InnerLayerResponse(m.a + m.b * ComplexF64(Q), zero(ComplexF64))

    model = RotTestModel(0.0im, 1.0 + 0im)

    @testset "q_shift offsets the layer Q argument on the scalar residual" begin
        shift = 0.75
        sc0 = surface_coupling(model, nothing, 1.0 + 0im; scale=1.0, tauk=1.0)
        scs = surface_coupling(model, nothing, 1.0 + 0im; scale=1.0, tauk=1.0,
            q_shift=shift)
        @test scs.q_shift == shift
        for Q in (0.0 + 0im, 2.0 - 1.0im, -3.5 + 0.25im)
            @test scs(Q) ≈ sc0(Q + shift)
        end
    end

    @testset "Coupled determinant applies each surface's own shift" begin
        # Diagonal Δ', so det = Π_k (dp_kk - Δ_k(Q·tauk_k/tauk_ref + shift_k)).
        s1, s2 = 0.5, -1.25
        sc1 = surface_coupling(model, nothing, 1.0 + 0im; scale=1.0, tauk=1.0,
            q_shift=s1)
        sc2 = surface_coupling(model, nothing, 2.0 + 0im; scale=1.0, tauk=2.0,
            q_shift=s2)
        dp = ComplexF64[1.0 0.0; 0.0 2.0]
        mc = multi_surface_coupling([sc1, sc2], dp)

        Q = 1.5 + 0.5im
        expected = (1.0 - (Q * (1.0 / 1.0) + s1)) * (2.0 - (Q * (2.0 / 1.0) + s2))
        @test mc(Q) ≈ expected
    end

    @testset "Real SLAYER surfaces: the lab-frame root moves with the E×B rotation" begin
        # Q_layer = τ_k·ω_lab − τ_k·n·Ω_E, so the rotating determinant at Q + τ_ref·n·Ω_E equals the
        # static one at Q: every root keeps its γ and moves by ω_lab(Ω_E) − ω_lab(0) = +n·Ω_E.
        mk(; qval, rs, m) = slayer_parameters(; n_e=5.0e19, t_e=1000.0, t_i=1000.0,
            omega_e=1.0e4, omega_i=5.0e3,
            qval=qval, sval_r=1.0, bt=2.0, rs=rs, R0=1.7,
            mu_i=2.0, zeff=1.0, chi_perp=1.0, chi_tor=1.0, m=m, n=2)
        params = [mk(; qval=1.5, rs=0.5, m=3), mk(; qval=2.0, rs=0.6, m=4)]
        slayer = SLAYERModel(; variant=:fitzpatrick)
        dp = ComplexF64[-2.0 0.5; 0.5 -3.0]
        lab_shift = 0.5                                        # τ_ref·n·Ω_E in the reference Q
        Ω_rigid = lab_shift / (params[1].tauk * params[1].n)   # rad/s per unit n, on both surfaces
        build(Ω) = [_build_surface_coupling(slayer, params[k], dp[k, k], _q_shift(params[k], Ω[k])) for k in 1:2]
        static = multi_surface_coupling(build(zeros(2)), dp; ref_idx=1, msing_max=2)
        rotating = multi_surface_coupling(build(fill(Ω_rigid, 2)), dp; ref_idx=1, msing_max=2)
        for Q in (0.3 + 0.2im, -0.4 + 0.6im, 0.1 + 1.1im)
            # The layer arguments agree to rounding, which the adaptive layer ODE amplifies to ≲ 3e-6.
            @test rotating(Q + lab_shift) ≈ static(Q) rtol = 1e-4
            # The opposite Doppler sign (mode counter-rotating with the plasma) must not match.
            @test !isapprox(rotating(Q - lab_shift), static(Q); rtol=1e-3)
        end
    end

    @testset "Coupled roots do not depend on the reference surface" begin
        # Diagonal Δ′ decouples the surfaces, so each coupled root must be that surface's own
        # root in physical units whichever surface normalizes Q.
        d = ComplexF64[1.0+1.0im, 3.0+2.0im]
        tauk = [2.0e-4, 5.0e-4]
        scs = [surface_coupling(model, nothing, d[k]; scale=1.0, tauk=tauk[k]) for k in 1:2]
        # Δ(Q) = Q makes surface k's own root Q_k = d_k, i.e. ω + iγ = d_k/τ_k.
        expected = sort(d ./ tauk; by=real)
        for ref in 1:2
            mc = multi_surface_coupling(scs, diagm(d); ref_idx=ref, msing_max=2)
            # det is quadratic in Q: recover it from three samples and solve.
            xs = ComplexF64[0, 1, 2]
            c = [x^j for x in xs, j in 0:2] \ [mc(x) for x in xs]
            disc = sqrt(c[2]^2 - 4c[3] * c[1])
            roots = [(-c[2] + disc) / (2c[3]), (-c[2] - disc) / (2c[3])]
            @test sort(roots ./ tauk[ref]; by=real) ≈ expected rtol = 1e-10
        end
    end
end
