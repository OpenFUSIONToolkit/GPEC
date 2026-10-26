@testset "Dispersion GGJ growth rates (full matching, winding classification)" begin
    using GeneralizedPerturbedEquilibrium.InnerLayer
    using GeneralizedPerturbedEquilibrium.InnerLayer: GGJModel, solve_inner
    using GeneralizedPerturbedEquilibrium.Dispersion
    GGJ = GeneralizedPerturbedEquilibrium.InnerLayer.GGJ

    # Glasser & Wang (2020) Eq. 55 layer, one surface. The 2 × 2 raw Δ' block is
    # built in the parity basis P = M⁻¹·Dᵀ·M (M = [1 −1; 1 1]) with P₁₂ = P₂₁ = 0,
    # for which the 4 × 4 determinant vanishes where Δ_tear(s) = P₂₂. Setting
    # P₂₂ = Δ_tear(s₀) puts a root at the real growth rate s₀.
    p = GGJ.glasser_wang_2020_eq55()
    Q0 = GGJ.q0(p)
    model = GGJModel(; solver=:ray)
    s0 = 0.1234 * Q0
    r0 = solve_inner(model, p, s0)
    P = ComplexF64[real(r0.interchange)/2 0; 0 r0.tearing]
    M = ComplexF64[1 -1; 1 1]
    dp_raw = Matrix(transpose(M * P / M))
    mc = multi_surface_coupling_full([surface_coupling(model, p, r0.tearing)], dp_raw)

    @testset "GGJDispersion: plane convention s = −i·Q·Q₀" begin
        g = GGJDispersion(mc)
        @test g.Q0 ≈ Q0
        Q = 0.03 + 0.11im
        @test g(Q) == mc(-im * Q * Q0)
        @test abs(g(0.1234im)) < 1e-8 * abs(g(0.03 + 0.11im))     # the constructed root
        @test_throws ArgumentError GGJDispersion(
            multi_surface_coupling_full(
                [surface_coupling(GeneralizedPerturbedEquilibrium.InnerLayer.SLAYERModel(), nothing, 0.0im; scale=1.0, tauk=1.0)], dp_raw)
        )
    end

    @testset "ggj_growth_rates finds the constructed root, winding +1" begin
        gr, amr = ggj_growth_rates(mc; gamma_range=(0.08Q0, 0.17Q0), omega_range=(-0.031Q0, 0.047Q0),
            nre0=6, nim0=6, passes=3, parallel=false)
        @test gr.classification === :winding
        @test !(:no_root in gr.warning_flags)
        @test all(w -> w ≈ 1, gr.valid_roots_winding)
        @test gr.gamma_Hz ≈ s0 rtol = 1e-6
        @test abs(gr.omega_Hz) < 1e-6 * Q0
        @test abs(gr.Q_root - 0.1234im) < 1e-6
    end
end
