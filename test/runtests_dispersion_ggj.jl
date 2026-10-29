struct GGJDomainTestResidual{G,F}
    dispersion::G
    residual::F
    evaluated::Vector{ComplexF64}
end

function (f::GGJDomainTestResidual)(Q)
    push!(f.evaluated, ComplexF64(Q))
    return f.residual(Q)
end

GeneralizedPerturbedEquilibrium.Tearing.Dispersion._residual_domain_radius(f::GGJDomainTestResidual, Q::ComplexF64) =
    GeneralizedPerturbedEquilibrium.Tearing.Dispersion._residual_domain_radius(f.dispersion, Q)

@testset "Dispersion GGJ growth rates (full matching, winding classification)" begin
    using GeneralizedPerturbedEquilibrium.InnerLayer
    using GeneralizedPerturbedEquilibrium.InnerLayer: GGJModel, solve_inner
    using GeneralizedPerturbedEquilibrium.Tearing.Dispersion
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

    @testset "GGJ classification and polishing stay at positive growth" begin
        D = GeneralizedPerturbedEquilibrium.Tearing.Dispersion
        g = GGJDispersion(mc)
        center = 0.01im
        re_paths = [ComplexF64[-1 + center, 1 + center]]
        im_paths = [ComplexF64[0.001im, 0.02im]]
        # A coarse contour estimate must not pull a stable root into a positive-growth search.
        f = GGJDomainTestResidual(g, Q -> Q + 0.1im, ComplexF64[])
        gr = D._run_analysis(re_paths, im_paths, [zeros(2)], 1.0;
            pole_threshold=10.0, filter_above_poles=false, filter_outside_re=false,
            residual=f, classification=:winding, zero_winding=1)
        @test :no_root in gr.warning_flags
        @test isempty(gr.valid_roots)
        @test !isempty(f.evaluated) && all(Q -> imag(Q) > 0, f.evaluated)

        # Expansion may find a nearby unstable root, but its returned trust disk must remain in-domain.
        nearby = GGJDomainTestResidual(g, Q -> Q - 0.014im, ComplexF64[])
        kind, W, radius = D._winding_classify(nearby, ComplexF64[center], 1, re_paths, im_paths, 2.0;
            zero_winding=1, rfrac=0.001)
        @test kind === :root
        @test W ≈ 1
        @test 0 < radius < imag(center)
        root, _, _ = D._polish_root(nearby, center, radius)
        @test root ≈ 0.014im atol = 1e-10
        @test all(Q -> imag(Q) > 0, nearby.evaluated)

        # Even an oversized requested trust region cannot move the iterate or its probes below γ = 0.
        empty!(f.evaluated)
        root, _, _ = D._polish_root(f, center, 1.0)
        @test imag(root) > 0
        @test !isempty(f.evaluated) && all(Q -> imag(Q) > 0, f.evaluated)
        empty!(f.evaluated)
        @test D._polish_root(f, 0.0im, 1.0) == (0.0im, 0, false)
        kind, _, _ = D._winding_classify(f, ComplexF64[0.0im], 1, re_paths, im_paths, 2.0;
            zero_winding=1, rfrac=0.5)
        @test kind === :unresolved
        @test isempty(f.evaluated)
    end

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
        @test gr.gamma ≈ s0 rtol = 1e-6
        @test abs(gr.omega) < 1e-6 * Q0
        @test abs(gr.Q_root - 0.1234im) < 1e-6
        # The GGJ layer's branch cut runs along γ ≤ 0, so a box reaching there is refused.
        @test_throws ArgumentError ggj_growth_rates(mc; gamma_range=(-0.01Q0, 0.17Q0), omega_range=(-0.031Q0, 0.047Q0))
        @test_throws ArgumentError ggj_growth_rates(mc; gamma_range=(0.08Q0, 0.17Q0), omega_range=(0.047Q0, -0.031Q0))
    end

    @testset "two coupled surfaces: the root agrees with a direct solve of the 8 × 8 determinant" begin
        # A second surface with its own constructed root far outside the box, and a symmetric coupling of the two
        # surfaces' tearing channels (the off-diagonal block c·[1 −1; −1 1] reaches d₋ through the matching identity),
        # which moves surface 1's root off its uncoupled value.
        p2 = GGJ.q4_surface_benchmark()
        r02 = solve_inner(model, p2, 0.5 * Q0)
        blk(r) = Matrix(transpose(M * ComplexF64[real(r.interchange)/2 0; 0 r.tearing] / M))
        B = 0.1 * abs(r0.tearing) * ComplexF64[1 -1; -1 1]
        dp2 = [blk(r0) B; transpose(B) blk(r02)]
        mc2 = multi_surface_coupling_full([surface_coupling(model, p, r0.tearing), surface_coupling(model, p2, r02.tearing)], dp2)
        g2 = GGJDispersion(mc2)
        # Independent reference: a complex secant iteration on the determinant, started at the uncoupled root.
        a, b = 0.1234im, 0.1234im * 1.01
        fa, fb = g2(a), g2(b)
        for _ in 1:40
            a, fa, b = b, fb, b - fb * (b - a) / (fb - fa)
            fb = g2(b)
            abs(b - a) < 1e-14 && break
        end
        @test abs(b - 0.1234im) > 1e-5                 # the coupling moved the root
        gr2, _ = ggj_growth_rates(mc2; gamma_range=(0.08Q0, 0.17Q0), omega_range=(-0.031Q0, 0.047Q0),
            nre0=6, nim0=6, passes=3, parallel=false)
        @test length(gr2.valid_roots) == 1 && all(w -> w ≈ 1, gr2.valid_roots_winding)
        @test abs(gr2.Q_root - b) < 1e-8 * abs(b)
    end
end
