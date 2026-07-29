# runtests_innerlayer.jl
#
# Inner-layer GGJ solver checks on the Glasser & Wang (2020, Phys. Plasmas 27, 012506) Eq. 55
# benchmark equilibrium (D-shaped, aspect ratio 2, q = 2 surface).
#
# SCOPE — what this paper pins: Glasser & Wang (2020) does NOT tabulate an inner-region matching
# Δ(Q) (its "Δ_±" Eq. 54 is a convergence-error norm; its physical outputs are growth rates from
# full resistive DCON). So we test (a) the paper-grounded, platform-independent Mercier index D_I
# and Frobenius matching powers r_± = 3/2 ± √(−D_I) (Eq. 49), pure functions of the coefficients,
# and (b) the Galerkin Δ(Q) at the paper's operating point Q = 0.1234, pinned to a value
# cross-checked against the Fortran rmatch deltac solver (inps basis).
#
# The Δ(Q) cross-check: at Q = 0.1234 the layer is well-conditioned and Δ is purely real; the
# Julia GGJ Galerkin solver and Fortran rmatch deltac agree to ~1e-8 there
# (Δ_odd ≈ 3.698368e4, Δ_even ≈ 14.759721). This is a different, well-behaved operating point than
# the original ill-conditioned pin (Q ≈ 6e5·i) that was order-1 irreproducible across architecture.

const IL = GeneralizedPerturbedEquilibrium.InnerLayer
const GGJ = IL.GGJ

@testset "InnerLayer GGJ (Glasser & Wang 2020, Eq. 55)" begin
    p = IL.glasser_wang_2020_eq55()

    @testset "Mercier index and matching powers (Eq. 49)" begin
        # D_I = E + F + H − 1/4 from the verbatim Eq. 55 coefficients.
        D_I = GGJ.mercier_di(p)
        @test D_I ≈ -0.268361 rtol = 1e-4
        @test D_I < 0                              # Mercier-stable: the inner-layer model's premise

        # p1 = √(−D_I) sets the large-x Frobenius exponents r_± = 3/2 ± √(−D_I) (Eq. 49).
        @test GGJ.p1(p) ≈ sqrt(-D_I) rtol = 1e-12
        @test GGJ.p1(p) ≈ 0.518036 rtol = 1e-4
    end

    @testset "Galerkin Δ(Q) at Q = 0.1234, cross-checked vs Fortran rmatch deltac" begin
        gal = IL.GGJModel(; solver=:galerkin)
        # Eq. 55's companion point is the scaled growth rate Q = 0.1234 (real); build the physical
        # rate γ = Q·Q₀ so inner_Q(p, γ) lands exactly there.
        Q_paper = 0.1234
        γ = Q_paper * GGJ.q0(p)
        @test GGJ.inner_Q(p, γ) ≈ Q_paper rtol = 1e-12
        Δ = IL.solve_inner(gal, p, γ)
        @test all(isfinite, (Δ.tearing, Δ.interchange))
        # Δ is purely real at this real Q; values cross-checked against an independent
        # inner-layer reference — the two codes agree to ~1e-8. rtol absorbs cross-platform jitter.
        # `solve_inner` returns the named-field `InnerLayerResponse`; the interchange branch is
        # the large root, the tearing branch the small one.
        @test real(Δ.interchange) ≈ 3.698368e4 rtol = 1e-3
        @test real(Δ.tearing) ≈ 14.759721 rtol = 1e-3
        @test abs(imag(Δ.interchange)) < 1e-3 * abs(Δ.interchange)
        @test abs(imag(Δ.tearing)) < 1e-3 * abs(Δ.tearing)
    end
end
