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

@testset "InnerLayer GGJ :ray backend (rotated-contour collocation)" begin
    # The method was validated before landing here: manufactured Δ* to 3e-14,
    # Fortran rmatch pins, 96-equilibrium robustness scans to Q = 500i. These
    # tests pin the implementation, not the method.
    p = IL.glasser_wang_2020_eq55()

    @testset "agrees with :galerkin at the paper point Q = 0.1234" begin
        γ = 0.1234 * GGJ.q0(p)
        Δ = IL.solve_inner(IL.GGJModel(; solver=:ray), p, γ)
        # Same Fortran-cross-checked pins as the Galerkin testset above.
        @test real(Δ.interchange) ≈ 3.698368e4 rtol = 1e-3
        @test real(Δ.tearing) ≈ 14.759721 rtol = 1e-3
        @test abs(imag(Δ.interchange)) < 1e-3 * abs(Δ.interchange)
        @test abs(imag(Δ.tearing)) < 1e-3 * abs(Δ.tearing)
        # :ray is the default backend.
        @test IL.GGJModel() === IL.GGJModel{:ray}()
    end

    @testset "q4 physical benchmark at Q = 500i (regime beyond :galerkin)" begin
        q4 = IL.q4_surface_benchmark()
        γ = 500.0im * GGJ.q0(q4)
        Δ = IL.solve_inner(IL.GGJModel(), q4, γ)
        # Pins from the pre-port validation suite (post extended-precision
        # march fix; S-invariant to 3e-4 / 7e-9 and θ-stable there).
        @test Δ.interchange ≈ 2.4720608737 + 13.354123514im rtol = 1e-4
        @test Δ.tearing ≈ 0.13749694953 + 0.74275468725im rtol = 1e-4

        # Δ is an analytic invariant of the contour angle: the outward
        # θ-check drift is a direct numerical error measurement. `solve_ray`
        # returns the raw pair (Δ₁, Δ₂) = (interchange, tearing).
        Q = GGJ.inner_Q(q4, γ)
        r2 = IL.solve_ray(q4, Q; θ=1.2 * angle(Q) / 4)
        @test abs(r2.Δ[2] - Δ.tearing) / abs(Δ.tearing) < 1e-5
        @test abs(r2.Δ[1] - Δ.interchange) / abs(Δ.interchange) < 1e-3
    end
end

@testset "InnerLayer GGJ :ray internal machinery" begin
    p = IL.q4_surface_benchmark()

    @testset "cheblobatto nodes and differentiation matrix" begin
        t, D = GGJ.cheblobatto(8)
        @test length(t) == 9
        @test issorted(t)                         # ascending, per the reflected convention
        @test t[1] ≈ -1 && t[end] ≈ 1
        @test D * ones(9) ≈ zeros(9) atol = 1e-10  # d/dt of a constant is zero
        @test D * t ≈ ones(9) atol = 1e-10         # d/dt of the linear function is one
    end

    @testset "ode_matrix: ordinary point and type-generic build" begin
        Q = 5.0im
        # x = 0 is an ordinary point: the coefficient matrix is finite there.
        M0 = GGJ.ode_matrix(p, Q, 0.0)
        @test all(isfinite, M0)
        @test M0[1, 4] == 1 && M0[2, 5] == 1 && M0[3, 6] == 1   # v' = (Ψ',Ξ',Υ') block
        # The extended-precision build agrees with the Float64 build.
        Md = GGJ.ode_matrix(Complex{GGJ.Double64}, p, Q, 0.3)
        @test ComplexF64.(Md) ≈ GGJ.ode_matrix(p, Q, 0.3) rtol = 1e-12
    end

    @testset "parity_rows match the deltac boundary convention" begin
        @test GGJ.parity_rows(1) == [4, 2, 3]     # odd:  Ψ'(0)=Ξ(0)=Υ(0)=0
        @test GGJ.parity_rows(2) == [1, 5, 6]     # even: Ψ(0)=Ξ'(0)=Υ'(0)=0
    end

    @testset "decaying_pair is an orthonormal 6×2 frame" begin
        Q = 5.0im
        θ = angle(Q) / 4
        E = GGJ.decaying_pair(p, Q, θ, 60.0)
        @test size(E) == (6, 2)
        @test all(isfinite, E)
        @test E' * E ≈ I(2) atol = 1e-10          # columns orthonormal
    end

    @testset "profile diagnostics reconstruct finite fields" begin
        res = IL.solve_ray(p, 5.0im)
        prof = IL.solution_profile(res; npc=6)
        @test size(prof.Ψ, 2) == 2 && length(prof.s) == size(prof.Ψ, 1)
        @test all(isfinite, prof.Ψ) && all(isfinite, prof.Ξ) && all(isfinite, prof.Υ)
        # Analytic tail evaluates on the trusted series radius.
        asy = IL.asymptotic_profile(p, res, [res.S, 2 * res.S])
        @test all(isfinite, asy.Ψ)
    end

    @testset "delta_convergence: small spread, consistent with solve_inner" begin
        Q = 5.0im
        conv = IL.delta_convergence(p, Q; verbose=false)
        Δ = IL.solve_inner(IL.GGJModel(; solver=:ray), p, Q * GGJ.q0(p))
        # conv.Δ is the raw solve_ray pair (Δ₁, Δ₂) = (interchange, tearing).
        @test conv.Δ[1] ≈ Δ.interchange rtol = 1e-6   # baseline == the plain solve
        @test conv.Δ[2] ≈ Δ.tearing rtol = 1e-6
        @test maximum(conv.spread) < 1e-4         # honest error bar is small here
    end
end

@testset "solve_inner_profile interface (matching-driver contract)" begin
    p = IL.glasser_wang_2020_eq55()
    γ = 0.1234 * GGJ.q0(p)
    for model in (IL.GGJModel(; solver=:ray), IL.GGJModel(; solver=:galerkin))
        prof = IL.solve_inner_profile(model, p, γ)
        # Δ agrees with the plain matching solve of the same backend (identical solve path).
        # prof.Δ stays the positional pair (Δ₁, Δ₂) = (interchange, tearing) that feeds deltar.
        Δ_plain = IL.solve_inner(model, p, γ)
        @test prof.Δ[1] ≈ Δ_plain.interchange rtol = 1e-12
        @test prof.Δ[2] ≈ Δ_plain.tearing rtol = 1e-12
        # Real ascending inner-coordinate grid from the rational surface, profiles npts × 2.
        @test issorted(prof.x)
        @test prof.x[1] ≈ 0 atol = 1e-12
        @test size(prof.Ψ) == (length(prof.x), 2) && size(prof.Ξ) == (length(prof.x), 2)
        @test all(isfinite, prof.Ψ) && all(isfinite, prof.Ξ)
        # Parity at the layer center: Ψ(0) ≠ 0 odd-parity column, Ψ(0) = 0 even-parity column.
        @test abs(prof.Ψ[1, 2]) < 1e-6 * abs(prof.Ψ[1, 1])
        # Conversion factors match their GGJ definitions.
        @test prof.dψdx ≈ GGJ.x0(p) / p.v1
        @test prof.rescale ≈ (p.v1 / GGJ.x0(p))^(0.5 + GGJ.p1(p))
    end
    # Ray backend certificate: at real Q the optimal contour is θ = 0, so the two solves coincide.
    ray = IL.solve_inner_profile(IL.GGJModel(; solver=:ray), p, γ)
    @test ray.certΔ < 1e-12
end
