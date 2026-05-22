# Tests for the InnerLayer module: the shared WasowGalerkin engine and the GGJ
# single-fluid model that is its first client / regression oracle.

using Test
using StaticArrays
using GeneralizedPerturbedEquilibrium.InnerLayer
const WG = GeneralizedPerturbedEquilibrium.InnerLayer.WasowGalerkin
const ILGGJ = GeneralizedPerturbedEquilibrium.InnerLayer.GGJ

@testset "InnerLayer" begin
    p = glasser_wang_2020_eq55()
    spec = ILGGJ.ggj_system_spec(p)

    @testset "SystemSpec construction" begin
        @test spec.n == 6
        @test spec.ncomp == 3
        @test spec.n_alg == 2
        @test spec.nterms == 3
        @test spec.alg_idx == [1, 2]
        @test spec.exp_idx == [3, 4, 5, 6]
        @test spec.domain == :half
        @test WG.n_exp(spec) == 4
        @test WG.nparities(spec.bc) == 2
        # Algebraic block must be contiguous and first.
        @test_throws ArgumentError WG.SystemSpec(; n=6, ncomp=3, np=3, nterms=3, n_alg=2,
            alg_idx=[1, 3], exp_idx=[2, 4, 5, 6],
            asymptotic_matrices=spec.asymptotic_matrices, eigenbasis=spec.eigenbasis,
            exponents=spec.exponents, coeffs=spec.coeffs, physical=spec.physical,
            rescale=spec.rescale, bc=spec.bc)
    end

    @testset "Wasow asymptotic basis" begin
        Q = inner_Q(p, ComplexF64(1.0, 1.0))
        cache = WG.build_wasow(spec, p, Q; kmax=8)
        @test cache isa WG.WasowCache
        @test cache.n == 6 && cache.n_alg == 2
        @test length(cache.R) == 2
        # The basis satisfies the asymptotic ODE far out (the series converges for
        # x ≳ a few hundred for this case); the residual decreases there.
        @test maximum(WG.wasow_residual(cache, 500.0)) < 1e-9
        @test maximum(WG.wasow_residual(cache, 500.0)) < maximum(WG.wasow_residual(cache, 250.0))
        # pick_xmax returns a point where the residual meets tolerance.
        xm, c2 = WG.pick_xmax(spec, p, Q; eps=1e-7)
        @test maximum(WG.wasow_residual(c2, xm)) < 1e-7
        # Evaluation returns the right shapes.
        U, dU = WG.evaluate_wasow(cache, 50.0; derivative=true)
        @test size(U) == (6, 2)
        @test size(dU) == (6, 2)
        @test all(isfinite, U)
    end

    @testset "GGJ Galerkin golden (Glasser & Wang 2020 Eq. 55)" begin
        # Golden matching data captured from develop prior to the WasowGalerkin
        # refactor. The engine reproduces these within the solver's xmax-bisection
        # floating-point reproducibility floor (see ggj_reference.toml).
        Δ = solve_inner(GGJModel(; solver=:galerkin), p, ComplexF64(1.0, 1.0))
        @test Δ isa SVector{2,ComplexF64}
        gold = (ComplexF64(-2.210957642095927e+01, -3.881739471602441e+00),
            ComplexF64(-9.204746144445668e-01, 6.257857379918319e-01))
        @test real(Δ[1]) ≈ real(gold[1]) atol = 1e-3
        @test imag(Δ[1]) ≈ imag(gold[1]) atol = 1e-3
        @test real(Δ[2]) ≈ real(gold[2]) atol = 1e-5
        @test imag(Δ[2]) ≈ imag(gold[2]) atol = 1e-5
    end

    @testset "Backward-compatible asymptotic API" begin
        Q = inner_Q(p, ComplexF64(1.0, 1.0))
        cache = build_asymptotics(p, Q; kmax=8)
        @test cache isa InnerAsymptoticsCache
        U, _ = evaluate_asymptotics(cache, 50.0; derivative=true)
        @test size(U) == (6, 2)
        xm, _ = pick_xmax(p, Q; eps=1e-7)
        @test xm > 0
    end

    @testset "component_parity derived from bc" begin
        # NEUMANN component → even (+1), DIRICHLET → odd (−1); GGJ odd sector.
        @test spec.component_parity == [1, -1, -1]
        @test WG.component_parity_from_bc(spec.bc) == [1, -1, -1]
    end

    @testset "Full-domain recombines to half-domain (parity-symmetric GGJ)" begin
        # The two-sided full-domain solve on [-xmax,+xmax] must, for the
        # parity-symmetric GGJ system, produce a SYMMETRIC end-matching matrix M,
        # whose symmetric/antisymmetric parity combinations reproduce the
        # half-domain (Δ_odd, Δ_even) matching data to solver tolerance.
        for γ in (ComplexF64(1.0, 1.0), ComplexF64(0.5, 0.3))
            fm = solve_inner_full(GGJModel(; solver=:galerkin), p, γ; nx=256)
            @test fm isa WG.FullDomainMatching
            @test all(isfinite, fm.M)
            relasym = abs(fm.M[1, 2] - fm.M[2, 1]) / max(abs(fm.M[1, 2]), abs(fm.M[2, 1]))
            @test relasym < 1e-7   # M symmetric for the parity-symmetric system
            # Recombine and map to the half-domain output convention (λ_sym = odd,
            # λ_anti = even; solve_inner returns rescale(even, odd)).
            λs, λa = WG.parity_recombine(fm)
            full_phys = ILGGJ.rescale_delta(SVector(λa, λs), p)
            half = solve_inner(GGJModel(; solver=:galerkin), p, γ; nx=256)
            @test full_phys[1] ≈ half[1] rtol = 1e-4
            @test full_phys[2] ≈ half[2] rtol = 1e-4
        end
    end

    @testset "Full-domain matching matrix on a parity-broken system" begin
        # A synthetic system whose FEM coefficients break x → -x symmetry: the
        # half-domain even/odd decomposition no longer applies and the 2×2 matrix
        # M is asymmetric by design — the full matrix is then the matching
        # deliverable. The two-sided assembly must run and yield a finite,
        # genuinely-asymmetric M.
        base = ILGGJ.ggj_system_spec(p)
        broken_coeffs(params, Q, x) = begin
            Imat, U, V = base.coeffs(params, Q, x)
            bump = ComplexF64[0 0 0; 0 0 0; 0 0 0.1*x]  # parity-breaking interior term
            return Imat, U + bump, V
        end
        specb = WG.SystemSpec(; n=base.n, ncomp=base.ncomp, np=base.np, nterms=base.nterms,
            n_alg=base.n_alg, alg_idx=base.alg_idx, exp_idx=base.exp_idx,
            asymptotic_matrices=base.asymptotic_matrices, eigenbasis=base.eigenbasis,
            exponents=base.exponents, coeffs=broken_coeffs, physical=base.physical,
            rescale=base.rescale, bc=base.bc, domain=:full, component_parity=[1, -1, -1])
        Q = inner_Q(p, ComplexF64(1.0, 1.0))
        fm = WG.solve_galerkin_full(specb, p, Q; nx=128)
        @test fm isa WG.FullDomainMatching
        @test all(isfinite, fm.M)
        relasym = abs(fm.M[1, 2] - fm.M[2, 1]) / max(abs(fm.M[1, 2]), abs(fm.M[2, 1]))
        @test relasym > 1e-6   # genuinely asymmetric
    end

    @testset "Shooting cross-check at small Q" begin
        # The backward shoot is consistent only for very small Q; at γ=1e-3 both
        # solvers should agree on the dominant matching component.
        γ = ComplexF64(1e-3, 0.0)
        Δs = solve_inner(GGJModel(; solver=:shooting), p, γ)
        Δg = solve_inner(GGJModel(; solver=:galerkin), p, γ)
        @test Δs isa SVector{2,ComplexF64}
        rel = abs(Δs[1] - Δg[1]) / abs(Δs[1])
        @test rel < 1e-2
    end
end
