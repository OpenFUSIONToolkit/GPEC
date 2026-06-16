using LinearAlgebra, Random, TOML

const FFS = GeneralizedPerturbedEquilibrium.ForceFreeStates

# Configure a fresh ForceFreeStatesInternal from an already-built equilibrium.
# Cheap (sing_lim! + sing_find! + field assignment). Separate from equil/ffit
# setup because intr is mutated by each integration (sing[s].delta_prime etc.).
function make_solovev_intr(inputs, ctrl, equil, ex)
    intr = FFS.ForceFreeStatesInternal(; dir_path=ex)
    intr.wall_settings = GeneralizedPerturbedEquilibrium.Vacuum.WallShapeSettings(;
        (Symbol(k) => v for (k, v) in inputs["Wall"])...)
    FFS.sing_lim!(intr, ctrl, equil)
    intr.nlow = ctrl.nn_low;
    intr.nhigh = ctrl.nn_high;
    intr.npert = 1
    FFS.sing_find!(intr, equil)
    intr.mlow = min(intr.nlow * equil.params.qmin, 0) - 4 - ctrl.delta_mlow
    intr.mhigh = trunc(Int, intr.nhigh * equil.params.qmax) + ctrl.delta_mhigh
    intr.mpert = intr.mhigh - intr.mlow + 1
    intr.numpert_total = intr.mpert * intr.npert
    return intr
end

@testset "Riccati Integration Tests" begin

    # ── Pure matrix unit tests — no equilibrium needed ────────────────────────

    @testset "renormalize_riccati_inplace!" begin
        N = 4
        # Build a random (U₁, U₂) pair and verify renorm gives S = U₁·U₂⁻¹ with U₂_new = I
        rng = [1.0+0.5im 0.2im 0.1 0.3im;
            0.0 1.2+0.1im 0.0im 0.2;
            0.1+0.1im 0.0 0.9+0.3im 0.1im;
            0.0im 0.2 0.0 1.1+0.2im]
        U1 = rng .+ 0.5*I(N)
        U2 = 0.5*rng .+ I(N)  # near-identity to ensure invertibility

        u = zeros(ComplexF64, N, N, 2)
        u[:, :, 1] .= U1
        u[:, :, 2] .= U2

        S_expected = U1 / U2  # = U₁ · U₂⁻¹

        FFS.renormalize_riccati_inplace!(u, N)

        @test u[:, :, 2] ≈ I(N)
        @test u[:, :, 1] ≈ S_expected rtol=1e-12
    end

    @testset "renormalize_riccati_inplace! idempotent" begin
        N = 3
        # If U₂ = I already, renorm should leave u unchanged
        S = [1.0+0.5im 0.2im 0.1;
            0.0im 1.2+0.1im 0.0;
            0.1+0.1im 0.0 0.9+0.3im]
        u = zeros(ComplexF64, N, N, 2)
        u[:, :, 1] .= S
        u[:, :, 2] .= I(N)

        FFS.renormalize_riccati_inplace!(u, N)

        @test u[:, :, 2] ≈ I(N)
        @test u[:, :, 1] ≈ S rtol=1e-12
    end

    @testset "renormalize_riccati! (OdeState)" begin
        N = 3
        rng = [1.0+0.5im 0.2im 0.1;
            0.0im 1.2+0.1im 0.0;
            0.1+0.1im 0.0 0.9+0.3im]
        U1 = rng .+ 0.5*I(N)
        U2 = 0.2*rng .+ I(N)

        odet = FFS.OdeState(N, 10, 5, 1)
        odet.u[:, :, 1] .= U1
        odet.u[:, :, 2] .= U2

        S_expected = U1 / U2
        intr = FFS.ForceFreeStatesInternal(; mpert=N, numpert_total=N)

        FFS.renormalize_riccati!(odet, intr)

        @test odet.u[:, :, 2] ≈ I(N)
        @test odet.u[:, :, 1] ≈ S_expected rtol=1e-12
    end

    # ── Shared Solovev setup ──────────────────────────────────────────────────
    #
    # equil (Grad-Shafranov solve) and ffit (metric matrices) are expensive and
    # immutable after construction — built ONCE and shared across all tests below.
    # intr is cheap to (re)initialize but is mutated by each integration run
    # (sing[s].delta_prime etc.), so a fresh copy is made for each integration.
    #
    # Integration runs:
    #   intr_ric / odet_ric — Riccati path (shared by most tests)
    #   intr_std / odet_std — Standard path (energy comparison only)

    ex = joinpath(@__DIR__, "test_data", "regression_solovev_ideal_example")
    inputs = TOML.parsefile(joinpath(ex, "gpec.toml"))
    inputs["ForceFreeStates"]["verbose"] = false

    ctrl = FFS.ForceFreeStatesControl(;
        (Symbol(k) => v for (k, v) in inputs["ForceFreeStates"])...)
    equil = GeneralizedPerturbedEquilibrium.Equilibrium.setup_equilibrium(
        GeneralizedPerturbedEquilibrium.Equilibrium.EquilibriumConfig(inputs["Equilibrium"], ex))

    intr_tmp = make_solovev_intr(inputs, ctrl, equil, ex)
    metric = FFS.make_metric(equil, intr_tmp.mpert)
    ffit = FFS.make_matrix(equil, intr_tmp, metric)
    N = intr_tmp.numpert_total

    # Riccati integration
    intr_ric = make_solovev_intr(inputs, ctrl, equil, ex)
    odet_ric = FFS.riccati_eulerlagrange_integration(ctrl, equil, ffit, intr_ric)

    # Save inline Δ' values before any test that calls compute_delta_prime_from_ca!
    # (which overwrites intr_ric.sing[s].delta_prime)
    delta_prime_inline = [copy(intr_ric.sing[s].delta_prime) for s in 1:intr_ric.msing]

    vac_ric = FFS.free_run!(odet_ric, ctrl, equil, ffit, intr_ric)
    et_ric = real(vac_ric.et[1])

    # Standard integration (needed only for energy comparison).  eulerlagrange_integration
    # returns (odet, propagators, chunks, S_at_surface_left); only odet is used here.
    intr_std = make_solovev_intr(inputs, ctrl, equil, ex)
    odet_std, _, _, _ = FFS.eulerlagrange_integration(ctrl, equil, ffit, intr_std)
    vac_std = FFS.free_run!(odet_std, ctrl, equil, ffit, intr_std)
    et_std = real(vac_std.et[1])

    # ─────────────────────────────────────────────────────────────────────────

    @testset "Riccati integration matches standard ODE — Solovev example" begin
        # PR description claims Solovev energy eigenvalue error 0.006 % vs standard path.
        # Tightened to rtol=1e-4 (matches the PR's headline claim within ≈2×). A regression
        # of the Riccati/renormalization algorithm to ~1 % error would fail here loudly.
        @test isapprox(et_ric, et_std; rtol=1e-4)

        # Riccati uses no more than 2x as many steps as standard
        @test odet_ric.step <= 2 * odet_std.step
    end

    # Note: a Solovev per-surface Δ' regression testset previously lived here,
    # exercising the (1 - ca_l[res,res,2]) / (4π²·psio) calculation from the
    # Riccati path. Per-surface Δ' is now treated as a stub (left in the code
    # for future work but de-emphasized): not reported, not output, and not
    # regression-tested on any actual equilibrium. The canonical Δ' is the
    # STRIDE BVP Δ' matrix (see runtests_parallel_integration.jl).

    @testset "Riccati end state has U₂ ≈ I" begin
        # After riccati_eulerlagrange_integration, odet.u[:,:,2] should be identity
        # (canonical Riccati convention after final renorm)
        @test odet_ric.u[:, :, 2] ≈ I(N) rtol=1e-10
    end

    @testset "riccati_der! formula — Glasser 2018 Eq. 19" begin
        # Verify riccati_der! correctly evaluates dS/dψ = w†·F̄⁻¹·w − S·Ḡ·S, w = Q − K̄·S.
        #
        # Test states are Hermitian (physical constraint: the EL system preserves S†=S from
        # the axis). Non-Hermitian states would give ~5% disagreement — not a bug, but a
        # consequence of the derivation assuming the physical symmetry.
        #
        # See benchmarks/benchmark_riccati_der.jl for the extended version with output.

        # Use an initialized OdeState just for spline_hint and chunk bounds
        odet_tmp = FFS.OdeState(N, ctrl.numsteps_init, ctrl.numunorms_init, intr_ric.msing)
        FFS.initialize_el_at_axis!(odet_tmp, ctrl, ffit, equil.profiles, intr_ric)
        chunks = FFS.chunk_el_integration_bounds(odet_tmp, ctrl, intr_ric)

        # 30% into each chunk: away from singularities at psi_end
        test_psis = [c.psi_start + 0.3 * (c.psi_end - c.psi_start) for c in chunks]

        rng = Random.MersenneTwister(42)
        for psi in test_psis
            # Hermitian S: physical Riccati matrix is Hermitian (preserved by EL symmetry)
            A = randn(rng, ComplexF64, N, N)
            S = (A + A') / 2

            # Manual RHS: w†·F̄⁻¹·w − S·Ḡ·S
            L = zeros(ComplexF64, N, N)
            Kmat = zeros(ComplexF64, N, N)
            Gmat = zeros(ComplexF64, N, N)
            ffit.fmats_lower(vec(L), psi; hint=ffit._hint)
            ffit.kmats(vec(Kmat), psi; hint=ffit._hint)
            ffit.gmats(vec(Gmat), psi; hint=ffit._hint)
            q = equil.profiles.q_spline(psi)
            singfac = vec(1.0 ./ ((intr_ric.mlow:intr_ric.mhigh) .-
                                  q .* (intr_ric.nlow:intr_ric.nhigh)'))
            w = -Kmat * S
            for i in 1:N
                ;
                w[i, i] += singfac[i];
            end
            v = copy(w)
            ldiv!(LowerTriangular(L), v)
            ldiv!(UpperTriangular(L'), v)
            dS_manual = adjoint(w) * v - S * Gmat * S

            # riccati_der! RHS
            u_ric = zeros(ComplexF64, N, N, 2)
            du_ric = zeros(ComplexF64, N, N, 2)
            u_ric[:, :, 1] .= S
            u_ric[:, :, 2] .= Matrix{ComplexF64}(I, N, N)
            dummy = FFS.IntegrationChunk(psi, psi, false, 0, 1)
            params = (ctrl, equil, ffit, intr_ric, odet_tmp, dummy)
            FFS.riccati_der!(du_ric, u_ric, params, psi)

            rel_err = norm(du_ric[:, :, 1] - dS_manual) / max(norm(dS_manual), 1e-10)
            @test rel_err < 1e-10
        end
    end

    @testset "compute_delta_prime_from_ca! matches inline Δ'" begin
        # Verify the standalone Δ' formula matches the inline Riccati crossing computation.
        # Both apply the identical diagonal formula to the same ca_l/ca_r arrays, so the
        # result must be bit-for-bit identical (not just approximately equal).
        #
        # Note: this call overwrites intr_ric.sing[s].delta_prime; delta_prime_inline was
        # saved before free_run! above so it holds the original inline values.
        #
        # See benchmarks/benchmark_delta_prime_methods.jl for the extended version.
        FFS.compute_delta_prime_from_ca!(odet_ric, intr_ric, equil)
        for s in 1:intr_ric.msing
            @test intr_ric.sing[s].delta_prime == delta_prime_inline[s]
        end
    end

end
