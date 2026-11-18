using LinearAlgebra, Logging, TOML

const FFS = GeneralizedPerturbedEquilibrium.ForceFreeStates

# A fixture (Solovev by default) prepared as solve prepares it, up to the integration. Each integration mutates its
# intr (sing[s].delta_prime etc.), so every run takes a freshly prepared one.
function prepared_solovev(; fixture="regression_solovev_ideal_example", integrator="Riccati", singfac_min=nothing, nchunks=0, over=Dict{String,Any}())
    GPEC = GeneralizedPerturbedEquilibrium
    ex = joinpath(@__DIR__, "test_data", fixture)
    inputs = TOML.parsefile(joinpath(ex, "gpec.toml"))
    # The fixture's [Forward] tolerance drives both integrators; the Riccati control also carries the chunking.
    inputs["ForceFreeStates"]["integrator"] = integrator
    if integrator == "Riccati"
        inputs["Riccati"] = delete!(pop!(inputs, "Forward"), "renorm_threshold")
        inputs["Riccati"]["nchunks"] = nchunks
    end
    singfac_min === nothing || (inputs["ForceFreeStates"]["singfac_min"] = singfac_min)
    merge!(inputs["ForceFreeStates"], over)
    equil = GPEC.Equilibrium.PlasmaEquilibrium(GPEC.equilibrium_config(inputs, ex), GPEC.analytic_config(inputs))
    prob = GPEC.ForceBalanceProblem(equil; GPEC.force_balance_keywords(inputs, ex)...)
    ctrl, intr, equil, metric, mats = GPEC._prepare_solve(prob, GPEC.force_balance_algorithm(inputs))
    return ctrl, intr, equil, metric, mats
end

# Free-boundary energy eigenvalues of a Solovev fixture, crossing each surface at `singfac_min`.
function solovev_energies(integrator, singfac_min; fixture="regression_solovev_ideal_example")
    ctrl, intr, equil, _, mats = prepared_solovev(; fixture, integrator, singfac_min)
    odet = first(FFS.eulerlagrange_integration(ctrl, equil, mats, intr))
    return FFS.free_run(odet, ctrl, equil, mats, intr).et
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
        U1 = rng .+ 0.5 * I(N)
        U2 = 0.5 * rng .+ I(N)  # near-identity to ensure invertibility

        u = zeros(ComplexF64, N, N, 2)
        u[:, :, 1] .= U1
        u[:, :, 2] .= U2

        S_expected = U1 / U2  # = U₁ · U₂⁻¹

        FFS.renormalize_riccati_inplace!(u, N)

        @test u[:, :, 2] ≈ I(N)
        @test u[:, :, 1] ≈ S_expected rtol = 1e-12
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
        @test u[:, :, 1] ≈ S rtol = 1e-12
    end

    # ── Shared Solovev setup ──────────────────────────────────────────────────
    #
    # equil (Grad-Shafranov solve) and mats (metric matrices) are expensive and
    # immutable after construction — built ONCE and shared across all tests below.
    # intr is cheap to (re)initialize but is mutated by each integration run
    # (sing[s].delta_prime etc.), so a fresh copy is made for each integration.

    ctrl, intr_ric, equil, metric, mats = prepared_solovev()
    N = intr_ric.numpert_total

    odet_ric = first(FFS.riccati_eulerlagrange_integration(ctrl, equil, mats, intr_ric))

    # ─────────────────────────────────────────────────────────────────────────

    @testset "energy does not depend on the crossing distance" begin
        et_far = solovev_energies("Riccati", 1e-3)
        et_near = solovev_energies("Riccati", 1e-5)
        @test isapprox(et_far[1], et_near[1]; rtol=1e-6)
        # Forward crosses the same way, so it gives the same energy at either distance.
        @test isapprox(solovev_energies("Forward", 1e-3)[1], et_far[1]; rtol=1e-6)
        @test isapprox(solovev_energies("Forward", 1e-5)[1], et_far[1]; rtol=1e-6)
    end

    @testset "surface with two resonances" begin
        fixture = "regression_solovev_ideal_example_multi_n"
        et_far = solovev_energies("Riccati", 1e-3; fixture)
        et_near = solovev_energies("Riccati", 1e-5; fixture)
        et_forward = solovev_energies("Forward", 1e-3; fixture)
        # The least-stable multi-n eigenvalue is near marginal, so its relative tolerances are looser.
        @test isapprox(et_far[1], et_near[1]; rtol=1e-4)
        @test isapprox(et_far[1], et_forward[1]; rtol=1e-4)
    end

    @testset "stored solution" begin
        # Every solver step of every chunk is stored, in increasing ψ, ending at the edge state.
        @test odet_ric.step == length(odet_ric.psi_store) == size(odet_ric.u_store, 4)
        @test all(diff(odet_ric.psi_store) .> 0)
        @test odet_ric.u_store[:, :, :, end] ≈ odet_ric.u

        # Each stored column is a solution: integrated directly from one stored node it lands on the next.
        psi = odet_ric.psi_store
        for k in (1, odet_ric.step ÷ 4, odet_ric.step ÷ 2, odet_ric.step - 1)
            any(s -> psi[k] < s.psifac < psi[k+1], intr_ric.sing) && continue
            params = (ctrl, equil, mats, intr_ric, FFS.OdeState(N, 1, 1, 0), FFS.IntegrationChunk(psi[k], psi[k+1], false, 0, 1))
            direct = FFS.solve_el(FFS.ODEProblem(FFS.sing_der!, odet_ric.u_store[:, :, :, k], (psi[k], psi[k+1]), params),
                ctrl.algorithm.eulerlagrange_tolerance).u[end]
            @test isapprox(direct, odet_ric.u_store[:, :, :, k+1]; rtol=1e-5)
        end

        # Inside the crossing distance of each surface the stored states come from its asymptotic series, and they
        # continue the integrated solution: integrating from where the integration stopped lands on the first of them.
        for s in intr_ric.sing
            near = findall(p -> abs(p - s.psifac) < (1 - 1e-9) * ctrl.singfac_min / abs(s.n[1] * s.q1), psi)
            @test length(near) == 2 * ctrl.crossing_fill_points
            k = first(near) - 1
            params = (ctrl, equil, mats, intr_ric, FFS.OdeState(N, 1, 1, 0), FFS.IntegrationChunk(psi[k], psi[k+1], false, 0, 1))
            direct = FFS.solve_el(FFS.ODEProblem(FFS.sing_der!, odet_ric.u_store[:, :, :, k], (psi[k], psi[k+1]), params),
                ctrl.algorithm.eulerlagrange_tolerance).u[end]
            @test isapprox(direct, odet_ric.u_store[:, :, :, k+1]; rtol=1e-5)
        end

        # They are the solutions Forward stores: scaled to a unit edge displacement, ξ and its conjugate momentum
        # agree at the nodes the two integrators share, which lie on each side of every surface.
        function edge_scaled(integrator)
            ctrl, intr, equil, _, mats = prepared_solovev(; integrator)
            odet = first(FFS.eulerlagrange_integration(ctrl, equil, mats, intr))
            edge = odet.u_store[:, :, 1, odet.step]
            return odet, intr, [cat(odet.u_store[:, :, 1, k] / edge, odet.u_store[:, :, 2, k] / edge; dims=3) for k in 1:odet.step]
        end
        ric, intr, state_ric = edge_scaled("Riccati")
        fwd, _, state_fwd = edge_scaled("Forward")
        shared = [(findfirst(==(p), ric.psi_store), k) for (k, p) in enumerate(fwd.psi_store[1:fwd.step]) if p in ric.psi_store]
        @test length(shared) >= 2 * intr.msing
        for component in 1:2
            @test maximum(norm(state_ric[i][:, :, component] - state_fwd[k][:, :, component]) / norm(state_fwd[k][:, :, component]) for (i, k) in shared) < 1e-4
        end
    end

    # Note: a Solovev per-surface Δ' regression testset previously lived here,
    # exercising the (1 - ca_l[res,res,2]) / (4π²·psio) calculation from the
    # Riccati path. Per-surface Δ' is now treated as a stub (left in the code
    # for future work but de-emphasized): not reported, not output, and not
    # regression-tested on any actual equilibrium. The canonical Δ' is the
    # STRIDE BVP Δ' matrix (see runtests_parallel_integration.jl).

    @testset "Riccati end state has U₂ ≈ I" begin
        # The edge state is reduced to the canonical Riccati form (S, I).
        @test odet_ric.u[:, :, 2] ≈ I(N) rtol = 1e-10
    end

    @testset "Δ′ does not depend on the chunk layout" begin
        # The TJ fixture has a well-conditioned Δ′; the smallest layout has two chunks per segment.
        function delta_prime(nchunks)
            ctrl, intr, equil, _, mats = prepared_solovev(; fixture="regression_tj_ideal_example", nchunks)
            odet, riccati = FFS.riccati_eulerlagrange_integration(ctrl, equil, mats, intr)
            wv = FFS.free_run(odet, ctrl, equil, mats, intr).wv
            result, _ = FFS.compute_delta_prime_matrix(intr, riccati.propagators, riccati.chunks, riccati.columns.axis_state; wv, psio=equil.psio)
            return result.matrix, riccati.chunks
        end
        dp_auto, chunks_auto = delta_prime(0)
        dp_few, chunks_few = delta_prime(1)
        @test length(chunks_few) < length(chunks_auto)
        @test !isempty(dp_auto) && all(isfinite, dp_auto)
        @test isapprox(dp_few, dp_auto; rtol=1e-4)
    end

    @testset "a surface whose small solution the tolerance cannot resolve is named" begin
        function surface_warnings(fixture)
            ctrl, intr, equil, _, mats = prepared_solovev(; fixture)
            _, riccati = FFS.riccati_eulerlagrange_integration(ctrl, equil, mats, intr)
            return () -> FFS.warn_unresolved_surfaces(riccati.bases, intr.sing[1:intr.msing], ctrl.algorithm.eulerlagrange_tolerance)
        end
        @test_logs (:warn, r"big solution is .* times the small one") match_mode = :any surface_warnings("regression_solovev_ideal_example")()
        @test_logs min_level = Logging.Warn surface_warnings("regression_tj_ideal_example")()
    end

    @testset "the Δ′ solutions on the stored grid" begin
        ctrl, intr, equil, _, mats = prepared_solovev(; fixture="regression_tj_ideal_example")
        odet, riccati = FFS.riccati_eulerlagrange_integration(ctrl, equil, mats, intr)
        wv = FFS.free_run(odet, ctrl, equil, mats, intr).wv
        delta_prime, coefficients = FFS.compute_delta_prime_matrix(intr, riccati.propagators, riccati.chunks, riccati.columns.axis_state; wv, psio=equil.psio)
        basis = FFS.outer_solution_basis(coefficients, riccati.propagators, riccati.chunks, riccati.columns, riccati.bases, intr.sing, equil, mats,
            FFS.ModeIndices(intr), FFS.crossing_fill_fractions(ctrl))
        N, msing = intr.numpert_total, intr.msing
        ideal = (2msing+1):(2msing+N)
        @test basis.psi == odet.psi_store && size(basis.xi) == size(basis.xi_deriv) == size(basis.xi_cut) == (N, odet.step, 2msing + N)
        # The Δ_EF solutions carry no big solution and a unit edge displacement: the ideal solution, which the
        # integrator's own columns give when scaled the same way.
        @test basis.xi[:, end, ideal] ≈ I(N)
        edge = odet.u_store[:, :, 1, end]
        @test all(k -> isapprox(basis.xi[:, k, ideal], odet.u_store[:, :, 1, k] / edge; rtol=1e-6), 1:odet.step)
        # The cut solution differs from the solution only next to the surfaces, where it is what remains of the
        # resonant harmonic once the leading big and small terms are gone.
        for (j, s) in enumerate(intr.sing)
            lo, hi = basis.cut_range[j, :]
            @test lo < s.psifac < hi
            ires = 1 + s.m[1] - intr.mlow
            right = findfirst(>(s.psifac), basis.psi)
            @test abs(basis.xi_cut[ires, right, 2j]) < 1e-2 * abs(basis.xi[ires, right, 2j])
            @test abs(basis.xi_cut[ires, right-1, 2j-1]) < 1e-2 * abs(basis.xi[ires, right-1, 2j-1])
        end
        outside = [all(j -> !(basis.cut_range[j, 1] <= p <= basis.cut_range[j, 2]), 1:msing) for p in basis.psi]
        @test any(outside) && basis.xi_cut[:, outside, :] == basis.xi[:, outside, :]
    end

    @testset "truncation at the edge energy peak" begin
        # The chunks, their propagators and the Δ′ system follow the edge to the peak.
        ctrl, intr, equil, _, mats = prepared_solovev(; over=Dict{String,Any}("truncate_at_dW_peak" => true))
        odet, riccati = FFS.riccati_eulerlagrange_integration(ctrl, equil, mats, intr)
        (; propagators, chunks) = riccati
        axis_state = riccati.columns.axis_state
        @test intr.psilim == odet.psi_store[end] == chunks[end].psi_end
        @test length(propagators) == length(chunks) && count(c -> c.needs_crossing, chunks) == intr.msing
        @test odet.u_store[:, :, :, end] ≈ odet.u
        free = FFS.free_run(odet, ctrl, equil, mats, intr)
        @test isfinite(free.et[1])
        # The Δ′ system reproduces the plasma response of the axis-to-edge products, and says so when it does not.
        delta_prime, coefficients = @test_logs min_level = Logging.Warn FFS.compute_delta_prime_matrix(intr, propagators, chunks, axis_state;
            wv=free.wv, wp=free.wp, psio=equil.psio)
        @test size(delta_prime.matrix) == (intr.msing, intr.msing) && all(isfinite, delta_prime.matrix)
        @test size(delta_prime.coil) == (2 * intr.msing, intr.numpert_total)
        basis = FFS.outer_solution_basis(
            coefficients,
            propagators,
            chunks,
            riccati.columns,
            riccati.bases,
            intr.sing,
            equil,
            mats,
            FFS.ModeIndices(intr),
            FFS.crossing_fill_fractions(ctrl)
        )
        @test basis.psi == odet.psi_store && all(isfinite, basis.xi)
        @test_logs (:warn, r"differs from the axis-to-edge one") FFS.compute_delta_prime_matrix(intr, propagators, chunks, axis_state;
            wv=free.wv, wp=1.01 .* free.wp, psio=equil.psio)
    end

end
