using LinearAlgebra
using TOML

# The Solovev fixture prepared exactly as solve prepares it, up to the Euler-Lagrange integration.
function _prepared_solovev(integrator::String="Forward"; nchunks=nothing)
    GPEC = GeneralizedPerturbedEquilibrium
    ex = joinpath(@__DIR__, "test_data", "regression_solovev_ideal_example")
    inputs = TOML.parsefile(joinpath(ex, "gpec.toml"))
    if integrator != "Forward"
        inputs["ForceFreeStates"]["integrator"] = integrator
        integrator == "Galerkin" ? delete!(inputs, "Forward") : (inputs[integrator] = pop!(inputs, "Forward"))
        integrator == "Riccati" && delete!(inputs["Riccati"], "renorm_threshold")
    end
    nchunks === nothing || (inputs[integrator]["nchunks"] = nchunks)
    equil = GPEC.Equilibrium.PlasmaEquilibrium(GPEC.equilibrium_config(inputs, ex), GPEC.analytic_config(inputs))
    prob = GPEC.ForceBalanceProblem(equil; GPEC.force_balance_keywords(inputs, ex)...)
    ctrl, intr, equil, metric, mats = GPEC._prepare_solve(prob, GPEC.force_balance_algorithm(inputs))
    return ctrl, intr, equil, metric, mats
end

@testset "BVP solve precision" begin
    FFS = GeneralizedPerturbedEquilibrium.ForceFreeStates
    M = ComplexF64[1 / (i + j - 1) + 0.02im / (i + j) for i in 1:6, j in 1:6]
    reference = setprecision(BigFloat, 256) do
        factor = lu(Complex{BigFloat}.(M))
        raw = zeros(Complex{BigFloat}, 2, 2)
        for row in 1:2
            rhs = zeros(Complex{BigFloat}, 6)
            rhs[4+row] = 1
            x = factor \ rhs
            raw[row, :] = x[[3, 5]]
        end
        delta = reshape([raw[2, 2] - raw[2, 1] - raw[1, 2] + raw[1, 1]], 1, 1)
        return ComplexF64.(delta), ComplexF64.(raw)
    end
    extended = FFS._solve_bvp_and_combine_pest3(M, 1, 1, [1], true)
    ordinary = FFS._solve_bvp_and_combine_pest3(M, 1, 1, [1], false)
    @test extended == reference
    @test ordinary != reference
    @test all(isapprox.(ordinary, reference; rtol=1e-8))
    @test !Riccati().extended_precision_bvp
end

@testset "Riccati FM Integration Tests" begin

    @testset "apply_propagator! and its inverse" begin
        FFS = GeneralizedPerturbedEquilibrium.ForceFreeStates
        N = 3
        A = I(N) .+ 0.15 * [1.0+0.2im 0.1im 0.05; 0.0im 0.9+0.3im 0.1; 0.2+0.1im 0.0 1.0+0.1im]
        B = 0.1 * [0.8+0.1im 0.1im 0.0; 0.0im 1.2+0.2im 0.1; 0.0im 0.1 0.9+0.1im]
        C = 0.1 * [0.5+0.1im 0.0im 0.1; 0.1im 0.8+0.2im 0.0; 0.0im 0.0 0.7+0.1im]
        D = I(N) .+ 0.15 * [0.9+0.1im 0.0im 0.05; 0.0im 1.0+0.2im 0.0; 0.1+0.1im 0.0 0.95+0.1im]
        # Fundamental matrix [A B; C D] in the propagator layout: U₁ rows in slice 1, U₂ rows in slice 2.
        prop = FFS.ChunkPropagator(cat([A B], [C D]; dims=3), Float64[], Array{ComplexF64,3}[])

        u1_in = [1.0+0.5im 0.2im 0.0; 0.1+0.1im 1.2+0.1im 0.0; 0.0im 0.0 0.9+0.3im]
        u2_in = I(N) .+ 0.1im * ones(N, N)
        odet = FFS.OdeState(N, 10, 5, 0)
        odet.u[:, :, 1] .= u1_in
        odet.u[:, :, 2] .= u2_in

        FFS.apply_propagator!(odet, prop)
        @test odet.u[:, :, 1] ≈ A * u1_in + B * u2_in rtol = 1e-12
        @test odet.u[:, :, 2] ≈ C * u1_in + D * u2_in rtol = 1e-12

        FFS.apply_propagator_inverse!(odet, prop)
        @test odet.u[:, :, 1] ≈ u1_in rtol = 1e-12
        @test odet.u[:, :, 2] ≈ u2_in rtol = 1e-12
    end

    @testset "propagator steps reproduce a direct integration" begin
        # The fundamental matrix at a solver step, applied to a start state, is the solution from that state.
        FFS = GeneralizedPerturbedEquilibrium.ForceFreeStates
        ctrl, intr, equil, metric, mats = _prepared_solovev("Riccati")
        N = intr.numpert_total
        odet = FFS.OdeState(N, 1, 1, intr.msing)
        FFS.initialize_el_at_axis!(odet, ctrl, mats, equil.profiles, intr)
        chunks = FFS.balance_integration_chunks(FFS.chunk_el_integration_bounds(odet, ctrl, intr), ctrl, intr)
        chunk = chunks[1]
        start = FFS.propagator_start(chunks, 1, Dict{Int,Any}(), N)
        prop = FFS.chunk_propagator(chunk, start, ctrl.algorithm.eulerlagrange_tolerance, equil, mats, FFS.ModeIndices(intr))
        @test length(prop.psi) == length(prop.steps)
        @test prop.psi[1] == chunk.psi_start && prop.psi[end] == chunk.psi_end
        @test prop.steps[1][:, :, 1] == [I(N) zeros(N, N)] && prop.steps[1][:, :, 2] == [zeros(N, N) I(N)]
        @test prop.steps[end] == prop.u

        k = length(prop.psi) ÷ 2
        direct = FFS.solve_el(FFS.ODEProblem(FFS.sing_der!, copy(odet.u), (chunk.psi_start, prop.psi[k]),
                (ctrl, equil, mats, intr, FFS.OdeState(N, 1, 1, 0), chunk)), ctrl.algorithm.eulerlagrange_tolerance).u[end]
        x = [odet.u[:, :, 1]; odet.u[:, :, 2]]
        @test prop.steps[k][:, :, 1] * x ≈ direct[:, :, 1] rtol = 1e-5
        @test prop.steps[k][:, :, 2] * x ≈ direct[:, :, 2] rtol = 1e-5
    end

    @testset "balance_integration_chunks produces target count" begin
        # Verify that balance_integration_chunks creates 8*(msing+1)+msing chunks from a small set of
        # base chunks, and never fewer than the base chunks with every crossing chunk split once.
        ctrl, intr, equil, metric, mats = _prepared_solovev("Riccati")

        odet = GeneralizedPerturbedEquilibrium.ForceFreeStates.OdeState(intr.numpert_total, 1, 1, intr.msing)
        GeneralizedPerturbedEquilibrium.ForceFreeStates.initialize_el_at_axis!(odet, ctrl, mats, equil.profiles, intr)

        base_chunks = GeneralizedPerturbedEquilibrium.ForceFreeStates.chunk_el_integration_bounds(odet, ctrl, intr)
        balanced = GeneralizedPerturbedEquilibrium.ForceFreeStates.balance_integration_chunks(base_chunks, ctrl, intr)

        # Must mirror balance_integration_chunks' internal target_n formula for nchunks = 0
        # (src/ForceFreeStates/EulerLagrange.jl). Keep this in sync. The formula reads only
        # intr.msing — no thread count enters it, which is what makes Riccati outputs
        # independent of how many threads `julia -t` provides.
        min_chunks = length(base_chunks) + count(c -> c.needs_crossing, base_chunks)
        target_n = max(min_chunks, 8 * (intr.msing + 1) + intr.msing)

        # After balancing, chunk count equals target_n: the while-loop adds exactly one
        # chunk per iteration (a bisection split) and exits when length(result) >= target_n,
        # so the post-loop count is target_n under normal conditions. (The function can
        # produce fewer if every remaining chunk is unsplittable — width < 1e-8 — but that
        # never happens in the regression cases here.)
        @test length(balanced) == target_n

        # First chunk starts at the correct position, last chunk ends at the edge
        @test balanced[1].psi_start ≈ base_chunks[1].psi_start
        @test balanced[end].psi_end ≈ base_chunks[end].psi_end

        # Consecutive chunks are contiguous UNLESS the previous chunk ends with a
        # crossing (needs_crossing=true), in which case there is an intentional inner-layer
        # gap of ≈2·singfac_min/|n·q1| between the pre-crossing and post-crossing intervals.
        for i in eachindex(balanced)[2:end]
            if !balanced[i-1].needs_crossing
                @test balanced[i].psi_start ≈ balanced[i-1].psi_end rtol = 1e-10
            else
                # Inner-layer gap: post-crossing chunk starts AFTER the rational surface
                @test balanced[i].psi_start > balanced[i-1].psi_end
            end
        end

        # The total number of needs_crossing=true chunks should equal the original
        n_crossings_base = count(c -> c.needs_crossing, base_chunks)
        n_crossings_bal = count(c -> c.needs_crossing, balanced)
        @test n_crossings_bal == n_crossings_base

        # Chunking is a pure function of (chunks, ctrl, intr): repeated calls on the same
        # inputs give bit-identical boundaries. Together with the thread-free target_n
        # formula above, this is what guarantees Riccati results do not move with `julia -t`.
        balanced_again = GeneralizedPerturbedEquilibrium.ForceFreeStates.balance_integration_chunks(base_chunks, ctrl, intr)
        @test length(balanced_again) == length(balanced)
        @test all(balanced_again[i].psi_start == balanced[i].psi_start for i in eachindex(balanced))
        @test all(balanced_again[i].psi_end == balanced[i].psi_end for i in eachindex(balanced))

        # An explicit nchunks steers the target count directly.
        ctrl_more = first(_prepared_solovev("Riccati"; nchunks=target_n + 7))
        balanced_more = GeneralizedPerturbedEquilibrium.ForceFreeStates.balance_integration_chunks(base_chunks, ctrl_more, intr)
        @test length(balanced_more) == target_n + 7

        # An nchunks below the singular-surface floor is clamped up, with a warning. At the floor a
        # forward chunk still separates every crossing chunk from the previous surface or the axis.
        ctrl_few = first(_prepared_solovev("Riccati"; nchunks=1))
        balanced_few = @test_logs (:warn,) match_mode = :any GeneralizedPerturbedEquilibrium.ForceFreeStates.balance_integration_chunks(base_chunks, ctrl_few, intr)
        @test length(balanced_few) == min_chunks
        @test all(i > 1 && !balanced_few[i-1].needs_crossing for i in eachindex(balanced_few) if balanced_few[i].needs_crossing)
    end

    @testset "chunk_el_integration_bounds direction field — bidirectional mode" begin
        # Verify that bidirectional=true sets direction=-1 on crossing chunks and direction=+1
        # on non-crossing chunks, and that balance_integration_chunks propagates these correctly:
        # the right sub-chunk inherits direction from the parent, the left sub-chunk is always +1.
        ctrl, intr, equil, metric, mats = _prepared_solovev("Riccati")

        odet = GeneralizedPerturbedEquilibrium.ForceFreeStates.OdeState(intr.numpert_total, 1, 1, intr.msing)
        GeneralizedPerturbedEquilibrium.ForceFreeStates.initialize_el_at_axis!(odet, ctrl, mats, equil.profiles, intr)

        # Default (bidirectional=false): all chunks should have direction=+1
        chunks_fwd = GeneralizedPerturbedEquilibrium.ForceFreeStates.chunk_el_integration_bounds(odet, ctrl, intr)
        @test all(c -> c.direction == 1, chunks_fwd)

        # bidirectional=true: crossing chunks direction=-1, non-crossing direction=+1
        chunks_bidi = GeneralizedPerturbedEquilibrium.ForceFreeStates.chunk_el_integration_bounds(odet, ctrl, intr; bidirectional=true)
        @test count(c -> c.needs_crossing, chunks_bidi) > 0  # at least one crossing chunk
        for chunk in chunks_bidi
            if chunk.needs_crossing
                @test chunk.direction == -1
            else
                @test chunk.direction == 1
            end
        end

        # balance_integration_chunks preserves direction: right sub-chunk inherits parent direction,
        # left sub-chunk is always +1 regardless of parent
        balanced_bidi = GeneralizedPerturbedEquilibrium.ForceFreeStates.balance_integration_chunks(chunks_bidi, ctrl, intr)
        for chunk in balanced_bidi
            if chunk.needs_crossing
                @test chunk.direction == -1
            else
                @test chunk.direction == 1
            end
        end
    end

    @testset "Riccati FM integration matches forward ODE — Solovev example" begin
        # Run forward and Riccati FM integrations on the Solovev regression test.
        # The energy eigenvalue et[1] should match to within 2%.
        #
        # Bidirectional FM integration (crossing chunks integrated backward) is what the
        # Riccati path uses. It keeps FM propagators well-conditioned for both small-N
        # on this small multi-surface fixture.
        function run_solovev(integrator)
            ctrl, intr, equil, metric, mats = _prepared_solovev(integrator)
            odet = first(GeneralizedPerturbedEquilibrium.ForceFreeStates.eulerlagrange_integration(ctrl, equil, mats, intr))
            vac = GeneralizedPerturbedEquilibrium.ForceFreeStates.free_run(odet, ctrl, equil, mats, intr)
            return real(vac.et[1]), intr
        end

        et_fwd, intr_fwd = run_solovev("Forward")
        et_ric, intr_ric = run_solovev("Riccati")

        # Energy eigenvalue matches to 2%
        @test isapprox(et_ric, et_fwd; rtol=0.02)
    end

    @testset "ode_itime_cost is additive over sub-intervals" begin
        # Verify cost(a, c) ≈ cost(a, b) + cost(b, c) for b ∈ (a, c) where no
        # rational surface is inside [a, c]. The cost function uses abs(Δlog) for
        # each reference point; this is additive only when |psi - ref| is monotone
        # on [a, c], i.e., when no reference (rational surface, axis, edge) lies
        # strictly inside the interval. We use the first integration chunk from
        # chunk_el_integration_bounds, which is guaranteed to contain no rational
        # surfaces in its interior.
        ctrl, intr, equil, metric, mats = _prepared_solovev()

        # Use the first chunk from chunk_el_integration_bounds: guaranteed rational-free interior
        odet_tmp = GeneralizedPerturbedEquilibrium.ForceFreeStates.OdeState(intr.numpert_total, 10, 5, intr.msing)
        GeneralizedPerturbedEquilibrium.ForceFreeStates.initialize_el_at_axis!(odet_tmp, ctrl, mats, equil.profiles, intr)
        chunks_tmp = GeneralizedPerturbedEquilibrium.ForceFreeStates.chunk_el_integration_bounds(odet_tmp, ctrl, intr)
        chunk1 = chunks_tmp[1]
        a = chunk1.psi_start
        c = chunk1.psi_end
        b = (a + c) / 2.0

        cost_ac = GeneralizedPerturbedEquilibrium.ForceFreeStates.ode_itime_cost(a, c, intr)
        cost_ab = GeneralizedPerturbedEquilibrium.ForceFreeStates.ode_itime_cost(a, b, intr)
        cost_bc = GeneralizedPerturbedEquilibrium.ForceFreeStates.ode_itime_cost(b, c, intr)

        @test isapprox(cost_ac, cost_ab + cost_bc; rtol=1e-10)
    end

    # Note: a Solovev BVP Δ' regression testset previously lived here, but the
    # Solovev fixture (q₀ = 1.9, e = 1.6, close conformal wall) is near marginal
    # external-kink stability (et[1] ≈ +0.24), where Δ' diverges — the pinned
    # values were order 10⁵-10¹¹ with |Im/Re| ≫ 1 and didn't track anything
    # physically meaningful. BVP Δ' regression is concentrated on the DIIID-like
    # fixture below (intrinsically stable, well-conditioned BVP Δ').

    @testset "both ODE integrators leave solution profiles" begin
        # Either store materializes into dΞ_ψ/dψ and Ξ_s, which is what PerturbedEquilibrium consumes.
        for integrator in ("Forward", "Riccati")
            ctrl, intr, equil, metric, mats = _prepared_solovev(integrator)
            odet = first(GeneralizedPerturbedEquilibrium.ForceFreeStates.eulerlagrange_integration(ctrl, equil, mats, intr))
            @test GeneralizedPerturbedEquilibrium.ForceFreeStates.materialize_derivative_stores!(odet, equil, mats, intr)
            @test odet.du_store_populated
            @test size(odet.du_store) == size(odet.xi_s_store) == (intr.numpert_total, intr.numpert_total, odet.step)
            @test all(isfinite, odet.du_store) && all(isfinite, odet.xi_s_store)
        end
    end

    # Production DIII-D energies and Δ′ are tracked by the diiid_n1_riccati harness cases.

end
