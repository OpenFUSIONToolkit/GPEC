using LinearAlgebra
using TOML

@testset "BVP without controls preserves extended precision" begin
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
    default = FFS._solve_bvp_and_combine_pest3(M, 1, 1, 6, false, [1], nothing, nothing, false)
    extended = FFS._solve_bvp_and_combine_pest3(M, 1, 1, 6, false, [1], nothing, FFS.ForceFreeStatesControl(), false)
    ordinary = FFS._solve_bvp_and_combine_pest3(M, 1, 1, 6, false, [1], nothing, FFS.ForceFreeStatesControl(; algorithm=Riccati(; extended_precision_bvp=false)), false)
    @test default == reference
    @test extended == reference
    @test ordinary != reference
end

@testset "Riccati FM Integration Tests" begin

    @testset "ChunkPropagator identity on trivial interval" begin
        # Integrating over a zero-width interval should give the identity propagator.
        # We test that apply_propagator! on an identity state preserves the state.
        N = 3
        prop = GeneralizedPerturbedEquilibrium.ForceFreeStates.ChunkPropagator(N)

        # Set propagator to identity (block_upper_ic = (I, 0), block_lower_ic = (0, I))
        for i in 1:N
            prop.block_upper_ic[i, i, 1] = 1  # U1 block from IC=(I,0)
            prop.block_lower_ic[i, i, 2] = 1  # U2 block from IC=(0,I)
        end

        # Apply identity propagator to an arbitrary state
        odet = GeneralizedPerturbedEquilibrium.ForceFreeStates.OdeState(N, 10, 5, 0)
        u1_in = [1.0+0.5im 0.2im 0.0;
            0.1+0.1im 1.2+0.1im 0.0;
            0.0im 0.0 0.9+0.3im]
        u2_in = [0.8+0.1im 0.1im 0.0;
            0.0im 1.0+0.2im 0.1;
            0.1im 0.0 1.1+0.0im]
        odet.u[:, :, 1] .= u1_in
        odet.u[:, :, 2] .= u2_in

        GeneralizedPerturbedEquilibrium.ForceFreeStates.apply_propagator!(odet, prop)

        @test odet.u[:, :, 1] ≈ u1_in rtol=1e-12
        @test odet.u[:, :, 2] ≈ u2_in rtol=1e-12
    end

    @testset "apply_propagator! linearity" begin
        # Verify that apply_propagator! applies the correct linear map.
        N = 3
        prop = GeneralizedPerturbedEquilibrium.ForceFreeStates.ChunkPropagator(N)

        # Fill block_upper_ic and block_lower_ic with random data
        rng_upper = [1.1+0.2im 0.1im 0.05;
            0.0im 0.9+0.3im 0.1;
            0.2+0.1im 0.0 1.0+0.1im]
        rng_lower = [0.8+0.1im 0.1im 0.0;
            0.0im 1.2+0.2im 0.1;
            0.0im 0.1 0.9+0.1im]
        prop.block_upper_ic[:, :, 1] .= rng_upper
        prop.block_upper_ic[:, :, 2] .= 0.5 * rng_upper
        prop.block_lower_ic[:, :, 1] .= 0.3 * rng_lower
        prop.block_lower_ic[:, :, 2] .= rng_lower

        odet = GeneralizedPerturbedEquilibrium.ForceFreeStates.OdeState(N, 10, 5, 0)
        u1_in = 0.5 * I(N) .+ 0.1im * ones(N, N)
        u2_in = I(N) .+ 0.2im * ones(N, N)
        odet.u[:, :, 1] .= u1_in
        odet.u[:, :, 2] .= u2_in

        GeneralizedPerturbedEquilibrium.ForceFreeStates.apply_propagator!(odet, prop)

        # Manual computation of expected result
        U1_upper = prop.block_upper_ic[:, :, 1]
        U2_upper = prop.block_upper_ic[:, :, 2]
        U1_lower = prop.block_lower_ic[:, :, 1]
        U2_lower = prop.block_lower_ic[:, :, 2]
        u1_expected = U1_upper * u1_in + U1_lower * u2_in
        u2_expected = U2_upper * u1_in + U2_lower * u2_in

        @test odet.u[:, :, 1] ≈ u1_expected rtol=1e-12
        @test odet.u[:, :, 2] ≈ u2_expected rtol=1e-12
    end

    @testset "apply_propagator_inverse! is inverse of apply_propagator!" begin
        # Verify that apply_propagator_inverse! is the algebraic inverse of apply_propagator!:
        # applying inverse then forward should recover the original state exactly.
        # This checks the LU-solve path: Φ \ (Φ * u) = u for an arbitrary invertible Φ.
        N = 3
        prop = GeneralizedPerturbedEquilibrium.ForceFreeStates.ChunkPropagator(N)

        # Near-identity blocks guarantee the 2N×2N matrix [A B; C D] is invertible
        A = I(N) .+ 0.15 * [1.0+0.2im 0.1im 0.05; 0.0im 0.9+0.3im 0.1; 0.2+0.1im 0.0 1.0+0.1im]
        B = 0.1 * [0.8+0.1im 0.1im 0.0; 0.0im 1.2+0.2im 0.1; 0.0im 0.1 0.9+0.1im]
        C = 0.1 * [0.5+0.1im 0.0im 0.1; 0.1im 0.8+0.2im 0.0; 0.0im 0.0 0.7+0.1im]
        D = I(N) .+ 0.15 * [0.9+0.1im 0.0im 0.05; 0.0im 1.0+0.2im 0.0; 0.1+0.1im 0.0 0.95+0.1im]

        prop.block_upper_ic[:, :, 1] .= A
        prop.block_lower_ic[:, :, 1] .= B
        prop.block_upper_ic[:, :, 2] .= C
        prop.block_lower_ic[:, :, 2] .= D

        u1_in = [1.0+0.5im 0.2im 0.0;
            0.1+0.1im 1.2+0.1im 0.0;
            0.0im 0.0 0.9+0.3im]
        u2_in = I(N) .+ 0.1im * ones(N, N)

        odet = GeneralizedPerturbedEquilibrium.ForceFreeStates.OdeState(N, 10, 5, 0)
        odet.u[:, :, 1] .= u1_in
        odet.u[:, :, 2] .= u2_in

        # Round-trip: inverse then forward = identity
        GeneralizedPerturbedEquilibrium.ForceFreeStates.apply_propagator_inverse!(odet, prop)
        GeneralizedPerturbedEquilibrium.ForceFreeStates.apply_propagator!(odet, prop)

        @test odet.u[:, :, 1] ≈ u1_in rtol=1e-12
        @test odet.u[:, :, 2] ≈ u2_in rtol=1e-12
    end

    @testset "balance_integration_chunks produces target count" begin
        # Verify that balance_integration_chunks creates at least
        # max(2*msing+3, 8*(msing+1)+msing) chunks from a small set of base chunks.
        ex = joinpath(@__DIR__, "test_data", "regression_solovev_ideal_example")
        inputs = TOML.parsefile(joinpath(ex, "gpec.toml"))
        intr = GeneralizedPerturbedEquilibrium.ForceFreeStates.ForceFreeStatesInternal()
        ctrl = GeneralizedPerturbedEquilibrium.force_balance_control(inputs, ex)
        eq_config = GeneralizedPerturbedEquilibrium.equilibrium_config(inputs, ex)
        equil = GeneralizedPerturbedEquilibrium.Equilibrium.PlasmaEquilibrium(
            eq_config,
            haskey(inputs, "SOL_INPUT") ? GeneralizedPerturbedEquilibrium.Equilibrium.SolovevConfig(inputs["SOL_INPUT"]) : nothing
        )
        intr.nlow = ctrl.nn_low;
        intr.nhigh = ctrl.nn_high;
        intr.npert = 1
        GeneralizedPerturbedEquilibrium.ForceFreeStates.sing_lim!(intr, ctrl, equil)
        GeneralizedPerturbedEquilibrium.ForceFreeStates.sing_find!(intr, equil)
        intr.mlow = min(intr.nlow * equil.params.qmin, 0) - 4 - ctrl.delta_mlow
        intr.mhigh = trunc(Int, intr.nhigh * equil.params.qmax) + ctrl.delta_mhigh
        intr.mpert = intr.mhigh - intr.mlow + 1
        intr.numpert_total = intr.mpert * intr.npert

        metric = GeneralizedPerturbedEquilibrium.ForceFreeStates.make_metric(equil, intr.mpert)
        mats = GeneralizedPerturbedEquilibrium.ForceFreeStates.build_matrix_splines(equil, intr, metric)

        odet = GeneralizedPerturbedEquilibrium.ForceFreeStates.OdeState(intr.numpert_total, ctrl.algorithm.numsteps_init, ctrl.algorithm.numunorms_init, intr.msing)
        GeneralizedPerturbedEquilibrium.ForceFreeStates.initialize_el_at_axis!(odet, ctrl, mats, equil.profiles, intr)

        base_chunks = GeneralizedPerturbedEquilibrium.ForceFreeStates.chunk_el_integration_bounds(odet, ctrl, intr)
        balanced = GeneralizedPerturbedEquilibrium.ForceFreeStates.balance_integration_chunks(base_chunks, ctrl, intr)

        # Must mirror balance_integration_chunks' internal target_n formula for nchunks = 0
        # (src/ForceFreeStates/EulerLagrange.jl). Keep this in sync. The formula reads only
        # intr.msing — no thread count enters it, which is what makes Riccati outputs
        # independent of how many threads `julia -t` provides.
        target_n = max(2 * intr.msing + 3, 8 * (intr.msing + 1) + intr.msing)

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
                @test balanced[i].psi_start ≈ balanced[i-1].psi_end rtol=1e-10
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
        ctrl_more = GeneralizedPerturbedEquilibrium.force_balance_control(merge(inputs, Dict("Forward" => merge(inputs["Forward"], Dict{String,Any}("nchunks" => target_n + 7)))), ex)
        balanced_more = GeneralizedPerturbedEquilibrium.ForceFreeStates.balance_integration_chunks(base_chunks, ctrl_more, intr)
        @test length(balanced_more) == target_n + 7

        # An nchunks below the singular-surface floor is clamped up, with a warning.
        min_chunks = 2 * intr.msing + 3
        ctrl_few = GeneralizedPerturbedEquilibrium.force_balance_control(merge(inputs, Dict("Forward" => merge(inputs["Forward"], Dict{String,Any}("nchunks" => 1)))), ex)
        balanced_few = @test_logs (:warn,) match_mode=:any GeneralizedPerturbedEquilibrium.ForceFreeStates.balance_integration_chunks(base_chunks, ctrl_few, intr)
        @test length(balanced_few) == max(min_chunks, length(base_chunks))
    end

    @testset "chunk_el_integration_bounds direction field — bidirectional mode" begin
        # Verify that bidirectional=true sets direction=-1 on crossing chunks and direction=+1
        # on non-crossing chunks, and that balance_integration_chunks propagates these correctly:
        # the right sub-chunk inherits direction from the parent, the left sub-chunk is always +1.
        ex = joinpath(@__DIR__, "test_data", "regression_solovev_ideal_example")
        inputs = TOML.parsefile(joinpath(ex, "gpec.toml"))
        intr = GeneralizedPerturbedEquilibrium.ForceFreeStates.ForceFreeStatesInternal()
        ctrl = GeneralizedPerturbedEquilibrium.force_balance_control(inputs, ex)
        eq_config = GeneralizedPerturbedEquilibrium.equilibrium_config(inputs, ex)
        equil = GeneralizedPerturbedEquilibrium.Equilibrium.PlasmaEquilibrium(
            eq_config,
            haskey(inputs, "SOL_INPUT") ? GeneralizedPerturbedEquilibrium.Equilibrium.SolovevConfig(inputs["SOL_INPUT"]) : nothing
        )
        intr.nlow = ctrl.nn_low;
        intr.nhigh = ctrl.nn_high;
        intr.npert = 1
        GeneralizedPerturbedEquilibrium.ForceFreeStates.sing_lim!(intr, ctrl, equil)
        GeneralizedPerturbedEquilibrium.ForceFreeStates.sing_find!(intr, equil)
        intr.mlow = min(intr.nlow * equil.params.qmin, 0) - 4 - ctrl.delta_mlow
        intr.mhigh = trunc(Int, intr.nhigh * equil.params.qmax) + ctrl.delta_mhigh
        intr.mpert = intr.mhigh - intr.mlow + 1
        intr.numpert_total = intr.mpert * intr.npert

        metric = GeneralizedPerturbedEquilibrium.ForceFreeStates.make_metric(equil, intr.mpert)
        mats = GeneralizedPerturbedEquilibrium.ForceFreeStates.build_matrix_splines(equil, intr, metric)

        odet = GeneralizedPerturbedEquilibrium.ForceFreeStates.OdeState(intr.numpert_total, ctrl.algorithm.numsteps_init, ctrl.algorithm.numunorms_init, intr.msing)
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
        ex = joinpath(@__DIR__, "test_data", "regression_solovev_ideal_example")

        function run_solovev(integrator)
            inputs = TOML.parsefile(joinpath(ex, "gpec.toml"))
            inputs["ForceFreeStates"]["integrator"] = integrator
            integrator == "Galerkin" ? delete!(inputs, "Forward") : (inputs[integrator] = pop!(inputs, "Forward"))
            intr = GeneralizedPerturbedEquilibrium.ForceFreeStates.ForceFreeStatesInternal()
            ctrl = GeneralizedPerturbedEquilibrium.force_balance_control(inputs, ex)
            eq_config = GeneralizedPerturbedEquilibrium.equilibrium_config(inputs, ex)
            equil = GeneralizedPerturbedEquilibrium.Equilibrium.PlasmaEquilibrium(
                eq_config,
                haskey(inputs, "SOL_INPUT") ? GeneralizedPerturbedEquilibrium.Equilibrium.SolovevConfig(inputs["SOL_INPUT"]) : nothing
            )
            intr.wall_settings = GeneralizedPerturbedEquilibrium.Vacuum.parse_wall(get(inputs, "Wall", nothing))
            intr.nlow = ctrl.nn_low;
            intr.nhigh = ctrl.nn_high;
            intr.npert = 1
            GeneralizedPerturbedEquilibrium.ForceFreeStates.sing_lim!(intr, ctrl, equil)
            GeneralizedPerturbedEquilibrium.ForceFreeStates.sing_find!(intr, equil)
            intr.mlow = min(intr.nlow * equil.params.qmin, 0) - 4 - ctrl.delta_mlow
            intr.mhigh = trunc(Int, intr.nhigh * equil.params.qmax) + ctrl.delta_mhigh
            intr.mpert = intr.mhigh - intr.mlow + 1
            intr.numpert_total = intr.mpert * intr.npert
            metric = GeneralizedPerturbedEquilibrium.ForceFreeStates.make_metric(equil, intr.mpert)
            mats = GeneralizedPerturbedEquilibrium.ForceFreeStates.build_matrix_splines(equil, intr, metric)
            odet, _, _, _ = GeneralizedPerturbedEquilibrium.ForceFreeStates.eulerlagrange_integration(ctrl, equil, mats, intr)
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
        ex = joinpath(@__DIR__, "test_data", "regression_solovev_ideal_example")
        inputs = TOML.parsefile(joinpath(ex, "gpec.toml"))
        intr = GeneralizedPerturbedEquilibrium.ForceFreeStates.ForceFreeStatesInternal()
        ctrl = GeneralizedPerturbedEquilibrium.force_balance_control(inputs, ex)
        eq_config = GeneralizedPerturbedEquilibrium.equilibrium_config(inputs, ex)
        equil = GeneralizedPerturbedEquilibrium.Equilibrium.PlasmaEquilibrium(
            eq_config,
            haskey(inputs, "SOL_INPUT") ? GeneralizedPerturbedEquilibrium.Equilibrium.SolovevConfig(inputs["SOL_INPUT"]) : nothing
        )
        intr.nlow = ctrl.nn_low;
        intr.nhigh = ctrl.nn_high;
        intr.npert = 1
        GeneralizedPerturbedEquilibrium.ForceFreeStates.sing_lim!(intr, ctrl, equil)
        GeneralizedPerturbedEquilibrium.ForceFreeStates.sing_find!(intr, equil)
        intr.mlow = min(intr.nlow * equil.params.qmin, 0) - 4 - ctrl.delta_mlow
        intr.mhigh = trunc(Int, intr.nhigh * equil.params.qmax) + ctrl.delta_mhigh
        intr.mpert = intr.mhigh - intr.mlow + 1
        intr.numpert_total = intr.mpert * intr.npert

        metric = GeneralizedPerturbedEquilibrium.ForceFreeStates.make_metric(equil, intr.mpert)
        mats = GeneralizedPerturbedEquilibrium.ForceFreeStates.build_matrix_splines(equil, intr, metric)

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

    @testset "Riccati leaves a sparse u_store in the Riccati basis" begin
        # The Riccati path stores only chunk-endpoint snapshots, so u_store / psi_store have
        # strictly fewer entries than the forward path's dense saved steps, and the stored
        # state is not in the Euler-Lagrange axis basis. Downstream ξ consumers must therefore
        # be fed by the forward integrator; this test pins that contract.
        function run_and_capture(example_dir, integrator)
            inputs = TOML.parsefile(joinpath(example_dir, "gpec.toml"))
            inputs["ForceFreeStates"]["integrator"] = integrator
            integrator == "Galerkin" ? delete!(inputs, "Forward") : (inputs[integrator] = pop!(inputs, "Forward"))
            intr = GeneralizedPerturbedEquilibrium.ForceFreeStates.ForceFreeStatesInternal()
            ctrl = GeneralizedPerturbedEquilibrium.force_balance_control(inputs, example_dir)
            eq_config = GeneralizedPerturbedEquilibrium.equilibrium_config(inputs, example_dir)
            equil = GeneralizedPerturbedEquilibrium.Equilibrium.PlasmaEquilibrium(
                eq_config,
                haskey(inputs, "SOL_INPUT") ? GeneralizedPerturbedEquilibrium.Equilibrium.SolovevConfig(inputs["SOL_INPUT"]) : nothing
            )
            intr.wall_settings = GeneralizedPerturbedEquilibrium.Vacuum.parse_wall(get(inputs, "Wall", nothing))
            intr.nlow = ctrl.nn_low;
            intr.nhigh = ctrl.nn_high;
            intr.npert = 1
            GeneralizedPerturbedEquilibrium.ForceFreeStates.sing_lim!(intr, ctrl, equil)
            GeneralizedPerturbedEquilibrium.ForceFreeStates.sing_find!(intr, equil)
            intr.mlow = min(intr.nlow * equil.params.qmin, 0) - 4 - ctrl.delta_mlow
            intr.mhigh = trunc(Int, intr.nhigh * equil.params.qmax) + ctrl.delta_mhigh
            intr.mpert = intr.mhigh - intr.mlow + 1
            intr.numpert_total = intr.mpert * intr.npert
            metric = GeneralizedPerturbedEquilibrium.ForceFreeStates.make_metric(equil, intr.mpert)
            mats = GeneralizedPerturbedEquilibrium.ForceFreeStates.build_matrix_splines(equil, intr, metric)
            odet, _, _, _ = GeneralizedPerturbedEquilibrium.ForceFreeStates.eulerlagrange_integration(ctrl, equil, mats, intr)
            # Derivatives are recomputed on demand; materialize so the stores can be compared.
            GeneralizedPerturbedEquilibrium.ForceFreeStates.materialize_derivative_stores!(odet, equil, mats, intr)
            return odet
        end

        ex = joinpath(@__DIR__, "test_data", "regression_solovev_ideal_example")
        odet_fwd = run_and_capture(ex, "Forward")
        odet_ric = run_and_capture(ex, "Riccati")

        @test odet_ric.step < odet_fwd.step
        @test length(odet_ric.psi_store) < length(odet_fwd.psi_store)

        # The forward path returns dense ξ in the axis basis, ready for PerturbedEquilibrium.
        @test odet_fwd.u_store_el_basis
        @test odet_fwd.du_store_populated

        # The Riccati solution is in the Riccati basis, so the derivative stores cannot be
        # materialized from it and stay empty rather than holding unusable values.
        @test !odet_ric.u_store_el_basis
        @test !odet_ric.du_store_populated
        @test isempty(odet_ric.du_store)
        @test isempty(odet_ric.xi_s_store)
    end

    # Production DIII-D energies and Δ′ are tracked by the diiid_n1_riccati harness cases.

end
