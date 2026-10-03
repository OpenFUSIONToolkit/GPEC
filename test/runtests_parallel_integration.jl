using LinearAlgebra
using TOML

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
        inputs["ForceFreeStates"]["verbose"] = false
        intr = GeneralizedPerturbedEquilibrium.ForceFreeStates.ForceFreeStatesInternal(; dir_path=ex)
        ctrl = GeneralizedPerturbedEquilibrium.ForceFreeStates.ForceFreeStatesControl(;
            (Symbol(k) => v for (k, v) in inputs["ForceFreeStates"])...)
        eq_config = GeneralizedPerturbedEquilibrium.Equilibrium.EquilibriumConfig(inputs["Equilibrium"], ex)
        equil = GeneralizedPerturbedEquilibrium.Equilibrium.setup_equilibrium(eq_config, haskey(inputs, "SOL_INPUT") ? GeneralizedPerturbedEquilibrium.Equilibrium.SolovevConfig(inputs["SOL_INPUT"]) : nothing)
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

        odet = GeneralizedPerturbedEquilibrium.ForceFreeStates.OdeState(intr.numpert_total, ctrl.numsteps_init, ctrl.numunorms_init, intr.msing)
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
        ctrl_more = GeneralizedPerturbedEquilibrium.ForceFreeStates.ForceFreeStatesControl(;
            (Symbol(k) => v for (k, v) in inputs["ForceFreeStates"])..., nchunks=target_n + 7)
        balanced_more = GeneralizedPerturbedEquilibrium.ForceFreeStates.balance_integration_chunks(base_chunks, ctrl_more, intr)
        @test length(balanced_more) == target_n + 7

        # An nchunks below the singular-surface floor is clamped up, with a warning.
        min_chunks = 2 * intr.msing + 3
        ctrl_few = GeneralizedPerturbedEquilibrium.ForceFreeStates.ForceFreeStatesControl(;
            (Symbol(k) => v for (k, v) in inputs["ForceFreeStates"])..., nchunks=1)
        balanced_few = @test_logs (:warn,) match_mode=:any GeneralizedPerturbedEquilibrium.ForceFreeStates.balance_integration_chunks(base_chunks, ctrl_few, intr)
        @test length(balanced_few) == max(min_chunks, length(base_chunks))
    end

    @testset "chunk_el_integration_bounds direction field — bidirectional mode" begin
        # Verify that bidirectional=true sets direction=-1 on crossing chunks and direction=+1
        # on non-crossing chunks, and that balance_integration_chunks propagates these correctly:
        # the right sub-chunk inherits direction from the parent, the left sub-chunk is always +1.
        ex = joinpath(@__DIR__, "test_data", "regression_solovev_ideal_example")
        inputs = TOML.parsefile(joinpath(ex, "gpec.toml"))
        inputs["ForceFreeStates"]["verbose"] = false
        intr = GeneralizedPerturbedEquilibrium.ForceFreeStates.ForceFreeStatesInternal(; dir_path=ex)
        ctrl = GeneralizedPerturbedEquilibrium.ForceFreeStates.ForceFreeStatesControl(;
            (Symbol(k) => v for (k, v) in inputs["ForceFreeStates"])...)
        eq_config = GeneralizedPerturbedEquilibrium.Equilibrium.EquilibriumConfig(inputs["Equilibrium"], ex)
        equil = GeneralizedPerturbedEquilibrium.Equilibrium.setup_equilibrium(eq_config, haskey(inputs, "SOL_INPUT") ? GeneralizedPerturbedEquilibrium.Equilibrium.SolovevConfig(inputs["SOL_INPUT"]) : nothing)
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

        odet = GeneralizedPerturbedEquilibrium.ForceFreeStates.OdeState(intr.numpert_total, ctrl.numsteps_init, ctrl.numunorms_init, intr.msing)
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
        # (Solovev N=8, tested here) and large-N (DIIID N=26, tested below).
        ex = joinpath(@__DIR__, "test_data", "regression_solovev_ideal_example")

        function run_solovev(integrator)
            inputs = TOML.parsefile(joinpath(ex, "gpec.toml"))
            inputs["ForceFreeStates"]["verbose"] = false
            inputs["ForceFreeStates"]["integrator"] = integrator
            intr = GeneralizedPerturbedEquilibrium.ForceFreeStates.ForceFreeStatesInternal(; dir_path=ex)
            ctrl = GeneralizedPerturbedEquilibrium.ForceFreeStates.ForceFreeStatesControl(;
                (Symbol(k) => v for (k, v) in inputs["ForceFreeStates"])...)
            eq_config = GeneralizedPerturbedEquilibrium.Equilibrium.EquilibriumConfig(inputs["Equilibrium"], ex)
            equil = GeneralizedPerturbedEquilibrium.Equilibrium.setup_equilibrium(eq_config, haskey(inputs, "SOL_INPUT") ? GeneralizedPerturbedEquilibrium.Equilibrium.SolovevConfig(inputs["SOL_INPUT"]) : nothing)
            intr.wall_settings = GeneralizedPerturbedEquilibrium.Vacuum.WallShapeSettings(;
                (Symbol(k) => v for (k, v) in inputs["Wall"])...)
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

        et_fwd, intr_fwd = run_solovev("forward")
        et_ric, intr_ric = run_solovev("riccati")

        # Energy eigenvalue matches to 2%
        @test isapprox(et_ric, et_fwd; rtol=0.02)
        # Per-surface Δ' assertions were removed: per-surface Δ' is a stub calculation
        # left in the code for future work but no longer reported, output, or tested.
        # The STRIDE BVP Δ' matrix (`SingularSurfaces/Delta_prime_matrix`) is the canonical
        # Δ', regression-tested via the DIIID-like fixture which has well-conditioned
        # values; Solovev is near marginal stability and BVP Δ' is pathological there.
    end

    @testset "Riccati FM integration matches forward ODE — DIIID-like example (large N)" begin
        # Run forward and Riccati FM integrations on the DIIID-like example (N≈26 modes).
        # Before bidirectional integration, the all-forward FM propagators were ill-conditioned
        # for large N, producing ~10% energy error. Bidirectional integration (backward crossing
        # chunks + forward intermediate chunks) restores accuracy to within 2%.
        #
        # This is the key regression test for the bidirectional parallel FM fix.
        ex = joinpath(@__DIR__, "..", "examples", "DIIID-like_ideal_example")

        function run_diiid(integrator)
            inputs = TOML.parsefile(joinpath(ex, "gpec.toml"))
            inputs["ForceFreeStates"]["verbose"] = false
            inputs["ForceFreeStates"]["integrator"] = integrator
            inputs["ForceFreeStates"]["write_outputs_to_HDF5"] = false
            intr = GeneralizedPerturbedEquilibrium.ForceFreeStates.ForceFreeStatesInternal(; dir_path=ex)
            ctrl = GeneralizedPerturbedEquilibrium.ForceFreeStates.ForceFreeStatesControl(;
                (Symbol(k) => v for (k, v) in inputs["ForceFreeStates"])...)
            eq_config = GeneralizedPerturbedEquilibrium.Equilibrium.EquilibriumConfig(inputs["Equilibrium"], ex)
            equil = GeneralizedPerturbedEquilibrium.Equilibrium.setup_equilibrium(eq_config, haskey(inputs, "SOL_INPUT") ? GeneralizedPerturbedEquilibrium.Equilibrium.SolovevConfig(inputs["SOL_INPUT"]) : nothing)
            # Apply the two-pass auto grid exactly as the main driver does (the example ships
            # grid_type="auto", mpsi=0): measured-curvature refinement with rational surfaces
            # pinned as mandatory knots, re-formed from the captured ingest. The pinned values
            # below are for this grid, which is the production default.
            if GeneralizedPerturbedEquilibrium.Equilibrium.wants_two_pass(eq_config)
                mand = GeneralizedPerturbedEquilibrium.ForceFreeStates.rational_psi_nodes(equil; nlow=ctrl.nn_low, nhigh=ctrl.nn_high)
                psi_nodes = GeneralizedPerturbedEquilibrium.Equilibrium.refined_psi_grid(equil; tau=eq_config.psi_accuracy, mandatory=mand)
                rerun_input = GeneralizedPerturbedEquilibrium.Equilibrium.build_direct_from_ingest(eq_config, equil.ingest)
                equil = GeneralizedPerturbedEquilibrium.Equilibrium.setup_equilibrium(eq_config, rerun_input; override_psi_nodes=psi_nodes)
            end
            intr.wall_settings = GeneralizedPerturbedEquilibrium.Vacuum.WallShapeSettings(;
                (Symbol(k) => v for (k, v) in inputs["Wall"])...)
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

        et_ric, intr_ric = run_diiid("riccati")

        # Riccati FM et[1] regression — pinned tightly, NOT bracketed. et[1] is grid- and
        # equilibrium-sensitive (auto-mpsi gives a spurious value; a wrong grid/Ip shifts it), so
        # a loose bracket would mask exactly that accuracy regression. The Riccati-path value is
        # deterministic and reproducible.
        @test isapprox(et_ric, 0.800637; rtol=2e-2)
        # Per-surface Δ' assertions removed (stub calculation; see Solovev testset
        # comment above). BVP Δ' matrix regression for DIIID-like is in the
        # `delta_prime_matrix — STRIDE BVP DIIID-like regression (large N)` testset.

        # No explicit Riccati-vs-forward cross-path check here: the two paths share the
        # equilibrium grid (so a cross-path comparison is blind to grid/accuracy regressions),
        # and their agreement is already verified on the lighter Solovev case above. The tight
        # absolute pin above is the guard for grid/equilibrium regressions on this case.
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
        inputs["ForceFreeStates"]["verbose"] = false
        intr = GeneralizedPerturbedEquilibrium.ForceFreeStates.ForceFreeStatesInternal(; dir_path=ex)
        ctrl = GeneralizedPerturbedEquilibrium.ForceFreeStates.ForceFreeStatesControl(;
            (Symbol(k) => v for (k, v) in inputs["ForceFreeStates"])...)
        eq_config = GeneralizedPerturbedEquilibrium.Equilibrium.EquilibriumConfig(inputs["Equilibrium"], ex)
        equil = GeneralizedPerturbedEquilibrium.Equilibrium.setup_equilibrium(eq_config, haskey(inputs, "SOL_INPUT") ? GeneralizedPerturbedEquilibrium.Equilibrium.SolovevConfig(inputs["SOL_INPUT"]) : nothing)
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
            inputs["ForceFreeStates"]["verbose"] = false
            inputs["ForceFreeStates"]["integrator"] = integrator
            inputs["ForceFreeStates"]["write_outputs_to_HDF5"] = false
            intr = GeneralizedPerturbedEquilibrium.ForceFreeStates.ForceFreeStatesInternal(; dir_path=example_dir)
            ctrl = GeneralizedPerturbedEquilibrium.ForceFreeStates.ForceFreeStatesControl(;
                (Symbol(k) => v for (k, v) in inputs["ForceFreeStates"])...)
            eq_config = GeneralizedPerturbedEquilibrium.Equilibrium.EquilibriumConfig(inputs["Equilibrium"], example_dir)
            equil = GeneralizedPerturbedEquilibrium.Equilibrium.setup_equilibrium(eq_config, haskey(inputs, "SOL_INPUT") ? GeneralizedPerturbedEquilibrium.Equilibrium.SolovevConfig(inputs["SOL_INPUT"]) : nothing)
            intr.wall_settings = GeneralizedPerturbedEquilibrium.Vacuum.WallShapeSettings(;
                (Symbol(k) => v for (k, v) in inputs["Wall"])...)
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
        odet_fwd = run_and_capture(ex, "forward")
        odet_ric = run_and_capture(ex, "riccati")

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

    @testset "delta_prime_matrix — STRIDE BVP DIIID-like regression (large N)" begin
        # Verify that the parallel FM path computes a well-formed inter-surface Δ' matrix
        # for the DIIID-like case (N≈26 modes, multiple rational surfaces). This complements
        # the Solovev test above by exercising the BVP assembly with more surfaces and larger
        # mode space, where ill-conditioned (non-bidirectional) FM propagators would fail.
        ex = joinpath(@__DIR__, "..", "examples", "DIIID-like_ideal_example")
        inputs = TOML.parsefile(joinpath(ex, "gpec.toml"))
        inputs["ForceFreeStates"]["verbose"] = false
        inputs["ForceFreeStates"]["integrator"] = "riccati"
        inputs["ForceFreeStates"]["write_outputs_to_HDF5"] = false
        intr = GeneralizedPerturbedEquilibrium.ForceFreeStates.ForceFreeStatesInternal(; dir_path=ex)
        ctrl = GeneralizedPerturbedEquilibrium.ForceFreeStates.ForceFreeStatesControl(;
            (Symbol(k) => v for (k, v) in inputs["ForceFreeStates"])...)
        eq_config = GeneralizedPerturbedEquilibrium.Equilibrium.EquilibriumConfig(inputs["Equilibrium"], ex)
        equil = GeneralizedPerturbedEquilibrium.Equilibrium.setup_equilibrium(eq_config, haskey(inputs, "SOL_INPUT") ? GeneralizedPerturbedEquilibrium.Equilibrium.SolovevConfig(inputs["SOL_INPUT"]) : nothing)
        # Apply the two-pass auto grid (measured-curvature refinement, rational surfaces pinned
        # as mandatory knots) exactly as the main driver does (see the FM testset above); the
        # pinned values below are for this grid, the production default.
        if GeneralizedPerturbedEquilibrium.Equilibrium.wants_two_pass(eq_config)
            mand = GeneralizedPerturbedEquilibrium.ForceFreeStates.rational_psi_nodes(equil; nlow=ctrl.nn_low, nhigh=ctrl.nn_high)
            psi_nodes = GeneralizedPerturbedEquilibrium.Equilibrium.refined_psi_grid(equil; tau=eq_config.psi_accuracy, mandatory=mand)
            rerun_input = GeneralizedPerturbedEquilibrium.Equilibrium.build_direct_from_ingest(eq_config, equil.ingest)
            equil = GeneralizedPerturbedEquilibrium.Equilibrium.setup_equilibrium(eq_config, rerun_input; override_psi_nodes=psi_nodes)
        end
        intr.wall_settings = GeneralizedPerturbedEquilibrium.Vacuum.WallShapeSettings(;
            (Symbol(k) => v for (k, v) in inputs["Wall"])...)
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
        odet, fm_propagators, fm_chunks, fm_S_left =
            GeneralizedPerturbedEquilibrium.ForceFreeStates.eulerlagrange_integration(ctrl, equil, mats, intr)
        vac = GeneralizedPerturbedEquilibrium.ForceFreeStates.free_run(odet, ctrl, equil, mats, intr)
        GeneralizedPerturbedEquilibrium.ForceFreeStates.compute_delta_prime_matrix!(
            intr, fm_propagators, fm_chunks;
            wv=vac.wv, psio=equil.psio,
            S_at_surface_left=fm_S_left, ctrl=ctrl, equil=equil, mats=mats)

        msing = intr.msing
        dpm = intr.delta_prime_matrix

        # Matrix is populated with correct shape (msing × msing); see Solovev test above
        # for why this is msing × msing rather than 2·msing × 2·msing.
        @test !isempty(dpm)
        @test size(dpm) == (msing, msing)

        # All elements are finite
        @test all(isfinite, dpm)

        # Diagonal (self-response) elements are non-zero
        for j in 1:msing
            @test abs(dpm[j, j]) > 1e-10
        end

        # Pinned diagonal `delta_prime_matrix` REAL parts, PEST3-convention self-response Δ' from
        # the STRIDE BVP with vacuum coupling, on the two-pass measured-curvature grid (rational
        # surfaces now *bracketed* rather than pinned on — see `bracket_mandatory_nodes`).
        #
        # The q=2 pin is now a CONVERGED value, which closes the "replace this with a converged-Δ′
        # pin on a fixed ldp grid" item the previous pin carried. Three independent grid families
        # on this fixture agree to ~1%:
        #     ldp     mpsi 512 / 1024 / 2048   :  8.478 / 9.142 / 9.244
        #     uniform mpsi 1024 / 2048         :  9.146 / 9.143
        #     auto    τ 2e-3/1e-3/5e-4/2.5e-4  :  9.198 / 9.239 / 9.206 / 9.157
        # i.e. dpm[1,1] → ≈9.2. Before this grid fix the τ-scan swung ~50% (6.2–9.9) and the pin
        # had to be a mid-spread snapshot; it is now flat in τ to 0.9%, so the default
        # psi_accuracy lands on the converged value rather than near it by luck.
        # See the "pinning grid-sensitive Δ′ robustly" open problem in
        # docs/src/developer_notes.md for the plateau criterion meant to replace single-point pins.
        # (et[1], NTV torque, and ‖resonant flux‖ stay grid-robust to <1% — the sensitivity is
        # local to the singular-layer matching, not the global response.) Only real parts are
        # pinned; the imaginary parts are dominated by the PEST3 four-term cancellation and are
        # FP/platform-sensitive. Near-separatrix surfaces q=5,6 keep only the finiteness/non-zero
        # checks above. Values use this testset's mode range (mpert=27, vs full-pipeline mpert=35).
        #
        # The q=2 platform spread also collapsed with this fix: because the matching stencil no
        # longer straddles a knot, q‴ is single-valued across it and the surface no longer sits on
        # the steepest part of the refinement curve. Same grid and inputs now give 9.239
        # (macOS/aarch64), 9.333 (linux/julia 1.11) and 9.356 (linux/julia 1.x) — 1.3% end to end,
        # against ~25% before (6.14 / 6.69 / 7.70). rtol=3e-2 spans that with margin and is a 5×
        # TIGHTENING of the old 1.5e-1. q=3 and q=4 are unchanged: the same ladder puts them at
        # −5.716 and −16.033 (ldp 2048), inside their existing tolerances, and they reproduce
        # across platforms.
        @test isapprox(real(dpm[1, 1]), +9.300000e+00; rtol=3e-2)   # q=2, converged (ldp2048 9.244)
        @test isapprox(real(dpm[2, 2]), -5.344199e+00; rtol=1e-1)   # q=3
        @test isapprox(real(dpm[3, 3]), -1.590034e+01; rtol=1e-1)   # q=4
    end

end
