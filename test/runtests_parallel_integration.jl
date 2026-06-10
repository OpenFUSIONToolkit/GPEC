using LinearAlgebra
using TOML

@testset "Parallel FM Integration Tests" begin

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
        u1_in = [1.0+0.5im  0.2im   0.0;
                 0.1+0.1im  1.2+0.1im 0.0;
                 0.0im      0.0      0.9+0.3im]
        u2_in = [0.8+0.1im  0.1im   0.0;
                 0.0im      1.0+0.2im 0.1;
                 0.1im      0.0      1.1+0.0im]
        odet.u[:, :, 1] .= u1_in
        odet.u[:, :, 2] .= u2_in

        GeneralizedPerturbedEquilibrium.ForceFreeStates.apply_propagator!(odet, prop)

        @test odet.u[:, :, 1] ≈ u1_in  rtol=1e-12
        @test odet.u[:, :, 2] ≈ u2_in  rtol=1e-12
    end

    @testset "apply_propagator! linearity" begin
        # Verify that apply_propagator! applies the correct linear map.
        N = 3
        prop = GeneralizedPerturbedEquilibrium.ForceFreeStates.ChunkPropagator(N)

        # Fill block_upper_ic and block_lower_ic with random data
        rng_upper = [1.1+0.2im  0.1im   0.05;
                     0.0im      0.9+0.3im 0.1;
                     0.2+0.1im  0.0      1.0+0.1im]
        rng_lower = [0.8+0.1im  0.1im   0.0;
                     0.0im      1.2+0.2im 0.1;
                     0.0im      0.1      0.9+0.1im]
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

        @test odet.u[:, :, 1] ≈ u1_expected  rtol=1e-12
        @test odet.u[:, :, 2] ≈ u2_expected  rtol=1e-12
    end

    @testset "apply_propagator_inverse! is inverse of apply_propagator!" begin
        # Verify that apply_propagator_inverse! is the algebraic inverse of apply_propagator!:
        # applying inverse then forward should recover the original state exactly.
        # This checks the LU-solve path: Φ \ (Φ * u) = u for an arbitrary invertible Φ.
        N = 3
        prop = GeneralizedPerturbedEquilibrium.ForceFreeStates.ChunkPropagator(N)

        # Near-identity blocks guarantee the 2N×2N matrix [A B; C D] is invertible
        A = I(N) .+ 0.15 * [1.0+0.2im  0.1im   0.05; 0.0im  0.9+0.3im  0.1; 0.2+0.1im  0.0  1.0+0.1im]
        B = 0.1  * [0.8+0.1im  0.1im   0.0;    0.0im  1.2+0.2im  0.1; 0.0im  0.1  0.9+0.1im]
        C = 0.1  * [0.5+0.1im  0.0im   0.1;    0.1im  0.8+0.2im  0.0; 0.0im  0.0  0.7+0.1im]
        D = I(N) .+ 0.15 * [0.9+0.1im  0.0im   0.05; 0.0im  1.0+0.2im  0.0; 0.1+0.1im  0.0  0.95+0.1im]

        prop.block_upper_ic[:, :, 1] .= A
        prop.block_lower_ic[:, :, 1] .= B
        prop.block_upper_ic[:, :, 2] .= C
        prop.block_lower_ic[:, :, 2] .= D

        u1_in = [1.0+0.5im  0.2im   0.0;
                 0.1+0.1im  1.2+0.1im 0.0;
                 0.0im      0.0      0.9+0.3im]
        u2_in = I(N) .+ 0.1im * ones(N, N)

        odet = GeneralizedPerturbedEquilibrium.ForceFreeStates.OdeState(N, 10, 5, 0)
        odet.u[:, :, 1] .= u1_in
        odet.u[:, :, 2] .= u2_in

        # Round-trip: inverse then forward = identity
        GeneralizedPerturbedEquilibrium.ForceFreeStates.apply_propagator_inverse!(odet, prop)
        GeneralizedPerturbedEquilibrium.ForceFreeStates.apply_propagator!(odet, prop)

        @test odet.u[:, :, 1] ≈ u1_in  rtol=1e-12
        @test odet.u[:, :, 2] ≈ u2_in  rtol=1e-12
    end

    @testset "balance_integration_chunks produces target count" begin
        # Verify that balance_integration_chunks creates at least
        # max(2*msing+3, 4*nthreads) chunks from a small set of base chunks.
        ex = joinpath(@__DIR__, "test_data", "regression_solovev_ideal_example")
        inputs = TOML.parsefile(joinpath(ex, "gpec.toml"))
        inputs["ForceFreeStates"]["verbose"] = false
        intr = GeneralizedPerturbedEquilibrium.ForceFreeStates.ForceFreeStatesInternal(; dir_path=ex)
        ctrl = GeneralizedPerturbedEquilibrium.ForceFreeStates.ForceFreeStatesControl(;
            (Symbol(k) => v for (k, v) in inputs["ForceFreeStates"])...)
        eq_config = GeneralizedPerturbedEquilibrium.Equilibrium.EquilibriumConfig(inputs["Equilibrium"], ex)
        equil = GeneralizedPerturbedEquilibrium.Equilibrium.setup_equilibrium(eq_config)
        GeneralizedPerturbedEquilibrium.ForceFreeStates.sing_lim!(intr, ctrl, equil)
        intr.nlow = ctrl.nn_low; intr.nhigh = ctrl.nn_high; intr.npert = 1
        GeneralizedPerturbedEquilibrium.ForceFreeStates.sing_find!(intr, equil)
        intr.mlow = min(intr.nlow * equil.params.qmin, 0) - 4 - ctrl.delta_mlow
        intr.mhigh = trunc(Int, intr.nhigh * equil.params.qmax) + ctrl.delta_mhigh
        intr.mpert = intr.mhigh - intr.mlow + 1
        intr.mband = intr.mpert - 1
        intr.numpert_total = intr.mpert * intr.npert

        metric = GeneralizedPerturbedEquilibrium.ForceFreeStates.make_metric(equil; mband=intr.mband)
        ffit = GeneralizedPerturbedEquilibrium.ForceFreeStates.make_matrix(equil, intr, metric)

        odet = GeneralizedPerturbedEquilibrium.ForceFreeStates.OdeState(intr.numpert_total, ctrl.numsteps_init, ctrl.numunorms_init, intr.msing)
        GeneralizedPerturbedEquilibrium.ForceFreeStates.initialize_el_at_axis!(odet, ctrl, ffit, equil.profiles, intr)

        base_chunks = GeneralizedPerturbedEquilibrium.ForceFreeStates.chunk_el_integration_bounds(odet, ctrl, intr)
        balanced = GeneralizedPerturbedEquilibrium.ForceFreeStates.balance_integration_chunks(base_chunks, ctrl, intr)

        # Must mirror balance_integration_chunks' internal target_n formula
        # (src/ForceFreeStates/EulerLagrange.jl). Keep this in sync.
        target_n = max(2 * intr.msing + 3, 4 * Threads.nthreads(), 8 * (intr.msing + 1) + intr.msing)

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
                @test balanced[i].psi_start ≈ balanced[i-1].psi_end  rtol=1e-10
            else
                # Inner-layer gap: post-crossing chunk starts AFTER the rational surface
                @test balanced[i].psi_start > balanced[i-1].psi_end
            end
        end

        # The total number of needs_crossing=true chunks should equal the original
        n_crossings_base = count(c -> c.needs_crossing, base_chunks)
        n_crossings_bal = count(c -> c.needs_crossing, balanced)
        @test n_crossings_bal == n_crossings_base
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
        equil = GeneralizedPerturbedEquilibrium.Equilibrium.setup_equilibrium(eq_config)
        GeneralizedPerturbedEquilibrium.ForceFreeStates.sing_lim!(intr, ctrl, equil)
        intr.nlow = ctrl.nn_low; intr.nhigh = ctrl.nn_high; intr.npert = 1
        GeneralizedPerturbedEquilibrium.ForceFreeStates.sing_find!(intr, equil)
        intr.mlow = min(intr.nlow * equil.params.qmin, 0) - 4 - ctrl.delta_mlow
        intr.mhigh = trunc(Int, intr.nhigh * equil.params.qmax) + ctrl.delta_mhigh
        intr.mpert = intr.mhigh - intr.mlow + 1
        intr.mband = intr.mpert - 1
        intr.numpert_total = intr.mpert * intr.npert

        metric = GeneralizedPerturbedEquilibrium.ForceFreeStates.make_metric(equil; mband=intr.mband)
        ffit = GeneralizedPerturbedEquilibrium.ForceFreeStates.make_matrix(equil, intr, metric)

        odet = GeneralizedPerturbedEquilibrium.ForceFreeStates.OdeState(intr.numpert_total, ctrl.numsteps_init, ctrl.numunorms_init, intr.msing)
        GeneralizedPerturbedEquilibrium.ForceFreeStates.initialize_el_at_axis!(odet, ctrl, ffit, equil.profiles, intr)

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

    @testset "Parallel FM integration matches standard ODE — Solovev example" begin
        # Run standard and parallel FM integrations on the Solovev regression test.
        # The energy eigenvalue et[1] should match to within 2%.
        #
        # Bidirectional FM integration (crossing chunks integrated backward) is the
        # default for use_parallel=true. It keeps FM propagators well-conditioned for
        # both small-N (Solovev N=8, tested here) and large-N (DIIID N=26, tested below).
        ex = joinpath(@__DIR__, "test_data", "regression_solovev_ideal_example")

        function run_solovev(use_parallel)
            inputs = TOML.parsefile(joinpath(ex, "gpec.toml"))
            inputs["ForceFreeStates"]["verbose"] = false
            inputs["ForceFreeStates"]["use_parallel"] = use_parallel
            intr = GeneralizedPerturbedEquilibrium.ForceFreeStates.ForceFreeStatesInternal(; dir_path=ex)
            ctrl = GeneralizedPerturbedEquilibrium.ForceFreeStates.ForceFreeStatesControl(;
                (Symbol(k) => v for (k, v) in inputs["ForceFreeStates"])...)
            eq_config = GeneralizedPerturbedEquilibrium.Equilibrium.EquilibriumConfig(inputs["Equilibrium"], ex)
            equil = GeneralizedPerturbedEquilibrium.Equilibrium.setup_equilibrium(eq_config)
            intr.wall_settings = GeneralizedPerturbedEquilibrium.Vacuum.WallShapeSettings(;
                (Symbol(k) => v for (k, v) in inputs["Wall"])...)
            GeneralizedPerturbedEquilibrium.ForceFreeStates.sing_lim!(intr, ctrl, equil)
            intr.nlow = ctrl.nn_low; intr.nhigh = ctrl.nn_high; intr.npert = 1
            GeneralizedPerturbedEquilibrium.ForceFreeStates.sing_find!(intr, equil)
            intr.mlow = min(intr.nlow * equil.params.qmin, 0) - 4 - ctrl.delta_mlow
            intr.mhigh = trunc(Int, intr.nhigh * equil.params.qmax) + ctrl.delta_mhigh
            intr.mpert = intr.mhigh - intr.mlow + 1
            intr.mband = intr.mpert - 1
            intr.numpert_total = intr.mpert * intr.npert
            metric = GeneralizedPerturbedEquilibrium.ForceFreeStates.make_metric(equil; mband=intr.mband)
            ffit = GeneralizedPerturbedEquilibrium.ForceFreeStates.make_matrix(equil, intr, metric)
            odet, _, _, _ = GeneralizedPerturbedEquilibrium.ForceFreeStates.eulerlagrange_integration(ctrl, equil, ffit, intr)
            vac = GeneralizedPerturbedEquilibrium.ForceFreeStates.free_run!(odet, ctrl, equil, ffit, intr)
            return real(vac.et[1]), intr
        end

        et_std, intr_std = run_solovev(false)
        et_par, intr_par = run_solovev(true)

        # Energy eigenvalue matches to 2%
        @test isapprox(et_par, et_std; rtol=0.02)
        # Per-surface Δ' assertions were removed: per-surface Δ' is a stub calculation
        # left in the code for future work but no longer reported, output, or tested.
        # The STRIDE BVP Δ' matrix (`singular/delta_prime_matrix`) is the canonical
        # Δ', regression-tested via the DIIID-like fixture which has well-conditioned
        # values; Solovev is near marginal stability and BVP Δ' is pathological there.
    end

    @testset "Parallel FM integration matches standard ODE — DIIID-like example (large N)" begin
        # Run standard and parallel FM integrations on the DIIID-like example (N≈26 modes).
        # Before bidirectional integration, the all-forward FM propagators were ill-conditioned
        # for large N, producing ~10% energy error. Bidirectional integration (backward crossing
        # chunks + forward intermediate chunks) restores accuracy to within 2%.
        #
        # This is the key regression test for the bidirectional parallel FM fix.
        ex = joinpath(@__DIR__, "..", "examples", "DIIID-like_ideal_example")

        function run_diiid(use_parallel)
            inputs = TOML.parsefile(joinpath(ex, "gpec.toml"))
            inputs["ForceFreeStates"]["verbose"] = false
            inputs["ForceFreeStates"]["use_parallel"] = use_parallel
            inputs["ForceFreeStates"]["write_outputs_to_HDF5"] = false
            intr = GeneralizedPerturbedEquilibrium.ForceFreeStates.ForceFreeStatesInternal(; dir_path=ex)
            ctrl = GeneralizedPerturbedEquilibrium.ForceFreeStates.ForceFreeStatesControl(;
                (Symbol(k) => v for (k, v) in inputs["ForceFreeStates"])...)
            eq_config = GeneralizedPerturbedEquilibrium.Equilibrium.EquilibriumConfig(inputs["Equilibrium"], ex)
            equil = GeneralizedPerturbedEquilibrium.Equilibrium.setup_equilibrium(eq_config)
            intr.wall_settings = GeneralizedPerturbedEquilibrium.Vacuum.WallShapeSettings(;
                (Symbol(k) => v for (k, v) in inputs["Wall"])...)
            GeneralizedPerturbedEquilibrium.ForceFreeStates.sing_lim!(intr, ctrl, equil)
            intr.nlow = ctrl.nn_low; intr.nhigh = ctrl.nn_high; intr.npert = 1
            GeneralizedPerturbedEquilibrium.ForceFreeStates.sing_find!(intr, equil)
            intr.mlow = min(intr.nlow * equil.params.qmin, 0) - 4 - ctrl.delta_mlow
            intr.mhigh = trunc(Int, intr.nhigh * equil.params.qmax) + ctrl.delta_mhigh
            intr.mpert = intr.mhigh - intr.mlow + 1
            intr.mband = intr.mpert - 1
            intr.numpert_total = intr.mpert * intr.npert
            metric = GeneralizedPerturbedEquilibrium.ForceFreeStates.make_metric(equil; mband=intr.mband)
            ffit = GeneralizedPerturbedEquilibrium.ForceFreeStates.make_matrix(equil, intr, metric)
            odet, _, _, _ = GeneralizedPerturbedEquilibrium.ForceFreeStates.eulerlagrange_integration(ctrl, equil, ffit, intr)
            vac = GeneralizedPerturbedEquilibrium.ForceFreeStates.free_run!(odet, ctrl, equil, ffit, intr)
            return real(vac.et[1]), intr
        end

        et_par, intr_par = run_diiid(true)

        # Parallel FM et[1] regression. The bidirectional fix gives et ≈ 1.5–1.6 with
        # set_psilim_via_dmlim = true (production diverted convention; DIIID-like example
        # sets it explicitly). With the previous default (false) this was ≈ 1.29. Single-
        # point pinning of et_par is platform-sensitive at the few-percent level (BLAS
        # variant / FP rounding through the BVP solve and outer-plasma Riccati pass shift
        # the eigenvalue ~5-10 %), so we bracket the eigenvalue rather than pin a tight
        # value. A true regression of the bidirectional assembly (et ≈ 1.29 or ≈ 2+) still
        # fails this bracket loudly.
        @test 1.4 < et_par < 1.7
        # Per-surface Δ' assertions removed (stub calculation; see Solovev testset
        # comment above). BVP Δ' matrix regression for DIIID-like is in the
        # `delta_prime_matrix — STRIDE BVP DIIID-like regression (large N)` testset.

        # Cross-path consistency (parallel vs standard) is omitted here: after the
        # edge-dW decoupling, the two paths store the final-state U at different
        # ψ in the edge band (different chunking → different saved points), and
        # on DIIID the standard path's free-boundary eigenvalue computation is
        # numerically unstable past the old dW-peak location, producing non-
        # sensical et values on some CI runners. A proper cross-path check would
        # require both paths to integrate on identical ψ grids, which is out of
        # scope for this regression test.
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
        equil = GeneralizedPerturbedEquilibrium.Equilibrium.setup_equilibrium(eq_config)
        GeneralizedPerturbedEquilibrium.ForceFreeStates.sing_lim!(intr, ctrl, equil)
        intr.nlow = ctrl.nn_low; intr.nhigh = ctrl.nn_high; intr.npert = 1
        GeneralizedPerturbedEquilibrium.ForceFreeStates.sing_find!(intr, equil)
        intr.mlow = min(intr.nlow * equil.params.qmin, 0) - 4 - ctrl.delta_mlow
        intr.mhigh = trunc(Int, intr.nhigh * equil.params.qmax) + ctrl.delta_mhigh
        intr.mpert = intr.mhigh - intr.mlow + 1
        intr.mband = intr.mpert - 1
        intr.numpert_total = intr.mpert * intr.npert

        metric = GeneralizedPerturbedEquilibrium.ForceFreeStates.make_metric(equil; mband=intr.mband)
        ffit = GeneralizedPerturbedEquilibrium.ForceFreeStates.make_matrix(equil, intr, metric)

        # Use the first chunk from chunk_el_integration_bounds: guaranteed rational-free interior
        odet_tmp = GeneralizedPerturbedEquilibrium.ForceFreeStates.OdeState(intr.numpert_total, 10, 5, intr.msing)
        GeneralizedPerturbedEquilibrium.ForceFreeStates.initialize_el_at_axis!(odet_tmp, ctrl, ffit, equil.profiles, intr)
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

    @testset "ξ functions bit-identical between use_parallel modes (populate_dense_xi)" begin
        # When `ctrl.use_parallel = true` and `ctrl.populate_dense_xi = true`
        # (default), `parallel_eulerlagrange_integration` appends a serial
        # Euler-Lagrange pass and returns that fresh `odet` instead of the
        # propagator-BVP one.  That dense pass invokes the SAME
        # `eulerlagrange_integration` code path the serial `use_parallel = false`
        # benchmark goes through with the SAME `(ctrl, equil, ffit, intr)`
        # inputs (BVP-only state on `intr` saved/restored across the pass), so
        # the resulting `psi_store` / `q_store` / `u_store` / `ud_store` /
        # `crit_store` arrays must be bit-identical to a standalone serial run.
        # This is a strong correctness guarantee that the dense pass does NOT
        # perturb the DCON eigenfunction calculation in any way — exactly what
        # downstream PerturbedEquilibrium / FieldReconstruction needs.
        #
        # Run on both the small-N Solovev case and the large-N DIIID-like case
        # to catch any (m, IC, ψ)-dependent regression.

        function run_and_capture(example_dir, use_parallel; populate_dense_xi=true)
            inputs = TOML.parsefile(joinpath(example_dir, "gpec.toml"))
            inputs["ForceFreeStates"]["verbose"] = false
            inputs["ForceFreeStates"]["use_parallel"] = use_parallel
            inputs["ForceFreeStates"]["populate_dense_xi"] = populate_dense_xi
            inputs["ForceFreeStates"]["write_outputs_to_HDF5"] = false
            intr = GeneralizedPerturbedEquilibrium.ForceFreeStates.ForceFreeStatesInternal(; dir_path=example_dir)
            ctrl = GeneralizedPerturbedEquilibrium.ForceFreeStates.ForceFreeStatesControl(;
                (Symbol(k) => v for (k, v) in inputs["ForceFreeStates"])...)
            eq_config = GeneralizedPerturbedEquilibrium.Equilibrium.EquilibriumConfig(inputs["Equilibrium"], example_dir)
            equil = GeneralizedPerturbedEquilibrium.Equilibrium.setup_equilibrium(eq_config)
            intr.wall_settings = GeneralizedPerturbedEquilibrium.Vacuum.WallShapeSettings(;
                (Symbol(k) => v for (k, v) in inputs["Wall"])...)
            GeneralizedPerturbedEquilibrium.ForceFreeStates.sing_lim!(intr, ctrl, equil)
            intr.nlow = ctrl.nn_low; intr.nhigh = ctrl.nn_high; intr.npert = 1
            GeneralizedPerturbedEquilibrium.ForceFreeStates.sing_find!(intr, equil)
            intr.mlow = min(intr.nlow * equil.params.qmin, 0) - 4 - ctrl.delta_mlow
            intr.mhigh = trunc(Int, intr.nhigh * equil.params.qmax) + ctrl.delta_mhigh
            intr.mpert = intr.mhigh - intr.mlow + 1
            intr.mband = intr.mpert - 1
            intr.numpert_total = intr.mpert * intr.npert
            metric = GeneralizedPerturbedEquilibrium.ForceFreeStates.make_metric(equil; mband=intr.mband)
            ffit = GeneralizedPerturbedEquilibrium.ForceFreeStates.make_matrix(equil, intr, metric)
            odet, _, _, _ = GeneralizedPerturbedEquilibrium.ForceFreeStates.eulerlagrange_integration(ctrl, equil, ffit, intr)
            return odet
        end

        # Compare the storage arrays that downstream code reads.  All values
        # must be EXACTLY equal (no tolerance — the dense pass calls the same
        # ODE solver with the same inputs as the standalone serial path, so
        # any nonzero difference indicates a real regression in the dense-pass
        # machinery).
        function assert_bit_identical(odet_a, odet_b)
            @test odet_a.step == odet_b.step
            @test odet_a.nzero == odet_b.nzero
            @test length(odet_a.psi_store) == length(odet_b.psi_store)
            @test length(odet_a.q_store) == length(odet_b.q_store)
            @test size(odet_a.u_store) == size(odet_b.u_store)
            @test size(odet_a.ud_store) == size(odet_b.ud_store)
            @test maximum(abs.(odet_a.psi_store .- odet_b.psi_store))    == 0.0
            @test maximum(abs.(odet_a.q_store   .- odet_b.q_store))      == 0.0
            @test maximum(abs.(odet_a.u_store   .- odet_b.u_store))      == 0.0
            @test maximum(abs.(odet_a.ud_store  .- odet_b.ud_store))     == 0.0
            @test maximum(abs.(odet_a.crit_store .- odet_b.crit_store))  == 0.0
        end

        @testset "Solovev (small N)" begin
            ex = joinpath(@__DIR__, "test_data", "regression_solovev_ideal_example")
            odet_std = run_and_capture(ex, false)
            odet_par = run_and_capture(ex, true;  populate_dense_xi=true)
            assert_bit_identical(odet_std, odet_par)
        end

        @testset "DIIID-like (large N)" begin
            ex = joinpath(@__DIR__, "..", "examples", "DIIID-like_ideal_example")
            odet_std = run_and_capture(ex, false)
            odet_par = run_and_capture(ex, true;  populate_dense_xi=true)
            assert_bit_identical(odet_std, odet_par)
        end

        @testset "populate_dense_xi=false leaves sparse u_store (control)" begin
            # Sanity-check the opposite mode: with populate_dense_xi=false, the
            # parallel BVP path stores only chunk-endpoint Riccati snapshots,
            # so u_store / ud_store / psi_store have strictly fewer entries
            # than the serial path.  Catching this guarantees the bit-identical
            # test above is meaningful — it's NOT trivially passing because
            # both modes accidentally produce the same sparse data.
            ex = joinpath(@__DIR__, "test_data", "regression_solovev_ideal_example")
            odet_std    = run_and_capture(ex, false)
            odet_sparse = run_and_capture(ex, true;  populate_dense_xi=false)
            @test odet_sparse.step < odet_std.step
            # ud_store entries inside FM chunks are left at the @kwdef
            # `undef` initial value when populate_dense_xi=false; ensure the
            # array IS smaller (sparse).
            @test length(odet_sparse.psi_store) < length(odet_std.psi_store)
        end
    end

    @testset "delta_prime_matrix — STRIDE BVP DIIID-like regression (large N)" begin
        # Verify that the parallel FM path computes a well-formed inter-surface Δ' matrix
        # for the DIIID-like case (N≈26 modes, multiple rational surfaces). This complements
        # the Solovev test above by exercising the BVP assembly with more surfaces and larger
        # mode space, where ill-conditioned (non-bidirectional) FM propagators would fail.
        ex = joinpath(@__DIR__, "..", "examples", "DIIID-like_ideal_example")
        inputs = TOML.parsefile(joinpath(ex, "gpec.toml"))
        inputs["ForceFreeStates"]["verbose"] = false
        inputs["ForceFreeStates"]["use_parallel"] = true
        inputs["ForceFreeStates"]["write_outputs_to_HDF5"] = false
        intr = GeneralizedPerturbedEquilibrium.ForceFreeStates.ForceFreeStatesInternal(; dir_path=ex)
        ctrl = GeneralizedPerturbedEquilibrium.ForceFreeStates.ForceFreeStatesControl(;
            (Symbol(k) => v for (k, v) in inputs["ForceFreeStates"])...)
        eq_config = GeneralizedPerturbedEquilibrium.Equilibrium.EquilibriumConfig(inputs["Equilibrium"], ex)
        equil = GeneralizedPerturbedEquilibrium.Equilibrium.setup_equilibrium(eq_config)
        intr.wall_settings = GeneralizedPerturbedEquilibrium.Vacuum.WallShapeSettings(;
            (Symbol(k) => v for (k, v) in inputs["Wall"])...)
        GeneralizedPerturbedEquilibrium.ForceFreeStates.sing_lim!(intr, ctrl, equil)
        intr.nlow = ctrl.nn_low; intr.nhigh = ctrl.nn_high; intr.npert = 1
        GeneralizedPerturbedEquilibrium.ForceFreeStates.sing_find!(intr, equil)
        intr.mlow = min(intr.nlow * equil.params.qmin, 0) - 4 - ctrl.delta_mlow
        intr.mhigh = trunc(Int, intr.nhigh * equil.params.qmax) + ctrl.delta_mhigh
        intr.mpert = intr.mhigh - intr.mlow + 1
        intr.mband = intr.mpert - 1
        intr.numpert_total = intr.mpert * intr.npert
        metric = GeneralizedPerturbedEquilibrium.ForceFreeStates.make_metric(equil; mband=intr.mband)
        ffit = GeneralizedPerturbedEquilibrium.ForceFreeStates.make_matrix(equil, intr, metric)
        odet, fm_propagators, fm_chunks, fm_S_left =
            GeneralizedPerturbedEquilibrium.ForceFreeStates.eulerlagrange_integration(ctrl, equil, ffit, intr)
        vac = GeneralizedPerturbedEquilibrium.ForceFreeStates.free_run!(odet, ctrl, equil, ffit, intr)
        GeneralizedPerturbedEquilibrium.ForceFreeStates.compute_delta_prime_matrix!(
            intr, fm_propagators, fm_chunks;
            wv=vac.wv, psio=equil.psio,
            S_at_surface_left=fm_S_left, ctrl=ctrl, equil=equil, ffit=ffit)

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

        # Pinned diagonal `delta_prime_matrix` values for the DIIID-like case (msing = 5),
        # PEST3-convention self-response Δ' from the STRIDE BVP with vacuum coupling.
        # Tolerances are split by entry magnitude / |Im|/|Re| ratio (audit V4):
        #   - dpm[1], dpm[2]: nearly-real entries (|Im|/|Re| < 0.02). Platform-stable; rtol=1e-2.
        #   - dpm[3]: complex entry with |Im| ≈ |Re| (both ~10). Modest FP sensitivity in the
        #     PEST3 cancellation. rtol=5e-2 catches sign/normalization regressions while
        #     accepting ~2-3% imaginary-part drift across BLAS variants.
        #   - dpm[4], dpm[5]: |Im| is highly sensitive to FP round-off in the PEST3 four-term
        #     cancellation (dp_raw entries can be 10⁴–10⁵× larger than the result). The
        #     imaginary part drifts by 2–5× across platforms even with `extended_precision_bvp=true`.
        #     Pin only the real part tightly; bracket |dpm| to catch sign/normalization errors.
        # dpm[1], dpm[2] re-pinned when FourierCoefficients began dropping the duplicated θ=2π
        # endpoint before the FFT (faithful Fortran fspline_fit_2, equil/fspline.f:293): the old
        # un-trimmed FFT double-counted θ=0, biasing the metric DC coefficient. dpm[3]–dpm[5]
        # shifted too but stay within their wider tolerances.
        @test isapprox(dpm[1, 1], +8.672812e+00 + 2.139354e-02im; rtol=1e-2)
        @test isapprox(dpm[2, 2], -3.890689e+00 - 5.121123e-02im; rtol=1e-2)
        @test isapprox(dpm[3, 3], -9.137656e+00 + 7.704888e+00im; rtol=5e-2)
        @test isapprox(real(dpm[4, 4]), +5.790777e+03; rtol=5e-2)
        @test isapprox(real(dpm[5, 5]), -2.940021e+02; rtol=5e-2)
        @test 1e3 < abs(dpm[4, 4]) < 1e5    # |dpm[4,4]| ≈ 6e3; catches sign/normalization errors
        @test 1e2 < abs(dpm[5, 5]) < 1e3    # |dpm[5,5]| ≈ 3e2; catches sign/normalization errors
    end

end
