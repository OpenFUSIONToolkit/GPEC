using LinearAlgebra
using TOML

@testset "Parallel FM Integration Tests" begin

    @testset "ChunkPropagator identity on trivial interval" begin
        # Integrating over a zero-width interval should give the identity propagator.
        # We test that apply_propagator! on an identity state preserves the state.
        N = 3
        prop = JPEC.ForceFreeStates.ChunkPropagator(N)

        # Set propagator to identity (block_upper_ic = (I, 0), block_lower_ic = (0, I))
        for i in 1:N
            prop.block_upper_ic[i, i, 1] = 1  # U1 block from IC=(I,0)
            prop.block_lower_ic[i, i, 2] = 1  # U2 block from IC=(0,I)
        end

        # Apply identity propagator to an arbitrary state
        odet = JPEC.ForceFreeStates.OdeState(N, 10, 5, 0)
        u1_in = [1.0+0.5im  0.2im   0.0;
                 0.1+0.1im  1.2+0.1im 0.0;
                 0.0im      0.0      0.9+0.3im]
        u2_in = [0.8+0.1im  0.1im   0.0;
                 0.0im      1.0+0.2im 0.1;
                 0.1im      0.0      1.1+0.0im]
        odet.u[:, :, 1] .= u1_in
        odet.u[:, :, 2] .= u2_in

        JPEC.ForceFreeStates.apply_propagator!(odet, prop)

        @test odet.u[:, :, 1] ≈ u1_in  rtol=1e-12
        @test odet.u[:, :, 2] ≈ u2_in  rtol=1e-12
    end

    @testset "apply_propagator! linearity" begin
        # Verify that apply_propagator! applies the correct linear map.
        N = 3
        prop = JPEC.ForceFreeStates.ChunkPropagator(N)

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

        odet = JPEC.ForceFreeStates.OdeState(N, 10, 5, 0)
        u1_in = 0.5 * I(N) .+ 0.1im * ones(N, N)
        u2_in = I(N) .+ 0.2im * ones(N, N)
        odet.u[:, :, 1] .= u1_in
        odet.u[:, :, 2] .= u2_in

        JPEC.ForceFreeStates.apply_propagator!(odet, prop)

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
        prop = JPEC.ForceFreeStates.ChunkPropagator(N)

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

        odet = JPEC.ForceFreeStates.OdeState(N, 10, 5, 0)
        odet.u[:, :, 1] .= u1_in
        odet.u[:, :, 2] .= u2_in

        # Round-trip: inverse then forward = identity
        JPEC.ForceFreeStates.apply_propagator_inverse!(odet, prop)
        JPEC.ForceFreeStates.apply_propagator!(odet, prop)

        @test odet.u[:, :, 1] ≈ u1_in  rtol=1e-12
        @test odet.u[:, :, 2] ≈ u2_in  rtol=1e-12
    end

    @testset "balance_integration_chunks produces target count" begin
        # Verify that balance_integration_chunks creates at least
        # max(2*msing+3, 4*nthreads) chunks from a small set of base chunks.
        ex = joinpath(@__DIR__, "test_data", "regression_solovev_ideal_example")
        inputs = TOML.parsefile(joinpath(ex, "jpec.toml"))
        inputs["ForceFreeStates"]["verbose"] = false
        intr = JPEC.ForceFreeStates.ForceFreeStatesInternal(; dir_path=ex)
        ctrl = JPEC.ForceFreeStates.ForceFreeStatesControl(;
            (Symbol(k) => v for (k, v) in inputs["ForceFreeStates"])...)
        eq_config = JPEC.Equilibrium.EquilibriumConfig(inputs["Equilibrium"], ex)
        equil = JPEC.Equilibrium.setup_equilibrium(eq_config)
        JPEC.ForceFreeStates.sing_lim!(intr, ctrl, equil)
        intr.nlow = ctrl.nn_low; intr.nhigh = ctrl.nn_high; intr.npert = 1
        JPEC.ForceFreeStates.sing_find!(intr, equil)
        intr.mlow = min(intr.nlow * equil.params.qmin, 0) - 4 - ctrl.delta_mlow
        intr.mhigh = trunc(Int, intr.nhigh * equil.params.qmax) + ctrl.delta_mhigh
        intr.mpert = intr.mhigh - intr.mlow + 1
        intr.mband = intr.mpert - 1
        intr.numpert_total = intr.mpert * intr.npert

        odet = JPEC.ForceFreeStates.OdeState(intr.numpert_total, ctrl.numsteps_init, ctrl.numunorms_init, intr.msing)
        JPEC.ForceFreeStates.initialize_el_at_axis!(odet, ctrl, equil.profiles, intr)

        base_chunks = JPEC.ForceFreeStates.chunk_el_integration_bounds(odet, ctrl, intr)
        balanced = JPEC.ForceFreeStates.balance_integration_chunks(base_chunks, ctrl, intr)

        target_n = max(2 * intr.msing + 3, 4 * Threads.nthreads())

        # After balancing, should have at least target_n chunks
        @test length(balanced) >= min(target_n, length(base_chunks) * 50)

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
        inputs = TOML.parsefile(joinpath(ex, "jpec.toml"))
        inputs["ForceFreeStates"]["verbose"] = false
        intr = JPEC.ForceFreeStates.ForceFreeStatesInternal(; dir_path=ex)
        ctrl = JPEC.ForceFreeStates.ForceFreeStatesControl(;
            (Symbol(k) => v for (k, v) in inputs["ForceFreeStates"])...)
        eq_config = JPEC.Equilibrium.EquilibriumConfig(inputs["Equilibrium"], ex)
        equil = JPEC.Equilibrium.setup_equilibrium(eq_config)
        JPEC.ForceFreeStates.sing_lim!(intr, ctrl, equil)
        intr.nlow = ctrl.nn_low; intr.nhigh = ctrl.nn_high; intr.npert = 1
        JPEC.ForceFreeStates.sing_find!(intr, equil)
        intr.mlow = min(intr.nlow * equil.params.qmin, 0) - 4 - ctrl.delta_mlow
        intr.mhigh = trunc(Int, intr.nhigh * equil.params.qmax) + ctrl.delta_mhigh
        intr.mpert = intr.mhigh - intr.mlow + 1
        intr.mband = intr.mpert - 1
        intr.numpert_total = intr.mpert * intr.npert

        odet = JPEC.ForceFreeStates.OdeState(intr.numpert_total, ctrl.numsteps_init, ctrl.numunorms_init, intr.msing)
        JPEC.ForceFreeStates.initialize_el_at_axis!(odet, ctrl, equil.profiles, intr)

        # Default (bidirectional=false): all chunks should have direction=+1
        chunks_fwd = JPEC.ForceFreeStates.chunk_el_integration_bounds(odet, ctrl, intr)
        @test all(c -> c.direction == 1, chunks_fwd)

        # bidirectional=true: crossing chunks direction=-1, non-crossing direction=+1
        chunks_bidi = JPEC.ForceFreeStates.chunk_el_integration_bounds(odet, ctrl, intr; bidirectional=true)
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
        balanced_bidi = JPEC.ForceFreeStates.balance_integration_chunks(chunks_bidi, ctrl, intr)
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
            inputs = TOML.parsefile(joinpath(ex, "jpec.toml"))
            inputs["ForceFreeStates"]["verbose"] = false
            inputs["ForceFreeStates"]["use_parallel"] = use_parallel
            intr = JPEC.ForceFreeStates.ForceFreeStatesInternal(; dir_path=ex)
            ctrl = JPEC.ForceFreeStates.ForceFreeStatesControl(;
                (Symbol(k) => v for (k, v) in inputs["ForceFreeStates"])...)
            eq_config = JPEC.Equilibrium.EquilibriumConfig(inputs["Equilibrium"], ex)
            equil = JPEC.Equilibrium.setup_equilibrium(eq_config)
            intr.wall_settings = JPEC.Vacuum.WallShapeSettings(;
                (Symbol(k) => v for (k, v) in inputs["Wall"])...)
            JPEC.ForceFreeStates.sing_lim!(intr, ctrl, equil)
            intr.nlow = ctrl.nn_low; intr.nhigh = ctrl.nn_high; intr.npert = 1
            JPEC.ForceFreeStates.sing_find!(intr, equil)
            intr.mlow = min(intr.nlow * equil.params.qmin, 0) - 4 - ctrl.delta_mlow
            intr.mhigh = trunc(Int, intr.nhigh * equil.params.qmax) + ctrl.delta_mhigh
            intr.mpert = intr.mhigh - intr.mlow + 1
            intr.mband = intr.mpert - 1
            intr.numpert_total = intr.mpert * intr.npert
            metric = JPEC.ForceFreeStates.make_metric(equil; mband=intr.mband, fft_flag=ctrl.fft_flag)
            ffit = JPEC.ForceFreeStates.make_matrix(equil, intr, metric)
            odet = JPEC.ForceFreeStates.eulerlagrange_integration(ctrl, equil, ffit, intr)
            vac = JPEC.ForceFreeStates.free_run!(odet, ctrl, equil, ffit, intr)
            return real(vac.et[1]), intr
        end

        et_std, intr_std = run_solovev(false)
        et_par, intr_par = run_solovev(true)

        # Energy eigenvalue matches to 2%
        @test isapprox(et_par, et_std; rtol=0.02)

        # Δ' is populated for every singular surface (finite values)
        # Note: the FM parallel path computes Δ' from ca_l/ca_r accumulated in (S,I)
        # normalization (Riccati-style crossings). This differs from the sequential path's
        # (U1,U2) normalization, so absolute Δ' values are not compared here.
        @test all(s -> !isempty(s.delta_prime), intr_par.sing)
        @test all(s -> all(isfinite, s.delta_prime), intr_par.sing)

        # delta_prime_col is populated and has the correct shape (N × n_res_modes)
        N = intr_par.numpert_total
        @test all(s -> !isempty(s.delta_prime_col), intr_par.sing)
        @test all(s -> size(s.delta_prime_col, 1) == N, intr_par.sing)
        @test all(s -> size(s.delta_prime_col, 2) == length(s.delta_prime), intr_par.sing)

        # Diagonal of delta_prime_col matches delta_prime (consistency check)
        for s in intr_par.sing
            ipert_res_vals = 1 .+ s.m .- intr_par.mlow .+ (s.n .- intr_par.nlow) .* intr_par.mpert
            for (i, ipr) in enumerate(ipert_res_vals)
                @test s.delta_prime_col[ipr, i] ≈ s.delta_prime[i]  rtol=1e-10
            end
        end
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
            inputs = TOML.parsefile(joinpath(ex, "jpec.toml"))
            inputs["ForceFreeStates"]["verbose"] = false
            inputs["ForceFreeStates"]["use_parallel"] = use_parallel
            inputs["ForceFreeStates"]["write_outputs_to_HDF5"] = false
            intr = JPEC.ForceFreeStates.ForceFreeStatesInternal(; dir_path=ex)
            ctrl = JPEC.ForceFreeStates.ForceFreeStatesControl(;
                (Symbol(k) => v for (k, v) in inputs["ForceFreeStates"])...)
            eq_config = JPEC.Equilibrium.EquilibriumConfig(inputs["Equilibrium"], ex)
            equil = JPEC.Equilibrium.setup_equilibrium(eq_config)
            intr.wall_settings = JPEC.Vacuum.WallShapeSettings(;
                (Symbol(k) => v for (k, v) in inputs["Wall"])...)
            JPEC.ForceFreeStates.sing_lim!(intr, ctrl, equil)
            intr.nlow = ctrl.nn_low; intr.nhigh = ctrl.nn_high; intr.npert = 1
            JPEC.ForceFreeStates.sing_find!(intr, equil)
            intr.mlow = min(intr.nlow * equil.params.qmin, 0) - 4 - ctrl.delta_mlow
            intr.mhigh = trunc(Int, intr.nhigh * equil.params.qmax) + ctrl.delta_mhigh
            intr.mpert = intr.mhigh - intr.mlow + 1
            intr.mband = intr.mpert - 1
            intr.numpert_total = intr.mpert * intr.npert
            metric = JPEC.ForceFreeStates.make_metric(equil; mband=intr.mband, fft_flag=ctrl.fft_flag)
            ffit = JPEC.ForceFreeStates.make_matrix(equil, intr, metric)
            odet = JPEC.ForceFreeStates.eulerlagrange_integration(ctrl, equil, ffit, intr)
            vac = JPEC.ForceFreeStates.free_run!(odet, ctrl, equil, ffit, intr)
            return real(vac.et[1])
        end

        et_std = run_diiid(false)
        et_par = run_diiid(true)

        # Energy eigenvalue matches to 2% (bidirectional fix: was ~10% error without it)
        @test isapprox(et_par, et_std; rtol=0.02)
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
        inputs = TOML.parsefile(joinpath(ex, "jpec.toml"))
        inputs["ForceFreeStates"]["verbose"] = false
        intr = JPEC.ForceFreeStates.ForceFreeStatesInternal(; dir_path=ex)
        ctrl = JPEC.ForceFreeStates.ForceFreeStatesControl(;
            (Symbol(k) => v for (k, v) in inputs["ForceFreeStates"])...)
        eq_config = JPEC.Equilibrium.EquilibriumConfig(inputs["Equilibrium"], ex)
        equil = JPEC.Equilibrium.setup_equilibrium(eq_config)
        JPEC.ForceFreeStates.sing_lim!(intr, ctrl, equil)
        intr.nlow = ctrl.nn_low; intr.nhigh = ctrl.nn_high; intr.npert = 1
        JPEC.ForceFreeStates.sing_find!(intr, equil)
        intr.mpert = 8; intr.numpert_total = 8

        # Use the first chunk from chunk_el_integration_bounds: guaranteed rational-free interior
        odet_tmp = JPEC.ForceFreeStates.OdeState(8, 10, 5, intr.msing)
        JPEC.ForceFreeStates.initialize_el_at_axis!(odet_tmp, ctrl, equil.profiles, intr)
        chunks_tmp = JPEC.ForceFreeStates.chunk_el_integration_bounds(odet_tmp, ctrl, intr)
        chunk1 = chunks_tmp[1]
        a = chunk1.psi_start
        c = chunk1.psi_end
        b = (a + c) / 2.0

        cost_ac = JPEC.ForceFreeStates.ode_itime_cost(a, c, intr)
        cost_ab = JPEC.ForceFreeStates.ode_itime_cost(a, b, intr)
        cost_bc = JPEC.ForceFreeStates.ode_itime_cost(b, c, intr)

        @test isapprox(cost_ac, cost_ab + cost_bc; rtol=1e-10)
    end

    @testset "delta_prime_matrix — STRIDE BVP Solovev regression" begin
        # Verify that the parallel FM path computes a well-formed inter-surface Δ' matrix
        # via the STRIDE global BVP [Glasser 2018 Phys. Plasmas 25, 032501].
        # Shape: (2·msing × 2·msing), where index 2j-1 = left side and 2j = right side
        # of surface j. Each entry is the U₂[ipert_res] response amplitude for one
        # driving configuration.
        ex = joinpath(@__DIR__, "test_data", "regression_solovev_ideal_example")
        inputs = TOML.parsefile(joinpath(ex, "jpec.toml"))
        inputs["ForceFreeStates"]["verbose"] = false
        inputs["ForceFreeStates"]["use_parallel"] = true
        intr = JPEC.ForceFreeStates.ForceFreeStatesInternal(; dir_path=ex)
        ctrl = JPEC.ForceFreeStates.ForceFreeStatesControl(;
            (Symbol(k) => v for (k, v) in inputs["ForceFreeStates"])...)
        eq_config = JPEC.Equilibrium.EquilibriumConfig(inputs["Equilibrium"], ex)
        equil = JPEC.Equilibrium.setup_equilibrium(eq_config)
        intr.wall_settings = JPEC.Vacuum.WallShapeSettings(;
            (Symbol(k) => v for (k, v) in inputs["Wall"])...)
        JPEC.ForceFreeStates.sing_lim!(intr, ctrl, equil)
        intr.nlow = ctrl.nn_low; intr.nhigh = ctrl.nn_high; intr.npert = 1
        JPEC.ForceFreeStates.sing_find!(intr, equil)
        intr.mlow = min(intr.nlow * equil.params.qmin, 0) - 4 - ctrl.delta_mlow
        intr.mhigh = trunc(Int, intr.nhigh * equil.params.qmax) + ctrl.delta_mhigh
        intr.mpert = intr.mhigh - intr.mlow + 1
        intr.mband = intr.mpert - 1
        intr.numpert_total = intr.mpert * intr.npert
        metric = JPEC.ForceFreeStates.make_metric(equil; mband=intr.mband, fft_flag=ctrl.fft_flag)
        ffit = JPEC.ForceFreeStates.make_matrix(equil, intr, metric)
        JPEC.ForceFreeStates.eulerlagrange_integration(ctrl, equil, ffit, intr)

        msing = intr.msing
        dpm = intr.delta_prime_matrix

        # Matrix is populated with correct shape (2·msing × 2·msing)
        @test !isempty(dpm)
        @test size(dpm) == (2 * msing, 2 * msing)

        # All elements are finite
        @test all(isfinite, dpm)

        # Diagonal (self-response) elements are non-zero for each surface side
        for j in 1:msing
            @test abs(dpm[2j-1, 2j-1]) > 1e-10
            @test abs(dpm[2j,   2j  ]) > 1e-10
        end
    end

    @testset "delta_prime_matrix — STRIDE BVP DIIID-like regression (large N)" begin
        # Verify that the parallel FM path computes a well-formed inter-surface Δ' matrix
        # for the DIIID-like case (N≈26 modes, multiple rational surfaces). This complements
        # the Solovev test above by exercising the BVP assembly with more surfaces and larger
        # mode space, where ill-conditioned (non-bidirectional) FM propagators would fail.
        ex = joinpath(@__DIR__, "..", "examples", "DIIID-like_ideal_example")
        inputs = TOML.parsefile(joinpath(ex, "jpec.toml"))
        inputs["ForceFreeStates"]["verbose"] = false
        inputs["ForceFreeStates"]["use_parallel"] = true
        inputs["ForceFreeStates"]["write_outputs_to_HDF5"] = false
        intr = JPEC.ForceFreeStates.ForceFreeStatesInternal(; dir_path=ex)
        ctrl = JPEC.ForceFreeStates.ForceFreeStatesControl(;
            (Symbol(k) => v for (k, v) in inputs["ForceFreeStates"])...)
        eq_config = JPEC.Equilibrium.EquilibriumConfig(inputs["Equilibrium"], ex)
        equil = JPEC.Equilibrium.setup_equilibrium(eq_config)
        intr.wall_settings = JPEC.Vacuum.WallShapeSettings(;
            (Symbol(k) => v for (k, v) in inputs["Wall"])...)
        JPEC.ForceFreeStates.sing_lim!(intr, ctrl, equil)
        intr.nlow = ctrl.nn_low; intr.nhigh = ctrl.nn_high; intr.npert = 1
        JPEC.ForceFreeStates.sing_find!(intr, equil)
        intr.mlow = min(intr.nlow * equil.params.qmin, 0) - 4 - ctrl.delta_mlow
        intr.mhigh = trunc(Int, intr.nhigh * equil.params.qmax) + ctrl.delta_mhigh
        intr.mpert = intr.mhigh - intr.mlow + 1
        intr.mband = intr.mpert - 1
        intr.numpert_total = intr.mpert * intr.npert
        metric = JPEC.ForceFreeStates.make_metric(equil; mband=intr.mband, fft_flag=ctrl.fft_flag)
        ffit = JPEC.ForceFreeStates.make_matrix(equil, intr, metric)
        JPEC.ForceFreeStates.eulerlagrange_integration(ctrl, equil, ffit, intr)

        msing = intr.msing
        dpm = intr.delta_prime_matrix

        # Matrix is populated with correct shape (2·msing × 2·msing)
        @test !isempty(dpm)
        @test size(dpm) == (2 * msing, 2 * msing)

        # All elements are finite
        @test all(isfinite, dpm)

        # Diagonal (self-response) elements are non-zero for each surface side
        for j in 1:msing
            @test abs(dpm[2j-1, 2j-1]) > 1e-10
            @test abs(dpm[2j,   2j  ]) > 1e-10
        end
    end

end
