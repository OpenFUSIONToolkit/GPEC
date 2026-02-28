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

    @testset "Parallel FM integration matches standard ODE — Solovev example" begin
        # Run standard and parallel FM integrations on the Solovev regression test.
        # The energy eigenvalue et[1] should match to within 2%.
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

end
