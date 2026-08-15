using Test
using TOML

# Thread-invariance of the parallel FM/BVP path.
#
# `ForceFreeStatesControl` documents that the parallel path produces bit-identical Δ′ across
# thread counts, and the example decks disagree about whether to rely on it: the SLAYER deck
# pins `parallel_threads = 1` for reproducibility while the DIII-D-like ideal deck — whose Δ′
# the regression harness tracks — runs `parallel_threads = 2`. These tests hold the source and
# the deck fixed and vary only the BVP thread cap, so a golden Δ′ is a property of the physics
# rather than of the machine it was measured on.
#
# Two axes are reachable through `parallel_threads`:
#
#   - scheduling: `Threads.@threads` over chunks vs a serial loop (any cap ≥ 2)
#   - decomposition: `balance_integration_chunks` targets
#     `max(2·msing+3, 4·effective_threads, 8·(msing+1)+msing)` sub-chunks, so once
#     `4·effective_threads` exceeds the `min_bvp_intervals` floor the chunk boundaries
#     themselves move and the propagator products reassociate
#
# The decomposition axis needs `effective_threads > (9·msing+8)/4` — about 14 threads for the
# DIII-D-like deck's msing=5 — so it is out of reach of a typical CI runner and is exercised by
# the nightly harness instead. Measured on this deck at parallel_threads = 1, 2, 4 and 16
# (53 vs 64 chunks, confirmed different boundaries): Δ′ and et[1] were bit-identical throughout.
#
# `effective_threads = min(Threads.nthreads(), parallel_threads)` collapses every cap to 1 in a
# single-threaded session, which would make these comparisons trivially true, so they are skipped
# there rather than passing vacuously.

const GP_TI = GeneralizedPerturbedEquilibrium

"""
Run the ideal stability pipeline on `dir` at a given BVP thread cap.

Mirrors the standalone setup used by the parallel-integration tests: build the equilibrium
(applying the two-pass auto grid when the deck asks for it), integrate the Euler-Lagrange
system, then assemble the STRIDE BVP Δ′ matrix. Returns the Δ′ matrix, the leading energy
eigenvalue, and the chunk boundaries, so a caller can tell scheduling changes from
decomposition changes.
"""
function _run_at_thread_cap(dir::String, parallel_threads::Int)
    inputs = TOML.parsefile(joinpath(dir, "gpec.toml"))
    inputs["ForceFreeStates"]["verbose"] = false
    inputs["ForceFreeStates"]["use_parallel"] = true
    inputs["ForceFreeStates"]["write_outputs_to_HDF5"] = false
    inputs["ForceFreeStates"]["parallel_threads"] = parallel_threads

    intr = GP_TI.ForceFreeStates.ForceFreeStatesInternal(; dir_path=dir)
    ctrl = GP_TI.ForceFreeStates.ForceFreeStatesControl(; (Symbol(k) => v for (k, v) in inputs["ForceFreeStates"])...)
    eq_config = GP_TI.Equilibrium.EquilibriumConfig(inputs["Equilibrium"], dir)
    sol_config = haskey(inputs, "SOL_INPUT") ? GP_TI.Equilibrium.SolovevConfig(inputs["SOL_INPUT"]) : nothing
    equil = GP_TI.Equilibrium.setup_equilibrium(eq_config, sol_config)
    if GP_TI.Equilibrium.wants_two_pass(eq_config)
        mand = GP_TI.ForceFreeStates.rational_psi_nodes(equil; nlow=ctrl.nn_low, nhigh=ctrl.nn_high)
        psi_nodes = GP_TI.Equilibrium.refined_psi_grid(equil; tau=eq_config.psi_accuracy, mandatory=mand)
        rerun_input = GP_TI.Equilibrium.build_direct_from_ingest(eq_config, equil.ingest)
        equil = GP_TI.Equilibrium.setup_equilibrium(eq_config, rerun_input; override_psi_nodes=psi_nodes)
    end
    intr.wall_settings = GP_TI.Vacuum.WallShapeSettings(; (Symbol(k) => v for (k, v) in inputs["Wall"])...)
    # The toroidal range must be resolved before sing_lim!: under set_psilim_via_dmlim it
    # truncates at (last_rational_q + dmlim)/n and so needs n.
    intr.nlow = ctrl.nn_low
    intr.nhigh = ctrl.nn_high
    intr.npert = 1
    GP_TI.ForceFreeStates.sing_lim!(intr, ctrl, equil)
    GP_TI.ForceFreeStates.sing_find!(intr, equil)
    intr.mlow = min(intr.nlow * equil.params.qmin, 0) - 4 - ctrl.delta_mlow
    intr.mhigh = trunc(Int, intr.nhigh * equil.params.qmax) + ctrl.delta_mhigh
    intr.mpert = intr.mhigh - intr.mlow + 1
    intr.numpert_total = intr.mpert * intr.npert
    metric = GP_TI.ForceFreeStates.make_metric(equil, intr.mpert)
    ffit = GP_TI.ForceFreeStates.make_matrix(equil, intr, metric)
    odet, fm_propagators, fm_chunks, fm_S_left = GP_TI.ForceFreeStates.eulerlagrange_integration(ctrl, equil, ffit, intr)
    vac = GP_TI.ForceFreeStates.free_run!(odet, ctrl, equil, ffit, intr)
    GP_TI.ForceFreeStates.compute_delta_prime_matrix!(intr, fm_propagators, fm_chunks;
        wv=vac.wv, psio=equil.psio, S_at_surface_left=fm_S_left, ctrl=ctrl, equil=equil, ffit=ffit)
    return (dpm=copy(intr.delta_prime_matrix), et1=vac.et[1], msing=intr.msing,
        bounds=[(c.psi_start, c.psi_end) for c in fm_chunks])
end

@testset "Thread invariance of the parallel BVP path" begin
    if Threads.nthreads() < 2
        @info "Thread-invariance tests skipped: effective_threads collapses to 1 in a single-threaded session. Run with `julia -t 4` (CI covers this in its multi-threaded matrix leg)."
        @test true
    else
        @testset "Solovev — leading eigenvalue" begin
            dir = joinpath(@__DIR__, "test_data", "regression_solovev_ideal_example")
            serial = _run_at_thread_cap(dir, 1)
            threaded = _run_at_thread_cap(dir, 2)
            # Bit-identical, not approximate: the parallel path claims exactness, and a
            # tolerance here would hide precisely the reassociation this test exists to catch.
            @test threaded.et1 === serial.et1
        end

        @testset "DIII-D-like — Δ′ diagonal and leading eigenvalue" begin
            # The deck whose Δ′ the regression harness pins, and where the BVP Δ′ is
            # well-conditioned (Solovev sits near marginal stability and its BVP Δ′ is not).
            dir = joinpath(@__DIR__, "..", "examples", "DIIID-like_ideal_example")
            serial = _run_at_thread_cap(dir, 1)
            threaded = _run_at_thread_cap(dir, 2)

            @test threaded.msing == serial.msing
            @test size(threaded.dpm) == size(serial.dpm)
            @test threaded.et1 === serial.et1
            for j in 1:serial.msing
                @test threaded.dpm[j, j] === serial.dpm[j, j]
            end
            @test threaded.dpm == serial.dpm

            # At these caps the min_bvp_intervals floor fixes the chunk count, so the
            # boundaries should be untouched and only scheduling differs. If this fails the
            # decomposition moved and the Δ′ comparison above became a stronger claim than
            # the one this testset intends to make.
            @test threaded.bounds == serial.bounds
        end
    end
end
