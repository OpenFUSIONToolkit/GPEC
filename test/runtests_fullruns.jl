using HDF5
using TOML

# Run GeneralizedPerturbedEquilibrium.main on the provided example directories and assert it completes without throwing.
@testset "Full ForceFreeStates runs" begin
    ex1 = joinpath(@__DIR__, "test_data", "regression_solovev_ideal_example")
    # A renorm_threshold reduction landing just before an ideal crossing must not change which solution the crossing removes.
    @info "Running Solovev ideal example with frequent Gaussian reductions"
    @testset "et[1] is independent of the reduction frequency" begin
        deck = TOML.parsefile(joinpath(ex1, "gpec.toml"))
        function total_energy(renorm_threshold)
            deck["Forward"]["renorm_threshold"] = renorm_threshold
            return mktempdir() do workdir
                open(io -> TOML.print(io, deck), joinpath(workdir, "gpec.toml"), "w")
                GeneralizedPerturbedEquilibrium.main([workdir])
                return h5open(h5 -> read(h5["ForceFreeStates/FreeBoundaryStability/eigenmode_energies"])[1], joinpath(workdir, "gpec.h5"), "r")
            end
        end
        et_ref = total_energy(1e3)
        for renorm_threshold in (2.0, 3.0, 5.0, 10.0)
            @test total_energy(renorm_threshold) ≈ et_ref rtol = 1e-6
        end
    end

    ex2 = joinpath(@__DIR__, "test_data", "regression_solovev_ideal_example_multi_n")
    @info "Running Solovev ideal multi-n example"
    @test begin
        GeneralizedPerturbedEquilibrium.main([ex2])
        true
    end

    # A fixture deck solved through the API with synthetic test kinetic matrices, which no deck can select; saved to `dir`.
    function test_kinetic_run(dir, fixture)
        GPEC = GeneralizedPerturbedEquilibrium
        inputs, config, analytic = GPEC.build_inputs_from_toml(joinpath(@__DIR__, "test_data", fixture))
        kwargs = merge(GPEC.force_balance_keywords(inputs, joinpath(@__DIR__, "test_data", fixture)), Dict(:kinetic => TestKineticMatrices(; scale=1e-9)))
        prob = ForceBalanceProblem(PlasmaEquilibrium(config, analytic); kwargs...)
        path = joinpath(dir, "gpec.h5")
        save_to_hdf5(path, solve(prob, GPEC.force_balance_algorithm(inputs)))
        return path
    end

    @info "Running Solovev with synthetic test kinetic matrices (scale 1e-9)"
    @test mktempdir() do dir
        h5open(test_kinetic_run(dir, "regression_solovev_ideal_example"), "r") do h5
            # Smoke test: this nerfed deck (mpsi=16, delta_m=0) is for "does it run / no NaN",
            # not numeric regression — the mode-converged physical value is pinned in the
            # regression harness (examples/Solovev_kinetic_calculated_example). Assert only
            # nerfed-grid-robust facts: finite and positive (Solovev is stable).
            et = read(h5["ForceFreeStates/FreeBoundaryStability/eigenmode_energies"])
            @test isfinite(real(et[1]))
            @test real(et[1]) > 0
            # Kinetic runs never populate the asymptotic ca coefficients; the writer must
            # emit deterministic zero-extent sentinels, not uninitialized memory.
            @test isempty(read(h5["SingularSurfaces/ca_left"]))
            @test isempty(read(h5["SingularSurfaces/ca_right"]))
            # The Euler-Lagrange matrices are opt-in; a default run writes no group.
            @test !haskey(h5, "ForceFreeStates/EulerLagrangeMatrices")
        end
        true
    end

    @info "Running Solovev multi-n with synthetic test kinetic matrices (scale 1e-9, n = 1:2)"
    @test mktempdir() do dir
        h5open(test_kinetic_run(dir, "regression_solovev_ideal_example_multi_n"), "r") do h5
            # Smoke test only (nerfed mpsi=16, delta_m=0 deck): runs without faulting and
            # produces a finite leading eigenvalue. Numeric regression tracking lives in the
            # harness on the mode-converged deck, not here — et[1] is a near-marginal,
            # ill-conditioned, FP-reassociation-sensitive quantity on this grid (sign not even
            # robust across platforms), so no value is pinned.
            et = read(h5["ForceFreeStates/FreeBoundaryStability/eigenmode_energies"])
            @test isfinite(real(et[1]))
        end
        true
    end

    ex5 = joinpath(@__DIR__, "test_data", "regression_solovev_kinetic_calculated")
    @info "Running Solovev self-consistent kinetic-MHD example (kinetic = KineticForces)"
    @test begin
        GeneralizedPerturbedEquilibrium.main([ex5])
        h5open(joinpath(ex5, "gpec.h5"), "r") do h5
            et = read(h5["ForceFreeStates/FreeBoundaryStability/eigenmode_energies"])
            # Smoke test (nerfed mpsi=16, delta_m=0 deck): exercises the full self-consistent
            # KF→FFS kinetic-MHD path end-to-end. NO numeric value is pinned here — the prior
            # imag(et[1]) ≈ -0.711 rtol=0.08 pin was platform-fragile (failed on macOS aarch64
            # at -0.856; issue #273). Physical regression is tracked in the harness on the
            # mode-converged deck (examples/Solovev_kinetic_calculated_example, mpert=32).
            # Assert only nerfed-grid-robust physics: finite, positive total energy,
            # negative kinetic damping (imag sign is robust across configs; only its
            # magnitude is FP-sensitive).
            @test isfinite(real(et[1]))
            @test isfinite(imag(et[1]))
            @test real(et[1]) > 0
            @test imag(et[1]) < 0
        end
        rm(joinpath(ex5, "gpec.h5"); force=true)
        true
    end

    ex6 = joinpath(@__DIR__, "test_data", "regression_solovev_kinetic_nuzero")
    @info "Running Solovev self-consistent kinetic-MHD example (kinetic = KineticForces, nutype=zero)"
    @test begin
        GeneralizedPerturbedEquilibrium.main([ex6])
        h5open(joinpath(ex6, "gpec.h5"), "r") do h5
            et = read(h5["ForceFreeStates/FreeBoundaryStability/eigenmode_energies"])
            # Smoke test (nerfed mpsi=16, delta_m=0 deck): exercises the collisionless
            # (nutype="zero") real-x-space energy-integral path end-to-end — the #281 fix —
            # without faulting/NaN (the bug this guards against). The precise collisionless
            # physics is locked down by the deterministic runtests_kinetic.jl unit tests
            # (tail-pole, ν→0⁺ limit, Ω′<0); the mode-converged eigenvalue is pinned in the
            # harness (calculated example deck run collisionless via override). No value pinned here — only
            # finite, positive total energy, negative resonant (Landau) damping.
            @test isfinite(real(et[1]))
            @test isfinite(imag(et[1]))
            @test real(et[1]) > 0
            @test imag(et[1]) < 0
        end
        rm(joinpath(ex6, "gpec.h5"); force=true)
        true
    end
end
