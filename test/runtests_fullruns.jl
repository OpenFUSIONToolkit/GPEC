using HDF5

# Run GeneralizedPerturbedEquilibrium.main on the provided example directories and assert it completes without throwing.
@testset "Full ForceFreeStates runs" begin
    ex1 = joinpath(@__DIR__, "test_data", "regression_solovev_ideal_example")
    @info "Running Solovev ideal example"
    @test begin
        GeneralizedPerturbedEquilibrium.main([ex1])
        true
    end

    ex2 = joinpath(@__DIR__, "test_data", "regression_solovev_ideal_example_multi_n")
    @info "Running Solovev ideal multi-n example"
    @test begin
        GeneralizedPerturbedEquilibrium.main([ex2])
        true
    end

    ex3 = joinpath(@__DIR__, "test_data", "regression_solovev_kinetic_example")
    @info "Running Solovev kinetic example (kinetic_source=fixed, kinetic_factor=1e-9)"
    @test begin
        GeneralizedPerturbedEquilibrium.main([ex3])
        h5open(joinpath(ex3, "gpec.h5"), "r") do h5
            # Smoke test: this nerfed deck (mpsi=16, delta_m=0) is for "does it run / no NaN",
            # not numeric regression — the mode-converged physical value is pinned in the
            # regression harness (examples/Solovev_kinetic_calculated_example). Assert only
            # nerfed-grid-robust facts: finite and positive (Solovev is stable).
            et = read(h5["FreeBoundaryStability/XiNorm/eigenmode_energies"])
            @test isfinite(real(et[1]))
            @test real(et[1]) > 0
        end
        rm(joinpath(ex3, "gpec.h5"); force=true)
        true
    end

    ex4 = joinpath(@__DIR__, "test_data", "regression_solovev_kinetic_multi_n")
    @info "Running Solovev kinetic multi-n example (kinetic_factor=1e-9, nn_low=1, nn_high=2)"
    @test begin
        GeneralizedPerturbedEquilibrium.main([ex4])
        h5open(joinpath(ex4, "gpec.h5"), "r") do h5
            # Smoke test only (nerfed mpsi=16, delta_m=0 deck): runs without faulting and
            # produces a finite leading eigenvalue. Numeric regression tracking lives in the
            # harness on the mode-converged deck, not here — et[1] is a near-marginal,
            # ill-conditioned, FP-reassociation-sensitive quantity on this grid (sign not even
            # robust across platforms), so no value is pinned.
            et = read(h5["FreeBoundaryStability/XiNorm/eigenmode_energies"])
            @test isfinite(real(et[1]))
        end
        rm(joinpath(ex4, "gpec.h5"); force=true)
        true
    end

    ex5 = joinpath(@__DIR__, "test_data", "regression_solovev_kinetic_calculated")
    @info "Running Solovev self-consistent kinetic-MHD example (kinetic_source=calculated)"
    @test begin
        GeneralizedPerturbedEquilibrium.main([ex5])
        h5open(joinpath(ex5, "gpec.h5"), "r") do h5
            et = read(h5["FreeBoundaryStability/XiNorm/eigenmode_energies"])
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
    @info "Running Solovev self-consistent kinetic-MHD example (kinetic_source=calculated, nutype=zero)"
    @test begin
        GeneralizedPerturbedEquilibrium.main([ex6])
        h5open(joinpath(ex6, "gpec.h5"), "r") do h5
            et = read(h5["FreeBoundaryStability/XiNorm/eigenmode_energies"])
            # Smoke test (nerfed mpsi=16, delta_m=0 deck): exercises the collisionless
            # (nutype="zero") real-x-space energy-integral path end-to-end — the #281 fix —
            # without faulting/NaN (the bug this guards against). The precise collisionless
            # physics is locked down by the deterministic runtests_kinetic.jl unit tests
            # (tail-pole, ν→0⁺ limit, Ω′<0); the mode-converged eigenvalue is pinned in the
            # harness (examples/Solovev_kinetic_nuzero_example). No value pinned here — only
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
