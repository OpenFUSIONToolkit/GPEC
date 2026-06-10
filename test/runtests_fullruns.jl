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
            et = read(h5["FreeBoundaryStability/eigenmode_energies"])
            @test isfinite(real(et[1]))
            @test real(et[1]) > 0  # Solovev is stable (positive total energy)
            @test real(et[1]) ≈ 1527.65 rtol = 0.01
        end
        rm(joinpath(ex3, "gpec.h5"); force=true)
        true
    end

    ex4 = joinpath(@__DIR__, "test_data", "regression_solovev_kinetic_multi_n")
    @info "Running Solovev kinetic multi-n example (kinetic_factor=1e-9, nn_low=1, nn_high=2)"
    @test begin
        GeneralizedPerturbedEquilibrium.main([ex4])
        h5open(joinpath(ex4, "gpec.h5"), "r") do h5
            et = read(h5["FreeBoundaryStability/eigenmode_energies"])
            @test isfinite(real(et[1]))
            # et[1] is the single unstable, near-marginal kinetic eigenvalue; the rest of
            # the spectrum is large and positive (stable). Being a small difference of large
            # plasma/vacuum energies, et[1] is ill-conditioned: @inbounds @simd floating-point
            # reassociation (active under check-bounds=auto, disabled under Pkg.test's
            # --check-bounds=yes) perturbs every eigenvalue by ~0.1%, which the marginal et[1]
            # amplifies into a platform-dependent swing — observed real(et[1]) is -0.1936
            # (macOS aarch64, check-bounds=auto), -0.1612 (macOS aarch64, check-bounds=yes),
            # and -0.1480 (Linux x86 CI, check-bounds=yes). All are the same physics. We
            # bracket the marginal et[1] loosely and rely on the well-conditioned eigenvalues
            # et[2]/et[3] (which vary only ~0.1%) pinned tightly to catch real regressions
            # (kinetic factor, edge-dW path, parallel BVP).
            @test real(et[1]) < 0                            # genuinely unstable
            @test -0.30 < real(et[1]) < -0.10                # marginal: platform/FP sensitive
            @test isapprox(real(et[2]), 17.74; rtol=1e-2)    # well-conditioned stable mode
            @test isapprox(real(et[3]), 17.49; rtol=1e-2)    # well-conditioned stable mode
        end
        rm(joinpath(ex4, "gpec.h5"); force=true)
        true
    end
end
