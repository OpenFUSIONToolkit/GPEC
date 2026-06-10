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
            et = read(h5["vacuum/et"])
            @test isfinite(real(et[1]))
            @test real(et[1]) > 0  # Solovev is stable (positive total energy)
            @test real(et[1]) ≈ 16.480 rtol = 0.01
        end
        rm(joinpath(ex3, "gpec.h5"); force=true)
        true
    end

    ex4 = joinpath(@__DIR__, "test_data", "regression_solovev_kinetic_multi_n")
    @info "Running Solovev kinetic multi-n example (kinetic_factor=1e-9, nn_low=1, nn_high=2)"
    @test begin
        GeneralizedPerturbedEquilibrium.main([ex4])
        h5open(joinpath(ex4, "gpec.h5"), "r") do h5
            et = read(h5["vacuum/et"])
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

    ex5 = joinpath(@__DIR__, "test_data", "regression_solovev_kinetic_calculated")
    @info "Running Solovev self-consistent kinetic-MHD example (kinetic_source=calculated)"
    @test begin
        GeneralizedPerturbedEquilibrium.main([ex5])
        h5open(joinpath(ex5, "gpec.h5"), "r") do h5
            et = read(h5["vacuum/et"])
            # Self-consistent KF→FFS kinetic-MHD eigenvalue with full-strength kinetic
            # matrices (kinetic_factor=1.0). The Julia kinetic-DCON path is validated
            # against Fortran kinetic DCON on the DIIID benchmark to <2% on both Re and
            # Im of et[1] (PR #112). Here the Solovev value is the regression anchor:
            # Re(et[1]) is a large, well-conditioned eigenvalue (pinned tight); Im(et[1])
            # is the kinetic damping rate, which is more FP-sensitive (bracketed loosely).
            @test isfinite(real(et[1]))
            @test isfinite(imag(et[1]))
            @test real(et[1]) ≈ 15.888 rtol = 0.01
            @test imag(et[1]) < 0                       # kinetic damping
            @test imag(et[1]) ≈ -0.711 rtol = 0.05
        end
        rm(joinpath(ex5, "gpec.h5"); force=true)
        true
    end
end
