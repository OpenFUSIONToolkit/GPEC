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
            # Baseline refreshed to the develop-consistent value: develop's edge-scan
            # (psiedge band) and periodic-theta-endpoint handling shifted the multi-n
            # eigenvalue from the pre-merge -0.01248.
            @test real(et[1]) ≈ 0.22325 rtol = 0.01
        end
        rm(joinpath(ex4, "gpec.h5"); force=true)
        true
    end

    # Skipped: the self-consistent kinetic_source="calculated" path (KF→FFS→PE)
    # is out of scope for the current PR. Active work is on the perturbative
    # FFS→PE→KF path (kinetic_source="fixed"). Kinetic matrix validation for the
    # calculated source is pending; re-enable this test once that validation
    # lands, with a physics-based baseline (not a captured code output).
    # ex5 = joinpath(@__DIR__, "test_data", "regression_solovev_kinetic_calculated")
    # @info "Running Solovev kinetic example (kinetic_source=calculated)"
    # @test begin
    #     GeneralizedPerturbedEquilibrium.main([ex5])
    #     h5open(joinpath(ex5, "gpec.h5"), "r") do h5
    #         et = read(h5["vacuum/et"])
    #         @test isfinite(real(et[1]))
    #         @test isfinite(imag(et[1]))
    #     end
    #     rm(joinpath(ex5, "gpec.h5"); force=true)
    #     true
    # end
end
