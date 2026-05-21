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
            # Edge-dW scan is now diagnostic-only; integration always reaches qhigh/psihigh.
            # Previous value (-0.01248) reflected the old truncated-integration behaviour.
            # The earlier "rtol=0.2 because thread-count sensitive" comment is now stale:
            # a sweep over julia_nthreads ∈ {1,2,4} × parallel_threads ∈ {1,2,4} ×
            # use_parallel ∈ {true,false} (9 runs total) on this exact test case
            # produced et_re = -0.193593591803846 bit-identical to 15 digits in every
            # configuration. The 15% drift was historical and is resolved by the
            # edge-dW truncation decoupling (5d5b8eed). rtol=1e-6 leaves cross-platform
            # floating-point headroom while still catching any real regression.
            @test real(et[1]) ≈ -0.193593591803846 rtol = 1e-6
        end
        rm(joinpath(ex4, "gpec.h5"); force=true)
        true
    end
end
