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
            # Previous truncated-integration value was -0.01248; current full-domain value
            # is ≈ -0.18 on Linux x86 (CI baseline -0.193593591803846 across julia_nthreads ×
            # parallel_threads × use_parallel sweeps, bit-identical to 15 digits). Apple
            # silicon / non-x86 BLAS variants drift by up to ~20 % on this kinetic multi-n
            # eigenvalue. We bracket the sign and order of magnitude rather than pin tightly:
            # the eigenvalue must remain negative (kinetic-driven instability) and within
            # an order-of-magnitude band; tight regressions in the edge-dW or kinetic path
            # would still fall outside this bracket.
            @test real(et[1]) < 0          # sign sanity: kinetic-driven instability
            @test -0.30 < real(et[1]) < -0.10  # order-of-magnitude bracket (CI -0.194; Apple ~-0.16)
        end
        rm(joinpath(ex4, "gpec.h5"); force=true)
        true
    end
end
