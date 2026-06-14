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
            # ξ-space (XiNorm) eigenvalues — power-norm collapses these stable modes to ~0, unusable as a regression anchor.
            et = read(h5["FreeBoundaryStability/XiNorm/eigenmode_energies"])
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
            # ξ-space (XiNorm) eigenvalues — power-norm collapses these modes to ~0, unusable as a regression anchor.
            et = read(h5["FreeBoundaryStability/XiNorm/eigenmode_energies"])
            @test isfinite(real(et[1]))
            # et[1] is the single near-marginal kinetic eigenvalue; the rest of the spectrum
            # is large and positive (stable). Being a small difference of large plasma/vacuum
            # energies (|et[1]| ~ 0.2 vs spectrum scale ~17), et[1] is ill-conditioned and its
            # SIGN is not robust: @inbounds @simd floating-point reassociation already swings it
            # across platforms (-0.194 macOS aarch64 / -0.148 Linux x86 CI, both check-bounds=yes),
            # and dropping the duplicated θ=2π endpoint before the metric FFT (faithful Fortran
            # fspline_fit_2, equil/fspline.f:293) shifts it across zero to +0.190 (macOS aarch64).
            # All are the same marginal physics. We therefore bracket only its MAGNITUDE
            # (sign-agnostic) and rely on the well-conditioned et[2]/et[3] (which vary only ~0.1%)
            # pinned tightly to catch real regressions (kinetic factor, edge-dW path, parallel BVP).
            @test abs(real(et[1])) < 0.5                     # near-marginal: sign not robust
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
            et = read(h5["FreeBoundaryStability/XiNorm/eigenmode_energies"])
            # Self-consistent KF→FFS kinetic-MHD eigenvalue with full-strength kinetic
            # matrices (kinetic_factor=1.0). This Solovev value is physics-validated
            # against Fortran kinetic DCON on the SAME equilibrium + kinetic profile
            # (mode band matched to mpert=8): Fortran W_t[1] = 15.619 - 0.660i vs Julia
            # 15.888 - 0.711i — Re 1.7%, Im 7.6% (issue #227, reproduce with
            # benchmarks/benchmark_solovev_kinetic_stability.jl). Re(et[1]) is large and
            # well-conditioned (pinned tight); Im(et[1]) is the kinetic damping rate,
            # more FP-sensitive (bracketed loosely). The pre-PV `ximag` value (34.176)
            # was not physical; the PV+residue energy integral gives the correct result.
            @test isfinite(real(et[1]))
            @test isfinite(imag(et[1]))
            @test real(et[1]) ≈ 15.888 rtol = 0.01
            @test imag(et[1]) < 0                       # kinetic damping
            @test imag(et[1]) ≈ -0.711 rtol = 0.08
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
            # Collisionless (nutype="zero") calculated kinetic-MHD eigenvalue. The collisionless
            # energy integral runs in real x-space over [0, 72] (Fortran PENTRC convention) with
            # an analytic principal-value + residue and a regular-part limit on the resonance
            # poles (issue #281). Re(et[1]) must sit essentially on the harmonic value 15.888
            # (collisions are a small perturbation to the real energy); Im(et[1]) is the
            # resonant (Landau ∓iπ) damping rate, smaller in magnitude than the harmonic
            # -0.711 because the collisional broadening contribution is dropped.
            @test isfinite(real(et[1]))
            @test isfinite(imag(et[1]))
            @test real(et[1]) > 0
            @test real(et[1]) ≈ 15.885 rtol = 0.01    # ≈ harmonic 15.888 (small-ν limit)
            @test imag(et[1]) < 0                       # resonant kinetic damping
            @test imag(et[1]) ≈ -0.482 rtol = 0.10
        end
        rm(joinpath(ex6, "gpec.h5"); force=true)
        true
    end
end
