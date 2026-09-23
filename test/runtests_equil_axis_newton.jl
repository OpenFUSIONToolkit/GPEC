@testset "Direct equilibrium: magnetic-axis search robustness" begin
    using Logging
    using GeneralizedPerturbedEquilibrium.Equilibrium
    using GeneralizedPerturbedEquilibrium.Equilibrium: EquilibriumConfig, read_efit, direct_position!

    data_dir = joinpath(@__DIR__, "test_data")
    fallback = r"restarting Newton from a first-derivative bisection"

    # Two 257x257 TJ circular geqdsks (R0 = 2 m, axis at Z = 0) whose ψ has a curvature spike
    # in the cells at the axis. Newton from the midplane guess meets a singular Hessian on the
    # first and cycles on the second; both must be found through the warned fallback, to the
    # same step-size convergence test as the plain path.
    @testset "non-smooth ψ near the axis: $name" for name in ("regression", "cycling")
        cfg = EquilibriumConfig(; eq_filename=joinpath(data_dir, "TJ_circular_axis_newton_$name.geqdsk"), eq_type="efit")
        rp = read_efit(cfg)
        ro, zo, _, _ = @test_logs (:warn, fallback) match_mode = :any direct_position!(rp)
        # The spline's own critical point sits ~3e-4 m from the header axis (Z ≠ 0 despite up-down
        # symmetry), a symptom of the defective near-axis ψ; it is well inside one grid cell.
        @test isapprox(ro, 2.0; atol=1e-3)
        @test abs(zo) < 1e-3
    end

    # A healthy file takes the plain Newton path with no warning, so its axis is unchanged.
    @testset "well-behaved geqdsk takes the plain Newton path" begin
        cfg = EquilibriumConfig(; eq_filename=joinpath(data_dir, "CHEASE_test_data", "EQDSK_COCOS_02"), eq_type="efit")
        rp = read_efit(cfg)
        ro, zo, _, _ = @test_logs min_level = Logging.Warn direct_position!(rp)
        @test 6.5 < ro < 7.5
        @test isfinite(zo)
    end
end
