@testset "Direct equilibrium: magnetic-axis search robustness" begin
    using GeneralizedPerturbedEquilibrium.Equilibrium
    using GeneralizedPerturbedEquilibrium.Equilibrium: EquilibriumConfig, read_efit, direct_position!

    data_dir = joinpath(@__DIR__, "test_data")

    # A 257x257 TJ circular geqdsk (R0 = 2 m) on which the 2-D psi spline's
    # d(Bz)/dR passes through zero at Newton's second iterate. Undamped Newton
    # previously stepped -4.3 m and died with "Jacobian matrix is singular".
    # The step cap keeps the iterate near the axis until the Hessian recovers, and
    # capped Newton then converges on its own (no fallback in the log).
    @testset "previously divergent geqdsk now converges to the axis" begin
        cfg = EquilibriumConfig(;
            eq_filename=joinpath(data_dir, "TJ_circular_axis_newton_regression.geqdsk"),
            eq_type="efit")
        rp = read_efit(cfg)
        ro, zo, _, _ = @test_logs (:info, r"^Magnetic axis found at R = [0-9.]+, Z = -?[0-9.]+$") match_mode=:any direct_position!(rp)
        @test isapprox(ro, 2.0; atol=1e-3)
        @test abs(zo) < 1e-3
    end

    # A second failure mode: the Hessian is indefinite over several cells around the
    # axis (dB_z/dR swings between ~5 and ~0), so even a capped Newton cycles without
    # converging. The first-derivative bisection stage resolves it.
    @testset "cycling Newton is rescued by the bisection stage" begin
        cfg = EquilibriumConfig(;
            eq_filename=joinpath(data_dir, "TJ_circular_axis_newton_cycling.geqdsk"),
            eq_type="efit")
        rp = read_efit(cfg)
        ro, zo, _, _ = @test_logs (:info, r"first-derivative bisection") match_mode=:any direct_position!(rp)
        @test isapprox(ro, 2.0; atol=1e-3)
        @test abs(zo) < 1e-3
    end

    # A healthy file must take the plain Newton path (the log carries no fallback
    # suffix), so its axis -- and everything downstream -- is unchanged.
    @testset "well-behaved geqdsk takes the plain Newton path" begin
        cfg = EquilibriumConfig(;
            eq_filename=joinpath(data_dir, "CHEASE_test_data", "EQDSK_COCOS_02"), eq_type="efit")
        rp = read_efit(cfg)
        ro, zo, _, _ = @test_logs (:info, r"^Magnetic axis found at R = [0-9.]+, Z = -?[0-9.]+$") match_mode=:any direct_position!(rp)
        @test 6.5 < ro < 7.5
        @test isfinite(zo)
    end
end
