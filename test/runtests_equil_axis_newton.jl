@testset "Direct equilibrium: magnetic-axis Newton step cap" begin
    using GeneralizedPerturbedEquilibrium.Equilibrium
    using GeneralizedPerturbedEquilibrium.Equilibrium: EquilibriumConfig, read_efit, direct_position!

    data_dir = joinpath(@__DIR__, "test_data")

    # A 257x257 TJ circular geqdsk (R0 = 2 m) on which the 2-D psi spline's
    # d(Bz)/dR passes through zero at Newton's second iterate. Undamped Newton
    # previously stepped -4.3 m and died with "Jacobian matrix is singular".
    # The step cap keeps the iterate near the axis until the Hessian recovers.
    @testset "previously divergent geqdsk now converges to the axis" begin
        cfg = EquilibriumConfig(;
            eq_filename=joinpath(data_dir, "TJ_circular_axis_newton_regression.geqdsk"),
            eq_type="efit")
        rp = read_efit(cfg)
        ro, zo, _, _ = direct_position!(rp)
        @test isapprox(ro, 2.0; atol=1e-3)
        @test abs(zo) < 1e-3
    end

    # The cap must be inert on a healthy file: the axis it finds is the one the
    # uncapped iteration found before the change (pinned to the previous value).
    @testset "cap is inactive on a well-behaved geqdsk" begin
        cfg = EquilibriumConfig(;
            eq_filename=joinpath(data_dir, "CHEASE_test_data", "EQDSK_COCOS_02"), eq_type="efit")
        rp = read_efit(cfg)
        ro, zo, _, _ = direct_position!(rp)
        @test 6.5 < ro < 7.5
        @test isfinite(zo)
    end
end
