using LinearAlgebra

# The NTV-limited error-field-correction model on hand-built couplings: the residual spectrum
# projection, the linear and NTV-limited correction currents against the closed-form quadratic,
# and the two correctable-overlap limits.
@testset "NTV limits of error-field correction" begin
    GPEC = GeneralizedPerturbedEquilibrium
    EF = GPEC.ErrorFields
    PE = GPEC.PerturbedEquilibrium

    @testset "residual spectrum" begin
        v = normalize(ComplexF64[1, 2im, -1, 0.5])
        dom = PE.DominantCoupling([3.0, 1.0], hcat(v, normalize(ComplexF64[0, 1, 1, 0])), ones(ComplexF64, 2, 2), [1, 2])
        b = ComplexF64[0.3, -0.2im, 0.7, 1.0]
        r = EF.residual_spectrum(dom, b)
        @test abs(dot(v, r)) < 1e-14                       # nothing left along the mode
        @test r + v * dot(v, b) ≈ b                        # the projection is exact
        @test EF.residual_spectrum(dom, v) ≈ zeros(4) atol = 1e-14
    end

    c = EF.EFCCoupling("efcc", 2.0e-5, 40.0, 0.05, 0.02)    # δ per kAt, %, N·m per kAt² (full, residual)
    δt, T0 = 1.0e-4, 4.0

    @testset "correction current" begin
        # Below the threshold nothing is needed; the linear current is (δ − δ_t)/C_c.
        @test EF.correction_current(0.5δt, c; delta_threshold=δt, torque_budget=T0) == 0.0
        @test EF.correction_current(3δt, c; delta_threshold=δt, torque_budget=T0, ntv=false) ≈ 2δt / c.delta_per_kat
        # With NTV the current is the smaller root of a I² − C I + (δ − δ_t) = 0, a = δ_t T_r / T_0.
        a = δt * c.torque_residual_per_kat2 / T0
        I = EF.correction_current(2δt, c; delta_threshold=δt, torque_budget=T0)
        @test a * I^2 - c.delta_per_kat * I + δt ≈ 0 atol = 1e-18
        @test I > δt / c.delta_per_kat                     # NTV always costs extra current
        # Safety factor scales the target; zero residual torque recovers the linear current.
        @test EF.correction_current(3δt, c; delta_threshold=δt, torque_budget=T0, safety_factor=2.0, ntv=false) ≈ δt / c.delta_per_kat
        c0 = EF.EFCCoupling("x", c.delta_per_kat, 40.0, 0.05, 0.0)
        @test EF.correction_current(3δt, c0; delta_threshold=δt, torque_budget=T0) ≈ 2δt / c.delta_per_kat
        # Beyond the correctable limit there is no real root.
        lim = EF.max_correctable_overlap(c; delta_threshold=δt, torque_budget=T0)
        @test isnan(EF.correction_current(1.01 * lim.with_ntv, c; delta_threshold=δt, torque_budget=T0))
        @test !isnan(EF.correction_current(0.99 * lim.with_ntv, c; delta_threshold=δt, torque_budget=T0))
    end

    @testset "limits and curve" begin
        lim = EF.max_correctable_overlap(c; delta_threshold=δt, torque_budget=T0)
        @test lim.with_ntv ≈ δt + c.delta_per_kat^2 * T0 / (4 * δt * c.torque_residual_per_kat2)
        @test lim.torque_only ≈ c.delta_per_kat * sqrt(T0 / c.torque_full_per_kat2)
        # At the NTV limit the discriminant vanishes and the current tends to C_c / (2a).
        a = δt * c.torque_residual_per_kat2 / T0
        @test EF.correction_current((1 - 1e-6) * lim.with_ntv, c; delta_threshold=δt, torque_budget=T0) ≈ c.delta_per_kat / (2a) rtol = 1e-2
        curve = EF.efc_current_curve(c; delta_threshold=δt, torque_budget=T0, delta_max=20, npoints=200)
        @test length(curve.delta_ef) == 200 && curve.delta_ef[end] ≈ 20δt
        @test all(curve.current_linear .>= 0)
        @test all(isnan.(curve.current_ntv[curve.delta_ef.>lim.with_ntv]))
        @test all(.!isnan.(curve.current_ntv[curve.delta_ef.<lim.with_ntv]))
        @test all(curve.current_ntv[.!isnan.(curve.current_ntv)] .>= curve.current_linear[.!isnan.(curve.current_ntv)] .- 1e-12)
        @test curve.with_ntv == lim.with_ntv && curve.torque_only == lim.torque_only
        # A negative torque (the other rotation sense) consumes the budget like a positive one.
        negative = EF.EFCCoupling("z", c.delta_per_kat, 40.0, -c.torque_full_per_kat2, -c.torque_residual_per_kat2)
        @test EF.max_correctable_overlap(negative; delta_threshold=δt, torque_budget=T0) == lim
        @test EF.correction_current(2δt, negative; delta_threshold=δt, torque_budget=T0) == EF.correction_current(2δt, c; delta_threshold=δt, torque_budget=T0)
        no_torque = EF.EFCCoupling("y", c.delta_per_kat, 40.0, 0.0, 0.0)
        @test EF.max_correctable_overlap(no_torque; delta_threshold=δt, torque_budget=T0) == (; with_ntv=Inf, torque_only=Inf)
    end
end
