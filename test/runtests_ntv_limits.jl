using LinearAlgebra
using HDF5

# The NTV-limited error-field-correction model on hand-built couplings: the residual spectrum
# projection, the linear and NTV-limited correction currents against the closed-form quadratic,
# the two correctable-overlap limits, and the torque-balance model on tabulated torques.
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
        # For this coupling the residual torque exhausts the budget before the quadratic's tangency:
        # the limit is C_c √(T_0/T_res) and the current there is the budget-exhausting √(T_0/T_res).
        exhausted = c.delta_per_kat * sqrt(T0 / c.torque_residual_per_kat2)
        @test exhausted > 2δt && lim.with_ntv ≈ exhausted
        @test lim.torque_only ≈ c.delta_per_kat * sqrt(T0 / c.torque_full_per_kat2)
        @test EF.correction_current((1 - 1e-6) * lim.with_ntv, c; delta_threshold=δt, torque_budget=T0) ≈ sqrt(T0 / c.torque_residual_per_kat2) rtol = 1e-2
        # A weaker coupling reaches the tangency first: there the discriminant vanishes and the current tends to C_c / (2a).
        weak = EF.EFCCoupling("weak", 1.0e-5, 40.0, 0.05, 0.02)
        limw = EF.max_correctable_overlap(weak; delta_threshold=δt, torque_budget=T0)
        @test weak.delta_per_kat * sqrt(T0 / weak.torque_residual_per_kat2) <= 2δt
        @test limw.with_ntv ≈ δt + weak.delta_per_kat^2 * T0 / (4 * δt * weak.torque_residual_per_kat2)
        a = δt * weak.torque_residual_per_kat2 / T0
        @test EF.correction_current((1 - 1e-6) * limw.with_ntv, weak; delta_threshold=δt, torque_budget=T0) ≈ weak.delta_per_kat / (2a) rtol = 1e-2
        @test isnan(EF.correction_current(1.01 * limw.with_ntv, weak; delta_threshold=δt, torque_budget=T0))
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

    # A tabulated torque: linear in the rotation shift, braking at the nominal rotation, zero at the offset.
    ω_ref, ω_off = 5.0e4, -2.0e4                     # rad/s: co-rotating reference, counter offset
    Δ = collect(range(-6.0e4, 6.0e4; length=13))
    # A positive torque brakes the co-rotation (TORQUE_ROTATION_SIGN = −1): +0.02 N·m/kAt² at Δ = 0, zero at the offset, reversed beyond.
    Tlin(Δ) = 0.02 * (1 - Δ / ω_off)
    scanned = EF.EFCCoupling("efcc", 2.0e-5, 40.0, 0.05, 0.02, Δ, 2.5 .* Tlin.(Δ), Tlin.(Δ), ω_ref, ω_off, [0.0, 0.5, 1.0], zeros(3, 13), zeros(3, 13))
    constant = EF.EFCCoupling("const", 2.0e-5, 40.0, 0.05, 0.02, Δ, fill(0.05, 13), fill(0.02, 13), ω_ref, NaN, Float64[], zeros(0, 1), zeros(0, 1))

    @testset "torque table" begin
        @test EF.has_rotation_scan(scanned) && !EF.has_rotation_scan(c)
        @test EF.torque_at(scanned, 0.0) == 0.02
        @test EF.torque_at(scanned, ω_off) ≈ 0 atol = 1e-12
        @test EF.torque_at(scanned, 1.23e4) ≈ Tlin(1.23e4)                     # linear table interpolates exactly
        @test isnan(EF.torque_at(scanned, 7e4)) && EF.torque_at(c, 1e9) == c.torque_residual_per_kat2
        @test EF.torque_at(scanned, 0.0; field=:full) == 0.05
        @test EF.torque_zero_crossings(scanned) ≈ [ω_off] atol = 1e-9
        @test isempty(EF.torque_zero_crossings(constant))
        span = EF.rotation_scan_span(3.0e4, -1.0e4; span_factor=1.5, offset_factor=2.0)
        @test span.span ≈ 4.5e4 && span.offset_estimate ≈ -2.0e4
        @test_throws ArgumentError EF.rotation_scan_span(0.0, 0.0)
    end

    @testset "torque balance" begin
        # Balance Δ·T_0/ω_ref = T(Δ)·I² with the linear table has the closed form Δ = −a I² ω_off / (ω_off T_0/ω_ref ... ) — check it numerically instead.
        I = 3.0
        Δb = EF.rotation_shift(scanned, I; torque_budget=T0)
        @test Δb < 0 && Δb > ω_off                                              # brakes toward, not past, the offset
        @test Δb * T0 / ω_ref ≈ EF.TORQUE_ROTATION_SIGN * Tlin(Δb) * I^2 rtol = 1e-6
        @test EF.rotation_shift(scanned, 0.0; torque_budget=T0) == 0.0
        # As I grows the balance point approaches the offset and never crosses it: the torque vanishes there.
        Δbig = EF.rotation_shift(scanned, 50.0; torque_budget=T0)
        @test ω_off < Δbig < Δb
        # A constant braking torque reproduces the linear budget: Δ/ω_ref = −|T| I²/T_0, and the same threshold factor.
        Δc = EF.rotation_shift(constant, I; torque_budget=T0)
        @test Δc / ω_ref ≈ -0.02 * I^2 / T0
        @test EF.threshold_factor(constant, I; torque_budget=T0) ≈ EF.threshold_factor(constant, I; torque_budget=T0, model=:linear)
        # Rotation exponent: α = 2 lowers the threshold more than α = 1; α = 0 not at all.
        f1 = EF.threshold_factor(scanned, I; torque_budget=T0)
        @test 0 < EF.threshold_factor(scanned, I; torque_budget=T0, rotation_exponent=2.0) < f1 < 1
        @test EF.threshold_factor(scanned, I; torque_budget=T0, rotation_exponent=0.0) == 1
        # Beyond the scanned span the balance has no root.
        @test isnan(EF.rotation_shift(constant, 1e3; torque_budget=T0))
        @test_throws ArgumentError EF.rotation_shift(c, 1.0; torque_budget=T0)
    end

    @testset "correction current with the balance" begin
        # The constant table with the balance equals the closed-form quadratic (α = 1, braking).
        for d in (1.5δt, 2.5δt)
            @test EF.correction_current(d, constant; delta_threshold=δt, torque_budget=T0) ≈
                  EF.correction_current(d, constant; delta_threshold=δt, torque_budget=T0, model=:linear) rtol = 1e-6
        end
        limc = EF.max_correctable_overlap(constant; delta_threshold=δt, torque_budget=T0)
        liml = EF.max_correctable_overlap(constant; delta_threshold=δt, torque_budget=T0, model=:linear)
        @test limc.with_ntv ≈ liml.with_ntv rtol = 1e-4
        @test limc.torque_only ≈ liml.torque_only rtol = 1e-4
        # With the offset-limited table the torque saturates, so more overlap is correctable than the linear model says.
        # (2.5 δ_t is inside the linear model's limit of exactly 3 δ_t for this coupling.)
        Ib = EF.correction_current(2.5δt, scanned; delta_threshold=δt, torque_budget=T0)
        Il = EF.correction_current(2.5δt, scanned; delta_threshold=δt, torque_budget=T0, model=:linear)
        @test Ib > 1.5δt / scanned.delta_per_kat
        @test Ib < Il
        lims = EF.max_correctable_overlap(scanned; delta_threshold=δt, torque_budget=T0)
        @test lims.with_ntv > EF.max_correctable_overlap(scanned; delta_threshold=δt, torque_budget=T0, model=:linear).with_ntv
        curve = EF.efc_current_curve(scanned; delta_threshold=δt, torque_budget=T0, delta_max=10, npoints=50)
        @test curve.model === :torque_balance && length(curve.threshold_factor) == 50
        @test all(0 .< filter(!isnan, curve.threshold_factor) .<= 1)
        # No scan: the model falls back to the linear budget.
        @test EF.efc_current_curve(c; delta_threshold=δt, torque_budget=T0, npoints=5).model === :linear
        @test_throws ArgumentError EF.correction_current(3δt, scanned; delta_threshold=δt, torque_budget=T0, model=:foo)
    end

    @testset "HDF5 round trip with a scan" begin
        tmp = tempname() * ".h5"
        h5open(tmp, "w") do f
            EF.write_to_hdf5!(f, [scanned, c])
        end
        back = EF.read_efc_couplings(tmp)
        @test back[1].rotation_shift == Δ && back[1].torque_residual_scan == Tlin.(Δ) && back[1].omega_reference == ω_ref
        @test size(back[1].torque_full_profile) == (3, 13) && back[1].psi == [0.0, 0.5, 1.0]
        @test !EF.has_rotation_scan(back[2]) && back[2].torque_full_per_kat2 == c.torque_full_per_kat2
        rm(tmp)
    end
end
