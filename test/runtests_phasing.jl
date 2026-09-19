using LinearAlgebra

# Coil-array phasing maps on synthetic sensitivities: the closed-form two-array map, its
# extremes, the three-array grid shape, the resonant-fraction bound, and the argument guards.
@testset "coil-array phasing" begin
    GPEC = GeneralizedPerturbedEquilibrium
    EF = GPEC.ErrorFields
    PE = GPEC.PerturbedEquilibrium

    # A dominant mode on a 4-mode basis and three arrays with known spectra and ampere-turns.
    v = normalize(ComplexF64[1, 2im, -1, 0.5])
    dom = PE.DominantCoupling([3.0, 1.0], hcat(v, normalize(ComplexF64[0, 1, 1, 0])), ones(ComplexF64, 2, 2), [1, 2])
    b1 = ComplexF64[1.0, 0.0, 0.0, 0.0] .* 2e-3
    b2 = ComplexF64[0.0, 1.0, 0.0, 0.0] .* 3e-3
    b3 = ComplexF64[0.5, 0.5, 0.5, 0.5] .* 1e-3
    nominal = hcat(b1, b2, b3)
    b_t0 = 2.0
    sens = EF.CoilSensitivities(["L", "M", "U"], [1, 2, 3, 4], [1, 1, 1, 1], b_t0, nominal, zeros(ComplexF64, 4, 3, 3), zeros(ComplexF64, 4, 3, 3),
        zeros(3, 3), zeros(3, 3), [1000.0, 2000.0, 500.0], [10.0, 5.0, 4.0])
    kat = [10.0 * 1.0, 5.0 * 2.0, 4.0 * 0.5]           # kA·turns
    δ = [dot(v, nominal[:, j]) / kat[j] / b_t0 for j in 1:3]

    @testset "two arrays: closed form" begin
        map = EF.phasing_map(sens, dom, ["L", "M"]; nphase=360)
        @test length(map.phase_deg) == 1 && size(map.delta_per_kat) == (360,)
        @test map.delta_per_kat_each ≈ abs.(δ[1:2])
        φ = deg2rad.(map.phase_deg[1])
        @test map.delta_per_kat ≈ abs.(δ[1] .+ δ[2] .* cis.(φ))
        # Resonant fraction is bounded by Cauchy–Schwarz and equals |Vᴴb̃|/‖b̃‖ in percent.
        @test all(0 .<= map.overlap_percent .<= 100 + 1e-9)
        b = nominal[:, 1] ./ kat[1] .+ nominal[:, 2] ./ kat[2] .* cis(φ[10])
        @test map.overlap_percent[10] ≈ 100 * abs(dot(v, b)) / norm(b)
        # The maximum of |δ1 + δ2 e^{iφ}| is at φ = arg(δ1) − arg(δ2), value |δ1| + |δ2|.
        vmax, ph = EF.extreme_phasing(map)
        @test vmax ≈ abs(δ[1]) + abs(δ[2]) rtol = 1e-3
        @test isapprox(mod(deg2rad(ph[1]) - (angle(δ[1]) - angle(δ[2])), 2π), 0; atol=deg2rad(1.01)) ||
              isapprox(mod(deg2rad(ph[1]) - (angle(δ[1]) - angle(δ[2])), 2π), 2π; atol=deg2rad(1.01))
        vmin, _ = EF.extreme_phasing(map; which=:min)
        @test vmin ≈ abs(abs(δ[1]) - abs(δ[2])) rtol = 1e-2
    end

    @testset "three arrays: cumulative phases" begin
        map = EF.phasing_map(sens, dom, ["L", "M", "U"]; nphase=90)
        @test size(map.delta_per_kat) == (90, 90) && size(map.overlap_percent) == (90, 90)
        i, j = 7, 31
        φ1, φ2 = deg2rad(map.phase_deg[1][i]), deg2rad(map.phase_deg[2][j])
        @test map.delta_per_kat[i, j] ≈ abs(δ[1] + δ[2] * cis(φ1) + δ[3] * cis(φ1 + φ2))
        v_o, ph = EF.extreme_phasing(map; quantity=:overlap_percent)
        @test 0 < v_o <= 100 + 1e-9 && length(ph) == 2
    end

    @testset "winding sense is kept in the map" begin
        # The same geometry with a negative winding multiplier is the same array wound the other
        # way: its per-kAt spectrum flips sign, so the map is the original one rotated by 180°.
        flipped = EF.CoilSensitivities(sens.coil_names, sens.m_modes, sens.n_modes, b_t0, hcat(b1, -b2, b3), sens.shift_sensitivity, sens.tilt_sensitivity,
            sens.shift_linearity_residual, sens.tilt_linearity_residual, sens.peak_current, [10.0, -5.0, 4.0])
        m0 = EF.phasing_map(sens, dom, ["L", "M"]; nphase=36)
        m1 = EF.phasing_map(flipped, dom, ["L", "M"]; nphase=36)
        @test m1.delta_per_kat ≈ circshift(m0.delta_per_kat, 18)
        @test m1.delta_per_kat_each ≈ m0.delta_per_kat_each
    end

    @testset "guards" begin
        @test_throws ArgumentError EF.phasing_map(sens, dom, ["L"])
        @test_throws ArgumentError EF.phasing_map(sens, dom, ["L", "X"])
        @test_throws ArgumentError EF.phasing_map(sens, dom, ["L", "M"]; nphase=1)
        @test_throws ArgumentError EF.phasing_map(sens, dom, ["L", "M"]; mode=3)
        dead = EF.CoilSensitivities(sens.coil_names, sens.m_modes, sens.n_modes, b_t0, nominal, sens.shift_sensitivity, sens.tilt_sensitivity,
            sens.shift_linearity_residual, sens.tilt_linearity_residual, [0.0, 2000.0, 500.0], sens.winding_multiplier)
        @test_throws ArgumentError EF.phasing_map(dead, dom, ["L", "M"])
        map = EF.phasing_map(sens, dom, ["L", "M"]; nphase=12)
        @test_throws ArgumentError EF.extreme_phasing(map; quantity=:foo)
        @test_throws ArgumentError EF.extreme_phasing(map; which=:median)
    end
end
