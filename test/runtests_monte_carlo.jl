using Random
using Statistics
using LinearAlgebra

# The tolerance Monte Carlo kernel on synthetic sensitivity tables: the closed-form |δ| density
# of one axisymmetric coil under a Flat shift tolerance, agreement with a literal transcription
# of the OMFIT tool's model in the limit where the two coincide (S_y = ±i·S_x), the corrected
# histogram, coherent groups, the coil subset, and bit-for-bit reproducibility under a seed.
@testset "tolerance Monte Carlo" begin
    GPEC = GeneralizedPerturbedEquilibrium
    EF = GPEC.ErrorFields
    FT = GPEC.ForcingTerms

    # A synthetic table: names, nominal overlaps, S = (Sx, Sy, Sz) per m, T per deg.
    function synthetic_table(names, δ0, S, T)
        n = length(names)
        rms(M) = [sqrt((abs2(M[1, j]) + abs2(M[2, j])) / 2) for j in 1:n]
        canc(M) = [EF.cancelling_offset(δ0[j], M[1, j], M[2, j])[i] for i in 1:2, j in 1:n]
        return EF.SensitivityTable(names, 1, ComplexF64.(δ0), ComplexF64.(S), ComplexF64.(T), rms(S), rms(T), canc(S), canc(T))
    end
    hoops(names; heights=zeros(length(names))) = [FT.make_pf_hoop(; radius=1.5, height=heights[i], name=nm) for (i, nm) in enumerate(names)]
    pdf_moments(res) = begin
        centers = (res.bin_edges[1:end-1] .+ res.bin_edges[2:end]) ./ 2
        w = diff(res.bin_edges)
        (sum(res.pdf .* w), sum(centers .* res.pdf .* w))
    end

    @testset "one axisymmetric coil, Flat shift tolerance: box density" begin
        # δ = S·conj(Δ) with |Δ| = R·u uniform in radius → |δ| uniform on [0, |S|R].
        Sx = 4.0 + 3.0im
        table = synthetic_table(["a"], [0.0], [Sx -im*Sx; 0.0 0.0]', zeros(3, 1))
        ts = EF.parse_tolerance_toml("""
            [[ErrorFields.coil]]
            name = "a"
            shift_tol_mm = 2.0
            radial_shape = "flat"
            """)
        ctrl = EF.MonteCarloControl(; nsample=200_000, nbatch=2, seed=3, nbins=100, delta_max=abs(Sx) * 2e-3)
        res = EF.run_monte_carlo(table, ts, hoops(["a"]), ctrl)
        total, mean_δ = pdf_moments(res)
        @test total ≈ 1 atol = 1e-2
        @test isapprox(mean_δ, abs(Sx) * 1e-3; rtol=2e-2)
        @test res.mean_abs_delta ≈ abs(Sx) * 1e-3 rtol = 2e-2
        @test all(abs.(res.pdf .- 1 / (abs(Sx) * 2e-3)) .< 0.15 / (abs(Sx) * 2e-3))
        @test res.clamped_fraction == 0
        @test res.delta_nominal == 0
        # Everything is correctable: the corrected |δ| is the intrinsic one halved, a box on [0, |S|R/2].
        half = ctrl.nbins ÷ 2
        @test all(abs.(res.pdf_efc[1:half] .- 2 / (abs(Sx) * 2e-3)) .< 0.3 / (abs(Sx) * 2e-3))
        @test all(res.pdf_efc[half+1:end] .== 0)
        @test res.mean_abs_delta_efc ≈ res.mean_abs_delta / 2 rtol = 2e-2
        @test size(res.pdf_batches) == (100, 2)
    end

    @testset "reduction to the OMFIT model and seed reproducibility" begin
        # Three coils with S_y = −i·S_x and T_y = +i·T_x (the axisymmetric identities) so the
        # complex model equals OMFIT's |S|·r·e^{iφ} with a free phase; compare |δ| histograms of
        # the kernel and of a literal transcription of the OMFIT sampling on the same tolerances.
        names = ["c1", "c2", "c3"]
        δ0 = [1e-4 + 2e-4im, -3e-4 + 0.5e-4im, 2e-4 - 1e-4im]
        Sx = [0.3 + 0.1im, -0.2 + 0.25im, 0.15 - 0.3im]
        Tx = [0.01 + 0.02im, 0.03 - 0.01im, -0.02 + 0.015im]
        S = hcat([[Sx[j], -im * Sx[j], 0.0] for j in 1:3]...)
        T = hcat([[Tx[j], im * Tx[j], 0.0] for j in 1:3]...)
        table = synthetic_table(names, δ0, S, T)
        tol = """
            [ErrorFields.defaults]
            radial_shape = "flat"
            [[ErrorFields.coil]]
            name = "c1"
            shift_tol_mm = 1.0
            tilt_tol = 0.02
            shift_sigma_mm = 0.2
            [[ErrorFields.coil]]
            name = "c2"
            shift_tol_mm = 0.5
            tilt_tol = 0.01
            radial_shape = "hollow"
            [[ErrorFields.coil]]
            name = "c3"
            shift_tol_mm = 2.0
            tilt_tol = 0.0
            [ErrorFields.correctability]
            uncorrectable_coils = ["c3"]
            efc_factor = 2.0
            [ErrorFields.other_field]
            magnitude = 5e-5
            radial_shape = "ring"
            """
        ts = EF.parse_tolerance_toml(tol)
        ctrl = EF.MonteCarloControl(; nsample=300_000, nbatch=2, seed=11, nbins=200)
        res = EF.run_monte_carlo(table, ts, hoops(names), ctrl)

        # Literal OMFIT transcription: dfac = tol·rand^p·e^{iφ}, dunc = σ·randn·e^{iφ}, deltas =
        # dc_nerr + dc_disp·(dfac + dunc) + dc_tilt·(tfac + tunc), other = mag·rand^0·e^{iφ},
        # EFC divides correctable rows and the other budget.
        rng = Xoshiro(99)
        N = 600_000
        dc_disp = abs.(Sx)
        dc_tilt = abs.(Tx)
        disp_tol = [1e-3, 0.5e-3, 2e-3]
        tilt_tol = [0.02, 0.01, 0.0]
        disp_unc = [0.2e-3, 0.0, 0.0]
        pw = [1.0, 1 / 3, 1.0]
        correct = [true, true, false]
        δ_all = zeros(N)
        δ_efc = zeros(N)
        for s in 1:N
            tot = 0.0im
            corr = 0.0im
            unc = 0.0im
            for c in 1:3
                dfac = disp_tol[c] * rand(rng)^pw[c] * cis(2π * rand(rng))
                tfac = tilt_tol[c] * rand(rng)^pw[c] * cis(2π * rand(rng))
                dunc = disp_unc[c] * randn(rng) * cis(2π * rand(rng))
                d = δ0[c] + dc_disp[c] * (dfac + dunc) + dc_tilt[c] * tfac
                correct[c] ? (corr += d) : (unc += d)
            end
            other = 5e-5 * cis(2π * rand(rng))
            δ_all[s] = abs(corr + unc + other)
            δ_efc[s] = abs((corr + other) / 2 + unc)
        end
        edges = res.bin_edges
        hist(x) = [count(v -> edges[i] <= v < edges[i+1], x) for i in 1:length(edges)-1] ./ (length(x) * (edges[2] - edges[1]))
        h_all = hist(δ_all)
        h_efc = hist(δ_efc)
        # Histograms agree to Monte Carlo noise: compare CDFs (max deviation) and means.
        cdf(p) = cumsum(p .* (edges[2] - edges[1]))
        @test maximum(abs.(cdf(res.pdf) .- cdf(h_all))) < 0.01
        @test maximum(abs.(cdf(res.pdf_efc) .- cdf(h_efc))) < 0.01
        @test isapprox(res.mean_abs_delta, mean(δ_all); rtol=1e-2)
        @test isapprox(res.mean_abs_delta_efc, mean(δ_efc); rtol=1e-2)
        @test res.delta_nominal ≈ abs(sum(δ0))

        # Same seed, any thread count or batch order: bit-identical.
        again = EF.run_monte_carlo(table, ts, hoops(names), ctrl)
        @test again.pdf == res.pdf && again.pdf_efc == res.pdf_efc
        @test again.pdf_batches == res.pdf_batches
        # A different seed differs; the tolerance scale widens the distribution.
        other_seed = EF.run_monte_carlo(table, ts, hoops(names), EF.MonteCarloControl(; nsample=300_000, nbatch=2, seed=12, nbins=200))
        @test other_seed.pdf != res.pdf
        wide = EF.run_monte_carlo(table, ts, hoops(names), EF.MonteCarloControl(; nsample=100_000, nbatch=1, seed=11, nbins=200, tolerance_scale=3.0))
        @test wide.mean_abs_delta > res.mean_abs_delta
        # Coil subset: only c1 sampled; c2 and c3 contribute their nominal overlap.
        sub = EF.run_monte_carlo(table, ts, hoops(names), EF.MonteCarloControl(; nsample=100_000, nbatch=1, seed=11, nbins=200, coil_subset=["c1"]))
        @test sub.delta_worst < res.delta_worst
        none = EF.run_monte_carlo(table, ts, hoops(names), EF.MonteCarloControl(; nsample=20_000, nbatch=1, seed=11, nbins=200, coil_subset=["zzz"]))
        # With nothing sampled but the ring budget, |δ| = |Σδ0 + 5e-5·e^{iφ}| lies within 5e-5 of the nominal.
        @test none.mean_abs_delta ≈ res.delta_nominal rtol = 0.5
        @test all(abs.(none.bin_edges[findall(>(0), none.pdf)] .- res.delta_nominal) .< 5e-5 + 2 * (none.bin_edges[2] - none.bin_edges[1]))
    end

    @testset "coherent group: shared draw and rigid rotation" begin
        # Two identical coils at heights ±h in one group: a pure group tilt about z = 0 shifts
        # them oppositely, so with equal S the rotation-induced shifts cancel and only 2·T·θ
        # remains; with the pivot at the lower coil they do not cancel.
        names = ["u", "l"]
        Sx = 1.0 + 0.0im
        Tx = 0.0 + 0.0im
        S = [Sx -im*Sx 0.0; Sx -im*Sx 0.0]'
        T = zeros(3, 2)
        table = synthetic_table(names, [0.0, 0.0], S, T)
        sets = hoops(names; heights=[0.5, -0.5])
        centered = EF.parse_tolerance_toml("""
            [[ErrorFields.coherent_group]]
            name = "pair"
            members = ["u", "l"]
            shift_tol_mm = 0.0
            tilt_tol = 1.0
            radial_shape = "ring"
            rotation_center_z_m = 0.0
            """)
        res0 = EF.run_monte_carlo(table, centered, sets, EF.MonteCarloControl(; nsample=20_000, nbatch=1, seed=2, nbins=50, delta_max=1e-2))
        @test res0.mean_abs_delta < 1e-12
        offset = EF.parse_tolerance_toml("""
            [[ErrorFields.coherent_group]]
            name = "pair"
            members = ["u", "l"]
            shift_tol_mm = 0.0
            tilt_tol = 1.0
            radial_shape = "ring"
            rotation_center_z_m = -0.5
            """)
        # Rotation about the lower coil by 1° moves the upper coil by 1.0 m·deg2rad(1): |δ| = |S|·1.0·deg2rad(1).
        res1 = EF.run_monte_carlo(table, offset, sets, EF.MonteCarloControl(; nsample=20_000, nbatch=1, seed=2, nbins=50, delta_max=0.05))
        @test res1.mean_abs_delta ≈ deg2rad(1.0) rtol = 1e-2
        # Two groups sharing a phase_group move in the same direction: their shifts add coherently.
        shared = EF.parse_tolerance_toml("""
            [[ErrorFields.coherent_group]]
            name = "gu"
            members = ["u"]
            shift_tol_mm = 1.0
            radial_shape = "ring"
            phase_group = "same"
            [[ErrorFields.coherent_group]]
            name = "gl"
            members = ["l"]
            shift_tol_mm = 1.0
            radial_shape = "ring"
            phase_group = "same"
            """)
        res_s = EF.run_monte_carlo(table, shared, sets, EF.MonteCarloControl(; nsample=20_000, nbatch=1, seed=2, nbins=50, delta_max=5e-3))
        @test res_s.mean_abs_delta ≈ 2e-3 rtol = 1e-2
        independent = EF.parse_tolerance_toml(replace(shared.raw, "phase_group = \"same\"" => ""))
        res_i = EF.run_monte_carlo(table, independent, sets, EF.MonteCarloControl(; nsample=50_000, nbatch=1, seed=2, nbins=50, delta_max=5e-3))
        # |e^{iφ1} + e^{iφ2}| averages 4/π for independent phases.
        @test res_i.mean_abs_delta ≈ 1e-3 * 4 / π rtol = 2e-2
    end

    @testset "guards" begin
        table = synthetic_table(["a"], [1e-4], reshape([1.0, -1.0im, 0.0], 3, 1), zeros(3, 1))
        ts = EF.parse_tolerance_toml("[[ErrorFields.coil]]\nname = \"b\"\nshift_tol_mm = 1.0\n")
        @test_throws ArgumentError EF.run_monte_carlo(table, ts, hoops(["a"]))   # unknown coil name
        ok = EF.parse_tolerance_toml("[[ErrorFields.coil]]\nname = \"a\"\nshift_tol_mm = 1.0\n")
        @test_throws ArgumentError EF.run_monte_carlo(table, ok, hoops(["zzz"]))  # no geometry for the coil
        @test_throws ArgumentError EF.run_monte_carlo(table, ok, hoops(["a"]), EF.MonteCarloControl(; nsample=0))
        empty = EF.parse_tolerance_toml("[[ErrorFields.coil]]\nname = \"a\"\nshift_tol_mm = 0.0\n")
        zero_table = synthetic_table(["a"], [0.0], reshape([1.0, -1.0im, 0.0], 3, 1), zeros(3, 1))
        @test_throws ArgumentError EF.run_monte_carlo(zero_table, empty, hoops(["a"]))  # nothing to sample
    end
end
