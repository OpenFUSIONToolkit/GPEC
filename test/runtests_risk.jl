using HDF5
using Random
using SpecialFunctions
using Statistics

# The locking-risk model: the ITPA threshold table and its nominal value, the sampled threshold
# distribution, the convolution with a Monte Carlo overlap distribution in closed-form limits
# (a sharp threshold, a distribution entirely below or above it), the tolerance scan, and the
# log-space inversion for an allowable tolerance.
@testset "locking risk" begin
    GPEC = GeneralizedPerturbedEquilibrium
    EF = GPEC.ErrorFields
    FT = GPEC.ForcingTerms

    scen = EF.ScenarioParameters(; n_e=2.0, b_t0=2.0, r_0=1.7, beta_n=1.8, l_i=1.0)   # 1e19 m^-3, T, m, -, -

    @testset "scaling table and nominal threshold" begin
        sc = EF.threshold_scaling(; n=1, dataset="O,L", fit="WLS")
        @test sc.alpha_c == (-3.46, 0.05) && sc.alpha_beta == (0.15, 0.07)
        @test EF.fitted_threshold(sc, scen) ≈ 10.0^-3.46 * 2.0^0.64 * 2.0^-1.14 * 1.7^0.20 * 1.8^0.15
        @test_throws ArgumentError EF.threshold_scaling(; n=3)
        @test_throws ArgumentError EF.threshold_scaling(; n=1, fit="XYZ")
        @test length(EF.ITPA_THRESHOLD_SCALINGS) == 11
        # n=2 "O,L,N" is the n=1 O,L WLS fit doubled: 10^(-3.16) ≈ 2·10^(-3.46).
        @test 10^EF.threshold_scaling(; n=2, dataset="O,L,N").alpha_c[1] ≈ 2 * 10^EF.threshold_scaling(; n=1, dataset="O,L").alpha_c[1] rtol = 5e-3
        @test_throws ArgumentError EF.ScenarioParameters(; n_e=0.0, b_t0=2.0, r_0=1.7, beta_n=1.8, l_i=1.0)
        # Five positive scalars of similar size: a positional form would take them in any order.
        @test_throws MethodError EF.ScenarioParameters(2.0, 2.0, 1.7, 1.8, 1.0)
        # Sampled thresholds: median near the nominal, log-normal-ish spread from the exponent errors.
        t = EF.threshold_samples(Xoshiro(1), sc, scen; nsample=100_000)
        @test abs(log10(median(t)) - log10(EF.fitted_threshold(sc, scen))) < 0.01
        @test all(>(0), t)
        flat = EF.threshold_samples(Xoshiro(1), sc, scen; nsample=50_000, dist="flat")
        @test maximum(abs.(log10.(flat) .- log10(EF.fitted_threshold(sc, scen)))) < 0.05 + 0.09 * log10(2) + 0.12 * log10(2) + 0.08 * log10(1.7) + 0.07 * log10(1.8) + 1e-9
        trunc = EF.threshold_samples(Xoshiro(1), sc, scen; nsample=50_000, dist="normal_truncated")
        @test std(log10.(trunc)) < std(log10.(t))
        @test_throws ArgumentError EF.threshold_samples(Xoshiro(1), sc, scen; nsample=10, dist="cauchy")
    end

    @testset "2026 n=1 fits with a plasma-current term" begin
        # Bursch et al., PPCF 2026 (doi:10.1088/1361-6587/aea7d6), Eqs. 7 (OLS) and 8 (WLS), exponent by exponent.
        ols = EF.threshold_scaling(; n=1, year=2026, dataset="O,L", fit="OLS")
        wls = EF.threshold_scaling(; n=1, year=2026, dataset="O,L", fit="WLS")
        @test (ols.alpha_c, ols.alpha_n, ols.alpha_b, ols.alpha_r, ols.alpha_beta, ols.alpha_ip) ==
              ((-4.31, 0.09), (0.77, 0.08), (0.19, 0.09), (1.88, 0.16), (0.25, 0.08), (-0.97, 0.08))
        @test (wls.alpha_c, wls.alpha_n, wls.alpha_b, wls.alpha_r, wls.alpha_beta, wls.alpha_ip) ==
              ((-4.26, 0.09), (0.56, 0.08), (0.30, 0.10), (1.57, 0.15), (0.13, 0.06), (-1.01, 0.07))
        @test_throws ArgumentError EF.threshold_scaling(; n=1, year=2026, dataset="O,L", fit="DSOLS")
        # The 2020 fits carry no current term.
        @test all(sc -> sc.year == 2026 || sc.alpha_ip == (0.0, 0.0), values(EF.ITPA_THRESHOLD_SCALINGS))
        # Every key names its year, and the default lookup stays the 2020 n=1 O,L WLS fit.
        @test all(((k, sc),) -> k == EF.scaling_label(sc), EF.ITPA_THRESHOLD_SCALINGS)
        @test EF.scaling_label(EF.threshold_scaling()) == "n=1 2020 O,L WLS"
        @test EF.scaling_label(wls) == "n=1 2026 O,L WLS"
        @test_throws ArgumentError EF.threshold_scaling(; n=1, year=2026, dataset="O,L,H")

        scen_ip = EF.ScenarioParameters(; n_e=2.0, b_t0=2.0, r_0=1.7, beta_n=1.8, l_i=1.0, i_p=1.2)   # I_p in MA
        @test EF.fitted_threshold(ols, scen_ip) ≈ 10.0^-4.31 * 2.0^0.77 * 2.0^0.19 * 1.7^1.88 * 1.8^0.25 * 1.2^-0.97
        @test EF.fitted_threshold(wls, scen_ip) ≈ 10.0^-4.26 * 2.0^0.56 * 2.0^0.30 * 1.7^1.57 * 1.8^0.13 * 1.2^-1.01
        # Without a current the 2026 fits refuse to evaluate; the 2020 fits never need one.
        @test isnan(scen.i_p)
        @test_throws ArgumentError EF.fitted_threshold(wls, scen)
        @test_throws ArgumentError EF.threshold_samples(Xoshiro(1), ols, scen; nsample=10)
        @test_throws ArgumentError EF.ScenarioParameters(; n_e=2.0, b_t0=2.0, r_0=1.7, beta_n=1.8, l_i=1.0, i_p=-1.2)
        # A fit without a current term draws no current exponent, so its sampled stream does not
        # depend on whether the scenario carries a current.
        sc20 = EF.threshold_scaling(; n=1, year=2020, dataset="O,L", fit="WLS")
        @test EF.threshold_samples(Xoshiro(3), sc20, scen; nsample=1000) == EF.threshold_samples(Xoshiro(3), sc20, scen_ip; nsample=1000)
        @test EF.fitted_threshold(sc20, scen) == EF.fitted_threshold(sc20, scen_ip)
        # With the current term the sampled median still sits at the nominal threshold.
        t = EF.threshold_samples(Xoshiro(1), wls, scen_ip; nsample=100_000)
        @test abs(log10(median(t)) - log10(EF.fitted_threshold(wls, scen_ip))) < 0.01
    end

    # A Monte Carlo result with a known |δ| density: uniform on [a, b].
    function uniform_mc(a, b; nbins=200, nbatch=2, edge_max=2b)
        edges = collect(range(0.0, edge_max; length=nbins + 1))
        centers = (edges[1:end-1] .+ edges[2:end]) ./ 2
        pdf = [a <= c <= b ? 1 / (b - a) : 0.0 for c in centers]
        pdf ./= sum(pdf .* diff(edges))
        batches = repeat(pdf, 1, nbatch)
        EF.MonteCarloResult(edges, pdf, pdf ./ 2, batches, batches ./ 2, (a + b) / 2, b, (a + b) / 2, (a + b) / 4, 0.0, 1000, nbatch, 1)
    end
    sc = EF.threshold_scaling(; n=1)

    @testset "risk in closed-form limits" begin
        mc = uniform_mc(1e-4, 3e-4)
        # A sharp threshold at δ_t: P_lock = fraction of the distribution above δ_t.
        δt = mc.abs_delta_total_as_designed                                # = 2e-4, the midpoint
        risk = EF.locking_risk(mc, fill(δt, 1000), sc, scen)
        @test risk.locking_probability_percent ≈ 50.0 atol = 1.0
        @test risk.locking_probability_as_designed_percent == 100.0                    # δ_nominal = δ_t counts as locked
        @test risk.locking_probability_batches_percent ≈ fill(risk.locking_probability_percent, 2)
        @test risk.locking_probability_given_delta[1] == 0.0 && risk.locking_probability_given_delta[end] == 1.0
        @test sum(risk.threshold_pdf .* diff(risk.abs_delta_bin_edges)) ≈ 1.0
        # Thresholds entirely above the distribution: no risk; entirely below: certain.
        @test EF.locking_risk(mc, fill(1e-3, 100), sc, scen).locking_probability_percent == 0.0
        @test EF.locking_risk(mc, fill(1e-6, 100), sc, scen).locking_probability_percent ≈ 100.0 atol = 1e-9
        # Uniform thresholds on [1e-4, 3e-4] against a uniform |δ| on the same interval: P = 1/2.
        thr = collect(range(1e-4, 3e-4; length=20_001))
        @test EF.locking_risk(mc, thr, sc, scen).locking_probability_percent ≈ 50.0 atol = 1.0
        # Sampled ITPA thresholds through the RiskControl path are reproducible and bounded.
        r1 = EF.locking_risk(mc, sc, scen; ctrl=EF.RiskControl(; nsample_threshold=50_000, seed=4))
        r2 = EF.locking_risk(mc, sc, scen; ctrl=EF.RiskControl(; nsample_threshold=50_000, seed=4))
        @test r1.locking_probability_percent == r2.locking_probability_percent && 0 <= r1.locking_probability_percent <= 100 &&
              r1.locking_probability_efc_percent <= r1.locking_probability_percent
        @test r1.threshold_fit == EF.fitted_threshold(sc, scen)
        @test 0 <= r1.locking_probability_fit_threshold_percent <= 100
    end

    @testset "tolerance scan and allowable tolerance" begin
        # One coil, S real, δ_nominal = 0, Flat 1 mm tolerance: |δ| uniform on [0, |S|·scale·1e-3].
        # With a sharp threshold the risk is analytic: P = 1 − δ_t / (|S|·scale·1e-3) once the edge passes δ_t.
        S = 0.2
        # Columns are abs_delta_shift_per_mm, abs_delta_tilt_per_deg, abs_delta_rim_per_mm; this coil has no
        # tilt sensitivity, so the last two are zero.
        table = EF.SensitivityTable(["a"], 1, [0.0im], ComplexF64[S; -im*S; 0.0;;], zeros(ComplexF64, 3, 1),
            [S], [0.0], [0.0], zeros(2, 1), zeros(2, 1))
        ts = EF.parse_tolerance_toml("[[ErrorFields.coil]]\nname = \"a\"\nshift_tol_mm = 1.0\nradial_shape = \"flat\"\n")
        sets = [FT.make_pf_hoop(; radius=1.5, height=0.0, name="a")]
        mc_ctrl = EF.MonteCarloControl(; nsample=100_000, nbatch=2, seed=7, nbins=200)
        scales = [0.5, 0.6, 0.7, 0.8, 1.0, 2.0, 4.0]
        δt = 1e-4   # = |S| · 0.5 mm: the scale-0.5 edge
        # A degenerate scaling whose exponents have no spread gives a sharp threshold; pick one so
        # that fitted_threshold == δt by construction.
        sharp = EF.ThresholdScaling(1, 0, "test", "sharp", (log10(δt), 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0))
        scan = EF.tolerance_scan(table, ts, sets, mc_ctrl, sharp, scen; scales, risk_ctrl=EF.RiskControl(; nsample_threshold=1000))
        expected = [max(0.0, 1 - δt / (S * s * 1e-3)) * 100 for s in scales]
        @test scan.tolerance_scale == scales
        @test all(abs.(scan.locking_probability_percent .- expected) .< 1.5)
        @test all(scan.locking_probability_efc_percent .<= scan.locking_probability_percent .+ 1e-9)
        @test all(scan.locking_probability_spread_percent .>= 0)
        @test scan.locking_probability_as_designed_percent == 0.0
        # Inversion: the scale at which the risk reaches 25 %, from the analytic curve, is 0.5/(1−0.25) = 2/3,
        # bracketed by the 0.6 (16.7 %) and 0.7 (28.6 %) points.
        s25 = EF.allowable_tolerance(scan, 25.0)
        @test isapprox(s25, 2 / 3; rtol=0.05)
        @test isnan(EF.allowable_tolerance(scan, 99.0))     # never reached within the scan
        @test_throws ArgumentError EF.allowable_tolerance(scan, 0.0)
        @test_throws ArgumentError EF.tolerance_scan(table, ts, sets, mc_ctrl, sharp, scen; scales=Float64[])
    end

    @testset "far-tail risk against the closed form, binning and sample convergence" begin
        # One coil with no tolerance disk and a Gaussian placement uncertainty σ: δ = S·conj(u) with
        # u = σ·r·e^{iφ} and r ~ N(0, 1), so |δ| = |S|·σ·|r| is half-normal with scale a = |S|·σ and
        # the locking probability against a sharp threshold t is erfc(t / (a√2)) exactly; against a
        # Gaussian threshold it is that integrated over the threshold density. The regime is the
        # far tail, where the risk requirement lives and where a binning bias could hide behind a
        # small batch spread. Tolerances are in units of the batch standard error plus the bin
        # discretization of P(lock|δ), never magic numbers.
        S = 0.2
        σ = 1e-3
        a = S * σ
        table = EF.SensitivityTable(["a"], 1, [0.0im], ComplexF64[S; -im*S; 0.0;;], zeros(ComplexF64, 3, 1), [S], [0.0], [0.0], zeros(2, 1), zeros(2, 1))
        ts = EF.parse_tolerance_toml("[[ErrorFields.coil]]\nname = \"a\"\nshift_tol_mm = 0.0\nshift_sigma_mm = $(1e3 * σ)\n")
        sets = [FT.make_pf_hoop(; radius=1.5, height=0.0, name="a")]
        ctrl = EF.MonteCarloControl(; nsample=400_000, nbatch=10, seed=21, nbins=300)
        mc = EF.run_monte_carlo(table, ts, sets, ctrl)
        @test mc.abs_delta_sampled_mean ≈ a * sqrt(2 / π) rtol = 2e-2
        exact_sharp(t) = 100 * erfc(t / (a * sqrt(2)))
        bin = mc.abs_delta_bin_edges[2] - mc.abs_delta_bin_edges[1]
        for k in (3.0, 4.0)
            t = k * a
            risk = EF.locking_risk(mc, fill(t, 100), sc, scen)
            se = std(risk.locking_probability_batches_percent) / sqrt(ctrl.nbatch)
            # P(lock|δ) is a step on the bin edges, so the sharp threshold is resolved to one bin.
            @test abs(risk.locking_probability_percent - exact_sharp(t)) < 4 * se + abs(exact_sharp(t + bin) - exact_sharp(t - bin))
        end
        # Gaussian threshold T ~ N(4a, a/2): the expected risk is ∫ pdf_T(t) erfc(t/(a√2)) dt.
        μ, w = 4a, a / 2
        thr = μ .+ w .* randn(Xoshiro(5), 2_000_000)
        tgrid = range(μ - 8w, μ + 8w; length=20_001)
        pdf_t = exp.(-((tgrid .- μ) ./ w) .^ 2 ./ 2) ./ (w * sqrt(2π))
        expected = sum(pdf_t .* exact_sharp.(max.(tgrid, 0.0))) * step(tgrid)
        risk_g = EF.locking_risk(mc, thr, sc, scen)
        se_g = std(risk_g.locking_probability_batches_percent) / sqrt(ctrl.nbatch)
        @test abs(risk_g.locking_probability_percent - expected) < 4 * se_g + 0.05 * expected
        # Binning and sampling: 300 against 3000 bins agree within the batch spread, the value stays
        # at the closed form at every sample count, and the spread falls as 1/√N.
        sharp = EF.ThresholdScaling(1, 0, "test", "sharp", (log10(3a), 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0))
        conv = EF.risk_convergence(table, ts, sets, sharp, scen; nsamples=[25_000, 100_000, 400_000], nbins_list=[300, 3000], ctrl=ctrl,
            risk_ctrl=EF.RiskControl(; nsample_threshold=1000))
        @test conv.nsample == [25_000, 100_000, 400_000] && conv.nbins == [300, 3000]
        p300, p3000 = conv.locking_probability_percent_by_nbins
        @test abs(p300 - p3000) < maximum(conv.locking_probability_spread_percent_by_nbins) + abs(exact_sharp(3a + bin) - exact_sharp(3a - bin))
        @test all(
            abs.(conv.locking_probability_percent_by_nsample .- exact_sharp(3a)) .<
            conv.locking_probability_spread_percent_by_nsample .+ abs(exact_sharp(3a + bin) - exact_sharp(3a - bin))
        )
        s = conv.locking_probability_spread_percent_by_nsample
        @test 2.0 < s[1] / s[3] < 8.0            # sixteen times the samples: the spread falls by about four
    end

    @testset "a threshold scaling describes one toroidal mode number" begin
        @test EF.single_toroidal_mode(2, 2) == 2
        @test_throws ArgumentError EF.single_toroidal_mode(1, 3; where="a multi-n run")
        # The ITPA fits are per n, but a dominant mode spans every n the run carried. Silently
        # taking the lowest would apply an n = 1 threshold to a partly n = 2 mode.
        mktempdir() do dir
            single = joinpath(dir, "single.h5")
            HDF5.h5open(single, "w") do f
                f["Info/nlow"] = 1
                f["Info/nhigh"] = 1
            end
            @test EF.scaling_toroidal_mode(single) == 1

            multi = joinpath(dir, "multi.h5")
            HDF5.h5open(multi, "w") do f
                f["Info/nlow"] = 1
                f["Info/nhigh"] = 2
            end
            @test_throws ArgumentError EF.scaling_toroidal_mode(multi)
        end
    end
end
