using Random
using Statistics

# The locking-risk model: the ITPA threshold table and its nominal value, the sampled threshold
# distribution, the convolution with a Monte Carlo overlap distribution in closed-form limits
# (a sharp threshold, a distribution entirely below or above it), the tolerance scan, and the
# log-space inversion for an allowable tolerance.
@testset "locking risk" begin
    GPEC = GeneralizedPerturbedEquilibrium
    EF = GPEC.ErrorFields
    FT = GPEC.ForcingTerms

    scen = EF.ScenarioParameters(2.0, 2.0, 1.7, 1.8, 1.0)   # n_e [1e19], B_T [T], R_0 [m], β_N, l_i

    @testset "scaling table and nominal threshold" begin
        sc = EF.threshold_scaling(; n=1, dataset="O,L", fit="WLS")
        @test sc.alpha_c == (-3.46, 0.05) && sc.alpha_beta == (0.15, 0.07)
        @test EF.nominal_threshold(sc, scen) ≈ 10.0^-3.46 * 2.0^0.64 * 2.0^-1.14 * 1.7^0.20 * 1.8^0.15
        @test_throws ArgumentError EF.threshold_scaling(; n=3)
        @test_throws ArgumentError EF.threshold_scaling(; n=1, fit="XYZ")
        @test length(EF.ITPA_THRESHOLD_SCALINGS) == 9
        # n=2 "O,L,N" is the n=1 O,L WLS fit doubled: 10^(-3.16) ≈ 2·10^(-3.46).
        @test 10^EF.threshold_scaling(; n=2, dataset="O,L,N").alpha_c[1] ≈ 2 * 10^EF.threshold_scaling(; n=1, dataset="O,L").alpha_c[1] rtol = 5e-3
        @test_throws ArgumentError EF.ScenarioParameters(0.0, 2.0, 1.7, 1.8, 1.0)
        # Sampled thresholds: median near the nominal, log-normal-ish spread from the exponent errors.
        t = EF.threshold_samples(Xoshiro(1), sc, scen; nsample=100_000)
        @test abs(log10(median(t)) - log10(EF.nominal_threshold(sc, scen))) < 0.01
        @test all(>(0), t)
        flat = EF.threshold_samples(Xoshiro(1), sc, scen; nsample=50_000, dist="flat")
        @test maximum(abs.(log10.(flat) .- log10(EF.nominal_threshold(sc, scen)))) < 0.05 + 0.09 * log10(2) + 0.12 * log10(2) + 0.08 * log10(1.7) + 0.07 * log10(1.8) + 1e-9
        trunc = EF.threshold_samples(Xoshiro(1), sc, scen; nsample=50_000, dist="normal_truncated")
        @test std(log10.(trunc)) < std(log10.(t))
        @test_throws ArgumentError EF.threshold_samples(Xoshiro(1), sc, scen; nsample=10, dist="cauchy")
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
        δt = mc.delta_nominal                                # = 2e-4, the midpoint
        risk = EF.locking_risk(mc, fill(δt, 1000), sc, scen)
        @test risk.plock ≈ 50.0 atol = 1.0
        @test risk.plock_nominal == 100.0                    # δ_nominal = δ_t counts as locked
        @test risk.plock_batches ≈ fill(risk.plock, 2)
        @test risk.p_lock_given_delta[1] == 0.0 && risk.p_lock_given_delta[end] == 1.0
        @test sum(risk.threshold_pdf .* diff(risk.bin_edges)) ≈ 1.0
        # Thresholds entirely above the distribution: no risk; entirely below: certain.
        @test EF.locking_risk(mc, fill(1e-3, 100), sc, scen).plock == 0.0
        @test EF.locking_risk(mc, fill(1e-6, 100), sc, scen).plock ≈ 100.0 atol = 1e-9
        # Uniform thresholds on [1e-4, 3e-4] against a uniform |δ| on the same interval: P = 1/2.
        thr = collect(range(1e-4, 3e-4; length=20_001))
        @test EF.locking_risk(mc, thr, sc, scen).plock ≈ 50.0 atol = 1.0
        # Sampled ITPA thresholds through the RiskControl path are reproducible and bounded.
        r1 = EF.locking_risk(mc, sc, scen; ctrl=EF.RiskControl(; nsample_threshold=50_000, seed=4))
        r2 = EF.locking_risk(mc, sc, scen; ctrl=EF.RiskControl(; nsample_threshold=50_000, seed=4))
        @test r1.plock == r2.plock && 0 <= r1.plock <= 100 && r1.plock_efc <= r1.plock
        @test r1.threshold_nominal == EF.nominal_threshold(sc, scen)
        @test 0 <= r1.plock_sharp <= 100
    end

    @testset "tolerance scan and allowable tolerance" begin
        # One coil, S real, δ_nominal = 0, Flat 1 mm tolerance: |δ| uniform on [0, |S|·scale·1e-3].
        # With a sharp threshold the risk is analytic: P = 1 − δ_t / (|S|·scale·1e-3) once the edge passes δ_t.
        S = 0.2
        table = EF.SensitivityTable(["a"], 1, [0.0im], ComplexF64[S; -im*S; 0.0;;], zeros(ComplexF64, 3, 1), [S], [0.0], zeros(2, 1), zeros(2, 1))
        ts = EF.parse_tolerance_toml("[[ErrorFields.coil]]\nname = \"a\"\nshift_tol_mm = 1.0\nradial_shape = \"flat\"\n")
        sets = [FT.make_pf_hoop(; radius=1.5, height=0.0, name="a")]
        mc_ctrl = EF.MonteCarloControl(; nsample=100_000, nbatch=2, seed=7, nbins=200)
        scales = [0.5, 0.6, 0.7, 0.8, 1.0, 2.0, 4.0]
        δt = 1e-4   # = |S| · 0.5 mm: the scale-0.5 edge
        # A degenerate scaling whose exponents have no spread gives a sharp threshold; pick one so
        # that nominal_threshold == δt by construction.
        sharp = EF.ThresholdScaling(1, "test", "sharp", (log10(δt), 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0))
        scan = EF.tolerance_scan(table, ts, sets, mc_ctrl, sharp, scen; scales, risk_ctrl=EF.RiskControl(; nsample_threshold=1000))
        expected = [max(0.0, 1 - δt / (S * s * 1e-3)) * 100 for s in scales]
        @test scan.scale == scales
        @test all(abs.(scan.plock .- expected) .< 1.5)
        @test all(scan.plock_efc .<= scan.plock .+ 1e-9)
        @test all(scan.plock_spread .>= 0)
        @test scan.plock_nominal == 0.0
        # Inversion: the scale at which the risk reaches 25 %, from the analytic curve, is 0.5/(1−0.25) = 2/3,
        # bracketed by the 0.6 (16.7 %) and 0.7 (28.6 %) points.
        s25 = EF.allowable_tolerance(scan, 25.0)
        @test isapprox(s25, 2 / 3; rtol=0.05)
        @test isnan(EF.allowable_tolerance(scan, 99.0))     # never reached within the scan
        @test_throws ArgumentError EF.allowable_tolerance(scan, 0.0)
        @test_throws ArgumentError EF.tolerance_scan(table, ts, sets, mc_ctrl, sharp, scen; scales=Float64[])
    end
end
