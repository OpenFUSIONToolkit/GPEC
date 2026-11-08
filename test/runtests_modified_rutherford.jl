using TOML

# The modified Rutherford terms: the limits and signs each term must show
# alone, and the surface geometry against Fortran RDCON (MRE_flag output) on the DIII-D-like example.
@testset "ModifiedRutherford terms" begin
    MR = GeneralizedPerturbedEquilibrium.ModifiedRutherford

    @testset "classical term: Mercier power and its zero-pressure limit" begin
        @test MR.classical_term(-3.0, -0.25, 0.01) ≈ -3.0
        @test MR.classical_term(-3.0, -0.3, 0.01; model=:constant) == -3.0
        # Slope of log Δ′(w) is −2α₋ = 2√(−D_I) − 1.
        D_I = -0.3
        slope = log(MR.classical_term(2.0, D_I, 0.02) / MR.classical_term(2.0, D_I, 0.01)) / log(2)
        @test slope ≈ 2 * sqrt(-D_I) - 1 rtol = 1e-12
    end

    @testset "bootstrap term: positive, peaked, k1·D_nc/w far above w_d" begin
        D_nc, w_d, k1 = 0.4, 1e-3, 1.7
        w = exp.(range(log(1e-5), log(0.2); length=400))
        b = MR.bootstrap_term.(D_nc, w_d, w; k1)
        @test all(>(0), b)
        @test w[argmax(b)] ≈ sqrt(k1 / 0.5) * w_d rtol = 0.02
        @test MR.bootstrap_term(D_nc, w_d, 1e3 * w_d; k1) ≈ k1 * D_nc / (1e3 * w_d) rtol = 1e-5
        @test MR.bootstrap_term(D_nc, w_d, 1e-3 * w_d; k1) ≈ 0.5 * D_nc * 1e-3 * w_d / w_d^2 rtol = 1e-5
    end

    @testset "curvature term: stabilizing for D_R < 0" begin
        @test MR.curvature_term(-0.02, 0.01, -0.27, 1e-3, 0.01) < 0
        @test MR.curvature_term(0.02, 0.01, -0.27, 1e-3, 0.01) > 0
    end

    @testset "polarization term: sign follows ω̂(ω̂ − ω̂_pi), zero at ω̂ = 0" begin
        args = (; c_pol=1.0, eps=0.2, rho_theta_psi=0.012, beta_p=0.5, gradient_ratio=-0.8, omega_hat_pi=1.5, w_ib=1e-3)
        pol(ω̂) = MR.polarization_term(0.02; args..., omega_hat=ω̂)
        @test pol(0.0) == 0
        @test pol(1.5) == 0
        @test pol(0.7) > 0          # ω̂ inside (0, ω̂_pi): destabilizing
        @test pol(2.0) < 0 && pol(-1.0) < 0
        @test MR.reduced_polarization_term(0.01; c_pol=1.0, C_pol=1.9, w_ib=1e-3) ≈ -(1.9e-3)^2 / 1e-6
    end

    @testset "Lütjens curvature: SLAYER's critical offset at small w, 6.35 D_R/W at large w" begin
        D_R, D_c, K = -0.02, 1.5, 0.9
        W_d = MR.layer_transport_width(D_R, D_c)
        @test MR.lutjens_curvature_term(D_R, -0.25, K, W_d, 1e-9) * K ≈ -D_c rtol = 1e-3
        @test MR.lutjens_curvature_term(D_R, -0.25, K, W_d, 1e3 * W_d * K) ≈ 6.35 * D_R / (1e3 * W_d * K) rtol = 1e-6
        # Away from D_I = −¼, classical plus curvature still change sign where SLAYER's gate does, at any small w.
        for D_I in (-0.3, -0.4), w in (1e-6, 1e-3 * W_d * K), f in (0.99, 1.01)
            dp = f * D_c * K^(-2 * sqrt(-D_I))
            @test sign(MR.classical_term(dp, D_I, w) + MR.lutjens_curvature_term(D_R, D_I, K, W_d, w)) == sign(f - 1)
        end
        @test MR.layer_transport_width(D_R, 0.0) == Inf
    end

    @testset "saturation term: Wesson cylinder of Fitzpatrick 2023 Fig. 9.2" begin
        # q0 = 0.8, q(a) = 4.4, r = √ψ (a = 1), J_z = (2/q0)(1 − r²)^ν; reference α_s, β_s from the independent shooting script.
        q0, qa = 0.8, 4.4
        nu = qa / q0 - 1
        q(x) = qa * x / (1 - (1 - x)^(nu + 1))
        dq(x) = (q(x + 1e-7) - q(x - 1e-7)) / 2e-7
        lo, hi = 0.1, 0.9
        for _ in 1:60
            mid = (lo + hi) / 2
            q(mid) < 2 ? (lo = mid) : (hi = mid)
        end
        psi_s = (lo + hi) / 2
        sc = MR.saturation_coefficient(x -> (2 / q0) * (1 - x)^nu, sqrt, x -> 0.5 / sqrt(x), q, dq, psi_s; window=0.005)
        @test sc.converged
        @test sqrt(psi_s) ≈ 0.65942681 rtol = 1e-6
        @test sc.alpha_s ≈ 1.642673 rtol = 1e-4
        @test sc.beta_s ≈ -7.204691 rtol = 1e-3
        # Classical plus saturation vanish at W_pw/r_s = r_sΔ′(0)/C_sat (Eq. 9.21); r_sΔ′(0) = 1.64 from the cylindrical
        # tearing equation, which puts W_pw/a at the 0.27 peak of the r_w = a curve. K = r_s dψ/dr = 2ψ_s.
        E_ss, K = 1.64, 2 * psi_s
        grid = exp.(range(log(1e-4), log(2.0); length=200))
        r = MR.mre_roots(w -> MR.classical_term(E_ss / K, -0.25, w) + MR.saturation_term(sc.C_sat, K, -0.25, w), 1e-4, grid)
        @test r.w_sat / K ≈ E_ss / sc.C_sat rtol = 1e-8
        @test sqrt(psi_s) * r.w_sat / K ≈ 0.27 rtol = 0.02
    end

    @testset "transport width grows as w^(1/4)" begin
        kw = (; c_wd=1.0, chi_perp=1.0, v_te=5e7, R0=1.7, q=2.0, q1=3.0, n=1, m=2, Wc_prefac=3e4)
        @test MR.transport_width(16e-3; kw...) / MR.transport_width(1e-3; kw...) ≈ 2 rtol = 1e-12
    end

    @testset "roots: seed threshold rises through zero, saturation falls through it" begin
        grid = exp.(range(log(1e-5), log(0.2); length=60))
        r = MR.mre_roots(w -> (w - 3e-3) * (0.05 - w) / w^2, 1e-5, grid)
        @test r.w_m ≈ 3e-3 rtol = 1e-10
        @test r.w_sat ≈ 0.05 rtol = 1e-10
        @test !r.fully_stabilized && !r.unstable_any_seed
        @test MR.mre_roots(w -> -1 / w, 1e-5, grid).fully_stabilized
        s = MR.mre_roots(w -> (0.05 - w) / w, 1e-5, grid)
        @test s.unstable_any_seed && isnan(s.w_m)
        @test s.w_sat ≈ 0.05 rtol = 1e-10
    end

    @testset "Callen cubic: both positive roots solve it and merge at the critical vacuum width" begin
        a, c = -40.0, -(1.5e-2)^2
        x_min, x_sat = MR.callen_cubic_roots(a, 2.0, c)
        @test 0 < x_min < x_sat
        @test abs(a * x_min^3 + 2.0 * x_min + c) < 1e-12 && abs(a * x_sat^3 + 2.0 * x_sat + c) < 1e-12
        # The double root sits where 4p³ + 27q² = 0, i.e. b = 3|a| (c/2a)^(2/3).
        b_crit = 3 * abs(a) * (c / (2a))^(2 / 3)
        @test all(isnan, MR.callen_cubic_roots(a, 0.99 * b_crit, c))
        lo, hi = MR.callen_cubic_roots(a, 1.0001 * b_crit, c)
        @test hi / lo < 1.05
        # Callen 2016 Fig. 8 (DIII-D 158115, ρ = 71.88 cm): full widths w_min ≈ 1.28 cm, w_sat ≈ 2.24 cm.
        w_min, w_sat = MR.callen_cubic_roots(-16 / 71.88, 26.4 / 71.88 * 2.45^2, -1.5^2)
        @test w_min ≈ 1.28 rtol = 0.1
        @test w_sat ≈ 2.24 rtol = 0.1
    end

    @testset "surface geometry matches Fortran RDCON on the DIII-D-like example" begin
        # Fortran values at q = 2 (ψ_N = 0.5183), interpolated from rdcon_output_n1.nc with MRE_flag = t and Z_eff = 1.52.
        dir = joinpath(@__DIR__, "..", "examples", "DIIID-like_gal_resistive_example")
        deck = TOML.parsefile(joinpath(dir, "gpec.toml"))
        equil = GeneralizedPerturbedEquilibrium.Equilibrium.PlasmaEquilibrium(GeneralizedPerturbedEquilibrium.equilibrium_config(deck, dir),
            GeneralizedPerturbedEquilibrium.analytic_config(deck))
        psi = 0.51830565
        g = MR.mre_geometry(equil, psi, equil.profiles.q_deriv(psi))
        @test g.f_trap ≈ 0.61419 rtol = 2e-3
        @test g.overbar_Rsq ≈ 3.0113 rtol = 2e-3
        @test g.avg_bpol ≈ 0.28709 rtol = 2e-3
        @test g.Wc_prefac ≈ 33565 rtol = 2e-3
        D_nc, mu_ratio = MR.bootstrap_coefficient(g, 1.52)
        @test mu_ratio ≈ 0.68255 rtol = 2e-3
        @test D_nc ≈ 0.38065 rtol = 2e-3
    end
end
