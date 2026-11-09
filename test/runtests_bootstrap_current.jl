using Test
using TOML
using GeneralizedPerturbedEquilibrium

@testset "Bootstrap current" begin
    GPEC = GeneralizedPerturbedEquilibrium
    BC = GPEC.Utilities.BootstrapCurrent
    Eq = GPEC.Equilibrium
    EC = 1.602176634e-19

    # (f_t, Z_eff, nu_e*, nu_i*, L31, L32, alpha, jdotB) from OpenFUSIONToolkit TokaMaker `redl_bootstrap` (commit 637ddb5), jboot1, use_legacy_L34=False
    redl_oracle = [
        (0.1, 1.0, 0.01, 0.01, 0.13545033708131518, -0.08305470624423099, -1.0483168453295013, 53865.33136495606),
        (0.25, 1.5, 0.1, 0.05, 0.2359363283734479, -0.0715742719918062, -0.7167460948754181, 130260.29898267234),
        (0.4, 2.0, 1.0, 0.5, 0.24743537097926352, 0.06276713461942436, -0.04000751746519661, 209222.55685617123),
        (0.55, 1.2, 3.0, 1.0, 0.184850306311777, 0.14119356432215222, -0.03275606630693376, 183761.98078772676),
        (0.7, 2.5, 10.0, 5.0, 0.10564861564801808, 0.151141790537118, 2.0418011480760123, 168358.79959828986),
        (0.35, 3.0, 0.5, 10.0, 0.27010731762521395, 0.020969260841897194, 2.1182499511424036, 329096.2052590009)
    ]
    # (n_e, n_i, T_e, T_i, lnLambda_e, lnLambda_ii) from TokaMaker `calculate_ln_lambda` (NRL electron, Zavg ion)
    lnl_oracle = [
        (4e+19, 3.5e+19, 1500.0, 1200.0, 15.653266337878941, 17.73358120809945),
        (1e+20, 9e+19, 5000.0, 4000.0, 16.399094471655047, 19.151822241068018),
        (2e+19, 2e+19, 300.0, 300.0, 14.390400380045524, 16.334541738260896),
        (5e+18, 4e+18, 20000.0, 18000.0, 19.283255507827988, 22.611346883868467)
    ]

    @testset "Redl coefficients and assembly match TokaMaker" begin
        for (ft, Z, nue, nui, L31, L32, alpha, jB) in redl_oracle
            c = BC.redl_coefficients(ft, Z, nue, nui)
            @test c.L31 ≈ L31 rtol = 1e-12
            @test c.L32 ≈ L32 rtol = 1e-12
            @test c.L34 ≈ L31 rtol = 1e-12
            @test c.alpha ≈ alpha rtol = 1e-12
            Te, Ti, ne, ni = 1500.0, 1200.0, 4.0e19, 3.5e19
            r = BC.jdotB_bootstrap(; T_e=Te, T_i=Ti, n_e=ne, n_i=ni, Z_eff=Z, f_t=ft, nu_e_star=nue, nu_i_star=nui, I_psi=5.0,
                dT_e=-9000.0, dT_i=-7000.0, dn_e=-1.5e20, dn_i=-1.2e20)
            @test r.jdotB ≈ jB rtol = 1e-12
        end
    end

    @testset "Coulomb logarithms and collisionality" begin
        for (ne, ni, Te, Ti, le, li) in lnl_oracle
            @test max(BC.coulomb_log_e(ne, Te), 10.0) ≈ le rtol = 1e-12
            @test BC.coulomb_log_i(ne, ni, Ti) ≈ li rtol = 1e-12
        end
        # Floors at 10
        @test BC.coulomb_log_i(1e26, 1e26, 1.0) == 10.0
        # Koh form with Z_ion⁴ = Z_dom² (n_e/n_i) Z_eff, written out
        ne, ni, Te, Ti, R, eps, q, Z = 4e19, 3.5e19, 1500.0, 1200.0, 1.7, 0.3, 2.0, 1.8
        lnii = 30 - log((ne / ni)^3 * sqrt(ni) / Ti^1.5)
        @test BC.nu_star_i_koh(ne, ni, Ti, R, eps, q, Z) ≈ 4.90e-18 * q * R * ni * (ne / ni) * Z * lnii / (Ti^2 * eps^1.5) rtol = 1e-12
        @test BC.nu_star_i_koh(ne, ni, Ti, R, eps, -q, Z; Z_dom=2) ≈ 4 * BC.nu_star_i_koh(ne, ni, Ti, R, eps, q, Z) rtol = 1e-12
        lne = max(23.5 - log(sqrt(ne / 1e6) * Te^-1.25) - sqrt(1e-5 + (log(Te) - 2)^2 / 16), 10.0)
        @test BC.nu_star_e_floored(ne, Te, R, eps, q, Z) ≈ 6.921e-18 * q * R * ne * Z * lne / (Te^2 * eps^1.5) rtol = 1e-12
    end

    @testset "Circulating fraction" begin
        # Model surface B = B0/(1 + ε cos θ) with uniform weights; the reference evaluates the definition term by term with plain loops
        function reference_fc(eps, N)
            thetas = [2π * (k - 1) / N for k in 1:N]
            Bs = [1 / (1 + eps * cos(t)) for t in thetas]
            Bmax = maximum(Bs)
            h = sum(Bs) / N / Bmax
            h2 = sum(Bs .^ 2) / N / Bmax^2
            ftu = 1 - h2 / h^2 * (1 - sqrt(1 - h) * (1 + h / 2))
            hf = sum((1 - sqrt(1 - Bi / Bmax) * (1 + Bi / Bmax / 2)) / (Bi / Bmax)^2 for Bi in Bs) / N
            return 1 - (0.75ftu + 0.25 * (1 - h2 * hf))
        end
        n = 512
        th = range(0, 2π; length=n + 1)[1:n]
        for eps in (0.05, 0.2, 0.5)
            B = 1 ./ (1 .+ eps .* cos.(th))
            @test BC.circulating_fraction(B, ones(n)) ≈ reference_fc(eps, n) rtol = 1e-12
            # Grid convergence against a 100x finer grid; the √(1−b) cusp at B_max limits the periodic trapezoid to O(h²)
            @test BC.circulating_fraction(B, ones(n)) ≈ reference_fc(eps, 100n) rtol = 1e-5
        end
        # Small-ε limit f_t → 1.46 √ε
        eps = 1e-4
        B = 1 ./ (1 .+ eps .* cos.(th))
        @test 1 - BC.circulating_fraction(B, ones(n)) ≈ 1.46 * sqrt(eps) rtol = 0.1
        # The 3/4-1/4 Sauter form approximates the exact f_c = 3/4 ⟨B²⟩ ∫_0^{1/Bmax} λ dλ / ⟨√(1 − λB)⟩ to about a percent
        for eps in (0.1, 0.3)
            B = 1 ./ (1 .+ eps .* cos.(th))
            lam = ((1:20000) .- 0.5) ./ 20000 ./ maximum(B)
            integral = sum(l / (sum(sqrt.(1 .- l .* B)) / n) for l in lam) * step(lam)
            fc_exact = 0.75 * sum(B .^ 2) / n * integral
            @test BC.circulating_fraction(B, ones(n)) ≈ fc_exact rtol = 0.02
        end
    end

    @testset "End to end on a DIII-D-like equilibrium with kinetic profiles" begin
        dir = joinpath(@__DIR__, "..", "examples", "DIIID-like_SLAYER_example")
        inputs = TOML.parsefile(joinpath(dir, "gpec.toml"))
        equil = Eq.PlasmaEquilibrium(GPEC.equilibrium_config(inputs, dir))
        x = collect(range(0.05, 0.95; length=91))
        r = BC.bootstrap_current(equil; psi_N=x)
        @test all(isfinite, r.jdotB)
        @test all(0 .< r.f_t .< 1)
        @test all(r.eps .> 0)
        # Equilibrium ⟨j·B⟩ = −(F p′ + F′⟨B²⟩/μ0) and toroidal ⟨j_φ/R⟩ = −(p′ + F F′⟨R⁻²⟩/μ0) in the same ψ convention (profiles hold μ0 dp/dψ_N and 2πF)
        prof, psio = equil.profiles, equil.psio
        jdotB_eq = [-(r.F[i] * prof.P_deriv(xi) / psio + prof.F_deriv(xi) / (2π) / psio * r.avg_B2[i]) for (i, xi) in enumerate(x)] ./ (4π * 1e-7)
        @test sign(sum(r.jdotB)) == sign(sum(jdotB_eq))
        # Bootstrap fraction of the toroidal current, I = ∫⟨j_φ/R⟩ dV/2π
        jphiR_bs = r.jdotB .* r.F .* r.avg_invR2 ./ r.avg_B2
        jphiR_tot = [-(prof.P_deriv(xi) + r.F[i] * prof.F_deriv(xi) / (2π) * r.avg_invR2[i]) / psio for (i, xi) in enumerate(x)] ./ (4π * 1e-7)
        vp = [prof.dVdpsi_spline(xi) for xi in x]
        trapz(f) = sum((f[1:(end-1)] .+ f[2:end]) ./ 2 .* diff(x))
        frac = trapz(jphiR_bs .* vp) / trapz(jphiR_tot .* vp)
        @info "Bootstrap end-to-end" fraction = frac jdotB_at = Dict(xi => r.jdotB[findmin(abs.(x .- xi))[2]] for xi in (0.2, 0.5, 0.8, 0.9)) f_t_mid = r.f_t[46] nu_e_mid =
            r.nu_e_star[46]
        @test 0.02 < frac < 0.8
    end
end
