using TOML
using LinearAlgebra

# Perpendicular inertia (ELITE's kinetic energy K = ½∫ρ|ξ_⊥|² dV) as a kinetic source of the force-balance
# equations: the A, C, H bands it adds, the B² identity they rely on, the γ²-scaled solves and growth_rate.
@testset "perpendicular inertia" begin
    GPEC = GeneralizedPerturbedEquilibrium
    FFS = GPEC.ForceFreeStates
    template = joinpath(@__DIR__, "test_data", "regression_tj_ideal_example")
    deck = TOML.parsefile(joinpath(template, "gpec.toml"))
    equil = GPEC.Equilibrium.PlasmaEquilibrium(GPEC.equilibrium_config(deck, template), GPEC.analytic_config(deck))
    wall = GPEC.Vacuum.parse_wall(get(deck, "Wall", nothing))
    ffs_kwargs = Dict{Symbol,Any}(p.first => p.second for p in GPEC.force_balance_keywords(deck, template) if !(p.first in (:nn, :wall)))
    ffs_kwargs[:vac_flag] = true
    forward = Forward(; eulerlagrange_tolerance=1e-8)

    mlow, mhigh = -3, 4
    mpert = mhigh - mlow + 1
    metric = FFS.make_metric(equil, mpert)
    intr = FFS.ForceFreeStatesInternal(; mpert, numpert_total=mpert, mlow, mhigh, nlow=1, nhigh=1, npert=1)
    ipsi_mid = cld(metric.mpsi, 2)
    bands(kw, ic, ipsi) = reshape(kw[ipsi, :, ic], mpert, mpert)

    @testset "A, C, H bands" begin
        kw, kt = FFS.inertia_kinetic_matrices(FFS.PerpendicularInertia(; rho=1e-7), equil, intr, metric)
        @test size(kw) == (metric.mpsi, mpert^2, 6)
        @test all(iszero, kt)
        @test all(iszero, kw[:, :, [2, 4, 5]])
        for ic in (1, 3, 6)
            m = bands(kw, ic, ipsi_mid)
            @test norm(m - m') <= 1e-12 * norm(m)
            @test norm(m) > 0
        end
        @test all(>(0), eigvals(Hermitian(bands(kw, 1, ipsi_mid))))
        @test all(>(0), eigvals(Hermitian(bands(kw, 6, ipsi_mid))))

        kw2, _ = FFS.inertia_kinetic_matrices(FFS.PerpendicularInertia(; rho=2e-7), equil, intr, metric)
        @test kw2 ≈ 2kw rtol = 1e-12

        # A ψ-dependent density scales each surface's bands by ρ(ψ)/ρ₀ and so lowers them toward the edge
        kwf, _ = FFS.inertia_kinetic_matrices(FFS.PerpendicularInertia(; rho=psi -> 1e-7 * (1 - psi)), equil, intr, metric)
        for ipsi in (2, ipsi_mid, metric.mpsi - 1), ic in (1, 3, 6)
            @test bands(kwf, ic, ipsi) ≈ (1 - metric.xs[ipsi]) * bands(kw, ic, ipsi) rtol = 1e-12
        end
        diagA = [real(kwf[ipsi, 1, 1]) / real(kw[ipsi, 1, 1]) for ipsi in 1:metric.mpsi]
        @test issorted(diagA; rev=true)
    end

    @testset "B² from the metric matches the equilibrium field" begin
        chi1 = 2π * equil.psio
        bnodes = equil.eqfun_B.nodal_derivs.partials
        @test size(bnodes)[2:3] == (metric.mpsi, metric.mtheta)
        for ipsi in (3, ipsi_mid, metric.mpsi - 2), jtheta in (1, cld(metric.mtheta, 3), cld(metric.mtheta, 2))
            g11, g22, g33, g23, g31, g12, jac = metric.fs[ipsi, jtheta, 1:7]
            q = equil.profiles.q_spline(metric.xs[ipsi])
            @test chi1^2 * (g22 + 2q * g23 + q^2 * g33) / jac ≈ bnodes[1, ipsi, jtheta]^2 rtol = 1e-3
        end
    end

    lambda_min(r) = minimum(real, r.free_boundary.et)
    ideal = solve(equil, forward; nn=1, wall, ffs_kwargs...)
    inertial_kwargs = merge(ffs_kwargs, Dict{Symbol,Any}(:kinetic => FFS.PerpendicularInertia(; rho=1e-7), :singfac_min => 0.0))

    @testset "inertial solves" begin
        @test equil.kinetic === nothing
        weak = solve(equil, forward; nn=1, wall, inertial_kwargs..., kinetic_factor=1e4)
        @test weak.closure === :regular
        @test weak.kinetic.kmsing == 0

        # The inertial layers at the rational surfaces add a term linear in γ, so λ_min approaches the
        # ideal value as √γ²: a decade in γ² shrinks the offset by √10, and the offset at γ² = 1e2 is ~1.5e-4
        # of λ (measured 6.5e-5 absolute on this fixture).
        weaker = solve(equil, forward; nn=1, wall, inertial_kwargs..., kinetic_factor=1e2)
        offset_weak = lambda_min(weak) - lambda_min(ideal)
        offset_weaker = lambda_min(weaker) - lambda_min(ideal)
        @test 0 < offset_weaker < 3e-4 * lambda_min(ideal)
        @test 8 < offset_weak / offset_weaker < 12

        strong = solve(equil, forward; nn=1, wall, inertial_kwargs..., kinetic_factor=1e10)
        @test lambda_min(strong) > lambda_min(weak)

        # The kinetic crossing machinery (singfac_min > 0, det F̄ near-zeros) and the single chunk agree
        crossed = solve(equil, forward; nn=1, wall, merge(inertial_kwargs, Dict{Symbol,Any}(:singfac_min => 1e-4))..., kinetic_factor=1e4)
        @test crossed.kinetic.kmsing > 0
        @test lambda_min(crossed) ≈ lambda_min(weak) rtol = 1e-5
    end

    @testset "growth_rate" begin
        @test_throws ErrorException solve(equil, forward; nn=1, wall, ffs_kwargs..., kinetic_factor=2.0)

        # The fixture is ideal-stable, so growth_rate stops after the marginal solve
        @test lambda_min(ideal) > 0
        prob = ForceBalanceProblem(equil; nn=1, wall, ffs_kwargs...)
        gr = FFS.growth_rate(prob, forward; inertia=FFS.PerpendicularInertia(; rho=1e-7))
        @test gr.gamma == 0
        @test gr.gamma_sq == 0
        @test gr.result === nothing
        @test length(gr.history) == 1
        @test gr.history[1].lambda ≈ lambda_min(ideal) rtol = 1e-12
    end

    @testset "rho and omega_star_pi from kinetic profiles" begin
        ktemplate = joinpath(@__DIR__, "test_data", "regression_tj_kinetic_calculated")
        kdeck = TOML.parsefile(joinpath(ktemplate, "gpec.toml"))
        kequil = GPEC.Equilibrium.PlasmaEquilibrium(GPEC.equilibrium_config(kdeck, ktemplate), GPEC.analytic_config(kdeck))
        @test kequil.kinetic !== nothing
        src = FFS.PerpendicularInertia(kequil)
        # The density the SLAYER layer uses at a rational surface
        @test FFS._rho_at(src, 0.5) ≈ only(GPEC.InnerLayer.layer_rho(kequil, [(psifac=0.5,)])) rtol = 1e-12
        @test FFS._rho_at(src, 0.3) > 0
        # ω_*pi = n (dp_i/dΨ)/(Z e n_i) with p_i = n_e T_i, against a finite difference of the profile splines
        p = GPEC.InnerLayer.SLAYER._slayer_profiles(kequil.kinetic, nothing).profiles
        p_i(ψ) = p.n_e(ψ) * p.T_i(ψ)
        h = 1e-4
        for n in (1, 3)
            omega = FFS.ion_diamagnetic_frequency(kequil, n)
            fd = -n / kequil.psio * (p_i(0.5 + h) - p_i(0.5 - h)) / (2h) / p.n_e(0.5)
            @test omega(0.5) ≈ fd rtol = 1e-6
        end
        @test FFS.ion_diamagnetic_frequency(kequil, 2)(0.5) ≈ 2 * FFS.ion_diamagnetic_frequency(kequil, 1)(0.5)
        @test FFS.ion_diamagnetic_frequency(kequil, 1)(0.5) > 0   # outward-decreasing p_i: the layer sign convention
        @test_throws ArgumentError FFS.ion_diamagnetic_frequency(equil, 1)   # the ideal fixture carries no kinetic data
        @test_throws ArgumentError FFS.PerpendicularInertia(equil)

        # The kinetic TJ fixture is ideal-stable too: γ = 0 with the domain's peak ω_*pi reported
        kkw = Dict{Symbol,Any}(q.first => q.second for q in GPEC.force_balance_keywords(kdeck, ktemplate) if !(q.first in (:nn, :wall, :kinetic, :kinetic_factor)))
        kkw[:vac_flag] = true
        kprob = ForceBalanceProblem(kequil; nn=1, wall=GPEC.Vacuum.parse_wall(get(kdeck, "Wall", nothing)), kkw...)
        gr = FFS.growth_rate(kprob, forward; inertia=src)
        @test gr.gamma == 0 && isfinite(gr.omega_star_pi) && gr.omega_star_pi > 0
        omega1 = FFS.ion_diamagnetic_frequency(kequil, 1)
        @test gr.omega_star_pi ≈ maximum(abs ∘ omega1, filter(x -> gr.marginal.psilow <= x <= gr.marginal.psilim, gr.marginal.metric.xs))
        # The config form reuses a marginal solve and derives the same inertia
        cfg = FFS.GrowthRateConfig()
        gr2 = FFS.growth_rate(kprob, forward, cfg; marginal=gr.marginal)
        @test gr2.gamma == gr.gamma && gr2.omega_star_pi == gr.omega_star_pi && gr2.marginal === gr.marginal
        @test FFS._rho_at(FFS.inertia_source(cfg, kequil), 0.4) == FFS._rho_at(src, 0.4)
        @test FFS._rho_at(FFS.inertia_source(FFS.GrowthRateConfig(; rho=2.0), kequil), 0.4) == 2.0
    end
end
