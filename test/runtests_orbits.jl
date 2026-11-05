using TOML
using HDF5

@testset "KineticForces guiding-center orbits" begin
    GPEC = GeneralizedPerturbedEquilibrium
    KF = GPEC.KineticForces
    template = joinpath(@__DIR__, "test_data", "regression_solovev_kinetic_calculated")
    deck = TOML.parsefile(joinpath(template, "gpec.toml"))
    equil = GPEC.Equilibrium.PlasmaEquilibrium(GPEC.equilibrium_config(deck, template), GPEC.analytic_config(deck))
    prof = KF.ntv_profiles(KF.KineticForcesControl(), equil)
    g = KF.guiding_center_field(equil)
    mass, chrg = 2 * KF.mp, KF.e
    T = prof.Ti_spline(0.5)

    @testset "field table reproduces q and |B|" begin
        θs = range(0, 1, 257)[1:end-1]
        for ψ in (0.2, 0.5, 0.8)
            q = sum(θ -> g.F(ψ) * g.jac((ψ, θ)) / (g.s * g.psio * g.R((ψ, θ))^2), θs) / length(θs) / 2π
            @test q ≈ equil.profiles.q_spline(ψ) rtol = 1e-3
            for θ in θs[1:32:end]
                Bm = g.bcov_theta((ψ, θ)) * g.s * g.psio / g.jac((ψ, θ)) + (g.F(ψ) / g.R((ψ, θ)))^2 / g.B((ψ, θ))
                @test Bm ≈ g.B((ψ, θ)) rtol = 1e-3
            end
        end
    end

    @testset "classification and conservation" begin
        lam_max = g.bo / g.B((0.5, 0.0))
        trapped = KF.guiding_center_orbit(g, 0.5, 0.99 * lam_max, T, 1; mass, chrg)
        passing = KF.guiding_center_orbit(g, 0.5, 0.2, T, 1; mass, chrg)
        @test trapped.status == :trapped
        @test passing.status == :passing
        for o in (trapped, passing)
            @test o.energy_error < 1e-8
            @test o.pphi_error < 1e-8
        end
        @test KF.guiding_center_orbit(g, 0.5, 1.1 * lam_max, T, 1; mass, chrg) === nothing
    end

    @testset "start-point reflection on an up-down symmetric equilibrium" begin
        for (lam, sv) in ((0.2, 1), (0.2, -1))
            a = KF.guiding_center_orbit(g, 0.5, lam, T, sv; mass, chrg, theta0=0.05)
            b = KF.guiding_center_orbit(g, 0.5, lam, T, sv; mass, chrg, theta0=0.95)
            @test a.wb ≈ b.wb rtol = 1e-6
            @test a.psibar ≈ b.psibar atol = 1e-8
        end
    end

    @testset "co-current passing ions sit outward in R" begin
        # Co-current ⇔ v∥ F s > 0; starting at the outboard midplane, an outward-shifted orbit has ⟨ψ⟩ < ψ₀.
        sco = sign(g.F(0.5)) * g.s
        co = KF.guiding_center_orbit(g, 0.5, 0.2, T, sco; mass, chrg)
        counter = KF.guiding_center_orbit(g, 0.5, 0.2, T, -sco; mass, chrg)
        @test co.psibar < 0.5 < counter.psibar
    end

    @testset "KineticForces/Orbits output" begin
        mktempdir() do dir
            file = joinpath(dir, "orbits.h5")
            h5open(file, "w") do h5
                KF.write_orbits!(h5, KF.KineticForcesControl(; orbit_diagnostic=true), prof, equil; psi=[0.3, 0.6], pitch=[0.2, 0.95], x=[1.0])
            end
            h5open(file) do h5
                @test size(read(h5["KineticForces/Orbits/wb"])) == (2, 2, 1, 2)
                st = read(h5["KineticForces/Orbits/status"])
                @test all(st[:, 1, :, :] .== 2)
                @test all(isfinite, read(h5["KineticForces/Orbits/psibar"])[in.(st, Ref((1, 2)))])
            end
        end
    end
end
