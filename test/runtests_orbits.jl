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
        @test (trapped.ncross, passing.ncross) == (2, 1)
        # Narrow orbits: the precession is integrated relative to the field line, so it keeps the ODE tolerance.
        narrow(rtol) = KF.guiding_center_orbit(g, 0.5, 0.99 * lam_max, T, 1; mass, chrg=4096chrg, rtol).wphi
        @test narrow(1e-8) ≈ narrow(1e-11) rtol = 1e-5
    end

    @testset "orbits trapped by E×B close as trapped" begin
        # R₀ω_E ≈ v_t: counter-going ions are trapped about v∥ ≈ −R ω_E without v∥ reversing (θ̇ reverses instead).
        vt = sqrt(2T / mass)
        gE = KF.guiding_center_field(equil; omegaE=ψ -> vt / g.ro)
        for sv in (-1, 1)
            o = KF.guiding_center_orbit(gE, 0.5, 0.2, T, sv; mass, chrg)
            ws = [o.sol(τ)[4] for τ in range(0, o.sol.t[end]; length=401)]
            @test o.status in (:trapped, :passing)
            @test o.energy_error < 1e-8
            o.status == :trapped && @test (o.ncross, all(<(0), ws) || all(>(0), ws)) == (2, true)
        end
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

    @testset "resonance roots per spline cell" begin
        # Two roots 1e-5 apart (a near-tangent resonance) are both found; a quadratic that misses zero has none.
        r = KF._cell_roots(s -> (s - 0.3) * (s - 0.30001) * (s - 1.7) * (s^2 + 1), 0.0, 2.0, 5)
        @test length(r) == 3 && isapprox(r, [0.3, 0.30001, 1.7]; atol=1e-10)
        @test isempty(KF._cell_roots(s -> (s - 0.5)^2 + 1e-6, 0.0, 1.0, 2))
        @test isempty(KF._cell_roots(s -> 2.0, 0.0, 1.0, 5))
    end

    @testset "orbit-averaged torque reduces to the surface path" begin
        # Synthetic two-mode perturbation: the torque reads only the δB/B and ∇·ξ⊥ Fourier modes.
        intr = KF.KineticForcesInternal(equil)
        intr.mfac, intr.mpert = collect(1:4), 4
        ψg = collect(range(0.0, 1.0, 65))
        dbob = ComplexF64[ψ * (1 - ψ) * (m == 2 ? 1e-3 : 2e-4) for ψ in ψg, m in 1:4]
        divx = ComplexF64[ψ * (1 - ψ) * (m == 3 ? 1e-3im : 1e-4) for ψ in ψg, m in 1:4]
        intr.dbob_m = cubic_interp(ψg, Series(dbob); extrap=ExtendExtrap())
        intr.divx_m = cubic_interp(ψg, Series(divx); extrap=ExtendExtrap())
        intr.orbit_field, intr.orbit_lmax, intr.orbit_nenergy, intr.orbit_nlambda = g, 1, 6, 32
        ψ, n = 0.5, 1
        torque(surf, zi, method) = sum(l -> (r = Ref{ComplexF64}(0); KF.tpsi!(r, ψ, n, l, zi, 2, 1.0, 1.0, false, method, equil, intr, prof; surface=surf); r[]), -1:1)

        # Surface values in the orbit table, one full-range segment per pitch and sign, reproduce fgar on the same
        # pitch grid; they differ only in splitting the pitch spline at bo/bmax, which vanishes as the grid is refined.
        function replay_table(sf, geo)
            sdj = reduce(hcat, [KF.bounce_harmonic(geo, l, n, sf.q, intr.ro, intr.mpert).dJdJ for l in -1:1])
            us = [0.5, 1.0, 2.0, 4.0]
            segs = map(1:geo.nlmda) do il
                geo.valid[il] || return KF.OrbitSegment[]
                passing = geo.sigma[il] != 0
                [
                    KF.OrbitSegment(passing, dir, 0.0, sqrt(KF.X_ENERGY_MAX), us, fill(geo.wb[il], 4), fill(geo.wd[il], 4), repeat(sdj[il:il, :], 4))
                    for dir in (passing ? (-1, 1) : (0, 0))
                ]
            end
            return KF.OrbitSurfaceTable(geo.lambda[1:geo.nlmda], segs, 1, zeros(Int, 6))
        end
        orbit_torque(tab, sf; classes=:all) = sum(
            l -> KF.calculate_orbit(tab, n, l, sf.q, sf.epsr, sf.wdian, sf.wdiat, sf.welec, sf.nuk,
                intr.bo, sf.bmax, sf.n_s, sf.T_s; chi1=intr.chi1, ro=intr.ro, energy_atol=1e-11, energy_rtol=1e-8, classes), -1:1)
        sf = KF._torque_surface(ψ, 1, 2, false, "fgar", equil, intr, prof; n)
        geo512 = KF.compute_bounce_geometry(ψ, n, sf.q, intr.bo, sf.bmax, sf.bmin, sf.theta_bmax, sf.tspl, sf.B_vpar, intr.mfac,
            intr.chi1, intr.ro, sf.dbob_m_f, sf.divx_m_f, 1.0, 1.0, sf.mass, sf.chrg, sf.T_s, "fgar"; nlmda=512, surface_field=sf.bf)
        sf512 = merge(sf, (; geometry=geo512))
        @test orbit_torque(replay_table(sf512, geo512), sf512) ≈ torque(sf512, 1, "fgar") rtol = 1e-4

        # At small width each orbit class carries the surface path's share, so the class bookkeeping keys off the orbit.
        # Passing orbits carry their own ⟨φ̇⟩ − qω_θ, which differs from the surface ω_D at O(ω_D), hence the wider band.
        surfaces(zi) = (KF._torque_surface(ψ, zi, 2, false, "fgar", equil, intr, prof; n), KF._torque_surface(ψ, zi, 2, false, "forb", equil, intr, prof; n))
        sf, so = surfaces(1024)
        replay = replay_table(sf, so.geometry)
        @test orbit_torque(so.orbit, sf; classes=:passing) ≈ orbit_torque(replay, sf; classes=:passing) rtol = 5e-2
        @test orbit_torque(so.orbit, sf; classes=:trapped) ≈ orbit_torque(replay, sf; classes=:trapped) rtol = 1e-3

        # Orbit width ∝ 1/Z: the difference from the surface values on the same pitch grid vanishes with it.
        err(zi, sf, so) = abs(orbit_torque(so.orbit, sf) / orbit_torque(replay_table(sf, so.geometry), sf) - 1)
        e1, e2 = err(4, surfaces(4)...), err(1024, sf, so)
        @test e2 < 1e-4
        @test e2 < e1 / 10
    end
end
