using Test
using GeneralizedPerturbedEquilibrium

# The straight-fieldline coordinate a traced equilibrium produces must not depend on where the ODE
# solver happened to place its steps. Sampling each surface at its own solver steps and resampling
# leaves an error that varies from surface to surface (white noise in ψ). These checks fail by
# 10–1000× when that happens, against either tracer.
@testset "Straight-fieldline sampling is independent of solver steps" begin
    Eq = GeneralizedPerturbedEquilibrium.Equilibrium
    psis = range(0.21, 0.89; length=23)
    thetas = range(0.013, 0.97; length=17)
    # Max over off-grid points of |a − b|, relative to the largest |a|.
    function max_rel_diff(a, b)
        va = [a((p, t)) for p in psis, t in thetas]
        vb = [b((p, t)) for p in psis, t in thetas]
        return maximum(abs.(va .- vb)) / maximum(abs.(va))
    end

    @testset "Solovev: rzphi independent of ψ resolution and on the exact surfaces" begin
        sc = Eq.SolovevConfig(64, 64, 64, 1.6, 0.33, 1.0, 1.9, 1.0, 1.0, 1.0)
        sol_eq(mpsi) = Eq.equilibrium_solver(Eq.sol_run(
            Eq.EquilibriumConfig(; eq_type="sol", eq_filename="unused", jac_type="pest", grid_type="core_edge_packed",
                psilow=1e-4, psihigh=0.99999, mpsi=mpsi, mtheta=128),
            sc))
        coarse, fine = sol_eq(64), sol_eq(128)
        # ν is identically zero for this equilibrium, so it carries no signal.
        for field in (:rzphi_offset, :rzphi_jac, :rzphi_rsquared)
            # Measured: ≤ 2.2e-7 with common SFL abscissae, ≥ 2.1e-4 with per-surface resampling.
            @test max_rel_diff(getfield(coarse, field), getfield(fine, field)) < 1e-5
        end

        # Each (ψ, θ) point maps to (R, Z) on the exact analytic surface.
        # Normalized flux of the Solovev solution that `sol_run` generates (0 on the axis).
        psi_exact(R, Z) = ((R * Z / sc.elongation)^2 + (R^2 - sc.r0^2)^2 / 4) / (sc.a * sc.r0)^2
        for eq in (coarse, fine)
            function rz(p, t)
                r, eta = sqrt(eq.rzphi_rsquared((p, t))), 2π * (t + eq.rzphi_offset((p, t)))
                return eq.ro + r * cos(eta), eq.zo + r * sin(eta)
            end
            # Measured: 1.4e-5 (the floor of the ψ(R, Z) spline) with common SFL abscissae, ≥ 4.2e-4 with per-surface resampling.
            @test maximum(abs(psi_exact(rz(p, t)...) - p) for p in psis, t in thetas) < 1e-4
        end
    end

    @testset "EFIT: efit and efit_arclength tracers agree" begin
        efit_file = joinpath(@__DIR__, "test_data", "CHEASE_test_data", "EQDSK_COCOS_02")
        efit_eq(kind) = Eq.setup_equilibrium(Eq.EquilibriumConfig(; eq_filename=efit_file, eq_type=kind,
            jac_type="boozer", grid_type="core_edge_packed", psilow=0.01, psihigh=0.994, mpsi=64, mtheta=128))
        direct, arclength = efit_eq("efit"), efit_eq("efit_arclength")
        # Two integrators with different step sequences along the same surfaces. Measured: Jacobian 8.7e-8
        # and ν 1.7e-7 when both sample at common abscissae; 2.9e-4 and 1.1e-5 with per-surface resampling.
        @test max_rel_diff(direct.rzphi_jac, arclength.rzphi_jac) < 2e-6
        @test max_rel_diff(direct.rzphi_nu, arclength.rzphi_nu) < 2e-6
    end
end
