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

    @testset "Solovev: rzphi independent of ψ resolution" begin
        sol_eq(mpsi) = Eq.equilibrium_solver(Eq.sol_run(
            Eq.EquilibriumConfig(; eq_type="sol", eq_filename="unused", jac_type="pest", grid_type="ldp",
                psilow=1e-4, psihigh=0.99999, mpsi=mpsi, mtheta=128),
            Eq.SolovevConfig(64, 64, 64, 1.6, 0.33, 1.0, 1.9, 1.0, 1.0, 1.0)))
        coarse, fine = sol_eq(64), sol_eq(128)
        # ν is identically zero for this equilibrium, so it carries no signal.
        for field in (:rzphi_offset, :rzphi_jac, :rzphi_rsquared)
            # Measured: ≤ 2.2e-7 with common SFL abscissae, ≥ 2.1e-4 with per-surface resampling.
            @test max_rel_diff(getfield(coarse, field), getfield(fine, field)) < 1e-5
        end
    end

    @testset "EFIT: efit and efit_arclength tracers agree" begin
        efit_file = joinpath(@__DIR__, "test_data", "CHEASE_test_data", "EQDSK_COCOS_02")
        efit_eq(kind) = Eq.setup_equilibrium(Eq.EquilibriumConfig(; eq_filename=efit_file, eq_type=kind,
            jac_type="boozer", grid_type="ldp", psilow=0.01, psihigh=0.994, mpsi=64, mtheta=128))
        direct, arclength = efit_eq("efit"), efit_eq("efit_arclength")
        # Two integrators with different step sequences along the same surfaces. Measured on the SFL
        # Jacobian: 2.3e-6 when both sample at common abscissae, ≥ 2.4e-4 if either resamples its steps.
        @test max_rel_diff(direct.rzphi_jac, arclength.rzphi_jac) < 2e-5
    end
end
