using Test
using HDF5
using GeneralizedPerturbedEquilibrium
using GeneralizedPerturbedEquilibrium.PerturbedEquilibrium
using GeneralizedPerturbedEquilibrium.ForceFreeStates
using GeneralizedPerturbedEquilibrium.Equilibrium

# Solovev forward fixture (mpsi = 16, vac_flag = true) with the [EnergyDecomposition] section
# appended, run once in a scratch directory and shared by every testset below.
const _ED_FIXTURE = joinpath(@__DIR__, "test_data", "regression_solovev_ideal_example")
const _ED_SECTION = """

[EnergyDecomposition]
eigenmodes = [1]              # Free-boundary eigenmodes to decompose (1 = least stable)
effective_field_form = true   # Decompose δW_p in the effective-field form
standard_form = true          # Decompose δW_p in the standard form
write_densities = true        # Store the per-(ψ,θ) densities so their shapes can be checked
"""

function _run_energy_fixture()
    dir = mktempdir()
    deck = read(joinpath(_ED_FIXTURE, "gpec.toml"), String)
    write(joinpath(dir, "gpec.toml"), deck * _ED_SECTION)
    return dir, GeneralizedPerturbedEquilibrium.main([dir])
end

const _ED_DIR, _ED_RES = _run_energy_fixture()
const _ED_FFS = _ED_RES.ffs

@testset "EnergyDecomposition: contra displacements take reg_spot as a keyword" begin
    ffs = _ED_FFS
    sol = ffs.solution
    equil = ffs.equil
    psi_grid = sol.psi_store[1:sol.step]
    u1_edge = Matrix(@view sol.u_store[:, :, 1, sol.step])
    edge = Vector{ComplexF64}(ffs.free_boundary.wt[:, 1])
    xi_psi, xi_psi1, xi_s = PerturbedEquilibrium.sum_eigenmode_contributions(edge, u1_edge, sol, ffs)
    alpha_over_chi1 = xi_s ./ (2π * equil.psio)
    xwp, xwt, xwz, xmt, xmz = PerturbedEquilibrium.compute_contra_displacements(
        xi_psi, xi_psi1, alpha_over_chi1, psi_grid, equil, ffs, ffs.metric; reg_spot=0.0)
    @test xmt == xwt
    @test xmz == xwz
    xwp_r, xwt_r, _, xmt_r, _ = PerturbedEquilibrium.compute_contra_displacements(
        xi_psi, xi_psi1, alpha_over_chi1, psi_grid, equil, ffs, ffs.metric; reg_spot=0.05)
    @test xwp_r == xwp
    @test xwt_r == xwt
    @test xmt_r != xwt_r
end

@testset "EnergyDecomposition: surface geometry" begin
    equil = _ED_FFS.equil
    mtheta = length(equil.rzphi_ys) - 1
    thetas = [(k - 1) / mtheta for k in 1:mtheta]
    geom = PerturbedEquilibrium.SurfaceGeometry(mtheta)
    psi = 0.5
    PerturbedEquilibrium.surface_geometry!(geom, equil, psi, thetas)
    hint = (Ref(1), Ref(1))
    for k in 1:mtheta
        m = Equilibrium.flux_surface_metric(equil, psi, thetas[k]; hint=hint)
        @test geom.jac[k] ≈ m.jac rtol = 1e-12
        @test sqrt(geom.delpsi2[k]) ≈ m.delpsi rtol = 1e-12
        @test hypot(geom.R[k] - equil.ro, geom.Z[k] - equil.zo)^2 ≈ equil.rzphi_rsquared((psi, thetas[k]); hint=hint) rtol = 1e-10
    end
    @test all(geom.bsq .> 0)
    @test all(geom.g22 .> 0) && all(geom.g33 .> 0)
    @test all(geom.g22 .* geom.g33 .- geom.g23 .^ 2 .> 0)
    @test all(isfinite, geom.jac_psi) && all(isfinite, geom.bsq_psi) && all(isfinite, geom.bsq_theta)
    ctrl = PerturbedEquilibrium.EnergyDecompositionControl()
    @test ctrl.eigenmodes == [1] && ctrl.effective_field_form && ctrl.standard_form && !ctrl.write_densities
    res = PerturbedEquilibrium.EnergyDecompositionResult()
    @test isempty(res.psi) && isempty(res.dW_plasma) && isempty(res.effective_b_squared_density)
end
