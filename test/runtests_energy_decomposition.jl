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

@testset "EnergyDecomposition: curvature kernel" begin
    equil = _ED_FFS.equil
    thetas_ext = collect(equil.rzphi_ys)
    mtheta = length(thetas_ext) - 1
    geom = PerturbedEquilibrium.SurfaceGeometry(mtheta)
    kern = PerturbedEquilibrium.CurvatureKernel(mtheta)
    psi = 0.5
    PerturbedEquilibrium.surface_geometry!(geom, equil, psi, thetas_ext[1:mtheta])
    PerturbedEquilibrium.curvature_kernel!(kern, geom, equil, psi, thetas_ext)
    @test all(isfinite, kern.shear) && all(isfinite, kern.curvature) && all(isfinite, kern.sigma)
    @test kern.K2 ≈ PerturbedEquilibrium.MU_0 .* geom.bsq .* kern.sigma .^ 2
    @test kern.K1 ≈ geom.delpsi2 .* kern.sigma .* kern.shear
    # p' = (μ₀p)'/μ₀ on this surface; K3 = 2 p' κ_ψ
    p1 = equil.profiles.P_deriv(psi) / PerturbedEquilibrium.MU_0
    @test kern.K3 ≈ 2 .* p1 .* kern.curvature
    # The Solovev equilibrium carries a finite parallel current: σ must not vanish identically.
    @test maximum(abs, kern.sigma) > 0
    # Flux-surface average of the shear geometry term vanishes, so ⟨J S⟩/χ'² reduces to q' ⟨1⟩.
    chi1 = 2π * equil.psio
    q1 = equil.profiles.q_deriv(psi)
    @test sum(kern.shear .* geom.jac) / mtheta ≈ chi1^2 * q1 rtol = 1e-6
end

@testset "EnergyDecomposition: effective field satisfies (∇×b_eff)·∇ψ = 0" begin
    ffs = _ED_FFS
    equil = ffs.equil
    psi, is_knot = PerturbedEquilibrium.decomposition_grid(ffs)
    @test all(diff(psi) .> 0)
    @test psi[end] == ffs.solution.psi_store[ffs.solution.step]
    @test count(is_knot) >= 2
    modes = PerturbedEquilibrium.eigenmode_modes(ffs, 1, psi)
    @test size(modes.Jxi_psi) == (length(psi), ffs.mpert)
    thetas_ext = collect(equil.rzphi_ys)
    mtheta = length(thetas_ext) - 1
    mvals = collect(ffs.mlow:ffs.mhigh)
    ft = GeneralizedPerturbedEquilibrium.Utilities.FourierTransforms.FourierTransform(mtheta, ffs.mpert, ffs.mlow)
    geom = PerturbedEquilibrium.SurfaceGeometry(mtheta)
    sf = PerturbedEquilibrium.SurfaceFields(mtheta)
    eff = PerturbedEquilibrium.EffectiveField(mtheta)
    res = zeros(ComplexF64, mtheta)
    bufs = (zeros(ComplexF64, ffs.mpert), zeros(ComplexF64, ffs.mpert), zeros(ComplexF64, ffs.mpert), zeros(ComplexF64, mtheta))
    worst = 0.0
    for i in findall(is_knot)
        PerturbedEquilibrium.surface_geometry!(geom, equil, psi[i], thetas_ext[1:mtheta])
        PerturbedEquilibrium.surface_fields!(sf, ft, modes, i, mvals, geom, bufs[3])
        PerturbedEquilibrium.effective_field!(eff, geom, equil, psi[i], sf)
        @test eff.Jc_psi == sf.Jb_psi
        scale = PerturbedEquilibrium.curl_residual!(res, ft, eff, geom, mvals, ffs.nlow, bufs...)
        @test scale > 0
        worst = max(worst, maximum(abs, res) / scale)
    end
    @test worst < 1e-8
end
