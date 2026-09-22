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

@testset "EnergyDecomposition: surface densities" begin
    ffs = _ED_FFS
    equil = ffs.equil
    psi_grid, _ = PerturbedEquilibrium.decomposition_grid(ffs)
    modes = PerturbedEquilibrium.eigenmode_modes(ffs, 1, psi_grid)
    thetas_ext = collect(equil.rzphi_ys)
    mtheta = length(thetas_ext) - 1
    mvals = collect(ffs.mlow:ffs.mhigh)
    ft = GeneralizedPerturbedEquilibrium.Utilities.FourierTransforms.FourierTransform(mtheta, ffs.mpert, ffs.mlow)
    geom = PerturbedEquilibrium.SurfaceGeometry(mtheta)
    kern = PerturbedEquilibrium.CurvatureKernel(mtheta)
    sf = PerturbedEquilibrium.SurfaceFields(mtheta)
    eff = PerturbedEquilibrium.EffectiveField(mtheta)
    den = PerturbedEquilibrium.SurfaceDensities(mtheta)
    work = zeros(ComplexF64, ffs.mpert)
    ipsi = length(psi_grid) ÷ 2
    psi = psi_grid[ipsi]
    PerturbedEquilibrium.surface_geometry!(geom, equil, psi, thetas_ext[1:mtheta])
    PerturbedEquilibrium.curvature_kernel!(kern, geom, equil, psi, thetas_ext)
    PerturbedEquilibrium.surface_fields!(sf, ft, modes, ipsi, mvals, geom, work)
    PerturbedEquilibrium.effective_field!(eff, geom, equil, psi, sf)
    PerturbedEquilibrium.surface_densities!(den, geom, kern, sf, eff, equil, psi, ffs.nlow; effective_field_form=true, standard_form=true)
    @test den.effective_b_squared ≈ den.effective_b_squared_psi .+ den.effective_b_squared_theta .+ den.effective_b_squared_zeta
    @test all(den.effective_b_squared .>= 0)
    @test all(den.b_squared .>= 0)
    @test all(den.parallel_current_squared .>= 0)
    @test PerturbedEquilibrium.surface_integral(den.effective_b_squared, geom.jac) ≈ sum(den.effective_b_squared .* geom.jac) / mtheta
    # The standard-form current term is real up to roundoff at the surface-integral level.
    cc = PerturbedEquilibrium.surface_integral(den.current_coupling, geom.jac)
    @test abs(imag(cc)) < 1e-8 * max(abs(real(cc)), 1e-300)
    # Disabling a form leaves its densities untouched (zero).
    fill!(den.b_squared, 0.0)
    PerturbedEquilibrium.surface_densities!(den, geom, kern, sf, eff, equil, psi, ffs.nlow; effective_field_form=true, standard_form=false)
    @test all(iszero, den.b_squared)
end

@testset "EnergyDecomposition: decompose_energy totals, checks, and toggles" begin
    ffs = _ED_FFS
    res = PerturbedEquilibrium.decompose_energy(ffs; eigenmodes=[1], write_densities=true, verbose=false)
    npsi = length(res.psi)
    mtheta = length(ffs.equil.rzphi_ys) - 1
    @test res.eigenmode_index == [1]
    @test res.psi == first(PerturbedEquilibrium.decomposition_grid(ffs))
    @test res.effective_b_curl_residual_relative[1] < 1e-8
    @test res.effective_b_curl_residual_rms[1] <= res.effective_b_curl_residual_max[1]
    @test res.dW_plasma_reference[1] ≈ real(ffs.free_boundary.ep[1])
    # mpsi = 16 fixture: 1.4 % / 1.2 % observed; both forms converge to 2e-3 / 3e-4 at mpsi = 128.
    @test abs(res.dW_plasma_relative_error[1]) < 0.03
    @test abs(res.dW_plasma_standard_form_relative_error[1]) < 0.03
    @test res.dW_cumulative[1, 1] == 0.0
    @test res.dW_cumulative[end, 1] ≈ res.dW_plasma[1]
    @test res.dW_cumulative_standard_form[end, 1] ≈ res.dW_plasma_standard_form[1]
    @test res.effective_b_squared ≈ res.effective_b_squared_psi .+ res.effective_b_squared_theta .+ res.effective_b_squared_zeta
    @test size(res.effective_b_squared_density) == (npsi, mtheta, 1)
    @test size(res.current_coupling_density) == (npsi, mtheta, 1)
    @test size(res.R) == (npsi, mtheta)
    @test res.theta == [(k - 1) / mtheta for k in 1:mtheta]
    ipsi = npsi ÷ 2
    jac = [ffs.equil.rzphi_jac((res.psi[ipsi], t)) for t in res.theta]
    @test sum(res.effective_b_squared_density[ipsi, :, 1] .* jac) / mtheta ≈ res.effective_b_squared[ipsi, 1] rtol = 1e-10
    @test sum(res.b_squared_density[ipsi, :, 1] .* jac) / mtheta ≈ res.b_squared[ipsi, 1] rtol = 1e-10

    plain = PerturbedEquilibrium.decompose_energy(ffs; verbose=false)
    @test isempty(plain.theta) && isempty(plain.R) && isempty(plain.effective_b_squared_density)
    @test plain.dW_plasma == res.dW_plasma

    only_eff = PerturbedEquilibrium.decompose_energy(ffs; standard_form=false, verbose=false)
    @test isempty(only_eff.b_squared) && isempty(only_eff.dW_plasma_standard_form) && isempty(only_eff.dW_cumulative_standard_form)
    @test only_eff.dW_plasma == res.dW_plasma

    only_std = PerturbedEquilibrium.decompose_energy(ffs; effective_field_form=false, verbose=false)
    @test isempty(only_std.dW_plasma) && isempty(only_std.effective_b_squared) && isempty(only_std.effective_b_curl_residual_max)
    @test only_std.dW_plasma_standard_form == res.dW_plasma_standard_form

    two = PerturbedEquilibrium.decompose_energy(ffs; eigenmodes=[1, 2], verbose=false)
    @test size(two.effective_b_squared) == (npsi, 2)
    @test two.dW_plasma[1] == res.dW_plasma[1]
    @test two.dW_plasma_reference[2] ≈ real(ffs.free_boundary.ep[2])
    # mpsi = 16 fixture: 3.3 % observed for the second mode; 1e-3 at mpsi = 128.
    @test abs(two.dW_plasma_relative_error[2]) < 0.07

    @test_throws ArgumentError PerturbedEquilibrium.decompose_energy(ffs; effective_field_form=false, standard_form=false)
    @test_throws ArgumentError PerturbedEquilibrium.decompose_energy(ffs; eigenmodes=[0])
    @test_throws ArgumentError PerturbedEquilibrium.decompose_energy(ffs; eigenmodes=[size(ffs.free_boundary.wt, 2) + 1])
end
