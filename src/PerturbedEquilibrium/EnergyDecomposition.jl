"""
Energy decomposition of free-boundary eigenmodes on the (ψ,θ) grid.

Evaluates the ideal-MHD plasma potential energy δW_p of a ForceFreeStates eigenmode in two
forms and splits it into its physical terms (Fortran GPEC `gpout_recon`/`gpout_recon2`, theory
note `docs/tex/recon/main.tex` of the Fortran repository, Bernstein et al. 1958):

  - effective-field form  2δW_p = ∫dψ ∮dθ J [ |b_eff|²/μ₀ − K|ξ_n|² ],  b_eff = b + ξ_n (μ₀ j × n̂)
  - standard form         2δW_p = ∫dψ ∮dθ J [ |b|²/μ₀ − j·(b×ξ*) + (∇·ξ)*(ξ·∇p) ]

both without the compressional γp|∇·ξ|² term (a reconstruction assumption shared with the
Fortran). K = K1 + K2 + K3 with K1 = |∇ψ|²σS (shear-current), K2 = μ₀B²σ² (parallel current),
K3 = 2p'κ_ψ (pressure-curvature) and σ = j·B/B². Every stored energy is scaled by 2μ₀/ψ₀² so the
totals read in the normalization of `eigenmode_plasma_energies`.

Array conventions follow the gpeq arrays: `Jxi_psi` = Jξ^ψ, `Jb_*` = Jb^i (Jacobian-weighted
contravariant), `b_cov_*` = b_i (covariant), `xi_n` = ξ·n̂.
"""

using ..Utilities.PhysicalConstants: MU_0

"""
    EnergyDecompositionControl

User-facing controls of the `[EnergyDecomposition]` section of gpec.toml.

## Fields

  - `eigenmodes::Vector{Int}` - Free-boundary eigenmodes to decompose (1 = least stable)
  - `effective_field_form::Bool` - Decompose δW_p in the effective-field form ½∫[|b_eff|²/μ₀ − K|ξ_n|²]
  - `standard_form::Bool` - Decompose δW_p in the standard form ½∫[|b|²/μ₀ − j·(b×ξ*) + (∇·ξ)*(ξ·∇p)]
  - `write_densities::Bool` - Also store the per-(ψ,θ) energy densities with R,Z for 2-D maps
"""
@kwdef struct EnergyDecompositionControl
    eigenmodes::Vector{Int} = [1]
    effective_field_form::Bool = true
    standard_form::Bool = true
    write_densities::Bool = false
end

"""
    EnergyDecompositionResult

Energy decomposition of the requested free-boundary eigenmodes. Field names are the dataset
names under `ForceFreeStates/EnergyDecomposition/`; profiles are `(psi × eigenmode)` matrices,
totals `(eigenmode)` vectors, densities `(psi × theta × eigenmode)` arrays. Energies are scaled
by 2μ₀/ψ₀² to the normalization of `eigenmode_plasma_energies`. Fields of a disabled form or of
unrequested densities stay empty.

## Fields

  - `psi` - ψ_N of the ξ solution grid
  - `eigenmode_index` - free-boundary eigenmode index of each decomposed mode
  - `effective_b_squared`, `effective_b_squared_psi/theta/zeta` - ∮J|b_eff|²/μ₀ dθ and its component split
  - `shear_current`, `parallel_current_squared`, `pressure_curvature` - ∮J K_i|ξ_n|² dθ for K1, K2, K3
  - `b_squared`, `current_coupling`, `pressure_compression` - standard-form pieces ∮J|b|²/μ₀, ∮J j·(b×ξ*), ∮J(∇·ξ)*(ξ·∇p) dθ
  - `dW_cumulative`, `dW_cumulative_standard_form` - running ½∫dψ of each form
  - `dW_plasma`, `dW_plasma_standard_form` - the two totals
  - `dW_plasma_reference` - `ep[k]` of the same eigenmode from FreeBoundaryStability
  - `dW_plasma_relative_error`, `dW_plasma_standard_form_relative_error` - (total − reference)/|reference|
  - `effective_b_curl_residual_max/rms/relative` - the (∇×b_eff)·∇ψ = 0 check
  - `theta`, `R`, `Z` - the θ grid and cylindrical coordinates of the densities
  - `*_density` - per-point integrands of the profiles above (integral = Σ density·J/mtheta)
"""
@kwdef mutable struct EnergyDecompositionResult
    psi::Vector{Float64} = Float64[]
    eigenmode_index::Vector{Int} = Int[]
    effective_b_squared::Matrix{Float64} = zeros(0, 0)
    effective_b_squared_psi::Matrix{Float64} = zeros(0, 0)
    effective_b_squared_theta::Matrix{Float64} = zeros(0, 0)
    effective_b_squared_zeta::Matrix{Float64} = zeros(0, 0)
    shear_current::Matrix{Float64} = zeros(0, 0)
    parallel_current_squared::Matrix{Float64} = zeros(0, 0)
    pressure_curvature::Matrix{Float64} = zeros(0, 0)
    b_squared::Matrix{Float64} = zeros(0, 0)
    current_coupling::Matrix{ComplexF64} = zeros(ComplexF64, 0, 0)
    pressure_compression::Matrix{ComplexF64} = zeros(ComplexF64, 0, 0)
    dW_cumulative::Matrix{Float64} = zeros(0, 0)
    dW_cumulative_standard_form::Matrix{ComplexF64} = zeros(ComplexF64, 0, 0)
    dW_plasma::Vector{Float64} = Float64[]
    dW_plasma_standard_form::Vector{ComplexF64} = ComplexF64[]
    dW_plasma_reference::Vector{Float64} = Float64[]
    dW_plasma_relative_error::Vector{Float64} = Float64[]
    dW_plasma_standard_form_relative_error::Vector{Float64} = Float64[]
    effective_b_curl_residual_max::Vector{Float64} = Float64[]
    effective_b_curl_residual_rms::Vector{Float64} = Float64[]
    effective_b_curl_residual_relative::Vector{Float64} = Float64[]
    theta::Vector{Float64} = Float64[]
    R::Matrix{Float64} = zeros(0, 0)
    Z::Matrix{Float64} = zeros(0, 0)
    effective_b_squared_density::Array{Float64,3} = zeros(0, 0, 0)
    effective_b_squared_psi_density::Array{Float64,3} = zeros(0, 0, 0)
    effective_b_squared_theta_density::Array{Float64,3} = zeros(0, 0, 0)
    effective_b_squared_zeta_density::Array{Float64,3} = zeros(0, 0, 0)
    shear_current_density::Array{Float64,3} = zeros(0, 0, 0)
    parallel_current_squared_density::Array{Float64,3} = zeros(0, 0, 0)
    pressure_curvature_density::Array{Float64,3} = zeros(0, 0, 0)
    effective_b_curl_residual_density::Array{ComplexF64,3} = zeros(ComplexF64, 0, 0, 0)
    b_squared_density::Array{Float64,3} = zeros(0, 0, 0)
    current_coupling_density::Array{ComplexF64,3} = zeros(ComplexF64, 0, 0, 0)
    pressure_compression_density::Array{ComplexF64,3} = zeros(ComplexF64, 0, 0, 0)
end

"""
    SurfaceGeometry(mtheta)

Equilibrium geometry of one flux surface on the periodic θ grid, filled by
`surface_geometry!`. Cylindrical `R`, `Z`; Jacobian `jac` and its ψ-derivative `jac_psi`;
`delpsi2` = |∇ψ|², `dpdt` = ∇ψ·∇θ, `dpdz` = ∇ψ·∇ζ; covariant metric `g22`, `g23`, `g33` of the
tangent basis e_i = ∂x/∂q^i; `bsq` = B² with its ψ- and θ-derivatives `bsq_psi`, `bsq_theta`.
"""
struct SurfaceGeometry
    R::Vector{Float64}
    Z::Vector{Float64}
    jac::Vector{Float64}
    jac_psi::Vector{Float64}
    delpsi2::Vector{Float64}
    dpdt::Vector{Float64}
    dpdz::Vector{Float64}
    g22::Vector{Float64}
    g23::Vector{Float64}
    g33::Vector{Float64}
    bsq::Vector{Float64}
    bsq_psi::Vector{Float64}
    bsq_theta::Vector{Float64}
end

SurfaceGeometry(mtheta::Int) = SurfaceGeometry((zeros(Float64, mtheta) for _ in 1:13)...)

"""
    surface_geometry!(geom, equil, psi, thetas; hint=(Ref(1), Ref(1))) -> geom

Fill `geom` at flux surface `psi` on the θ points `thetas`. The gradient basis (∇ψ, ∇θ, ∇ζ)
and the tangent basis (e_θ, e_ζ) are the `w` and `v` matrices of the Fortran `gpeq_c`; |∇ψ| and
the Jacobian agree with `Equilibrium.flux_surface_metric`. Derivatives are with respect to
normalized ψ_N, as in the Fortran bicubic evaluations.
"""
function surface_geometry!(geom::SurfaceGeometry, equil::Equilibrium.PlasmaEquilibrium, psi::Float64, thetas::AbstractVector{Float64};
    hint=(Ref(1), Ref(1)))
    ro = equil.ro
    zo = equil.zo
    for (k, theta) in enumerate(thetas)
        pt = (psi, theta)
        r2 = equil.rzphi_rsquared(pt; hint=hint)
        r2_x = equil.rzphi_rsquared(pt; deriv=DerivOp(1, 0), hint=hint)
        r2_y = equil.rzphi_rsquared(pt; deriv=DerivOp(0, 1), hint=hint)
        deta = equil.rzphi_offset(pt; hint=hint)
        deta_x = equil.rzphi_offset(pt; deriv=DerivOp(1, 0), hint=hint)
        deta_y = equil.rzphi_offset(pt; deriv=DerivOp(0, 1), hint=hint)
        nu_x = equil.rzphi_nu(pt; deriv=DerivOp(1, 0), hint=hint)
        nu_y = equil.rzphi_nu(pt; deriv=DerivOp(0, 1), hint=hint)
        jac = equil.rzphi_jac(pt; hint=hint)
        jac_x = equil.rzphi_jac(pt; deriv=DerivOp(1, 0), hint=hint)
        B = equil.eqfun_B(pt; hint=hint)
        B_x = equil.eqfun_B(pt; deriv=DerivOp(1, 0), hint=hint)
        B_y = equil.eqfun_B(pt; deriv=DerivOp(0, 1), hint=hint)

        rfac = sqrt(abs(r2))
        eta = 2π * (theta + deta)
        R = ro + rfac * cos(eta)

        # Gradient basis in the poloidal plane: rows ∇ψ, ∇θ, ∇ζ (Fortran gpeq_c w-matrix).
        w11 = (1.0 + deta_y) * 4π^2 * rfac * R / jac
        w12 = -r2_y * π * R / (rfac * jac)
        w21 = -4π^2 * rfac * R * deta_x / jac
        w22 = π * R * r2_x / (rfac * jac)
        w31 = (2π * R * rfac / jac) * (deta_x * nu_y - nu_x * (1.0 + deta_y))
        w32 = (R / (2.0 * rfac * jac)) * (nu_x * r2_y - r2_x * nu_y)

        # Tangent basis e_θ, e_ζ without the Jacobian factor (Fortran gpeq_c v-matrix).
        v21 = r2_y / (2.0 * rfac)
        v22 = (1.0 + deta_y) * 2π * rfac
        v23 = nu_y * R
        v33 = 2π * R

        geom.R[k] = R
        geom.Z[k] = zo + rfac * sin(eta)
        geom.jac[k] = jac
        geom.jac_psi[k] = jac_x
        geom.delpsi2[k] = w11^2 + w12^2
        geom.dpdt[k] = w11 * w21 + w12 * w22
        geom.dpdz[k] = w11 * w31 + w12 * w32
        geom.g22[k] = v21^2 + v22^2 + v23^2
        geom.g23[k] = v23 * v33
        geom.g33[k] = v33^2
        geom.bsq[k] = B^2
        geom.bsq_psi[k] = 2.0 * B * B_x
        geom.bsq_theta[k] = 2.0 * B * B_y
    end
    return geom
end

"""
    CurvatureKernel(mtheta)

Per-θ pieces of the destabilizing kernel on one surface, filled by `curvature_kernel!`:
the DCON shear `shear` = S, the curvature projection `curvature` = κ_ψ = κ·∇ψ, the parallel
current density ratio `sigma` = j·B/B², and K1 = |∇ψ|²σS, K2 = μ₀B²σ², K3 = 2p'κ_ψ.
"""
struct CurvatureKernel
    shear::Vector{Float64}
    curvature::Vector{Float64}
    sigma::Vector{Float64}
    K1::Vector{Float64}
    K2::Vector{Float64}
    K3::Vector{Float64}
end

CurvatureKernel(mtheta::Int) = CurvatureKernel((zeros(Float64, mtheta) for _ in 1:6)...)

"""
    curvature_kernel!(kern, geom, equil, psi, thetas_ext; hint=Ref(1)) -> kern

Fill the curvature kernel at `psi` from the geometry `geom` of the periodic θ points
`thetas_ext[1:end-1]` (`thetas_ext` closes the period at 1.0). Fortran `gpeq_shear`,
`gpeq_curvature`, `gpeq_K`; main.tex §"Expression of K in DCON coordinates":

    S   = (χ'²/J) [ q' + ∂_θ( (q ∇ψ·∇θ − ∇ψ·∇ζ)/|∇ψ|² ) ]      (periodic cubic-spline derivative)
    κ_ψ = (|∇ψ|²/B²) [ (μ₀p)' + ½∂_ψB² + ½∂_θB² (∇ψ·∇θ)/|∇ψ|² ]
    σ   = j·B/B²  with  j^θ = −F'/(μ₀J),  j^ζ = q j^θ − p'/χ',  B^θ = χ'/J,  B^ζ = qχ'/J

where F' = d(2πF)/dψ and p' = (μ₀p)'/μ₀ come from the equilibrium profile splines.
"""
function curvature_kernel!(kern::CurvatureKernel, geom::SurfaceGeometry, equil::Equilibrium.PlasmaEquilibrium, psi::Float64,
    thetas_ext::Vector{Float64}; hint=Ref(1))
    profiles = equil.profiles
    q = profiles.q_spline(psi; hint=hint)
    q1 = profiles.q_deriv(psi; hint=hint)
    F1 = profiles.F_deriv(psi; hint=hint)
    mu0p1 = profiles.P_deriv(psi; hint=hint)
    p1 = mu0p1 / MU_0
    chi1 = 2π * equil.psio
    mtheta = length(geom.jac)

    # Shear geometry term on the closed θ grid; its θ-derivative comes from a periodic cubic spline.
    ratio = Vector{Float64}(undef, mtheta + 1)
    for k in 1:mtheta
        ratio[k] = (q * geom.dpdt[k] - geom.dpdz[k]) / geom.delpsi2[k]
    end
    ratio[mtheta+1] = ratio[1]
    dratio = deriv1(cubic_interp(thetas_ext, ratio; bc=PeriodicBC()))

    for k in 1:mtheta
        jac = geom.jac[k]
        kern.shear[k] = (chi1^2 / jac) * (q1 + dratio(thetas_ext[k]))
        kern.curvature[k] = (geom.delpsi2[k] / geom.bsq[k]) * (mu0p1 + 0.5 * geom.bsq_psi[k] + 0.5 * geom.bsq_theta[k] * geom.dpdt[k] / geom.delpsi2[k])
        jt = -F1 / (jac * MU_0)
        jz = q * jt - p1 / chi1
        bt = chi1 / jac
        bz = q * chi1 / jac
        jdotb = geom.g22[k] * bt * jt + geom.g33[k] * bz * jz + geom.g23[k] * (bt * jz + bz * jt)
        sigma = jdotb / geom.bsq[k]
        kern.sigma[k] = sigma
        kern.K1[k] = geom.delpsi2[k] * sigma * kern.shear[k]
        kern.K2[k] = MU_0 * geom.bsq[k] * sigma^2
        kern.K3[k] = 2.0 * p1 * kern.curvature[k]
    end
    return kern
end

"""
    decomposition_grid(ffs) -> (psi, is_knot)

Radial nodes of the decomposition: the equilibrium ψ knots inside the ξ solution domain, where
the metric coefficients, the Euler-Lagrange matrices and the geometry are all exact (the Fortran
`gpout_recon` loops over the same knots), plus the solution edge appended when the last knot
falls short of it so the radial integral reaches the plasma edge. `is_knot` marks the exact
nodes; the (∇×b_eff)·∇ψ residual is an algebraic identity there and is reported on them only.
"""
function decomposition_grid(ffs::ForceFreeStatesResult)
    solution = ffs.solution
    psi_lo = solution.psi_store[1]
    psi_hi = solution.psi_store[solution.step]
    psi = [x for x in ffs.equil.rzphi_xs if psi_lo <= x <= psi_hi]
    is_knot = trues(length(psi))
    if isempty(psi) || psi[end] < psi_hi - 1e-10
        push!(psi, psi_hi)
        push!(is_knot, false)
    end
    return psi, is_knot
end

"""
    eigenmode_modes(ffs, k, psi) -> NamedTuple

Mode-space displacement and perturbed field of free-boundary eigenmode `k` on the radial nodes
`psi`, `(length(psi) × mpert)` matrices named after the gpeq arrays: `xi_psi` = ξ^ψ,
`xi_psi1` = ∂ξ^ψ/∂ψ, `Jxi_psi/theta/zeta` = Jξ^i, `Jb_psi/theta/zeta` = Jb^i,
`b_cov_psi/theta/zeta` = b_i. The edge boundary condition is the eigenvector `wt[:, k]` (unit
power norm); ξ and ξ' are cubic-spline interpolated from the ξ solution grid and ξ_s is
recomputed at every node as −A⁻¹(Bξ' + Cξ) from the ideal Euler-Lagrange matrices there
(Fortran `gpeq_sol`), which keeps the fields algebraically consistent with the metric at the
equilibrium knots. The rest of the chain is the unregularized (`reg_spot = 0`) one
`reconstruct_physical_fields` runs for a forced response.
"""
function eigenmode_modes(ffs::ForceFreeStatesResult, k::Int, psi::Vector{Float64})
    solution = ffs.solution
    equil = ffs.equil
    mats = ffs.mats
    nsol = solution.step
    psi_store = solution.psi_store[1:nsol]
    u1_edge = Matrix(@view solution.u_store[:, :, 1, nsol])
    edge = Vector{ComplexF64}(@view ffs.free_boundary.wt[:, k])
    xi_sol, xi1_sol, _ = sum_eigenmode_contributions(edge, u1_edge, solution, ffs)
    itp_xi = cubic_interp(psi_store, Series(xi_sol); bc=CubicFit(), extrap=ExtendExtrap())
    itp_xi1 = cubic_interp(psi_store, Series(xi1_sol); bc=CubicFit(), extrap=ExtendExtrap())

    nnode = length(psi)
    mpert = ffs.mpert
    N = ffs.numpert_total
    xi_psi = zeros(ComplexF64, nnode, mpert)
    xi_psi1 = zeros(ComplexF64, nnode, mpert)
    xi_s = zeros(ComplexF64, nnode, mpert)
    amat = zeros(ComplexF64, N, N)
    bmat = zeros(ComplexF64, N, N)
    cmat = zeros(ComplexF64, N, N)
    rhs = zeros(ComplexF64, N)
    hint = Ref(1)
    for (i, p) in enumerate(psi)
        xi_psi[i, :] .= itp_xi(p)
        xi_psi1[i, :] .= itp_xi1(p)
        mats.ideal.A_spline(view(amat, :), p; hint=hint)
        mats.ideal.B_spline(view(bmat, :), p; hint=hint)
        mats.ideal.C_spline(view(cmat, :), p; hint=hint)
        mul!(rhs, bmat, view(xi_psi1, i, :))
        mul!(rhs, cmat, view(xi_psi, i, :), 1.0 + 0.0im, 1.0 + 0.0im)
        ldiv!(cholesky!(Hermitian(amat, :L)), rhs)
        xi_s[i, :] .= .-rhs
    end

    chi1 = 2π * equil.psio
    Jb_psi, Jb_theta, Jb_zeta = compute_perturbed_field_modes(xi_psi, xi_psi1, xi_s, psi, equil, ffs)
    Jxi_psi, Jxi_theta, Jxi_zeta, _, _ = compute_contra_displacements(xi_psi, xi_psi1, xi_s ./ chi1, psi, equil, ffs, ffs.metric; reg_spot=0.0)
    _, _, _, b_cov_psi, b_cov_theta, b_cov_zeta = compute_cova_components(Jxi_psi, Jxi_theta, Jxi_zeta, Jb_psi, Jb_theta, Jb_zeta, psi, ffs, ffs.metric)
    return (; xi_psi, xi_psi1, Jxi_psi, Jxi_theta, Jxi_zeta, Jb_psi, Jb_theta, Jb_zeta, b_cov_psi, b_cov_theta, b_cov_zeta)
end

"""
    SurfaceFields(mtheta)

θ-space eigenmode fields on one surface: the inverse transform of one row of
`eigenmode_modes` (same names), `dJxi_theta` = ∂_θ(Jξ^θ), and the normal displacement
`xi_n` = Jξ^ψ/(J|∇ψ|) band-limited to the mode range like the Fortran `xno_mn`. Filled by
`surface_fields!`.
"""
struct SurfaceFields
    xi_psi::Vector{ComplexF64}
    xi_psi1::Vector{ComplexF64}
    Jxi_psi::Vector{ComplexF64}
    Jxi_theta::Vector{ComplexF64}
    Jxi_zeta::Vector{ComplexF64}
    dJxi_theta::Vector{ComplexF64}
    Jb_psi::Vector{ComplexF64}
    Jb_theta::Vector{ComplexF64}
    Jb_zeta::Vector{ComplexF64}
    b_cov_psi::Vector{ComplexF64}
    b_cov_theta::Vector{ComplexF64}
    b_cov_zeta::Vector{ComplexF64}
    xi_n::Vector{ComplexF64}
end

SurfaceFields(mtheta::Int) = SurfaceFields((zeros(ComplexF64, mtheta) for _ in 1:13)...)

"""
    surface_fields!(sf, ft, modes, ipsi, mvals, geom, work_modes) -> sf

Inverse-transform row `ipsi` of every matrix in `modes` onto the θ grid of `ft`, take the
θ-derivative of Jξ^θ spectrally (2πi m per mode, exact for the band-limited field where the
Fortran `gpeq_firstform` fits a periodic spline), and form ξ_n from Jξ^ψ and the geometry
`geom` with the Fortran `gpeq_normal` round trip through mode space. `work_modes` is an
`mpert` scratch vector.
"""
function surface_fields!(sf::SurfaceFields, ft::Utilities.FourierTransforms.FourierTransform, modes::NamedTuple, ipsi::Int,
    mvals::Vector{Int}, geom::SurfaceGeometry, work_modes::Vector{ComplexF64})
    for name in (:xi_psi, :xi_psi1, :Jxi_psi, :Jxi_theta, :Jxi_zeta, :Jb_psi, :Jb_theta, :Jb_zeta, :b_cov_psi, :b_cov_theta, :b_cov_zeta)
        Utilities.FourierTransforms.inverse_transform!(getfield(sf, name), ft, view(getfield(modes, name), ipsi, :))
    end
    for i in eachindex(mvals)
        work_modes[i] = 2π * im * mvals[i] * modes.Jxi_theta[ipsi, i]
    end
    Utilities.FourierTransforms.inverse_transform!(sf.dJxi_theta, ft, work_modes)
    for k in eachindex(geom.jac)
        sf.xi_n[k] = sf.Jxi_psi[k] / (geom.jac[k] * sqrt(geom.delpsi2[k]))
    end
    Utilities.FourierTransforms.transform!(work_modes, ft, sf.xi_n)
    Utilities.FourierTransforms.inverse_transform!(sf.xi_n, ft, work_modes)
    return sf
end

"""
    EffectiveField(mtheta)

θ-space components of the effective field b_eff = b + ξ_n(μ₀ j × n̂) on one surface:
Jacobian-weighted contravariant `Jc_psi/theta/zeta` = J b_eff^i and covariant
`c_cov_psi/theta/zeta` = b_eff,i. Filled by `effective_field!`.
"""
struct EffectiveField
    Jc_psi::Vector{ComplexF64}
    Jc_theta::Vector{ComplexF64}
    Jc_zeta::Vector{ComplexF64}
    c_cov_psi::Vector{ComplexF64}
    c_cov_theta::Vector{ComplexF64}
    c_cov_zeta::Vector{ComplexF64}
end

EffectiveField(mtheta::Int) = EffectiveField((zeros(ComplexF64, mtheta) for _ in 1:6)...)

"""
    effective_field!(eff, geom, equil, psi, sf; hint=Ref(1)) -> eff

Build b_eff = b + ξ_n(μ₀ j × n̂) from the perturbed field and Jξ^ψ of `sf` (Fortran `gpeq_c`,
main.tex Eqs. 148–151), with μ₀j^θ = −F'/J and μ₀j^ζ = −(μ₀p)'/χ' − qF'/J:

    J b_eff^ψ = J b^ψ
    J b_eff^θ = J b^θ + (Jξ^ψ)/(J|∇ψ|²) [ μ₀j^θ g_θζ + μ₀j^ζ g_ζζ ]
    J b_eff^ζ = J b^ζ − (Jξ^ψ)/(J|∇ψ|²) [ μ₀j^θ g_θθ + μ₀j^ζ g_θζ ]
    b_eff,ψ   = b_ψ + (Jξ^ψ)/|∇ψ|² [ μ₀j^θ (∇ψ·∇ζ) − μ₀j^ζ (∇ψ·∇θ) ]
    b_eff,θ   = b_θ + (Jξ^ψ) μ₀j^ζ
    b_eff,ζ   = b_ζ − (Jξ^ψ) μ₀j^θ
"""
function effective_field!(eff::EffectiveField, geom::SurfaceGeometry, equil::Equilibrium.PlasmaEquilibrium, psi::Float64, sf::SurfaceFields;
    hint=Ref(1))
    profiles = equil.profiles
    F1 = profiles.F_deriv(psi; hint=hint)
    mu0p1 = profiles.P_deriv(psi; hint=hint)
    q = profiles.q_spline(psi; hint=hint)
    chi1 = 2π * equil.psio
    for k in eachindex(geom.jac)
        jac = geom.jac[k]
        mu0jt = -F1 / jac
        mu0jz = -mu0p1 / chi1 - F1 * q / jac
        d2 = geom.delpsi2[k]
        xw = sf.Jxi_psi[k]
        eff.Jc_psi[k] = sf.Jb_psi[k]
        eff.Jc_theta[k] = sf.Jb_theta[k] + xw / (d2 * jac) * (mu0jt * geom.g23[k] + mu0jz * geom.g33[k])
        eff.Jc_zeta[k] = sf.Jb_zeta[k] - xw / (d2 * jac) * (mu0jt * geom.g22[k] + mu0jz * geom.g23[k])
        eff.c_cov_psi[k] = sf.b_cov_psi[k] + xw / d2 * (mu0jt * geom.dpdz[k] - mu0jz * geom.dpdt[k])
        eff.c_cov_theta[k] = sf.b_cov_theta[k] + xw * mu0jz
        eff.c_cov_zeta[k] = sf.b_cov_zeta[k] - xw * mu0jt
    end
    return eff
end

"""
    curl_residual!(res, ft, eff, geom, mvals, n, theta_modes, zeta_modes, work_modes, work_fun) -> scale

Evaluate (∇×b_eff)·∇ψ = (∂_θ b_eff,ζ − ∂_ζ b_eff,θ)/J on the θ grid into `res` (Fortran
`gpeq_cveri`): the covariant θ and ζ components are transformed to mode space, differentiated
exactly (∂_θ → 2πi m, ∂_ζ → −2πi n), transformed back and divided by J. Returns
max|∂_θ b_eff,ζ/J|, the size of the cancelling terms, for the relative residual. The four
trailing arguments are scratch buffers (`mpert`, `mpert`, `mpert`, `mtheta`).
"""
function curl_residual!(res::Vector{ComplexF64}, ft::Utilities.FourierTransforms.FourierTransform, eff::EffectiveField, geom::SurfaceGeometry,
    mvals::Vector{Int}, n::Int, theta_modes::Vector{ComplexF64}, zeta_modes::Vector{ComplexF64}, work_modes::Vector{ComplexF64},
    work_fun::Vector{ComplexF64})
    Utilities.FourierTransforms.transform!(theta_modes, ft, eff.c_cov_theta)
    Utilities.FourierTransforms.transform!(zeta_modes, ft, eff.c_cov_zeta)
    for i in eachindex(mvals)
        work_modes[i] = 2π * im * (mvals[i] * zeta_modes[i] + n * theta_modes[i])
    end
    Utilities.FourierTransforms.inverse_transform!(res, ft, work_modes)
    res ./= geom.jac
    for i in eachindex(mvals)
        work_modes[i] = 2π * im * mvals[i] * zeta_modes[i]
    end
    Utilities.FourierTransforms.inverse_transform!(work_fun, ft, work_modes)
    work_fun ./= geom.jac
    return maximum(abs, work_fun)
end

"""
    SurfaceDensities(mtheta)

Per-θ energy densities on one surface (integral = Σ density·J/mtheta, see `surface_integral`),
filled by `surface_densities!`: the effective-field pieces `effective_b_squared`
(= |b_eff|²/μ₀, with its `_psi/_theta/_zeta` contributions), `shear_current`,
`parallel_current_squared`, `pressure_curvature` (= K_i|ξ_n|²), and the standard-form pieces
`b_squared` (= |b|²/μ₀), `current_coupling` (= j·(b×ξ*)), `pressure_compression`
(= (∇·ξ)*(ξ·∇p)).
"""
struct SurfaceDensities
    effective_b_squared::Vector{Float64}
    effective_b_squared_psi::Vector{Float64}
    effective_b_squared_theta::Vector{Float64}
    effective_b_squared_zeta::Vector{Float64}
    shear_current::Vector{Float64}
    parallel_current_squared::Vector{Float64}
    pressure_curvature::Vector{Float64}
    b_squared::Vector{Float64}
    current_coupling::Vector{ComplexF64}
    pressure_compression::Vector{ComplexF64}
end

SurfaceDensities(mtheta::Int) = SurfaceDensities((zeros(Float64, mtheta) for _ in 1:8)..., zeros(ComplexF64, mtheta), zeros(ComplexF64, mtheta))

"""
    surface_densities!(den, geom, kern, sf, eff, equil, psi, n; effective_field_form, standard_form, hint=Ref(1)) -> den

|b_eff|² = Re[(J b_eff^i)* b_eff,i]/J and |b|² = Re[(J b^i)* b_i]/J; the current term is
j·(b×ξ*) = [ −μ₀j^θ (Jb^ψ (Jξ^ζ)* − Jb^ζ (Jξ^ψ)*) + μ₀j^ζ (Jb^ψ (Jξ^θ)* − Jb^θ (Jξ^ψ)*) ]/(μ₀J), and
∇·ξ = ∂_ψξ^ψ + (∂_ψJ/J)ξ^ψ + [∂_θ(Jξ^θ) − 2πi n Jξ^ζ]/J with ξ·∇p = p'ξ^ψ.
"""
function surface_densities!(den::SurfaceDensities, geom::SurfaceGeometry, kern::CurvatureKernel, sf::SurfaceFields, eff::EffectiveField,
    equil::Equilibrium.PlasmaEquilibrium, psi::Float64, n::Int; effective_field_form::Bool, standard_form::Bool, hint=Ref(1))
    profiles = equil.profiles
    F1 = profiles.F_deriv(psi; hint=hint)
    mu0p1 = profiles.P_deriv(psi; hint=hint)
    q = profiles.q_spline(psi; hint=hint)
    p1 = mu0p1 / MU_0
    chi1 = 2π * equil.psio
    for k in eachindex(geom.jac)
        jac = geom.jac[k]
        if effective_field_form
            dpsi = real(conj(eff.Jc_psi[k]) * eff.c_cov_psi[k]) / (MU_0 * jac)
            dthe = real(conj(eff.Jc_theta[k]) * eff.c_cov_theta[k]) / (MU_0 * jac)
            dzet = real(conj(eff.Jc_zeta[k]) * eff.c_cov_zeta[k]) / (MU_0 * jac)
            den.effective_b_squared_psi[k] = dpsi
            den.effective_b_squared_theta[k] = dthe
            den.effective_b_squared_zeta[k] = dzet
            den.effective_b_squared[k] = dpsi + dthe + dzet
            xin2 = abs2(sf.xi_n[k])
            den.shear_current[k] = kern.K1[k] * xin2
            den.parallel_current_squared[k] = kern.K2[k] * xin2
            den.pressure_curvature[k] = kern.K3[k] * xin2
        end
        if standard_form
            mu0jt = -F1 / jac
            mu0jz = -mu0p1 / chi1 - F1 * q / jac
            den.b_squared[k] = real(conj(sf.Jb_psi[k]) * sf.b_cov_psi[k] + conj(sf.Jb_theta[k]) * sf.b_cov_theta[k] + conj(sf.Jb_zeta[k]) * sf.b_cov_zeta[k]) / (MU_0 * jac)
            det_term =
                -mu0jt * (sf.Jb_psi[k] * conj(sf.Jxi_zeta[k]) - sf.Jb_zeta[k] * conj(sf.Jxi_psi[k])) +
                mu0jz * (sf.Jb_psi[k] * conj(sf.Jxi_theta[k]) - sf.Jb_theta[k] * conj(sf.Jxi_psi[k]))
            den.current_coupling[k] = det_term / (MU_0 * jac)
            divxi = sf.xi_psi1[k] + (geom.jac_psi[k] / jac) * sf.xi_psi[k] + sf.dJxi_theta[k] / jac - (2π * im * n) * sf.Jxi_zeta[k] / jac
            den.pressure_compression[k] = conj(divxi) * (p1 * sf.xi_psi[k])
        end
    end
    return den
end

"""
    surface_integral(density, jac) -> Number

∮ J·density dθ over the periodic θ grid as the Riemann sum Σ density·J/mtheta, exact for the
band-limited integrands of the decomposition and identical to the Fortran sums.
"""
function surface_integral(density::AbstractVector, jac::AbstractVector{Float64})
    acc = zero(eltype(density))
    for k in eachindex(density)
        acc += density[k] * jac[k]
    end
    return acc / length(density)
end
