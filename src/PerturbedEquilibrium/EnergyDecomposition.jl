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
