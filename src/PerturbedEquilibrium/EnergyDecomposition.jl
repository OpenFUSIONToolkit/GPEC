"""
Energy decomposition of free-boundary eigenmodes on the (ψ,θ) grid.

Evaluates the ideal-MHD plasma potential energy δW_p of a ForceFreeStates eigenmode in two
forms and splits it into its physical terms (Fortran GPEC `effective_field_flag`/`standard_form_flag`,
routines `gpout_recon`/`gpout_recon2`, theory note `docs/tex/recon/main.tex` there, Bernstein et al. 1958):

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
names under `PerturbedEquilibrium/EnergyDecomposition/`; profiles are `(psi × eigenmode)` matrices,
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

Equilibrium geometry of one flux surface on the periodic θ grid, filled by `surface_geometry!`.

## Fields

  - `R`, `Z` - cylindrical coordinates of the θ points
  - `jac`, `jac_psi` - Jacobian J and its ψ_N-derivative
  - `delpsi2`, `dpdt`, `dpdz` - |∇ψ|², ∇ψ·∇θ and ∇ψ·∇ζ
  - `g11`, `g12`, `g13`, `g22`, `g23`, `g33` - covariant metric of the tangent basis e_i = ∂x/∂q^i
  - `bsq`, `bsq_psi`, `bsq_theta` - metric B² and its ψ_N- and θ-derivatives
  - `mu0_j_theta`, `mu0_j_zeta` - equilibrium current μ₀j^θ = −F'/J and μ₀j^ζ = −(μ₀p)'/χ' − qF'/J
"""
struct SurfaceGeometry
    R::Vector{Float64}
    Z::Vector{Float64}
    jac::Vector{Float64}
    jac_psi::Vector{Float64}
    delpsi2::Vector{Float64}
    dpdt::Vector{Float64}
    dpdz::Vector{Float64}
    g11::Vector{Float64}
    g12::Vector{Float64}
    g13::Vector{Float64}
    g22::Vector{Float64}
    g23::Vector{Float64}
    g33::Vector{Float64}
    bsq::Vector{Float64}
    bsq_psi::Vector{Float64}
    bsq_theta::Vector{Float64}
    mu0_j_theta::Vector{Float64}
    mu0_j_zeta::Vector{Float64}
end

# Per-surface scratch struct with one zeroed length-`mtheta` vector per field.
_zeros_struct(::Type{T}, mtheta::Int) where {T} = T((zeros(eltype(fieldtype(T, i)), mtheta) for i in 1:fieldcount(T))...)

SurfaceGeometry(mtheta::Int) = _zeros_struct(SurfaceGeometry, mtheta)

"""
    surface_geometry!(geom, equil, psi, thetas; bsq, knot=0, hint=(Ref(1), Ref(1))) -> geom

Fill `geom` at flux surface `psi` on the θ points `thetas` from `Equilibrium.flux_surface_basis`,
with B² and its derivatives from the metric interpolant `bsq` (`Equilibrium.metric_bsq_interpolant`).
When `psi` is the equilibrium knot `equil.rzphi_xs[knot]` and `thetas` are the θ knots, the geometry
and B² are read from the values stored at the knots rather than evaluated through the interpolants.
"""
function surface_geometry!(geom::SurfaceGeometry, equil::Equilibrium.PlasmaEquilibrium, psi::Float64, thetas::AbstractVector{Float64};
    bsq::FastInterpolations.CubicInterpolantND, knot::Int=0, hint=(Ref(1), Ref(1)))
    profiles = equil.profiles
    F1 = profiles.F_deriv(psi)
    mu0p1 = profiles.P_deriv(psi)
    q = profiles.q_spline(psi)
    chi1 = 2π * equil.psio
    bsq_knots = bsq.nodal_derivs.partials
    for (k, theta) in enumerate(thetas)
        if knot > 0
            b = Equilibrium.flux_surface_basis(equil, knot, k)
            geom.bsq[k] = bsq_knots[1, knot, k]
            geom.bsq_psi[k] = bsq_knots[2, knot, k]
            geom.bsq_theta[k] = bsq_knots[3, knot, k]
        else
            b = Equilibrium.flux_surface_basis(equil, psi, theta; hint=hint)
            geom.bsq[k] = bsq((psi, theta); hint=hint)
            geom.bsq_psi[k] = bsq((psi, theta); deriv=DerivOp(1, 0), hint=hint)
            geom.bsq_theta[k] = bsq((psi, theta); deriv=DerivOp(0, 1), hint=hint)
        end
        geom.R[k] = b.r
        geom.Z[k] = b.z
        geom.jac[k] = b.jac
        geom.jac_psi[k] = b.jac_psi
        geom.delpsi2[k] = b.delpsi2
        geom.dpdt[k] = b.dpdt
        geom.dpdz[k] = b.dpdz
        geom.g11[k] = b.g11
        geom.g12[k] = b.g12
        geom.g13[k] = b.g13
        geom.g22[k] = b.g22
        geom.g23[k] = b.g23
        geom.g33[k] = b.g33
        geom.mu0_j_theta[k] = -F1 / b.jac
        geom.mu0_j_zeta[k] = -mu0p1 / chi1 - F1 * q / b.jac
    end
    return geom
end

"""
    CurvatureKernel(mtheta)

Per-θ pieces of the destabilizing kernel on one surface, filled by `curvature_kernel!`.

## Fields

  - `shear` - DCON local shear S
  - `curvature` - curvature projection κ_ψ = κ·∇ψ
  - `sigma` - parallel current density ratio σ = j·B/B²
  - `K1`, `K2`, `K3` - K1 = |∇ψ|²σS, K2 = μ₀B²σ², K3 = 2p'κ_ψ
"""
struct CurvatureKernel
    shear::Vector{Float64}
    curvature::Vector{Float64}
    sigma::Vector{Float64}
    K1::Vector{Float64}
    K2::Vector{Float64}
    K3::Vector{Float64}
end

CurvatureKernel(mtheta::Int) = _zeros_struct(CurvatureKernel, mtheta)

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
        # Deviation from Fortran gpeq_curvature/gpeq_K: κ_ψ, σ and K2 use the metric B², not eqfun B², so K2|ξ_n|² matches |V|².
        kern.curvature[k] = (geom.delpsi2[k] / geom.bsq[k]) * (mu0p1 + 0.5 * geom.bsq_psi[k] + 0.5 * geom.bsq_theta[k] * geom.dpdt[k] / geom.delpsi2[k])
        jt = geom.mu0_j_theta[k] / MU_0
        jz = geom.mu0_j_zeta[k] / MU_0
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

# ψ_N gap below which the last equilibrium knot counts as the solution edge.
const _EDGE_KNOT_TOL = 1e-10

"""
    decomposition_grid(ffs) -> (psi, is_knot)

Radial nodes of the decomposition: the equilibrium ψ knots inside the ξ solution domain, where
the metric coefficients, the Euler-Lagrange matrices and the geometry are all exact and the
standard-form integrand equals the DCON Lagrangian density (1e-11 pointwise), plus the solution
edge appended when the last knot falls short of it so the radial integral reaches the plasma edge.
The Fortran `gpout_recon` uses these knots for its profiles but integrates its energy check on
the DCON `psifac` grid. `is_knot` marks the exact
nodes; the (∇×b_eff)·∇ψ residual is an algebraic identity there and is reported on them only.
"""
function decomposition_grid(ffs::ForceFreeStatesResult)
    solution = ffs.solution
    psi_lo = solution.psi_store[1]
    psi_hi = solution.psi_store[solution.step]
    psi = [x for x in ffs.equil.rzphi_xs if psi_lo <= x <= psi_hi]
    is_knot = trues(length(psi))
    if isempty(psi) || psi[end] < psi_hi - _EDGE_KNOT_TOL
        push!(psi, psi_hi)
        push!(is_knot, false)
    end
    return psi, is_knot
end

"""
    node_solution(ffs, psi) -> (u1, du1)

Euler-Lagrange basis Ξ_ψ and dΞ_ψ/dψ at the radial nodes `psi`, `(numpert_total × numpert_total ×
length(psi))` in the basis of the stored solution. Each node state comes from
`ForceFreeStates.solution_state_at`, so ξ is the ODE's own value there rather than a spline through
the saved steps, whose spacing follows the integrator's tolerance control.
"""
function node_solution(ffs::ForceFreeStatesResult, psi::Vector{Float64})
    npert = ffs.numpert_total
    u1 = zeros(ComplexF64, npert, npert, length(psi))
    du1 = zeros(ComplexF64, npert, npert, length(psi))
    du = zeros(ComplexF64, npert, npert, 2)
    hints = (Ref(1), Ref(1))
    for (i, p) in enumerate(psi)
        u = ForceFreeStates.solution_state_at(ffs, p)
        ForceFreeStates.el_derivatives!(du, u, false, ffs.equil, ffs.mats, ffs, p, hints...)
        u1[:, :, i] .= @view u[:, :, 1]
        du1[:, :, i] .= @view du[:, :, 1]
    end
    return u1, du1
end

"""
    eigenmode_modes(ffs, k, psi, u1, du1) -> NamedTuple

Mode-space displacement and perturbed field of free-boundary eigenmode `k` on the radial nodes
`psi`, `(length(psi) × mpert)` matrices named after the gpeq arrays: `xi_psi` = ξ^ψ,
`xi_psi1` = ∂ξ^ψ/∂ψ, `Jxi_psi/theta/zeta` = Jξ^i, `Jb_psi/theta/zeta` = Jb^i. The edge boundary
condition is the eigenvector `wt[:, k]` (unit power norm), which fixes the combination of the node
basis `u1`, `du1` from `node_solution`. ξ_s = −A⁻¹(Bξ' + Cξ) is recomputed at every node
from the ideal Euler-Lagrange matrices there (`ForceFreeStates.compute_node_xi_s!`, Fortran
`gpeq_sol`), which keeps the fields algebraically consistent with the metric at the equilibrium
knots. The rest of the chain is the unregularized one `reconstruct_physical_fields` runs for a
forced response.
"""
function eigenmode_modes(ffs::ForceFreeStatesResult, k::Int, psi::Vector{Float64}, u1::Array{ComplexF64,3}, du1::Array{ComplexF64,3})
    solution = ffs.solution
    equil = ffs.equil
    u1_edge = @view solution.u_store[:, :, 1, solution.step]
    alpha = u1_edge \ Vector{ComplexF64}(@view ffs.free_boundary.wt[:, k])

    nnode = length(psi)
    mpert = ffs.mpert
    xi_psi = zeros(ComplexF64, nnode, mpert)
    xi_psi1 = zeros(ComplexF64, nnode, mpert)
    xi_s = zeros(ComplexF64, nnode, mpert)
    hint = Ref(1)
    for (i, p) in enumerate(psi)
        mul!(view(xi_psi, i, :), view(u1, :, :, i), alpha)
        mul!(view(xi_psi1, i, :), view(du1, :, :, i), alpha)
        ForceFreeStates.compute_node_xi_s!(view(xi_s, i, :), view(xi_psi1, i, :), view(xi_psi, i, :), ffs.mats, p; hint=hint)
    end

    chi1 = 2π * equil.psio
    Jb_psi, Jb_theta, Jb_zeta = compute_perturbed_field_modes(xi_psi, xi_psi1, xi_s, psi, equil, ffs)
    # Deviation from Fortran gpeq_sol/gpeq_contra: always unregularized (reg_spot = 0), the Fortran regularizes when reg_flag is set.
    Jxi_psi, Jxi_theta, Jxi_zeta, _, _ = compute_contra_displacements(xi_psi, xi_psi1, xi_s ./ chi1, psi, equil, ffs, ffs.metric; reg_spot=0.0)
    return (; xi_psi, xi_psi1, Jxi_psi, Jxi_theta, Jxi_zeta, Jb_psi, Jb_theta, Jb_zeta)
end

"""
    SurfaceFields(mtheta)

θ-space eigenmode fields on one surface, filled by `surface_fields!`.

## Fields

  - `xi_psi`, `xi_psi1` - ξ^ψ and ∂ξ^ψ/∂ψ
  - `Jxi_psi`, `Jxi_theta`, `Jxi_zeta` - Jacobian-weighted contravariant displacement Jξ^i
  - `dJxi_theta` - ∂_θ(Jξ^θ)
  - `Jb_psi`, `Jb_theta`, `Jb_zeta` - Jacobian-weighted contravariant perturbed field Jb^i
  - `b_cov_psi`, `b_cov_theta`, `b_cov_zeta` - covariant perturbed field b_i = g_ij Jb^j/J
  - `xi_n` - normal displacement ξ_n = Jξ^ψ/(J|∇ψ|)
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

SurfaceFields(mtheta::Int) = _zeros_struct(SurfaceFields, mtheta)

"""
    surface_fields!(sf, ft, modes, ipsi, mvals, geom, work_modes) -> sf

Inverse-transform row `ipsi` of every matrix in `modes` onto the θ grid of `ft`, take the
θ-derivative of Jξ^θ spectrally, lower the field with the local metric of `geom`
(b_i = g_ij Jb^j/J), and form Jξ^ψ = J·ξ^ψ and ξ_n = ξ^ψ/|∇ψ| pointwise. The pointwise products
keep every θ harmonic, so |V|² and K₂|ξ_n|² cancel exactly in the effective-field form; truncating
them to the mode range leaves a ψ-resolution-independent, mpert-dependent deficit. Jξ^θ and Jξ^ζ
still come from the mode-truncated `jmat` convolution of `compute_contra_displacements`.
`work_modes` is an `mpert` scratch vector.
"""
function surface_fields!(sf::SurfaceFields, ft::Utilities.FourierTransforms.FourierTransform, modes::NamedTuple, ipsi::Int,
    mvals::Vector{Int}, geom::SurfaceGeometry, work_modes::Vector{ComplexF64})
    for name in (:xi_psi, :xi_psi1, :Jxi_theta, :Jxi_zeta, :Jb_psi, :Jb_theta, :Jb_zeta)
        Utilities.FourierTransforms.inverse_transform!(getfield(sf, name), ft, view(getfield(modes, name), ipsi, :))
    end
    # Deviation from Fortran gpeq_firstform: ∂_θ(Jξ^θ) is spectral (2πi m per mode), not a periodic-spline derivative.
    for i in eachindex(mvals)
        work_modes[i] = 2π * im * mvals[i] * modes.Jxi_theta[ipsi, i]
    end
    Utilities.FourierTransforms.inverse_transform!(sf.dJxi_theta, ft, work_modes)
    # Deviation from Fortran gpeq_contra/gpeq_cova/gpeq_normal: Jξ^ψ, b_i and ξ_n are pointwise, not band-truncated.
    for k in eachindex(geom.jac)
        jac = geom.jac[k]
        sf.Jxi_psi[k] = jac * sf.xi_psi[k]
        sf.b_cov_psi[k] = (geom.g11[k] * sf.Jb_psi[k] + geom.g12[k] * sf.Jb_theta[k] + geom.g13[k] * sf.Jb_zeta[k]) / jac
        sf.b_cov_theta[k] = (geom.g12[k] * sf.Jb_psi[k] + geom.g22[k] * sf.Jb_theta[k] + geom.g23[k] * sf.Jb_zeta[k]) / jac
        sf.b_cov_zeta[k] = (geom.g13[k] * sf.Jb_psi[k] + geom.g23[k] * sf.Jb_theta[k] + geom.g33[k] * sf.Jb_zeta[k]) / jac
        sf.xi_n[k] = sf.Jxi_psi[k] / (jac * sqrt(geom.delpsi2[k]))
    end
    return sf
end

"""
    EffectiveField(mtheta)

θ-space components of the effective field b_eff = b + ξ_n(μ₀ j × n̂) on one surface, filled by
`effective_field!`.

## Fields

  - `Jc_psi`, `Jc_theta`, `Jc_zeta` - Jacobian-weighted contravariant components J b_eff^i
  - `c_cov_psi`, `c_cov_theta`, `c_cov_zeta` - covariant components b_eff,i
"""
struct EffectiveField
    Jc_psi::Vector{ComplexF64}
    Jc_theta::Vector{ComplexF64}
    Jc_zeta::Vector{ComplexF64}
    c_cov_psi::Vector{ComplexF64}
    c_cov_theta::Vector{ComplexF64}
    c_cov_zeta::Vector{ComplexF64}
end

EffectiveField(mtheta::Int) = _zeros_struct(EffectiveField, mtheta)

"""
    effective_field!(eff, geom, sf) -> eff

Build b_eff = b + ξ_n(μ₀ j × n̂) from the perturbed field and Jξ^ψ of `sf` and the equilibrium
current of `geom` (Fortran `gpeq_c`, main.tex Eqs. 148–151):

    J b_eff^ψ = J b^ψ
    J b_eff^θ = J b^θ + (Jξ^ψ)/(J|∇ψ|²) [ μ₀j^θ g_θζ + μ₀j^ζ g_ζζ ]
    J b_eff^ζ = J b^ζ − (Jξ^ψ)/(J|∇ψ|²) [ μ₀j^θ g_θθ + μ₀j^ζ g_θζ ]
    b_eff,ψ   = b_ψ + (Jξ^ψ)/|∇ψ|² [ μ₀j^θ (∇ψ·∇ζ) − μ₀j^ζ (∇ψ·∇θ) ]
    b_eff,θ   = b_θ + (Jξ^ψ) μ₀j^ζ
    b_eff,ζ   = b_ζ − (Jξ^ψ) μ₀j^θ
"""
function effective_field!(eff::EffectiveField, geom::SurfaceGeometry, sf::SurfaceFields)
    # Deviation from Fortran gpeq_c: b_eff^i and b_eff,i are formed pointwise in θ, not band-truncated.
    for k in eachindex(geom.jac)
        jac = geom.jac[k]
        mu0jt = geom.mu0_j_theta[k]
        mu0jz = geom.mu0_j_zeta[k]
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
filled by `surface_densities!`. Field names match the profiles of `EnergyDecompositionResult`.

## Fields

  - `effective_b_squared` - |b_eff|²/μ₀
  - `effective_b_squared_psi`, `effective_b_squared_theta`, `effective_b_squared_zeta` - its component contributions
  - `shear_current`, `parallel_current_squared`, `pressure_curvature` - K1|ξ_n|², K2|ξ_n|², K3|ξ_n|²
  - `b_squared` - standard form |b|²/μ₀
  - `current_coupling` - standard form j·(b×ξ*)
  - `pressure_compression` - standard form (∇·ξ)*(ξ·∇p)
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

SurfaceDensities(mtheta::Int) = _zeros_struct(SurfaceDensities, mtheta)

# Per-surface profiles of each form; the names are shared by SurfaceDensities and EnergyDecompositionResult.
const _ED_EFFECTIVE_PROFILES = (:effective_b_squared, :effective_b_squared_psi, :effective_b_squared_theta, :effective_b_squared_zeta,
    :shear_current, :parallel_current_squared, :pressure_curvature)
const _ED_STANDARD_PROFILES = (:b_squared, :current_coupling, :pressure_compression)

# Result field holding the per-(ψ,θ) density of a profile.
_density_field(name::Symbol) = Symbol(name, :_density)

"""
    surface_densities!(den, geom, kern, sf, eff, equil, psi, n; effective_field_form, standard_form, hint=Ref(1)) -> den

|b_eff|² = Re[(J b_eff^i)* b_eff,i]/J and |b|² = Re[(J b^i)* b_i]/J; the current term is
j·(b×ξ*) = [ −μ₀j^θ (Jb^ψ (Jξ^ζ)* − Jb^ζ (Jξ^ψ)*) + μ₀j^ζ (Jb^ψ (Jξ^θ)* − Jb^θ (Jξ^ψ)*) ]/(μ₀J), and
∇·ξ = ∂_ψξ^ψ + (∂_ψJ/J)ξ^ψ + [∂_θ(Jξ^θ) − 2πi n Jξ^ζ]/J with ξ·∇p = p'ξ^ψ.
"""
function surface_densities!(den::SurfaceDensities, geom::SurfaceGeometry, kern::CurvatureKernel, sf::SurfaceFields, eff::EffectiveField,
    equil::Equilibrium.PlasmaEquilibrium, psi::Float64, n::Int; effective_field_form::Bool, standard_form::Bool, hint=Ref(1))
    p1 = equil.profiles.P_deriv(psi; hint=hint) / MU_0
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
            mu0jt = geom.mu0_j_theta[k]
            mu0jz = geom.mu0_j_zeta[k]
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

# Relative (∇×b_eff)·∇ψ residual above which the effective field is reported as inconsistent.
const _CURL_RESIDUAL_WARN_TOL = 1e-8
# Relative mismatch between a reconstructed δW_p and eigenmode_plasma_energies that triggers a warning.
const _ENERGY_MISMATCH_WARN_TOL = 0.05

# Scale from the Bernstein integrals (2μ₀δW form) to the normalization of eigenmode_plasma_energies.
_energy_scale(equil::Equilibrium.PlasmaEquilibrium) = 2 * MU_0 / equil.psio^2

# Cubic-spline integral over ψ with the running value at every node (Fortran recon_int = "spline").
# Repeated ψ nodes are collapsed for the fit and the running value is carried across them.
function _radial_integral(psi::Vector{Float64}, y::AbstractVector{<:Real})
    keep = trues(length(psi))
    for i in 2:length(psi)
        keep[i] = psi[i] > psi[i-1]
    end
    itp = cubic_interp(psi[keep], Float64.(y[keep]); bc=CubicFit())
    total = FastInterpolations.integrate(itp)
    cum_keep = vec(FastInterpolations.cumulative_integrate(itp))
    cum = Vector{Float64}(undef, length(psi))
    j = 0
    for i in eachindex(psi)
        j += keep[i]
        cum[i] = cum_keep[j]
    end
    return total, cum
end

function _radial_integral(psi::Vector{Float64}, y::AbstractVector{<:Complex})
    tr, cr = _radial_integral(psi, real.(y))
    ti, ci = _radial_integral(psi, imag.(y))
    return complex(tr, ti), complex.(cr, ci)
end

# Replace result field `f` by zeros of its element type and shape `dims`.
_zero_field!(res::EnergyDecompositionResult, f::Symbol, dims::Int...) = setfield!(res, f, zeros(eltype(fieldtype(EnergyDecompositionResult, f)), dims...))

# Allocate the result arrays of the enabled forms; disabled forms keep their empty defaults.
function _allocate_result!(res::EnergyDecompositionResult, npsi::Int, mtheta::Int, nk::Int, thetas::Vector{Float64},
    effective_field_form::Bool, standard_form::Bool, write_densities::Bool)
    res.dW_plasma_reference = zeros(nk)
    if effective_field_form
        for f in (_ED_EFFECTIVE_PROFILES..., :dW_cumulative)
            _zero_field!(res, f, npsi, nk)
        end
        for f in (:dW_plasma, :dW_plasma_relative_error, :effective_b_curl_residual_max, :effective_b_curl_residual_rms, :effective_b_curl_residual_relative)
            _zero_field!(res, f, nk)
        end
    end
    if standard_form
        for f in (_ED_STANDARD_PROFILES..., :dW_cumulative_standard_form)
            _zero_field!(res, f, npsi, nk)
        end
        for f in (:dW_plasma_standard_form, :dW_plasma_standard_form_relative_error)
            _zero_field!(res, f, nk)
        end
    end
    if write_densities
        res.theta = copy(thetas)
        res.R = zeros(npsi, mtheta)
        res.Z = zeros(npsi, mtheta)
        effective_field_form && _zero_field!(res, :effective_b_curl_residual_density, npsi, mtheta, nk)
        for f in _enabled_profiles(effective_field_form, standard_form)
            _zero_field!(res, _density_field(f), npsi, mtheta, nk)
        end
    end
    return res
end

_enabled_profiles(effective_field_form::Bool, standard_form::Bool) =
    ((effective_field_form ? _ED_EFFECTIVE_PROFILES : ())..., (standard_form ? _ED_STANDARD_PROFILES : ())...)

"""
    decompose_energy(ffs; eigenmodes=[1], effective_field_form=true, standard_form=true, write_densities=false, verbose=ffs.control.verbose)
        -> EnergyDecompositionResult

Decompose the plasma potential energy of the free-boundary eigenmodes `eigenmodes` of a forward
ForceFreeStates solve on the (ψ,θ) grid (Fortran `gpout_recon` and `gpout_recon2`). For each
eigenmode the unregularized gpeq chain gives ξ and b in mode space on the radial nodes of
`decomposition_grid`; every node is transformed to θ space, the effective field
b_eff = b + ξ_n(μ₀ j × n̂) and the curvature kernel K are built, and the per-surface integrals of
the enabled forms are integrated in ψ with a cubic spline. All energies are scaled by 2μ₀/ψ₀² so
`dW_plasma` reads in the normalization of `eigenmode_plasma_energies`, whose entry `ep[k]` is
stored as `dW_plasma_reference`. Two checks run every time: the relative (∇×b_eff)·∇ψ residual
over the equilibrium knots and the relative mismatch of each total against the reference; both
warn above their tolerance and are stored as data.

It covers the free-boundary DCON eigenmodes only: the edge condition is the eigenvector
`wt[:, k]` and the reference is `ep[k]`, which fixed-boundary or forced-response perturbations
do not supply. Requires the dense ξ solution (`integrator = "forward"`), the free-boundary
eigenmodes (`vac_flag = true`), an ideal solve, a single toroidal mode number, and at least one
enabled form; anything else throws an `ArgumentError`.
"""
function decompose_energy(ffs::ForceFreeStatesResult; eigenmodes::Vector{Int}=[1], effective_field_form::Bool=true, standard_form::Bool=true,
    write_densities::Bool=false, verbose::Bool=ffs.control.verbose)
    (effective_field_form || standard_form) || throw(ArgumentError("EnergyDecomposition: enable at least one of effective_field_form and standard_form"))
    ffs.solution === nothing && throw(ArgumentError("EnergyDecomposition needs the dense ξ solution of a forward run (integrator = \"forward\")"))
    ffs.free_boundary === nothing && throw(ArgumentError("EnergyDecomposition needs the free-boundary eigenmodes (vac_flag = true)"))
    ffs.nlow == ffs.nhigh || throw(ArgumentError("EnergyDecomposition handles a single toroidal mode number (nn_low == nn_high)"))
    ffs.mats.kinetic === nothing || throw(ArgumentError("EnergyDecomposition decomposes ideal eigenmodes only (kinetic_factor = 0)"))
    free = ffs.free_boundary
    equil = ffs.equil
    nmodes = size(free.wt, 2)
    for k in eigenmodes
        1 <= k <= nmodes || throw(ArgumentError("EnergyDecomposition: eigenmode $k is outside 1:$nmodes"))
    end

    # Deviation from Fortran gpout_recon: the totals integrate over the equilibrium knots, the Fortran ones over the ODE psifac grid.
    psi, is_knot = decomposition_grid(ffs)
    knots = [on_knot ? searchsortedfirst(equil.rzphi_xs, p) : 0 for (p, on_knot) in zip(psi, is_knot)]
    npsi = length(psi)
    thetas_ext = collect(equil.rzphi_ys)
    mtheta = length(thetas_ext) - 1
    thetas = thetas_ext[1:mtheta]
    mpert = ffs.mpert
    mvals = collect(ffs.mlow:ffs.mhigh)
    nn = ffs.nlow
    scale = _energy_scale(equil)
    nk = length(eigenmodes)
    profiles = _enabled_profiles(effective_field_form, standard_form)

    res = EnergyDecompositionResult(; psi=copy(psi), eigenmode_index=copy(eigenmodes))
    _allocate_result!(res, npsi, mtheta, nk, thetas, effective_field_form, standard_form, write_densities)

    # Per-thread scratch; the ψ loops are threaded :static so threadid() indexes it safely.
    ft = Utilities.FourierTransforms.FourierTransform(mtheta, mpert, ffs.mlow)
    nt = Threads.maxthreadid()
    sfs = [SurfaceFields(mtheta) for _ in 1:nt]
    effs = [EffectiveField(mtheta) for _ in 1:nt]
    dens = [SurfaceDensities(mtheta) for _ in 1:nt]
    curls = [zeros(ComplexF64, mtheta) for _ in 1:nt]
    mode_bufs = [(zeros(ComplexF64, mpert), zeros(ComplexF64, mpert), zeros(ComplexF64, mpert)) for _ in 1:nt]
    fun_bufs = [zeros(ComplexF64, mtheta) for _ in 1:nt]
    hints2d = [(Ref(1), Ref(1)) for _ in 1:nt]
    hints1d = [Ref(1) for _ in 1:nt]

    # Equilibrium geometry and curvature kernel of every surface, shared by all eigenmodes.
    bsq = Equilibrium.metric_bsq_interpolant(equil)
    geoms = [SurfaceGeometry(mtheta) for _ in 1:npsi]
    kerns = [CurvatureKernel(mtheta) for _ in 1:npsi]
    Threads.@threads :static for ipsi in 1:npsi
        tid = Threads.threadid()
        surface_geometry!(geoms[ipsi], equil, psi[ipsi], thetas; bsq, knot=knots[ipsi], hint=hints2d[tid])
        effective_field_form && curvature_kernel!(kerns[ipsi], geoms[ipsi], equil, psi[ipsi], thetas_ext; hint=hints1d[tid])
    end
    if write_densities
        for ipsi in 1:npsi
            res.R[ipsi, :] .= geoms[ipsi].R
            res.Z[ipsi, :] .= geoms[ipsi].Z
        end
    end

    # Per-surface (∇×b_eff)·∇ψ statistics of the current eigenmode, filled by the threaded loop.
    curl_max = zeros(npsi)
    curl_sumsq = zeros(npsi)
    curl_scale = zeros(npsi)

    # The θ transforms are small BLAS matvecs; one BLAS thread per Julia thread avoids oversubscription.
    blas_threads = BLAS.get_num_threads()
    BLAS.set_num_threads(1)
    try
        u1, du1 = node_solution(ffs, psi)
        for (ik, k) in enumerate(eigenmodes)
            modes = eigenmode_modes(ffs, k, psi, u1, du1)
            Threads.@threads :static for ipsi in 1:npsi
                tid = Threads.threadid()
                geom = geoms[ipsi]
                sf = sfs[tid]
                eff = effs[tid]
                den = dens[tid]
                theta_modes, zeta_modes, work_modes = mode_bufs[tid]
                surface_fields!(sf, ft, modes, ipsi, mvals, geom, work_modes)
                if effective_field_form
                    effective_field!(eff, geom, sf)
                    curl_scale[ipsi] = curl_residual!(curls[tid], ft, eff, geom, mvals, nn, theta_modes, zeta_modes, work_modes, fun_bufs[tid])
                    curl_max[ipsi] = maximum(abs, curls[tid])
                    curl_sumsq[ipsi] = sum(abs2, curls[tid])
                end
                surface_densities!(den, geom, kerns[ipsi], sf, eff, equil, psi[ipsi], nn; effective_field_form, standard_form, hint=hints1d[tid])
                for name in profiles
                    getfield(res, name)[ipsi, ik] = scale * surface_integral(getfield(den, name), geom.jac)
                end
                if write_densities
                    for name in profiles
                        getfield(res, _density_field(name))[ipsi, :, ik] .= scale .* getfield(den, name)
                    end
                    effective_field_form && (res.effective_b_curl_residual_density[ipsi, :, ik] .= curls[tid])
                end
            end

            reference = real(free.ep[k])
            res.dW_plasma_reference[ik] = reference
            if effective_field_form
                tot_eb, cum_eb = _radial_integral(psi, @view res.effective_b_squared[:, ik])
                tot_k1, cum_k1 = _radial_integral(psi, @view res.shear_current[:, ik])
                tot_k2, cum_k2 = _radial_integral(psi, @view res.parallel_current_squared[:, ik])
                tot_k3, cum_k3 = _radial_integral(psi, @view res.pressure_curvature[:, ik])
                res.dW_cumulative[:, ik] .= 0.5 .* (cum_eb .- cum_k1 .- cum_k2 .- cum_k3)
                res.dW_plasma[ik] = 0.5 * (tot_eb - tot_k1 - tot_k2 - tot_k3)
                res.dW_plasma_relative_error[ik] = (res.dW_plasma[ik] - reference) / abs(reference)
                res.effective_b_curl_residual_max[ik] = maximum(curl_max[is_knot])
                res.effective_b_curl_residual_rms[ik] = sqrt(sum(curl_sumsq[is_knot]) / (count(is_knot) * mtheta))
                res.effective_b_curl_residual_relative[ik] = res.effective_b_curl_residual_max[ik] / maximum(curl_scale[is_knot])
            end
            if standard_form
                tot_b2, cum_b2 = _radial_integral(psi, @view res.b_squared[:, ik])
                tot_cc, cum_cc = _radial_integral(psi, @view res.current_coupling[:, ik])
                tot_pc, cum_pc = _radial_integral(psi, @view res.pressure_compression[:, ik])
                res.dW_cumulative_standard_form[:, ik] .= 0.5 .* (cum_b2 .- cum_cc .+ cum_pc)
                res.dW_plasma_standard_form[ik] = 0.5 * (tot_b2 - tot_cc + tot_pc)
                res.dW_plasma_standard_form_relative_error[ik] = (real(res.dW_plasma_standard_form[ik]) - reference) / abs(reference)
            end

            verbose && print_energy_decomposition_summary(res, ik)
            if effective_field_form
                rel = res.effective_b_curl_residual_relative[ik]
                rel > _CURL_RESIDUAL_WARN_TOL &&
                    @warn "EnergyDecomposition: (∇×b_eff)·∇ψ of eigenmode $k does not vanish (relative residual $(@sprintf("%.2e", rel)) > $(_CURL_RESIDUAL_WARN_TOL))"
                err = res.dW_plasma_relative_error[ik]
                abs(err) > _ENERGY_MISMATCH_WARN_TOL &&
                    @warn "EnergyDecomposition: effective-field δW_p of eigenmode $k differs from eigenmode_plasma_energies by $(@sprintf("%.2f", 100 * err)) %"
            end
            if standard_form
                err = res.dW_plasma_standard_form_relative_error[ik]
                abs(err) > _ENERGY_MISMATCH_WARN_TOL &&
                    @warn "EnergyDecomposition: standard-form δW_p of eigenmode $k differs from eigenmode_plasma_energies by $(@sprintf("%.2f", 100 * err)) %"
            end
        end
    finally
        BLAS.set_num_threads(blas_threads)
    end
    return res
end

# Total of a stored profile over ψ (the profiles are already scaled).
_profile_total(res::EnergyDecompositionResult, profile::AbstractMatrix, ik::Int) = first(_radial_integral(res.psi, @view profile[:, ik]))

"""
    print_energy_decomposition_summary(res, ik)

Log the totals of decomposed eigenmode number `ik` of `res` and the two consistency checks.
"""
function print_energy_decomposition_summary(res::EnergyDecompositionResult, ik::Int)
    k = res.eigenmode_index[ik]
    lines = ["Energy decomposition of free-boundary eigenmode $k (power-normalized, per unit ⟨|ξ|²⟩)"]
    if !isempty(res.dW_plasma)
        eb = _profile_total(res, res.effective_b_squared, ik)
        kk = _profile_total(res, res.shear_current, ik) + _profile_total(res, res.parallel_current_squared, ik) + _profile_total(res, res.pressure_curvature, ik)
        push!(lines, @sprintf("  |b_eff|²/μ₀ = %+.4e   K|ξ_n|² = %+.4e   δW_p = %+.4e", eb, kk, res.dW_plasma[ik]))
        push!(
            lines,
            @sprintf("  shear = %+.4e   parallel = %+.4e   curvature = %+.4e",
                _profile_total(res, res.shear_current, ik), _profile_total(res, res.parallel_current_squared, ik), _profile_total(res, res.pressure_curvature, ik))
        )
    end
    if !isempty(res.dW_plasma_standard_form)
        push!(
            lines,
            @sprintf("  standard form: |b|²/μ₀ = %+.4e   current coupling = %+.4e   pressure = %+.4e   δW_p = %+.4e",
                _profile_total(res, res.b_squared, ik), real(_profile_total(res, res.current_coupling, ik)),
                real(_profile_total(res, res.pressure_compression, ik)), real(res.dW_plasma_standard_form[ik]))
        )
    end
    ref_line = @sprintf("  ForceFreeStates ep[%d] = %+.4e", k, res.dW_plasma_reference[ik])
    isempty(res.dW_plasma) || (ref_line *= @sprintf("   relative difference = %.2e", res.dW_plasma_relative_error[ik]))
    isempty(res.dW_plasma_standard_form) || (ref_line *= @sprintf("   (standard form %.2e)", res.dW_plasma_standard_form_relative_error[ik]))
    push!(lines, ref_line)
    if !isempty(res.effective_b_curl_residual_max)
        push!(
            lines,
            @sprintf("  (∇×b_eff)·∇ψ residual: max %.2e, rms %.2e, relative %.2e",
                res.effective_b_curl_residual_max[ik], res.effective_b_curl_residual_rms[ik], res.effective_b_curl_residual_relative[ik])
        )
    end
    @info join(lines, "\n")
    return nothing
end

# Datasets of the group, in write order; names are the struct fields.
const _ED_MAIN_FIELDS = (:psi, :eigenmode_index, :effective_b_squared, :effective_b_squared_psi, :effective_b_squared_theta,
    :effective_b_squared_zeta, :shear_current, :parallel_current_squared, :pressure_curvature, :b_squared, :current_coupling,
    :pressure_compression, :dW_cumulative, :dW_cumulative_standard_form, :dW_plasma, :dW_plasma_standard_form, :dW_plasma_reference,
    :dW_plasma_relative_error, :dW_plasma_standard_form_relative_error, :effective_b_curl_residual_max, :effective_b_curl_residual_rms,
    :effective_b_curl_residual_relative)
const _ED_DENSITY_FIELDS = (map(_density_field, _ED_EFFECTIVE_PROFILES)..., :effective_b_curl_residual_density, map(_density_field, _ED_STANDARD_PROFILES)...)

const _ED_NORM = "power-normalized like eigenmode_plasma_energies (per unit ⟨|ξ|²⟩, scaled by 2μ₀/ψ₀²)"
const _ED_TOTAL = (; dims=("eigenmode",), attach=(1 => "eigenmode_index",))
const _ED_PROFILE = (; dims=("psi", "eigenmode"), attach=(1 => "psi", 2 => "eigenmode_index"))
const _ED_DENSITY = (; dims=("psi", "theta", "eigenmode"), attach=(1 => "psi", 2 => "Densities/theta", 3 => "eigenmode_index"))

# Metadata table for PerturbedEquilibrium/EnergyDecomposition/ (paths relative to the group).
const ENERGY_DECOMPOSITION_H5_ANNOTATIONS = [
    "psi" => (; long_name="normalized poloidal flux ψ_N of the decomposition nodes (equilibrium knots plus the solution edge)", scale="psi"),
    "eigenmode_index" => (; long_name="free-boundary eigenmode index (1 = least stable) of each decomposed mode, the eigenmode coordinate", scale="eigenmode"),
    "effective_b_squared" => (; long_name="∮J|b_eff|²/μ₀ dθ with b_eff = b + ξ_n μ₀ j×n̂, $_ED_NORM", _ED_PROFILE...),
    "effective_b_squared_psi" => (; long_name="ψ-component contribution Re[(J b_eff^ψ)* b_eff,ψ]/(μ₀J) to effective_b_squared", _ED_PROFILE...),
    "effective_b_squared_theta" => (; long_name="θ-component contribution Re[(J b_eff^θ)* b_eff,θ]/(μ₀J) to effective_b_squared", _ED_PROFILE...),
    "effective_b_squared_zeta" => (; long_name="ζ-component contribution Re[(J b_eff^ζ)* b_eff,ζ]/(μ₀J) to effective_b_squared", _ED_PROFILE...),
    "shear_current" => (; long_name="∮J K1|ξ_n|² dθ with K1 = |∇ψ|²σS (shear-current coupling), $_ED_NORM", _ED_PROFILE...),
    "parallel_current_squared" => (; long_name="∮J K2|ξ_n|² dθ with K2 = μ₀B²σ² (parallel current squared), $_ED_NORM", _ED_PROFILE...),
    "pressure_curvature" => (; long_name="∮J K3|ξ_n|² dθ with K3 = 2p'κ_ψ (pressure-curvature drive), $_ED_NORM", _ED_PROFILE...),
    "b_squared" => (; long_name="standard form: ∮J|b|²/μ₀ dθ, $_ED_NORM", _ED_PROFILE...),
    "current_coupling" => (; long_name="standard form: ∮J j·(b×ξ*) dθ (complex per surface; its ψ-integral is real to roundoff), $_ED_NORM", _ED_PROFILE...),
    "pressure_compression" => (; long_name="standard form: ∮J (∇·ξ)*(ξ·∇p) dθ (complex per surface; its ψ-integral is real to roundoff), $_ED_NORM", _ED_PROFILE...),
    "dW_cumulative" => (; long_name="running ½∫dψ [effective_b_squared − shear_current − parallel_current_squared − pressure_curvature]", _ED_PROFILE...),
    "dW_cumulative_standard_form" => (; long_name="running ½∫dψ [b_squared − current_coupling + pressure_compression]", _ED_PROFILE...),
    "dW_plasma" => (; long_name="plasma potential energy δW_p in the effective-field form, $_ED_NORM", _ED_TOTAL...),
    "dW_plasma_standard_form" => (; long_name="plasma potential energy δW_p in the standard form (imaginary part is roundoff), $_ED_NORM", _ED_TOTAL...),
    "dW_plasma_reference" => (; long_name="eigenmode_plasma_energies entry of the same eigenmode, the reference for both forms", _ED_TOTAL...),
    "dW_plasma_relative_error" => (; long_name="(dW_plasma − dW_plasma_reference)/|dW_plasma_reference|", _ED_TOTAL...),
    "dW_plasma_standard_form_relative_error" => (; long_name="(Re dW_plasma_standard_form − dW_plasma_reference)/|dW_plasma_reference|", _ED_TOTAL...),
    "effective_b_curl_residual_max" =>
        (; long_name="max over the equilibrium knots and θ of |(∇×b_eff)·∇ψ|, which vanishes for a consistent effective field", _ED_TOTAL...),
    "effective_b_curl_residual_rms" => (; long_name="rms over the equilibrium knots and θ of |(∇×b_eff)·∇ψ|", _ED_TOTAL...),
    "effective_b_curl_residual_relative" => (; long_name="effective_b_curl_residual_max divided by max|∂_θ b_eff,ζ/J|, the size of the cancelling terms", _ED_TOTAL...),
    "Densities/theta" => (; long_name="poloidal angle θ ∈ [0,1) of the density grid", scale="theta"),
    "Densities/R" => (; long_name="major radius of the (ψ,θ) grid points", units="m", dims=("psi", "theta"), attach=(1 => "psi", 2 => "Densities/theta")),
    "Densities/Z" => (; long_name="vertical position of the (ψ,θ) grid points", units="m", dims=("psi", "theta"), attach=(1 => "psi", 2 => "Densities/theta")),
    "Densities/effective_b_squared_density" => (; long_name="|b_eff|²/μ₀ per point (profile = Σ density·J/mtheta), $_ED_NORM", _ED_DENSITY...),
    "Densities/effective_b_squared_psi_density" => (; long_name="ψ-component of |b_eff|²/μ₀ per point", _ED_DENSITY...),
    "Densities/effective_b_squared_theta_density" => (; long_name="θ-component of |b_eff|²/μ₀ per point", _ED_DENSITY...),
    "Densities/effective_b_squared_zeta_density" => (; long_name="ζ-component of |b_eff|²/μ₀ per point", _ED_DENSITY...),
    "Densities/shear_current_density" => (; long_name="K1|ξ_n|² per point, K1 = |∇ψ|²σS", _ED_DENSITY...),
    "Densities/parallel_current_squared_density" => (; long_name="K2|ξ_n|² per point, K2 = μ₀B²σ²", _ED_DENSITY...),
    "Densities/pressure_curvature_density" => (; long_name="K3|ξ_n|² per point, K3 = 2p'κ_ψ", _ED_DENSITY...),
    "Densities/effective_b_curl_residual_density" => (; long_name="(∇×b_eff)·∇ψ per point", _ED_DENSITY...),
    "Densities/b_squared_density" => (; long_name="standard form: |b|²/μ₀ per point", _ED_DENSITY...),
    "Densities/current_coupling_density" => (; long_name="standard form: j·(b×ξ*) per point", _ED_DENSITY...),
    "Densities/pressure_compression_density" => (; long_name="standard form: (∇·ξ)*(ξ·∇p) per point", _ED_DENSITY...)
]

"""
    write_energy_decomposition!(h5, res) -> nothing

Write `res` to `PerturbedEquilibrium/EnergyDecomposition/` of the open file `h5`, replacing the group
if it exists, and apply the metadata table. Fields of a disabled form are written zero-extent;
`Densities/` appears only when the densities were requested.
"""
function write_energy_decomposition!(h5::HDF5.File, res::EnergyDecompositionResult)
    pe_group = haskey(h5, "PerturbedEquilibrium") ? h5["PerturbedEquilibrium"] : create_group(h5, "PerturbedEquilibrium")
    haskey(pe_group, "EnergyDecomposition") && delete_object(pe_group, "EnergyDecomposition")
    g = create_group(pe_group, "EnergyDecomposition")
    # Disabled forms hold zero-extent arrays of the documented rank, the file's not-computed sentinel.
    for name in _ED_MAIN_FIELDS
        g[String(name)] = getfield(res, name)
    end
    if !isempty(res.theta)
        d = create_group(g, "Densities")
        d["theta"] = res.theta
        d["R"] = res.R
        d["Z"] = res.Z
        for name in _ED_DENSITY_FIELDS
            d[String(name)] = getfield(res, name)
        end
    end
    Utilities.HDF5Annotations.annotate!(g, ENERGY_DECOMPOSITION_H5_ANNOTATIONS)
    return nothing
end
