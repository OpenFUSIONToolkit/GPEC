"""
    EnergyDecomposition

Decomposition of the plasma potential energy δW_p of the free-boundary DCON eigenmodes of a forward
ForceFreeStates (FFS) solve on the (ψ, θ) grid, in two forms:

    effective-field form   2δW_p = ∫ [ |b_eff|²/μ₀ − K|ξ_n|² ] dV,   K = K₁ + K₂ + K₃
    standard form          2δW_p = ∫ [ |b|²/μ₀ − j·(b×ξ*) + (∇·ξ)*(ξ·∇p) ] dV

with b_eff = b + ξ_n(μ₀ j × n̂) and K₁ = |∇ψ|²σS (shear current), K₂ = μ₀B²σ² (parallel current),
K₃ = 2p'κ_ψ (pressure curvature). The Fortran counterpart is `gpout_recon` with `effective_field_flag`
and `standard_form_flag`. Both totals are scaled by 2μ₀/ψ₀² so they read in the normalization of
`ffs.free_boundary.ep`.

Call [`decompose_plasma_energy`](@ref) on a `ForceFreeStatesResult` and plot the result with
[`plot_energy_decomposition`](@ref).

## References

  - I. B. Bernstein, E. A. Frieman, M. D. Kruskal and R. M. Kulsrud, Proc. R. Soc. A 244, 17 (1958):
    the energy principle and its standard form.
  - I. B. Bernstein, Handbook of Plasma Physics Vol. 1, p. 421 (North-Holland, 1983): the effective-field form.
  - M. S. Chance et al., PPPL-CFP-2687 (1992), Eqs. 12a–12c and 15–17: both forms and the split of K.
  - A. H. Boozer, Rev. Mod. Phys. 76, 1071 (2004), Sec. V.C, Eqs. 98–100: the effective field and
    (∇×b_eff)·∇ψ = 0.
"""
module EnergyDecomposition

using FastInterpolations
using LinearAlgebra
using Plots
import ...ForceFreeStates as FFS
import ...PerturbedEquilibrium as PE
import ...Utilities.FourierTransforms: FourierTransform, inverse
import ...Utilities.PhysicalConstants: MU_0

export decompose_plasma_energy, plot_energy_decomposition

"""
Names of the seven density terms, in the order of the third axis of `density` and the second of `profile`.
"""
const TERMS = (:b_eff_sq, :K1, :K2, :K3, :b_sq, :j_b_xi, :divxi_gradp)

# ψ_N gap below which an equilibrium knot counts as an end of the solution domain.
const EDGE_KNOT_TOL = 1e-10

"""
    radial_nodes(ffs) -> Vector{Float64}

The equilibrium ψ knots inside the ξ solution domain, plus each end of the domain (the ODE start,
e.g. a `qlow` crossing, and the solution edge) that falls between knots. On the knots the metric,
the Euler-Lagrange matrices and the geometry are exact.
"""
function radial_nodes(ffs::FFS.ForceFreeStatesResult)
    sol = ffs.solution
    # The sweep starts at max(psilow, first profile knot), as initialize_el_at_axis! does, but saves its first state one step later.
    lo, hi = max(ffs.psilow, ffs.equil.profiles.xs[1]), sol.psi_store[sol.step]
    psi = [x for x in ffs.equil.rzphi_xs if lo <= x <= hi]
    (isempty(psi) || psi[1] > lo + EDGE_KNOT_TOL) && pushfirst!(psi, lo)
    psi[end] < hi - EDGE_KNOT_TOL && push!(psi, hi)
    return psi
end

"""
    surface_geometry(ffs, psi) -> NamedTuple of (length(psi), mtheta) matrices

Geometry, equilibrium current and curvature-kernel coefficients at the nodes `psi` on the θ knots,
all from `ffs.metric` (columns e_i·e_j/J, J, ∂J/∂ψ of `make_metric`). The contravariant products
follow from the covariant metric by cofactors, |∇ψ|² = (g₂₂g₃₃ − g₂₃²)/J², ∇ψ·∇θ = (g₁₃g₂₃ − g₁₂g₃₃)/J²,
∇ψ·∇ζ = (g₁₂g₂₃ − g₁₃g₂₂)/J², and B² = (χ'/J)²(g₂₂ + 2q g₂₃ + q² g₃₃). The kernel follows Fortran
`gpeq_shear`, `gpeq_curvature` and `gpeq_K`:

    S   = (χ'²/J) [ q' + ∂_θ( (q ∇ψ·∇θ − ∇ψ·∇ζ)/|∇ψ|² ) ]
    κ_ψ = (|∇ψ|²/B²) [ (μ₀p)' + ½∂_ψB² + ½∂_θB² (∇ψ·∇θ)/|∇ψ|² ]
    σ   = j·B/B²,  μ₀j^θ = −F'/J,  μ₀j^ζ = −(μ₀p)'/χ' − qF'/J
"""
function surface_geometry(ffs::FFS.ForceFreeStatesResult, psi::Vector{Float64})
    equil = ffs.equil
    profiles = equil.profiles
    fs = ffs.metric.fs
    xs = equil.rzphi_xs
    theta_ext = collect(equil.rzphi_ys)
    mtheta = length(theta_ext) - 1
    chi1 = 2π * equil.psio
    r2 = equil.rzphi_rsquared.nodal_derivs.partials
    deta = equil.rzphi_offset.nodal_derivs.partials

    # Knot rows of the eight metric columns, B², R and Z, splined in ψ to reach the edge node off the knots.
    knot_vals = zeros(length(xs), 11 * mtheta)
    for i in eachindex(xs), k in 1:mtheta
        J = fs[i, k, 7]
        q = profiles.q_spline(xs[i])
        rfac = sqrt(abs(r2[1, i, k]))
        eta = 2π * (theta_ext[k] + deta[1, i, k])
        # Deviation from Fortran gpeq_K/gpeq_curvature: B² of the metric, not eqfun B², so K₂|ξ_n|² cancels |b_eff|² exactly.
        bsq = (chi1 / J)^2 * J * (fs[i, k, 2] + 2q * fs[i, k, 4] + q^2 * fs[i, k, 3])
        for (c, v) in enumerate((fs[i, k, 1:8]..., bsq, equil.ro + rfac * cos(eta), equil.zo + rfac * sin(eta)))
            knot_vals[i, (c-1)*mtheta+k] = v
        end
    end
    itp = cubic_interp(xs, Series(knot_vals); bc=CubicFit(), extrap=ExtendExtrap())

    names = (:R, :Z, :jac, :jac_psi, :g11, :g12, :g13, :g22, :g23, :g33, :delpsi2, :dpdt, :dpdz, :mu0_j_theta, :mu0_j_zeta, :shear, :K1, :K2, :K3)
    geo = NamedTuple{names}(ntuple(_ -> zeros(length(psi), mtheta), length(names)))
    ratio = zeros(mtheta + 1)
    for (i, p) in enumerate(psi)
        v = itp(p)
        dv = itp(p; deriv=DerivOp(1))
        col(c, k) = v[(c-1)*mtheta+k]
        q, q1 = profiles.q_spline(p), profiles.q_deriv(p)
        F1, mu0p1 = profiles.F_deriv(p), profiles.P_deriv(p)
        bsq = [col(9, k) for k in 1:mtheta]
        dbsq_theta = deriv1(cubic_interp(theta_ext, [bsq; bsq[1]]; bc=PeriodicBC()))
        for k in 1:mtheta
            J = col(7, k)
            g11, g22, g33, g23, g13, g12 = (col(c, k) * J for c in 1:6)
            d2 = (g22 * g33 - g23^2) / J^2
            dpdt = (g13 * g23 - g12 * g33) / J^2
            dpdz = (g12 * g23 - g13 * g22) / J^2
            geo.R[i, k], geo.Z[i, k] = col(10, k), col(11, k)
            geo.jac[i, k], geo.jac_psi[i, k] = J, col(8, k)
            geo.g11[i, k], geo.g12[i, k], geo.g13[i, k], geo.g22[i, k], geo.g23[i, k], geo.g33[i, k] = g11, g12, g13, g22, g23, g33
            geo.delpsi2[i, k], geo.dpdt[i, k], geo.dpdz[i, k] = d2, dpdt, dpdz
            geo.mu0_j_theta[i, k] = -F1 / J
            geo.mu0_j_zeta[i, k] = -mu0p1 / chi1 - F1 * q / J
            ratio[k] = (q * dpdt - dpdz) / d2
        end
        ratio[end] = ratio[1]
        dratio = deriv1(cubic_interp(theta_ext, ratio; bc=PeriodicBC()))
        for k in 1:mtheta
            J, d2, b2 = geo.jac[i, k], geo.delpsi2[i, k], bsq[k]
            S = chi1^2 / J * (q1 + dratio(theta_ext[k]))
            kappa = d2 / b2 * (mu0p1 + 0.5 * dv[8*mtheta+k] + 0.5 * dbsq_theta(theta_ext[k]) * geo.dpdt[i, k] / d2)
            jt, jz = geo.mu0_j_theta[i, k] / MU_0, geo.mu0_j_zeta[i, k] / MU_0
            bt, bz = chi1 / J, q * chi1 / J
            sigma = (geo.g22[i, k] * bt * jt + geo.g33[i, k] * bz * jz + geo.g23[i, k] * (bt * jz + bz * jt)) / b2
            geo.shear[i, k] = S
            geo.K1[i, k] = d2 * sigma * S
            geo.K2[i, k] = MU_0 * b2 * sigma^2
            geo.K3[i, k] = 2 * mu0p1 / MU_0 * kappa
        end
    end
    return geo
end

"""
    node_states(ffs, psi) -> (u1, du1, xi_s)

Euler-Lagrange basis Ξ_ψ, dΞ_ψ/dψ and Ξ_s = −A⁻¹(BΞ'_ψ + CΞ_ψ) at the nodes `psi`, each
`(numpert_total, numpert_total, length(psi))`. Each node state is integrated from the nearest saved
step (`ForceFreeStates.solution_state_at`), so it does not depend on how densely the sweep saved.
"""
function node_states(ffs::FFS.ForceFreeStatesResult, psi::Vector{Float64})
    N = ffs.numpert_total
    u1, du1, xi_s = (zeros(ComplexF64, N, N, length(psi)) for _ in 1:3)
    du = zeros(ComplexF64, N, N, 2)
    for (i, p) in enumerate(psi)
        u = FFS.solution_state_at(ffs, p)
        FFS.el_derivatives!(du, u, false, ffs.equil, ffs.mats, ffs, p, Ref(1), Ref(1))
        u1[:, :, i] .= @view u[:, :, 1]
        du1[:, :, i] .= @view du[:, :, 1]
        FFS.compute_node_xi_s!(view(xi_s, :, :, i), view(du1, :, :, i), view(u1, :, :, i), ffs.mats, p)
    end
    return u1, du1, xi_s
end

"""
    eigenmode_fields(ffs, k, psi, states, ft) -> NamedTuple of (length(psi), mtheta) matrices

θ-space fields of free-boundary eigenmode `k`: ξ^ψ (`xi`), ∂ξ^ψ/∂ψ (`xi1`), Jξ^θ, Jξ^ζ, ∂_θ(Jξ^θ)
and Jb^ψ, Jb^θ, Jb^ζ. The edge condition is the unit-power eigenvector `wt[:, k]`; the mode-space
chain is the one `PerturbedEquilibrium` runs for a forced response.
"""
function eigenmode_fields(ffs::FFS.ForceFreeStatesResult, k::Int, psi::Vector{Float64}, states, ft::FourierTransform)
    u1, du1, xs = states
    sol = ffs.solution
    alpha = sol.u_store[:, :, 1, sol.step] \ Vector{ComplexF64}(ffs.free_boundary.wt[:, k])
    rows(A) = permutedims(reduce(hcat, (A[:, :, i] * alpha for i in eachindex(psi))))
    xi, xi1, xi_s = rows(u1), rows(du1), rows(xs)
    chi1 = 2π * ffs.equil.psio
    Jbp, Jbt, Jbz = PE.compute_perturbed_field_modes(xi, xi1, xi_s, psi, ffs.equil, ffs)
    # Deviation from Fortran gpeq_contra: always unregularized (regularization width 0).
    _, Jxt, Jxz, _, _ = PE.compute_contra_displacements(xi, xi1, xi_s ./ chi1, psi, ffs.equil, ffs, ffs.metric, 0.0)
    # Deviation from Fortran gpeq_firstform: ∂_θ(Jξ^θ) is spectral, not a periodic-spline derivative.
    dJxt = Jxt .* transpose(2π * im .* (ffs.mlow:ffs.mhigh))
    theta(X) = permutedims(inverse(ft, permutedims(X)))
    return (; xi=theta(xi), xi1=theta(xi1), Jxt=theta(Jxt), Jxz=theta(Jxz), dJxt=theta(dJxt), Jbp=theta(Jbp), Jbt=theta(Jbt), Jbz=theta(Jbz))
end

"""
    effective_field(geo, f, i, k) -> (Jc_psi, Jc_theta, Jc_zeta, c_psi, c_theta, c_zeta, b_psi, b_theta, b_zeta, xi_n)

Contravariant J b_eff^i and covariant b_eff,i of b_eff = b + ξ_n(μ₀ j × n̂) at node `i`, θ point `k`
(Fortran `gpeq_c`), with the covariant field b_i = g_ij Jb^j/J and ξ_n = ξ^ψ/|∇ψ|:

    J b_eff^θ = J b^θ + Jξ^ψ/(J|∇ψ|²) (μ₀j^θ g_θζ + μ₀j^ζ g_ζζ)     b_eff,θ = b_θ + Jξ^ψ μ₀j^ζ
    J b_eff^ζ = J b^ζ − Jξ^ψ/(J|∇ψ|²) (μ₀j^θ g_θθ + μ₀j^ζ g_θζ)     b_eff,ζ = b_ζ − Jξ^ψ μ₀j^θ
    b_eff,ψ = b_ψ + Jξ^ψ/|∇ψ|² (μ₀j^θ ∇ψ·∇ζ − μ₀j^ζ ∇ψ·∇θ),          J b_eff^ψ = J b^ψ
"""
function effective_field(geo::NamedTuple, f::NamedTuple, i::Int, k::Int)
    J, d2 = geo.jac[i, k], geo.delpsi2[i, k]
    g11, g12, g13, g22, g23, g33 = geo.g11[i, k], geo.g12[i, k], geo.g13[i, k], geo.g22[i, k], geo.g23[i, k], geo.g33[i, k]
    jt, jz = geo.mu0_j_theta[i, k], geo.mu0_j_zeta[i, k]
    Jbp, Jbt, Jbz = f.Jbp[i, k], f.Jbt[i, k], f.Jbz[i, k]
    # Deviation from Fortran gpeq_cova/gpeq_normal/gpeq_c: b_i, Jξ^ψ, ξ_n and b_eff are pointwise in θ, not band-truncated.
    xw = J * f.xi[i, k]
    bp = (g11 * Jbp + g12 * Jbt + g13 * Jbz) / J
    bt = (g12 * Jbp + g22 * Jbt + g23 * Jbz) / J
    bz = (g13 * Jbp + g23 * Jbt + g33 * Jbz) / J
    Jct = Jbt + xw / (d2 * J) * (jt * g23 + jz * g33)
    Jcz = Jbz - xw / (d2 * J) * (jt * g22 + jz * g23)
    cp = bp + xw / d2 * (jt * geo.dpdz[i, k] - jz * geo.dpdt[i, k])
    return (Jbp, Jct, Jcz, cp, bt + xw * jz, bz - xw * jt, bp, bt, bz, xw / (J * sqrt(d2)))
end

# Relative (∇×b_eff)·∇ψ residual above which the effective field is reported as inconsistent (Fortran cveri_tol).
const CURL_RESIDUAL_TOL = 1e-8
# Relative mismatch of a δW_p total against ep[k] above which the radial grid is reported as too coarse.
const ENERGY_MISMATCH_TOL = 0.05

"""
    curl_maxima(c_theta, c_zeta, jac, ft, n) -> (residual, scale)

Largest |(∇×b_eff)·∇ψ| = |∂_θ b_eff,ζ − ∂_ζ b_eff,θ|/J on one surface and the largest
(|∂_θ b_eff,ζ| + |∂_ζ b_eff,θ|)/J that it cancels, from the covariant components `c_theta`, `c_zeta`
on the θ grid of `ft`. Perturbations vary as exp(2πi(mθ − nζ)), so ∂_ζ → −2πin. As in Fortran
`gpeq_cveri`, the reported residual is the largest `residual` over all surfaces divided by the
largest `scale`.
"""
function curl_maxima(c_theta::Vector{ComplexF64}, c_zeta::Vector{ComplexF64}, jac::AbstractVector{Float64}, ft::FourierTransform, n::Int)
    m = ft.mlow:(ft.mlow+ft.mpert-1)
    dtheta_c_zeta = inverse(ft, 2π * im .* m .* ft(c_zeta)) ./ jac
    dzeta_c_theta = inverse(ft, -2π * im * n .* ft(c_theta)) ./ jac
    return maximum(abs, dtheta_c_zeta .- dzeta_c_theta), maximum(abs.(dtheta_c_zeta) .+ abs.(dzeta_c_theta))
end

"""
    decompose_plasma_energy(ffs; eigenmodes=[1]) -> NamedTuple

Decompose δW_p of the free-boundary eigenmodes `eigenmodes` of a forward, ideal, single-n FFS solve
with the vacuum response (`vac_flag = true`). Fields of the result:

  - `psi` - radial nodes (`radial_nodes`); `theta` - θ knots in [0, 1); `R`, `Z` - node positions in m
  - `terms` - `TERMS`, the names along the term axis
  - `eigenmodes` - the decomposed eigenmode indices
  - `density` - `(npsi, mtheta, 7, neig)` pointwise integrands, scaled by 2μ₀/ψ₀²
  - `profile` - `(npsi, 7, neig)` surface integrals ∮J·density dθ
  - `dW_effective`, `dW_standard` - ½∫ of the effective-field and standard-form combinations
  - `relative_error_effective`, `relative_error_standard` - `(dW − real(ep[k])) / abs(ep[k])` of each total
  - `curl_residual` - largest (∇×b_eff)·∇ψ on the knots relative to the largest term it cancels (`curl_maxima`)

These last two are the success checks of the decomposition, and both are warned. On the knots
(∇×b_eff)·∇ψ vanishes identically, so a curl residual above `CURL_RESIDUAL_TOL` flags b_eff inconsistent
with the metric and the equilibrium current; near the axis of a reconstructed equilibrium the radial grid
and the coordinates can raise it too. A mismatch against `ep[k]` above `ENERGY_MISMATCH_TOL` means the
radial grid is too coarse for the ψ quadrature (it converges with `mpsi`); it is not a check of the
algebra, since an error that moves a total by a few percent stays below it.
"""
function decompose_plasma_energy(ffs::FFS.ForceFreeStatesResult; eigenmodes::AbstractVector{<:Integer}=[1])
    ffs.integrator === :forward && ffs.solution !== nothing || throw(ArgumentError("energy decomposition needs a forward-integrator solve"))
    ffs.free_boundary === nothing && throw(ArgumentError("energy decomposition needs the free-boundary eigenmodes (vac_flag = true)"))
    ffs.nlow == ffs.nhigh || throw(ArgumentError("energy decomposition handles a single toroidal mode number"))
    ffs.mats.kinetic === nothing || throw(ArgumentError("energy decomposition handles ideal solves only, not kinetic ones"))
    nwt = size(ffs.free_boundary.wt, 2)
    all(k -> 1 <= k <= nwt, eigenmodes) || throw(ArgumentError("eigenmodes must lie in 1:$nwt"))

    equil = ffs.equil
    # Deviation from Fortran gpout_recon: the radial integral runs over the equilibrium knots, not the ODE psifac grid.
    psi = radial_nodes(ffs)
    mtheta = length(equil.rzphi_ys) - 1
    geo = surface_geometry(ffs, psi)
    states = node_states(ffs, psi)
    ft = FourierTransform(mtheta, ffs.mpert, ffs.mlow)
    n = ffs.nlow
    scale = 2 * MU_0 / equil.psio^2

    density = zeros(ComplexF64, length(psi), mtheta, length(TERMS), length(eigenmodes))
    curl_max, curl_scale = zeros(length(eigenmodes)), zeros(length(eigenmodes))
    c_theta, c_zeta = zeros(ComplexF64, mtheta), zeros(ComplexF64, mtheta)
    for (m, k) in enumerate(eigenmodes)
        f = eigenmode_fields(ffs, k, psi, states, ft)
        for i in eachindex(psi)
            p1 = equil.profiles.P_deriv(psi[i]) / MU_0
            for j in 1:mtheta
                Jcp, Jct, Jcz, cp, ct, cz, bp, bt, bz, xi_n = effective_field(geo, f, i, j)
                c_theta[j], c_zeta[j] = ct, cz
                J, xw = geo.jac[i, j], geo.jac[i, j] * f.xi[i, j]
                jt, jz = geo.mu0_j_theta[i, j], geo.mu0_j_zeta[i, j]
                divxi = f.xi1[i, j] + geo.jac_psi[i, j] / J * f.xi[i, j] + f.dJxt[i, j] / J - 2π * im * n * f.Jxz[i, j] / J
                terms = (real(conj(Jcp) * cp + conj(Jct) * ct + conj(Jcz) * cz) / (MU_0 * J),
                    geo.K1[i, j] * abs2(xi_n), geo.K2[i, j] * abs2(xi_n), geo.K3[i, j] * abs2(xi_n),
                    real(conj(f.Jbp[i, j]) * bp + conj(f.Jbt[i, j]) * bt + conj(f.Jbz[i, j]) * bz) / (MU_0 * J),
                    (-jt * (f.Jbp[i, j] * conj(f.Jxz[i, j]) - f.Jbz[i, j] * conj(xw)) + jz * (f.Jbp[i, j] * conj(f.Jxt[i, j]) - f.Jbt[i, j] * conj(xw))) / (MU_0 * J),
                    conj(divxi) * p1 * f.xi[i, j])
                for t in eachindex(terms)
                    density[i, j, t, m] = scale * terms[t]
                end
            end
            if psi[i] in equil.rzphi_xs
                residual, cancelled = curl_maxima(c_theta, c_zeta, view(geo.jac, i, :), ft, n)
                curl_max[m], curl_scale[m] = max(curl_max[m], residual), max(curl_scale[m], cancelled)
            end
        end
    end
    curl = curl_max ./ curl_scale
    profile = dropdims(sum(density .* geo.jac; dims=2); dims=2) ./ mtheta
    # The imaginary parts of the surface integrals are roundoff, so the ψ integrals take the real part.
    total(t, m) = FastInterpolations.integrate(cubic_interp(psi, real.(profile[:, t, m]); bc=CubicFit()))
    dW_effective = [0.5 * (total(1, m) - total(2, m) - total(3, m) - total(4, m)) for m in eachindex(eigenmodes)]
    dW_standard = [0.5 * (total(5, m) - total(6, m) + total(7, m)) for m in eachindex(eigenmodes)]
    ep = real.(ffs.free_boundary.ep[eigenmodes])
    relative_error_effective = (dW_effective .- ep) ./ abs.(ep)
    relative_error_standard = (dW_standard .- ep) ./ abs.(ep)
    for (m, k) in enumerate(eigenmodes)
        curl[m] > CURL_RESIDUAL_TOL && @warn "Energy decomposition: (∇×b_eff)·∇ψ of eigenmode $k does not vanish (relative residual $(curl[m]))"
        for (form, err) in (("effective-field", relative_error_effective[m]), ("standard-form", relative_error_standard[m]))
            abs(err) > ENERGY_MISMATCH_TOL &&
                @warn "Energy decomposition: $form δW_p of eigenmode $k differs from ep[$k] by $(round(100 * err; sigdigits=3)) %; refine mpsi"
        end
    end
    return (; psi, theta=equil.rzphi_ys[1:mtheta], R=geo.R, Z=geo.Z, terms=TERMS, eigenmodes=collect(eigenmodes), density, profile,
        dW_effective, dW_standard, relative_error_effective, relative_error_standard, curl_residual=curl)
end

"""
    plot_energy_decomposition(ed; eigenmode=first(ed.eigenmodes), save_path=nothing) -> Plots.Plot

Radial profiles of the seven terms of `ed` (from `decompose_plasma_energy`) for one eigenmode, over
the cumulative effective-field and standard-form δW_p.
"""
function plot_energy_decomposition(ed::NamedTuple; eigenmode::Integer=first(ed.eigenmodes), save_path=nothing)
    m = findfirst(==(eigenmode), ed.eigenmodes)
    m === nothing && throw(ArgumentError("eigenmode $eigenmode was not decomposed"))
    prof = real.(ed.profile[:, :, m])
    cum = reduce(hcat, (vec(cumulative_integrate(cubic_interp(ed.psi, prof[:, t]; bc=CubicFit()))) for t in eachindex(ed.terms)))
    eff = 0.5 .* (cum[:, 1] .- cum[:, 2] .- cum[:, 3] .- cum[:, 4])
    std = 0.5 .* (cum[:, 5] .- cum[:, 6] .+ cum[:, 7])
    p1 = plot(ed.psi, prof; label=permutedims(collect(string.(ed.terms))), xlabel="ψ_N", ylabel="∮ J·density dθ", title="Eigenmode $eigenmode")
    p2 = plot(ed.psi, [eff std]; label=["effective field" "standard form"], xlabel="ψ_N", ylabel="cumulative δW_p")
    p = plot(p1, p2; layout=(2, 1), size=(800, 800), left_margin=12Plots.mm, bottom_margin=4Plots.mm)
    if save_path !== nothing
        savefig(p, save_path)
        println("Saved energy decomposition plot to ", abspath(save_path))
    end
    return p
end

end # module EnergyDecomposition
