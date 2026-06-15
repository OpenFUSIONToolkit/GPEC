"""
EnergyReconstructionRealSpace.jl — v1 (real-space) reconstruction of the ideal MHD
energy principle δW, following `docs/tex/recon/main.tex` §"Reconstruction v1"
(Sunjae Lee) and the Fortran DCON `recon` flag (`gpout_recon`/`gpeq_*` in `~/GPEC/gpec`).

For a chosen DCON eigenmode this reconstructs, on the (ψ,θ) grid, the Bernstein
energy integrand

    2μ₀ dδW/dψ = ∫dθ [ |C|² − μ₀ K (ξ·n̂)² ]

decomposed exactly as in `gpout_recon`:
  - `epf(ψ) = (1/μ₀)∫dθ |C|²`           (`gpeq_epf`; |C|² split into ψ/θ/ζ pieces)
  - `dst(ψ) = ∫dθ J K |ξ_n|²`           (`gpeq_dst`; K split into shear/B²σ²/curvature)
  - `dW(ψ)  = ½(epf − dst)`             so ∫dW dψ is the plasma δW.

C is the Bernstein cross-field `C = Q + (ξ·n̂)(μ₀ j×n̂)` (`gpeq_c`), built from the
perturbed field Q (= `compute_perturbed_field_modes`, the gpeq_sol contravariant
J Qⁱ, with covariant Qᵢ from `compute_cova_components`) plus the equilibrium current
j (j^θ=−F'/J, j^ζ=−μ₀p'/χ'−F'q/J). K = |∇ψ|²σ·shear + μ₀B²σ² + 2p'κ^ψ (`gpeq_K`).

The ψ-integral of `dW` equals the DCON boundary-term plasma energy (verified by the
v2 matrix reconstruction, `ForceFreeStates.integrate_energy_v2`), giving a built-in
acceptance test independent of v2's quadrature.
"""

const _MU0 = 4.0e-7 * π

"""
    _theta_derivative_matrix(nθ) -> Matrix{Float64}

Real `nθ×nθ` spectral differentiation operator d/dθ on the uniform grid
θⱼ=(j-1)/nθ, θ∈[0,1) (periodic). Used for the `gpeq_shear` poloidal derivative.
"""
function _theta_derivative_matrix(nθ::Int)
    F = [exp(-2π * im * k * ((j - 1) / nθ)) for k in 0:(nθ-1), j in 1:nθ]
    Finv = [exp(2π * im * k * ((j - 1) / nθ)) / nθ for j in 1:nθ, k in 0:(nθ-1)]
    wn = [k <= nθ ÷ 2 ? k : k - nθ for k in 0:(nθ-1)]
    return real(Finv * (Diagonal(2π * im .* wn) * F))
end

"""
    reconstruct_energy_realspace(equil, ffs_intr, ffit, odet, metric, mode_vec) -> NamedTuple

v1 real-space reconstruction of the energy-principle density for the boundary
displacement `mode_vec` (normal-displacement Ξ_ψ at the edge, `vac_data.wt`-basis).

### Returns
  - `psi`        — ODE ψ grid (1:`odet.step`).
  - `dW`         — ½(epf − dst) per surface; ∫dW dψ is the plasma δW.
  - `epf`, `dst` — the C²/μ₀ and K|ξ_n|² densities per surface.
  - `dW_total`   — ∫dW dψ (trapezoidal).
"""
function reconstruct_energy_realspace(equil::Equilibrium.PlasmaEquilibrium, ffs_intr::ForceFreeStatesInternal, ffit::FourFitVars, odet::OdeState, metric::MetricData, mode_vec::AbstractVector; ntheta::Int=max(512, 4 * ffs_intr.mpert))
    npsi = size(odet.u_store, 4)
    mpert = ffs_intr.mpert
    mlow = ffs_intr.mlow
    nn = ffs_intr.nlow
    chi1 = 2π * equil.psio
    ro = equil.ro
    psi_grid = collect(@view odet.psi_store[1:npsi])

    # --- Eigenmode amplitudes Ξ_ψ, Ξ_ψ', Ξ_s in mode space (gpeq_sol input) ---
    alpha = (@view odet.u_store[:, :, 1, odet.step]) \ Vector{ComplexF64}(mode_vec)
    xi_psi = Matrix{ComplexF64}(undef, npsi, mpert)
    xi_psi1 = Matrix{ComplexF64}(undef, npsi, mpert)
    xi_s = Matrix{ComplexF64}(undef, npsi, mpert)
    for ipsi in 1:npsi
        @views mul!(xi_psi[ipsi, :], odet.u_store[:, :, 1, ipsi], alpha)
        @views mul!(xi_psi1[ipsi, :], odet.ud_store[:, :, 1, ipsi], alpha)
        @views mul!(xi_s[ipsi, :], odet.ud_store[:, :, 2, ipsi], alpha)
    end

    # --- Q field (gpeq_sol/contra/cova) reusing the tested FieldReconstruction helpers ---
    bwp, bwt, bwz = compute_perturbed_field_modes(xi_psi, xi_psi1, xi_s, psi_grid, equil, ffs_intr)
    ctrl0 = PerturbedEquilibriumControl(; reg_spot=0.0)   # unregularized Bernstein form (reg_flag=false)
    clebsch_psi1 = xi_psi1
    clebsch_alpha = xi_s ./ chi1
    xwp, _, _, xmt, xmz = compute_contra_displacements(xi_psi, clebsch_psi1, clebsch_alpha, psi_grid, equil, ffs_intr, metric, ctrl0)
    _, _, _, bvp, bvt, bvz = compute_cova_components(xwp, xmt, xmz, bwp, bwt, bwz, psi_grid, ffs_intr, metric)

    # --- θ grid and inverse-DFT phase (mode → θ) ---
    # |C|² is not band-limited (the gpeq_c current corrections multiply by metric
    # functions of θ), so the θ-integral needs a fine grid (Fortran's mthsurf), not
    # the coarse equilibrium grid, to avoid m-dependent quadrature error.
    nθ = ntheta
    thetas = [(k - 1) / nθ for k in 1:nθ]
    m_vals = [mlow + i - 1 for i in 1:mpert]
    phase = [exp(2π * im * m_vals[i] * thetas[k]) for k in 1:nθ, i in 1:mpert]   # IDFT
    Dθ = _theta_derivative_matrix(nθ)

    epf = zeros(Float64, npsi)
    dst = zeros(Float64, npsi)
    epf_p = zeros(Float64, npsi); epf_t = zeros(Float64, npsi); epf_z = zeros(Float64, npsi)
    dst1 = zeros(Float64, npsi); dst2 = zeros(Float64, npsi); dst3 = zeros(Float64, npsi)

    # per-θ scratch
    delpsi2 = zeros(Float64, nθ)
    dpdt = zeros(Float64, nθ)
    dpdz = zeros(Float64, nθ)
    g22 = zeros(Float64, nθ)
    g23 = zeros(Float64, nθ)
    g33 = zeros(Float64, nθ)
    jacv = zeros(Float64, nθ)
    bsq = zeros(Float64, nθ)
    bsq_psi = zeros(Float64, nθ)
    bsq_theta = zeros(Float64, nθ)
    comp5 = zeros(Float64, nθ)

    for ipsi in 1:npsi
        psi = psi_grid[ipsi]
        hint = (Ref(1), Ref(1))
        q = equil.profiles.q_spline(psi)
        q1 = equil.profiles.q_deriv(psi)
        fp = equil.profiles.F_deriv(psi)      # F' (tex 2πf' convention)
        pp = equil.profiles.P_deriv(psi)      # μ₀p'

        # θ-local geometry (recon_metric w/v convention, matching gpeq_c/curvature/K)
        for k in 1:nθ
            θ = thetas[k]
            pt = (psi, θ)
            r2 = equil.rzphi_rsquared(pt; hint=hint)
            deta = equil.rzphi_offset(pt; hint=hint)
            jac = equil.rzphi_jac(pt; hint=hint)
            fy1 = equil.rzphi_rsquared(pt; deriv=DerivOp(0, 1), hint=hint)
            fy2 = equil.rzphi_offset(pt; deriv=DerivOp(0, 1), hint=hint)
            fy3 = equil.rzphi_nu(pt; deriv=DerivOp(0, 1), hint=hint)
            fx1 = equil.rzphi_rsquared(pt; deriv=DerivOp(1, 0), hint=hint)
            fx2 = equil.rzphi_offset(pt; deriv=DerivOp(1, 0), hint=hint)
            fx3 = equil.rzphi_nu(pt; deriv=DerivOp(1, 0), hint=hint)
            rfac = sqrt(abs(r2))
            eta = 2π * (θ + deta)
            r = ro + rfac * cos(eta)

            w11 = (1.0 + fy2) * (2π)^2 * rfac * r / jac
            w12 = -fy1 * π * r / (rfac * jac)
            w21 = -(2π)^2 * rfac * r * fx2 / jac
            w22 = π * r * fx1 / (rfac * jac)
            w31 = (2π * r * rfac / jac) * (fx2 * fy3 - fx3 * (1.0 + fy2))
            w32 = (r / (2 * rfac * jac)) * (fx3 * fy1 - fx1 * fy3)

            delpsi2[k] = w11^2 + w12^2
            dpdt[k] = w11 * w21 + w12 * w22
            dpdz[k] = w11 * w31 + w12 * w32

            v21 = fy1 / (2 * rfac)
            v22 = (1.0 + fy2) * 2π * rfac
            v23 = fy3 * r
            v33 = 2π * r
            g22[k] = v21^2 + v22^2 + v23^2
            g33[k] = v33^2
            g23[k] = v23 * v33
            jacv[k] = jac

            B = equil.eqfun_B(pt; hint=hint)
            Bx = equil.eqfun_B(pt; deriv=DerivOp(1, 0), hint=hint)
            By = equil.eqfun_B(pt; deriv=DerivOp(0, 1), hint=hint)
            bsq[k] = B^2
            bsq_psi[k] = 2 * B * Bx
            bsq_theta[k] = 2 * B * By

            comp5[k] = (q * dpdt[k] - dpdz[k]) / delpsi2[k]   # gpeq_shear component 5
        end

        dcomp5 = Dθ * comp5   # d/dθ of component 5 (gpeq_shear)

        # θ-space fields (gpeq_sol/cova → θ) for this surface
        bwp_f = phase * @view(bwp[ipsi, :])
        bwt_f = phase * @view(bwt[ipsi, :])
        bwz_f = phase * @view(bwz[ipsi, :])
        bvp_f = phase * @view(bvp[ipsi, :])
        bvt_f = phase * @view(bvt[ipsi, :])
        bvz_f = phase * @view(bvz[ipsi, :])
        xwp_f = phase * @view(xwp[ipsi, :])

        epf_acc = 0.0; epfp_acc = 0.0; epft_acc = 0.0; epfz_acc = 0.0
        dst_acc = 0.0; dst1_acc = 0.0; dst2_acc = 0.0; dst3_acc = 0.0
        for k in 1:nθ
            jac = jacv[k]
            d2 = delpsi2[k]

            # --- gpeq_c: C = Q + current term ---
            jwt_c = -fp / jac                       # μ₀ j^θ
            jwz_c = -pp / chi1 - fp * q / jac       # μ₀ j^ζ
            xwpk = xwp_f[k]
            cwp = bwp_f[k]                                                    # J C^ψ = J Q^ψ
            cwt = bwt_f[k] + xwpk / (d2 * jac) * (jwt_c * g23[k] + jwz_c * g33[k])
            cwz = bwz_f[k] - xwpk / (d2 * jac) * (jwt_c * g22[k] + jwz_c * g23[k])
            cvp = bvp_f[k] + xwpk / d2 * (jwt_c * dpdz[k] - jwz_c * dpdt[k])  # C_ψ
            cvt = bvt_f[k] + xwpk * jwz_c                                     # C_θ
            cvz = bvz_f[k] - xwpk * jwt_c                                     # C_ζ

            # gpeq_epf: |C|² = Σ Re(conj(J Cⁱ)·Cᵢ); the J cancels the 1/(μ₀ jac)·jac integration
            ep_p = real(conj(cwp) * cvp) / _MU0
            ep_t = real(conj(cwt) * cvt) / _MU0
            ep_z = real(conj(cwz) * cvz) / _MU0
            epfp_acc += ep_p; epft_acc += ep_t; epfz_acc += ep_z
            epf_acc += ep_p + ep_t + ep_z

            # --- gpeq_curvature / shear / K ---
            kappa = (d2 / bsq[k]) * (pp + 0.5 * bsq_psi[k] + 0.5 * bsq_theta[k] * dpdt[k] / d2)
            shear = (chi1^2 / jac) * (q1 + dcomp5[k])
            jwt_k = -fp / (jac * _MU0)              # physical j^θ
            jwz_k = q * jwt_k - (pp / _MU0) / chi1  # physical j^ζ  (pp/μ₀ = p')
            bth = chi1 / jac
            bze = q * chi1 / jac
            jdotb = g22[k] * bth * jwt_k + g33[k] * bze * jwz_k + g23[k] * (bth * jwz_k + bze * jwt_k)
            sigma = jdotb / bsq[k]
            K_t1 = d2 * sigma * shear
            K_t2 = bsq[k] * sigma^2 * _MU0
            K_t3 = 2.0 * kappa * (pp / _MU0)        # 2 p' κ^ψ

            # gpeq_normal + gpeq_dst: ξ_n = J ξ^ψ /(J|∇ψ|); ∫ J K |ξ_n|²
            xno = xwpk / (jac * sqrt(d2))
            w = abs2(xno) * jac
            dst1_acc += K_t1 * w; dst2_acc += K_t2 * w; dst3_acc += K_t3 * w
            dst_acc += (K_t1 + K_t2 + K_t3) * w
        end
        epf[ipsi] = epf_acc / nθ
        dst[ipsi] = dst_acc / nθ
        epf_p[ipsi] = epfp_acc / nθ; epf_t[ipsi] = epft_acc / nθ; epf_z[ipsi] = epfz_acc / nθ
        dst1[ipsi] = dst1_acc / nθ; dst2[ipsi] = dst2_acc / nθ; dst3[ipsi] = dst3_acc / nθ
    end

    dW = 0.5 .* (epf .- dst)
    dW_total = _trapz_real(psi_grid, dW)
    return (; psi=psi_grid, dW, epf, dst, dW_total, epf_p, epf_t, epf_z, dst1, dst2, dst3)
end

# Trapezoidal integral over a (possibly nonuniform) grid.
function _trapz_real(x::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    s = 0.0
    @inbounds for i in 1:(length(x)-1)
        s += (y[i] + y[i+1]) * (x[i+1] - x[i]) / 2
    end
    return s
end
