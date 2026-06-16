"""
EnergyReconstruction.jl — reconstruction of the ideal MHD energy principle δW for a
DCON eigenmode, both ways used by the Fortran DCON `recon` flag (`gpout_recon`/`gpeq_*`
in `~/GPEC/gpec`; `docs/tex/recon/main.tex`, Sunjae Lee):

  - **v1 (real-space)** — on the (ψ,θ) grid, the Bernstein integrand
    `2μ₀ dδW/dψ = ∫dθ [|C|² − μ₀ K (ξ·n̂)²]`, split into `epf=(1/μ₀)∫J|C|²`
    (with ψ/θ/ζ pieces) and `dst=∫J K|ξ_n|²` (with shear/B²σ²/curvature pieces).
    `C = Q + (ξ·n̂)(μ₀ j×n̂)` (`gpeq_c`) is built from the perturbed field Q (the tested
    `compute_perturbed_field_modes`/`compute_cova_components` helpers) plus the
    equilibrium current. `K = |∇ψ|²σ·shear + μ₀B²σ² + 2p'κ^ψ` (`gpeq_K`).
  - **v2 (matrix form)** — the same δW as the DCON/Newcomb radial matrix form
    `2μ₀δW = ∫dψ [Ξ_s†AΞ_s + Ξ_s†(BΞ_ψ'+CΞ_ψ) + h.c. + Ξ_ψ'†DΞ_ψ' + (Ξ_ψ'†EΞ_ψ + h.c.) + Ξ_ψ†HΞ_ψ]`.
    A,B,C,H reuse `ffit.amats/bmats/cmats/hmats` (identical to the tex with χ'=const); the
    tex D,E are built by `build_tex_de_matrices`; Ξ_s = -A⁻¹(BΞ_ψ'+CΞ_ψ) is formed analytically.

The v2 boundary term equals the DCON plasma energy `psio²·ep[k]` to machine precision, so
it is the authoritative δW; v1 reconstructs the real-space decomposition (validated against
the Fortran `gpec_recon_*` reference term-by-term). Integrals use cubic-spline integration
(`FastInterpolations.integrate`, matching the Fortran `recon_int="spline"`).
"""

const _MU0 = 4.0e-7 * π

# Definite ∫ y dx via cubic-spline integration (Fortran recon_int="spline"). Used for the
# smooth real-space v1 densities. For the v2 *matrix* density, which spikes at rational
# surfaces, a cubic spline rings — `_trapz` is used there instead (it is only a self-
# consistency diagnostic; the authoritative v2 value is the exact boundary term).
_spline_int(x::AbstractVector, y::AbstractVector) =
    FastInterpolations.integrate(cubic_interp(collect(float.(x)), collect(real.(y)); extrap=ExtendExtrap()))

function _trapz(x::AbstractVector{<:Real}, y::AbstractVector{<:Number})
    s = zero(eltype(y))
    @inbounds for i in 1:(length(x)-1)
        s += (y[i] + y[i+1]) * (x[i+1] - x[i]) / 2
    end
    return s
end

# ============================  v2 matrix form  ============================

"""
    build_tex_de_matrices(equil, intr, metric) -> NamedTuple

Build the tex `Reconstruction v2` D and E coefficient matrices as cubic series
interpolants over the metric ψ-grid (χ' = 2π·psio = const):

  D[m,m'] = χ'²(g₂₂ + 2q g₂₃ + q² g₃₃)
  E[m,m'] = χ'²q'(g₂₃ + q g₃₃) - 2πi χ'²(g₁₂ + q g₃₁)(m'-nq) + μ₀p' ⟨1⟩

with gᵢⱼ the Fourier coefficients ⟨gᵢⱼ/J²⟩ at Fourier index dm=m'-m. Returns `dmats`, `emats`.
"""
function build_tex_de_matrices(equil::Equilibrium.PlasmaEquilibrium, intr::ForceFreeStatesInternal, metric::MetricData)
    mpsi = metric.mpsi
    profiles = equil.profiles
    N = intr.numpert_total
    nb = 2 * intr.mband + 1
    mid = intr.mband + 1

    dmats_flat = zeros(ComplexF64, mpsi, N^2)
    emats_flat = zeros(ComplexF64, mpsi, N^2)
    g22 = Vector{ComplexF64}(undef, nb)
    g33 = Vector{ComplexF64}(undef, nb)
    g23 = Vector{ComplexF64}(undef, nb)
    g31 = Vector{ComplexF64}(undef, nb)
    g12 = Vector{ComplexF64}(undef, nb)
    jmat = Vector{ComplexF64}(undef, nb)

    chi1 = 2π * equil.psio
    hint = Ref(1)
    fc = metric.fourier_coeffs
    for ipsi in 1:mpsi
        psi = profiles.xs[ipsi]
        q = profiles.q_spline.y[ipsi]
        q1 = profiles.q_deriv(psi; hint=hint)
        p1 = profiles.P_deriv(psi; hint=hint)

        for m in 0:intr.mband
            g22[mid-m] = Utilities.get_complex_coeff(fc, ipsi, m, 2)
            g33[mid-m] = Utilities.get_complex_coeff(fc, ipsi, m, 3)
            g23[mid-m] = Utilities.get_complex_coeff(fc, ipsi, m, 4)
            g31[mid-m] = Utilities.get_complex_coeff(fc, ipsi, m, 5)
            g12[mid-m] = Utilities.get_complex_coeff(fc, ipsi, m, 6)
            jmat[mid-m] = Utilities.get_complex_coeff(fc, ipsi, m, 7)
        end
        for k in 1:intr.mband
            g22[mid+k] = conj(g22[mid-k])
            g33[mid+k] = conj(g33[mid-k])
            g23[mid+k] = conj(g23[mid-k])
            g31[mid+k] = conj(g31[mid-k])
            g12[mid+k] = conj(g12[mid-k])
            jmat[mid+k] = conj(jmat[mid-k])
        end

        dview = @view dmats_flat[ipsi, :]
        eview = @view emats_flat[ipsi, :]
        for n in intr.nlow:intr.nhigh
            ipert_n = n - intr.nlow + 1
            nq = n * q
            for m1 in intr.mlow:intr.mhigh
                ipert_m = m1 - intr.mlow + 1
                for dm in max(1 - ipert_m, -intr.mband):min(intr.mpert - ipert_m, intr.mband)
                    m2 = m1 + dm
                    singfac2 = m2 - nq
                    jpert_m = ipert_m + dm
                    dmidx = dm + mid
                    ipert = ipert_m + (ipert_n - 1) * intr.mpert
                    jpert = jpert_m + (ipert_n - 1) * intr.mpert
                    ipert_flat = ipert + (jpert - 1) * N
                    dview[ipert_flat] = chi1^2 * (g22[dmidx] + 2q * g23[dmidx] + q^2 * g33[dmidx])
                    eview[ipert_flat] =
                        chi1^2 * q1 * (g23[dmidx] + q * g33[dmidx]) -
                        2π * im * chi1^2 * (g12[dmidx] + q * g31[dmidx]) * singfac2 +
                        p1 * jmat[dmidx]
                end
            end
        end
    end

    itp_opts = (; extrap=ExtendExtrap())
    dmats = cubic_interp(metric.xs, Series(dmats_flat); itp_opts...)
    emats = cubic_interp(metric.xs, Series(emats_flat); itp_opts...)
    return (; dmats, emats)
end

"""
    reconstruct_energy_density_v2(equil, intr, ffit, odet, mode_vec; de=nothing) -> NamedTuple

v2 energy-principle integrand `2μ₀ dδW/dψ` over the stored ψ grid for boundary
displacement `mode_vec`. Returns `psi` and `density`. Pass a prebuilt `de` to avoid
rebuilding the metric.
"""
function reconstruct_energy_density_v2(equil::Equilibrium.PlasmaEquilibrium, intr::ForceFreeStatesInternal, ffit::FourFitVars, odet::OdeState, mode_vec::AbstractVector; de=nothing)
    n = intr.numpert_total
    nstep = odet.step
    if de === nothing
        metric = ForceFreeStates.make_metric(equil; mband=intr.mband)
        de = build_tex_de_matrices(equil, intr, metric)
    end

    u1_edge = @view odet.u_store[:, :, 1, nstep]
    alpha = u1_edge \ Vector{ComplexF64}(mode_vec)

    psi = collect(@view odet.psi_store[1:nstep])
    density = Vector{ComplexF64}(undef, nstep)

    A = Matrix{ComplexF64}(undef, n, n)
    B = similar(A)
    C = similar(A)
    D = similar(A)
    E = similar(A)
    H = similar(A)
    hint = Ref(1)
    for i in 1:nstep
        p = psi[i]
        A .= reshape(ffit.amats(p; hint=hint), n, n)
        B .= reshape(ffit.bmats(p; hint=hint), n, n)
        C .= reshape(ffit.cmats(p; hint=hint), n, n)
        H .= reshape(ffit.hmats(p; hint=hint), n, n)
        D .= reshape(de.dmats(p; hint=hint), n, n)
        E .= reshape(de.emats(p; hint=hint), n, n)

        xpsi = @view(odet.u_store[:, :, 1, i]) * alpha    # Ξ_ψ
        xpsi1 = @view(odet.ud_store[:, :, 1, i]) * alpha   # Ξ_ψ'
        bcx = B * xpsi1 + C * xpsi
        xs = -(A \ bcx)                                    # Ξ_s = -A⁻¹(BΞ_ψ'+CΞ_ψ)

        t1 = dot(xs, A, xs)
        t2 = dot(xs, bcx)
        t3 = dot(xpsi1, D, xpsi1)
        t4 = dot(xpsi1, E, xpsi)
        t5 = dot(xpsi, H, xpsi)
        density[i] = t1 + (t2 + conj(t2)) + t3 + (t4 + conj(t4)) + t5
    end
    return (; psi, density)
end

"""
    integrate_energy_v2(equil, intr, ffit, odet, mode_vec; de=nothing) -> NamedTuple

Cubic-spline-integrate the v2 density and also return the exact DCON boundary-term
energy `Ξ_ψ(edge)†·u₂(edge)` (= psio²·ep). Returns `dW_volume`, `dW_boundary`, `density`, `psi`.
"""
function integrate_energy_v2(equil::Equilibrium.PlasmaEquilibrium, intr::ForceFreeStatesInternal, ffit::FourFitVars, odet::OdeState, mode_vec::AbstractVector; de=nothing)
    rec = reconstruct_energy_density_v2(equil, intr, ffit, odet, mode_vec; de=de)
    dW_volume = _trapz(rec.psi, rec.density)

    nstep = odet.step
    u1_edge = @view odet.u_store[:, :, 1, nstep]
    u2_edge = @view odet.u_store[:, :, 2, nstep]
    alpha = u1_edge \ Vector{ComplexF64}(mode_vec)
    dW_boundary = dot(u1_edge * alpha, u2_edge * alpha)

    # Running cumulative δW(ψ) = Ξ_ψ(ψ)†·u₂(ψ) over the full grid. This is the exact
    # ψ-cumulative energy (u₂ = canonical momentum F̃Ξ_ψ'+K̃Ξ_ψ; ∫density dψ telescopes to
    # it), regular through rational surfaces — unlike the naive A⁻¹ matrix density which
    # spikes there. Its endpoint equals dW_boundary (= psio²·ep).
    dW_running = [real(dot(@view(odet.u_store[:, :, 1, i]) * alpha, @view(odet.u_store[:, :, 2, i]) * alpha)) for i in 1:nstep]

    return (; dW_volume, dW_boundary, dW_running, rec.density, rec.psi)
end

"""
    dW_of_shape(equil, intr, ffit, psi_grid, xi_psi_modes; de=nothing) -> NamedTuple

Evaluate the energy principle for an ARBITRARY perturbation SHAPE given as the normal-
displacement harmonics Ξ_ψ(ψ, m) on a radial grid (`xi_psi_modes` is `[length(psi_grid), mpert]`,
columns ordered m = mlow … mhigh). Unlike `energy_for_perturbation` (which builds the DCON
energy-minimising interior from a boundary displacement), this evaluates the v2 matrix energy
functional for the *given* radial shape — so an externally-specified mode can be scored (e.g. an
analytic m=2 kink profile, or an ELITE eigenfunction converted to GPEC `(ψ, m)`). The surface
variable is minimised analytically (Ξ_s = -A⁻¹(BΞ_ψ'+CΞ_ψ)); Ξ_ψ' is obtained by per-harmonic
cubic-spline differentiation of the input in ψ.

For an exact Euler-Lagrange solution this returns the true δW; for any other trial shape it is the
variational (≥ minimum) energy of that shape, in the same DCON normalisation as
`integrate_energy_v2`'s `dW_volume`. Returns `(; psi, density, dW)`.
"""
function dW_of_shape(equil::Equilibrium.PlasmaEquilibrium, intr::ForceFreeStatesInternal, ffit::FourFitVars, psi_grid::AbstractVector, xi_psi_modes::AbstractMatrix; de=nothing)
    n = intr.numpert_total
    np = length(psi_grid)
    @assert size(xi_psi_modes, 1) == np "xi_psi_modes must be [length(psi_grid), mpert]"
    @assert size(xi_psi_modes, 2) == n "xi_psi_modes must have numpert_total columns"
    if de === nothing
        metric = ForceFreeStates.make_metric(equil; mband=intr.mband)
        de = build_tex_de_matrices(equil, intr, metric)
    end
    psi = collect(float.(psi_grid))

    # Radial derivative Ξ_ψ'(ψ, m) by per-harmonic cubic spline.
    dxi = Matrix{ComplexF64}(undef, np, n)
    for j in 1:n
        cj = cubic_interp(psi, ComplexF64.(@view xi_psi_modes[:, j]); extrap=ExtendExtrap())
        for i in 1:np
            dxi[i, j] = cj(psi[i]; deriv=DerivOp(1))
        end
    end

    density = Vector{ComplexF64}(undef, np)
    A = Matrix{ComplexF64}(undef, n, n)
    B = similar(A); C = similar(A); D = similar(A); E = similar(A); H = similar(A)
    hint = Ref(1)
    for i in 1:np
        p = psi[i]
        A .= reshape(ffit.amats(p; hint=hint), n, n)
        B .= reshape(ffit.bmats(p; hint=hint), n, n)
        C .= reshape(ffit.cmats(p; hint=hint), n, n)
        H .= reshape(ffit.hmats(p; hint=hint), n, n)
        D .= reshape(de.dmats(p; hint=hint), n, n)
        E .= reshape(de.emats(p; hint=hint), n, n)

        xpsi = ComplexF64.(@view xi_psi_modes[i, :])
        xpsi1 = @view dxi[i, :]
        bcx = B * xpsi1 + C * xpsi
        xs = -(A \ bcx)                       # minimising surface variable
        t1 = dot(xs, A, xs)
        t2 = dot(xs, bcx)
        t3 = dot(xpsi1, D, xpsi1)
        t4 = dot(xpsi1, E, xpsi)
        t5 = dot(xpsi, H, xpsi)
        density[i] = t1 + (t2 + conj(t2)) + t3 + (t4 + conj(t4)) + t5
    end
    dW = _spline_int(psi, density)
    return (; psi, density, dW)
end

# ============================  v1 real-space  ============================

# d/dθ of a periodic θ-grid function via a periodic cubic spline (matches the Fortran
# gpeq_shear spline derivative; avoids the Gibbs ringing of a global spectral derivative).
function _dtheta_periodic(thetas::AbstractVector, f::AbstractVector)
    itp = cubic_interp(collect(thetas), collect(f); bc=PeriodicBC(; endpoint=:exclusive, period=1.0))
    return [itp(θ; deriv=DerivOp(1)) for θ in thetas]
end

"""
    reconstruct_energy_realspace(equil, ffs_intr, ffit, odet, metric, mode_vec;
                                 ntheta=max(512,4·mpert), store_maps=false) -> NamedTuple

v1 real-space reconstruction of the energy-principle density for boundary displacement
`mode_vec`. Returns per-surface `psi`, `dW=½(epf−dst)`, `epf`, `dst`, their ψ/θ/ζ and
shear/B²σ²/curvature splits, the spline-integrated `dW_total`, and (when `store_maps`)
`maps` with per-(ψ,θ) `R, Z, C2_density, K, Kxin2_density` for R-Z heatmaps.
"""
function reconstruct_energy_realspace(equil::Equilibrium.PlasmaEquilibrium, ffs_intr::ForceFreeStatesInternal, ffit::FourFitVars, odet::OdeState, metric::MetricData, mode_vec::AbstractVector; ntheta::Int=max(512, 4 * ffs_intr.mpert), store_maps::Bool=false)
    npsi = size(odet.u_store, 4)
    mpert = ffs_intr.mpert
    mlow = ffs_intr.mlow
    nn = ffs_intr.nlow
    chi1 = 2π * equil.psio
    ro = equil.ro
    zo = equil.zo
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

    # --- θ grid and inverse-DFT phase (mode → θ); fine grid avoids m-dependent quadrature error ---
    nθ = ntheta
    thetas = [(k - 1) / nθ for k in 1:nθ]
    m_vals = [mlow + i - 1 for i in 1:mpert]
    phase = [exp(2π * im * m_vals[i] * thetas[k]) for k in 1:nθ, i in 1:mpert]   # IDFT

    epf = zeros(Float64, npsi)
    dst = zeros(Float64, npsi)
    epf_p = zeros(Float64, npsi); epf_t = zeros(Float64, npsi); epf_z = zeros(Float64, npsi)
    dst1 = zeros(Float64, npsi); dst2 = zeros(Float64, npsi); dst3 = zeros(Float64, npsi)

    # Optional 2D (ψ,θ) maps for R-Z heatmaps.
    Rmap = store_maps ? Matrix{Float64}(undef, npsi, nθ) : nothing
    Zmap = store_maps ? Matrix{Float64}(undef, npsi, nθ) : nothing
    C2map = store_maps ? Matrix{Float64}(undef, npsi, nθ) : nothing
    Kmap = store_maps ? Matrix{Float64}(undef, npsi, nθ) : nothing
    KXmap = store_maps ? Matrix{Float64}(undef, npsi, nθ) : nothing

    # per-θ scratch
    delpsi2 = zeros(Float64, nθ); dpdt = zeros(Float64, nθ); dpdz = zeros(Float64, nθ)
    g22 = zeros(Float64, nθ); g23 = zeros(Float64, nθ); g33 = zeros(Float64, nθ)
    jacv = zeros(Float64, nθ); bsq = zeros(Float64, nθ); bsq_psi = zeros(Float64, nθ); bsq_theta = zeros(Float64, nθ)
    rvec = zeros(Float64, nθ); zvec = zeros(Float64, nθ); comp5 = zeros(Float64, nθ)

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

            v21 = fy1 / (2 * rfac); v22 = (1.0 + fy2) * 2π * rfac; v23 = fy3 * r; v33 = 2π * r
            g22[k] = v21^2 + v22^2 + v23^2
            g33[k] = v33^2
            g23[k] = v23 * v33
            jacv[k] = jac
            rvec[k] = r
            zvec[k] = zo + rfac * sin(eta)

            B = equil.eqfun_B(pt; hint=hint)
            Bx = equil.eqfun_B(pt; deriv=DerivOp(1, 0), hint=hint)
            By = equil.eqfun_B(pt; deriv=DerivOp(0, 1), hint=hint)
            bsq[k] = B^2
            bsq_psi[k] = 2 * B * Bx
            bsq_theta[k] = 2 * B * By

            comp5[k] = (q * dpdt[k] - dpdz[k]) / delpsi2[k]   # gpeq_shear component 5
        end

        dcomp5 = _dtheta_periodic(thetas, comp5)   # d/dθ of component 5 (gpeq_shear)

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

            if store_maps
                Rmap[ipsi, k] = rvec[k]
                Zmap[ipsi, k] = zvec[k]
                C2map[ipsi, k] = ep_p + ep_t + ep_z        # |C|²/μ₀ density
                Kmap[ipsi, k] = K_t1 + K_t2 + K_t3         # K kernel
                KXmap[ipsi, k] = (K_t1 + K_t2 + K_t3) * w  # K|ξ_n|² density
            end
        end
        epf[ipsi] = epf_acc / nθ
        dst[ipsi] = dst_acc / nθ
        epf_p[ipsi] = epfp_acc / nθ; epf_t[ipsi] = epft_acc / nθ; epf_z[ipsi] = epfz_acc / nθ
        dst1[ipsi] = dst1_acc / nθ; dst2[ipsi] = dst2_acc / nθ; dst3[ipsi] = dst3_acc / nθ
    end

    dW = 0.5 .* (epf .- dst)
    dW_total = _spline_int(psi_grid, dW)
    maps = store_maps ? (; psi=psi_grid, theta=thetas, R=Rmap, Z=Zmap, C2_density=C2map, K=Kmap, Kxin2_density=KXmap) : nothing
    return (; psi=psi_grid, dW, epf, dst, dW_total, epf_p, epf_t, epf_z, dst1, dst2, dst3, maps)
end

# ============================  driver + HDF5 writer  ============================

"""
    plasma_energy(vac_data, equil, xi_b) -> Float64

Exact plasma δW for an ARBITRARY boundary perturbation `xi_b` (normal-displacement vector in
the (m) Fourier basis), evaluated directly from the DCON plasma-energy operator:
`δW = psio²·(xi_b† W_p xi_b)` where `W_p = vac_data.wp`. This is the cheap "reverse" δW — it
needs no spatial reconstruction. Every `xi_b` is an admissible perturbation: it maps to the
unique regular Euler-Lagrange solution `ξ_ψ(ψ) = u_store(ψ)·(u_store(edge)⁻¹ xi_b)`, since the
stored fundamental solutions span all regular EL solutions. The free-boundary eigenmodes
`vac_data.wt[:,k]` are the special perturbations that diagonalise `W_p` (giving `ep[k]`).
"""
plasma_energy(vac_data::VacuumData, equil::Equilibrium.PlasmaEquilibrium, xi_b::AbstractVector) =
    equil.psio^2 * real(dot(Vector{ComplexF64}(xi_b), vac_data.wp * Vector{ComplexF64}(xi_b)))

"""
    energy_for_perturbation(equil, ffs_intr, ffit, odet, vac_data, xi_b; store_maps=false, ntheta=..., de=nothing, metric=nothing) -> NamedTuple

Reverse-direction reconstruction: for an ARBITRARY perturbation `xi_b` (boundary
normal-displacement, (m) basis), reconstruct the regular Euler-Lagrange solution it generates
and its energy principle both ways (v1 real-space, v2 matrix). The exact plasma δW is
`dW = v2.dW_boundary = psio²·(xi_b† W_p xi_b)` (= `plasma_energy(vac_data, equil, xi_b)`).
Pass prebuilt `metric`/`de` to reuse across a scan over many perturbations.
Returns `(; v1, v2, xi_b, dW)`.
"""
function energy_for_perturbation(equil::Equilibrium.PlasmaEquilibrium, ffs_intr::ForceFreeStatesInternal, ffit::FourFitVars, odet::OdeState, vac_data::VacuumData, xi_b::AbstractVector; store_maps::Bool=false, ntheta::Int=max(512, 4 * ffs_intr.mpert), de=nothing, metric=nothing)
    m = metric === nothing ? ForceFreeStates.make_metric(equil; mband=ffs_intr.mband) : metric
    v2 = integrate_energy_v2(equil, ffs_intr, ffit, odet, xi_b; de=de)
    v1 = reconstruct_energy_realspace(equil, ffs_intr, ffit, odet, m, xi_b; ntheta=ntheta, store_maps=store_maps)
    return (; v1, v2, xi_b=Vector{ComplexF64}(xi_b), dW=real(v2.dW_boundary))
end

"""
    reconstruct_energy_principle(equil, ffs_intr, ffit, odet, vac_data; mode_index=1, perturbation=nothing, store_maps=true, ntheta=...) -> NamedTuple

Run both reconstructions for one perturbation. By default uses the `mode_index`-th
free-boundary eigenmode (`vac_data.wt[:,mode_index]`); pass `perturbation` (an arbitrary
boundary displacement vector) to reconstruct an arbitrary EL-satisfying perturbation instead
(see `energy_for_perturbation`). Returns `(; v1, v2, mode_index, xi_b)`.
"""
function reconstruct_energy_principle(equil::Equilibrium.PlasmaEquilibrium, ffs_intr::ForceFreeStatesInternal, ffit::FourFitVars, odet::OdeState, vac_data::VacuumData; mode_index::Int=1, perturbation::Union{Nothing,AbstractVector}=nothing, store_maps::Bool=true, ntheta::Int=max(512, 4 * ffs_intr.mpert))
    xi_b = perturbation === nothing ? vac_data.wt[:, mode_index] : perturbation
    res = energy_for_perturbation(equil, ffs_intr, ffit, odet, vac_data, xi_b; store_maps=store_maps, ntheta=ntheta)
    return (; res.v1, res.v2, mode_index, res.xi_b)
end

"""
    write_recon_group!(f, equil, ffs_intr, rec, vac_data)

Write the energy-principle reconstruction `rec` (from `reconstruct_energy_principle`) into
an open HDF5 handle `f` under the `recon/` group: v1 densities (`epf`, `dst`) with their
ψ/θ/ζ and shear/B²σ²/curvature splits, `dW_total`, v2 `dW_volume`/`dW_boundary`, the
equilibrium context, and (when present) the 2D (ψ,θ) `maps/` for R-Z heatmaps.
"""
function write_recon_group!(f, equil::Equilibrium.PlasmaEquilibrium, ffs_intr::ForceFreeStatesInternal, rec, vac_data::VacuumData)
    v1 = rec.v1
    v2 = rec.v2
    f["recon/mode_index"] = rec.mode_index
    f["recon/psi"] = v1.psi
    f["recon/dW"] = v1.dW
    f["recon/epf"] = v1.epf
    f["recon/dst"] = v1.dst
    f["recon/epf_psi"] = v1.epf_p
    f["recon/epf_theta"] = v1.epf_t
    f["recon/epf_zeta"] = v1.epf_z
    f["recon/dst_shear"] = v1.dst1
    f["recon/dst_bsq_sigma2"] = v1.dst2
    f["recon/dst_curvature"] = v1.dst3
    f["recon/dW_total_v1"] = v1.dW_total
    f["recon/dW_volume_v2"] = real(v2.dW_volume)
    f["recon/dW_boundary_v2"] = real(v2.dW_boundary)
    f["recon/psio"] = equil.psio
    f["recon/ep1"] = real(vac_data.ep[1])
    f["recon/et1"] = real(vac_data.et[1])
    # Boundary perturbation used + the plasma-energy operator, so δW for ANY perturbation ξ_b
    # is reproducible offline as psio²·(ξ_b† W_plasma ξ_b) (the reverse-direction energy).
    haskey(rec, :xi_b) && (f["recon/xi_b"] = rec.xi_b)
    f["recon/W_plasma"] = vac_data.wp
    if v1.maps !== nothing
        f["recon/maps/psi"] = v1.maps.psi
        f["recon/maps/theta"] = v1.maps.theta
        f["recon/maps/R"] = v1.maps.R
        f["recon/maps/Z"] = v1.maps.Z
        f["recon/maps/C2_density"] = v1.maps.C2_density
        f["recon/maps/K"] = v1.maps.K
        f["recon/maps/Kxin2_density"] = v1.maps.Kxin2_density
    end
    return nothing
end
