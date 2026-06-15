"""
EnergyReconstruction.jl — v2 (matrix-form) reconstruction of the ideal MHD
energy principle δW.

Follows `docs/tex/recon/main.tex` §"Reconstruction v2: Matrix representation of
δW" (Sunjae Lee) and the Fortran DCON `recon` flag (`gpout_recon` in
`~/GPEC/gpec/gpout.f`). The Bernstein energy is written in the DCON/Newcomb
radial matrix structure

    2μ₀δW = ∫dψ [ Ξ_s†AΞ_s + Ξ_s†(BΞ_ψ'+CΞ_ψ) + (Ξ_ψ'†B†+Ξ_ψ†C†)Ξ_s
                + Ξ_ψ'†DΞ_ψ' + (Ξ_ψ'†EΞ_ψ + Ξ_ψ†E†Ξ_ψ') + Ξ_ψ†HΞ_ψ ]   (recon eq. for 2μ₀δW)

with the tex coefficient matrices A,B,C,D,E,H assembled from the Fourier-projected
metric Gᵢⱼ = ⟨gᵢⱼ/J²⟩ (= `metric.fourier_coeffs`) and the equilibrium profiles.

Correspondence to `make_matrix` (Fourfit.jl), using χ' = 2π·psio = const (so χ''=0)
and the code convention 2πf' ↔ `F_deriv`:
  - A = `ffit.amats`, B = `ffit.bmats`, C = `ffit.cmats`, H = `ffit.hmats`  (identical).
  - D, E are **not** `ffit.dmats_prim`/`emats_prim` — those are Glasser-2016 A7/A8
    subterms in a different (equally valid) decomposition whose surface variable is
    `-A⁻¹(D_glasser Ξ_ψ' + C Ξ_ψ)`, stored in `ud_store[:,:,2,:]`. The tex D, E are
    built here by `build_tex_de_matrices`.

The tex surface variable Ξ_s is the energy-minimizer of the tex form,
Ξ_s = -A⁻¹(BΞ_ψ'+CΞ_ψ); it is computed analytically here rather than read from
storage. The physical normal displacement Ξ_ψ and its derivative Ξ_ψ' (the
convention-clean component-1 quantities) are read from `u_store`/`ud_store`. The
ψ-integral of the form equals the DCON boundary-term plasma energy Ξ_ψ(edge)†·u₂(edge).
"""

"""
    build_tex_de_matrices(equil, intr, metric) -> NamedTuple

Build the tex `Reconstruction v2` D and E coefficient matrices as cubic series
interpolants over the metric ψ-grid, mirroring the element-wise assembly of
`make_matrix` (Fourfit.jl). With χ' = 2π·psio = const,

  D[m,m'] = χ'²(g₂₂ + 2q g₂₃ + q² g₃₃)
  E[m,m'] = χ'²q'(g₂₃ + q g₃₃) - 2πi χ'²(g₁₂ + q g₃₁)(m'-nq) + μ₀p' ⟨1⟩

where gᵢⱼ here are the Fourier coefficients ⟨gᵢⱼ/J²⟩ at Fourier index dm=m'-m.

### Returns
  - `dmats`, `emats` — `CubicSeriesInterpolant`s of the flattened (numpert_total²) matrices over ψ.
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

Reconstruct the v2 energy-principle integrand `2μ₀ dδW/dψ` over the stored ψ grid
for the boundary displacement `mode_vec` (a normal-displacement vector Ξ_ψ at the
plasma edge, in the same mode basis as `vac_data.wt`).

The solution coefficient `alpha` is chosen so the reconstructed Ξ_ψ at the edge
equals `mode_vec`; Ξ_ψ and Ξ_ψ' are then read from the stored fundamental-solution
basis (`u_store[:,:,1,:]`, `ud_store[:,:,1,:]`) and the tex surface variable is
formed as Ξ_s = -A⁻¹(BΞ_ψ'+CΞ_ψ). Pass a prebuilt `de` (from
`build_tex_de_matrices`) to avoid recomputing the metric on repeated calls.

### Returns
  - `psi::Vector{Float64}` — stored flux grid (1:`odet.step`).
  - `density::Vector{ComplexF64}` — integrand `2μ₀ dδW/dψ` at each ψ.
"""
function reconstruct_energy_density_v2(equil::Equilibrium.PlasmaEquilibrium, intr::ForceFreeStatesInternal, ffit::FourFitVars, odet::OdeState, mode_vec::AbstractVector; de=nothing)
    n = intr.numpert_total
    nstep = odet.step
    if de === nothing
        metric = make_metric(equil; mband=intr.mband)
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

Integrate the v2 density (trapezoidal over the stored ψ grid) and also return the
exact DCON boundary-term energy for the same mode, `Ξ_ψ(edge)†·u₂(edge)`, which the
ψ-integral should reproduce up to quadrature error.

### Returns
  - `dW_volume::ComplexF64` — ∫ density dψ (the reconstructed 2μ₀δW).
  - `dW_boundary::ComplexF64` — boundary-term energy Ξ_ψ(edge)†·u₂(edge).
  - `density`, `psi` — as in `reconstruct_energy_density_v2`.
"""
function integrate_energy_v2(equil::Equilibrium.PlasmaEquilibrium, intr::ForceFreeStatesInternal, ffit::FourFitVars, odet::OdeState, mode_vec::AbstractVector; de=nothing)
    rec = reconstruct_energy_density_v2(equil, intr, ffit, odet, mode_vec; de=de)
    dW_volume = _trapz(rec.psi, rec.density)

    nstep = odet.step
    u1_edge = @view odet.u_store[:, :, 1, nstep]
    u2_edge = @view odet.u_store[:, :, 2, nstep]
    alpha = u1_edge \ Vector{ComplexF64}(mode_vec)
    dW_boundary = dot(u1_edge * alpha, u2_edge * alpha)

    return (; dW_volume, dW_boundary, rec.density, rec.psi)
end

# Trapezoidal integral of complex y over a (possibly nonuniform) grid x.
function _trapz(x::AbstractVector{<:Real}, y::AbstractVector{<:Number})
    s = zero(eltype(y))
    @inbounds for i in 1:(length(x)-1)
        s += (y[i] + y[i+1]) * (x[i+1] - x[i]) / 2
    end
    return s
end
