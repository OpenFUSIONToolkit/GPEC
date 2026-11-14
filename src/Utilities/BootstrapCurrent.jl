"""
    BootstrapCurrent

Flux-surface-averaged bootstrap current ⟨j_bs·B⟩ from the Redl et al. 2021 closure, assembled the way
OpenFUSIONToolkit TokaMaker assembles it: exact Redl coefficients with L34 = L31, Koh ion collisionality,
NRL electron and ion Coulomb logarithms floored at 10, and the Sauter circulating fraction from surface moments of |B|.

Conventions: ψ is the poloidal flux per radian [Wb/rad] increasing from the axis to the edge, so ψ = psio ψ_N and the
gradients passed to [`jdotB_bootstrap`](@ref) are d/dψ of profiles that fall outward. I(ψ) = R B_φ is signed and
⟨j_bs·B⟩ = −I p_e [L31 p′/p_e + L32 T_e′/T_e + L34 α (1 − R_pe)/R_pe T_i′/T_i] is co-directed with the equilibrium
⟨j·B⟩ = −(F p′ + F′⟨B²⟩/μ₀) of the same convention. Temperatures are in eV, densities in m⁻³, lengths in m.

References: Redl et al., Phys. Plasmas 28, 022502 (2021); Sauter et al., Phys. Plasmas 6, 2834 (1999).
"""
module BootstrapCurrent

using FastInterpolations
using ..NeoclassicalResistivity: coulomb_log_e

export redl_coefficients, coulomb_log_i, nu_star_i_koh, nu_star_e_floored, circulating_fraction, jdotB_bootstrap, surface_moments, bootstrap_current

# Elementary charge as used by TokaMaker (CODATA 2018)
const E_CHARGE = 1.602176634e-19

"""
    redl_coefficients(f_t, Z_eff, nu_e_star, nu_i_star) -> (; L31, L32, L34, alpha)

Redl 2021 Eqs. 10-21 with L34 = L31, as in TokaMaker `redl_bootstrap`.
"""
function redl_coefficients(f_t::Real, Z_eff::Real, nu_e_star::Real, nu_i_star::Real)
    fT, Z, nue, nui = f_t, Z_eff, nu_e_star, nu_i_star
    sZ1 = sqrt(max(Z - 1, 0.0))
    # L31 (Eqs. 10-11)
    d1 = 0.67 * (1 - 0.7fT) * sqrt(nue) / (0.56 + 0.44Z)
    d2 = (0.52 + 0.086 * sqrt(nue)) * (1 + 0.87fT) * nue / (1 + 1.13sZ1)
    X31 = fT / (1 + d1 + d2)
    dZ = Z^1.2 - 0.71
    L31 = (1 + 0.15 / dZ) * X31 - (0.22 / dZ) * X31^2 + (0.01 / dZ) * X31^3 + (0.06 / dZ) * X31^4
    # L32 (Eqs. 12-16)
    dee2 = 0.23 * (1 - 0.96fT) * sqrt(nue) / sqrt(Z)
    dee3 = 0.13 * (1 - 0.38fT) * nue / Z^2 * (sqrt(1 + 2sZ1) + fT^2 * sqrt((0.075 + 0.25 * (Z - 1)^2) * nue))
    Xee = fT / (1 + dee2 + dee3)
    Fee = (0.1 + 0.6Z) / (Z * (0.77 + 0.63 * (1 + max(Z - 1, 0.0)^1.1))) * (Xee - Xee^4) +
          0.7 / (1 + 0.2Z) * (Xee^2 - Xee^4 - 1.2 * (Xee^3 - Xee^4)) + 1.3 / (1 + 0.5Z) * Xee^4
    dei2 = 0.87 * (1 + 0.39fT) * sqrt(nue) / (1 + 2.95 * (Z - 1)^2)
    dei3 = 1.53 * (1 - 0.37fT) * nue * (2 + 0.375 * (Z - 1))
    Xei = fT / (1 + dei2 + dei3)
    Fei = -(0.4 + 1.93Z) / (Z * (0.8 + 0.6Z)) * (Xei - Xei^4) + 5.5 / (1.5 + 2Z) * (Xei^2 - Xei^4 - 0.8 * (Xei^3 - Xei^4)) - 1.3 / (1 + 0.5Z) * Xei^4
    L32 = Fee + Fei
    # α (Eqs. 20-21)
    alpha0 = -(0.62 + 0.055 * (Z - 1)) / (0.53 + 0.17 * (Z - 1)) * (1 - fT) / (1 - (0.31 - 0.065 * (Z - 1)) * fT - 0.25fT^2)
    alpha = ((alpha0 + 0.7Z * sqrt(fT) * sqrt(nui)) / (1 + 0.18 * sqrt(nui)) - 0.002 * nui^2 * fT^6) / (1 + 0.004 * nui^2 * fT^6)
    return (; L31, L32, L34=L31, alpha)
end

"""
    coulomb_log_i(n_e, n_i, T_i) -> Float64

Ion-ion Coulomb logarithm `30 − ln(Z_avg³ √n_i / T_i^1.5)` with `Z_avg = max(n_e/n_i, 1)`, floored at 10 (TokaMaker form).
`n` in m⁻³, `T_i` in eV.
"""
function coulomb_log_i(n_e::Real, n_i::Real, T_i::Real)
    Z_avg = max(n_e / n_i, 1.0)
    return max(30.0 - log(Z_avg^3 * sqrt(n_i) / T_i^1.5), 10.0)
end

"""
    nu_star_e_floored(n_e, T_e, R_geo, eps, q, Z_eff) -> Float64

Electron collisionality of `nu_star_e` with the NRL Coulomb logarithm floored at 10, as TokaMaker.
"""
function nu_star_e_floored(n_e::Real, T_e::Real, R_geo::Real, eps::Real, q::Real, Z_eff::Real)
    lnL = max(coulomb_log_e(n_e, T_e), 10.0)
    return 6.921e-18 * abs(q) * R_geo * n_e * Z_eff * lnL / (T_e^2 * eps^1.5)
end

"""
    nu_star_i_koh(n_e, n_i, T_i, R_geo, eps, q, Z_eff; Z_dom=1) -> Float64

Ion collisionality, Koh form: `4.90e-18 |q| R n_i Z_ion⁴ lnΛ_ii / (T_i² ε^1.5)` with
`Z_ion = (Z_dom² Z_avg Z_eff)^(1/4)` and `Z_avg = n_e/n_i`.
"""
function nu_star_i_koh(n_e::Real, n_i::Real, T_i::Real, R_geo::Real, eps::Real, q::Real, Z_eff::Real; Z_dom::Real=1)
    Z_ion4 = Z_dom^2 * (n_e / n_i) * Z_eff
    return 4.90e-18 * abs(q) * R_geo * n_i * Z_ion4 * coulomb_log_i(n_e, n_i, T_i) / (T_i^2 * eps^1.5)
end

"""
    circulating_fraction(B, weights) -> Float64

Circulating fraction `f_c = 1 − (3/4 f_tu + 1/4 f_tl)` of Sauter 1999 as TokaMaker evaluates it from moments of |B| on a surface,
with `f_tu = 1 − ⟨B²⟩/⟨B⟩² (1 − √(1−h)(1+h/2))`, `f_tl = 1 − ⟨B²⟩/B_max² ⟨(1 − √(1−b)(1+b/2))/b²⟩`, `h = ⟨B⟩/B_max`, `b = B/B_max`.
`B` samples |B| on a periodic uniform θ grid (no repeated endpoint) and `weights` are the flux-surface-average weights (the Jacobian).
"""
function circulating_fraction(B::AbstractVector{<:Real}, weights::AbstractVector{<:Real})
    wsum = sum(weights)
    Bmax = maximum(B)
    avgB = sum(weights .* B) / wsum
    avgB2 = sum(weights .* B .^ 2) / wsum
    h = avgB / Bmax
    h2 = avgB2 / Bmax^2
    ftu = 1 - h2 / h^2 * (1 - sqrt(1 - h) * (1 + h / 2))
    g(b) = (1 - sqrt(1 - b) * (1 + b / 2)) / b^2
    hf = sum(weights .* g.(min.(B ./ Bmax, 1.0))) / wsum
    ftl = 1 - h2 * hf
    return 1 - (0.75ftu + 0.25ftl)
end

"""
    jdotB_bootstrap(; T_e, T_i, n_e, n_i, Z_eff, f_t, nu_e_star, nu_i_star, I_psi, dT_e, dT_i, dn_e, dn_i) -> NamedTuple

TokaMaker `jboot1` assembly of ⟨j_bs·B⟩ [T A/m²]. Temperatures in eV, densities in m⁻³, `I_psi = R B_φ` [T m], and the four
gradients are d/dψ with ψ in Wb/rad increasing outward. Returns `(; jdotB, L31, L32, L34, alpha)`.
"""
function jdotB_bootstrap(; T_e, T_i, n_e, n_i, Z_eff, f_t, nu_e_star, nu_i_star, I_psi, dT_e, dT_i, dn_e, dn_i)
    c = redl_coefficients(f_t, Z_eff, nu_e_star, nu_i_star)
    p_e = E_CHARGE * n_e * T_e
    p_i = E_CHARGE * n_i * T_i
    R_pe = p_e / (p_e + p_i)
    dp = (n_e * dT_e + T_e * dn_e + n_i * dT_i + T_i * dn_i) * E_CHARGE
    bra = c.L31 * dp / p_e + c.L32 * dT_e / T_e + c.L34 * c.alpha * (1 - R_pe) / R_pe * dT_i / T_i
    return (; jdotB=-I_psi * p_e * bra, c.L31, c.L32, c.L34, c.alpha)
end

"""
    surface_moments(equil, psi_N) -> NamedTuple

Geometry of the flux surface `psi_N` of a `PlasmaEquilibrium`, sampled on its θ grid: `B` (|B| per θ), `weights` (Jacobian), `R`,
and `q`, `F = R B_φ`, `R_max`, `R_min`, `R_geo = (R_max + R_min)/2`, `eps = (R_max − R_min)/(R_max + R_min)`, `avg_B2 = ⟨B²⟩`,
`avg_invR = ⟨1/R⟩`, `avg_invR2 = ⟨1/R²⟩`. Hamada-style metric as in the equilibrium construction; `profiles.F_spline` is 2πF.
"""
function surface_moments(equil, psi_N::Real)
    profiles = equil.profiles
    chi1 = 2π * equil.psio
    psi = Float64(psi_N)
    q = profiles.q_spline(psi)
    F = profiles.F_spline(psi) / 2π
    ys = equil.rzphi_ys[1:(end-1)]
    n = length(ys)
    B = zeros(n)
    w = zeros(n)
    R = zeros(n)
    for (i, theta) in enumerate(ys)
        rfac = sqrt(equil.rzphi_rsquared((psi, theta)))
        jac = equil.rzphi_jac((psi, theta))
        fy1 = FastInterpolations.deriv_view(equil.rzphi_rsquared, (0, 1))((psi, theta))
        fy2 = FastInterpolations.deriv_view(equil.rzphi_offset, (0, 1))((psi, theta))
        fy3 = FastInterpolations.deriv_view(equil.rzphi_nu, (0, 1))((psi, theta))
        eta = 2π * (theta + equil.rzphi_offset((psi, theta)))
        r = equil.ro + rfac * cos(eta)
        v21 = fy1 / (2 * rfac * jac)
        v22 = (1 + fy2) * 2π * rfac / jac
        v23 = fy3 * r / jac
        v33 = 2π * r / jac
        B[i] = chi1 * sqrt(v21^2 + v22^2 + (v23 + q * v33)^2)
        w[i] = jac
        R[i] = r
    end
    wsum = sum(w)
    avg(x) = sum(w .* x) / wsum
    R_max, R_min = maximum(R), minimum(R)
    return (; B, weights=w, R, q, F, R_max, R_min, R_geo=(R_max + R_min) / 2, eps=(R_max - R_min) / (R_max + R_min),
        avg_B2=avg(B .^ 2), avg_invR=avg(1 ./ R), avg_invR2=avg(1 ./ R .^ 2))
end

# Second-order differences on a non-uniform grid, as TokaMaker's gradient_ (numpy.gradient, edge_order=2)
function _gradient(x::AbstractVector, y::AbstractVector)
    n = length(x)
    d = zeros(n)
    for i in 2:(n-1)
        hl, hr = x[i] - x[i-1], x[i+1] - x[i]
        d[i] = (hl^2 * y[i+1] - hr^2 * y[i-1] + (hr^2 - hl^2) * y[i]) / (hl * hr * (hl + hr))
    end
    h1, h2 = x[2] - x[1], x[3] - x[2]
    d[1] = -(2h1 + h2) / (h1 * (h1 + h2)) * y[1] + (h1 + h2) / (h1 * h2) * y[2] - h1 / (h2 * (h1 + h2)) * y[3]
    h1, h2 = x[n-1] - x[n-2], x[n] - x[n-1]
    d[n] = h2 / (h1 * (h1 + h2)) * y[n-2] - (h1 + h2) / (h1 * h2) * y[n-1] + (2h2 + h1) / (h2 * (h1 + h2)) * y[n]
    return d
end

"""
    bootstrap_current(equil, kinetic=equil.kinetic; psi_N=range(0.05, 0.95; length=91), Z_dom=1) -> NamedTuple

Bootstrap current profiles on `psi_N` for a `PlasmaEquilibrium` and its `KineticProfileData` (needs `n_e`, `n_i`, `T_e`, `T_i`; `Z_eff` is the
`zeff` profile, else quasi-neutrality with one impurity of `impurity_charge` and a charge-1 main ion). The kinetic columns are cubic-spline interpolated onto `psi_N` and differentiated by
second-order finite differences in ψ = psio ψ_N, as TokaMaker does, so a fine `psi_N` grid is needed for smooth gradients.
`psi_N` must hold at least three strictly increasing points inside (0, 1).
Returns `psi_N`, `jdotB` (⟨j_bs·B⟩ [T A/m²]), `jphi_bs = ⟨j_bs·B⟩ F⟨1/R⟩/⟨B²⟩` [A/m²], `f_t`, `nu_e_star`, `nu_i_star`, `eps`, `R_geo`, `q`, `F`,
`avg_B2`, `avg_invR`, `avg_invR2`, `L31`, `L32`, `L34`, `alpha`. `jphi_bs` is the field-aligned part only; TokaMaker's `jphi_bs`
also adds the pressure-driven term P′(⟨R⟩ − F²⟨1/R⟩/⟨B²⟩).
"""
function bootstrap_current(equil, kinetic=equil.kinetic; psi_N=range(0.05, 0.95; length=91), Z_dom::Real=1)
    kinetic === nothing && throw(ArgumentError("bootstrap_current: no kinetic profiles attached"))
    for name in (:n_e, :n_i, :T_e, :T_i)
        getfield(kinetic, name) === nothing && throw(ArgumentError("bootstrap_current: kinetic profile $name is missing"))
    end
    x = collect(Float64, psi_N)
    (length(x) >= 3 && all(>(0), diff(x)) && 0 < x[1] && x[end] < 1) ||
        throw(ArgumentError("bootstrap_current: psi_N needs at least three strictly increasing points inside (0, 1)"))
    interp(y) = (itp = cubic_interp(kinetic.psi, y); [itp(xi) for xi in x])
    ne, ni, Te, Ti = interp(kinetic.n_e), interp(kinetic.n_i), interp(kinetic.T_e), interp(kinetic.T_i)
    # Z_eff as the inner layers take it: the file's profile, else quasi-neutrality with one impurity and a charge-1 main ion.
    # An all-zero `zeff` column counts as absent.
    Zeff = if kinetic.zeff !== nothing && !all(iszero, kinetic.zeff)
        minimum(kinetic.zeff) >= 1 - 1e-8 || throw(ArgumentError("bootstrap_current: zeff = $(minimum(kinetic.zeff)) < 1 is unphysical"))
        interp(kinetic.zeff)
    else
        zimp = Float64(kinetic.impurity_charge)
        zimp >= 1 || throw(ArgumentError("bootstrap_current: impurity_charge = $zimp < 1 is unphysical"))
        [max(zimp - (nii / nee) * (zimp - 1), 1.0) for (nii, nee) in zip(ni, ne)]
    end
    psi = equil.psio .* x
    dne, dni, dTe, dTi = _gradient(psi, ne), _gradient(psi, ni), _gradient(psi, Te), _gradient(psi, Ti)
    geo = [surface_moments(equil, xi) for xi in x]
    n = length(x)
    out = (; psi_N=x, jdotB=zeros(n), jphi_bs=zeros(n), f_t=zeros(n), nu_e_star=zeros(n), nu_i_star=zeros(n), eps=[g.eps for g in geo], R_geo=[g.R_geo for g in geo],
        q=[g.q for g in geo], F=[g.F for g in geo], avg_B2=[g.avg_B2 for g in geo], avg_invR=[g.avg_invR for g in geo], avg_invR2=[g.avg_invR2 for g in geo],
        L31=zeros(n), L32=zeros(n), L34=zeros(n), alpha=zeros(n))
    for i in 1:n
        g = geo[i]
        f_t = 1 - circulating_fraction(g.B, g.weights)
        nue = nu_star_e_floored(ne[i], Te[i], g.R_geo, g.eps, g.q, Zeff[i])
        nui = nu_star_i_koh(ne[i], ni[i], Ti[i], g.R_geo, g.eps, g.q, Zeff[i]; Z_dom)
        r = jdotB_bootstrap(; T_e=Te[i], T_i=Ti[i], n_e=ne[i], n_i=ni[i], Z_eff=Zeff[i], f_t, nu_e_star=nue, nu_i_star=nui, I_psi=g.F,
            dT_e=dTe[i], dT_i=dTi[i], dn_e=dne[i], dn_i=dni[i])
        out.jdotB[i] = r.jdotB
        out.jphi_bs[i] = r.jdotB * g.F * g.avg_invR / g.avg_B2
        out.f_t[i], out.nu_e_star[i], out.nu_i_star[i] = f_t, nue, nui
        out.L31[i], out.L32[i], out.L34[i], out.alpha[i] = r.L31, r.L32, r.L34, r.alpha
    end
    return out
end

end # module BootstrapCurrent
