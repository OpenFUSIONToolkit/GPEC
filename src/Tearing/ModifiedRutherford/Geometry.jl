# Geometry.jl
#
# Flux-surface averages the modified Rutherford terms need beyond `ResistGeometry`: the Hegna 1999 R̄²,
# poloidal-field averages, the mean minor radius and the Sauter 2002 trapped fraction, as in Fortran RDCON `mercier.f`.
#
# θ-integrands, each weighted by `jac / v1` before integration:
#   1: B²/|∇ψ|²   2: B²   3: |∇ψ|²   4: R² v1/jac   5: B_pol   6: B_pol²   7: r (minor radius)

"""
    MRESurfaceGeometry

Per-surface geometry of the modified Rutherford terms. Averages ⟨·⟩ are normalized flux-surface averages (⟨1⟩ = 1)
and `ψ_N` is normalized poloidal flux, as in [`ResistGeometry`](@ref GeneralizedPerturbedEquilibrium.ForceFreeStates.ResistGeometry).

| field                    | meaning                                                             |
|:------------------------ |:------------------------------------------------------------------- |
| `q`, `q1`                | safety factor and dq/dψ_N                                           |
| `p1`                     | μ₀ dp/dψ_N                                                          |
| `v1`                     | dV/dψ_N [m³]                                                        |
| `psio`                   | poloidal flux per radian at the boundary [Wb]                       |
| `avg_bsq_over_dpsisq`    | ⟨B²/‖∇ψ_N‖²⟩                                                        |
| `avg_bsq`                | ⟨B²⟩ [T²]                                                           |
| `avg_dpsisq`             | ⟨‖∇ψ_N‖²⟩                                                           |
| `overbar_Rsq`            | Hegna 1999 R̄² = ⟨R² V′/𝒥⟩ [m²]                                     |
| `avg_bpol`, `avg_bpolsq` | ⟨B_pol⟩ [T] and ⟨B_pol²⟩ [T²]                                       |
| `r_minor`                | ⟨r⟩, mean distance from the magnetic axis [m]                       |
| `R_major`, `eps`         | (R_max + R_min)/2 [m] and (R_max − R_min)/(R_max + R_min)           |
| `delta`                  | triangularity, mean of the upper and lower values                   |
| `f_trap`                 | Sauter 2002 trapped fraction with the triangularity-corrected ε_eff |
| `Wc_prefac`              | Schlutt & Hegna 2012 w_d⁴ prefactor in ψ_N⁴, before χ⊥/χ∥ and 1/m²  |
"""
struct MRESurfaceGeometry
    q::Float64
    q1::Float64
    p1::Float64
    v1::Float64
    psio::Float64
    avg_bsq_over_dpsisq::Float64
    avg_bsq::Float64
    avg_dpsisq::Float64
    overbar_Rsq::Float64
    avg_bpol::Float64
    avg_bpolsq::Float64
    r_minor::Float64
    R_major::Float64
    eps::Float64
    delta::Float64
    f_trap::Float64
    Wc_prefac::Float64
end

"""
    sauter2002_trapped_fraction(eps, delta) -> Float64

Trapped fraction of Sauter 2002 (as Fortran RDCON uses it): ε_eff = max(0, 0.67(1 − 1.4δ|δ|)ε) and
`f_t = 1 − √((1 − ε)/(1 + ε)) (1 − ε_eff)/(1 + 2√ε_eff)`, capped at 1.
"""
function sauter2002_trapped_fraction(eps::Real, delta::Real)
    eps_eff = max(0.0, 0.67 * (1 - 1.4 * delta * abs(delta)) * eps)
    return min(1 - sqrt((1 - eps) / (1 + eps)) * (1 - eps_eff) / (1 + 2 * sqrt(eps_eff)), 1.0)
end

"""
    bootstrap_coefficient(g::MRESurfaceGeometry, zeff) -> (D_nc, mu_ratio)

Bootstrap drive coefficient of Hegna 1999 as Fortran RDCON evaluates it,
`D_nc = −(q/q′)(μ₀p′/⟨B²⟩) R̄² ⟨B²/‖∇ψ_N‖²⟩ ψ_o⁻² · μ_e/(ν_e + μ_e)`, with the banana-limit viscosity ratio
`μ_e/(ν_e + μ_e) = f_t a/((1 − f_t) + f_t a)`, `a = 1 + 0.533/Z_eff`.
"""
function bootstrap_coefficient(g::MRESurfaceGeometry, zeff::Real)
    a = 1 + 0.533 / zeff
    mu_ratio = g.f_trap * a / ((1 - g.f_trap) + g.f_trap * a)
    return -g.q * (g.p1 / (g.q1 * g.avg_bsq)) * g.overbar_Rsq * g.avg_bsq_over_dpsisq / g.psio^2 * mu_ratio, mu_ratio
end

"""
    mre_geometry(equil, psifac, q1) -> MRESurfaceGeometry

Flux-surface geometry of the modified Rutherford terms at the interior surface `psifac` with dq/dψ_N `q1`.
"""
function mre_geometry(equil, psifac::Real, q1::Real)
    profiles = equil.profiles
    chi1 = 2π * equil.psio
    psi = Float64(psifac)
    twopif = profiles.F_spline(psi)
    p1 = profiles.P_deriv(psi)
    v1 = profiles.dVdpsi_spline(psi)
    q = profiles.q_spline(psi)

    ys = equil.rzphi_ys
    ff = zeros(Float64, length(ys), 7)
    R_max, R_min, Z_max, Z_min = -Inf, Inf, -Inf, Inf
    R_at_Zmax = R_at_Zmin = equil.ro
    for (i, theta) in enumerate(ys)
        rfac = sqrt(equil.rzphi_rsquared((psi, theta)))
        jac = equil.rzphi_jac((psi, theta))
        fy1 = FastInterpolations.deriv_view(equil.rzphi_rsquared, (0, 1))((psi, theta))
        fy2 = FastInterpolations.deriv_view(equil.rzphi_offset, (0, 1))((psi, theta))
        fy3 = FastInterpolations.deriv_view(equil.rzphi_nu, (0, 1))((psi, theta))
        eta = 2π * (theta + equil.rzphi_offset((psi, theta)))
        r = equil.ro + rfac * cos(eta)
        z = equil.zo + rfac * sin(eta)

        v21 = fy1 / (2 * rfac * jac)
        v22 = (1 + fy2) * 2π * rfac / jac
        v23 = fy3 * r / jac
        v33 = 2π * r / jac
        bsq = chi1^2 * (v21^2 + v22^2 + (v23 + q * v33)^2)
        dpsisq = (2π * r)^2 * (v21^2 + v22^2)
        bpolsq = max(bsq - (twopif / (2π * r))^2, 0.0)

        R_max, R_min = max(R_max, r), min(R_min, r)
        z > Z_max && ((Z_max, R_at_Zmax) = (z, r))
        z < Z_min && ((Z_min, R_at_Zmin) = (z, r))
        ff[i, :] .= (bsq / dpsisq, bsq, dpsisq, r^2 * v1 / jac, sqrt(bpolsq), bpolsq, rfac)
        @views ff[i, :] .*= jac / v1
    end
    @views ff[end, :] .= ff[1, :]
    avg = FastInterpolations.integrate(cubic_interp(ys, Series(ff); bc=PeriodicBC()))

    R_major = 0.5 * (R_max + R_min)
    a_minor = 0.5 * (R_max - R_min)
    eps = a_minor / R_major
    delta = 0.5 * ((R_major - R_at_Zmax) + (R_major - R_at_Zmin)) / a_minor
    # Schlutt & Hegna 2012 ⟨𝒥B²⟩⟨𝒥|∇ψ|²⟩/(dι/dψ)² in toroidal flux, converted to ψ_N⁴.
    Wc_prefac = (v1 * avg[2] / (q * equil.psio)) * v1 * avg[3] * (q * equil.psio) * (q^6 / (q1 / equil.psio)^2) / (q * equil.psio)^4
    return MRESurfaceGeometry(q, q1, p1, v1, equil.psio, avg[1], avg[2], avg[3], avg[4], avg[5], avg[6], avg[7],
        R_major, eps, delta, sauter2002_trapped_fraction(eps, delta), Wc_prefac)
end

"""
    current_density_profile(equil) -> Function

Dimensionless toroidal current density `J_z(ψ_N) = (R₀²/B₀) ⟨μ₀ j_φ/R⟩` of the cylindrical saturation theory, from the
Grad–Shafranov flux-surface identity `⟨μ₀ j_φ/R⟩ = μ₀ dp/dψ + F dF/dψ ⟨R⁻²⟩` (ψ the poloidal flux per radian), signed
positive in the core. It reduces to `R₀μ₀j_z/B_z` in the large-aspect-ratio cylinder.
"""
function current_density_profile(equil)
    profiles = equil.profiles
    R0, B0, psio = Float64(equil.ro), Float64(equil.params.b0), Float64(equil.psio)
    ys = equil.rzphi_ys
    function avg_inv_Rsq(psi)
        ff = zeros(Float64, length(ys), 2)
        for (i, theta) in enumerate(ys)
            rfac = sqrt(equil.rzphi_rsquared((psi, theta)))
            jac = equil.rzphi_jac((psi, theta))
            r = equil.ro + rfac * cos(2π * (theta + equil.rzphi_offset((psi, theta))))
            ff[i, 1], ff[i, 2] = jac / r^2, jac
        end
        @views ff[end, :] .= ff[1, :]
        avg = FastInterpolations.integrate(cubic_interp(ys, Series(ff); bc=PeriodicBC()))
        return avg[1] / avg[2]
    end
    raw(psi) = (profiles.P_deriv(psi) + profiles.F_spline(psi) * profiles.F_deriv(psi) / (4π^2) * avg_inv_Rsq(psi)) / psio
    sgn = sign(raw(0.05))
    return psi -> sgn * R0^2 / B0 * raw(Float64(psi))
end
