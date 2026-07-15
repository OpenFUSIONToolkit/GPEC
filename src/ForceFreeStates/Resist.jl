# Resist.jl
#
# Per-rational-surface resistive inner-layer parameters (Glasser–Greene–Johnson).
# Julia port of RDCON's `resist_eval` (resist.f): computes the flux-surface-averaged
# coefficients E, F, H, M, G, K and the local time scales τ_A, τ_R at a singular surface.
#
# Returns an `InnerLayer.GGJParameters`, so a resistive ForceFreeStates run can pass its
# per-surface coefficients straight to the GGJ inner solver. (InnerLayer is included first
# so this type is available here.)
#
# Resistivity η and mass density ρ are required per-surface inputs. The Fortran Spitzer block
# (hardcoded ne/Te → η, ρ) is intentionally not ported; the caller supplies η and ρ from its
# own transport/profile model. τ_A and τ_R are built from a purely geometric factor times the
# explicit η and ρ (no RMATCH-style divide-out / re-multiply).
#
# Deliberate differences from the Fortran reference:
#   - Rational surfaces lie between grid nodes (q = m/n), so geometry is evaluated with the
#     callable bicubic interpolants at the exact (ψ_s, θ) — using DerivOp(0,1) for θ-derivatives —
#     rather than the nodal-grid access used by mercier_scan!.
#   - η and ρ are caller-supplied (no Spitzer model, no hardcoded defaults).
#
# Reference: Glasser, Greene & Johnson, Phys. Fluids 18, 875 (1975); Glasser, PoP 23, 072505 (2016).

"""
    resist_eval(sing, equil, intr; eta, rho, gamma, ising=0) -> InnerLayer.GGJParameters

Compute the GGJ resistive inner-layer parameters at the rational surface `sing`.
Port of RDCON `resist_eval` (resist.f), but with η and ρ supplied by the caller (no
built-in Spitzer/density model). Returns an `InnerLayer.GGJParameters` carrying the curvature/mass
coefficients E, F, G, H, K, M, the local time scales τ_A (`taua`), τ_R (`taur`), `v1` (V′ normalized
by total volume), and the surface index `ising`.

# Required keyword arguments
  - `eta::Real` — plasma resistivity η at this surface [Ω·m]
  - `rho::Real` — mass density ρ at this surface [kg/m³]
  - `gamma::Real` — ratio of specific heats (enters G; physical thermodynamic value, e.g. 5/3)

# Physics
The six flux-surface averages (Fortran `avg(1:6)`) are
`⟨B²/|∇ψ|²⟩, ⟨1/|∇ψ|²⟩, ⟨1/B²⟩, ⟨1/(B²|∇ψ|²)⟩, ⟨B²⟩, ⟨|∇ψ|²/B²⟩`, each weighted by `J/V′`
and integrated over θ ∈ [0,1) — exactly the resist.f integrands. The timescales are
`τ_A = √(ρ·M·μ₀) / |2π n q₁ χ₁ / V′|` and `τ_R = (⟨B²/|∇ψ|²⟩/⟨B²⟩)·μ₀/η`, with η and ρ entering
once, explicitly.
"""
function resist_eval(sing::SingType, equil::Equilibrium.PlasmaEquilibrium,
    intr::ForceFreeStatesInternal; eta::Real, rho::Real, gamma::Real, ising::Int=0)

    profiles = equil.profiles
    psifac = sing.psifac
    nn = sing.n[1]

    # --- 1D surface quantities at the (off-grid) rational ψ ---
    hint = Ref(1)
    twopif = profiles.F_spline(psifac; hint=hint)
    p = profiles.P_spline(psifac; hint=hint)
    p1 = profiles.P_deriv(psifac; hint=hint)
    v1 = profiles.dVdpsi_spline(psifac; hint=hint)
    v2 = profiles.dVdpsi_deriv(psifac; hint=hint)
    q = sing.q
    q1 = sing.q1
    chi1 = 2π * equil.psio

    # --- θ-integrate the six geometric averages (resist.f) ---
    nth = length(equil.rzphi_ys)
    ff = zeros(nth, 6)
    hint2d = (Ref(1), Ref(1))
    for itheta in 1:nth
        theta = equil.rzphi_ys[itheta]
        f1 = equil.rzphi_rsquared((psifac, theta); hint=hint2d)
        f2 = equil.rzphi_offset((psifac, theta); hint=hint2d)
        jac = equil.rzphi_jac((psifac, theta); hint=hint2d)
        fy1 = equil.rzphi_rsquared((psifac, theta); deriv=DerivOp(0, 1), hint=hint2d)
        fy2 = equil.rzphi_offset((psifac, theta); deriv=DerivOp(0, 1), hint=hint2d)
        fy3 = equil.rzphi_nu((psifac, theta); deriv=DerivOp(0, 1), hint=hint2d)

        rfac = sqrt(f1)
        eta_geo = 2π * (theta + f2)   # geometric poloidal angle (local; not the resistivity η)
        r = equil.ro + rfac * cos(eta_geo)

        v21 = fy1 / (2.0 * rfac * jac)
        v22 = (1.0 + fy2) * 2π * rfac / jac
        v23 = fy3 * r / jac
        v33 = 2π * r / jac
        bsq = chi1^2 * (v21^2 + v22^2 + (v23 + q * v33)^2)
        dpsisq = (2π * r)^2 * (v21^2 + v22^2)

        ff[itheta, 1] = bsq / dpsisq
        ff[itheta, 2] = 1.0 / dpsisq
        ff[itheta, 3] = 1.0 / bsq
        ff[itheta, 4] = 1.0 / (bsq * dpsisq)
        ff[itheta, 5] = bsq
        ff[itheta, 6] = dpsisq / bsq
        @views ff[itheta, :] .*= jac / v1
    end
    itp = cubic_interp(equil.rzphi_ys, Series(ff); bc=PeriodicBC())
    avg = FastInterpolations.integrate(itp)

    # --- curvature terms E, F, H (resist.f) ---
    E = p1 * v1 / (q1 * chi1^2)^2 * avg[1] * (twopif * q1 * chi1 / avg[5] - v2)
    F = (p1 * v1 / (q1 * chi1^2))^2 * (avg[1] * avg[3] + (twopif / chi1)^2 * (avg[1] * avg[4] - avg[2]^2))
    H = twopif * p1 * v1 / (q1 * chi1^3) * (avg[2] - avg[1] / avg[5])

    # --- mass factor M and derived G, K (resist.f) ---
    M = avg[1] * (avg[6] + (twopif / chi1)^2 * (avg[3] - 1 / avg[5]))
    G = avg[5] / (M * gamma * p)
    K = (q1 * chi1^2 / (p1 * v1))^2 * avg[5] / (M * avg[1])

    # --- Alfvén and resistive times from caller-supplied η, ρ (resist.f) ---
    # τ_A = √(ρ·M·μ₀) / |2π n q₁ χ₁ / V′|   — geometric factor × √ρ.
    # τ_R = (⟨B²/|∇ψ|²⟩ / ⟨B²⟩) · μ₀ / η     — geometric factor × 1/η.
    # η, ρ enter once, explicitly (no RMATCH divide-out / re-multiply).
    mu0 = 4π * 1e-7
    geom_taua = sqrt(M * mu0) / abs(2π * nn * q1 * chi1 / v1)
    taua = sqrt(rho) * geom_taua
    geom_taur = avg[1] / avg[5] * mu0
    taur = geom_taur / eta

    v1norm = equil.params.volume === nothing ? v1 : v1 / equil.params.volume

    return InnerLayer.GGJParameters(; E=E, F=F, G=G, H=H, K=K, M=M,
        taua=taua, taur=taur, v1=v1norm, ising=ising)
end
