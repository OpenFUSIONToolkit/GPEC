"""
SlowingDown.jl - the fusion-alpha species of the NTV sum: an isotropic slowing-down distribution
f₀ = C(ψ)/(E^{3/2} + E_c^{3/2}) below the birth energy.
"""

const ε₀ = 8.854_187_8128e-12   # vacuum permittivity (F/m)

"""
    dt_reactivity(T_keV) → Float64

D–T fusion reactivity ⟨σv⟩ in m³/s at ion temperature `T_keV` (keV), from the Bosch & Hale (Nucl. Fusion 32, 611, 1992)
fit, Eqs. 12–14 with the Table VII coefficients; valid for 0.2–100 keV, extrapolated below (where it is negligible) and
held at 100 keV above.
"""
function dt_reactivity(T_keV::Real)
    T = min(Float64(T_keV), 100.0)
    BG, mrc2 = 34.3827, 1124656.0
    C1, C2, C3, C4, C5, C6, C7 = 1.17302e-9, 1.51361e-2, 7.51886e-2, 4.60643e-3, 1.35e-2, -1.06750e-4, 1.366e-5
    θ = T / (1 - T * (C2 + T * (C4 + T * C6)) / (1 + T * (C3 + T * (C5 + T * C7))))
    ξ = (BG^2 / (4θ))^(1 / 3)
    return 1e-6 * C1 * θ * sqrt(ξ / (mrc2 * T^3)) * exp(-3ξ)
end

"""
    alpha_species(species, kp; ebirth=3.52e6*eV) → ResolvedNTVSpecies

The fusion-alpha species (Z = 2, A = 4) of a resolved D–T species set `species` with electron and shared profiles `kp`.
Its profile view carries, on the grid of `kp`:

  - density n_α = S τ_s ln(1 + (E_b/E_c)^{3/2})/3, the slowing-down steady state of the source S = n_D n_T ⟨σv⟩(T_i),
  - the critical energy E_c = 14.8 A T_e (Σ_j n_j Z_j²/(n_e A_j))^{2/3} in the temperature slot (J),
  - the pitch-angle scattering rate on the ions at v = v_c, ν = Σ_j n_j Z_j² Z_α² e⁴ lnΛ/(4π ε₀² m_α² v_c³), as the
    Krook rate of the harmonic collision model, whose x^{-3/2} is the fast-ion v^{-3} law with x = E/E_c,

with τ_s = 3(2π)^{3/2} ε₀² m_α T_e^{3/2}/(n_e Z_α² e⁴ m_e^{1/2} lnΛ) the Spitzer slowing-down time on the electrons.
D and T are the species with (z, m) = (1, 2) and (1, 3).
"""
function alpha_species(species::AbstractVector{<:Equilibrium.ResolvedNTVSpecies}, kp::Equilibrium.KineticProfileSplines;
    ebirth::Float64=3.52e6 * eV)
    ions = [sp for sp in species if !sp.electron]
    iD = findfirst(sp -> (sp.z, sp.m) == (1, 2), ions)
    iT = findfirst(sp -> (sp.z, sp.m) == (1, 3), ions)
    (iD === nothing || iT === nothing) && error("the alpha species needs D (z = 1, m = 2) and T (z = 1, m = 3) in ion_species")
    zα, mα = 2, 4 * mp
    xs = kp.xs
    nα, Ec, nu = (zeros(length(xs)) for _ in 1:3)
    for (k, ψ) in enumerate(xs)
        ne, Te, lnΛ = kp.ne_spline(ψ), kp.Te_spline(ψ), kp.loglam_spline(ψ)
        nD, nT = ions[iD].profiles.ni_spline(ψ), ions[iT].profiles.ni_spline(ψ)
        nz2 = sum(sp.profiles.ni_spline(ψ) * sp.z^2 for sp in ions)
        nz2a = sum(sp.profiles.ni_spline(ψ) * sp.z^2 / sp.m for sp in ions)
        Ec[k] = 14.8 * 4 * Te * (nz2a / ne)^(2 / 3)
        τs = 3 * (2π)^1.5 * ε₀^2 * mα * Te^1.5 / (ne * zα^2 * e^4 * sqrt(me) * lnΛ)
        S = nD * nT * dt_reactivity(kp.Ti_spline(ψ) / (1e3 * eV))
        nα[k] = S * τs * log1p((ebirth / Ec[k])^1.5) / 3
        vc = sqrt(2 * Ec[k] / mα)
        nu[k] = nz2 * zα^2 * e^4 * lnΛ / (4π * ε₀^2 * mα^2 * vc^3)
    end
    all(>(0), nα) || error("alpha density is not positive everywhere")
    ne, Te, ω, lnΛ, nue, zeff = ([f(ψ) for ψ in xs] for f in (kp.ne_spline, kp.Te_spline, kp.omegaE_spline, kp.loglam_spline, kp.nue_spline, kp.zeff_spline))
    view = Equilibrium.KineticProfileSplines(xs, nα, ne, Ec, Te, ω, lnΛ, nu, nue, zeff)
    return Equilibrium.ResolvedNTVSpecies(zα, 4, false, "alpha", view)
end
