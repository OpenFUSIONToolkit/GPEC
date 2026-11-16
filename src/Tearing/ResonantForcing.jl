# The resonant forcing of a single-n coil-driven perturbed equilibrium, per rational surface, in the form a steady island
# equation takes it (Callen, UW-CPTC 16-4, 2016): the vacuum and total resonant fields, Δ_RMP and the vacuum island width,
# on the toroidal-flux radius ρ that Callen's widths use. The modified Rutherford equation's forcing term and Callen preset use it.

module ResonantForcing

import FastInterpolations

import ...ForcingTerms
using ...Utilities.PhysicalConstants: M_P, E_CHG

export toroidal_flux_radius, callen_polarization_width, resonant_forcing, forcing_thresholds

"""
    toroidal_flux_radius(equil, psi, q) -> (rho, drho)

Toroidal-flux radius ρ = (Φ_tor/πB₀)^(1/2) [m] at `psi`, with Φ_tor = 2πψ₀ ∫q dψ_N and B₀ the vacuum field `bt0`, and
dρ/dψ_N = qψ₀/(B₀ρ) [m] with the surface's safety factor `q`.
"""
function toroidal_flux_radius(equil, psi::Real, q::Real)
    B0 = equil.params.bt0
    rho = sqrt(FastInterpolations.integrate(equil.profiles.q_spline, first(equil.profiles.xs), psi) * 2 * equil.psio / B0)
    return rho, q * equil.psio / (B0 * rho)
end

"""
    callen_polarization_width(equil, plasma, psi, q) -> (rho_i, w_pol)

Ion gyroradius ρ_i = (2T_i/m_i)^(1/2)/(Z_i e B₀/m_i) and Callen's polarization width w_pol = ½ q ρ_i/(ρ/R₀)^(1/2), both in
metres, from a `layer_plasma` view of the kinetic data; R₀ is the axis major radius and B₀ the vacuum field `bt0`.
"""
function callen_polarization_width(equil, plasma, psi::Real, q::Real)
    m_i, z_i, B0 = plasma.mu_i * M_P, plasma.z_i, equil.params.bt0
    rho_i = sqrt(2 * E_CHG * Float64(plasma.profiles.T_i(psi)) / m_i) / (z_i * E_CHG * B0 / m_i)
    rho, _ = toroidal_flux_radius(equil, psi, q)
    return rho_i, 0.5 * q * rho_i / sqrt(rho / equil.ro)
end

# Vacuum resonant field |B_vac| [T] of the coil forcing on the surface ψ, harmonic m.
function _vacuum_resonant_field(pe, equil, psi, n, m, area)
    rmp = pe.forcing
    rmp isa ForcingTerms.RMPSource && rmp.ctrl.forcing_data_format == "coil" ||
        throw(ArgumentError("the resonant forcing needs a single coil forcing (forcing_data_format = \"coil\") for the vacuum resonant field"))
    cfg = ForcingTerms.CoilConfig(rmp.ctrl)
    grid = ForcingTerms.CoilForcingGrid(equil, cfg, n; psi)
    amplitude = only(ForcingTerms.coil_forcing_modes(pe.coil_sets, grid, n, m, m)).amplitude
    return abs(rmp.scale * amplitude) / area
end

"""
    resonant_forcing(pe) -> NamedTuple of vectors

The resonant forcing on every resonant row of the single-n, coil-driven perturbed equilibrium `pe`:

  - `rational_psi`, `rational_q`, `m`, `n`, `surface_idx` - Surface identity; `surface_idx` indexes `pe.response.ffs.surfaces`.
  - `rho`, `drho` - Toroidal-flux radius and dρ/dψ_N [m] ([`toroidal_flux_radius`](@ref)).
  - `b_res`, `b_vac` - |total| and |vacuum| resonant area-weighted field [T].
  - `delta_rmp` - Δ_RMP = |Δ|/(2πR₀q) B₀/|B_res| [1/m of ρ], Δ the singular-current jump over 2πχ₁.
  - `hw_isl`, `hw_v` - Total-field and vacuum island half-widths in ψ_N, hw_v = (4 B_vac A/(2π n q′ χ₁))^(1/2).

In ψ_N the forcing of a full width w is then Δ_RMP (dρ/dψ_N) (2 hw_v/w)², which scales with B_vac.
"""
function resonant_forcing(pe)
    resp = pe.response
    ffs = resp.ffs
    equil = ffs.equil
    ffs.npert == 1 || throw(ArgumentError("the resonant forcing is single-n; the solve spans n = $(ffs.nlow):$(ffs.nhigh)"))
    isempty(pe.island_half_width) && throw(ArgumentError("the resonant forcing needs the singular coupling of the perturbed equilibrium"))
    R0, B0, chi1 = equil.ro, equil.params.bt0, 2π * equil.psio
    nrow = length(resp.rational_psi)
    rows = map(1:nrow) do row
        psi, q, m, n, area = resp.rational_psi[row], resp.rational_q[row], resp.rational_m_res[row], resp.rational_n[row], resp.rational_area[row]
        idx = resp.rational_surface_idx[row]
        q1 = ffs.surfaces[idx].q1
        rho, drho = toroidal_flux_radius(equil, psi, q)
        b_res = abs(pe.resonant_area_weighted_field[row])
        b_vac = _vacuum_resonant_field(pe, equil, psi, n, m, area)
        return (; rational_psi=psi, rational_q=q, m, n, surface_idx=idx, rho, drho, b_res, b_vac, delta_rmp=abs(pe.delta_prime[row]) / (2π * R0 * q) * B0 / b_res,
            hw_isl=pe.island_half_width[row], hw_v=sqrt(abs(4 * b_vac * area / (2π * n * q1 * chi1))))
    end
    keys = (:rational_psi, :rational_q, :m, :n, :surface_idx, :rho, :drho, :b_res, :b_vac, :delta_rmp, :hw_isl, :hw_v)
    return NamedTuple{keys}(Tuple([getfield(r, k) for r in rows] for k in keys))
end

# Golden-section extremum of f on [a, b]: the minimum, or with `sign = -1` the maximum.
function _golden(f, a, b; sign=1, tol=1e-10)
    phi = (sqrt(5) - 1) / 2
    c, d = b - phi * (b - a), a + phi * (b - a)
    fc, fd = sign * f(c), sign * f(d)
    while b - a > tol * (abs(a) + abs(b))
        if fc < fd
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = sign * f(c)
        else
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = sign * f(d)
        end
    end
    x = (a + b) / 2
    return x, f(x)
end

"""
    forcing_thresholds(F0, k, w_lo, w_grid, w_m0) -> (; b_crit, b_trigger, w_crit)

Critical fields of a forcing k B_vac/w² added to an unforced steady island equation `F0(w)`, with the island locked in
phase with the field (Callen's assumption). The forced F is positive where k B_vac > G(w) = −F0(w) w², so:

  - `b_crit` - The vacuum field at which a forced window F > 0 first opens below the intrinsic seed threshold `w_m0`
    (anywhere on the grid when `w_m0` is `NaN`), apart from the intrinsic region: the smallest local minimum of G,
    `w_lo` included, at width `w_crit`. For Callen's F0 it is his B_crit. It is 0 when F0 > 0 at `w_lo`, or when G
    falls all the way to `w_m0` (any field lowers the seed threshold); `NaN` when it falls all the way to the top of
    the grid without a `w_m0`. Without a term that beats 1/w² at small w (Callen's −w_pol²/w³ does), G rises from
    `w_lo`, the window opens there (`w_crit == w_lo`) and `b_crit` depends on `w_lo`, the layer width when the gate
    sets it.
  - `b_trigger` - The field at which the window merges with the region above `w_m0`, so that an island in it grows
    past `w_m0` towards the intrinsic saturated width: the maximum of G on `[w_crit, w_m0]` (on `[w_lo, w_m0]`
    without a window, the field at which no seed is needed). It is 0 when F0 > 0 at `w_lo`, and `NaN` without `w_m0`.

G is sampled on `w_lo` and the grid points above it, and each interior extremum is refined by golden-section search.
"""
function forcing_thresholds(F0, k::Real, w_lo::Real, w_grid::AbstractVector, w_m0::Real)
    G(w) = -F0(w) * w^2
    none = (; b_crit=NaN, b_trigger=NaN, w_crit=NaN)
    (k > 0 && isfinite(k)) || return none
    top = isfinite(w_m0) ? w_m0 : maximum(w_grid)
    nodes = vcat(Float64(w_lo), filter(w -> w_lo < w < top, w_grid), top)
    vals = G.(nodes)
    any(!isfinite, vals) && return none
    vals[1] < 0 && return (; b_crit=0.0, b_trigger=0.0, w_crit=NaN)
    n = length(nodes)
    interior = [_golden(G, nodes[j-1], nodes[j+1]) for j in 2:(n-1) if vals[j] <= vals[j-1] && vals[j] <= vals[j+1]]
    minima = n > 1 && vals[1] <= vals[2] ? vcat([(nodes[1], vals[1])], interior) : interior
    if isempty(minima)
        isfinite(w_m0) || return none
        w_crit, b_crit = NaN, 0.0
    else
        w_crit, g_crit = minima[argmin(last.(minima))]
        b_crit = g_crit / k
        isfinite(w_m0) || return (; b_crit, b_trigger=NaN, w_crit)
    end
    # The highest barrier of G between the window (or w_lo) and the intrinsic seed threshold.
    start = isnan(w_crit) ? nodes[1] : w_crit
    j = argmax([w >= start ? v : -Inf for (w, v) in zip(nodes, vals)])
    g_max = 1 < j < n ? last(_golden(G, max(nodes[j-1], start), nodes[j+1]; sign=-1)) : vals[j]
    return (; b_crit, b_trigger=max(g_max, vals[j]) / k, w_crit)
end

end # module ResonantForcing
