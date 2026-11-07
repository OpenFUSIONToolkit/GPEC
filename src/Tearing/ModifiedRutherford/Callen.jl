# Callen.jl
#
# `solve(CallenProblem(pe))`: Callen's reduced modified Rutherford equation for RMP-driven islands (UW-CPTC 16-4, 2016),
# in the Fortran GPEC 1.6.0 form, on every resonant row of a single-n perturbed equilibrium driven by coils.

"""
    CallenProblem(pe)

Callen's reduced MRE for the vacuum-island threshold on every resonant row of the perturbed equilibrium `pe`
(a single-n state from `perturbed_equilibrium` with a coil forcing). Half-widths are in metres of the toroidal-flux
radius ρ = (Φ_tor/πB₀)^(1/2) inside, and in ψ_N on output, as in Fortran GPEC 1.6.0:

  - w_pol = ½ q ρ_i/(ρ/R₀)^(1/2) with ρ_i = (2T_i/m_i)^(1/2)/(Z_i e B₀/m_i), R₀ and B₀ on the axis;
  - Δ_RMP = |Δ|/(2πR₀q) B₀/|B_res|, Δ the singular-current jump over 2πχ₁ and B_res the total resonant field;
  - Δ_c = −2m/ρ;
  - critical vacuum half-width h_v,crit = ½ w_pol^(2/3) ((27/4)|Δ_c|)^(1/6)/Δ_RMP^(1/2);
  - the cubic Δ_c x³ + Δ_RMP (2h_v)² x − w_pol² = 0 for the half-width x, with h_v the vacuum island half-width; its
    positive roots are h_min < h_sat;
  - critical resonant field B_crit = |B_vac| (h_v,crit/h_v)².

The vacuum resonant field B_vac is the coils' resonant harmonic on the rational surface over its area.

## Fields

  - `pe` - Perturbed-equilibrium state, single n, from a coil forcing.
"""
struct CallenProblem{S}
    pe::S
end

"""
    CallenResult

Output of `solve(::CallenProblem)`, one entry per resonant row. Half-widths are in ψ_N; `NaN` marks a root that
does not exist.

## Fields

  - `problem` - The producing problem.
  - `psi`, `q`, `m`, `n` - Surface identity.
  - `rho_tor`, `drho_dpsi` - Toroidal-flux radius ρ [m] and dρ/dψ_N [m].
  - `rho_i`, `w_pol` - Ion gyroradius and polarization width [m].
  - `delta_rmp`, `delta_c` - Δ_RMP and Δ_c [1/m].
  - `b_res`, `b_vac` - |total| and |vacuum| resonant field [T].
  - `hw_isl`, `hw_v` - Total-field and vacuum island half-widths.
  - `hw_v_crit` - Critical vacuum half-width.
  - `hw_min`, `hw_sat` - Smaller and larger positive cubic roots.
  - `n_roots` - Number of positive cubic roots (0 or 2).
  - `b_crit` - Critical vacuum resonant field [T].
"""
struct CallenResult{P}
    problem::P
    psi::Vector{Float64}
    q::Vector{Float64}
    m::Vector{Int}
    n::Vector{Int}
    rho_tor::Vector{Float64}
    drho_dpsi::Vector{Float64}
    rho_i::Vector{Float64}
    w_pol::Vector{Float64}
    delta_rmp::Vector{Float64}
    delta_c::Vector{Float64}
    b_res::Vector{Float64}
    b_vac::Vector{Float64}
    hw_isl::Vector{Float64}
    hw_v::Vector{Float64}
    hw_v_crit::Vector{Float64}
    hw_min::Vector{Float64}
    hw_sat::Vector{Float64}
    n_roots::Vector{Int}
    b_crit::Vector{Float64}
end

"""
    callen_cubic_roots(a, b, c) -> (x_min, x_sat)

The positive roots of `a x³ + b x + c = 0` with `a < 0`, `b ≥ 0`, `c < 0`, by the trigonometric solution
x_k = 2R cos((θ + 2πk)/3), R = (−p/3)^(1/2), cos θ = −q/(2R³) of the depressed cubic x³ + px + q; `(NaN, NaN)`
when it has no positive root.
"""
function callen_cubic_roots(a::Real, b::Real, c::Real)
    p, q = b / a, c / a
    p < 0 || return (NaN, NaN)
    R = sqrt(-p / 3)
    arg = -q / (2 * R^3)
    abs(arg) <= 1 || return (NaN, NaN)
    theta = acos(arg)
    roots = sort(filter(>(0), [2R * cos((theta + 2π * k) / 3) for k in 0:2]))
    length(roots) == 2 || return (NaN, NaN)
    return (roots[1], roots[2])
end

# Vacuum resonant field |B_vac| [T] of the coil forcing on the surface ψ, harmonic m.
function _vacuum_resonant_field(pe, equil, psi, n, m, area)
    rmp = pe.forcing
    rmp isa ForcingTerms.RMPSource && rmp.ctrl.forcing_data_format == "coil" ||
        throw(ArgumentError("CallenProblem needs a single coil forcing (forcing_data_format = \"coil\") for the vacuum resonant field"))
    cfg = ForcingTerms.CoilConfig(rmp.ctrl)
    grid = ForcingTerms.CoilForcingGrid(equil, cfg, n; psi)
    amplitude = only(ForcingTerms.coil_forcing_modes(pe.coil_sets, grid, n, m, m)).amplitude
    return abs(rmp.scale * amplitude) / area
end

function CommonSolve.solve(prob::CallenProblem)
    pe = prob.pe
    resp = pe.response
    ffs = resp.ffs
    equil = ffs.equil
    ffs.npert == 1 || throw(ArgumentError("CallenProblem is single-n; the solve spans n = $(ffs.nlow):$(ffs.nhigh)"))
    data = equil.kinetic
    data === nothing && throw(ArgumentError("CallenProblem needs kinetic profiles; set kinetic_file in [Equilibrium]"))
    isempty(pe.island_half_width) && throw(ArgumentError("CallenProblem needs the singular coupling of the perturbed equilibrium"))

    T_i_of = _slayer_profiles(data, nothing).profiles.T_i
    species = data.ion_species
    m_i = sum(s.fraction * s.m for s in species) / sum(s.fraction for s in species) * M_P
    z_i = Float64(first(species).z)
    R0, B0 = equil.ro, equil.params.bt0
    chi1 = 2π * equil.psio
    qspl = equil.profiles.q_spline
    psi0 = first(equil.profiles.xs)

    nrow = length(resp.rational_psi)
    out = Dict(
        k => fill(NaN, nrow) for k in (:psi, :q, :rho_tor, :drho_dpsi, :rho_i, :w_pol, :delta_rmp, :delta_c, :b_res, :b_vac,
            :hw_isl, :hw_v, :hw_v_crit, :hw_min, :hw_sat, :b_crit)
    )
    m_out, n_out, n_roots = zeros(Int, nrow), zeros(Int, nrow), zeros(Int, nrow)
    for row in 1:nrow
        psi, q, m, n, area = resp.rational_psi[row], resp.rational_q[row], resp.rational_m_res[row], resp.rational_n[row], resp.rational_area[row]
        q1 = ffs.surfaces[resp.rational_surface_idx[row]].q1
        # ρ = (Φ_tor/πB₀)^(1/2) with Φ_tor = 2πψ₀ ∫q dψ_N; dρ/dψ_N = qψ₀/(B₀ρ).
        rho = sqrt(FastInterpolations.integrate(qspl, psi0, psi) * 2 * equil.psio / B0)
        drho = q * equil.psio / (B0 * rho)
        rho_i = sqrt(2 * E_CHG * Float64(T_i_of(psi)) / m_i) / (z_i * E_CHG * B0 / m_i)
        w_pol = 0.5 * q * rho_i / sqrt(rho / R0)
        b_res = abs(pe.resonant_area_weighted_field[row])
        delta_rmp = abs(pe.delta_prime[row]) / (2π * R0 * q) * B0 / b_res
        delta_c = -2 * m / rho
        hw_v_crit = 0.5 * w_pol^(2 / 3) * ((27 / 4) * abs(delta_c))^(1 / 6) / sqrt(delta_rmp)
        b_vac = _vacuum_resonant_field(pe, equil, psi, n, m, area)
        shear = n * q1 / q
        hw_v = sqrt(abs(4 * b_vac * area / (2π * shear * q * chi1)))
        x_min, x_sat = callen_cubic_roots(delta_c, delta_rmp * (2 * hw_v * drho)^2, -w_pol^2)
        for (k, v) in pairs((; psi, q, rho_tor=rho, drho_dpsi=drho, rho_i, w_pol, delta_rmp, delta_c, b_res, b_vac,
            hw_isl=pe.island_half_width[row], hw_v, hw_v_crit=hw_v_crit / drho, hw_min=x_min / drho, hw_sat=x_sat / drho,
            b_crit=b_vac * (hw_v_crit / drho / hw_v)^2))
            out[k][row] = v
        end
        m_out[row], n_out[row], n_roots[row] = m, n, isnan(x_min) ? 0 : 2
    end
    return CallenResult(prob, out[:psi], out[:q], m_out, n_out, out[:rho_tor], out[:drho_dpsi], out[:rho_i], out[:w_pol], out[:delta_rmp],
        out[:delta_c], out[:b_res], out[:b_vac], out[:hw_isl], out[:hw_v], out[:hw_v_crit], out[:hw_min], out[:hw_sat], n_roots, out[:b_crit])
end
