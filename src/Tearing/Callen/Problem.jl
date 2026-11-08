# Problem.jl
#
# `solve(CallenProblem(pe))`: Callen's reduced modified Rutherford equation for RMP-driven islands (UW-CPTC 16-4, 2016),
# in the Fortran GPEC 1.6.0 form, on every resonant row of a single-n perturbed equilibrium driven by coils.

"""
    CallenProblem(pe; delta_prime=nothing, delta_prime_source=:computed)

Callen's reduced MRE for the vacuum-island threshold on every resonant row of the perturbed equilibrium `pe`
(a single-n state from `perturbed_equilibrium` with a coil forcing). Half-widths are in metres of the toroidal-flux
radius ρ = (Φ_tor/πB₀)^(1/2) inside, and in ψ_N on output, as in Fortran GPEC 1.6.0:

  - w_pol = ½ q ρ_i/(ρ/R₀)^(1/2) with ρ_i = (2T_i/m_i)^(1/2)/(Z_i e B₀/m_i), R₀ the axis major radius and B₀ the
    vacuum toroidal field `bt0`, as Fortran GPEC 1.6.0 takes them;
  - Δ_RMP = |Δ|/(2πR₀q) B₀/|B_res|, Δ the singular-current jump over 2πχ₁ and B_res the total resonant field;
  - Δ_c the intrinsic tearing index in 1/m of ρ: the true Δ′_mn of `delta_prime` (a Riccati or Galerkin solve on the
    same equilibrium and n), referred to ρ as K^(2α) Δ′_mn/ρ with K = ρ dψ_N/dρ and α = (−D_I)^(1/2) as for SLAYER, with
    `delta_prime_source = :computed`; or Callen's cylindrical −2m/ρ (Fortran GPEC 1.6.0) with `:cylindrical`;
  - critical vacuum half-width h_v,crit = ½ w_pol^(2/3) ((27/4)|Δ_c|)^(1/6)/Δ_RMP^(1/2);
  - the steady state of Callen's Eq. 17, Δ_c w³ + Δ_RMP w_vac² w − w_pol² = 0, a cubic in the full width w with
    w_vac = 2h_v; its positive roots halved are h_min < h_sat (with Δ_c ≥ 0 only h_min exists, and h_v,crit = 0);
  - critical resonant field B_crit = |B_vac| (h_v,crit/h_v)².

The vacuum resonant field B_vac is the coils' resonant harmonic on the rational surface over its area.

## Fields

  - `pe` - Perturbed-equilibrium state, single n, from a coil forcing.
  - `delta_prime` - Force-free-states result carrying Δ′ (Riccati or Galerkin) for `:computed`; set by the deck.
  - `delta_prime_source` - `:computed` (true Δ′_mn) or `:cylindrical` (−2m/ρ, the Fortran 1.6.0 reference).
"""
@kwdef struct CallenProblem{S,D}
    pe::S
    delta_prime::D = nothing
    delta_prime_source::Symbol = :computed
end

CallenProblem(pe; kwargs...) = CallenProblem(; pe=pe, kwargs...)

"""
    CallenResult

Output of `solve(::CallenProblem)`, one entry per resonant row. Half-widths are in ψ_N; `NaN` marks a root that
does not exist.

## Fields

  - `problem` - The producing problem.
  - `rational_psi`, `rational_q`, `m`, `n` - Surface identity.
  - `rho_tor`, `drho_dpsi` - Toroidal-flux radius ρ [m] and dρ/dψ_N [m].
  - `rho_i`, `w_pol` - Ion gyroradius and polarization width [m].
  - `delta_rmp`, `delta_c` - Δ_RMP and Δ_c [1/m].
  - `b_res`, `b_vac` - |total| and |vacuum| resonant field [T].
  - `hw_isl`, `hw_v` - Total-field and vacuum island half-widths.
  - `hw_v_crit` - Critical vacuum half-width.
  - `hw_min`, `hw_sat` - Half of the smaller and larger positive (full-width) cubic roots.
  - `n_roots` - Number of positive cubic roots (0 or 2; 1 when Δ_c ≥ 0).
  - `b_crit` - Critical vacuum resonant field [T].
"""
struct CallenResult{P}
    problem::P
    rational_psi::Vector{Float64}
    rational_q::Vector{Float64}
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

The positive roots of `a x³ + b x + c = 0` with `b ≥ 0`, `c < 0`. For `a < 0`, the trigonometric solution
x_k = 2R cos((θ + 2πk)/3), R = (−p/3)^(1/2), cos θ = −q/(2R³) of the depressed cubic x³ + px + q gives two positive
roots or none (`(NaN, NaN)`); for `a > 0` there is one positive root (Cardano), returned as `(x, NaN)`.
"""
function callen_cubic_roots(a::Real, b::Real, c::Real)
    a == 0 && return (b > 0 ? -c / b : NaN, NaN)
    p, q = b / a, c / a
    if a > 0
        s = sqrt(q^2 / 4 + p^3 / 27)
        return (cbrt(-q / 2 + s) + cbrt(-q / 2 - s), NaN)
    end
    p < 0 || return (NaN, NaN)
    R = sqrt(-p / 3)
    arg = -q / (2 * R^3)
    abs(arg) <= 1 || return (NaN, NaN)
    theta = acos(arg)
    roots = sort(filter(>(0), [2R * cos((theta + 2π * k) / 3) for k in 0:2]))
    length(roots) == 2 || return (NaN, NaN)
    return (roots[1], roots[2])
end

# Δ′_mn of the surface at ψ from a force-free-states result, referred to the radius ρ in 1/m: K^(2α) Δ′/ρ, K = ρ dψ_N/dρ.
function _callen_delta_c(ric, psi, rho, drho)
    k = findfirst(s -> isapprox(s.psifac, psi; atol=1e-8), ric.surfaces)
    k === nothing && throw(ArgumentError("CallenProblem: delta_prime has no rational surface at ψ = $psi"))
    rg = ric.surfaces[k].restype
    alpha = sqrt(max(0.25 - (rg.E + rg.F + rg.H), 0.0))
    return (rho / drho)^(2 * alpha) * real(ric.delta_prime.matrix[k, k]) / rho
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
    prob.delta_prime_source in (:computed, :cylindrical) ||
        throw(ArgumentError("delta_prime_source must be :computed or :cylindrical, got :$(prob.delta_prime_source)"))
    ric = prob.delta_prime
    prob.delta_prime_source === :cylindrical || (ric !== nothing && !isempty(ric.delta_prime.matrix)) ||
        throw(ArgumentError("CallenProblem with delta_prime_source = :computed needs delta_prime, a Riccati or Galerkin solve carrying Δ′"))

    plasma = layer_plasma(equil; log=false)
    rf = resonant_forcing(pe)
    nrow = length(rf.rational_psi)
    out = Dict(k => fill(NaN, nrow) for k in (:rho_i, :w_pol, :delta_c, :hw_v_crit, :hw_min, :hw_sat, :b_crit))
    n_roots = zeros(Int, nrow)
    for row in 1:nrow
        psi, m, rho, drho, delta_rmp, hw_v, b_vac = rf.rational_psi[row], rf.m[row], rf.rho[row], rf.drho[row], rf.delta_rmp[row], rf.hw_v[row], rf.b_vac[row]
        rho_i, w_pol = callen_polarization_width(equil, plasma, psi, rf.rational_q[row])
        delta_c = prob.delta_prime_source === :cylindrical ? -2 * m / rho : _callen_delta_c(ric, psi, rho, drho)
        # With Δ_c ≥ 0 any vacuum island lets a seed above h_min grow, so the critical vacuum width is zero.
        hw_v_crit = delta_c < 0 ? 0.5 * w_pol^(2 / 3) * ((27 / 4) * abs(delta_c))^(1 / 6) / sqrt(delta_rmp) : 0.0
        # Roots are full widths in metres; the outputs are half-widths in ψ_N.
        x_min, x_sat = callen_cubic_roots(delta_c, delta_rmp * (2 * hw_v * drho)^2, -w_pol^2)
        for (k, v) in pairs((; rho_i, w_pol, delta_c, hw_v_crit=hw_v_crit / drho, hw_min=x_min / (2 * drho), hw_sat=x_sat / (2 * drho),
            b_crit=b_vac * (hw_v_crit / drho / hw_v)^2))
            out[k][row] = v
        end
        n_roots[row] = count(!isnan, (x_min, x_sat))
    end
    return CallenResult(prob, rf.rational_psi, rf.rational_q, rf.m, rf.n, rf.rho, rf.drho, out[:rho_i], out[:w_pol], rf.delta_rmp, out[:delta_c], rf.b_res,
        rf.b_vac, rf.hw_isl, rf.hw_v, out[:hw_v_crit], out[:hw_min], out[:hw_sat], n_roots, out[:b_crit])
end
