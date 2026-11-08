# Problem.jl
#
# `solve(ModifiedRutherfordProblem(ffs; ...))`: the terms of the steady modified Rutherford equation and their sum F(w)
# on a log-spaced grid of full island widths, per rational surface, and its roots w_m and w_sat.

"""
    ModifiedRutherfordProblem(ffs; kwargs...)

The steady modified Rutherford equation (MRE) on every rational surface of a finished force-balance solve,

    F(w) = s_Δ Δ′(w) + s_sat Δ_sat(w) + s_nc Δ_nc(w) + s_GGJ Δ_GGJ(w) + s_pol Δ_pol(w) + s_RMP Δ_RMP(w),

with `w` the full island width in normalized poloidal flux ψ_N. Each term is evaluated on the grid whether or not its
switch is on; the switches select which terms `F` sums. Δ′ is the real part of the diagonal of `ffs.delta_prime`
(`delta_prime_source = :computed`) or the cylindrical vacuum value r_s Δ′ = −2m referred to ψ_N as SLAYER refers Δ′ to
r_s (`:cylindrical`, which needs no Δ′ solve);
the Mercier, resistive-interchange and Pfirsch-Schlüter indices D_I, D_R = E + F + H², H come from the surfaces'
`ResistGeometry`, and the plasma (densities, temperatures, Z_eff, χ⊥,e, E×B rotation, ion species) is the
equilibrium's kinetic data, as every tearing layer reads it. A deck states it as `[Tearing.ModifiedRutherford]`.

The roots of F are refined between grid points from `w_lo = max(δ_r, w_min)`: the seed threshold `w_m` is the
smallest crossing from F < 0 to F > 0 and the saturated width `w_sat` the first crossing from F > 0 to F < 0 above where
the island starts (`w_lo` when F(w_lo) > 0, else `w_m`). Every saturated width and the seed threshold that reaches it are
kept as well (`w_sat_all`, `w_seed_all`), so a surface with several saturated solutions shows all of them. Surfaces that
are resistive-interchange unstable (D_R > 0) get no roots; linearly tearing unstable ones (γ > 0 in `tearing`) grow
from noise, so they get no seed threshold but keep `w_sat`.

With a coil-driven perturbed equilibrium `pe`, the applied field forces the island: Δ_RMP(w) = Δ_RMP (dρ/dψ_N) (w_v/w)²
(Callen's Eq. 17, [`resonant_forcing`](@ref)), with w_v the vacuum island width and the island locked in phase with the
field. It scales with the vacuum resonant field B_vac, so the roots and branches are those of the forced F, and the
critical fields `b_crit` and `b_trigger` follow from the unforced F ([`forcing_thresholds`](@ref)).

`CALLEN_PRESET` (deck `preset = "callen"`) reduces F to Callen's steady state Δ_c + Δ_RMP (w_v/w)² − w_pol²/w³ in his
units, so `solve` reproduces `CallenProblem`'s roots (full widths) and critical field.

## Fields

  - `ffs` - Outer solution, single n, with a Δ′ matrix (Riccati with `vac_flag=true`, or Galerkin with `rpec_flag=true`)
    unless `delta_prime_source = :cylindrical`.
  - `tearing` - Linear tearing solve on the same Δ′ (the gate and the layer width δ_r), or `nothing`; set by the deck
    when `[Tearing]` is present.
  - `pe` - Single-n, coil-driven perturbed equilibrium on the same equilibrium and n, or `nothing`; set by the deck when
    `[PerturbedEquilibrium]` is present. Its rows are matched to the surfaces by ψ.
  - `classical`, `saturation`, `bootstrap`, `curvature`, `polarization`, `forcing` - Term switches s_Δ, s_sat, s_nc, s_GGJ,
    s_pol, s_RMP. `saturation` is the finite-width correction to Δ′ ([`saturation_term`](@ref)), off by default;
    `forcing` acts only with `pe`.
  - `saturation_window` - Half-width in ψ_N of the quadratic fit that gives J_z′ and J_z″ in [`saturation_coefficient`](@ref).
  - `delta_prime_source` - `:computed` (Δ′_mn of `ffs`) or `:cylindrical` (r_s Δ′_mn = −2m, a reference case).
  - `delta_prime_model` - `:mercier_power` (Benjamin Eq. 10), `:constant` (Δ′(w) = Δ′_mn), or `:callen` (constant,
    Callen's Δ_c = K^(2α) Δ′_mn/ρ, or −2m/ρ with `:cylindrical`, in ψ_N; K = ρ dψ_N/dρ on the toroidal-flux radius ρ).
  - `delta_prime_factor`, `D_nc_factor`, `rho_theta_factor` - Multipliers on Δ′_mn, D_nc and the poloidal ion Larmor
    radius ρ_θi (which also sets w_ib), for scans.
  - `curvature_model` - `:lutjens` (default; Lütjens et al. 2001, [`lutjens_curvature_term`](@ref), whose small-w limit is
    the SLAYER toroidal critical offset D_c of the linear gate) or `:benjamin` (Benjamin Eq. 12, [`curvature_term`](@ref),
    for comparison with the tearing-physics suite). `:lutjens` takes W_d and D_c from SLAYER's toroidal layer parameters
    built on each surface with this problem's χ⊥.
  - `k1` - Bootstrap coefficient k₁, and curvature coefficient of `curvature_model = :benjamin`.
  - `c_wd` - Prefactor of the transport threshold width w_d.
  - `chi_perp_factor` - Multiplier on χ⊥,e (the kinetic `chi_e` profile, else its `chi_e_default`).
  - `pol_model` - `:full` (Benjamin Eq. C.1), `:reduced` (−c_pol w_pol²/w³, w_pol = C_pol w_ib) or `:callen`
    (−c_pol w_pol²/w³ with Callen's w_pol, [`callen_polarization_width`](@ref)).
  - `c_pol`, `C_pol` - Polarization coefficients.
  - `pol_sign` - ±1, multiplies the polarization term.
  - `omega_model` - Island frequency ω in the E_r = 0 frame: `:electron_diamagnetic`, ω = ω_*e(1 + η_e/4) (Wilson
    1996), or `:exb`, the lab-frame ω = n Ω_E from the kinetic `omega_E`.
  - `omega_factor` - Multiplier on the ω of `omega_model`; 0 is a stalled island.
  - `omega_ratio` - ω/ω_*i on every surface, replacing `omega_model`; `nothing` (default) uses the model.
  - `w_min`, `w_max`, `nw` - Log-spaced width grid in ψ_N.
"""
@kwdef struct ModifiedRutherfordProblem{R,T,S}
    ffs::R
    tearing::T = nothing
    pe::S = nothing
    classical::Bool = true
    bootstrap::Bool = true
    curvature::Bool = true
    polarization::Bool = false
    saturation::Bool = false
    forcing::Bool = true
    saturation_window::Float64 = 0.02
    curvature_model::Symbol = :lutjens
    delta_prime_source::Symbol = :computed
    delta_prime_model::Symbol = :mercier_power
    delta_prime_factor::Float64 = 1.0
    D_nc_factor::Float64 = 1.0
    rho_theta_factor::Float64 = 1.0
    k1::Float64 = 1.7
    c_wd::Float64 = 1.0
    chi_perp_factor::Float64 = 1.0
    pol_model::Symbol = :full
    c_pol::Float64 = 1.0
    C_pol::Float64 = 1.9
    pol_sign::Float64 = 1.0
    omega_model::Symbol = :electron_diamagnetic
    omega_factor::Float64 = 1.0
    omega_ratio::Union{Nothing,Float64} = nothing
    w_min::Float64 = 1e-5
    w_max::Float64 = 0.2
    nw::Int = 200
end

ModifiedRutherfordProblem(ffs; kwargs...) = ModifiedRutherfordProblem(; ffs=ffs, kwargs...)

"""
    CALLEN_PRESET

`ModifiedRutherfordProblem` keywords that reduce F to Callen's steady state Δ_c + Δ_RMP (w_v/w)² − w_pol²/w³: a
constant classical term in Callen's units, his polarization width, the forcing, and no other term. Keywords given with
the preset override it, e.g. `ModifiedRutherfordProblem(ffs; pe, CALLEN_PRESET..., delta_prime_source=:cylindrical)`.
"""
const CALLEN_PRESET = (; classical=true, bootstrap=false, curvature=false, saturation=false, polarization=true, forcing=true,
    delta_prime_model=:callen, pol_model=:callen, c_pol=1.0, pol_sign=1.0)

"""
    ModifiedRutherfordResult

Output of `solve(::ModifiedRutherfordProblem)`, built by keyword. Per-surface vectors run core to edge; matrices are
surface × width.

## Fields

  - `problem` - The producing problem.
  - `w` - Width grid in ψ_N.
  - `rational_psi`, `rational_q`, `m`, `n` - Surface identity.
  - `delta_prime`, `delta_prime_imag`, `D_I`, `D_R`, `H` - Outer-region and layer indices (Δ′ after `delta_prime_factor`).
  - `D_nc`, `f_trap`, `mu_ratio`, `zeff` - Bootstrap coefficient, Sauter 2002 trapped fraction, μ_e/(ν_e + μ_e), Z_eff.
  - `Wc_prefac`, `chi_perp`, `v_te`, `R0` - Inputs of w_d.
  - `omega_star_i`, `omega_star_e`, `omega_hat`, `omega_hat_pi` - Ion and electron diamagnetic frequencies [rad/s] from
    the density gradients, ω/ω_*i and ω_*pi/ω_*i.
  - `beta_p`, `rho_theta`, `w_ib`, `eps` - Polarization inputs: poloidal ion beta, poloidal ion Larmor radius [m],
    banana width in ψ_N, inverse aspect ratio.
  - `gamma`, `delta_r` - Linear growth rate [1/s] and resistive layer width in ψ_N from `tearing` (`NaN` without it).
  - `w_lo` - Lower end of the root search.
  - `D_c`, `W_d_layer` - SLAYER toroidal critical offset (r_s reference) and the transport width it implies, in the
    (r − r_s)/r_s reference, used by `curvature_model = :lutjens` (`NaN` for `:benjamin`).
  - `alpha_s`, `beta_s`, `C_sat` - Current-gradient coefficients of the saturation term and its coefficient.
  - `w_m`, `w_sat` - Seed threshold and saturated full widths in ψ_N; `NaN` when absent or excluded, and `w_m` `NaN` when γ > 0.
  - `w_sat_all`, `w_seed_all` - Every saturated width per surface (surface × branch, increasing, `NaN`-padded) and the
    seed threshold that reaches each: seeds between `w_seed_all[s, k]` and the next threshold settle at `w_sat_all[s, k]`;
    `w_seed_all` is `NaN` on a branch F > 0 reaches from `w_lo`. `w_sat` is the first column.
  - `w_seed_unsaturated` - Seed above which the island grows past `w_max`, when F > 0 there; `NaN` otherwise.
  - `n_sat` - Number of saturated widths; `multiple_saturated` when more than one.
  - `w_m_star` - Smallest `w_m` over the surfaces.
  - `b_m`, `b_seed_all`, `b_seed_unsaturated`, `b_m_star` - Resonant area-weighted field [T] whose vacuum island is as wide as
    `w_m`, each `w_seed_all` and `w_seed_unsaturated` ([`vacuum_island_field`](@ref)), comparable with the perturbed
    equilibrium's resonant field and with error-field thresholds; `b_m_star` is the smallest `b_m` over the surfaces (not
    necessarily at the `w_m_star` surface).
  - `b_vac`, `b_res`, `delta_rmp`, `w_v` - Vacuum and total resonant fields [T], Δ_RMP [1/m] and the full vacuum island
    width in ψ_N from `pe` ([`resonant_forcing`](@ref)); `NaN` without it.
  - `b_crit`, `b_trigger`, `w_crit` - Critical vacuum resonant fields [T] of the forcing, from the unforced F
    ([`forcing_thresholds`](@ref)): a forced window opens at `b_crit` (at full width `w_crit`) and reaches the intrinsic
    seed threshold at `b_trigger`; `NaN` without `pe`. `b_crit_star` is the smallest `b_crit` over the surfaces.
  - `fully_stabilized` - F < 0 on the whole search range.
  - `unstable_any_seed` - F > 0 at `w_lo`.
  - `unsaturated` - F > 0 at `w_max`: an island past its last rising root does not saturate in the search range.
  - `outside_plasma` - ψ ± `w_sat`/2 leaves the equilibrium's ψ grid.
  - `gate_disagrees` - F(w_lo) > 0 with γ < 0, or F(w_lo) < 0 with γ > 0.
  - `interchange_unstable`, `linear_unstable` - D_R > 0 (excluded), or γ > 0 (no seed threshold; `w_sat` kept).
  - `below_w_ib` - `w_m` below the ion banana width, where the polarization ordering fails.
  - `large_delta_prime_w` - |Δ′(w)| w ≥ 1 at `w_m` or `w_sat`, with Δ′(w) the classical term.
  - `complex_delta_prime` - |Im Δ′_mn| > 10⁻³ |Re Δ′_mn|.
  - `saturation_unconverged` - `C_sat` not converged in the fit window (see [`saturation_coefficient`](@ref)).
  - `F`, `classical`, `saturation`, `bootstrap`, `curvature`, `polarization`, `forcing`, `w_d` - F(w), each term, and
    w_d(w); `forcing` is 0 without `pe`.
"""
@kwdef struct ModifiedRutherfordResult{P}
    problem::P
    w::Vector{Float64}
    rational_psi::Vector{Float64}
    rational_q::Vector{Float64}
    m::Vector{Int}
    n::Vector{Int}
    delta_prime::Vector{Float64}
    delta_prime_imag::Vector{Float64}
    D_I::Vector{Float64}
    D_R::Vector{Float64}
    H::Vector{Float64}
    D_nc::Vector{Float64}
    f_trap::Vector{Float64}
    mu_ratio::Vector{Float64}
    zeff::Vector{Float64}
    Wc_prefac::Vector{Float64}
    chi_perp::Vector{Float64}
    v_te::Vector{Float64}
    R0::Vector{Float64}
    omega_star_i::Vector{Float64}
    omega_star_e::Vector{Float64}
    omega_hat::Vector{Float64}
    omega_hat_pi::Vector{Float64}
    beta_p::Vector{Float64}
    rho_theta::Vector{Float64}
    w_ib::Vector{Float64}
    eps::Vector{Float64}
    gamma::Vector{Float64}
    delta_r::Vector{Float64}
    w_lo::Vector{Float64}
    D_c::Vector{Float64}
    W_d_layer::Vector{Float64}
    alpha_s::Vector{Float64}
    beta_s::Vector{Float64}
    C_sat::Vector{Float64}
    w_m::Vector{Float64}
    w_sat::Vector{Float64}
    w_sat_all::Matrix{Float64}
    w_seed_all::Matrix{Float64}
    w_seed_unsaturated::Vector{Float64}
    n_sat::Vector{Int}
    b_m::Vector{Float64}
    b_seed_all::Matrix{Float64}
    b_seed_unsaturated::Vector{Float64}
    w_m_star::Float64
    b_m_star::Float64
    b_vac::Vector{Float64}
    b_res::Vector{Float64}
    delta_rmp::Vector{Float64}
    w_v::Vector{Float64}
    b_crit::Vector{Float64}
    b_trigger::Vector{Float64}
    w_crit::Vector{Float64}
    b_crit_star::Float64
    fully_stabilized::Vector{Bool}
    unstable_any_seed::Vector{Bool}
    unsaturated::Vector{Bool}
    multiple_saturated::Vector{Bool}
    outside_plasma::Vector{Bool}
    gate_disagrees::Vector{Bool}
    interchange_unstable::Vector{Bool}
    linear_unstable::Vector{Bool}
    below_w_ib::Vector{Bool}
    large_delta_prime_w::Vector{Bool}
    complex_delta_prime::Vector{Bool}
    saturation_unconverged::Vector{Bool}
    F::Matrix{Float64}
    classical::Matrix{Float64}
    saturation::Matrix{Float64}
    bootstrap::Matrix{Float64}
    curvature::Matrix{Float64}
    polarization::Matrix{Float64}
    forcing::Matrix{Float64}
    w_d::Matrix{Float64}
end

function validate(prob::ModifiedRutherfordProblem)
    ffs = prob.ffs
    ffs.npert == 1 || throw(ArgumentError("ModifiedRutherfordProblem is single-n; the solve spans n = $(ffs.nlow):$(ffs.nhigh)"))
    prob.delta_prime_source in (:computed, :cylindrical) || throw(ArgumentError("delta_prime_source must be :computed or :cylindrical"))
    if prob.delta_prime_source === :computed
        dp = ffs.delta_prime
        (dp === nothing || isempty(dp.matrix)) &&
            throw(
                ArgumentError(
                    "ModifiedRutherfordProblem needs Δ′: solve with Riccati() and vac_flag=true, Galerkin(; rpec_flag=true), " *
                    "or set delta_prime_source = :cylindrical"
                )
            )
        size(dp.matrix, 1) == length(ffs.surfaces) ||
            throw(ArgumentError("ModifiedRutherfordProblem: the Δ′ matrix is $(size(dp.matrix, 1))×$(size(dp.matrix, 2)) for $(length(ffs.surfaces)) surfaces"))
    end
    ffs.equil.kinetic === nothing && throw(ArgumentError("ModifiedRutherfordProblem needs kinetic profiles; set kinetic_file in [Equilibrium]"))
    prob.delta_prime_model in (:mercier_power, :constant, :callen) || throw(ArgumentError("delta_prime_model must be :mercier_power, :constant or :callen"))
    prob.curvature_model in (:lutjens, :benjamin) || throw(ArgumentError("curvature_model must be :lutjens or :benjamin"))
    prob.pol_model in (:full, :reduced, :callen) || throw(ArgumentError("pol_model must be :full, :reduced or :callen"))
    pe = prob.pe
    if pe !== nothing
        pffs = pe.response.ffs
        pffs.npert == 1 && pffs.nlow == ffs.nlow || throw(ArgumentError("ModifiedRutherfordProblem: pe must be single-n with the n of ffs"))
    end
    prob.omega_model in (:electron_diamagnetic, :exb) || throw(ArgumentError("omega_model must be :electron_diamagnetic or :exb"))
    abs(prob.pol_sign) == 1 || throw(ArgumentError("pol_sign must be +1 or -1"))
    0 < prob.w_min < prob.w_max && prob.nw >= 2 || throw(ArgumentError("the width grid needs 0 < w_min < w_max and nw ≥ 2"))
    prob.k1 > 0 && prob.c_wd > 0 && prob.chi_perp_factor > 0 && prob.rho_theta_factor > 0 ||
        throw(ArgumentError("k1, c_wd, chi_perp_factor and rho_theta_factor must be positive"))
    return nothing
end

# Value and ψ-derivative of a kinetic profile column at ψ.
function _profile_at(psi_xs, values, psi)
    itp = cubic_interp(psi_xs, collect(Float64, values))
    return Float64(itp(psi)), Float64(itp(psi; deriv=DerivOp(1)))
end


"""
    mre_roots(F, w_lo, w_grid) -> (; w_m, w_sat, w_sat_all, w_seed_all, w_seed_unsaturated, fully_stabilized, unstable_any_seed, unsaturated)

Roots of `F` on `[w_lo, maximum(w_grid)]`, bracketed on `w_lo` and the grid points above it and refined by Brent's
method: `w_m` is the smallest crossing from negative to positive, and `w_sat` the first crossing from positive to
negative above where an island starts, `w_lo` when F(w_lo) > 0, else `w_m` (`NaN` when absent). `unsaturated` is
F > 0 at the top of the grid.

Every saturated branch is returned too: `w_sat_all` holds each falling crossing in increasing order, and `w_seed_all[k]`
the rising crossing just below `w_sat_all[k]` (`NaN` when F > 0 from `w_lo` up to it). An island seeded between
`w_seed_all[k]` and the next seed threshold settles at `w_sat_all[k]`. `w_seed_unsaturated` is the rising crossing
above the last falling one when F > 0 at the top of the grid: seeds above it grow past the grid (`NaN` otherwise).
"""
function mre_roots(F, w_lo::Real, w_grid::AbstractVector)
    nodes = vcat(Float64(w_lo), filter(>(w_lo), w_grid))
    vals = F.(nodes)
    rising, falling = Float64[], Float64[]
    for j in 1:(length(nodes)-1)
        a, b = vals[j], vals[j+1]
        isfinite(a) && isfinite(b) && a != 0 && (a * b < 0 || b == 0) || continue
        push!(a < 0 ? rising : falling, find_zero(F, (nodes[j], nodes[j+1]), Brent()))
    end
    w_m = isempty(rising) ? NaN : first(rising)
    start = vals[1] > 0 ? nodes[1] : w_m
    above = filter(>(start), falling)
    w_sat = isempty(above) ? NaN : first(above)
    # F is continuous, so rising and falling crossings alternate; each falling one is reached from the rising one below it.
    seed_below(x) = (b = filter(<(x), rising); isempty(b) ? NaN : last(b))
    w_seed_all = seed_below.(above)
    top = isempty(above) ? filter(>(-Inf), rising) : filter(>(last(above)), rising)
    w_seed_unsaturated = vals[end] > 0 && !isempty(top) ? last(top) : NaN
    return (; w_m, w_sat, w_sat_all=above, w_seed_all, w_seed_unsaturated, fully_stabilized=all(<(0), vals), unstable_any_seed=vals[1] > 0,
        unsaturated=vals[end] > 0)
end

"""
    vacuum_island_field(equil, psi, n, q1, w) -> Float64

Resonant area-weighted field [T] whose vacuum island at the surface `psi` (toroidal mode number `n`, dq/dψ_N `q1`) has
full width `w` in ψ_N: the inverse of the perturbed equilibrium's island width, B = π |n q1| χ₁ w²/(8A), with χ₁ = 2πψ₀
and A the flux-surface area.
"""
function vacuum_island_field(equil, psi::Real, n::Integer, q1::Real, w::Real)
    area = Equilibrium.flux_surface_area(equil, Float64(psi), length(equil.rzphi_ys) - 1)
    return π * abs(n * q1) * 2π * equil.psio * w^2 / (8 * area)
end

# Per-surface scalars, flags and width rows of the result, in field order.
const _MRE_SCALARS = (:rational_psi, :rational_q, :m, :n, :delta_prime, :delta_prime_imag, :D_I, :D_R, :H, :D_nc, :f_trap, :mu_ratio, :zeff,
    :Wc_prefac, :chi_perp, :v_te, :R0, :omega_star_i, :omega_star_e, :omega_hat, :omega_hat_pi, :beta_p, :rho_theta, :w_ib, :eps, :gamma,
    :delta_r, :w_lo, :D_c, :W_d_layer, :alpha_s, :beta_s, :C_sat, :w_m, :w_sat, :w_seed_unsaturated, :n_sat, :b_m, :b_seed_unsaturated, :b_vac, :b_res, :delta_rmp, :w_v,
    :b_crit, :b_trigger, :w_crit)
const _MRE_FLAGS = (:fully_stabilized, :unstable_any_seed, :unsaturated, :multiple_saturated, :outside_plasma, :gate_disagrees, :interchange_unstable,
    :linear_unstable, :below_w_ib, :large_delta_prime_w, :complex_delta_prime, :saturation_unconverged)
const _MRE_TERMS = (:F, :classical, :saturation, :bootstrap, :curvature, :polarization, :forcing, :w_d)

# SLAYER's toroidal critical offset D_c and the W_d it implies at one surface, with χ⊥ for every χ input.
function _layer_critical_offset(equil, s, plasma, chi_perp)
    p = only(
        build_slayer_inputs(equil, [s], plasma.profiles; plasma.mu_i, plasma.mu_rho, plasma.zeff, chi_perp_e=chi_perp, chi_perp_i=chi_perp,
            chi_perp_Dc=chi_perp, chi_tor=chi_perp, tau_E=plasma.data.tau_E, P_perp_model=:chi_perp_e, dc_type=:toroidal,
            plasma.resistivity_model, plasma.lnLambda_form)
    )
    return p.dc_tmp, layer_transport_width(p.dr_val, p.dc_tmp)
end

# The linear tearing row at ψ in `tearing`, or `nothing`.
_tearing_row(tearing, psi) = findfirst(p -> isapprox(p, psi; atol=1e-8), tearing.rational_psi)

# What every surface shares: the plasma, the radial label, the current profile and the gate.
function _mre_context(prob::ModifiedRutherfordProblem)
    equil = prob.ffs.equil
    plasma = layer_plasma(equil; log=false)
    data = plasma.data
    usable(v) = v !== nothing && !all(iszero, v)
    usable(data.chi_e) || data.chi_e_default !== nothing ||
        throw(ArgumentError("ModifiedRutherfordProblem needs χ⊥,e: a chi_e profile or chi_e_default in the kinetic data"))
    gate = prob.tearing
    # Metres map to ψ_N with the toroidal-flux radial label's dr/dψ, the label SLAYER's layer widths use.
    r_at, dr_dpsi = radial_label(equil)
    return (; equil, plasma, data, psi_xs=collect(Float64, data.psi), usable, gate, r_at, dr_dpsi, J_z=current_density_profile(equil),
        forcing=prob.pe === nothing ? nothing : resonant_forcing(prob.pe),
        psi_lim=(first(equil.profiles.xs), last(equil.profiles.xs)),
        chi_gate=gate === nothing || !hasproperty(gate.model, :chi_perp_Dc) ? nothing : gate.model.chi_perp_Dc)
end

# Geometry, indices, plasma and term inputs at the k-th rational surface.
function _surface_inputs(prob::ModifiedRutherfordProblem, ctx, k, s)
    (; equil, plasma, data, psi_xs, usable, r_at, dr_dpsi) = ctx
    rg = s.restype
    rg === nothing && throw(ArgumentError("ModifiedRutherfordProblem: the surface at ψ = $(s.psifac) has no ResistGeometry"))
    psi = s.psifac
    m, n = s.m[1], s.n[1]
    g = mre_geometry(equil, psi, s.q1)
    D_I = rg.E + rg.F + rg.H - 0.25
    D_R = rg.E + rg.F + rg.H^2
    D_I < 0 || @warn "ModifiedRutherfordProblem: D_I = $D_I ≥ 0 at the $m/$n surface; its terms are NaN"

    # Plasma at the surface; the ion density falls back to n_e where the data have no n_i.
    prof = plasma.profiles
    n_e, dn_e = Float64(prof.n_e(psi)), Float64(prof.n_e(psi; deriv=DerivOp(1)))
    T_e, dT_e = Float64(prof.T_e(psi)), Float64(prof.T_e(psi; deriv=DerivOp(1)))
    T_i, dT_i = Float64(prof.T_i(psi)), Float64(prof.T_i(psi; deriv=DerivOp(1)))
    n_i, dn_i = usable(data.n_i) ? _profile_at(psi_xs, data.n_i, psi) : (n_e, dn_e)
    zeff = plasma.zeff isa Real ? Float64(plasma.zeff) : Float64(plasma.zeff(psi))
    chi_perp = prob.chi_perp_factor * (usable(data.chi_e) ? first(_profile_at(psi_xs, data.chi_e, psi)) : data.chi_e_default)
    ctx.chi_gate === nothing || isapprox(ctx.chi_gate, chi_perp; rtol=1e-6) ||
        @warn "ModifiedRutherfordProblem: the gate's chi_perp_Dc = $(ctx.chi_gate) differs from the w_d χ⊥ = $chi_perp at the $m/$n surface"
    D_nc, mu_ratio = bootstrap_coefficient(g, zeff)

    # Polarization inputs (Benjamin App. C); lengths in metres map to ψ_N with dr/dψ.
    m_i, z_i = plasma.mu_i * M_P, plasma.z_i
    v_te = sqrt(2 * E_CHG * T_e / M_E)
    v_ti = sqrt(2 * E_CHG * T_i / m_i)
    rho_theta = prob.rho_theta_factor * m_i * v_ti / (g.avg_bpol * E_CHG)
    rho_theta_psi = rho_theta / abs(dr_dpsi(psi))
    # Diamagnetic frequencies from the density gradients, in SLAYER's sign convention (charge −1 for electrons).
    omega_star_i = -n * T_i * (dn_i / n_i) / (z_i * g.psio)
    omega_star_e = n * T_e * (dn_e / n_e) / g.psio
    omega_model =
        prob.omega_model === :electron_diamagnetic ? omega_star_e * (1 + 0.25 * (dT_e / T_e) / (dn_e / n_e)) :
        n * (plasma.has_omega_E ? Float64(prof.omega_E(psi)) : 0.0)

    # Cylindrical: r_s Δ′ = −2m, referred back to ψ_N by K^(−2α), K = r_s dψ_N/dr, α = √(−D_I) (as delta_prime_to_rs_reference).
    K = r_at(psi) / abs(dr_dpsi(psi))
    dp_complex = prob.delta_prime_source === :computed ? prob.ffs.delta_prime.matrix[k, k] : complex(-2 * m * K^(-2 * sqrt(max(-D_I, 0.0))))
    # Saturation coefficients from the equilibrium current in the same radial label.
    sc = saturation_coefficient(ctx.J_z, r_at, x -> abs(dr_dpsi(x)), x -> Float64(equil.profiles.q_spline(x)),
        x -> Float64(equil.profiles.q_deriv(x)), psi; window=prob.saturation_window, ctx.psi_lim)
    # SLAYER's toroidal critical offset at this surface with this χ⊥, so the curvature term meets the gate at small w.
    D_c, W_d_layer = prob.curvature_model === :lutjens ? _layer_critical_offset(equil, s, plasma, chi_perp) : (NaN, NaN)
    # Callen's toroidal-flux radius (without the F/B₀R₀ factor of the layer label), his Δ_c and w_pol, in ψ_N.
    rho, drho = toroidal_flux_radius(equil, psi, s.q)
    delta_c = prob.delta_prime_source === :computed ? (rho / drho)^(2 * sqrt(max(-D_I, 0.0))) * real(dp_complex) / rho : -2 * m / rho
    w_pol_callen = callen_polarization_width(equil, plasma, psi, s.q)[2] / drho
    delta_prime = prob.delta_prime_factor * (prob.delta_prime_model === :callen ? delta_c * drho : real(dp_complex))
    # The coil forcing at this surface: Δ_RMP (dρ/dψ_N) w_v² is the coefficient of 1/w², proportional to B_vac.
    rf = ctx.forcing
    row = rf === nothing ? nothing : findfirst(p -> isapprox(p, psi; atol=1e-8), rf.rational_psi)
    b_vac, b_res, delta_rmp, w_v = row === nothing ? (NaN, NaN, NaN, NaN) : (rf.b_vac[row], rf.b_res[row], rf.delta_rmp[row], 2 * rf.hw_v[row])
    forcing_amp = row === nothing ? 0.0 : delta_rmp * drho * w_v^2
    return (; rational_psi=psi, rational_q=s.q, m, n, g, b_per_w2=vacuum_island_field(equil, psi, n, s.q1, 1.0), D_I, D_R, H=rg.H, D_nc=prob.D_nc_factor * D_nc, mu_ratio, zeff,
        chi_perp, v_te,
        omega_star_i, omega_star_e, omega_hat=something(prob.omega_ratio, prob.omega_factor * omega_model / omega_star_i),
        omega_hat_pi=1 + (dT_i / T_i) / (dn_i / n_i), beta_p=2 * MU_0 * n_i * E_CHG * T_i / g.avg_bpolsq, rho_theta, rho_theta_psi,
        w_ib=sqrt(g.eps) * rho_theta_psi, gradient_ratio=(dn_i / n_i) / (g.q1 / g.q), K, dp_complex,
        delta_prime, sc, D_c, W_d_layer, w_pol_callen, b_vac, b_res, delta_rmp, w_v, forcing_amp, has_forcing=row !== nothing)
end

# Each term and their switched sum F at the full width w, from a surface's inputs.
function _mre_terms(prob::ModifiedRutherfordProblem, si, w)
    (; g, D_I, D_R, n, m) = si
    model = prob.delta_prime_model === :callen ? :constant : prob.delta_prime_model
    wd = transport_width(w; prob.c_wd, si.chi_perp, si.v_te, R0=g.R_major, g.q, g.q1, n, m, g.Wc_prefac)
    cl = D_I < 0 ? classical_term(si.delta_prime, D_I, w; model) : NaN
    st = D_I < 0 ? saturation_term(si.sc.C_sat, si.K, D_I, w; model) : NaN
    bs = bootstrap_term(si.D_nc, wd, w; prob.k1)
    cv =
        D_I >= 0 ? NaN :
        prob.curvature_model === :lutjens ? lutjens_curvature_term(D_R, D_I, si.K, si.W_d_layer, w; model) :
        curvature_term(D_R, si.H, D_I, wd, w; prob.k1)
    pl =
        prob.pol_sign * (
            prob.pol_model === :full ?
            polarization_term(w; prob.c_pol, g.eps, si.rho_theta_psi, si.beta_p, si.gradient_ratio, si.omega_hat, si.omega_hat_pi, si.w_ib) :
            prob.pol_model === :reduced ? reduced_polarization_term(w; prob.c_pol, prob.C_pol, si.w_ib) :
            reduced_polarization_term(w; prob.c_pol, C_pol=1.0, w_ib=si.w_pol_callen)
        )
    fc = si.forcing_amp / w^2
    F0 = prob.classical * cl + prob.saturation * st + prob.bootstrap * bs + prob.curvature * cv + prob.polarization * pl
    return (; F=F0 + prob.forcing * fc, F0, classical=cl, saturation=st, bootstrap=bs, curvature=cv, polarization=pl, forcing=fc, w_d=wd)
end

# The linear tearing growth rate and resistive layer width in ψ_N from `tearing`, and the bottom of the root search.
function _gate(prob::ModifiedRutherfordProblem, ctx, psi)
    gate = ctx.gate
    row = gate === nothing ? nothing : _tearing_row(gate, psi)
    gamma = row === nothing ? NaN : gate.gamma[row]
    delta_r = row === nothing || isempty(gate.layer_widths) ? NaN : gate.layer_widths[row].delta_s_m / abs(ctx.dr_dpsi(psi))
    return (; gamma, delta_r, w_lo=isnan(delta_r) ? prob.w_min : max(delta_r, prob.w_min))
end

# Roots and flags at one surface.
function _surface_roots(prob::ModifiedRutherfordProblem, ctx, si, gt, w)
    F(x) = _mre_terms(prob, si, x).F
    r = mre_roots(F, gt.w_lo, w)
    interchange_unstable = si.D_R > 0
    linear_unstable = gt.gamma > 0
    excluded = interchange_unstable || si.D_I >= 0
    # A linearly unstable surface grows from noise, so it has no seed threshold; the MRE still sets its saturation.
    w_m, w_sat = excluded ? (NaN, NaN) : (linear_unstable ? NaN : r.w_m, r.w_sat)
    w_sat_all, w_seed_all = excluded ? (Float64[], Float64[]) : (r.w_sat_all, r.w_seed_all)
    w_seed_unsaturated = excluded ? NaN : r.w_seed_unsaturated
    psi = si.rational_psi
    # Resonant field of the vacuum island as wide as each seed threshold; B ∝ w².
    b(x) = si.b_per_w2 * x^2
    # Critical fields of the forcing from the unforced F; the forcing is (forcing_amp/b_vac) B_vac/w².
    F0(x) = _mre_terms(prob, si, x).F0
    th =
        excluded || !si.has_forcing ? (; b_crit=NaN, b_trigger=NaN, w_crit=NaN) :
        forcing_thresholds(F0, si.forcing_amp / si.b_vac, gt.w_lo, w, mre_roots(F0, gt.w_lo, w).w_m)
    return (; th..., w_m, w_sat, w_sat_all, w_seed_all, w_seed_unsaturated, n_sat=length(w_sat_all), b_m=b(w_m), b_seed_all=b.(w_seed_all),
        b_seed_unsaturated=b(w_seed_unsaturated),
        multiple_saturated=length(w_sat_all) > 1, fully_stabilized=!excluded && r.fully_stabilized, unstable_any_seed=!excluded && r.unstable_any_seed,
        unsaturated=!excluded && r.unsaturated, outside_plasma=isfinite(w_sat) && (psi - w_sat / 2 < ctx.psi_lim[1] || psi + w_sat / 2 > ctx.psi_lim[2]),
        gate_disagrees=!excluded && isfinite(gt.gamma) && (r.unstable_any_seed ? gt.gamma < 0 : gt.gamma > 0), interchange_unstable,
        linear_unstable, below_w_ib=w_m < si.w_ib,
        # Δ′(w) w at the root, dimensionless at any D_I.
        large_delta_prime_w=any(x -> isfinite(x) && abs(_mre_terms(prob, si, x).classical) * x >= 1, (w_m, w_sat)),
        complex_delta_prime=abs(imag(si.dp_complex)) > 1e-3 * abs(real(si.dp_complex)), saturation_unconverged=!si.sc.converged)
end

function CommonSolve.solve(prob::ModifiedRutherfordProblem)
    validate(prob)
    ctx = _mre_context(prob)
    w = exp.(range(log(prob.w_min), log(prob.w_max); length=prob.nw))
    rows = map(enumerate(prob.ffs.surfaces)) do (k, s)
        si = _surface_inputs(prob, ctx, k, s)
        gt = _gate(prob, ctx, si.rational_psi)
        terms = [_mre_terms(prob, si, wj) for wj in w]
        extra = (; si.g.f_trap, si.g.Wc_prefac, R0=si.g.R_major, si.g.eps, delta_prime_imag=imag(si.dp_complex), si.sc.alpha_s, si.sc.beta_s, si.sc.C_sat)
        rowterms = NamedTuple{_MRE_TERMS}(Tuple(getfield.(terms, key) for key in _MRE_TERMS))
        merge(si, extra, gt, _surface_roots(prob, ctx, si, gt, w), rowterms)
    end
    column(key) = [getfield(r, key) for r in rows]
    matrix(key) = isempty(rows) ? zeros(0, length(w)) : permutedims(reduce(hcat, column(key)))
    # Branches padded with NaN to the most found on any surface.
    nbranch = maximum((r.n_sat for r in rows); init=1)
    branches(key) = [k <= length(getfield(r, key)) ? getfield(r, key)[k] : NaN for r in rows, k in 1:nbranch]
    w_m = column(:w_m)
    finite_min(v) = all(isnan, v) ? NaN : minimum(filter(!isnan, v))
    return ModifiedRutherfordResult(; problem=prob, w=collect(w), (key => column(key) for key in _MRE_SCALARS)..., b_crit_star=finite_min(column(:b_crit)),
        w_sat_all=branches(:w_sat_all), w_seed_all=branches(:w_seed_all), b_seed_all=branches(:b_seed_all),
        b_m_star=all(isnan, column(:b_m)) ? NaN : minimum(filter(!isnan, column(:b_m))),
        w_m_star=all(isnan, w_m) ? NaN : minimum(filter(!isnan, w_m)), (key => Vector{Bool}(column(key)) for key in _MRE_FLAGS)...,
        (key => matrix(key) for key in _MRE_TERMS)...)
end
