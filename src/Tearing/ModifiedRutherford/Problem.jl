# Problem.jl
#
# `solve(ModifiedRutherfordProblem(ffs; ...))`: the terms of the steady modified Rutherford equation and their sum F(w)
# on a log-spaced grid of full island widths, per rational surface. Diagnostic only: no roots.

"""
    ModifiedRutherfordProblem(ffs; kwargs...)

The steady modified Rutherford equation (MRE) on every rational surface of a finished force-balance solve,

    F(w) = s_Δ Δ′(w) + s_sat Δ_sat(w) + s_nc Δ_nc(w) + s_GGJ Δ_GGJ(w) + s_pol Δ_pol(w),

with `w` the full island width in normalized poloidal flux ψ_N. Each term is evaluated on the grid whether or not its
switch is on; the switches select which terms `F` sums. Δ′ is the real part of the diagonal of `ffs.delta_prime`
(`delta_prime_source = :computed`) or the cylindrical vacuum value r_s Δ′ = −2m referred to ψ_N as SLAYER refers Δ′ to
r_s (`:cylindrical`, which needs no Δ′ solve);
the Mercier, resistive-interchange and Pfirsch-Schlüter indices D_I, D_R = E + F + H², H come from the surfaces'
`ResistGeometry`, and the plasma (densities, temperatures, Z_eff, χ⊥,e, E×B rotation, ion species) is the
equilibrium's kinetic data, as every tearing layer reads it. A deck states it as `[Tearing.ModifiedRutherford]`.

The roots of F are refined between grid points from `w_lo = max(δ_r, w_min)`: the seed threshold `w_m` is the
smallest crossing from F < 0 to F > 0 and the saturated width `w_sat` the largest from F > 0 to F < 0. Surfaces that
are resistive-interchange unstable (D_R > 0) get no roots; linearly tearing unstable ones (γ > 0 in `tearing`) grow
from noise, so they get no seed threshold but keep `w_sat`.

## Fields

  - `ffs` - Outer solution, single n, with a Δ′ matrix (Riccati with `vac_flag=true`, or Galerkin with `rpec_flag=true`)
    unless `delta_prime_source = :cylindrical`.
  - `tearing` - Linear tearing solve on the same Δ′ (the gate and the layer width δ_r), or `nothing`; set by the deck
    when `[Tearing]` is present.
  - `classical`, `saturation`, `bootstrap`, `curvature`, `polarization` - Term switches s_Δ, s_sat, s_nc, s_GGJ, s_pol.
    `saturation` is the finite-width correction to Δ′ ([`saturation_term`](@ref)), off by default.
  - `saturation_window` - Half-width in ψ_N of the quadratic fit that gives J_z′ and J_z″ in [`saturation_coefficient`](@ref).
  - `delta_prime_source` - `:computed` (Δ′_mn of `ffs`) or `:cylindrical` (r_s Δ′_mn = −2m, a reference case).
  - `delta_prime_model` - `:mercier_power` (Benjamin Eq. 10) or `:constant` (Δ′(w) = Δ′_mn).
  - `delta_prime_factor`, `D_nc_factor`, `rho_theta_factor` - Multipliers on Δ′_mn, D_nc and the poloidal ion Larmor
    radius ρ_θi (which also sets w_ib), for scans.
  - `curvature_model` - `:lutjens` (default; Lütjens et al. 2001, [`lutjens_curvature_term`](@ref), whose small-w limit is
    the SLAYER toroidal critical offset D_c of the linear gate) or `:benjamin` (Benjamin Eq. 12, [`curvature_term`](@ref),
    for comparison with the tearing-physics suite). `:lutjens` takes W_d and D_c from SLAYER's toroidal layer parameters
    built on each surface with this problem's χ⊥.
  - `k1` - Bootstrap coefficient k₁, and curvature coefficient of `curvature_model = :benjamin`.
  - `c_wd` - Prefactor of the transport threshold width w_d.
  - `chi_perp_factor` - Multiplier on χ⊥,e (the kinetic `chi_e` profile, else its `chi_e_default`).
  - `pol_model` - `:full` (Benjamin Eq. C.1) or `:reduced` (−c_pol w_pol²/w³, w_pol = C_pol w_ib).
  - `c_pol`, `C_pol` - Polarization coefficients.
  - `pol_sign` - ±1, multiplies the polarization term.
  - `omega_model` - Island frequency ω in the E_r = 0 frame: `:electron_diamagnetic`, ω = ω_*e(1 + η_e/4) (Wilson
    1996), or `:exb`, the lab-frame ω = n Ω_E from the kinetic `omega_E`.
  - `omega_factor` - Multiplier on the ω of `omega_model`; 0 is a stalled island.
  - `omega_ratio` - ω/ω_*i on every surface, replacing `omega_model`; `nothing` (default) uses the model.
  - `w_min`, `w_max`, `nw` - Log-spaced width grid in ψ_N.
"""
@kwdef struct ModifiedRutherfordProblem{R,T}
    ffs::R
    tearing::T = nothing
    classical::Bool = true
    bootstrap::Bool = true
    curvature::Bool = true
    polarization::Bool = false
    saturation::Bool = false
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
    ModifiedRutherfordResult

Output of `solve(::ModifiedRutherfordProblem)`. Per-surface vectors run core to edge; matrices are surface × width.

## Fields

  - `problem` - The producing problem.
  - `w` - Width grid in ψ_N.
  - `psi`, `q`, `m`, `n` - Surface identity.
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
  - `w_m_star` - Smallest `w_m` over the surfaces.
  - `fully_stabilized` - F < 0 on the whole search range.
  - `unstable_any_seed` - F > 0 at `w_lo`.
  - `interchange_unstable`, `linear_unstable` - D_R > 0 (excluded), or γ > 0 (no seed threshold; `w_sat` kept).
  - `below_w_ib` - `w_m` below the ion banana width, where the polarization ordering fails.
  - `large_delta_prime_w` - |Δ′| w ≥ 1 at `w_m` or `w_sat`.
  - `complex_delta_prime` - |Im Δ′_mn| > 10⁻³ |Re Δ′_mn|.
  - `saturation_unconverged` - `C_sat` not converged in the fit window (see [`saturation_coefficient`](@ref)).
  - `F`, `classical`, `saturation`, `bootstrap`, `curvature`, `polarization`, `w_d` - F(w), each term, and w_d(w).
"""
struct ModifiedRutherfordResult{P}
    problem::P
    w::Vector{Float64}
    psi::Vector{Float64}
    q::Vector{Float64}
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
    w_m_star::Float64
    fully_stabilized::Vector{Bool}
    unstable_any_seed::Vector{Bool}
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
    prob.delta_prime_model in (:mercier_power, :constant) || throw(ArgumentError("delta_prime_model must be :mercier_power or :constant"))
    prob.curvature_model in (:lutjens, :benjamin) || throw(ArgumentError("curvature_model must be :lutjens or :benjamin"))
    prob.pol_model in (:full, :reduced) || throw(ArgumentError("pol_model must be :full or :reduced"))
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
    mre_roots(F, w_lo, w_grid) -> (; w_m, w_sat, fully_stabilized, unstable_any_seed)

Roots of `F` on `[w_lo, maximum(w_grid)]`, bracketed on `w_lo` and the grid points above it and refined by Brent's
method: `w_m` is the smallest crossing from negative to positive, `w_sat` the largest from positive to negative
(`NaN` when absent).
"""
function mre_roots(F, w_lo::Real, w_grid::AbstractVector)
    nodes = vcat(Float64(w_lo), filter(>(w_lo), w_grid))
    vals = F.(nodes)
    w_m, w_sat = NaN, NaN
    for j in 1:(length(nodes)-1)
        a, b = vals[j], vals[j+1]
        isfinite(a) && isfinite(b) && a != 0 && (a * b < 0 || b == 0) || continue
        root = find_zero(F, (nodes[j], nodes[j+1]), Brent())
        a < 0 ? (isnan(w_m) && (w_m = root)) : (w_sat = root)
    end
    return (; w_m, w_sat, fully_stabilized=all(<(0), vals), unstable_any_seed=vals[1] > 0)
end

# SLAYER's toroidal critical offset D_c and the W_d it implies at one surface, with χ⊥ for every χ input.
function _layer_critical_offset(equil, s, loaded, data, chi_perp)
    zeff = _layer_zeff(loaded, data)
    p = only(
        build_slayer_inputs(equil, [s], loaded.profiles; mu_i=_layer_ion_mass(data), mu_rho=_layer_mass_per_electron(data, zeff), zeff,
            chi_perp_e=chi_perp, chi_perp_i=chi_perp, chi_perp_Dc=chi_perp, chi_tor=chi_perp, tau_E=data.tau_E, P_perp_model=:chi_perp_e,
            dc_type=:toroidal, resistivity_model=_build_resistivity_model(data.resistivity_model), lnLambda_form=data.lnLambda_form)
    )
    return p.dc_tmp, layer_transport_width(p.dr_val, p.dc_tmp)
end

# The linear tearing row at ψ in `tearing`, or `nothing`.
_tearing_row(tearing, psi) = findfirst(p -> isapprox(p, psi; atol=1e-8), tearing.rational_psi)

function CommonSolve.solve(prob::ModifiedRutherfordProblem)
    validate(prob)
    ffs = prob.ffs
    equil = ffs.equil
    data = equil.kinetic
    surfaces = ffs.surfaces
    ns = length(surfaces)
    w = exp.(range(log(prob.w_min), log(prob.w_max); length=prob.nw))

    # One plasma for every tearing consumer: the SLAYER view supplies n_e, T_e, T_i, Ω_E and the Z_eff closure.
    loaded = _slayer_profiles(data, nothing)
    psi_xs = collect(Float64, data.psi)
    usable(v) = v !== nothing && !all(iszero, v)
    usable(data.chi_e) || data.chi_e_default !== nothing ||
        throw(ArgumentError("ModifiedRutherfordProblem needs χ⊥,e: a chi_e profile or chi_e_default in the kinetic data"))
    species = data.ion_species
    m_i = sum(s.fraction * s.m for s in species) / sum(s.fraction for s in species) * M_P
    z_i = Float64(first(species).z)
    gate = prob.tearing
    # Metres map to ψ_N with the toroidal-flux radial label's dr/dψ, the label SLAYER's layer widths use.
    r_at, dr_dpsi = radial_label(equil)
    J_z = current_density_profile(equil)
    psi_lim = (first(equil.profiles.xs), last(equil.profiles.xs))
    chi_gate = gate === nothing || !hasproperty(gate.model, :chi_perp_Dc) ? nothing : gate.model.chi_perp_Dc

    vec0() = zeros(ns)
    flag0() = falses(ns)
    out = (; psi=vec0(), q=vec0(), m=zeros(Int, ns), n=zeros(Int, ns), delta_prime=vec0(), delta_prime_imag=vec0(), D_I=vec0(), D_R=vec0(),
        H=vec0(), D_nc=vec0(), f_trap=vec0(), mu_ratio=vec0(), zeff=vec0(), Wc_prefac=vec0(), chi_perp=vec0(), v_te=vec0(), R0=vec0(),
        omega_star_i=vec0(), omega_star_e=vec0(), omega_hat=vec0(), omega_hat_pi=vec0(), beta_p=vec0(), rho_theta=vec0(), w_ib=vec0(),
        eps=vec0(), gamma=vec0(), delta_r=vec0(), w_lo=vec0(), D_c=vec0(), W_d_layer=vec0(), alpha_s=vec0(), beta_s=vec0(), C_sat=vec0(), w_m=vec0(), w_sat=vec0())
    flags = (; fully_stabilized=flag0(), unstable_any_seed=flag0(), interchange_unstable=flag0(), linear_unstable=flag0(),
        below_w_ib=flag0(), large_delta_prime_w=flag0(), complex_delta_prime=flag0(),
        saturation_unconverged=flag0())
    mats = Dict(k => zeros(ns, prob.nw) for k in (:F, :classical, :saturation, :bootstrap, :curvature, :polarization, :w_d))

    for (k, s) in enumerate(surfaces)
        rg = s.restype
        rg === nothing && throw(ArgumentError("ModifiedRutherfordProblem: the surface at ψ = $(s.psifac) has no ResistGeometry"))
        psi = s.psifac
        m, n = s.m[1], s.n[1]
        g = mre_geometry(equil, psi, s.q1)
        D_I = rg.E + rg.F + rg.H - 0.25
        D_R = rg.E + rg.F + rg.H^2
        D_I < 0 || @warn "ModifiedRutherfordProblem: D_I = $D_I ≥ 0 at the $m/$n surface; its terms are NaN"

        # Plasma at the surface; the ion density falls back to n_e where the data have no n_i.
        n_e, dn_e = Float64(loaded.profiles.n_e(psi)), Float64(loaded.profiles.n_e(psi; deriv=DerivOp(1)))
        T_e, dT_e = Float64(loaded.profiles.T_e(psi)), Float64(loaded.profiles.T_e(psi; deriv=DerivOp(1)))
        T_i, dT_i = Float64(loaded.profiles.T_i(psi)), Float64(loaded.profiles.T_i(psi; deriv=DerivOp(1)))
        n_i, dn_i = usable(data.n_i) ? _profile_at(psi_xs, data.n_i, psi) : (n_e, dn_e)
        zeff = loaded.zeff === nothing ? data.zeff_default : Float64(loaded.zeff(psi))
        chi_perp = prob.chi_perp_factor * (usable(data.chi_e) ? first(_profile_at(psi_xs, data.chi_e, psi)) : data.chi_e_default)
        chi_gate === nothing || isapprox(chi_gate, chi_perp; rtol=1e-6) ||
            @warn "ModifiedRutherfordProblem: the gate's chi_perp_Dc = $chi_gate differs from the w_d χ⊥ = $chi_perp at the $m/$n surface"

        D_nc, mu_ratio = bootstrap_coefficient(g, zeff)
        D_nc *= prob.D_nc_factor

        # Polarization inputs (Benjamin App. C); lengths in metres map to ψ_N with dr/dψ.
        v_te = sqrt(2 * E_CHG * T_e / M_E)
        v_ti = sqrt(2 * E_CHG * T_i / m_i)
        rho_theta = prob.rho_theta_factor * m_i * v_ti / (g.avg_bpol * E_CHG)
        rho_theta_psi = rho_theta / abs(dr_dpsi(psi))
        w_ib = sqrt(g.eps) * rho_theta_psi
        beta_p = 2 * MU_0 * n_i * E_CHG * T_i / g.avg_bpolsq
        gradient_ratio = (dn_i / n_i) / (g.q1 / g.q)
        # Diamagnetic frequencies from the density gradients, in SLAYER's sign convention (charge −1 for electrons).
        omega_star_i = -n * T_i * (dn_i / n_i) / (z_i * g.psio)
        omega_star_e = n * T_e * (dn_e / n_e) / g.psio
        omega_hat_pi = 1 + (dT_i / T_i) / (dn_i / n_i)
        omega_model =
            prob.omega_model === :electron_diamagnetic ? omega_star_e * (1 + 0.25 * (dT_e / T_e) / (dn_e / n_e)) :
            n * (loaded.has_omega_E ? Float64(loaded.profiles.omega_E(psi)) : 0.0)
        omega_hat = something(prob.omega_ratio, prob.omega_factor * omega_model / omega_star_i)

        # Cylindrical: r_s Δ′ = −2m, referred back to ψ_N by K^(−2α), K = r_s dψ_N/dr, α = √(−D_I) (as delta_prime_to_rs_reference).
        K = r_at(psi) / abs(dr_dpsi(psi))
        dp_complex = prob.delta_prime_source === :computed ? ffs.delta_prime.matrix[k, k] : complex(-2 * m * K^(-2 * sqrt(max(-D_I, 0.0))))
        dp_mn = prob.delta_prime_factor * real(dp_complex)
        # Saturation coefficients from the equilibrium current in the same radial label.
        sc = saturation_coefficient(J_z, r_at, x -> abs(dr_dpsi(x)), x -> Float64(equil.profiles.q_spline(x)),
            x -> Float64(equil.profiles.q_deriv(x)), psi; window=prob.saturation_window, psi_lim)
        flags.saturation_unconverged[k] = !sc.converged
        # SLAYER's toroidal critical offset at this surface with this χ⊥, so the curvature term meets the gate at small w.
        D_c, W_d_layer = prob.curvature_model === :lutjens ? _layer_critical_offset(equil, s, loaded, data, chi_perp) : (NaN, NaN)
        function terms(wj)
            wd = transport_width(wj; prob.c_wd, chi_perp, v_te, R0=g.R_major, g.q, g.q1, n, m, g.Wc_prefac)
            cl = D_I < 0 ? classical_term(dp_mn, D_I, wj; model=prob.delta_prime_model) : NaN
            st = D_I < 0 ? saturation_term(sc.C_sat, K, D_I, wj; model=prob.delta_prime_model) : NaN
            bs = bootstrap_term(D_nc, wd, wj; prob.k1)
            cv =
                D_I >= 0 ? NaN :
                prob.curvature_model === :lutjens ? lutjens_curvature_term(D_R, D_I, K, W_d_layer, wj) :
                curvature_term(D_R, rg.H, D_I, wd, wj; prob.k1)
            pl =
                prob.pol_sign * (
                    prob.pol_model === :full ?
                    polarization_term(wj; prob.c_pol, g.eps, rho_theta_psi, beta_p,
                        gradient_ratio, omega_hat, omega_hat_pi, w_ib) :
                    reduced_polarization_term(wj; prob.c_pol, prob.C_pol, w_ib)
                )
            F = prob.classical * cl + prob.saturation * st + prob.bootstrap * bs + prob.curvature * cv + prob.polarization * pl
            return (; F, cl, st, bs, cv, pl, wd)
        end
        for (j, wj) in enumerate(w)
            t = terms(wj)
            mats[:F][k, j], mats[:w_d][k, j] = t.F, t.wd
            mats[:classical][k, j], mats[:saturation][k, j], mats[:bootstrap][k, j] = t.cl, t.st, t.bs
            mats[:curvature][k, j], mats[:polarization][k, j] = t.cv, t.pl
        end

        # Gates: the linear tearing growth rate and resistive layer width from `tearing`, and resistive interchange.
        row = gate === nothing ? nothing : _tearing_row(gate, psi)
        gamma = row === nothing ? NaN : gate.gamma[row]
        delta_r = row === nothing || isempty(gate.layer_widths) ? NaN : gate.layer_widths[row].delta_s_m / abs(dr_dpsi(psi))
        w_lo = isnan(delta_r) ? prob.w_min : max(delta_r, prob.w_min)
        flags.interchange_unstable[k] = D_R > 0
        flags.linear_unstable[k] = gamma > 0
        flags.complex_delta_prime[k] = abs(imag(dp_complex)) > 1e-3 * abs(real(dp_complex))
        r = mre_roots(wj -> terms(wj).F, w_lo, w)
        excluded = flags.interchange_unstable[k] || D_I >= 0
        # A linearly unstable surface grows from noise, so it has no seed threshold; the MRE still sets its saturation.
        w_m, w_sat = excluded ? (NaN, NaN) : (flags.linear_unstable[k] ? NaN : r.w_m, r.w_sat)
        flags.fully_stabilized[k], flags.unstable_any_seed[k] = !excluded && r.fully_stabilized, !excluded && r.unstable_any_seed
        flags.below_w_ib[k] = w_m < w_ib
        flags.large_delta_prime_w[k] = abs(dp_mn) * w_m >= 1 || abs(dp_mn) * w_sat >= 1

        for (key, val) in pairs((; psi, q=s.q, m, n, delta_prime=dp_mn, delta_prime_imag=imag(dp_complex), D_I, D_R, H=rg.H, D_nc,
            f_trap=g.f_trap, mu_ratio, zeff, g.Wc_prefac, chi_perp, v_te, R0=g.R_major, omega_star_i, omega_star_e, omega_hat, omega_hat_pi,
            beta_p, rho_theta, w_ib, g.eps, gamma, delta_r, w_lo, D_c, W_d_layer, sc.alpha_s, sc.beta_s, sc.C_sat, w_m, w_sat))
            getfield(out, key)[k] = val
        end
    end
    valid = filter(!isnan, out.w_m)
    w_m_star = isempty(valid) ? NaN : minimum(valid)
    return ModifiedRutherfordResult(prob, collect(w), out.psi, out.q, out.m, out.n, out.delta_prime, out.delta_prime_imag, out.D_I, out.D_R,
        out.H, out.D_nc, out.f_trap, out.mu_ratio, out.zeff, out.Wc_prefac, out.chi_perp, out.v_te, out.R0, out.omega_star_i,
        out.omega_star_e, out.omega_hat, out.omega_hat_pi, out.beta_p, out.rho_theta, out.w_ib, out.eps, out.gamma, out.delta_r, out.w_lo,
        out.D_c, out.W_d_layer, out.alpha_s, out.beta_s, out.C_sat, out.w_m, out.w_sat, w_m_star, Vector{Bool}(flags.fully_stabilized), Vector{Bool}(flags.unstable_any_seed),
        Vector{Bool}(flags.interchange_unstable), Vector{Bool}(flags.linear_unstable), Vector{Bool}(flags.below_w_ib),
        Vector{Bool}(flags.large_delta_prime_w), Vector{Bool}(flags.complex_delta_prime),
        Vector{Bool}(flags.saturation_unconverged),
        mats[:F], mats[:classical], mats[:saturation], mats[:bootstrap], mats[:curvature], mats[:polarization], mats[:w_d])
end
