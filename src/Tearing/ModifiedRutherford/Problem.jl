# Problem.jl
#
# `solve(ModifiedRutherfordProblem(ffs; ...))`: the terms of the steady modified Rutherford equation and their sum F(w)
# on a log-spaced grid of full island widths, per rational surface. Diagnostic only: no roots.

"""
    ModifiedRutherfordProblem(ffs; kwargs...)

The steady modified Rutherford equation (MRE) on every rational surface of a finished force-balance solve,

    F(w) = s_Δ Δ′(w) + s_nc Δ_nc(w) + s_GGJ Δ_GGJ(w) + s_pol Δ_pol(w),

with `w` the full island width in normalized poloidal flux ψ_N. Each term is evaluated on the grid whether or not its
switch is on; the switches select which terms `F` sums. Δ′ is the real part of the diagonal of `ffs.delta_prime`,
the Mercier, resistive-interchange and Pfirsch-Schlüter indices D_I, D_R = E + F + H², H come from the surfaces'
`ResistGeometry`, and the plasma (densities, temperatures, Z_eff, χ⊥,e, E×B rotation, ion species) is the
equilibrium's kinetic data, as every tearing layer reads it. A deck states it as `[Tearing.ModifiedRutherford]`.

## Fields

  - `ffs` - Outer solution with a Δ′ matrix (Riccati with `vac_flag=true`, or Galerkin with `rpec_flag=true`), single n.
  - `classical`, `bootstrap`, `curvature`, `polarization` - Term switches s_Δ, s_nc, s_GGJ, s_pol.
  - `delta_prime_model` - `:mercier_power` (Benjamin Eq. 10) or `:constant` (Δ′(w) = Δ′_mn).
  - `k1` - Bootstrap and curvature coefficient k₁.
  - `c_wd` - Prefactor of the transport threshold width w_d.
  - `chi_perp_factor` - Multiplier on χ⊥,e (the kinetic `chi_e` profile, else its `chi_e_default`).
  - `pol_model` - `:full` (Benjamin Eq. C.1) or `:reduced` (−c_pol w_pol²/w³, w_pol = C_pol w_ib).
  - `c_pol`, `C_pol` - Polarization coefficients.
  - `pol_sign` - ±1, multiplies the polarization term.
  - `omega_factor` - Multiplier on the island frequency ω = n Ω_E from the kinetic `omega_E`; 0 is a stalled island.
  - `omega_ratio` - ω/ω_*i on every surface, replacing the one from `omega_E`; `nothing` (default) uses `omega_E`.
  - `w_min`, `w_max`, `nw` - Log-spaced width grid in ψ_N.
"""
@kwdef struct ModifiedRutherfordProblem{R}
    ffs::R
    classical::Bool = true
    bootstrap::Bool = true
    curvature::Bool = true
    polarization::Bool = false
    delta_prime_model::Symbol = :mercier_power
    k1::Float64 = 1.7
    c_wd::Float64 = 1.0
    chi_perp_factor::Float64 = 1.0
    pol_model::Symbol = :full
    c_pol::Float64 = 1.0
    C_pol::Float64 = 1.9
    pol_sign::Float64 = 1.0
    omega_factor::Float64 = 1.0
    omega_ratio::Union{Nothing,Float64} = nothing
    w_min::Float64 = 1e-5
    w_max::Float64 = 0.2
    nw::Int = 200
end

ModifiedRutherfordProblem(ffs; kwargs...) = ModifiedRutherfordProblem{typeof(ffs)}(; ffs=ffs, kwargs...)

"""
    ModifiedRutherfordResult

Output of `solve(::ModifiedRutherfordProblem)`. Per-surface vectors run core to edge; matrices are surface × width.

## Fields

  - `problem` - The producing problem.
  - `w` - Width grid in ψ_N.
  - `psi`, `q`, `m`, `n` - Surface identity.
  - `delta_prime`, `D_I`, `D_R`, `H` - Outer-region and layer indices.
  - `D_nc`, `f_trap`, `mu_ratio`, `zeff` - Bootstrap coefficient, Sauter 2002 trapped fraction, μ_e/(ν_e + μ_e), Z_eff.
  - `Wc_prefac`, `chi_perp`, `v_te`, `R0` - Inputs of w_d.
  - `omega_star_i`, `omega_hat`, `omega_hat_pi` - Ion diamagnetic frequency [rad/s], ω/ω_*i and ω_*pi/ω_*i.
  - `beta_p`, `rho_theta`, `w_ib`, `eps` - Polarization inputs: poloidal ion beta, poloidal ion Larmor radius [m],
    banana width in ψ_N, inverse aspect ratio.
  - `F`, `classical`, `bootstrap`, `curvature`, `polarization`, `w_d` - F(w), each term, and w_d(w).
"""
struct ModifiedRutherfordResult{P}
    problem::P
    w::Vector{Float64}
    psi::Vector{Float64}
    q::Vector{Float64}
    m::Vector{Int}
    n::Vector{Int}
    delta_prime::Vector{Float64}
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
    omega_hat::Vector{Float64}
    omega_hat_pi::Vector{Float64}
    beta_p::Vector{Float64}
    rho_theta::Vector{Float64}
    w_ib::Vector{Float64}
    eps::Vector{Float64}
    F::Matrix{Float64}
    classical::Matrix{Float64}
    bootstrap::Matrix{Float64}
    curvature::Matrix{Float64}
    polarization::Matrix{Float64}
    w_d::Matrix{Float64}
end

function validate(prob::ModifiedRutherfordProblem)
    ffs = prob.ffs
    ffs.npert == 1 || throw(ArgumentError("ModifiedRutherfordProblem is single-n; the solve spans n = $(ffs.nlow):$(ffs.nhigh)"))
    dp = ffs.delta_prime
    dp === nothing && throw(ArgumentError("ModifiedRutherfordProblem needs Δ′: solve with Riccati() and vac_flag=true, or Galerkin(; rpec_flag=true)"))
    size(dp.matrix, 1) == length(ffs.surfaces) ||
        throw(ArgumentError("ModifiedRutherfordProblem: the Δ′ matrix is $(size(dp.matrix, 1))×$(size(dp.matrix, 2)) for $(length(ffs.surfaces)) surfaces"))
    ffs.equil.kinetic === nothing && throw(ArgumentError("ModifiedRutherfordProblem needs kinetic profiles; set kinetic_file in [Equilibrium]"))
    prob.delta_prime_model in (:mercier_power, :constant) || throw(ArgumentError("delta_prime_model must be :mercier_power or :constant"))
    prob.pol_model in (:full, :reduced) || throw(ArgumentError("pol_model must be :full or :reduced"))
    abs(prob.pol_sign) == 1 || throw(ArgumentError("pol_sign must be +1 or -1"))
    0 < prob.w_min < prob.w_max && prob.nw >= 2 || throw(ArgumentError("the width grid needs 0 < w_min < w_max and nw ≥ 2"))
    prob.k1 > 0 && prob.c_wd > 0 && prob.chi_perp_factor > 0 || throw(ArgumentError("k1, c_wd and chi_perp_factor must be positive"))
    return nothing
end

# Value and ψ-derivative of a kinetic profile column at ψ.
function _profile_at(psi_xs, values, psi)
    itp = cubic_interp(psi_xs, collect(Float64, values))
    return Float64(itp(psi)), Float64(itp(psi; deriv=DerivOp(1)))
end

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

    vec0() = zeros(ns)
    out = (; psi=vec0(), q=vec0(), m=zeros(Int, ns), n=zeros(Int, ns), delta_prime=vec0(), D_I=vec0(), D_R=vec0(), H=vec0(),
        D_nc=vec0(), f_trap=vec0(), mu_ratio=vec0(), zeff=vec0(), Wc_prefac=vec0(), chi_perp=vec0(), v_te=vec0(), R0=vec0(),
        omega_star_i=vec0(), omega_hat=vec0(), omega_hat_pi=vec0(), beta_p=vec0(), rho_theta=vec0(), w_ib=vec0(), eps=vec0())
    mats = Dict(k => zeros(ns, prob.nw) for k in (:F, :classical, :bootstrap, :curvature, :polarization, :w_d))

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
        n_e = Float64(loaded.profiles.n_e(psi))
        T_e = Float64(loaded.profiles.T_e(psi))
        T_i, dT_i = Float64(loaded.profiles.T_i(psi)), Float64(loaded.profiles.T_i(psi; deriv=DerivOp(1)))
        n_i, dn_i = usable(data.n_i) ? _profile_at(psi_xs, data.n_i, psi) : (n_e, Float64(loaded.profiles.n_e(psi; deriv=DerivOp(1))))
        zeff = loaded.zeff === nothing ? data.zeff_default : Float64(loaded.zeff(psi))
        chi_perp = prob.chi_perp_factor * (usable(data.chi_e) ? first(_profile_at(psi_xs, data.chi_e, psi)) : data.chi_e_default)

        D_nc, mu_ratio = bootstrap_coefficient(g, zeff)

        # Polarization inputs (Benjamin App. C); widths in metres map to ψ_N by w/(2ψ_s) ↔ ρ/r_s.
        v_te = sqrt(2 * E_CHG * T_e / M_E)
        v_ti = sqrt(2 * E_CHG * T_i / m_i)
        rho_theta = m_i * v_ti / (g.avg_bpol * E_CHG)
        w_ib = sqrt(g.eps) * rho_theta * 2 * psi / g.r_minor
        beta_p = 2 * MU_0 * n_i * E_CHG * T_i / g.avg_bpolsq
        gradient_ratio = (dn_i / n_i) / (g.q1 / g.q)
        # Ion diamagnetic frequency from the density gradient, in SLAYER's sign convention; ω_*pi adds the temperature gradient.
        omega_star_i = -n * T_i * (dn_i / n_i) / (z_i * g.psio)
        omega_hat_pi = 1 + (dT_i / T_i) / (dn_i / n_i)
        omega = prob.omega_factor * n * (loaded.has_omega_E ? Float64(loaded.profiles.omega_E(psi)) : 0.0)
        omega_hat = something(prob.omega_ratio, omega / omega_star_i)

        dp_mn = real(ffs.delta_prime.matrix[k, k])
        for (j, wj) in enumerate(w)
            wd = transport_width(wj; prob.c_wd, chi_perp, v_te, R0=g.R_major, g.q, g.q1, n, m, g.Wc_prefac)
            cl = D_I < 0 ? classical_term(dp_mn, D_I, wj; model=prob.delta_prime_model) : NaN
            bs = bootstrap_term(D_nc, wd, wj; prob.k1)
            cv = D_I < 0 ? curvature_term(D_R, rg.H, D_I, wd, wj; prob.k1) : NaN
            pl = prob.pol_sign * (prob.pol_model === :full ?
                                  polarization_term(wj; prob.c_pol, g.eps, rho_theta_over_rs=rho_theta / g.r_minor, psi_s=psi, beta_p,
                gradient_ratio, omega_hat, omega_hat_pi, w_ib) :
                                  reduced_polarization_term(wj; prob.c_pol, prob.C_pol, w_ib))
            mats[:w_d][k, j] = wd
            mats[:classical][k, j], mats[:bootstrap][k, j], mats[:curvature][k, j], mats[:polarization][k, j] = cl, bs, cv, pl
            mats[:F][k, j] = prob.classical * cl + prob.bootstrap * bs + prob.curvature * cv + prob.polarization * pl
        end
        for (key, val) in pairs((; psi, q=s.q, m, n, delta_prime=dp_mn, D_I, D_R, H=rg.H, D_nc, f_trap=g.f_trap, mu_ratio, zeff,
            g.Wc_prefac, chi_perp, v_te, R0=g.R_major, omega_star_i, omega_hat, omega_hat_pi, beta_p, rho_theta, w_ib, g.eps))
            getfield(out, key)[k] = val
        end
    end
    return ModifiedRutherfordResult(prob, collect(w), out.psi, out.q, out.m, out.n, out.delta_prime, out.D_I, out.D_R, out.H,
        out.D_nc, out.f_trap, out.mu_ratio, out.zeff, out.Wc_prefac, out.chi_perp, out.v_te, out.R0, out.omega_star_i,
        out.omega_hat, out.omega_hat_pi, out.beta_p, out.rho_theta, out.w_ib, out.eps,
        mats[:F], mats[:classical], mats[:bootstrap], mats[:curvature], mats[:polarization], mats[:w_d])
end
