# LayerParameters.jl
#
# `SLAYERParameters` carries the dimensionless layer-physics parameters
# that the Fitzpatrick layer Riccati ODE consumes for one rational surface,
# plus the dimensional conversion factors needed to translate normalized
# frequencies and Δ values back to physical units.
#
# The constructor builds the per-surface state from dimensional equilibrium
# and kinetic-profile inputs. The Fitzpatrick two-fluid layer uses
# P_perp/P_tor/D_norm rather than the older magnetic/electron Prandtl
# (pr/pe) and ρ_s-based (ds) parametrization. Q is not stored — it is
# passed directly to `solve_inner`.

"""
    SLAYERParameters

Dimensionless layer-physics parameters at one rational surface for the
Fitzpatrick two-fluid drift-MHD SLAYER inner-layer model (Fitzpatrick
2023; Park et al. 2022), plus dimensional auxiliaries required for
de-normalization. The parametrization uses `P_perp`, `P_tor`, and
`D_norm` (not the older `pr`/`pe`/`ds` set).

| field      | meaning                                                           |
|:---------- |:----------------------------------------------------------------- |
| `ising`    | Singular-surface index (traceability only)                        |
| `m`, `n`   | Poloidal / toroidal mode numbers at this surface                  |
| `tau`      | T_i / T_e                                                         |
| `lu`       | Lundquist number S = τ_R / τ_H                                    |
| `c_beta`   | Compressibility √(β_local / (1 + β_local))                        |
| `D_norm`   | (d_β/r_s) · S^(1/3) · √(τ/(1+τ))  (Fitzpatrick normalized scale)  |
| `P_perp`   | Perpendicular Prandtl number τ_R / τ_⊥                            |
| `P_tor`    | Toroidal-direction Prandtl number τ_R / τ_‖tor                    |
| `Q_e`      | Normalized electron diamagnetic: −tauk · ω_*e                     |
| `Q_i`      | Normalized ion diamagnetic:      −tauk · ω_*i                     |
| `iota_e`   | Q_e / (Q_e − Q_i)                                                 |
| `tauk`     | Q-conversion factor S^(1/3) · τ_H  [s] — multiplies ω to get Q    |
| `tau_r`    | Resistive diffusion time [s]                                      |
| `delta_n`  | Δ-normalization factor S^(1/3) / r_s [m⁻¹]                        |
| `rs`       | Minor radius at this surface [m]                                  |
| `R0`       | Major radius [m]                                                  |
| `bt`       | Toroidal field [T]                                                |
| `sval_r`   | r-based magnetic shear r_s · (dq/dr) / q (Fitzpatrick convention) |
| `dr_val`   | Radial width parameter at surface (input to dc_tmp)               |
| `dgeo_val` | Geometric Δ (Shafranov shift factor)                              |
| `eta`      | Parallel resistivity entering τ_R = μ₀r_s²/η [Ω·m]                |
| `d_beta`   | Beta-weighted ion length scale c_β · d_i [m]                      |
| `dc_tmp`   | Critical-Δ offset from chi_parallel matching                      |
| `dc_type`  | Selector for `dc_tmp` formula                                     |

The complex normalized growth rate `Q = ω + iγ` is **not** stored here;
it is passed as a separate argument to `solve_inner`.
"""
Base.@kwdef struct SLAYERParameters <: InnerLayerParameters
    # Surface identity
    ising::Int = 0
    m::Int = 0
    n::Int = 0

    # Normalized layer parameters consumed by riccati_f
    tau::Float64
    lu::Float64
    c_beta::Float64
    D_norm::Float64
    P_perp::Float64
    P_tor::Float64
    Q_e::Float64
    Q_i::Float64
    iota_e::Float64

    # Conversion factors (Q ↔ ω in rad/s)
    tauk::Float64
    tau_r::Float64
    delta_n::Float64

    # Geometric / fluid auxiliaries
    rs::Float64
    R0::Float64
    bt::Float64
    sval_r::Float64
    dr_val::Float64 = 0.0
    dgeo_val::Float64 = 0.0
    eta::Float64
    d_beta::Float64

    # Critical-Δ offset
    dc_tmp::Float64 = 0.0
    dc_type::Symbol = :none
end

# Allowed dc_type values for the critical-Δ offset. `:none` is the default
# `dc_tmp = 0` branch.
const ALLOWED_DC_TYPES = (:none, :lar, :rfitzp, :toroidal)

"""
    r_based_shear(rs, q, dq_dpsi, da_dpsi) -> Float64

Convert a ψ-based shear to the r-based (Fitzpatrick) convention used
throughout SLAYER:

```
s_r = r_s · (dq/dr) / q  =  r_s · (dq/dψ) / (q · da/dψ)
```

`rs` is the minor radius at the surface, `q` the safety factor,
`dq_dpsi` the radial derivative of q with respect to ψ, and `da_dpsi`
the derivative of the surface minor radius with respect to ψ. The two
ψ derivatives must use the **same** ψ convention (i.e., both with
respect to ψ_norm or both with respect to physical ψ — the conversion
factor cancels in the ratio).

Equivalent to the conversion `s_Fitz = s_psiN · r_s / (psi_N · da_dpsiN)`.
"""
function r_based_shear(rs::Real, q::Real, dq_dpsi::Real, da_dpsi::Real)
    da_dpsi != 0 || throw(ArgumentError("r_based_shear: da/dψ must be non-zero"))
    q != 0 || throw(ArgumentError("r_based_shear: q must be non-zero"))
    return rs * dq_dpsi / (q * da_dpsi)
end

# Internal: solve the Wd self-consistency loop for the chi_parallel-based
# critical Δ (Connor-Hastie-Helander 2015). Returns dc_tmp as a Float64.
function _solve_dc_tmp(; dc_type::Symbol, dr_val::Real, dgeo_val::Real,
    chi_perp::Real, t_e::Real, zeff::Real, tau_ee::Real,
    rs::Real, R0::Real, sval_r::Real, n_tor::Integer,
    max_iter::Integer=100, tol::Real=1e-10)
    dc_type in ALLOWED_DC_TYPES ||
        throw(ArgumentError("SLAYERParameters: unknown dc_type=$dc_type. " *
                            "Allowed: $(ALLOWED_DC_TYPES)"))
    (dc_type === :none || dr_val == 0.0) && return 0.0

    vte = sqrt(2.0 * t_e * E_CHG / M_E)
    chi_par_smfp = (1.581 * tau_ee * vte^2) / (1.0 + 0.2535 * zeff)

    Wd = 0.1
    converged = false
    for _ in 1:max_iter
        chi_par_lmfp = (2.0 * R0 * vte) / (sqrt(π) * n_tor * sval_r * Wd)
        chi_par = (chi_par_smfp * chi_par_lmfp) /
                  (chi_par_smfp + chi_par_lmfp)
        Wd_new = sqrt(8.0) * (chi_perp / chi_par)^0.25 *
                 (1.0 / sqrt((rs / R0) * sval_r * n_tor))
        if abs(Wd_new - Wd) / max(abs(Wd), 1e-30) < tol
            Wd = Wd_new
            converged = true
            break
        end
        Wd = Wd_new
    end
    converged || error("SLAYERParameters: Wd iteration failed to converge")

    chi_par_lmfp = (2.0 * R0 * vte) / (sqrt(π) * n_tor * sval_r * Wd)
    chi_par = (chi_par_smfp * chi_par_lmfp) / (chi_par_smfp + chi_par_lmfp)

    if dc_type === :lar
        return 0.5 * (-dr_val) * π^1.5 *
               (chi_par / chi_perp)^0.25 *
               sqrt((n_tor * sval_r) / (R0 * rs))
    elseif dc_type === :rfitzp
        return -(sqrt(2.0) * π^1.5 * dr_val) / Wd
    elseif dc_type === :toroidal
        return 0.5 * (-dr_val) * π^1.5 *
               (chi_par / chi_perp)^0.25 * dgeo_val
    end
    return 0.0
end

"""
    slayer_parameters(; n_e, t_e, t_i, omega, omega_e, omega_i,
                        qval, sval_r, bt, rs, R0, mu_i, zeff,
                        chi_perp, chi_tor,
                        m, n,
                        dr_val=0.0, dgeo_val=0.0,
                        dc_type=:none, ising=0,
                        resistivity_model=SauterNeoModel(),
                        f_trap=nothing, nu_e_star=nothing,
                        R_major_eff=nothing,
                        lnLambda_form=:nrl)
        -> SLAYERParameters

Build a `SLAYERParameters` for one rational surface from dimensional
equilibrium and kinetic-profile inputs, in the Fitzpatrick two-fluid layer
parametrization (P_perp/P_tor/D_norm; the older magnetic/electron Prandtl
`pr`/`pe` and ρ_s-based `ds` parameters are not used).

# Arguments

  - `n_e` -- electron density [m⁻³]
  - `t_e` -- electron temperature [eV]
  - `t_i` -- ion temperature [eV]
  - `omega`   -- toroidal rotation frequency at the surface [rad/s]
  - `omega_e` -- electron diamagnetic frequency [rad/s]
  - `omega_i` -- ion diamagnetic frequency [rad/s]
  - `qval`    -- safety factor q at the surface
  - `sval_r`  -- **r-based** magnetic shear r·(dq/dr)/q (Fitzpatrick).
    Use `r_based_shear` to convert from ψ-based shear.
  - `bt`      -- toroidal field [T]
  - `rs`      -- minor radius at the surface [m]
  - `R0`      -- major radius [m]
  - `mu_i`    -- ion mass in proton-mass units (e.g. 2.0 for D)
  - `zeff`    -- effective charge
  - `chi_perp`, `chi_tor` -- perpendicular / toroidal heat diffusivity [m²/s]
  - `m`, `n`  -- poloidal / toroidal mode numbers at the surface
  - `dr_val`, `dgeo_val` -- inputs for the critical-Δ formula
  - `dc_type` -- one of `:none`, `:lar`, `:rfitzp`, `:toroidal`
  - `ising`   -- singular-surface index for traceability

# Resistivity kwargs

The selected resistivity sets the resistive diffusion time
`τ_R = μ₀ r_s² / η` and hence the Lundquist number and every normalized
layer parameter derived from it.

  - `resistivity_model` -- `SauterNeoModel()` (default: Sauter 1999 F_33
    trapped-particle correction, the physically-appropriate choice for
    H-mode tearing stability), `RedlNeoModel()` (improved high-ν* fit),
    `SpitzerModel()` (Sauter 18a fit, no trapped-particle correction), or
    `SpitzerHarmModel()` (Fitzpatrick Spitzer-Härm σ_∥ — the legacy SLAYER
    closure; pair with `lnLambda_form=:wesson` to reproduce legacy τ_R
    exactly).
  - `f_trap`  -- trapped-particle fraction at this surface. If not provided
    with a neoclassical model, falls back to Lin-Liu-Miller ε-only form
    with `ε = rs / (R_major_eff or R0)`.
  - `nu_e_star` -- electron collisionality. If `nothing` with a neoclassical
    model, computed from Sauter 1999 Eq. 18b using the same ε.
  - `R_major_eff` -- ⟨R⟩ at the surface for the ν*_e formula (default `R0`).
  - `lnLambda_form` -- `:nrl` (default), `:sauter`, or `:wesson` (legacy
    form).

# Sign convention for diamagnetic frequencies

The diamagnetic normalization uses

```
Q_e = -tauk · ω_*e
Q_i = -tauk · ω_*i
```

For the standard plasma-physics input where ω_*e is tabulated negative and
ω_*i positive (electrons and ions drifting in opposite directions), this
produces `Q_e > 0, Q_i < 0`, matching the opposite-drift expectation of the
dispersion relation.
"""
function slayer_parameters(;
    n_e::Real, t_e::Real, t_i::Real,
    omega::Real, omega_e::Real, omega_i::Real,
    qval::Real, sval_r::Real, bt::Real,
    rs::Real, R0::Real, mu_i::Real, zeff::Real,
    chi_perp::Real, chi_tor::Real,
    m::Integer, n::Integer,
    dr_val::Real=0.0, dgeo_val::Real=0.0,
    dc_type::Symbol=:none, ising::Integer=0,
    resistivity_model::NeoResistivityModel=SauterNeoModel(),
    f_trap::Union{Real,Nothing}=nothing,
    nu_e_star::Union{Real,Nothing}=nothing,
    R_major_eff::Union{Real,Nothing}=nothing,
    lnLambda_form::Symbol=:nrl)

    # Coulomb logarithm shared by the resistivity closure and τ_ee.
    lnLamb = coulomb_log_e(n_e, t_e; form=lnLambda_form)

    # Parallel resistivity entering τ_R. The model selects the closure:
    #   SpitzerHarmModel — Fitzpatrick Spitzer-Härm σ_∥; with
    #     lnLambda_form=:wesson this is bit-identical to the legacy SLAYER η.
    #   SpitzerModel     — Sauter 18a fit (legacy 1.65e-9 form under :wesson).
    #   SauterNeoModel / RedlNeoModel — F_33 trapped-particle correction
    #     using f_trap and ν*_e (from ResistGeometry or ε-only fallback).
    if resistivity_model isa SpitzerModel && lnLambda_form === :wesson
        # Preserve bit-identical legacy η diagnostic behaviour.
        eta = 1.65e-9 * lnLamb / (t_e / 1e3)^1.5
    elseif resistivity_model isa Union{SpitzerModel,SpitzerHarmModel}
        eta = eta_neoclassical(resistivity_model, n_e, t_e, zeff,
            0.0, 0.0; lnLamb=lnLamb)
    else
        R_eff = R_major_eff === nothing ? R0 : Float64(R_major_eff)
        eps_here = clamp(rs / R_eff, 1e-6, 1.0 - 1e-6)
        ft_here = f_trap === nothing ? trapped_fraction_eps(eps_here) :
                  Float64(f_trap)
        nue_here = nu_e_star === nothing ?
                   nu_star_e(n_e, t_e, R_eff, eps_here, qval, zeff;
            lnLamb=lnLamb) :
                   Float64(nu_e_star)
        eta = eta_neoclassical(resistivity_model, n_e, t_e, zeff,
            ft_here, nue_here; lnLamb=lnLamb)
    end

    # Basic plasma quantities
    tau = t_i / t_e
    rho = mu_i * M_P * n_e

    # Electron-electron collision time (still needed by the χ_∥-matching
    # critical-Δ regardless of the resistivity model).
    tau_ee = tau_ee_spitzer_harm(n_e, t_e; lnLamb=lnLamb)

    # Characteristic field, Alfven speed, length scales, fundamental
    # timescales.
    rho_s = 1.02e-4 * sqrt(mu_i * t_e) / bt                 # ion Larmor [m]
    d_i = sqrt((mu_i * M_P) / (n_e * E_CHG^2 * MU_0))     # ion skin depth [m]

    # Alfven time uses minor-radius shear directly (sval enters the
    # b_l = (n/m) r_s sval bt / R0 expression and cancels through to
    # tau_h = R0 sqrt(mu0 rho) / (n sval bt)).
    tau_h = R0 * sqrt(MU_0 * rho) / (n * sval_r * bt)
    # Resistive diffusion time τ_R = μ₀ r_s² / η (Fitzpatrick 2023), with
    # the selected η closure — neoclassical by default — setting the
    # Lundquist number.
    tau_r = MU_0 * rs^2 / eta

    # Lundquist number and Q-conversion factor
    lu = tau_r / tau_h
    tauk = lu^(1.0 / 3.0) * tau_h         # = Qconv

    # Normalized diamagnetic frequencies, Q = -tauk·ω; see docstring sign
    # convention.
    Q_e = -tauk * omega_e
    Q_i = -tauk * omega_i
    Q_e_minus_Q_i = Q_e - Q_i
    # iota_e = Q_e/(Q_e - Q_i) is singular when the electron and ion
    # diamagnetic frequencies are degenerate (ω_*e == ω_*i). The downstream
    # Riccati coefficients divide by iota_e, so silently setting it to 0 (or
    # Inf) produces NaN/Inf; surface the degeneracy as a clear error instead.
    Q_e_minus_Q_i != 0 ||
        throw(
            ArgumentError(
                "slayer_parameters: Q_e == Q_i (degenerate " *
                "electron/ion diamagnetic frequencies) makes " *
                "iota_e = Q_e/(Q_e - Q_i) singular. Check the " *
                "diamagnetic-frequency inputs ω_*e, ω_*i."
            )
        )
    iota_e = Q_e / Q_e_minus_Q_i

    # Plasma beta and compressibility
    lbeta = (5.0 / 3.0) * MU_0 * n_e * E_CHG * (t_e + t_i) / bt^2
    c_beta = sqrt(lbeta / (1.0 + lbeta))

    # Effective Prandtl-like transport ratios
    tau_perp = rs^2 / chi_perp
    P_perp = tau_r / tau_perp
    tau_tor = rs^2 / chi_tor
    P_tor = tau_r / tau_tor

    # Normalized beta-related width and Δ-normalization
    d_beta = c_beta * d_i
    D_norm = (d_beta / rs) * lu^(1.0 / 3.0) * sqrt(tau / (1.0 + tau))
    delta_n = lu^(1.0 / 3.0) / rs

    # Critical-Δ offset from chi_parallel matching
    dc_tmp = _solve_dc_tmp(; dc_type=dc_type, dr_val=dr_val, dgeo_val=dgeo_val,
        chi_perp=chi_perp, t_e=t_e, zeff=zeff,
        tau_ee=tau_ee, rs=rs, R0=R0, sval_r=sval_r,
        n_tor=n)

    return SLAYERParameters(;
        ising=ising, m=m, n=n,
        tau=tau, lu=lu, c_beta=c_beta, D_norm=D_norm,
        P_perp=P_perp, P_tor=P_tor,
        Q_e=Q_e, Q_i=Q_i, iota_e=iota_e,
        tauk=tauk, tau_r=tau_r, delta_n=delta_n,
        rs=rs, R0=R0, bt=bt, sval_r=sval_r,
        dr_val=dr_val, dgeo_val=dgeo_val,
        eta=eta, d_beta=d_beta,
        dc_tmp=dc_tmp, dc_type=dc_type
    )
end
