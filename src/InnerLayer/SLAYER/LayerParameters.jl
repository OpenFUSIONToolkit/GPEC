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
2023; Park 2022), plus dimensional auxiliaries required for
de-normalization. The parametrization uses `P_perp`, `P_tor`, and
`D_norm` (not the older `pr`/`pe`/`ds` set).

| field      | meaning                                                           |
|:---------- |:----------------------------------------------------------------- |
| `ising`    | Singular-surface index (traceability only)                        |
| `m`, `n`   | Poloidal / toroidal mode numbers at this surface                  |
| `tau`      | T_i / T_e                                                         |
| `lu`       | Lundquist number S = τ_R / τ_H                                    |
| `c_beta`   | Compressibility √(β_local / (1 + β_local))                        |
| `D_norm`   | (d_β/r_s) · S^(1/3) · √ι_e  (Fitzpatrick normalized scale)        |
| `P_perp`   | Perpendicular Prandtl number τ_R / τ_⊥ (set by `P_perp_model`)    |
| `P_tor`    | Momentum Prandtl number τ_R / τ_φ, τ_φ = r_s²/χ_φ                 |
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
| `dr_val`   | Resistive interchange D_R = E + F + H² (critical-Δ input; auto-derived from GGJ coefficients unless overridden) |
| `dgeo_val` | Connor et al. (2015) Eq. 59 geometric factor (0 unless supplied) |
| `eta`      | Parallel resistivity entering τ_R = μ₀r_s²/η [Ω·m]                |
| `d_beta`   | Beta-weighted ion length scale c_β · d_i [m]                      |
| `dc_tmp`   | Critical-Δ offset from chi_parallel matching                      |
| `dc_type`  | Selector for `dc_tmp` formula                                     |
| `k_ref`    | Reference-length ratio K = r_s · (dψ_N/dr) at this surface (1 = no Δ' conversion) |
| `alpha_mercier` | Mercier Frobenius exponent α = √(−D_I) (Glasser-Greene-Johnson 1975 Eq. 48) governing the Δ' reference-length conversion (1/2 = slab/cylindrical value) |

`k_ref` and `alpha_mercier` feed the ψ_N → r_s reference-length conversion of
the outer Δ' matrix (see `delta_prime_to_rs_reference` in the Tearing
runner): the slab layer, its `dc_tmp` critical-Δ, and the `S^(1/3)` Δ(Q)
scale are all referenced to unit `x̂ = (r−r_s)/r_s`, while the outer BVP Δ'
is referenced to unit Δψ_N. Hand-built parameters default to `k_ref = 1`,
which makes the conversion the identity.

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

    # Reference-length conversion inputs for the outer Δ' (ψ_N → r_s-based x̂)
    k_ref::Float64 = 1.0
    alpha_mercier::Float64 = 0.5
end

# Allowed dc_type values for the critical-Δ offset. `:none` is the default
# `dc_tmp = 0` branch.
const ALLOWED_DC_TYPES = (:none, :lar, :fitzpatrick, :toroidal)

# Allowed P_perp_model values selecting how P_perp is built. `:chi_perp_e` is the default.
const ALLOWED_P_PERP_MODELS = (:chi_perp_e, :chi_perp_i, :P_phi, :D_perp, :c_beta, :tau_E)

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

# Internal: solve the Wd self-consistency loop for the chi_parallel-based critical Δ. W_d and the
# smfp/lmfp chi_parallel blend follow Fitzpatrick, Phys. Plasmas 32, 062509 (2025); the toroidal and
# large-aspect-ratio offsets are Connor, Ham, Hastie & Liu, PPCF 57, 065001 (2015) Eqs. (59) and (61),
# after Lütjens, Luciani & Garbet, Phys. Plasmas 8, 4267 (2001). Returns dc_tmp as a Float64.
function _solve_dc_tmp(; dc_type::Symbol, dr_val::Real, dgeo_val::Real,
    chi_perp_e::Real, t_e::Real, zeff::Real, tau_ee::Real,
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
        chi_par_lmfp = (2.0 * R0 * vte) / (sqrt(π) * n_tor * abs(sval_r) * Wd)
        chi_par = (chi_par_smfp * chi_par_lmfp) /
                  (chi_par_smfp + chi_par_lmfp)
        Wd_new = sqrt(8.0) * (chi_perp_e / chi_par)^0.25 *
                 (1.0 / sqrt((rs / R0) * abs(sval_r) * n_tor))
        if abs(Wd_new - Wd) / max(abs(Wd), 1e-30) < tol
            Wd = Wd_new
            converged = true
            break
        end
        Wd = Wd_new
    end
    converged || error("SLAYERParameters: Wd iteration failed to converge")

    chi_par_lmfp = (2.0 * R0 * vte) / (sqrt(π) * n_tor * abs(sval_r) * Wd)
    chi_par = (chi_par_smfp * chi_par_lmfp) / (chi_par_smfp + chi_par_lmfp)

    if dc_type === :lar
        return 0.5 * (-dr_val) * π^1.5 *
               (chi_par / chi_perp_e)^0.25 *
               sqrt((n_tor * abs(sval_r)) / (R0 * rs))
    elseif dc_type === :fitzpatrick
        return -(sqrt(2.0) * π^1.5 * dr_val) / Wd
    elseif dc_type === :toroidal
        return 0.5 * (-dr_val) * π^1.5 *
               (chi_par / chi_perp_e)^0.25 * dgeo_val
    end
    return 0.0
end

"""
    slayer_parameters(; n_e, t_e, t_i, omega_e, omega_i,
                        qval, sval_r, bt, rs, R0, mu_i, zeff,
                        chi_perp_e, chi_tor,
                        m, n,
                        chi_perp_i=nothing, tau_E=nothing, a_minor=nothing,
                        dlnn=nothing, dlnTe=nothing, dlnTi=nothing,
                        P_perp_model=:D_perp,
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
  - `chi_perp_e`, `chi_tor` -- electron perpendicular heat / toroidal momentum diffusivity [m²/s]
  - `chi_perp_i` -- ion perpendicular heat diffusivity [m²/s], required by `P_perp_model=:chi_perp_i` and `:D_perp`
  - `dlnn`, `dlnTe`, `dlnTi` -- radial log-gradients of n, T_e, T_i in any common radial variable
    (only their ratios enter; required by `P_perp_model=:D_perp`)
  - `tau_E`   -- whole-plasma energy confinement time τ̄_E [s], required by `P_perp_model=:tau_E`
  - `a_minor` -- plasma minor radius `a` [m], required by `P_perp_model=:tau_E` (`build_slayer_inputs` takes it from the flux label)

  An input a model does not use may be left as `nothing`; one it uses must be given.
  - `P_perp_model` -- how `P_perp` is built (default `:D_perp`, as in the `[SLAYER]` deck):
      + `:D_perp` -- τ_R/τ_⊥ with τ_⊥ = r_s²/D_⊥, D_⊥ the perpendicular particle diffusivity of
        Fitzpatrick, Phys. Plasmas 29, 032507 (2022), Eq. 16 (with Eqs. 4-6 and η_⊥ = η_∥). Its
        τ = (T_e/T_i)(1+η_e)/(1+η_i) is the inverse orientation of the `tau` = T_i/T_e field
        of `SLAYERParameters`. With flat temperatures it falls to the `:c_beta` value
      + `:chi_perp_e` -- τ_R·χ⊥,e/r_s²
      + `:chi_perp_i` -- τ_R·χ⊥,i/r_s²
      + `:P_phi`  -- P_perp = P_tor (Fitzpatrick, Phys. Plasmas 30, 092512 (2023), Eq. 143)
      + `:c_beta` -- P_perp = c_β² (J.-K. Park, Phys. Plasmas 29, 072506 (2022), Eq. 9 with K = 0),
        the χ⊥ → 0 limit of `:D_perp`
      + `:tau_E`  -- P_perp = P_tor = τ_R/τ_E(r_s) (Fitzpatrick 2023, Eqs. 37-38, 81-83 and 131), from
        a uniform χ_E = κ_e a²/τ̄_E with κ_e = 3/16, the value for T_e ∝ 1 − r̂² (Eq. 38). Other
        profile shapes change κ_e by a factor of order 2-3. Both Prandtl numbers come from χ_E, so
        `chi_tor` is ignored in this mode
    `chi_perp_e` still sets the critical-Δ `dc_tmp` in every mode.
  - `m`, `n`  -- poloidal / toroidal mode numbers at the surface
  - `dr_val`, `dgeo_val` -- inputs for the critical-Δ formula
  - `dc_type` -- one of `:none`, `:lar`, `:fitzpatrick`, `:toroidal`
  - `ising`   -- singular-surface index for traceability
  - `k_ref`   -- reference-length ratio K = r_s·(dψ_N/dr) at the surface,
    used by the Tearing runner to convert the ψ_N-referenced outer Δ' to
    the r_s-referenced convention this layer works in (default `1.0`,
    i.e. no conversion; `build_slayer_inputs` fills it from the equilibrium)
  - `alpha_mercier` -- Mercier Frobenius exponent α = √(−D_I) for the same conversion
    (default `0.5`, the slab/cylindrical value at D_I = −1/4)

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
    omega_e::Real, omega_i::Real,
    qval::Real, sval_r::Real, bt::Real,
    rs::Real, R0::Real, mu_i::Real, zeff::Real,
    chi_perp_e::Real, chi_tor::Real,
    m::Integer, n::Integer,
    chi_perp_i::Union{Real,Nothing}=nothing, tau_E::Union{Real,Nothing}=nothing, a_minor::Union{Real,Nothing}=nothing,
    dlnn::Union{Real,Nothing}=nothing, dlnTe::Union{Real,Nothing}=nothing, dlnTi::Union{Real,Nothing}=nothing,
    P_perp_model::Symbol=:D_perp,
    dr_val::Real=0.0, dgeo_val::Real=0.0,
    dc_type::Symbol=:none, ising::Integer=0,
    resistivity_model::NeoResistivityModel=SauterNeoModel(),
    f_trap::Union{Real,Nothing}=nothing,
    nu_e_star::Union{Real,Nothing}=nothing,
    R_major_eff::Union{Real,Nothing}=nothing,
    lnLambda_form::Symbol=:nrl,
    k_ref::Real=1.0,
    alpha_mercier::Real=0.5)

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
    # tau_h = R0 sqrt(mu0 rho) / (n sval bt)). Magnitude only: the layer timescales and
    # widths depend on |dq/dr|, not its sign, and a reverse-shear surface would otherwise
    # give tau_h < 0, hence lu < 0 and a DomainError in lu^(1/3) below.
    tau_h = R0 * sqrt(MU_0 * rho) / (n * abs(sval_r) * bt)
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
    iota_e > 0 ||
        throw(
            ArgumentError(
                "slayer_parameters: iota_e = Q_e/(Q_e - Q_i) = $iota_e is " *
                "not positive, so the ion-sound-radius normalization " *
                "D_norm = (d_beta/r_s)·S^(1/3)·sqrt(iota_e) is undefined. " *
                "Check the diamagnetic-frequency inputs ω_*e, ω_*i — they " *
                "are expected to carry opposite signs."
            )
        )

    # Plasma beta and compressibility
    lbeta = (5.0 / 3.0) * MU_0 * n_e * E_CHG * (t_e + t_i) / bt^2
    c_beta = sqrt(lbeta / (1.0 + lbeta))

    # Effective Prandtl-like transport ratios. A diffusivity the model uses must be given, finite and positive.
    _need(name, v) = (v !== nothing && isfinite(v) && v > 0) ||
                     throw(ArgumentError("slayer_parameters: P_perp_model=:$P_perp_model needs a positive, finite $name; got $v"))
    _need("chi_perp_e", chi_perp_e)
    P_perp_model === :tau_E || _need("chi_tor", chi_tor)
    P_perp_model in (:chi_perp_i, :D_perp) && _need("chi_perp_i", chi_perp_i)
    tau_tor = rs^2 / chi_tor
    P_tor = tau_r / tau_tor
    if P_perp_model === :tau_E
        _need("tau_E", tau_E)
        _need("a_minor", a_minor)
        # Fitzpatrick 2023, Eqs. 37-38, 81-83 and 131.
        chi_E = (3 / 16) * a_minor^2 / tau_E
        P_tor = tau_r / (rs^2 / chi_E)
    end
    P_perp = if P_perp_model === :chi_perp_e
        tau_r / (rs^2 / chi_perp_e)
    elseif P_perp_model === :chi_perp_i
        tau_r / (rs^2 / chi_perp_i)
    elseif P_perp_model === :P_phi || P_perp_model === :tau_E
        P_tor
    elseif P_perp_model === :D_perp
        # Fitzpatrick, Phys. Plasmas 29, 032507 (2022), Eqs. 4-6 and 16 with η_⊥ = η_∥, written over the pressure gradient p′/n.
        all(x -> x !== nothing && isfinite(x), (dlnn, dlnTe, dlnTi)) ||
            throw(ArgumentError("slayer_parameters: P_perp_model=:D_perp needs finite dlnn, dlnTe and dlnTi; got $dlnn, $dlnTe, $dlnTi"))
        dp = t_e * (dlnn + dlnTe) + t_i * (dlnn + dlnTi)
        dp != 0 || throw(ArgumentError("slayer_parameters: P_perp_model=:D_perp is undefined where the pressure gradient vanishes " *
                                       "(t_e(dlnn+dlnTe) + t_i(dlnn+dlnTi) = 0); choose another P_perp_model for this surface"))
        D_perp = c_beta^2 * eta / MU_0 +
                 (2 / 3) * (1 - c_beta^2) * (t_e * dlnTe * chi_perp_e + t_i * dlnTi * chi_perp_i) / dp
        tau_r / (rs^2 / D_perp)
    elseif P_perp_model === :c_beta
        c_beta^2
    else
        throw(ArgumentError("slayer_parameters: unknown P_perp_model=$P_perp_model. " *
                            "Allowed: $(ALLOWED_P_PERP_MODELS)"))
    end
    isfinite(P_perp) && P_perp > 0 ||
        throw(ArgumentError("slayer_parameters: P_perp_model=:$P_perp_model gave P_perp=$P_perp; check its inputs " *
                            "(chi_perp_e=$chi_perp_e, chi_perp_i=$chi_perp_i, dlnn=$dlnn, dlnTe=$dlnTe, dlnTi=$dlnTi)"))

    # Normalized beta-related width and Δ-normalization
    d_beta = c_beta * d_i
    D_norm = (d_beta / rs) * lu^(1.0 / 3.0) * sqrt(iota_e)
    delta_n = lu^(1.0 / 3.0) / rs

    # Critical-Δ offset from chi_parallel matching
    dc_tmp = _solve_dc_tmp(; dc_type=dc_type, dr_val=dr_val, dgeo_val=dgeo_val,
        chi_perp_e=chi_perp_e, t_e=t_e, zeff=zeff,
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
        dc_tmp=dc_tmp, dc_type=dc_type,
        k_ref=k_ref, alpha_mercier=alpha_mercier
    )
end
