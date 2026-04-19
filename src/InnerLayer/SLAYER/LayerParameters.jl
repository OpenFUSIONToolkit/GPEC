# LayerParameters.jl
#
# `SLAYERParameters` carries the dimensionless layer-physics parameters
# that the Fitzpatrick `riccati_f` ODE consumes for one rational surface,
# plus the dimensional conversion factors needed to translate normalized
# frequencies and Δ values back to physical units.
#
# Constructor `SLAYERParameters(; ...)` ports `params.f::SUBROUTINE
# params` (modified): no pr, no pe, no ds (those entered only the
# legacy `riccati()` / `riccati_del_s()` paths which are not implemented
# here). Q is not stored — it is passed directly to `solve_inner`.

"""
    SLAYERParameters

Dimensionless layer-physics parameters at one rational surface for the
Fitzpatrick (`riccati_f`) SLAYER inner-layer model, plus dimensional
auxiliaries required for de-normalization.

Mirrors the Fortran SLAYER per-surface state (`sglobal_mod` +
`slayer_inputs_type`) restricted to the quantities consumed by
`riccati_f`. The legacy magnetic Prandtl `pr`, electron Prandtl `pe`,
and `ρ_s`-based `ds` parameters are intentionally absent — the
`riccati_f` formulation uses `P_perp`, `P_tor`, and `D_norm` instead.

| field      | meaning                                                           |
|------------|-------------------------------------------------------------------|
| `ising`    | Singular-surface index (traceability only)                        |
| `m`, `n`   | Poloidal / toroidal mode numbers at this surface                  |
| `tau`      | T_i / T_e                                                         |
| `lu`       | Lundquist number S = τ_R / τ_H                                    |
| `c_beta`   | Compressibility √(β_local / (1 + β_local))                        |
| `D_norm`   | (d_β/r_s) · S^(1/3) · √(τ/(1+τ))  (Fitzpatrick normalized scale)  |
| `P_perp`   | Perpendicular Prandtl number τ_R / τ_⊥                            |
| `P_tor`    | Toroidal-direction Prandtl number τ_R / τ_‖tor                    |
| `Q_e`      | Normalized electron diamagnetic: −tauk · ω_*e                     |
| `Q_i`      | Normalized ion diamagnetic:      +tauk · ω_*i                     |
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
| `eta`      | Spitzer resistivity [Ω·m]                                         |
| `d_beta`   | Beta-weighted ion length scale c_β · d_i [m]                      |
| `dc_tmp`   | Critical-Δ offset from chi_parallel matching                      |
| `dc_type`  | Selector for `dc_tmp` formula                                     |

The complex normalized growth rate `Q = ω + iγ` is **not** stored here;
it is passed as a separate argument to `solve_inner`.
"""
Base.@kwdef struct SLAYERParameters
    # Surface identity
    ising::Int = 0
    m::Int     = 0
    n::Int     = 0

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
    dr_val::Float64    = 0.0
    dgeo_val::Float64  = 0.0
    eta::Float64
    d_beta::Float64

    # Critical-Δ offset
    dc_tmp::Float64    = 0.0
    dc_type::Symbol    = :none
end

# Allowed dc_type values (ports the Fortran `dc_type` SELECT CASE in
# params.f:230-242). `:none` reproduces the default `dc_tmp = 0` branch.
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

This is the Julia analogue of the conversion `s_Fitz = s_psiN · r_s /
(psi_N · da_dpsiN)` performed at `layerinputs.f:488`.
"""
function r_based_shear(rs::Real, q::Real, dq_dpsi::Real, da_dpsi::Real)
    da_dpsi != 0 || throw(ArgumentError("r_based_shear: da/dψ must be non-zero"))
    q       != 0 || throw(ArgumentError("r_based_shear: q must be non-zero"))
    return rs * dq_dpsi / (q * da_dpsi)
end

# Internal: solve the Wd self-consistency loop for the chi_parallel-based
# critical Δ. Ports params.f:204-246. Returns dc_tmp as a Float64.
function _solve_dc_tmp(; dc_type::Symbol, dr_val::Real, dgeo_val::Real,
                        chi_perp::Real, t_e::Real, zeff::Real, tau_ee::Real,
                        rs::Real, R0::Real, sval_r::Real, n_tor::Integer,
                        max_iter::Integer=100, tol::Real=1e-10)
    dc_type in ALLOWED_DC_TYPES ||
        throw(ArgumentError("SLAYERParameters: unknown dc_type=$dc_type. " *
                            "Allowed: $(ALLOWED_DC_TYPES)"))
    (dc_type === :none || dr_val == 0.0) && return 0.0

    vte           = sqrt(2.0 * t_e * E_CHG / M_E)
    chi_par_smfp  = (1.581 * tau_ee * vte^2) / (1.0 + 0.2535 * zeff)

    Wd = 0.1
    converged = false
    for _ in 1:max_iter
        chi_par_lmfp = (2.0 * R0 * vte) / (sqrt(π) * n_tor * sval_r * Wd)
        chi_par      = (chi_par_smfp * chi_par_lmfp) /
                       (chi_par_smfp + chi_par_lmfp)
        Wd_new       = sqrt(8.0) * (chi_perp / chi_par)^0.25 *
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
    chi_par      = (chi_par_smfp * chi_par_lmfp) / (chi_par_smfp + chi_par_lmfp)

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
                        dc_type=:none, ising=0)
        -> SLAYERParameters

Build a `SLAYERParameters` for one rational surface from dimensional
equilibrium and kinetic-profile inputs. Mirrors `params.f::SUBROUTINE
params` restricted to the Fitzpatrick (`riccati_f`) path: drops the
magnetic Prandtl `pr`, electron Prandtl `pe`, and ρ_s-based `ds` (those
parameters entered only the legacy `riccati()` and `riccati_del_s()`
formulations).

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

# Sign convention for diamagnetic frequencies

Following the Fortran `layerinputs.f:540-541` convention used by the
SLAYER dispersion solver:

```
Q_e = -tauk · ω_*e
Q_i = +tauk · ω_*i
```

i.e. callers pass `omega_e` and `omega_i` as raw diamagnetic frequencies
in the convention used by the kinetic-profile splines. The sign flip on
`Q_e` is intrinsic to the dispersion-relation derivation.
"""
function slayer_parameters(;
        n_e::Real, t_e::Real, t_i::Real,
        omega::Real, omega_e::Real, omega_i::Real,
        qval::Real, sval_r::Real, bt::Real,
        rs::Real, R0::Real, mu_i::Real, zeff::Real,
        chi_perp::Real, chi_tor::Real,
        m::Integer, n::Integer,
        dr_val::Real=0.0, dgeo_val::Real=0.0,
        dc_type::Symbol=:none, ising::Integer=0)

    # Coulomb logarithm (params.f:91)
    lnLamb = 24.0 + 3.0 * log(10.0) - 0.5 * log(n_e) + log(t_e)

    # Basic plasma quantities (params.f:93-97)
    tau = t_i / t_e
    eta = 1.65e-9 * lnLamb / (t_e / 1e3)^1.5
    rho = mu_i * M_P * n_e

    # Electron-electron collision time and Spitzer-Härm conductivity
    # (params.f:103-111). T_e enters in eV; the chag^(-2.5) factor in
    # the denominator absorbs the eV→J conversion (see params.f
    # comments for derivation).
    tau_ee_num   = 6.0 * sqrt(2.0) * π^1.5 *
                   EPS_0^2 * sqrt(M_E) * t_e^1.5
    tau_ee_denom = lnLamb * E_CHG^2.5 * n_e
    tau_ee       = tau_ee_num / tau_ee_denom

    sigma_par_1 = (sqrt(2.0) + 13.0 * (zeff / 4.0)) /
                  (zeff * (sqrt(2.0) + zeff))
    sigma_par_2 = (n_e * E_CHG^2 * tau_ee) / M_E
    sigma_par   = sigma_par_1 * sigma_par_2

    # Characteristic field, Alfven speed, length scales, fundamental
    # timescales (params.f:119-126).
    rho_s = 1.02e-4 * sqrt(mu_i * t_e) / bt                 # ion Larmor [m]
    d_i   = sqrt((mu_i * M_P) / (n_e * E_CHG^2 * MU_0))     # ion skin depth [m]

    # Alfven time uses minor-radius shear directly (sval enters the
    # b_l = (n/m) r_s sval bt / R0 expression and cancels through to
    # tau_h = R0 sqrt(mu0 rho) / (n sval bt)).
    tau_h = R0 * sqrt(MU_0 * rho) / (n * sval_r * bt)
    tau_r = MU_0 * rs^2 * sigma_par                          # Fitzpatrick

    # Lundquist number and Q-conversion factor (params.f:136, 143-144)
    lu    = tau_r / tau_h
    tauk  = lu^(1.0 / 3.0) * tau_h         # = Qconv

    # Normalized diamagnetic frequencies (layerinputs.f:540-541
    # convention; see docstring sign convention discussion).
    Q_e = -tauk * omega_e
    Q_i = +tauk * omega_i
    Q_e_minus_Q_i = Q_e - Q_i
    iota_e = Q_e_minus_Q_i == 0 ? 0.0 : Q_e / Q_e_minus_Q_i

    # Plasma beta and compressibility (params.f:164-165)
    lbeta  = (5.0 / 3.0) * MU_0 * n_e * E_CHG * (t_e + t_i) / bt^2
    c_beta = sqrt(lbeta / (1.0 + lbeta))

    # Effective Prandtl-like transport ratios (params.f:177-182)
    tau_perp = rs^2 / chi_perp
    P_perp   = tau_r / tau_perp
    tau_tor  = rs^2 / chi_tor
    P_tor    = tau_r / tau_tor

    # Normalized beta-related width and Δ-normalization (params.f:187-192)
    d_beta  = c_beta * d_i
    D_norm  = (d_beta / rs) * lu^(1.0 / 3.0) * sqrt(tau / (1.0 + tau))
    delta_n = lu^(1.0 / 3.0) / rs

    # Critical-Δ offset from chi_parallel matching (params.f:204-246)
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
        dc_tmp=dc_tmp, dc_type=dc_type,
    )
end
