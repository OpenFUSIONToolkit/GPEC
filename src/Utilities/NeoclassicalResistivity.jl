# NeoclassicalResistivity.jl
#
# Shared neoclassical-resistivity utilities used by both the GGJ and
# SLAYER inner-layer models. All formulas follow Sauter, Angioni & Lin-Liu
# Phys. Plasmas 6, 2834 (1999) and its errata, with an optional Redl et al.
# Phys. Plasmas 28, 022502 (2021) variant that improves the fit at high
# collisionality.
#
# Two external references were cross-checked during implementation:
#   - OpenFUSIONToolkit `TokaMaker/bootstrap.py`  (Redl 2021 path)
#   - OMFIT `omfit_classes/utils_fusion.py::nclass_conductivity-style
#     block` around lines 1255-1319 (Sauter 1999 and `neo_2021` paths)
#
# Formula provenance:
#   - eq 18a (Spitzer):       Sauter et al. 1999, Eq. (18a)
#   - eq 18b (nu*_e):         Sauter et al. 1999, Eq. (18b)
#   - eq 13 (F_33 Sauter):    Sauter et al. 1999, Eqs. (13a)-(13b)
#   - eq 17 (F_33 Redl):      Redl et al. 2021, Eqs. (17)-(18)
#   - f_t (Lin-Liu & Miller): Phys. Plasmas 2, 1666 (1995), Eq. (6)
#   - NRL Coulomb log:        NRL Plasma Formulary 2009

"""
    NeoclassicalResistivity

Spitzer + Sauter / Redl neoclassical resistivity closures, shared between
the GGJ and SLAYER inner-layer models so both see identical plasma-input
physics when the same `NeoResistivityModel` is selected.

# Exports

| symbol                 | role                                                   |
|:---------------------- |:------------------------------------------------------ |
| `NeoResistivityModel`  | abstract tag                                           |
| `SpitzerModel`         | plain Spitzer (no trapped-particle correction)         |
| `SpitzerHarmModel`     | Fitzpatrick/TJ Spitzer-Härm σ_∥ (legacy SLAYER τ_R)    |
| `SauterNeoModel`       | Sauter 1999 F_33 neoclassical correction               |
| `RedlNeoModel`         | Redl 2021 F_33 neoclassical correction                 |
| `coulomb_log_e`        | ln Λ_e (NRL or Sauter form)                            |
| `eta_spitzer`          | Sauter 18a Spitzer resistivity [Ω·m]                   |
| `tau_ee_spitzer_harm`  | electron-electron collision time τ_ee [s]              |
| `eta_spitzer_harm`     | Fitzpatrick/TJ Spitzer-Härm resistivity 1/σ_∥ [Ω·m]    |
| `trapped_fraction`     | Lin-Liu & Miller 1995 f_t from ⟨B⟩, ⟨B²⟩, B_min, B_max |
| `trapped_fraction_eps` | simple ε-only f_t fallback                             |
| `nu_star_e`            | Sauter 18b electron collisionality                     |
| `eta_neoclassical`     | dispatched: Spitzer(-Härm) or F_33 · Spitzer           |
"""
module NeoclassicalResistivity

using ..PhysicalConstants: EPS_0, M_E, E_CHG

export NeoResistivityModel, SpitzerModel, SpitzerHarmModel, SauterNeoModel, RedlNeoModel
export coulomb_log_e, eta_spitzer, tau_ee_spitzer_harm, eta_spitzer_harm
export trapped_fraction, trapped_fraction_eps
export nu_star_e, eta_neoclassical

"""
Abstract tag for a neoclassical-resistivity closure.
"""
abstract type NeoResistivityModel end

"""
Plain Spitzer resistivity — no trapped-particle correction.
"""
struct SpitzerModel <: NeoResistivityModel end

"""
Spitzer-Härm parallel resistivity 1/σ_∥ as used by Fitzpatrick's TJ
(LayerParameters.tex Eqs. 7-8) and legacy SLAYER for τ_R. No
trapped-particle correction. Pair with `lnLambda_form=:wesson` to
reproduce legacy SLAYER behaviour bit-identically.
"""
struct SpitzerHarmModel <: NeoResistivityModel end

"""
Sauter, Angioni & Lin-Liu 1999 F_33 neoclassical correction (Eqs. 13a,b).
"""
struct SauterNeoModel <: NeoResistivityModel end

"""
Redl et al. 2021 F_33 neoclassical correction (Eqs. 17-18). Improved
high-collisionality fit vs SauterNeoModel.
"""
struct RedlNeoModel <: NeoResistivityModel end

# --------------------------------------------------------------------------
# Coulomb logarithm
# --------------------------------------------------------------------------

"""
    coulomb_log_e(n_e, T_e; form=:nrl) -> Float64

Electron Coulomb logarithm. `n_e` in m⁻³, `T_e` in eV.

`form=:nrl` (default) uses the NRL Plasma Formulary 2009 expression, which
OpenFUSIONToolkit's `bootstrap.py` also selects as the "more accurate"
option. `form=:sauter` uses the simpler Sauter 1999 Eq. 18d form.
"""
function coulomb_log_e(n_e::Real, T_e::Real; form::Symbol=:nrl)
    n_e > 0 || throw(ArgumentError("coulomb_log_e: n_e must be > 0 (got $n_e)"))
    T_e > 0 || throw(ArgumentError("coulomb_log_e: T_e must be > 0 (got $T_e)"))
    if form === :nrl
        # NRL 2009, n_e in cm⁻³; matches utils_fusion.py:1262-1264
        return 23.5 - log(sqrt(n_e / 1e6) * T_e^(-1.25)) -
               sqrt(1e-5 + (log(T_e) - 2)^2 / 16.0)
    elseif form === :sauter
        # Sauter 1999 Eq. 18d; matches utils_fusion.py:1255
        return 31.3 - log(sqrt(n_e) / T_e)
    elseif form === :wesson
        # Legacy Wesson form used by previous Julia code & SLAYER
        return 24.0 + 3.0 * log(10.0) - 0.5 * log(n_e) + log(T_e)
    else
        throw(ArgumentError("coulomb_log_e: unknown form=$form " *
                            "(expected :nrl, :sauter, or :wesson)"))
    end
end

# --------------------------------------------------------------------------
# Spitzer resistivity (Sauter 1999 Eq. 18a)
# --------------------------------------------------------------------------

# Sauter 1999 Eq. 18a line 2 — Spitzer conductivity Zeff correction
_N_Z(Z::Real) = 0.58 + 0.74 / (0.76 + Z)

"""
    eta_spitzer(n_e, T_e, Z_eff; lnLamb=nothing) -> Float64

Spitzer resistivity in Ω·m, using the Sauter 1999 Eq. 18a form

```
σ_Sp = 1.9012e4 · T_e^1.5 / (Z_eff · N(Z_eff) · lnΛ_e)
N(Z) = 0.58 + 0.74 / (0.76 + Z)
η_Sp = 1 / σ_Sp
```

`n_e` [m⁻³], `T_e` [eV]. `lnLamb` defaults to `coulomb_log_e(n_e, T_e)` (NRL).
"""
function eta_spitzer(n_e::Real, T_e::Real, Z_eff::Real;
    lnLamb::Union{Real,Nothing}=nothing)
    T_e > 0 || throw(ArgumentError("eta_spitzer: T_e must be > 0 (got $T_e)"))
    Z_eff > 0 || throw(ArgumentError("eta_spitzer: Z_eff must be > 0 (got $Z_eff)"))
    lnL = lnLamb === nothing ? coulomb_log_e(n_e, T_e) : Float64(lnLamb)
    sigma_sp = 1.9012e4 * T_e^1.5 / (Z_eff * _N_Z(Z_eff) * lnL)
    return 1.0 / sigma_sp
end

# --------------------------------------------------------------------------
# Spitzer-Härm parallel resistivity (Fitzpatrick TJ LayerParameters.tex Eqs. 7-8)
# --------------------------------------------------------------------------

"""
    tau_ee_spitzer_harm(n_e, T_e; lnLamb=nothing) -> Float64

Electron-electron collision time per Fitzpatrick TJ LayerParameters.tex Eq. 7:

```
τ_ee = 6√2 π^1.5 ε₀² √m_e T_e^1.5 / (lnΛ e^2.5 n_e)
```

`n_e` [m⁻³], `T_e` [eV] (the `e^2.5` denominator absorbs the eV→J
conversion). `lnLamb` defaults to `coulomb_log_e(n_e, T_e)` (NRL).
"""
function tau_ee_spitzer_harm(n_e::Real, T_e::Real;
    lnLamb::Union{Real,Nothing}=nothing)
    n_e > 0 || throw(ArgumentError("tau_ee_spitzer_harm: n_e must be > 0"))
    T_e > 0 || throw(ArgumentError("tau_ee_spitzer_harm: T_e must be > 0"))
    lnL = lnLamb === nothing ? coulomb_log_e(n_e, T_e) : Float64(lnLamb)
    return 6.0 * sqrt(2.0) * π^1.5 * EPS_0^2 * sqrt(M_E) * T_e^1.5 /
           (lnL * E_CHG^2.5 * n_e)
end

"""
    eta_spitzer_harm(n_e, T_e, Z_eff; lnLamb=nothing) -> Float64

Spitzer-Härm parallel resistivity η_∥ = 1/σ_∥ in Ω·m, per Fitzpatrick TJ
LayerParameters.tex Eq. 8:

```
σ_∥ = (√2 + 13 Z_eff/4) / (Z_eff (√2 + Z_eff)) · n_e e² τ_ee / m_e
```

This is the resistivity entering the legacy SLAYER / TJ resistive
diffusion time τ_R = μ₀ r_s² σ_∥ (LayerParameters.tex Eq. 17). Agrees
with `eta_spitzer` to the fit accuracy of the two formulas (~1% at Z=1).
"""
function eta_spitzer_harm(n_e::Real, T_e::Real, Z_eff::Real;
    lnLamb::Union{Real,Nothing}=nothing)
    Z_eff > 0 || throw(ArgumentError("eta_spitzer_harm: Z_eff must be > 0"))
    tau_ee = tau_ee_spitzer_harm(n_e, T_e; lnLamb=lnLamb)
    sigma_par = (sqrt(2.0) + 13.0 * (Z_eff / 4.0)) /
                (Z_eff * (sqrt(2.0) + Z_eff)) *
                (n_e * E_CHG^2 * tau_ee) / M_E
    return 1.0 / sigma_par
end

# --------------------------------------------------------------------------
# Trapped fraction
# --------------------------------------------------------------------------

"""
    trapped_fraction(avg_B, avg_Bsq, B_min, B_max) -> Float64

Lin-Liu & Miller 1995, Phys. Plasmas **2**, 1666, Eq. (6):

```
f_t = 1 − ⟨B⟩² / ⟨B²⟩ · (1 − √(1 − h) · (1 + h/2)),   h = B_min / B_max
```

Equivalent to the OMFIT `f_t` / `f_c` pair at full geometric accuracy (uses
both the average-B ratio and the min/max extremes). Arguments are
flux-surface averages computed from the θ-loop in the equilibrium.
"""
function trapped_fraction(avg_B::Real, avg_Bsq::Real,
    B_min::Real, B_max::Real)
    B_max > 0 || throw(ArgumentError("trapped_fraction: B_max must be > 0"))
    avg_Bsq > 0 || throw(ArgumentError("trapped_fraction: avg_Bsq must be > 0"))
    h = clamp(B_min / B_max, 0.0, 1.0)
    factor = 1.0 - sqrt(1.0 - h) * (1.0 + 0.5 * h)
    ft = 1.0 - (avg_B^2 / avg_Bsq) * factor
    return clamp(ft, 0.0, 1.0)
end

"""
    trapped_fraction_eps(eps) -> Float64

Simple ε-only trapped-fraction approximation (OMFIT `f_t`):

```
f_c ≈ (1 − ε)² / (√(1 − ε²) · (1 + 1.46·√ε + 0.2·ε))
f_t = 1 − f_c
```

Used as a fallback when the full (⟨B⟩, ⟨B²⟩, B_min, B_max) moments are
unavailable — e.g. when feeding SLAYER directly from minor-radius geometry
without having evaluated `ResistGeometry` first.
"""
function trapped_fraction_eps(eps::Real)
    e = clamp(eps, 0.0, 1.0 - 1e-12)
    fc = (1.0 - e)^2 / (sqrt(1.0 - e^2) * (1.0 + 1.46 * sqrt(e) + 0.2 * e))
    return clamp(1.0 - fc, 0.0, 1.0)
end

# --------------------------------------------------------------------------
# Electron collisionality (Sauter 1999 Eq. 18b)
# --------------------------------------------------------------------------

"""
    nu_star_e(n_e, T_e, R_major, eps, q, Z_eff; lnLamb=nothing) -> Float64

Electron collisionality ν*_e per Sauter 1999 Eq. 18b:

```
ν*_e = 6.921e-18 · |q| · R · n_e · Z_eff · lnΛ_e / (T_e² · ε^1.5)
```

`n_e` [m⁻³], `T_e` [eV], `R_major` [m]. Matches OFT `bootstrap.py:640` and
OMFIT `utils_fusion.py:1278`.
"""
function nu_star_e(n_e::Real, T_e::Real, R_major::Real,
    eps::Real, q::Real, Z_eff::Real;
    lnLamb::Union{Real,Nothing}=nothing)
    eps > 0 || throw(ArgumentError("nu_star_e: eps must be > 0"))
    T_e > 0 || throw(ArgumentError("nu_star_e: T_e must be > 0"))
    lnL = lnLamb === nothing ? coulomb_log_e(n_e, T_e) : Float64(lnLamb)
    return 6.921e-18 * abs(q) * R_major * n_e * Z_eff * lnL /
           (T_e^2 * eps^1.5)
end

# --------------------------------------------------------------------------
# Neoclassical resistivity (F_33 · η_Sp)
# --------------------------------------------------------------------------

# Sauter 1999 Eqs. 13a-13b
function _F33_sauter(f_t::Real, nu_star::Real, Z_eff::Real)
    x = f_t / (1.0 + (0.55 - 0.1 * f_t) * sqrt(nu_star) +
               0.45 * (1.0 - f_t) * nu_star * Z_eff^(-1.5))
    return 1.0 - (1.0 + 0.36 / Z_eff) * x +
           (0.59 / Z_eff) * x^2 - (0.23 / Z_eff) * x^3
end

# Redl 2021 Eqs. 17-18
function _F33_redl(f_t::Real, nu_star::Real, Z_eff::Real)
    dZm1 = sqrt(max(Z_eff - 1.0, 0.0))
    x = f_t / (1.0 + 0.25 * (1.0 - 0.7 * f_t) * sqrt(nu_star) *
                     (1.0 + 0.45 * dZm1) +
               0.61 * (1.0 - 0.41 * f_t) * nu_star / sqrt(Z_eff))
    return 1.0 - (1.0 + 0.21 / Z_eff) * x +
           (0.54 / Z_eff) * x^2 - (0.33 / Z_eff) * x^3
end

"""
    eta_neoclassical(model, n_e, T_e, Z_eff, f_t, nu_e_star;
                     lnLamb=nothing) -> Float64

Neoclassical resistivity η [Ω·m] under the chosen closure.

  - `SpitzerModel()`   -- returns `eta_spitzer(n_e, T_e, Z_eff; lnLamb)`
    unchanged; `f_t` and `nu_e_star` are ignored.
  - `SpitzerHarmModel()` -- returns `eta_spitzer_harm(n_e, T_e, Z_eff; lnLamb)`
    (Fitzpatrick/TJ legacy); `f_t` and `nu_e_star` are ignored.
  - `SauterNeoModel()` -- Sauter 1999 Eq. 13: η = η_Sp / F_33(Sauter).
  - `RedlNeoModel()`   -- Redl 2021 Eq. 17: η = η_Sp / F_33(Redl).

Note that σ_neo = σ_Sp · F_33, so η_neo = η_Sp / F_33. For a banana-regime
plasma with f_t ≈ 0.5 and ν*_e ≪ 1, F_33 ≈ 0.4–0.5, so η_neo is a factor
of ~2 larger than η_Sp — this is the standard H-mode tearing correction.
"""
function eta_neoclassical(::SpitzerModel, n_e::Real, T_e::Real, Z_eff::Real,
    f_t::Real, nu_e_star::Real;
    lnLamb::Union{Real,Nothing}=nothing)
    return eta_spitzer(n_e, T_e, Z_eff; lnLamb=lnLamb)
end

function eta_neoclassical(::SpitzerHarmModel, n_e::Real, T_e::Real, Z_eff::Real,
    f_t::Real, nu_e_star::Real;
    lnLamb::Union{Real,Nothing}=nothing)
    return eta_spitzer_harm(n_e, T_e, Z_eff; lnLamb=lnLamb)
end

function eta_neoclassical(::SauterNeoModel, n_e::Real, T_e::Real, Z_eff::Real,
    f_t::Real, nu_e_star::Real;
    lnLamb::Union{Real,Nothing}=nothing)
    eta_sp = eta_spitzer(n_e, T_e, Z_eff; lnLamb=lnLamb)
    F33 = _F33_sauter(clamp(f_t, 0.0, 1.0), max(nu_e_star, 0.0), Z_eff)
    F33 > 0 || throw(DomainError(F33, "eta_neoclassical: F_33 non-positive — " *
                                      "inputs outside Sauter fit range"))
    return eta_sp / F33
end

function eta_neoclassical(::RedlNeoModel, n_e::Real, T_e::Real, Z_eff::Real,
    f_t::Real, nu_e_star::Real;
    lnLamb::Union{Real,Nothing}=nothing)
    eta_sp = eta_spitzer(n_e, T_e, Z_eff; lnLamb=lnLamb)
    F33 = _F33_redl(clamp(f_t, 0.0, 1.0), max(nu_e_star, 0.0), Z_eff)
    F33 > 0 || throw(DomainError(F33, "eta_neoclassical: F_33 non-positive — " *
                                      "inputs outside Redl fit range"))
    return eta_sp / F33
end

end # module NeoclassicalResistivity
