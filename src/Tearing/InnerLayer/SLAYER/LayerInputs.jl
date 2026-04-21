# LayerInputs.jl
#
# Build per-surface `SLAYERParameters` from an in-memory `PlasmaEquilibrium`,
# the `SingType` rational-surface data produced by `ForceFreeStates`, and a
# `KineticProfiles` object. Replaces the STRIDE-NetCDF path that the Fortran
# SLAYER (`layerinputs.f`) uses — julia_GPEC already holds everything we
# need in memory.
#
# Geometry extraction:
#   - Minor radius at the outboard midplane (θ = 0) via
#     `equil.rzphi_rsquared((ψ, 0.0))`.
#   - `da/dψ` via central finite difference on the same bicubic.
#   - r-based magnetic shear via `r_based_shear(rs, q, q1, da/dψ)` (defined
#     in LayerParameters.jl).

using ..Utilities: KineticProfiles
using ...Utilities.NeoclassicalResistivity: NeoResistivityModel, SpitzerModel,
    coulomb_log_e, nu_star_e

"""
    surface_minor_radius(equil, psi; theta=0.0) -> Float64

Minor radius at normalized flux `psi` and poloidal angle `theta`,
computed from `equil.rzphi_rsquared` as `√((R − R₀)² + (Z − Z₀)²)`.
`theta = 0.0` (outboard midplane) is the default; pass `θ = π` to measure
the inboard side if you want an average.
"""
function surface_minor_radius(equil, psi::Real; theta::Real=0.0)
    r_sq = equil.rzphi_rsquared((Float64(psi), Float64(theta)))
    return sqrt(r_sq)
end

"""
    surface_da_dpsi(equil, psi; theta=0.0, h=1e-5) -> Float64

Central finite-difference approximation of `d(minor radius)/dψ` at `psi`.
Falls back to one-sided differences near the flux-coordinate boundaries
(0 or 1).
"""
function surface_da_dpsi(equil, psi::Real; theta::Real=0.0, h::Real=1e-5)
    psi_f = Float64(psi)
    # Clamp to safe sampling range within (0, 1)
    eps_edge = 10 * h
    lo = psi_f - h
    hi = psi_f + h
    if lo < eps_edge
        # one-sided forward
        a0 = surface_minor_radius(equil, max(psi_f, eps_edge); theta=theta)
        a1 = surface_minor_radius(equil, max(psi_f, eps_edge) + h; theta=theta)
        return (a1 - a0) / h
    elseif hi > 1.0 - eps_edge
        # one-sided backward
        a0 = surface_minor_radius(equil, min(psi_f, 1.0 - eps_edge) - h; theta=theta)
        a1 = surface_minor_radius(equil, min(psi_f, 1.0 - eps_edge); theta=theta)
        return (a1 - a0) / h
    else
        a_plus  = surface_minor_radius(equil, psi_f + h; theta=theta)
        a_minus = surface_minor_radius(equil, psi_f - h; theta=theta)
        return (a_plus - a_minus) / (2h)
    end
end

"""
    build_slayer_inputs(equil, sings, profiles; …) -> Vector{SLAYERParameters}

Build a `SLAYERParameters` for each rational surface in `sings`, pulling
geometry (minor radius, r-based shear, q, dq/dψ, R₀) from the in-memory
`equil::PlasmaEquilibrium` and kinetic data (n_e, T_e, T_i, ω, ω\\_\\*e,
ω\\_\\*i) from `profiles::KineticProfiles`.

This is the Julia analogue of the Fortran SLAYER `layerinputs.f` path,
without the intermediate STRIDE NetCDF round-trip.

# Arguments

  - `equil`    -- `PlasmaEquilibrium`
  - `sings`    -- `Vector{SingType}` (one per resonant surface)
  - `profiles` -- `KineticProfiles` valid across all `sings` ψ values

# Keyword arguments

  - `bt`        -- toroidal field [T]. Scalar, callable of `psi`, or
    `nothing` (default). When `nothing`, the physical `B_T = F(ψ) / (2π·R₀)`
    is computed per surface from the equilibrium's F-spline. Note:
    `equil.config.b0exp` is a *normalization* (often just `1.0`), not the
    physical field, so passing it as a scalar is almost always wrong.
  - `mu_i`      -- ion mass in proton-mass units (default `2.0` for D).
  - `zeff`      -- effective charge (default `1.0`).
  - `chi_perp`  -- perpendicular heat diffusivity [m²/s]. Scalar or a
    callable of `psi` (default `1.0`).
  - `chi_tor`   -- toroidal heat diffusivity [m²/s]. Scalar or a callable
    of `psi` (default `1.0`).
  - `dr_val`    -- radial width for the critical-Δ offset. Scalar or a
    callable of `psi` (default `0.0`, which turns the offset off).
  - `dgeo_val`  -- geometric Shafranov shift factor for the toroidal
    dc_type. Scalar or a callable of `psi` (default `0.0`).
  - `dc_type`   -- `:none` (default), `:lar`, `:rfitzp`, or `:toroidal`.
  - `theta`     -- poloidal angle at which to measure minor radius (default
    `0.0`, outboard midplane).
  - `resistivity_model` -- `SpitzerModel()` (default), `SauterNeoModel()`,
    or `RedlNeoModel()`. When non-Spitzer, `f_trap` and ν*_e are taken
    from the surface's `ResistGeometry` if populated (via
    `ForceFreeStates.resist_eval_all!`), otherwise fall back to the ε-only
    Lin-Liu-Miller form and `rs/R_0` aspect ratio.
  - `lnLambda_form` -- Coulomb-log form passed through to `slayer_parameters`
    (default `:wesson` to match legacy SLAYER exactly when
    `resistivity_model=SpitzerModel()`).
"""
function build_slayer_inputs(equil, sings, profiles::KineticProfiles;
                              bt = nothing,
                              mu_i::Real = 2.0,
                              zeff::Real = 1.0,
                              chi_perp = 1.0,
                              chi_tor  = 1.0,
                              dr_val   = 0.0,
                              dgeo_val = 0.0,
                              dc_type::Symbol = :none,
                              theta::Real = 0.0,
                              resistivity_model::NeoResistivityModel = SpitzerModel(),
                              lnLambda_form::Symbol = :wesson)
    R0 = equil.ro
    _eval(x, ψ) = x isa Real ? Float64(x) : Float64(x(ψ))

    # Compute physical B_T = F(ψ) / (2π·R₀) per surface from the F spline
    # when `bt` is not explicitly supplied.
    _bt_at(ψ) = if bt === nothing
        Float64(equil.profiles.F_spline(ψ)) / (2π * R0)
    elseif bt isa Real
        Float64(bt)
    else
        Float64(bt(ψ))
    end

    out = Vector{SLAYERParameters}(undef, length(sings))
    for (k, sing) in enumerate(sings)
        psi = sing.psifac
        q   = sing.q
        q1  = sing.q1

        rs       = surface_minor_radius(equil, psi; theta=theta)
        da_dpsi  = surface_da_dpsi(equil, psi; theta=theta)
        sval_r   = r_based_shear(rs, q, q1, da_dpsi)

        prof = profiles(psi)

        # Resonant (m, n): take the first element of the mode-number vectors.
        # Parallel-FM `sing.m`/`sing.n` hold exactly one entry each; ideal
        # DCON may hold multiple — we pick the first and document the choice.
        m_res = sing.m[1]
        n_res = sing.n[1]

        # Pull geometric trapped-fraction inputs from ResistGeometry when
        # available (populated by ForceFreeStates.resist_eval_all!); else
        # fall back to nothing and let slayer_parameters compute them from
        # aspect ratio + Lin-Liu-Miller ε-only form.
        rg = sing.restype
        f_trap_kw    = rg === nothing ? nothing : rg.f_trap
        R_major_eff  = rg === nothing ? nothing : rg.R_major
        nu_e_star_kw = if rg === nothing || resistivity_model isa SpitzerModel
            nothing
        else
            lnL = coulomb_log_e(prof.n_e, prof.T_e; form=lnLambda_form)
            nu_star_e(prof.n_e, prof.T_e, rg.R_major, rg.eps_local,
                      q, zeff; lnLamb=lnL)
        end

        out[k] = slayer_parameters(;
            n_e = prof.n_e, t_e = prof.T_e, t_i = prof.T_i,
            omega = prof.omega, omega_e = prof.omega_e, omega_i = prof.omega_i,
            qval = q, sval_r = sval_r, bt = _bt_at(psi),
            rs = rs, R0 = R0, mu_i = mu_i, zeff = zeff,
            chi_perp = _eval(chi_perp, psi),
            chi_tor  = _eval(chi_tor,  psi),
            m = m_res, n = n_res,
            dr_val   = _eval(dr_val,   psi),
            dgeo_val = _eval(dgeo_val, psi),
            dc_type = dc_type, ising = k,
            resistivity_model = resistivity_model,
            f_trap = f_trap_kw,
            nu_e_star = nu_e_star_kw,
            R_major_eff = R_major_eff,
            lnLambda_form = lnLambda_form,
        )
    end
    return out
end
