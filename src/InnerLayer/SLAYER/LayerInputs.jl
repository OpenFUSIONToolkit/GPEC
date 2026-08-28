# LayerInputs.jl
#
# Build per-surface `SLAYERParameters` from an in-memory `PlasmaEquilibrium`,
# the `SingType` rational-surface data produced by `ForceFreeStates`, and a
# `KineticProfiles` object. Everything needed is already held in memory, so
# the layer inputs are assembled directly without an intermediate file
# round-trip.
#
# Geometry extraction:
#   - Minor radius at the outboard midplane (θ = 0) via
#     `equil.rzphi_rsquared((ψ, 0.0))`.
#   - `da/dψ` from the interpolant's own analytic ψ-derivative.
#   - r-based magnetic shear via `r_based_shear(rs, q, q1, da/dψ)` (defined
#     in LayerParameters.jl).

using ..Utilities: KineticProfiles
using ...Utilities.NeoclassicalResistivity: NeoResistivityModel, SpitzerModel,
    coulomb_log_e, nu_star_e
using FastInterpolations: DerivOp, integrate, cubic_interp, cumulative_integrate, ExtendExtrap

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
    surface_da_dpsi(equil, psi; theta=0.0) -> Float64

Analytic ψ-derivative of the minor radius at `psi` and poloidal angle
`theta`, taken from the `rzphi_rsquared` interpolant's own ψ-derivative as
`da/dψ = (∂r²/∂ψ) / (2a)`. Valid wherever the interpolant is, including
under extrapolation past the ψ grid. Diverges at the magnetic axis, where
`a ~ √ψ`; callers evaluating near `ψ = 0` must check `isfinite`.
"""
function surface_da_dpsi(equil, psi::Real; theta::Real=0.0)
    return _da_dpsi_at_theta(equil, Float64(psi), Float64(theta))
end

# d(√r²)/dψ at one (ψ, θ) from the interpolant's own ψ-derivative. Shared by all radial-label
# conventions so none carries its own stencil.
@inline function _da_dpsi_at_theta(equil, psi::Float64, theta::Float64)
    r_sq = equil.rzphi_rsquared((psi, theta))
    a = sqrt(max(r_sq, 0.0))
    a > 0 || return Inf   # magnetic axis: a ~ √ψ, so da/dψ genuinely diverges
    return equil.rzphi_rsquared((psi, theta); deriv=DerivOp(1, 0)) / (2a)
end

"""
    radial_label(equil; rs_method=:midplane, theta=0.0) -> (r_at, dr_dpsi_at)

Build the pair of closures `r_at(ψ)` and `dr_dpsi_at(ψ)` defining one radial
label for the layer stack. Both closures must come from the same label,
because the r-based shear `(r/q)dq/dr`, `τ_R = μ₀r²/η`, `τ_E = r²/χ`, the
`d_β/r` normalization, the Δ' reference-length factor `k_ref = r_s/(da/dψ)`,
and any metre-to-ψ width conversion all have to live in one coordinate; a
mismatched `r` and `dr/dψ` silently corrupts every one of them. All
derivatives are analytic — no label carries a finite-difference stencil.

# Labels

  - `:midplane` -- outboard-midplane chord from the magnetic axis at `theta`
    (historical default), natural for comparison with midplane diagnostics.
  - `:halfwidth` -- midplane half-chord, the mean of the outboard and inboard
    chords at `θ = 0` and `θ = 0.5`; shift-free. Coincides with the flux label
    on circular equilibria but is its own convention on shaped ones.
  - `:fsa` -- θ-mean surface radius, a 128-point midpoint mean of the local
    minor radius; the closest geometric approximation to `:flux` at interior
    surfaces of shaped equilibria.
  - `:volume` -- cylinder-equivalent label `√(V(ψ)/(2π²R₀))`, the
    Rutherford-literature convention.
  - `:flux` -- toroidal-flux label. Fitzpatrick, Nucl. Fusion (2025),
    Eq. 30: `dψ_p/dr = B₀ r g/q` integrates to `ψ_t = B₀r²/2`, so
    `r = √(2ψ_t/B₀)` with `ψ_t = psio·∫₀^ψ (q/g) dψ′` and `g = F/(B₀R₀)`.
    Defined from flux alone, it carries no circular-cross-section assumption,
    and its derivative `dr/dψ ∝ q` grows toward a separatrix where the
    geometric labels' `da/dψ` collapses.

On shaped equilibria the labels agree at low-q surfaces and diverge strongly
near the edge, where the slab-layer matching is label-ambiguous regardless of
choice. The label is selected programmatically; it is not exposed via TOML.
"""
function radial_label(equil; rs_method::Symbol=:midplane, theta::Real=0.0)
    theta_f = Float64(theta)

    _flux_r, _flux_dr = if rs_method === :flux
        b0f = Float64(equil.params.b0)
        R0f = Float64(equil.ro)
        psiof = Float64(equil.psio)
        xs_f = collect(Float64, equil.profiles.xs)
        # Carry g = F/(B0 R0) rather than assuming g = 1: r² = 2∫(q/g)dψ_p/B0.
        _g_at(x) = Float64(equil.profiles.F_spline(x)) / (2π * b0f * R0f)
        qg = [Float64(equil.profiles.q_spline(x)) / _g_at(x) for x in xs_f]
        Phi = collect(Float64, cumulative_integrate(cubic_interp(xs_f, qg)))
        r_knots = sqrt.(max.(2 .* psiof .* Phi ./ b0f, 0.0))
        rspl = cubic_interp(xs_f, r_knots; extrap=ExtendExtrap())
        (ψ -> Float64(rspl(Float64(ψ))),
            ψ -> psiof * Float64(equil.profiles.q_spline(Float64(ψ))) /
                 (b0f * _g_at(Float64(ψ)) * max(Float64(rspl(Float64(ψ))), eps())))
    else
        (nothing, nothing)
    end

    _a_at(ψ, θ) = sqrt(max(equil.rzphi_rsquared((Float64(ψ), Float64(θ))), 0.0))

    _rs_at(ψ) =
        if rs_method === :fsa
            N = 128
            s = 0.0
            @inbounds for k in 1:N
                s += _a_at(ψ, (k - 0.5) / N)
            end
            s / N
        elseif rs_method === :halfwidth
            0.5 * (_a_at(ψ, 0.0) + _a_at(ψ, 0.5))
        elseif rs_method === :volume
            # Lower bound just off the axis, where dV/dψ is extrapolated; the omitted sliver
            # of core volume is negligible for an edge-surface label.
            V = integrate(equil.profiles.dVdpsi_spline, 1e-4, Float64(ψ))
            sqrt(max(V, 0.0) / (2π^2 * equil.ro))
        elseif rs_method === :flux
            _flux_r(ψ)
        else
            surface_minor_radius(equil, ψ; theta=theta_f)
        end

    _da_dpsi_at(ψ) =
        if rs_method === :fsa
            N = 128
            s = 0.0
            @inbounds for k in 1:N
                s += _da_dpsi_at_theta(equil, Float64(ψ), (k - 0.5) / N)
            end
            s / N
        elseif rs_method === :halfwidth
            0.5 * (_da_dpsi_at_theta(equil, Float64(ψ), 0.0) + _da_dpsi_at_theta(equil, Float64(ψ), 0.5))
        elseif rs_method === :volume
            Float64(equil.profiles.dVdpsi_spline(ψ)) / (4π^2 * equil.ro * max(_rs_at(ψ), eps()))
        elseif rs_method === :flux
            _flux_dr(ψ)
        else
            _da_dpsi_at_theta(equil, Float64(ψ), theta_f)
        end

    return (_rs_at, _da_dpsi_at)
end

"""
    build_slayer_inputs(equil, sings, profiles; …) -> Vector{SLAYERParameters}

Build a `SLAYERParameters` for each rational surface in `sings`, pulling
geometry (minor radius, r-based shear, q, dq/dψ, R₀) from the in-memory
`equil::PlasmaEquilibrium` and kinetic data (n_e, T_e, T_i, ω, ω\\_\\*e,
ω\\_\\*i) from `profiles::KineticProfiles`.

Layer inputs are assembled directly from the in-memory equilibrium and
profiles, without an intermediate file round-trip.

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
  - `chi_tor`   -- Fitzpatrick's χ_φ: anomalous perpendicular ion MOMENTUM
    diffusivity `[m²/s]` (a viscosity entering the vorticity and parallel-flow
    equations — not a heat diffusivity). Scalar or a callable of `psi`
    (default `1.0`).
  - `dr_val`    -- resistive interchange index `D_R = E + F + H²`
    (Glasser-Greene-Johnson 1975) feeding the critical-Δ formulas
    (`:lar`, `:rfitzp`, `:toroidal`). When `nothing` (default), Julia
    derives it per-surface from the equilibrium as
    `dr_val_k = D_R(ψ_k) = E_k + F_k + H_k²`,
    consistent with Connor-Hastie-Helander 2015 (PPCF 57 065001) Eq. 59
    which uses `(−D_R)` in the χ_‖-matching critical-Δ. Pass a scalar /
    vector / callable to override.

    **NOTE**: the χ_‖-matching critical-Δ requires the resistive
    interchange index `D_R = E + F + H²` (Glasser-Greene-Johnson 1975),
    NOT the Mercier index `D_I = E + F + H − 1/4`. The two differ by
    `(H − 1/2)²`, which is non-trivial on shaped equilibria (~factor 3 on
    DIII-D); this code uses the physically correct `D_R`.
  - `dgeo_val`  -- Connor 2015 (PPCF 57 065001) Eq. 59 geometric factor
    used by `dc_type=:toroidal`. When `nothing` (default), an error is
    raised if `dc_type=:toroidal` is also requested — the auto-derived
    formula additionally needs ⟨|∇ψ|²⟩ FSA which `ResistGeometry`
    doesn't currently expose. Pass a scalar / vector / callable to use
    a prescribed value. (For `dc_type=:rfitzp` and `:lar`, dgeo_val is
    not consulted.)
  - `dc_type`   -- `:none` (default), `:lar`, `:rfitzp`, or `:toroidal`.
  - `rs_method` -- radial label defining `r_s` for the whole layer stack:
    `:midplane` (default), `:halfwidth`, `:fsa`, `:volume`, or `:flux`. See
    [`radial_label`](@ref) for the definitions. S, the r-based shear, W_d,
    and the Δ' reference-length factor `k_ref` all follow the choice
    together, so every option is self-consistent. Not exposed via TOML —
    programmatic use only.
  - `theta`     -- poloidal angle at which to measure minor radius (default
    `0.0`, outboard midplane).
  - `resistivity_model` -- `SauterNeoModel()` (default), `RedlNeoModel()`,
    `SpitzerModel()`, or `SpitzerHarmModel()` (legacy Fitzpatrick σ_∥).
    Sets the η entering τ_R = μ₀r_s²/η. With a neoclassical model, `f_trap`
    and ν*_e are taken from the surface's `ResistGeometry` if populated
    (via `ForceFreeStates.resist_eval_all!`), otherwise fall back to the
    ε-only Lin-Liu-Miller form and `rs/R_0` aspect ratio.
  - `lnLambda_form` -- Coulomb-log form passed through to `slayer_parameters`
    (default `:nrl`; `:wesson` + `SpitzerHarmModel()` reproduces legacy
    SLAYER exactly).
"""
function build_slayer_inputs(equil, sings, profiles::KineticProfiles;
    bt=nothing,
    R0=nothing,
    rs_method::Symbol=:midplane,
    mu_i::Real=2.0,
    zeff::Real=1.0,
    z_i::Real=1.0,
    chi_perp=1.0,
    chi_tor=1.0,
    dr_val=nothing,
    dgeo_val=nothing,
    dc_type::Symbol=:none,
    theta::Real=0.0,
    compute_omega_star::Bool=true,
    resistivity_model::NeoResistivityModel=SauterNeoModel(),
    lnLambda_form::Symbol=:nrl)
    R0_use = R0 === nothing ? equil.ro : Float64(R0)
    _eval(x, ψ) = x isa Real ? Float64(x) : Float64(x(ψ))

    # Compute physical B_T = F(ψ) / (2π·R₀) per surface from the F spline
    # when `bt` is not explicitly supplied.
    _bt_at(ψ) =
        if bt === nothing
            Float64(equil.profiles.F_spline(ψ)) / (2π * R0_use)
        elseif bt isa Real
            Float64(bt)
        else
            Float64(bt(ψ))
        end

    _rs_at, _da_dpsi_at = radial_label(equil; rs_method=rs_method, theta=theta)

    # Per-surface ω_*e, ω_*i (diamagnetic frequencies) from spline
    # derivatives. When `compute_omega_star=true` we override any ω_*e/ω_*i
    # carried in `profiles`. Main-ion density is
    # taken equal to the electron density (quasi-neutrality, matching the
    # staging step).
    chi1 = 2π * equil.psio
    _omega_star_at(ψ) = begin
        n_e = Float64(profiles.n_e(ψ))
        dn_e = Float64(profiles.n_e(ψ; deriv=DerivOp(1)))
        T_e = Float64(profiles.T_e(ψ))
        dT_e = Float64(profiles.T_e(ψ; deriv=DerivOp(1)))
        T_i = Float64(profiles.T_i(ψ))
        dT_i = Float64(profiles.T_i(ψ; deriv=DerivOp(1)))
        ω_star_e = (2π / chi1) * (T_e * dn_e / n_e + dT_e)
        ω_star_i = -(2π / (Float64(z_i) * chi1)) * (T_i * dn_e / n_e + dT_i)
        return (ω_star_e, ω_star_i)
    end

    out = Vector{SLAYERParameters}(undef, length(sings))
    for (k, sing) in enumerate(sings)
        psi = sing.psifac
        q = sing.q
        q1 = sing.q1

        rs = _rs_at(psi)
        da_dpsi = _da_dpsi_at(psi)
        sval_r = r_based_shear(rs, q, q1, da_dpsi)

        prof = profiles(psi)
        # Override ω_*e, ω_*i with spline-derivative values when requested.
        ω_e_use, ω_i_use = if compute_omega_star
            _omega_star_at(psi)
        else
            (prof.omega_e, prof.omega_i)
        end

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
        f_trap_kw = rg === nothing ? nothing : rg.f_trap
        R_major_eff = rg === nothing ? nothing : rg.R_major
        nu_e_star_kw = if rg === nothing ||
                          resistivity_model isa Union{SpitzerModel,SpitzerHarmModel}
            nothing
        else
            lnL = coulomb_log_e(prof.n_e, prof.T_e; form=lnLambda_form)
            nu_star_e(prof.n_e, prof.T_e, rg.R_major, rg.eps_local,
                q, zeff; lnLamb=lnL)
        end

        # dr_val: per-surface resistive interchange index D_R = E + F + H²
        # (Glasser-Greene-Johnson 1975). Used by `_solve_dc_tmp` to compute
        # the χ_‖-matching critical-Δ via Connor-Hastie-Helander 2015 Eq. 59,
        # which has `(−D_R)` as a multiplier. NOT the Mercier index
        # D_I = E + F + H − 1/4 (see this function's docstring); we use the
        # physically correct D_R here.
        dr_val_k = if dr_val === nothing
            rg === nothing &&
                throw(
                    ArgumentError(
                        "build_slayer_inputs: dr_val=nothing " *
                        "requires `sing.restype` populated by " *
                        "ForceFreeStates.resist_eval_all!. " *
                        "Surface k=$k has restype=nothing."
                    )
                )
            rg.E + rg.F + rg.H^2
        else
            _eval(dr_val, psi)
        end

        # dgeo_val: only used by dc_type=:toroidal (the Connor-Hastie-
        # Helander 2015 formula). Auto-derivation requires ⟨|∇ψ|²⟩ FSA
        # which the current `ResistGeometry` doesn't expose; for now we
        # require an explicit value if the toroidal dc_type is selected.
        dgeo_val_k = if dgeo_val === nothing
            dc_type === :toroidal &&
                throw(
                    ArgumentError(
                        "build_slayer_inputs: dc_type=:toroidal " *
                        "needs `dgeo_val` (Connor 2015 PPCF 57 " *
                        "065001 Eq. 59 geometric factor). " *
                        "Auto-derivation from equilibrium not " *
                        "yet implemented; pass a scalar / vector " *
                        "/ callable explicitly."
                    )
                )
            0.0
        else
            _eval(dgeo_val, psi)
        end

        # Reference-length conversion inputs for the outer Δ': K = r_s·(dψ_N/dr)|_s and
        # μ = √(−D_I) (Glasser-Greene-Johnson 1975), with μ clamped to 0 on Mercier-unstable
        # surfaces (the factor turns complex there) and K = 1 whenever da/dψ is not a usable
        # positive number (zero/non-finite/negative would corrupt the Δ' diagonal).
        k_ref_k = if isfinite(da_dpsi) && da_dpsi > 0.0
            rs / da_dpsi
        else
            @warn("build_slayer_inputs: da/dψ = $da_dpsi at ψ = $psi is not usable; leaving " *
                  "Δ' unconverted (k_ref = 1) at this surface.", maxlog=3)
            1.0
        end
        mu_k = if rg === nothing
            @warn("build_slayer_inputs: sing.restype not populated; using the " *
                  "slab Mercier exponent μ = 1/2 for the Δ' reference-length " *
                  "conversion at all such surfaces.", maxlog=1)
            0.5
        else
            sqrt(max(-(rg.E + rg.F + rg.H - 0.25), 0.0))
        end

        out[k] = slayer_parameters(;
            n_e=prof.n_e, t_e=prof.T_e, t_i=prof.T_i,
            omega=prof.omega, omega_e=ω_e_use, omega_i=ω_i_use,
            qval=q, sval_r=sval_r, bt=_bt_at(psi),
            rs=rs, R0=R0_use, mu_i=mu_i, zeff=zeff,
            chi_perp=_eval(chi_perp, psi),
            chi_tor=_eval(chi_tor, psi),
            m=m_res, n=n_res,
            dr_val=dr_val_k,
            dgeo_val=dgeo_val_k,
            dc_type=dc_type, ising=k,
            resistivity_model=resistivity_model,
            f_trap=f_trap_kw,
            nu_e_star=nu_e_star_kw,
            R_major_eff=R_major_eff,
            lnLambda_form=lnLambda_form,
            k_ref=k_ref_k,
            mu_mercier=mu_k
        )
    end
    return out
end
