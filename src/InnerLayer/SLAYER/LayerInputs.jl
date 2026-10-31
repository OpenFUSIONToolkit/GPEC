# LayerInputs.jl
#
# Build per-surface `SLAYERParameters` from an in-memory `PlasmaEquilibrium`, the `SingType` rational-surface
# data from `ForceFreeStates`, and a `KineticProfiles` object.

using ..Utilities: KineticProfiles
using ...Utilities.NeoclassicalResistivity: NeoResistivityModel, SpitzerModel,
    coulomb_log_e, nu_star_e
using FastInterpolations: DerivOp, integrate, cubic_interp, cumulative_integrate, ExtendExtrap

"""
    surface_minor_radius(equil, psi; theta=0.0) -> Float64

Minor radius at normalized flux `psi` and poloidal angle `theta`,
computed from `equil.rzphi_rsquared` as `√((R − R₀)² + (Z − Z₀)²)`.
`theta` is normalized to [0, 1): `0.0` (outboard midplane) is the default and
`0.5` is the inboard midplane.
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
    radial_label(equil; rs_method=:flux, theta=0.0, q_beyond=nothing) -> (r_at, dr_dpsi_at)

Build the pair of closures `r_at(ψ)` and `dr_dpsi_at(ψ)` defining one radial
label for the layer stack. Both closures must come from the same label,
because the r-based shear `(r/q)dq/dr`, `τ_R = μ₀r²/η`, `τ_E = r²/χ`, the
`d_β/r` normalization, the Δ' reference-length factor `k_ref = r_s/(da/dψ)`,
and any metre-to-ψ width conversion all have to live in one coordinate; a
mismatched `r` and `dr/dψ` silently corrupts every one of them. All
derivatives are analytic — no label carries a finite-difference stencil.

# Labels

  - `:midplane` -- outboard-midplane chord from the magnetic axis at `theta`
    natural for comparison with midplane diagnostics.
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
    `r = √(2ψ_t/B₀)` with `ψ_t = psio·∫₀^ψ (q/g) dψ′` and `g = F/(B₀R₀)`,
    `F = R·B_tor`. Note `profiles.F_spline` stores `2πF`, so the
    implementation divides it by 2π to form `g`.
    Defined from flux alone, it carries no circular-cross-section assumption,
    and its derivative `dr/dψ ∝ q` grows toward a separatrix where the
    geometric labels' `da/dψ` collapses. Beyond the ψ grid the q spline is a
    cubic extrapolation that does not follow that growth, so `q_beyond`, a
    callable of ψ, supplies q there when the caller knows it.

On shaped equilibria the labels agree at low-q surfaces and diverge strongly
near the edge, where the slab-layer matching is label-ambiguous regardless of
choice. The label is selected programmatically; it is not exposed via TOML.
"""
function radial_label(equil; rs_method::Symbol=:flux, theta::Real=0.0, q_beyond=nothing)
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
        _q_at(ψ) = (q_beyond !== nothing && ψ > xs_f[end]) ? Float64(q_beyond(ψ)) : Float64(equil.profiles.q_spline(ψ))
        (ψ -> Float64(rspl(Float64(ψ))),
            ψ -> psiof * _q_at(Float64(ψ)) /
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
    toroidal_dgeo(; chi1, v1, q, q1, n, avg_bsq, avg_dpsisq, k_ref) -> Float64

Geometric factor of the toroidal critical-Δ, Connor, Ham, Hastie & Liu 2015
(PPCF 57 065001) Eq. 59, `V_s·(α²Λ²/(⟨B²⟩⟨|∇V|²⟩))^{1/4}`, converted from the
paper's `Y = (V−V_s)/V_s` reference to the `x̂ = (r−r_s)/r_s` reference shared
by the slab layer, the `:fitzpatrick` critical-Δ, and the reference-length-converted
outer Δ'. In GPEC quantities, with `V` the flux-surface volume, `ψ_N` the
normalized poloidal flux, and `'` = d/dV:

  - `α = 2πn/χ'` with `χ' = chi1/v1` (`χ` the full poloidal flux,
    `chi1 = 2π·psio`, `v1 = dV/dψ_N`)
  - `Λ = ψ_t'²·(ι/2π)'` with `ψ_t' = q·chi1/v1` (toroidal flux) and
    `(ι/2π)' = −q1/(q²·v1)` (`q1 = dq/dψ_N`)
  - `⟨|∇V|²⟩ = v1²·⟨|∇ψ_N|²⟩`, both averages being normalized flux-surface
    averages (⟨1⟩ = 1)
  - the reference conversion `r_s·(dV/dr)/V_s = k_ref·v1/V_s` with
    `k_ref = r_s·dψ_N/dr`, so `V_s` cancels and is never integrated.

At large aspect ratio this reduces to `√(n·s·r_s/R₀)` with `s = (r_s/q)·dq/dr`,
so `dc_type=:toroidal` coincides with `:fitzpatrick` there; the paper's own Eq. 61
is recovered after dividing by `r_s`.

The reference conversion is deliberately linear (Frobenius exponent ½, the
paper's H = 0 ordering): the critical-Δ is a layer-side quantity like the slab
Δ̂(Q) and W_d, whereas the outer Δ' carries the Mercier exponent `α_M = √(−D_I)`
and is converted with `K^(2α_M)`.
"""
function toroidal_dgeo(; chi1::Real, v1::Real, q::Real, q1::Real, n::Integer,
    avg_bsq::Real, avg_dpsisq::Real, k_ref::Real)
    alpha, lambda = _connor_alpha_lambda("toroidal_dgeo", chi1, v1, q, q1, n)
    grad_v_sq = v1^2 * avg_dpsisq
    return k_ref * v1 * (alpha^2 * lambda^2 / (avg_bsq * grad_v_sq))^0.25
end

"""
    toroidal_kpar(; chi1, v1, q, q1, n, avg_bsq, k_ref) -> Float64

Parallel-wavenumber gradient `K∥` [1/m] of the toroidal critical-Δ closure: an island of normalized
width `W` (in the `x̂ = (r−r_s)/r_s` reference) sees `k∥ = K∥·W`. From Connor, Ham, Hastie & Liu 2015
(PPCF 57 065001) Eq. 20, whose transport balance reads `χ⊥⟨|∇V|²⟩ ∂²_x = χ∥ (α²Λ²/⟨B²⟩) x²` in
`x = V − V_s`, the parallel wavenumber is `k∥ = |αΛ|·x/√⟨B²⟩`; with `x = W·r_s·(dV/dr) = W·k_ref·v1`,

```
K∥ = |α·Λ|·k_ref·v1 / √⟨B²⟩,
```

with `α`, `Λ` as in [`toroidal_dgeo`](@ref). At large aspect ratio it reduces to Fitzpatrick's
(1995, Phys. Plasmas 2 825, Eq. 132) cylindrical `n·s/R₀`. It is not a function of `D_geo` alone:
`K∥ = (D_geo²/r_s)·r_s√⟨|∇ψ_N|²⟩/k_ref`, and the last factor is 1 only in a cylinder.
"""
function toroidal_kpar(; chi1::Real, v1::Real, q::Real, q1::Real, n::Integer, avg_bsq::Real, k_ref::Real)
    alpha, lambda = _connor_alpha_lambda("toroidal_kpar", chi1, v1, q, q1, n)
    return abs(alpha * lambda) * k_ref * v1 / sqrt(avg_bsq)
end

# Connor et al. 2015 α = 2πn/χ' and Λ = ψ_t'²·(ι/2π)' (' = d/dV) from ψ_N-grid quantities.
function _connor_alpha_lambda(caller, chi1, v1, q, q1, n)
    v1 > 0 || throw(ArgumentError("$caller: dV/dψ_N must be positive, got $v1"))
    chi1 != 0 || throw(ArgumentError("$caller: chi1 must be non-zero"))
    alpha = 2π * n * v1 / chi1
    psit1 = q * chi1 / v1
    lambda = psit1^2 * (-q1 / (q^2 * v1))
    return alpha, lambda
end

"""
    build_slayer_inputs(equil, sings, profiles; …) -> Vector{SLAYERParameters}

Build a `SLAYERParameters` for each rational surface in `sings` from the in-memory `equil::PlasmaEquilibrium`
(geometry, q, dq/dψ, R₀) and `profiles::KineticProfiles` (n_e, T_e, T_i, and ω\\_\\*e, ω\\_\\*i from their gradients).
Keywords documented as scalar-or-callable are evaluated per surface at ψ.

# Arguments

  - `equil` -- `PlasmaEquilibrium`
  - `sings` -- `Vector{SingType}`, one per resonant surface
  - `profiles` -- `KineticProfiles` valid across all `sings` ψ values

# Keyword arguments

  - `bt` -- toroidal field in T, scalar or callable; `nothing` (default) uses B_T = F(ψ)/(2π·R₀). `equil.config.b0_norm` is a
    normalization, not this field
  - `R0` -- major radius in m (default `equil.ro`)
  - `rs_method` -- radial label of the whole layer stack: `:flux` (default), `:midplane`, `:halfwidth`, `:fsa` or `:volume`;
    see [`radial_label`](@ref). Programmatic only, not exposed via TOML
  - `q_beyond` -- callable q(ψ) the `:flux` label uses beyond the ψ grid (default `nothing`)
  - `mu_i` -- ion mass in proton masses (default `2.0`, D)
  - `zeff` -- effective charge, scalar or callable (default `1.0`)
  - `z_i` -- main-ion charge entering ω_*i (default `1.0`)
  - `chi_perp_e` -- electron perpendicular heat diffusivity in m²/s, scalar or callable (default `1.0`)
  - `chi_perp_i` -- ion perpendicular heat diffusivity in m²/s, scalar or callable; needed by `P_perp_model = :chi_perp_i` and `:D_perp`
  - `chi_perp_Dc` -- electron χ⊥ in m²/s used only by the critical Δ, scalar or callable; `nothing` (default) uses `chi_perp_e`
  - `tau_E` -- whole-plasma energy confinement time τ̄_E in s, needed by `P_perp_model = :tau_E`, which takes `a` from the `:flux` label
  - `P_perp_model` -- how `P_perp` is built (default `:D_perp`); see [`slayer_parameters`](@ref)
  - `chi_tor` -- toroidal momentum diffusivity χ_φ in m²/s, scalar or callable, setting P_tor = τ_R/τ_φ with τ_φ = r_s²/χ_φ
    (Fitzpatrick, Phys. Plasmas 30, 092512 (2023)) (default `1.0`)
  - `dr_val` -- resistive interchange index D_R = E + F + H² (Glasser-Greene-Johnson 1975) of the critical Δ, **not** the Mercier
    D_I; `nothing` (default) derives it from `sing.restype`, a scalar or callable overrides it
  - `dgeo_val` -- Connor et al. 2015 (PPCF 57 065001) Eq. 59 factor in the `r_s` reference ([`toroidal_dgeo`](@ref)), used only by
    `dc_type = :toroidal`; `nothing` (default) derives it from `sing.restype` (required then). A prescribed value must already be in
    the `r_s` reference, i.e. Eq. 59 times `k_ref·v1/V_s` (≈ 2 at large aspect ratio)
  - `dc_type` -- critical-Δ formula: `:none` (default), `:lar`, `:fitzpatrick` or `:toroidal`; `:toroidal` takes K∥ from
    [`toroidal_kpar`](@ref), or the cylindrical `n·|s|/R₀` with a warning when the surface has no `ResistGeometry`
  - `theta` -- poloidal angle of the minor-radius chord, used only by `rs_method = :midplane` (default `0.0`, outboard midplane)
  - `resistivity_model` -- `SauterNeoModel()` (default), `RedlNeoModel()`, `SpitzerModel()` or `SpitzerHarmModel()`; the
    neoclassical `f_trap` and ν*_e come from `sing.restype` when populated, else the ε-only forms
  - `lnLambda_form` -- Coulomb-log form (default `:nrl`; `:wesson` with `SpitzerHarmModel()` reproduces the legacy SLAYER)
"""
function build_slayer_inputs(equil, sings, profiles::KineticProfiles;
    bt=nothing,
    R0=nothing,
    rs_method::Symbol=:flux,
    q_beyond=nothing,
    mu_i::Real=2.0,
    zeff=1.0,
    z_i::Real=1.0,
    chi_perp_e=1.0,
    chi_perp_i=nothing,
    chi_perp_Dc=nothing,
    tau_E::Union{Real,Nothing}=nothing,
    P_perp_model::Symbol=:D_perp,
    chi_tor=1.0,
    dr_val=nothing,
    dgeo_val=nothing,
    dc_type::Symbol=:none,
    theta::Real=0.0,
    resistivity_model::NeoResistivityModel=SauterNeoModel(),
    lnLambda_form::Symbol=:nrl)
    R0_use = R0 === nothing ? equil.ro : Float64(R0)
    _eval(x, ψ) = x === nothing ? nothing : x isa Real ? Float64(x) : Float64(x(ψ))

    # Physical B_T = F(ψ) / (2π·R₀) per surface when `bt` is not supplied.
    _bt_at(ψ) =
        if bt === nothing
            Float64(equil.profiles.F_spline(ψ)) / (2π * R0_use)
        elseif bt isa Real
            Float64(bt)
        else
            Float64(bt(ψ))
        end

    _rs_at, _da_dpsi_at = radial_label(equil; rs_method=rs_method, theta=theta, q_beyond=q_beyond)
    # Boundary minor radius on the flux label, needed only to turn a whole-plasma τ̄_E into a uniform χ_E.
    a_minor = P_perp_model === :tau_E ? first(radial_label(equil; rs_method=:flux, theta=theta, q_beyond=q_beyond))(1.0) : nothing

    # Per-surface ω_*e, ω_*i from spline derivatives: ω_* = n·(dp/dψ)/(e·n_e) in flux coordinates, with n_i = n_e.
    chi1 = 2π * equil.psio
    _kinetic_at(ψ) = (n_e=Float64(profiles.n_e(ψ)), dn_e=Float64(profiles.n_e(ψ; deriv=DerivOp(1))),
        T_e=Float64(profiles.T_e(ψ)), dT_e=Float64(profiles.T_e(ψ; deriv=DerivOp(1))),
        T_i=Float64(profiles.T_i(ψ)), dT_i=Float64(profiles.T_i(ψ; deriv=DerivOp(1))))
    _omega_star_at(ψ, n_tor) = begin
        (; n_e, dn_e, T_e, dT_e, T_i, dT_i) = _kinetic_at(ψ)
        ω_star_e = n_tor * (2π / chi1) * (T_e * dn_e / n_e + dT_e)
        ω_star_i = -n_tor * (2π / (Float64(z_i) * chi1)) * (T_i * dn_e / n_e + dT_i)
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

        # Resonant (m, n): the first entry of `sing.m`/`sing.n`, which may hold several.
        m_res = sing.m[1]
        n_res = sing.n[1]

        prof = profiles(psi)
        kin = _kinetic_at(psi)
        # ω_*e, ω_*i from the density and temperature spline derivatives, at this surface's n.
        ω_e_use, ω_i_use = _omega_star_at(psi, n_res)

        # Trapped-fraction inputs from ResistGeometry when populated; else slayer_parameters uses the ε-only forms.
        rg = sing.restype
        (; f_trap, nu_e_star, R_major_eff) = _resistivity_geometry(rg, prof.n_e, prof.T_e, q, _eval(zeff, psi);
            resistivity_model=resistivity_model, lnLambda_form=lnLambda_form)

        # Resistive interchange index D_R = E + F + H² (Glasser-Greene-Johnson 1975), the (−D_R) of Connor et al. 2015 Eq. 59;
        # not the Mercier index D_I = E + F + H − 1/4.
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

        # Reference-length conversion inputs for the outer Δ': K = r_s·(dψ_N/dr)|_s and
        # α = √(−D_I) (Glasser-Greene-Johnson 1975 Eq. 48), with α clamped to 0 on Mercier-unstable
        # surfaces (the factor turns complex there) and K = 1 whenever da/dψ is not a usable
        # positive number (zero/non-finite/negative would corrupt the Δ' diagonal).
        k_ref_k = if isfinite(da_dpsi) && da_dpsi > 0.0
            rs / da_dpsi
        else
            @warn("build_slayer_inputs: da/dψ = $da_dpsi at ψ = $psi is not usable; leaving " *
                  "Δ' and the toroidal critical-Δ unconverted (k_ref = 1) at this surface.", maxlog = 3)
            1.0
        end

        # dgeo_val and kpar_val: Connor et al. 2015 Eq. 59 geometric factor in the r_s reference and
        # the Eq. 20 parallel-wavenumber gradient (see `toroidal_dgeo`, `toroidal_kpar`), derived
        # whenever the surface carries a ResistGeometry; only dc_type=:toroidal consumes them.
        kpar_val_k = if rg === nothing
            dc_type === :toroidal && @warn(
                "build_slayer_inputs: no ResistGeometry, so the toroidal critical-Δ χ∥ closure uses " *
                "the cylindrical parallel wavenumber n·|s|/R₀.", maxlog = 1)
            nothing
        else
            toroidal_kpar(; chi1=chi1, v1=rg.v1_local, q=q, q1=q1, n=n_res, avg_bsq=rg.avg_bsq, k_ref=k_ref_k)
        end
        dgeo_val_k = if dgeo_val === nothing
            if rg !== nothing
                toroidal_dgeo(; chi1=chi1, v1=rg.v1_local, q=q, q1=q1, n=n_res,
                    avg_bsq=rg.avg_bsq, avg_dpsisq=rg.avg_dpsisq, k_ref=k_ref_k)
            elseif dc_type === :toroidal
                throw(
                    ArgumentError(
                        "build_slayer_inputs: dc_type=:toroidal with " *
                        "dgeo_val=nothing requires `sing.restype` populated " *
                        "by ForceFreeStates.resist_eval_all!. " *
                        "Surface k=$k has restype=nothing."
                    )
                )
            else
                0.0
            end
        else
            _eval(dgeo_val, psi)
        end

        alpha_k = if rg === nothing
            @warn(
                "build_slayer_inputs: sing.restype not populated; using the " *
                "slab Mercier exponent α = 1/2 for the Δ' reference-length " *
                "conversion at all such surfaces.", maxlog=1)
            0.5
        else
            sqrt(max(-(rg.E + rg.F + rg.H - 0.25), 0.0))
        end

        out[k] = slayer_parameters(;
            n_e=prof.n_e, t_e=prof.T_e, t_i=prof.T_i,
            omega_e=ω_e_use, omega_i=ω_i_use,
            qval=q, sval_r=sval_r, bt=_bt_at(psi),
            rs=rs, R0=R0_use, mu_i=mu_i, zeff=_eval(zeff, psi),
            chi_perp_e=_eval(chi_perp_e, psi),
            chi_tor=_eval(chi_tor, psi),
            chi_perp_i=_eval(chi_perp_i, psi),
            chi_perp_Dc=_eval(chi_perp_Dc, psi),
            dlnn=kin.dn_e / kin.n_e, dlnTe=kin.dT_e / kin.T_e, dlnTi=kin.dT_i / kin.T_i,
            tau_E=tau_E, a_minor=a_minor,
            P_perp_model=P_perp_model,
            m=m_res, n=n_res,
            dr_val=dr_val_k,
            dgeo_val=dgeo_val_k,
            kpar_val=kpar_val_k,
            dc_type=dc_type, ising=k,
            resistivity_model=resistivity_model,
            f_trap=f_trap,
            nu_e_star=nu_e_star,
            R_major_eff=R_major_eff,
            lnLambda_form=lnLambda_form,
            k_ref=k_ref_k,
            alpha_mercier=alpha_k
        )
    end
    return out
end

# Trapped fraction, electron collisionality and effective major radius of the neoclassical resistivity
# closure at one surface, from its `ResistGeometry` when populated; `nothing` lets `surface_resistivity`
# fall back to the ε = r_s/R form.
function _resistivity_geometry(rg, n_e, T_e, q, zeff; resistivity_model::NeoResistivityModel, lnLambda_form::Symbol)
    rg === nothing && return (f_trap=nothing, nu_e_star=nothing, R_major_eff=nothing)
    nu_e_star = if resistivity_model isa Union{SpitzerModel,SpitzerHarmModel}
        nothing
    else
        lnL = coulomb_log_e(n_e, T_e; form=lnLambda_form)
        nu_star_e(n_e, T_e, rg.R_major, rg.eps_local, q, zeff; lnLamb=lnL)
    end
    return (f_trap=rg.f_trap, nu_e_star=nu_e_star, R_major_eff=rg.R_major)
end

"""
    layer_resistivity(equil, sings, profiles::KineticProfiles; zeff=1.0, R0=nothing, rs_method=:midplane,
                      theta=0.0, q_beyond=nothing, resistivity_model=SauterNeoModel(), lnLambda_form=:nrl) -> Vector{Float64}

Parallel resistivity η in Ω·m at each surface of `sings`, with the same closure and per-surface inputs
`build_slayer_inputs` gives the SLAYER layer, so every resistive inner layer of a run sees one η.
`zeff` is a scalar or a callable of ψ.
"""
function layer_resistivity(equil, sings, profiles::KineticProfiles;
    zeff=1.0, R0=nothing, rs_method::Symbol=:midplane, theta::Real=0.0, q_beyond=nothing,
    resistivity_model::NeoResistivityModel=SauterNeoModel(), lnLambda_form::Symbol=:nrl)
    R0_use = R0 === nothing ? equil.ro : Float64(R0)
    rs_at, _ = radial_label(equil; rs_method=rs_method, theta=theta, q_beyond=q_beyond)
    return map(sings) do sing
        psi = sing.psifac
        prof = profiles(psi)
        z = zeff isa Real ? Float64(zeff) : Float64(zeff(psi))
        geometry = _resistivity_geometry(sing.restype, prof.n_e, prof.T_e, sing.q, z;
            resistivity_model=resistivity_model, lnLambda_form=lnLambda_form)
        surface_resistivity(; n_e=prof.n_e, t_e=prof.T_e, zeff=z, qval=sing.q, rs=rs_at(psi), R0=R0_use,
            resistivity_model=resistivity_model, lnLambda_form=lnLambda_form, geometry...)
    end
end
