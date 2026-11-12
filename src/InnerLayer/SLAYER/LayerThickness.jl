# LayerThickness.jl
#
# Resistive inner-layer *width* (del_s) diagnostic for the SLAYER two-fluid drift-MHD slab layer (Fitzpatrick,
# Tearing Mode Dynamics in Tokamak Plasmas, IOP 2023; Park 2022, Phys. Plasmas 29, 072506; Burgess et al. 2026).
# Distinct from the matching Δ of Riccati.jl: a reduced Riccati ODE at the electron diamagnetic frequency gives the
# ratio delta_s/d_beta, which depends only on Q_e, P_perp, P_tor, tau and D_norm.

using OrdinaryDiffEq

# ---------------------------------------------------------------------
# Pre-computed q-independent constants for the del_s Riccati ODE.
# Normalisation block:
#   Q_hat      = (Q_e (1+tau)/tau) / D_norm^4
#   P_perp_hat = P_perp / D_norm^6
#   P_tor_hat  = P_tor  / D_norm^6
# `one_plus = 1 + 1/tau`, `inv_c = 1/(1+1/tau)` recur in E, F and the
# boundary/extraction prefactor, so they are cached here too.
# ---------------------------------------------------------------------
struct _DelSConsts
    Q_hat::Float64       # normalised electron diamagnetic frequency
    Pperp_hat::Float64   # normalised perpendicular Prandtl number
    Ptor_hat::Float64    # normalised toroidal Prandtl number
    one_plus::Float64    # 1 + 1/tau
    inv_c::Float64       # 1 / (1 + 1/tau)
end

@inline function _build_dels_consts(p::SLAYERParameters)
    one_plus = 1.0 + 1.0 / p.tau                       # (1+tau)/tau
    D2 = p.D_norm * p.D_norm
    D4 = D2 * D2
    D6 = D4 * D2
    return _DelSConsts(
        (p.Q_e * one_plus) / D4,
        p.P_perp / D6,
        p.P_tor / D6,
        one_plus,
        1.0 / one_plus
    )
end

# Scalar ODE right-hand side dW/dq of the del_s Riccati; E and F are complex through the `im·Q_hat` terms.
@inline function _dels_rhs(W::Number, c::_DelSConsts, q::Real)
    q2 = q * q
    q4 = q2 * q2
    E = -(c.Q_hat^2) * c.inv_c -
        im * c.Q_hat * (c.Pperp_hat + c.Ptor_hat) * q2 +
        c.Pperp_hat * c.Ptor_hat * q4
    F = c.Pperp_hat - im * c.Q_hat + c.one_plus * c.Ptor_hat * q2
    return W / q - (W * W) / q + (q * E) / F
end

# Analytic Jacobian dF/dW: the q·E/F term is W-independent, leaving
# d/dW(W/q - W²/q) = 1/q - 2W/q.
@inline _dels_jac(W::Number, c::_DelSConsts, q::Real) = 1.0 / q - 2.0 * W / q

"""
    riccati_del_s(p::SLAYERParameters;
                  q_start=5*p.D_norm, q_min=1e-5,
                  reltol=1e-10, abstol=1e-10, maxiters=50_000,
                  solver=Rodas5P(autodiff=false)) -> ComplexF64

Solve the SLAYER `del_s` inner-layer Riccati ODE and return the
**dimensionless** layer-thickness ratio `δ_s / d_β` at one rational
surface.

Integrates `dW/dq = W/q − W²/q + q·E/F` inward from `q_start = 5·D_norm`
to `q_min`, with asymptotic boundary value `W = −α·q_start² − 0.5`,
`α = √(P̂_⊥ / (1 + 1/τ))`. The result is
`−(π / √(1 + 1/τ)) · W′(q_min)`, evaluated from a single RHS call at the
inner endpoint.

Returns `NaN + NaN·im` if the stiff integration does not converge (the
caller treats this as a missing diagnostic rather than a hard error).

Multiply the result by `p.d_beta` to obtain the resistive layer
thickness in meters (see [`slayer_layer_thickness`](@ref)).
"""
function riccati_del_s(p::SLAYERParameters;
    q_start::Real=5.0 * p.D_norm,
    q_min::Real=1e-5,
    reltol::Real=1e-10,
    abstol::Real=1e-10,
    maxiters::Integer=50_000,
    solver=Rodas5P(; autodiff=false))
    if !(q_start > q_min)
        @debug "riccati_del_s: degenerate integration span" q_start q_min p.D_norm
        return ComplexF64(NaN, NaN)
    end

    c = _build_dels_consts(p)

    # Asymptotic boundary condition at large q (α and W₀ are real).
    α = sqrt(c.Pperp_hat * c.inv_c)
    W0 = ComplexF64(-α * q_start^2 - 0.5)

    f = ODEFunction{false}(_dels_rhs; jac=_dels_jac)
    prob = ODEProblem(f, W0, (q_start, q_min), c)
    sol = solve(prob, solver;
        reltol=reltol, abstol=abstol, maxiters=maxiters,
        save_everystep=false, dense=false)

    if sol.retcode != ReturnCode.Success
        @debug "SLAYER riccati_del_s did not return Success" sol.retcode
        return ComplexF64(NaN, NaN)
    end

    W_end = sol.u[end]
    dW_end = _dels_rhs(W_end, c, q_min)
    return -(π / sqrt(c.one_plus)) * dW_end
end

"""
    LayerWidths

Resistive inner-layer length scales at one rational surface, in meters.
The primary quantity is `delta_s_m`, the resistive layer thickness from
the SLAYER `del_s` Riccati solve; `d_beta` is the β-weighted ion scale it
is built from, retained as a drift-scale reference.

# Fields

  - `ising`, `m`, `n` -- surface index and mode numbers (traceability)
  - `dels_db`   -- dimensionless `δ_s / d_β` from [`riccati_del_s`](@ref)
  - `delta_s`   -- complex layer thickness `δ_s = dels_db · d_β` [m]
  - `delta_s_m` -- `|δ_s|`, the resistive layer thickness in meters (primary)
  - `d_beta`    -- β-weighted ion scale `c_β·d_i` in meters (drift reference)
  - `delta_norm`  -- layer normalization length `r_s · S^(-1/3)` in meters (`Δ̂' = Δ'·delta_norm`), the reciprocal of
    `SLAYERParameters.delta_n`, shared by all four regimes of Burgess et al. (2026) Eqs. (10)-(13); **not** the
    Furth-Killeen-Rosenbluth `S^(-2/5)` tearing width
  - `delta_dr` -- diffusive-resistive layer thickness in meters, Fitzpatrick (2025) Eq. (100), equal to
    `rs · lu^(-1/2) P_perp^(1/4) / d_beta_hat^(1/2)` with `d_beta_hat = d_beta/rs` (SLAYER's `lu` already carries the paper's
    `(n|s|)^(1/2)`). Strong shear forces edge layers into this (`nu = 1/4`) regime, so it is the width the edge overlap
    criterion compares against surface spacing
  - `delta_visco` -- viscous-resistive scale `delta_norm · P_perp^(1/6)` in meters,
    from the `P^(1/6)` broadening of the VR-regime growth rate, Burgess et al.
    (2026) Eq. (11). The paper gives the growth rate rather than an explicit width;
    this is the corresponding length

`delta_s_m` should sit within a few orders of magnitude of `d_beta` for a
well-posed surface (`dels_db` is O(1)); a large gap flags a normalisation
or input problem.
"""
struct LayerWidths
    ising::Int
    m::Int
    n::Int
    dels_db::ComplexF64
    delta_s::ComplexF64
    delta_s_m::Float64
    d_beta::Float64
    delta_norm::Float64
    delta_visco::Float64
    delta_dr::Float64
end

"""
    slayer_algebraic_widths(p::SLAYERParameters) -> (; delta_norm, delta_visco, delta_dr)

The three comparison scales that follow from `p` alone, with no Riccati solve. `delta_dr` is
the channel the resistive-layer overlap criterion uses, so a scan needs only this.
"""
function slayer_algebraic_widths(p::SLAYERParameters)
    # Layer normalization length: exactly 1/delta_n (delta_n = S^(1/3)/r_s), computed
    # from rs and lu so it reads as a length. Burgess et al. (2026), the Δ̂' = Δ'/S^(1/3)
    # normalization preceding Eq. (10).
    delta_norm = p.rs * p.lu^(-1.0 / 3.0)
    # Viscous-resistive broadening, P^(1/6) coefficient of Burgess et al. (2026) Eq. (11).
    delta_visco = delta_norm * p.P_perp^(1.0 / 6.0)
    # Diffusive-resistive width, Fitzpatrick (2025) Eq. (100). The paper's tau_A (Eq. 74) has no shear, while SLAYER's
    # tau_h divides by n|s|, so lu = (n|s|)·tau_R/tau_A already supplies Eq. (100)'s (n|s|)^(-1/2).
    d_beta_hat = p.d_beta / p.rs
    delta_dr = d_beta_hat > 0 ? p.rs * p.lu^(-0.5) * p.P_perp^(0.25) / sqrt(d_beta_hat) : NaN
    return (; delta_norm, delta_visco, delta_dr)
end

"""
    slayer_layer_thickness(p::SLAYERParameters; kwargs...) -> LayerWidths

Compute the resistive inner-layer thickness in meters at one rational
surface.

Runs [`riccati_del_s`](@ref) for the dimensionless `δ_s / d_β` and scales
by `p.d_beta` to obtain `δ_s` in meters. Keyword arguments are forwarded
to `riccati_del_s`.

The algebraic comparison scales `delta_norm`, `delta_visco` and `delta_dr` come from `p`
directly and do not depend on the Riccati solve; [`slayer_algebraic_widths`](@ref) returns
just those, for callers that do not need `delta_s`.
"""
function slayer_layer_thickness(p::SLAYERParameters; kwargs...)
    dels_db = riccati_del_s(p; kwargs...)
    delta_s = dels_db * p.d_beta
    w = slayer_algebraic_widths(p)
    return LayerWidths(p.ising, p.m, p.n,
        dels_db, delta_s, abs(delta_s),
        p.d_beta, w.delta_norm, w.delta_visco, w.delta_dr)
end
