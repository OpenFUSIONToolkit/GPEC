# LayerThickness.jl
#
# Resistive inner-layer thickness via the SLAYER `del_s` Riccati
# formulation. Ports the Fortran `riccati_del_s` / `w_der_del_s` /
# `jac_del_s` routines (branch `slayer_growthrate`) and the
# meters-scaling `delta_s = (delta_s/d_beta) * d_beta`.
#
# Unlike `riccati_f` (which feeds the dispersion-relation root find for
# the tearing growth rate), `riccati_del_s` is a one-shot diagnostic
# evaluated at the electron diamagnetic frequency Q_e. It returns the
# dimensionless ratio delta_s/d_beta; multiplying by the beta-weighted
# ion scale d_beta gives the resistive layer thickness in meters at each
# rational surface.
#
# Q_i, c_beta, and the scanned Q are NOT referenced by this formulation;
# the layer width is set by Q_e, P_perp, P_tor, tau, and D_norm alone.

using OrdinaryDiffEq

# ---------------------------------------------------------------------
# Pre-computed q-independent constants for the del_s Riccati ODE.
# Mirrors the normalisation block of the Fortran `w_der_del_s` routine:
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
        1.0 / one_plus,
    )
end

# Scalar ODE right-hand side dW/dq for the del_s Riccati (port of the
# Fortran `w_der_del_s` routine). The E, F dispersion coefficients are
# q-dependent and complex (via the `im·Q_hat` terms); everything else is
# cached in `_DelSConsts`.
@inline function _dels_rhs(W::Number, c::_DelSConsts, q::Real)
    q2 = q * q
    q4 = q2 * q2
    E = -(c.Q_hat^2) * c.inv_c -
        im * c.Q_hat * (c.Pperp_hat + c.Ptor_hat) * q2 +
        c.Pperp_hat * c.Ptor_hat * q4
    F = c.Pperp_hat - im * c.Q_hat + c.one_plus * c.Ptor_hat * q2
    return W / q - (W * W) / q + (q * E) / F
end

# Analytic Jacobian dF/dW (port of the Fortran `jac_del_s` routine): the q·E/F
# term is W-independent, leaving d/dW(W/q - W²/q) = 1/q - 2W/q.
@inline _dels_jac(W::Number, c::_DelSConsts, q::Real) = 1.0 / q - 2.0 * W / q

"""
    riccati_del_s(p::SLAYERParameters;
                  q_start=5*p.D_norm, q_min=1e-5,
                  reltol=1e-10, abstol=1e-10, maxiters=50_000,
                  solver=Rodas5P(autodiff=false)) -> ComplexF64

Solve the SLAYER `del_s` inner-layer Riccati ODE and return the
**dimensionless** layer-thickness ratio `δ_s / d_β` at one rational
surface. Ports the Fortran `riccati_del_s` routine.

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
                       solver=Rodas5P(autodiff=false))
    if !(q_start > q_min)
        @debug "riccati_del_s: degenerate integration span" q_start q_min p.D_norm
        return ComplexF64(NaN, NaN)
    end

    c = _build_dels_consts(p)

    # Asymptotic boundary condition at large q. P_hat is
    # the normalised P_perp, i.e. c.Pperp_hat; α and W₀ are real.
    α  = sqrt(c.Pperp_hat * c.inv_c)
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
  - `delta_s_m` -- `|δ_s|`, the resistive layer thickness [m] (primary)
  - `d_beta`    -- β-weighted ion scale `c_β·d_i` [m] (drift reference)

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
end

"""
    slayer_layer_thickness(p::SLAYERParameters; kwargs...) -> LayerWidths

Compute the resistive inner-layer thickness in meters at one rational
surface.

Runs [`riccati_del_s`](@ref) for the dimensionless `δ_s / d_β` and scales
by `p.d_beta` to obtain `δ_s` in meters (matching the Fortran SLAYER
meters-scaling). Keyword arguments are forwarded to `riccati_del_s`.
"""
function slayer_layer_thickness(p::SLAYERParameters; kwargs...)
    dels_db = riccati_del_s(p; kwargs...)
    delta_s = dels_db * p.d_beta
    return LayerWidths(p.ising, p.m, p.n,
                       dels_db, delta_s, abs(delta_s),
                       p.d_beta)
end
