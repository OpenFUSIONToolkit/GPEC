# TorqueBalance.jl
#
# Error-field penetration threshold from the steady-state torque balance at a
# rational surface (Cole and Fitzpatrick, Phys. Plasmas 13, 032503 (2006), Sec. IV).
# The electromagnetic and viscous torques balance when (Cole Eq. 61)
#
#   Im[Δ(Q)] / |α + Δ(Q)|² = 2·P·(Q0 − Q) / (S·κ̂·(b_r/B_φ)²),
#
# with κ̂ = [2/s(r_s)]² ∫_{r_s}^{a} [μ(r_s)/μ(r)] dr/r. Balance is lost when no Q
# satisfies this, so the critical field is (Cole Eq. 62)
#
#   (b_r/B_φ)²_crit = max_Q 2·P·(Q0 − Q) / (S·κ̂·Im[−1/(α + Δ(Q))]).
#
# The viscosity integral is taken as 1/2 (κ̂ = 2/s²) and α is kept finite at 1e-2.
# Q is on Cole's axis: the electron and ion diamagnetic frequencies sit at Q = Q_e
# and Q = Q_i (see `cole_delta`).

# Finite stand-in for α = S^(-1/3)·(−r_s Δ'_s) ≪ 1; b_crit moves ~4% over 1e-4 to 1e-1.
const TORQUE_BALANCE_ALPHA = 1e-2
# Default half-width around Q_e, Q_i within which a torque-balance maximum is treated as pole-dominated.
const TORQUE_BALANCE_POLE_WIDTH = 0.02

"""
    TorqueBalance{M<:InnerLayerModel,T}

Torque-balance inputs at one rational surface.

## Fields

  - `model`     -- inner-layer model passed to `solve_inner`
  - `params`    -- that model's layer parameters at the surface; must carry `Q_e`, `Q_i`, `m`, `n`
  - `Q0`        -- normalized natural E×B rotation of the m/n mode, τ_k·n·ω_E
  - `prandtl`   -- magnetic Prandtl number τ_R/τ_V (the layer's P_φ)
  - `lu`        -- Lundquist number S
  - `kappa_hat` -- Cole's slab-to-tokamak factor κ̂ (Eq. 61)
"""
struct TorqueBalance{M<:InnerLayerModel,T}
    model::M
    params::T
    Q0::Float64
    prandtl::Float64
    lu::Float64
    kappa_hat::Float64
end

"""
    cole_delta(model, params, Q::Real) -> ComplexF64

Inner-layer Δ at real frequency `Q` on Cole's axis. `solve_inner` evaluates the layer
at `i·conj(Q)`, which mirrors the real axis, so Cole's Δ(Q) is `conj(Δ_solve_inner(−Q))`.
"""
cole_delta(model::InnerLayerModel, params, Q::Real) =
    conj(solve_inner(model, params, ComplexF64(-Q)).tearing)

"""
    torque_balance_value(tb::TorqueBalance, Q::Real) -> (bal, Δ)

Right-hand side of Cole Eq. 62 before the maximum, `bal = 2·P·(Q0 − Q) / Im[−1/(α + Δ)]`,
and the layer `Δ(Q)` it used.
"""
function torque_balance_value(tb::TorqueBalance, Q::Real)
    Δ = cole_delta(tb.model, tb.params, Q)
    jxb = -imag(1.0 / (Δ + TORQUE_BALANCE_ALPHA))
    return 2.0 * tb.prandtl * (tb.Q0 - Q) / jxb, Δ
end

"""
    torque_balance_window(Q0, Q_e, Q_i) -> (Qmin, Qmax)

Q range bracketing the torque-balance branch between the natural rotation `Q0` and the
electron diamagnetic pole `Q_e`. It assumes `Q_i < 0 < Q_e`, which `torque_balance_scan` checks.
"""
function torque_balance_window(Q0::Real, Q_e::Real, Q_i::Real)
    Q0 > Q_e && return (1.05 * Q_e, 2.0 * Q0)
    Qmin = Q0 > 0 ? 0.8 * Q_i : 1.5 * min(Q0, Q_i)
    return (Qmin, 0.95 * Q_e)
end

"""
    torque_balance_scan(tb::TorqueBalance; Qmin=nothing, Qmax=nothing, n=2000,
                        pole_width=TORQUE_BALANCE_POLE_WIDTH)
        -> (Qs, bal, Qpeak, br_crit, idx_peak, Δs)

Sample `torque_balance_value` on `n` uniform real Q points and return the critical
normalized field `br_crit = b_r/B_φ` (Cole Eq. 62) at the largest positive interior local
maximum of `bal`, located at `Qpeak = Qs[idx_peak]`. A maximum within `pole_width` (or one
grid step, if larger) of the diamagnetic poles `Q_e`, `Q_i` is discarded as a pole artifact.
`Qmin`/`Qmax` default to `torque_balance_window`. Returns NaN for `Qpeak`, `br_crit` and 0 for
`idx_peak`, with a warning, when no valid maximum is found, when `Q0` is not finite, or when
the diamagnetic ordering `Q_i < 0 < Q_e` that the window assumes does not hold.
"""
function torque_balance_scan(tb::TorqueBalance; Qmin=nothing, Qmax=nothing, n::Integer=2000,
    pole_width::Real=TORQUE_BALANCE_POLE_WIDTH)
    p = tb.params
    skip(why) = (@warn "CriticalResonantField: $(p.m)/$(p.n) surface skipped: $why"; (Float64[], Float64[], NaN, NaN, 0, ComplexF64[]))
    isfinite(tb.Q0) || return skip("its natural rotation Q0 = $(tb.Q0) is not finite.")
    p.Q_i < 0 < p.Q_e || return skip("the scan window assumes Q_i < 0 < Q_e, but Q_e = $(p.Q_e), Q_i = $(p.Q_i).")
    wmin, wmax = torque_balance_window(tb.Q0, p.Q_e, p.Q_i)
    Qs = range(something(Qmin, wmin), something(Qmax, wmax); length=n)
    out = [torque_balance_value(tb, Q) for Q in Qs]
    bal = first.(out)
    Δs = last.(out)

    w = max(step(Qs), pole_width)
    near_pole(Q) = abs(Q - p.Q_e) <= w || abs(Q - p.Q_i) <= w
    peaks = [i for i in 2:(n-1) if isfinite(bal[i]) && bal[i] > 0 &&
             bal[i-1] < bal[i] && bal[i] >= bal[i+1] && !near_pole(Qs[i])]
    if isempty(peaks)
        @warn "CriticalResonantField: no positive torque-balance maximum on " *
              "Q ∈ [$(first(Qs)), $(last(Qs))] at the $(p.m)/$(p.n) surface."
        return Qs, bal, NaN, NaN, 0, Δs
    end
    idx = peaks[argmax(bal[peaks])]
    br_crit = sqrt(bal[idx] / (tb.lu * tb.kappa_hat))
    @info @sprintf("CriticalResonantField: %d/%d surface b_r/B_φ crit = %.3e at Q = %.3f",
        p.m, p.n, br_crit, Qs[idx])
    return Qs, bal, Qs[idx], br_crit, idx, Δs
end
