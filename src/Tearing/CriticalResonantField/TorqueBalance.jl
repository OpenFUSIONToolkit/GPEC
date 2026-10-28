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
# The viscosity integral is taken as 1/2 (κ̂ = 2/s²). α = S^(-1/3)·(−r_s Δ'_s) is each
# surface's own value (Cole Sec. IV C), which needs a tearing-stable surface (α > 0).
# Q is on Cole's axis: the electron and ion diamagnetic frequencies sit at Q = Q_e
# and Q = Q_i (see `cole_delta`).


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
  - `alpha`     -- outer stability α = S^(-1/3)·(−r_s Δ'_s) (Eq. 61), positive on a tearing-stable surface
"""
struct TorqueBalance{M<:InnerLayerModel,T}
    model::M
    params::T
    Q0::Float64
    prandtl::Float64
    lu::Float64
    kappa_hat::Float64
    alpha::Float64
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
    jxb = -imag(1.0 / (Δ + tb.alpha))
    return 2.0 * tb.prandtl * (tb.Q0 - Q) / jxb, Δ
end

# A maximum counts as a fold only if the balance drops by more than this fraction after it, above sampling noise.
const TORQUE_BALANCE_FOLD_DROP = 1e-3

"""
    fold_search_range(Q0, Q_e, Q_i) -> (Qa, Qb)

The segment the steady state follows as the error field grows: from the natural rotation `Q0`
toward the electron-fluid point `Q_e`, where the electromagnetic torque vanishes, ending at `Q_i`
instead if it lies in between (Fitzpatrick, Nucl. Fusion 33, 1049 (1993), Sec. 7.4).
"""
function fold_search_range(Q0::Real, Q_e::Real, Q_i::Real)
    return (Q0, min(Q0, Q_e) < Q_i < max(Q0, Q_e) ? Q_i : Q_e)
end

"""
    torque_balance_scan(tb::TorqueBalance; Qmin=nothing, Qmax=nothing, n=2000)
        -> (Qs, bal, Qpeak, br_crit, idx_peak, Δs)

Sample `torque_balance_value` on `n` uniform real Q points and return the critical normalized
field `br_crit = b_r/B_φ` (Cole Eq. 62) at the penetration fold: the first maximum of `bal` on
the path from `Q0` along `fold_search_range` after which the balance drops by more than
`TORQUE_BALANCE_FOLD_DROP`, above which the shielded branch followed from `Q0` no longer exists as
the field grows. Points where |Δ| ≤ α, the unshielded zone next to Q_e, are not on the path. With
no resolved fold there is no discrete threshold. `Qmin`/`Qmax` override the
sampled range, but the fold is only sought between `Q0` and the end of `fold_search_range`.
Returns NaN for `Qpeak`, `br_crit` and 0 for `idx_peak`, with a warning, when there is no fold,
when `Q0` is not finite, when the surface is tearing-unstable (α ≤ 0), or when the diamagnetic
ordering Q_i ≤ 0 ≤ Q_e does not hold.
"""
function torque_balance_scan(tb::TorqueBalance; Qmin=nothing, Qmax=nothing, n::Integer=2000)
    p = tb.params
    skip(why) = (@warn "CriticalResonantField: $(p.m)/$(p.n) surface skipped: $why"; (Float64[], Float64[], NaN, NaN, 0, ComplexF64[]))
    isfinite(tb.Q0) || return skip("its natural rotation Q0 = $(tb.Q0) is not finite.")
    tb.alpha > 0 || return skip("it is tearing-unstable or marginal (α = $(tb.alpha)), so there is no penetration threshold.")
    p.Q_i <= 0 <= p.Q_e && p.Q_i < p.Q_e ||
        return skip("the fold search assumes Q_i ≤ 0 ≤ Q_e, but Q_e = $(p.Q_e), Q_i = $(p.Q_i).")
    Qa, Qb = fold_search_range(tb.Q0, p.Q_e, p.Q_i)
    Qs = range(something(Qmin, Qa), something(Qmax, Qb); length=n + 2)[2:(end-1)]   # open at both ends
    out = [torque_balance_value(tb, Q) for Q in Qs]
    bal = first.(out)
    Δs = last.(out)

    # Following the shielded branch from Q0 toward Qb, the threshold is the first saddle-node: the first
    # running maximum of the balance that the path then drops below by more than the fold criterion.
    path = [i for i in sort(eachindex(Qs); by=i -> abs(Qs[i] - Qa))
            if min(Qa, Qb) < Qs[i] < max(Qa, Qb) && isfinite(bal[i]) && bal[i] > 0 && abs(Δs[i]) > tb.alpha]
    idx = 0
    peak = isempty(path) ? 0 : path[1]
    for i in path
        if bal[i] > bal[peak]
            peak = i
        elseif bal[i] < (1 - TORQUE_BALANCE_FOLD_DROP) * bal[peak]
            peak == path[1] || (idx = peak; break)
            peak = i   # a fall from the first sample is not a fold; restart the running maximum
        end
    end
    if idx == 0
        @warn "CriticalResonantField: no penetration fold between Q0 = $(tb.Q0) and Q = $Qb at the $(p.m)/$(p.n) " *
              "surface: the balance never falls more than $(100 * TORQUE_BALANCE_FOLD_DROP) % below a maximum on the " *
              "sampled path, so no discrete threshold is resolved."
        return Qs, bal, NaN, NaN, 0, Δs
    end
    br_crit = sqrt(bal[idx] / (tb.lu * tb.kappa_hat))
    @info @sprintf("CriticalResonantField: %d/%d surface b_r/B_φ crit = %.3e at Q = %.3f",
        p.m, p.n, br_crit, Qs[idx])
    return Qs, bal, Qs[idx], br_crit, idx, Δs
end
