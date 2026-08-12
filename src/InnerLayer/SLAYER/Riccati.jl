# Riccati.jl
#
# Inner-layer Δ via the Fitzpatrick two-fluid drift-MHD layer Riccati ODE
# (Fitzpatrick, Tearing Mode Dynamics in Tokamak Plasmas, IOP 2023; Park
# et al. 2022, Phys. Plasmas 29, 122505; Burgess et al. 2026), for the
# pressureless tearing channel (no parallel flow, pe = 0). The A, B, C
# coefficients, the dW/dp Riccati equation, the large-p asymptotic boundary
# conditions, the regime test D² > ι_e P_⊥/P_tor^(2/3), the small-p
# Δ = π/W' extraction, and the analytic Jacobian all follow that layer
# formulation; the full C(p) form is retained in every regime (no low-D
# reduction).
#
# The complex normalized growth rate `Q = ω + iγ` is passed directly to
# `solve_inner` rather than carried on the parameter struct. All other
# inputs come from `SLAYERParameters` (see `LayerParameters.jl`).
#
# Returns the parity-projected matching data as an `InnerLayerResponse`
# with only the `tearing` channel populated so callers can treat SLAYER and
# GGJ interchangeably through the shared `InnerLayerModel` interface.
# SLAYER's inner-layer dispersion relation produces a single complex Δ
# (the tearing channel); the interchange channel is zero.

using OrdinaryDiffEq

# ---------------------------------------------------------------------
# Coefficient evaluation for the two-fluid layer A, B, C terms (Fitzpatrick 2023).
#
# All x-independent quantities are bundled in `_RiccatiConsts` and computed
# once per `solve_inner` call (see line ~200). The hot RHS / Jacobian
# evaluations then access only the bundled constants and `x`, avoiding the
# tens of thousands of redundant complex muls/adds the prior code did.
# ---------------------------------------------------------------------

# Pre-computed x-independent constants for the Fitzpatrick Riccati ODE.
# Derived from `(p::SLAYERParameters, Q::ComplexF64)` once per solve. Used as
# the integrator `params` so `_riccati_f_rhs` and `_riccati_f_jac` only need
# the x-dependent algebra.
struct _RiccatiConsts
    Q_plus_iQe::ComplexF64    # constant part of denom = Q + iQe + x²
    A::ComplexF64             # Q·(Q + iQi)               — fB constant term
    B::ComplexF64             # (Q + iQi)·(P_perp + P_tor) — fB · x² coefficient
    C::Float64                # P_perp · P_tor            — fB · x⁴ coefficient
    E::ComplexF64             # (Q + iQi) · D² + P_perp   — fC · x² coefficient
    G::Float64                # P_tor · D² / iota_e       — fC · x⁴ coefficient
end

@inline function _build_riccati_consts(p::SLAYERParameters, Q::ComplexF64)
    Q_plus_iQe = Q + im * p.Q_e
    Q_plus_iQi = Q + im * p.Q_i
    D2 = p.D_norm * p.D_norm
    return _RiccatiConsts(
        Q_plus_iQe,
        Q * Q_plus_iQi,                                   # A
        Q_plus_iQi * (p.P_perp + p.P_tor),                # B
        p.P_perp * p.P_tor,                               # C
        p.P_perp + Q_plus_iQi * D2,                       # E
        p.P_tor * D2 / p.iota_e                          # G
    )
end

# Riccati RHS coefficients fA, fA', fB, fC at point x. Receives the
# pre-built `_RiccatiConsts` so each call costs only a handful of muls/adds
# plus one complex division (the fA = p²/denom).
@inline function _riccati_f_coeffs(c::_RiccatiConsts, x::Real)
    p2 = x * x
    p4 = p2 * p2
    denom = c.Q_plus_iQe + p2

    fA = p2 / denom
    # Use the numerator-subtracts-twice-p² form rather than the algebraic
    # identity 1 − 2·fA. The two are mathematically equal, but the
    # integrator's adaptive stepping near marginal stability compounds
    # ULP-level differences in fA' over thousands of steps; this form keeps
    # the result tighter than the identity, which drifts to ~3e-3 relative
    # (within abs-tolerance, but tighter is better).
    fA_prime = (denom - 2 * p2) / denom

    fB = c.A + c.B * p2 + c.C * p4
    fC = c.Q_plus_iQe + c.E * p2 + c.G * p4

    return fA, fA_prime, fB, fC
end

# Scalar ODE right-hand side dW/dp for OrdinaryDiffEq.
#
# This is a 1-equation ODE — modeling W(x) as a `ComplexF64` scalar (rather
# than a 1-element `Vector{ComplexF64}`) lets the integrator's stage updates
# stay on the stack with no per-step allocations. SDIRK + Rosenbrock + BDF
# methods in OrdinaryDiffEq all support scalar `u`.
@inline function _riccati_f_rhs(W::Number, consts::_RiccatiConsts, x::Real)
    fA, fA_prime, fB, fC = _riccati_f_coeffs(consts, x)
    return -(fA_prime / x) * W - W * W / x + (fB / (fA * fC)) * (x * x * x)
end

# Analytic Jacobian of the Riccati RHS. The full RHS has
# both the explicit (fA'/p, fB·p³) terms and the W² term; for the
# Jacobian only the W-dependent pieces survive. Returns a scalar — the
# 1×1 Jacobian of the scalar ODE.
@inline function _riccati_f_jac(W::Number, consts::_RiccatiConsts, x::Real)
    p2 = x * x
    denom = consts.Q_plus_iQe + p2
    fA_prime = (denom - 2 * p2) / denom
    return -(fA_prime / x) - 2 * W / x
end

# ---------------------------------------------------------------------
# Boundary-condition selection (large-p asymptotics, Fitzpatrick 2023).
# Two regimes selected by D_norm² vs.
# iota_e·P_perp/P_tor^(2/3).
# ---------------------------------------------------------------------

# Returns (p_start, W_at_p_start, branch) where `branch ∈ (:large_D, :small_D)`.
function _riccati_f_initial(p::SLAYERParameters, Q::ComplexF64;
    p_floor::Real=6.0)
    D2 = p.D_norm * p.D_norm
    Pperp_over_Ptor23 = p.P_perp / p.P_tor^(2 / 3)

    if D2 > p.iota_e * Pperp_over_Ptor23
        # Large-D_norm branch. In ((P_tor·D²)/(iota_e·P_tor·P_perp))^(1/4)
        # the P_tor factor cancels — written out for traceability.
        p_start = max(((p.P_tor * D2) / (p.iota_e * p.P_tor * p.P_perp))^0.25,
            p_floor)

        ak = -(Q + im * p.Q_e)
        bk = (p.iota_e * p.P_perp * p.P_tor) / (p.P_tor * D2)
        ck = bk * (1 + (Q + im * p.Q_i) * ((p.P_tor + p.P_perp) /
                                           (p.P_tor * p.P_perp))
                   -
                   (p.P_perp + (Q + im * p.Q_i) * D2) *
                   (p.iota_e / (p.P_tor * D2)))
        sqrt_bk = sqrt(bk)
        xk = (ck - sqrt_bk * (1 - sqrt_bk * ak)) / (2 * sqrt_bk)

        W_bound = xk - sqrt_bk * p_start
        return p_start, W_bound, :large_D
    else
        # Small-D_norm branch.
        p_start = max(1.0 / p.P_tor^(1 / 6), p_floor)

        ak = -(Q + im * p.Q_e)
        bk = ComplexF64(p.P_tor)        # promoted to ComplexF64 for sqrt below
        ck = -im * (p.Q_e - p.Q_i) * (p.P_tor / p.P_perp) + (Q + im * p.Q_i)
        sqrt_bk = sqrt(bk)
        xk = (ak * bk - ck) / (2 * sqrt_bk)

        W_bound = -1.0 + xk * p_start - sqrt_bk * p_start^3
        return p_start, W_bound, :small_D
    end
end

# ---------------------------------------------------------------------
# solve_inner dispatch for SLAYERModel{:fitzpatrick}.
# ---------------------------------------------------------------------

"""
    solve_inner(::SLAYERModel{:fitzpatrick},
                p::SLAYERParameters, Q::Number;
                pmin=1e-6, p_floor=6.0,
                reltol=1e-10, abstol=1e-10,
                maxiters=50_000,
                solver=Rodas5P(autodiff=false)) -> InnerLayerResponse

Solve the Fitzpatrick SLAYER inner-layer Riccati ODE for the complex
normalized growth rate `Q = ω + iγ`. Returns an `InnerLayerResponse` with
`tearing = Δ` and `interchange = 0`, so the result is interface-compatible
with `GGJModel.solve_inner` (which populates both channels); SLAYER's
pressureless layer produces only the tearing channel.

# Algorithm

Implements the Fitzpatrick two-fluid drift-MHD layer formulation
(Fitzpatrick 2023; Park et al. 2022) for the pressureless tearing channel
(no parallel flow, pe = 0). Integrates
`dW/dp = -(fA'/p)·W − W²/p + (fB/(fA·fC))·p³` from a large `p_start`
(selected by `_riccati_f_initial` according to whether
`D_norm² ≷ iota_e·P_perp/P_tor^(2/3)`) inward to `pmin`, then computes
`Δ = π / W'(pmin)` from a single RHS evaluation at the inner endpoint.

# Solver

Default `Rodas5P(autodiff=false)` (Rosenbrock, stiff-friendly). The
analytic Jacobian wired via the `ODEFunction(jac=...)` field accelerates
the Newton solves. AD is disabled because complex `Dual` propagation
through the chained denominators incurs allocations in this regime;
finite-difference fallback is fast enough for the 1-equation system.

**Note on solver swaps:** sub-percent floating-point differences between
ODE solvers cascade through the outer AMR's cell-flagging decisions
(`ContourSearchAMR.jl::_crosses_zero`) and produce **structurally
different** AMR cell trees. Swapping the ODE solver has been observed to
change the classified root/pole inventory substantially even when the
most-unstable γ agrees to within ~1e-5 relative. So solver choice is not
just a per-call optimization — it affects the downstream root/pole
inventory. Future solver swaps need to be validated against the topology
fields (`n_valid_roots`, `n_poles`), not just γ.

# Keyword arguments

  - `pmin`     -- inner-layer cutoff (default 1e-6)
  - `p_floor`  -- floor on `p_start` (default 6.0)
  - `reltol`,`abstol`,`maxiters` -- stiff-solver tolerance/iteration limits
  - `solver`   -- any OrdinaryDiffEq algorithm; pass `Tsit5()` for the
    non-stiff path (rarely needed here)
"""
function solve_inner(::SLAYERModel{:fitzpatrick},
    p::SLAYERParameters, Q::Number;
    pmin::Real=1e-6,
    p_floor::Real=6.0,
    reltol::Real=1e-10,
    abstol::Real=1e-10,
    maxiters::Integer=50_000,
    solver=Rodas5P(; autodiff=false))
    # Convention bridge between our scan plane and the layer eigenvalue.
    # We scan in the Park plane `Q = ω + iγ` (+γ on the +imag axis; Park
    # et al. 2022). Fitzpatrick's normalized layer eigenvalue is
    # `ĝ = (γ − iω)·τ_k` in the co-rotating frame (Fitzpatrick 2023:
    # γ = Re(ĝ)/τ_k, ω = −Im(ĝ)/τ_k), so +γ sits on his +real axis. The two
    # planes differ by a pure −90° rotation, `ĝ = −i·Q` — relabeling which
    # axis is growth, no physics.
    #
    # The layer ODE coefficients are analytic in ĝ with Q_e, Q_i, D, P_⊥,
    # P_φ all real, so Schwarz reflection gives `Δ̂_s(conj ĝ) = conj(Δ̂_s(ĝ))`.
    # We feed `Q_c = i·conj(Q) = conj(−i·Q) = conj(ĝ)`, hence evaluate
    # `conj(Δ̂_s(ĝ))`. Conjugation maps zeros to zeros, so the extracted
    # growth-rate roots are identical to those of Δ̂_s(ĝ); it only reflects
    # the off-root residual surface, fixing the integration-path orientation
    # so |Δ| contours are consistent across the scan plane.
    Q_c = im * conj(ComplexF64(Q))

    # Boundary condition at p_start
    p_start, W_bound, _ = _riccati_f_initial(p, Q_c; p_floor=p_floor)

    # Pre-compute x-independent constants ONCE; the integrator threads this
    # through to every RHS / Jacobian call instead of recomputing per-step.
    rhs_params = _build_riccati_consts(p, Q_c)

    # Scalar `u0`: the ODE state is a single `ComplexF64`, not a 1-element
    # vector. OrdinaryDiffEq supports scalar problems via the out-of-place
    # form (`ODEFunction{false}`). This eliminates the per-step heap-
    # allocation of intermediate `dW` vectors that the in-place form
    # incurred for every stage of every accepted/rejected step.
    u0 = ComplexF64(W_bound)
    f = ODEFunction{false}(_riccati_f_rhs; jac=_riccati_f_jac)
    prob = ODEProblem(f, u0, (p_start, pmin), rhs_params)
    sol = solve(prob, solver;
        reltol=reltol, abstol=abstol, maxiters=maxiters,
        save_everystep=false, dense=false)

    if sol.retcode != ReturnCode.Success
        # Unconverged solve: return a NaN sentinel so the dispersion scan / AMR
        # flags this Q-cell (via its isfinite checks) rather than ingesting a
        # bogus finite Δ built from an unconverged W_end. @debug not @warn: in a
        # dense Q-plane scan failures cluster near poles and would flood the log.
        @debug "SLAYER Riccati integration did not return Success" sol.retcode
        return InnerLayerResponse(ComplexF64(NaN, NaN), zero(ComplexF64))
    end

    # Δ = π / W'(pmin) — single RHS evaluation at the inner endpoint
    W_end = sol.u[end]
    dW_end = _riccati_f_rhs(W_end, rhs_params, pmin)
    Δ::ComplexF64 = π / dW_end

    # Fitzpatrick / pressureless SLAYER has no interchange channel
    # (the Δ_− / even-parity matching quantity is identically zero in
    # the pressureless limit), so populate only the tearing field.
    return InnerLayerResponse(Δ, zero(ComplexF64))
end
