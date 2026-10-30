# Riccati.jl
#
# Inner-layer Δ from the Fitzpatrick two-fluid drift-MHD layer Riccati ODE (Fitzpatrick, Tearing Mode Dynamics in
# Tokamak Plasmas, IOP 2023; Park 2022, Phys. Plasmas 29, 072506; Burgess et al. 2026) for the pressureless tearing
# channel (no parallel flow, pe = 0), keeping the full C(p) form in every regime.
#
# The layer is evaluated at Fitzpatrick's normalized eigenvalue ĝ = (γ − iω)·τ_k: `solve_inner` takes the
# growth rate s = γ − iω (ĝ = s·τ_k) and `layer_response` the dispersion scans' Q = (ω + iγ)·τ_k (ĝ = −i·Q).
# Only the `tearing` channel of the returned `InnerLayerResponse` is populated; the interchange channel is zero.

using OrdinaryDiffEq

# ---------------------------------------------------------------------
# Coefficients of the two-fluid layer A, B, C terms (Fitzpatrick 2023).
# ---------------------------------------------------------------------

# x-independent constants of the Riccati ODE, built once per solve from (p, ĝ) and passed as the integrator `params`.
struct _RiccatiConsts
    g_minus_iQe::ComplexF64   # constant part of denom = g − iQe + x²
    A::ComplexF64             # g·(g − iQi)               — fB constant term
    B::ComplexF64             # (g − iQi)·(P_perp + P_tor) — fB · x² coefficient
    C::Float64                # P_perp · P_tor            — fB · x⁴ coefficient
    E::ComplexF64             # (g − iQi) · D² + P_perp   — fC · x² coefficient
    G::Float64                # P_tor · D² / iota_e       — fC · x⁴ coefficient
end

@inline function _build_riccati_consts(p::SLAYERParameters, g::ComplexF64)
    g_minus_iQe = g - im * p.Q_e
    g_minus_iQi = g - im * p.Q_i
    D2 = p.D_norm * p.D_norm
    return _RiccatiConsts(
        g_minus_iQe,
        g * g_minus_iQi,                                  # A
        g_minus_iQi * (p.P_perp + p.P_tor),               # B
        p.P_perp * p.P_tor,                               # C
        p.P_perp + g_minus_iQi * D2,                      # E
        p.P_tor * D2 / p.iota_e                          # G
    )
end

# Riccati RHS coefficients fA, fA', fB, fC at x.
@inline function _riccati_f_coeffs(c::_RiccatiConsts, x::Real)
    p2 = x * x
    p4 = p2 * p2
    denom = c.g_minus_iQe + p2

    fA = p2 / denom
    # (denom − 2p²)/denom, not the equal 1 − 2·fA: the identity drifts ~3e-3 relative near marginal stability.
    fA_prime = (denom - 2 * p2) / denom

    fB = c.A + c.B * p2 + c.C * p4
    fC = c.g_minus_iQe + c.E * p2 + c.G * p4

    return fA, fA_prime, fB, fC
end

# Scalar ODE right-hand side dW/dp; a scalar `ComplexF64` state keeps the integrator allocation-free.
@inline function _riccati_f_rhs(W::Number, consts::_RiccatiConsts, x::Real)
    fA, fA_prime, fB, fC = _riccati_f_coeffs(consts, x)
    return -(fA_prime / x) * W - W * W / x + (fB / (fA * fC)) * (x * x * x)
end

# Analytic 1×1 Jacobian of the Riccati RHS (only the W-dependent terms survive).
@inline function _riccati_f_jac(W::Number, consts::_RiccatiConsts, x::Real)
    p2 = x * x
    denom = consts.g_minus_iQe + p2
    fA_prime = (denom - 2 * p2) / denom
    return -(fA_prime / x) - 2 * W / x
end

# ---------------------------------------------------------------------
# Boundary-condition selection (large-p asymptotics, Fitzpatrick 2023): regime set by D_norm² vs iota_e·P_perp/P_tor^(2/3).
# ---------------------------------------------------------------------

# Returns (p_start, W_at_p_start, branch) where `branch ∈ (:large_D, :small_D)`.
function _riccati_f_initial(p::SLAYERParameters, g::ComplexF64;
    p_floor::Real=6.0)
    D2 = p.D_norm * p.D_norm
    Pperp_over_Ptor23 = p.P_perp / p.P_tor^(2 / 3)

    if D2 > p.iota_e * Pperp_over_Ptor23
        # Large-D_norm branch. In ((P_tor·D²)/(iota_e·P_tor·P_perp))^(1/4)
        # the P_tor factor cancels — written out for traceability.
        p_start = max(((p.P_tor * D2) / (p.iota_e * p.P_tor * p.P_perp))^0.25,
            p_floor)

        ak = -(g - im * p.Q_e)
        bk = (p.iota_e * p.P_perp * p.P_tor) / (p.P_tor * D2)
        ck = bk * (1 + (g - im * p.Q_i) * ((p.P_tor + p.P_perp) /
                                           (p.P_tor * p.P_perp))
                   -
                   (p.P_perp + (g - im * p.Q_i) * D2) *
                   (p.iota_e / (p.P_tor * D2)))
        sqrt_bk = sqrt(bk)
        xk = (ck - sqrt_bk * (1 - sqrt_bk * ak)) / (2 * sqrt_bk)

        W_bound = xk - sqrt_bk * p_start
        return p_start, W_bound, :large_D
    else
        # Small-D_norm branch.
        p_start = max(1.0 / p.P_tor^(1 / 6), p_floor)

        ak = -(g - im * p.Q_e)
        bk = ComplexF64(p.P_tor)        # promoted to ComplexF64 for sqrt below
        ck = im * (p.Q_e - p.Q_i) * (p.P_tor / p.P_perp) + (g - im * p.Q_i)
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
    solve_inner(::SLAYERModel{:fitzpatrick}, p::SLAYERParameters, s::Number;
                pmin=1e-6, p_floor=6.0, reltol=1e-10, abstol=1e-10, maxiters=50_000,
                solver=Rodas5P(autodiff=false)) -> InnerLayerResponse

Solve the Fitzpatrick SLAYER inner-layer Riccati ODE at the complex growth rate `s = γ − iω` in 1/s, in the
surface's E×B frame. The layer is evaluated at Fitzpatrick's normalized eigenvalue ĝ = s·τ_k (Fitzpatrick 2023:
γ = Re(ĝ)/τ_k, ω = −Im(ĝ)/τ_k), and the diamagnetic drifts enter as ĝ − i·Q_e and ĝ − i·Q_i, so the layer co-rotates
with the electron fluid at ω = ω_*e. Returns an `InnerLayerResponse` with `tearing = Δ` and `interchange = 0`: SLAYER's
pressureless layer produces only the tearing channel.

Integrates `dW/dp = -(fA'/p)·W − W²/p + (fB/(fA·fC))·p³` (Fitzpatrick 2023; Park 2022) from a large `p_start`
inward to `pmin` and returns `Δ = π / W'(pmin)`.

**Solver swaps:** tiny differences between ODE solvers change the outer AMR's cell flagging and hence the
classified root/pole inventory, so validate a new `solver` against `n_valid_roots` and `n_poles`, not just γ.

# Keyword arguments

  - `pmin`     -- inner-layer cutoff (default 1e-6)
  - `p_floor`  -- floor on `p_start` (default 6.0)
  - `reltol`,`abstol`,`maxiters` -- stiff-solver tolerance/iteration limits
  - `solver`   -- any OrdinaryDiffEq algorithm; pass `Tsit5()` for the
    non-stiff path (rarely needed here)
"""
solve_inner(::SLAYERModel{:fitzpatrick}, p::SLAYERParameters, s::Number; kwargs...) = _slayer_delta(p, ComplexF64(s) * p.tauk; kwargs...)

# The dispersion scans run in the plane Q = (ω + iγ)·τ_k (Park et al. 2022), a −90° rotation of ĝ: ĝ = −i·Q.
layer_response(::SLAYERModel{:fitzpatrick}, p::SLAYERParameters, Q::Number; kwargs...) =
    _slayer_delta(p, ComplexF64(imag(Q), -real(Q)); kwargs...)

# Δ of the layer at the normalized eigenvalue ĝ.
function _slayer_delta(p::SLAYERParameters, g::ComplexF64;
    pmin::Real=1e-6,
    p_floor::Real=6.0,
    reltol::Real=1e-10,
    abstol::Real=1e-10,
    maxiters::Integer=50_000,
    solver=Rodas5P(; autodiff=false))

    p_start, W_bound, _ = _riccati_f_initial(p, g; p_floor=p_floor)

    rhs_params = _build_riccati_consts(p, g)

    # Scalar state with the out-of-place form (`ODEFunction{false}`) avoids per-step allocations.
    u0 = ComplexF64(W_bound)
    f = ODEFunction{false}(_riccati_f_rhs; jac=_riccati_f_jac)
    prob = ODEProblem(f, u0, (p_start, pmin), rhs_params)
    sol = solve(prob, solver;
        reltol=reltol, abstol=abstol, maxiters=maxiters,
        save_everystep=false, dense=false)

    if sol.retcode != ReturnCode.Success
        # NaN sentinel so the scan/AMR isfinite checks flag this cell; @debug since failures cluster near poles.
        @debug "SLAYER Riccati integration did not return Success" sol.retcode
        return InnerLayerResponse(ComplexF64(NaN, NaN), zero(ComplexF64))
    end

    # Δ = π / W'(pmin) from one RHS evaluation at the inner endpoint.
    W_end = sol.u[end]
    dW_end = _riccati_f_rhs(W_end, rhs_params, pmin)
    Δ::ComplexF64 = π / dW_end

    # The pressureless layer has no interchange (Δ₊) channel.
    return InnerLayerResponse(Δ, zero(ComplexF64))
end
