# Riccati.jl
#
# Inner-layer Δ via the Fitzpatrick (`riccati_f`) Riccati ODE. Ports the
# Fortran SLAYER `riccati_f` / `w_der_f` / `jac_f` from delta.f:323-494
# under the simplifying assumptions that have been agreed for this Julia
# port:
#
#   - PeOhmOnly_flag = .TRUE.  (Fortran default; the alternate path is
#     not ported)
#   - parflow_flag   = .FALSE. (Fortran default; the alternate path is
#     not ported)
#   - pe = 0
#
# The complex normalized growth rate `Q = ω + iγ` is passed directly to
# `solve_inner` rather than carried on the parameter struct. All other
# inputs come from `SLAYERParameters` (see `LayerParameters.jl`).
#
# Returns the parity-projected matching data as `SVector{2,ComplexF64}`
# in `(Δ, 0)` form so callers can treat SLAYER and GGJ interchangeably
# through the shared `InnerLayerModel` interface. SLAYER's inner-layer
# dispersion relation produces a single complex Δ, hence the second slot
# is unused.

using OrdinaryDiffEq

# ---------------------------------------------------------------------
# Coefficient evaluation (port of w_der_f, delta.f:461-494).
# Inlined wherever called in the hot ODE RHS.
# ---------------------------------------------------------------------

# Riccati RHS coefficients fA, fA', fB, fC at point p for normalized
# growth rate Q. Returns a 4-tuple of complex numbers.
@inline function _riccati_f_coeffs(p::SLAYERParameters, Q::ComplexF64, x::Real)
    p2    = x * x
    p4    = p2 * p2
    D2    = p.D_norm * p.D_norm
    denom = Q + im * p.Q_e + p2

    fA       = p2 / denom
    fA_prime = (denom - 2 * p2) / denom

    Q_plus_iQi = Q + im * p.Q_i
    fB = Q * Q_plus_iQi +
         Q_plus_iQi * (p.P_perp + p.P_tor) * p2 +
         p.P_perp * p.P_tor * p4

    fC = (Q + im * p.Q_e) +
         (p.P_perp + Q_plus_iQi * D2) * p2 +
         (p.P_tor * D2 / p.iota_e) * p4

    return fA, fA_prime, fB, fC
end

# In-place ODE right-hand side dW/dp for OrdinaryDiffEq.
function _riccati_f_rhs!(dW, W, params, x)
    p, Q = params
    fA, fA_prime, fB, fC = _riccati_f_coeffs(p, Q, x)
    W1 = W[1]
    dW[1] = -(fA_prime / x) * W1 - W1 * W1 / x + (fB / (fA * fC)) * (x * x * x)
    return nothing
end

# Analytic Jacobian (port of jac_f, delta.f:442-455). The full RHS has
# both the explicit (fA'/p, fB·p³) terms and the W² term; for the
# Jacobian only the W-dependent pieces survive.
function _riccati_f_jac!(J, W, params, x)
    p, Q = params
    p2     = x * x
    denom  = Q + im * p.Q_e + p2
    fA_prime = (denom - 2 * p2) / denom
    J[1, 1] = -(fA_prime / x) - 2 * W[1] / x
    return nothing
end

# ---------------------------------------------------------------------
# Boundary-condition selection (port of riccati_f initialisation,
# delta.f:369-400). Two regimes selected by D_norm² vs.
# iota_e·P_perp/P_tor^(2/3).
# ---------------------------------------------------------------------

# Returns (p_start, W_at_p_start, branch) where `branch ∈ (:large_D, :small_D)`.
function _riccati_f_initial(p::SLAYERParameters, Q::ComplexF64;
                             p_floor::Real=6.0)
    D2 = p.D_norm * p.D_norm
    Pperp_over_Ptor23 = p.P_perp / p.P_tor^(2 / 3)

    if D2 > p.iota_e * Pperp_over_Ptor23
        # Large-D_norm branch (delta.f:373-387). Note: in the Fortran
        # expression ((P_tor·D²)/(iota_e·P_tor·P_perp))^(1/4) the
        # P_tor factor cancels — preserved here for traceability.
        p_start = max(((p.P_tor * D2) / (p.iota_e * p.P_tor * p.P_perp))^0.25,
                      p_floor)

        ak = -(Q + im * p.Q_e)
        bk = (p.iota_e * p.P_perp * p.P_tor) / (p.P_tor * D2)
        ck = bk * (1 + (Q + im * p.Q_i) * ((p.P_tor + p.P_perp) /
                                            (p.P_tor * p.P_perp))
                     - (p.P_perp + (Q + im * p.Q_i) * D2) *
                       (p.iota_e / (p.P_tor * D2)))
        sqrt_bk = sqrt(bk)
        xk = (ck - sqrt_bk * (1 - sqrt_bk * ak)) / (2 * sqrt_bk)

        W_bound = xk - sqrt_bk * p_start
        return p_start, W_bound, :large_D
    else
        # Small-D_norm branch (delta.f:389-399).
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
                solver=Rodas5P(autodiff=false)) -> SVector{2,ComplexF64}

Solve the Fitzpatrick SLAYER inner-layer Riccati ODE for the complex
normalized growth rate `Q = ω + iγ`. Returns `SVector(Δ, 0+0im)` so the
result is interface-compatible with `GGJModel.solve_inner` (which
returns a parity-projected pair); SLAYER produces a single Δ, hence the
second slot is zero.

# Algorithm

Ports `riccati_f` (delta.f:323-438) with PeOhmOnly + parflow off and
pe=0. Integrates `dW/dp = -(fA'/p)·W − W²/p + (fB/(fA·fC))·p³` from a
large `p_start` (selected by `_riccati_f_initial` according to whether
`D_norm² ≷ iota_e·P_perp/P_tor^(2/3)`) inward to `pmin`, then computes
`Δ = π / W'(pmin)` from a single RHS evaluation at the inner endpoint.

# Solver

Default `Rodas5P(autodiff=false)` (Rosenbrock, stiff-friendly). The
analytic Jacobian wired via the `ODEFunction(jac=...)` field accelerates
the Newton solves. AD is disabled because complex `Dual` propagation
through the chained denominators incurs allocations in this regime;
finite-difference fallback is fast enough for the 1-equation system.

# Keyword arguments

  - `pmin`     -- inner-layer cutoff (Fortran `xmin = 1e-6`)
  - `p_floor`  -- floor on `p_start` (Fortran `MAX(my_p, 6.0)`)
  - `reltol`,`abstol`,`maxiters` -- LSODE defaults from delta.f:354-363
  - `solver`   -- any OrdinaryDiffEq algorithm; pass `Tsit5()` for the
    non-stiff path (rarely needed for `riccati_f`)
"""
function solve_inner(::SLAYERModel{:fitzpatrick},
                     p::SLAYERParameters, Q::Number;
                     pmin::Real=1e-6,
                     p_floor::Real=6.0,
                     reltol::Real=1e-10,
                     abstol::Real=1e-10,
                     maxiters::Integer=50_000,
                     solver=Rodas5P(autodiff=false))
    Q_c = ComplexF64(Q)

    # Boundary condition at p_start
    p_start, W_bound, _ = _riccati_f_initial(p, Q_c; p_floor=p_floor)

    # Pack params for the closure-free RHS
    rhs_params = (p, Q_c)
    u0 = ComplexF64[W_bound]

    # ODEFunction with analytic Jacobian for the stiff Rosenbrock solver
    f = ODEFunction{true}(_riccati_f_rhs!; jac=_riccati_f_jac!)
    prob = ODEProblem(f, u0, (p_start, pmin), rhs_params)
    sol = solve(prob, solver;
                reltol=reltol, abstol=abstol, maxiters=maxiters,
                save_everystep=false, dense=false)

    sol.retcode == ReturnCode.Success ||
        @warn "SLAYER Riccati integration did not return Success" sol.retcode

    # Δ = π / W'(pmin) — recompute the RHS once at the final endpoint
    W_end = sol.u[end]
    dW_end = similar(W_end)
    _riccati_f_rhs!(dW_end, W_end, rhs_params, pmin)
    Δ = π / dW_end[1]

    return SVector{2,ComplexF64}(Δ, zero(ComplexF64))
end
