# Uncoupled.jl
#
# Per-surface complex Newton root-finder for the uncoupled tearing dispersion
# relation `r(Q) = 0`. Mirrors the Fortran `coupling_flag = .FALSE.` path
# (slayer.f:301, growthrates.f single-surface branch).
#
# The residual `r(Q)` is supplied as a callable (typically a `SurfaceCoupling`).
# Q is treated as a single complex number; the derivative is approximated by a
# small complex step, and Newton iterates until |r(Q)| falls below `tol` or
# `maxiter` is exhausted. Convergence and final residual are reported via
# `NewtonResult` so callers can decide how to handle non-convergence (typical
# follow-up: retry from a different Q0, or fall back to the AMR/brute-force
# scans in PRs 5/6).

"""
    NewtonResult

Result of a single complex-Newton root-find:

| field         | meaning                                                  |
|---------------|----------------------------------------------------------|
| `Q`           | Final iterate (the root, if `converged == true`)         |
| `residual`    | Residual `r(Q)` at the final iterate                     |
| `iterations`  | Number of Newton steps actually performed                |
| `converged`   | `true` iff `|residual| < tol` or `|step| < step_tol`     |
"""
struct NewtonResult
    Q::ComplexF64
    residual::ComplexF64
    iterations::Int
    converged::Bool
end

"""
    solve_uncoupled(sc::SurfaceCoupling, Q0::Number;
                    tol=1e-6, step_tol=1e-7, stall_iters=3,
                    maxiter=50, h_rel=1e-4, on_failure=:warn)
        -> NewtonResult

Find a complex root `Q` of the per-surface dispersion residual `sc(Q) = 0`
by complex Newton iteration starting from `Q0`. The derivative `r'(Q)` is
estimated by central differences of step size `max(|Q|, 1) * h_rel`.

Convergence is accepted on **any** of three criteria:

  - **residual** -- `|sc(Q)| < tol`
  - **step**     -- `|ΔQ| < step_tol`
  - **stall**    -- `|sc(Q)|` does not decrease for `stall_iters` iterations
    in a row (Newton has hit the ODE-residual noise floor; the current
    iterate is the best available root)

# Keyword arguments

  - `tol`         -- absolute residual tolerance (default `1e-6`)
  - `step_tol`    -- absolute Newton-step tolerance (default `1e-7`)
  - `stall_iters` -- consecutive non-improvements before declaring the
    noise floor reached (default `3`)
  - `maxiter`     -- maximum Newton iterations
  - `h_rel`       -- finite-difference step relative to `max(|Q|, 1)`.
    The default `1e-4` balances truncation error (∝ h²) against amplification
    of the ~1e-3·|Δ| ODE noise (∝ 1/h) when computing `r'`.
  - `on_failure`  -- `:warn` (default), `:error`, or `:silent` action when
    none of the three criteria fire within `maxiter`.
"""
function solve_uncoupled(sc::SurfaceCoupling, Q0::Number;
                         tol::Real=1e-6, step_tol::Real=1e-7,
                         stall_iters::Integer=3,
                         maxiter::Integer=50,
                         h_rel::Real=1e-4, on_failure::Symbol=:warn)
    Q = ComplexF64(Q0)
    f = sc(Q)
    iter = 0
    no_improve = 0
    while iter < maxiter
        if abs(f) < tol
            return NewtonResult(Q, f, iter, true)
        end
        h  = max(abs(Q), 1.0) * h_rel
        df = (sc(Q + h) - sc(Q - h)) / (2h)             # central difference
        if df == 0
            error("solve_uncoupled: zero derivative at Q=$Q (try a different Q0)")
        end
        ΔQ    = f / df
        Q    -= ΔQ
        f_new = sc(Q)
        iter += 1

        if abs(ΔQ) < step_tol
            return NewtonResult(Q, f_new, iter, true)
        end

        # Track stagnation at the ODE noise floor
        if abs(f_new) >= abs(f)
            no_improve += 1
            if no_improve >= stall_iters
                return NewtonResult(Q, f_new, iter, true)
            end
        else
            no_improve = 0
        end
        f = f_new
    end

    converged = abs(f) < tol
    if !converged
        msg = "solve_uncoupled: did not converge in $maxiter iterations " *
              "(|residual|=$(abs(f)), tol=$tol)"
        if on_failure === :warn
            @warn msg Q residual=f
        elseif on_failure === :error
            error(msg)
        elseif on_failure !== :silent
            throw(ArgumentError("solve_uncoupled: on_failure=$on_failure not " *
                                 "in (:warn, :error, :silent)"))
        end
    end
    return NewtonResult(Q, f, iter, converged)
end

"""
    solve_uncoupled(scs::AbstractVector{<:SurfaceCoupling}, Q0;
                    kwargs...) -> Vector{NewtonResult}

Solve the uncoupled dispersion relation surface-by-surface, returning a
`NewtonResult` for each. `Q0` may be a scalar (used for every surface) or a
vector of per-surface starting guesses.
"""
function solve_uncoupled(scs::AbstractVector{<:SurfaceCoupling},
                         Q0::Number; kwargs...)
    return [solve_uncoupled(sc, Q0; kwargs...) for sc in scs]
end

function solve_uncoupled(scs::AbstractVector{<:SurfaceCoupling},
                         Q0s::AbstractVector{<:Number}; kwargs...)
    length(Q0s) == length(scs) ||
        throw(ArgumentError("solve_uncoupled: length(Q0s) ≠ length(scs)"))
    return [solve_uncoupled(sc, Q0; kwargs...) for (sc, Q0) in zip(scs, Q0s)]
end
