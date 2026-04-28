#!/usr/bin/env julia
# test_riccati_solver_convergence.jl — Sweep ODE solvers across the SLAYER
# linear-tearing growth-rate regimes to identify which converge robustly,
# at what cost.
#
# Parameter grid (per the SLAYER inner-layer normalization):
#   D       12 log-spaced points in [0.1, 5]
#                — covers TJ q=3 (D=0.18), TJ q=2 (D=0.63), DIII-D (D ~ 0.1-2)
#   Q_*/D⁴  6 linear points in [0, 2]
#                — Q_* = 2|Q_e| = 2|Q_i|; Q_e = Q_i = (qr × D⁴) / 2
#   P/D⁶    6 linear points in [0, 4]
#                — P = P_tor = P_perp = pr × D⁶
#   Q       4 representative complex points (typical / small / larger / pure-iγ)
#   x0      3 starting-point factors {0.5, 1.0, 1.5} × x0_natural
#
# Skip rules:
#   - P=0 (boundary `P_tor^(1/6)` floor in `_riccati_f_initial`)
#   - Q_* > Q_STAR_CAP (default 500) — extreme diamagnetic regime
#   - P > P_CAP (default 2000)        — extreme pressure regime
#   These caps prevent the high-D corner of the grid from running expensive
#   solves at unphysically large coefficients.
#
# Convergence: a combo "converges" if the 3 Δ values across x0 factors agree
# to relative spread < threshold. Three thresholds reported:
#   tight  1e-5 — catches solver-precision regressions
#   medium 1e-4 — between tight and loose
#   loose  1e-3 — catches catastrophic failures only
# At smallest x0 the asymptotic BC truncation error is O(1/x_start²) or
# O(1/x_start⁴), so tight may fail on BC noise (not solver noise) at small
# x0 ratios — in that case ALL solvers fail similarly on the same combos.
#
# For each solver, reports:
#   - convergence rate at each threshold
#   - median + p95 walltime per solve
#   - mean integrator step count
#
# Usage:
#   julia --project=. profiling/test_riccati_solver_convergence.jl \
#       [--solvers Rodas5P,Rodas4,KenCarp4,QNDF,...] \
#       [--coarse]                 # quick smoke (3 D × 2 qr × 2 pr × 1 Q)
#       [--Qstar-cap 500]          # cap |Q_*| (default 500)
#       [--P-cap 2000]             # cap |P|   (default 2000)
#       [--out /tmp/riccati_solver_test.tsv]
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using GeneralizedPerturbedEquilibrium
using GeneralizedPerturbedEquilibrium.Tearing.InnerLayer.SLAYER:
    SLAYERParameters, SLAYERModel
using OrdinaryDiffEq
using LinearAlgebra, Printf, Statistics

# Pull the private Riccati helpers via internal accessors. They live in the
# SLAYER module — we import them by qualified name for the test only.
const RC = GeneralizedPerturbedEquilibrium.Tearing.InnerLayer.SLAYER
const _riccati_f_rhs        = getfield(RC, :_riccati_f_rhs)
const _riccati_f_jac        = getfield(RC, :_riccati_f_jac)
const _riccati_f_initial    = getfield(RC, :_riccati_f_initial)
const _build_riccati_consts = getfield(RC, :_build_riccati_consts)

# CLI ---------------------------------------------------------------------
function get_arg(args, name, default=nothing; parser=identity)
    for (i, a) in enumerate(args)
        a == "--$name" && return parser(args[i+1])
    end
    return default
end
args = ARGS

solvers_str = get_arg(args, "solvers", "Rodas5P,Rodas4,Rodas3,KenCarp4,TRBDF2,QNDF,FBDF")
out_path    = get_arg(args, "out", "/tmp/riccati_solver_test.tsv")
Qstar_cap   = get_arg(args, "Qstar-cap", 500.0; parser=x->parse(Float64, x))
P_cap       = get_arg(args, "P-cap",     2000.0; parser=x->parse(Float64, x))
const COARSE_MODE = "--coarse" in args

solver_names = String.(strip.(split(solvers_str, ',')))
solver_factory = Dict(
    "Rodas5P"  => () -> Rodas5P(autodiff=false),
    "Rodas4"   => () -> Rodas4(autodiff=false),
    "Rodas3"   => () -> Rodas3(autodiff=false),
    "KenCarp4" => () -> KenCarp4(autodiff=false),
    "TRBDF2"   => () -> TRBDF2(autodiff=false),
    "QNDF"     => () -> QNDF(autodiff=false),
    "FBDF"     => () -> FBDF(autodiff=false),
)

# Parameter grid ----------------------------------------------------------
# D log-spaced over [0.1, 5] — covers TJ q=3 (D=0.18), TJ q=2 (D=0.63),
# DIII-D surfaces (D ~ 0.1-2) AND the original D ∈ [0.5, 5] regime.
D_grid = COARSE_MODE ? [0.18, 0.63, 2.0] :
                       round.(exp.(range(log(0.1), log(5.0), length=12)), digits=4)
Qstar_ratio = COARSE_MODE ? [0.0, 1.0] : collect(range(0.0, 2.0, length=6))
P_ratio     = COARSE_MODE ? [0.0, 2.0] : collect(range(0.0, 4.0, length=6))

# Q sweep: 4 representative complex points covering small/large/typical/pure-iγ.
Q_test_grid = COARSE_MODE ? [ComplexF64(1.0, 0.1)] :
              [ComplexF64(1.0, 0.1),    # typical (mid-Q, mostly real)
               ComplexF64(0.1, 0.01),   # small Q
               ComplexF64(3.0, 0.5),    # larger Q
               ComplexF64(0.0, 1.0)]    # pure imaginary (γ-mode, ω=0)

x0_factors = [0.5, 1.0, 1.5]

# Pre-enumerate combos (with caps applied) so we can size + log up front
combos = []   # Vector of (D, qr, pr, Q_star, P, Q_pt)
for D in D_grid, qr in Qstar_ratio, pr in P_ratio, Q_pt in Q_test_grid
    Q_star = qr * D^4
    P      = pr * D^6
    P == 0.0     && continue           # boundary-condition floor
    Q_star > Qstar_cap && continue     # absolute Q_* cap
    P      > P_cap     && continue     # absolute P cap
    push!(combos, (D, qr, pr, Q_star, P, Q_pt))
end

@info @sprintf("Grid: %d D × %d Q*/D⁴ × %d P/D⁶ × %d Q = %d raw combos",
               length(D_grid), length(Qstar_ratio), length(P_ratio),
               length(Q_test_grid),
               length(D_grid)*length(Qstar_ratio)*length(P_ratio)*length(Q_test_grid))
@info @sprintf("After P=0 / Q*>%.0f / P>%.0f cuts: %d combos × %d x0 = %d Δs per solver",
               Qstar_cap, P_cap, length(combos),
               length(x0_factors), length(combos)*length(x0_factors))
@info @sprintf("Across %d solvers: ~%d total ODE solves",
               length(solver_names),
               length(combos)*length(x0_factors)*length(solver_names))

# Build SLAYERParameters with only the Riccati-relevant fields populated
# meaningfully. Outer-only fields (rs, R0, bt, etc.) get harmless dummy values.
function _build_params(D::Float64, Q_e::Float64, Q_i::Float64,
                       P_perp::Float64, P_tor::Float64;
                       iota_e::Float64=1.0)
    return SLAYERParameters(
        ising=1, m=2, n=1,
        tau=1.0, lu=1.0, c_beta=1.0,
        D_norm=D, P_perp=P_perp, P_tor=P_tor,
        Q_e=Q_e, Q_i=Q_i, iota_e=iota_e,
        tauk=1.0, tau_r=1.0, delta_n=0.01,
        rs=0.5, R0=1.0, bt=1.0, sval_r=1.5,
        dr_val=0.0, dgeo_val=0.0,
        eta=1e-8, d_beta=0.0,
    )
end

# Solve the Riccati ODE for a given x0_start (overriding _riccati_f_initial's
# natural choice). Returns (Δ, success, walltime_s, n_steps).
function _solve_riccati_at_x0(p::SLAYERParameters, Q::ComplexF64,
                              x0_factor::Float64, solver_factory_fn;
                              pmin::Real=1e-6, p_floor::Real=6.0,
                              reltol::Real=1e-10, abstol::Real=1e-10,
                              maxiters::Integer=50_000)
    # Mirror solve_inner's Wick rotation
    Q_c = im * conj(Q)

    # Natural x0 from the asymptotic expansion, then rescale.
    x0_natural, _, _ = _riccati_f_initial(p, Q_c; p_floor=p_floor)
    p_start = x0_factor * x0_natural

    # Recompute the asymptotic boundary value AT THIS x0 (not at x0_natural).
    # The asymptotic W(x) = xk - sqrt_bk·x  (large-D) or
    # W(x) = -1 + xk·x - sqrt_bk·x³        (small-D).
    D2 = p.D_norm^2
    Pperp_over_Ptor23 = p.P_perp / p.P_tor^(2/3)
    if D2 > p.iota_e * Pperp_over_Ptor23
        ak = -(Q_c + im * p.Q_e)
        bk = (p.iota_e * p.P_perp * p.P_tor) / (p.P_tor * D2)
        ck = bk * (1 + (Q_c + im * p.Q_i) * ((p.P_tor + p.P_perp) /
                                              (p.P_tor * p.P_perp))
                     - (p.P_perp + (Q_c + im * p.Q_i) * D2) *
                       (p.iota_e / (p.P_tor * D2)))
        sqrt_bk = sqrt(bk)
        xk = (ck - sqrt_bk * (1 - sqrt_bk * ak)) / (2 * sqrt_bk)
        W_bound = xk - sqrt_bk * p_start
    else
        ak = -(Q_c + im * p.Q_e)
        bk = ComplexF64(p.P_tor)
        ck = -im * (p.Q_e - p.Q_i) * (p.P_tor / p.P_perp) + (Q_c + im * p.Q_i)
        sqrt_bk = sqrt(bk)
        xk = (ak * bk - ck) / (2 * sqrt_bk)
        W_bound = -1.0 + xk * p_start - sqrt_bk * p_start^3
    end

    rhs_params = _build_riccati_consts(p, Q_c)
    u0 = ComplexF64(W_bound)
    f = ODEFunction{false}(_riccati_f_rhs; jac=_riccati_f_jac)
    prob = ODEProblem(f, u0, (p_start, pmin), rhs_params)

    success = true
    Δ = NaN + im * NaN
    walltime = NaN
    n_steps = 0
    try
        t0 = time_ns()
        sol = solve(prob, solver_factory_fn();
                    reltol=reltol, abstol=abstol, maxiters=maxiters,
                    save_everystep=false, dense=false)
        walltime = (time_ns() - t0) / 1e9
        n_steps = sol.stats.naccept + sol.stats.nreject
        success = sol.retcode == ReturnCode.Success
        if success
            W_end = sol.u[end]
            dW_end = _riccati_f_rhs(W_end, rhs_params, pmin)
            Δ = π / dW_end
        end
    catch e
        success = false
    end
    return (Δ=Δ, success=success, walltime=walltime, n_steps=n_steps)
end

# Run the full sweep ------------------------------------------------------
results = Dict{String,Vector{NamedTuple}}()
for sname in solver_names
    haskey(solver_factory, sname) ||
        (println("[skip] unknown solver $sname"); continue)
    @info "=== Solver: $sname ==="
    sfac = solver_factory[sname]

    # Warm-up (JIT) on one combo
    p_warm = _build_params(1.0, 0.25, 0.25, 1.0, 1.0)
    _solve_riccati_at_x0(p_warm, ComplexF64(1.0, 0.1), 1.0, sfac)

    rows = NamedTuple[]
    n_done = 0; n_total = length(combos)
    for (D, qr, pr, Q_star, P, Q_pt) in combos
        Q_e = Q_star / 2
        Q_i = Q_star / 2
        p = _build_params(D, Q_e, Q_i, P, P)
        outs = [_solve_riccati_at_x0(p, Q_pt, fac, sfac) for fac in x0_factors]
        Δs = [o.Δ for o in outs]
        successes = [o.success for o in outs]
        walls = [o.walltime for o in outs]
        steps_arr = [o.n_steps for o in outs]
        all_success = all(successes)
        spread_rel = NaN
        if all_success && all(isfinite, Δs)
            ref = Δs[2]   # x0_factor=1.0 reference
            if abs(ref) > 0
                spread_rel = maximum(abs.(Δs .- ref)) / abs(ref)
            end
        end
        converged_tight  = all_success && isfinite(spread_rel) && spread_rel < 1e-5
        converged_medium = all_success && isfinite(spread_rel) && spread_rel < 1e-4
        converged_loose  = all_success && isfinite(spread_rel) && spread_rel < 1e-3
        push!(rows, (D=D, Qratio=qr, Pratio=pr, Qstar=Q_star, P=P,
                     Q_re=real(Q_pt), Q_im=imag(Q_pt),
                     Δ=Δs, success=successes, walltime=walls, n_steps=steps_arr,
                     spread_rel=spread_rel,
                     converged_tight=converged_tight,
                     converged_medium=converged_medium,
                     converged_loose=converged_loose))
        n_done += 1
        if n_done % 200 == 0
            @info @sprintf("  [%s] %d/%d", sname, n_done, n_total)
        end
    end
    results[sname] = rows
    n_tight  = count(r->r.converged_tight, rows)
    n_medium = count(r->r.converged_medium, rows)
    n_loose  = count(r->r.converged_loose, rows)
    n_succ   = count(r->all(r.success), rows)
    walls_all = vcat([collect(r.walltime) for r in rows]...)
    median_wall = median(walls_all)
    p95_wall    = quantile(walls_all, 0.95)
    mean_steps  = mean(vcat([collect(r.n_steps) for r in rows]...))
    @info @sprintf("  [%s] tight<1e-5 %.1f%%  med<1e-4 %.1f%%  loose<1e-3 %.1f%%  all-succ %.1f%%  walltime med=%.2fms p95=%.2fms  mean steps=%.0f",
                   sname,
                   100*n_tight/length(rows),
                   100*n_medium/length(rows),
                   100*n_loose/length(rows),
                   100*n_succ/length(rows),
                   1e3*median_wall, 1e3*p95_wall, mean_steps)
end

# Write a tab-separated row-per-test output. Easier for downstream
# pandas / awk / spreadsheet inspection than nested JSON, and avoids
# pulling JSON.jl as a direct dep.
open(out_path, "w") do f
    println(f, "# Riccati solver convergence test")
    println(f, "# Q test grid = $Q_test_grid")
    println(f, "# x0_factors = $x0_factors")
    println(f, "# Caps: Q_* ≤ $Qstar_cap, P ≤ $P_cap")
    println(f, "# Convergence criterion: max|Δᵢ−Δ_ref|/|Δ_ref|, thresholds 1e-5/1e-4/1e-3")
    println(f, "")
    println(f, join(["solver", "D", "Qratio", "Pratio", "Qstar", "P",
                     "Q_re", "Q_im",
                     "Δ_re_x0lo", "Δ_im_x0lo", "Δ_re_x0med", "Δ_im_x0med",
                     "Δ_re_x0hi", "Δ_im_x0hi",
                     "success_lo", "success_med", "success_hi",
                     "walltime_lo", "walltime_med", "walltime_hi",
                     "steps_lo", "steps_med", "steps_hi",
                     "spread_rel", "conv_tight_1e-5",
                     "conv_med_1e-4", "conv_loose_1e-3"], '\t'))
    for (sname, rs) in results
        for r in rs
            println(f, join([sname, r.D, r.Qratio, r.Pratio, r.Qstar, r.P,
                             r.Q_re, r.Q_im,
                             real(r.Δ[1]), imag(r.Δ[1]),
                             real(r.Δ[2]), imag(r.Δ[2]),
                             real(r.Δ[3]), imag(r.Δ[3]),
                             Int(r.success[1]), Int(r.success[2]), Int(r.success[3]),
                             r.walltime[1], r.walltime[2], r.walltime[3],
                             r.n_steps[1], r.n_steps[2], r.n_steps[3],
                             r.spread_rel,
                             Int(r.converged_tight),
                             Int(r.converged_medium),
                             Int(r.converged_loose)], '\t'))
        end
    end
end
@info "Wrote $out_path"

# Brief summary table to stdout
println("\n  Solver summary (rows = solvers, columns = metrics):")
println(@sprintf("  %-10s  %-10s  %-10s  %-10s  %-10s  %-12s  %-12s  %-10s",
                 "solver", "tight<1e-5", "med<1e-4", "loose<1e-3",
                 "any-fail", "med wall(ms)", "p95 wall(ms)", "mean steps"))
println("  " * "-"^104)
for sname in solver_names
    haskey(results, sname) || continue
    rs = results[sname]
    n_tight  = count(r->r.converged_tight, rs)
    n_med    = count(r->r.converged_medium, rs)
    n_loose  = count(r->r.converged_loose, rs)
    n_fail   = count(r->!all(r.success), rs)
    walls_all = vcat([collect(r.walltime) for r in rs]...)
    median_wall = median(walls_all)
    p95_wall    = quantile(walls_all, 0.95)
    mean_steps  = mean(vcat([collect(r.n_steps) for r in rs]...))
    println(@sprintf("  %-10s  %5.1f%%      %5.1f%%      %5.1f%%      %3d/%-3d    %6.2f       %6.2f        %4.0f",
                     sname,
                     100*n_tight/length(rs),
                     100*n_med/length(rs),
                     100*n_loose/length(rs),
                     n_fail, length(rs),
                     1e3*median_wall, 1e3*p95_wall, mean_steps))
end
