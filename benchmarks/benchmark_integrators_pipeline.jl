#!/usr/bin/env julia
"""
benchmark_integrators_pipeline.jl - Whole-pipeline integrator comparison for the Euler-Lagrange sweep.

Runs the Force-Free States stage of one case (equilibrium and matrix splines prepared once, untimed)
with every requested OrdinaryDiffEq algorithm and relative tolerance, selected through the
`ode_solver` / `ode_abstol` control fields. Times the Euler-Lagrange integration, the free-boundary energies and
the Δ′ BVP separately, and compares the products every downstream consumer reads — the free-boundary
energy eigenvalues `et` and the singular-surface Δ′ matrix — against two references:

  - `ref`:   Vern9 at the case's shipped `eulerlagrange_tolerance` (today's production numbers), and
  - `truth`: Vern9 at reltol 1e-13, the best answer this pipeline can produce.

The acceptance rule is "no loss in accuracy": a candidate is only interesting if its distance from
`truth` is no larger than the production reference's own distance from `truth`.

# Usage

```bash
julia --project=. -t 1 benchmarks/benchmark_integrators_pipeline.jl --case <dir> [options]

  --case <dir>           directory holding eq.geqdsk and gpec.toml (required)
  --algs a,b,c           OrdinaryDiffEq algorithm names (default: the explicit non-stiff set below)
  --rtols 1e-8,1e-10     relative tolerances to sweep (default: the case's eulerlagrange_tolerance)
  --reps <n>             timed repetitions per (alg, rtol); the minimum is reported (default 2)
  --mode riccati|forward Euler-Lagrange formalism (default riccati, the STRIDE-equivalent path)
  --blas-threads <k>     BLAS.set_num_threads(k) before timing (default: leave as started)
  --abstol <a>           absolute tolerance for every candidate solve (default 1e-6, the OrdinaryDiffEq default)
  --out <csv>            output path (default benchmarks/integrator_results/<case>_pipeline.csv)
  --no-truth             skip the reltol 1e-13 truth run
```

Inputs are read only from the case directory; outputs are written under `benchmarks/`.
"""

using LinearAlgebra
using Printf
using Statistics
using TOML
using OrdinaryDiffEq
using GeneralizedPerturbedEquilibrium

const GPE = GeneralizedPerturbedEquilibrium
const FFS = GPE.ForceFreeStates

const DEFAULT_ALGS = ["Vern9", "Vern8", "Vern7", "Vern6", "Tsit5", "BS5", "DP5", "DP8", "TanYam7", "TsitPap8",
    "Feagin10", "Feagin12", "Feagin14", "VCAB4", "VCAB5", "VCABM3", "VCABM4", "VCABM5", "VCABM", "AN5"]
const TRUTH_RTOL = 1e-13
const TRUTH_ABSTOL = 1e-15

function parse_args(args)
    opts = Dict{String,Any}("algs" => DEFAULT_ALGS, "rtols" => nothing, "reps" => 2, "mode" => "riccati",
        "blas" => nothing, "out" => nothing, "truth" => true, "case" => nothing, "abstol" => 1e-6)
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--case"
            opts["case"] = abspath(args[i+1])
            i += 2
        elseif a == "--algs"
            opts["algs"] = split(args[i+1], ",")
            i += 2
        elseif a == "--rtols"
            opts["rtols"] = parse.(Float64, split(args[i+1], ","))
            i += 2
        elseif a == "--reps"
            opts["reps"] = parse(Int, args[i+1])
            i += 2
        elseif a == "--mode"
            opts["mode"] = args[i+1]
            i += 2
        elseif a == "--blas-threads"
            opts["blas"] = parse(Int, args[i+1])
            i += 2
        elseif a == "--out"
            opts["out"] = abspath(args[i+1])
            i += 2
        elseif a == "--abstol"
            opts["abstol"] = parse(Float64, args[i+1])
            i += 2
        elseif a == "--no-truth"
            opts["truth"] = false
            i += 1
        else
            error("unknown argument $a")
        end
    end
    opts["case"] === nothing && error("--case <dir> is required")
    opts["mode"] in ("riccati", "forward") || error("--mode must be riccati or forward")
    return opts
end

# Equilibrium, wall and control keywords from the case's gpec.toml; the two-pass auto grid is
# re-formed once here so every timed run integrates against the same equilibrium.
function prepare_case(case::String, mode::String)
    inputs = TOML.parsefile(joinpath(case, "gpec.toml"))
    eq_config = GPE.Equilibrium.EquilibriumConfig(inputs["Equilibrium"], case)
    wall = GPE.Vacuum.WallShapeSettings(; (Symbol(k) => v for (k, v) in inputs["Wall"])...)
    kw = Dict{Symbol,Any}(Symbol(k) => v for (k, v) in inputs["ForceFreeStates"])
    kw[:verbose] = false
    kw[:write_outputs_to_HDF5] = false
    kw[:integrator] = mode
    ctrl = FFS.ForceFreeStatesControl(; kw...)

    intr = FFS.ForceFreeStatesInternal(; dir_path=case)
    intr.wall_settings = wall
    GPE.resolve_mode_space!(intr, ctrl)
    equil = GPE.Equilibrium.setup_equilibrium(eq_config, nothing)
    equil = GPE.maybe_reform_equilibrium(equil, eq_config, nothing, intr, ctrl, nothing)
    kf_ctrl = GPE.KineticForces.KineticForcesControl()
    t_prep = @elapsed metric, mats = GPE.prepare_force_free_states!(intr, ctrl, equil, kf_ctrl, nothing)
    return (; equil, wall, kw, intr, metric, mats, t_prep)
end

# One Force-Free States run mirroring run_force_free_states, with the three stages timed separately.
function run_once(prep, alg::AbstractString, rtol::Float64, case::String; abstol::Float64=1e-6)
    kw = copy(prep.kw)
    kw[:eulerlagrange_tolerance] = rtol
    kw[:ode_solver] = String(alg)
    kw[:ode_abstol] = abstol
    ctrl = FFS.ForceFreeStatesControl(; kw...)
    intr = deepcopy(prep.intr)
    equil, mats, metric = prep.equil, prep.mats, prep.metric

    t_int = @elapsed odet, fm_propagators, fm_chunks, fm_S_left = FFS.eulerlagrange_integration(ctrl, equil, mats, intr)
    free_energies = nothing
    t_free = 0.0
    t_dp = 0.0
    if ctrl.vac_flag
        t_free = @elapsed begin
            free_energies = FFS.free_run(odet, ctrl, equil, mats, intr)
            FFS.normalize_eigenfunctions!(odet, free_energies.wt, equil.psio)
        end
        if intr.msing > 0 && fm_propagators !== nothing
            t_dp = @elapsed FFS.compute_delta_prime_matrix!(intr, fm_propagators, fm_chunks;
                wv=free_energies.wv, psio=equil.psio, debug=false, S_at_surface_left=fm_S_left, ctrl=ctrl, equil=equil, mats=mats)
        end
    end
    res = FFS.build_result(Symbol(ctrl.integrator), ctrl, equil, intr, metric, mats, odet, free_energies, nothing, nothing)
    et = free_energies === nothing ? ComplexF64[] : free_energies.et
    dp = res.delta_prime === nothing ? nothing : res.delta_prime.matrix
    return (; t_int, t_free, t_dp, total_steps=odet.total_steps, nzero=odet.nzero, et, dp, nchunks=fm_chunks === nothing ? 0 : length(fm_chunks))
end

relerr(a, b) = norm(a - b) / max(norm(b), eps())
et_err(r, ref) = (isempty(r.et) || isempty(ref.et)) ? NaN : relerr(r.et[1:min(5, end)], ref.et[1:min(5, end)])
dp_err(r, ref) = (r.dp === nothing || ref.dp === nothing) ? NaN : relerr(r.dp, ref.dp)

function main(args)
    opts = parse_args(args)
    case = opts["case"]
    casename = basename(case)
    opts["blas"] !== nothing && BLAS.set_num_threads(opts["blas"])
    out = something(opts["out"], joinpath(@__DIR__, "integrator_results", "$(casename)_pipeline.csv"))
    mkpath(dirname(out))

    @info "Preparing $casename (mode=$(opts["mode"]))"
    prep = prepare_case(case, opts["mode"])
    rtol0 = Float64(prep.kw[:eulerlagrange_tolerance])
    rtols = something(opts["rtols"], [rtol0])
    N = prep.intr.numpert_total
    @printf("case %s: N=%d msing=%d threads=%d blas=%d prep=%.2fs shipped rtol=%.1e abstol=%.1e\n",
        casename, N, prep.intr.msing, Threads.nthreads(), BLAS.get_num_threads(), prep.t_prep, rtol0, opts["abstol"])

    # JIT warm-up and the two references.
    run_once(prep, "Vern9", rtol0, case)
    ref = run_once(prep, "Vern9", rtol0, case)
    truth = opts["truth"] ? run_once(prep, "Vern9", TRUTH_RTOL, case; abstol=TRUTH_ABSTOL) : ref
    @printf("reference Vern9@%.0e: steps=%d t_int=%.2fs et1=%.6e  | truth Vern9@%.0e/abstol 1e-15: steps=%d t_int=%.2fs et1=%.6e  | ref-vs-truth et=%.2e dp=%.2e\n",
        rtol0, ref.total_steps, ref.t_int, real(ref.et[1]), TRUTH_RTOL, truth.total_steps, truth.t_int, real(truth.et[1]),
        et_err(ref, truth), dp_err(ref, truth))

    header =
        "case,alg,rtol,abstol,status,mode,nthreads,blas_threads,N,msing,nchunks,total_steps,t_int_s,t_free_s,t_dp_s,t_total_s," *
        "et1_re,nzero,err_et_vs_ref,err_dp_vs_ref,err_et_vs_truth,err_dp_vs_truth"
    rows = String[]
    for rtol in rtols, name in opts["algs"]
        if !isdefined(OrdinaryDiffEq, Symbol(name))
            @warn "unknown OrdinaryDiffEq algorithm $name"
            push!(
                rows,
                "$casename,$name,$rtol,$(opts["abstol"]),failed,$(opts["mode"]),$(Threads.nthreads()),$(BLAS.get_num_threads()),$N,$(prep.intr.msing)," * join(fill("", 12), ",")
            )
            continue
        end
        alg = name
        best = nothing
        status = "ok"
        try
            run_once(prep, alg, rtol, case; abstol=opts["abstol"])  # warm-up for this algorithm's compiled code
            for _ in 1:opts["reps"]
                r = run_once(prep, alg, rtol, case; abstol=opts["abstol"])
                best = (best === nothing || r.t_int + r.t_free + r.t_dp < best.t_int + best.t_free + best.t_dp) ? r : best
            end
        catch err
            status = "failed"
            @warn "$name @ $rtol failed: $(sprint(showerror, err))"
        end
        if best === nothing
            push!(
                rows,
                "$casename,$name,$rtol,$(opts["abstol"]),$status,$(opts["mode"]),$(Threads.nthreads()),$(BLAS.get_num_threads()),$N,$(prep.intr.msing)," * join(fill("", 12), ",")
            )
            continue
        end
        t_total = best.t_int + best.t_free + best.t_dp
        line = @sprintf("%s,%s,%.1e,%.1e,%s,%s,%d,%d,%d,%d,%d,%d,%.4f,%.4f,%.4f,%.4f,%.10e,%d,%.3e,%.3e,%.3e,%.3e",
            casename, name, rtol, opts["abstol"], status, opts["mode"], Threads.nthreads(), BLAS.get_num_threads(), N, prep.intr.msing, best.nchunks,
            best.total_steps, best.t_int, best.t_free, best.t_dp, t_total, isempty(best.et) ? NaN : real(best.et[1]), best.nzero,
            et_err(best, ref), dp_err(best, ref), et_err(best, truth), dp_err(best, truth))
        push!(rows, line)
        @printf("%-9s rtol=%.0e steps=%6d t_int=%7.3fs t_tot=%7.3fs et1=%.6e  vs_ref: et=%.1e dp=%.1e  vs_truth: et=%.1e dp=%.1e\n",
            name, rtol, best.total_steps, best.t_int, t_total, isempty(best.et) ? NaN : real(best.et[1]),
            et_err(best, ref), dp_err(best, ref), et_err(best, truth), dp_err(best, truth))
    end
    open(out, "w") do io
        println(io, header)
        foreach(r -> println(io, r), rows)
    end
    @info "Wrote $out"
end

main(ARGS)
