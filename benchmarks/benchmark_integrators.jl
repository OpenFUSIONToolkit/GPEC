# Work-precision comparison of OrdinaryDiffEq integrators on the STRIDE-equivalent step of
# julia_GPEC: the fundamental-matrix (FM) propagator chunks of the Euler-Lagrange ODE.
#
# A chunk propagator is the fundamental matrix Φ(ψ₂,ψ₁) of the Euler-Lagrange system over one
# sub-interval of the outer region, obtained by integrating from the two identity-block initial
# conditions (I,0) and (0,I) [Glasser 2018 Phys. Plasmas 25, 032507]. Assembling the chunk
# propagators serially with periodic renormalization is what makes the chunked BVP well
# conditioned, and it is where essentially all of the Riccati solver's ODE time is spent.
#
# Accuracy criterion: each candidate is compared against a Vern9 reference at reltol=1e-13,
# abstol=1e-15 through the relative Frobenius error of the end-state blocks, maxed over the two
# initial conditions. The row `Vern9` at the case's own `eulerlagrange_tolerance` is the
# production setting and is the accuracy floor every candidate must be judged against — an
# integrator is only a drop-in replacement if it is at least as accurate as that row.
#
# Usage:
#   julia --project=. -t 1 benchmarks/benchmark_integrators.jl --case <case_dir> [options]
#
#   --case <dir>            case directory containing eq.geqdsk and gpec.toml (required)
#   --algs <a,b,c>          comma-separated OrdinaryDiffEq algorithm names
#   --rtols <r1,r2>         comma-separated relative tolerances (default 1e-6,1e-8,1e-10)
#   --reps <n>              timed repetitions, minimum is kept (default 3)
#   --out <path>            CSV output path (default benchmarks/integrator_results/<case>.csv)
#   --quick                 smoke test: 3 chunks, Vern9/Tsit5/VCABM, rtol 1e-8, 1 rep
#   --include-outer         also time the serial outer-plasma Riccati solve with its callback

using LinearAlgebra, Printf, TOML
using GeneralizedPerturbedEquilibrium
using OrdinaryDiffEq

const GPE = GeneralizedPerturbedEquilibrium
const FFS = GeneralizedPerturbedEquilibrium.ForceFreeStates

# Absolute tolerance for candidate solves; `nothing` keeps the production default (OrdinaryDiffEq 1e-6).
const CANDIDATE_ABSTOL = Ref{Union{Nothing,Float64}}(nothing)

const DEFAULT_ALGS = ["Tsit5", "BS5", "DP5", "Vern6", "Vern7", "Vern8", "Vern9", "DP8", "TanYam7", "TsitPap8",
    "Feagin10", "Feagin12", "Feagin14", "VCAB4", "VCAB5", "VCABM3", "VCABM4", "VCABM5", "VCABM", "AN5"]

# ---------------------------------------------------------------- CLI

function parse_args(args)
    opts = Dict{String,Any}("case" => nothing, "algs" => DEFAULT_ALGS, "rtols" => [1e-6, 1e-8, 1e-10],
        "reps" => 3, "out" => nothing, "quick" => false, "include_outer" => false, "blas" => nothing, "abstol" => nothing)
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--case"
            opts["case"] = args[i+1]
            i += 2
        elseif a == "--algs"
            opts["algs"] = String.(split(args[i+1], ","))
            i += 2
        elseif a == "--rtols"
            opts["rtols"] = parse.(Float64, split(args[i+1], ","))
            i += 2
        elseif a == "--reps"
            opts["reps"] = parse(Int, args[i+1])
            i += 2
        elseif a == "--out"
            opts["out"] = args[i+1]
            i += 2
        elseif a == "--quick"
            opts["quick"] = true
            i += 1
        elseif a == "--include-outer"
            opts["include_outer"] = true
            i += 1
        elseif a == "--blas-threads"
            opts["blas"] = parse(Int, args[i+1])
            i += 2
        elseif a == "--abstol"
            opts["abstol"] = parse(Float64, args[i+1])
            i += 2
        else
            error("Unknown argument: $a")
        end
    end
    opts["case"] === nothing && error("--case <dir> is required")
    if opts["quick"]
        opts["algs"] = ["Vern9", "Tsit5", "VCABM"]
        opts["rtols"] = [1e-8]
        opts["reps"] = 1
    end
    return opts
end

# ---------------------------------------------------------------- case setup

# Build (ctrl, equil, mats, intr) exactly as the production pipeline does: mode space from
# resolve_mode_space!, the two-pass auto grid re-formed once, matrices from prepare_force_free_states!.
function setup_case(dir::AbstractString)
    inputs = TOML.parsefile(joinpath(dir, "gpec.toml"))
    inputs["ForceFreeStates"]["verbose"] = false
    inputs["ForceFreeStates"]["write_outputs_to_HDF5"] = false
    inputs["ForceFreeStates"]["integrator"] = "riccati"
    ctrl = FFS.ForceFreeStatesControl(; (Symbol(k) => v for (k, v) in inputs["ForceFreeStates"])...)
    eq_config = GPE.Equilibrium.EquilibriumConfig(inputs["Equilibrium"], dir)
    intr = FFS.ForceFreeStatesInternal(; dir_path=dir)
    intr.wall_settings = GPE.Vacuum.WallShapeSettings(; (Symbol(k) => v for (k, v) in inputs["Wall"])...)
    GPE.resolve_mode_space!(intr, ctrl)
    equil = GPE.Equilibrium.setup_equilibrium(eq_config, nothing)
    equil = GPE.maybe_reform_equilibrium(equil, eq_config, nothing, intr, ctrl, nothing)
    metric, mats = GPE.prepare_force_free_states!(intr, ctrl, equil, GPE.KineticForces.KineticForcesControl(), nothing)
    return ctrl, equil, mats, intr
end

# ---------------------------------------------------------------- one chunk

# Reproduces integrate_propagator_chunk! standalone: two identity-block ICs, hints reset before
# each solve, tspan reversed for backward (direction=-1) chunks.
function solve_chunk(chunk, alg, ctrl, equil, mats, intr, proxy; reltol, abstol=nothing)
    N = intr.numpert_total
    tspan = chunk.direction == 1 ? (chunk.psi_start, chunk.psi_end) : (chunk.psi_end, chunk.psi_start)
    params = (ctrl, equil, mats, intr, proxy, chunk)
    blocks = Vector{Array{ComplexF64,3}}(undef, 2)
    nf = naccept = nreject = 0
    for (ic, slot) in ((1, 1), (2, 2))
        u0 = zeros(ComplexF64, N, N, 2)
        for i in 1:N
            u0[i, i, ic] = 1
        end
        proxy.spline_hint[] = 1
        proxy.mats_hint[] = 1
        prob = ODEProblem(FFS.sing_der!, u0, tspan, params)
        atol = abstol === nothing ? CANDIDATE_ABSTOL[] : abstol
        sol =
            atol === nothing ? solve(prob, alg; reltol=reltol, save_everystep=false, save_end=true) :
            solve(prob, alg; reltol=reltol, abstol=atol, save_everystep=false, save_end=true)
        blocks[slot] = sol.u[end]
        nf += sol.stats.nf
        naccept += sol.stats.naccept
        nreject += sol.stats.nreject
    end
    return blocks, nf, naccept, nreject
end

relerr(a, b) = norm(a .- b) / norm(b)
chunk_error(blocks, ref) = max(relerr(blocks[1], ref[1]), relerr(blocks[2], ref[2]))

# ---------------------------------------------------------------- sweep

function sweep(algname, rtol, chunks, ctrl, equil, mats, intr, refs, reps)
    N = intr.numpert_total
    proxy = FFS.OdeState(N, 1, 1, 0)
    alg = try
        getfield(OrdinaryDiffEq, Symbol(algname))()
    catch err
        @warn "construction failed for $algname: $(sprint(showerror, err))"
        return nothing
    end
    try
        solve_chunk(chunks[1], alg, ctrl, equil, mats, intr, proxy; reltol=rtol)  # JIT warm-up, untimed
    catch err
        @warn "first solve failed for $algname @ rtol=$rtol: $(sprint(showerror, err))"
        return nothing
    end
    best = Inf
    nf = naccept = nreject = 0
    held = Vector{Vector{Array{ComplexF64,3}}}(undef, length(chunks))
    per_naccept = zeros(Int, length(chunks))
    for _ in 1:reps
        nf_r = naccept_r = nreject_r = 0
        t = @elapsed for (i, ch) in enumerate(chunks)
            blocks, a, b, c = solve_chunk(ch, alg, ctrl, equil, mats, intr, proxy; reltol=rtol)
            held[i] = blocks
            per_naccept[i] = b
            nf_r += a
            naccept_r += b
            nreject_r += c
        end
        if t < best
            best = t
            nf, naccept, nreject = nf_r, naccept_r, nreject_r
        end
    end
    errs = [chunk_error(held[i], refs[i]) for i in eachindex(chunks)]
    return (; wall=best, nf=nf, naccept=naccept, nreject=nreject, errs=errs, per_naccept=per_naccept)
end

# Serial outer-plasma Riccati solve (axis to the first crossing) with its DiscreteCallback.
# Only the first chunk is run: it is the representative callback-carrying solve and needs no
# singular-surface crossing machinery.
function sweep_outer(algname, rtol, ctrl, equil, mats, intr, reps)
    alg = try
        getfield(OrdinaryDiffEq, Symbol(algname))()
    catch err
        @warn "construction failed for $algname (outer): $(sprint(showerror, err))"
        return nothing
    end
    saved = FFS.EL_ODE_ALGORITHM[]
    FFS.EL_ODE_ALGORITHM[] = alg
    ctrl_local = ctrl
    best = Inf
    ustate = nothing
    try
        for r in 0:reps
            odet = FFS._initialize_parallel_odet(ctrl_local, equil, mats, intr)
            serial = FFS.chunk_el_integration_bounds(odet, ctrl_local, intr; bidirectional=false)
            t = @elapsed FFS.riccati_integrate_chunk!(odet, ctrl_local, equil, mats, intr, serial[1])
            r == 0 && continue  # warm-up
            best = min(best, t)
            ustate = copy(odet.u)
        end
    catch err
        @warn "outer solve failed for $algname @ rtol=$rtol: $(sprint(showerror, err))"
        FFS.EL_ODE_ALGORITHM[] = saved
        return nothing
    end
    FFS.EL_ODE_ALGORITHM[] = saved
    return (; wall=best, u=ustate)
end

# ---------------------------------------------------------------- driver

function main(args)
    opts = parse_args(args)
    casedir = abspath(opts["case"])
    casename = basename(rstrip(casedir, '/'))
    outpath = opts["out"] === nothing ? joinpath(@__DIR__, "integrator_results", casename * ".csv") : abspath(opts["out"])
    mkpath(dirname(outpath))
    chunkpath = replace(outpath, r"\.csv$" => "") * "_chunks.csv"

    opts["blas"] !== nothing && BLAS.set_num_threads(opts["blas"])
    CANDIDATE_ABSTOL[] = opts["abstol"]
    ctrl, equil, mats, intr = setup_case(casedir)
    N = intr.numpert_total
    odet = FFS._initialize_parallel_odet(ctrl, equil, mats, intr)
    chunks, _, _ = FFS._setup_parallel_chunks_and_proxies(odet, ctrl, intr)
    opts["quick"] && (chunks = chunks[1:min(3, length(chunks))])

    @printf("\ncase              %s\n", casename)
    @printf("N (numpert_total) %d\n", N)
    @printf("BLAS threads      %d\n", BLAS.get_num_threads())
    @printf("candidate abstol  %s\n", CANDIDATE_ABSTOL[] === nothing ? "default (1e-6)" : string(CANDIDATE_ABSTOL[]))
    @printf("msing             %d\n", intr.msing)
    @printf("nchunks           %d\n", length(chunks))
    @printf("psi range         [%.6f, %.6f] over %d chunk boundaries\n",
        minimum(c.psi_start for c in chunks), maximum(c.psi_end for c in chunks), length(chunks) + 1)
    @printf("production rtol   %.1e (eulerlagrange_tolerance, Vern9)\n\n", ctrl.eulerlagrange_tolerance)

    println("computing Vern9 reference (reltol=1e-13, abstol=1e-15) ...")
    refproxy = FFS.OdeState(N, 1, 1, 0)
    refs = [solve_chunk(ch, Vern9(), ctrl, equil, mats, intr, refproxy; reltol=1e-13, abstol=1e-15)[1] for ch in chunks]
    println("reference done.\n")

    rows = Any[]
    chunkrows = Dict{Tuple{String,Float64},Any}()
    # The production setting first: it defines the accuracy floor the candidates are judged against.
    prod_rtol = ctrl.eulerlagrange_tolerance
    pairs = [("Vern9", prod_rtol)]
    for rtol in opts["rtols"], a in opts["algs"]
        (a, rtol) in pairs || push!(pairs, (a, rtol))
    end

    for (algname, rtol) in pairs
        res = sweep(algname, rtol, chunks, ctrl, equil, mats, intr, refs, opts["reps"])
        if res === nothing
            push!(rows, (casename, algname, rtol, "failed", length(chunks), N, intr.msing, NaN, 0, 0, 0, NaN, NaN))
            continue
        end
        us_per_rhs = res.nf > 0 ? res.wall / res.nf * 1e6 : NaN
        push!(rows, (casename, algname, rtol, "ok", length(chunks), N, intr.msing, res.wall, res.nf,
            res.naccept, res.nreject, maximum(res.errs), median_of(res.errs)))
        chunkrows[(algname, rtol)] = res
        @printf("  %-10s rtol=%.0e  %7.3f s  nf=%-8d err_max=%.3e\n", algname, rtol, res.wall, res.nf, maximum(res.errs))
    end

    if opts["include_outer"]
        for rtol in opts["rtols"], algname in opts["algs"]
            r = sweep_outer(algname, rtol, ctrl, equil, mats, intr, opts["reps"])
            r === nothing && continue
            push!(rows, (casename, algname * "+outer", rtol, "ok", 1, N, intr.msing, r.wall, 0, 0, 0, NaN, NaN))
        end
    end

    write_csv(outpath, rows)
    write_chunk_csv(chunkpath, casename, chunks, chunkrows, rows)
    print_table(rows, prod_rtol)
    @printf("\nwrote %s\n      %s\n", outpath, chunkpath)
    return nothing
end

function median_of(v)
    isempty(v) && return NaN
    s = sort(v)
    n = length(s)
    return isodd(n) ? s[(n+1)÷2] : 0.5 * (s[n÷2] + s[n÷2+1])
end

const HEADER = "case,alg,rtol,status,nchunks,N,msing,wall_s_min,nf,naccept,nreject,us_per_rhs,err_max,err_median"

function write_csv(path, rows)
    open(path, "w") do io
        println(io, HEADER)
        for r in rows
            us = r[9] > 0 ? r[8] / r[9] * 1e6 : NaN
            @printf(io, "%s,%s,%.3e,%s,%d,%d,%d,%.6f,%d,%d,%d,%.4f,%.6e,%.6e\n",
                r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9], r[10], r[11], us, r[12], r[13])
        end
    end
end

# Per-chunk detail for the three fastest algorithms, so step concentration is visible.
function write_chunk_csv(path, casename, chunks, chunkrows, rows)
    ok = filter(r -> r[4] == "ok" && !occursin("+outer", r[2]), rows)
    ranked = sort(unique(r[2] for r in ok); by=a -> minimum(r[8] for r in ok if r[2] == a))
    best3 = ranked[1:min(3, length(ranked))]
    open(path, "w") do io
        println(io, "case,alg,rtol,chunk,psi_start,psi_end,direction,naccept,err")
        for ((algname, rtol), res) in sort(collect(chunkrows); by=kv -> (kv[1][1], kv[1][2]))
            algname in best3 || continue
            for (i, ch) in enumerate(chunks)
                @printf(io, "%s,%s,%.3e,%d,%.8f,%.8f,%d,%d,%.6e\n",
                    casename, algname, rtol, i, ch.psi_start, ch.psi_end, ch.direction, res.per_naccept[i], res.errs[i])
            end
        end
    end
end

function print_table(rows, prod_rtol)
    println("\n", "="^108)
    @printf("%-12s %-9s %-7s %10s %10s %9s %9s %10s %11s %11s\n",
        "alg", "rtol", "status", "wall_s", "nf", "naccept", "nreject", "us/rhs", "err_max", "err_med")
    println("="^108)
    for rtol in sort(unique(r[3] for r in rows))
        sub = sort(filter(r -> r[3] == rtol, rows); by=r -> (r[4] == "ok" ? 0 : 1, isnan(r[8]) ? Inf : r[8]))
        for r in sub
            us = r[9] > 0 ? r[8] / r[9] * 1e6 : NaN
            mark = (r[2] == "Vern9" && rtol == prod_rtol) ? " <- production" : ""
            @printf("%-12s %-9.0e %-7s %10.3f %10d %9d %9d %10.4f %11.3e %11.3e%s\n",
                r[2], r[3], r[4], r[8], r[9], r[10], r[11], us, r[12], r[13], mark)
        end
        println("-"^108)
    end
end

main(ARGS)
