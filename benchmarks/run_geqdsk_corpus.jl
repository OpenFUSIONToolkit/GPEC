#!/usr/bin/env julia
"""
run_geqdsk_corpus.jl - Run the Force-Free States stage on every geqdsk in a list and tabulate the outcome.

Robustness sweep for the Riccati Euler-Lagrange path: each geqdsk is run with one template `gpec.toml`
(the shipped DIII-D-like deck by default) on a pool of worker processes, with a wall-clock cap per case.
A worker that exceeds the cap is killed and replaced, so one pathological equilibrium cannot stall the
sweep, and the JIT cost is paid once per worker rather than once per case. One CSV row per case:
q-profile shape from the geqdsk, mode space, rational surfaces, step count, stage timings, the lowest
free-boundary eigenvalues and the Δ′ diagonal.

# Usage

```bash
julia --project=. benchmarks/run_geqdsk_corpus.jl --list <file> --root <dir> --out <csv> \\
      [--template <gpec.toml>] [--workers 2] [--threads 4] [--timeout 600] [--limit n]
```

`--list` holds one geqdsk path per line, relative to `--root`. Outputs are written under `benchmarks/`.
"""

using Distributed
using Printf
using TOML

function parse_args(args)
    opts = Dict{String,Any}("list" => nothing, "root" => ".", "out" => nothing, "workers" => 2, "threads" => 4,
        "timeout" => 600.0, "limit" => typemax(Int),
        "template" => joinpath(@__DIR__, "..", "examples", "DIIID-like_ideal_example", "gpec.toml"))
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--list"
            opts["list"] = abspath(args[i+1])
            i += 2
        elseif a == "--root"
            opts["root"] = abspath(args[i+1])
            i += 2
        elseif a == "--out"
            opts["out"] = abspath(args[i+1])
            i += 2
        elseif a == "--template"
            opts["template"] = abspath(args[i+1])
            i += 2
        elseif a == "--workers"
            opts["workers"] = parse(Int, args[i+1])
            i += 2
        elseif a == "--threads"
            opts["threads"] = parse(Int, args[i+1])
            i += 2
        elseif a == "--timeout"
            opts["timeout"] = parse(Float64, args[i+1])
            i += 2
        elseif a == "--limit"
            opts["limit"] = parse(Int, args[i+1])
            i += 2
        else
            error("unknown argument $a")
        end
    end
    opts["list"] === nothing && error("--list <file> is required")
    opts["out"] === nothing && (opts["out"] = joinpath(@__DIR__, "integrator_results", "geqdsk_corpus.csv"))
    return opts
end

# Template deck with the Force-Free States stage only (no PerturbedEquilibrium / KineticForces).
function corpus_template(path)
    inputs = TOML.parsefile(path)
    keep = Dict{String,Any}(k => inputs[k] for k in ("Equilibrium", "Wall", "ForceFreeStates") if haskey(inputs, k))
    keep["Equilibrium"]["eq_filename"] = "eq.geqdsk"
    keep["ForceFreeStates"]["integrator"] = "riccati"
    keep["ForceFreeStates"]["verbose"] = false
    keep["ForceFreeStates"]["write_outputs_to_HDF5"] = false
    keep["ForceFreeStates"]["local_stability_flag"] = false
    return keep
end

const HEADER = "idx,geqdsk,status,elapsed_s,nw,q0,qmin,psi_qmin,qedge,N,msing,surfaces,total_steps," *
               "t_equil_s,t_prep_s,t_int_s,t_free_s,t_dp_s,et1,et2,et3,dp_diag,message"

function main(args)
    opts = parse_args(args)
    files = filter(!isempty, readlines(opts["list"]))[1:min(end, opts["limit"])]
    mkpath(dirname(opts["out"]))
    template = corpus_template(opts["template"])
    exeflags = ["--project=$(Base.active_project())", "--threads=$(opts["threads"])"]

    # Worker-side case runner; the imports are evaluated first so the macros below resolve.
    worker_imports = quote
        using LinearAlgebra, TOML, Printf
        using GeneralizedPerturbedEquilibrium
    end
    worker_setup = quote
        const GPE = GeneralizedPerturbedEquilibrium
        const FFS = GPE.ForceFreeStates
        function read_q_profile(path)
            lines = readlines(path)
            # nw, nh are the last two integers of the header line; the leading label field is
            # not column-aligned across EFIT and TokaMaker writers.
            ints = [parse(Int, m.match) for m in eachmatch(r"(?<![\d.eE+-])\d+(?![\d.eE+-])", lines[1])]
            length(ints) >= 2 || error("cannot read nw, nh from geqdsk header: $(lines[1])")
            nw, nh = ints[end-1], ints[end]
            nums = Float64[]
            for l in lines[2:end]
                for m in eachmatch(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?", l)
                    push!(nums, parse(Float64, m.match))
                end
            end
            off = 20 + 4nw + nw * nh
            q = abs.(nums[off+1:off+nw])
            imin = argmin(q)
            return nw, q[1], q[imin], (imin - 1) / (nw - 1), q[end]
        end
        function run_case(geqdsk::String, template::Dict{String,Any})
            t0 = time()
            nw, q0, qmin, psi_qmin, qedge = read_q_profile(geqdsk)
            dir = mktempdir()
            cp(geqdsk, joinpath(dir, "eq.geqdsk"))
            open(joinpath(dir, "gpec.toml"), "w") do io
                TOML.print(io, template)
            end
            inputs = TOML.parsefile(joinpath(dir, "gpec.toml"))
            eq_config = GPE.Equilibrium.EquilibriumConfig(inputs["Equilibrium"], dir)
            wall = GPE.Vacuum.WallShapeSettings(; (Symbol(k) => v for (k, v) in inputs["Wall"])...)
            ctrl = FFS.ForceFreeStatesControl(; (Symbol(k) => v for (k, v) in inputs["ForceFreeStates"])...)
            intr = FFS.ForceFreeStatesInternal(; dir_path=dir)
            intr.wall_settings = wall
            GPE.resolve_mode_space!(intr, ctrl)
            t_equil = @elapsed begin
                equil = GPE.Equilibrium.setup_equilibrium(eq_config, nothing)
                equil = GPE.maybe_reform_equilibrium(equil, eq_config, nothing, intr, ctrl, nothing)
            end
            t_prep = @elapsed metric, mats = GPE.prepare_force_free_states!(intr, ctrl, equil, GPE.KineticForces.KineticForcesControl(), nothing)
            t_int = @elapsed odet, props, chunks, S_left = FFS.eulerlagrange_integration(ctrl, equil, mats, intr)
            t_free = @elapsed begin
                free = FFS.free_run(odet, ctrl, equil, mats, intr)
                FFS.normalize_eigenfunctions!(odet, free.wt, equil.psio)
            end
            t_dp = 0.0
            if intr.msing > 0 && props !== nothing
                t_dp = @elapsed FFS.compute_delta_prime_matrix!(intr, props, chunks; wv=free.wv, psio=equil.psio, debug=false,
                    S_at_surface_left=S_left, ctrl=ctrl, equil=equil, mats=mats)
            end
            res = FFS.build_result(Symbol(ctrl.integrator), ctrl, equil, intr, metric, mats, odet, free, nothing, nothing)
            surfaces = join([@sprintf("%d@%.4f", s.m[1], s.psifac) for s in intr.sing], ";")
            dp = res.delta_prime === nothing ? "" : join([@sprintf("%.6g%+.6gi", real(z), imag(z)) for z in LinearAlgebra.diag(res.delta_prime.matrix)], ";")
            et = real.(free.et)
            rm(dir; recursive=true, force=true)
            return (status="ok", elapsed=time() - t0, nw, q0, qmin, psi_qmin, qedge, N=intr.numpert_total, msing=intr.msing, surfaces,
                total_steps=odet.total_steps, t_equil, t_prep, t_int, t_free, t_dp,
                et1=get(et, 1, NaN), et2=get(et, 2, NaN), et3=get(et, 3, NaN), dp, message="")
        end
        # Errors are turned into a row on the worker: a raised exception would carry method
        # instances the driver process cannot deserialize.
        function run_case_safe(geqdsk::String, template::Dict{String,Any})
            t0 = time()
            try
                return run_case(geqdsk, template)
            catch err
                msg = first(sprint(showerror, err), 400)
                return (status="failed", elapsed=time() - t0, nw=0, q0=NaN, qmin=NaN, psi_qmin=NaN, qedge=NaN, N=0, msing=0, surfaces="",
                    total_steps=0, t_equil=NaN, t_prep=NaN, t_int=NaN, t_free=NaN, t_dp=NaN, et1=NaN, et2=NaN, et3=NaN, dp="", message=msg)
            end
        end
        nothing  # the eval's return value travels back to the driver; a function object would not deserialize there
    end

    os_pids = Dict{Int,Int}()

    function spawn_worker()
        pid = only(addprocs(1; exeflags=exeflags))
        os_pids[pid] = remotecall_fetch(getpid, pid)
        remotecall_fetch(Core.eval, pid, Main, worker_imports)
        remotecall_fetch(Core.eval, pid, Main, worker_setup)
        remotecall_fetch(Core.eval, pid, Main, :(const TEMPLATE = $template))
        return pid
    end

    # A worker stuck inside a solve does not answer the cooperative shutdown and `rmprocs` then
    # throws; SIGKILL the OS process so one pathological equilibrium cannot end the whole sweep.
    function kill_worker(pid)
        try
            rmprocs(pid; waitfor=5)
        catch
            ospid = get(os_pids, pid, 0)
            if ospid > 0
                try
                    run(`kill -9 $ospid`)
                catch
                end
            end
            try
                rmprocs(pid; waitfor=0)
            catch
            end
        end
        delete!(os_pids, pid)
        return nothing
    end

    io = open(opts["out"], "w")
    println(io, HEADER)
    flush(io)
    write_row(idx, rel, r) = begin
        println(
            io,
            join(
                [idx, rel, r.status, @sprintf("%.1f", r.elapsed), r.nw, @sprintf("%.4f", r.q0), @sprintf("%.4f", r.qmin),
                    @sprintf("%.3f", r.psi_qmin), @sprintf("%.3f", r.qedge), r.N, r.msing, r.surfaces, r.total_steps,
                    @sprintf("%.2f", r.t_equil), @sprintf("%.2f", r.t_prep), @sprintf("%.2f", r.t_int), @sprintf("%.2f", r.t_free),
                    @sprintf("%.2f", r.t_dp), @sprintf("%.8e", r.et1), @sprintf("%.8e", r.et2), @sprintf("%.8e", r.et3), r.dp,
                    replace(r.message, "," => ";", "\n" => " ")], ",")
        )
        flush(io)
    end
    blank(status, elapsed, msg) = (status, elapsed, nw=0, q0=NaN, qmin=NaN, psi_qmin=NaN, qedge=NaN, N=0, msing=0, surfaces="", total_steps=0,
        t_equil=NaN, t_prep=NaN, t_int=NaN, t_free=NaN, t_dp=NaN, et1=NaN, et2=NaN, et3=NaN, dp="", message=msg)

    queue = Channel{Tuple{Int,String}}(length(files))
    for (i, f) in enumerate(files)
        put!(queue, (i, f))
    end
    close(queue)
    lock = ReentrantLock()

    @sync for w in 1:opts["workers"]
        @async begin
            pid = spawn_worker()
            for (idx, rel) in queue
                path = joinpath(opts["root"], rel)
                t0 = time()
                # The task must never throw: killing a timed-out worker makes its pending
                # fetch fail, and an exception here would escape @sync and end the sweep.
                task = @async try
                    remotecall_fetch(Core.eval, pid, Main, :(run_case_safe($path, TEMPLATE)))
                catch err
                    err
                end
                while !istaskdone(task) && time() - t0 < opts["timeout"]
                    sleep(1)
                end
                r = if istaskdone(task)
                    out = fetch(task)
                    out isa Exception ? blank("failed", time() - t0, first(sprint(showerror, out), 300)) : out
                else
                    kill_worker(pid)
                    pid = spawn_worker()
                    blank("timeout", time() - t0, "exceeded $(opts["timeout"]) s")
                end
                @lock lock write_row(idx, rel, r)
                @printf("%3d/%d %-8s %6.1fs  %s\n", idx, length(files), r.status, r.elapsed, rel)
            end
            kill_worker(pid)
        end
    end
    close(io)
    @info "Wrote $(opts["out"])"
end

main(ARGS)
