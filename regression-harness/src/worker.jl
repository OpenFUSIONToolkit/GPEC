"""
Warm workers: long-lived `julia` subprocesses that run case scripts one after another.

A fresh subprocess per case pays GPEC's load and first-use compilation on every case, which
dominates the runtime of most cases. A warm worker loads GPEC once per (project, build mode) and
runs each case script in a fresh anonymous module, so only the first case at a ref compiles the
shared code paths. Pass `--isolate` to fall back to one subprocess per case.
"""

"""
Script run by each warm worker. Signals `READY` once GPEC is loaded, then reads one request per
line from stdin (`script<TAB>logfile<TAB>args...`) and answers `OK` or `ERROR` on its original
stdout. Each case's stdout, stderr and log records go to its own logfile.
"""
const WORKER_SCRIPT = raw"""
using Pkg, Logging
ARGS[1] == "1" && Pkg.instantiate()
using GeneralizedPerturbedEquilibrium
const CHANNEL = stdout
println(CHANNEL, "READY")
flush(CHANNEL)
for line in eachline(stdin)
    fields = split(line, '\t')
    script, logfile, args = String(fields[1]), String(fields[2]), String.(fields[3:end])
    ok = open(logfile, "w") do io
        redirect_stdio(; stdout=io, stderr=io) do
            with_logger(ConsoleLogger(io)) do
                try
                    m = Module(:RegressCase)
                    Core.eval(m, :(const ARGS = $args))
                    Base.include(m, script)
                    true
                catch e
                    showerror(io, e, catch_backtrace())
                    println(io)
                    false
                end
            end
        end
    end
    println(CHANNEL, ok ? "OK" : "ERROR")
    flush(CHANNEL)
end
"""

"""
A case script failed; its output has been written to the caller's error buffer.
"""
struct ScriptFailed <: Exception end

"""
A running warm worker.

## Fields

  - `proc::Base.Process` — the subprocess, opened read/write on its stdin and stdout
  - `script::String` — path of the worker script, removed on shutdown
  - `log::String` — the worker's own stderr (instantiate and precompile output), "" when inherited
"""
struct WarmWorker
    proc::Base.Process
    script::String
    log::String
end

"""
The warm workers serving one execution lane at one project root, one per build mode.

## Fields

  - `project_root::String` — the worktree (or live checkout) the workers load GPEC from
  - `no_instantiate::Bool` — skip `Pkg.instantiate()` when a worker starts
  - `verbose::Bool` — pass worker and case output through to the terminal
  - `workers::Dict{Bool,WarmWorker}` — running workers keyed by `precompile_workload`
"""
struct WorkerSlot
    project_root::String
    no_instantiate::Bool
    verbose::Bool
    workers::Dict{Bool,WarmWorker}
end

WorkerSlot(project_root::String; no_instantiate::Bool=false, verbose::Bool=false) =
    WorkerSlot(project_root, no_instantiate, verbose, Dict{Bool,WarmWorker}())

_tail(s::AbstractString, n::Int=2000) = length(s) > n ? "..." * last(s, n) : s

"""
Next protocol word from a worker, skipping any stray output; "" once the worker has exited.
"""
function _await_reply(proc::Base.Process)::String
    while true
        line = readline(proc)
        line in ("READY", "OK", "ERROR") && return line
        eof(proc) && return ""
    end
end

"""
Start a worker for `case_spec`'s build mode and wait until it has loaded GPEC.
"""
function start_worker(slot::WorkerSlot, case_spec::CaseSpec)::WarmWorker
    script = tempname() * ".jl"
    write(script, WORKER_SCRIPT)
    log = slot.verbose ? "" : tempname() * ".worker.log"
    cmd = with_workload_preference(
        `julia --startup-file=no -t $SUBPROCESS_THREADS --project=$(slot.project_root) $script $(slot.no_instantiate ? "0" : "1")`,
        case_spec)
    proc = open(pipeline(cmd; stderr=slot.verbose ? stderr : log), "r+")
    if _await_reply(proc) != "READY"
        wait(proc)
        msg = isempty(log) ? "" : read(log, String)
        rm(script; force=true)
        isempty(log) || rm(log; force=true)
        error("warm worker failed to start (exit code $(proc.exitcode)):\n" * _tail(msg))
    end
    return WarmWorker(proc, script, log)
end

"""
The slot's worker for `case_spec`'s build mode, started on first use.
"""
get_worker!(slot::WorkerSlot, case_spec::CaseSpec) =
    get!(() -> start_worker(slot, case_spec), slot.workers, case_spec.precompile_workload)

"""
Stop a worker: close its stdin so its request loop ends, wait for it, and remove its files.
"""
function stop_worker(w::WarmWorker)
    process_running(w.proc) && close(w.proc.in)
    wait(w.proc)
    rm(w.script; force=true)
    isempty(w.log) || rm(w.log; force=true)
end

function shutdown!(slot::WorkerSlot)
    foreach(stop_worker, values(slot.workers))
    empty!(slot.workers)
end
shutdown!(::Nothing) = nothing

"""
Run one case script in the slot's warm worker. On failure the case log goes to `stderr_buf` and
`ScriptFailed` is thrown; a worker that dies mid-case is discarded, so the next case starts a new one.
"""
function run_in_worker(slot::WorkerSlot, script::String, args::Vector{String}, case_spec::CaseSpec, stderr_buf::IO)
    w = get_worker!(slot, case_spec)
    logfile = tempname() * ".case.log"
    try
        reply = try
            println(w.proc, join([script; logfile; args], '\t'))
            flush(w.proc)
            _await_reply(w.proc)
        catch e
            e isa Base.IOError || rethrow()
            ""
        end
        out = isfile(logfile) ? read(logfile, String) : ""
        slot.verbose && print(out)
        reply == "OK" && return
        if reply != "ERROR"
            delete!(slot.workers, case_spec.precompile_workload)
            stop_worker(w)
            out *= "\nwarm worker exited (code $(w.proc.exitcode)) during this case"
        end
        write(stderr_buf, out)
        throw(ScriptFailed())
    finally
        rm(logfile; force=true)
    end
end
