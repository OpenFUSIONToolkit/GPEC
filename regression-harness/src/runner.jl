"""
Runner: orchestrates checking out commits, running GPEC, and extracting results.
"""

"""Read the GPEC-only runtime from the timing file written by the subprocess."""
function _read_timing_file(path::String)::Float64
    if isfile(path)
        return parse(Float64, strip(read(path, String)))
    end
    return NaN
end

"""
Materialize the directory GPEC will actually run in.

With no `overrides`, the example deck runs in place (returns it untouched). With overrides,
the deck is copied to a throwaway temp dir and the named gpec.toml keys are patched there,
so one shared example can serve several cases (e.g. a collisionless variant via
`"KineticForces.nutype" => "zero"`). Override keys are dotted paths into the TOML; missing
intermediate tables are created. Returns `(rundir, is_temp)`; the caller removes the temp
tree when `is_temp`.
"""
function _materialize_rundir(example_path::String, overrides::Dict{String,Any})
    isempty(overrides) && return (example_path, false)
    rundir = joinpath(mktempdir(), "deck")
    cp(example_path, rundir)
    rm(joinpath(rundir, "gpec.h5"); force=true)   # drop any stale output copied along
    cfg = TOML.parsefile(joinpath(rundir, "gpec.toml"))
    for (dotted, val) in overrides
        ks = split(dotted, ".")
        d = cfg
        for k in ks[1:(end - 1)]
            d = get!(d, k, Dict{String,Any}())
        end
        d[ks[end]] = val
    end
    open(joinpath(rundir, "gpec.toml"), "w") do io
        TOML.print(io, cfg)
    end
    return (rundir, true)
end

const RUNNER_SCRIPT_TEMPLATE = """
using Pkg
%INSTANTIATE%
using GeneralizedPerturbedEquilibrium
t_start = time()
GeneralizedPerturbedEquilibrium.main([ARGS[1]])
elapsed = time() - t_start
# Write GPEC-only runtime to a file the harness reads back
open(ARGS[2], "w") do f
    println(f, elapsed)
end
"""

# Self-contained computation for the GGJ inner-layer reference benchmark.
# Runs the Galerkin solver on the Glasser & Wang 2020 Eq. 55 parameter set
# at γ = 1 + i and writes (Δ_odd, Δ_even) into a small HDF5 file at ARGS[1].
# Runtime is written to ARGS[2] for parity with the GPEC runner.
# `solve_inner` returns the named-field `InnerLayerResponse`; the historical
# delta_odd / delta_even slots map to interchange / tearing respectively, which
# preserves the numeric identity of each tracked quantity.
const COMPUTED_GGJ_SCRIPT_TEMPLATE = """
using Pkg
%INSTANTIATE%
using GeneralizedPerturbedEquilibrium
using GeneralizedPerturbedEquilibrium.InnerLayer
using HDF5
γ = ComplexF64(1.0, 1.0)
p = glasser_wang_2020_eq55()
t_start = time()
Δ = solve_inner(GGJModel(solver=:galerkin), p, γ)
elapsed = time() - t_start
h5open(ARGS[1], "w") do fid
    fid["ggj/delta_odd_real"]  = real(Δ.interchange)
    fid["ggj/delta_odd_imag"]  = imag(Δ.interchange)
    fid["ggj/delta_even_real"] = real(Δ.tearing)
    fid["ggj/delta_even_imag"] = imag(Δ.tearing)
end
open(ARGS[2], "w") do f
    println(f, elapsed)
end
"""

# GGJ rotated-ray backend at Q = 500i on the q=4 benchmark surface — a regime beyond the
# :galerkin backend. Builds the physical rate γ = 500i·Q₀ so inner_Q lands exactly on the
# imaginary axis at 500i, then writes the parity matching data. Runtime to ARGS[2] as usual.
# As in the galerkin template, the delta_odd / delta_even slots map to interchange / tearing.
const COMPUTED_GGJ_RAY_SCRIPT_TEMPLATE = """
using Pkg
%INSTANTIATE%
using GeneralizedPerturbedEquilibrium
using GeneralizedPerturbedEquilibrium.InnerLayer
using HDF5
p = q4_surface_benchmark()
γ = 500.0im * InnerLayer.GGJ.q0(p)
t_start = time()
Δ = solve_inner(GGJModel(solver=:ray), p, γ)
elapsed = time() - t_start
h5open(ARGS[1], "w") do fid
    fid["ggj/delta_odd_real"]  = real(Δ.interchange)
    fid["ggj/delta_odd_imag"]  = imag(Δ.interchange)
    fid["ggj/delta_even_real"] = real(Δ.tearing)
    fid["ggj/delta_even_imag"] = imag(Δ.tearing)
end
open(ARGS[2], "w") do f
    println(f, elapsed)
end
"""

# Self-contained separatrix-finder regression (PR #296). Loads a fixed-boundary EFIT whose
# computational box hugs the LCFS (eps=0.05 TokaMaker aspect-scan g-file): outside the prescribed
# LCFS the coil-vacuum flux turns back above the boundary value before the grid edge, so the old
# bracketed-Brent separatrix finder could not bracket psi=sibry and equilibrium setup threw. Calls
# setup_equilibrium directly and writes the leading equilibrium scalars: errors on the buggy code,
# passes with the Newton finder. Runtime is written to ARGS[2] for parity with the GPEC runner.
const COMPUTED_SEPARATRIX_SCRIPT_TEMPLATE = """
using Pkg
%INSTANTIATE%
using GeneralizedPerturbedEquilibrium
using GeneralizedPerturbedEquilibrium.Equilibrium
using HDF5
root = dirname(Base.active_project())
gfile = joinpath(root, "examples", "efit_fixedbdy_separatrix_example", "eq_eps0.0500000_k1.000_d0.000.geqdsk")
cfg = Equilibrium.EquilibriumConfig(; eq_type="efit", eq_filename=gfile)
t_start = time()
pe = Equilibrium.setup_equilibrium(cfg)
elapsed = time() - t_start
h5open(ARGS[1], "w") do fid
    fid["equil/psio"]  = pe.psio
    fid["equil/q0"]    = pe.params.q0
    fid["equil/q95"]   = pe.params.q95
    fid["equil/betat"] = pe.params.betat
    fid["equil/betan"] = pe.params.betan
end
open(ARGS[2], "w") do f
    println(f, elapsed)
end
"""

"""
Run GPEC for a single commit/ref and case. Dispatches to run_local for
the working tree or run_at_commit for a git ref.
"""
function run_commit(db::SQLite.DB, commit_hash::String, ref_name::String,
                    case_spec::CaseSpec, repo_root::String;
                    force::Bool=false, verbose::Bool=false,
                    no_instantiate::Bool=false)
    if case_spec.kind == "computed"
        if commit_hash == LOCAL_REF
            return run_computed_local(db, case_spec, repo_root;
                                      verbose=verbose, no_instantiate=no_instantiate)
        end
        return run_computed_at_commit(db, commit_hash, ref_name, case_spec, repo_root;
                                      force=force, verbose=verbose, no_instantiate=no_instantiate)
    end
    if commit_hash == LOCAL_REF
        return run_local(db, case_spec, repo_root;
                         force=force, verbose=verbose, no_instantiate=no_instantiate)
    end
    return run_at_commit(db, commit_hash, ref_name, case_spec, repo_root;
                         force=force, verbose=verbose, no_instantiate=no_instantiate)
end

"""
Pick the subprocess script template for a `kind="computed"` case.
"""
function _computed_script_template(case_spec::CaseSpec)
    if case_spec.name == "ggj_reference"
        return COMPUTED_GGJ_SCRIPT_TEMPLATE
    elseif case_spec.name == "ggj_ray_q500i"
        return COMPUTED_GGJ_RAY_SCRIPT_TEMPLATE
    elseif case_spec.name == "efit_fixedbdy_separatrix"
        return COMPUTED_SEPARATRIX_SCRIPT_TEMPLATE
    end
    error("No computed-script template registered for case '$(case_spec.name)'")
end

"""
Shared implementation for kind="computed" cases. Runs the case's
self-contained Julia script in a subprocess against `project_root` (so it can
import GeneralizedPerturbedEquilibrium), reads the resulting tempfile h5 with
`extract_quantities`, and returns `(extracted, runtime_s)`. Throws on failure
so callers can handle store_failed_run uniformly.
"""
function _execute_computed(case_spec::CaseSpec, project_root::String;
                           verbose::Bool, no_instantiate::Bool,
                           stderr_buf::IO)
    instantiate_line = no_instantiate ? "" : "Pkg.instantiate()"
    script_content = replace(_computed_script_template(case_spec),
                             "%INSTANTIATE%" => instantiate_line)
    tmpscript = tempname() * ".jl"
    h5path = tempname() * ".h5"
    timingfile = tempname() * ".timing"
    try
        write(tmpscript, script_content)
        if verbose
            run(pipeline(`julia --project=$project_root $tmpscript $h5path $timingfile`))
        else
            run(pipeline(`julia --project=$project_root $tmpscript $h5path $timingfile`,
                         stdout=devnull, stderr=stderr_buf))
        end
        runtime_s = _read_timing_file(timingfile)
        if !isfile(h5path)
            error("Computed case '$(case_spec.name)' produced no output h5")
        end
        extracted = extract_quantities(h5path, case_spec.quantities, runtime_s)
        return extracted, runtime_s
    finally
        rm(tmpscript; force=true)
        rm(h5path; force=true)
        rm(timingfile; force=true)
    end
end

"""
Run a kind="computed" case against the working tree.
"""
function run_computed_local(db::SQLite.DB, case_spec::CaseSpec, repo_root::String;
                            verbose::Bool=false, no_instantiate::Bool=false)
    delete_cached(db, LOCAL_REF, case_spec.name)
    date = Dates.format(Dates.now(), "yyyy-mm-ddTHH:MM:SS")
    @info "Running: $(case_spec.name) @ local (working tree, computed)"
    stderr_buf = IOBuffer()
    try
        extracted, runtime_s = _execute_computed(case_spec, repo_root;
                                                 verbose=verbose,
                                                 no_instantiate=no_instantiate,
                                                 stderr_buf=stderr_buf)
        store_run(db, LOCAL_REF, "local", date, "working tree", case_spec.name,
                  runtime_s, extracted)
        @info "  Completed in $(round(runtime_s, digits=3))s — $(length(extracted)) quantities extracted"
    catch e
        err_msg = if e isa ProcessFailedException
            stderr_str = String(take!(stderr_buf))
            isempty(stderr_str) ? "Subprocess failed" : stderr_str
        else
            sprint(showerror, e)
        end
        # Keep the tail of the message: Julia errors usually appear at the end
        # of the subprocess output, not the start (Pkg.instantiate output dominates the head).
        # `last` is unicode-safe and won't split a multibyte char like `err_msg[end-N:end]` could.
        err_msg_short = length(err_msg) > 2000 ? "..." * last(err_msg, 2000) : err_msg
        @warn "Run failed (local computed): $(first(err_msg_short, 200))"
        store_failed_run(db, LOCAL_REF, "local", date, "working tree", case_spec.name,
                         err_msg_short)
    end
end

"""
Run a kind="computed" case at a specific git commit via worktree.
"""
function run_computed_at_commit(db::SQLite.DB, commit_hash::String, ref_name::String,
                                case_spec::CaseSpec, repo_root::String;
                                force::Bool=false, verbose::Bool=false,
                                no_instantiate::Bool=false)
    if !force && is_cached(db, commit_hash, case_spec.name)
        info = get_run_info(db, commit_hash, case_spec.name)
        if info !== nothing
            @info "Cached: $(case_spec.name) @ $(info.commit_short) ($(info.commit_date))"
            return
        end
    end
    if force
        delete_cached(db, commit_hash, case_spec.name)
    end

    commit_info = get_commit_info(commit_hash, repo_root)
    @info "Running: $(case_spec.name) @ $(commit_info.short) ($(commit_info.date)) [computed]"
    @info "  $(commit_info.msg)"

    worktree_path = nothing
    stderr_buf = IOBuffer()
    try
        worktree_path = create_worktree(commit_hash, repo_root)
        extracted, runtime_s = _execute_computed(case_spec, worktree_path;
                                                 verbose=verbose,
                                                 no_instantiate=no_instantiate,
                                                 stderr_buf=stderr_buf)
        store_run(db, commit_hash, commit_info.short, commit_info.date,
                  commit_info.msg, case_spec.name, runtime_s, extracted)
        @info "  Completed in $(round(runtime_s, digits=3))s — $(length(extracted)) quantities extracted"
    catch e
        err_msg = if e isa ProcessFailedException
            stderr_str = String(take!(stderr_buf))
            isempty(stderr_str) ? "Subprocess failed" : stderr_str
        else
            sprint(showerror, e)
        end
        # Keep the tail of the message: Julia errors usually appear at the end
        # of the subprocess output, not the start (Pkg.instantiate output dominates the head).
        # `last` is unicode-safe and won't split a multibyte char like `err_msg[end-N:end]` could.
        err_msg_short = length(err_msg) > 2000 ? "..." * last(err_msg, 2000) : err_msg
        @warn "Run failed (computed) for $(commit_info.short): $(first(err_msg_short, 200))"
        store_failed_run(db, commit_hash, commit_info.short, commit_info.date,
                         commit_info.msg, case_spec.name, err_msg_short)
    finally
        if worktree_path !== nothing
            remove_worktree(worktree_path, repo_root)
        end
    end
end

"""
Run GPEC in the current working tree (uncommitted changes included).
Always re-runs (local results are never cached since the working tree is mutable).
"""
function run_local(db::SQLite.DB, case_spec::CaseSpec, repo_root::String;
                   force::Bool=false, verbose::Bool=false,
                   no_instantiate::Bool=false)
    # Always delete previous local results and re-run
    delete_cached(db, LOCAL_REF, case_spec.name)

    date = Dates.format(Dates.now(), "yyyy-mm-ddTHH:MM:SS")
    @info "Running: $(case_spec.name) @ local (working tree)"

    example_path = joinpath(repo_root, case_spec.example_dir)
    if !isdir(example_path)
        @warn "Example directory not found: $(case_spec.example_dir)"
        store_failed_run(db, LOCAL_REF, "local", date, "working tree", case_spec.name,
                         "Example directory not found: $(case_spec.example_dir)")
        return
    end

    tmpscript = nothing
    timingfile = nothing
    rundir = nothing
    rundir_is_temp = false
    stderr_buf = IOBuffer()

    try
        (rundir, rundir_is_temp) = _materialize_rundir(example_path, case_spec.overrides)

        instantiate_line = no_instantiate ? "" : "Pkg.instantiate()"
        script_content = replace(RUNNER_SCRIPT_TEMPLATE, "%INSTANTIATE%" => instantiate_line)
        tmpscript = tempname() * ".jl"
        timingfile = tempname() * ".timing"
        write(tmpscript, script_content)

        if verbose
            run(pipeline(`julia --project=$repo_root $tmpscript $rundir $timingfile`))
        else
            run(pipeline(`julia --project=$repo_root $tmpscript $rundir $timingfile`,
                         stdout=devnull, stderr=stderr_buf))
        end
        runtime_s = _read_timing_file(timingfile)

        h5path = joinpath(rundir, "gpec.h5")
        if !isfile(h5path)
            @warn "gpec.h5 not produced"
            store_failed_run(db, LOCAL_REF, "local", date, "working tree", case_spec.name,
                             "gpec.h5 not produced after successful run")
            return
        end

        extracted = extract_quantities(h5path, case_spec.quantities, runtime_s)
        store_run(db, LOCAL_REF, "local", date, "working tree", case_spec.name,
                  runtime_s, extracted)

        @info "  Completed in $(round(runtime_s, digits=1))s — $(length(extracted)) quantities extracted"

    catch e
        err_msg = if e isa ProcessFailedException
            stderr_str = String(take!(stderr_buf))
            isempty(stderr_str) ? "Subprocess failed (stderr was printed to terminal in verbose mode)" : stderr_str
        else
            sprint(showerror, e)
        end
        # Keep the tail of the message: Julia errors usually appear at the end
        # of the subprocess output, not the start (Pkg.instantiate output dominates the head).
        # `last` is unicode-safe and won't split a multibyte char like `err_msg[end-N:end]` could.
        err_msg_short = length(err_msg) > 2000 ? "..." * last(err_msg, 2000) : err_msg
        @warn "Run failed (local): $(first(err_msg_short, 200))"
        store_failed_run(db, LOCAL_REF, "local", date, "working tree", case_spec.name,
                         err_msg_short)
    finally
        if tmpscript !== nothing
            rm(tmpscript; force=true)
        end
        if timingfile !== nothing
            rm(timingfile; force=true)
        end
        if rundir_is_temp && rundir !== nothing
            rm(dirname(rundir); recursive=true, force=true)
        end
    end
end

"""
Run GPEC for a specific git commit via worktree. Stores results in the database.
Skips if already cached (unless force=true).
"""
function run_at_commit(db::SQLite.DB, commit_hash::String, ref_name::String,
                       case_spec::CaseSpec, repo_root::String;
                       force::Bool=false, verbose::Bool=false,
                       no_instantiate::Bool=false)
    # Check cache
    if !force && is_cached(db, commit_hash, case_spec.name)
        info = get_run_info(db, commit_hash, case_spec.name)
        if info !== nothing
            @info "Cached: $(case_spec.name) @ $(info.commit_short) ($(info.commit_date))"
            return
        end
    end

    # Delete existing cached data if force re-running
    if force
        delete_cached(db, commit_hash, case_spec.name)
    end

    # Get commit metadata
    commit_info = get_commit_info(commit_hash, repo_root)
    @info "Running: $(case_spec.name) @ $(commit_info.short) ($(commit_info.date))"
    @info "  $(commit_info.msg)"

    worktree_path = nothing
    tmpscript = nothing
    timingfile = nothing
    rundir = nothing
    rundir_is_temp = false
    stderr_buf = IOBuffer()

    try
        # Create worktree
        worktree_path = create_worktree(commit_hash, repo_root)

        # Check example directory exists in this commit
        example_path = joinpath(worktree_path, case_spec.example_dir)
        if !isdir(example_path)
            @warn "Example directory not found at commit $(commit_info.short): $(case_spec.example_dir)"
            store_failed_run(db, commit_hash, commit_info.short, commit_info.date,
                             commit_info.msg, case_spec.name,
                             "Example directory not found: $(case_spec.example_dir)")
            return
        end

        (rundir, rundir_is_temp) = _materialize_rundir(example_path, case_spec.overrides)

        # Write temp runner script
        instantiate_line = no_instantiate ? "" : "Pkg.instantiate()"
        script_content = replace(RUNNER_SCRIPT_TEMPLATE, "%INSTANTIATE%" => instantiate_line)
        tmpscript = tempname() * ".jl"
        timingfile = tempname() * ".timing"
        write(tmpscript, script_content)

        # Run GPEC in subprocess
        project_root = worktree_path

        if verbose
            run(pipeline(`julia --project=$project_root $tmpscript $rundir $timingfile`))
        else
            run(pipeline(`julia --project=$project_root $tmpscript $rundir $timingfile`,
                         stdout=devnull, stderr=stderr_buf))
        end
        runtime_s = _read_timing_file(timingfile)

        # Check for gpec.h5
        h5path = joinpath(rundir, "gpec.h5")
        if !isfile(h5path)
            @warn "gpec.h5 not produced at $(commit_info.short)"
            store_failed_run(db, commit_hash, commit_info.short, commit_info.date,
                             commit_info.msg, case_spec.name,
                             "gpec.h5 not produced after successful run")
            return
        end

        # Extract quantities
        extracted = extract_quantities(h5path, case_spec.quantities, runtime_s)

        # Store in database
        store_run(db, commit_hash, commit_info.short, commit_info.date,
                  commit_info.msg, case_spec.name, runtime_s, extracted)

        @info "  Completed in $(round(runtime_s, digits=1))s — $(length(extracted)) quantities extracted"

    catch e
        err_msg = if e isa ProcessFailedException
            stderr_str = String(take!(stderr_buf))
            isempty(stderr_str) ? "Subprocess failed (stderr was printed to terminal in verbose mode)" : stderr_str
        else
            sprint(showerror, e)
        end
        # Truncate for display and storage
        # Keep the tail of the message: Julia errors usually appear at the end
        # of the subprocess output, not the start (Pkg.instantiate output dominates the head).
        # `last` is unicode-safe and won't split a multibyte char like `err_msg[end-N:end]` could.
        err_msg_short = length(err_msg) > 2000 ? "..." * last(err_msg, 2000) : err_msg
        @warn "Run failed for $(commit_info.short): $(first(err_msg_short, 200))"
        store_failed_run(db, commit_hash, commit_info.short, commit_info.date,
                         commit_info.msg, case_spec.name, err_msg_short)
    finally
        # Clean up
        if tmpscript !== nothing
            rm(tmpscript; force=true)
        end
        if timingfile !== nothing
            rm(timingfile; force=true)
        end
        if rundir_is_temp && rundir !== nothing
            rm(dirname(rundir); recursive=true, force=true)
        end
        if worktree_path !== nothing
            remove_worktree(worktree_path, repo_root)
        end
    end
end
