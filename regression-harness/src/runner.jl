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

"""
Run GPEC for a single commit/ref and case. Dispatches to run_local for
the working tree or run_at_commit for a git ref.
"""
function run_commit(db::SQLite.DB, commit_hash::String, ref_name::String,
                    case_spec::CaseSpec, repo_root::String;
                    force::Bool=false, verbose::Bool=false,
                    no_instantiate::Bool=false)
    if commit_hash == LOCAL_REF
        return run_local(db, case_spec, repo_root;
                         force=force, verbose=verbose, no_instantiate=no_instantiate)
    end
    return run_at_commit(db, commit_hash, ref_name, case_spec, repo_root;
                         force=force, verbose=verbose, no_instantiate=no_instantiate)
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
    stderr_buf = IOBuffer()

    try
        instantiate_line = no_instantiate ? "" : "Pkg.instantiate()"
        script_content = replace(RUNNER_SCRIPT_TEMPLATE, "%INSTANTIATE%" => instantiate_line)
        tmpscript = tempname() * ".jl"
        timingfile = tempname() * ".timing"
        write(tmpscript, script_content)

        if verbose
            run(pipeline(`julia --project=$repo_root $tmpscript $example_path $timingfile`))
        else
            run(pipeline(`julia --project=$repo_root $tmpscript $example_path $timingfile`,
                         stdout=devnull, stderr=stderr_buf))
        end
        runtime_s = _read_timing_file(timingfile)

        h5path = joinpath(example_path, "gpec.h5")
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
            isempty(stderr_str) ? "Subprocess failed (no stderr captured)" : stderr_str
        else
            sprint(showerror, e)
        end
        err_msg_short = length(err_msg) > 2000 ? err_msg[1:2000] * "..." : err_msg
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

        # Write temp runner script
        instantiate_line = no_instantiate ? "" : "Pkg.instantiate()"
        script_content = replace(RUNNER_SCRIPT_TEMPLATE, "%INSTANTIATE%" => instantiate_line)
        tmpscript = tempname() * ".jl"
        timingfile = tempname() * ".timing"
        write(tmpscript, script_content)

        # Run GPEC in subprocess
        project_root = worktree_path

        if verbose
            run(pipeline(`julia --project=$project_root $tmpscript $example_path $timingfile`))
        else
            run(pipeline(`julia --project=$project_root $tmpscript $example_path $timingfile`,
                         stdout=devnull, stderr=stderr_buf))
        end
        runtime_s = _read_timing_file(timingfile)

        # Check for gpec.h5
        h5path = joinpath(example_path, "gpec.h5")
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
            isempty(stderr_str) ? "Subprocess failed (no stderr captured)" : stderr_str
        else
            sprint(showerror, e)
        end
        # Truncate for display and storage
        err_msg_short = length(err_msg) > 2000 ? err_msg[1:2000] * "..." : err_msg
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
        if worktree_path !== nothing
            remove_worktree(worktree_path, repo_root)
        end
    end
end
