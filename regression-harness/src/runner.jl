"""
Runner: orchestrates checking out commits, running GPEC, and extracting results.
"""

"""
Epilogue appended to every subprocess script: records the GPEC-only runtime *and* the
environment that produced it (Julia, host, resolved package set, thread counts) as `key=value`
lines. Measured inside the subprocess, after `Pkg.instantiate()` has resolved the environment,
so it describes what actually ran rather than what the harness intended to run.
"""
const RUNINFO_EPILOGUE = """
using SHA
using LinearAlgebra: BLAS
let manifest = joinpath(dirname(Base.active_project()), "Manifest.toml")
    open(ARGS[2], "w") do f
        println(f, "runtime_s=", elapsed)
        println(f, "julia_version=", string(VERSION))
        println(f, "os_arch=", Sys.MACHINE)
        println(f, "manifest_sha=", isfile(manifest) ? bytes2hex(SHA.sha256(read(manifest))) : "")
        println(f, "nthreads=", Threads.nthreads())
        println(f, "blas_threads=", BLAS.get_num_threads())
    end
end
"""

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
%RUNINFO%
"""

# Self-contained computation for the GGJ inner-layer reference benchmark.
# Runs the Galerkin solver on the Glasser & Wang 2020 Eq. 55 parameter set
# at γ = 1 + i and writes (Δ_odd, Δ_even) into a small HDF5 file at ARGS[1].
# Runtime and environment are written to ARGS[2] by the shared run-info epilogue.
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
%RUNINFO%
"""

# GGJ rotated-ray backend at Q = 500i on the q=4 benchmark surface — a regime beyond the
# :galerkin backend. Builds the physical rate γ = 500i·Q₀ so inner_Q lands exactly on the
# imaginary axis at 500i, then writes the parity matching data. Runtime and environment are
# written to ARGS[2] by the shared run-info epilogue.
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
%RUNINFO%
"""

# Self-contained separatrix-finder regression (PR #296). Loads a fixed-boundary EFIT whose
# computational box hugs the LCFS (eps=0.05 TokaMaker aspect-scan g-file): outside the prescribed
# LCFS the coil-vacuum flux turns back above the boundary value before the grid edge, so the old
# bracketed-Brent separatrix finder could not bracket psi=sibry and equilibrium setup threw. Calls
# setup_equilibrium directly and writes the leading equilibrium scalars: errors on the buggy code,
# passes with the Newton finder. Runtime and environment are written to ARGS[2] by the run-info epilogue.
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
    fid["Equilibrium/psio"]  = pe.psio
    fid["Equilibrium/q0"]    = pe.params.q0
    fid["Equilibrium/q95"]   = pe.params.q95
    fid["Equilibrium/betat"] = pe.params.betat
    fid["Equilibrium/betan"] = pe.params.betan
end
%RUNINFO%
"""

"""
Expand a subprocess script template: the optional `Pkg.instantiate()` call and the run-info
epilogue that records runtime and environment.
"""
function _render_script(template::String; no_instantiate::Bool)::String
    rendered = replace(template, "%INSTANTIATE%" => (no_instantiate ? "" : "Pkg.instantiate()"))
    return replace(rendered, "%RUNINFO%" => RUNINFO_EPILOGUE)
end

"""
Run GPEC for a single commit/ref and case. Dispatches to run_local for
the working tree or run_at_commit for a git ref.

`pin_manifest` is the path to a resolved `Manifest.toml` copied into each worktree so every ref
runs against the same package set; `nothing` disables pinning. `expected_key` is the environment
key a fresh run will carry — a cached run whose key differs is re-run rather than reused.
"""
function run_commit(db::SQLite.DB, commit_hash::String, ref_name::String,
                    case_spec::CaseSpec, repo_root::String;
                    force::Bool=false, verbose::Bool=false,
                    no_instantiate::Bool=false,
                    pin_manifest::Union{String,Nothing}=nothing,
                    expected_key::Union{String,Nothing}=nothing)
    if case_spec.kind == "computed"
        if commit_hash == LOCAL_REF
            return run_computed_local(db, case_spec, repo_root;
                                      verbose=verbose, no_instantiate=no_instantiate,
                                      pin_manifest=pin_manifest)
        end
        return run_computed_at_commit(db, commit_hash, ref_name, case_spec, repo_root;
                                      force=force, verbose=verbose, no_instantiate=no_instantiate,
                                      pin_manifest=pin_manifest, expected_key=expected_key)
    end
    if commit_hash == LOCAL_REF
        return run_local(db, case_spec, repo_root;
                         force=force, verbose=verbose, no_instantiate=no_instantiate,
                         pin_manifest=pin_manifest)
    end
    return run_at_commit(db, commit_hash, ref_name, case_spec, repo_root;
                         force=force, verbose=verbose, no_instantiate=no_instantiate,
                         pin_manifest=pin_manifest, expected_key=expected_key)
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
                           stderr_buf::IO, pin_manifest::Union{String,Nothing}=nothing)
    script_content = _render_script(_computed_script_template(case_spec); no_instantiate=no_instantiate)
    tmpscript = tempname() * ".jl"
    h5path = tempname() * ".h5"
    runinfo_file = tempname() * ".runinfo"
    try
        write(tmpscript, script_content)
        if verbose
            run(pipeline(`julia --project=$project_root $tmpscript $h5path $runinfo_file`))
        else
            run(pipeline(`julia --project=$project_root $tmpscript $h5path $runinfo_file`,
                         stdout=devnull, stderr=stderr_buf))
        end
        runtime_s, fingerprint = read_runinfo(runinfo_file, pin_manifest !== nothing)
        isempty(fingerprint.julia_version) && error("subprocess wrote no run-info metadata — does the script template end with %RUNINFO%?")
        _warn_pin_broken(pin_manifest, fingerprint, case_spec.name)
        if !isfile(h5path)
            error("Computed case '$(case_spec.name)' produced no output h5")
        end
        extracted = extract_quantities(h5path, case_spec.quantities, runtime_s)
        return extracted, runtime_s, fingerprint
    finally
        rm(tmpscript; force=true)
        rm(h5path; force=true)
        rm(runinfo_file; force=true)
    end
end

"""
Add a remedy hint when a run failed because the pinned Manifest is missing a direct dependency
declared at the checked-out commit — the one incompatibility `Pkg.instantiate()` refuses to run under.
Detects on the full error text, since tail-keeping truncation can drop Pkg's ERROR line from `short_err`.
"""
function _hint_pin_incompatible(full_err::AbstractString, short_err::AbstractString)::String
    occursin("is a direct dependency, but does not appear in the manifest", full_err) || return String(short_err)
    return "Pinned Manifest lacks a direct dependency declared at this commit — re-run with --no-pin-manifest " *
           "to let this ref resolve its own package set.\n" * short_err
end

"""
Run a kind="computed" case against the working tree.
"""
function run_computed_local(db::SQLite.DB, case_spec::CaseSpec, repo_root::String;
                            verbose::Bool=false, no_instantiate::Bool=false,
                            pin_manifest::Union{String,Nothing}=nothing)
    delete_cached(db, LOCAL_REF, case_spec.name)
    date = Dates.format(Dates.now(), "yyyy-mm-ddTHH:MM:SS")
    @info "Running: $(case_spec.name) @ local (working tree, computed)"
    stderr_buf = IOBuffer()
    try
        extracted, runtime_s, fingerprint = _execute_computed(case_spec, repo_root;
                                                              verbose=verbose,
                                                              no_instantiate=no_instantiate,
                                                              stderr_buf=stderr_buf,
                                                              pin_manifest=pin_manifest)
        store_run(db, LOCAL_REF, "local", date, "working tree", case_spec.name,
                  runtime_s, extracted; fingerprint=fingerprint)
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
        err_msg_short = _hint_pin_incompatible(err_msg, length(err_msg) > 2000 ? "..." * last(err_msg, 2000) : err_msg)
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
                                no_instantiate::Bool=false,
                                pin_manifest::Union{String,Nothing}=nothing,
                                expected_key::Union{String,Nothing}=nothing)
    if !force && is_cached(db, commit_hash, case_spec.name; expected_key=expected_key)
        info = get_run_info(db, commit_hash, case_spec.name)
        if info !== nothing
            @info "Cached: $(case_spec.name) @ $(info.commit_short) ($(info.commit_date))"
            return
        end
    end
    _warn_env_invalidated(db, commit_hash, case_spec.name, expected_key, force)
    if force
        delete_cached(db, commit_hash, case_spec.name)
    end

    commit_info = get_commit_info(commit_hash, repo_root)
    @info "Running: $(case_spec.name) @ $(commit_info.short) ($(commit_info.date)) [computed]"
    @info "  $(commit_info.msg)"

    worktree_path = nothing
    stderr_buf = IOBuffer()
    try
        worktree_path = create_worktree(commit_hash, repo_root; pin_manifest_from=pin_manifest)
        extracted, runtime_s, fingerprint = _execute_computed(case_spec, worktree_path;
                                                              verbose=verbose,
                                                              no_instantiate=no_instantiate,
                                                              stderr_buf=stderr_buf,
                                                              pin_manifest=pin_manifest)
        store_run(db, commit_hash, commit_info.short, commit_info.date,
                  commit_info.msg, case_spec.name, runtime_s, extracted; fingerprint=fingerprint)
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
        err_msg_short = _hint_pin_incompatible(err_msg, length(err_msg) > 2000 ? "..." * last(err_msg, 2000) : err_msg)
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
Warn if a pinned package set did not survive `Pkg.instantiate()`.

Contingency insurance rather than a description of current behavior: on Julia 1.11/1.12,
instantiate never rewrites an out-of-sync pinned Manifest — it warns and proceeds, or errors when
the commit declares a direct dependency the Manifest lacks (which fails the run loudly). Should a
future Pkg re-resolve in place instead, this catches the pin silently not holding.
"""
function _warn_pin_broken(pin_manifest::Union{String,Nothing}, fp::EnvFingerprint, label::AbstractString)
    pin_manifest === nothing && return
    isempty(fp.manifest_sha) && return
    pinned_sha = file_sha256(pin_manifest)
    (isempty(pinned_sha) || pinned_sha == fp.manifest_sha) && return
    @warn "Pinned Manifest was re-resolved for $label — its package set differs from the working tree" pinned = pinned_sha[1:8] resolved = fp.manifest_sha[1:8]
end

"""
Explain a cache miss caused by the environment rather than by absence.

A cached run exists for this (commit, case) but was produced under a different Julia, host, or
package set — the exact situation that used to be silently reused and reported as a physics
regression. Says so, once, before re-running.
"""
function _warn_env_invalidated(db::SQLite.DB, commit_hash::String, case_name::String,
                               expected_key::Union{String,Nothing}, force::Bool)
    (force || expected_key === nothing) && return
    stored = cached_env_key(db, commit_hash, case_name)
    stored == expected_key && return
    if stored === nothing
        is_cached(db, commit_hash, case_name) || return
        @warn "Cached result for $case_name predates environment fingerprinting — re-running"
    else
        @warn "Cached result for $case_name was produced in a different environment — re-running" stored_key = stored current_key = expected_key
    end
end

"""
Run GPEC in the current working tree (uncommitted changes included).
Always re-runs (local results are never cached since the working tree is mutable).
"""
function run_local(db::SQLite.DB, case_spec::CaseSpec, repo_root::String;
                   force::Bool=false, verbose::Bool=false,
                   no_instantiate::Bool=false,
                   pin_manifest::Union{String,Nothing}=nothing)
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
    runinfo_file = nothing
    rundir = nothing
    rundir_is_temp = false
    stderr_buf = IOBuffer()

    try
        (rundir, rundir_is_temp) = _materialize_rundir(example_path, case_spec.overrides)

        script_content = _render_script(RUNNER_SCRIPT_TEMPLATE; no_instantiate=no_instantiate)
        tmpscript = tempname() * ".jl"
        runinfo_file = tempname() * ".runinfo"
        write(tmpscript, script_content)

        if verbose
            run(pipeline(`julia --project=$repo_root $tmpscript $rundir $runinfo_file`))
        else
            run(pipeline(`julia --project=$repo_root $tmpscript $rundir $runinfo_file`,
                         stdout=devnull, stderr=stderr_buf))
        end
        runtime_s, fingerprint = read_runinfo(runinfo_file, pin_manifest !== nothing)
        isempty(fingerprint.julia_version) && error("subprocess wrote no run-info metadata — does the script template end with %RUNINFO%?")
        _warn_pin_broken(pin_manifest, fingerprint, case_spec.name)

        h5path = joinpath(rundir, "gpec.h5")
        if !isfile(h5path)
            @warn "gpec.h5 not produced"
            store_failed_run(db, LOCAL_REF, "local", date, "working tree", case_spec.name,
                             "gpec.h5 not produced after successful run")
            return
        end

        extracted = extract_quantities(h5path, case_spec.quantities, runtime_s)
        store_run(db, LOCAL_REF, "local", date, "working tree", case_spec.name,
                  runtime_s, extracted; fingerprint=fingerprint)

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
        err_msg_short = _hint_pin_incompatible(err_msg, length(err_msg) > 2000 ? "..." * last(err_msg, 2000) : err_msg)
        @warn "Run failed (local): $(first(err_msg_short, 200))"
        store_failed_run(db, LOCAL_REF, "local", date, "working tree", case_spec.name,
                         err_msg_short)
    finally
        if tmpscript !== nothing
            rm(tmpscript; force=true)
        end
        if runinfo_file !== nothing
            rm(runinfo_file; force=true)
        end
        if rundir_is_temp && rundir !== nothing
            rm(dirname(rundir); recursive=true, force=true)
        end
    end
end

"""
Run GPEC for a specific git commit via worktree. Stores results in the database.
Skips if already cached in the same environment (unless force=true).
"""
function run_at_commit(db::SQLite.DB, commit_hash::String, ref_name::String,
                       case_spec::CaseSpec, repo_root::String;
                       force::Bool=false, verbose::Bool=false,
                       no_instantiate::Bool=false,
                       pin_manifest::Union{String,Nothing}=nothing,
                       expected_key::Union{String,Nothing}=nothing)
    # Check cache
    if !force && is_cached(db, commit_hash, case_spec.name; expected_key=expected_key)
        info = get_run_info(db, commit_hash, case_spec.name)
        if info !== nothing
            @info "Cached: $(case_spec.name) @ $(info.commit_short) ($(info.commit_date))"
            return
        end
    end
    _warn_env_invalidated(db, commit_hash, case_spec.name, expected_key, force)

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
    runinfo_file = nothing
    rundir = nothing
    rundir_is_temp = false
    stderr_buf = IOBuffer()

    try
        # Create worktree, pinning the package set when the caller supplied a Manifest
        worktree_path = create_worktree(commit_hash, repo_root; pin_manifest_from=pin_manifest)

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
        script_content = _render_script(RUNNER_SCRIPT_TEMPLATE; no_instantiate=no_instantiate)
        tmpscript = tempname() * ".jl"
        runinfo_file = tempname() * ".runinfo"
        write(tmpscript, script_content)

        # Run GPEC in subprocess
        project_root = worktree_path

        if verbose
            run(pipeline(`julia --project=$project_root $tmpscript $rundir $runinfo_file`))
        else
            run(pipeline(`julia --project=$project_root $tmpscript $rundir $runinfo_file`,
                         stdout=devnull, stderr=stderr_buf))
        end
        runtime_s, fingerprint = read_runinfo(runinfo_file, pin_manifest !== nothing)
        isempty(fingerprint.julia_version) && error("subprocess wrote no run-info metadata — does the script template end with %RUNINFO%?")
        _warn_pin_broken(pin_manifest, fingerprint, commit_info.short)

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
                  commit_info.msg, case_spec.name, runtime_s, extracted; fingerprint=fingerprint)

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
        err_msg_short = _hint_pin_incompatible(err_msg, length(err_msg) > 2000 ? "..." * last(err_msg, 2000) : err_msg)
        @warn "Run failed for $(commit_info.short): $(first(err_msg_short, 200))"
        store_failed_run(db, commit_hash, commit_info.short, commit_info.date,
                         commit_info.msg, case_spec.name, err_msg_short)
    finally
        # Clean up
        if tmpscript !== nothing
            rm(tmpscript; force=true)
        end
        if runinfo_file !== nothing
            rm(runinfo_file; force=true)
        end
        if rundir_is_temp && rundir !== nothing
            rm(dirname(rundir); recursive=true, force=true)
        end
        if worktree_path !== nothing
            remove_worktree(worktree_path, repo_root)
        end
    end
end
