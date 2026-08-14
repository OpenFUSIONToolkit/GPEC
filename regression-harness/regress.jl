#!/usr/bin/env julia

using Dates, HDF5, JSON, Printf, SHA, SQLite, Tables, TOML

const HARNESS_DIR = @__DIR__
const REPO_ROOT = abspath(joinpath(HARNESS_DIR, ".."))
const DEFAULT_DB_PATH = joinpath(HARNESS_DIR, ".regress_cache.sqlite")
const CASES_DIR = joinpath(HARNESS_DIR, "cases")

include("src/types.jl")
include("src/env.jl")
include("src/config.jl")
include("src/database.jl")
include("src/utils.jl")
include("src/extractor.jl")
include("src/runner.jl")
include("src/reporter.jl")

function parse_args(args)
    cases = String[]
    refs = String[]
    ref_range = nothing
    force = false
    list_cases = false
    show_qty = nothing
    show_case = nothing
    db_path = nothing
    verbose = false
    no_instantiate = false
    no_pin_manifest = false
    allow_env_mismatch = false
    fail_on_change = false
    help = false

    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "--help" || arg == "-h"
            help = true
            i += 1
        elseif arg == "--force"
            force = true
            i += 1
        elseif arg == "--list-cases"
            list_cases = true
            i += 1
        elseif arg == "--verbose"
            verbose = true
            i += 1
        elseif arg == "--no-instantiate"
            no_instantiate = true
            i += 1
        elseif arg == "--no-pin-manifest"
            no_pin_manifest = true
            i += 1
        elseif arg == "--allow-env-mismatch"
            allow_env_mismatch = true
            i += 1
        elseif arg == "--fail-on-change"
            fail_on_change = true
            i += 1
        elseif arg == "--cases" && i < length(args)
            cases = split(args[i+1], ",") |> collect .|> strip
            i += 2
        elseif arg == "--refs" && i < length(args)
            refs = split(args[i+1], ",") |> collect .|> strip
            i += 2
        elseif arg == "--ref-range" && i < length(args)
            ref_range = args[i+1]
            i += 2
        elseif arg == "--show" && i < length(args)
            show_qty = args[i+1]
            i += 2
        elseif arg == "--case" && i < length(args)
            show_case = args[i+1]
            i += 2
        elseif arg == "--db" && i < length(args)
            db_path = args[i+1]
            i += 2
        else
            error("Unknown argument: $arg. Use --help for usage.")
        end
    end

    return CLIOptions(cases, refs, ref_range, force, list_cases, show_qty, show_case, db_path, verbose,
        no_instantiate, no_pin_manifest, allow_env_mismatch, fail_on_change, help)
end

const HELP_TEXT = """
GPEC Regression Harness

Runs GPEC test cases across multiple git commits and compares numerical outputs
to detect unintended changes. Results are cached in a local SQLite database.
All cases in a single invocation share one git worktree (and one instantiate/
precompile) per commit, so batching cases into one command is substantially
faster than running them in separate invocations.

Usage:
    julia --project=regression-harness regression-harness/regress.jl [OPTIONS]

Options:
    --cases name1,name2     Comma-separated case names (default: all)
    --refs ref1,ref2,...    Comma-separated git refs (branches, tags, SHAs)
                            Use "local" to run on the current working tree
    --ref-range from..to   Expand to all commits in range via git rev-list
    --force                Re-run even if cached
    --list-cases           Print available cases and exit
    --show qty_name        Show historical values for a quantity (requires --case)
    --case name            Single case name (used with --show)
    --db path              Override database path
    --verbose              Print subprocess output
    --no-instantiate       Skip Pkg.instantiate() in subprocess
    --no-pin-manifest      Let each ref resolve its own package set (default: pin the working
                           tree's Manifest.toml into every worktree so source code is the only
                           variable in a comparison)
    --allow-env-mismatch   Reuse cached results produced in a different environment instead of
                           re-running them
    --fail-on-change       Exit non-zero if any tracked quantity changed (for CI use; a failed
                           run always exits non-zero regardless)
    --help                 Print this help message

Environment:
    GPEC_REGRESS_THREADS   Threads for GPEC subprocesses (default "auto" = all cores);
                           the actual count is recorded in each run's env fingerprint

Exit status:
    0  all runs completed (and, with --fail-on-change, nothing changed)
    1  a run failed, or a quantity changed under --fail-on-change

Examples:
    # Compare two refs
    julia --project=regression-harness regression-harness/regress.jl \\
        --cases solovev_n1 --refs develop,HEAD

    # Track across many versions
    julia --project=regression-harness regression-harness/regress.jl \\
        --cases solovev_n1 --refs v0.1.0,v0.2.0,develop

    # Expand a commit range
    julia --project=regression-harness regression-harness/regress.jl \\
        --cases solovev_n1 --ref-range develop~10..develop

    # List available cases
    julia --project=regression-harness regression-harness/regress.jl --list-cases

    # Show one quantity's history
    julia --project=regression-harness regression-harness/regress.jl \\
        --show et_real --case solovev_n1

    # Compare current working tree against develop
    julia --project=regression-harness regression-harness/regress.jl \\
        --cases solovev_n1 --refs develop,local
"""

function main(args=ARGS)
    opts = parse_args(args)

    if opts.help
        print(HELP_TEXT)
        return
    end

    # Load all available cases
    all_cases = load_all_cases(CASES_DIR)

    if opts.list_cases
        print_cases(all_cases)
        return
    end

    # Open database
    db_path = something(opts.db_path, DEFAULT_DB_PATH)
    db = open_database(db_path)

    try
        # Handle --show mode
        if opts.show_qty !== nothing
            if opts.show_case === nothing
                error("--show requires --case")
            end
            show_quantity_history(db, opts.show_case, opts.show_qty)
            return
        end

        # Resolve which cases to run
        case_specs = if isempty(opts.cases)
            collect(values(all_cases))
        else
            [all_cases[name] for name in opts.cases if haskey(all_cases, name) || error("Unknown case: $name")]
        end

        # Resolve refs
        resolved_refs = if opts.ref_range !== nothing
            expand_ref_range(opts.ref_range, REPO_ROOT)
        elseif !isempty(opts.refs)
            [resolve_ref(ref, REPO_ROOT) for ref in opts.refs]
        else
            error("Must specify --refs or --ref-range")
        end

        if isempty(resolved_refs)
            error("No commits resolved from the given refs")
        end

        warn_stale_refs(resolved_refs, REPO_ROOT)

        # Pin every ref to the working tree's package set unless asked not to, so that a
        # comparison varies source code alone.
        manifest_path = joinpath(REPO_ROOT, "Manifest.toml")
        pin_manifest = if opts.no_pin_manifest
            @warn "Manifest pinning disabled — refs may resolve different package sets, and differences below may not be caused by source changes"
            nothing
        elseif !isfile(manifest_path)
            @warn "No Manifest.toml in the working tree; cannot pin the package set. Run Pkg.instantiate() first."
            nothing
        else
            manifest_path
        end
        expected_key = opts.allow_env_mismatch ? nothing : expected_env_key(pin_manifest)

        n_failed = 0
        n_changed = 0

        # Run every case against each ref, grouped by ref so cases sharing a commit share
        # one worktree (and one Pkg.instantiate/precompile) instead of paying for it per case.
        for ref in resolved_refs
            run_cases_at_ref(db, ref, case_specs, REPO_ROOT;
                force=opts.force, verbose=opts.verbose,
                no_instantiate=opts.no_instantiate,
                pin_manifest=pin_manifest, expected_key=expected_key)
        end

        # Report, grouped by case as before
        for case_spec in case_specs
            println("\n", "="^64)
            println("Case: $(case_spec.name) — $(case_spec.description)")
            println("="^64)

            summary = if length(resolved_refs) == 2
                report_two_ref_comparison(db, case_spec,
                    resolved_refs[1], resolved_refs[2])
            else
                report_multi_ref(db, case_spec, resolved_refs)
            end
            n_failed += summary.n_failed
            n_changed += summary.n_changed
        end

        if n_failed > 0
            @error "$n_failed run(s) failed — see the reports above"
            exit(1)
        end
        if opts.fail_on_change && n_changed > 0
            @error "$n_changed quantity/quantities changed (--fail-on-change)"
            exit(1)
        end
    finally
        close_database(db)
    end
end

main()
