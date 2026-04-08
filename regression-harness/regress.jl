#!/usr/bin/env julia

using Dates, HDF5, JSON, Printf, SHA, SQLite, Tables, TOML

const HARNESS_DIR = @__DIR__
const REPO_ROOT = abspath(joinpath(HARNESS_DIR, ".."))
const DEFAULT_DB_PATH = joinpath(HARNESS_DIR, ".regress_cache.sqlite")
const CASES_DIR = joinpath(HARNESS_DIR, "cases")

include("src/types.jl")
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

    return CLIOptions(cases, refs, ref_range, force, list_cases, show_qty, show_case, db_path, verbose, no_instantiate, help)
end

const HELP_TEXT = """
GPEC Regression Harness

Runs GPEC test cases across multiple git commits and compares numerical outputs
to detect unintended changes. Results are cached in a local SQLite database.

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
    --help                 Print this help message

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

        # Run each case at each commit
        for case_spec in case_specs
            println("\n", "="^64)
            println("Case: $(case_spec.name) — $(case_spec.description)")
            println("="^64)

            for ref in resolved_refs
                run_commit(db, ref.commit_hash, ref.name, case_spec, REPO_ROOT;
                           force=opts.force, verbose=opts.verbose,
                           no_instantiate=opts.no_instantiate)
            end

            # Report
            if length(resolved_refs) == 1
                report_multi_ref(db, case_spec, resolved_refs)
            elseif length(resolved_refs) == 2
                report_two_ref_comparison(db, case_spec,
                    resolved_refs[1], resolved_refs[2])
            else
                report_multi_ref(db, case_spec, resolved_refs)
            end
        end
    finally
        close_database(db)
    end
end

main()
