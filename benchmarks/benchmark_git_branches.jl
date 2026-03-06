#!/usr/bin/env julia
"""
benchmark_git_branches.jl - Generic Git branch benchmarking tool for GPEC

Compares performance between two Git branches or commits by running a specified
example multiple times to warm up Julia's JIT compiler, then measuring runtime.

# Usage

Basic comparison of two branches:
```bash
julia benchmarks/benchmark_git_branches.jl --example examples/DIIID-like_ideal_example --branch1 develop --branch2 feature-branch
```

Compare specific commits:
```bash
julia benchmarks/benchmark_git_branches.jl --example examples/DIIID-like_ideal_example --commit1 abc123 --commit2 def456
```

Compare current state against old state of same branch:
```bash
julia benchmarks/benchmark_git_branches.jl --example examples/DIIID-like_ideal_example --branch1 develop --commit1 HEAD~10 --branch2 develop
```

# Options

- `--example PATH`: Path to example directory (required)
- `--branch1 NAME`: First branch name (default: "develop")
- `--branch2 NAME`: Second branch name (default: current branch)
- `--commit1 HASH`: Specific commit for branch1 (default: branch HEAD)
- `--commit2 HASH`: Specific commit for branch2 (default: branch HEAD)
- `--runs N`: Number of warm runs to average (default: 2)
- `--output FILE`: Write results to file (default: stdout only)

# Requirements

- Example directory must contain a `gpec.toml` config file runnable via: `using GeneralizedPerturbedEquilibrium; GeneralizedPerturbedEquilibrium.main(["path"])`
- Working directory must be clean or changes will be stashed during branch switching

# Output Metrics

For each branch/commit, reports:
- Eigenmode energy (et[1]) from jpec.h5
- Integration steps from jpec.h5
- Runtime (averaged over warm runs)
- Git commit hash

# Example Output

```
============================================================
GPEC Branch Benchmark Comparison
============================================================
Example: examples/DIIID-like_ideal_example

Branch 1: develop @ abc123
  Eigenmode energy (et[1]):  1.7005
  Integration steps:          911
  Runtime (warmed, avg):      145.3 s

Branch 2: feature-branch @ def456
  Eigenmode energy (et[1]):  1.7072
  Integration steps:          136
  Runtime (warmed, avg):      142.8 s

Comparison:
  Eigenmode Δ:    +0.0067 (+0.4%)
  Steps Δ:        -775 (-85.1%)
  Runtime Δ:      -2.5 s (-1.7% faster)
```
"""

using Pkg, HDF5, Printf

# Parse command-line arguments
function parse_args(args)
    options = Dict(
        "example" => nothing,
        "branch1" => "develop",
        "branch2" => nothing,  # Default to current branch
        "commit1" => nothing,
        "commit2" => nothing,
        "runs" => 2,
        "output" => nothing
    )

    i = 1
    while i <= length(args)
        if startswith(args[i], "--")
            key = args[i][3:end]
            if haskey(options, key)
                if i < length(args) && !startswith(args[i+1], "--")
                    value = args[i+1]
                    if key == "runs"
                        options[key] = parse(Int, value)
                    else
                        options[key] = value
                    end
                    i += 2
                else
                    error("Option --$key requires a value")
                end
            else
                error("Unknown option: $(args[i])")
            end
        else
            error("Unexpected argument: $(args[i])")
        end
    end

    # Validate required arguments
    if options["example"] === nothing
        error("--example is required")
    end

    # Default branch2 to current branch if not specified
    if options["branch2"] === nothing
        options["branch2"] = strip(read(`git branch --show-current`, String))
    end

    return options
end

# Get commit hash for a branch/commit reference
function get_commit_hash(ref)
    return strip(read(`git rev-parse --short $ref`, String))
end

# Check if working directory is clean
function check_clean_workdir()
    status = read(`git status --porcelain`, String)
    if !isempty(status)
        println("Warning: Working directory has uncommitted changes. Stashing...")
        run(`git stash`)
        return true
    end
    return false
end

# Restore stashed changes
function restore_stash(was_stashed)
    if was_stashed
        println("Restoring stashed changes...")
        run(`git stash pop`)
    end
end

# Checkout a specific branch/commit
function checkout_ref(branch, commit)
    if commit !== nothing
        println("Checking out $branch @ $commit...")
        run(`git checkout $commit`)
    else
        println("Checking out $branch...")
        run(`git checkout $branch`)
    end
end

# Run the example benchmark
function run_example_benchmark(example_path, num_runs)
    abs_example_path = abspath(example_path)
    project_root = abspath(joinpath(example_path, "../.."))
    println("\nRunning example: $abs_example_path")
    println("Warming up with $(num_runs + 1) runs...")

    # Write a temp script so `using GeneralizedPerturbedEquilibrium` is at top-level in each subprocess.
    # Pkg.instantiate() handles branches whose environments haven't been resolved yet.
    tmpscript = tempname() * ".jl"
    write(tmpscript, "using Pkg; Pkg.instantiate(); using GeneralizedPerturbedEquilibrium; GeneralizedPerturbedEquilibrium.main([ARGS[1]])\n")

    try
        # First run for JIT compilation
        println("\n[1/$(num_runs+1)] First run (JIT compilation)...")
        run(`julia --project=$project_root $tmpscript $abs_example_path`)

        # Warm runs for timing
        runtimes = Float64[]
        for i in 1:num_runs
            println("\n[$((i+1))/$(num_runs+1)] Warm run $i...")
            t_start = time()
            run(`julia --project=$project_root $tmpscript $abs_example_path`)
            runtime = time() - t_start
            push!(runtimes, runtime)
            println("  Runtime: $(round(runtime, digits=2)) s")
        end

        # Extract metrics from gpec.h5
        gpec_path = joinpath(abs_example_path, "gpec.h5")
        if !isfile(gpec_path)
            error("gpec.h5 not found at $gpec_path")
        end

        h5 = h5open(gpec_path, "r")
        et = read(h5["vacuum/et"])
        nsteps = read(h5["integration/nstep"])
        close(h5)

        avg_runtime = sum(runtimes) / length(runtimes)

        return (
            eigenvalue = real(et[1]),
            steps = nsteps,
            runtime = avg_runtime
        )
    finally
        rm(tmpscript, force=true)
    end
end

# Main benchmarking function
function benchmark_branches(options)
    println("="^60)
    println("GPEC Branch Benchmark Comparison")
    println("="^60)
    println("Example: $(options["example"])")
    println()

    # Store original branch
    original_branch = strip(read(`git branch --show-current`, String))
    was_stashed = check_clean_workdir()

    results = Dict()

    try
        # Benchmark branch 1
        println("\n" * "="^60)
        println("Benchmarking Branch 1: $(options["branch1"])")
        println("="^60)
        checkout_ref(options["branch1"], options["commit1"])
        commit1 = get_commit_hash("HEAD")
        results["branch1"] = (
            name = options["branch1"],
            commit = commit1,
            metrics = run_example_benchmark(options["example"], options["runs"])
        )

        # Benchmark branch 2
        println("\n" * "="^60)
        println("Benchmarking Branch 2: $(options["branch2"])")
        println("="^60)
        checkout_ref(options["branch2"], options["commit2"])
        commit2 = get_commit_hash("HEAD")
        results["branch2"] = (
            name = options["branch2"],
            commit = commit2,
            metrics = run_example_benchmark(options["example"], options["runs"])
        )

    finally
        # Restore original state
        println("\nRestoring original branch: $original_branch")
        run(`git checkout $original_branch`)
        restore_stash(was_stashed)
    end

    # Print comparison
    println("\n" * "="^60)
    println("BENCHMARK RESULTS")
    println("="^60)
    println()

    r1 = results["branch1"]
    r2 = results["branch2"]

    println("Branch 1: $(r1.name) @ $(r1.commit)")
    @printf("  Eigenmode energy (et[1]):  %.4f\n", r1.metrics.eigenvalue)
    @printf("  Integration steps:          %d\n", r1.metrics.steps)
    @printf("  Runtime (warmed, avg):      %.2f s\n", r1.metrics.runtime)
    println()

    println("Branch 2: $(r2.name) @ $(r2.commit)")
    @printf("  Eigenmode energy (et[1]):  %.4f\n", r2.metrics.eigenvalue)
    @printf("  Integration steps:          %d\n", r2.metrics.steps)
    @printf("  Runtime (warmed, avg):      %.2f s\n", r2.metrics.runtime)
    println()

    # Comparison
    println("Comparison (Branch 2 vs Branch 1):")

    ev_delta = r2.metrics.eigenvalue - r1.metrics.eigenvalue
    ev_pct = 100 * ev_delta / r1.metrics.eigenvalue
    @printf("  Eigenmode Δ:    %+.4f (%+.2f%%)\n", ev_delta, ev_pct)

    steps_delta = r2.metrics.steps - r1.metrics.steps
    steps_pct = 100 * steps_delta / r1.metrics.steps
    @printf("  Steps Δ:        %+d (%+.1f%%)\n", steps_delta, steps_pct)

    time_delta = r2.metrics.runtime - r1.metrics.runtime
    time_pct = 100 * time_delta / r1.metrics.runtime
    speedup_desc = time_delta < 0 ? "faster" : "slower"
    @printf("  Runtime Δ:      %+.2f s (%+.1f%% %s)\n", time_delta, abs(time_pct), speedup_desc)

    println("\n" * "="^60)

    # Write to output file if requested
    if options["output"] !== nothing
        open(options["output"], "w") do f
            write(f, "# GPEC Benchmark Comparison\n")
            write(f, "Example: $(options["example"])\n\n")
            write(f, "## Branch 1: $(r1.name) @ $(r1.commit)\n")
            write(f, @sprintf("- Eigenmode energy: %.4f\n", r1.metrics.eigenvalue))
            write(f, @sprintf("- Integration steps: %d\n", r1.metrics.steps))
            write(f, @sprintf("- Runtime: %.2f s\n\n", r1.metrics.runtime))
            write(f, "## Branch 2: $(r2.name) @ $(r2.commit)\n")
            write(f, @sprintf("- Eigenmode energy: %.4f\n", r2.metrics.eigenvalue))
            write(f, @sprintf("- Integration steps: %d\n", r2.metrics.steps))
            write(f, @sprintf("- Runtime: %.2f s\n\n", r2.metrics.runtime))
            write(f, "## Comparison\n")
            write(f, @sprintf("- Eigenmode Δ: %+.4f (%+.2f%%)\n", ev_delta, ev_pct))
            write(f, @sprintf("- Steps Δ: %+d (%+.1f%%)\n", steps_delta, steps_pct))
            write(f, @sprintf("- Runtime Δ: %+.2f s (%+.1f%% %s)\n", time_delta, abs(time_pct), speedup_desc))
        end
        println("Results written to: $(options["output"])")
    end
end

# Entry point
if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) == 0
        println("""
        Usage: julia benchmarks/benchmark_git_branches.jl [OPTIONS]

        Required:
          --example PATH        Path to example directory

        Optional:
          --branch1 NAME        First branch (default: develop)
          --branch2 NAME        Second branch (default: current branch)
          --commit1 HASH        Specific commit for branch1
          --commit2 HASH        Specific commit for branch2
          --runs N              Number of warm runs (default: 2)
          --output FILE         Write results to file

        Examples:
          # Compare two branches
          julia benchmarks/benchmark_git_branches.jl --example examples/DIIID-like_ideal_example --branch1 develop --branch2 feature-branch

          # Compare develop now vs 10 commits ago
          julia benchmarks/benchmark_git_branches.jl --example examples/DIIID-like_ideal_example --branch1 develop --commit1 HEAD~10 --branch2 develop

          # Compare specific commits
          julia benchmarks/benchmark_git_branches.jl --example examples/DIIID-like_ideal_example --commit1 abc123 --commit2 def456
        """)
        exit(1)
    end

    options = parse_args(ARGS)
    benchmark_branches(options)
end
