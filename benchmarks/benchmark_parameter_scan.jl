#!/usr/bin/env julia
"""
benchmark_parameter_scan.jl — General GPEC parameter scan and eigenmode comparison tool

Runs GPEC multiple times with a single TOML parameter swept over a list of values,
then produces overplot comparisons of eigenmode radial profiles, poloidal spectra,
and edge stability scans.  Designed to be general: any scalar TOML parameter can be
swept, e.g. `edge_layer_width`, EL tolerance, or vacuum grid resolution.

# Usage

```bash
# Sweep edge_layer_width over three values
julia --project=. benchmarks/benchmark_parameter_scan.jl \\
    --example examples/DIIID-like_ideal_example \\
    --param ForceFreeStates.edge_layer_width \\
    --values "1e-4,1e-3,5e-3"

# Use custom labels
julia --project=. benchmarks/benchmark_parameter_scan.jl \\
    --example examples/DIIID-like_ideal_example \\
    --param ForceFreeStates.edge_layer_width \\
    --values "1e-4,1e-3,5e-3" \\
    --labels "tight,medium,wide"

# Skip re-running GPEC (plot from existing output directories)
julia --project=. benchmarks/benchmark_parameter_scan.jl \\
    --no-run \\
    --output-dir /tmp/my_scan \\
    --param ForceFreeStates.edge_layer_width \\
    --values "1e-4,1e-3,5e-3"
```

# Options

- `--example PATH`     Path to example directory containing `gpec.toml` (required unless --no-run)
- `--param KEY`        Dot-path TOML key to sweep, e.g. `ForceFreeStates.edge_layer_width`
- `--values LIST`      Comma-separated numeric values to test
- `--labels LIST`      Comma-separated legend labels (default: param=value)
- `--output-dir PATH`  Directory for per-run subdirectories (default: <example_dir>/scan_<param>)
- `--no-run`           Skip GPEC runs; read existing gpec.h5 files from output-dir subdirectories
- `--modes LIST`       Comma-separated m values for radial profile plot (default: 1,2,3,4,5)

# Output

All plots are written to `<output-dir>/plots/`:
  - `edge_scan_comparison.png`     — et vs q overplot (numerical stability check)
  - `mode_profiles.png`            — ξ_m(ψ) radial profiles for requested m values
  - `poloidal_spectrum.png`        — |ξ(m)| at truncation point
  - `summary.txt`                  — Table of et[1], psilim, qlim per run
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using GeneralizedPerturbedEquilibrium
using GeneralizedPerturbedEquilibrium: Analysis
using HDF5
using TOML
using Plots
using Printf

project_root = joinpath(@__DIR__, "..")

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

function parse_scan_args(raw_args)
    opts = Dict{String,Any}(
        "example"    => nothing,
        "param"      => nothing,
        "values"     => nothing,
        "labels"     => nothing,
        "output-dir" => nothing,
        "no-run"     => false,
        "modes"      => "1,2,3,4,5",
    )
    i = 1
    while i <= length(raw_args)
        arg = raw_args[i]
        if arg == "--no-run"
            opts["no-run"] = true
        elseif startswith(arg, "--") && i < length(raw_args)
            opts[arg[3:end]] = raw_args[i+1]
            i += 1
        end
        i += 1
    end
    return opts
end

opts = parse_scan_args(ARGS)

param_key   = opts["param"]
values_str  = opts["values"]
labels_opt  = opts["labels"]
no_run      = opts["no-run"]
modes_str   = opts["modes"]

isnothing(param_key)  && error("--param is required")
isnothing(values_str) && error("--values is required")

# Parse values (floats)
param_values = parse.(Float64, strip.(split(values_str, ",")))
nvals = length(param_values)

# Labels default to "param=value"
param_short = split(param_key, ".")[end]   # e.g. "edge_layer_width"
labels = isnothing(labels_opt) ?
    ["$param_short=$(v)" for v in param_values] :
    strip.(split(labels_opt, ","))
length(labels) == nvals || error("--labels count ($(length(labels))) must match --values count ($nvals)")

# Modes for radial profile plot
modes_vec = parse.(Int, strip.(split(modes_str, ",")))

println("="^70)
println("GPEC Parameter Scan")
println("="^70)
println("  Parameter: $param_key")
println("  Values:    $(join(param_values, ", "))")
println("  Labels:    $(join(labels, ", "))")
println()

# ---------------------------------------------------------------------------
# Output directory setup
# ---------------------------------------------------------------------------

if no_run
    isnothing(opts["output-dir"]) && error("--output-dir is required with --no-run")
    scan_dir = abspath(opts["output-dir"])
else
    example_dir = abspath(opts["example"])
    isdir(example_dir) || error("Example directory not found: $example_dir")
    param_slug = replace(param_short, r"[^a-zA-Z0-9_]" => "_")
    scan_dir = isnothing(opts["output-dir"]) ?
        joinpath(@__DIR__, "scan_$(param_slug)") :
        abspath(opts["output-dir"])
end

plots_dir = joinpath(scan_dir, "plots")
mkpath(plots_dir)

# ---------------------------------------------------------------------------
# TOML patching helpers
# ---------------------------------------------------------------------------

"""
Patch a nested TOML dict at a dot-separated key path, e.g. "ForceFreeStates.edge_layer_width".
Returns a deep-copied dict with the leaf value set.
"""
function patch_toml(base::Dict, param_path::String, value)
    patched = deepcopy(base)
    parts = split(param_path, ".")
    node = patched
    for part in parts[1:end-1]
        node = node[part]
    end
    node[parts[end]] = value
    return patched
end

# ---------------------------------------------------------------------------
# Per-value run loop
# ---------------------------------------------------------------------------

run_dirs  = String[]
h5paths   = String[]
runtimes  = Float64[]

if !no_run
    base_toml_path = joinpath(example_dir, "gpec.toml")
    isfile(base_toml_path) || error("gpec.toml not found in $example_dir")
    base_toml = TOML.parsefile(base_toml_path)

    # Files to copy from example dir (everything except .toml and existing .h5/.png)
    copy_exts = [".geqdsk", ".dat", ".gfile", ".jl", ".txt", ".h5"]
    files_to_copy = filter(readdir(example_dir)) do f
        any(endswith(f, ext) for ext in copy_exts) && f != "gpec.h5"
    end

    tmpscript = tempname() * ".jl"
    write(tmpscript, """
        using Pkg; Pkg.instantiate()
        using GeneralizedPerturbedEquilibrium
        GeneralizedPerturbedEquilibrium.main([ARGS[1]])
        """)

    for (i, (val, lbl)) in enumerate(zip(param_values, labels))
        run_dir = joinpath(scan_dir, "run_$(i)")
        mkpath(run_dir)
        push!(run_dirs, run_dir)

        # Copy support files
        for f in files_to_copy
            cp(joinpath(example_dir, f), joinpath(run_dir, f); force=true)
        end

        # Write patched toml
        patched = patch_toml(base_toml, param_key, val)
        open(joinpath(run_dir, "gpec.toml"), "w") do io
            TOML.print(io, patched)
        end

        h5path = joinpath(run_dir, "gpec.h5")
        push!(h5paths, h5path)

        println("Run $i/$nvals: $lbl")
        println("  Directory: $run_dir")

        t0 = time()
        try
            run(`julia --project=$project_root $tmpscript $run_dir`)
        catch e
            @warn "Run $i failed: $e"
            push!(runtimes, NaN)
            continue
        end
        push!(runtimes, time() - t0)
        println(@sprintf("  Runtime: %.1f s", runtimes[end]))
        println()
    end
    rm(tmpscript; force=true)
else
    # Collect existing run directories and h5 files
    for i in 1:nvals
        run_dir = joinpath(scan_dir, "run_$(i)")
        push!(run_dirs, run_dir)
        push!(h5paths, joinpath(run_dir, "gpec.h5"))
        push!(runtimes, NaN)
    end
end

# Filter to successful runs (h5 file exists)
valid = isfile.(h5paths)
if !all(valid)
    @warn "Missing gpec.h5 for runs: $(findall(.!valid))"
end
h5paths_ok = h5paths[valid]
labels_ok  = labels[valid]

isempty(h5paths_ok) && error("No successful runs to plot.")

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

println("="^70)
println("Summary")
println("="^70)
@printf("  %-30s  %10s  %10s  %10s  %8s\n", "Label", "et[1]", "psilim", "qlim", "time(s)")
println("  " * "-"^72)

summary_lines = String[]
for (i, (h5path, lbl)) in enumerate(zip(h5paths, labels))
    if !isfile(h5path)
        @printf("  %-30s  %10s\n", lbl, "FAILED")
        continue
    end
    et1, psilim, qlim = h5open(h5path, "r") do f
        real(read(f["vacuum/et"])[1]),
        read(f["info/psilim"]),
        read(f["info/qlim"])
    end
    rt = isnan(runtimes[i]) ? "  -" : @sprintf("%.1f", runtimes[i])
    line = @sprintf("  %-30s  %10.4f  %10.6f  %10.4f  %8s", lbl, et1, psilim, qlim, rt)
    println(line)
    push!(summary_lines, line)
end
println()

open(joinpath(plots_dir, "summary.txt"), "w") do io
    println(io, "GPEC Parameter Scan: $param_key")
    println(io, "Values: $(join(param_values, ", "))")
    println(io)
    for l in summary_lines; println(io, l); end
end

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

println("Generating plots → $plots_dir")

# Plot 1: Edge stability scan — total energy vs q
p1 = Analysis.ForceFreeStates.plot_edge_scan_comparison(
    h5paths_ok, labels_ok;
    save_path = joinpath(plots_dir, "edge_scan_comparison.png")
)
println("  Saved: edge_scan_comparison.png")

# Plot 2: Radial profiles of individual ξ_m modes
p2 = Analysis.ForceFreeStates.plot_mode_radial_profiles_comparison(
    h5paths_ok, labels_ok;
    modes     = modes_vec,
    save_path = joinpath(plots_dir, "mode_profiles.png")
)
println("  Saved: mode_profiles.png")

# Plot 3: Poloidal spectrum |ξ(m)| at truncation point
p3 = Analysis.ForceFreeStates.plot_poloidal_spectrum_at_truncation(
    h5paths_ok, labels_ok;
    save_path = joinpath(plots_dir, "poloidal_spectrum.png")
)
println("  Saved: poloidal_spectrum.png")

println()
println("All plots saved to: $plots_dir")
