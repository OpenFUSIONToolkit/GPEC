# NOTE: Driver script for fixed_axis=true vs fixed_axis=false comparison. Remove before merging.
#
# Compares the original Glasser initialization (U₁=0, U₂=I; fixed_axis=true) against
# the Frobenius initialization (fixed_axis=false, the new default). Saves outputs to
# gpec_fixed.h5 and gpec_free.h5 in the example directory, then generates the comparison
# plot via compare_fixed_axis.jl.
#
# Usage:
#   julia --project=. benchmarks/run_fixed_axis_comparison.jl

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using GeneralizedPerturbedEquilibrium
using TOML
using Printf

# ── Configuration ──────────────────────────────────────────────────────────────
const EXAMPLE_DIR = joinpath(@__DIR__, "..", "examples", "DIIID-like_ideal_example")
# ───────────────────────────────────────────────────────────────────────────────

function run_scenario(fixed_axis::Bool, output_filename::String, psilow::Float64, nn::Int)
    label = fixed_axis ? "fixed_axis=true" : "fixed_axis=false"
    @info "Running $label → $output_filename"

    tmp_dir = mktempdir()
    try
        # Symlink all files except gpec.toml (rewritten below) and any prior h5 outputs
        for f in readdir(EXAMPLE_DIR)
            (f == "gpec.toml" || endswith(f, ".h5")) && continue
            symlink(joinpath(EXAMPLE_DIR, f), joinpath(tmp_dir, f))
        end

        cfg = TOML.parsefile(joinpath(EXAMPLE_DIR, "gpec.toml"))
        cfg["Equilibrium"]["psilow"] = psilow
        cfg["ForceFreeStates"]["nn_low"]  = nn
        cfg["ForceFreeStates"]["nn_high"] = nn
        cfg["ForceFreeStates"]["fixed_axis"] = fixed_axis
        cfg["ForceFreeStates"]["HDF5_filename"] = output_filename
        cfg["ForceFreeStates"]["force_termination"] = true  # skip PE for speed
        open(joinpath(tmp_dir, "gpec.toml"), "w") do io
            TOML.print(io, cfg)
        end

        t = @elapsed GeneralizedPerturbedEquilibrium.main([tmp_dir])
        @info "$label completed in $(round(t; digits=1)) s"

        src = joinpath(tmp_dir, output_filename)
        dst = joinpath(EXAMPLE_DIR, output_filename)
        cp(src, dst; force=true)
        @info "Saved $dst"
    finally
        rm(tmp_dir; recursive=true, force=true)
    end
end

example_tag = basename(EXAMPLE_DIR)
psilow_tag  = replace(@sprintf("%.0e", 1e-4), "+" => "", "-0" => "-")

global PLOT_TAG = "$(example_tag)_psilow$(psilow_tag)_n1"
global EXAMPLE_DIR_FOR_PLOT = EXAMPLE_DIR

@info "=== Case: $PLOT_TAG ==="
run_scenario(true,  "gpec_fixed.h5", 1e-4, 1)
run_scenario(false, "gpec_free.h5",  1e-4, 1)

@info "Running comparison plot for $PLOT_TAG..."
include(joinpath(@__DIR__, "compare_fixed_axis.jl"))
