#!/usr/bin/env julia
"""
    run_sensitivity.jl — STRIDE Parameter Sensitivity Study

Systematically sweep every tunable parameter and record Δ' values to
characterize convergence, stability, and identify optimal settings.

Usage:
    julia --project=../.. run_sensitivity.jl                    # Full study
    julia --project=../.. run_sensitivity.jl --category ode     # One category
    julia --project=../.. run_sensitivity.jl --category mode    # Mode truncation only
    julia --project=../.. run_sensitivity.jl --dry-run          # List all runs
    julia --project=../.. run_sensitivity.jl --equil 147131     # One equilibrium only
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, "../.."))

using GeneralizedPerturbedEquilibrium
using HDF5
using TOML
using Printf

const SCRIPT_DIR = @__DIR__
const OUTPUT_H5 = joinpath(SCRIPT_DIR, "sensitivity_results.h5")

# ── Equilibrium definitions ─────────────────────────────────────────────

struct EquilDef
    label::String
    geqdsk::String
    psilow::Float64
    psihigh::Float64
end

const EQUILIBRIA = [
    EquilDef("147131",
        expanduser("~/Desktop/plasma/GPEC/docs/examples/DIIID_ideal_example/g147131.02300_DIIID_KEFIT"),
        1e-4, 0.993),
    EquilDef("LAR_0.4072",
        joinpath(dirname(SCRIPT_DIR), "LAR_epsilon_scan/equilibria/TJ_epsilon_scan_0.4072.geqdsk"),
        0.01, 0.995),
]

const N_VALUES = [1, 2]

# ── Baseline configuration ──────────────────────────────────────────────

function baseline_config(eq::EquilDef, n::Int, run_dir::String)
    Dict(
        "Equilibrium" => Dict(
            "eq_type" => "efit",
            "eq_filename" => eq.geqdsk,
            "jac_type" => "hamada",
            "grid_type" => "ldp",
            "psilow" => eq.psilow,
            "psihigh" => eq.psihigh,
            "mpsi" => 128,
            "mtheta" => 256,
            "etol" => 1e-7,
        ),
        "Wall" => Dict(
            "shape" => "conformal",
            "a" => 20.0,
        ),
        "ForceFreeStates" => Dict(
            "bal_flag" => false,
            "mat_flag" => true,
            "ode_flag" => true,
            "vac_flag" => true,
            "mer_flag" => true,
            "qlow" => 1.02,
            "qhigh" => 1e3,
            "sing_start" => 0,
            "nn_low" => n,
            "nn_high" => n,
            "delta_mlow" => 8,
            "delta_mhigh" => 8,
            "delta_mband" => 0,
            "mthvac" => 512,
            "thmax0" => 1,
            "eulerlagrange_tolerance" => 1e-12,
            "singfac_min" => 1e-4,
            "ucrit" => 1e4,
            "sing_order" => 6,
            "use_parallel" => true,
            "verbose" => false,
            "force_termination" => true,
            "write_outputs_to_HDF5" => true,
            "HDF5_filename" => joinpath(run_dir, "gpec.h5"),
            "save_interval" => 3,
            "force_wv_symmetry" => true,
            "use_double64_bvp" => true,
        ),
    )
end

# ── Parameter sweep definitions ──────────────────────────────────────────

struct ParamSweep
    category::String
    param_path::Vector{String}   # e.g., ["ForceFreeStates", "sing_order"]
    values::Vector{Any}
    label::String                # short name for HDF5 group
end

function build_sweeps()
    sweeps = ParamSweep[]

    # Category 1: ODE Integration
    push!(sweeps, ParamSweep("ode", ["ForceFreeStates", "eulerlagrange_tolerance"],
        [1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13, 1e-14], "el_tol"))
    push!(sweeps, ParamSweep("ode", ["ForceFreeStates", "singfac_min"],
        [1e-2, 1e-3, 1e-4, 1e-5, 1e-6], "singfac_min"))
    push!(sweeps, ParamSweep("ode", ["ForceFreeStates", "sing_order"],
        [2, 4, 6, 8, 10, 12], "sing_order"))
    push!(sweeps, ParamSweep("ode", ["ForceFreeStates", "ucrit"],
        [1e2, 1e3, 1e4, 1e5, 1e6], "ucrit"))

    # Category 2: Mode Truncation
    # delta_mlow and delta_mhigh swept together
    push!(sweeps, ParamSweep("mode", ["ForceFreeStates", "delta_m_symmetric"],
        [2, 4, 6, 8, 10, 12, 16], "delta_m"))
    push!(sweeps, ParamSweep("mode", ["ForceFreeStates", "delta_mband"],
        [0, 2, 4, 8], "delta_mband"))

    # Category 3: Equilibrium Grid
    push!(sweeps, ParamSweep("grid", ["Equilibrium", "mpsi"],
        [32, 64, 128, 256, 512], "mpsi"))
    push!(sweeps, ParamSweep("grid", ["Equilibrium", "mtheta"],
        [64, 128, 256, 512], "mtheta"))
    push!(sweeps, ParamSweep("grid", ["Equilibrium", "etol"],
        [1e-5, 1e-6, 1e-7, 1e-8, 1e-9], "etol"))

    # Category 4: Integration Domain
    push!(sweeps, ParamSweep("domain", ["Equilibrium", "psilow"],
        [1e-2, 1e-3, 1e-4, 1e-5], "psilow"))
    push!(sweeps, ParamSweep("domain", ["Equilibrium", "psihigh"],
        [0.98, 0.99, 0.993, 0.995, 0.999], "psihigh"))
    push!(sweeps, ParamSweep("domain", ["ForceFreeStates", "qhigh"],
        [3.6, 5.0, 10.0, 100.0, 1e3], "qhigh"))

    # Category 5: Vacuum / Wall
    push!(sweeps, ParamSweep("vacuum", ["ForceFreeStates", "mthvac"],
        [128, 256, 512, 960, 1920], "mthvac"))
    push!(sweeps, ParamSweep("vacuum", ["Wall", "a"],
        [1.5, 5.0, 10.0, 20.0, 100.0], "wall_a"))

    # Category 6: Algorithmic
    push!(sweeps, ParamSweep("algo", ["ForceFreeStates", "force_wv_symmetry"],
        [true, false], "force_wv_sym"))
    push!(sweeps, ParamSweep("algo", ["ForceFreeStates", "use_double64_bvp"],
        [true, false], "double64_bvp"))

    return sweeps
end

# ── Run a single configuration ───────────────────────────────────────────

function run_single(config::Dict)
    run_dir = mktempdir(; prefix="sens_")
    config["ForceFreeStates"]["HDF5_filename"] = joinpath(run_dir, "gpec.h5")

    open(joinpath(run_dir, "gpec.toml"), "w") do io
        TOML.print(io, config)
    end

    t = @elapsed begin
        try
            GeneralizedPerturbedEquilibrium.main([run_dir])
        catch e
            @warn "Run failed" exception=(e, catch_backtrace())
            rm(run_dir; force=true, recursive=true)
            return nothing
        end
    end

    h5_path = joinpath(run_dir, "gpec.h5")
    if !isfile(h5_path)
        rm(run_dir; force=true, recursive=true)
        return nothing
    end

    result = h5open(h5_path, "r") do f
        msing = read(f, "singular/msing")
        m_sing = read(f, "singular/m")
        q_sing = read(f, "singular/q")
        psi_sing = read(f, "singular/psi")
        dp_mat = haskey(f, "singular/delta_prime_matrix") ? read(f, "singular/delta_prime_matrix") : zeros(ComplexF64, 0, 0)
        ep = read(f, "vacuum/ep")
        ev = read(f, "vacuum/ev")
        et = read(f, "vacuum/et")
        nstep_saved = haskey(f, "integration/nstep") ? read(f, "integration/nstep") : 0
        nstep_total = haskey(f, "integration/nstep_total") ? read(f, "integration/nstep_total") : 0

        Dict(
            "msing" => msing,
            "m_sing" => m_sing,
            "q_sing" => q_sing,
            "psi_sing" => psi_sing,
            "dp_matrix" => dp_mat,
            "ep" => ep,
            "ev" => ev,
            "et" => et,
            "nstep_saved" => nstep_saved,
            "nstep_total" => nstep_total,
            "runtime" => t,
        )
    end

    rm(run_dir; force=true, recursive=true)
    return result
end

# ── Store results to HDF5 ────────────────────────────────────────────────

function store_result!(h5_path::String, group_path::String, result::Dict)
    h5open(h5_path, isfile(h5_path) ? "r+" : "w") do f
        # Delete existing group if present (re-run safety)
        if haskey(f, group_path)
            delete_object(f, group_path)
        end
        g = create_group(f, group_path)
        g["msing"] = result["msing"]
        g["m_sing"] = result["m_sing"]
        g["q_sing"] = result["q_sing"]
        g["psi_sing"] = result["psi_sing"]
        g["dp_matrix"] = result["dp_matrix"]
        g["ep"] = result["ep"]
        g["ev"] = result["ev"]
        g["et"] = result["et"]
        g["nstep_saved"] = result["nstep_saved"]
        g["nstep_total"] = result["nstep_total"]
        g["runtime"] = result["runtime"]
    end
end

# ── Apply a parameter sweep value to a config ────────────────────────────

function apply_param!(config::Dict, sweep::ParamSweep, value)
    if sweep.label == "delta_m"
        # Special: set both delta_mlow and delta_mhigh
        config["ForceFreeStates"]["delta_mlow"] = value
        config["ForceFreeStates"]["delta_mhigh"] = value
    else
        section = sweep.param_path[1]
        key = sweep.param_path[2]
        config[section][key] = value
    end
end

function value_label(v)
    if v isa Float64
        if v >= 1.0 && v == round(v)
            return @sprintf("%.1f", v)
        else
            return @sprintf("%.0e", v)
        end
    elseif v isa Bool
        return v ? "true" : "false"
    else
        return string(v)
    end
end

# ── Main ─────────────────────────────────────────────────────────────────

function main()
    dry_run = "--dry-run" in ARGS
    category_filter = nothing
    equil_filter = nothing
    for (i, a) in enumerate(ARGS)
        if a == "--category" && i < length(ARGS)
            category_filter = ARGS[i+1]
        elseif a == "--equil" && i < length(ARGS)
            equil_filter = ARGS[i+1]
        end
    end

    sweeps = build_sweeps()
    if category_filter !== nothing
        sweeps = filter(s -> s.category == category_filter, sweeps)
    end

    equils = EQUILIBRIA
    if equil_filter !== nothing
        equils = filter(e -> occursin(equil_filter, e.label), equils)
    end

    # Count total runs
    total = 0
    for eq in equils, n in N_VALUES
        total += 1  # baseline
        for sweep in sweeps
            total += length(sweep.values)
        end
    end

    if dry_run
        println("Parameter Sensitivity Study: $total total runs")
        println("\nEquilibria: $(join([e.label for e in equils], ", "))")
        println("n values: $N_VALUES")
        println("\nSweeps:")
        for s in sweeps
            println("  [$(s.category)] $(s.label): $(length(s.values)) values = $(s.values)")
        end
        return
    end

    @info "Parameter Sensitivity Study: $total runs → $OUTPUT_H5"

    # Remove old results
    isfile(OUTPUT_H5) && rm(OUTPUT_H5)

    run_idx = 0
    for eq in equils
        if !isfile(eq.geqdsk)
            @warn "Skipping $(eq.label): geqdsk not found at $(eq.geqdsk)"
            continue
        end

        for n in N_VALUES
            case_label = "$(eq.label)_n$(n)"

            # Baseline run
            run_idx += 1
            @info "[$run_idx/$total] $case_label baseline"
            config = baseline_config(eq, n, "")
            result = run_single(config)
            if result !== nothing
                store_result!(OUTPUT_H5, "$case_label/baseline", result)
                dp = result["dp_matrix"]
                msing = result["msing"]
                dp_diag = [size(dp,1) >= s ? dp[s,s] : NaN+NaN*im for s in 1:msing]
                @info "  baseline: msing=$msing, dp_diag=$([@sprintf("%.2f%+.2fi", real(d), imag(d)) for d in dp_diag])"
            else
                @warn "  baseline FAILED"
            end

            # Parameter sweeps
            for sweep in sweeps
                for val in sweep.values
                    run_idx += 1
                    vl = value_label(val)
                    h5_group = "$case_label/$(sweep.label)/$vl"
                    @info "[$run_idx/$total] $case_label $(sweep.label)=$vl"

                    config = baseline_config(eq, n, "")
                    apply_param!(config, sweep, val)
                    result = run_single(config)

                    if result !== nothing
                        store_result!(OUTPUT_H5, h5_group, result)
                        dp = result["dp_matrix"]
                        msing_r = result["msing"]
                        if size(dp, 1) > 0
                            dp1 = dp[1, 1]
                            @info "  dp[1,1]=$(@sprintf("%.4f%+.4fi", real(dp1), imag(dp1))), msing=$msing_r, t=$(@sprintf("%.1fs", result["runtime"]))"
                        end
                    else
                        @warn "  FAILED: $(sweep.label)=$vl"
                    end
                end
            end
        end
    end

    @info "Done. Results saved to $OUTPUT_H5"
end

main()
