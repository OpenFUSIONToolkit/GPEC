#!/usr/bin/env julia
"""
    run_comparison.jl — DIII-D STRIDE comparison: Julia GPEC vs Fortran reference

Runs Julia GPEC on a set of DIII-D geqdsk equilibria for specified toroidal mode numbers,
extracts Δ' values at each rational surface, and compares against Fortran STRIDE reference
data (pre-computed NetCDF outputs or manually entered values).

Usage:
    julia --project=../.. run_comparison.jl              # Full run
    julia --project=../.. run_comparison.jl --shots 147131  # Single shot
    julia --project=../.. run_comparison.jl --dry-run    # List shots without running

The script produces:
  - A summary table printed to stdout
  - A CSV file at outputs/diiid_stride_comparison.csv
  - Individual HDF5 outputs in outputs/<shot>_n<n>/gpec.h5
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, "../.."))

using GeneralizedPerturbedEquilibrium
using HDF5
using TOML
using Printf
using DelimitedFiles
import Pkg
try
    @eval using JSON
catch
    Pkg.add("JSON")
    @eval using JSON
end

# ============================================================================
# Shot definitions: each shot has a geqdsk path, n values to run, and optional
# Fortran reference data.
# ============================================================================

const SCRIPT_DIR = @__DIR__
const OUTPUT_DIR = joinpath(SCRIPT_DIR, "outputs")

"""Shot configuration: geqdsk path, modes to compute, and STRIDE parameters."""
@kwdef struct ShotConfig
    shot::String
    geqdsk::String
    n_values::Vector{Int} = [1]
    psilow::Float64 = 1e-4
    psihigh::Float64 = 0.993
    mpsi::Int = 128
    mtheta::Int = 256
    delta_mlow::Int = 8
    delta_mhigh::Int = 8
    mthvac::Int = 512
    qhigh::Float64 = 1e3
    sing_order::Int = 6
    wall_a::Float64 = 20.0   # large = effectively no wall
    description::String = ""
end

"""Fortran reference data for a single (shot, n) combination."""
@kwdef struct FortranReference
    shot::String
    n::Int
    msing::Int
    m_values::Vector{Int}        # m at each rational surface
    q_values::Vector{Float64}    # q at each rational surface
    dp_diag_real::Vector{Float64}  # Δ' diagonal (real part)
    dp_diag_imag::Vector{Float64}  # Δ' diagonal (imag part)
    dW_plasma::Float64
    dW_vacuum::Float64
    dW_total::Float64
    source::String = "pre-computed"  # where this data came from
end

# ============================================================================
# Shot registry
# ============================================================================

function find_geqdsk(patterns...)
    for p in patterns
        expanded = expanduser(p)
        if isfile(expanded)
            return expanded
        end
    end
    return nothing
end

function build_shot_configs()
    shots = ShotConfig[]

    # 147131: DIII-D KEFIT H-mode, the canonical GPEC example
    # Use the ACTUAL EFIT geqdsk (same file Fortran uses), not the TokaMaker synthetic
    g147131 = find_geqdsk(
        "~/Desktop/plasma/GPEC/docs/examples/DIIID_ideal_example/g147131.02300_DIIID_KEFIT",
        joinpath(dirname(SCRIPT_DIR), "DIIID-like_ideal_example/TkMkr_D3Dlike_Hmode.geqdsk"),
    )
    if g147131 !== nothing
        push!(shots, ShotConfig(shot="147131", geqdsk=g147131, n_values=[1, 2],
            description="DIII-D KEFIT H-mode (canonical GPEC example)"))
    end

    # 204441: IDA_run MSE-constrained with Er
    g204441 = find_geqdsk(
        "~/Desktop/plasma/IDA_run/MSE_constraint/test_data/204441_4400_withEr/g204441.04409.geqdsk",
    )
    if g204441 !== nothing
        push!(shots, ShotConfig(shot="204441", geqdsk=g204441, n_values=[1, 2],
            description="DIII-D IDA MSE-constrained with Er"))
    end

    # 153833: IDA workflow example
    g153833 = find_geqdsk(
        "~/Desktop/plasma/IDA_run/IDA_workflow_examples/153833_3450/g153833.3450_results_baseline.geqdsk",
        "~/Desktop/plasma/IDA_run/IDA_workflow_examples/153833_3450/g153833.3450.EFIT13",
    )
    if g153833 !== nothing
        push!(shots, ShotConfig(shot="153833", geqdsk=g153833, n_values=[1, 2],
            description="DIII-D IDA workflow example"))
    end

    # 153072: Reversed shear (qmin=1.86) — excluded, Julia doesn't yet support
    # multi-resonance surfaces from non-monotonic q profiles

    return shots
end

# ============================================================================
# Fortran reference data (extracted from pre-existing STRIDE output files)
# ============================================================================

function load_fortran_references()
    # Fortran reference data is loaded from outputs/fortran_stride_results.json
    # (produced by run_fortran_stride.py). No hardcoded values here.
    json_path = joinpath(OUTPUT_DIR, "fortran_stride_results.json")
    if !isfile(json_path)
        @warn "No Fortran reference: $json_path not found. Run run_fortran_stride.py first."
        return FortranReference[]
    end

    raw = JSON.parsefile(json_path)
    refs = FortranReference[]
    for r in raw
        push!(refs, FortranReference(
            shot=r["shot"], n=r["n"], msing=r["msing"],
            m_values=[s["m"] for s in r["surfaces"]],
            q_values=[s["q"] for s in r["surfaces"]],
            dp_diag_real=[s["dp_real"] for s in r["surfaces"]],
            dp_diag_imag=[s["dp_imag"] for s in r["surfaces"]],
            dW_plasma=r["dW_plasma"],
            dW_vacuum=r["dW_vacuum"],
            dW_total=r["dW_total"],
            source="run_fortran_stride.py"
        ))
    end
    return refs
end

# ============================================================================
# Julia GPEC runner
# ============================================================================

"""Run Julia GPEC for a single (shot, n) and return extracted results."""
function run_julia_gpec(shot_config::ShotConfig, n::Int; verbose::Bool=true)
    run_dir = joinpath(OUTPUT_DIR, "$(shot_config.shot)_n$(n)")
    mkpath(run_dir)

    # Build gpec.toml
    config = Dict(
        "Equilibrium" => Dict(
            "eq_type" => "efit",
            "eq_filename" => shot_config.geqdsk,
            "jac_type" => "hamada",
            "grid_type" => "ldp",
            "psilow" => shot_config.psilow,
            "psihigh" => shot_config.psihigh,
            "mpsi" => shot_config.mpsi,
            "mtheta" => shot_config.mtheta,
        ),
        "Wall" => Dict(
            "shape" => "conformal",
            "a" => shot_config.wall_a,
        ),
        "ForceFreeStates" => Dict(
            "bal_flag" => false,
            "mat_flag" => true,
            "ode_flag" => true,
            "vac_flag" => true,
            "mer_flag" => true,
            "qlow" => 1.02,
            "qhigh" => shot_config.qhigh,
            "sing_start" => 0,
            "nn_low" => n,
            "nn_high" => n,
            "delta_mlow" => shot_config.delta_mlow,
            "delta_mhigh" => shot_config.delta_mhigh,
            "delta_mband" => 0,
            "mthvac" => shot_config.mthvac,
            "thmax0" => 1,
            "eulerlagrange_tolerance" => 1e-8,
            "singfac_min" => 1e-4,
            "ucrit" => 1e4,
            "sing_order" => shot_config.sing_order,
            "use_parallel" => true,
            "verbose" => false,
            "force_termination" => true,
            "write_outputs_to_HDF5" => true,
            "HDF5_filename" => joinpath(run_dir, "gpec.h5"),
            "save_interval" => 3,
        ),
    )

    toml_path = joinpath(run_dir, "gpec.toml")
    open(toml_path, "w") do io
        TOML.print(io, config)
    end

    if verbose
        @info "Running Julia GPEC: shot=$(shot_config.shot) n=$n"
    end

    try
        GeneralizedPerturbedEquilibrium.main([run_dir])
        return extract_julia_results(joinpath(run_dir, "gpec.h5"), n)
    catch e
        @warn "Julia GPEC failed for shot=$(shot_config.shot) n=$n" exception=(e, catch_backtrace())
        return nothing
    end
end

"""Extract results from Julia GPEC HDF5 output."""
function extract_julia_results(h5_path::String, n::Int)
    h5open(h5_path, "r") do f
        ep = read(f, "vacuum/ep")
        ev = read(f, "vacuum/ev")
        et = read(f, "vacuum/et")

        msing = read(f, "singular/msing")
        m_sing = read(f, "singular/m")      # [msing, max_modes]
        q_sing = read(f, "singular/q")      # [msing]
        psi_sing = read(f, "singular/psi")  # [msing]

        dp_mat = haskey(f, "singular/delta_prime_matrix") ? read(f, "singular/delta_prime_matrix") : nothing
        dp_ca = haskey(f, "singular/delta_prime") ? read(f, "singular/delta_prime") : nothing

        q0 = read(f, "equil/q0")
        qmax = read(f, "equil/qmax")
        qlim = haskey(f, "info/qlim") ? read(f, "info/qlim") : qmax

        surfaces = NamedTuple[]
        for s in 1:msing
            m_val = m_sing[s, 1]
            dp_val = if dp_mat !== nothing && s <= size(dp_mat, 1)
                dp_mat[s, s]
            elseif dp_ca !== nothing
                dp_ca[s, 1]
            else
                NaN + NaN * im
            end
            push!(surfaces, (m=m_val, n=n, q=q_sing[s], psi=psi_sing[s],
                dp_real=real(dp_val), dp_imag=imag(dp_val)))
        end

        return (
            msing=msing,
            surfaces=surfaces,
            dW_plasma=real(ep[1]),
            dW_vacuum=real(ev[1]),
            dW_total=real(et[1]),
            q0=q0, qmax=qmax, qlim=qlim,
        )
    end
end

# ============================================================================
# Comparison table
# ============================================================================

"""Print a formatted comparison table."""
function print_comparison_table(all_results, fortran_refs; io::IO=stdout)
    println(io, "")
    println(io, "="^110)
    println(io, "  DIII-D STRIDE Comparison: Julia GPEC vs Fortran STRIDE")
    println(io, "="^110)

    # Group results by shot
    shots_seen = String[]
    for (shot, n, julia_res) in all_results
        if !(shot in shots_seen)
            push!(shots_seen, shot)
        end
    end

    for shot in shots_seen
        shot_results = filter(x -> x[1] == shot, all_results)
        desc = ""  # TODO: get from config

        println(io, "\n", "-"^110)
        println(io, "  Shot: $shot")
        println(io, "-"^110)
        @printf(io, "  %-6s %-4s | %-12s %-12s %-12s | %-12s %-12s %-12s | %-8s\n",
            "m/n", "q", "Julia dp_Re", "Fortran dp_Re", "Error %",
            "Julia dp_Im", "Fortran dp_Im", "Error %", "Source")
        println(io, "  ", "-"^106)

        for (_, n, julia_res) in shot_results
            if julia_res === nothing
                println(io, "  n=$n: Julia GPEC FAILED")
                continue
            end

            # Find matching Fortran reference
            ref = nothing
            for r in fortran_refs
                if r.shot == shot && r.n == n
                    ref = r
                    break
                end
            end

            # Print dW comparison first
            @printf(io, "  n=%d: dW_p=%.4f  dW_v=%.4f  dW_t=%.4f  (qlim=%.2f, q0=%.3f, msing=%d)\n",
                n, julia_res.dW_plasma, julia_res.dW_vacuum, julia_res.dW_total,
                julia_res.qlim, julia_res.q0, julia_res.msing)
            if ref !== nothing
                @printf(io, "       dW_p=%.4f  dW_v=%.4f  dW_t=%.4f  [Fortran: %s]\n",
                    ref.dW_plasma, ref.dW_vacuum, ref.dW_total, ref.source)
            end

            for surf in julia_res.surfaces
                m_n_str = "$(surf.m)/$(surf.n)"
                q_str = @sprintf("%.3f", surf.q)

                # Find matching Fortran surface
                f_dp_re = NaN
                f_dp_im = NaN
                source = "--"
                if ref !== nothing
                    for (fi, fm) in enumerate(ref.m_values)
                        if fm == surf.m
                            f_dp_re = ref.dp_diag_real[fi]
                            f_dp_im = ref.dp_diag_imag[fi]
                            source = "Fortran"
                            break
                        end
                    end
                end

                err_re = isnan(f_dp_re) ? NaN : 100 * abs(surf.dp_real - f_dp_re) / max(abs(f_dp_re), 1e-30)
                err_im = isnan(f_dp_im) ? NaN : 100 * abs(surf.dp_imag - f_dp_im) / max(abs(f_dp_im), 1e-30)

                j_re_str = @sprintf("%+12.4f", surf.dp_real)
                j_im_str = @sprintf("%+12.4f", surf.dp_imag)
                f_re_str = isnan(f_dp_re) ? "    --      " : @sprintf("%+12.4f", f_dp_re)
                f_im_str = isnan(f_dp_im) ? "    --      " : @sprintf("%+12.4f", f_dp_im)
                e_re_str = isnan(err_re) ? "  --  " : @sprintf("%6.1f%%", err_re)
                e_im_str = isnan(err_im) ? "  --  " : @sprintf("%6.1f%%", err_im)

                @printf(io, "  %-6s %-4s | %s %s %s | %s %s %s | %s\n",
                    m_n_str, q_str, j_re_str, f_re_str, e_re_str,
                    j_im_str, f_im_str, e_im_str, source)
            end
        end
    end

    println(io, "\n", "="^110)
end

"""Save results to CSV."""
function save_csv(all_results, fortran_refs, output_path::String)
    header = "shot,n,m,q,psi,julia_dp_real,julia_dp_imag," *
             "fortran_dp_real,fortran_dp_imag,error_dp_real_pct,error_dp_imag_pct," *
             "julia_dW_p,julia_dW_v,julia_dW_t,fortran_dW_p,fortran_dW_v,fortran_dW_t"

    open(output_path, "w") do io
        println(io, header)
        for (shot, n, julia_res) in all_results
            julia_res === nothing && continue

            ref = nothing
            for r in fortran_refs
                r.shot == shot && r.n == n && (ref = r; break)
            end

            for surf in julia_res.surfaces
                f_re = NaN; f_im = NaN
                if ref !== nothing
                    for (fi, fm) in enumerate(ref.m_values)
                        if fm == surf.m
                            f_re = ref.dp_diag_real[fi]
                            f_im = ref.dp_diag_imag[fi]
                            break
                        end
                    end
                end

                err_re = isnan(f_re) ? NaN : 100 * abs(surf.dp_real - f_re) / max(abs(f_re), 1e-30)
                err_im = isnan(f_im) ? NaN : 100 * abs(surf.dp_imag - f_im) / max(abs(f_im), 1e-30)

                f_dWp = ref !== nothing ? ref.dW_plasma : NaN
                f_dWv = ref !== nothing ? ref.dW_vacuum : NaN
                f_dWt = ref !== nothing ? ref.dW_total : NaN

                @printf(io, "%s,%d,%d,%.6f,%.8f,%.8e,%.8e,%.8e,%.8e,%.4f,%.4f,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e\n",
                    shot, n, surf.m, surf.q, surf.psi,
                    surf.dp_real, surf.dp_imag, f_re, f_im, err_re, err_im,
                    julia_res.dW_plasma, julia_res.dW_vacuum, julia_res.dW_total,
                    f_dWp, f_dWv, f_dWt)
            end
        end
    end
    @info "CSV saved to $output_path"
end

# ============================================================================
# Main
# ============================================================================

function main()
    dry_run = "--dry-run" in ARGS
    shot_filter = nothing
    for (i, a) in enumerate(ARGS)
        if a == "--shots" && i < length(ARGS)
            shot_filter = split(ARGS[i+1], ",")
        end
    end

    shots = build_shot_configs()
    if shot_filter !== nothing
        shots = filter(s -> s.shot in shot_filter, shots)
    end

    if isempty(shots)
        @error "No shots found! Check geqdsk paths."
        return
    end

    println("\nShots to process:")
    for s in shots
        println("  $(s.shot): $(s.geqdsk)")
        println("    n_values=$(s.n_values), $(s.description)")
    end

    if dry_run
        println("\n[dry-run] Exiting without running.")
        return
    end

    mkpath(OUTPUT_DIR)

    fortran_refs = load_fortran_references()

    # Run Julia GPEC for each (shot, n)
    all_results = Tuple{String, Int, Any}[]
    total_runs = sum(length(s.n_values) for s in shots)
    run_idx = 0
    for shot_config in shots
        for n in shot_config.n_values
            run_idx += 1
            @info "[$run_idx/$total_runs] $(shot_config.shot) n=$n"
            result = run_julia_gpec(shot_config, n)
            push!(all_results, (shot_config.shot, n, result))
        end
    end

    # Print comparison table
    print_comparison_table(all_results, fortran_refs)

    # Also write to file
    table_path = joinpath(OUTPUT_DIR, "comparison_table.txt")
    open(table_path, "w") do io
        print_comparison_table(all_results, fortran_refs; io=io)
    end
    @info "Table saved to $table_path"

    # Save CSV
    save_csv(all_results, fortran_refs, joinpath(OUTPUT_DIR, "diiid_stride_comparison.csv"))

    @info "Done. Outputs in $OUTPUT_DIR"
end

main()
