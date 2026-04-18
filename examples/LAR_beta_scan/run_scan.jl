#!/usr/bin/env julia
"""
    run_scan.jl — LAR pressure-factor (β) scan

Runs Julia GPEC on each TJ benchmark equilibrium, extracts Δ'(2/1), Δ'(3/1),
and δW components, then compares against the Fortran STRIDE reference CSV.

Usage:
    julia --project=../.. run_scan.jl              # Full scan (42 points)
    julia --project=../.. run_scan.jl --test        # Quick test (3 points)
    julia --project=../.. run_scan.jl --compare     # Plot comparison with Fortran
    julia --project=../.. run_scan.jl --test --compare
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, "../.."))

using GeneralizedPerturbedEquilibrium
using HDF5
using TOML
using Printf
using DelimitedFiles
using Plots

# ============================================================================
# Scan parameters — matches Fortran run_stride_beta_scan.py
# ============================================================================

const PRESSURE_FACTORS = [
    0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08,
    0.1, 0.11, 0.12, 0.13, 0.14, 0.145, 0.15, 0.1525, 0.155, 0.1575,
    0.16, 0.1625, 0.165, 0.16625, 0.1674, 0.1675, 0.17, 0.1725, 0.175,
    0.1775, 0.18, 0.18225, 0.1825, 0.18275, 0.183, 0.18325, 0.1835,
    0.18375, 0.18425, 0.1845, 0.18475, 0.185,
]

const PRESSURE_FACTORS_TEST = [0.1, 0.16, 0.175]

const SCAN_DIR = @__DIR__
const EQUIL_DIR = joinpath(SCAN_DIR, "equilibria")
const REF_DIR = joinpath(SCAN_DIR, "reference")
const OUTPUT_DIR = joinpath(SCAN_DIR, "outputs")

# ============================================================================
# Core scan logic
# ============================================================================

"""
    run_single(pressure_factor; verbose=true) -> NamedTuple or nothing

Run Julia GPEC for a single pressure factor value. Returns extracted results
or nothing on failure.
"""
function run_single(pf::Float64; verbose::Bool=true)
    geqdsk = joinpath(EQUIL_DIR, "TJ_betascan_$(pf).geqdsk")
    if !isfile(geqdsk)
        @warn "geqdsk not found for pressure_factor=$pf: $geqdsk"
        return nothing
    end

    # Create a temporary working directory for this run
    run_dir = mktempdir(; prefix="gpec_beta_$(pf)_")

    try
        # Copy and patch gpec.toml
        config = TOML.parsefile(joinpath(SCAN_DIR, "gpec.toml"))
        config["Equilibrium"]["eq_filename"] = geqdsk
        config["ForceFreeStates"]["HDF5_filename"] = joinpath(run_dir, "gpec.h5")

        # Write patched config
        open(joinpath(run_dir, "gpec.toml"), "w") do io
            TOML.print(io, config)
        end

        if verbose
            @info "Running GPEC for pressure_factor=$pf"
        end

        # Run GPEC
        GeneralizedPerturbedEquilibrium.main([run_dir])

        # Read results from HDF5
        h5_path = joinpath(run_dir, "gpec.h5")
        if !isfile(h5_path)
            @warn "HDF5 output not found for pressure_factor=$pf"
            return nothing
        end

        return extract_results(h5_path)
    catch e
        @warn "GPEC failed for pressure_factor=$pf" exception=(e, catch_backtrace())
        return nothing
    finally
        rm(run_dir; force=true, recursive=true)
    end
end

"""
    extract_results(h5_path) -> NamedTuple

Extract Δ', δW, and q_edge from GPEC HDF5 output.

Uses the STRIDE BVP `delta_prime_matrix` diagonal (matching Fortran STRIDE output)
when available, falling back to the per-surface ca-based `delta_prime`.
"""
function extract_results(h5_path::String)
    h5open(h5_path, "r") do f
        # Energy components
        ep = read(f, "vacuum/ep")  # plasma energy
        ev = read(f, "vacuum/ev")  # vacuum energy
        et = read(f, "vacuum/et")  # total energy

        # Singular surface data
        msing = read(f, "singular/msing")
        m_sing = read(f, "singular/m")  # [msing, max_modes]

        # STRIDE BVP Δ' matrix: [msing, msing] complex — diagonal = per-surface Δ'
        dp_mat = haskey(f, "singular/delta_prime_matrix") ? read(f, "singular/delta_prime_matrix") : nothing

        # Per-surface ca-based Δ': [msing, max_modes] complex (fallback)
        dp_ca = haskey(f, "singular/delta_prime") ? read(f, "singular/delta_prime") : nothing

        # q_edge from equilibrium parameters (qmax ≈ q_edge for monotonic q profiles)
        # Use the truncated qlim (integration limit) to match Fortran's q_edge reporting
        q_edge = haskey(f, "info/qlim") ? read(f, "info/qlim") : read(f, "equil/qmax")

        # Extract Δ'(2/1) and Δ'(3/1) from singular surface data
        dp_21_real = NaN
        dp_21_imag = NaN
        dp_31_real = NaN
        dp_31_imag = NaN

        if msing > 0
            for s in 1:msing
                m_val = m_sing[s, 1]
                # Prefer STRIDE BVP matrix diagonal, fall back to ca-based
                dp_val = if dp_mat !== nothing && s <= size(dp_mat, 1)
                    dp_mat[s, s]
                elseif dp_ca !== nothing
                    dp_ca[s, 1]
                else
                    NaN + NaN*im
                end
                if m_val == 2
                    dp_21_real = real(dp_val)
                    dp_21_imag = imag(dp_val)
                elseif m_val == 3
                    dp_31_real = real(dp_val)
                    dp_31_imag = imag(dp_val)
                end
            end
        end

        return (
            delta_prime_21_real = dp_21_real,
            delta_prime_21_imag = dp_21_imag,
            delta_prime_31_real = dp_31_real,
            delta_prime_31_imag = dp_31_imag,
            delta_W_plasma = real(ep[1]),
            delta_W_vacuum = real(ev[1]),
            delta_W_total = real(et[1]),
            q_edge = q_edge,
        )
    end
end

"""
    run_scan(pressure_factors; verbose=true) -> Dict

Run the full scan and collect results into arrays.
"""
function run_scan(pfs::Vector{Float64}; verbose::Bool=true)
    n = length(pfs)
    results = Dict{String,Vector{Float64}}(
        "pressure_factor"     => copy(pfs),
        "delta_prime_21_real" => fill(NaN, n),
        "delta_prime_21_imag" => fill(NaN, n),
        "delta_prime_31_real" => fill(NaN, n),
        "delta_prime_31_imag" => fill(NaN, n),
        "delta_W_plasma"      => fill(NaN, n),
        "delta_W_vacuum"      => fill(NaN, n),
        "delta_W_total"       => fill(NaN, n),
        "q_edge"              => fill(NaN, n),
    )

    for (i, pf) in enumerate(pfs)
        @info "[$(i)/$(n)] pressure_factor = $pf"
        r = run_single(pf; verbose=verbose)
        if r !== nothing
            results["delta_prime_21_real"][i] = r.delta_prime_21_real
            results["delta_prime_21_imag"][i] = r.delta_prime_21_imag
            results["delta_prime_31_real"][i] = r.delta_prime_31_real
            results["delta_prime_31_imag"][i] = r.delta_prime_31_imag
            results["delta_W_plasma"][i]      = r.delta_W_plasma
            results["delta_W_vacuum"][i]      = r.delta_W_vacuum
            results["delta_W_total"][i]       = r.delta_W_total
            results["q_edge"][i]              = r.q_edge
        end
    end

    return results
end

# ============================================================================
# I/O
# ============================================================================

function save_summary(results::Dict, output_path::String)
    header = "pressure_factor,delta_prime_21_real,delta_prime_21_imag," *
             "delta_prime_31_real,delta_prime_31_imag," *
             "delta_W_plasma,delta_W_vacuum,delta_W_total,q_edge"
    n = length(results["pressure_factor"])
    data = hcat(
        results["pressure_factor"],
        results["delta_prime_21_real"],
        results["delta_prime_21_imag"],
        results["delta_prime_31_real"],
        results["delta_prime_31_imag"],
        results["delta_W_plasma"],
        results["delta_W_vacuum"],
        results["delta_W_total"],
        results["q_edge"],
    )
    open(output_path, "w") do io
        println(io, header)
        for i in 1:n
            println(io, join([@sprintf("%.10e", data[i, j]) for j in 1:9], ","))
        end
    end
    @info "Summary saved to $output_path"
end

function load_reference()
    ref_path = joinpath(REF_DIR, "pressure_factor_scan_summary.csv")
    if !isfile(ref_path)
        @warn "Reference CSV not found: $ref_path"
        return nothing
    end
    data, header = readdlm(ref_path, ',', Float64; header=true)
    cols = vec(header)
    return Dict(cols[j] => data[:, j] for j in 1:length(cols))
end

# ============================================================================
# Comparison plotting
# ============================================================================

function plot_comparison(julia_results::Dict, fortran_ref::Dict)
    pf_j = julia_results["pressure_factor"]
    pf_f = fortran_ref["pressure_factor"]

    p1 = Plots.plot(pf_f, fortran_ref["delta_prime_21_real"];
        label="Fortran Δ'(2/1)", lw=2, xlabel="pressure factor", ylabel="Δ'(2/1) real",
        left_margin=12Plots.mm, bottom_margin=4Plots.mm)
    Plots.scatter!(p1, pf_j, julia_results["delta_prime_21_real"];
        label="Julia Δ'(2/1)", ms=4)

    p2 = Plots.plot(pf_f, fortran_ref["delta_prime_31_real"];
        label="Fortran Δ'(3/1)", lw=2, xlabel="pressure factor", ylabel="Δ'(3/1) real",
        left_margin=12Plots.mm, bottom_margin=4Plots.mm)
    Plots.scatter!(p2, pf_j, julia_results["delta_prime_31_real"];
        label="Julia Δ'(3/1)", ms=4)

    p3 = Plots.plot(pf_f, fortran_ref["delta_W_total"];
        label="Fortran δW total", lw=2, xlabel="pressure factor", ylabel="δW",
        left_margin=12Plots.mm, bottom_margin=4Plots.mm)
    Plots.plot!(p3, pf_f, fortran_ref["delta_W_plasma"]; label="Fortran δW plasma", lw=1, ls=:dash)
    Plots.plot!(p3, pf_f, fortran_ref["delta_W_vacuum"]; label="Fortran δW vacuum", lw=1, ls=:dash)
    Plots.scatter!(p3, pf_j, julia_results["delta_W_total"]; label="Julia δW total", ms=4)
    Plots.scatter!(p3, pf_j, julia_results["delta_W_plasma"]; label="Julia δW plasma", ms=3)
    Plots.scatter!(p3, pf_j, julia_results["delta_W_vacuum"]; label="Julia δW vacuum", ms=3)

    fig = Plots.plot(p1, p2, p3; layout=(3, 1), size=(800, 900),
        plot_title="Beta Scan: Julia GPEC vs Fortran STRIDE")

    fig_path = joinpath(OUTPUT_DIR, "beta_scan_comparison.png")
    Plots.savefig(fig, fig_path)
    @info "Comparison plot saved to $fig_path"
end

# ============================================================================
# Main
# ============================================================================

function main_scan()
    test_mode = "--test" in ARGS
    compare_mode = "--compare" in ARGS

    pfs = test_mode ? PRESSURE_FACTORS_TEST : PRESSURE_FACTORS

    @info "LAR beta scan: $(length(pfs)) points" * (test_mode ? " (test mode)" : "")

    mkpath(OUTPUT_DIR)

    results = run_scan(pfs)
    save_summary(results, joinpath(OUTPUT_DIR, "julia_beta_scan_summary.csv"))

    if compare_mode
        ref = load_reference()
        if ref !== nothing
            plot_comparison(results, ref)
        end
    end

    @info "Done."
end

main_scan()
