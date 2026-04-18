#!/usr/bin/env julia
"""
    run_scan.jl — LAR epsilon (inverse aspect ratio) scan

Runs Julia GPEC on each TJ benchmark equilibrium, extracts Δ'(2/1), Δ'(3/1),
and δW components, then compares against the Fortran STRIDE reference CSV.

Usage:
    julia --project=../.. run_scan.jl              # Full scan (49+ points)
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
# Scan parameters — matches Fortran run_stride_epsilon_scan.py
# ============================================================================

# Downsampled from 109 original points: NaNs removed, cropped before the
# Δ' pole at eps~0.662, thinned in flat regions, full resolution kept
# in the steep gradient region.
const EPSILONS = [
    # Flat region (every ~3rd point)
    0.125, 0.1499, 0.1748, 0.1997, 0.2246, 0.2495, 0.2744, 0.2993,
    0.3242, 0.3491,
    # Moderate decline (every ~2nd point)
    0.3574, 0.374, 0.3906, 0.4072, 0.4238, 0.4404, 0.457, 0.4736,
    0.4902, 0.5005, 0.5151, 0.5317, 0.54275,
    # Slow approach (every ~2nd point)
    0.551, 0.55475, 0.55925, 0.56475, 0.57025, 0.57575, 0.58125,
    0.58675, 0.59225, 0.59775, 0.60325, 0.60875, 0.61425, 0.61975,
    # Steep rise to pole (all points kept)
    0.6225, 0.62525, 0.628, 0.63075, 0.6335, 0.63625, 0.639,
    0.64175, 0.6445, 0.64725, 0.65, 0.65125, 0.65375, 0.655,
    0.65625, 0.6575, 0.65875, 0.66, 0.66125,
]

const EPSILONS_TEST = [0.2495, 0.4072, 0.595]

const SCAN_DIR = @__DIR__
const EQUIL_DIR = joinpath(SCAN_DIR, "equilibria")
const REF_DIR = joinpath(SCAN_DIR, "reference")
const OUTPUT_DIR = joinpath(SCAN_DIR, "outputs")

# ============================================================================
# Core scan logic
# ============================================================================

"""
    run_single(epsilon; verbose=true) -> NamedTuple or nothing

Run Julia GPEC for a single epsilon value. Returns extracted results
or nothing on failure.
"""
function run_single(eps::Float64; verbose::Bool=true)
    geqdsk = joinpath(EQUIL_DIR, "TJ_epsilon_scan_$(eps).geqdsk")
    if !isfile(geqdsk)
        @warn "geqdsk not found for epsilon=$eps: $geqdsk"
        return nothing
    end

    run_dir = mktempdir(; prefix="gpec_eps_$(eps)_")

    try
        config = TOML.parsefile(joinpath(SCAN_DIR, "gpec.toml"))
        config["Equilibrium"]["eq_filename"] = geqdsk
        config["ForceFreeStates"]["HDF5_filename"] = joinpath(run_dir, "gpec.h5")

        open(joinpath(run_dir, "gpec.toml"), "w") do io
            TOML.print(io, config)
        end

        if verbose
            @info "Running GPEC for epsilon=$eps"
        end

        GeneralizedPerturbedEquilibrium.main([run_dir])

        h5_path = joinpath(run_dir, "gpec.h5")
        if !isfile(h5_path)
            @warn "HDF5 output not found for epsilon=$eps"
            return nothing
        end

        return extract_results(h5_path)
    catch e
        @warn "GPEC failed for epsilon=$eps" exception=(e, catch_backtrace())
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
        ep = read(f, "vacuum/ep")
        ev = read(f, "vacuum/ev")
        et = read(f, "vacuum/et")

        msing = read(f, "singular/msing")
        m_sing = read(f, "singular/m")

        # STRIDE BVP Δ' matrix: [msing, msing] complex — diagonal = per-surface Δ'
        dp_mat = haskey(f, "singular/delta_prime_matrix") ? read(f, "singular/delta_prime_matrix") : nothing

        # Per-surface ca-based Δ': [msing, max_modes] complex (fallback)
        dp_ca = haskey(f, "singular/delta_prime") ? read(f, "singular/delta_prime") : nothing

        # Use the truncated qlim (integration limit) to match Fortran's q_edge reporting
        q_edge = haskey(f, "info/qlim") ? read(f, "info/qlim") : read(f, "equil/qmax")

        dp_21_real = NaN
        dp_21_imag = NaN
        dp_31_real = NaN
        dp_31_imag = NaN

        if msing > 0
            for s in 1:msing
                m_val = m_sing[s, 1]
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
    run_scan(epsilons; verbose=true) -> Dict

Run the full scan and collect results into arrays.
"""
function run_scan(epsilons::Vector{Float64}; verbose::Bool=true)
    n = length(epsilons)
    results = Dict{String,Vector{Float64}}(
        "epsilon"             => copy(epsilons),
        "delta_prime_21_real" => fill(NaN, n),
        "delta_prime_21_imag" => fill(NaN, n),
        "delta_prime_31_real" => fill(NaN, n),
        "delta_prime_31_imag" => fill(NaN, n),
        "delta_W_plasma"      => fill(NaN, n),
        "delta_W_vacuum"      => fill(NaN, n),
        "delta_W_total"       => fill(NaN, n),
        "q_edge"              => fill(NaN, n),
    )

    for (i, eps) in enumerate(epsilons)
        @info "[$(i)/$(n)] epsilon = $eps"
        r = run_single(eps; verbose=verbose)
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
    header = "epsilon,delta_prime_21_real,delta_prime_21_imag," *
             "delta_prime_31_real,delta_prime_31_imag," *
             "delta_W_plasma,delta_W_vacuum,delta_W_total,q_edge"
    n = length(results["epsilon"])
    data = hcat(
        results["epsilon"],
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
    ref_path = joinpath(REF_DIR, "epsilon_scan_summary.csv")
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
    eps_j = julia_results["epsilon"]
    eps_f = fortran_ref["epsilon"]

    p1 = Plots.plot(eps_f, fortran_ref["delta_prime_21_real"];
        label="Fortran Δ'(2/1)", lw=2, xlabel="epsilon (a/R₀)", ylabel="Δ'(2/1) real",
        left_margin=12Plots.mm, bottom_margin=4Plots.mm)
    Plots.scatter!(p1, eps_j, julia_results["delta_prime_21_real"];
        label="Julia Δ'(2/1)", ms=4)

    p2 = Plots.plot(eps_f, fortran_ref["delta_prime_31_real"];
        label="Fortran Δ'(3/1)", lw=2, xlabel="epsilon (a/R₀)", ylabel="Δ'(3/1) real",
        left_margin=12Plots.mm, bottom_margin=4Plots.mm)
    Plots.scatter!(p2, eps_j, julia_results["delta_prime_31_real"];
        label="Julia Δ'(3/1)", ms=4)

    p3 = Plots.plot(eps_f, fortran_ref["delta_W_total"];
        label="Fortran δW total", lw=2, xlabel="epsilon (a/R₀)", ylabel="δW",
        left_margin=12Plots.mm, bottom_margin=4Plots.mm)
    Plots.plot!(p3, eps_f, fortran_ref["delta_W_plasma"]; label="Fortran δW plasma", lw=1, ls=:dash)
    Plots.plot!(p3, eps_f, fortran_ref["delta_W_vacuum"]; label="Fortran δW vacuum", lw=1, ls=:dash)
    Plots.scatter!(p3, eps_j, julia_results["delta_W_total"]; label="Julia δW total", ms=4)
    Plots.scatter!(p3, eps_j, julia_results["delta_W_plasma"]; label="Julia δW plasma", ms=3)
    Plots.scatter!(p3, eps_j, julia_results["delta_W_vacuum"]; label="Julia δW vacuum", ms=3)

    fig = Plots.plot(p1, p2, p3; layout=(3, 1), size=(800, 900),
        plot_title="Epsilon Scan: Julia GPEC vs Fortran STRIDE")

    fig_path = joinpath(OUTPUT_DIR, "epsilon_scan_comparison.png")
    Plots.savefig(fig, fig_path)
    @info "Comparison plot saved to $fig_path"
end

# ============================================================================
# Main
# ============================================================================

function main_scan()
    test_mode = "--test" in ARGS
    compare_mode = "--compare" in ARGS

    epsilons = test_mode ? EPSILONS_TEST : EPSILONS

    @info "LAR epsilon scan: $(length(epsilons)) points" * (test_mode ? " (test mode)" : "")

    mkpath(OUTPUT_DIR)

    results = run_scan(epsilons)
    save_summary(results, joinpath(OUTPUT_DIR, "julia_epsilon_scan_summary.csv"))

    if compare_mode
        ref = load_reference()
        if ref !== nothing
            plot_comparison(results, ref)
        end
    end

    @info "Done."
end

main_scan()
