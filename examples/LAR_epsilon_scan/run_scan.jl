#!/usr/bin/env julia
"""
    run_scan.jl — TJ-model epsilon (inverse aspect ratio) scan

Uses the built-in TJ analytic equilibrium model (eq_type="tj") adapted from
R. Fitzpatrick's TJ code. No geqdsk files needed.

Usage:
    julia --project=../.. run_scan.jl              # Full scan
    julia --project=../.. run_scan.jl --test        # Quick test (3 points)
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, "../.."))

using GeneralizedPerturbedEquilibrium
using GeneralizedPerturbedEquilibrium.Equilibrium: TJConfig, EquilibriumConfig, setup_equilibrium
using HDF5
using TOML
using Printf

# ============================================================================
# Scan parameters (matching TJ benchmark)
# ============================================================================

const EPSILONS_FULL = [
    0.125, 0.1499, 0.1748, 0.1997, 0.2246, 0.2495, 0.2744, 0.2993,
    0.3242, 0.3491, 0.3574, 0.3740, 0.3906, 0.4072, 0.4238, 0.4404,
    0.4570, 0.4736, 0.4902, 0.5005, 0.5151, 0.5317, 0.5428, 0.5510,
    0.5548, 0.5593, 0.5648, 0.5703, 0.5758, 0.5813, 0.5868, 0.5923,
    0.5978, 0.6033, 0.6088, 0.6143, 0.6198, 0.6225, 0.6253, 0.6280,
    0.6308, 0.6335, 0.6363, 0.6390, 0.6418, 0.6445, 0.6473, 0.6500,
    0.6513, 0.6538, 0.6550, 0.6563, 0.6575, 0.6588, 0.6600, 0.6613,
]

const EPSILONS_TEST = [0.2495, 0.4072, 0.5510]

const SCAN_DIR = @__DIR__
const OUTPUT_H5 = joinpath(SCAN_DIR, "epsilon_scan.h5")

# TJ benchmark parameters (from TJ/Inputs/Equilibrium.json)
const QC = 1.5      # On-axis safety factor
const QA = 3.6      # Edge safety factor
const PC = 0.001    # Normalized pressure (very low for epsilon scan)
const MU = 2.0      # Pressure peaking exponent
const B0 = 12.0     # Toroidal field [T]
const LAR_A = 1.0   # Minor radius [m] (fixed)

# ============================================================================
# Run a single epsilon point
# ============================================================================

function run_single(epsilon::Float64)
    run_dir = mktempdir(; prefix="gpec_tj_")
    try
        # Write TJ config
        tj_dict = Dict("TJ_INPUT" => Dict(
            "lar_r0" => LAR_A / epsilon,
            "lar_a" => LAR_A,
            "qc" => QC, "qa" => QA, "pc" => PC,
            "mu" => MU, "B0" => B0,
            "ma" => 128, "mtau" => 128,
        ))
        open(joinpath(run_dir, "tj.toml"), "w") do io; TOML.print(io, tj_dict); end

        config = TOML.parsefile(joinpath(SCAN_DIR, "gpec.toml"))
        config["Equilibrium"]["eq_filename"] = joinpath(run_dir, "tj.toml")
        config["ForceFreeStates"]["HDF5_filename"] = joinpath(run_dir, "gpec.h5")
        open(joinpath(run_dir, "gpec.toml"), "w") do io; TOML.print(io, config); end

        GeneralizedPerturbedEquilibrium.main([run_dir])
        return extract_results(joinpath(run_dir, "gpec.h5"))
    catch e
        @warn "Failed for ε=$epsilon" exception=(e, catch_backtrace())
        return nothing
    finally
        rm(run_dir; force=true, recursive=true)
    end
end

function extract_results(h5_path::String)
    h5open(h5_path, "r") do f
        ep = read(f, "vacuum/ep"); ev = read(f, "vacuum/ev"); et = read(f, "vacuum/et")
        msing = read(f, "singular/msing")
        m_sing = read(f, "singular/m")
        dp_mat = haskey(f, "singular/delta_prime_matrix") ? read(f, "singular/delta_prime_matrix") : nothing
        qlim = haskey(f, "info/qlim") ? read(f, "info/qlim") : read(f, "equil/qmax")
        q0 = read(f, "equil/q0"); qmax = read(f, "equil/qmax")

        dp_21 = NaN + NaN*im; dp_31 = NaN + NaN*im
        if dp_mat !== nothing && msing > 0
            for s in 1:min(msing, size(dp_mat, 1))
                m_val = size(m_sing, 1) == msing ? m_sing[s, 1] : m_sing[1, s]
                if m_val == 2; dp_21 = dp_mat[s, s]; end
                if m_val == 3; dp_31 = dp_mat[s, s]; end
            end
        end
        return (dp_21=dp_21, dp_31=dp_31,
                dW_plasma=real(ep[1]), dW_vacuum=real(ev[1]), dW_total=real(et[1]),
                q0=q0, qmax=qmax, qlim=qlim, msing=msing, dp_matrix=dp_mat)
    end
end

# ============================================================================
# Main
# ============================================================================

function main()
    test_mode = "--test" in ARGS
    epsilons = test_mode ? EPSILONS_TEST : EPSILONS_FULL

    @info "TJ epsilon scan: $(length(epsilons)) points, B0=$(B0)T, qc=$(QC), qa=$(QA), pc=$(PC)" *
          (test_mode ? " (test mode)" : "")

    isfile(OUTPUT_H5) && rm(OUTPUT_H5)

    for (i, eps) in enumerate(epsilons)
        @info "[$(i)/$(length(epsilons))] ε=$eps (R0=$(@sprintf("%.3f", LAR_A/eps)))"
        result = run_single(eps)
        if result !== nothing
            h5open(OUTPUT_H5, isfile(OUTPUT_H5) ? "r+" : "w") do f
                gname = @sprintf("eps_%.4f", eps)
                haskey(f, gname) && delete_object(f, gname)
                g = create_group(f, gname)
                g["epsilon"] = eps
                g["dp_21_real"] = real(result.dp_21); g["dp_21_imag"] = imag(result.dp_21)
                g["dp_31_real"] = real(result.dp_31); g["dp_31_imag"] = imag(result.dp_31)
                g["dW_plasma"] = result.dW_plasma; g["dW_vacuum"] = result.dW_vacuum; g["dW_total"] = result.dW_total
                g["q0"] = result.q0; g["qmax"] = result.qmax; g["qlim"] = result.qlim; g["msing"] = result.msing
                if result.dp_matrix !== nothing; g["dp_matrix"] = result.dp_matrix; end
            end
            @printf("  dp21=%+.4f%+.4fi  dp31=%+.4f%+.4fi  dW_t=%+.6f  qa=%.3f\n",
                real(result.dp_21), imag(result.dp_21), real(result.dp_31), imag(result.dp_31),
                result.dW_total, result.qmax)
        end
    end

    @info "Results saved to $OUTPUT_H5"
end

main()
