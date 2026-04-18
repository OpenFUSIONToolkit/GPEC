#!/usr/bin/env julia
"""
    run_scan.jl — TJ-model beta (pressure factor) scan

Fixed geometry (ε=0.2), varying pressure via pc parameter.
Uses the built-in TJ analytic equilibrium model.

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
# Scan parameters — TJ benchmark pressure factors
# ============================================================================

# Pressure scan: pc grid ends just before the ideal-kink pole at pc ≈ 0.174
# (where δW_t → 0 and Δ' diverges).  Grid is power-law warped so the spacing
# is approximately uniform over most of the range and smoothly tightens as
# the pole is approached, giving an even visual cadence without wasting
# points on the flat-slope region far from the pole.
function _warped_grid(x_start::Float64, x_end::Float64, N::Int; p::Float64 = 2.0)
    return [x_start + (x_end - x_start) * (1 - (1 - i / (N - 1))^p) for i in 0:N-1]
end

const PC_FULL = _warped_grid(0.001, 0.1735, 40; p = 2.0)

const PC_TEST = [0.001, 0.10, 0.17]

const SCAN_DIR = @__DIR__
const OUTPUT_H5 = joinpath(SCAN_DIR, "beta_scan.h5")

# Fixed TJ parameters for beta scan (ε = 0.2, matching paper: R0=2m, a=0.4m)
const LAR_R0 = 2.0    # Major radius [m]
const LAR_A = 0.4      # Minor radius [m] → ε = 0.2
const QC = 1.5
const QA = 3.6
const MU = 2.0
const B0 = 12.0

# ============================================================================
# Run a single pressure point
# ============================================================================

function run_single(pc::Float64)
    run_dir = mktempdir(; prefix="gpec_tj_beta_")
    try
        tj_dict = Dict("TJ_INPUT" => Dict(
            "lar_r0" => LAR_R0, "lar_a" => LAR_A,
            "qc" => QC, "qa" => QA, "pc" => pc,
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
        @warn "Failed for pc=$pc" exception=(e, catch_backtrace())
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
    pcs = test_mode ? PC_TEST : PC_FULL

    @info "TJ beta scan: $(length(pcs)) points, ε=$(LAR_A/LAR_R0), B0=$(B0)T, qc=$(QC), qa=$(QA)" *
          (test_mode ? " (test mode)" : "")

    isfile(OUTPUT_H5) && rm(OUTPUT_H5)

    for (i, pc) in enumerate(pcs)
        @info "[$(i)/$(length(pcs))] pc=$pc"
        result = run_single(pc)
        if result !== nothing
            h5open(OUTPUT_H5, isfile(OUTPUT_H5) ? "r+" : "w") do f
                gname = @sprintf("pc_%.5f", pc)
                haskey(f, gname) && delete_object(f, gname)
                g = create_group(f, gname)
                g["pressure_factor"] = pc
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
