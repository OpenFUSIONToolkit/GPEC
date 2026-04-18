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

# Aspect-ratio scan: ε grid ends just before the ideal-kink pole at
# ε ≈ 0.665 (where δW_t → 0 and Δ' diverges).  Grid is power-law warped so
# spacing tightens smoothly as the pole is approached — the flat low-ε
# region is covered with even cadence, and more points land in the final
# few percent where Δ' rises by orders of magnitude.
function _warped_grid(x_start::Float64, x_end::Float64, N::Int; p::Float64 = 2.0)
    return [x_start + (x_end - x_start) * (1 - (1 - i / (N - 1))^p) for i in 0:N-1]
end

const EPSILONS_FULL = _warped_grid(0.125, 0.660, 56; p = 2.0)

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
        # Option B: use tj_direct (ψ(R,Z) grid + direct-GS solver) rather than
        # the inverse pipeline.  Required to capture the ideal external-kink
        # pole (δW_t → 0 as ε → ε_crit); the inverse path bypasses the
        # line-integrated q and shows no such pole.
        config["Equilibrium"]["eq_type"] = "tj_direct"
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
