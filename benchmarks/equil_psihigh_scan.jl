"""
equil_psihigh_scan.jl

Scans psihigh from 0.980 to 0.999, testing robustness of all three equilibrium
methods near the separatrix.

Key outputs:
  - Last successful psihigh for each method
  - q(ψ) panel at each psihigh (all methods on same axes)
  - Max roundtrip error in outer 10% (ψ > 0.90) vs. psihigh
  - q monotonicity violations in outer 10%

Physics note: In a diverted tokamak (DIIID), q → ∞ as ψ → 1. The EFIT q-profile
extrapolates to a finite qa, which is physically wrong near the separatrix. This
benchmark therefore does NOT use the EFIT q-profile as ground truth for ψ > 0.95.
Instead, methods are compared against each other, and the divergence of q toward
the separatrix is tracked as a physical indicator.

Usage:
  julia --project=. benchmarks/equil_psihigh_scan.jl [example_path]
"""

using GeneralizedPerturbedEquilibrium
using GeneralizedPerturbedEquilibrium.Equilibrium
using TOML, Printf, Statistics, CSV, DataFrames

example_path = length(ARGS) > 0 ? ARGS[1] : joinpath(@__DIR__, "../examples/DIIID-like_ideal_example")
config_path  = joinpath(example_path, "gpec.toml")

psihigh_values = [0.980, 0.985, 0.990, 0.993, 0.995, 0.996, 0.997, 0.998, 0.999]
methods = ["efit", "efit_arclength", "efit_by_inversion"]

function make_config(path::String, eq_type::String, psihigh::Float64)
    raw = TOML.parsefile(path)
    raw["Equilibrium"]["eq_type"] = eq_type
    raw["Equilibrium"]["psihigh"] = psihigh
    base = dirname(path)
    return Equilibrium.EquilibriumConfig(raw["Equilibrium"], base)
end

# JIT warmup with default config
println("JIT warmup...")
warmup_cfg = make_config(config_path, "efit", 0.990)
setup_equilibrium(warmup_cfg)

println("=" ^ 70)
println("psihigh robustness scan")
println("=" ^ 70)

# Storage: DataFrame for CSV output
rows = []

for method in methods
    println("\n--- Method: $method ---")
    for psihigh in psihigh_values
        cfg = make_config(config_path, method, psihigh)
        local pe = nothing
        local runtime = NaN
        local success = false
        local err_msg = ""
        try
            t = @elapsed pe = setup_equilibrium(cfg)
            runtime = t
            success = true
        catch e
            err_msg = string(e)[1:min(80, length(string(e)))]
        end

        # Compute diagnostics if successful
        roundtrip_max_edge = NaN
        q_mono_violations_edge = NaN
        q_at_psihigh = NaN
        q_edge_slope = NaN

        if success
            raw_profile = Equilibrium.read_efit(cfg)
            psi_xs = pe.rzphi_xs
            mtheta = length(pe.rzphi_ys) - 1
            psio = pe.psio

            # Roundtrip error in outer 10%
            errors = Float64[]
            for ipsi in 1:length(psi_xs)
                psi_xs[ipsi] < 0.90 && continue
                ψ = psi_xs[ipsi]
                for itheta in 1:8:mtheta+1
                    θ = pe.rzphi_ys[itheta]
                    r2  = pe.rzphi_rsquared((ψ, θ))
                    off = pe.rzphi_offset((ψ, θ))
                    rfac = sqrt(max(r2, 0.0))
                    η = 2π * (θ + off)
                    R = pe.ro + rfac * cos(η)
                    Z = pe.zo + rfac * sin(η)
                    ψ_recon = 1.0 - raw_profile.psi_in((R, Z)) / psio
                    push!(errors, abs(ψ_recon - ψ))
                end
            end
            roundtrip_max_edge = isempty(errors) ? NaN : maximum(errors)

            # q monotonicity in outer 10%
            q_profile = [pe.profiles.q_spline(ψ) for ψ in psi_xs]
            dq = diff(q_profile)
            mask_edge = psi_xs[2:end] .> 0.90
            q_mono_violations_edge = count(dq[mask_edge] .< 0)

            # q at outermost grid point and slope
            q_at_psihigh = pe.profiles.q_spline.y[end]
            n = length(psi_xs)
            if n >= 2
                q_edge_slope = (pe.profiles.q_spline.y[end] - pe.profiles.q_spline.y[end-1]) /
                               (psi_xs[end] - psi_xs[end-1])
            end
        end

        status = success ? "OK" : "FAIL"
        @printf("  psihigh=%.3f  %-4s  t=%.2fs  rt_edge=%.2e  q_mono_viol=%d  q(end)=%.2f  q_slope=%.1f\n",
            psihigh, status, runtime, roundtrip_max_edge, Int(isnan(q_mono_violations_edge) ? -1 : q_mono_violations_edge),
            isnan(q_at_psihigh) ? -1.0 : q_at_psihigh,
            isnan(q_edge_slope) ? -1.0 : q_edge_slope)

        push!(rows, (
            method=method, psihigh=psihigh, success=success, runtime=runtime,
            roundtrip_max_edge=roundtrip_max_edge,
            q_mono_violations_edge=q_mono_violations_edge,
            q_at_psihigh=q_at_psihigh, q_edge_slope=q_edge_slope,
            error_msg=err_msg
        ))
    end
end

# Write CSV
df = DataFrame(rows)
output_csv = joinpath(example_path, "equil_psihigh_scan.csv")
CSV.write(output_csv, df)
println("\nResults saved to: $output_csv")

# Summary: last successful psihigh per method
println("\n" * "=" ^ 70)
println("Last successful psihigh per method:")
for method in methods
    method_rows = filter(r -> r.method == method && r.success, rows)
    if isempty(method_rows)
        println("  $method: ALL FAILED")
    else
        last_ok = maximum(r.psihigh for r in method_rows)
        @printf("  %-22s  psihigh = %.3f\n", method, last_ok)
    end
end
println()
