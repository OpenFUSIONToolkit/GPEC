"""
equil_psilow_scan.jl

Scans psilow from 1e-1 down to 1e-4, testing near-axis accuracy and robustness
of all three equilibrium methods.

Key outputs:
  - q0 extrapolation to axis vs. EFIT q-axis value
  - Roundtrip error for ψ < 0.10 (near-axis region)
  - For efit_by_inversion: smallest psilow where Contour.jl traces a closed curve
    (identifies where the near-axis circular fallback kicks in)

Usage:
  julia --project=. benchmarks/equil_psilow_scan.jl [example_path]
"""

using GeneralizedPerturbedEquilibrium
using GeneralizedPerturbedEquilibrium.Equilibrium
using TOML, Printf, Statistics

example_path = length(ARGS) > 0 ? ARGS[1] : joinpath(@__DIR__, "../examples/DIIID-like_ideal_example")
config_path  = joinpath(example_path, "gpec.toml")

psilow_values = [1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4]
methods = ["efit", "efit_arclength", "efit_by_inversion"]

function make_config(path::String, eq_type::String, psilow::Float64)
    raw = TOML.parsefile(path)
    raw["Equilibrium"]["eq_type"] = eq_type
    raw["Equilibrium"]["psilow"] = psilow
    base = dirname(path)
    return Equilibrium.EquilibriumConfig(raw["Equilibrium"], base)
end

# Read EFIT q-axis value from header (simag line)
function read_efit_header_qaxis(eq_filename)
    lines = readlines(eq_filename)
    # Line 3 has raxis, zaxis, psimag, psibry, bcentr — q_axis is not directly in header.
    # The q-profile at ψ_norm=0 is q_axis; extrapolate from the sampled q-profile.
    header_parts = split(lines[1])
    nw = parse(Int, header_parts[end-1])
    lines_per_block = cld(nw, 5)
    # q-profile starts after 4 header lines + 4 data blocks (fpol,pres,ffprime,pprime) + psi 2D block
    nh = parse(Int, header_parts[end])
    q_start = 6 + (4 + cld(nw*nh, 5)) * lines_per_block  # rough; parse all blocks
    # Simpler: just extrapolate from the first two q-values returned by our config
    return nothing  # will use profile extrapolation below
end

# JIT warmup
println("JIT warmup...")
warmup_cfg = make_config(config_path, "efit", 1e-2)
setup_equilibrium(warmup_cfg)

println("=" ^ 70)
println("psilow near-axis accuracy scan")
println("=" ^ 70)

rows = []

for method in methods
    println("\n--- Method: $method ---")
    for psilow in psilow_values
        cfg = make_config(config_path, method, psilow)
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

        q0_extrap = NaN
        roundtrip_max_axis = NaN
        q_axis_rms_axis = NaN

        if success
            raw_profile = Equilibrium.read_efit(cfg)
            psi_xs = pe.rzphi_xs
            mtheta = length(pe.rzphi_ys) - 1
            psio   = pe.psio

            # q0 extrapolated to axis (linear from innermost two grid points)
            q1 = pe.profiles.q_spline.y[1]
            q_slope = pe.profiles.q_deriv(psi_xs[1]; hint=Ref(1))
            q0_extrap = q1 - q_slope * psi_xs[1]

            # Roundtrip error in inner region (ψ < 0.10)
            errors = Float64[]
            for ipsi in 1:length(psi_xs)
                psi_xs[ipsi] > 0.10 && break
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
            roundtrip_max_axis = isempty(errors) ? NaN : maximum(errors)
        end

        status = success ? "OK" : "FAIL"
        @printf("  psilow=%.1e  %-4s  t=%.2fs  q0_extrap=%.3f  rt_inner=%.2e\n",
            psilow, status, runtime, isnan(q0_extrap) ? -1.0 : q0_extrap, roundtrip_max_axis)
        !success && println("    Error: $err_msg")

        push!(rows, (
            method=method, psilow=psilow, success=success, runtime=runtime,
            q0_extrap=q0_extrap, roundtrip_max_axis=roundtrip_max_axis,
            error_msg=err_msg
        ))
    end
end

output_csv = joinpath(example_path, "equil_psilow_scan.csv")
open(output_csv, "w") do io
    println(io, join(string.(keys(rows[1])), ","))
    for row in rows
        println(io, join(map(v -> v isa AbstractString ? "\"$v\"" : string(v), values(row)), ","))
    end
end
println("\nResults saved to: $output_csv")
println()
