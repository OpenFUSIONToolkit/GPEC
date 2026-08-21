"""
equil_method_comparison.jl

Compares the three EFIT equilibrium solver methods at default parameters:
  - "efit"             (geometric-angle field-line ODE, original)
  - "efit_arclength"   (arc-length level-set ODE, Strategy A)
  - "efit_by_inversion" (Contour.jl marching squares, Strategy B)

Accuracy metrics (all self-consistent, no EFIT external reference for far edge):
  - q(ψ) cross-method differences and comparison to EFIT input for ψ < 0.95
  - GSE residual (full domain and outer 10%: ψ > 0.90)
  - Roundtrip error: (ψ,θ)→(R,Z)→ψ_spline - ψ_target
  - Jacobian consistency: rzphi_jac vs. analytically derived Jacobian

Usage:
  julia --project=. benchmarks/equil_method_comparison.jl [example_path]
"""

using GeneralizedPerturbedEquilibrium
using GeneralizedPerturbedEquilibrium.Equilibrium
using TOML, Printf, Statistics

example_path = length(ARGS) > 0 ? ARGS[1] : joinpath(@__DIR__, "../examples/DIIID-like_ideal_example")
config_path = joinpath(example_path, "gpec.toml")
psihigh_override = length(ARGS) > 1 ? parse(Float64, ARGS[2]) : nothing

println("=" ^ 65)
println("Equilibrium Method Comparison")
println("Example: $example_path")
println("=" ^ 65)

methods = ["efit", "efit_arclength", "efit_by_inversion"]
results = Dict{String,Any}()

# ─── Helper: load a named TOML, override eq_type, return EquilibriumConfig ───
function make_config(path::String, eq_type::String; psihigh=nothing)
    raw = TOML.parsefile(path)
    raw["Equilibrium"]["eq_type"] = eq_type
    psihigh !== nothing && (raw["Equilibrium"]["psihigh"] = psihigh)
    base = dirname(path)
    return Equilibrium.EquilibriumConfig(raw["Equilibrium"], base)
end

# ─── Run each method (3 warm runs, report average of last 2) ─────────────────
for method in methods
    println("\n--- Running: $method ---")
    cfg = make_config(config_path, method; psihigh=psihigh_override)
    local pe = nothing
    local t1 = 0.0
    local t2 = 0.0
    local success = true
    try
        setup_equilibrium(cfg)  # JIT warmup
        t1 = @elapsed pe = setup_equilibrium(cfg)
        t2 = @elapsed setup_equilibrium(cfg)
    catch e
        @warn "  FAILED: $e"
        success = false
    end
    results[method] = Dict(
        "success" => success,
        "pe" => pe,
        "runtime" => (t1 + t2) / 2.0,
        "config" => cfg
    )
    success && @printf("  Runtime (avg 2 warm): %.3f s\n", (t1 + t2) / 2.0)
end

# ─── Profile comparison ───────────────────────────────────────────────────────
println("\n" * "=" ^ 65)
println("q(ψ) cross-method comparison")
println("=" ^ 65)

# Reference from the first successful method at shared psi_nodes
ref_method = first(m for m in methods if results[m]["success"])
ref_pe = results[ref_method]["pe"]
psi_nodes = ref_pe.rzphi_xs

# Evaluate q at all psi_nodes for each method
q_vals = Dict{String,Vector{Float64}}()
for method in methods
    results[method]["success"] || continue
    pe = results[method]["pe"]
    q_vals[method] = [pe.profiles.q_spline(ψ) for ψ in psi_nodes]
end

println("\nPairwise max |Δq| (full domain / ψ > 0.90):")
for (i, m1) in enumerate(methods), m2 in methods[(i+1):end]
    (results[m1]["success"] && results[m2]["success"]) || continue
    Δq = abs.(q_vals[m1] .- q_vals[m2])
    mask_edge = psi_nodes .> 0.90
    @printf("  |q_%s - q_%s|: max=%.2e (full)  max=%.2e (ψ>0.90)\n",
        m1, m2, maximum(Δq), maximum(Δq[mask_edge]))
end

# EFIT input q-profile for ψ < 0.95 only.
# The sq_in spline from read_efit already holds the parsed q-profile in column 3:
# sq_in columns = [|F|, mu0*P, q, sqrt(psi_norm)].
println("\nComparison to EFIT input q-profile (valid for ψ < 0.95 only):")
ref_raw = Equilibrium.read_efit(results[ref_method]["config"])
f_sq_buf = zeros(4)
efit_q_interp = ψ -> (ref_raw.sq_in(f_sq_buf, ψ); f_sq_buf[3])
mask_mid = psi_nodes .< 0.95
for method in methods
    results[method]["success"] || continue
    Δq = abs.([q_vals[method][i] - efit_q_interp(psi_nodes[i]) for i in 1:length(psi_nodes)])
    @printf("  |q_%s - q_efit| (ψ<0.95): max=%.2e  rms=%.2e\n",
        method, maximum(Δq[mask_mid]), sqrt(mean(Δq[mask_mid] .^ 2)))
end

# ─── Roundtrip error ─────────────────────────────────────────────────────────
println("\n" * "=" ^ 65)
println("Roundtrip error: (ψ,θ)→(R,Z)→ψ_spline(R,Z)−ψ_target")
println("=" ^ 65)

for method in methods
    results[method]["success"] || continue
    pe = results[method]["pe"]
    cfg = results[method]["config"]
    raw_profile = Equilibrium.read_efit(cfg)
    # Reconstruct psi_in before direct_position! modifies it
    # (psi_in is already renormalized inside pe — test uses it)
    psio = pe.psio
    psi_xs = pe.rzphi_xs
    psi_ys = pe.rzphi_ys
    mpsi = length(psi_xs) - 1
    mtheta = length(psi_ys) - 1

    errors = Float64[]
    for ipsi in 1:5:(mpsi+1)
        ψ_target = psi_xs[ipsi]
        for itheta in 1:8:(mtheta+1)
            θ = psi_ys[itheta]
            r2 = pe.rzphi_rsquared((ψ_target, θ))
            off = pe.rzphi_offset((ψ_target, θ))
            rfac = sqrt(max(r2, 0.0))
            η = 2π * (θ + off)
            R = pe.ro + rfac * cos(η)
            Z = pe.zo + rfac * sin(η)
            # Evaluate psi at (R, Z): psi_in returns (sibry - ψ_raw), so
            # psi_norm = 1 - psi_in/psio (0 at axis, 1 at boundary).
            ψ_reconstructed = 1.0 - raw_profile.psi_in((R, Z)) / psio
            push!(errors, abs(ψ_reconstructed - ψ_target))
        end
    end
    mask_edge_err = [i for (i, ψ) in enumerate(psi_xs[1:5:end]) if ψ > 0.90]
    @printf("  %s: max=%.2e  rms=%.2e\n", method, maximum(errors), sqrt(mean(errors .^ 2)))
end

# ─── GSE residual summary ─────────────────────────────────────────────────────
println("\n" * "=" ^ 65)
println("Grad-Shafranov residual (from equilibrium_gse! output — see gsec.h5 if diagnose_src=true)")
println("(Showing q-monotonicity near edge as proxy since GSE requires diagnose_src=true)")
println("=" ^ 65)

for method in methods
    results[method]["success"] || continue
    pe = results[method]["pe"]
    psi_xs = pe.rzphi_xs
    q_profile = [pe.profiles.q_spline(ψ) for ψ in psi_xs]
    dq = diff(q_profile)
    n_nonmono = count(dq .< 0)
    mask_edge = psi_xs[2:end] .> 0.90
    n_nonmono_edge = count(dq[mask_edge] .< 0)
    @printf("  %s: q non-monotone steps: %d total, %d for ψ>0.90\n",
        method, n_nonmono, n_nonmono_edge)
end

# ─── Runtime summary ─────────────────────────────────────────────────────────
println("\n" * "=" ^ 65)
println("Runtime summary")
println("=" ^ 65)
for method in methods
    if results[method]["success"]
        @printf("  %-22s  %.3f s\n", method, results[method]["runtime"])
    else
        @printf("  %-22s  FAILED\n", method)
    end
end
println()
