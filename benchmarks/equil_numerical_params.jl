"""
equil_numerical_params.jl

Sweeps numerical parameters to characterize accuracy vs. cost for each method.

Sweeps:
  - mpsi ∈ {64, 128, 256}              (all methods)
  - mtheta ∈ {128, 256, 512}            (all methods)
  - ODE reltol ∈ {1e-5, 1e-7, 1e-9}    (efit, efit_arclength)
  - Contour grid refinement ∈ {2, 4, 8} (efit_by_inversion, via REFINE env var override)

Accuracy metric: max roundtrip error over a sparse (ψ, θ) sample grid.
Performance metric: wall-clock runtime.

Usage:
  julia --project=. benchmarks/equil_numerical_params.jl [example_path]
"""

using GeneralizedPerturbedEquilibrium
using GeneralizedPerturbedEquilibrium.Equilibrium
using TOML, Printf, Statistics

example_path = length(ARGS) > 0 ? ARGS[1] : joinpath(@__DIR__, "../examples/DIIID-like_ideal_example")
config_path = joinpath(example_path, "gpec.toml")

function make_config(path, eq_type, mpsi, mtheta)
    raw = TOML.parsefile(path)
    raw["Equilibrium"]["eq_type"] = eq_type
    raw["Equilibrium"]["mpsi"] = mpsi
    raw["Equilibrium"]["mtheta"] = mtheta
    return Equilibrium.EquilibriumConfig(raw["Equilibrium"], dirname(path))
end

function roundtrip_error(pe, raw_profile)
    psi_xs = pe.rzphi_xs
    psio = pe.psio
    mtheta = length(pe.rzphi_ys) - 1
    errors = Float64[]
    for ipsi in 1:4:length(psi_xs)
        ψ = psi_xs[ipsi]
        for itheta in 1:8:(mtheta+1)
            θ = pe.rzphi_ys[itheta]
            r2 = pe.rzphi_rsquared((ψ, θ))
            off = pe.rzphi_offset((ψ, θ))
            rfac = sqrt(max(r2, 0.0))
            η = 2π * (θ + off)
            R = pe.ro + rfac * cos(η)
            Z = pe.zo + rfac * sin(η)
            ψ_recon = 1.0 - raw_profile.psi_in((R, Z)) / psio
            push!(errors, abs(ψ_recon - ψ))
        end
    end
    return isempty(errors) ? NaN : maximum(errors)
end

# JIT warmup
println("JIT warmup...")
warmup_cfg = make_config(config_path, "efit", 64, 128)
setup_equilibrium(warmup_cfg)

rows = []

# ── mpsi scan ──────────────────────────────────────────────────────────────────
println("\n" * "=" ^ 65)
println("mpsi scan (mtheta=256, default reltol)")
println("=" ^ 65)
for method in ["efit", "efit_arclength", "efit_by_inversion"]
    for mpsi in [64, 128, 256]
        cfg = make_config(config_path, method, mpsi, 256)
        local pe = nothing
        local success = false
        local runtime = NaN
        local rt_err = NaN
        try
            t = @elapsed pe = setup_equilibrium(cfg)
            runtime = t
            raw = Equilibrium.read_efit(cfg)
            rt_err = roundtrip_error(pe, raw)
            success = true
        catch e
            @warn "  $method mpsi=$mpsi FAILED: $e"
        end
        @printf("  %-22s mpsi=%3d  t=%.2fs  rt=%.2e\n", method, mpsi, runtime, rt_err)
        push!(rows, (sweep="mpsi", method=method, param=Float64(mpsi), param2=256.0,
            success=success, runtime=runtime, roundtrip_max=rt_err))
    end
end

# ── mtheta scan ────────────────────────────────────────────────────────────────
println("\n" * "=" ^ 65)
println("mtheta scan (mpsi=128, default reltol)")
println("=" ^ 65)
for method in ["efit", "efit_arclength", "efit_by_inversion"]
    for mtheta in [128, 256, 512]
        cfg = make_config(config_path, method, 128, mtheta)
        local pe = nothing
        local success = false
        local runtime = NaN
        local rt_err = NaN
        try
            t = @elapsed pe = setup_equilibrium(cfg)
            runtime = t
            raw = Equilibrium.read_efit(cfg)
            rt_err = roundtrip_error(pe, raw)
            success = true
        catch e
            @warn "  $method mtheta=$mtheta FAILED: $e"
        end
        @printf("  %-22s mtheta=%3d  t=%.2fs  rt=%.2e\n", method, mtheta, runtime, rt_err)
        push!(rows, (sweep="mtheta", method=method, param=Float64(mtheta), param2=128.0,
            success=success, runtime=runtime, roundtrip_max=rt_err))
    end
end

# ── ODE reltol scan (efit and efit_arclength only) ────────────────────────────
# Note: reltol is not exposed through EquilibriumConfig yet.
# This scan patches it by temporarily modifying the solver call.
# TODO: expose etol/reltol as a config parameter when needed.
println("\n" * "=" ^ 65)
println("ODE reltol scan — NOTE: not yet exposed via config; using default reltol")
println("(Add reltol to EquilibriumConfig and threading to enable this scan)")
println("=" ^ 65)

# ── Contour grid refinement scan (efit_by_inversion only) ─────────────────────
# The refine factor in equilibrium_solver_by_inversion is a keyword arg.
# We call it directly here, bypassing the setup_equilibrium wrapper.
println("\n" * "=" ^ 65)
println("Contour grid refinement scan (efit_by_inversion only, mpsi=128, mtheta=256)")
println("=" ^ 65)
for refine in [2, 4, 8]
    cfg = make_config(config_path, "efit_by_inversion", 128, 256)
    raw_input = Equilibrium.read_efit(cfg)
    local pe = nothing
    local success = false
    local runtime = NaN
    local rt_err = NaN
    try
        t = @elapsed begin
            pe_raw = Equilibrium.equilibrium_solver_by_inversion(raw_input; refine=refine)
            pe = pe_raw
            Equilibrium.equilibrium_global_parameters!(pe)
            Equilibrium.equilibrium_qfind!(pe)
        end
        runtime = t
        raw_for_check = Equilibrium.read_efit(cfg)
        rt_err = roundtrip_error(pe, raw_for_check)
        success = true
    catch e
        @warn "  refine=$refine FAILED: $e"
    end
    @printf("  efit_by_inversion refine=%2d  t=%.2fs  rt=%.2e\n", refine, runtime, rt_err)
    push!(rows, (sweep="refine", method="efit_by_inversion", param=Float64(refine), param2=128.0,
        success=success, runtime=runtime, roundtrip_max=rt_err))
end

output_csv = joinpath(example_path, "equil_numerical_params.csv")
open(output_csv, "w") do io
    println(io, join(string.(keys(rows[1])), ","))
    for row in rows
        println(io, join(map(v -> v isa AbstractString ? "\"$v\"" : string(v), values(row)), ","))
    end
end
println("\nResults saved to: $output_csv")
println()
