"""
Diagnostic script: compare eigenvalues across Jacobian types (hamada, pest, park, boozer).

Runs the DIIID-like example with each Jacobian, then plots:
  - ξ-space eigenvalues (should differ across Jacobians)
  - Power-normalized flux eigenvalues (should agree across Jacobians)

Usage:
    julia --project=. scripts/compare_jacobians_power_norm.jl

Requires ~3 min per Jacobian (~12 min total).
"""

using TOML
using HDF5
using Plots
using Printf
using Statistics
using GeneralizedPerturbedEquilibrium

const JACOBIANS = ["hamada", "pest", "park", "boozer"]
const EXAMPLE_DIR = "examples/DIIID-like_ideal_example"
const COLORS = Dict("hamada" => :blue, "pest" => :red, "park" => :green, "boozer" => :orange)

# Toroidal mode number for this comparison — embedded into output filenames
# so artefacts from different n can be kept side by side.
const NN = 1

function run_with_jacobian(jac_type::String, base_dir::String, work_dir::String)
    # Copy example to temp directory
    cp(base_dir, work_dir; force=true)

    # Read and modify TOML — settings chosen for apples-to-apples Jacobian comparison:
    #   psihigh=0.994, mtheta=1024: match the proven-invariant regime in
    #     scripts/test_power_norm_invariance.jl (floor ~6.9e-7).
    #   delta_mlow/delta_mhigh=32: give a generous auto-expanded m-band. q(ψ) is a
    #     flux-surface invariant, so qmin/qmax are identical across jacs — the
    #     auto-computed [mlow, mhigh] should match as well, which the mpert print
    #     in main() verifies.
    toml_path = joinpath(work_dir, "gpec.toml")
    config = TOML.parsefile(toml_path)
    config["Equilibrium"]["jac_type"] = jac_type
    config["Equilibrium"]["psihigh"] = 0.994
    config["Equilibrium"]["mtheta"] = 1024
    config["ForceFreeStates"]["nn_low"] = NN
    config["ForceFreeStates"]["nn_high"] = NN
    config["ForceFreeStates"]["psiedge"] = 0.98
    config["ForceFreeStates"]["delta_mlow"] = 32
    config["ForceFreeStates"]["delta_mhigh"] = 32
    config["ForceFreeStates"]["force_termination"] = true
    open(toml_path, "w") do io
        TOML.print(io, config)
    end

    # Run GPEC
    @info "Running with jac_type = $jac_type"
    GeneralizedPerturbedEquilibrium.main([work_dir])
    return joinpath(work_dir, "gpec.h5")
end

function main()
    tmpdir = mktempdir()
    results = Dict{String,Dict{String,Any}}()

    for jac in JACOBIANS
        work_dir = joinpath(tmpdir, jac)
        h5_path = run_with_jacobian(jac, EXAMPLE_DIR, work_dir)
        h5open(h5_path, "r") do f
            mpert_here = haskey(f, "info/mpert") ? read(f, "info/mpert") : -1
            mlow_here = haskey(f, "info/mlow") ? read(f, "info/mlow") : 0
            mhigh_here = haskey(f, "info/mhigh") ? read(f, "info/mhigh") : 0
            results[jac] = Dict(
                "psi" => read(f, "edge_scan/psi"),
                "et" => real.(read(f, "edge_scan/total_energy")),
                "ep" => real.(read(f, "edge_scan/plasma_energy")),
                "ev" => real.(read(f, "edge_scan/vacuum_energy")),
                "pn_et" => real.(read(f, "edge_scan/pn_total_energy")),
                "pn_ep" => real.(read(f, "edge_scan/pn_plasma_energy")),
                "pn_ev" => real.(read(f, "edge_scan/pn_vacuum_energy")),
                "mpert" => mpert_here,
                "mlow" => mlow_here,
                "mhigh" => mhigh_here
            )
        end
    end

    # Apples-to-apples diagnostic: report mlow/mhigh/mpert per jac. Because q(ψ) is
    # a flux-surface invariant, qmin/qmax and thus the auto-computed m-range should
    # match across jacs. A mismatch here would invalidate the comparison.
    println("\n--- m-range per Jacobian (should all match — q is a flux-surface invariant) ---")
    @Printf.printf("  %-7s  %6s  %6s  %6s\n", "jac", "mlow", "mhigh", "mpert")
    for jac in JACOBIANS
        r = results[jac]
        @Printf.printf("  %-7s  %6d  %6d  %6d\n", jac, r["mlow"], r["mhigh"], r["mpert"])
    end

    # Symmetric ylims keyed on 3× median |value| — a robust trend scale that ignores
    # rational-surface spikes. ylim = (-ymax*1.1, ymax*1.1).
    function symmetric_ylims(datasets)
        all_vals = Float64[]
        for d in datasets
            append!(all_vals, filter(!isnan, d))
        end
        isempty(all_vals) && return (-1.0, 1.0)
        ymax = 3.0 * median(abs.(all_vals))
        ymax <= 0 && return (-1.0, 1.0)
        return (-ymax * 1.1, ymax * 1.1)
    end

    pn_data = [results[j]["pn_et"] for j in JACOBIANS]

    # Plot ξ-space eigenvalues (expected to differ)
    p1 = plot(; xlabel="ψ", ylabel="δW (ξ-space)", title="ξ-space eigenvalues by Jacobian", legend=:topright, ylims=(-1.0, 1.0))
    for jac in JACOBIANS
        r = results[jac]
        valid = .!isnan.(r["et"])
        plot!(p1, r["psi"][valid], r["et"][valid]; label=jac, lw=2, color=COLORS[jac])
    end
    hline!(p1, [0.0]; label="", color=:black, ls=:dot, alpha=0.5)

    # Plot power-norm eigenvalues (expected to agree)
    p2 = plot(; xlabel="ψ", ylabel="δW (power-norm flux)", title="Power-norm flux eigenvalues by Jacobian", legend=:topright, ylims=symmetric_ylims(pn_data))
    for jac in JACOBIANS
        r = results[jac]
        valid = .!isnan.(r["pn_et"])
        plot!(p2, r["psi"][valid], r["pn_et"][valid]; label=jac, lw=2, color=COLORS[jac])
    end
    hline!(p2, [0.0]; label="", color=:black, ls=:dot, alpha=0.5)

    fig = plot(p1, p2; layout=(2, 1), size=(1600, 1000), left_margin=12Plots.mm, bottom_margin=4Plots.mm)

    outfile = joinpath(EXAMPLE_DIR, "jacobian_comparison_n$(NN).png")
    savefig(fig, outfile)
    println("Saved: $(abspath(outfile))")

    # Edge-scan eigenvalue plot for the hamada run — same content as
    # scripts/compare_eigenvalues_edge_scan.jl, inlined here so we don't need to
    # preserve the (multi-GB) tmpdir h5 after the script exits.
    r_h = results["hamada"]
    valid_h = .!isnan.(r_h["et"]) .& .!isnan.(r_h["pn_et"])
    p3 = plot(r_h["psi"][valid_h], r_h["et"][valid_h]; label="total (et)", lw=2, xlabel="ψ",
        ylabel="δW (ξ-space)", title="ξ-space eigenvalues (hamada)", ylims=(-1.0, 1.0))
    plot!(p3, r_h["psi"][valid_h], r_h["ep"][valid_h]; label="plasma (ep)", lw=1.5, ls=:dash)
    plot!(p3, r_h["psi"][valid_h], r_h["ev"][valid_h]; label="vacuum (ev)", lw=1.5, ls=:dash)
    hline!(p3, [0.0]; label="", color=:black, ls=:dot, alpha=0.5)

    pn_ylims_h = symmetric_ylims([r_h["pn_et"][valid_h], r_h["pn_ep"][valid_h], r_h["pn_ev"][valid_h]])
    p4 = plot(r_h["psi"][valid_h], r_h["pn_et"][valid_h]; label="total (pn_et)", lw=2, xlabel="ψ",
        ylabel="δW (power-norm flux)", title="Power-norm flux eigenvalues (hamada)", ylims=pn_ylims_h)
    plot!(p4, r_h["psi"][valid_h], r_h["pn_ep"][valid_h]; label="plasma (pn_ep)", lw=1.5, ls=:dash)
    plot!(p4, r_h["psi"][valid_h], r_h["pn_ev"][valid_h]; label="vacuum (pn_ev)", lw=1.5, ls=:dash)
    hline!(p4, [0.0]; label="", color=:black, ls=:dot, alpha=0.5)

    fig_edge = plot(p3, p4; layout=(2, 1), size=(1200, 900), left_margin=12Plots.mm, bottom_margin=4Plots.mm)
    edge_out = joinpath(EXAMPLE_DIR, "gpec_edge_eigenvalues_n$(NN).png")
    savefig(fig_edge, edge_out)
    println("Saved: $(abspath(edge_out))")

    # Compare pn_et across Jacobians at matched ψ values (interpolated to a common grid).
    # Each Jacobian has its own edge-scan truncation, so "last point" lands at different ψ
    # per case — we must compare at the same physical ψ to test invariance.
    psi_min_common = maximum(results[j]["psi"][1] for j in JACOBIANS)
    psi_max_common = minimum(results[j]["psi"][end] for j in JACOBIANS)
    psi_probes = range(psi_min_common, psi_max_common; length=5)

    println("\n--- Eigenvalue comparison at matched ψ (linear interp of valid points) ---")
    println("(test_power_norm_invariance.jl floor on these surfaces is ~6.9e-7; spreads of that")
    println("order indicate full-stack invariance matches the unit-level operator test.)")
    println("ψ           | hamada         | pest           | park           | boozer")
    println("----------- | -------------- | -------------- | -------------- | --------------")
    function interp_at(r, psi_q)
        psi = r["psi"]
        vals = r["pn_et"]
        valid = .!isnan.(vals)
        isempty(filter(identity, valid)) && return NaN
        psi_v = psi[valid]
        vals_v = vals[valid]
        (psi_q < psi_v[1] || psi_q > psi_v[end]) && return NaN
        i = searchsortedlast(psi_v, psi_q)
        i == length(psi_v) && return vals_v[end]
        t = (psi_q - psi_v[i]) / (psi_v[i+1] - psi_v[i])
        return (1 - t) * vals_v[i] + t * vals_v[i+1]
    end
    for psi_q in psi_probes
        vals = [interp_at(results[j], psi_q) for j in JACOBIANS]
        @Printf.printf("%+.4e | %+.6e | %+.6e | %+.6e | %+.6e\n", psi_q, vals...)
        finite = filter(isfinite, vals)
        if length(finite) >= 2
            spread = (maximum(finite) - minimum(finite)) / maximum(abs.(finite))
            @Printf.printf("            relative spread = %.2e %s\n", spread, spread < 1e-3 ? "(invariant)" : "(NOT invariant)")
        end
    end
end

main()
