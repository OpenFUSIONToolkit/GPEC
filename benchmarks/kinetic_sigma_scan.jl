"""
kinetic_sigma_scan.jl

Scans the fixed kinetic matrix scaling parameter σ and plots:
  1. et[1] (least stable eigenvalue) vs σ — shows kinetic damping effect
  2. Eigenmode |ξ_ψ(ψ)| profiles for m=0..4 — shows kinetic modification near rational surfaces

Uses the continuous one-chunk integration path (kinetic_factor > 0 enables kinetic mode).

Usage:
  julia --project=. benchmarks/kinetic_sigma_scan.jl [example_path]
"""

using GeneralizedPerturbedEquilibrium
using TOML, Printf, HDF5, Plots

example_path = length(ARGS) > 0 ? ARGS[1] : joinpath(@__DIR__, "../examples/DIIID-like_ideal_example")
config_path = joinpath(example_path, "gpec.toml")
output_dir = @__DIR__

sigma_values = [1e-12, 1e-9, 1e-6]

"""
Run GPEC with kinetic fixed matrices at the given σ. Returns (et, psi, xi_psi, nstep_total)
or nothing on failure. Uses a temporary directory to avoid modifying the example.
"""
function run_kinetic(config_path::String, sigma::Float64)
    example_dir = dirname(config_path)
    raw = TOML.parsefile(config_path)
    raw["ForceFreeStates"]["kinetic_source"] = "fixed"
    raw["ForceFreeStates"]["kinetic_factor"] = sigma
    raw["ForceFreeStates"]["save_interval"] = 1
    raw["ForceFreeStates"]["verbose"] = false

    mktempdir() do tmpdir
        open(joinpath(tmpdir, "gpec.toml"), "w") do io
            TOML.print(io, raw)
        end
        for f in readdir(example_dir)
            src = abspath(joinpath(example_dir, f))
            isfile(src) && !endswith(f, ".toml") && symlink(src, joinpath(tmpdir, f))
        end
        try
            GeneralizedPerturbedEquilibrium.main([tmpdir])
            h5open(joinpath(tmpdir, "gpec.h5"), "r") do h5
                et = read(h5["vacuum/et"])
                psi = read(h5["integration/psi"])
                xi_psi = read(h5["integration/xi_psi"])
                nstep_total = read(h5["integration/nstep_total"])
                return (et=et, psi=psi, xi_psi=xi_psi, nstep_total=nstep_total)
            end
        catch e
            @warn "sigma=$sigma FAILED" exception = e
            return nothing
        end
    end
end

# Also run ideal reference (kinetic_factor=0)
function run_ideal(config_path::String)
    example_dir = dirname(config_path)
    raw = TOML.parsefile(config_path)
    raw["ForceFreeStates"]["kinetic_factor"] = 0.0
    raw["ForceFreeStates"]["save_interval"] = 1
    raw["ForceFreeStates"]["verbose"] = false

    mktempdir() do tmpdir
        open(joinpath(tmpdir, "gpec.toml"), "w") do io
            TOML.print(io, raw)
        end
        for f in readdir(example_dir)
            src = abspath(joinpath(example_dir, f))
            isfile(src) && !endswith(f, ".toml") && symlink(src, joinpath(tmpdir, f))
        end
        try
            GeneralizedPerturbedEquilibrium.main([tmpdir])
            h5open(joinpath(tmpdir, "gpec.h5"), "r") do h5
                et = read(h5["vacuum/et"])
                psi = read(h5["integration/psi"])
                xi_psi = read(h5["integration/xi_psi"])
                nstep_total = read(h5["integration/nstep_total"])
                return (et=et, psi=psi, xi_psi=xi_psi, nstep_total=nstep_total)
            end
        catch e
            @warn "Ideal run FAILED" exception = e
            return nothing
        end
    end
end

# ---- JIT warmup ----
println("JIT warmup...")
run_ideal(config_path)

# ---- Run ideal reference ----
println("\nRunning ideal reference...")
ideal = run_ideal(config_path)
if ideal !== nothing
    @printf("  Ideal: et[1] = %+.6f %+.6fi, steps = %d, saved = %d\n",
        real(ideal.et[1]), imag(ideal.et[1]), ideal.nstep_total, length(ideal.psi))
end

# ---- Run kinetic sigma scan ----
println("\nRunning kinetic sigma scan...")
results = Dict{Float64,Any}()
for sigma in sigma_values
    @printf("  sigma = %.0e ... ", sigma)
    result = run_kinetic(config_path, sigma)
    if result !== nothing
        results[sigma] = result
        @printf("et[1] = %+.6f %+.6fi, steps = %d, saved = %d\n",
            real(result.et[1]), imag(result.et[1]), result.nstep_total, length(result.psi))
    else
        println("FAILED")
    end
end

# ---- Figure 1: et[1] vs sigma ----
println("\nGenerating eigenvalue plot...")
sigmas_plot = Float64[]
et_real = Float64[]
et_imag = Float64[]
for sigma in sigma_values
    haskey(results, sigma) || continue
    push!(sigmas_plot, sigma)
    push!(et_real, real(results[sigma].et[1]))
    push!(et_imag, imag(results[sigma].et[1]))
end

p1 = plot(sigmas_plot, et_real;
    xscale=:log10, label="Re(et[1])", marker=:circle, lw=2,
    xlabel="σ (kinetic scaling)", ylabel="et[1]",
    title="Kinetic Damping: Eigenvalue vs σ",
    legend=:topleft, left_margin=12Plots.mm, bottom_margin=6Plots.mm)
plot!(p1, sigmas_plot, et_imag;
    label="Im(et[1])", marker=:diamond, lw=2, ls=:dash)
if ideal !== nothing
    hline!(p1, [real(ideal.et[1])]; label="Ideal Re(et[1])", ls=:dot, color=:gray, lw=1)
end

fig1_path = joinpath(output_dir, "kinetic_sigma_scan_et1.png")
savefig(p1, fig1_path)
println("  Saved: $fig1_path")

# ---- Figure 2: Eigenmode profiles ----
println("Generating eigenmode profile plots...")

# Read mlow from the HDF5 to determine m indices
mlow = h5open(joinpath(example_path, "gpec.h5"), "r") do h5
    haskey(h5, "info/mlow") ? read(h5["info/mlow"]) : -8
end

m_values = [0, 1, 2, 3, 4]
sigma_colors = palette(:viridis, length(sigma_values))

p2 = plot(; layout=(length(m_values), 1), size=(900, 250 * length(m_values)),
    left_margin=14Plots.mm, bottom_margin=6Plots.mm, right_margin=4Plots.mm)

for (im, m) in enumerate(m_values)
    m_idx = m - mlow + 1
    ideal_ymax = 0.0

    # Plot ideal reference first
    if ideal !== nothing
        if 1 <= m_idx <= size(ideal.xi_psi, 1)
            # xi_psi is (numpert_total, numpert_total, nstep) — first column is leading eigenvector
            profile = abs.(ideal.xi_psi[m_idx, 1, :])
            ideal_ymax = maximum(profile)
            plot!(p2, ideal.psi, profile;
                subplot=im, label=(im == 1 ? "Ideal" : ""), color=:black, lw=2, ls=:dash,
                marker=:circle, markersize=1, markerstrokewidth=0)
        end
    end

    # Plot kinetic results
    for (isig, sigma) in enumerate(sigma_values)
        haskey(results, sigma) || continue
        r = results[sigma]
        if 1 <= m_idx <= size(r.xi_psi, 1)
            profile = abs.(r.xi_psi[m_idx, 1, :])
            lbl = (im == 1) ? @sprintf("σ=%.0e", sigma) : ""
            plot!(p2, r.psi, profile;
                subplot=im, label=lbl, color=sigma_colors[isig], lw=1.5,
                marker=:circle, markersize=1, markerstrokewidth=0)
        end
    end

    # Scale y-axis based on ideal profile maximum (with 20% headroom)
    ymax = ideal_ymax > 0 ? 1.5 * ideal_ymax : nothing
    plot!(p2; subplot=im, ylabel="m=$m |ξ_ψ|",
        xlabel=(im == length(m_values) ? "ψ" : ""),
        ylims=(ymax !== nothing ? (0, ymax) : :auto))
end
plot!(p2; plot_title="Eigenmode Profiles: Kinetic σ Scan")

fig2_path = joinpath(output_dir, "kinetic_sigma_scan_eigenmodes.png")
savefig(p2, fig2_path)
println("  Saved: $fig2_path")

# ---- Summary table ----
println("\n" * "=" ^ 70)
println("Summary: et[1] vs σ")
println("=" ^ 70)
@printf("  %-12s  %-24s  %-24s  %s\n", "σ", "Re(et[1])", "Im(et[1])", "Steps")
println("  " * "-" ^ 66)
if ideal !== nothing
    @printf("  %-12s  %+24.10f  %+24.10f  %d\n", "ideal", real(ideal.et[1]), imag(ideal.et[1]), ideal.nstep_total)
end
for sigma in sigma_values
    haskey(results, sigma) || continue
    r = results[sigma]
    @printf("  %-12.0e  %+24.10f  %+24.10f  %d\n", sigma, real(r.et[1]), imag(r.et[1]), r.nstep_total)
end
println()
