# Synthetic 2/1 → 4/2 island-bifurcation demonstration on the DIII-D-like equilibrium.
#
# Uses ONLY the I-coils (iu, il). The current in each 6-coil array is
#     I(β) = A1·cos(β)·[n=1 pattern] + A2·sin(β)·[n=2 pattern]
# so the n=1 (2,1) and n=2 (4,2) drives are anti-correlated — each β has one dominant
# resonance. At the q=2 surface (ψ_N ≈ 0.518) this gives
#     β = 0    → 2/1 island  (2 O-points)   [pure n=1]
#     β = π/2  → 4/2 island  (4 O-points)   [pure n=2]
#     β = π    → 2/1 island  (2 O-points)   [pure n=1, flipped]
# The vacuum field-line trace resolves the multi-n coil field (n = 1…3), so the
# competing resonances — and the bifurcation — appear directly in the Poincaré section.

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))
using GeneralizedPerturbedEquilibrium, HDF5, Plots
using GeneralizedPerturbedEquilibrium: Analysis
gr()

const HERE = @__DIR__

# Six-coil toroidal phasings (φ_j = 60°·j).
const N1 = [cos(2π * 1 * j / 6) for j in 0:5]   # n=1: [1, .5, -.5, -1, -.5, .5]
const N2 = [cos(2π * 2 * j / 6) for j in 0:5]   # n=2: [1, -.5, -.5, 1, -.5, -.5]
const A1 = 5000.0                                # n=1 amplitude [A]
const A2 = 3000.0                                # n=2 amplitude [A]

# The n=1 and n=2 drives are anti-correlated so each panel has a single dominant resonance:
# I(β) = A1·cos(β)·N1 + A2·sin(β)·N2 → β=0 pure n=1 (2/1), β=π/2 pure n=2 (4/2), β=π pure n=1.
# The vacuum flux trace uses the coil field directly, so the 4/2 case works even with 0 n=1 forcing.
const CASES = [(β=0.0, label="2over1_a", title="β=0 : 2/1 island (pure n=1)"),
               (β=π / 2, label="4over2", title="β=π/2 : 4/2 island (pure n=2)"),
               (β=π, label="2over1_b", title="β=π : 2/1 island (pure n=1, flipped)")]

_arr(v) = "[" * join(string.(round.(v; digits=1)), ", ") * "]"

function write_toml(dir, β)
    I = A1 * cos(β) .* N1 .+ A2 * sin(β) .* N2   # engineered I-coil currents [A]
    base = read(joinpath(HERE, "gpec.toml"), String)
    # Replace the two headline current lines with the swept currents (same for iu and il),
    # and point the equilibrium path up one level (configs are written into a subdirectory).
    base = replace(base,
        r"currents = \[3000.0, -1500.0, -1500.0, 3000.0, -1500.0, -1500.0\][^\n]*" =>
            "currents = $(_arr(I))")
    base = replace(base,
        "eq_filename = \"TkMkr_D3Dlike_Hmode.geqdsk\"" =>
            "eq_filename = \"../TkMkr_D3Dlike_Hmode.geqdsk\"")
    write(joinpath(dir, "gpec.toml"), base)
    return I
end

function plot_panel(h5, ttl, i)
    f = h5open(h5, "r")
    theta = read(f["field_line_tracing/punctures/theta"])
    psi = read(f["field_line_tracing/punctures/psi"])
    line = read(f["field_line_tracing/punctures/line_id"])
    close(f)
    keep = (psi .>= 0.44) .& (psi .<= 0.60) .& .!isnan.(theta)
    p = plot(; ylims=(0.44, 0.60), xlims=(0, 360), ylabel="ψ_N", legend=false, grid=false,
             title=ttl, titlefontsize=10, left_margin=8Plots.mm,
             bottom_margin=(i == 3 ? 5Plots.mm : 1Plots.mm))
    for lid in unique(line[keep])
        m = keep .& (line .== lid)
        scatter!(p, 360 .* theta[m], psi[m]; markersize=0.5, markerstrokewidth=0, color=:black)
    end
    i == 3 && plot!(p; xlabel="Poloidal Angle (θ) [deg]")
    return p
end

panels = Plots.Plot[]
for (i, c) in enumerate(CASES)
    dir = joinpath(HERE, c.label)
    mkpath(dir)
    I = write_toml(dir, c.β)
    @info "==== BIFURCATION RUN: $(c.title) ====\n   I-coil currents [A] = $(round.(I; digits=1))"
    GeneralizedPerturbedEquilibrium.main([dir])
    h5 = joinpath(dir, "gpec.h5")
    push!(panels, plot_panel(h5, c.title, i))
    # Per-case standard FieldLineTracing figures.
    savefig(Analysis.FieldLineTracing.plot_poincare(h5), joinpath(dir, "poincare_RZ_$(c.label).png"))
    savefig(Analysis.FieldLineTracing.plot_poincare_flux(h5), joinpath(dir, "poincare_flux_$(c.label).png"))
    savefig(Analysis.FieldLineTracing.plot_island_widths(h5), joinpath(dir, "island_widths_$(c.label).png"))
    savefig(Analysis.FieldLineTracing.plot_field_line_tracing_summary(h5), joinpath(dir, "summary_$(c.label).png"))
end

fig = plot(panels...; layout=(3, 1), size=(720, 940),
           plot_title="DIII-D-like 2/1 → 4/2 island bifurcation (I-coil only, vacuum trace)",
           plot_titlefontsize=11)
out = joinpath(HERE, "bifurcation_2to4to2.png")
savefig(fig, out)

println("\nFigures written:")
println("  ", out, "   (3-panel 2/1 → 4/2 → 2/1 bifurcation)")
for c in CASES
    println("  ", joinpath(HERE, c.label), "/{poincare_RZ,poincare_flux,island_widths,summary}_$(c.label).png")
end
