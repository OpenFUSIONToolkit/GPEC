# Visualizes where the two-pass auto grid places its radial knots compared to the fixed
# ldp (sin²) grids, on the DIII-D-like example. Two panels: cumulative knot fraction
# index/mpsi vs ψ_N (packing profile), and local knot spacing Δψ vs ψ_N (log scale),
# with the n=1 rational surfaces marked. Documents the auto-grid PR.
#
# Usage: julia --project=. benchmarks/plot_grid_knot_placement.jl
# Output (not committed): benchmarks/grid_knot_placement.png

using Pkg;
Pkg.activate(joinpath(@__DIR__, ".."))
using GeneralizedPerturbedEquilibrium
using Plots

const GPE = GeneralizedPerturbedEquilibrium
const EXAMPLE_DIR = joinpath(@__DIR__, "..", "examples", "DIIID-like_ideal_example")

function build_grids()
    _, eq_config, additional_input = GPE.build_inputs_from_toml(EXAMPLE_DIR)
    tau = eq_config.psi_accuracy

    # Pass 1 (coarse layout) and the two-pass refined grid, as the driver builds them
    equil1 = GPE.Equilibrium.setup_equilibrium(eq_config, additional_input)
    pass1 = copy(equil1.profiles.xs)
    rationals = GPE.ForceFreeStates.rational_psi_nodes(equil1; nlow=1, nhigh=1)
    auto = GPE.Equilibrium.refined_psi_grid(equil1; tau, mandatory=rationals)

    ldp(mpsi) = [eq_config.psilow + (eq_config.psihigh - eq_config.psilow) * sin((i / mpsi) * (π / 2))^2 for i in 0:mpsi]

    grids = [
        ("auto two-pass, τ=$(tau), mpsi=$(length(auto) - 1)", auto),
        ("pass-1 layout, mpsi=$(length(pass1) - 1)", pass1),
        ("ldp, mpsi=256", ldp(256)),
        ("ldp, mpsi=512", ldp(512))
    ]
    return grids, rationals
end

function main()
    grids, rationals = build_grids()
    # Okabe-Ito colorblind-safe hues, fixed order; pass-1 recessive gray
    colors = ["#0072B2", "#999999", "#E69F00", "#009E73"]
    styles = [:solid, :dash, :solid, :solid]

    p1 = plot(; xlabel="ψ_N", ylabel="knot index / mpsi", title="Radial knot packing — DIIID-like example (n=1)",
        legend=:bottomright, left_margin=12Plots.mm, bottom_margin=4Plots.mm, size=(900, 420))
    p2 = plot(; xlabel="ψ_N", ylabel="local knot spacing Δψ_N", yscale=:log10,
        legend=:bottomleft, left_margin=12Plots.mm, bottom_margin=6Plots.mm, size=(900, 420))

    for (i, (label, g)) in enumerate(grids)
        n = length(g) - 1
        plot!(p1, g, (0:n) ./ n; lw=2, color=colors[i], ls=styles[i], label=label)
        mids = 0.5 .* (g[1:(end-1)] .+ g[2:end])
        plot!(p2, mids, diff(g); lw=2, color=colors[i], ls=styles[i], label=label)
    end
    for (j, r) in enumerate(rationals)
        for p in (p1, p2)
            vline!(p, [r]; color="#CC79A7", ls=:dot, lw=1.5, label=(j == 1 && p === p1) ? "rational surfaces q=m/1" : "")
        end
    end

    fig = plot(p1, p2; layout=(2, 1), size=(900, 840))
    out = joinpath(@__DIR__, "grid_knot_placement.png")
    savefig(fig, out)
    println("Wrote ", abspath(out))
end

main()
