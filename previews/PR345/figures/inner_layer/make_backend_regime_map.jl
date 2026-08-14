# make_backend_regime_map.jl
#
# Figure: accuracy of the three GGJ backends along the imaginary-Q axis. The
# :ray backend is the reference (its own error stays below ~1e-5 out to
# |Q| = 500 — see the convergence figure). The relative error of :shooting and
# :galerkin against it is plotted versus |Q|: :shooting holds to |Q| ~ 1,
# :galerkin to |Q| ~ 4, and both then lose all accuracy, while :ray continues.
# The crossovers of each curve with the 1% line are the practical reach of each
# method — this is what motivates the :ray backend. A few tens of solves; ~1 min.
#
# Run manually:  julia --project=. docs/src/figures/inner_layer/make_backend_regime_map.jl

include(joinpath(@__DIR__, "..", "..", "..", "figure_tools.jl"))

using GeneralizedPerturbedEquilibrium
using LinearAlgebra
using Printf

const IL = GeneralizedPerturbedEquilibrium.InnerLayer
const GGJ = IL.GGJ

p = IL.q4_surface_benchmark()
γ(aq) = im * aq * GGJ.q0(p)                       # imaginary axis: Q = i|Q|

# Worst-case relative error over the two parity components vs the :ray reference.
relerr(Δ, Δref) = maximum(abs.(Δ .- Δref) ./ abs.(Δref))

absQ = 10 .^ range(-1, log10(500); length=22)
ref = [IL.solve_inner(IL.GGJModel(; solver=:ray), p, γ(aq)) for aq in absQ]

gal_x, gal_e = Float64[], Float64[]
shoot_x, shoot_e = Float64[], Float64[]
for (k, aq) in enumerate(absQ)
    try
        Δ = IL.solve_inner(IL.GGJModel(; solver=:galerkin), p, γ(aq))
        all(isfinite, Δ) && (push!(gal_x, aq); push!(gal_e, max(relerr(Δ, ref[k]), 1e-16)))
    catch
    end
    aq > 10 && continue                            # :shooting is meaningless past its regime
    try
        Δ = IL.solve_inner(IL.GGJModel(; solver=:shooting), p, γ(aq))
        all(isfinite, Δ) && (push!(shoot_x, aq); push!(shoot_e, max(relerr(Δ, ref[k]), 1e-16)))
    catch
    end
end

@printf("computed %d :ray refs, %d :galerkin, %d :shooting points\n",
    length(absQ), length(gal_x), length(shoot_x))

plt = plot(; size=(820, 560), legend=:bottomright, xscale=:log10, yscale=:log10,
    xlabel="|Q|  (imaginary axis, Q = i|Q|)",
    ylabel="relative error vs :ray reference",
    title="Backend accuracy along the imaginary-Q axis  (q=4 surface)",
    ylims=(1e-7, 3e0), left_margin=9Plots.mm, bottom_margin=5Plots.mm)

plot!(plt, shoot_x, shoot_e; lw=2, marker=:diamond, ms=5, color=:seagreen, label=":shooting")
plot!(plt, gal_x, gal_e; lw=2, marker=:circle, ms=5, color=:orange, label=":galerkin")
hline!(plt, [1e-2]; color=:red, ls=:dash, lw=1.5, label="1% accuracy")
annotate!(plt, 8.0, 3e-6,
    Plots.text(":ray reference\n(error < 1e-5 to |Q| = 500)", 8, :center, RGB(0.2, 0.3, 0.6)))

save_doc_figure(plt, "inner_layer", "backend_regime_map";
    script=basename(@__FILE__),
    depends=["src/InnerLayer/GGJ/Ray.jl", "src/InnerLayer/GGJ/Galerkin.jl",
        "src/InnerLayer/GGJ/Shooting.jl"])
