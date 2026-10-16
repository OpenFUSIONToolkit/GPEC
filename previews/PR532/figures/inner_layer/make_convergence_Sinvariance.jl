# make_convergence_Sinvariance.jl
#
# Figure: honest numerical error bar on the matching data Δ at Q = 500i. The
# `delta_convergence` battery re-solves with each numerical knob perturbed on
# an independent axis (contour angle θ, spectral order p, series order/radius,
# refinement depth, march tolerance, handoff radius, purification e-folds). Δ
# is an analytic invariant of the contour, so the worst-case spread across
# these orthogonal perturbations bounds the true error. Eight solves; ~30 s.
#
# Run manually:  julia --project=. docs/src/figures/inner_layer/make_convergence_Sinvariance.jl

include(joinpath(@__DIR__, "..", "..", "..", "figure_tools.jl"))

using GeneralizedPerturbedEquilibrium
using Printf

const IL = GeneralizedPerturbedEquilibrium.InnerLayer

p = IL.q4_surface_benchmark()
Q = 500.0im

conv = IL.delta_convergence(p, Q; verbose=false)
names = [r.name for r in conv.table]
d1 = [max(r.d1, 1e-16) for r in conv.table]      # relative change of Δ_odd
d2 = [max(r.d2, 1e-16) for r in conv.table]      # relative change of Δ_even
n = length(names)

plt = plot(; size=(820, 540), legend=:topright, yscale=:log10,
    xticks=(1:n, names), xrotation=30, ylims=(1e-12, 1e-2),
    ylabel="relative change of Δ per knob", xlabel="perturbed numerical knob",
    title=@sprintf("Convergence & contour-invariance of Δ,  Q = %gi  (q=4)", imag(Q)),
    left_margin=10Plots.mm, bottom_margin=14Plots.mm)

scatter!(plt, 1:n, d1; marker=:circle, ms=7, color=1, label="δΔ_odd")
scatter!(plt, 1:n, d2; marker=:diamond, ms=7, color=2, label="δΔ_even")

# Worst-case spread — the reported error bar on each parity.
hline!(plt, [conv.spread[1]]; color=1, ls=:dash, lw=1.5,
    label=@sprintf("worst-case Δ_odd  %.1e", conv.spread[1]))
hline!(plt, [conv.spread[2]]; color=2, ls=:dash, lw=1.5,
    label=@sprintf("worst-case Δ_even  %.1e", conv.spread[2]))

save_doc_figure(plt, "inner_layer", "convergence_Sinvariance";
    script=basename(@__FILE__),
    depends=["src/InnerLayer/GGJ/Ray.jl"])
