"""
Generate Test 3 convergence plot from edge_jacobian_test_plan.jl output.
Saved to benchmarks/edge_jacobian_test3_psihigh_convergence.png
"""

using Plots

psihigh_vals = [0.985, 0.990, 0.993, 0.995, 0.997, 0.999]
et_std    = [2.155422, 1.242193, 1.210462, 1.182732, 1.145024, 0.733580]
et_eq_arc = [1.828996, 0.721240, 0.702160, 0.685396, 0.662933, 4.196613]

p = plot(
    psihigh_vals, et_std,
    label="hamada standard",
    marker=:circle, markersize=6, linewidth=2, color=:steelblue,
    xlabel="psihigh", ylabel="et[1] (real part)",
    title="use_equal_arc_vacuum: psihigh convergence scan\n(DIIID-like hamada, mpsi=128, ldp grid)",
    legend=:topleft, grid=true, minorgrid=true,
    size=(800, 500), dpi=150,
    ylims=(-0.5, 5.0)
)
plot!(p, psihigh_vals, et_eq_arc,
    label="hamada + equal-arc vacuum",
    marker=:square, markersize=6, linewidth=2, color=:darkorange,
    linestyle=:dash
)

# Annotate the cap-hit warning at psihigh=0.999
annotate!(p, 0.999, 4.196613 + 0.15,
    text("X-point cap hit\n(max_rate=9.9 > cap 8.0)", 8, :darkorange, :center))

# Add shaded region to indicate expected convergence zone
vline!(p, [0.999], linestyle=:dot, color=:red, alpha=0.5, label="X-point regime")

outpath = joinpath(@__DIR__, "edge_jacobian_test3_psihigh_convergence.png")
savefig(p, outpath)
println("Saved: $outpath")
