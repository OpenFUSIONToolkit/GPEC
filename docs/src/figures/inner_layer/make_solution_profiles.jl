# make_solution_profiles.jl
#
# Figure: the inner-layer fields (Ψ, Ξ, Υ) reconstructed on the rotated ray for
# the q=4 benchmark surface. The collocation solution on [0, s_m] (solid) and
# the analytic u_small + Δ·u_big asymptotic representation for s ≥ S (dashed)
# share the same power-law tail — the numeric↔asymptotic overlap the outer-
# region matching relies on. One BVP solve; a few minutes at large imaginary Q.
#
# Run manually:  julia --project=. docs/src/figures/inner_layer/make_solution_profiles.jl

include(joinpath(@__DIR__, "..", "..", "..", "figure_tools.jl"))

using GeneralizedPerturbedEquilibrium
using Printf

const IL = GeneralizedPerturbedEquilibrium.InnerLayer
const GGJ = IL.GGJ

p = IL.q4_surface_benchmark()
Q = 2.0im                  # imaginary axis, beyond the |Q|≪1 shooting regime,
#                            small enough that the collocation domain reaches
#                            the series radius directly (no march) — so the
#                            numeric and asymptotic segments join seamlessly.
isol = 1                   # "odd" parity solution (Ψ'(0)=Ξ(0)=Υ(0)=0)

res = IL.solve_ray(p, Q)
s_m = res.breaks[end]
@printf("solve_ray: Q=%s  θ=%.3f  S=%.4g  s_m=%.4g  resid=%.1e\n",
    string(Q), res.θ, res.S, s_m, res.resid)

# Collocation solution over the BVP domain [0, s_m].
prof = IL.solution_profile(res; npc=8)

# Analytic tail on [S, few·S] where the inps series is trusted.
srange = 10 .^ range(log10(res.S), log10(4 * res.S); length=80)
asy = IL.asymptotic_profile(p, res, srange)

fields = ((:Ψ, prof.Ψ, asy.Ψ, 1), (:Ξ, prof.Ξ, asy.Ξ, 2), (:Υ, prof.Υ, asy.Υ, 3))

plt = plot(; size=(760, 560), legend=:bottomleft, xscale=:log10, yscale=:log10,
    ylims=(1e-3, 1e2),
    xlabel="ray parameter s   (x = e^{iθ}s)", ylabel="|field|  (odd parity)",
    title=@sprintf("Inner-layer fields on the rotated ray, Q = %gi  (q=4)", imag(Q)),
    left_margin=8Plots.mm, bottom_margin=5Plots.mm)

for (nm, col, acol, ci) in fields
    plot!(plt, prof.s[2:end], abs.(col[2:end, isol]); lw=2.5, color=ci,
        label="|$(nm)|  collocation")
    plot!(plt, asy.s, abs.(acol[:, isol]); lw=2, ls=:dash, color=ci, label="|$(nm)|  asymptotic")
end

# Matching radius: with no march the BVP edge s_m coincides with the series
# radius S, so numeric and asymptotic meet at a single point.
vline!(plt, [res.S]; color=:black, ls=:dot, lw=1,
    label=@sprintf("S = s_m ≈ %.1f  (match point)", res.S))

save_doc_figure(plt, "inner_layer", "solution_profiles";
    script=basename(@__FILE__),
    depends=["src/InnerLayer/GGJ/Ray.jl", "src/InnerLayer/GGJ/RayAsymptotics.jl",
        "src/InnerLayer/GGJ/InnerAsymptotics.jl"])
