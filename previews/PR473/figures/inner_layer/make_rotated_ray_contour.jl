# make_rotated_ray_contour.jl
#
# Figure: the rotated integration ray in the complex layer-coordinate (x) plane.
# Shows why the :ray backend rotates the contour by θ = arg(Q)/4: the real-axis
# contour (θ = 0) runs straight through the pseudo-resonance at
# x² = −Q²(G + K F) (real and large on the imaginary-Q axis), while the rotated
# ray clears it. Geometry only — no BVP solve — so it is cheap to regenerate.
#
# Run manually:  julia --project=. docs/src/figures/inner_layer/make_rotated_ray_contour.jl

include(joinpath(@__DIR__, "..", "..", "..", "figure_tools.jl"))

using GeneralizedPerturbedEquilibrium
using Printf

const IL = GeneralizedPerturbedEquilibrium.InnerLayer
const GGJ = IL.GGJ

# q = 4 rational-surface benchmark on the imaginary-Q axis (the regime that
# defeats :galerkin). Q is the scaled growth rate; θ makes the parabolic
# exponent real.
p = IL.q4_surface_benchmark()
Q = 500.0im
θ = angle(Q) / 4                                   # = π/8 = 22.5°

# Pseudo-resonance location: x² = −Q²(G + K F). On the imaginary-Q axis this is
# real and large, so it sits ON the real-x contour.
x_pr = sqrt(-Q^2 * (p.G + p.K * p.F))
xpr = real(x_pr)                                   # imaginary part ≈ 0 here

# Matching radius from the series-residual criterion along the ray (annotation).
S, _, ok = IL.pick_smax(p, Q; θ=θ)

# Draw both contours out to ~1.8× the pseudo-resonance radius so it is in frame.
smax = 1.8 * xpr
s = range(0, smax; length=400)
ray = cis(θ) .* s                                  # x = e^{iθ} s

plt = plot(; size=(720, 560), legend=:topleft, framestyle=:zerolines,
    xlabel="Re x", ylabel="Im x", aspect_ratio=:equal,
    title="Rotated integration ray, Q = 500i  (q=4 surface)",
    left_margin=6Plots.mm, bottom_margin=4Plots.mm)

# Real-axis contour (θ = 0) and the rotated ray.
plot!(plt, [0, smax], [0, 0]; lw=2, ls=:dash, color=:gray,
    label="real-axis contour (θ = 0)")
plot!(plt, real.(ray), imag.(ray); lw=3, color=1,
    label=@sprintf("rotated ray  x = e^{iθ}s,  θ = %.1f°", rad2deg(θ)))

# Pseudo-resonance on the real axis — the point the rotation steps around.
scatter!(plt, [xpr], [0.0]; marker=:xcross, ms=9, color=:red, msw=3,
    label="pseudo-resonance  x² = −Q²(G+KF)")

# Clearance note above the ray near the pseudo-resonance radius, plus the true
# matching radius. Positions are tied to xpr so they stay inside the frame.
annotate!(plt, 1.02 * xpr, 0.56 * xpr,
    Plots.text("ray stays clear of the\nreal-axis pseudo-resonance", 8, :left,
        RGB(0.15, 0.15, 0.15)))
annotate!(plt, 0.45 * xpr, -0.14 * xpr,
    Plots.text(@sprintf("matching radius S ≈ %.2g%s", S, ok ? "" : " (series tol not met)"),
        8, :gray, :left))

save_doc_figure(plt, "inner_layer", "rotated_ray_contour";
    script=basename(@__FILE__),
    depends=["src/InnerLayer/GGJ/Ray.jl"])
