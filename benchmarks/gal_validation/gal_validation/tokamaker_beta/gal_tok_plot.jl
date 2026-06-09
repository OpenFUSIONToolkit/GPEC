# Plot the TokaMaker beta-scan pfac convergence study. Data: /tmp/gal_tok_scan.csv
# (beta,pfac,m,q,gal,stride). Produces: per-pfac error summary, a heatmap (beta x pfac)
# of median-over-surfaces tracking error, and per-q tracking-vs-beta at the chosen pfac.
using Pkg; Pkg.activate("/Users/pharr/Projects/GPEC_dev/docs/GPEC_julia")
using Plots, DelimitedFiles, Printf; gr()

d, h = readdlm("/tmp/gal_tok_scan.csv", ','; header=true)
col(n) = d[:, findfirst(==(n), strip.(vec(h)))]
beta = col("beta"); pfac = col("pfac"); m = Int.(col("m")); q = col("q"); g = col("gal"); s = col("stride")
# Exclude q=1: the internal-kink surface where the outer-region Δ′ construction does not apply
# (gal and STRIDE diverge structurally, pfac-independent). Keep the tearing surfaces q≥2.
keep = round.(q; digits=0) .>= 2
beta = beta[keep]; pfac = pfac[keep]; m = m[keep]; q = q[keep]; g = g[keep]; s = s[keep]
finite = isfinite.(g) .& isfinite.(s) .& (abs.(s) .> 1e-9)
relerr = fill(NaN, length(g))
relerr[finite] = 100 .* abs.(g[finite] .- s[finite]) ./ abs.(s[finite])

betas = sort(unique(beta)); pfacs = sort(unique(pfac))
med(v) = (w = sort(filter(isfinite, v)); isempty(w) ? NaN : (n = length(w); iseven(n) ? (w[n÷2]+w[n÷2+1])/2 : w[(n+1)÷2]))
qq(v, p) = (w = sort(filter(isfinite, v)); isempty(w) ? NaN : w[clamp(ceil(Int, p*length(w)), 1, length(w))])

# --- per-pfac summary over ALL (beta, surface) ---
println("pfac        median%   p90%    finite-max%   n")
for pf in pfacs
    r = relerr[pfac .≈ pf]
    @printf("%.2e    %7.3f  %7.2f  %9.1f   %d\n", pf, med(r), qq(r, 0.9), maximum(filter(isfinite, r); init=NaN), count(isfinite, r))
end

# --- heatmap: median over surfaces, per (beta, pfac) ---
Z = fill(NaN, length(pfacs), length(betas))
for (i, pf) in enumerate(pfacs), (j, b) in enumerate(betas)
    Z[i, j] = med(relerr[(pfac .≈ pf) .& (beta .≈ b)])
end
Zc = clamp.(Z, 0.01, 100)
p1 = heatmap(betas, pfacs, log10.(Zc); yscale=:log10, xlabel="β_N", ylabel="packing pfac",
    title="TokaMaker β scan: gal tracking vs STRIDE (median over q≥2 surfaces)\nlog₁₀ |gal−STRIDE|/STRIDE (%)",
    colorbar_title="log₁₀(% err)", c=:viridis,
    left_margin=12Plots.mm, bottom_margin=6Plots.mm, right_margin=10Plots.mm)

# --- per-pfac error curves vs beta (median over surfaces) ---
p2 = plot(; xlabel="β_N", ylabel="median |gal−STRIDE|/STRIDE (%)", yscale=:log10, legend=:topright,
    left_margin=12Plots.mm, bottom_margin=6Plots.mm, right_margin=4Plots.mm,
    title="median tracking error vs β per packing ratio")
sel = [findmin(abs.(log10.(pfacs) .- t))[2] for t in (-4, -3, -2, -1.5, -1, 0)]
for i in unique(sel)
    yv = [max(Z[i, j], 1e-3) for j in eachindex(betas)]
    plot!(p2, betas, yv; lw=2, marker=:circle, ms=2, label=@sprintf("pfac=%.1e", pfacs[i]))
end
fig = plot(p1, p2; layout=(2, 1), size=(1000, 1050))
savefig(fig, "/tmp/gal_tok_scan.png")
println("Saved /tmp/gal_tok_scan.png")
println("surfaces present (q): ", sort(unique(round.(q; digits=1))))
println("DONE")
