# Plot the (pc x pfac) gal-vs-STRIDE q=3 tracking scan: heatmap + per-pfac curves.
using Pkg; Pkg.activate("/Users/pharr/Projects/GPEC_dev/docs/GPEC_julia")
using Plots, DelimitedFiles, Printf; gr()

data, hdr = readdlm("/tmp/gal_pfac_scan2d.csv", ','; header=true)
col(name) = data[:, findfirst(==(name), strip.(vec(hdr)))]
pc = col("pc"); pfac = col("pfac"); gq3 = col("gal_q3"); sq3 = col("stride_q3")
relerr = 100 .* abs.(gq3 .- sq3) ./ abs.(sq3)

pcs = sort(unique(pc)); pfacs = sort(unique(pfac))
# Build relerr grid Z[i_pfac, j_pc]
Z = fill(NaN, length(pfacs), length(pcs))
for k in eachindex(pc)
    i = findfirst(≈(pfac[k]), pfacs); j = findfirst(≈(pc[k]), pcs)
    Z[i, j] = relerr[k]
end
Zc = clamp.(Z, 0.01, 100)  # for log color

# Panel 1: heatmap, log color
p1 = heatmap(pcs, pfacs, log10.(Zc);
    yscale=:log10, xlabel="pₐ", ylabel="packing pfac",
    title="gal q=3 tracking error vs STRIDE:  log₁₀ |gal−STRIDE|/STRIDE (%)",
    colorbar_title="log₁₀(% err)", c=:viridis,
    left_margin=12Plots.mm, bottom_margin=6Plots.mm, right_margin=8Plots.mm)

# Panel 2: per-pfac tracking curves
p2 = plot(; xlabel="pₐ", ylabel="|gal−STRIDE|/STRIDE (%)", yscale=:log10, legend=:topright,
    left_margin=12Plots.mm, bottom_margin=6Plots.mm, right_margin=4Plots.mm,
    title="q=3 tracking error per packing ratio")
for (i, pf) in enumerate(pfacs)
    plot!(p2, pcs, max.(Z[i, :], 1e-3); lw=2, marker=:circle, ms=3, label=@sprintf("pfac=%.0e", pf))
end

fig = plot(p1, p2; layout=(2, 1), size=(900, 1000))
savefig(fig, "/tmp/gal_pfac_scan2d.png")
println("Saved /tmp/gal_pfac_scan2d.png")

# Per-pfac median/max summary
println("\npfac      median%   max%   (q=3 over the pc grid)")
med(v)=(w=sort(filter(isfinite,v));isempty(w) ? NaN : (n=length(w); iseven(n) ? (w[n÷2]+w[n÷2+1])/2 : w[(n+1)÷2]))
for (i, pf) in enumerate(pfacs)
    r = Z[i, :]
    @printf("%.0e   %7.2f  %6.1f\n", pf, med(r), maximum(filter(isfinite, r)))
end
println("DONE")
