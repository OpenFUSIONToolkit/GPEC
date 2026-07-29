# Plot the fine (50 pc x 30 pfac) gal-vs-STRIDE q=3 tracking scan: heatmap + per-pfac curves.
using Pkg; Pkg.activate(normpath(joinpath(@__DIR__, "..", "..", "..")))
using Plots, DelimitedFiles, Printf; gr()

data, hdr = readdlm(joinpath(@__DIR__, "gal_pfac_scan2d_big.csv"), ','; header=true)
col(name) = data[:, findfirst(==(name), strip.(vec(hdr)))]
pc = col("pc"); pfac = col("pfac"); gq3 = col("gal_q3"); sq3 = col("stride_q3")
relerr = 100 .* abs.(gq3 .- sq3) ./ max.(abs.(sq3), 1e-12)

pcs = sort(unique(round.(pc; digits=6))); pfacs = sort(unique(pfac))
Z = fill(NaN, length(pfacs), length(pcs))
for k in eachindex(pc)
    i = findfirst(≈(pfac[k]), pfacs); j = findfirst(≈(round(pc[k]; digits=6)), pcs)
    (i === nothing || j === nothing) && continue
    Z[i, j] = relerr[k]
end
Zc = clamp.(Z, 0.01, 100)

p1 = heatmap(pcs, pfacs, log10.(Zc); yscale=:log10, xlabel="pₐ", ylabel="packing pfac",
    title="gal q=3 tracking vs STRIDE: log₁₀ |gal−STRIDE|/STRIDE (%)   [50×30]",
    colorbar_title="log₁₀(% err)", c=:viridis,
    left_margin=12Plots.mm, bottom_margin=6Plots.mm, right_margin=10Plots.mm)

# per-pfac error vs pc, subset of pfacs for legibility
p2 = plot(; xlabel="pₐ", ylabel="|gal−STRIDE|/STRIDE (%)", yscale=:log10, legend=:topright,
    left_margin=12Plots.mm, bottom_margin=6Plots.mm, right_margin=4Plots.mm,
    title="q=3 tracking error per packing ratio")
sel = [findmin(abs.(log10.(pfacs) .- t))[2] for t in (-4, -3, -2, -1.5, -1, 0)]
for i in unique(sel)
    plot!(p2, pcs, [isfinite(z) ? max(z, 1e-3) : NaN for z in Z[i, :]]; lw=2, label=@sprintf("pfac=%.1e", pfacs[i]))
end
fig = plot(p1, p2; layout=(2, 1), size=(1000, 1050))
savefig(fig, joinpath(@__DIR__, "gal_pfac_scan2d_big.png"))
println("Saved gal_pfac_scan2d_big.png to ", @__DIR__)

med(v) = (w = sort(filter(isfinite, v)); isempty(w) ? NaN : (n = length(w); iseven(n) ? (w[n÷2]+w[n÷2+1])/2 : w[(n+1)÷2]))
println("\npfac        median%   max%")
for (i, pf) in enumerate(pfacs)
    @printf("%.2e    %7.3f  %7.1f\n", pf, med(Z[i, :]), maximum(filter(isfinite, Z[i, :]); init=NaN))
end
println("DONE")
