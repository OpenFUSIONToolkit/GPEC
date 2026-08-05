#!/usr/bin/env julia
# replot_resistivity_bpen.jl -- regenerate the per-surface ||bpen|| vs eta figure from an existing
# resistivity_scan CSV (no GPEC rerun). STRIDE (ric_bpen, solid) vs Galerkin (gal_bpen, dashed); legend
# placed inside the plot (top-right empty space).
#
# Usage: julia --project=<GPEC> scripts/replot_resistivity_bpen.jl <resistivity_scan_*.csv> [outname]

using DelimitedFiles, Plots

length(ARGS) >= 1 || error("usage: julia ... scripts/replot_resistivity_bpen.jl <csv> [outname]")
csv = ARGS[1]; isfile(csv) || error("no such file: $csv")

data = readdlm(csv, ','; header=true)
rows, hdr = data[1], vec(data[2])
col(name) = (i = findfirst(==(name), hdr); i === nothing ? nothing : i)
eta = Float64.(rows[:, col("eta")])
qs = sort(unique(parse.(Float64, [split(h, "_q")[2] for h in hdr if startswith(h, "ric_bpen_q")])))

gr()
p = plot(; xscale=:log10, yscale=:log10, xlabel="resistivity  eta   (decreasing to the right = more strongly driven inner layer)",
    ylabel="penetrated resonant field  ||bpen||", title="Matched penetrated field vs resistivity eta, n=1",
    legend=:topright, left_margin=12Plots.mm, bottom_margin=4Plots.mm, right_margin=4Plots.mm, size=(880, 560))
scols = palette(:tab10)
getcol(name) = (c = col(name); c === nothing ? nothing : [v isa Number ? Float64(v) : NaN for v in rows[:, c]])
for (i, q) in enumerate(qs)
    c = scols[mod1(i, length(scols))]
    ys = getcol("ric_bpen_q$(q)"); yg = getcol("gal_bpen_q$(q)")
    ys === nothing || plot!(p, eta, ys; color=c, lw=2, marker=:circle, ms=4, label="q = $q  STRIDE")
    yg === nothing || plot!(p, eta, yg; color=c, lw=2, ls=:dash, marker=:diamond, ms=4, label="q = $q  Galerkin")
end
xflip!(p)

outname = length(ARGS) >= 2 ? ARGS[2] : replace(basename(csv), "resistivity_scan_" => "resistivity_scan_bpen_", ".csv" => "")
figdir = joinpath(@__DIR__, "..", "figures"); mkpath(figdir)
figpath = joinpath(figdir, "$(outname).png")
savefig(p, figpath)
println("wrote: $(abspath(figpath))")
