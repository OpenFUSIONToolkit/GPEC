#!/usr/bin/env julia
# plot_deltamn_panelA.jl -- single-panel |Delta_mn| vs resistivity eta per rational surface, from an
# existing deltamn_merge CSV (dmn_q* columns). The full merged resonant response only (panel (a) of the
# two-panel merge figure); saturation at low eta is the robustness signature.
#
# Usage: julia --project=<GPEC> scripts/plot_deltamn_panelA.jl <deltamn_merge_*.csv> [outname]

using DelimitedFiles, Plots

length(ARGS) >= 1 || error("usage: julia ... scripts/plot_deltamn_panelA.jl <csv> [outname]")
csv = ARGS[1]; isfile(csv) || error("no such file: $csv")

raw, hdr = (d = readdlm(csv, ','; header=true); (d[1], vec(d[2])))
col(name) = findfirst(==(name), hdr)
eta = Float64.(raw[:, col("eta")])
qs = sort(unique(parse.(Float64, [split(h, "_q")[2] for h in hdr if startswith(h, "dmn_q")])))

gr()
p = plot(; xscale=:log10, yscale=:log10, xflip=true,
    xlabel="resistivity  eta   (low eta = high |Q|, to the right)",
    ylabel="resonant metric  |Delta_mn|", title="Resonant metric |Delta_mn| vs resistivity (DIII-D, preliminary)",
    legend=:right, left_margin=10Plots.mm, bottom_margin=5Plots.mm, size=(760, 500))
scols = palette(:tab10)
for (i, q) in enumerate(qs)
    y = Float64.(raw[:, col("dmn_q$(q)")])
    plot!(p, eta, y; color=scols[mod1(i, length(scols))], lw=2, marker=:circle, ms=4, label="q = $q")
end

outname = length(ARGS) >= 2 ? ARGS[2] : "deltamn_panelA_diiid"
figdir = joinpath(@__DIR__, "..", "figures"); mkpath(figdir)
figpath = joinpath(figdir, "$(outname).png")
savefig(p, figpath)
println("wrote: $(abspath(figpath))")
