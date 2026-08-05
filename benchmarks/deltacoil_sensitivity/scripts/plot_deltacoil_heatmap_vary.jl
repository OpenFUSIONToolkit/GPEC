#!/usr/bin/env julia
# plot_deltacoil_heatmap_vary.jl -- relabeled heatmap of the coil-driven small-solution coefficient
# Δ_RMP (delta_coil) from the DIII-D showcase scan (vary.h5 base group, STRIDE singular/delta_coil_matrix).
# Rows = per-surface small solutions (2L,2R,...,5L,5R); x = externally forced poloidal mode m;
# color = |Δ_RMP|. Cyan dotted lines mark the resonant m = n·q of each surface.
#
# Usage: julia --project=<GPEC> scripts/plot_deltacoil_heatmap_vary.jl [vary.h5] [outname]

using HDF5, LinearAlgebra, Plots

h5 = length(ARGS) >= 1 ? ARGS[1] : "/Users/viaweber/Desktop/deltacoil_default_showcase/vary.h5"
isfile(h5) || error("no such file: $h5")

dc, q, m = h5open(h5, "r") do f
    b = f["base"]
    (read(b, "dc_re") .+ im .* read(b, "dc_im"), vec(read(b, "q")), vec(read(b, "m")))
end
dc = size(dc, 1) < size(dc, 2) ? dc : permutedims(dc)   # -> (2msing, m)
msing = length(q)
ord = sortperm(q)                                       # surfaces inner -> outer
rows = vcat([[2s - 1, 2s] for s in ord]...)             # per surface: LEFT then RIGHT
A = abs.(dc[rows, :])
ylabels = String[]
for s in ord
    qs = Int(round(q[s])); push!(ylabels, "$(qs)L"); push!(ylabels, "$(qs)R")
end

gr()
p = heatmap(collect(m), 1:2msing, A;
    color=:magma, xlabel="m,  externally forced poloidal mode number",
    ylabel="small-solution coefficient  Δ_RMP", colorbar_title="|Δ_RMP|",
    yticks=(1:2msing, ylabels), title="Coil-driven small solution Δ_RMP (delta_coil), STRIDE, DIII-D",
    left_margin=10Plots.mm, bottom_margin=6Plots.mm, right_margin=6Plots.mm, size=(1000, 430))
for s in ord
    vline!(p, [q[s]]; color=:cyan, ls=:dot, lw=0.8, alpha=0.6, label="")   # resonant m = n·q (n=1)
end

outname = length(ARGS) >= 2 ? ARGS[2] : "deltacoil_heatmap_diiid_relabeled"
figdir = joinpath(@__DIR__, "..", "figures"); mkpath(figdir)
figpath = joinpath(figdir, "$(outname).png")
savefig(p, figpath)
println("wrote: $(abspath(figpath))")
