#!/usr/bin/env julia
# plot_deltacoil_heatmap.jl -- heatmap of the coil-driven small-solution coefficients Δ_RMP (delta_coil).
# Rows = the 2·msing small solutions (per rational surface, LEFT/RIGHT), columns = externally forced
# poloidal mode number m. Value = |Δ_RMP|, the amplitude of the small (reconnecting) solution excited at
# each rational surface by a unit coil perturbation at the edge (Wang 2020 Fig. 1b picture).
#
# Usage: julia --project=<GPEC> scripts/plot_deltacoil_heatmap.jl <gpec.h5> [outname]

using GeneralizedPerturbedEquilibrium, HDF5, LinearAlgebra, Plots

length(ARGS) >= 1 || error("usage: julia --project=<GPEC> scripts/plot_deltacoil_heatmap.jl <gpec.h5> [outname]")
h5 = ARGS[1]; isfile(h5) || error("no such file: $h5")

dc, q, mlow = h5open(h5, "r") do fid
    haskey(fid, "singular/delta_coil_matrix") || error("$h5 has no singular/delta_coil_matrix")
    d = read(fid, "singular/delta_coil_matrix")
    d = size(d, 1) < size(d, 2) ? d : permutedims(d)   # -> (2msing, m)
    (d, vec(read(fid, "singular/q")), Int(read(fid, "info/mlow")))
end

msing = length(q)
size(dc, 1) == 2msing || error("delta_coil rows $(size(dc,1)) != 2msing $(2msing)")
nm = size(dc, 2)
mvals = mlow:(mlow + nm - 1)

# y labels: per surface, LEFT then RIGHT (Riccati loop_edge convention: row 2s-1 = L, 2s = R)
ylabels = String[]
for s in 1:msing
    qs = Int(round(q[s]))
    push!(ylabels, "$(qs)L"); push!(ylabels, "$(qs)R")
end

Z = abs.(dc)   # |Δ_RMP|
gr()
p = heatmap(collect(mvals), 1:2msing, Z;
    color=:viridis, xlabel="m,  externally forced poloidal mode number",
    ylabel="small-solution coefficient  Δ_RMP", colorbar_title="|Δ_RMP|",
    yticks=(1:2msing, ylabels), yflip=true, title="Coil-driven small solution Δ_RMP (delta_coil)",
    left_margin=10Plots.mm, bottom_margin=6Plots.mm, right_margin=6Plots.mm, size=(920, 360))

outname = length(ARGS) >= 2 ? ARGS[2] : "deltacoil_heatmap"
figdir = joinpath(@__DIR__, "..", "figures"); mkpath(figdir)
figpath = joinpath(figdir, "$(outname).png")
savefig(p, figpath)
println("wrote: $(abspath(figpath))")
