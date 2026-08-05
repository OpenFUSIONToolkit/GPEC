#!/usr/bin/env julia
# deltacoil_dmlim_riccati_galerkin.jl -- domain-truncation scan on dmlim (integration domain truncated at
# (last_rational_q + dmlim)/n, via set_psilim_via_dmlim=true). Records BOTH the Riccati/STRIDE delta_coil
# (singular/delta_coil_matrix) and the Galerkin/RDCON delta_coil (galerkin/delta_coil) at every dmlim, LEFT
# and RIGHT small solution separately, so both methods and both sides can be compared on one plot.
# Companion to deltacoil_psihigh_riccati_galerkin.jl (same read/columns), a second edge-truncation view.
# STRIDE-only self-convergence family (gal_match_flag=false, driven, delta_coil is eta-independent).
#
# Usage: julia --project=<GPEC> scripts/deltacoil_dmlim_riccati_galerkin.jl <example_dir> [dmlim_list_csv]
#        (dmlim_list_csv defaults to "0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5"; set QMAX to keep q<=QMAX)

using GeneralizedPerturbedEquilibrium, HDF5, LinearAlgebra, Printf, Plots

length(ARGS) >= 1 || error("usage: julia --project=<GPEC> scripts/deltacoil_dmlim_riccati_galerkin.jl <example_dir> [dmlim_list_csv]")
base = ARGS[1]; isdir(base) || error("no such dir: $base")
DMLIMS = length(ARGS) >= 2 ? parse.(Float64, split(ARGS[2], ",")) : [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
ENV["DELTACOIL_MODE"] = "driven"; delete!(ENV, "DELTACOIL_PROJECT")

base_toml = read(joinpath(base, "gpec.toml"), String)
scratch = mktempdir()
eqm = match(r"eq_filename\s*=\s*\"([^\"]+)\"", base_toml)
eqfile = eqm === nothing ? nothing : eqm.captures[1]
eqsrc  = eqfile === nothing ? nothing : (isabspath(eqfile) ? eqfile : joinpath(base, eqfile))

# per-surface delta_coil block, keyed by q, for one method's HDF5 dataset (2 rows: 2s-1 = L, 2s = R)
function blocks(fid, path, q)
    haskey(fid, path) || return nothing
    dc = read(fid, path)
    dc = size(dc, 1) < size(dc, 2) ? dc : permutedims(dc)       # -> (2msing, ncoil)
    out = Dict{Float64,Matrix{ComplexF64}}()
    for s in 1:length(q); out[round(q[s]; digits=2)] = Matrix{ComplexF64}(dc[2s-1:2s, :]); end
    out
end

function run_one(dmlim)
    tag = "dmlim_$(dmlim)"
    dir = joinpath(scratch, tag); mkpath(dir)
    if eqfile !== nothing
        dst = joinpath(dir, basename(eqfile)); islink(dst) || symlink(realpath(eqsrc), dst)
    end
    toml = base_toml
    for key in ("force_termination", "HDF5_filename", "set_psilim_via_dmlim", "dmlim", "gal_match_flag",
        "truncate_at_dW_peak", "thmax0")
        toml = replace(toml, Regex("(?m)^[ \\t]*$key[ \\t]*=.*\\n") => "")
    end
    toml = replace(toml, "[ForceFreeStates]" =>
        "[ForceFreeStates]\nforce_termination = true\nHDF5_filename = \"$tag.h5\"\nset_psilim_via_dmlim = true\n" *
        "dmlim = $dmlim\ntruncate_at_dW_peak = false\ngal_match_flag = false"; count=1)
    eqfile !== nothing && (toml = replace(toml, r"eq_filename\s*=\s*\"[^\"]+\"" => "eq_filename = \"$(basename(eqfile))\""))
    write(joinpath(dir, "gpec.toml"), toml)
    @info ">>> dmlim = $dmlim"
    try
        GeneralizedPerturbedEquilibrium.main([dir])
    catch err
        @warn "run failed at dmlim=$dmlim" exception = (err, catch_backtrace())
        return nothing
    end
    h5path = joinpath(dir, "$tag.h5")
    isfile(h5path) || return nothing
    h5open(h5path, "r") do fid
        haskey(fid, "singular/q") || return nothing
        q = vec(read(fid, "singular/q")); isempty(q) && return nothing
        ric = blocks(fid, "singular/delta_coil_matrix", q)
        gal = blocks(fid, "galerkin/delta_coil", q)
        (q=sort(round.(q; digits=2)), ric=ric, gal=gal)
    end
end

results = [(dm, run_one(dm)) for dm in DMLIMS]
allq = sort(collect(reduce(union, (Set(r[2].q) for r in results if r[2] !== nothing); init=Set{Float64}())))

QMAX = parse(Float64, get(ENV, "QMAX", "Inf"))
allq = filter(q -> q <= QMAX, allq)
suffix = isfinite(QMAX) ? "_core" : ""

getblk(r, meth, q) = (d = getfield(r, meth); (d === nothing || !haskey(d, q)) ? nothing : d[q])
nrmL(r, meth, q) = (b = getblk(r, meth, q); b === nothing ? NaN : norm(@view b[1, :]))
nrmR(r, meth, q) = (b = getblk(r, meth, q); b === nothing ? NaN : norm(@view b[2, :]))

outdir = joinpath(@__DIR__, "..", "results"); mkpath(outdir)
tag = replace(basename(base), r"[^A-Za-z0-9]" => "")
csv = joinpath(outdir, "deltacoil_dmlim_rg_$(tag)$(suffix).csv")
open(csv, "w") do io
    cols = String["dmlim"]
    for q in allq, meth in ("ric", "gal"), side in ("L", "R"); push!(cols, "norm$(side)_$(meth)_q$(q)"); end
    println(io, join(cols, ","))
    for (dm, r) in results
        vals = String[string(dm)]
        for q in allq
            if r === nothing
                append!(vals, ["", "", "", ""])
            else
                push!(vals, isnan(nrmL(r, :ric, q)) ? "" : string(nrmL(r, :ric, q)))
                push!(vals, isnan(nrmR(r, :ric, q)) ? "" : string(nrmR(r, :ric, q)))
                push!(vals, isnan(nrmL(r, :gal, q)) ? "" : string(nrmL(r, :gal, q)))
                push!(vals, isnan(nrmR(r, :gal, q)) ? "" : string(nrmR(r, :gal, q)))
            end
        end
        println(io, join(vals, ","))
    end
end

nfail = count(r -> r[2] === nothing, results)
println("\n" * "="^80)
println("  delta_coil dmlim truncation ($(basename(base))): Riccati vs Galerkin, L/R, dmlim in $(DMLIMS)")
println("  points that produced data: $(length(DMLIMS) - nfail)/$(length(DMLIMS))")
println("="^80)
@printf("  %-8s", "dmlim")
for q in allq; @printf(" | q%-4.1f ricL | q%-4.1f galL", q, q); end; println()
for (dm, r) in results
    @printf("  %-8.3f", dm)
    for q in allq
        @printf(" | %8s | %8s",
            r === nothing || isnan(nrmL(r, :ric, q)) ? "-" : @sprintf("%.2e", nrmL(r, :ric, q)),
            r === nothing || isnan(nrmL(r, :gal, q)) ? "-" : @sprintf("%.2e", nrmL(r, :gal, q)))
    end
    println()
end
println("\nwrote: $(abspath(csv))")

# ---- per-surface panels: |delta_coil| vs dmlim, Riccati vs Galerkin, LEFT and RIGHT ----
dms = [dm for (dm, r) in results]
panels = Plots.Plot[]
for q in allq
    ricL = [r === nothing ? NaN : nrmL(r, :ric, q) for (dm, r) in results]
    ricR = [r === nothing ? NaN : nrmR(r, :ric, q) for (dm, r) in results]
    galL = [r === nothing ? NaN : nrmL(r, :gal, q) for (dm, r) in results]
    galR = [r === nothing ? NaN : nrmR(r, :gal, q) for (dm, r) in results]
    p = plot(dms, ricL; label="Riccati L", color=:green, ls=:dash, lw=2, marker=:circle, ms=3,
             yscale=:log10, title="q = $q", xlabel="dmlim", ylabel="|delta_coil|", legend=:best)
    plot!(p, dms, ricR; label="Riccati R", color=:blue, lw=2, marker=:circle, ms=3)
    plot!(p, dms, galL; label="Galerkin L", color=:purple, ls=:dash, lw=2, marker=:square, ms=3)
    plot!(p, dms, galR; label="Galerkin R", color=:red, lw=2, marker=:square, ms=3)
    push!(panels, p)
end
ncol = min(length(panels), 4)
nrow = cld(length(panels), ncol)
figdir = joinpath(@__DIR__, "..", "figures"); mkpath(figdir)
fig = plot(panels...; layout=(nrow, ncol), size=(460 * ncol, 400 * nrow),
           left_margin=10Plots.mm, bottom_margin=6Plots.mm)
figpath = joinpath(figdir, "deltacoil_dmlim_rg_$(tag)$(suffix)_panels_norm_LR.png")
savefig(fig, figpath)
println("wrote: $(abspath(figpath))")
