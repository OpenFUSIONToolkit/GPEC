#!/usr/bin/env julia
# deltacoil_psihigh_riccati_galerkin.jl -- unified psihigh truncation scan that records BOTH the
# Riccati/STRIDE delta_coil (singular/delta_coil_matrix) and the Galerkin/RDCON delta_coil
# (galerkin/delta_coil) at every psihigh, so both methods can be compared on one plot.
#
# One GPEC run per psihigh serves both the norm view and the shape (cosine) view, so this replaces
# running deltacoil_psihigh_scan.jl and deltacoil_psihigh_metric.jl separately. Per surface and per
# method it writes:
#   norm  = ||delta_coil block for that surface||
#   cos   = |<v_nominal, v>| / (||v_nominal|| ||v||), aligned by poloidal mode m, nominal = largest psihigh
# cos = 1 means the coil-response pattern is unchanged (only rescaled); cos < 1 means it rotated.
#
# Config matches the earlier psihigh scans: set_psilim_via_dmlim=false, psiedge=1, truncate_at_dW_peak=false,
# gal_match_flag=false (delta_coil only). psihigh is log-packed toward the edge.
#
# Usage: julia --project=<GPEC> scripts/deltacoil_psihigh_riccati_galerkin.jl <example_dir> [N] [psihigh_max] [psihigh_min]

using GeneralizedPerturbedEquilibrium, HDF5, LinearAlgebra, Printf

length(ARGS) >= 1 || error("usage: julia --project=<GPEC> scripts/deltacoil_psihigh_riccati_galerkin.jl <example_dir> [N] [psihigh_max] [psihigh_min]")
base = ARGS[1]; isdir(base) || error("no such dir: $base")
N        = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 100
PSIH_MAX = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 0.9999
PSIH_MIN = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 0.9
ENV["DELTACOIL_MODE"] = "driven"; delete!(ENV, "DELTACOIL_PROJECT")

# logarithmic packing toward the edge: dense near psi -> 1, first point = PSIH_MIN, last = PSIH_MAX
PSIHIGHS = round.(1 .- 10 .^ range(log10(1 - PSIH_MIN), log10(1 - PSIH_MAX); length=N); digits=6)

base_toml = read(joinpath(base, "gpec.toml"), String)
scratch = mktempdir()
eqm = match(r"eq_filename\s*=\s*\"([^\"]+)\"", base_toml)
eqfile = eqm === nothing ? nothing : eqm.captures[1]
eqsrc  = eqfile === nothing ? nothing : (isabspath(eqfile) ? eqfile : joinpath(base, eqfile))

# per-surface delta_coil block, keyed by q, for one method's HDF5 dataset
function blocks(fid, path, q)
    haskey(fid, path) || return nothing
    dc = read(fid, path)
    dc = size(dc, 1) < size(dc, 2) ? dc : permutedims(dc)       # -> (2msing, ncoil), columns m=mlow..mhigh
    out = Dict{Float64,Matrix{ComplexF64}}()
    for s in 1:length(q); out[round(q[s]; digits=2)] = Matrix{ComplexF64}(dc[2s-1:2s, :]); end
    out
end

# run GPEC at one psihigh; return (q-values, ric blocks, gal blocks, mlow), or nothing on failure.
function run_one(ph)
    tag = "ph_$(ph)"
    dir = joinpath(scratch, tag); mkpath(dir)
    if eqfile !== nothing
        dst = joinpath(dir, basename(eqfile)); islink(dst) || symlink(realpath(eqsrc), dst)
    end
    toml = base_toml
    for key in ("force_termination", "HDF5_filename", "psiedge", "set_psilim_via_dmlim", "gal_match_flag",
        "truncate_at_dW_peak", "thmax0", "delta_mband")
        toml = replace(toml, Regex("(?m)^[ \\t]*$key[ \\t]*=.*\\n") => "")
    end
    toml = replace(toml, "[ForceFreeStates]" =>
        "[ForceFreeStates]\nforce_termination = true\nHDF5_filename = \"$tag.h5\"\npsiedge = 1.0\n" *
        "set_psilim_via_dmlim = false\ntruncate_at_dW_peak = false\ngal_match_flag = false"; count=1)
    toml = replace(toml, r"(?m)^([ \t]*)psihigh[ \t]*=[ \t]*[\d.eE+-]+" => SubstitutionString("\\1psihigh = $ph"))
    eqfile !== nothing && (toml = replace(toml, r"eq_filename\s*=\s*\"[^\"]+\"" => "eq_filename = \"$(basename(eqfile))\""))
    write(joinpath(dir, "gpec.toml"), toml)
    @info ">>> psihigh = $ph"
    try
        GeneralizedPerturbedEquilibrium.main([dir])
    catch err
        @warn "run failed at psihigh=$ph" exception = (err, catch_backtrace())
        return nothing
    end
    h5path = joinpath(dir, "$tag.h5")
    isfile(h5path) || return nothing
    h5open(h5path, "r") do fid
        haskey(fid, "singular/q") || return nothing
        q = vec(read(fid, "singular/q")); isempty(q) && return nothing
        mlow = Int(read(fid, "info/mlow"))
        ric = blocks(fid, "singular/delta_coil_matrix", q)
        gal = blocks(fid, "galerkin/delta_coil", q)
        (q=sort(round.(q; digits=2)), ric=ric, gal=gal, mlow=mlow)
    end
end

# cosine between a run block and the nominal block, aligned by common poloidal modes m
function cosine_aligned(br, mlr, bn, mln)
    ncr = size(br, 2); ncn = size(bn, 2)
    mlo = max(mlr, mln); mhi = min(mlr + ncr - 1, mln + ncn - 1)
    mhi >= mlo || return NaN
    a = vec(br[:, (mlo - mlr + 1):(mhi - mlr + 1)])
    b = vec(bn[:, (mlo - mln + 1):(mhi - mln + 1)])
    (norm(a) > 0 && norm(b) > 0) ? abs(dot(b, a)) / (norm(a) * norm(b)) : NaN
end

results = [(ph, run_one(ph)) for ph in PSIHIGHS]

# nominal = the largest psihigh that produced data (full-domain reference for the cosine metric)
nom_idx = findlast(r -> r[2] !== nothing, results)
nom_idx === nothing && error("no successful runs")
nom = results[nom_idx][2]
@info "nominal (full-domain) psihigh = $(results[nom_idx][1]); surfaces present: $(nom.q)"

allq = sort(collect(reduce(union, (Set(r[2].q) for r in results if r[2] !== nothing); init=Set{Float64}())))

# helper: norm / cosine for a method (:ric or :gal) at a given surface for a run r
getblk(r, meth, q) = (d = getfield(r, meth); (d === nothing || !haskey(d, q)) ? nothing : d[q])
nrm(r, meth, q) = (b = getblk(r, meth, q); b === nothing ? NaN : norm(b))
function cosm(r, meth, q)
    b = getblk(r, meth, q); bn = getblk(nom, meth, q)
    (b === nothing || bn === nothing) ? NaN : cosine_aligned(b, r.mlow, bn, nom.mlow)
end

outdir = joinpath(@__DIR__, "..", "results"); mkpath(outdir)
tag = replace(basename(base), r"[^A-Za-z0-9]" => "")
csv = joinpath(outdir, "deltacoil_psihigh_rg_$tag.csv")
open(csv, "w") do io
    cols = String["psihigh"]
    for q in allq, meth in ("ric", "gal"), metric in ("norm", "cos"); push!(cols, "$(metric)_$(meth)_q$(q)"); end
    println(io, join(cols, ","))
    for (ph, r) in results
        vals = String[string(ph)]
        for q in allq
            if r === nothing
                append!(vals, ["", "", "", ""])
            else
                nr = nrm(r, :ric, q); cr = cosm(r, :ric, q)
                ng = nrm(r, :gal, q); cg = cosm(r, :gal, q)
                push!(vals, isnan(nr) ? "" : string(nr)); push!(vals, isnan(cr) ? "" : string(cr))
                push!(vals, isnan(ng) ? "" : string(ng)); push!(vals, isnan(cg) ? "" : string(cg))
            end
        end
        println(io, join(vals, ","))
    end
end

# console summary
nfail = count(r -> r[2] === nothing, results)
println("\n" * "="^80)
println("  delta_coil psihigh truncation ($(basename(base))): Riccati vs Galerkin, $N points, "
        * "psihigh $PSIH_MIN -> $PSIH_MAX")
println("  points that produced data: $(N - nfail)/$N   (failures recorded as blanks)")
println("="^80)
@printf("  %-10s", "psihigh")
for q in allq; @printf(" | q%-4.1f ric | q%-4.1f gal", q, q); end; println()
for (ph, r) in results
    @printf("  %-10.5f", ph)
    for q in allq
        @printf(" | %8s | %8s",
            r === nothing || isnan(nrm(r, :ric, q)) ? "-" : @sprintf("%.2e", nrm(r, :ric, q)),
            r === nothing || isnan(nrm(r, :gal, q)) ? "-" : @sprintf("%.2e", nrm(r, :gal, q)))
    end
    println()
end
println("\nwrote: $(abspath(csv))")
