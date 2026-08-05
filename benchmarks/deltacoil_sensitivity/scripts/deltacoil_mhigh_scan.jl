#!/usr/bin/env julia
# deltacoil_mhigh_scan.jl -- Fourier-truncation self-convergence of the STRIDE delta_coil. Holds psihigh
# at the example default (full domain) and varies only delta_mhigh (the upper poloidal-mode band bound),
# so it measures whether adding more upper harmonics moves the coil response. Per surface it records
#   ||delta_coil block||  (singular/delta_coil_matrix), LEFT and RIGHT small solution separately.
# The plotter normalizes each surface to its largest-delta_mhigh value, giving a convergence metric -> 1.
# STRIDE-only (gal_match_flag=false, driven, delta_coil is eta-independent); companion to the psihigh scan.
#
# Usage: julia --project=<GPEC> scripts/deltacoil_mhigh_scan.jl <example_dir> [mhigh_list_csv]
#        (mhigh_list_csv defaults to "4,6,8,10,12,16,20,25,30")

using GeneralizedPerturbedEquilibrium, HDF5, LinearAlgebra, Printf

length(ARGS) >= 1 || error("usage: julia --project=<GPEC> scripts/deltacoil_mhigh_scan.jl <example_dir> [mhigh_list_csv]")
base = ARGS[1]; isdir(base) || error("no such dir: $base")
MHIGHS = length(ARGS) >= 2 ? parse.(Int, split(ARGS[2], ",")) : [4, 6, 8, 10, 12, 16, 20, 25, 30]
ENV["DELTACOIL_MODE"] = "driven"; delete!(ENV, "DELTACOIL_PROJECT")

base_toml = read(joinpath(base, "gpec.toml"), String)
scratch = mktempdir()
eqm = match(r"eq_filename\s*=\s*\"([^\"]+)\"", base_toml)
eqfile = eqm === nothing ? nothing : eqm.captures[1]
eqsrc  = eqfile === nothing ? nothing : (isabspath(eqfile) ? eqfile : joinpath(base, eqfile))

# per-surface delta_coil block, keyed by q (2 rows per surface: 2s-1 = LEFT, 2s = RIGHT small solution)
function blocks(fid, path, q)
    haskey(fid, path) || return nothing
    dc = read(fid, path)
    dc = size(dc, 1) < size(dc, 2) ? dc : permutedims(dc)       # -> (2msing, ncoil)
    out = Dict{Float64,Matrix{ComplexF64}}()
    for s in 1:length(q); out[round(q[s]; digits=2)] = Matrix{ComplexF64}(dc[2s-1:2s, :]); end
    out
end

function run_one(mhigh)
    tag = "mhigh_$(mhigh)"
    dir = joinpath(scratch, tag); mkpath(dir)
    if eqfile !== nothing
        dst = joinpath(dir, basename(eqfile)); islink(dst) || symlink(realpath(eqsrc), dst)
    end
    toml = base_toml
    for key in ("force_termination", "HDF5_filename", "gal_match_flag", "delta_mhigh", "delta_mband", "thmax0")
        toml = replace(toml, Regex("(?m)^[ \\t]*$key[ \\t]*=.*\\n") => "")
    end
    toml = replace(toml, "[ForceFreeStates]" =>
        "[ForceFreeStates]\nforce_termination = true\nHDF5_filename = \"$tag.h5\"\ngal_match_flag = false\ndelta_mhigh = $mhigh\n"; count=1)
    eqfile !== nothing && (toml = replace(toml, r"eq_filename\s*=\s*\"[^\"]+\"" => "eq_filename = \"$(basename(eqfile))\""))
    write(joinpath(dir, "gpec.toml"), toml)
    @info ">>> delta_mhigh = $mhigh"
    try
        GeneralizedPerturbedEquilibrium.main([dir])
    catch err
        @warn "run failed at delta_mhigh=$mhigh" exception = (err, catch_backtrace())
        return nothing
    end
    h5path = joinpath(dir, "$tag.h5")
    isfile(h5path) || return nothing
    h5open(h5path, "r") do fid
        haskey(fid, "singular/q") || return nothing
        q = vec(read(fid, "singular/q")); isempty(q) && return nothing
        (q=sort(round.(q; digits=2)), ric=blocks(fid, "singular/delta_coil_matrix", q))
    end
end

results = [(mh, run_one(mh)) for mh in MHIGHS]
allq = sort(collect(reduce(union, (Set(r[2].q) for r in results if r[2] !== nothing); init=Set{Float64}())))

getblk(r, q) = (r.ric === nothing || !haskey(r.ric, q)) ? nothing : r.ric[q]
nrmL(r, q) = (b = getblk(r, q); b === nothing ? NaN : norm(@view b[1, :]))
nrmR(r, q) = (b = getblk(r, q); b === nothing ? NaN : norm(@view b[2, :]))

outdir = joinpath(@__DIR__, "..", "results"); mkpath(outdir)
tag = replace(basename(base), r"[^A-Za-z0-9]" => "")
csv = joinpath(outdir, "deltacoil_mhigh_$(tag).csv")
open(csv, "w") do io
    cols = String["delta_mhigh"]
    for q in allq, side in ("L", "R"); push!(cols, "norm$(side)_ric_q$(q)"); end
    println(io, join(cols, ","))
    for (mh, r) in results
        vals = String[string(mh)]
        for q in allq
            if r === nothing
                append!(vals, ["", ""])
            else
                l = nrmL(r, q); rr = nrmR(r, q)
                push!(vals, isnan(l) ? "" : string(l)); push!(vals, isnan(rr) ? "" : string(rr))
            end
        end
        println(io, join(vals, ","))
    end
end

println("\n" * "="^72)
println("  delta_coil Fourier-truncation scan ($(basename(base))): STRIDE norm vs delta_mhigh")
println("="^72)
@printf("  %-12s", "delta_mhigh")
for q in allq; @printf(" | q%-4.1f L | q%-4.1f R", q, q); end; println()
for (mh, r) in results
    @printf("  %-12d", mh)
    for q in allq
        @printf(" | %7s | %7s",
            r === nothing || isnan(nrmL(r, q)) ? "-" : @sprintf("%.3e", nrmL(r, q)),
            r === nothing || isnan(nrmR(r, q)) ? "-" : @sprintf("%.3e", nrmR(r, q)))
    end
    println()
end
println("\nwrote: $(abspath(csv))")
