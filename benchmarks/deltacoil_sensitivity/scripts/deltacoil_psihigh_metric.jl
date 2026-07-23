#!/usr/bin/env julia
# deltacoil_psihigh_metric.jl — the dot-product (cosine) shape metric of plot 2, but with the truncation
# scanned by psihigh (absolute outer flux boundary) instead of dmlim.
#
# For each rational surface s and each psihigh, we form v_s = vec(delta_coil rows for that surface) and
# report the cosine similarity with the NOMINAL (full-domain) run:
#     cos = |<v_s^nom, v_s>| / (‖v_s^nom‖ ‖v_s‖)
# 1.0 means the response pattern over coil modes is identical (only rescaled); < 1.0 means it ROTATED.
# This catches shape changes that the norm ‖v_s‖ hides.
#
# Wrinkle handled: as psihigh grows, surfaces enter, and the coil poloidal-mode grid (mlow..mhigh) changes,
# so the raw column count differs between runs. We therefore ALIGN by poloidal mode m: the cosine for a
# surface uses only the m-columns common to that run and the nominal. Nominal = the largest psihigh (full
# domain). Runs with gal_match_flag=false (delta_coil only). psihigh is log-packed toward the edge.
#
# Usage: julia --project=<GPEC> scripts/deltacoil_psihigh_metric.jl <example_dir> [N] [psihigh_max]

using GeneralizedPerturbedEquilibrium, HDF5, LinearAlgebra, Printf

length(ARGS) >= 1 || error("usage: julia --project=<GPEC> scripts/deltacoil_psihigh_metric.jl <example_dir> [N] [psihigh_max]")
base = ARGS[1]; isdir(base) || error("no such dir: $base")
N        = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 20
PSIH_MAX = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 0.993
PSIH_MIN = 0.1
ENV["DELTACOIL_MODE"] = "driven"; delete!(ENV, "DELTACOIL_PROJECT")

PSIHIGHS = round.(1 .- 10 .^ range(log10(1 - PSIH_MIN), log10(1 - PSIH_MAX); length=N); digits=5)

base_toml = read(joinpath(base, "gpec.toml"), String)
scratch = mktempdir()
eqm = match(r"eq_filename\s*=\s*\"([^\"]+)\"", base_toml)
eqfile = eqm === nothing ? nothing : eqm.captures[1]
eqsrc  = eqfile === nothing ? nothing : (isabspath(eqfile) ? eqfile : joinpath(base, eqfile))

# run GPEC at one psihigh; return Dict q => (block::Matrix (2×ncoil), mlow::Int), or nothing on failure.
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
    isfile(h5path) || return Dict{Float64,Tuple{Matrix{ComplexF64},Int}}()
    h5open(h5path, "r") do fid
        haskey(fid, "singular/q") || return Dict{Float64,Tuple{Matrix{ComplexF64},Int}}()
        q = vec(read(fid, "singular/q"))
        isempty(q) && return Dict{Float64,Tuple{Matrix{ComplexF64},Int}}()
        dc = read(fid, "singular/delta_coil_matrix")
        dc = size(dc, 1) < size(dc, 2) ? dc : permutedims(dc)      # -> (2msing, ncoil), columns m=mlow..mhigh
        mlow = Int(read(fid, "info/mlow"))
        out = Dict{Float64,Tuple{Matrix{ComplexF64},Int}}()
        for s in 1:length(q); out[round(q[s]; digits=2)] = (Matrix{ComplexF64}(dc[2s-1:2s, :]), mlow); end
        out
    end
end

# cosine of surface blocks aligned by poloidal mode m (use only the m-columns common to both)
function cosine_aligned(run, nom)
    (br, mlr) = run; (bn, mln) = nom
    ncr = size(br, 2); ncn = size(bn, 2)
    mhr = mlr + ncr - 1; mhn = mln + ncn - 1
    mlo = max(mlr, mln); mhi = min(mhr, mhn)
    mhi >= mlo || return NaN
    cr = (mlo - mlr + 1):(mhi - mlr + 1)
    cn = (mlo - mln + 1):(mhi - mln + 1)
    a = vec(br[:, cr]); b = vec(bn[:, cn])
    (norm(a) > 0 && norm(b) > 0) ? abs(dot(b, a)) / (norm(a) * norm(b)) : NaN
end

results = [(ph, run_one(ph)) for ph in PSIHIGHS]
# nominal = the largest psihigh with data (full domain)
nom_idx = findlast(r -> r[2] !== nothing && !isempty(r[2]), results)
nom_idx === nothing && error("no successful runs")
nominal = results[nom_idx][2]
@info "nominal (full-domain) psihigh = $(results[nom_idx][1]), surfaces present: $(sort(collect(keys(nominal))))"

allq = sort(collect(reduce(union, (Set(keys(r[2])) for r in results if r[2] !== nothing); init=Set{Float64}())))

println("\n" * "="^78)
println("  Plot-2 dot-product (cosine) metric vs psihigh truncation  (nominal = full domain)")
println("  cos = |<v_nom, v>| / (‖v_nom‖‖v‖) per surface, aligned by poloidal mode m")
println("="^78)
@printf("  %-9s", "psihigh")
for q in allq; @printf(" | q=%-6.1f", q); end; println()
println("  " * "-"^(11 + 11 * length(allq)))
for (ph, r) in results
    @printf("  %-9.4f", ph)
    for q in allq
        if r === nothing || !haskey(r, q) || !haskey(nominal, q)
            @printf(" | %8s", "—")
        else
            c = cosine_aligned(r[q], nominal[q])
            @printf(" | %8s", isnan(c) ? "n/a" : @sprintf("%.4f", c))
        end
    end
    println()
end

outdir = joinpath(@__DIR__, "..", "results"); mkpath(outdir)
csv = joinpath(outdir, "deltacoil_psihigh_metric.csv")
open(csv, "w") do io
    println(io, "psihigh," * join(["q$(q)" for q in allq], ","))
    for (ph, r) in results
        vals = String[]
        for q in allq
            if r === nothing || !haskey(r, q) || !haskey(nominal, q)
                push!(vals, "")
            else
                c = cosine_aligned(r[q], nominal[q])
                push!(vals, isnan(c) ? "" : string(c))
            end
        end
        println(io, "$ph," * join(vals, ","))
    end
end
println("\nwrote: $(abspath(csv))")
println("read: cos ≈ 1 ⇒ shape unchanged (only rescaled by truncation); cos < 1 ⇒ the coil-response")
println("pattern ROTATED as the truncation boundary moved. The norm alone would hide this.")
