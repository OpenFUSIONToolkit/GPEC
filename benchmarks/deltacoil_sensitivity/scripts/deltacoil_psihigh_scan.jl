#!/usr/bin/env julia
# deltacoil_psihigh_scan.jl — sensitivity of delta_coil to the outer truncation, scanned via the
# ABSOLUTE flux boundary psihigh (instead of the relative dmlim).
#
# Setup per the requested convention:
#   set_psilim_via_dmlim = false   → psilim follows psihigh (dmlim truncation OFF)
#   psiedge              = 1        → disable the edge-dW scan band
#   truncate_at_dW_peak  = false    → psihigh is the SOLE truncation control
#   gal_match_flag       = false    → delta_coil only (no inner match; no msing/array guard; surface
#                                     count free to change as psihigh moves past each rational ψ)
#
# psihigh is swept from PSIH_MIN(0.1) to PSIH_MAX(~edge) with LOGARITHMIC packing toward the edge
# (points cluster near ψ→1, where truncation bites). Results aligned BY q-VALUE, so surfaces
# appearing/disappearing across the sweep are handled cleanly.
#
# Usage (repo root, project env):
#   julia --project=/abs/path/to/GPEC scripts/deltacoil_psihigh_scan.jl <example_dir> [N] [psihigh_max]
#
# Runs GPEC once per point (~2 min each). Each point re-solves the equilibrium out to psihigh.

using GeneralizedPerturbedEquilibrium, HDF5, LinearAlgebra, Printf

length(ARGS) >= 1 || error("usage: julia --project=<GPEC> scripts/deltacoil_psihigh_scan.jl <example_dir> [N] [psihigh_max]")
base = ARGS[1]; isdir(base) || error("no such dir: $base")
N        = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 10
PSIH_MAX = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 0.993   # numerically-safe edge (separatrix ~1 is unreasonable)
PSIH_MIN = 0.1
ENV["DELTACOIL_MODE"] = "driven"; delete!(ENV, "DELTACOIL_PROJECT")

# logarithmic packing toward the edge: dense near ψ→1, first point = PSIH_MIN, last = PSIH_MAX
PSIHIGHS = round.(1 .- 10 .^ range(log10(1 - PSIH_MIN), log10(1 - PSIH_MAX); length=N); digits=5)

base_toml = read(joinpath(base, "gpec.toml"), String)
scratch = mktempdir()
eqm = match(r"eq_filename\s*=\s*\"([^\"]+)\"", base_toml)
eqfile = eqm === nothing ? nothing : eqm.captures[1]
eqsrc  = eqfile === nothing ? nothing : (isabspath(eqfile) ? eqfile : joinpath(base, eqfile))

# run GPEC at one psihigh; return (sorted q-values, |delta_coil| per surface keyed by q). nothing on failure.
function run_one(ph)
    tag = "ph_$(ph)"
    dir = joinpath(scratch, tag); mkpath(dir)
    if eqfile !== nothing
        dst = joinpath(dir, basename(eqfile)); islink(dst) || symlink(realpath(eqsrc), dst)
    end
    toml = base_toml
    # Strip every FFS knob we override (examples differ: some set force_termination/HDF5_filename/psiedge,
    # some don't — stripping first avoids TOML duplicate-key errors), then inject one canonical block after
    # the section header. delta_coil is computed inside ForceFreeStates, so force_termination=true stops
    # before PerturbedEquilibrium and the explicit HDF5_filename is the file we read back.
    # also drop keys removed from ForceFreeStatesControl on this branch (thmax0, delta_mband) — some
    # example tomls (e.g. LAR) still carry them and would otherwise fail the control constructor.
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
    isfile(h5path) || (@warn "no output h5 at psihigh=$ph (likely no singular surfaces inside psihigh)"; return (Float64[], Dict{Float64,Float64}()))
    h5open(h5path, "r") do fid
        haskey(fid, "singular/q") || return (Float64[], Dict{Float64,Float64}())
        q = vec(read(fid, "singular/q"))
        isempty(q) && return (Float64[], Dict{Float64,Float64}())
        dc = read(fid, "singular/delta_coil_matrix")
        dc = size(dc, 1) < size(dc, 2) ? dc : permutedims(dc)       # -> (2msing, ncoil)
        mag = Dict{Float64,Float64}()
        for s in 1:length(q); mag[round(q[s]; digits=2)] = norm(dc[2s-1:2s, :]); end
        (sort(round.(q; digits=2)), mag)
    end
end

runs = [(ph, run_one(ph)) for ph in PSIHIGHS]

# ---- report ----
allq = sort(collect(reduce(union, (Set(keys(r[2][2])) for r in runs if r[2] !== nothing); init=Set{Float64}())))
println("\n" * "="^78)
println("  delta_coil vs OUTER TRUNCATION scanned by psihigh  (log-packed toward edge)")
println("  set_psilim_via_dmlim=false, psiedge=1, truncate_at_dW_peak=false, gal_match_flag=false")
println("  |delta_coil| per surface (blank = surface not present at that psihigh)")
println("="^78)
@printf("  %-9s", "psihigh")
for q in allq; @printf(" | q=%-6.1f", q); end; println()
println("  " * "-"^(11 + 11 * length(allq)))
for (ph, r) in runs
    @printf("  %-9.4f", ph)
    if r === nothing
        for _ in allq; @printf(" | %8s", "FAIL"); end
    else
        for q in allq
            m = get(r[2], q, nothing)
            m === nothing ? @printf(" | %8s", "—") : @printf(" | %8.2e", m)
        end
    end
    println()
end

# CSV for plotting
outdir = joinpath(@__DIR__, "..", "results"); mkpath(outdir)
csv = joinpath(outdir, "deltacoil_psihigh_result.csv")
open(csv, "w") do io
    println(io, "psihigh," * join(["q$(q)" for q in allq], ","))
    for (ph, r) in runs
        vals = [r === nothing ? "" : (haskey(r[2], q) ? string(r[2][q]) : "") for q in allq]
        println(io, "$ph," * join(vals, ","))
    end
end
println("\nwrote: $(abspath(csv))")
println("interpretation: as psihigh → edge, more surfaces enter and each surface's |delta_coil|")
println("settles; the edge-most surface is where the truncation boundary bites hardest.")
