#!/usr/bin/env julia
# deltacoil_sensitivity.jl — sensitivity of delta_coil to (a) the outer truncation point dmlim and
# (b) the rational-surface approach distance singfac_min.
#
# delta_coil (singular/delta_coil_matrix) is the outer small-solution response to unit edge sources.
# It is computed by the rpec edge loop INDEPENDENTLY of the resistive match, so this runs with
# gal_match_flag = false — no inner-layer match, hence no msing/array guard, and dmlim is free to
# change the surface count. Results are aligned BY q-VALUE (not row index), so a surface appearing or
# disappearing across the sweep is handled cleanly.
#
#   Sweep A: dmlim ∈ (0,1) with set_psilim_via_dmlim = true   (moves the outer truncation boundary)
#   Sweep B: singfac_min                                       (moves the rational-surface cutoff)
#
# Usage (repo root, project env):
#   julia --project=/abs/path/to/GPEC scripts/deltacoil_sensitivity.jl examples/DIIID-like_gal_resistive_pe_example
#
# Runs GPEC once per sweep point (~2 min each). Edit the value lists below to change the sampling.

using GeneralizedPerturbedEquilibrium, HDF5, LinearAlgebra, Printf

length(ARGS) >= 1 || error("usage: julia --project=<GPEC> scripts/deltacoil_sensitivity.jl <example_dir>")
base = ARGS[1]; isdir(base) || error("no such dir: $base")
ENV["DELTACOIL_MODE"] = "driven"; delete!(ENV, "DELTACOIL_PROJECT")

DMLIMS  = [0.1, 0.3, 0.5, 0.7, 0.9]        # open interval (0,1)
SINGFAC = [1e-5, 1e-4, 1e-3]               # rational-surface approach distance

base_toml = read(joinpath(base, "gpec.toml"), String)
scratch = mktempdir()
eqm = match(r"eq_filename\s*=\s*\"([^\"]+)\"", base_toml); eqm === nothing && error("no eq_filename")
eqfile = eqm.captures[1]; eqsrc = isabspath(eqfile) ? eqfile : joinpath(base, eqfile)

# run GPEC with the given toml patches; return (q-values, per-surface |delta_coil| dict keyed by q)
function run_one(tag, patches)
    dir = joinpath(scratch, tag); mkpath(dir)
    dst = joinpath(dir, basename(eqfile)); islink(dst) || symlink(realpath(eqsrc), dst)
    toml = base_toml
    for (r, s) in patches; toml = replace(toml, r => s); end
    toml = replace(toml, r"gal_match_flag\s*=\s*\w+" => "gal_match_flag = false")   # delta_coil only
    toml = replace(toml, r"eq_filename\s*=\s*\"[^\"]+\"" => "eq_filename = \"$(basename(eqfile))\"")
    # delta_coil is written by ForceFreeStates; terminate there (skip PE, which would otherwise send the
    # consolidated output to the default gpec.h5) and pin the FFS output file we read back. Strip any
    # pre-existing copies first to avoid TOML duplicate-key errors across example variants.
    for key in ("force_termination", "HDF5_filename")
        toml = replace(toml, Regex("(?m)^[ \\t]*$key[ \\t]*=.*\\n") => "")
    end
    toml = replace(toml, "[ForceFreeStates]" => "[ForceFreeStates]\nforce_termination = true\nHDF5_filename = \"$tag.h5\""; count=1)
    write(joinpath(dir, "gpec.toml"), toml)
    @info ">>> $tag"
    GeneralizedPerturbedEquilibrium.main([dir])
    h5open(joinpath(dir, "$tag.h5"), "r") do fid
        q  = vec(read(fid, "singular/q"))
        dc = read(fid, "singular/delta_coil_matrix")
        dc = size(dc, 1) < size(dc, 2) ? dc : permutedims(dc)   # -> (2msing, ncoil)
        mag = Dict{Float64,Float64}()
        for s in 1:length(q); mag[round(q[s]; digits=2)] = norm(dc[2s-1:2s, :]); end
        (sort(round.(q; digits=2)), mag)
    end
end

function report(title, param, vals, runs)
    qs = sort(collect(reduce(union, (Set(keys(r[2])) for r in runs))))
    println("\n" * "="^72); println("  $title"); println("  |delta_coil| per surface (blank = surface absent at that setting)"); println("="^72)
    @printf("  %-6s", "q\\$param")
    for v in vals; @printf(" | %9.3g", v); end; println()
    for q in qs
        @printf("  q=%-4.1f", q)
        for (i, v) in enumerate(vals)
            m = get(runs[i][2], q, nothing)
            m === nothing ? @printf(" | %9s", "—") : @printf(" | %9.3e", m)
        end
        println(q == maximum(qs) ? "   <- edge" : "")
    end
    # relative change vs the LAST column (reference)
    refm = runs[end][2]
    println("  " * "-"^66)
    println("  relative change of |delta_coil| vs $param=$(vals[end]) (reference):")
    for q in qs
        haskey(refm, q) || continue
        @printf("  q=%-4.1f", q)
        for (i, v) in enumerate(vals)
            m = get(runs[i][2], q, nothing)
            (m === nothing) ? @printf(" | %9s", "—") : @printf(" | %9.2e", abs(m - refm[q]) / max(refm[q], 1e-300))
        end
        println()
    end
end

# --- Sweep A: dmlim (truncation) ---
runsA = [run_one("dmlim_$(d)", [r"set_psilim_via_dmlim\s*=\s*\w+" => "set_psilim_via_dmlim = true",
                                r"(?m)^([ \t]*)dmlim[ \t]*=[ \t]*[\d.]+" => SubstitutionString("\\1dmlim = $d")])
         for d in DMLIMS]
# --- Sweep B: singfac_min (approach distance), truncation fixed at recommended dmlim=0.2 ---
runsB = [run_one("sfac_$(sf)", [r"set_psilim_via_dmlim\s*=\s*\w+" => "set_psilim_via_dmlim = true",
                                r"(?m)^([ \t]*)dmlim[ \t]*=[ \t]*[\d.]+" => SubstitutionString("\\1dmlim = 0.2"),
                                r"singfac_min\s*=\s*[\d.eE+-]+" => "singfac_min = $sf"])
         for sf in SINGFAC]

report("Sweep A — delta_coil vs outer truncation dmlim ∈ (0,1)  [set_psilim_via_dmlim=true]", "dmlim", DMLIMS, runsA)
report("Sweep B — delta_coil vs rational-surface cutoff singfac_min  [dmlim fixed at 0.2]", "sfac", SINGFAC, runsB)
println("\ninterpretation: flat rows ⇒ delta_coil robust to that knob at that surface; the edge-most")
println("surface is where truncation (dmlim) and approach distance (singfac_min) bite hardest.")
