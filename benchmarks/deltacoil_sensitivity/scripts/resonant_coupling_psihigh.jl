#!/usr/bin/env julia
# resonant_coupling_psihigh.jl -- truncation scan of the MATCHED resonant coupling (Wang 2020 Eq. 11
# output) versus the outer boundary psihigh, for BOTH match paths from a single run:
#   STRIDE path   : singular/resonant_match/cout   (Riccati outer delta_coil + inner layer)
#   galerkin path : galerkin/match/cout            (RDCON weak-form outer + inner layer)
# Per rational surface we report ||cout_surface|| (norm over coil modes of that surface's two sides).
# This is the physical matched coupling, downstream of the delta_coil we scanned before.
#
# The match needs per-surface resistive arrays (gal_eta/gal_rho/gal_rotation) of length msing, and msing
# changes as psihigh passes surfaces. So we first run the full domain (arrays already correct in the toml)
# to get the surface locations, then size the arrays to the surface count at each psihigh.
#
# Usage: julia --project=<GPEC> scripts/resonant_coupling_psihigh.jl <example_dir> [N] [psihigh_max]

using GeneralizedPerturbedEquilibrium, HDF5, LinearAlgebra, Printf

length(ARGS) >= 1 || error("usage: julia --project=<GPEC> scripts/resonant_coupling_psihigh.jl <example_dir> [N] [psihigh_max]")
base = ARGS[1]; isdir(base) || error("no such dir: $base")
N        = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 12
PSIH_MAX = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 0.993
ENV["DELTACOIL_MODE"] = "driven"; delete!(ENV, "DELTACOIL_PROJECT")

base_toml = read(joinpath(base, "gpec.toml"), String)
scratch = mktempdir()
eqm = match(r"eq_filename\s*=\s*\"([^\"]+)\"", base_toml)
eqfile = eqm === nothing ? nothing : eqm.captures[1]
eqsrc  = eqfile === nothing ? nothing : (isabspath(eqfile) ? eqfile : joinpath(base, eqfile))

# base resistive values (uniform across surfaces in the examples): parse first entry of each array
firstval(key, default) = (m = match(Regex("$key\\s*=\\s*\\[\\s*([\\d.eE+-]+)"), base_toml); m === nothing ? default : parse(Float64, m.captures[1]))
ETA = firstval("gal_eta", 8e-8); RHO = firstval("gal_rho", 3.3e-7); ROT = firstval("gal_rotation", 1.0)

# run the match at one psihigh with resistive arrays of length `ms` (nothing => keep toml arrays = full domain)
function run_match(ph, ms)
    tag = "ph_$(ph)"
    dir = joinpath(scratch, tag); mkpath(dir)
    if eqfile !== nothing
        dst = joinpath(dir, basename(eqfile)); islink(dst) || symlink(realpath(eqsrc), dst)
    end
    toml = base_toml
    strip_keys = ["force_termination", "HDF5_filename", "psiedge", "set_psilim_via_dmlim",
        "truncate_at_dW_peak", "thmax0", "delta_mband", "gal_match_flag"]
    ms === nothing || append!(strip_keys, ["gal_eta", "gal_rho", "gal_rotation"])   # only resize when ms given; keep toml arrays for full domain
    for key in strip_keys
        toml = replace(toml, Regex("(?m)^[ \\t]*$key[ \\t]*=.*\\n") => "")
    end
    arrblk = ms === nothing ? "" :
        "gal_eta = [" * join(fill(string(ETA), ms), ", ") * "]\n" *
        "gal_rho = [" * join(fill(string(RHO), ms), ", ") * "]\n" *
        "gal_rotation = [" * join(fill(string(ROT), ms), ", ") * "]\n"
    toml = replace(toml, "[ForceFreeStates]" =>
        "[ForceFreeStates]\nforce_termination = true\nHDF5_filename = \"$tag.h5\"\npsiedge = 1.0\n" *
        "set_psilim_via_dmlim = false\ntruncate_at_dW_peak = false\ngal_match_flag = true\n" * arrblk; count=1)
    toml = replace(toml, r"(?m)^([ \t]*)psihigh[ \t]*=[ \t]*[\d.eE+-]+" => SubstitutionString("\\1psihigh = $ph"))
    eqfile !== nothing && (toml = replace(toml, r"eq_filename\s*=\s*\"[^\"]+\"" => "eq_filename = \"$(basename(eqfile))\""))
    write(joinpath(dir, "gpec.toml"), toml)
    @info ">>> psihigh = $ph  (msing arrays = $(ms === nothing ? "toml" : ms))"
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
        psi = vec(read(fid, "singular/psi"))
        # per-surface norm of cout: cout is (ncoil, 2msing), columns are surface-sides
        percol(dc) = begin
            dc = size(dc, 1) >= size(dc, 2) ? dc : permutedims(dc)   # -> (ncoil, 2msing)
            Dict(round(q[s]; digits=2) => norm(dc[:, 2s-1:2s]) for s in 1:length(q))
        end
        stride_c = haskey(fid, "singular/resonant_match/cout") ? percol(read(fid, "singular/resonant_match/cout")) : Dict{Float64,Float64}()
        gal_c    = haskey(fid, "galerkin/match/cout")          ? percol(read(fid, "galerkin/match/cout"))          : Dict{Float64,Float64}()
        (q=sort(round.(q; digits=2)), psi=psi, stride=stride_c, gal=gal_c)
    end
end

# --- nominal at full domain first: gives surface locations and full msing (toml arrays already correct) ---
@info "=== nominal (full domain) ==="
nom = run_match(PSIH_MAX, nothing)
nom === nothing && error("nominal full-domain match failed")
psi_s = sort(nom.psi)
@info "surfaces (psi_s): $psi_s   full msing=$(length(psi_s))"

# grid: log-packed toward edge, from just past the innermost surface to the edge
PSIHIGHS = round.(1 .- 10 .^ range(log10(1 - (psi_s[1] + 1e-3)), log10(1 - PSIH_MAX); length=N); digits=5)

results = Tuple{Float64,Any}[]
for ph in PSIHIGHS
    ms = count(<(ph), psi_s)
    push!(results, (ph, ms == 0 ? nothing : (ph == PSIH_MAX ? nom : run_match(ph, ms))))
end
# make sure the nominal (max) point is present
any(r -> r[1] == PSIH_MAX, results) || push!(results, (PSIH_MAX, nom))

allq = sort(collect(nom.q))
outdir = joinpath(@__DIR__, "..", "results"); mkpath(outdir)
tag = replace(basename(base), r"[^A-Za-z0-9]" => "")
csv = joinpath(outdir, "resonant_coupling_psihigh_$tag.csv")
open(csv, "w") do io
    hdr = "psihigh," * join(["stride_q$(q)" for q in allq], ",") * "," * join(["gal_q$(q)" for q in allq], ",")
    println(io, hdr)
    for (ph, r) in sort(results, by=first)
        sv = [r === nothing ? "" : string(get(r.stride, q, "")) for q in allq]
        gv = [r === nothing ? "" : string(get(r.gal, q, ""))    for q in allq]
        println(io, "$ph," * join(sv, ",") * "," * join(gv, ","))
    end
end

println("\n" * "="^78)
println("  MATCHED resonant coupling ||cout|| per surface vs psihigh  ($(basename(base)))")
println("  STRIDE = singular/resonant_match/cout ; galerkin = galerkin/match/cout")
println("="^78)
@printf("  %-9s", "psihigh")
for q in allq; @printf(" | S:q%-4.1f", q); end
for q in allq; @printf(" | G:q%-4.1f", q); end; println()
for (ph, r) in sort(results, by=first)
    @printf("  %-9.4f", ph)
    for q in allq; @printf(" | %7s", r === nothing ? "-" : (haskey(r.stride, q) ? @sprintf("%.2e", r.stride[q]) : "-")); end
    for q in allq; @printf(" | %7s", r === nothing ? "-" : (haskey(r.gal, q)    ? @sprintf("%.2e", r.gal[q])    : "-")); end
    println()
end
println("\nwrote: $(abspath(csv))")
