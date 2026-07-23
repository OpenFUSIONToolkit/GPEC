#!/usr/bin/env julia
# resistivity_scan.jl -- resistivity (eta) scan of the matched inner-layer response, comparing the two
# inner-layer backends in the galerkin match: "ray" (rotated-contour collocation, robust for |Q| >~ 1)
# and "galerkin" (Hermite-cubic inps, drifts for |Q| >~ 1). psihigh is held at the full domain; only eta
# is varied (|Q| grows as eta falls, roughly |Q| ~ eta^(-1/3)). Per surface we report:
#   |deltar|  -- the inner-layer Delta(Q) (galerkin/match/deltar), where the two backends differ, and
#   ||bpen||  -- the matched penetrated field (galerkin/match/bpen), the shielding family.
# The branch result to reproduce: ray gives a smooth, monotonic shielding family; galerkin jumps at low eta.
#
# Usage: julia --project=<GPEC> scripts/resistivity_scan.jl <example_dir> [N_eta] [eta_hi] [eta_lo]

using GeneralizedPerturbedEquilibrium, HDF5, LinearAlgebra, Printf

length(ARGS) >= 1 || error("usage: julia --project=<GPEC> scripts/resistivity_scan.jl <example_dir> [N_eta] [eta_hi] [eta_lo]")
base = ARGS[1]; isdir(base) || error("no such dir: $base")
NETA   = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 8
ETA_HI = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 1e-6
ETA_LO = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 1e-13
ENV["DELTACOIL_MODE"] = "driven"; delete!(ENV, "DELTACOIL_PROJECT")

base_toml = read(joinpath(base, "gpec.toml"), String)
scratch = mktempdir()
eqm = match(r"eq_filename\s*=\s*\"([^\"]+)\"", base_toml)
eqfile = eqm === nothing ? nothing : eqm.captures[1]
eqsrc  = eqfile === nothing ? nothing : (isabspath(eqfile) ? eqfile : joinpath(base, eqfile))

# how many surfaces (length of the toml's gal_eta array = full-domain msing)
nsurf = (m = match(r"gal_eta\s*=\s*\[([^\]]*)\]", base_toml); m === nothing ? 0 : count(==(','), m.captures[1]) + 1)
nsurf > 0 || error("could not parse gal_eta length from toml")
RHO = (m = match(r"gal_rho\s*=\s*\[\s*([\d.eE+-]+)", base_toml); m === nothing ? 3.3e-7 : parse(Float64, m.captures[1]))
# rotation f [Hz] sets gamma=2pi n f, hence |Q| = 2pi n f / Q0 (Q0 ∝ eta^1/3). Override via ENV to reach
# the |Q| >~ 1 regime where the galerkin inner layer drifts (example default f=1 Hz keeps |Q| < 1).
ROT = (rv = get(ENV, "GAL_ROTATION", ""); rv != "" ? parse(Float64, rv) :
       (m = match(r"gal_rotation\s*=\s*\[\s*([\d.eE+-]+)", base_toml); m === nothing ? 1.0 : parse(Float64, m.captures[1])))

ETAS = 10 .^ range(log10(ETA_HI), log10(ETA_LO); length=NETA)

function run_eta(eta, solver)
    tag = "eta_$(eta)_$(solver)"
    dir = joinpath(scratch, tag); mkpath(dir)
    if eqfile !== nothing
        dst = joinpath(dir, basename(eqfile)); islink(dst) || symlink(realpath(eqsrc), dst)
    end
    toml = base_toml
    for key in ("force_termination", "HDF5_filename", "gal_match_flag", "gal_inner_solver",
        "thmax0", "delta_mband", "gal_eta", "gal_rho", "gal_rotation")
        toml = replace(toml, Regex("(?m)^[ \\t]*$key[ \\t]*=.*\\n") => "")
    end
    arrs = "gal_eta = [" * join(fill(string(eta), nsurf), ", ") * "]\n" *
           "gal_rho = [" * join(fill(string(RHO), nsurf), ", ") * "]\n" *
           "gal_rotation = [" * join(fill(string(ROT), nsurf), ", ") * "]\n"
    toml = replace(toml, "[ForceFreeStates]" =>
        "[ForceFreeStates]\nforce_termination = true\nHDF5_filename = \"$tag.h5\"\ngal_match_flag = true\n" *
        "gal_inner_solver = \"$solver\"\n" * arrs; count=1)
    eqfile !== nothing && (toml = replace(toml, r"eq_filename\s*=\s*\"[^\"]+\"" => "eq_filename = \"$(basename(eqfile))\""))
    write(joinpath(dir, "gpec.toml"), toml)
    @info ">>> eta = $eta  solver = $solver"
    try
        GeneralizedPerturbedEquilibrium.main([dir])
    catch err
        @warn "run failed at eta=$eta solver=$solver" exception = (err, catch_backtrace())
        return nothing
    end
    h5path = joinpath(dir, "$tag.h5")
    isfile(h5path) || return nothing
    h5open(h5path, "r") do fid
        haskey(fid, "singular/q") || return nothing
        q = vec(read(fid, "singular/q")); isempty(q) && return nothing
        dr = haskey(fid, "galerkin/match/deltar") ? read(fid, "galerkin/match/deltar") : nothing   # (msing,2)
        bp = haskey(fid, "galerkin/match/bpen")   ? read(fid, "galerkin/match/bpen")   : nothing   # (ncoil,msing)
        deltar = Dict{Float64,Float64}(); bpen = Dict{Float64,Float64}()
        for s in 1:length(q)
            qk = round(q[s]; digits=2)
            dr === nothing || (deltar[qk] = norm(dr[s, :]))
            if bp !== nothing
                b = size(bp, 1) >= size(bp, 2) ? bp : permutedims(bp)
                bpen[qk] = norm(b[:, s])
            end
        end
        (q=sort(round.(q; digits=2)), deltar=deltar, bpen=bpen)
    end
end

rows = Tuple{Float64,Any,Any}[]
for eta in ETAS
    push!(rows, (eta, run_eta(eta, "ray"), run_eta(eta, "galerkin")))
end
allq = sort(collect(reduce(union, (Set(keys(r[2].deltar)) for r in rows if r[2] !== nothing); init=Set{Float64}())))

outdir = joinpath(@__DIR__, "..", "results"); mkpath(outdir)
tag = replace(basename(base), r"[^A-Za-z0-9]" => "") * "_f" * replace(string(ROT), "." => "p")
csv = joinpath(outdir, "resistivity_scan_$tag.csv")
open(csv, "w") do io
    cols = String["eta"]
    for q in allq, m in ("ray_deltar","gal_deltar","ray_bpen","gal_bpen"); push!(cols, "$(m)_q$(q)"); end
    println(io, join(cols, ","))
    for (eta, rr, rg) in rows
        vals = String[string(eta)]
        for q in allq
            push!(vals, rr === nothing ? "" : string(get(rr.deltar, q, "")))
            push!(vals, rg === nothing ? "" : string(get(rg.deltar, q, "")))
            push!(vals, rr === nothing ? "" : string(get(rr.bpen, q, "")))
            push!(vals, rg === nothing ? "" : string(get(rg.bpen, q, "")))
        end
        println(io, join(vals, ","))
    end
end

println("\n" * "="^78)
println("  RESISTIVITY scan ($(basename(base))): inner-layer |deltar| vs eta, ray vs galerkin backend")
println("="^78)
@printf("  %-10s", "eta")
for q in allq; @printf(" | ray:q%-4.1f | gal:q%-4.1f", q, q); end; println()
for (eta, rr, rg) in rows
    @printf("  %-10.2e", eta)
    for q in allq
        @printf(" | %8s | %8s",
            rr === nothing ? "-" : (haskey(rr.deltar, q) ? @sprintf("%.2e", rr.deltar[q]) : "-"),
            rg === nothing ? "-" : (haskey(rg.deltar, q) ? @sprintf("%.2e", rg.deltar[q]) : "-"))
    end
    println()
end
println("\nwrote: $(abspath(csv))")
