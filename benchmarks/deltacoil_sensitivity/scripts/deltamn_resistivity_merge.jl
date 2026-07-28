#!/usr/bin/env julia
# deltamn_resistivity_merge.jl -- Task 5 capstone: merge the outer coupling matrix with the
# eta-dependent inner layer to get the full resonant response vs resistivity.
#
# The coefficient-based Delta_mn is assembled (GalerkinMatch.jl) as
#     S_small       = transpose(cout) * Dm[1:2msing, :]  +  Dm[2msing+1 : 2msing+mcoil, :]
#     delta_mn[i,j] = KR[i]*S_small[j,2i] - KL[i]*S_small[j,2i-1]
# with Dm = galerkin/delta the OUTER Delta-prime matrix. Only `cout` carries the resistivity
# dependence (through the matched inner-layer Delta), so Delta_mn splits into
#     delta_mn = delta_mn_coil (eta-independent, from the Dm coil block)
#              + delta_mn_plasma (eta-dependent, the cout-weighted plasma term).
# The per-surface kernels KL,KR are not written to HDF5, so we recover them per surface by least
# squares from the stored delta_mn and S_small (2 complex unknowns, mcoil equations) and report the
# fit residual so the split is verifiable rather than assumed.
#
# Scans eta with the "ray" inner backend (Task 10 established ray is the trustworthy one at |Q| >~ 1).
# Rotation override via ENV["GAL_ROTATION"] (|Q| ~ f * eta^(-1/3); f=1 Hz keeps |Q| < 1).
#
# Usage: julia --project=<GPEC> scripts/deltamn_resistivity_merge.jl <example_dir> [N_eta] [eta_hi] [eta_lo]

using GeneralizedPerturbedEquilibrium, HDF5, LinearAlgebra, Printf

length(ARGS) >= 1 || error("usage: julia --project=<GPEC> scripts/deltamn_resistivity_merge.jl <example_dir> [N_eta] [eta_hi] [eta_lo]")
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

nsurf = (m = match(r"gal_eta\s*=\s*\[([^\]]*)\]", base_toml); m === nothing ? 0 : count(==(','), m.captures[1]) + 1)
nsurf > 0 || error("could not parse gal_eta length from toml")
RHO = (m = match(r"gal_rho\s*=\s*\[\s*([\d.eE+-]+)", base_toml); m === nothing ? 3.3e-7 : parse(Float64, m.captures[1]))
ROT = (rv = get(ENV, "GAL_ROTATION", ""); rv != "" ? parse(Float64, rv) :
       (m = match(r"gal_rotation\s*=\s*\[\s*([\d.eE+-]+)", base_toml); m === nothing ? 1.0 : parse(Float64, m.captures[1])))

ETAS = 10 .^ range(log10(ETA_HI), log10(ETA_LO); length=NETA)

"Recover KL,KR per surface by least squares, then split delta_mn into its coil and plasma parts."
function decompose(dmn, cout, Dm)
    msing, mcoil = size(dmn)
    S_plasma = transpose(cout) * Dm[1:2msing, :]              # (mcoil × 2msing) eta-dependent
    S_coil   = Dm[(2msing+1):(2msing+mcoil), :]               # (mcoil × 2msing) eta-independent
    S_small  = S_plasma .+ S_coil
    n_tot = zeros(msing); n_coil = zeros(msing); n_plas = zeros(msing); fitres = zeros(msing)
    for i in 1:msing
        A = hcat(S_small[:, 2i], -S_small[:, 2i-1])           # columns multiply [KR; KL]
        b = vec(dmn[i, :])
        x = A \ b
        fitres[i] = norm(A * x - b) / max(norm(b), 1e-300)
        KR, KL = x[1], x[2]
        n_tot[i]  = norm(b)
        n_coil[i] = norm(KR .* S_coil[:, 2i]   .- KL .* S_coil[:, 2i-1])
        n_plas[i] = norm(KR .* S_plasma[:, 2i] .- KL .* S_plasma[:, 2i-1])
    end
    return (tot=n_tot, coil=n_coil, plasma=n_plas, fitres=fitres)
end

function run_eta(eta)
    tag = "eta_$(eta)_ray"
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
        "gal_inner_solver = \"ray\"\n" * arrs; count=1)
    eqfile !== nothing && (toml = replace(toml, r"eq_filename\s*=\s*\"[^\"]+\"" => "eq_filename = \"$(basename(eqfile))\""))
    write(joinpath(dir, "gpec.toml"), toml)
    @info ">>> eta = $eta  (ray, f = $ROT Hz)"
    try
        GeneralizedPerturbedEquilibrium.main([dir])
    catch err
        @warn "run failed at eta=$eta" exception = (err, catch_backtrace())
        return nothing
    end
    h5path = joinpath(dir, "$tag.h5")
    isfile(h5path) || return nothing
    h5open(h5path, "r") do fid
        haskey(fid, "singular/q") || return nothing
        q = vec(read(fid, "singular/q")); isempty(q) && return nothing
        (haskey(fid, "galerkin/match/delta_mn") && haskey(fid, "galerkin/match/cout") &&
         haskey(fid, "galerkin/delta")) || return nothing
        dmn  = read(fid, "galerkin/match/delta_mn")   # (msing × mcoil)
        cout = read(fid, "galerkin/match/cout")       # (2msing × mcoil)
        Dm   = read(fid, "galerkin/delta")            # (2msing+mcoil × 2msing)
        dr   = haskey(fid, "galerkin/match/deltar") ? read(fid, "galerkin/match/deltar") : nothing
        bp   = haskey(fid, "galerkin/match/bpen")   ? read(fid, "galerkin/match/bpen")   : nothing
        dec = decompose(dmn, cout, Dm)
        qk = round.(q; digits=2)
        deltar = Dict{Float64,Float64}(); bpen = Dict{Float64,Float64}()
        dmn_t = Dict{Float64,Float64}(); dmn_c = Dict{Float64,Float64}()
        dmn_p = Dict{Float64,Float64}(); fitr = Dict{Float64,Float64}()
        for s in 1:length(q)
            k = qk[s]
            dr === nothing || (deltar[k] = norm(dr[s, :]))
            if bp !== nothing
                b = size(bp, 1) >= size(bp, 2) ? bp : permutedims(bp)
                bpen[k] = norm(b[:, s])
            end
            dmn_t[k] = dec.tot[s]; dmn_c[k] = dec.coil[s]
            dmn_p[k] = dec.plasma[s]; fitr[k] = dec.fitres[s]
        end
        (q=sort(qk), deltar=deltar, bpen=bpen, dmn=dmn_t, dmn_coil=dmn_c, dmn_plasma=dmn_p, fitres=fitr)
    end
end

rows = Tuple{Float64,Any}[]
for eta in ETAS
    push!(rows, (eta, run_eta(eta)))
end
allq = sort(collect(reduce(union, (Set(keys(r[2].dmn)) for r in rows if r[2] !== nothing); init=Set{Float64}())))

outdir = joinpath(@__DIR__, "..", "results"); mkpath(outdir)
tag = replace(basename(base), r"[^A-Za-z0-9]" => "") * "_f" * replace(string(ROT), "." => "p")
csv = joinpath(outdir, "deltamn_merge_$tag.csv")
open(csv, "w") do io
    cols = String["eta"]
    for q in allq, m in ("dmn", "dmn_coil", "dmn_plasma", "deltar", "bpen", "fitres")
        push!(cols, "$(m)_q$(q)")
    end
    println(io, join(cols, ","))
    for (eta, r) in rows
        vals = String[string(eta)]
        for q in allq, d in (r === nothing ? [nothing for _ in 1:6] :
                             [r.dmn, r.dmn_coil, r.dmn_plasma, r.deltar, r.bpen, r.fitres])
            push!(vals, d === nothing ? "" : string(get(d, q, "")))
        end
        println(io, join(vals, ","))
    end
end

println("\n" * "="^92)
println("  Delta_mn MERGE scan ($(basename(base)), ray, f = $ROT Hz): full resonant response vs eta")
println("="^92)
@printf("  %-10s", "eta")
for q in allq; @printf(" | q%-4.1f dmn | q%-4.1f coil | q%-4.1f plas", q, q, q); end; println()
for (eta, r) in rows
    @printf("  %-10.2e", eta)
    for q in allq
        if r === nothing
            @printf(" | %9s | %10s | %10s", "-", "-", "-")
        else
            @printf(" | %9s | %10s | %10s",
                haskey(r.dmn, q) ? @sprintf("%.3e", r.dmn[q]) : "-",
                haskey(r.dmn_coil, q) ? @sprintf("%.3e", r.dmn_coil[q]) : "-",
                haskey(r.dmn_plasma, q) ? @sprintf("%.3e", r.dmn_plasma[q]) : "-")
        end
    end
    println()
end
mx = maximum((maximum(values(r.fitres)) for (_, r) in rows if r !== nothing); init=0.0)
@printf("\n  max KL/KR least-squares fit residual over all eta and surfaces: %.3e (should be ~1e-12 or below)\n", mx)
println("wrote: $(abspath(csv))")
