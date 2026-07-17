#!/usr/bin/env julia
# deltacoil_rdcon_convergence.jl — does STRIDE's delta_coil converge to RDCON as STRIDE refines?
#
# Sweeps STRIDE's singular-expansion order (sing_order) and, per inner surface (q2, q3), reports:
#   (a) STRIDE self-convergence — |dc(order) − dc(order_max)| / |dc(order_max)|  (should → 0)
#   (b) correlation of STRIDE |delta_coil| with the FIXED RDCON reference (galerkin/delta_coil).
# If STRIDE self-converges (a→0) but the RDCON correlation (b) stays FLAT below 1, the residual is
# the weak-form-vs-strong-form representation, not STRIDE under-resolution. If (b) climbs toward 1,
# it was resolution. Runs with the match off (delta_coil only). RDCON ref read from the example gpec.h5.
#
# Usage: julia --project=<GPEC> scripts/deltacoil_rdcon_convergence.jl <example_dir> [ord1 ord2 ...]

using GeneralizedPerturbedEquilibrium, HDF5, LinearAlgebra, Statistics, Printf

length(ARGS) >= 1 || error("usage: julia --project=<GPEC> scripts/deltacoil_rdcon_convergence.jl <example_dir> [orders...]")
base = ARGS[1]; isdir(base) || error("no such dir: $base")
ORDERS = length(ARGS) >= 2 ? parse.(Int, ARGS[2:end]) : [4, 6, 8]
ENV["DELTACOIL_MODE"] = "driven"; delete!(ENV, "DELTACOIL_PROJECT")

base_toml = read(joinpath(base, "gpec.toml"), String)
scratch = mktempdir()
eqm = match(r"eq_filename\s*=\s*\"([^\"]+)\"", base_toml); eqm === nothing && error("no eq_filename")
eqfile = eqm.captures[1]; eqsrc = isabspath(eqfile) ? eqfile : joinpath(base, eqfile)

align8(a) = size(a, 1) == 8 ? a : permutedims(a)                       # -> (8 surface-side, ncoil)
R = align8(h5read(joinpath(base, "gpec.h5"), "galerkin/delta_coil"))    # fixed RDCON reference
corr(a, b) = cor(abs.(vec(a)), abs.(vec(b)))

function run_order(ord)
    dir = joinpath(scratch, "ord_$ord"); mkpath(dir)
    dst = joinpath(dir, basename(eqfile)); islink(dst) || symlink(realpath(eqsrc), dst)
    toml = replace(base_toml, r"gal_match_flag\s*=\s*\w+" => "gal_match_flag = false")
    toml = replace(toml, r"gal_sing_order_ceiling\s*=\s*\w+" => "gal_sing_order_ceiling = false")
    if occursin(r"(?m)^[ \t]*sing_order[ \t]*=[ \t]*\d+", toml)
        toml = replace(toml, r"(?m)^([ \t]*)sing_order[ \t]*=[ \t]*\d+" => SubstitutionString("\\1sing_order = $ord"))
    else
        toml = replace(toml, "[ForceFreeStates]" => "[ForceFreeStates]\nsing_order = $ord"; count=1)
    end
    toml = replace(toml, r"eq_filename\s*=\s*\"[^\"]+\"" => "eq_filename = \"$(basename(eqfile))\"")
    toml = replace(toml, r"HDF5_filename\s*=\s*\"[^\"]+\"" => "HDF5_filename = \"ord_$ord.h5\"")
    write(joinpath(dir, "gpec.toml"), toml)
    @info ">>> sing_order = $ord"
    GeneralizedPerturbedEquilibrium.main([dir])
    align8(h5read(joinpath(dir, "ord_$ord.h5"), "singular/delta_coil_matrix"))
end

res = Dict(o => run_order(o) for o in ORDERS)
omax = maximum(ORDERS); ref = res[omax]
inner_rows = [1, 2, 3, 4]   # q2L,q2R,q3L,q3R

println("\n" * "="^70)
println("  STRIDE delta_coil: self-convergence vs correlation with RDCON (inner surfaces)")
println("="^70)
@printf("  %-6s | %-22s | %-22s\n", "order", "(a) self-conv vs ord $omax", "(b) corr with RDCON")
@printf("  %-6s | %-10s %-10s | %-10s %-10s\n", "", "q2", "q3", "q2", "q3")
println("  " * "-"^62)
for o in sort(ORDERS)
    sc(rows) = norm(abs.(res[o][rows,:]) .- abs.(ref[rows,:])) / max(norm(abs.(ref[rows,:])), 1e-300)
    cr(rows) = mean(corr(res[o][r,:], R[r,:]) for r in rows)
    @printf("  %-6d | %-10.2e %-10.2e | %-10.3f %-10.3f\n", o, sc(1:2), sc(3:4), cr(1:2), cr(3:4))
end
println("\ninterpretation: if (a) → 0 while (b) stays flat below 1, the STRIDE↔RDCON gap is representational")
println("(weak-form vs strong-form), NOT STRIDE resolution — so refining STRIDE will not close it.")
