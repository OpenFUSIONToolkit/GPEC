#!/usr/bin/env julia
# deltacoil_metrics.jl — per-singular-surface sensitivity of delta_coil to dmlim and singfac_min,
# reported with TWO complementary metrics against a nominal (baseline) run:
#   (1) ‖v_s‖               — magnitude of the surface's delta_coil vector (over all coil modes)
#   (2) |⟨v_s^nom, v_s⟩| / (‖v_s^nom‖‖v_s‖)  — normalized Hermitian dot product (cosine similarity)
#       with the nominal: 1.0 = identical pattern (only rescaled), <1.0 = the response ROTATED.
#
# delta_coil_matrix is (2·msing, ncoil): each singular surface = 2 rows (L/R small solutions), so a
# surface's vector v_s = vec(dc[2s-1:2s, :]) has length 2·ncoil. The dot product needs matching ncoil,
# so when a dmlim run drops q5 (changing the resonant m-band → ncoil), cosine is marked n/a.
# Run with gal_match_flag=false (delta_coil is computed by the rpec loop, independent of the match).
#
# Nominal = the baseline DIII-D case AS-SHIPPED (set_psilim_via_dmlim=false, dmlim=0.2, singfac_min=1e-4).
#
# Usage:  julia --project=<GPEC> scripts/deltacoil_metrics.jl examples/DIIID-like_gal_resistive_pe_example

using GeneralizedPerturbedEquilibrium, HDF5, LinearAlgebra, Printf

length(ARGS) >= 1 || error("usage: julia --project=<GPEC> scripts/deltacoil_metrics.jl <example_dir>")
base = ARGS[1]; isdir(base) || error("no such dir: $base")
ENV["DELTACOIL_MODE"] = "driven"; delete!(ENV, "DELTACOIL_PROJECT")

DMLIMS  = length(ARGS) >= 2 ? parse.(Float64, ARGS[2:end]) : [0.1, 0.3, 0.5, 0.7, 0.9]
SINGFAC = [1e-5, 1e-4, 1e-3]
SKIP_B  = get(ENV, "SKIP_SINGFAC", "") != ""   # set SKIP_SINGFAC=1 to run only the dmlim sweep

base_toml = read(joinpath(base, "gpec.toml"), String)
scratch = mktempdir()
eqm = match(r"eq_filename\s*=\s*\"([^\"]+)\"", base_toml); eqm === nothing && error("no eq_filename")
eqfile = eqm.captures[1]; eqsrc = isabspath(eqfile) ? eqfile : joinpath(base, eqfile)

# returns (sorted q-values, Dict q=>complex surface-vector, ncoil)
function run_one(tag, patches)
    dir = joinpath(scratch, tag); mkpath(dir)
    dst = joinpath(dir, basename(eqfile)); islink(dst) || symlink(realpath(eqsrc), dst)
    toml = base_toml
    for (r, s) in patches; toml = replace(toml, r => s); end
    toml = replace(toml, r"gal_match_flag\s*=\s*\w+" => "gal_match_flag = false")
    toml = replace(toml, r"eq_filename\s*=\s*\"[^\"]+\"" => "eq_filename = \"$(basename(eqfile))\"")
    toml = replace(toml, r"HDF5_filename\s*=\s*\"[^\"]+\"" => "HDF5_filename = \"$tag.h5\"")
    write(joinpath(dir, "gpec.toml"), toml)
    @info ">>> $tag"
    GeneralizedPerturbedEquilibrium.main([dir])
    h5open(joinpath(dir, "$tag.h5"), "r") do fid
        q  = vec(read(fid, "singular/q"))
        dc = read(fid, "singular/delta_coil_matrix")
        dc = size(dc, 1) < size(dc, 2) ? dc : permutedims(dc)     # -> (2msing, ncoil)
        ncoil = size(dc, 2)
        v = Dict{Float64,Vector{ComplexF64}}()
        for s in 1:length(q); v[round(q[s]; digits=2)] = vec(dc[2s-1:2s, :]); end
        (sort(round.(q; digits=2)), v, ncoil)
    end
end

cosine(a, b) = (length(a) == length(b) && norm(a) > 0 && norm(b) > 0) ? abs(dot(a, b)) / (norm(a) * norm(b)) : NaN

# --- nominal (baseline as-shipped) ---
@info "=== nominal (baseline) ==="
qn, vnom, nc_nom = run_one("nominal", Pair{Regex,String}[])

# --- Sweep A: dmlim (truncation on) ---
runsA = [run_one("dmlim_$(d)", [r"set_psilim_via_dmlim\s*=\s*\w+" => "set_psilim_via_dmlim = true",
                                r"(?m)^([ \t]*)dmlim[ \t]*=[ \t]*[\d.]+" => SubstitutionString("\\1dmlim = $d")])
         for d in DMLIMS]
# --- Sweep B: singfac_min (truncation on, dmlim fixed 0.2) ---
runsB = SKIP_B ? Nothing[] :
        [run_one("sfac_$(sf)", [r"set_psilim_via_dmlim\s*=\s*\w+" => "set_psilim_via_dmlim = true",
                                r"(?m)^([ \t]*)dmlim[ \t]*=[ \t]*[\d.]+" => SubstitutionString("\\1dmlim = 0.2"),
                                r"singfac_min\s*=\s*[\d.eE+-]+" => "singfac_min = $sf"])
         for sf in SINGFAC]

function report(title, param, vals, runs)
    qs = sort(collect(keys(vnom)))
    println("\n" * "="^80); println("  $title"); println("="^80)
    println("  nominal ‖v‖ per surface: " * join(["q$q=$(round(norm(vnom[q]);sigdigits=4))" for q in qs], "  "))
    println("\n  (1) ‖delta_coil‖ per surface (— = surface absent):")
    @printf("  %-7s", "q\\$param")
    for v in vals; @printf(" | %9.3g", v); end; println()
    for q in qs
        @printf("  q=%-5.1f", q)
        for r in runs; @printf(" | %9s", haskey(r[2], q) ? @sprintf("%.3e", norm(r[2][q])) : "—"); end
        println(q == maximum(qs) ? "   <- edge" : "")
    end
    println("\n  (2) cosine similarity with nominal  |<v_nom,v>|/(‖v_nom‖‖v‖)   (1.0 = same pattern):")
    @printf("  %-7s", "q\\$param")
    for v in vals; @printf(" | %9.3g", v); end; println()
    for q in qs
        @printf("  q=%-5.1f", q)
        for r in runs
            if haskey(r[2], q)
                c = cosine(vnom[q], r[2][q])
                @printf(" | %9s", isnan(c) ? "n/a(ncoil)" : @sprintf("%.6f", c))
            else
                @printf(" | %9s", "—")
            end
        end
        println()
    end
end

report("Sweep A — delta_coil vs dmlim ∈ (0,1)   [nominal ncoil=$nc_nom]", "dmlim", DMLIMS, runsA)
SKIP_B || report("Sweep B — delta_coil vs singfac_min   [dmlim=0.2]", "sfac", SINGFAC, runsB)
println("\nread: ‖v‖ moves ⇒ magnitude sensitivity; cosine<1 ⇒ the response PATTERN over coil modes rotated.")
println("cosine≈1 with ‖v‖ varying ⇒ pure rescaling; cosine falling ⇒ genuine change of the coil-response shape.")
