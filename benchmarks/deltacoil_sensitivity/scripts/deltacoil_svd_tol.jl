#!/usr/bin/env julia
# deltacoil_svd_tol.jl — completes the numerical-robustness metric set for delta_coil:
#   (A) SVD of the delta_coil matrix per run — singular VALUES (spectrum, spectral & Frobenius norm,
#       condition number) and singular VECTORS (leading left-vector alignment with the nominal).
#   (B) integration-tolerance scan (eulerlagrange_tolerance) with the per-surface norm + dot-product
#       metrics, to confirm delta_coil is insensitive to the ODE tolerance.
#
# delta_coil_matrix is (2·msing, ncoil). SVD works on any shape; the left singular vectors live in the
# (2·msing) surface-side space, so leading-vector alignment vs nominal is defined only when msing matches
# (marked n/a when a dmlim run has dropped q5). Runs use gal_match_flag=false (delta_coil is independent
# of the match). Nominal = baseline as-shipped.
#
# Usage: julia --project=<GPEC> scripts/deltacoil_svd_tol.jl <example_dir>

using GeneralizedPerturbedEquilibrium, HDF5, LinearAlgebra, Printf

length(ARGS) >= 1 || error("usage: julia --project=<GPEC> scripts/deltacoil_svd_tol.jl <example_dir>")
base = ARGS[1]; isdir(base) || error("no such dir: $base")
ENV["DELTACOIL_MODE"] = "driven"; delete!(ENV, "DELTACOIL_PROJECT")

DMLIMS = [0.30, 0.42, 0.50]                 # below / at / above the q5-drop threshold (0.426)
TOLS   = [1e-8, 1e-10, 1e-12]               # eulerlagrange_tolerance

base_toml = read(joinpath(base, "gpec.toml"), String)
scratch = mktempdir()
eqm = match(r"eq_filename\s*=\s*\"([^\"]+)\"", base_toml); eqm === nothing && error("no eq_filename")
eqfile = eqm.captures[1]; eqsrc = isabspath(eqfile) ? eqfile : joinpath(base, eqfile)

# returns (q-values, delta_coil matrix M as (2msing, ncoil))
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
        q  = sort(round.(vec(read(fid, "singular/q")); digits=2))
        M  = read(fid, "singular/delta_coil_matrix")
        M  = size(M, 1) < size(M, 2) ? M : permutedims(M)      # (2msing, ncoil)
        (q, Matrix{ComplexF64}(M))
    end
end

cosv(a, b) = (length(a) == length(b) && norm(a) > 0 && norm(b) > 0) ? abs(dot(a, b)) / (norm(a) * norm(b)) : NaN
persurf(q, M) = Dict(q[s] => vec(M[2s-1:2s, :]) for s in 1:length(q))

@info "=== nominal ==="; qn, Mn = run_one("nominal", Pair{Regex,String}[])
Fn = svd(Mn); u1n = Fn.U[:, 1]; vsurf_n = persurf(qn, Mn)

runsD = [(d, run_one("dmlim_$(d)", [r"set_psilim_via_dmlim\s*=\s*\w+" => "set_psilim_via_dmlim = true",
                                    r"(?m)^([ \t]*)dmlim[ \t]*=[ \t]*[\d.]+" => SubstitutionString("\\1dmlim = $d")])...)
         for d in DMLIMS]
runsT = [(t, run_one("tol_$(t)", [r"eulerlagrange_tolerance\s*=\s*[\d.eE+-]+" => "eulerlagrange_tolerance = $t"])...)
         for t in TOLS]

# ---------- (A) SVD metrics ----------
println("\n" * "="^84)
println("  (A) SVD of the delta_coil matrix  M = (2·msing × ncoil)")
println("="^84)
allruns = [("nominal", qn, Mn); [("dmlim=$d", q, M) for (d, q, M) in runsD]; [("tol=$t", q, M) for (t, q, M) in runsT]]
@printf("  %-11s | %-6s | %-10s | %-10s | %-9s | %-9s | %-10s\n",
        "run", "2msing", "‖M‖2 (σ1)", "‖M‖F", "σ_min", "cond", "u1·u1_nom")
println("  " * "-"^80)
for (name, q, M) in allruns
    F = svd(M)
    u1 = F.U[:, 1]
    align = cosv(u1n, u1)
    @printf("  %-11s | %-6d | %10.4e | %10.4e | %9.3e | %9.2e | %s\n",
            name, size(M, 1), F.S[1], norm(M), F.S[end], F.S[1] / F.S[end],
            isnan(align) ? "  n/a" : @sprintf("%8.5f", align))
end

println("\n  singular-value spectrum σ_i (first 6):")
@printf("  %-11s", "run")
for i in 1:6; @printf(" | σ%-7d", i); end; println()
for (name, q, M) in allruns
    S = svd(M).S
    @printf("  %-11s", name)
    for i in 1:6; @printf(" | %8.3e", i <= length(S) ? S[i] : NaN); end; println()
end

# ---------- (B) integration-tolerance robustness ----------
println("\n" * "="^84)
println("  (B) eulerlagrange_tolerance scan — per-surface ‖v‖ and cosine vs nominal")
println("="^84)
qs = sort(collect(keys(vsurf_n)))
@printf("  %-8s", "q\\tol"); for t in TOLS; @printf(" | %9.0e", t); end; println("   (nominal = as-shipped)")
println("  (1) ‖delta_coil‖ per surface:")
for q in qs
    @printf("  q=%-5.1f", q)
    for (t, qq, M) in runsT
        v = persurf(qq, M); @printf(" | %9s", haskey(v, q) ? @sprintf("%.3e", norm(v[q])) : "—")
    end; println()
end
println("  (2) cosine with nominal:")
for q in qs
    @printf("  q=%-5.1f", q)
    for (t, qq, M) in runsT
        v = persurf(qq, M); @printf(" | %9s", haskey(v, q) ? @sprintf("%.6f", cosv(vsurf_n[q], v[q])) : "—")
    end; println()
end
println("\nread: (A) stable σ-spectrum + u1·u1_nom≈1 ⇒ matrix conditioning robust; (B) cosine≈1 across")
println("tolerances ⇒ delta_coil independent of the ODE tolerance (as it is of singfac_min).")
