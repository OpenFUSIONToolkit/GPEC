#!/usr/bin/env julia
# Gal-enabled TJ-analytic beta (pc) scan: at each pc, run the full GPEC pipeline with gal_flag=true
# and capture three Δ′ measures per resonant surface (m=2 q=2, m=3 q=3):
#   gal      = SingularSurfaces/GalerkinDeltaPrime/pest3_Delta diagonal      (RDCON singular-Galerkin, this port)
#   stride   = SingularSurfaces/delta_prime_matrix diag   (STRIDE BVP)
#   cajump   = (ca_r-ca_l)/((2pi)^2 psio)          (ca-jump primary)
# Reuses examples/LAR_epsilon_scan/gpec.toml. Writes gal_epsilon_scan_results.h5 + CSV next to this script.
using Pkg
const REPO = normpath(joinpath(@__DIR__, "..", ".."))
Pkg.activate(REPO)
using GeneralizedPerturbedEquilibrium
using HDF5, TOML, Printf

_warped_grid(x0, x1, N; p=2.0) = [x0 + (x1 - x0) * (1 - (1 - i / (N - 1))^p) for i in 0:N-1]
const PC_FULL = _warped_grid(0.125, 0.660, 56; p=2.0)
const PC_TEST = [0.2, 0.4, 0.55]
const BASE = TOML.parsefile(joinpath(REPO, "examples", "LAR_epsilon_scan", "gpec.toml"))

function extract(h5)
    h5open(h5, "r") do f
        out = Dict{String,Any}()
        out["q0"] = read(f, "Equilibrium/q0"); out["qmax"] = read(f, "Equilibrium/qmax")
        out["dW_total"] = real(read(f, "ForceFreeStates/FreeBoundaryStability/eigenmode_energies")[1])
        psio = read(f, "Equilibrium/psio"); denom = (2pi)^2 * psio
        mlow = read(f, "Info/mlow"); mpert = read(f, "Info/mpert"); nlow = read(f, "Info/nlow")
        # surface m-values (STRIDE side)
        m_s = vec(read(f, "SingularSurfaces/m")); n_s = vec(read(f, "SingularSurfaces/n")); msing = length(m_s)
        # STRIDE delta_prime_matrix
        dpm = haskey(f, "SingularSurfaces/delta_prime_matrix") ? read(f, "SingularSurfaces/delta_prime_matrix") : nothing
        cal = read(f, "SingularSurfaces/ca_left"); car = read(f, "SingularSurfaces/ca_right")
        # gal
        gm = haskey(f, "SingularSurfaces/GalerkinDeltaPrime/sing_m") ? vec(read(f, "SingularSurfaces/GalerkinDeltaPrime/sing_m")) : Int[]
        gp = haskey(f, "SingularSurfaces/GalerkinDeltaPrime/pest3_Delta") ? read(f, "SingularSurfaces/GalerkinDeltaPrime/pest3_Delta") : nothing
        for (s, m) in enumerate(m_s)
            key = "m$(m)"
            # ca-jump
            ipr = 1 + m - mlow + (n_s[s] - nlow) * mpert
            out["cajump_$key"] = isempty(cal) ? NaN + NaN * im : (car[ipr, ipr, 2, s] - cal[ipr, ipr, 2, s]) / denom  # zero-extent when not computed
            # stride
            out["stride_$key"] = dpm !== nothing && s <= size(dpm, 1) ? dpm[s, s] : NaN + NaN*im
            # gal (match by m)
            gi = findfirst(==(m), gm)
            out["gal_$key"] = (gp !== nothing && gi !== nothing) ? gp[gi, gi] : NaN + NaN*im
        end
        return out
    end
end

function run_point(pc)
    dir = mktempdir(; prefix="gal_eps_")
    try
        c = deepcopy(BASE)
        c["TJ_ANALYTIC_INPUT"]["lar_r0"] = c["TJ_ANALYTIC_INPUT"]["lar_a"] / pc
        c["Equilibrium"]["eq_type"] = "tj_analytic_direct"
        c["ForceFreeStates"]["gal_flag"] = true
        c["ForceFreeStates"]["gal_solver"] = "LU"
        c["ForceFreeStates"]["HDF5_filename"] = joinpath(dir, "gpec.h5")
        open(joinpath(dir, "gpec.toml"), "w") do io; TOML.print(io, c); end
        GeneralizedPerturbedEquilibrium.main([dir])
        return extract(joinpath(dir, "gpec.h5"))
    catch e
        @warn "pc=$pc failed" exception=(e, catch_backtrace())
        return nothing
    finally
        rm(dir; force=true, recursive=true)
    end
end

pcs = ("--test" in ARGS) ? PC_TEST : PC_FULL
@info "Gal epsilon scan: $(length(pcs)) points (ε=0.2, qc=1.5, qa=3.6)"
results = Tuple{Float64,Dict{String,Any}}[]
t0 = time()
for (i, pc) in enumerate(pcs)
    ti = time()
    r = run_point(pc)
    el = time() - ti; tot = time() - t0
    if r !== nothing
        push!(results, (pc, r))
        g2 = get(r, "gal_m2", NaN); s2 = get(r, "stride_m2", NaN)
        g3 = get(r, "gal_m3", NaN); s3 = get(r, "stride_m3", NaN)
        @printf("[%2d/%2d] pc=%.5f  q=2: gal=%+9.3f stride=%+9.3f | q=3: gal=%+9.3f stride=%+9.3f  dW=%+.4f  (%.0fs, tot %.0fs)\n",
            i, length(pcs), pc, real(g2), real(s2), real(g3), real(s3), get(r, "dW_total", NaN), el, tot)
        flush(stdout)
    end
end

# Save
out_h5 = joinpath(@__DIR__, "gal_epsilon_scan_results.h5")
isfile(out_h5) && rm(out_h5)
h5open(out_h5, "w") do f
    f["pc"] = [r[1] for r in results]
    for key in ("gal_m2", "stride_m2", "cajump_m2", "gal_m3", "stride_m3", "cajump_m3", "dW_total")
        vals = [get(r[2], key, NaN+NaN*im) for r in results]
        if eltype(vals) <: Complex
            f["$(key)_re"] = real.(vals); f["$(key)_im"] = imag.(vals)
        else
            f[key] = Float64.(vals)
        end
    end
end
open(joinpath(@__DIR__, "gal_epsilon_scan_results.csv"), "w") do io
    println(io, "pc,gal_m2,stride_m2,cajump_m2,gal_m3,stride_m3,cajump_m3,dW_total")
    for (pc, r) in results
        @printf(io, "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n", pc,
            real(get(r,"gal_m2",NaN)), real(get(r,"stride_m2",NaN)), real(get(r,"cajump_m2",NaN)),
            real(get(r,"gal_m3",NaN)), real(get(r,"stride_m3",NaN)), real(get(r,"cajump_m3",NaN)),
            get(r,"dW_total",NaN))
    end
end
@info "Saved gal_epsilon_scan_results.h5 and .csv to $(@__DIR__) ($(length(results)) points)"
println("DONE")
