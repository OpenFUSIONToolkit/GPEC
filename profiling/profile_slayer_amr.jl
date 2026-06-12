#!/usr/bin/env julia
# profile_slayer_amr.jl — Phase 0 profiling harness for SLAYER coupled-AMR.
#
# Runs the SLAYER step ONLY (assumes a `gpec.h5` already exists from a prior
# `GeneralizedPerturbedEquilibrium.main()` run on the case dir, OR runs main()
# fresh if missing). Captures:
#
#   1. wall-time breakdown of each phase
#   2. allocation count + GC time
#   3. CPU profile (Profile.@profile) → flat report saved to stdout
#   4. Allocation profile (Profile.Allocs) → allocation hotspots saved to stdout
#
# Use a SHORT case (DIII-D coupled_rfitzp ~5-15 min, or one TJ βₚ run) so the
# profile is tractable. Defaults to the DIII-D coupled_rfitzp staged dir.
#
# Usage (from julia_GPEC repo root):
#   julia --project=. profiling/profile_slayer_amr.jl \
#       --case-dir /path/to/results/coupled_rfitzp \
#       --out /tmp/profile_slayer.txt
#
# The case dir must contain `julia/gpec.toml`, `julia/slayer.in`, the staged
# geqdsk, and `julia/tmp.gpeckf`. Re-using an existing scan dir avoids
# restaging.
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using GeneralizedPerturbedEquilibrium
using GeneralizedPerturbedEquilibrium.Equilibrium
using GeneralizedPerturbedEquilibrium.ForceFreeStates
using GeneralizedPerturbedEquilibrium.Tearing.Runner
using GeneralizedPerturbedEquilibrium.Tearing.InnerLayer:
    KineticProfiles, build_slayer_inputs
using HDF5, Printf, Base.Threads, LinearAlgebra, TOML, Profile

BLAS.set_num_threads(1)
@info "BLAS threads=1; Julia threads=$(Threads.nthreads())"

# -------------------------------------------------------------------------
# Self-contained namelist + geqdsk-header parsing (no external driver needed).

function _parse_g_line(line::AbstractString, n::Int=5, width::Int=16)
    [parse(Float64, strip(line[(k-1)*width+1 : min(k*width, length(line))]))
     for k in 1:n]
end
function geqdsk_header(path::AbstractString)
    lines = readlines(path)
    l3 = _parse_g_line(lines[3])
    return (rmaxis=l3[1], zmaxis=l3[2], simag=l3[3], sibry=l3[4], bcentr=l3[5])
end

function parse_namelist(path::AbstractString, keys::Vector{Symbol})
    out = Dict{Symbol,Any}()
    keys_set = Set(lowercase.(string.(keys)))
    for raw in readlines(path)
        s = split(raw, '!'; limit=2)[1]
        occursin('=', s) || continue
        k, v = split(s, '='; limit=2)
        kname = lowercase(strip(k))
        kname in keys_set || continue
        rhs = strip(replace(v, "," => " "))
        rhs = replace(rhs, "\"" => "", "'" => "")
        toks = split(rhs)
        isempty(toks) && continue
        parsed = Any[]
        for t in toks
            tt = lowercase(t)
            if tt == "t" || tt == ".true." || tt == "true"
                push!(parsed, true)
            elseif tt == "f" || tt == ".false." || tt == "false"
                push!(parsed, false)
            else
                x = tryparse(Float64, t)
                push!(parsed, x === nothing ? t : x)
            end
        end
        out[Symbol(kname)] = length(parsed) == 1 ? parsed[1] : parsed
    end
    return out
end

function read_gpeckf(path::AbstractString)
    psi_v = Float64[]; ne_v = Float64[]; te_v = Float64[]
    ti_v = Float64[]; wexb_v = Float64[]
    for line in eachline(path)
        s = strip(line)
        (isempty(s) || startswith(s, "#")) && continue
        parts = split(s)
        length(parts) < 5 && continue
        tp = tryparse(Float64, parts[1]); tp === nothing && continue
        push!(psi_v, tp)
        push!(ne_v, parse(Float64, parts[3]))
        push!(ti_v, parse(Float64, parts[4]))
        push!(te_v, parse(Float64, parts[5]))
        push!(wexb_v, length(parts) ≥ 6 ? parse(Float64, parts[6]) : 0.0)
    end
    return psi_v, ne_v, te_v, ti_v, wexb_v
end

function get_arg(args, name, default=nothing; parser=identity)
    for (i, a) in enumerate(args)
        a == "--$name" && return parser(args[i+1])
    end
    return default
end

# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
args = ARGS
case_dir = get_arg(args, "case-dir") :: AbstractString
out_path = get_arg(args, "out", "/tmp/profile_slayer.txt") :: AbstractString
warm     = get_arg(args, "warm", "true") == "true"
profile_amr_only = get_arg(args, "profile-amr-only", "true") == "true"

julia_dir = joinpath(case_dir, "julia")
isfile(joinpath(julia_dir, "gpec.toml")) ||
    error("Missing gpec.toml in $julia_dir")
isfile(joinpath(julia_dir, "slayer.in")) ||
    error("Missing slayer.in in $julia_dir")

function _find_staged_geqdsk(dir::AbstractString)
    for f in readdir(dir; join=true)
        base = basename(f)
        base in ("gpec.toml", "tmp.gpeckf", "slayer.in", "forcing.dat") && continue
        startswith(base, ".") && continue
        return f
    end
    return ""
end
geqdsk_path = _find_staged_geqdsk(julia_dir)
isempty(geqdsk_path) && error("No geqdsk in $julia_dir")
gpeckf_path = joinpath(julia_dir, "tmp.gpeckf")

# ---- Equilibrium phase ----
@info "[profile] Equilibrium + Force-Free States via main()"
t_main = @elapsed result = GeneralizedPerturbedEquilibrium.main([julia_dir])
equil = result.equil
intr  = result.intr
ForceFreeStates.resist_eval_all!(intr, equil)
@info @sprintf("[profile] main() in %.2fs", t_main)

msing = length(intr.sing)
q_values = [s.q for s in intr.sing]
m_values = [s.m[1] for s in intr.sing]

# ---- Read case selectors ----
nl = parse_namelist(joinpath(julia_dir, "slayer.in"),
                     [:mu_i, :zeff, :chi_p_prof, :chi_t_prof,
                      :mm, :coupling_flag, :dc_type, :msing_max])
mu_i_val   = Float64(get(nl, :mu_i, 2.0))
zeff_val   = Float64(get(nl, :zeff, 2.0))
chi_p_arr  = get(nl, :chi_p_prof, [0.2])
chi_t_arr  = get(nl, :chi_t_prof, [0.2])
chi_p_val  = Float64(chi_p_arr isa AbstractVector ? first(chi_p_arr) : chi_p_arr)
chi_t_val  = Float64(chi_t_arr isa AbstractVector ? first(chi_t_arr) : chi_t_arr)
mm_target  = Int(get(nl, :mm, 2))
coupling   = Bool(get(nl, :coupling_flag, true))
dc_type_s  = String(get(nl, :dc_type, "none"))
dc_type_sym = Symbol(lowercase(dc_type_s))
msing_max  = Int(get(nl, :msing_max, msing))

keep_range = if coupling
    1:min(msing, msing_max)
else
    idx = findfirst(==(mm_target), m_values)
    idx === nothing && error("uncoupled mm=$mm_target not in $m_values")
    idx:idx
end
keep = collect(keep_range)
msing_use = length(keep_range)
@info "[profile] msing_use=$msing_use  q=$(q_values[keep])  m=$(m_values[keep])  coupling=$coupling  dc=$dc_type_s"

# ---- Build SLAYER inputs ----
psi_kin, ne_kin, te_kin, ti_kin, wexb_kin = read_gpeckf(gpeckf_path)
zeros_kin = zeros(Float64, length(psi_kin))
profiles = KineticProfiles(
    psi=psi_kin, n_e=ne_kin, T_e=te_kin, T_i=ti_kin, omega=wexb_kin,
    omega_e=zeros_kin, omega_i=zeros_kin)
hdr = geqdsk_header(geqdsk_path)
bt = abs(hdr.bcentr); R0_geq = hdr.rmaxis

sings_kept = [intr.sing[k] for k in keep]
slayer_params = build_slayer_inputs(equil, sings_kept, profiles;
                                     bt=bt, R0=R0_geq, rs_method=:fsa,
                                     mu_i=mu_i_val, zeff=zeff_val,
                                     chi_perp=chi_p_val, chi_tor=chi_t_val,
                                     dc_type=dc_type_sym)
dp_full = intr.delta_prime_matrix
dp_matrix = ComplexF64.(dp_full[keep, keep])
tau_k_ref = slayer_params[1].tauk
kHz_per_Q = 1.0 / (tau_k_ref * 1e3)

# Q box: read from baseline (Q_HW_kHz attr in betascan_result.h5 if present),
# else use a sensible default based on the case.
function _read_q_hw_kHz(case_dir::AbstractString)
    for fname in ("betascan_result.h5", "diiid_result.h5")
        p = joinpath(case_dir, fname)
        isfile(p) || continue
        h5open(p, "r") do f
            haskey(attrs(f), "Q_HW_kHz") && return Float64(attrs(f)["Q_HW_kHz"])
            return nothing
        end
    end
    return nothing
end
q_hw_khz_baseline = _read_q_hw_kHz(case_dir)
Q_HW_kHz = q_hw_khz_baseline === nothing ? 50.0 : q_hw_khz_baseline
Q_HW = Q_HW_kHz / kHz_per_Q
@info @sprintf("[profile] τ_k_ref=%.4e  kHz/Q=%.4e  Q_HW=±%.3f (=±%.1f kHz)",
               tau_k_ref, kHz_per_Q, Q_HW, Q_HW_kHz)

# ---- SLAYERControl ----
# `--passes` lets us shrink AMR work for a fast first-pass profile (passes=2
# gives ~30s SLAYER calls; production scan uses passes=5 coupled / 4 uncoupled).
default_passes = coupling ? 5 : 4
amr_passes = Int(get_arg(args, "passes", default_passes; parser=x->parse(Int, x)))
control = SLAYERControl(;
    enabled=true, inner_model=:slayer_fitzpatrick, scan_mode=:amr,
    coupling_mode = coupling ? :coupled : :uncoupled,
    dc_type=dc_type_sym, msing_max=msing_use, bt=bt,
    mu_i=mu_i_val, zeff=zeff_val, chi_perp=chi_p_val, chi_tor=chi_t_val,
    Q_re_range=(-Q_HW, +Q_HW), Q_im_range=(-Q_HW, +Q_HW),
    nre=100, nim=100, amr_passes=amr_passes,
    pole_threshold_adaptive=true, filter_above_poles=true,
    filter_outside_re=true, store_scan=true)

# ---- Warm-up run (JIT compile) ----
if warm
    @info "[profile] Warm-up SLAYER run (JIT)"
    t_warm = @elapsed run_slayer_from_inputs(slayer_params, dp_matrix, control)
    @info @sprintf("[profile] warm-up SLAYER: %.2fs", t_warm)
end

# ---- Timed run + memory stats ----
@info "[profile] Timed SLAYER run + GC stats"
GC.gc()
stats = @timed slayer_result = run_slayer_from_inputs(slayer_params, dp_matrix, control)
@info @sprintf("[profile] SLAYER  time=%.2fs  alloc=%.2f GB  GC=%.2fs (%.1f%%)",
               stats.time, stats.bytes / 1e9, stats.gctime,
               100 * stats.gctime / max(stats.time, eps()))

# Best root sanity check
if !isempty(slayer_result.Q_root)
    bq = slayer_result.Q_root[1]
    γ = imag(bq) * kHz_per_Q
    ω = real(bq) * kHz_per_Q
    @info @sprintf("[profile] best root: γ=%+.4f kHz  ω=%+.4f kHz", γ, ω)
end

# ---- CPU profile of one more run ----
@info "[profile] CPU profile"
Profile.clear()
Profile.init(n=10_000_000, delay=0.001)
Profile.@profile run_slayer_from_inputs(slayer_params, dp_matrix, control)
@info "[profile] writing flat CPU profile to $out_path"
open(out_path, "w") do io
    println(io, "# CPU profile of run_slayer_from_inputs")
    println(io, "# case-dir=$case_dir")
    println(io, "# coupling=$coupling  dc_type=$dc_type_s  msing_use=$msing_use  passes=$amr_passes")
    println(io, "# JULIA_NUM_THREADS=$(Threads.nthreads())  BLAS=$(BLAS.get_num_threads())")
    println(io, "# Wall=$(round(stats.time, digits=2))s  Alloc=$(round(stats.bytes/1e9, digits=2)) GB")
    println(io, "")
    Profile.print(io; format=:flat, sortedby=:count, mincount=200)
end

# ---- Allocation profile ----
@info "[profile] Allocation profile"
alloc_out = replace(out_path, r"\.txt$" => "_allocs.txt")
Profile.Allocs.clear()
Profile.Allocs.@profile sample_rate=0.01 run_slayer_from_inputs(slayer_params, dp_matrix, control)
results = Profile.Allocs.fetch()
@info @sprintf("[profile] allocations sampled: %d (sample_rate=0.01)", length(results.allocs))
open(alloc_out, "w") do io
    println(io, "# Allocation profile of run_slayer_from_inputs (sample_rate=0.01)")
    # Aggregate allocation count + bytes by call site
    counts = Dict{String,Tuple{Int,Int}}()
    for a in results.allocs
        for sf in a.stacktrace
            key = "$(sf.func) at $(sf.file):$(sf.line)"
            n, b = get(counts, key, (0, 0))
            counts[key] = (n + 1, b + a.size)
            break  # innermost frame only
        end
    end
    sorted = sort(collect(counts), by=x->-x[2][2])  # sort by total bytes
    println(io, @sprintf("%-12s %-12s  %s", "count", "bytes", "site"))
    for (site, (n, b)) in sorted[1:min(50, length(sorted))]
        println(io, @sprintf("%-12d %-12d  %s", n, b, site))
    end
end
@info "[profile] flat profile → $out_path"
@info "[profile] alloc profile → $alloc_out"
@info "[profile] DONE"
