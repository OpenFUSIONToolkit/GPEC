#!/usr/bin/env julia
# convergence_amr_resolution.jl — Phase 2.8 study.
#
# For a given staged equilibrium, sweep the AMR initial-grid resolution
# `nre0 = nim0 ∈ {25, 50, 100, 200}` and intermediate refinement counts
# `pass ∈ 0..max_passes(nre0)`, recording γ at every (nre0, pass) tuple
# for each of three SLAYER configurations on the same equilibrium:
#
#   mm=2  coupling=false  → q=2 uncoupled (msing_use=1)
#   mm=3  coupling=false  → q=3 uncoupled (msing_use=1)
#   mm=*  coupling=true   → both surfaces coupled (msing_use=msing)
#
# Implementation: ONE AMR scan per (case, nre0). The new
# `snapshot_callback` kwarg of `amr_scan` captures the cell list at the
# end of each pass; we then call `find_growth_rates` on each snapshot to
# extract the most-unstable Q_root → γ. This is much cheaper than re-
# running AMR for every (nre0, pass) combination.
#
# Output: a tab-separated `convergence_amr.tsv` with one row per
# (case, nre0, pass) tuple.
#
# Usage:
#   julia --project=. profiling/convergence_amr_resolution.jl \
#       --case-dir <staged equilibrium dir> \
#       [--out /tmp/convergence_amr.tsv] \
#       [--q-hw-khz 25.0]                    # default 25 kHz
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using GeneralizedPerturbedEquilibrium
using GeneralizedPerturbedEquilibrium.Equilibrium
using GeneralizedPerturbedEquilibrium.ForceFreeStates
using GeneralizedPerturbedEquilibrium.Tearing.InnerLayer:
    KineticProfiles, build_slayer_inputs, SLAYERModel
using GeneralizedPerturbedEquilibrium.Tearing.Dispersion:
    amr_scan, AMRResult, AMRCell,
    multi_surface_coupling, surface_coupling, find_growth_rates
using GeneralizedPerturbedEquilibrium.Tearing.InnerLayer.SLAYER: SLAYERParameters
using HDF5, Printf, Base.Threads, LinearAlgebra, Statistics

BLAS.set_num_threads(1)
@info "BLAS threads=1; Julia threads=$(Threads.nthreads())"

# ---------------------------------------------------------------------
# Geqdsk header parser (RMAXIS, BCENTR — same as DIIID benchmark)
# ---------------------------------------------------------------------
function _parse_g_line(line::AbstractString, n::Int=5, width::Int=16)
    [parse(Float64, strip(line[(k-1)*width+1 : min(k*width, length(line))]))
     for k in 1:n]
end
function geqdsk_header(path::AbstractString)
    lines = readlines(path)
    l3 = _parse_g_line(lines[3])
    return (rmaxis=l3[1], zmaxis=l3[2], simag=l3[3], sibry=l3[4], bcentr=l3[5])
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

args = ARGS
case_dir = get_arg(args, "case-dir") :: AbstractString
out_path = get_arg(args, "out", "/tmp/convergence_amr.tsv")
Q_HW_kHz = get_arg(args, "q-hw-khz", 25.0; parser=x->parse(Float64, x))

julia_dir = joinpath(case_dir, "julia")
isfile(joinpath(julia_dir, "gpec.toml")) ||
    error("Missing gpec.toml in $julia_dir")

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

# ---------------------------------------------------------------------
# Equilibrium + Force-Free States ONCE
# ---------------------------------------------------------------------
@info "Running GPEC main()"
t0 = time()
result = GeneralizedPerturbedEquilibrium.main([julia_dir])
@info @sprintf("main() in %.2fs", time()-t0)
equil = result.equil
intr  = result.intr
ForceFreeStates.resist_eval_all!(intr, equil)

msing = length(intr.sing)
q_values = [s.q for s in intr.sing]
m_values = [s.m[1] for s in intr.sing]
@info "msing=$msing  q=$q_values  m=$m_values"

# Read kinetic profiles
psi_kin, ne_kin, te_kin, ti_kin, wexb_kin = read_gpeckf(gpeckf_path)
zeros_kin = zeros(Float64, length(psi_kin))
profiles = KineticProfiles(
    psi=psi_kin, n_e=ne_kin, T_e=te_kin, T_i=ti_kin, omega=wexb_kin,
    omega_e=zeros_kin, omega_i=zeros_kin)

hdr = geqdsk_header(geqdsk_path)
bt = abs(hdr.bcentr); R0_geq = hdr.rmaxis

# Build SLAYER inputs for ALL surfaces; per-case slicing happens below.
slayer_params_all = build_slayer_inputs(equil, intr.sing, profiles;
                                         bt=bt, R0=R0_geq, rs_method=:fsa,
                                         mu_i=2.0, zeff=2.0,
                                         chi_perp=0.2, chi_tor=0.2,
                                         dc_type=:rfitzp)
dp_full = ComplexF64.(intr.delta_prime_matrix)

# ---------------------------------------------------------------------
# Case configurations on the same equilibrium
# ---------------------------------------------------------------------
struct CaseConfig
    name::String
    coupling::Bool
    mm::Int           # used only when coupling=false (selects which surface)
end

all_cases = [
    CaseConfig("uncoupled_2over1", false, 2),
    CaseConfig("uncoupled_3over1", false, 3),
    CaseConfig("coupled",          true,  0),
]
cases = haskey(ENV, "RICCATI_CONV_SMOKE") ? all_cases[1:1] : all_cases
@info "Cases to run: $([c.name for c in cases])"

# ---------------------------------------------------------------------
# Resolution sweep
# ---------------------------------------------------------------------
# (nre0, max_passes) per the user's spec.
all_sweep = [(25, 8), (50, 7), (100, 6), (200, 5)]
sweep = haskey(ENV, "RICCATI_CONV_SMOKE") ? [(25, 2)] : all_sweep
@info "Sweep configs: $sweep"
max_cells = 1_000_000

# ---------------------------------------------------------------------
# Build mc(Q) for a case + run AMR with snapshots → collect γ per pass
# ---------------------------------------------------------------------
function _build_mc_and_qhw(case::CaseConfig)
    # Pick keep_range based on case
    if case.coupling
        keep_range = 1:msing
    else
        idx = findfirst(==(case.mm), m_values)
        idx === nothing && error("uncoupled mm=$(case.mm) not in $m_values")
        keep_range = idx:idx
    end
    keep = collect(keep_range)
    msing_use = length(keep_range)

    sings_kept = [intr.sing[k] for k in keep]
    sp_kept = [slayer_params_all[k] for k in keep]
    dp_kept = ComplexF64.(dp_full[keep, keep])

    # Build per-surface couplings (matches Tearing.Runner pattern)
    model = SLAYERModel(variant=:fitzpatrick)
    scs = [surface_coupling(model, sp_kept[k], dp_kept[k, k]; dc=sp_kept[k].dc_tmp)
            for k in 1:msing_use]
    mc = multi_surface_coupling(scs, dp_kept; ref_idx=1, msing_max=msing_use)

    # Q box conversion: ±Q_HW_kHz → ±Q_HW (dimensionless)
    tau_k_ref = sp_kept[1].tauk
    kHz_per_Q = 1.0 / (tau_k_ref * 1e3)
    Q_HW = Q_HW_kHz / kHz_per_Q
    return (mc=mc, sp_kept=sp_kept, dp_kept=dp_kept, msing_use=msing_use,
            tau_k_ref=tau_k_ref, kHz_per_Q=kHz_per_Q, Q_HW=Q_HW)
end

# Light-weight snapshot of (cells, cache) → AMRResult
function _flatten_to_amr(cells, cache)
    n = length(cache)
    Q = Vector{ComplexF64}(undef, n)
    Δ = Vector{ComplexF64}(undef, n)
    for (k, (q, d)) in enumerate(cache); Q[k] = q; Δ[k] = d; end
    return AMRResult(copy(cells), Q, Δ)
end

# Extract best (most-unstable) γ from a single snapshot.
# Returns (γ_kHz, ω_kHz, n_valid_roots, n_poles, n_cells)
function _gamma_from_snapshot(snap::AMRResult, tauk::Float64, kHz_per_Q::Float64)
    # Adaptive pole threshold = |mean(Δ)| over finite entries, matching
    # SLAYERControl's pole_threshold_adaptive=true production setting.
    finite_Δ = filter(z -> isfinite(z) && abs(z) < 1e30, snap.Δ)
    pole_thr = isempty(finite_Δ) ? 10.0 : abs(mean(finite_Δ))

    extraction = find_growth_rates(snap, tauk;
                                    pole_threshold=pole_thr,
                                    filter_above_poles=true,
                                    filter_outside_re=true)
    n_valid = length(extraction.valid_roots)
    n_poles_ = length(extraction.poles)
    bq = extraction.Q_root
    if !isfinite(bq)
        return (γ_kHz=NaN, ω_kHz=NaN, n_valid_roots=n_valid, n_poles=n_poles_,
                n_cells=length(snap.cells))
    end
    return (γ_kHz=extraction.gamma_Hz / 1e3,    # find_growth_rates already divided by tauk
            ω_kHz=extraction.omega_Hz / 1e3,
            n_valid_roots=n_valid,
            n_poles=n_poles_,
            n_cells=length(snap.cells))
end

# ---------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------
rows = NamedTuple[]

for case in cases
    @info "=== Case: $(case.name) ==="
    cinfo = _build_mc_and_qhw(case)
    @info @sprintf("  msing_use=%d  τ_k_ref=%.4e  Q box ±%.4f (= ±%.1f kHz)",
                   cinfo.msing_use, cinfo.tau_k_ref, cinfo.Q_HW, Q_HW_kHz)

    for (nre0, max_passes) in sweep
        @info @sprintf("  --- nre0=%d × max_passes=%d ---", nre0, max_passes)
        flush(stderr)
        snapshots = AMRResult[]
        t0 = time()
        amr_scan(cinfo.mc,
                 (-cinfo.Q_HW, +cinfo.Q_HW),
                 (-cinfo.Q_HW, +cinfo.Q_HW);
                 nre0=nre0, nim0=nre0, passes=max_passes,
                 max_cells=max_cells,
                 max_cells_action=:warn_truncate,
                 parallel=Threads.nthreads() > 1,
                 snapshot_callback=(p, cells, cache) -> begin
                     push!(snapshots, _flatten_to_amr(cells, cache))
                     @info "      pass=$p cells=$(length(cells)) cache=$(length(cache))"
                     flush(stderr)
                 end)
        wall = time() - t0
        @info @sprintf("    AMR done in %.1fs, captured %d snapshots", wall, length(snapshots))
        flush(stderr)

        for (pass_idx, snap) in enumerate(snapshots)
            pass = pass_idx - 1   # snapshot index 1 corresponds to pass 0
            t_extract = time()
            r = _gamma_from_snapshot(snap, cinfo.tau_k_ref, cinfo.kHz_per_Q)
            t_extract = time() - t_extract
            @info @sprintf("      extract pass=%d in %.2fs: γ=%+.5e nv=%d np=%d",
                           pass, t_extract, r.γ_kHz, r.n_valid_roots, r.n_poles)
            flush(stderr)
            push!(rows, (case=case.name, nre0=nre0, pass=pass,
                         n_cells=r.n_cells, γ_kHz=r.γ_kHz, ω_kHz=r.ω_kHz,
                         n_valid_roots=r.n_valid_roots, n_poles=r.n_poles,
                         amr_wall_s=wall))
        end
    end
end

# ---------------------------------------------------------------------
# Save TSV
# ---------------------------------------------------------------------
open(out_path, "w") do io
    println(io, "# convergence_amr_resolution.jl results")
    println(io, "# case-dir = $case_dir")
    println(io, "# Q_HW_kHz = $Q_HW_kHz")
    println(io, "# max_cells = $max_cells (max_cells_action=:warn_truncate)")
    println(io, "# JULIA_NUM_THREADS = $(Threads.nthreads())")
    println(io, "")
    cols = ["case", "nre0", "pass", "n_cells", "gamma_kHz", "omega_kHz",
            "n_valid_roots", "n_poles", "amr_wall_s"]
    println(io, join(cols, '\t'))
    for r in rows
        println(io, join([r.case, r.nre0, r.pass, r.n_cells,
                          r.γ_kHz, r.ω_kHz, r.n_valid_roots, r.n_poles,
                          r.amr_wall_s], '\t'))
    end
end
@info "Wrote $out_path  ($(length(rows)) rows)"

# ---------------------------------------------------------------------
# Quick text summary: γ at max_pass for each (case, nre0)
# ---------------------------------------------------------------------
println("\n  γ converged @ max_pass (kHz):")
println(@sprintf("  %-20s  %8s  %8s  %8s  %8s",
                 "case", "nre0=25", "nre0=50", "nre0=100", "nre0=200"))
for case in cases
    γs = [first([r.γ_kHz for r in rows if r.case == case.name && r.nre0 == n && r.pass == p])
          for (n, p) in sweep]
    print(@sprintf("  %-20s ", case.name))
    for γ in γs
        print(@sprintf(" %+8.5f", γ))
    end
    println()
end
