#!/usr/bin/env julia
"""
    compare_kinetic_matrices.jl

Compare the six kinetic matrices (Ak, Bk, Ck, Dk, Ek, Hk per Logan 2015
thesis Eqs 7.30–7.35) produced by Julia's `compute_kinetic_matrices_at_psi!`
against the Fortran PENTRC reference at three ψ surfaces.

# How to generate the Fortran reference (one-time, ~seconds)

The Fortran matrix dump lives inside pentrc's `wxyz_flag` block
(`pentrc/pentrc.F90:98-129`), **not** `fkmm_flag` as the original plan
text assumed (fkmm only writes scalar flux profiles).

1. In `~/Code/gpec/docs/examples/DIIID_kinetic_example/pentrc.in`:
       &PENT_OUTPUT
           wxyz_flag = .true.        ! enables the matrix dump
           fgar_flag = .false.       ! disable other methods for speed
           tgar_flag = .false.
           pgar_flag = .false.
           clar_flag = .false.
           rlar_flag = .false.
           fcgl_flag = .false.
           fkmm_flag = .false.
           ftmm_flag = .false.
           fwmm_flag = .false.
           psi_out   = 0.1, 0.5, 0.9
2. Run `pentrc` in that directory.
3. It writes `pentrc_tgar_elmat_n1.out` containing one (m_1, m_2,
   real/imag A..H) block per ψ. **Only ℓ=0 is written** — the Fortran
   wxyz loop `do l=0,0` (pentrc.F90:115) has a `!! should be all`
   TODO that hasn't been fixed. The Julia dumper records all
   ℓ ∈ {-1, 0, +1}, but this script compares only ℓ=0.

The modified `pentrc.in` and the resulting `.out` are NOT committed to
JPEC_ode — they live in the Fortran tree.

# What this script does

1. If `benchmarks/fortran_kinetic_matrices_n1.h5` doesn't exist, parse
   the Fortran ASCII dump into HDF5.
2. Load the Julia HDF5 dump (run `dump_julia_kinetic_matrices.jl` first).
3. For each (ψ, matrix) pair, report `rel_frob`, `max_abs_re_diff`,
   `max_abs_im_diff`. Threshold for "match": rel_frob ≤ 1e-3.
4. Emit heatmap PNGs comparing Re(Fortran), Re(Julia), |diff| and the
   same for Im.
5. Print a runtime + agreement summary table.

Usage:
    julia --project=. benchmarks/compare_kinetic_matrices.jl [fortran_dir]
"""

using Printf
using HDF5
using Plots
using LinearAlgebra

const DEFAULT_FORTRAN_DIR =
    expanduser("~/Code/gpec/docs/examples/DIIID_kinetic_example")
const JULIA_H5         = joinpath(@__DIR__, "julia_kinetic_matrices_n1.h5")
const JULIA_H5_QUADGK  = joinpath(@__DIR__, "julia_kinetic_matrices_n1_quadgk.h5")
const FORTRAN_H5       = joinpath(@__DIR__, "fortran_kinetic_matrices_n1.h5")
const FORTRAN_ASCII_NAME = "pentrc_tgar_elmat_n1.out"
const MATRIX_NAMES = ("Ak", "Bk", "Ck", "Dk", "Ek", "Hk")
const REL_FROB_THRESHOLD = 1e-3

"""
    parse_fortran_ascii(path) → Dict{Float64, NamedTuple}

Parse the Fortran `pentrc_tgar_elmat_n<n>.out` file. Returns a dict
keyed by ψ; each value is `(; mfac, matrices)` with
`matrices::NTuple{6,Matrix{ComplexF64}}` in order (Ak, Bk, Ck, Dk,
Ek, Hk).

File layout (one block per ψ):
    PERTURBED EQUILIBRIUM NONAMBIPOLAR TRANSPORT CODE:
    Kinetic additions to the ideal Euler-Lagrange matrices
        n =    1 l =    0
     psi =        1.00000000e-01
     m_1 m_2         real(A_k)   imag(A_k)   ...   imag(H_k)
     <mpert^2 rows of data>
     psi =        ...
     (next block)
"""
function parse_fortran_ascii(path::String)
    isfile(path) || error("Fortran matrix dump not found: $path\n" *
                          "Run pentrc with wxyz_flag=.true. (see script header).")
    out = Dict{Float64, NamedTuple}()
    current_psi = nothing
    m1_list = Int[]
    m2_list = Int[]
    rows = Vector{NTuple{12, Float64}}()

    function flush_block!()
        (current_psi === nothing || isempty(rows)) && return
        ms1 = sort!(unique(m1_list))
        ms2 = sort!(unique(m2_list))
        @assert ms1 == ms2 "Fortran ascii m_1 and m_2 indices disagree"
        mfac = ms1
        mpert = length(mfac)
        mats = ntuple(_ -> zeros(ComplexF64, mpert, mpert), 6)
        m_to_idx = Dict(m => i for (i, m) in enumerate(mfac))
        for (m1, m2, row) in zip(m1_list, m2_list, rows)
            i = m_to_idx[m1]
            j = m_to_idx[m2]
            for q in 1:6
                mats[q][i, j] = complex(row[2q - 1], row[2q])
            end
        end
        out[current_psi] = (; mfac, matrices=mats)
        empty!(m1_list); empty!(m2_list); empty!(rows)
    end

    for line in eachline(path)
        s = strip(line)
        isempty(s) && continue
        if occursin("psi =", s)
            flush_block!()
            # "psi =      1.00000000e-01"
            tok = split(s)
            current_psi = parse(Float64, tok[end])
        elseif startswith(s, "m_1")
            continue  # header
        elseif startswith(s, "n =") || startswith(s, "PERTURBED") ||
               startswith(s, "Kinetic")
            continue  # meta
        else
            toks = split(s)
            length(toks) == 14 || continue
            try
                m1 = parse(Int, toks[1])
                m2 = parse(Int, toks[2])
                vals = ntuple(k -> parse(Float64, toks[2 + k]), 12)
                push!(m1_list, m1)
                push!(m2_list, m2)
                push!(rows, vals)
            catch
                continue
            end
        end
    end
    flush_block!()
    isempty(out) && error("No ψ blocks found in $path")
    return out
end

"""
    fortran_ascii_to_hdf5(ascii_path, h5_path)

One-time conversion of the Fortran ASCII dump to HDF5. Skips if
`h5_path` is newer than `ascii_path`.
"""
function fortran_ascii_to_hdf5(ascii_path::String, h5_path::String;
                                fortran_wall_s::Union{Nothing,Float64}=nothing)
    if isfile(h5_path) && mtime(h5_path) > mtime(ascii_path)
        return h5_path
    end
    parsed = parse_fortran_ascii(ascii_path)
    h5open(h5_path, "w") do h5
        attributes(h5)["source_ascii"] = ascii_path
        attributes(h5)["ell"] = 0
        attributes(h5)["nn"] = 1
        if fortran_wall_s !== nothing
            attributes(h5)["fortran_wall_s"] = fortran_wall_s
        end
        first_psi = first(keys(parsed))
        write(h5, "mfac", parsed[first_psi].mfac)
        for (psi, block) in parsed
            g = create_group(h5, @sprintf("psi=%.3f", psi))
            attributes(g)["psi"] = psi
            for (k, name) in enumerate(MATRIX_NAMES)
                write(g, name, block.matrices[k])
            end
        end
    end
    @printf(stderr, "  wrote Fortran HDF5: %s  (%d surfaces)\n",
            h5_path, length(parsed))
    return h5_path
end

"""
    load_matrix_set(h5_path, psi) → (; mfac, matrices)

Load the six matrices at surface ψ. Both Fortran and Julia HDF5 files
use the same layout: `psi=<v>/{Ak, Bk, Ck, Dk, Ek, Hk}` at ℓ=0.
"""
function load_matrix_set(h5_path::String, psi::Float64)
    h5open(h5_path, "r") do h5
        mfac = Vector{Int}(read(h5, "mfac"))
        psi_group_name = @sprintf("psi=%.3f", psi)
        haskey(h5, psi_group_name) || error("$psi_group_name not in $h5_path")
        g = h5[psi_group_name]
        mats = ntuple(k -> Matrix{ComplexF64}(read(g, MATRIX_NAMES[k])), 6)
        return (; mfac, matrices=mats)
    end
end

"""
    compare_matrices(fmat, jmat) → (; rel_frob, max_abs_re, max_abs_im)

Frobenius-norm relative error and per-component sup norms.
"""
function compare_matrices(fmat::Matrix{ComplexF64}, jmat::Matrix{ComplexF64})
    denom = norm(fmat)
    rel_frob = denom > 0 ? norm(jmat - fmat) / denom : norm(jmat - fmat)
    max_abs_re = maximum(abs.(real.(jmat - fmat)))
    max_abs_im = maximum(abs.(imag.(jmat - fmat)))
    return (; rel_frob, max_abs_re, max_abs_im)
end

"""
    heatmap_panel(fmat, jmat, title_prefix, outpath)

Save a 6-panel PNG: Re(Fortran), Re(Julia), |Re diff| and matching Im
row, arranged 2 rows × 3 cols. Shared colour limits per row.
"""
function heatmap_panel(fmat, jmat, title_prefix::String, outpath::String)
    re_f = real.(fmat); re_j = real.(jmat); re_d = abs.(re_f .- re_j)
    im_f = imag.(fmat); im_j = imag.(jmat); im_d = abs.(im_f .- im_j)
    re_clim = (minimum([minimum(re_f), minimum(re_j)]),
               maximum([maximum(re_f), maximum(re_j)]))
    im_clim = (minimum([minimum(im_f), minimum(im_j)]),
               maximum([maximum(im_f), maximum(im_j)]))
    p1 = heatmap(re_f; clims=re_clim, title="Re Fortran", color=:balance)
    p2 = heatmap(re_j; clims=re_clim, title="Re Julia",   color=:balance)
    p3 = heatmap(re_d; title="|Re diff|", color=:viridis)
    p4 = heatmap(im_f; clims=im_clim, title="Im Fortran", color=:balance)
    p5 = heatmap(im_j; clims=im_clim, title="Im Julia",   color=:balance)
    p6 = heatmap(im_d; title="|Im diff|", color=:viridis)
    plt = plot(p1, p2, p3, p4, p5, p6;
        layout=(2, 3), size=(1200, 700),
        plot_title=title_prefix,
        left_margin=8Plots.mm, bottom_margin=4Plots.mm)
    savefig(plt, outpath)
    @printf(stderr, "  %s\n", abspath(outpath))
    return abspath(outpath)
end

"""
    read_julia_metadata(h5_path) → (; total_wall, setup_wall, per_psi, label)

Shared helper for extracting the runtime attributes stored by
`dump_julia_kinetic_matrices.jl`. `label` is "ODE" / "QuadGK" / "Julia"
based on the `pitch_integrator` HDF5 attribute (falls back to "Julia").
"""
function read_julia_metadata(h5_path::String)
    h5open(h5_path, "r") do h5
        tw = haskey(attributes(h5), "total_wall_s") ?
             read(attributes(h5)["total_wall_s"]) : NaN
        sw = haskey(attributes(h5), "setup_wall_s") ?
             read(attributes(h5)["setup_wall_s"]) : NaN
        label = haskey(attributes(h5), "pitch_integrator") ?
                uppercase(read(attributes(h5)["pitch_integrator"])) : "Julia"
        per = Dict{Float64, Float64}()
        for name in keys(h5)
            if startswith(name, "psi=")
                g = h5[name]
                psi = read(attributes(g)["psi"])
                per[psi] = haskey(attributes(g), "wall_time_s") ?
                           read(attributes(g)["wall_time_s"]) : NaN
            end
        end
        return (; total_wall=tw, setup_wall=sw, per_psi=per, label)
    end
end

"""
    main(fortran_dir)

Entry point. Converts the Fortran ASCII (if needed), loads Fortran and
Julia HDF5 datasets (ODE plus optional QuadGK), prints comparison table,
writes heatmap PNGs.
"""
function main(fortran_dir::String=DEFAULT_FORTRAN_DIR)
    println(stderr, "=" ^ 70)
    println(stderr, "  Per-surface kinetic matrix comparison: Julia vs Fortran")
    println(stderr, "  Matrix set (Logan 2015 Eqs 7.30-7.35): ",
            join(MATRIX_NAMES, ", "))
    println(stderr, "  Comparison at ℓ=0 only (Fortran wxyz writes only ℓ=0)")
    println(stderr, "=" ^ 70)

    isfile(JULIA_H5) || error("Julia dump not found: $JULIA_H5\n" *
        "Run: julia --project=. benchmarks/dump_julia_kinetic_matrices.jl")

    fortran_ascii = joinpath(fortran_dir, FORTRAN_ASCII_NAME)
    if !isfile(fortran_ascii) && !isfile(FORTRAN_H5)
        error("Fortran dump not available.\n" *
              "  Expected: $fortran_ascii\n" *
              "  To generate: set wxyz_flag=.true. in pentrc.in, set\n" *
              "               psi_out=0.1,0.5,0.9, and run pentrc in\n" *
              "               $fortran_dir.")
    end
    if isfile(fortran_ascii)
        fortran_ascii_to_hdf5(fortran_ascii, FORTRAN_H5)
    end

    # Load the primary Julia (ODE) dump metadata + optional QuadGK dump.
    ode_meta = read_julia_metadata(JULIA_H5)
    has_quadgk = isfile(JULIA_H5_QUADGK)
    quadgk_meta = has_quadgk ? read_julia_metadata(JULIA_H5_QUADGK) : nothing
    if has_quadgk
        println(stderr, "  Three-way comparison: Fortran / Julia ODE / Julia QuadGK")
    else
        println(stderr, "  Two-way comparison: Fortran / Julia ODE")
        println(stderr, "  (generate QuadGK dump with `dump_julia_kinetic_matrices.jl <dir> ",
                "$(basename(JULIA_H5_QUADGK)) quadgk`)")
    end

    fortran_wall = h5open(FORTRAN_H5, "r") do h5
        haskey(attributes(h5), "fortran_wall_s") ?
            read(attributes(h5)["fortran_wall_s"]) : NaN
    end

    # Identify ψ values present in the Fortran reference.
    fortran_psis = h5open(FORTRAN_H5, "r") do h5
        sort([read(attributes(h5[name])["psi"]) for name in keys(h5)
              if startswith(name, "psi=")])
    end

    println(stderr, "")
    println(stderr, "--- Per-matrix agreement (ℓ=0, rel_frob vs Fortran) ---")
    if has_quadgk
        @printf(stderr, "  %6s  %4s  %12s  %12s  %s\n",
                "ψ", "mat", "rel_frob ODE", "rel_frob QGK", "status")
        println(stderr, "  " * "-" ^ 60)
    else
        @printf(stderr, "  %6s  %4s  %10s  %10s  %10s  %s\n",
                "ψ", "mat", "rel_frob", "max|ΔRe|", "max|ΔIm|", "status")
        println(stderr, "  " * "-" ^ 62)
    end

    all_pass = true
    outdir = @__DIR__
    for psi in fortran_psis
        fset = load_matrix_set(FORTRAN_H5, psi)
        jset_ode = load_matrix_set(JULIA_H5, psi)
        @assert fset.mfac == jset_ode.mfac "mfac mismatch at ψ=$psi"
        jset_qgk = has_quadgk ? load_matrix_set(JULIA_H5_QUADGK, psi) : nothing
        if has_quadgk
            @assert fset.mfac == jset_qgk.mfac "mfac mismatch QuadGK at ψ=$psi"
        end
        for (k, name) in enumerate(MATRIX_NAMES)
            fmat = fset.matrices[k]
            cmp_ode = compare_matrices(fmat, jset_ode.matrices[k])
            if has_quadgk
                cmp_qgk = compare_matrices(fmat, jset_qgk.matrices[k])
                pass = cmp_ode.rel_frob <= REL_FROB_THRESHOLD &&
                       cmp_qgk.rel_frob <= REL_FROB_THRESHOLD
                all_pass &= pass
                status = pass ? "OK" : "FAIL"
                @printf(stderr, "  %6.3f  %4s  %12.2e  %12.2e  %s\n",
                        psi, name, cmp_ode.rel_frob, cmp_qgk.rel_frob, status)
            else
                pass = cmp_ode.rel_frob <= REL_FROB_THRESHOLD
                all_pass &= pass
                status = pass ? "OK" : "FAIL"
                @printf(stderr, "  %6.3f  %4s  %10.2e  %10.2e  %10.2e  %s\n",
                        psi, name, cmp_ode.rel_frob, cmp_ode.max_abs_re,
                        cmp_ode.max_abs_im, status)
            end
            heatmap_panel(fmat, jset_ode.matrices[k],
                @sprintf("%s @ ψ=%.3f (ℓ=0) — Fortran vs Julia ODE", name, psi),
                joinpath(outdir,
                    @sprintf("kfmm_%s_psi%.3f_ode.png", name, psi)))
            if has_quadgk
                heatmap_panel(fmat, jset_qgk.matrices[k],
                    @sprintf("%s @ ψ=%.3f (ℓ=0) — Fortran vs Julia QuadGK", name, psi),
                    joinpath(outdir,
                        @sprintf("kfmm_%s_psi%.3f_quadgk.png", name, psi)))
            end
        end
    end

    println(stderr, "")
    println(stderr, "--- Runtime summary ---")
    if has_quadgk
        @printf(stderr, "  %-18s %12s %12s %12s\n",
                "path", "Fortran", "Julia ODE", "Julia QuadGK")
        @printf(stderr, "  %-18s %12s %12.2f %12.2f\n",
                "setup (s)", "--", ode_meta.setup_wall, quadgk_meta.setup_wall)
        @printf(stderr, "  %-18s %12.2f %12.2f %12.2f\n",
                "total matrix (s)",
                isnan(fortran_wall) ? 0.0 : fortran_wall,
                ode_meta.total_wall, quadgk_meta.total_wall)
        for psi in sort(collect(keys(ode_meta.per_psi)))
            @printf(stderr, "  ψ=%.2f (s)           %12s %12.2f %12.2f\n",
                    psi, "--", ode_meta.per_psi[psi],
                    get(quadgk_meta.per_psi, psi, NaN))
        end
    else
        @printf(stderr, "  %-18s %12s %12s\n", "path", "Fortran", "Julia (ODE)")
        @printf(stderr, "  %-18s %12s %12.2f\n",
                "setup (s)", "--", ode_meta.setup_wall)
        @printf(stderr, "  %-18s %12.2f %12.2f\n",
                "total matrix (s)",
                isnan(fortran_wall) ? 0.0 : fortran_wall,
                ode_meta.total_wall)
        for psi in sort(collect(keys(ode_meta.per_psi)))
            @printf(stderr, "  ψ=%.2f (s)           %12s %12.2f\n",
                    psi, "--", ode_meta.per_psi[psi])
        end
    end

    println(stderr, "")
    println(stderr, all_pass ? "  ALL MATCHES WITHIN rel_frob ≤ $REL_FROB_THRESHOLD" :
                               "  DISAGREEMENT DETECTED — see table above")
    println(stderr, "=" ^ 70)

    return all_pass
end

if abspath(PROGRAM_FILE) == @__FILE__
    fortran_dir = length(ARGS) >= 1 ? ARGS[1] : DEFAULT_FORTRAN_DIR
    exit(main(fortran_dir) ? 0 : 1)
end
