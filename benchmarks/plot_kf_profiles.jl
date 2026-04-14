#!/usr/bin/env julia
"""
    plot_kf_profiles.jl

Standalone plot of KineticForces dT/dψ and cumulative T(ψ) vs Fortran PENTRC
reference. Reads `benchmarks/kf_<method>_profiles.dat` written by
`benchmark_diiid_kinetic.jl` and produces one PNG per panel.

Use when the in-benchmark savefig fails (GR backend layout bugs, etc.) or
when iterating on plot styling without rerunning the 5-minute benchmark.
"""

using Printf
using Plots

const BENCH_DIR = @__DIR__

function load_profiles(path::String)
    julia_psi = Float64[]; julia_dT = Float64[]; julia_T = Float64[]
    fortran_psi = Float64[]; fortran_dT = Float64[]; fortran_T = Float64[]
    in_fortran = false
    for line in eachline(path)
        s = strip(line)
        isempty(s) && continue
        if startswith(s, "#")
            if occursin("fortran", lowercase(s))
                in_fortran = true
            end
            continue
        end
        parts = parse.(Float64, split(s))
        if in_fortran
            push!(fortran_psi, parts[1]); push!(fortran_dT, parts[2]); push!(fortran_T, parts[3])
        else
            push!(julia_psi, parts[1]); push!(julia_dT, parts[2]); push!(julia_T, parts[3])
        end
    end
    return (; julia_psi, julia_dT, julia_T, fortran_psi, fortran_dT, fortran_T)
end

function plot_method(method::String)
    path = joinpath(BENCH_DIR, "kf_$(method)_profiles.dat")
    isfile(path) || (println(stderr, "  skip $method: $path missing"); return)
    d = load_profiles(path)
    println(stderr, "  $method: Julia=$(length(d.julia_psi)) pts, Fortran=$(length(d.fortran_psi)) pts")

    p1 = plot(; xlabel="ψ_n", ylabel="dT/dψ [N·m]",
                title="$(uppercase(method)): dT/dψ", legend=:topleft,
                left_margin=12Plots.mm, bottom_margin=4Plots.mm, size=(900, 500))
    plot!(p1, d.fortran_psi, d.fortran_dT; label="Fortran", lw=2, color=:black, linestyle=:dash)
    plot!(p1, d.julia_psi,   d.julia_dT;   label="Julia",   lw=1.5, color=:crimson)
    out1 = joinpath(BENCH_DIR, "kf_$(method)_dTdpsi_vs_fortran.png")
    savefig(p1, out1)
    println(stderr, "  → $out1")

    p2 = plot(; xlabel="ψ_n", ylabel="T [N·m]",
                title="$(uppercase(method)): cumulative T", legend=:topleft,
                left_margin=12Plots.mm, bottom_margin=4Plots.mm, size=(900, 500))
    plot!(p2, d.fortran_psi, d.fortran_T; label="Fortran", lw=2, color=:black, linestyle=:dash)
    plot!(p2, d.julia_psi,   d.julia_T;   label="Julia",   lw=1.5, color=:crimson)
    out2 = joinpath(BENCH_DIR, "kf_$(method)_T_vs_fortran.png")
    savefig(p2, out2)
    println(stderr, "  → $out2")
end

for method in ("fgar", "tgar")
    plot_method(method)
end
