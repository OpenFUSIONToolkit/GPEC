#!/usr/bin/env julia
#
# Plot median wall time vs thread count from equilibrium_threads_sweep.jl CSV.
#
# Usage (from JPEC repo root):
#   julia --project=. benchmarks/plot_equilibrium_threads.jl [results.csv] [output.png]

using DelimitedFiles
using Plots

function load_thread_results(path::String)
    mat = readdlm(path, ',')
    nrow = size(mat, 1)
    nrow >= 2 || error("expected header + ≥1 data row in $path")
    r0 = tryparse(Float64, string(mat[1, 1])) === nothing ? 2 : 1
    r0 <= nrow || error("no numeric rows in $path")
    body = mat[r0:end, :]
    nd = size(body, 1)
    nt = Int[round(Int, body[i, 1]) for i in 1:nd]
    med = Float64[body[i, 2] for i in 1:nd]
    tmin = Float64[body[i, 3] for i in 1:nd]
    tmax = Float64[body[i, 4] for i in 1:nd]
    return nt, med, tmin, tmax
end

function main()
    root = dirname(dirname(abspath(@__FILE__)))
    csv = get(ARGS, 1, joinpath(root, "benchmarks", "equilibrium_threads_results.csv"))
    png_out = get(ARGS, 2, joinpath(root, "benchmarks", "equilibrium_threads_bench.png"))

    isfile(csv) || error("CSV not found: $csv (run equilibrium_threads_sweep.jl first)")

    nt, med, tmin, tmax = load_thread_results(csv)

    ylow = med .- tmin
    yhigh = tmax .- med

    scatter(
        nt,
        med;
        yerror = (ylow, yhigh),
        marker = :circle,
        markersize = 5,
        label = "median ± sample min/max",
        xlabel = "Threads (-t N)",
        ylabel = "Time (s)",
        title = "Direct equilibrium_solver (BenchmarkTools @benchmark, evals=1)",
        legend = :topright,
    )
    plot!(nt, med; linewidth = 2, label = "", color = 1)

    mkpath(dirname(png_out))
    savefig(png_out)
    @info "Saved figure to $png_out"
end

main()
