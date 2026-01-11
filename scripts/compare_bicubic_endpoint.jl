#!/usr/bin/env julia
"""
Compare bicubic splines with ENDPOINT INCLUDED (matching equilibrium code).

In DirectEquilibrium.jl, theta_nodes = range(0.0, 1.0; length=mtheta+1)
which INCLUDES both endpoints (0 and 1).

The Fortran code expects this and forces fs[end] = fs[0].
The Julia code expects NO endpoint for periodic data.

This test checks what happens when endpoint-inclusive data is passed to both.
"""

using Pkg
Pkg.activate(".")

using LinearAlgebra
using Statistics
using Printf

include("../src/Splines/Splines.jl")
using .SplinesMod
const Spl = SplinesMod

println("="^70)
println("Test: Endpoint-Inclusive Periodic Data (matching equilibrium)")
println("="^70)

# Periodic function in y: f(x,y) = (1 + x) * cos(2π*y)
# Period is 1 in normalized coordinates (actual theta = 2π*y)
function test_function(x, y)
    return (1 + x) * cos(2π * y)
end

function test_fy(x, y)
    return -(1 + x) * 2π * sin(2π * y)
end

function test_fyy(x, y)
    return -(1 + x) * (2π)^2 * cos(2π * y)
end

function run_comparison(nx::Int, ny::Int)
    println("\n" * "-"^60)
    println("Grid: $(nx)×$(ny) points (endpoint INCLUDED)")
    println("-"^60)

    # Create grid MATCHING equilibrium: includes endpoint
    xs = collect(range(0.0, 1.0; length=nx))
    ys_with_endpoint = collect(range(0.0, 1.0; length=ny))  # Includes 1.0 endpoint

    # Create data with endpoint included
    fs = zeros(nx, ny, 1)
    for i in 1:nx, j in 1:ny
        fs[i, j, 1] = test_function(xs[i], ys_with_endpoint[j])
    end

    println("  y range: [$(ys_with_endpoint[1]), $(ys_with_endpoint[end])]")
    println("  y includes endpoint: yes (same as equilibrium code)")
    println("  fs[1, 1, 1] = $(fs[1, 1, 1]), fs[1, end, 1] = $(fs[1, end, 1])")
    println("  (should be equal for periodic function)")

    # Fortran Hermite with endpoint-inclusive data
    bf = Spl.BicubicSpline(xs, ys_with_endpoint, fs;
        bctypex="extrap", bctypey="periodic")

    # Julia B-spline: needs endpoint REMOVED for proper periodic handling
    ys_no_endpoint = ys_with_endpoint[1:(end-1)]  # Remove endpoint
    fs_no_endpoint = fs[:, 1:(end-1), :]          # Remove last theta slice
    bw = Spl.BicubicWrapper(xs, ys_no_endpoint, fs_no_endpoint; periodic_y=true)

    # Also test what happens if we incorrectly pass endpoint-inclusive data to Julia
    bw_wrong = Spl.BicubicWrapper(xs, ys_with_endpoint, fs; periodic_y=true)

    # Evaluation points
    n_eval = 20
    x_eval = collect(range(0.1, 0.9; length=n_eval))
    y_eval = collect(range(0.05, 0.95; length=n_eval))

    # Storage
    fy_fortran = zeros(n_eval, n_eval)
    fy_julia_correct = zeros(n_eval, n_eval)
    fy_julia_wrong = zeros(n_eval, n_eval)
    fy_exact = zeros(n_eval, n_eval)

    fyy_fortran = zeros(n_eval, n_eval)
    fyy_julia_correct = zeros(n_eval, n_eval)
    fyy_julia_wrong = zeros(n_eval, n_eval)
    fyy_exact = zeros(n_eval, n_eval)

    for (i, x) in enumerate(x_eval)
        for (j, y) in enumerate(y_eval)
            fy_exact[i, j] = test_fy(x, y)
            fyy_exact[i, j] = test_fyy(x, y)

            # Fortran
            _, _, fy, _, _, fyy = Spl.bicube_deriv2!(bf, x, y)
            fy_fortran[i, j] = fy[1]
            fyy_fortran[i, j] = fyy[1]

            # Julia with correct (no endpoint) data
            _, _, fy, _, _, fyy = Spl.deriv2!(bw, x, y)
            fy_julia_correct[i, j] = fy[1]
            fyy_julia_correct[i, j] = fyy[1]

            # Julia with wrong (endpoint included) data
            _, _, fy, _, _, fyy = Spl.deriv2!(bw_wrong, x, y)
            fy_julia_wrong[i, j] = fy[1]
            fyy_julia_wrong[i, j] = fyy[1]
        end
    end

    scale_fy = maximum(abs.(fy_exact))
    scale_fyy = maximum(abs.(fyy_exact))

    fy_err_fortran = maximum(abs.(fy_fortran .- fy_exact)) / scale_fy
    fy_err_julia_correct = maximum(abs.(fy_julia_correct .- fy_exact)) / scale_fy
    fy_err_julia_wrong = maximum(abs.(fy_julia_wrong .- fy_exact)) / scale_fy

    fyy_err_fortran = maximum(abs.(fyy_fortran .- fyy_exact)) / scale_fyy
    fyy_err_julia_correct = maximum(abs.(fyy_julia_correct .- fyy_exact)) / scale_fyy
    fyy_err_julia_wrong = maximum(abs.(fyy_julia_wrong .- fyy_exact)) / scale_fyy

    println("\nRelative errors for fy (first y-derivative):")
    @printf("  Fortran (endpoint included):        %.2e\n", fy_err_fortran)
    @printf("  Julia CORRECT (endpoint removed):   %.2e\n", fy_err_julia_correct)
    @printf("  Julia WRONG (endpoint included):    %.2e\n", fy_err_julia_wrong)

    println("\nRelative errors for fyy (second y-derivative):")
    @printf("  Fortran (endpoint included):        %.2e\n", fyy_err_fortran)
    @printf("  Julia CORRECT (endpoint removed):   %.2e\n", fyy_err_julia_correct)
    @printf("  Julia WRONG (endpoint included):    %.2e\n", fyy_err_julia_wrong)

    # Direct comparison Fortran vs Julia (correct)
    diff_fy = maximum(abs.(fy_fortran .- fy_julia_correct)) / scale_fy
    diff_fyy = maximum(abs.(fyy_fortran .- fyy_julia_correct)) / scale_fyy
    println("\nDifference Fortran vs Julia-correct:")
    @printf("  fy:  %.2e\n", diff_fy)
    @printf("  fyy: %.2e\n", diff_fyy)

    return Dict(
        "fy_fortran" => fy_err_fortran,
        "fy_julia_correct" => fy_err_julia_correct,
        "fy_julia_wrong" => fy_err_julia_wrong,
        "fyy_fortran" => fyy_err_fortran,
        "fyy_julia_correct" => fyy_err_julia_correct,
        "fyy_julia_wrong" => fyy_err_julia_wrong
    )
end

run_comparison(21, 33)
run_comparison(41, 65)
run_comparison(65, 129)  # Typical equilibrium size

println("\n" * "="^70)
println("CONCLUSION")
println("="^70)
println("""
The equilibrium code (DirectEquilibrium.jl) currently passes endpoint-inclusive
data to BicubicSpline (Fortran) which handles it correctly.

If we want to use BicubicWrapper (Julia) for rzphi, we need to either:
1. Pass data WITHOUT the endpoint to BicubicWrapper
2. Modify the equilibrium code to not include the endpoint
3. Modify BicubicWrapper to handle endpoint-inclusive periodic data

Currently, eqfun uses BicubicWrapper with endpoint-inclusive data, which may
be causing subtle errors in derivatives.
""")
