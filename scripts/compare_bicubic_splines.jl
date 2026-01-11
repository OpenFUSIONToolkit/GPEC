#!/usr/bin/env julia
"""
Compare Fortran Hermite Bicubic vs Julia B-spline Bicubic Interpolation

This script compares the two bicubic spline implementations to understand
their numerical differences:
- Fortran: Hermite bicubic using sequential 1D spline fitting for derivatives
- Julia: B-spline bicubic using Interpolations.jl

The key algorithmic difference is:
- Hermite: stores (f, fx, fy, fxy) at knots, constructs polynomial per cell
- B-spline: stores control points, uses B-spline basis for evaluation
"""

using Pkg
Pkg.activate(".")

using LinearAlgebra
using Statistics
using Printf

# Load JPEC spline modules
include("../src/Splines/Splines.jl")
using .SplinesMod
const Spl = SplinesMod

println("="^70)
println("Bicubic Spline Comparison: Fortran Hermite vs Julia B-spline")
println("="^70)

# Test function and its analytical derivatives
# f(x,y) = sin(2πx) * cos(πy)
# fx = 2π*cos(2πx) * cos(πy)
# fy = -π*sin(2πx) * sin(πy)
# fxy = -2π²*cos(2πx) * sin(πy)
# fxx = -4π²*sin(2πx) * cos(πy)
# fyy = -π²*sin(2πx) * cos(πy)

function test_function(x, y)
    return sin(2π * x) * cos(π * y)
end

function test_fx(x, y)
    return 2π * cos(2π * x) * cos(π * y)
end

function test_fy(x, y)
    return -π * sin(2π * x) * sin(π * y)
end

function test_fxx(x, y)
    return -4π^2 * sin(2π * x) * cos(π * y)
end

function test_fxy(x, y)
    return -2π^2 * cos(2π * x) * sin(π * y)
end

function test_fyy(x, y)
    return -π^2 * sin(2π * x) * cos(π * y)
end

function run_comparison(nx::Int, ny::Int; periodic_y::Bool=false, name::String="")
    println("\n" * "-"^60)
    println("Grid: $(nx)×$(ny), periodic_y=$(periodic_y) $(name)")
    println("-"^60)

    # Create grid (uniform spacing)
    xs = collect(range(0.0, 1.0; length=nx))
    if periodic_y
        # For periodic, don't include the endpoint
        ys = collect(range(0.0, 2.0; length=ny+1)[1:ny])
    else
        ys = collect(range(0.0, 2.0; length=ny))
    end

    # Create data (single quantity)
    fs = zeros(nx, ny, 1)
    for i in 1:nx, j in 1:ny
        fs[i, j, 1] = test_function(xs[i], ys[j])
    end

    # Create splines
    println("\nCreating splines...")

    # Julia B-spline
    bw = Spl.BicubicWrapper(xs, ys, fs; periodic_y=periodic_y)

    # Fortran Hermite (boundary conditions: extrap=3, periodic=2, not-a-knot=4)
    bcx = "extrap"  # 3
    bcy = periodic_y ? "periodic" : "extrap"
    bf = Spl.BicubicSpline(xs, ys, fs; bctypex=bcx, bctypey=bcy)

    # Evaluation points (interior, away from boundaries)
    n_eval = 50
    x_eval = collect(range(xs[2], xs[end-1]; length=n_eval))
    y_eval = collect(range(ys[2], ys[end-1]; length=n_eval))

    # Storage for results
    f_julia = zeros(n_eval, n_eval)
    f_fortran = zeros(n_eval, n_eval)
    f_exact = zeros(n_eval, n_eval)

    fx_julia = zeros(n_eval, n_eval)
    fx_fortran = zeros(n_eval, n_eval)
    fx_exact = zeros(n_eval, n_eval)

    fy_julia = zeros(n_eval, n_eval)
    fy_fortran = zeros(n_eval, n_eval)
    fy_exact = zeros(n_eval, n_eval)

    fxy_julia = zeros(n_eval, n_eval)
    fxy_fortran = zeros(n_eval, n_eval)
    fxy_exact = zeros(n_eval, n_eval)

    fxx_julia = zeros(n_eval, n_eval)
    fxx_fortran = zeros(n_eval, n_eval)
    fxx_exact = zeros(n_eval, n_eval)

    fyy_julia = zeros(n_eval, n_eval)
    fyy_fortran = zeros(n_eval, n_eval)
    fyy_exact = zeros(n_eval, n_eval)

    # Evaluate
    for (i, x) in enumerate(x_eval)
        for (j, y) in enumerate(y_eval)
            # Exact values
            f_exact[i, j] = test_function(x, y)
            fx_exact[i, j] = test_fx(x, y)
            fy_exact[i, j] = test_fy(x, y)
            fxy_exact[i, j] = test_fxy(x, y)
            fxx_exact[i, j] = test_fxx(x, y)
            fyy_exact[i, j] = test_fyy(x, y)

            # Julia B-spline
            f, fx, fy, fxx, fxy, fyy = Spl.deriv2!(bw, x, y)
            f_julia[i, j] = f[1]
            fx_julia[i, j] = fx[1]
            fy_julia[i, j] = fy[1]
            fxx_julia[i, j] = fxx[1]
            fxy_julia[i, j] = fxy[1]
            fyy_julia[i, j] = fyy[1]

            # Fortran Hermite
            f, fx, fy, fxx, fxy, fyy = Spl.bicube_deriv2!(bf, x, y)
            f_fortran[i, j] = f[1]
            fx_fortran[i, j] = fx[1]
            fy_fortran[i, j] = fy[1]
            fxx_fortran[i, j] = fxx[1]
            fxy_fortran[i, j] = fxy[1]
            fyy_fortran[i, j] = fyy[1]
        end
    end

    # Compute errors
    function compute_errors(name, julia_vals, fortran_vals, exact_vals)
        # Error vs exact
        julia_err = abs.(julia_vals .- exact_vals)
        fortran_err = abs.(fortran_vals .- exact_vals)

        # Difference between methods
        diff = abs.(julia_vals .- fortran_vals)

        # Normalize by max absolute value
        scale = maximum(abs.(exact_vals))
        if scale < 1e-10
            scale = 1.0
        end

        println("\n  $name:")
        @printf("    Julia   vs Exact:   max=%.2e, mean=%.2e, rel_max=%.2e\n",
            maximum(julia_err), mean(julia_err), maximum(julia_err)/scale)
        @printf("    Fortran vs Exact:   max=%.2e, mean=%.2e, rel_max=%.2e\n",
            maximum(fortran_err), mean(fortran_err), maximum(fortran_err)/scale)
        @printf("    Julia vs Fortran:   max=%.2e, mean=%.2e, rel_max=%.2e\n",
            maximum(diff), mean(diff), maximum(diff)/scale)

        return (julia_err=maximum(julia_err)/scale,
            fortran_err=maximum(fortran_err)/scale,
            diff=maximum(diff)/scale)
    end

    results = Dict{String,NamedTuple}()
    results["f"] = compute_errors("f (function)", f_julia, f_fortran, f_exact)
    results["fx"] = compute_errors("fx (∂f/∂x)", fx_julia, fx_fortran, fx_exact)
    results["fy"] = compute_errors("fy (∂f/∂y)", fy_julia, fy_fortran, fy_exact)
    results["fxx"] = compute_errors("fxx (∂²f/∂x²)", fxx_julia, fxx_fortran, fxx_exact)
    results["fxy"] = compute_errors("fxy (∂²f/∂x∂y)", fxy_julia, fxy_fortran, fxy_exact)
    results["fyy"] = compute_errors("fyy (∂²f/∂y²)", fyy_julia, fyy_fortran, fyy_exact)

    return results
end

# Run tests at various grid sizes
results_coarse = run_comparison(11, 17; periodic_y=false, name="(coarse)")
results_medium = run_comparison(21, 33; periodic_y=false, name="(medium)")
results_fine = run_comparison(41, 65; periodic_y=false, name="(fine)")

# Test with periodic boundary in y (like theta direction in equilibrium)
results_periodic = run_comparison(21, 32; periodic_y=true, name="(periodic-y)")

println("\n" * "="^70)
println("SUMMARY")
println("="^70)
println("""
The differences arise from fundamentally different algorithms:

1. **Fortran Hermite Bicubic**:
   - Fits 1D cubic splines separately in x and y
   - Stores (f, fx, fy, fxy) at each knot point
   - Constructs polynomial coefficients per cell from corner values
   - Uses Hermite interpolation formula

2. **Julia B-spline Bicubic** (Interpolations.jl):
   - Uses tensor-product B-splines
   - Stores B-spline control points
   - Evaluates basis functions for interpolation
   - Derivatives from basis function derivatives

Key observations:
- Both are valid bicubic interpolation methods
- They produce slightly different results, especially for derivatives
- The difference increases with derivative order
- Cross-derivative (fxy) typically shows largest discrepancy
""")

# Check if Fortran is consistently more accurate
println("\nAccuracy comparison (relative max error vs exact):")
println("-"^50)
for (name, r) in [("Coarse", results_coarse), ("Medium", results_medium), ("Fine", results_fine)]
    println("\n$name grid:")
    for deriv in ["f", "fx", "fy", "fxx", "fxy", "fyy"]
        julia_err = r[deriv].julia_err
        fortran_err = r[deriv].fortran_err
        winner = julia_err < fortran_err ? "Julia" : "Fortran"
        ratio = fortran_err > 1e-15 ? julia_err / fortran_err : Inf
        @printf("  %3s: Julia=%.2e, Fortran=%.2e, ratio=%.2f (%s wins)\n",
            deriv, julia_err, fortran_err, ratio, winner)
    end
end
