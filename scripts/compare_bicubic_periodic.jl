#!/usr/bin/env julia
"""
Compare bicubic splines with proper periodic boundary conditions.

Uses a function that is truly periodic in y (theta direction),
matching the equilibrium usage where theta ∈ [0, 2π) is periodic.
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
println("Bicubic Spline Comparison: Properly Periodic Test Case")
println("="^70)

# Test function that is periodic in y with period 2π:
# f(x,y) = (1 + x) * cos(y)
# This is non-periodic in x (extrap), periodic in y with period 2π
function test_function(x, y)
    return (1 + x) * cos(y)
end

function test_fx(x, y)
    return cos(y)
end

function test_fy(x, y)
    return -(1 + x) * sin(y)
end

function test_fxx(x, y)
    return 0.0
end

function test_fxy(x, y)
    return -sin(y)
end

function test_fyy(x, y)
    return -(1 + x) * cos(y)
end

function run_comparison(nx::Int, ny::Int)
    println("\n" * "-"^60)
    println("Grid: $(nx)×$(ny), periodic in y (theta)")
    println("-"^60)

    # Create grid
    # x (psi) direction: [0, 1] with extrap boundary
    # y (theta) direction: [0, 2π) periodic - don't include endpoint!
    xs = collect(range(0.0, 1.0; length=nx))
    ys = collect(range(0.0, 2π; length=ny+1)[1:ny])  # Exclude 2π endpoint

    # Create data
    fs = zeros(nx, ny, 1)
    for i in 1:nx, j in 1:ny
        fs[i, j, 1] = test_function(xs[i], ys[j])
    end

    println("  x range: [$(xs[1]), $(xs[end])]")
    println("  y range: [$(ys[1]), $(ys[end])], period = 2π = $(2π)")
    println("  dy = $(ys[2] - ys[1])")

    # Create splines
    # Julia B-spline with periodic in y
    bw = Spl.BicubicWrapper(xs, ys, fs; periodic_y=true)

    # Fortran Hermite with extrap in x, periodic in y
    bf = Spl.BicubicSpline(xs, ys, fs; bctypex="extrap", bctypey="periodic")

    # Evaluation points - interior in x, full range in y (including near 2π)
    n_eval = 30
    x_eval = collect(range(xs[2], xs[end-1]; length=n_eval))
    # Test y across the full period, including near the wrap point
    y_eval = collect(range(0.1, 2π - 0.1; length=n_eval))

    # Storage
    f_julia, f_fortran, f_exact = zeros(n_eval, n_eval), zeros(n_eval, n_eval), zeros(n_eval, n_eval)
    fx_julia, fx_fortran, fx_exact = zeros(n_eval, n_eval), zeros(n_eval, n_eval), zeros(n_eval, n_eval)
    fy_julia, fy_fortran, fy_exact = zeros(n_eval, n_eval), zeros(n_eval, n_eval), zeros(n_eval, n_eval)
    fxy_julia, fxy_fortran, fxy_exact = zeros(n_eval, n_eval), zeros(n_eval, n_eval), zeros(n_eval, n_eval)
    fyy_julia, fyy_fortran, fyy_exact = zeros(n_eval, n_eval), zeros(n_eval, n_eval), zeros(n_eval, n_eval)

    for (i, x) in enumerate(x_eval)
        for (j, y) in enumerate(y_eval)
            # Exact
            f_exact[i, j] = test_function(x, y)
            fx_exact[i, j] = test_fx(x, y)
            fy_exact[i, j] = test_fy(x, y)
            fxy_exact[i, j] = test_fxy(x, y)
            fyy_exact[i, j] = test_fyy(x, y)

            # Julia
            f, fx, fy, fxx, fxy, fyy = Spl.deriv2!(bw, x, y)
            f_julia[i, j] = f[1]
            fx_julia[i, j] = fx[1]
            fy_julia[i, j] = fy[1]
            fxy_julia[i, j] = fxy[1]
            fyy_julia[i, j] = fyy[1]

            # Fortran
            f, fx, fy, fxx, fxy, fyy = Spl.bicube_deriv2!(bf, x, y)
            f_fortran[i, j] = f[1]
            fx_fortran[i, j] = fx[1]
            fy_fortran[i, j] = fy[1]
            fxy_fortran[i, j] = fxy[1]
            fyy_fortran[i, j] = fyy[1]
        end
    end

    function report(name, julia, fortran, exact)
        scale = maximum(abs.(exact))
        if scale < 1e-10
            scale = 1.0
        end
        julia_err = maximum(abs.(julia .- exact)) / scale
        fortran_err = maximum(abs.(fortran .- exact)) / scale
        diff = maximum(abs.(julia .- fortran)) / scale

        winner = julia_err < fortran_err ? "Julia" : "Fortran"
        @printf("  %4s: Julia=%.2e, Fortran=%.2e, diff=%.2e (%s)\n",
            name, julia_err, fortran_err, diff, winner)

        return (julia_err=julia_err, fortran_err=fortran_err, diff=diff)
    end

    println("\nRelative errors (vs exact analytical):")
    results = Dict{String,NamedTuple}()
    results["f"] = report("f", f_julia, f_fortran, f_exact)
    results["fx"] = report("fx", fx_julia, fx_fortran, fx_exact)
    results["fy"] = report("fy", fy_julia, fy_fortran, fy_exact)
    results["fxy"] = report("fxy", fxy_julia, fxy_fortran, fxy_exact)
    results["fyy"] = report("fyy", fyy_julia, fyy_fortran, fyy_exact)

    return results
end

# Test at different resolutions
results = Dict{String,Any}()
results["coarse"] = run_comparison(11, 16)
results["medium"] = run_comparison(21, 32)
results["fine"] = run_comparison(41, 64)

# Also test with the typical equilibrium grid size
println("\n" * "="^60)
println("Typical equilibrium-like grid (65 psi x 128 theta)")
println("="^60)
results["equil"] = run_comparison(65, 128)

println("\n" * "="^70)
println("ANALYSIS")
println("="^70)
println("""
For truly periodic functions in y (like cos(θ) in theta):

Key question: Which method is more accurate for the equilibrium use case?
- rzphi contains (R, Z, Φ) as functions of (ψ, θ)
- θ is periodic with period 2π
- ψ is non-periodic with extrap boundary

If Fortran consistently wins, we should keep using Fortran bicubic for rzphi.
If Julia wins or is comparable, we need to investigate further why the
full stability benchmark shows 6% discrepancy with Julia-only splines.
""")

println("\nConvergence Summary:")
for (grid, r) in [("Coarse", results["coarse"]), ("Medium", results["medium"]),
    ("Fine", results["fine"]), ("Equil", results["equil"])]
    println("\n$grid:")
    for d in ["f", "fx", "fy", "fxy", "fyy"]
        winner = r[d].julia_err < r[d].fortran_err ? "J" : "F"
        ratio = r[d].fortran_err > 1e-15 ? r[d].julia_err / r[d].fortran_err : Inf
        @printf("  %3s: J/F ratio=%.2f (%s wins)\n", d, ratio, winner)
    end
end
