#!/usr/bin/env julia
"""
Benchmark for Solovev equilibrium with BSplineKit.jl splines

This benchmark measures:
1. Equilibrium setup time
2. Spline evaluation performance
3. Derivative calculation performance
4. Full DCON run time
"""

using JPEC
using BenchmarkTools
using Printf
using BSplineKit

println("="^70)
println("BSplineKit.jl Spline Performance Benchmark - Solovev Example")
println("="^70)
println()

# Setup equilibrium
println("Setting up Solovev equilibrium...")
equil_path = "test/test_data/regression_solovev_ideal_example/equil.toml"

@time equil = JPEC.Equilibrium.setup_equilibrium(equil_path)
println()

# Get number of grid points
npsi = length(equil.psi_grid)
println("Grid points: $npsi")
println("q-range: [$(equil.params.qmin), $(equil.params.qmax)]")
println()

# Benchmark spline evaluation
println("Benchmarking spline evaluation...")
psi_test = 0.5  # Mid-radius

println("\n1. Function evaluation at ψ = $psi_test:")
b_eval_q = @benchmark $equil.q_spline($psi_test)
println("   q_spline:      $(BenchmarkTools.prettytime(median(b_eval_q.times)))")

b_eval_F = @benchmark $equil.F_spline($psi_test)
println("   F_spline:      $(BenchmarkTools.prettytime(median(b_eval_F.times)))")

b_eval_P = @benchmark $equil.P_spline($psi_test)
println("   P_spline:      $(BenchmarkTools.prettytime(median(b_eval_P.times)))")

b_eval_dV = @benchmark $equil.dVdpsi_spline($psi_test)
println("   dVdpsi_spline: $(BenchmarkTools.prettytime(median(b_eval_dV.times)))")

# Benchmark derivatives
println("\n2. First derivative evaluation:")
D1 = BSplineKit.Derivative(1)
b_deriv1_q = @benchmark ($D1 * $equil.q_spline)($psi_test)
println("   q' (BSplineKit): $(BenchmarkTools.prettytime(median(b_deriv1_q.times)))")

println("\n3. Second derivative evaluation:")
D2 = BSplineKit.Derivative(2)
b_deriv2_q = @benchmark ($D2 * $equil.q_spline)($psi_test)
println("   q'' (BSplineKit): $(BenchmarkTools.prettytime(median(b_deriv2_q.times)))")

println("\n4. Third derivative evaluation:")
D3 = BSplineKit.Derivative(3)
b_deriv3_q = @benchmark ($D3 * $equil.q_spline)($psi_test)
println("   q''' (BSplineKit): $(BenchmarkTools.prettytime(median(b_deriv3_q.times)))")

# Benchmark array evaluation (common in Equilibrium module)
println("\n5. Vectorized evaluation on all grid points:")
b_array_eval = @benchmark [$(equil.q_spline)(psi) for psi in $(equil.psi_grid)]
println("   q_spline.(psi_grid): $(BenchmarkTools.prettytime(median(b_array_eval.times)))")
println("   Per-point average:   $(BenchmarkTools.prettytime(median(b_array_eval.times) ÷ npsi))")

# Memory allocations
println("\n6. Memory allocations:")
println("   Function eval:   $(b_eval_q.allocs) allocations")
println("   1st derivative:  $(b_deriv1_q.allocs) allocations")
println("   2nd derivative:  $(b_deriv2_q.allocs) allocations")
println("   3rd derivative:  $(b_deriv3_q.allocs) allocations")
println("   Array eval:      $(b_array_eval.allocs) allocations")

# Full DCON run benchmark
println("\n7. Full DCON equilibrium + stability run:")
println("   Running full Solovev ideal example...")
dcon_path = dirname(equil_path)
@time JPEC.DCON.Main(dcon_path)

println()
println("="^70)
println("Benchmark complete!")
println("="^70)
