# Thread scaling for `equilibrium_solver` (BenchmarkTools + plot):
#
#   1) Sweep threads 1…20 (each row is a separate `julia -t N` process):
#        julia --project=. benchmarks/equilibrium_threads_sweep.jl
#        julia --project=. benchmarks/equilibrium_threads_sweep.jl path/to/out.csv 20
#
#   2) Plot CSV from (1):
#        julia --project=. benchmarks/plot_equilibrium_threads.jl
#        julia --project=. benchmarks/plot_equilibrium_threads.jl path/to/out.csv plot.png
#
# Optional env (runone subprocess):
#   GPEC_EQUIL_BENCH_TOML  — gpec.toml path (default: benchmarks/DIIID_ideal_example/gpec.toml)
#   GPEC_EQUIL_BENCH_SAMPLES — BenchmarkTools samples per thread count (default: 10)
#
# Single-process REPL check (same thread count as the Julia you started):
#   using BenchmarkTools, TOML, GeneralizedPerturbedEquilibrium.Equilibrium as EQ
#   raw = ... # load_prepared_direct_input as in equilibrium_threads_runone.jl
#   @btime EQ.equilibrium_solver($raw; benchmark=false)  # raw must be fresh each call if reused

println(@__FILE__, ": see comments at top for sweep + plot workflow.")
