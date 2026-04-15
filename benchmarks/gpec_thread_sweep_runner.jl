# Invoked by gpec_thread_sweep.jl as:
#   julia -t N --project=<JPEC> benchmarks/gpec_thread_sweep_runner.jl <case_dir>
#
# BLAS threads default to 1 so Julia thread count is the main knob.
using LinearAlgebra
BLAS.set_num_threads(1)

using Pkg
const ROOT = dirname(dirname(abspath(@__FILE__)))
Pkg.activate(ROOT)

using GeneralizedPerturbedEquilibrium

length(ARGS) >= 1 || error("usage: julia -t N --project=. benchmarks/gpec_thread_sweep_runner.jl <case_dir>")
GeneralizedPerturbedEquilibrium.main([ARGS[1]]);
