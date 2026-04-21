# Invoked by gpec_thread_sweep.jl as:
#   julia -t N --project=<JPEC> benchmarks/gpec_thread_sweep_runner.jl <case_dir> [n_runs]
#
# Runs GPEC main n_runs times in the SAME process (pay JIT cost once).
# Prints one line per run to stdout:
#   GPEC_TIMING equil=<s> ffs=<s> pe=<s> total=<s>
# Run #1 includes JIT compilation; the sweep script discards it.
#
# BLAS threads pinned to 1 so Julia thread count is the only variable being swept.
using LinearAlgebra
BLAS.set_num_threads(1)

using Pkg
const ROOT = dirname(dirname(abspath(@__FILE__)))
Pkg.activate(ROOT)

using GeneralizedPerturbedEquilibrium
using Logging

length(ARGS) >= 1 || error("usage: julia -t N --project=. gpec_thread_sweep_runner.jl <case_dir> [n_runs]")
case_dir = ARGS[1]
n_runs = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 3

@info "gpec_thread_sweep_runner" julia_threads=Threads.nthreads() blas_threads=BLAS.get_num_threads() case_dir n_runs

for r in 1:n_runs
    buf = IOBuffer()
    with_logger(SimpleLogger(buf, Logging.Info)) do
        GeneralizedPerturbedEquilibrium.main([case_dir])
    end
    log_text = String(take!(buf))

    grab(rx) = let m = match(rx, log_text)
        m === nothing ? NaN : parse(Float64, m.captures[1])
    end
    equil_s = grab(r"Equilibrium construction completed in ([0-9.]+) s")
    ffs_s   = grab(r"Force-Free States completed in ([0-9.]+) s")
    pe_s    = grab(r"Perturbed Equilibrium completed in ([0-9.]+) s")
    total_s = grab(r"GPEC completed successfully in ([0-9.]+) s")

    println("GPEC_TIMING equil=$(equil_s) ffs=$(ffs_s) pe=$(pe_s) total=$(total_s)")
    flush(stdout)
end
