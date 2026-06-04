# Stage-1 validation (issue #251): compare the BVP/MIRK fundamental-matrix path
# against the production parallel-FM path on the Solovev example (no singular
# surfaces in the integration domain). Success = et[1]/ep[1]/ev[1] match.
#
# Usage:
#   julia --project=. benchmarks/bvp_stage1_solovev.jl [example_dir]
# Default example_dir = examples/Solovev_ideal_example.
# Outputs a comparison table; writes nothing permanent (uses temp run dirs).

using GeneralizedPerturbedEquilibrium
using HDF5
using TOML
using Printf

const EX_DIR = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "..", "examples", "Solovev_ideal_example")

"Copy an example dir to a fresh temp dir and patch its [ForceFreeStates] flags."
function prepare_case(srcdir::AbstractString, ffs_overrides::Dict)
    dst = mktempdir()
    for f in readdir(srcdir)
        cp(joinpath(srcdir, f), joinpath(dst, f); force=true)
    end
    cfg = TOML.parsefile(joinpath(dst, "gpec.toml"))
    ffs = get!(cfg, "ForceFreeStates", Dict{String,Any}())
    for (k, v) in ffs_overrides
        ffs[k] = v
    end
    # Skip perturbed equilibrium for Stage-1 speed (et/ep/ev come from free_run!).
    ffs["force_termination"] = true
    open(joinpath(dst, "gpec.toml"), "w") do io
        TOML.print(io, cfg)
    end
    return dst
end

"Run GPEC on dir and return (et1, ep1, ev1, nstep, walltime_s)."
function run_case(dir::AbstractString)
    t = @elapsed GeneralizedPerturbedEquilibrium.main([dir])
    h5 = joinpath(dir, "gpec.h5")
    et1 = ep1 = ev1 = NaN + NaN * im
    nstep = -1
    h5open(h5, "r") do f
        et = read(f["vacuum/et"])
        ep = read(f["vacuum/ep"])
        ev = read(f["vacuum/ev"])
        et1 = ComplexF64(et[1]); ep1 = ComplexF64(ep[1]); ev1 = ComplexF64(ev[1])
        if haskey(f, "integration/nstep")
            nstep = Int(read(f["integration/nstep"]))
        end
    end
    return (; et1, ep1, ev1, nstep, walltime=t)
end

println("Stage-1 BVP validation on: ", abspath(EX_DIR))

# Optional qhigh truncation to isolate a no-crossing domain (Stage-1a core-formulation
# check). Set BVP_QHIGH=1.95 to truncate below the q=2 rational on Solovev.
const QHIGH = haskey(ENV, "BVP_QHIGH") ? parse(Float64, ENV["BVP_QHIGH"]) : nothing
extra = QHIGH === nothing ? Dict{String,Any}() : Dict{String,Any}("qhigh" => QHIGH, "set_psilim_via_dmlim" => false)
QHIGH === nothing || println("Truncating integration domain at qhigh = ", QHIGH, " (no-crossing isolation)")

ref_dir = prepare_case(EX_DIR, merge(Dict{String,Any}("use_bvp" => false, "use_parallel" => true, "populate_dense_xi" => true), extra))
bvp_over = Dict{String,Any}("use_bvp" => true, "use_parallel" => false, "verbose" => true)
haskey(ENV, "BVP_NINT") && (bvp_over["bvp_init_intervals"] = parse(Int, ENV["BVP_NINT"]))
haskey(ENV, "BVP_ADAPT") && (bvp_over["bvp_adaptive"] = ENV["BVP_ADAPT"] == "1")
haskey(ENV, "BVP_TOL") && (bvp_over["eulerlagrange_tolerance"] = parse(Float64, ENV["BVP_TOL"]))
haskey(ENV, "BVP_MAXITERS") && (bvp_over["bvp_maxiters"] = parse(Int, ENV["BVP_MAXITERS"]))
bvp_dir = prepare_case(EX_DIR, merge(bvp_over, extra))

const SKIP_REF = haskey(ENV, "BVP_SKIP_REF")
if SKIP_REF
    println("--- BVP (MIRK6) [reference skipped] ---")
    bvp = run_case(bvp_dir)
    println("\nBVP et[1] = ", bvp.et1, "  nstep=", bvp.nstep, "  walltime=", round(bvp.walltime; digits=1), "s")
    println("BVP_ONLY_DONE")
    exit(0)
end
println("\n--- reference (parallel FM) ---")
ref = run_case(ref_dir)
println("--- BVP (MIRK6) ---")
bvp = run_case(bvp_dir)

@printf "\n%-14s %24s %24s %14s\n" "quantity" "reference(parallel)" "bvp(MIRK6)" "rel.diff"
function row(name, a, b)
    rd = abs(b - a) / max(abs(a), 1e-30)
    @printf "%-14s %+ .10e %+ .10e %14.3e\n" name real(a) real(b) rd
end
row("et[1] Re", ref.et1, bvp.et1)
row("ep[1] Re", ref.ep1, bvp.ep1)
row("ev[1] Re", ref.ev1, bvp.ev1)
@printf "%-14s %24d %24d\n" "nstep(saved)" ref.nstep bvp.nstep
@printf "%-14s %24.2f %24.2f\n" "walltime(s)" ref.walltime bvp.walltime

et_rd = abs(bvp.et1 - ref.et1) / max(abs(ref.et1), 1e-30)
println("\net[1] relative difference = ", et_rd)
println(et_rd < 1e-3 ? "STAGE1_PASS" : "STAGE1_FAIL")
