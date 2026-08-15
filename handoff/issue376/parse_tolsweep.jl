# Parse the eulerlagrange_tolerance sweep (fixed grid, mpsi=512): step counts, EL wall time,
# and the physics observables (et[1], Δ' matrix) that decide whether a looser tolerance is safe.
using HDF5, Printf, LinearAlgebra

# Dir holding run_<tol>/ and tol_<tol>.log; defaults to this script's directory.
const SWEEP_DIR = isempty(ARGS) ? (@__DIR__) : ARGS[1]
const TOLS = ["1e-6", "1e-7", "1e-8", "1e-10", "1e-12"]
const REFTOL = "1e-12"   # tightest point is the reference for physics deviations

function marks(logfile)
    m = Tuple{Float64,String}[]
    for line in eachline(logfile)
        startswith(line, "TSMARK ") || continue
        rest = split(line[8:end], " | "; limit=2)
        length(rest) == 2 || continue
        push!(m, (parse(Float64, rest[1]), String(rest[2])))
    end
    return m
end

function tsafter(m, prefix, i0)
    for i in i0:length(m)
        startswith(m[i][2], prefix) && return m[i][1], i
    end
    return NaN, length(m) + 1
end

function phases(logfile)
    m = marks(logfile)
    _, i0 = tsafter(m, "RUN2_START", 1)
    t = Dict{String,Float64}()
    order = [("equil_start", "  Equilibrium"),
        ("ffs_start", "  Force-Free States"),
        ("el", "Integrating Euler-Lagrange"),
        ("freeb", "Computing free boundary"),
        ("run2_end", "RUN2_END")]
    i = i0
    for (key, prefix) in order
        t[key], i = tsafter(m, prefix, i)
    end
    return (elint=t["freeb"] - t["el"], total=t["run2_end"] - t["equil_start"])
end

read_opt(h5, path) = haskey(h5, path) ? read(h5[path]) : nothing

function h5info(rundir)
    h5open(joinpath(rundir, "gpec.h5"), "r") do h5
        psi = read(h5["integration/psi"])
        return (nstep=sum(read(h5["integration/nstep"])),
            nstep_total=read(h5["integration/nstep_total"]),
            axis_steps=count(<(0.1), psi),
            et1=real(read(h5["FreeBoundaryStability/eigenmode_energies"])[1]),
            dp=read_opt(h5, "singular/delta_prime_matrix"),
            npsi=length(read(h5["splines/profiles/xs"])))
    end
end

# Deviation of a Δ' matrix from the reference: worst per-surface relative error on the diagonal
# (the physically tracked quantity), and the full-matrix deviation normalised by its largest element.
function dpdev(dp, ref)
    (dp === nothing || ref === nothing || size(dp) != size(ref)) && return (NaN, NaN)
    diagrel = maximum(abs.(diag(dp) .- diag(ref)) ./ abs.(diag(ref)))
    return (diagrel, maximum(abs.(dp .- ref)) / maximum(abs.(ref)))
end

info = Dict{String,Any}()
for T in TOLS
    d = joinpath(SWEEP_DIR, "run_$T")
    isfile(joinpath(d, "gpec.h5")) || continue
    info[T] = merge(h5info(d), phases(joinpath(SWEEP_DIR, "tol_$T.log")))
end
ref = haskey(info, REFTOL) ? info[REFTOL].dp : nothing

@printf("%6s | %8s %8s %8s | %8s %8s %8s | %14s | %10s %10s\n",
    "tol", "nstep", "tried", "psi<0.1", "EL (s)", "ms/step", "run (s)", "et[1]", "Δ'diag rel", "Δ'mat rel")
for T in TOLS
    haskey(info, T) || continue
    r = info[T]
    ddiag, dmat = dpdev(r.dp, ref)
    @printf("%6s | %8d %8d %8d | %8.1f %8.2f %8.1f | %14.8g | %10.2e %10.2e\n",
        T, r.nstep, r.nstep_total, r.axis_steps, r.elint, r.elint / r.nstep * 1e3, r.total, r.et1, ddiag, dmat)
end
