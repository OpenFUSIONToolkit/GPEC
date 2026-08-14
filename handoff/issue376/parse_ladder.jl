# Parse TSMARK phase timestamps from ladder logs + nstep/et from gpec.h5, print the ladder table.
using HDF5, Printf

const LADDER_DIR = @__DIR__

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

# Timestamp of first mark whose tag starts with `prefix`, at or after index i0; returns (ts, idx)
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
             ("equil_end", "Equilibrium construction completed"),
             ("ffs_start", "  Force-Free States"),
             ("runparams", "Run parameters:"),
             ("fgk", "Computing F, G, and K matrices"),
             ("el", "Integrating Euler-Lagrange"),
             ("freeb", "Computing free boundary"),
             ("ffs_end", "Force-Free States completed"),
             ("pe_start", "  Perturbed Equilibrium"),
             ("pe_end", "Perturbed Equilibrium completed"),
             ("run2_end", "RUN2_END")]
    i = i0
    for (key, prefix) in order
        t[key], i = tsafter(m, prefix, i)
    end
    return (equil=t["equil_end"] - t["equil_start"],
        balloon=t["runparams"] - t["ffs_start"],
        metric=t["fgk"] - t["runparams"],
        matrix=t["el"] - t["fgk"],
        elint=t["freeb"] - t["el"],
        ffs_rest=t["ffs_end"] - t["freeb"],
        pe=t["pe_end"] - t["pe_start"],
        total=t["run2_end"] - t["equil_start"])
end

function h5info(rundir)
    h5open(joinpath(rundir, "gpec.h5"), "r") do h5
        nstep = read(h5["integration/nstep"])
        et = read(h5["FreeBoundaryStability/eigenmode_energies"])
        npsi = try
            length(read(h5["splines/profiles/xs"]))
        catch
            -1
        end
        return sum(nstep), real(et[1]), npsi
    end
end

@printf("%6s | %6s | %7s %7s %7s %7s | %8s %8s | %7s %7s | %10s | %8s\n",
        "mpsi", "knots", "equil", "balloon", "matrix", "elint", "nstep", "ms/step", "ffsrest", "pe", "et1", "total")
for N in ["128", "256", "512", "1024", "auto"]
    log = joinpath(LADDER_DIR, "mpsi_$N.log")
    isfile(log) || continue
    p = phases(log)
    nstep, et1, npsi = try
        h5info(joinpath(LADDER_DIR, "run_$N"))
    catch e
        (-1, NaN, -1)
    end
    @printf("%6s | %6d | %7.1f %7.1f %7.1f %7.1f | %8d %8.2f | %7.1f %7.1f | %10.4g | %8.1f\n",
            N, npsi, p.equil, p.balloon, p.metric + p.matrix, p.elint, nstep,
            nstep > 0 ? p.elint / nstep * 1e3 : NaN, p.ffs_rest, p.pe, et1, p.total)
end
