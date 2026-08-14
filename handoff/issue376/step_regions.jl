# Where do the extra ODE steps go as mpsi grows? Bin accepted steps by psi region.
using HDF5, Printf

const LADDER_DIR = @__DIR__
const EDGES = [0.0, 0.1, 0.3, 0.5, 0.516, 0.52, 0.6, 0.769, 0.772, 0.85, 0.89, 0.93, 0.955, 0.97, 0.985, 0.9935, 1.0]
# bins isolate q=2 (~0.518) and q=3 (~0.770) singular surfaces and the packed edge (q=4,5,6 land above 0.85)

function stepcounts(rundir)
    h5open(joinpath(rundir, "gpec.h5"), "r") do h5
        psi = read(h5["integration/psi"])
        counts = zeros(Int, length(EDGES) - 1)
        for p in psi
            b = clamp(searchsortedlast(EDGES, p), 1, length(counts))
            counts[b] += 1
        end
        return counts
    end
end

rows = []
for N in ["128", "256", "512", "1024", "auto"]
    d = joinpath(LADDER_DIR, "run_$N")
    isfile(joinpath(d, "gpec.h5")) || continue
    push!(rows, (N, stepcounts(d)))
end

@printf("%13s |", "psi bin")
for (N, _) in rows
    @printf(" %6s", N)
end
println()
for i in 1:length(EDGES)-1
    @printf("%6.3f-%6.4f |", EDGES[i], EDGES[i+1])
    for (_, c) in rows
        @printf(" %6d", c[i])
    end
    println()
end
@printf("%13s |", "TOTAL")
for (_, c) in rows
    @printf(" %6d", sum(c))
end
println()
