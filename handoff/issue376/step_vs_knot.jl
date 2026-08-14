# Compare ODE step sizes with local knot spacing: is dpsi pinned to ~k knot intervals?
using HDF5, Printf, Statistics

const LADDER_DIR = @__DIR__

function analyze(rundir)
    h5open(joinpath(rundir, "gpec.h5"), "r") do h5
        psi = read(h5["integration/psi"])
        xs = read(h5["splines/profiles/xs"])
        dpsi = diff(psi)
        keep = dpsi .> 0            # drop chunk-boundary resets
        dpsi = dpsi[keep]
        pmid = psi[1:end-1][keep] .+ dpsi ./ 2
        # local knot spacing at each step midpoint
        dxs = diff(xs)
        spacing = [dxs[clamp(searchsortedlast(xs, p), 1, length(dxs))] for p in pmid]
        ratio = dpsi ./ spacing
        return length(psi), median(ratio), quantile(ratio, 0.1), quantile(ratio, 0.9), median(dpsi), median(spacing)
    end
end

@printf("%6s | %6s | %12s (%5s–%5s) | %10s %10s\n", "mpsi", "nstep", "med dψ/knot", "p10", "p90", "med dψ", "med knotΔ")
for N in ["128", "256", "512", "1024", "auto"]
    d = joinpath(LADDER_DIR, "run_$N")
    isfile(joinpath(d, "gpec.h5")) || continue
    n, med, p10, p90, mdp, msp = analyze(d)
    @printf("%6s | %6d | %12.2f (%5.2f–%5.2f) | %10.2e %10.2e\n", N, n, med, p10, p90, mdp, msp)
end
