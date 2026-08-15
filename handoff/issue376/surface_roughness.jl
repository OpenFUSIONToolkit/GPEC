# Source-level test for RESULTS.md section 10: are the per-surface geometry quantities smooth in psi?
#
# The EL coefficient matrices split cleanly (section 9): A/B/D/F/K use only g22/g23/g33, which are
# built from *theta*-derivatives of the rzphi spline, and they are clean. C/E/H additionally use
# g31/g11/g12/jmat1, which are built from *psi*-derivatives (`nodal_derivs.partials[2,...]` in
# Fourfit.jl), and they carry the ~1e-5 floor. If the rzphi node values carry a fixed-amplitude
# psi-direction error eps, then d/dpsi of them inherits eps/dpsi -- the amplification that makes
# C/E/H grid-sensitive while the theta-derived matrices stay clean.
#
# Usage: julia --project=. surface_roughness.jl <rundir> [<rundir> ...]
const ARGS_SAVE = copy(ARGS)
empty!(ARGS)
include(joinpath(@__DIR__, "roughness.jl"))
append!(ARGS, ARGS_SAVE)

const QUANTITIES = ("nu", "jac", "offset", "rcoords")
const THETA_FRACS = (0.15, 0.35, 0.6, 0.85)   # sample several poloidal angles, away from theta=0

"""
Median over sampled theta of the psi-direction roughness of one surface quantity:
r1 of the second divided difference, and the local-fit residual relative to |f|.
"""
function psi_roughness(psi, arr, idx)
    r1s, res = Float64[], Float64[]
    ntheta = size(arr, 2)
    for frac in THETA_FRACS
        col = view(arr, idx, clamp(round(Int, frac * ntheta), 1, ntheta))
        maximum(abs, col) == 0 && continue
        push!(r1s, lag1(second_divided_difference(psi[idx], collect(col))))
        push!(res, fit_residual(psi[idx], collect(col)))
    end
    return median(r1s), median(res)
end

for rundir in ARGS
    h5open(joinpath(rundir, "gpec.h5"), "r") do h5
        psi = read(h5["splines/rzphi/xs"])
        @printf("\n%s  (npsi = %d)\n", basename(rundir), length(psi))
        @printf("  %-22s | %-6s", "region", "nknot")
        for q in QUANTITIES
            @printf(" | %-22s", "$q: r1 / resid")
        end
        println()
        for (name, lo, hi) in REGIONS
            idx = window(psi, lo, hi)
            length(idx) < 12 && continue
            @printf("  %-22s | %6d", name, length(idx))
            for q in QUANTITIES
                r1, res = psi_roughness(psi, read(h5["splines/rzphi"][q]), idx)
                @printf(" | %+.3f / %10.3e", r1, res)
            end
            println()
        end
    end
end
