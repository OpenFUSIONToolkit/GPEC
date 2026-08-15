# Piece 1 verification: reconstructed gal ξ(ψ) and analytic ξ′(ψ).
#   1. shapes / finiteness sanity of ForceFreeStates/Solutions/GalerkinIntegration/Solution arrays
#   2. analytic ξ′ vs centered finite-difference of ξ — agree in the smooth interior, diverge at the
#      packed edge (the spline-endpoint-derivative artifact we deliberately avoid)
# Usage: julia --project=. verify_gal_solution.jl [path/to/gpec.h5]
using HDF5, Printf, Statistics

h5path = length(ARGS) >= 1 ? ARGS[1] : "examples/DIIID-like_gal_resistive_example/gpec.h5"
@info "Reading $h5path"

psi, issing, xi, dxi, sing_psi = h5open(h5path) do f
    (read(f["ForceFreeStates/Solutions/GalerkinIntegration/Solution/psi"]),
        read(f["ForceFreeStates/Solutions/GalerkinIntegration/Solution/is_rational"]), read(f["ForceFreeStates/Solutions/GalerkinIntegration/Solution/xi_psi"]),
        read(f["ForceFreeStates/Solutions/GalerkinIntegration/Solution/dxi_psidpsi"]), read(f["SingularSurfaces/GalerkinDeltaPrime/rational_psi"]))
end
issing = Bool.(issing)
mpert, ngrid, nsol = size(xi)
@printf("grid: mpert=%d  ngrid=%d  nsol=%d   psi∈[%.4f, %.4f]\n", mpert, ngrid, nsol, psi[1], psi[end])
@printf("singular surfaces at psi = %s\n", join((@sprintf("%.4f", p) for p in sing_psi), ", "))

# --- finiteness (skip on-surface points, which are intentionally left zero) ---
good = .!issing
nbad = count(!isfinite, xi[:, good, :]) + count(!isfinite, dxi[:, good, :])
@printf("non-finite entries (off-surface): %d   |xi|max=%.3e  |dxi|max=%.3e\n",
    nbad, maximum(abs, xi[:, good, :]), maximum(abs, dxi[:, good, :]))

# --- analytic ξ′ vs centered finite difference of ξ ---
# For each column, centered diff at interior grid points (using off-surface neighbours), compared to the
# analytic xi_deriv. Report median relative error binned by distance to the nearest singular surface and
# to the packed edge.
dist_sing(p) = isempty(sing_psi) ? Inf : minimum(abs.(p .- sing_psi))
relerr = Float64[]
dsurf = Float64[]
dedge = Float64[]
for isol in 1:nsol
    for ip in 2:(ngrid-1)
        (issing[ip] || issing[ip-1] || issing[ip+1]) && continue
        h1 = psi[ip] - psi[ip-1]
        h2 = psi[ip+1] - psi[ip]
        (h1 <= 0 || h2 <= 0) && continue
        for m in 1:mpert
            # nonuniform centered difference
            fd = (xi[m, ip+1, isol] - xi[m, ip-1, isol]) / (h2 + h1)
            an = dxi[m, ip, isol]
            scale = max(abs(an), abs(fd), 1e-30)
            push!(relerr, abs(fd - an) / scale)
            push!(dsurf, dist_sing(psi[ip]))
            push!(dedge, psi[end] - psi[ip])
        end
    end
end

@printf("\nFinite-diff(ξ) vs analytic ξ′   (%d samples)\n", length(relerr))
println("  binned by distance to nearest singular surface:")
for (lo, hi) in [(0.0, 1e-3), (1e-3, 1e-2), (1e-2, 1e-1), (1e-1, Inf)]
    sel = (dsurf .>= lo) .& (dsurf .< hi)
    n = count(sel)
    n == 0 && continue
    @printf("    |ψ-ψ_s|∈[%-7.0e,%-7.0e)  n=%-7d  median relerr=%.2e  p90=%.2e\n",
        lo, hi, n, median(relerr[sel]), quantile(relerr[sel], 0.9))
end
println("  binned by distance to packed edge ψ=psihigh:")
for (lo, hi) in [(0.0, 1e-4), (1e-4, 1e-3), (1e-3, 1e-2), (1e-2, Inf)]
    sel = (dedge .>= lo) .& (dedge .< hi)
    n = count(sel)
    n == 0 && continue
    @printf("    edge dist∈[%-7.0e,%-7.0e)  n=%-7d  median relerr=%.2e  p90=%.2e\n",
        lo, hi, n, median(relerr[sel]), quantile(relerr[sel], 0.9))
end

# Interior (away from both layer and edge) should agree well; near edge/layer the finite diff degrades.
interior = (dsurf .> 1e-2) .& (dedge .> 1e-2)
@printf("\nINTERIOR (>1e-2 from layer AND edge): median relerr = %.2e  (n=%d)\n",
    median(relerr[interior]), count(interior))
println(median(relerr[interior]) < 1e-3 ?
        "  ✓ analytic ξ′ matches finite-diff in the smooth interior" :
        "  ✗ interior mismatch — investigate")
