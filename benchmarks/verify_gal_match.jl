# Piece 2 verification: RPEC outer↔inner matched solution (closed profiles under
# GalerkinIntegration/xi_psi + matching diagnostics under GalerkinIntegration/Match/*).
#   1. linear-solve residual ‖mat·cof − rmat‖/‖rmat‖
#   2. matched ξ / ξ′ finiteness
#   3. edge column == identity basis: each coil drive j must give ξ_edge = e_j (the j-th harmonic),
#      since plasma columns vanish at the edge and coil column j carries the unit edge source
#   4. inner-layer Δ(Q) per surface (finite, nonzero); forced eigenvalues γ_s = 2πi·n·f
# Usage: julia --project=. benchmarks/verify_gal_match.jl [path/to/gpec.h5]
using HDF5, Printf, LinearAlgebra

h5path = length(ARGS) >= 1 ? ARGS[1] : "examples/DIIID-like_gal_resistive_example/gpec.h5"
@info "Reading $h5path"

xi, dxi, cout, cin, deltar, eig, resid, sing_psi = h5open(h5path) do f
    (read(f["ForceFreeStates/Solutions/GalerkinIntegration/xi_psi"]), read(f["ForceFreeStates/Solutions/GalerkinIntegration/dxi_psidpsi"]),
        read(f["ForceFreeStates/Solutions/GalerkinIntegration/Match/cout"]), read(f["ForceFreeStates/Solutions/GalerkinIntegration/Match/cin"]),
        read(f["ForceFreeStates/Solutions/GalerkinIntegration/Match/Delta_r"]), read(f["ForceFreeStates/Solutions/GalerkinIntegration/Match/rpec_eig"]),
        read(f["ForceFreeStates/Solutions/GalerkinIntegration/Match/residual"]), read(f["ForceFreeStates/Solutions/GalerkinIntegration/rational_psi"]))
end
# HDF5 stores ComplexF64 as a compound (re,im); convert if needed
to_c(a) = eltype(a) <: Complex ? a : map(x -> ComplexF64(x.re, x.im), a)
xi = to_c(xi); dxi = to_c(dxi); cout = to_c(cout); cin = to_c(cin); deltar = to_c(deltar); eig = to_c(eig)

mpert, mcoil, ngrid = size(xi)
msing = size(deltar, 1)
@printf("matched solution: mpert=%d  ngrid=%d  mcoil=%d   msing=%d\n", mpert, ngrid, mcoil, msing)

@printf("\n[1] linear-solve residual  ‖mat·cof − rmat‖/‖rmat‖ = %.3e   %s\n",
    resid, resid < 1e-8 ? "✓" : "✗ (large)")

nbad = count(!isfinite, xi) + count(!isfinite, dxi)
@printf("[2] finiteness: non-finite entries = %d   |ξ|max=%.3e  |ξ′|max=%.3e   %s\n",
    nbad, maximum(abs, xi), maximum(abs, dxi), nbad == 0 ? "✓" : "✗")

# [3] edge column == identity (last grid point = psihigh edge)
edge = xi[:, :, ngrid]                       # (mpert mode, mcoil drive)
id_err = norm(edge - Matrix{ComplexF64}(I, mpert, mcoil)) / sqrt(mpert)
@printf("[3] edge basis: ‖ξ(edge) − I‖/√mpert = %.3e   %s\n",
    id_err, id_err < 1e-8 ? "✓ identity-at-edge" : "✗")

# [4] inner-layer Δ and forced eigenvalues
@printf("\n[4] per-surface inner-layer Δ(Q) and forced eigenvalue γ = 2πi·n·f:\n")
@printf("    %-4s %-10s  %-26s %-26s\n", "s", "ψ_s", "Δ₁", "Δ₂")
for s in 1:msing
    @printf("    %-4d %-10.5f  %+.4e%+.4ei  %+.4e%+.4ei   γ=%+.3f%+.3fi\n",
        s, sing_psi[s], real(deltar[s, 1]), imag(deltar[s, 1]),
        real(deltar[s, 2]), imag(deltar[s, 2]), real(eig[s]), imag(eig[s]))
end
allgood = resid < 1e-8 && nbad == 0 && id_err < 1e-8 && all(isfinite, deltar) && all(!iszero, deltar)
println(allgood ? "\n✓ Piece 2 matching checks PASS" : "\n✗ some checks failed — investigate")
