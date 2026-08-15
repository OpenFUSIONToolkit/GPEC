# Verify the gal_ideal_flag path: the matched solution is the bare ideal coil column (cout=0).
#   1. cout / cin / deltar all zero (no resistive plasma combination, no inner layer)
#   2. matched ξ_j == the gal coil column sols(:,:,2·msing+j)   (and ξ′ likewise)
# Usage: julia --project=. benchmarks/verify_gal_ideal.jl [path/to/gpec.h5]
using HDF5, Printf, LinearAlgebra

h5path = length(ARGS) >= 1 ? ARGS[1] : "/tmp/gal_ideal_test/gpec.h5"
to_c(a) = eltype(a) <: Complex ? ComplexF64.(a) : map(x -> ComplexF64(x.re, x.im), a)

cout, deltar, mxi, mdxi, sols, sols_d, sing_psi = h5open(h5path) do f
    (to_c(read(f["ForceFreeStates/Solutions/GalerkinIntegration/Match/cout"])), to_c(read(f["ForceFreeStates/Solutions/GalerkinIntegration/Match/Delta_r"])),
        to_c(read(f["ForceFreeStates/Solutions/GalerkinIntegration/Match/xi"])), to_c(read(f["ForceFreeStates/Solutions/GalerkinIntegration/Match/dxidpsi"])),
        to_c(read(f["ForceFreeStates/Solutions/GalerkinIntegration/Solution/xi_psi"])), to_c(read(f["ForceFreeStates/Solutions/GalerkinIntegration/Solution/dxi_psidpsi"])),
        read(f["SingularSurfaces/GalerkinDeltaPrime/rational_psi"]))
end
msing = length(sing_psi)
mpert, ngrid, mcoil = size(mxi)

@printf("[1] ‖cout‖ = %.2e, ‖deltar‖ = %.2e   %s\n", norm(cout), norm(deltar),
    (norm(cout) == 0 && norm(deltar) == 0) ? "✓ no resistive combination (ideal)" : "✗")

# matched ξ_j should equal the coil column sols(:,:,2msing+j)
err = maximum(1:mcoil) do j
    csol = 2msing + j
    max(norm(mxi[:, :, j] - sols[:, :, csol]), norm(mdxi[:, :, j] - sols_d[:, :, csol]))
end
@printf("[2] max‖ξ_matched − coil column‖ = %.2e   %s\n", err,
    err < 1e-12 ? "✓ matched ξ == bare ideal coil column" : "✗")

println((norm(cout) == 0 && err < 1e-12) ? "\n✓ gal_ideal_flag path verified" : "\n✗ investigate")
