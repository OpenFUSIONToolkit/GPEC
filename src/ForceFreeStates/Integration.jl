"""
    forcefreestates_integration(ctrl::ForceFreeStatesControl, equil::Equilibrium.PlasmaEquilibrium, ffit::FourFitVars, intr::ForceFreeStatesInternal) -> IntegrationResult

Single entry point for ideal-MHD Euler-Lagrange integration. Returns an
[`IntegrationResult`](@ref) wrapping the integrated `OdeState` plus the optional
fundamental-matrix data (`propagators`, `chunks`, `S_at_surface_left`) that the
chunked-Riccati path produces for the deferred Δ' BVP.

Currently delegates to `eulerlagrange_integration`, which dispatches on the legacy
`use_parallel` / `use_riccati` flags. A later step replaces that dispatch with
`ctrl.integration_method` (`"ChunkedRiccati"` / `"LegacyEulerLagrange"`).
"""
function forcefreestates_integration(ctrl::ForceFreeStatesControl, equil::Equilibrium.PlasmaEquilibrium, ffit::FourFitVars, intr::ForceFreeStatesInternal)
    odet, propagators, chunks, S_at_surface_left = eulerlagrange_integration(ctrl, equil, ffit, intr)
    return IntegrationResult(; odet, propagators, chunks, S_at_surface_left)
end
