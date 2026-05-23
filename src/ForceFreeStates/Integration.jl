"""
    forcefreestates_integration(ctrl::ForceFreeStatesControl, equil::Equilibrium.PlasmaEquilibrium, ffit::FourFitVars, intr::ForceFreeStatesInternal) -> IntegrationResult

Single entry point for ideal-MHD Euler-Lagrange integration. Dispatches on
`ctrl.integration_method`:

  - `"ChunkedRiccati"` (default) — runs two integrations *concurrently*: the legacy
    Euler-Lagrange sweep on ≈ 1 thread (produces the trusted `u_store` for
    PerturbedEquilibrium) and the chunked-Riccati fundamental-matrix solve on the
    remaining N − 1 (produces accurate Δ' and a diagnostic Riccati-gauge eigenmode).
    The two least-stable eigenvalues are cross-checked downstream.
  - `"LegacyEulerLagrange"` — the standard forward Euler-Lagrange sweep alone, kept for
    benchmarking. Does not produce Δ'.

Returns an [`IntegrationResult`](@ref); the `odet_riccati` / `propagators` / `chunks` /
`S_at_surface_left` fields are populated only by the chunked-Riccati path.
"""
function forcefreestates_integration(ctrl::ForceFreeStatesControl, equil::Equilibrium.PlasmaEquilibrium, ffit::FourFitVars, intr::ForceFreeStatesInternal)
    # Kinetic runs regularize the rational-surface layers (Logan 2015 Eq. 7.46), so there
    # are no ideal jumps to cross; the chunked-Riccati Δ' BVP is ideal-MHD specific. Route
    # kinetic runs through the legacy Euler-Lagrange sweep regardless of integration_method.
    if ctrl.kinetic_factor > 0
        odet = eulerlagrange_integration(ctrl, equil, ffit, intr)
        return IntegrationResult(odet)
    end

    if ctrl.integration_method == "ChunkedRiccati"
        odet, odet_riccati, propagators, chunks, S_at_surface_left = parallel_eulerlagrange_integration(ctrl, equil, ffit, intr)
        return IntegrationResult(; odet, odet_riccati, propagators, chunks, S_at_surface_left)
    elseif ctrl.integration_method == "LegacyEulerLagrange"
        odet = eulerlagrange_integration(ctrl, equil, ffit, intr)
        return IntegrationResult(odet)
    else
        error("Unknown integration_method \"$(ctrl.integration_method)\"; expected \"ChunkedRiccati\" or \"LegacyEulerLagrange\"")
    end
end
