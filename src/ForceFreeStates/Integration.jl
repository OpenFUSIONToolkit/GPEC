"""
    forcefreestates_integration(ctrl::ForceFreeStatesControl, equil::Equilibrium.PlasmaEquilibrium, ffit::FourFitVars, intr::ForceFreeStatesInternal) -> IntegrationResult

Single entry point for ideal-MHD Euler-Lagrange integration. Dispatches on
`ctrl.integration_method`:

  - `"ChunkedRiccati"` (default) — chunks the domain at rational surfaces and integrates
    each chunk's fundamental matrix in parallel, with backward integration away from
    rationals so every sub-integration runs well-conditioned. Produces accurate Δ' and the
    `u_store` consumed by PerturbedEquilibrium. Runs on `ctrl.integration_threads` threads.
  - `"LegacyEulerLagrange"` — the standard forward Euler-Lagrange sweep, kept for
    benchmarking and cross-checking. Does not produce Δ'.

Returns an [`IntegrationResult`](@ref); the `propagators` / `chunks` /
`S_at_surface_left` fields are populated only by the chunked-Riccati path.
"""
function forcefreestates_integration(ctrl::ForceFreeStatesControl, equil::Equilibrium.PlasmaEquilibrium, ffit::FourFitVars, intr::ForceFreeStatesInternal)
    if ctrl.integration_method == "ChunkedRiccati"
        odet, propagators, chunks, S_at_surface_left = parallel_eulerlagrange_integration(ctrl, equil, ffit, intr)
        return IntegrationResult(; odet, propagators, chunks, S_at_surface_left)
    elseif ctrl.integration_method == "LegacyEulerLagrange"
        odet = eulerlagrange_integration(ctrl, equil, ffit, intr)
        return IntegrationResult(odet)
    else
        error("Unknown integration_method \"$(ctrl.integration_method)\"; expected \"ChunkedRiccati\" or \"LegacyEulerLagrange\"")
    end
end
