"""
    Diagnostics

Island and stochasticity diagnostics. Island half-widths and Chirikov overlap come from the
resonant `island_width_sq` that drives the field-line map, so they are consistent with the
Poincaré plot by construction and match GPEC's singular-coupling island diagnostics.
"""

"""
    compute_island_diagnostics!(state, res_m, res_n, res_psi, island_width_sq)

Fill `state.island_*` with, per rational surface (sorted by ψ_N), the island half-width
`w½ = sqrt(|island_width_sq|)` and the Chirikov overlap with the neighbouring islands,
`σ_i = (w½_i + w½_{i+1}) / |ψ_{i+1} − ψ_i|` (`σ > 1` signals overlap / stochasticity).
"""
function compute_island_diagnostics!(state::FieldLineTracingState,
                                     res_m::Vector{Int}, res_n::Vector{Int},
                                     res_psi::Vector{Float64},
                                     island_width_sq::Vector{ComplexF64})
    order = sortperm(res_psi)
    psi = res_psi[order]
    m = res_m[order]
    n = res_n[order]
    w = sqrt.(abs.(island_width_sq[order]))

    for i in eachindex(psi)
        chirikov = NaN
        if i < length(psi)
            sep = abs(psi[i + 1] - psi[i])
            sep > 0 && (chirikov = (w[i] + w[i + 1]) / sep)
        end
        push!(state.island_m, m[i])
        push!(state.island_n, n[i])
        push!(state.island_psi, psi[i])
        push!(state.island_halfwidth, w[i])
        push!(state.island_chirikov, chirikov)
    end
    return nothing
end
