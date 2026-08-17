# Result.jl
#
# `SLAYERResult` packages the output of a full SLAYER analysis run:
# per-surface layer parameters, the extracted tearing eigenvalues, and (if
# `control.store_scan`) the full Q-plane scan data for plotting.
#
# `CriticalResonantFieldResult` packages the output of the critical resonant field analysis run:
# per-surface critical resoant field. If `control.store_scan` is true, the full Q scan data,
# viscous torque, and electromagnetic torque are also stored for plotting.

"""
    SLAYERResult

Output of `run_slayer`. Carries both summary eigenvalues (ω_Hz, γ_Hz) and
full diagnostic detail (valid roots, poles, filtered roots, contours) for
downstream inspection and HDF5 output.

# Fields

  - `enabled`             -- `true` only when the analysis actually ran
  - `control`             -- the `SLAYERControl` used (frozen snapshot)
  - `params`              -- `Vector{SLAYERParameters}`, one per surface
  - `rational_psi`, `rational_q` -- normalized poloidal flux ψ_N and safety
    factor q of each analyzed surface, aligned with `params`. Empty when the
    analysis was built from bare parameters (`run_slayer_from_inputs` without
    the surface list), in which case the HDF5 writer skips them.
  - `dp_matrix`           -- outer-region Δ' matrix used in the analysis
  - `Q_root`              -- tearing eigenvalue(s) in normalized Q
      + length `nsurfaces` in `:uncoupled` mode
      + length `1` in `:coupled` mode (global eigenvalue normalized by
        `params[1].tauk`)
  - `omega_Hz`, `gamma_Hz` -- physical rotation frequency / growth rate
  - `per_surface_extraction` -- `Vector{GrowthRateResult}` of length
    `nsurfaces` in uncoupled mode (each includes polelines, pole list,
    valid roots, filtered roots). Empty in coupled mode.
  - `coupled_extraction`  -- single `GrowthRateResult` in coupled mode.
    `nothing` otherwise.
  - `layer_widths`        -- `Vector{LayerWidths}`, one per surface: the
    resistive layer thickness (in meters) from the `del_s` Riccati solve
    plus FKR / visco-resistive sanity scales. Empty when disabled.
  - `scan_data`           -- scan results (per-surface in uncoupled, single
    entry in coupled). Empty unless `control.store_scan == true`.
"""
struct SLAYERResult
    enabled::Bool
    control::SLAYERControl
    params::AbstractVector{<:InnerLayerParameters}
    rational_psi::Vector{Float64}
    rational_q::Vector{Float64}
    dp_matrix::Matrix{ComplexF64}
    Q_root::Vector{ComplexF64}
    omega_Hz::Vector{Float64}
    gamma_Hz::Vector{Float64}
    per_surface_extraction::Vector{GrowthRateResult}
    coupled_extraction::Union{Nothing,GrowthRateResult}
    layer_widths::Vector{LayerWidths}
    scan_data::Vector{Union{ScanResult,AMRResult}}
end

# Empty result (enabled=false path)
function empty_slayer_result(control::SLAYERControl)
    return SLAYERResult(false, control,
                        SLAYERParameters[],
                        Float64[], Float64[],
                        zeros(ComplexF64, 0, 0),
                        ComplexF64[], Float64[], Float64[],
                        GrowthRateResult[], nothing,
                        LayerWidths[],
                        Union{ScanResult,AMRResult}[])
end

"""
    CriticalResonantFieldResult

Output of `run_critical_resonant_field`. Carries both summary critical resonant field values and if `control.store_scan` is true, the full Q scan data, viscous torque, and electromagnetic torque for plotting.
"""

struct CriticalResonantFieldResult
    enabled::Bool
    params::AbstractVector{<:InnerLayerParameters}
    surface_index::Vector{Int}
    Qpeak::Vector{Float64}
    br_crit::Vector{Float64}
    Q0::Vector{Float64}
    P::Vector{Float64}
    scan_data::Vector{NamedTuple}
end

function empty_critical_resonant_field_result()
    return CriticalResonantFieldResult(
        false,
        InnerLayerParameters[],
        Int[],
        Float64[],
        Float64[],
        Float64[],
        Float64[],
        NamedTuple[]
    )
end
