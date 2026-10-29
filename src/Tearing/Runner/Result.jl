# Result.jl
#
# `SLAYERResult` packages the output of a full SLAYER analysis run:
# per-surface layer parameters, the extracted tearing eigenvalues, and (if
# `control.store_scan`) the full Q-plane scan data for plotting.

"""
    CriticalResonantFieldScan

Real-Q torque-balance samples at one surface.

## Fields

  - `Q`       -- sampled normalized frequencies (Cole's axis)
  - `balance` -- torque-balance value 2·P·(Q0 − Q)/Im[−1/(α + Δ)] at each Q
  - `Delta`   -- inner-layer Δ(Q) at each Q
"""
struct CriticalResonantFieldScan
    Q::Vector{Float64}
    balance::Vector{Float64}
    Delta::Vector{ComplexF64}
end

"""
    CriticalResonantFieldResult

Output of `run_critical_resonant_field`, one entry per SLAYER surface.

## Fields

  - `enabled`        -- the analysis ran
  - `rational_index` -- rational-surface index of each entry
  - `Q_peak`         -- normalized frequency Q at the torque-balance maximum (NaN if none or skipped)
  - `br_crit`        -- critical normalized resonant field b_r/B_φ (NaN if none or skipped)
  - `br_crit_tesla`  -- `br_crit`·|B0| in tesla, with the layer's B0, approximately comparable to the total resonant field
    `resonant_area_weighted_field` (Park et al., PRL 99, 195003 (2007))
  - `Q0`             -- normalized natural E×B rotation τ_k·n·Ω_E (NaN where Ω_E is unknown)
  - `scan`           -- per-surface Q scans; empty unless `store_scan`
"""
struct CriticalResonantFieldResult
    enabled::Bool
    rational_index::Vector{Int}
    Q_peak::Vector{Float64}
    br_crit::Vector{Float64}
    br_crit_tesla::Vector{Float64}
    Q0::Vector{Float64}
    scan::Vector{CriticalResonantFieldScan}
end

empty_critical_resonant_field_result() =
    CriticalResonantFieldResult(false, Int[], Float64[], Float64[], Float64[], Float64[], CriticalResonantFieldScan[])

"""
    SLAYERResult

Output of the tearing solve. Carries both summary eigenvalues (ω_Hz, γ_Hz) and
full diagnostic detail (valid roots, poles, filtered roots, contours) for
downstream inspection and HDF5 output.

# Fields

  - `enabled`             -- `true` only when the analysis actually ran
  - `control`             -- the `SLAYERControl` used (frozen snapshot)
  - `model`               -- the inner-layer model solved; `nothing` when the analysis did not run
  - `params`              -- `Vector{SLAYERParameters}`, one per surface
  - `rational_psi`, `rational_q` -- normalized poloidal flux ψ_N and safety
    factor q of each analyzed surface, aligned with `params`. Empty when the
    analysis was built from bare parameters (`run_slayer_from_inputs` without
    the surface list), in which case the HDF5 writer skips them.
  - `dp_matrix`           -- outer-region Δ' matrix used in the analysis.
    SLAYER path: r_s-referenced (the ψ_N BVP matrix transformed by
    `delta_prime_to_rs_reference`, written as `PerSurface/Delta_prime_matrix_rs`);
    GGJ path: the ψ_N matrix unchanged (written as `PerSurface/Delta_prime_matrix`)
  - `Q_root`              -- tearing eigenvalue(s) in normalized Q

      + length `nsurfaces` in `:uncoupled` mode
      + length `1` in `:coupled` mode (global eigenvalue normalized by
        `params[1].tauk`)
  - `omega_Hz`, `gamma_Hz` -- rotation angular frequency Re(Q)/τ_k [rad/s] and growth rate
    Im(Q)/τ_k [1/s]. `Q_root` and `omega_Hz` are in the lab frame in both modes; an uncoupled
    root is moved from its layer's E×B frame by τ_k·n·Ω_E, with Ω_E = 0 where it is unknown
  - `per_surface_extraction` -- `Vector{GrowthRateResult}` of length
    `nsurfaces` in uncoupled mode (each includes polelines, pole list,
    valid roots, filtered roots), in each surface's E×B frame. Empty in coupled mode.
  - `coupled_extraction`  -- single `GrowthRateResult` in coupled mode.
    `nothing` otherwise.
  - `layer_widths`        -- `Vector{LayerWidths}`, one per surface: the
    resistive layer thickness (in meters) from the `del_s` Riccati solve
    plus FKR / visco-resistive sanity scales. Empty when disabled.
  - `omega_E`             -- E×B angular frequency Ω_E per unit n [rad/s] resolved on each
    surface: the kinetic file's value, or the `control.omega_E_kHz` override for that m/n;
    NaN where neither gives one. Recorded in both coupling modes
  - `q_shift`             -- real Doppler offset `−τ_k·n·Ω_E` applied to each surface's
    inner-layer Q; zero outside `:coupled` mode, where rotation does not enter
  - `scan_data`           -- scan results (per-surface in uncoupled, single
    entry in coupled). Empty unless `control.store_scan == true`.
  - `critical_resonant_field` -- `CriticalResonantFieldResult`; `enabled=false`
    unless `control.critical_resonant_field.enabled`
"""
struct SLAYERResult
    enabled::Bool
    control::SLAYERControl
    model::Union{Nothing,InnerLayer.InnerLayerModel}
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
    omega_E::Vector{Float64}
    q_shift::Vector{Float64}
    scan_data::Vector{Union{ScanResult,AMRResult}}
    critical_resonant_field::CriticalResonantFieldResult
end

# Empty result (enabled=false path)
function empty_slayer_result(control::SLAYERControl)
    return SLAYERResult(false, control, nothing,
        SLAYERParameters[],
        Float64[], Float64[],
        zeros(ComplexF64, 0, 0),
        ComplexF64[], Float64[], Float64[],
        GrowthRateResult[], nothing,
        LayerWidths[],
        Float64[], Float64[],
        Union{ScanResult,AMRResult}[],
        empty_critical_resonant_field_result())
end
