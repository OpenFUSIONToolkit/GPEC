# Result.jl
#
# Result types of the tearing solve and the critical-resonant-field analysis.

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
    TearingResult

Output of the tearing solve: summary eigenvalues (ω, γ) and the full root-extraction diagnostics.

## Fields

  - `enabled`             -- `true` only when the analysis actually ran
  - `problem`             -- producing TearingProblem, or nothing for a low-level scan
  - `config`              -- resolved scan configuration used by the model
  - `model`               -- the inner-layer model solved; `nothing` when the analysis did not run
  - `params`              -- SLAYER or GGJ layer parameters, one per surface
  - `rational_psi`, `rational_q` -- ψ_N and q of each surface, aligned with `params`; empty when
    built from bare parameters, and then not written to HDF5
  - `dp_matrix`           -- outer-region Δ' matrix used: r_s-referenced for SLAYER (`delta_prime_to_rs_reference`,
    written as `PerSurface/Delta_prime_matrix_rs`), the ψ_N matrix for GGJ (`PerSurface/Delta_prime_matrix`)
  - `Q_root`              -- tearing eigenvalue(s) in normalized Q

      + length `nsurfaces` in `:uncoupled` mode
      + length `1` in `:coupled` mode (global eigenvalue normalized by
        `params[1].tauk`)
  - `omega`, `gamma` -- lab-frame angular frequency Re(Q)/τ_k in `rad/s` and growth rate Im(Q)/τ_k in `1/s`;
    positive gamma is unstable. An uncoupled root is moved from its layer's E×B frame by n·Ω_E
    (Ω_E = 0 where unknown); a coupled GGJ layer sees s + i·n·Ω_E
  - `per_surface_extraction` -- `Vector{GrowthRateResult}`, one per surface in uncoupled mode, in each
    surface's E×B frame. Empty in coupled mode.
  - `coupled_extraction`  -- single `GrowthRateResult` in coupled mode.
    `nothing` otherwise.
  - `layer_widths`        -- `Vector{LayerWidths}`, one per SLAYER surface: the `del_s` Riccati layer
    thickness in meters plus FKR / visco-resistive scales. Empty for GGJ.
  - `omega_E`             -- E×B angular frequency Ω_E per unit n in `rad/s` on each surface: the kinetic
    file's value or the `problem.omega_E_kHz` override; NaN where neither gives one
  - `q_shift`             -- real Doppler offset `−τ_k·n·Ω_E` applied to each surface's
    inner-layer Q; zero outside `:coupled` mode, where rotation does not enter. Zero for GGJ, whose
    shift n·Ω_E acts on the growth-rate variable
  - `scan_data`           -- scan results (per-surface in uncoupled, single
    entry in coupled). Empty unless `config.store_scan == true`.
  - `critical_resonant_field` -- `CriticalResonantFieldResult`; `enabled=false`
    when `problem.critical_resonant_field` is `nothing`
"""
struct TearingResult{P,C}
    problem::P
    enabled::Bool
    config::C
    model::Union{Nothing,InnerLayer.InnerLayerModel}
    params::AbstractVector{<:InnerLayerParameters}
    rational_psi::Vector{Float64}
    rational_q::Vector{Float64}
    dp_matrix::Matrix{ComplexF64}
    Q_root::Vector{ComplexF64}
    omega::Vector{Float64}
    gamma::Vector{Float64}
    per_surface_extraction::Vector{GrowthRateResult}
    coupled_extraction::Union{Nothing,GrowthRateResult}
    layer_widths::Vector{LayerWidths}
    omega_E::Vector{Float64}
    q_shift::Vector{Float64}
    scan_data::Vector{Union{ScanResult,AMRResult}}
    critical_resonant_field::CriticalResonantFieldResult
end

# Result of an analysis that did not run (enabled=false).
function empty_tearing_result(problem, config::TearingConfig, model)
    return TearingResult(problem, false, config, model,
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
