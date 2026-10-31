"""
    CriticalResonantFieldControl

Configuration for the critical resonant field (torque-balance) analysis, passed as
`TearingProblem(ffs; critical_resonant_field=CriticalResonantFieldControl(; ...))`; a deck states it as
`[Tearing.CriticalResonantField]`. The magnetic Prandtl number is the per-surface
`P_tor` of the SLAYER layer parameters, and the natural rotation is each surface's resolved Ω_E
(the kinetic file's `omega_E`, or the `omega_E_kHz` override), which it therefore requires.

## Fields

  - `Qmin`, `Qmax` -- override the sampled real-Q range; `nothing` samples from Q0 toward Q_e per surface
  - `n`          -- number of Q samples per surface
  - `store_scan` -- write the per-surface Q, torque balance and Δ samples to `gpec.h5`
"""
@kwdef struct CriticalResonantFieldControl
    Qmin::Union{Nothing,Float64} = nothing
    Qmax::Union{Nothing,Float64} = nothing
    n::Int = 2000
    store_scan::Bool = false
end

"""
    TearingConfig(; kwargs...)

Numerical scan and root-search settings for `solve(TearingProblem(...), model; config)`.
GGJ uses physical `gamma_range` and `omega_range`, with positive growth rates and frequencies
in the result's convention: lab-frame ω for a coupled solve, each surface's E×B-frame ω for an
uncoupled one. SLAYER uses its normalized Q box. The default
classification selects threshold classification for SLAYER and winding classification for GGJ.

## Fields

  - `gamma_range`, `omega_range` - GGJ search bounds in 1/s and rad/s; both required for a GGJ solve.
  - `scan_mode` - SLAYER scan method, `:amr` or `:brute_force`; GGJ uses AMR.
  - `Q_re_range`, `Q_im_range` - SLAYER normalized frequency search box.
  - `nre`, `nim` - Initial scan resolution.
  - `amr_passes`, `amr_max_cells` - AMR depth and cell cap.
  - `boxes`, `multi_box_prescreen_n` - Optional SLAYER multi-box scan and prescreen resolution.
  - `pole_threshold`, `pole_threshold_adaptive` - SLAYER threshold-classification controls.
  - `filter_above_poles`, `filter_outside_re` - SLAYER contour filters.
  - `gamma_gap_threshold` - SLAYER growth-rate separation threshold in 1/s.
  - `polish_roots`, `validity_rtol` - Root polishing and SLAYER residual acceptance tolerance.
  - `root_classification` - `:auto`, `:threshold`, or `:winding`; GGJ requires winding.
  - `store_scan` - Retain scan data in the result and HDF5 output.
"""
@kwdef struct TearingConfig
    gamma_range::Union{Nothing,Tuple{Float64,Float64}} = nothing
    omega_range::Union{Nothing,Tuple{Float64,Float64}} = nothing
    scan_mode::Symbol = :amr
    Q_re_range::Tuple{Float64,Float64} = (-10.0, 10.0)
    Q_im_range::Tuple{Float64,Float64} = (-2.0, 5.0)
    nre::Int = 41
    nim::Int = 31
    amr_passes::Int = 4
    amr_max_cells::Int = 10_000_000
    boxes::Vector{NTuple{4,Float64}} = NTuple{4,Float64}[]
    multi_box_prescreen_n::Int = 25
    pole_threshold::Float64 = 10.0
    pole_threshold_adaptive::Bool = false
    filter_above_poles::Bool = true
    filter_outside_re::Bool = true
    gamma_gap_threshold::Float64 = 1e3
    polish_roots::Bool = true
    validity_rtol::Float64 = 1e-3
    root_classification::Symbol = :auto
    store_scan::Bool = false
end
