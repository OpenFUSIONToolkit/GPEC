# Control.jl
#
# `SLAYERControl` holds every user-facing knob that drives the SLAYER
# growth-rate analysis. Populated either directly via the `@kwdef`
# constructor or by parsing the `[SLAYER]` (and nested `[SLAYER.*]`)
# section(s) of a `gpec.toml`.

"""
    SLAYERControl

Configuration for the SLAYER tearing-mode analysis. All fields are
user-facing: read from the `[SLAYER]` TOML section of a `gpec.toml` via
`slayer_control_from_toml`, or built directly via the `@kwdef` keyword
constructor.

# Core toggles

  - `enabled`       -- run the analysis at all (default `false`)
  - `inner_model`   -- `:slayer_fitzpatrick` (default), `:ggj_shooting`, or
    `:ggj_galerkin`
  - `scan_mode`     -- `:amr` (default) or `:brute_force`
  - `coupling_mode` -- `:uncoupled` (default, per-surface) or `:coupled`
    (multi-surface determinant)
  - `dc_type`       -- critical-Δ offset selector, one of `:none`, `:lar`,
    `:rfitzp`, `:toroidal` (see `params.f:230-242`)
  - `msing_max`     -- number of surfaces to include in the coupled
    determinant (default 3; capped at `length(sings)` at runtime)

# Physics knobs

  - `bt`       -- toroidal field [T]. `nothing` → use `equil.config.b0exp`
  - `mu_i`     -- ion mass in proton-mass units (default 2.0 for D)
  - `zeff`     -- effective charge
  - `chi_perp`, `chi_tor` -- perpendicular / toroidal heat diffusivity [m²/s]
  - `dr_val`, `dgeo_val`  -- critical-Δ formula inputs
  - `theta_sample` -- poloidal angle at which to sample minor radius
    (default 0.0, outboard midplane)

# Scan grid (used for both brute-force and AMR initial mesh)

  - `Q_re_range`, `Q_im_range` -- box in the normalized Q plane
  - `nre`, `nim`    -- grid resolution along each axis

# AMR refinement

  - `amr_passes`    -- max refinement levels
  - `amr_max_cells` -- hard safety cap

# Growth-rate-extraction filters

  - `pole_threshold`      -- threshold for pole classification (default 10)
  - `pole_threshold_adaptive` -- if true, pole_threshold is OVERRIDDEN per
    scan with `|mean(Δ)|` (the magnitude of the mean dispersion residual
    over the scan grid). Useful when |Δ| spans 8+ orders of magnitude
    (e.g. SLAYER scans where the hardcoded 10.0 default is too restrictive
    and classifies all intersections as poles). Validated against the
    omfit recipe and the Python `10·median(|d|)` heuristic — both
    converge to the same root identification on DIIID benchmark cases.
  - `filter_above_poles`  -- discard roots above the highest pole γ
  - `filter_outside_re`   -- condition the above-pole filter on the +γ
    step exiting the Re(Δ)=0 contour loop

# Kinetic-profile source

  - `profile_source` -- `:inline` (use the `[SLAYER.profiles]` TOML table)
    or `:h5` (read from a separate HDF5 file)
  - `profile_file`   -- HDF5 path (relative to the run dir), required if
    `profile_source === :h5`
  - `profile_group`  -- group within the HDF5 file (default `"/"`)

# Output control

  - `store_scan`  -- write the full Q/Δ scan grid to HDF5. `false` by
    default to keep the output file small.
"""
@kwdef struct SLAYERControl
    enabled::Bool = false

    inner_model::Symbol   = :slayer_fitzpatrick
    scan_mode::Symbol     = :amr
    coupling_mode::Symbol = :uncoupled
    dc_type::Symbol       = :none
    msing_max::Int        = 3

    bt::Union{Float64,Nothing} = nothing
    mu_i::Float64     = 2.0
    zeff::Float64     = 1.0
    chi_perp::Float64 = 1.0
    chi_tor::Float64  = 1.0
    dr_val::Float64   = 0.0
    dgeo_val::Float64 = 0.0
    theta_sample::Float64 = 0.0

    Q_re_range::Tuple{Float64,Float64} = (-10.0, 10.0)
    Q_im_range::Tuple{Float64,Float64} = (-2.0, 5.0)
    nre::Int = 41
    nim::Int = 31

    amr_passes::Int    = 4
    amr_max_cells::Int = 10_000_000

    # Multi-box stripe layout. When non-empty, `scan_mode=:amr` dispatches to
    # `multi_box_amr_scan` instead of single-box `amr_scan`. Each entry is a
    # dimensionless Q-space rectangle as `(omega_lo, omega_hi, gamma_lo,
    # gamma_hi)`. Activity criteria fire on Re(Δ) sign change, Im(Δ) sign
    # change, OR |Δ| ≥ pre-screen pole threshold. A typical 25-kHz stripe
    # layout for DIII-D-style equilibria (with kHz/Q given by the per-surface
    # τ_k, see run_julia_betascan.jl) is built externally by the driver,
    # converted to Q-units, and passed in here.
    boxes::Vector{NTuple{4, Float64}} = NTuple{4, Float64}[]
    multi_box_prescreen_n::Int = 25         # pre-screen grid resolution per box

    pole_threshold::Float64    = 10.0
    pole_threshold_adaptive::Bool = false
    filter_above_poles::Bool   = true
    filter_outside_re::Bool    = true
    gap_kHz_threshold::Float64 = 1.0       # forwarded to find_growth_rates

    profile_source::Symbol = :inline
    profile_file::String   = ""
    profile_group::String  = "/"

    store_scan::Bool = false
end

const _VALID_INNER_MODELS   = (:slayer_fitzpatrick, :ggj_shooting, :ggj_galerkin)
const _VALID_SCAN_MODES     = (:amr, :brute_force)
const _VALID_COUPLING_MODES = (:uncoupled, :coupled)
const _VALID_DC_TYPES       = (:none, :lar, :rfitzp, :toroidal)
const _VALID_PROFILE_SOURCES = (:inline, :h5)

function validate(ctrl::SLAYERControl)
    ctrl.inner_model   in _VALID_INNER_MODELS   ||
        throw(ArgumentError("SLAYERControl: inner_model=$(ctrl.inner_model) " *
                             "not in $(_VALID_INNER_MODELS)"))
    ctrl.scan_mode     in _VALID_SCAN_MODES     ||
        throw(ArgumentError("SLAYERControl: scan_mode=$(ctrl.scan_mode) " *
                             "not in $(_VALID_SCAN_MODES)"))
    ctrl.coupling_mode in _VALID_COUPLING_MODES ||
        throw(ArgumentError("SLAYERControl: coupling_mode=$(ctrl.coupling_mode) " *
                             "not in $(_VALID_COUPLING_MODES)"))
    ctrl.dc_type       in _VALID_DC_TYPES       ||
        throw(ArgumentError("SLAYERControl: dc_type=$(ctrl.dc_type) " *
                             "not in $(_VALID_DC_TYPES)"))
    ctrl.profile_source in _VALID_PROFILE_SOURCES ||
        throw(ArgumentError("SLAYERControl: profile_source=$(ctrl.profile_source) " *
                             "not in $(_VALID_PROFILE_SOURCES)"))
    ctrl.msing_max >= 1 ||
        throw(ArgumentError("SLAYERControl: msing_max=$(ctrl.msing_max) must be ≥ 1"))
    ctrl.nre >= 2 && ctrl.nim >= 2 ||
        throw(ArgumentError("SLAYERControl: nre and nim must both be ≥ 2"))
    ctrl.amr_passes >= 0 ||
        throw(ArgumentError("SLAYERControl: amr_passes must be ≥ 0"))
    return ctrl
end

# Helper: coerce range-like values to a 2-tuple of Float64
_as_range(x::NTuple{2,<:Real}) = (Float64(x[1]), Float64(x[2]))
_as_range(x::AbstractVector)   = begin
    length(x) == 2 || throw(ArgumentError("range must be length 2, got length $(length(x))"))
    (Float64(x[1]), Float64(x[2]))
end

"""
    slayer_control_from_toml(section::AbstractDict) -> SLAYERControl

Parse a `[SLAYER]` TOML section into a `SLAYERControl`. Known nested
subsections (`[SLAYER.scan_grid]`, `[SLAYER.amr]`,
`[SLAYER.growth_rate_filter]`) are flattened into the top-level fields.
Unknown keys raise an error so typos don't silently produce defaults.
"""
function slayer_control_from_toml(section::AbstractDict)
    # Flatten nested sections into the top-level key dictionary
    flat = Dict{String,Any}()
    for (k, v) in section
        if k == "scan_grid" && v isa AbstractDict
            # Promote scan_grid fields to top-level
            haskey(v, "Q_re_range") && (flat["Q_re_range"] = v["Q_re_range"])
            haskey(v, "Q_im_range") && (flat["Q_im_range"] = v["Q_im_range"])
            haskey(v, "nre") && (flat["nre"] = v["nre"])
            haskey(v, "nim") && (flat["nim"] = v["nim"])
        elseif k == "amr" && v isa AbstractDict
            haskey(v, "passes")    && (flat["amr_passes"]    = v["passes"])
            haskey(v, "max_cells") && (flat["amr_max_cells"] = v["max_cells"])
        elseif k == "growth_rate_filter" && v isa AbstractDict
            haskey(v, "pole_threshold")     && (flat["pole_threshold"]     = v["pole_threshold"])
            haskey(v, "filter_above_poles") && (flat["filter_above_poles"] = v["filter_above_poles"])
            haskey(v, "filter_outside_re")  && (flat["filter_outside_re"]  = v["filter_outside_re"])
        elseif k == "profiles"
            # Profiles are handled separately by the runner; skip here
            continue
        else
            flat[k] = v
        end
    end

    # Validate keys against the struct fields
    field_names = Set(String.(fieldnames(SLAYERControl)))
    unknown     = [k for k in keys(flat) if !(k in field_names)]
    isempty(unknown) ||
        throw(ArgumentError("slayer_control_from_toml: unknown keys " *
                             "$(unknown) in [SLAYER] section. Known: " *
                             "$(sort(collect(field_names)))."))

    # Coerce types where needed
    kwargs = Dict{Symbol,Any}()
    for (k, v) in flat
        sym = Symbol(k)
        if sym in (:inner_model, :scan_mode, :coupling_mode, :dc_type,
                   :profile_source)
            kwargs[sym] = v isa Symbol ? v : Symbol(String(v))
        elseif sym in (:Q_re_range, :Q_im_range)
            kwargs[sym] = _as_range(v)
        elseif sym === :bt
            # Allow explicit nothing or a number
            kwargs[sym] = v === nothing ? nothing : Float64(v)
        elseif sym === :boxes
            # `boxes` is a Vector{NTuple{4,Float64}}; from TOML this comes
            # in as a list of 4-element arrays. Coerce each.
            kwargs[sym] = NTuple{4,Float64}[
                let bb = collect(Float64, b)
                    length(bb) == 4 ||
                        throw(ArgumentError("SLAYER.boxes entry must have 4 " *
                                             "elements (omega_lo, omega_hi, " *
                                             "gamma_lo, gamma_hi); got $b"))
                    (bb[1], bb[2], bb[3], bb[4])
                end
                for b in v
            ]
        else
            kwargs[sym] = v
        end
    end
    return validate(SLAYERControl(; kwargs...))
end
