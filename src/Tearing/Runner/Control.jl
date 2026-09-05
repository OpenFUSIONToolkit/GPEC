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
    `:rfitzp`, `:toroidal` (χ_‖-matching critical-Δ formulas,
    Connor-Hastie-Helander 2015)
  - `msing_max`     -- number of surfaces to include in the coupled
    determinant (default 3; capped at `length(sings)` at runtime)

# Physics knobs

  - `bt`       -- toroidal field `[T]`. `nothing` (default) resolves the physical
    `B_T = F(ψ)/(2π·R₀)` per surface from the equilibrium's F-spline; a scalar or a
    callable of `psi` overrides it
  - `omega_E_kHz` -- per-surface E×B rotation frequency in kHz, ordered
    core→edge. Entry `k` shifts surface `k`'s inner-layer Q argument by
    `ΔRe(Q_k) = −tauk_k · 2π · 1e3 · omega_E_kHz[k]`, so that layer responds to
    the mode frequency in its own fluid frame (TJ's `ĝ = i(Q_E − ω·tau_k)`).
    Empty (default) means no rotation on any surface. Must be empty or have
    one entry per rational surface actually analysed
  - `tauk_rescale` -- direction of the inter-surface Q normalization in the
    coupled determinant: `:direct` (default, `Q·tauk_k/tauk_ref`, consistent
    with `Q = tauk·omega` and with TJ) or `:legacy` (`Q·tauk_ref/tauk_k`, the
    pre-correction behaviour, for reproducing older results only)
  - `mu_i`     -- ion mass in proton-mass units (default 2.0 for D)
  - `zeff`     -- effective charge
  - `chi_perp`, `chi_tor` -- fallback scalars [m²/s], used only when the
    kinetic file carries no usable `chi_e`/`chi_phi` profile (dataset absent
    or all-zero); otherwise the file's χ⊥(ψ)/χ_φ(ψ) take precedence.
    `chi_perp` is the perpendicular ENERGY diffusivity χ⊥; `chi_tor` is
    Fitzpatrick's χ_φ, the anomalous perpendicular ion MOMENTUM diffusivity
    (a viscosity: it enters the vorticity and parallel-flow equations, TJ
    Layer.tex — not a heat diffusivity)
  - `dr_val`, `dgeo_val`  -- critical-Δ formula inputs. `nothing` (default)
    auto-derives them from the equilibrium: `dr_val` from the resistive
    interchange index `D_R = E + F + H²` at each surface, `dgeo_val` from the
    Connor et al. 2015 Eq. 59 toroidal geometric factor in the `r_s` reference
    (consumed only by `dc_type=:toroidal`). Supply a
    scalar only to override the auto-derivation; an explicit `0.0` disables the
    critical-Δ offset (Δ_crit ≡ 0)
  - `theta_sample` -- poloidal angle at which to sample minor radius
    (default 0.0, outboard midplane)
  - `resistivity_model` -- η closure setting τ_R = μ₀r_s²/η: `:sauter`
    (default, neoclassical F_33), `:redl`, `:spitzer` (Sauter 18a, no
    trapped correction), or `:spitzer_harm` (legacy Fitzpatrick σ_∥)
  - `lnLambda_form` -- Coulomb-log form: `:nrl` (default), `:sauter`, or
    `:wesson` (legacy; pair with `:spitzer_harm` to reproduce old SLAYER)

# Scan grid (used for both brute-force and AMR initial mesh)

  - `Q_re_range`, `Q_im_range` -- box in the normalized Q plane
  - `nre`, `nim`    -- grid resolution along each axis

# AMR refinement

  - `amr_passes`    -- max refinement levels
  - `amr_max_cells` -- hard safety cap

# Growth-rate-extraction filters

  - `pole_threshold`      -- threshold for pole classification (default 10)
  - `pole_threshold_adaptive` -- if true, pole_threshold is OVERRIDDEN per
    scan with `10 × median(|Δ|)` over the scan grid. The median is robust
    where `|mean(Δ)|` is not: it resists inflation from the near-pole
    samples that dominate a mean, and avoids the phase-cancellation that
    can shrink `|mean|`. Useful when |Δ| spans 8+ orders of magnitude
    (e.g. SLAYER scans where the hardcoded 10.0 default is too restrictive
    and classifies all intersections as poles).
  - `filter_above_poles`  -- discard roots above the highest pole γ
  - `filter_outside_re`   -- condition the above-pole filter on the +γ
    step exiting the Re(Δ)=0 contour loop

# Kinetic-profile source

SLAYER reads kinetic profiles through the shared `Equilibrium.read_kinetic_file`
reader (the same standardized object used by the kinetic/NTV physics), so
there is one consistent interface for resistive and kinetic profiles.

  - `profile_file`   -- path to a kinetic-profile file (relative to the run
    dir), required when SLAYER is enabled. HDF5 (`.h5`) files use the GPEC
    kinetic schema and may carry `chi_e` (χ⊥) and `chi_phi` (χ_φ); ASCII
    tables are also accepted but carry no χ. When a χ dataset is absent or
    all-zero, the scalar `chi_perp`/`chi_tor` fallbacks below are used (so a
    file can keep the χ keys set to 0 to defer to the scalars).
  - `profile_group`  -- group within the HDF5 file (default `"/"`)

# Output control

  - `store_scan`  -- write the full Q/Δ scan grid to HDF5. `false` by
    default to keep the output file small.
"""
@kwdef struct SLAYERControl
    enabled::Bool = false

    inner_model::Symbol = :slayer_fitzpatrick
    scan_mode::Symbol = :amr
    coupling_mode::Symbol = :uncoupled
    dc_type::Symbol = :none
    msing_max::Int = 3

    bt::Union{Float64,Nothing} = nothing
    mu_i::Float64 = 2.0
    zeff::Float64 = 1.0
    chi_perp::Float64 = 1.0
    chi_tor::Float64 = 1.0
    dr_val::Union{Float64,Nothing} = nothing
    dgeo_val::Union{Float64,Nothing} = nothing
    theta_sample::Float64 = 0.0
    resistivity_model::Symbol = :sauter
    lnLambda_form::Symbol = :nrl

    Q_re_range::Tuple{Float64,Float64} = (-10.0, 10.0)
    Q_im_range::Tuple{Float64,Float64} = (-2.0, 5.0)
    nre::Int = 41
    nim::Int = 31

    amr_passes::Int = 4
    amr_max_cells::Int = 10_000_000

    # Multi-box stripe layout. When non-empty, `scan_mode=:amr` dispatches to
    # `multi_box_amr_scan` instead of single-box `amr_scan`. Each entry is a
    # dimensionless Q-space rectangle as `(omega_lo, omega_hi, gamma_lo,
    # gamma_hi)`. Activity criteria fire on Re(Δ) sign change, Im(Δ) sign
    # change, OR |Δ| ≥ pre-screen pole threshold. A typical 25-kHz stripe
    # layout for DIII-D-style equilibria (with kHz/Q given by the per-surface
    # τ_k) is built externally by the driver, converted to Q-units, and
    # passed in here.
    boxes::Vector{NTuple{4,Float64}} = NTuple{4,Float64}[]
    multi_box_prescreen_n::Int = 25         # pre-screen grid resolution per box

    pole_threshold::Float64 = 10.0
    pole_threshold_adaptive::Bool = false
    filter_above_poles::Bool = true
    filter_outside_re::Bool = true
    gap_kHz_threshold::Float64 = 1.0       # forwarded to find_growth_rates
    # Refine each contour-intersection root to the true zero of the dispersion
    # residual (isolate-then-polish). Makes the extracted γ resolution- and
    # thread-independent; adds ~a handful of residual evals per root. Set false
    # to report the raw marching-squares estimate. Applies to the UNCOUPLED
    # (per-surface scalar) path; the coupled determinant is ill-conditioned and
    # uses raw contour extraction regardless.
    polish_roots::Bool = true
    # Validity gate (uncoupled + polish_roots only): drop a root whose polished
    # |residual| is not below validity_rtol × median(|Δ|) — a contour near-miss
    # / failed-Δ'-BVP surface, not a real root. Flagged `:spurious`.
    validity_rtol::Float64 = 1e-3

    # Per-surface E×B rotation frequency [kHz], ordered core→edge to match the
    # rational-surface list. Entry k shifts surface k's inner layer by
    # ΔRe(Q_k) = −tauk_k · 2π · 1e3 · omega_E_kHz[k], so each rotating surface
    # sees the common lab-frame eigenvalue in its own fluid frame. Empty (the
    # default) means no rotation anywhere. A shorter-than-msing vector is an
    # error; pad with zeros to leave some surfaces static.
    omega_E_kHz::Vector{Float64} = Float64[]

    # Inter-surface Q normalization in the coupled determinant. `:direct`
    # (default) uses Q·tauk_k/tauk_ref: Q is defined as tauk·omega, so one
    # shared physical eigenvalue maps to surface k by MULTIPLYING by tauk_k.
    # `:legacy` restores the previous Q·tauk_ref/tauk_k for reproducing results
    # generated before the correction; it puts the scanned frequency and that
    # surface's Q_e/Q_i on opposite tauk scalings inside the same layer solve.
    tauk_rescale::Symbol = :direct

    profile_file::String = ""
    profile_group::String = "/"

    store_scan::Bool = false
end

const _VALID_INNER_MODELS = (:slayer_fitzpatrick, :ggj_shooting, :ggj_galerkin)
const _VALID_SCAN_MODES = (:amr, :brute_force)
const _VALID_COUPLING_MODES = (:uncoupled, :coupled)
const _VALID_DC_TYPES = (:none, :lar, :rfitzp, :toroidal)
const _VALID_RESISTIVITY_MODELS = (:sauter, :redl, :spitzer, :spitzer_harm)
const _VALID_LNLAMBDA_FORMS = (:nrl, :sauter, :wesson)
const _VALID_TAUK_RESCALE = (:legacy, :direct)

function validate(ctrl::SLAYERControl)
    ctrl.inner_model in _VALID_INNER_MODELS ||
        throw(ArgumentError("SLAYERControl: inner_model=$(ctrl.inner_model) " *
                            "not in $(_VALID_INNER_MODELS)"))
    ctrl.scan_mode in _VALID_SCAN_MODES ||
        throw(ArgumentError("SLAYERControl: scan_mode=$(ctrl.scan_mode) " *
                            "not in $(_VALID_SCAN_MODES)"))
    ctrl.coupling_mode in _VALID_COUPLING_MODES ||
        throw(ArgumentError("SLAYERControl: coupling_mode=$(ctrl.coupling_mode) " *
                            "not in $(_VALID_COUPLING_MODES)"))
    ctrl.dc_type in _VALID_DC_TYPES ||
        throw(ArgumentError("SLAYERControl: dc_type=$(ctrl.dc_type) " *
                            "not in $(_VALID_DC_TYPES)"))
    ctrl.resistivity_model in _VALID_RESISTIVITY_MODELS ||
        throw(ArgumentError("SLAYERControl: resistivity_model=$(ctrl.resistivity_model) " *
                            "not in $(_VALID_RESISTIVITY_MODELS)"))
    ctrl.lnLambda_form in _VALID_LNLAMBDA_FORMS ||
        throw(ArgumentError("SLAYERControl: lnLambda_form=$(ctrl.lnLambda_form) " *
                            "not in $(_VALID_LNLAMBDA_FORMS)"))
    ctrl.msing_max >= 1 ||
        throw(ArgumentError("SLAYERControl: msing_max=$(ctrl.msing_max) must be ≥ 1"))
    all(isfinite, ctrl.omega_E_kHz) ||
        throw(ArgumentError("SLAYERControl: omega_E_kHz contains a " *
                            "non-finite entry: $(ctrl.omega_E_kHz)"))
    ctrl.tauk_rescale in _VALID_TAUK_RESCALE ||
        throw(ArgumentError("SLAYERControl: tauk_rescale=$(ctrl.tauk_rescale) " *
                            "must be one of $(_VALID_TAUK_RESCALE)"))
    ctrl.nre >= 2 && ctrl.nim >= 2 ||
        throw(ArgumentError("SLAYERControl: nre and nim must both be ≥ 2"))
    ctrl.amr_passes >= 0 ||
        throw(ArgumentError("SLAYERControl: amr_passes must be ≥ 0"))
    return ctrl
end

# Helper: coerce range-like values to a 2-tuple of Float64
_as_range(x::NTuple{2,<:Real}) = (Float64(x[1]), Float64(x[2]))
_as_range(x::AbstractVector) = begin
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
            haskey(v, "passes") && (flat["amr_passes"] = v["passes"])
            haskey(v, "max_cells") && (flat["amr_max_cells"] = v["max_cells"])
        elseif k == "growth_rate_filter" && v isa AbstractDict
            haskey(v, "pole_threshold") && (flat["pole_threshold"] = v["pole_threshold"])
            haskey(v, "filter_above_poles") && (flat["filter_above_poles"] = v["filter_above_poles"])
            haskey(v, "filter_outside_re") && (flat["filter_outside_re"] = v["filter_outside_re"])
        else
            flat[k] = v
        end
    end

    # Validate keys against the struct fields
    field_names = Set(String.(fieldnames(SLAYERControl)))
    unknown = [k for k in keys(flat) if !(k in field_names)]
    isempty(unknown) ||
        throw(ArgumentError("slayer_control_from_toml: unknown keys " *
                            "$(unknown) in [SLAYER] section. Known: " *
                            "$(sort(collect(field_names)))."))

    # Coerce types where needed
    kwargs = Dict{Symbol,Any}()
    for (k, v) in flat
        sym = Symbol(k)
        if sym in (:inner_model, :scan_mode, :coupling_mode, :dc_type,
            :resistivity_model, :lnLambda_form, :tauk_rescale)
            kwargs[sym] = v isa Symbol ? v : Symbol(String(v))
        elseif sym in (:Q_re_range, :Q_im_range)
            kwargs[sym] = _as_range(v)
        elseif sym in (:bt, :dr_val, :dgeo_val)
            # Allow explicit nothing (auto-derive) or a number (override)
            kwargs[sym] = v === nothing ? nothing : Float64(v)
        elseif sym === :omega_E_kHz
            kwargs[sym] = Float64[Float64(x) for x in v]
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
