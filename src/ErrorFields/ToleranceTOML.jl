"""
    ToleranceTOML

The manufacturing-tolerance input of an error-field assessment: a TOML file, separate from
`gpec.toml` and named by `ErrorFieldsControl.tolerance_file`, holding one `[[ErrorFields.coil]]`
block per coil set that may move on its own, `[[ErrorFields.coherent_group]]` blocks for coil
sets that move together, and the `[ErrorFields.correctability]` and `[ErrorFields.other_field]`
tables. Tolerances are human-authored engineering data, so the format is diffable, commentable
text; the raw file is echoed into `Input/RawInputs/ErrorFields/tolerance_toml_raw` for replay.

```toml
[ErrorFields.defaults]           # fallbacks for keys a coil block leaves out (all optional)
tilt_units = "deg"
radial_shape = "hollow"

[[ErrorFields.coil]]
name = "PF1U"                    # a coil set name of the run
shift_tol_mm = 0.5               # radius of the in-plane displacement disk the coil may sit in
tilt_tol = 0.019                 # tilt tolerance, in tilt_units
tolerance_model = "cylinder"     # axis line inside a cylinder: shift and tilt drawn together
cylinder_half_height_m = 1.5

[[ErrorFields.coherent_group]]
name = "upper_pf_brace"
members = ["PF1U", "PF2U"]       # move together with one shared draw per sample
shift_tol_mm = 2.0
tilt_tol = 0.0
phase_group = "braces"           # groups naming the same phase_group share the random direction

[ErrorFields.correctability]
uncorrectable_coils = ["EFCC_U"] # excluded from the error-field-correction division
efc_factor = 2.0

[ErrorFields.other_field]
magnitude = 8.2e-6               # overlap budget of sources not attributed to any coil
```
"""

const RADIAL_SHAPES = ("flat", "uniform_area", "hollow", "ring")
const TOLERANCE_MODELS = ("additive", "cylinder")
const TILT_UNITS = ("deg", "m")

"""
    CoilTolerance

Manufacturing tolerance of one coil set that moves independently. Lengths are in metres and
the tilt tolerance in `tilt_units`; convert it to the degrees the sensitivities are stored in
with [`tilt_tolerance_deg`](@ref).

## Fields

  - `name`: coil set name; must match a coil set of the run
  - `shift_tol_m`: radius of the in-plane displacement disk the coil centre may lie in, metres
  - `shift_sigma_m`: Gaussian uncertainty added to the sampled shift, metres
  - `tilt_tol`: tilt tolerance, in `tilt_units`
  - `tilt_sigma`: Gaussian uncertainty added to the sampled tilt, in `tilt_units`
  - `tilt_units`: `"deg"`, or `"m"` for a rim displacement converted through the coil's
    nominal major radius
  - `radial_shape`: radial sampling density on the disk: `"flat"` (uniform in radius),
    `"uniform_area"`, `"hollow"` (peaked toward the edge), or `"ring"` (always at the edge)
  - `tolerance_model`: `"additive"` (independent shift and tilt draws) or `"cylinder"` (the
    coil axis line confined to a cylinder of radius `shift_tol_m`, shift and tilt drawn together)
  - `cylinder_half_height_m`: half-height of that cylinder, metres; sets the tilt reachable at
    the given radius
"""
struct CoilTolerance
    name::String
    shift_tol_m::Float64
    shift_sigma_m::Float64
    tilt_tol::Float64
    tilt_sigma::Float64
    tilt_units::String
    radial_shape::String
    tolerance_model::String
    cylinder_half_height_m::Float64
end

"""
    CoherentGroupTolerance

Tolerance of a set of coils that move together: every member receives the same displacement
(and the same rotation about a common pivot) from one random draw per sample. Groups that name
the same `phase_group` share the random direction of that draw while keeping independent
amplitudes.

## Fields

  - `name`: label of the group
  - `members`: coil set names moving together
  - `shift_tol_m`, `tilt_tol`, `tilt_units`, `radial_shape`, `tolerance_model`,
    `cylinder_half_height_m`: as in [`CoilTolerance`](@ref), applied to the shared draw
  - `phase_group`: groups with equal labels share the random direction; defaults to `name`
  - `rotation_center_z_m`: height of the pivot of the group's rigid rotation on the machine axis,
    metres; a member at height `z` tilted by θ also shifts laterally by `(z − z_pivot)·θ`
"""
struct CoherentGroupTolerance
    name::String
    members::Vector{String}
    shift_tol_m::Float64
    tilt_tol::Float64
    tilt_units::String
    radial_shape::String
    tolerance_model::String
    cylinder_half_height_m::Float64
    phase_group::String
    rotation_center_z_m::Float64
end

"""
    OtherFieldBudget

Overlap budget for error-field sources not attributed to any coil, sampled once per Monte
Carlo sample as an incoherent contribution with a random direction.

## Fields

  - `magnitude`: dimensionless overlap magnitude of the budget
  - `sigma`: Gaussian uncertainty on that magnitude
  - `radial_shape`: radial sampling density of the magnitude, as in [`CoilTolerance`](@ref)
"""
struct OtherFieldBudget
    magnitude::Float64
    sigma::Float64
    radial_shape::String
end

"""
    ToleranceSet

Everything a tolerance TOML file specifies, parsed and validated by [`read_tolerance_toml`](@ref).

## Fields

  - `coils`: independently moving coil sets `[ncoil]`
  - `groups`: coherently moving groups `[ngroup]`
  - `uncorrectable_coils`: coil set names whose error field the correction system cannot reduce
  - `efc_factor`: divisor applied to the correctable contributions when error-field correction is on
  - `other_field`: the unattributed budget
  - `raw`: the file's text, echoed into the output for replay
"""
struct ToleranceSet
    coils::Vector{CoilTolerance}
    groups::Vector{CoherentGroupTolerance}
    uncorrectable_coils::Vector{String}
    efc_factor::Float64
    other_field::OtherFieldBudget
    raw::String
end

# Built-in fallbacks for the per-coil keys, overridable by [ErrorFields.defaults]. The radial
# shape and the other-field shape follow the OMFIT tool's defaults.
const _COIL_KEY_DEFAULTS = Dict{String,Any}(
    "shift_sigma_mm" => 0.0, "tilt_sigma" => 0.0, "tilt_units" => "deg", "radial_shape" => "hollow",
    "tolerance_model" => "additive", "cylinder_half_height_m" => 0.0
)
const _GROUP_ONLY_KEYS = ("members", "phase_group", "rotation_center_z_m")

"""
    read_tolerance_toml(path) -> ToleranceSet
    parse_tolerance_toml(text) -> ToleranceSet

Read and validate a tolerance TOML file (see the module docstring for the schema). Every key is
checked against the schema so a misspelled key errors instead of silently taking a default;
tolerances must be non-negative, names unique, enumerated fields one of their allowed values,
`efc_factor ≥ 1`, and a cylinder model needs a positive half-height. Names are checked against
the run's coil sets separately by [`validate_tolerances`](@ref).
"""
read_tolerance_toml(path::AbstractString) = parse_tolerance_toml(read(path, String))

function parse_tolerance_toml(text::AbstractString)
    root = TOML.parse(String(text))
    haskey(root, "ErrorFields") || throw(ArgumentError("tolerance TOML has no [ErrorFields] tables (blocks are [[ErrorFields.coil]], ...)"))
    ef = root["ErrorFields"]
    _check_keys(ef, ("defaults", "coil", "coherent_group", "correctability", "other_field"), "ErrorFields")

    defaults = merge(_COIL_KEY_DEFAULTS, get(ef, "defaults", Dict{String,Any}()))
    _check_keys(defaults, collect(keys(_COIL_KEY_DEFAULTS)), "ErrorFields.defaults")

    coils = [_parse_coil(d, defaults) for d in get(ef, "coil", Dict{String,Any}[])]
    groups = [_parse_group(d, defaults) for d in get(ef, "coherent_group", Dict{String,Any}[])]
    _unique_names([c.name for c in coils], "[[ErrorFields.coil]]")
    _unique_names([g.name for g in groups], "[[ErrorFields.coherent_group]]")

    corr = get(ef, "correctability", Dict{String,Any}())
    _check_keys(corr, ("uncorrectable_coils", "efc_factor"), "ErrorFields.correctability")
    uncorrectable = String.(get(corr, "uncorrectable_coils", String[]))
    efc_factor = Float64(get(corr, "efc_factor", 2.0))
    efc_factor >= 1 || throw(ArgumentError("efc_factor must be ≥ 1 (got $efc_factor)"))

    other = get(ef, "other_field", Dict{String,Any}())
    _check_keys(other, ("magnitude", "sigma", "radial_shape"), "ErrorFields.other_field")
    other_field = OtherFieldBudget(_nonneg(get(other, "magnitude", 0.0), "other_field.magnitude"),
        _nonneg(get(other, "sigma", 0.0), "other_field.sigma"),
        _enum(get(other, "radial_shape", "ring"), RADIAL_SHAPES, "other_field.radial_shape"))

    return ToleranceSet(coils, groups, uncorrectable, efc_factor, other_field, String(text))
end

function _parse_coil(d::Dict{String,Any}, defaults::Dict{String,Any})
    _check_keys(d, vcat("name", "shift_tol_mm", "tilt_tol", collect(keys(_COIL_KEY_DEFAULTS))), "[[ErrorFields.coil]]")
    haskey(d, "name") || throw(ArgumentError("a [[ErrorFields.coil]] block has no name"))
    name = String(d["name"])
    get_key(k) = get(d, k, defaults[k])
    model = _enum(get_key("tolerance_model"), TOLERANCE_MODELS, "$name.tolerance_model")
    half_height = _nonneg(get_key("cylinder_half_height_m"), "$name.cylinder_half_height_m")
    model == "cylinder" && half_height <= 0 && throw(ArgumentError("coil $name: the cylinder model needs cylinder_half_height_m > 0"))
    return CoilTolerance(name,
        1e-3 * _nonneg(get(d, "shift_tol_mm", 0.0), "$name.shift_tol_mm"),
        1e-3 * _nonneg(get_key("shift_sigma_mm"), "$name.shift_sigma_mm"),
        _nonneg(get(d, "tilt_tol", 0.0), "$name.tilt_tol"),
        _nonneg(get_key("tilt_sigma"), "$name.tilt_sigma"),
        _enum(get_key("tilt_units"), TILT_UNITS, "$name.tilt_units"),
        _enum(get_key("radial_shape"), RADIAL_SHAPES, "$name.radial_shape"),
        model, half_height)
end

function _parse_group(d::Dict{String,Any}, defaults::Dict{String,Any})
    _check_keys(d, vcat("name", "shift_tol_mm", "tilt_tol", "tilt_units", "radial_shape", "tolerance_model",
            "cylinder_half_height_m", collect(_GROUP_ONLY_KEYS)), "[[ErrorFields.coherent_group]]")
    haskey(d, "name") || throw(ArgumentError("a [[ErrorFields.coherent_group]] block has no name"))
    name = String(d["name"])
    members = String.(get(d, "members", String[]))
    isempty(members) && throw(ArgumentError("coherent group $name lists no members"))
    _unique_names(members, "coherent group $name members")
    get_key(k) = get(d, k, defaults[k])
    model = _enum(get_key("tolerance_model"), TOLERANCE_MODELS, "$name.tolerance_model")
    half_height = _nonneg(get_key("cylinder_half_height_m"), "$name.cylinder_half_height_m")
    model == "cylinder" && half_height <= 0 && throw(ArgumentError("coherent group $name: the cylinder model needs cylinder_half_height_m > 0"))
    return CoherentGroupTolerance(name, members,
        1e-3 * _nonneg(get(d, "shift_tol_mm", 0.0), "$name.shift_tol_mm"),
        _nonneg(get(d, "tilt_tol", 0.0), "$name.tilt_tol"),
        _enum(get_key("tilt_units"), TILT_UNITS, "$name.tilt_units"),
        _enum(get_key("radial_shape"), RADIAL_SHAPES, "$name.radial_shape"),
        model, half_height,
        String(get(d, "phase_group", name)),
        Float64(get(d, "rotation_center_z_m", 0.0)))
end

function _check_keys(d::Dict{String,Any}, allowed, where::String)
    for k in keys(d)
        k in allowed || throw(ArgumentError("unknown key \"$k\" in $where (allowed: $(join(allowed, ", ")))"))
    end
end

function _unique_names(names, where::String)
    length(unique(names)) == length(names) || throw(ArgumentError("duplicate names in $where: $(join(unique(filter(n -> count(==(n), names) > 1, names)), ", "))"))
end

function _nonneg(v, what::String)
    x = Float64(v)
    x >= 0 || throw(ArgumentError("$what must be ≥ 0 (got $x)"))
    return x
end

function _enum(v, allowed, what::String)
    s = lowercase(String(v))
    s in allowed || throw(ArgumentError("$what must be one of $(join(allowed, ", ")) (got \"$v\")"))
    return s
end

"""
    validate_tolerances(ts::ToleranceSet, coil_names) -> ts

Check every coil, group member, and uncorrectable-coil name of `ts` against the coil set names
of a run (for instance `CoilSensitivities.coil_names`), throwing an `ArgumentError` that lists
the unknown names. Coil sets of the run that carry no tolerance are allowed: they contribute
only their nominal overlap.
"""
function validate_tolerances(ts::ToleranceSet, coil_names::AbstractVector{<:AbstractString})
    known = Set(String.(coil_names))
    unknown = String[]
    for c in ts.coils
        c.name in known || push!(unknown, "coil $(c.name)")
    end
    for g in ts.groups, m in g.members
        m in known || push!(unknown, "member $m of group $(g.name)")
    end
    for u in ts.uncorrectable_coils
        u in known || push!(unknown, "uncorrectable coil $u")
    end
    isempty(unknown) || throw(ArgumentError("tolerance names not among the run's coil sets ($(join(sort(collect(known)), ", "))): $(join(unknown, "; "))"))
    return ts
end

"""
    tilt_tolerance_deg(tilt, tilt_units, cs::CoilSet) -> Float64
    tilt_tolerance_deg(t::CoilTolerance, cs::CoilSet) -> Float64

A tilt tolerance in the degrees the stored sensitivities use. `"deg"` passes through; `"m"` is
a rim displacement converted through the coil set's arc-length-weighted major radius exactly as
`apply_transforms` converts `tilt_in_meters`: `asin(t / R_nom)`.
"""
function tilt_tolerance_deg(tilt::Real, tilt_units::AbstractString, cs::CoilSet)
    tilt_units == "deg" && return Float64(tilt)
    r_nom = ForcingTerms.nominal_major_radius(cs)
    tilt <= r_nom || throw(ArgumentError("tilt $tilt m exceeds the nominal radius $r_nom m of coil set $(cs.name)"))
    return rad2deg(asin(tilt / r_nom))
end
tilt_tolerance_deg(t::CoilTolerance, cs::CoilSet) = tilt_tolerance_deg(t.tilt_tol, t.tilt_units, cs)
