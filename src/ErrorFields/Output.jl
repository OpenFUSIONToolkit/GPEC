"""
    Output

HDF5 output for the ErrorFields coil linearization, under `ErrorFields/CoilSensitivities/`,
and the reader that rebuilds a `CoilSensitivities` from it.
"""

const _H5_GROUP = "ErrorFields/CoilSensitivities"

# Metadata table for ErrorFields/CoilSensitivities/ (paths relative to the group). Spectra are
# root-area-weighted control-surface fields b̃ on the Info/mn_index ordering; the DominantMode/
# summary is the full-window mode-1 projection, the windowed table being a post-hoc analysis.
const EF_H5_ANNOTATIONS = [
    "coil_name" => (; long_name="name of each coil set"),
    "nominal_field" => (; long_name="root-area-weighted control-surface field b̃ of each coil set as built", units="T", dims=("mode", "coil_set")),
    "shift_sensitivity" => (; long_name="∂b̃/∂(Δx, Δy, Δz) of each coil set under a rigid Cartesian shift", units="T/m", dims=("mode", "axis", "coil_set")),
    "tilt_sensitivity" => (; long_name="∂b̃/∂(θx, θy, θz) of each coil set under a rigid rotation about the machine axes", units="T/deg", dims=("mode", "axis", "coil_set")),
    "shift_linearity_residual" =>
        (; long_name="finite-difference curvature ‖b̃(+h)+b̃(−h)−2b̃(0)‖ of each shift tap relative to the set's largest first difference", dims=("axis", "coil_set")),
    "tilt_linearity_residual" =>
        (; long_name="finite-difference curvature ‖b̃(+h)+b̃(−h)−2b̃(0)‖ of each tilt tap relative to the set's largest first difference", dims=("axis", "coil_set")),
    "peak_current" => (; long_name="largest conductor current magnitude of each coil set at which the spectra were evaluated", units="A"),
    "winding_multiplier" => (; long_name="turns per conductor element of each coil set"),
    "DominantMode/delta_nominal" => (; long_name="overlap δ = Vᴴ₁·b̃ / B_T0 of each coil set with the full-window dominant mode"),
    "DominantMode/shift_sensitivity" => (; long_name="∂δ/∂(Δx, Δy, Δz) on the full-window dominant mode", units="1/m", dims=("axis", "coil_set")),
    "DominantMode/tilt_sensitivity" => (; long_name="∂δ/∂(θx, θy, θz) on the full-window dominant mode", units="1/deg", dims=("axis", "coil_set")),
    "DominantMode/shift_rms" => (; long_name="direction-averaged in-plane shift sensitivity √((|∂δ/∂Δx|²+|∂δ/∂Δy|²)/2)", units="1/m"),
    "DominantMode/tilt_rms" => (; long_name="direction-averaged in-plane tilt sensitivity √((|∂δ/∂θx|²+|∂δ/∂θy|²)/2)", units="1/deg"),
    "DominantMode/cancelling_shift" => (; long_name="in-plane shift (Δx, Δy) that cancels delta_nominal to linear order", units="m", dims=("axis", "coil_set")),
    "DominantMode/cancelling_tilt" => (; long_name="in-plane tilt (θx, θy) that cancels delta_nominal to linear order", units="deg", dims=("axis", "coil_set"))
]

"""
    write_to_hdf5!(h5file::HDF5.File, sens::CoilSensitivities, dom::DominantCoupling)

Write the coil linearization to `ErrorFields/CoilSensitivities/` — the spectra and their
derivatives, plus a `DominantMode/` summary projected onto mode 1 of `dom` (the run's
full-window decomposition). An existing group is replaced. The mode labels are the file's
`Info/mn_index` and the field normalization its `Equilibrium/B_T_axis`, neither duplicated here.
"""
function write_to_hdf5!(h5file::HDF5.File, sens::CoilSensitivities, dom::DominantCoupling)
    haskey(h5file, _H5_GROUP) && delete_object(h5file, _H5_GROUP)
    g = create_group(h5file, _H5_GROUP)
    g["coil_name"] = sens.coil_names
    g["nominal_field"] = sens.nominal_field
    g["shift_sensitivity"] = sens.shift_sensitivity
    g["tilt_sensitivity"] = sens.tilt_sensitivity
    g["shift_linearity_residual"] = sens.shift_linearity_residual
    g["tilt_linearity_residual"] = sens.tilt_linearity_residual
    g["peak_current"] = sens.peak_current
    g["winding_multiplier"] = sens.winding_multiplier

    table = sensitivity_table(sens, dom; mode=1)
    d = create_group(g, "DominantMode")
    d["delta_nominal"] = table.delta_nominal
    d["shift_sensitivity"] = table.shift
    d["tilt_sensitivity"] = table.tilt
    d["shift_rms"] = table.shift_rms
    d["tilt_rms"] = table.tilt_rms
    d["cancelling_shift"] = table.cancelling_shift
    d["cancelling_tilt"] = table.cancelling_tilt

    Utilities.HDF5Annotations.annotate!(g, EF_H5_ANNOTATIONS)
    return g
end

"""
    CoilSensitivities(h5path::AbstractString)

Read the coil linearization back from a `gpec.h5` written with an `[ErrorFields]` section, with
the mode labels from `Info/mn_index` and the normalization field from `Equilibrium/B_T_axis`.
"""
function CoilSensitivities(h5path::AbstractString)
    h5open(h5path, "r") do f
        haskey(f, _H5_GROUP) || throw(ArgumentError("$h5path has no $_H5_GROUP group (run with an [ErrorFields] section)"))
        haskey(f, "Info/mn_index") || throw(ArgumentError("$h5path has no Info/mn_index mode labels"))
        haskey(f, "Equilibrium/B_T_axis") || throw(ArgumentError("$h5path has no Equilibrium/B_T_axis"))
        g = f[_H5_GROUP]
        mn = read(f["Info/mn_index"])
        return CoilSensitivities(
            read(g["coil_name"]), mn[:, 1], mn[:, 2], Float64(read(f["Equilibrium/B_T_axis"])),
            read(g["nominal_field"]), read(g["shift_sensitivity"]), read(g["tilt_sensitivity"]),
            read(g["shift_linearity_residual"]), read(g["tilt_linearity_residual"]),
            read(g["peak_current"]), read(g["winding_multiplier"])
        )
    end
end

const _TOLERANCE_SNAPSHOT = "Input/RawInputs/ErrorFields/tolerance_toml_raw"

"""
    write_tolerance_snapshot!(h5file, ts::ToleranceSet)

Echo the tolerance file's text into `Input/RawInputs/ErrorFields/tolerance_toml_raw`, the raw
input snapshot a replay reads back with [`read_tolerance_snapshot`](@ref). Replaces an existing echo.
"""
function write_tolerance_snapshot!(h5file::HDF5.File, ts::ToleranceSet)
    haskey(h5file, _TOLERANCE_SNAPSHOT) && delete_object(h5file, _TOLERANCE_SNAPSHOT)
    h5file[_TOLERANCE_SNAPSHOT] = ts.raw
    return h5file
end

"""
    read_tolerance_snapshot(h5path) -> ToleranceSet

The tolerance set a run used, parsed from the raw echo in its `gpec.h5`; `nothing` when the run
named no tolerance file.
"""
function read_tolerance_snapshot(h5path::AbstractString)
    h5open(h5path, "r") do f
        haskey(f, _TOLERANCE_SNAPSHOT) || return nothing
        return parse_tolerance_toml(read(f[_TOLERANCE_SNAPSHOT]))
    end
end
