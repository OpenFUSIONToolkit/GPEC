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

const _MC_GROUP = "ErrorFields/MonteCarlo"

# Metadata table for ErrorFields/MonteCarlo/ (paths relative to the group). bin_edges has one
# more entry than the densities, so it is documented rather than attached as a dimension scale.
const MC_H5_ANNOTATIONS = [
    "bin_edges" => (; long_name="|δ| bin edges of the overlap histograms (nbins + 1); samples beyond the last edge are counted in the last bin"),
    "pdf" => (; long_name="probability density of the intrinsic dominant-mode overlap |δ| over the sampled misalignments, batch average", dims=("delta_bin",)),
    "pdf_efc" => (; long_name="probability density of the corrected overlap |δ| (correctable terms divided by efc_factor), batch average", dims=("delta_bin",)),
    "pdf_batches" => (; long_name="probability density of the intrinsic overlap |δ| per batch", dims=("delta_bin", "batch")),
    "pdf_efc_batches" => (; long_name="probability density of the corrected overlap |δ| per batch", dims=("delta_bin", "batch")),
    "delta_nominal" => (; long_name="|Σ δ_nominal|, the as-designed overlap with every coil set at its nominal position"),
    "delta_worst" => (; long_name="worst-case alignment bound Σ(|δ_nominal| + tolerance × |sensitivity|) used to size the histogram"),
    "mean_abs_delta" => (; long_name="sample mean of the intrinsic overlap |δ|"),
    "mean_abs_delta_efc" => (; long_name="sample mean of the corrected overlap |δ|"),
    "clamped_fraction" => (; long_name="fraction of samples beyond the last bin edge")
]

"""
    write_to_hdf5!(h5file::HDF5.File, mc::MonteCarloResult)

Write the tolerance Monte Carlo histograms to `ErrorFields/MonteCarlo/`. The sampling settings
(`nsample`, `nbatch`, `seed`) live in the run's `[ErrorFields.MonteCarlo]` table under
`Input/gpec_toml_raw`; the tolerances in `Input/RawInputs/ErrorFields/tolerance_toml_raw`.
An existing group is replaced.
"""
function write_to_hdf5!(h5file::HDF5.File, mc::MonteCarloResult)
    haskey(h5file, _MC_GROUP) && delete_object(h5file, _MC_GROUP)
    g = create_group(h5file, _MC_GROUP)
    g["bin_edges"] = mc.bin_edges
    g["pdf"] = mc.pdf
    g["pdf_efc"] = mc.pdf_efc
    g["pdf_batches"] = mc.pdf_batches
    g["pdf_efc_batches"] = mc.pdf_efc_batches
    g["delta_nominal"] = mc.delta_nominal
    g["delta_worst"] = mc.delta_worst
    g["mean_abs_delta"] = mc.mean_abs_delta
    g["mean_abs_delta_efc"] = mc.mean_abs_delta_efc
    g["clamped_fraction"] = mc.clamped_fraction
    Utilities.HDF5Annotations.annotate!(g, MC_H5_ANNOTATIONS)
    return g
end

"""
    MonteCarloResult(h5path::AbstractString)

Read the tolerance Monte Carlo of a run back from its `gpec.h5`, with the sampling settings
from the `[ErrorFields.MonteCarlo]` table of the stored deck.
"""
function MonteCarloResult(h5path::AbstractString)
    h5open(h5path, "r") do f
        haskey(f, _MC_GROUP) || throw(ArgumentError("$h5path has no $_MC_GROUP group (run with a tolerance_file)"))
        g = f[_MC_GROUP]
        inputs = TOML.parse(read(f["Input/gpec_toml_raw"]))
        ctrl = MonteCarloControl(; (Symbol(k) => v for (k, v) in get(get(inputs, "ErrorFields", Dict{String,Any}()), "MonteCarlo", Dict{String,Any}()))...)
        pdf_batches = read(g["pdf_batches"])
        return MonteCarloResult(read(g["bin_edges"]), read(g["pdf"]), read(g["pdf_efc"]), pdf_batches, read(g["pdf_efc_batches"]),
            read(g["delta_nominal"]), read(g["delta_worst"]), read(g["mean_abs_delta"]), read(g["mean_abs_delta_efc"]),
            read(g["clamped_fraction"]), ctrl.nsample, size(pdf_batches, 2), ctrl.seed)
    end
end

const _RISK_GROUP = "ErrorFields/Risk"

# Metadata table for ErrorFields/Risk/ (paths relative to the group). Percentages are stored as
# such; the threshold density and P(lock|δ) share the Monte Carlo's |δ| grid.
const RISK_H5_ANNOTATIONS = [
    "threshold_pdf" => (; long_name="probability density of the sampled ITPA penetration threshold on the Monte Carlo |δ| bins", dims=("delta_bin",)),
    "p_lock_given_delta" => (; long_name="probability that an overlap equal to each Monte Carlo bin edge locks (threshold cumulative distribution)", dims=("delta_edge",)),
    "threshold_nominal" => (; long_name="ITPA penetration threshold at the fitted exponents"),
    "plock_percent" => (; long_name="locking probability of the intrinsic overlap distribution, 100 ∫ pdf(δ) P(lock|δ) dδ, batch average", units="%"),
    "plock_efc_percent" => (; long_name="locking probability of the corrected overlap distribution, batch average", units="%"),
    "plock_batches_percent" => (; long_name="locking probability of the intrinsic distribution per Monte Carlo batch", units="%"),
    "plock_efc_batches_percent" => (; long_name="locking probability of the corrected distribution per Monte Carlo batch", units="%"),
    "plock_nominal_percent" => (; long_name="locking probability of the as-designed machine, 100 P(lock|δ_nominal)", units="%"),
    "plock_sharp_percent" => (; long_name="locking probability if the threshold were exactly its nominal value, 100 P(|δ| > threshold_nominal)", units="%"),
    "ToleranceScan/scale" => (; long_name="multiplier applied to every shift and tilt tolerance"),
    "ToleranceScan/plock_percent" => (; long_name="locking probability of the intrinsic distribution at each tolerance scale", units="%", dims=("scale",)),
    "ToleranceScan/plock_efc_percent" => (; long_name="locking probability of the corrected distribution at each tolerance scale", units="%", dims=("scale",)),
    "ToleranceScan/plock_spread_percent" => (; long_name="range of the intrinsic locking probability over the Monte Carlo batches at each scale", units="%", dims=("scale",)),
    "ToleranceScan/plock_efc_spread_percent" =>
        (; long_name="range of the corrected locking probability over the Monte Carlo batches at each scale", units="%", dims=("scale",))
]

"""
    write_to_hdf5!(h5file::HDF5.File, risk::RiskResult; scan=nothing)

Write the locking risk to `ErrorFields/Risk/`, with the tolerance scan under
`ErrorFields/Risk/ToleranceScan/` when given. The threshold fit, scenario and sampling settings
live in the run's `[ErrorFields.Risk]` and `[ErrorFields.scenario]` tables under
`Input/gpec_toml_raw`. An existing group is replaced.
"""
function write_to_hdf5!(h5file::HDF5.File, risk::RiskResult; scan::Union{Nothing,ToleranceScan}=nothing)
    haskey(h5file, _RISK_GROUP) && delete_object(h5file, _RISK_GROUP)
    g = create_group(h5file, _RISK_GROUP)
    g["threshold_pdf"] = risk.threshold_pdf
    g["p_lock_given_delta"] = risk.p_lock_given_delta
    g["threshold_nominal"] = risk.threshold_nominal
    g["plock_percent"] = risk.plock
    g["plock_efc_percent"] = risk.plock_efc
    g["plock_batches_percent"] = risk.plock_batches
    g["plock_efc_batches_percent"] = risk.plock_efc_batches
    g["plock_nominal_percent"] = risk.plock_nominal
    g["plock_sharp_percent"] = risk.plock_sharp
    if scan !== nothing
        sg = create_group(g, "ToleranceScan")
        sg["scale"] = scan.scale
        sg["plock_percent"] = scan.plock
        sg["plock_efc_percent"] = scan.plock_efc
        sg["plock_spread_percent"] = scan.plock_spread
        sg["plock_efc_spread_percent"] = scan.plock_efc_spread
    end
    Utilities.HDF5Annotations.annotate!(g, RISK_H5_ANNOTATIONS)
    return g
end

"""
    ToleranceScan(h5path::AbstractString)

Read a run's tolerance scan back from `ErrorFields/Risk/ToleranceScan/`.
"""
function ToleranceScan(h5path::AbstractString)
    h5open(h5path, "r") do f
        path = _RISK_GROUP * "/ToleranceScan"
        haskey(f, path) || throw(ArgumentError("$h5path has no $path group (set scan_scales in [ErrorFields.Risk])"))
        g = f[path]
        return ToleranceScan(read(g["scale"]), read(g["plock_percent"]), read(g["plock_efc_percent"]), read(g["plock_spread_percent"]),
            read(g["plock_efc_spread_percent"]), read(f[_RISK_GROUP*"/plock_nominal_percent"]))
    end
end
