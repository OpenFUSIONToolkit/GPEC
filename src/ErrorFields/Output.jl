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
    "field_as_designed" => (; long_name="root-area-weighted control-surface field b̃ of each coil set as built", units="T", dims=("mode", "coil_set")),
    "shift_sensitivity_per_m" => (; long_name="∂b̃/∂(Δx, Δy, Δz) of each coil set under a rigid Cartesian shift", units="T/m", dims=("mode", "axis", "coil_set")),
    "tilt_sensitivity_per_deg" =>
        (; long_name="∂b̃/∂(θx, θy, θz) of each coil set under a rigid rotation about the machine axes", units="T/deg", dims=("mode", "axis", "coil_set")),
    "shift_linearity_residual" =>
        (; long_name="finite-difference curvature ‖b̃(+h)+b̃(−h)−2b̃(0)‖ of each shift tap relative to the set's largest first difference", dims=("axis", "coil_set")),
    "tilt_linearity_residual" =>
        (; long_name="finite-difference curvature ‖b̃(+h)+b̃(−h)−2b̃(0)‖ of each tilt tap relative to the set's largest first difference", dims=("axis", "coil_set")),
    "peak_current" => (; long_name="largest conductor current magnitude of each coil set at which the spectra were evaluated", units="A"),
    "winding_multiplier" => (; long_name="turns per conductor element of each coil set"),
    "major_radius_m" => (; long_name="arc-length-weighted major radius of each coil set, the lever arm converting a tilt angle to rim displacement", units="m"),
    "DominantMode/delta_as_designed" => (; long_name="overlap δ = Vᴴ₁·b̃ / B_T0 of each coil set with the full-window dominant mode"),
    "DominantMode/shift_sensitivity_per_m" => (; long_name="∂δ/∂(Δx, Δy, Δz) on the full-window dominant mode", units="1/m", dims=("axis", "coil_set")),
    "DominantMode/tilt_sensitivity_per_deg" => (; long_name="∂δ/∂(θx, θy, θz) on the full-window dominant mode", units="1/deg", dims=("axis", "coil_set")),
    "DominantMode/abs_delta_shift_per_mm" => (; long_name="direction-averaged in-plane shift sensitivity √((|∂δ/∂Δx|²+|∂δ/∂Δy|²)/2)", units="1/mm"),
    "DominantMode/abs_delta_tilt_per_deg" => (; long_name="direction-averaged in-plane tilt sensitivity √((|∂δ/∂θx|²+|∂δ/∂θy|²)/2)", units="1/deg"),
    "DominantMode/abs_delta_rim_per_mm" => (; long_name="the same tilt sensitivity as rim displacement at major_radius_m", units="1/mm"),
    "DominantMode/cancelling_shift_m" => (; long_name="in-plane shift (Δx, Δy) that cancels delta_as_designed to linear order", units="m", dims=("axis", "coil_set")),
    "DominantMode/cancelling_tilt_deg" => (; long_name="in-plane tilt (θx, θy) that cancels delta_as_designed to linear order", units="deg", dims=("axis", "coil_set"))
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
    g["field_as_designed"] = sens.field_as_designed
    g["shift_sensitivity_per_m"] = sens.shift_sensitivity_per_m
    g["tilt_sensitivity_per_deg"] = sens.tilt_sensitivity_per_deg
    g["shift_linearity_residual"] = sens.shift_linearity_residual
    g["tilt_linearity_residual"] = sens.tilt_linearity_residual
    g["peak_current"] = sens.peak_current
    g["winding_multiplier"] = sens.winding_multiplier
    g["major_radius_m"] = sens.major_radius_m

    table = sensitivity_table(sens, dom; mode=1)
    d = create_group(g, "DominantMode")
    d["delta_as_designed"] = table.delta_as_designed
    d["shift_sensitivity_per_m"] = table.shift_sensitivity_per_m
    d["tilt_sensitivity_per_deg"] = table.tilt_sensitivity_per_deg
    d["abs_delta_shift_per_mm"] = table.abs_delta_shift_per_mm
    d["abs_delta_tilt_per_deg"] = table.abs_delta_tilt_per_deg
    d["abs_delta_rim_per_mm"] = table.abs_delta_rim_per_mm
    d["cancelling_shift_m"] = table.cancelling_shift_m
    d["cancelling_tilt_deg"] = table.cancelling_tilt_deg

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
            read(g["field_as_designed"]), read(g["shift_sensitivity_per_m"]), read(g["tilt_sensitivity_per_deg"]),
            read(g["shift_linearity_residual"]), read(g["tilt_linearity_residual"]),
            read(g["peak_current"]), read(g["winding_multiplier"]), read(g["major_radius_m"])
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

# Metadata table for ErrorFields/MonteCarlo/ (paths relative to the group). abs_delta_bin_edges has one
# more entry than the densities, so it is documented rather than attached as a dimension scale.
const MC_H5_ANNOTATIONS = [
    "abs_delta_bin_edges" => (; long_name="|δ| bin edges of the overlap histograms (nbins + 1); samples beyond the last edge are counted in the last bin"),
    "abs_delta_pdf" => (; long_name="probability density of the intrinsic dominant-mode overlap |δ| over the sampled misalignments, batch average", dims=("delta_bin",)),
    "abs_delta_efc_pdf" => (; long_name="probability density of the corrected overlap |δ| (correctable terms divided by efc_factor), batch average", dims=("delta_bin",)),
    "abs_delta_pdf_batches" => (; long_name="probability density of the intrinsic overlap |δ| per batch", dims=("delta_bin", "batch")),
    "abs_delta_efc_pdf_batches" => (; long_name="probability density of the corrected overlap |δ| per batch", dims=("delta_bin", "batch")),
    "abs_delta_total_as_designed" =>
        (; long_name="|Σ δ_as_designed|, the as-designed overlap with every error-field coil set at its design position (correction arrays and excluded sets left out)"),
    "abs_delta_worst_case" => (; long_name="worst-case alignment bound Σ(|δ_as_designed| + tolerance × |sensitivity|) used to size the histogram"),
    "abs_delta_sampled_mean" => (; long_name="sample mean of the intrinsic overlap |δ|"),
    "abs_delta_efc_sampled_mean" => (; long_name="sample mean of the corrected overlap |δ|"),
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
    g["abs_delta_bin_edges"] = mc.abs_delta_bin_edges
    g["abs_delta_pdf"] = mc.abs_delta_pdf
    g["abs_delta_efc_pdf"] = mc.abs_delta_efc_pdf
    g["abs_delta_pdf_batches"] = mc.abs_delta_pdf_batches
    g["abs_delta_efc_pdf_batches"] = mc.abs_delta_efc_pdf_batches
    g["abs_delta_total_as_designed"] = mc.abs_delta_total_as_designed
    g["abs_delta_worst_case"] = mc.abs_delta_worst_case
    g["abs_delta_sampled_mean"] = mc.abs_delta_sampled_mean
    g["abs_delta_efc_sampled_mean"] = mc.abs_delta_efc_sampled_mean
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
        # Read the two settings this needs by name rather than splatting the whole stored table into
        # MonteCarloControl: a file written by a version that knows one more key would otherwise be
        # unreadable here, and reading a result back should not depend on the writer's vintage.
        mc_tbl = get(get(inputs, "ErrorFields", Dict{String,Any}()), "MonteCarlo", Dict{String,Any}())
        defaults = MonteCarloControl()
        nsample = get(mc_tbl, "nsample", defaults.nsample)
        seed = get(mc_tbl, "seed", defaults.seed)
        abs_delta_pdf_batches = read(g["abs_delta_pdf_batches"])
        return MonteCarloResult(read(g["abs_delta_bin_edges"]), read(g["abs_delta_pdf"]), read(g["abs_delta_efc_pdf"]), abs_delta_pdf_batches, read(g["abs_delta_efc_pdf_batches"]),
            read(g["abs_delta_total_as_designed"]), read(g["abs_delta_worst_case"]), read(g["abs_delta_sampled_mean"]), read(g["abs_delta_efc_sampled_mean"]),
            read(g["clamped_fraction"]), nsample, size(abs_delta_pdf_batches, 2), seed)
    end
end

const _RISK_GROUP = "ErrorFields/Risk"

# Metadata table for ErrorFields/Risk/ (paths relative to the group). Percentages are stored as
# such; the threshold density and P(lock|δ) share the Monte Carlo's |δ| grid.
const RISK_H5_ANNOTATIONS = [
    "threshold_pdf" => (; long_name="probability density of the sampled ITPA penetration threshold on the Monte Carlo |δ| bins", dims=("delta_bin",)),
    "locking_probability_given_delta" =>
        (; long_name="probability that an overlap equal to each Monte Carlo bin edge locks (threshold cumulative distribution)", dims=("delta_edge",)),
    "threshold_fit" => (; long_name="ITPA penetration threshold at the fitted exponents"),
    "locking_probability_percent" => (; long_name="locking probability of the intrinsic overlap distribution, 100 ∫ pdf(δ) P(lock|δ) dδ, batch average", units="%"),
    "locking_probability_efc_percent" => (; long_name="locking probability of the corrected overlap distribution, batch average", units="%"),
    "locking_probability_batches_percent" => (; long_name="locking probability of the intrinsic distribution per Monte Carlo batch", units="%"),
    "locking_probability_efc_batches_percent" => (; long_name="locking probability of the corrected distribution per Monte Carlo batch", units="%"),
    "locking_probability_as_designed_percent" => (; long_name="locking probability of the as-designed machine, 100 P(lock | |Σ δ_as_designed|)", units="%"),
    "locking_probability_fit_threshold_percent" => (; long_name="locking probability if the threshold were exactly its fitted value, 100 P(|δ| > threshold_fit)", units="%"),
    "ToleranceScan/tolerance_scale" => (; long_name="multiplier applied to every shift and tilt tolerance"),
    "ToleranceScan/locking_probability_percent" =>
        (; long_name="locking probability of the intrinsic distribution at each tolerance scale", units="%", dims=("tolerance_scale",)),
    "ToleranceScan/locking_probability_efc_percent" =>
        (; long_name="locking probability of the corrected distribution at each tolerance scale", units="%", dims=("tolerance_scale",)),
    "ToleranceScan/locking_probability_spread_percent" =>
        (; long_name="range of the intrinsic locking probability over the Monte Carlo batches at each scale", units="%", dims=("tolerance_scale",)),
    "ToleranceScan/locking_probability_efc_spread_percent" =>
        (; long_name="range of the corrected locking probability over the Monte Carlo batches at each scale", units="%", dims=("tolerance_scale",))
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
    g["locking_probability_given_delta"] = risk.locking_probability_given_delta
    g["threshold_fit"] = risk.threshold_fit
    g["locking_probability_percent"] = risk.locking_probability_percent
    g["locking_probability_efc_percent"] = risk.locking_probability_efc_percent
    g["locking_probability_batches_percent"] = risk.locking_probability_batches_percent
    g["locking_probability_efc_batches_percent"] = risk.locking_probability_efc_batches_percent
    g["locking_probability_as_designed_percent"] = risk.locking_probability_as_designed_percent
    g["locking_probability_fit_threshold_percent"] = risk.locking_probability_fit_threshold_percent
    if scan !== nothing
        sg = create_group(g, "ToleranceScan")
        sg["tolerance_scale"] = scan.tolerance_scale
        sg["locking_probability_percent"] = scan.locking_probability_percent
        sg["locking_probability_efc_percent"] = scan.locking_probability_efc_percent
        sg["locking_probability_spread_percent"] = scan.locking_probability_spread_percent
        sg["locking_probability_efc_spread_percent"] = scan.locking_probability_efc_spread_percent
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
        return ToleranceScan(read(g["tolerance_scale"]), read(g["locking_probability_percent"]), read(g["locking_probability_efc_percent"]),
            read(g["locking_probability_spread_percent"]),
            read(g["locking_probability_efc_spread_percent"]), read(f[_RISK_GROUP*"/locking_probability_as_designed_percent"]))
    end
end

const _NTV_GROUP = "ErrorFields/NTV"

# Metadata table for ErrorFields/NTV/ (paths relative to the group): per correction-coil array,
# per kilo-ampere-turn of its current pattern.
const NTV_H5_ANNOTATIONS = [
    "coil_name" => (; long_name="name of each correction coil array"),
    "delta_per_kat" => (; long_name="dominant-mode overlap |δ| of each array per kilo-ampere-turn", units="1/kAt"),
    "resonant_fraction_percent" => (; long_name="resonant fraction of each array's field, 100·|Vᴴb̃|/‖b̃‖", units="%"),
    "torque_full_per_kat2" => (; long_name="NTV torque of each array's whole field per kilo-ampere-turn squared", units="N*m/kAt^2"),
    "torque_residual_per_kat2" => (; long_name="NTV torque of each array's field with the dominant mode projected out, per kilo-ampere-turn squared", units="N*m/kAt^2"),
    "omega_reference" => (; long_name="reference rotation ω_ref of each array's torque balance: ion toroidal rotation weighted by density and volume", units="rad/s"),
    "omega_offset_estimate" =>
        (; long_name="rough neoclassical offset rotation the scan span was sized against, offset_factor·ω_*T at the innermost kinetic surface", units="rad/s")
]

# ErrorFields/NTV/RotationScan/: the torques against a rigid E×B rotation shift, one column per
# array; the adaptive grids differ in length, so shorter scans are padded with NaN.
const NTV_SCAN_H5_ANNOTATIONS = [
    "coil_name" => (; long_name="name of each scanned correction coil array"),
    "rotation_shift" =>
        (; long_name="rigid shift Δω of the E×B rotation profile at each scan point of each array (NaN-padded)", units="rad/s", dims=("scan_point", "coil_set")),
    "torque_full" =>
        (; long_name="NTV torque of the whole field at each rotation shift, per kilo-ampere-turn squared (NaN-padded)", units="N*m/kAt^2", dims=("scan_point", "coil_set")),
    "torque_residual" => (;
        long_name="NTV torque of the field with the dominant mode projected out at each rotation shift, per kilo-ampere-turn squared (NaN-padded)",
        units="N*m/kAt^2",
        dims=("scan_point", "coil_set")
    ),
    "psi" => (; long_name="kinetic normalized poloidal flux grid of the torque profiles", scale="psi"),
    "torque_full_profile" => (;
        long_name="cumulative NTV torque of the whole field from the axis to ψ_N, at each rotation shift of each array, per kilo-ampere-turn squared (NaN-padded)",
        units="N*m/kAt^2",
        dims=("psi", "scan_point", "coil_set")
    ),
    "torque_residual_profile" => (;
        long_name="cumulative NTV torque of the residual field from the axis to ψ_N, at each rotation shift of each array, per kilo-ampere-turn squared (NaN-padded)",
        units="N*m/kAt^2",
        dims=("psi", "scan_point", "coil_set")
    )
]

"""
    write_to_hdf5!(h5file::HDF5.File, couplings::Vector{EFCCoupling})

Write the correction-coil couplings to `ErrorFields/NTV/`, and the torque-versus-rotation tables
of the arrays that have one to `ErrorFields/NTV/RotationScan/` (one column per array, shorter
scans NaN-padded). The torque budget, threshold, rotation exponent and safety factor that turn
them into a correction-current curve are analysis choices left to [`efc_current_curve`](@ref).
An existing group is replaced.
"""
function write_to_hdf5!(h5file::HDF5.File, couplings::Vector{EFCCoupling})
    haskey(h5file, _NTV_GROUP) && delete_object(h5file, _NTV_GROUP)
    g = create_group(h5file, _NTV_GROUP)
    g["coil_name"] = [c.coil_name for c in couplings]
    g["delta_per_kat"] = [c.delta_per_kat for c in couplings]
    g["resonant_fraction_percent"] = [c.resonant_fraction_percent for c in couplings]
    g["torque_full_per_kat2"] = [c.torque_full_per_kat2 for c in couplings]
    g["torque_residual_per_kat2"] = [c.torque_residual_per_kat2 for c in couplings]
    g["omega_reference"] = [c.omega_reference for c in couplings]
    g["omega_offset_estimate"] = [c.omega_offset_estimate for c in couplings]
    Utilities.HDF5Annotations.annotate!(g, NTV_H5_ANNOTATIONS)
    scanned = filter(has_rotation_scan, couplings)
    if !isempty(scanned)
        nmax = maximum(length(c.rotation_shift) for c in scanned)
        ψ = scanned[1].psi
        all(c.psi == ψ for c in scanned) || error("write_to_hdf5!: the rotation scans of the arrays are on different ψ grids")
        pad(v) = vcat(Float64.(v), fill(NaN, nmax - length(v)))
        padm(m) = hcat(Float64.(m), fill(NaN, size(m, 1), nmax - size(m, 2)))
        sg = create_group(g, "RotationScan")
        sg["coil_name"] = [c.coil_name for c in scanned]
        sg["rotation_shift"] = reduce(hcat, pad(c.rotation_shift) for c in scanned)
        sg["torque_full"] = reduce(hcat, pad(c.torque_full_scan) for c in scanned)
        sg["torque_residual"] = reduce(hcat, pad(c.torque_residual_scan) for c in scanned)
        sg["psi"] = ψ
        sg["torque_full_profile"] = cat((padm(c.torque_full_profile) for c in scanned)...; dims=3)
        sg["torque_residual_profile"] = cat((padm(c.torque_residual_profile) for c in scanned)...; dims=3)
        Utilities.HDF5Annotations.annotate!(sg, NTV_SCAN_H5_ANNOTATIONS)
    end
    return g
end

"""
    read_efc_couplings(h5path::AbstractString) -> Vector{EFCCoupling}

Read a run's correction-coil couplings back from `ErrorFields/NTV/`, with their
torque-versus-rotation tables when the run made them.
"""
function read_efc_couplings(h5path::AbstractString)
    h5open(h5path, "r") do f
        haskey(f, _NTV_GROUP) || throw(ArgumentError("$h5path has no $_NTV_GROUP group (set efc_coils in [ErrorFields.NTV])"))
        g = f[_NTV_GROUP]
        names = read(g["coil_name"])
        δ, ov = read(g["delta_per_kat"]), read(g["resonant_fraction_percent"])
        Tf, Tr = read(g["torque_full_per_kat2"]), read(g["torque_residual_per_kat2"])
        ω_ref = haskey(g, "omega_reference") ? read(g["omega_reference"]) : fill(NaN, length(names))
        ω_off = haskey(g, "omega_offset_estimate") ? read(g["omega_offset_estimate"]) : fill(NaN, length(names))
        scan = haskey(g, "RotationScan") ? g["RotationScan"] : nothing
        scan_names = scan === nothing ? String[] : read(scan["coil_name"])
        out = EFCCoupling[]
        for i in eachindex(names)
            j = findfirst(==(names[i]), scan_names)
            if j === nothing
                push!(out, EFCCoupling(names[i], δ[i], ov[i], Tf[i], Tr[i]))
            else
                shifts = read(scan["rotation_shift"])[:, j]
                n = count(!isnan, shifts)
                push!(
                    out,
                    EFCCoupling(names[i], δ[i], ov[i], Tf[i], Tr[i], shifts[1:n], read(scan["torque_full"])[1:n, j], read(scan["torque_residual"])[1:n, j],
                        ω_ref[i], ω_off[i], read(scan["psi"]), read(scan["torque_full_profile"])[:, 1:n, j], read(scan["torque_residual_profile"])[:, 1:n, j])
                )
            end
        end
        return out
    end
end
