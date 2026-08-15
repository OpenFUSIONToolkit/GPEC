# HDF5Output.jl
#
# Write a `SLAYERResult` into an HDF5 group. Designed to be called by the
# existing `PerturbedEquilibrium.write_outputs_to_HDF5` path — the
# top-level GPEC runner wires that up; this file only defines the pure
# writer.
#
# Output layout (relative to the parent group the caller provides;
# settings are not echoed here — inputs live only under Input/):
#
#   Tearing/
#   ├── PerSurface/         -- struct-of-arrays for SLAYERParameters fields
#   │   ├── psi, q, q1, ...
#   │   └── ...
#   ├── Roots/              -- complex Q_root, omega_Hz, gamma_Hz
#   ├── Diagnostics/        -- ValidRoots, Poles, FilteredRoots
#   │                           (flat-plus-offsets ragged encoding)
#   └── Scan/               -- optional: full Q/Δ scan data

using HDF5

"""
    write_slayer_hdf5!(parent::Union{HDF5.File,HDF5.Group},
                        result::SLAYERResult)

Write `result` into a `Tearing/` subgroup of `parent`. The subgroup is
created if missing and overwritten if it already exists (keeps the
output file reproducible across reruns).
"""
function write_slayer_hdf5!(parent::Union{HDF5.File,HDF5.Group},
    result::SLAYERResult)
    if haskey(parent, "Tearing")
        delete_object(parent, "Tearing")
    end
    g = create_group(parent, "Tearing")
    g["enabled"] = Int(result.enabled)
    # Which inner-layer model produced PerSurface/: the SLAYER and GGJ branches write
    # disjoint field sets, so readers must not have to infer it from the schema.
    attrs(g)["layer_model"] = result.enabled ? _layer_model_token(eltype(result.params)) : "none"

    if !result.enabled    # nothing else to write
        _annotate_tearing!(g)
        return g
    end

    _write_per_surface!(g, result.params, result.dp_matrix)
    _write_roots!(g, result)
    _write_layer_widths!(g, result.layer_widths)
    _write_diagnostics!(g, result)
    if result.control.store_scan && !isempty(result.scan_data)
        _write_scan_data!(g, result)
    end
    _annotate_tearing!(g)
    return g
end

# Token recorded in the Tearing group's layer_model attribute; keyed by the
# per-surface parameter type since the SLAYER and GGJ branches write disjoint fields.
_layer_model_token(::Type{SLAYERParameters}) = "slayer"
_layer_model_token(::Type{GGJParameters}) = "ggj"

# Metadata table for the Tearing group (paths relative to it); ragged Diagnostics
# subgroups and Scan/Surface_<k> groups are annotated by iteration below.
const TEARING_H5_ANNOTATIONS = [
    "enabled" => (; long_name="flag: SLAYER/tearing stage ran (1) or was disabled (0)"),
    "PerSurface/ising" => (; long_name="rational-surface index of each row", dims=("surface",)),
    "PerSurface/m" => (; long_name="resonant poloidal mode number m per surface", dims=("surface",)),
    "PerSurface/n" => (; long_name="resonant toroidal mode number n per surface", dims=("surface",)),
    "PerSurface/tau" => (; long_name="temperature ratio τ = T_i/T_e per surface", dims=("surface",)),
    "PerSurface/lu" => (; long_name="Lundquist number S per surface", dims=("surface",)),
    "PerSurface/c_beta" => (; long_name="compressibility factor c_β = √(β_local/(1+β_local)) per surface", dims=("surface",)),
    "PerSurface/D_norm" => (; long_name="Fitzpatrick normalized ion-sound/drift scale D = (d_β/r_s)·S^(1/3)·√(τ/(1+τ)) per surface", dims=("surface",)),
    "PerSurface/P_perp" => (; long_name="perpendicular magnetic Prandtl number per surface", dims=("surface",)),
    "PerSurface/P_tor" => (; long_name="toroidal (momentum) magnetic Prandtl number per surface", dims=("surface",)),
    "PerSurface/Q_e" => (; long_name="normalized electron diamagnetic frequency Q_e per surface", dims=("surface",)),
    "PerSurface/Q_i" => (; long_name="normalized ion diamagnetic frequency Q_i per surface", dims=("surface",)),
    "PerSurface/iota_e" => (; long_name="electron fraction ι_e = Q_e/(Q_e − Q_i) per surface", dims=("surface",)),
    "PerSurface/tauk" =>
        (; long_name="Q-normalization time S^(1/3)·τ_H per surface (Q = τ_k·ω; diamagnetic inputs Q_e, Q_i carry the opposite sign by convention)", units="s", dims=("surface",)),
    "PerSurface/tau_r" => (; long_name="resistive diffusion time τ_R = μ₀r_s²/η per surface (SLAYER layer parameters)", units="s", dims=("surface",)),
    "PerSurface/delta_n" => (; long_name="Δ'-normalization factor S^(1/3)/r_s per surface", units="1/m", dims=("surface",)),
    "PerSurface/rs" => (; long_name="minor radius of each rational surface", units="m", dims=("surface",)),
    "PerSurface/R0" => (; long_name="major radius", units="m", dims=("surface",)),
    "PerSurface/bt" => (; long_name="toroidal field", units="T", dims=("surface",)),
    "PerSurface/sval_r" => (; long_name="r-based magnetic shear r_s·(dq/dr)/q (Fitzpatrick convention)", dims=("surface",)),
    "PerSurface/D_R" =>
        (; long_name="resistive interchange D_R = E + F + H² for the critical-Δ formula (auto-derived from GGJ coefficients unless overridden)", dims=("surface",)),
    "PerSurface/D_geo" => (; long_name="Connor-Hastie-Helander 2015 Eq. 59 geometric factor (0 unless supplied)", dims=("surface",)),
    "PerSurface/eta" => (; long_name="parallel resistivity at each surface", units="Ohm*m", dims=("surface",)),
    "PerSurface/d_beta" => (; long_name="β-weighted ion drift scale d_β", units="m", dims=("surface",)),
    "PerSurface/dc_tmp" => (; long_name="critical-Δ offset from χ_∥/χ_⊥ matching (Connor-Hastie-Helander 2015 Eq. 59)", dims=("surface",)),
    "PerSurface/dc_type" => (; long_name="per-surface D_c prescription label", dims=("surface",)),
    "PerSurface/E" => (; long_name="Glasser-Greene-Johnson coefficient E per surface", dims=("surface",)),
    "PerSurface/F" => (; long_name="Glasser-Greene-Johnson coefficient F per surface", dims=("surface",)),
    "PerSurface/G" => (; long_name="Glasser-Greene-Johnson coefficient G per surface", dims=("surface",)),
    "PerSurface/H" => (; long_name="Glasser-Greene-Johnson coefficient H per surface", dims=("surface",)),
    "PerSurface/K" => (; long_name="Glasser-Greene-Johnson coefficient K per surface", dims=("surface",)),
    "PerSurface/M" => (; long_name="Glasser-Greene-Johnson coefficient M per surface", dims=("surface",)),
    "PerSurface/taua" => (; long_name="Alfvén time τ_A per surface (GGJ layer parameters)", units="s", dims=("surface",)),
    "PerSurface/taur" => (; long_name="resistive diffusion time τ_R per surface (GGJ layer parameters)", units="s", dims=("surface",)),
    "PerSurface/v1" => (; long_name="dV/dψ_N at each surface", units="m^3", dims=("surface",)),
    "PerSurface/dp_matrix" => (; long_name="full complex Δ' matrix coupling the rational surfaces", dims=("surface_row", "surface_col")),
    "Roots/Q_root" => (; long_name="complex dispersion-root normalized frequency Q (NaN = no root)", dims=("surface",)),
    "Roots/omega_Hz" =>
        (; long_name="mode rotation angular frequency ω = Re(Q)/τ_k of each root (dataset name is a misnomer: angular, not cycles/s)", units="rad/s", dims=("surface",)),
    "Roots/gamma_Hz" =>
        (; long_name="growth rate γ = Im(Q)/τ_k of each root, positive = unstable (an e-folding rate, so no 2π distinction applies)", units="1/s", dims=("surface",)),
    "Roots/no_root" => (; long_name="flag: no usable dispersion root found (Q_root is NaN, ω/γ are placeholders)", dims=("surface",)),
    "LayerWidths/ising" => (; long_name="rational-surface index of each row", dims=("surface",)),
    "LayerWidths/m" => (; long_name="resonant poloidal mode number m per surface", dims=("surface",)),
    "LayerWidths/n" => (; long_name="resonant toroidal mode number n per surface", dims=("surface",)),
    "LayerWidths/dels_db" => (; long_name="complex dimensionless layer thickness δ_s/d_β", dims=("surface",)),
    "LayerWidths/delta_s" => (; long_name="complex resistive layer thickness δ_s (Riccati)", dims=("surface",)),
    "LayerWidths/delta_s_m" => (; long_name="physical resistive layer thickness |δ_s|", units="m", dims=("surface",)),
    "LayerWidths/d_beta" => (; long_name="β-weighted ion drift scale d_β", units="m", dims=("surface",))
]

const TEARING_RAGGED_H5_ANNOTATIONS = [
    "flat" => (; long_name="concatenated complex entries (rows delimited by offsets)"),
    "offsets" => (; long_name="ragged-array offsets: row k spans offsets[k]+1:offsets[k+1]")
]

const TEARING_SCAN_H5_ANNOTATIONS = [
    "kind" => (; long_name="scan kind: brute_force or amr"),
    "Q" => (; long_name="sampled complex normalized frequency Q"),
    "Delta" => (; long_name="complex inner-layer matching Δ(Q)"),
    "re_axis" => (; long_name="Re(Q) axis of the brute-force scan grid"),
    "im_axis" => (; long_name="Im(Q) axis of the brute-force scan grid"),
    "n_cells" => (; long_name="number of AMR cells sampled"),
    "truncated" => (; long_name="flag: AMR refinement stopped at the cell cap")
]

# Attach long_name/units/dims to everything write_slayer_hdf5! wrote.
function _annotate_tearing!(g)
    ann = Utilities.HDF5Annotations
    ann.annotate!(g, TEARING_H5_ANNOTATIONS)
    if haskey(g, "Diagnostics")
        for sub in keys(g["Diagnostics"])
            ann.annotate!(g["Diagnostics"][sub], TEARING_RAGGED_H5_ANNOTATIONS)
        end
    end
    if haskey(g, "Scan")
        for sub in keys(g["Scan"])
            sg = g["Scan"][sub]
            ann.annotate!(sg, TEARING_SCAN_H5_ANNOTATIONS)
            # Brute-force scans store 2-D (re, im) grids; AMR stores flat samples.
            if haskey(sg, "Q") && ndims(sg["Q"]) == 2
                for a in ("Q", "Delta")
                    haskey(sg, a) && (attrs(sg[a])["dims"] = "(re_axis, im_axis)")
                end
            end
        end
    end
    return nothing
end

# ---------- per-surface layer parameters ----------
function _write_per_surface!(g, params::AbstractVector{SLAYERParameters},
    dp_matrix::Matrix{ComplexF64})
    ps = create_group(g, "PerSurface")

    # Scalar struct-of-arrays for all Float64 / Int fields
    for fname in (:ising, :m, :n)
        ps[String(fname)] = Int[getfield(p, fname) for p in params]
    end
    for fname in (:tau, :lu, :c_beta, :D_norm, :P_perp, :P_tor,
        :Q_e, :Q_i, :iota_e,
        :tauk, :tau_r, :delta_n,
        :rs, :R0, :bt, :sval_r,
        :eta, :d_beta, :dc_tmp)
        ps[String(fname)] = Float64[getfield(p, fname) for p in params]
    end
    # Physics names for the interchange/geometric inputs (struct fields keep *_val).
    ps["D_R"] = Float64[p.dr_val for p in params]
    ps["D_geo"] = Float64[p.dgeo_val for p in params]
    # Store dc_type per-surface as string array
    ps["dc_type"] = String[String(p.dc_type) for p in params]

    ps["dp_matrix"] = dp_matrix
    return nothing
end

# GGJ per-surface parameters carry the geometric coefficients (E,F,G,H,K,M)
# and resistive/Alfvén times rather than the SLAYER dimensionless set.
function _write_per_surface!(g, params::AbstractVector{GGJParameters},
    dp_matrix::Matrix{ComplexF64})
    ps = create_group(g, "PerSurface")
    ps["ising"] = Int[p.ising for p in params]
    for fname in (:E, :F, :G, :H, :K, :M, :taua, :taur, :v1)
        ps[String(fname)] = Float64[getfield(p, fname) for p in params]
    end
    ps["dp_matrix"] = dp_matrix
    return nothing
end

# ---------- eigenvalue roots ----------
function _write_roots!(g, r::SLAYERResult)
    roots = create_group(g, "Roots")
    roots["Q_root"] = r.Q_root
    roots["omega_Hz"] = r.omega_Hz
    roots["gamma_Hz"] = r.gamma_Hz
    # `no_root[k] == 1` flags entries where the extraction found NO usable
    # root (Q_root is NaN; omega_Hz/gamma_Hz are 0 placeholders, not a true
    # γ≈0 result). Aligned element-wise with Q_root/omega_Hz/gamma_Hz.
    extractions = r.coupled_extraction !== nothing ? [r.coupled_extraction] :
                  r.per_surface_extraction
    roots["no_root"] = Int[(:no_root in gr.warning_flags) ? 1 : 0
                           for gr in extractions]
    return nothing
end

# ---------- resistive layer thickness (del_s Riccati) ----------
function _write_layer_widths!(g, widths::Vector{LayerWidths})
    lw = create_group(g, "LayerWidths")
    for fname in (:ising, :m, :n)
        lw[String(fname)] = Int[getfield(w, fname) for w in widths]
    end
    # Dimensionless del_s/d_beta and the complex layer thickness.
    lw["dels_db"] = ComplexF64[w.dels_db for w in widths]
    lw["delta_s"] = ComplexF64[w.delta_s for w in widths]
    # Physical thickness [m] and the β-weighted ion drift scale [m].
    lw["delta_s_m"] = Float64[w.delta_s_m for w in widths]
    lw["d_beta"] = Float64[w.d_beta for w in widths]
    return nothing
end

# ---------- diagnostics: valid roots, poles, filtered roots ----------
function _write_diagnostics!(g, r::SLAYERResult)
    diag = create_group(g, "Diagnostics")
    # Uncoupled: one GrowthRateResult per surface. Coupled: one total.
    extractions = if r.coupled_extraction !== nothing
        [r.coupled_extraction]
    else
        r.per_surface_extraction
    end

    _write_ragged_complex!(diag, "ValidRoots",
        [gr.valid_roots for gr in extractions])
    _write_ragged_complex!(diag, "Poles",
        [gr.poles for gr in extractions])
    _write_ragged_complex!(diag, "FilteredRoots",
        [gr.filtered_roots for gr in extractions])
    return nothing
end

# Write a ragged vector-of-vectors of ComplexF64 as (flat, offsets) —
# `offsets[k+1] - offsets[k]` is the length of row `k`. This avoids HDF5 VLEN
# types, which have patchy cross-language support.
function _write_ragged_complex!(parent, name::String,
    data::Vector{Vector{ComplexF64}})
    g = create_group(parent, name)
    flat = ComplexF64[]
    offsets = Int[0]
    for v in data
        append!(flat, v)
        push!(offsets, offsets[end] + length(v))
    end
    g["flat"] = flat
    g["offsets"] = offsets
    return nothing
end

# ---------- full scan data (optional) ----------
function _write_scan_data!(g, r::SLAYERResult)
    sc = create_group(g, "Scan")
    for (k, data) in enumerate(r.scan_data)
        sk = create_group(sc, "Surface_$(k)")
        _write_single_scan!(sk, data)
    end
    return nothing
end

function _write_single_scan!(g, data::ScanResult)
    g["kind"] = "brute_force"
    g["Q"] = data.Q
    g["Delta"] = data.Δ
    g["re_axis"] = data.re_axis
    g["im_axis"] = data.im_axis
    return nothing
end

function _write_single_scan!(g, data::AMRResult)
    g["kind"] = "amr"
    g["Q"] = data.Q
    g["Delta"] = data.Δ
    g["n_cells"] = length(data.cells)
    g["truncated"] = Int(data.truncated)
    return nothing
end
