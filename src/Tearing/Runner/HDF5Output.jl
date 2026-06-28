# HDF5Output.jl
#
# Write a `SLAYERResult` into an HDF5 group. Designed to be called by the
# existing `PerturbedEquilibrium.write_outputs_to_HDF5` path — the
# top-level GPEC runner wires that up; this file only defines the pure
# writer.
#
# Output layout (relative to the parent group the caller provides):
#
#   slayer/
#   ├── settings/           -- control snapshot (strings, scalars)
#   ├── per_surface/        -- struct-of-arrays for SLAYERParameters fields
#   │   ├── psi, q, q1, ...
#   │   └── ...
#   ├── roots/              -- Q_root (real, imag), omega_Hz, gamma_Hz
#   ├── diagnostics/        -- all_valid_roots, poles, filtered_roots
#   │                           (flat-plus-offsets ragged encoding)
#   └── scan/               -- optional: full Q/Δ scan data

using HDF5

"""
    write_slayer_hdf5!(parent::Union{HDF5.File,HDF5.Group},
                        result::SLAYERResult)

Write `result` into a `slayer/` subgroup of `parent`. The subgroup is
created if missing and overwritten if it already exists (keeps the
output file reproducible across reruns).
"""
function write_slayer_hdf5!(parent::Union{HDF5.File,HDF5.Group},
    result::SLAYERResult)
    if haskey(parent, "slayer")
        delete_object(parent, "slayer")
    end
    g = create_group(parent, "slayer")
    g["enabled"] = Int(result.enabled)

    result.enabled || return g    # nothing else to write

    _write_settings!(g, result.control)
    _write_per_surface!(g, result.params, result.dp_matrix)
    _write_roots!(g, result)
    _write_layer_widths!(g, result.layer_widths)
    _write_diagnostics!(g, result)
    if result.control.store_scan && !isempty(result.scan_data)
        _write_scan_data!(g, result)
    end
    return g
end

# ---------- settings snapshot ----------
function _write_settings!(g, ctrl::SLAYERControl)
    s = create_group(g, "settings")
    s["inner_model"] = String(ctrl.inner_model)
    s["scan_mode"] = String(ctrl.scan_mode)
    s["coupling_mode"] = String(ctrl.coupling_mode)
    s["dc_type"] = String(ctrl.dc_type)
    s["msing_max"] = ctrl.msing_max
    s["bt"] = ctrl.bt === nothing ? NaN : ctrl.bt
    s["mu_i"] = ctrl.mu_i
    s["zeff"] = ctrl.zeff
    s["chi_perp"] = ctrl.chi_perp
    s["chi_tor"] = ctrl.chi_tor
    # NaN sentinel records the auto-derive (nothing) setting in a numeric dataset.
    s["dr_val"] = ctrl.dr_val === nothing ? NaN : ctrl.dr_val
    s["dgeo_val"] = ctrl.dgeo_val === nothing ? NaN : ctrl.dgeo_val
    s["theta_sample"] = ctrl.theta_sample
    s["resistivity_model"] = String(ctrl.resistivity_model)
    s["lnLambda_form"] = String(ctrl.lnLambda_form)
    s["Q_re_range"] = collect(ctrl.Q_re_range)
    s["Q_im_range"] = collect(ctrl.Q_im_range)
    s["nre"] = ctrl.nre
    s["nim"] = ctrl.nim
    s["amr_passes"] = ctrl.amr_passes
    s["amr_max_cells"] = ctrl.amr_max_cells
    # Multi-box stripe layout: flatten Vector{NTuple{4}} to an N×4 matrix
    # (empty → 0×4) so a rerun can reconstruct the exact scan boxes.
    s["boxes"] = isempty(ctrl.boxes) ? Matrix{Float64}(undef, 0, 4) :
                 permutedims(reduce(hcat, collect.(ctrl.boxes)))
    s["multi_box_prescreen_n"] = ctrl.multi_box_prescreen_n
    s["pole_threshold"] = ctrl.pole_threshold
    s["pole_threshold_adaptive"] = Int(ctrl.pole_threshold_adaptive)
    s["filter_above_poles"] = Int(ctrl.filter_above_poles)
    s["filter_outside_re"] = Int(ctrl.filter_outside_re)
    s["gap_kHz_threshold"] = ctrl.gap_kHz_threshold
    s["polish_roots"] = Int(ctrl.polish_roots)
    s["validity_rtol"] = ctrl.validity_rtol
    s["profile_file"] = ctrl.profile_file
    s["profile_group"] = ctrl.profile_group
    s["store_scan"] = Int(ctrl.store_scan)
    return nothing
end

# ---------- per-surface layer parameters ----------
function _write_per_surface!(g, params::AbstractVector{SLAYERParameters},
    dp_matrix::Matrix{ComplexF64})
    ps = create_group(g, "per_surface")

    # Scalar struct-of-arrays for all Float64 / Int fields
    for fname in (:ising, :m, :n)
        ps[String(fname)] = Int[getfield(p, fname) for p in params]
    end
    for fname in (:tau, :lu, :c_beta, :D_norm, :P_perp, :P_tor,
        :Q_e, :Q_i, :iota_e,
        :tauk, :tau_r, :delta_n,
        :rs, :R0, :bt, :sval_r, :dr_val, :dgeo_val,
        :eta, :d_beta, :dc_tmp)
        ps[String(fname)] = Float64[getfield(p, fname) for p in params]
    end
    # Store dc_type per-surface as string array
    ps["dc_type"] = String[String(p.dc_type) for p in params]

    # Full Δ' matrix, split real/imag
    dp = create_group(ps, "dp_matrix")
    dp["real"] = real.(dp_matrix)
    dp["imag"] = imag.(dp_matrix)
    return nothing
end

# GGJ per-surface parameters carry the geometric coefficients (E,F,G,H,K,M)
# and resistive/Alfvén times rather than the SLAYER dimensionless set.
function _write_per_surface!(g, params::AbstractVector{GGJParameters},
    dp_matrix::Matrix{ComplexF64})
    ps = create_group(g, "per_surface")
    ps["ising"] = Int[p.ising for p in params]
    for fname in (:E, :F, :G, :H, :K, :M, :taua, :taur, :v1)
        ps[String(fname)] = Float64[getfield(p, fname) for p in params]
    end
    dp = create_group(ps, "dp_matrix")
    dp["real"] = real.(dp_matrix)
    dp["imag"] = imag.(dp_matrix)
    return nothing
end

# ---------- eigenvalue roots ----------
function _write_roots!(g, r::SLAYERResult)
    roots = create_group(g, "roots")
    roots["Q_root_real"] = real.(r.Q_root)
    roots["Q_root_imag"] = imag.(r.Q_root)
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
    lw = create_group(g, "layer_widths")
    for fname in (:ising, :m, :n)
        lw[String(fname)] = Int[getfield(w, fname) for w in widths]
    end
    # Dimensionless del_s/d_beta and the complex layer thickness, split re/im.
    lw["dels_db_real"] = Float64[real(w.dels_db) for w in widths]
    lw["dels_db_imag"] = Float64[imag(w.dels_db) for w in widths]
    lw["delta_s_real"] = Float64[real(w.delta_s) for w in widths]
    lw["delta_s_imag"] = Float64[imag(w.delta_s) for w in widths]
    # Physical thickness [m] and the β-weighted ion drift scale [m].
    lw["delta_s_m"] = Float64[w.delta_s_m for w in widths]
    lw["d_beta"] = Float64[w.d_beta for w in widths]
    return nothing
end

# ---------- diagnostics: valid roots, poles, filtered roots ----------
function _write_diagnostics!(g, r::SLAYERResult)
    diag = create_group(g, "diagnostics")
    # Uncoupled: one GrowthRateResult per surface. Coupled: one total.
    extractions = if r.coupled_extraction !== nothing
        [r.coupled_extraction]
    else
        r.per_surface_extraction
    end

    _write_ragged_complex!(diag, "valid_roots",
        [gr.valid_roots for gr in extractions])
    _write_ragged_complex!(diag, "poles",
        [gr.poles for gr in extractions])
    _write_ragged_complex!(diag, "filtered_roots",
        [gr.filtered_roots for gr in extractions])
    return nothing
end

# Write a ragged vector-of-vectors of ComplexF64 as (flat_re, flat_im,
# offsets) — `offsets[k+1] - offsets[k]` is the length of row `k`. This
# avoids HDF5 VLEN types, which have patchy cross-language support.
function _write_ragged_complex!(parent, name::String,
    data::Vector{Vector{ComplexF64}})
    g = create_group(parent, name)
    flat_re = Float64[]
    flat_im = Float64[]
    offsets = Int[0]
    for v in data
        append!(flat_re, real.(v))
        append!(flat_im, imag.(v))
        push!(offsets, offsets[end] + length(v))
    end
    g["flat_real"] = flat_re
    g["flat_imag"] = flat_im
    g["offsets"] = offsets
    return nothing
end

# ---------- full scan data (optional) ----------
function _write_scan_data!(g, r::SLAYERResult)
    sc = create_group(g, "scan")
    for (k, data) in enumerate(r.scan_data)
        sk = create_group(sc, "surface_$(k)")
        _write_single_scan!(sk, data)
    end
    return nothing
end

function _write_single_scan!(g, data::ScanResult)
    g["kind"] = "brute_force"
    g["Q_real"] = real.(data.Q)
    g["Q_imag"] = imag.(data.Q)
    g["Delta_real"] = real.(data.Δ)
    g["Delta_imag"] = imag.(data.Δ)
    g["re_axis"] = data.re_axis
    g["im_axis"] = data.im_axis
    return nothing
end

function _write_single_scan!(g, data::AMRResult)
    g["kind"] = "amr"
    g["Q_real"] = real.(data.Q)
    g["Q_imag"] = imag.(data.Q)
    g["Delta_real"] = real.(data.Δ)
    g["Delta_imag"] = imag.(data.Δ)
    g["n_cells"] = length(data.cells)
    g["truncated"] = Int(data.truncated)
    return nothing
end
