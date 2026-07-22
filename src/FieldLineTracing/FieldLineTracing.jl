module FieldLineTracing

"""
FieldLineTracing — magnetic field-line tracing and topology diagnostics.

Integrates field lines through the GPEC-computed magnetic field to reveal 3D topology:
magnetic islands at rational surfaces, stochastic layers from island overlap, connection
lengths, and divertor footprints — the diagnostics common to TRIP3D and MAFOT. The traced
perturbation is selectable (vacuum / plasma-response / total), and integration runs either
in straight-field-line coordinates (interior) or real cylindrical space (inside and outside
the control surface).

## Module structure
- `FieldLineTracingStructs.jl`: Control / Internal / State data structures.
- `FieldModel.jl`: field assembly and evaluation (flux and real-space representations).
- `Tracer.jl`: `OrdinaryDiffEq` field-line integration; punctures, connection length, footprints.
- `Diagnostics.jl`: island half-widths and Chirikov overlap.
- `Output.jl`: HDF5 output under `field_line_tracing/`.
"""

using HDF5
using Printf
using LinearAlgebra
using FastInterpolations
using OrdinaryDiffEq

import ..Equilibrium
import ..ForcingTerms
import ..PerturbedEquilibrium
import ..ForceFreeStates
import ..ForceFreeStates: OdeState, ForceFreeStatesInternal, FourFitVars

include("FieldLineTracingStructs.jl")
include("FieldModel.jl")
include("Tracer.jl")
include("Diagnostics.jl")
include("Output.jl")

export FieldLineTracingControl, FieldLineTracingInternal, FieldLineTracingState
export compute_field_line_tracing, write_to_hdf5!

"""
    _source_amplitude(source, pe_state) -> Vector{ComplexF64}

Control-surface field amplitude (root-area-weighted b̃ basis) for the requested `source`:
`"vacuum"` is the applied field `Φ_x`, `"total"` is the plasma-response field `P·Φ_x`, and
`"plasma"` is their difference. This is the basis of `pe_state.C_island_width_sq`, so
`C_island_width_sq · amp` gives the resonant island_width_sq for that source.
"""
function _source_amplitude(source::String, pe_state)
    if source == "vacuum"
        return pe_state.forcing_b_rootarea
    elseif source == "total"
        return pe_state.response_b_rootarea
    else
        return pe_state.response_b_rootarea .- pe_state.forcing_b_rootarea
    end
end

"""
    _reconstruct_source_b_modes(source, vac_data, pe_odet, equil, ffs_intr, pe_intr,
                                metric, ffit, pe_ctrl, pe_state)

Perturbed-field mode NamedTuple `b_modes` for real-space tracing of a non-coil source:
`"total"` reuses the stored response fields, `"vacuum"` reconstructs from `Φ_x`, and
`"plasma"` is `total − vacuum` (reconstruction is linear in the input spectrum).
"""
function _reconstruct_source_b_modes(source::String, vac_data,
                                     pe_odet, equil, ffs_intr, pe_intr, metric, ffit, pe_ctrl,
                                     pe_state)
    source == "total" && return pe_state.b_modes
    flux_matrix = PerturbedEquilibrium.build_flux_matrix(equil, pe_odet, vac_data, ffs_intr)
    forcing_flux = PerturbedEquilibrium.map_forcing_to_eigenmodes(pe_intr.forcing_modes, ffs_intr)
    _, b_vac = PerturbedEquilibrium.reconstruct_physical_fields(
        forcing_flux, flux_matrix, pe_odet, equil, ffs_intr, pe_intr, metric, ffit, pe_ctrl)
    source == "vacuum" && return b_vac
    tot = pe_state.b_modes
    return (R=tot.R .- b_vac.R, Z=tot.Z .- b_vac.Z, phi=tot.phi .- b_vac.phi)
end

"""
    _resonant_surfaces_in_range(pe_state, psi_lo, psi_hi, source)
        -> (res_m, res_n, res_psi, island_width_sq)

Rational surfaces from the perturbed-equilibrium singular coupling that fall within
`[psi_lo, psi_hi]`, with the source-specific complex `island_width_sq =
C_island_width_sq · amp_source`.
"""
function _resonant_surfaces_in_range(pe_state, psi_lo::Float64, psi_hi::Float64, source::String)
    amp = _source_amplitude(source, pe_state)
    isq_all = pe_state.C_island_width_sq * amp
    keep = findall(p -> psi_lo <= p <= psi_hi, pe_state.rational_psi)
    return (pe_state.rational_m_res[keep], pe_state.rational_n[keep],
            pe_state.rational_psi[keep], isq_all[keep])
end

"""
    compute_field_line_tracing(equil, pe_odet, vac_data, ffs_intr, pe_intr, pe_ctrl,
                               flt_ctrl, flt_intr, metric, ffit, pe_state)

Entry point. Reconstructs the requested field source, builds the field model, traces the
field lines, and computes island diagnostics. Returns a filled `FieldLineTracingState`.
"""
function compute_field_line_tracing(equil, pe_odet::OdeState, vac_data,
                                    ffs_intr::ForceFreeStatesInternal,
                                    pe_intr, pe_ctrl,
                                    flt_ctrl::FieldLineTracingControl,
                                    flt_intr::FieldLineTracingInternal,
                                    metric, ffit::FourFitVars, pe_state)
    source = flt_ctrl.tracing_field
    source in ("vacuum", "plasma", "total") ||
        error("tracing_field must be \"vacuum\", \"plasma\", or \"total\" (got \"$source\")")
    flt_ctrl.tracing_coords in ("flux", "real") ||
        error("tracing_coords must be \"flux\" or \"real\" (got \"$(flt_ctrl.tracing_coords)\")")

    nn = ffs_intr.nlow
    mlow = ffs_intr.mlow
    mpert = ffs_intr.mpert
    psi_control = ffs_intr.psilim
    bt_sign = something(equil.params.bt_sign, 1)
    ip_sign = equil.params.crnt === nothing ? 1 : Int(sign(equil.params.crnt))
    helicity = bt_sign * ip_sign

    flt_ctrl.verbose && @info "FieldLineTracing: source=$source, coords=$(flt_ctrl.tracing_coords), n=$nn, lines=$(flt_ctrl.n_lines)"

    # Island-width diagnostic (independent of the trace): GPEC singular-coupling per source.
    psi_hi_isl = min(flt_ctrl.psi_end, psi_control)
    res_m, res_n, res_psi, isq = _resonant_surfaces_in_range(pe_state, flt_ctrl.psi_start, psi_hi_isl, source)

    # The vacuum source traces the exact coil field; total/plasma trace the reconstructed field.
    coil_sets = source == "vacuum" ? pe_intr.coil_sets : ForcingTerms.CoilSet[]
    b_modes = isempty(coil_sets) ?
              _reconstruct_source_b_modes(source, vac_data, pe_odet, equil, ffs_intr,
                                          pe_intr, metric, ffit, pe_ctrl, pe_state) : nothing
    if isempty(coil_sets) && source == "vacuum"
        error("vacuum field-line tracing needs coil geometry (forcing_data_format=\"coil\")")
    end

    state = FieldLineTracingState(; tracing_coords=flt_ctrl.tracing_coords,
                                  tracing_field=source, nn=nn)

    if flt_ctrl.tracing_coords == "flux"
        # True field-line trace: drive from the actual field's radial harmonics.
        # The vacuum coil field is fully 3D — keep n=1…flux_n_max so island bifurcation
        # (competing (2,1) and (4,2) resonances at q=2) is captured; total/plasma are single-n.
        n_vals = source == "vacuum" ? collect(1:max(1, flt_ctrl.flux_n_max)) : [nn]
        psi_lo = max(0.02, flt_ctrl.psi_start - 0.03)
        psi_hi = min(psi_control - 0.005, flt_ctrl.psi_end + 0.03)
        model = build_field_drive_flux_model(equil, nn, mlow, mpert, psi_control, helicity,
                                             psi_lo, psi_hi; n_vals=n_vals, coil_sets=coil_sets,
                                             b_modes=b_modes, pe_psi_grid=Vector{Float64}(pe_state.psi_grid))
        flt_intr.field_model = model
        trace_flux!(state, model, flt_ctrl)
    else
        model = build_real_field_model(equil, nn, mlow, mpert, psi_control, helicity;
                                       coil_sets=coil_sets,
                                       psi_grid=Vector{Float64}(pe_state.psi_grid), b_modes=b_modes)
        flt_intr.field_model = model
        trace_real!(state, model, flt_ctrl, flt_intr.wall_R, flt_intr.wall_Z)
    end

    if flt_ctrl.compute_island_width
        compute_island_diagnostics!(state, res_m, res_n, res_psi, isq)
    end

    flt_ctrl.verbose && @info "FieldLineTracing: $(length(state.punc_R)) punctures, $(length(state.island_m)) resonant surfaces"
    return state
end

end # module FieldLineTracing
