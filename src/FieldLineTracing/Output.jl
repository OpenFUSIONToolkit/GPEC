"""
    Output

HDF5 output for FieldLineTracing results. All results accumulate in a
`FieldLineTracingState` during the run and write to the `field_line_tracing/` group of
`gpec.h5` in a single append pass.
"""

# Idempotent group access so a rerun overwrites cleanly.
_group(parent, name) = haskey(parent, name) ? parent[name] : create_group(parent, name)

"""
    write_to_hdf5!(h5file::HDF5.File, state::FieldLineTracingState)

Write all FieldLineTracing results to the `field_line_tracing/` group in `gpec.h5`.

# Arguments
- `h5file::HDF5.File`: open HDF5 file handle (opened `"cw"` by the pipeline).
- `state::FieldLineTracingState`: accumulated results.
"""
function write_to_hdf5!(h5file::HDF5.File, state::FieldLineTracingState)
    g = _group(h5file, "field_line_tracing")

    cfg = _group(g, "config")
    cfg["tracing_coords"] = state.tracing_coords
    cfg["tracing_field"] = state.tracing_field
    cfg["nn"] = state.nn

    punc = _group(g, "punctures")
    punc["R"] = state.punc_R
    punc["Z"] = state.punc_Z
    punc["psi"] = state.punc_psi
    punc["theta"] = state.punc_theta
    punc["line_id"] = state.punc_line
    punc["phi"] = state.punc_phi

    cl = _group(g, "connection_length")
    cl["R"] = state.cl_R
    cl["Z"] = state.cl_Z
    cl["psi"] = state.cl_psi
    cl["length"] = state.cl_length
    cl["class"] = state.cl_class

    fp = _group(g, "footprints")
    fp["R"] = state.fp_R
    fp["Z"] = state.fp_Z
    fp["phi"] = state.fp_phi

    isl = _group(g, "islands")
    isl["m"] = state.island_m
    isl["n"] = state.island_n
    isl["psi"] = state.island_psi
    isl["half_width"] = state.island_halfwidth
    isl["chirikov"] = state.island_chirikov
    return nothing
end
