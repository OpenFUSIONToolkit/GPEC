"""
    save_equilibrium_ingest!(out_h5, equil)

Record the equilibrium's raw input arrays under `Input/RawInputs/Equilibrium/`, so a file-based rerun never needs the
original g-file, CHEASE file or IMAS source. Analytic equilibria are regenerated from their deck section instead.
"""
function save_equilibrium_ingest!(out_h5, equil)
    equil.ingest === nothing && return nothing
    eq_group = "Input/RawInputs/Equilibrium"  # read back by Rerun.read_equilibrium_ingest
    out_h5["$eq_group/ingest_kind"] = equil.ingest isa Equilibrium.DirectIngest ? "direct" : "inverse"
    for f in fieldnames(typeof(equil.ingest))
        out_h5["$eq_group/$f"] = getfield(equil.ingest, f)
    end
    return nothing
end
