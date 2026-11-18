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

"""
    save_input_files!(out_h5, equil, wall_settings)

Record the external files a run read, so a rerun and `equilibrium_from_h5` replay them without the originals:
the equilibrium's kinetic profiles under `Input/RawInputs/Kinetic/` (the kinetic HDF5 schema; the composition is
restated by the deck) and a file wall's text as `Input/RawInputs/Wall/wall_file_raw`.
"""
function save_input_files!(out_h5, equil, wall_settings)
    equil.kinetic === nothing || Equilibrium.write_kinetic_group!(create_group(out_h5, "Input/RawInputs/Kinetic"), equil.kinetic)
    wall_settings isa Vacuum.FileWall && (out_h5["Input/RawInputs/Wall/wall_file_raw"] = read(wall_settings.file, String))
    return nothing
end
