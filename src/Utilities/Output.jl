using HDF5
import TOML

"""
    save_to_hdf5(target, result; kwargs...)

Save a computed result to an HDF5 path, open file, or group. Each producing module
implements the method for its result type and an open `HDF5.File` or `HDF5.Group`.
The path method opens or creates the file, preserves existing contents, and closes
its handle after saving. An open target remains owned by the caller.
"""
function save_to_hdf5 end

function save_to_hdf5(path::AbstractString, result; kwargs...)
    return h5open(path, "cw") do target
        save_to_hdf5(target, result; kwargs...)
    end
end

"""
    deck_sections(result) -> Dict{String,Any}

The deck sections stating the inputs `result` was computed from, so a saved run records a deck that reproduces it.
The methods live with the deck translation of the top module, the inverse of its section parser.
"""
function deck_sections end

"""
    record_inputs!(target, result)

Merge the deck sections of `result` into the run record `Input/gpec_toml_raw` of `target`, replacing any section of
the same name, so every saved result adds the deck it was computed from.
"""
function record_inputs!(target::Union{HDF5.File,HDF5.Group}, result)
    path = "Input/gpec_toml_raw"
    deck = haskey(target, path) ? TOML.parse(read(target[path])) : Dict{String,Any}()
    merge!(deck, deck_sections(result))
    haskey(target, path) && HDF5.delete_object(target, path)
    target[path] = sprint(TOML.print, deck)
    return nothing
end
