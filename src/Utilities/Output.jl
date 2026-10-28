using HDF5

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
