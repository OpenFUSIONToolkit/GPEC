"""
    init_files(out::DconOutput, path::String)

Open requested output text files into `path` and store handles.
"""
function init_files(out::DconOutput, path::String)
    for pname in propertynames(out)
        if startswith(String(pname), "write_") && getproperty(out, pname)
            base_str = String(pname)[7:end]          # remove write_ prefix
            base = Symbol(base_str)                       # key for handles
            fname_field = Symbol("fname_" * base_str) # add fname_ prefix
            fname = getproperty(out, fname_field)

            full_path = joinpath(path, fname)

            if !endswith(fname, ".h5") # we don't pre-open h5 files
                if endswith(fname, ".out") || endswith(fname, ".txt")
                    out.handles[base] = open(full_path, "w")
                else
                    error("Unknown file extension for $fname, need to add to init_files!")
                end
            end
        end
    end
end

"""
    write_output(out::DconOutput, key::Symbol, data; dsetname=nothing, slice=:)

Writes `data` to the text file corresponding to `key`.
"""
function write_output(out::DconOutput, key::Symbol, data)
    handle = out.handles[key]

    if handle isa IOStream
        println(handle, data)
    else
        error("Unsupported handle type for key $key")
    end
end

"""
    close_files(out::DconOutput)

Closes all open text files.
"""
function close_files(out::DconOutput)
    for (key, handle) in out.handles
        close(handle)
    end
    empty!(out.handles)
end

"""
    resize_storage!(odet::OdeState)

Resize storage arrays in `odet` when the current step exceeds allocated size.
Doubles the size of the storage arrays for `u_store`, `ud_store`, `psi_store`,
and `q_store`, and copies over existing data to the new arrays.
"""
function resize_storage!(odet::OdeState)
    oldlen = size(odet.u_store, 4)
    newlen = 2 * oldlen

    # Allocate new arrays
    u_new = Array{ComplexF64,4}(undef, odet.mpert, odet.mpert, 2, newlen)
    ud_new = Array{ComplexF64,4}(undef, odet.mpert, odet.mpert, 2, newlen)
    psi_new = Vector{Float64}(undef, newlen)
    q_new = Vector{Float64}(undef, newlen)

    # Copy old data
    u_new[:, :, :, 1:odet.step] = odet.u_store[:, :, :, 1:odet.step]
    ud_new[:, :, :, 1:odet.step] = odet.ud_store[:, :, :, 1:odet.step]
    psi_new[1:odet.step] = odet.psi_store[1:odet.step]
    q_new[1:odet.step] = odet.q_store[1:odet.step]

    # Replace old arrays
    odet.u_store = u_new
    odet.ud_store = ud_new
    odet.psi_store = psi_new
    odet.q_store = q_new
end

"""
    trim_storage!(odet::OdeState)

Trim storage arrays in `odet` to the actual number of steps taken.
Resizes `u_store`, `ud_store`, `psi_store`, and `q_store` to the
current step count, removing any unused allocated space.
"""
function trim_storage!(odet::OdeState)
    resize!(odet.psi_store, odet.step)
    resize!(odet.q_store, odet.step)
    odet.u_store = odet.u_store[:, :, :, 1:odet.step]
    odet.ud_store = odet.ud_store[:, :, :, 1:odet.step]
end