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
    u_new = Array{ComplexF64,4}(undef, odet.numpert_total, odet.numpert_total, 2, newlen)
    ud_new = Array{ComplexF64,4}(undef, odet.numpert_total, odet.numpert_total, 2, newlen)
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