"""
    resize_storage!(odet::OdeState)

Resize storage arrays in `odet` when the current step exceeds allocated size.
Doubles the size of the storage arrays for `u_store`, `du_store`, `xi_s_store`,
`psi_store`, and `q_store`, and copies over existing data to the new arrays.
"""
function resize_storage!(odet::OdeState)
    oldlen = size(odet.u_store, 4)
    newlen = 2 * oldlen

    # Allocate new arrays
    u_new = Array{ComplexF64,4}(undef, odet.numpert_total, odet.numpert_total, 2, newlen)
    du_new = Array{ComplexF64,4}(undef, odet.numpert_total, odet.numpert_total, 2, newlen)
    xi_s_new = Array{ComplexF64,3}(undef, odet.numpert_total, odet.numpert_total, newlen)
    psi_new = Vector{Float64}(undef, newlen)
    q_new = Vector{Float64}(undef, newlen)

    # Copy old data
    u_new[:, :, :, 1:odet.step] = odet.u_store[:, :, :, 1:odet.step]
    du_new[:, :, :, 1:odet.step] = odet.du_store[:, :, :, 1:odet.step]
    xi_s_new[:, :, 1:odet.step] = odet.xi_s_store[:, :, 1:odet.step]
    psi_new[1:odet.step] = odet.psi_store[1:odet.step]
    q_new[1:odet.step] = odet.q_store[1:odet.step]

    # Replace old arrays
    odet.u_store = u_new
    odet.du_store = du_new
    odet.xi_s_store = xi_s_new
    odet.psi_store = psi_new
    odet.q_store = q_new
end

"""
    trim_storage!(odet::OdeState)

Trim storage arrays in `odet` to the actual number of steps taken.
Resizes `u_store`, `du_store`, `xi_s_store`, `psi_store`, and `q_store` to the
current step count, removing any unused allocated space.
"""
function trim_storage!(odet::OdeState)
    resize!(odet.psi_store, odet.step)
    resize!(odet.q_store, odet.step)
    odet.u_store = odet.u_store[:, :, :, 1:odet.step]
    odet.du_store = odet.du_store[:, :, :, 1:odet.step]
    odet.xi_s_store = odet.xi_s_store[:, :, 1:odet.step]
end

"""
    store_ode_data!(odet::OdeState, psi::Float64, u)

Save the current integration state at `psi`: `u` plus `odet.du`, `odet.xi_s`, and `odet.q`
from the latest `sing_der!` call. Callers must evaluate `sing_der!` at exactly `(psi, u)`
first, so the stored derivatives belong to the accepted point rather than the last
internal solver stage.
"""
function store_ode_data!(odet::OdeState, psi::Float64, u)
    if odet.step >= size(odet.u_store, 4)
        resize_storage!(odet)
    end
    odet.psi_store[odet.step] = psi
    odet.q_store[odet.step] = odet.q
    @views odet.u_store[:, :, :, odet.step] .= u
    @views odet.du_store[:, :, :, odet.step] .= odet.du
    @views odet.xi_s_store[:, :, odet.step] .= odet.xi_s
    odet.step += 1
end
