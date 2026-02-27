"""
    trim_storage!(odet::OdeState)

Trim storage arrays in `odet` to the actual number of steps taken.
Resizes `u_store`, `ud_store`, and `psi_store` to the current step count.
This is a no-op when arrays were pre-allocated to the exact size (normal case);
it only allocates when the integration was truncated early (psiedge < psilim path).
"""
function trim_storage!(odet::OdeState)
    odet.step == length(odet.psi_store) && return
    resize!(odet.psi_store, odet.step)
    odet.u_store = odet.u_store[:, :, 1:odet.step, :]
    odet.ud_store = odet.ud_store[:, :, 1:odet.step, :]
end
