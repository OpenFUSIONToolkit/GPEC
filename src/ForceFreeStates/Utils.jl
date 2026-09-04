"""
    resize_storage!(odet::OdeState)

Resize storage arrays in `odet` when the current step exceeds allocated size.
Doubles the size of the storage arrays for `u_store`, `psi_store`, and `q_store`, and
copies over existing data to the new arrays. The derivative stores are not grown here:
they are filled at exact size by `materialize_derivative_stores!` after integration.
"""
function resize_storage!(odet::OdeState)
    oldlen = size(odet.u_store, 4)
    newlen = 2 * oldlen

    # Allocate new arrays
    u_new = Array{ComplexF64,4}(undef, odet.numpert_total, odet.numpert_total, 2, newlen)
    psi_new = Vector{Float64}(undef, newlen)
    q_new = Vector{Float64}(undef, newlen)

    # Copy old data
    u_new[:, :, :, 1:odet.step] = odet.u_store[:, :, :, 1:odet.step]
    psi_new[1:odet.step] = odet.psi_store[1:odet.step]
    q_new[1:odet.step] = odet.q_store[1:odet.step]

    # Replace old arrays
    odet.u_store = u_new
    odet.psi_store = psi_new
    odet.q_store = q_new
end

"""
    trim_storage!(odet::OdeState)

Trim storage arrays in `odet` to the actual number of steps taken.
Resizes `u_store`, `psi_store`, and `q_store` to the current step count, removing any
unused allocated space. The derivative stores are left alone — they are either empty
(serial/Riccati, filled later by `materialize_derivative_stores!`) or already exact-size
(galerkin-matched states).
"""
function trim_storage!(odet::OdeState)
    resize!(odet.psi_store, odet.step)
    resize!(odet.q_store, odet.step)
    odet.u_store = odet.u_store[:, :, :, 1:odet.step]
end

"""
    store_ode_data!(odet::OdeState, psi::Float64, u)

Save the current integration state at `psi`: `u` plus `odet.q`, which callers set for the
accepted point. Derivatives are not stored — `materialize_derivative_stores!` recomputes
them from `(psi_store, u_store)` after integration, where they are actually consumed.
"""
function store_ode_data!(odet::OdeState, psi::Float64, u)
    if odet.step >= size(odet.u_store, 4)
        resize_storage!(odet)
    end
    odet.psi_store[odet.step] = psi
    odet.q_store[odet.step] = odet.q
    @views odet.u_store[:, :, :, odet.step] .= u
    odet.step += 1
end

"""
    materialize_derivative_stores!(odet, equil, mats, intr) -> Bool

Fill `odet.du_store` (dΞ_ψ/dψ) and `odet.xi_s_store` from the stored solution, returning
whether they hold valid data afterwards. Idempotent: a no-op when `du_store_populated` is
already true, so the galerkin-matched path keeps its analytic derivatives.

Derivatives are recomputed rather than accumulated during integration because they are
needed only at saved nodes, not at every Runge-Kutta stage. Recomputing after the Gaussian
fixup transforms and the free-boundary normalization is exact rather than merely close: the
Euler-Lagrange system is linear in `u`, and both operations right-multiply the solution by a
mixing matrix `T`, so `du(ψ, u·T) = du(ψ, u)·T`.

Returns `false` without allocating when there is nothing to work from — no `mats`, no stored
steps, or a solution held in a basis the Euler-Lagrange kernel does not apply to (the sparse
parallel path, which stores chunk-endpoint Riccati matrices).
"""
function materialize_derivative_stores!(
    odet::OdeState,
    equil::Equilibrium.PlasmaEquilibrium,
    mats::Union{MatrixSplines,Nothing},
    intr::ModeSpace
)
    odet.du_store_populated && return true
    (isnothing(mats) || odet.step == 0 || isempty(odet.u_store) || !odet.u_store_el_basis) && return false

    kinetic = mats.kinetic !== nothing
    nstep = min(odet.step, size(odet.u_store, 4))
    npert = odet.numpert_total
    odet.du_store = Array{ComplexF64}(undef, npert, npert, nstep)
    odet.xi_s_store = Array{ComplexF64}(undef, npert, npert, nstep)

    du = zeros(ComplexF64, npert, npert, 2)
    u = zeros(ComplexF64, npert, npert, 2)
    @views for istep in 1:nstep
        u .= odet.u_store[:, :, :, istep]
        odet.q = el_derivatives!(du, u, kinetic, equil, mats, intr, odet.psi_store[istep], odet.spline_hint, odet.mats_hint)
        odet.du_store[:, :, istep] .= du[:, :, 1]
        compute_node_xi_s!(odet.xi_s_store[:, :, istep], du[:, :, 1], u[:, :, 1], mats, odet.psi_store[istep];
            kinetic=kinetic, hint=odet.mats_hint)
    end

    odet.du_store_populated = true
    return true
end
