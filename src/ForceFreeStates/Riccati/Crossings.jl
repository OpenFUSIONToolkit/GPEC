# Singular-surface crossing of the Riccati integrator.

"""
    cross_singular_surface!(odet, intr, ising) -> Matrix{ComplexF64}

Cross singular surface `ising` [Glasser & Kolemen 2018, Phys. Plasmas 25, 082502, Eq. 36]. `odet.u` holds
asymptotic coefficients, in the left basis on entry and the right basis on return, with no big solution on
either side; both sets are recorded in `odet.ca_l` and `odet.ca_r`. Returns the N × N matrix that takes the
columns on return to their continuation left of the surface, in the columns on entry.
"""
function cross_singular_surface!(odet::OdeState, intr::ForceFreeStatesInternal, ising::Int)
    singp = intr.sing[ising]
    ipert_res = 1 .+ singp.m .- intr.mlow .+ (singp.n .- intr.nlow) .* intr.mpert
    pivots, factors = eliminate_big_solutions!(odet.u, ipert_res)
    odet.ca_l[:, :, :, ising] .= odet.u
    start_small_solutions!(odet.u, ipert_res, pivots)
    odet.ca_r[:, :, :, ising] .= odet.u
    odet.ca_populated = true

    continuation = Matrix{ComplexF64}(I, size(odet.u, 2), size(odet.u, 2))
    for (i, pivot) in enumerate(pivots)
        continuation .+= continuation[:, pivot] * transpose(@view factors[i, :])
    end
    # The small solutions started on the right do not exist on the left.
    continuation[:, pivots] .= 0
    return continuation
end
