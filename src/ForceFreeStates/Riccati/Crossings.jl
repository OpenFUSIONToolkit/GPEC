# Singular-surface crossing of the Riccati integrator, in the asymptotic basis.

"""
    asymptotic_bases(chunks, ctrl, equil, mats, intr) -> Dict{Int,NamedTuple}

Asymptotic basis `(; left, right)`, each N × 2N × 2, on both sides of every crossed surface, where its crossing
chunk ends. Keyed by the surface's index in `intr.sing`.
"""
function asymptotic_bases(chunks::Vector{IntegrationChunk}, ctrl::ForceFreeStatesControl,
    equil::Equilibrium.PlasmaEquilibrium, mats::MatrixSplines, intr::ForceFreeStatesInternal)
    bases = Dict{Int,@NamedTuple{left::Array{ComplexF64,3}, right::Array{ComplexF64,3}}}()
    for chunk in chunks
        chunk.needs_crossing || continue
        singp = intr.sing[chunk.ising]
        dpsi = singp.psifac - chunk.psi_end
        bases[chunk.ising] = (; left=sing_get_ua(compute_sing_asymptotics(singp, ctrl, equil, mats, intr; sig=-1.0), dpsi),
            right=sing_get_ua(compute_sing_asymptotics(singp, ctrl, equil, mats, intr; sig=1.0), dpsi))
    end
    return bases
end

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
    continuation = eliminate_big_solutions!(odet.u, ipert_res)
    odet.ca_l[:, :, :, ising] .= odet.u
    start_small_solutions!(odet.u, ipert_res)
    odet.ca_r[:, :, :, ising] .= odet.u
    odet.ca_populated = true
    # The small solutions introduced on the right do not exist on the left.
    continuation[:, ipert_res] .= 0
    return continuation
end

"""
    eliminate_big_solutions!(ca, ipert_res) -> Matrix{ComplexF64}

Column-reduce the asymptotic coefficients `ca` (N × N × 2: basis solution, solution, big or small half of the
basis) so that only column `ipert_res[i]` carries the big solution of resonance `i`. Returns the N × N column
transformation applied.
"""
function eliminate_big_solutions!(ca::Array{ComplexF64,3}, ipert_res)
    transform = Matrix{ComplexF64}(I, size(ca, 2), size(ca, 2))
    for (i, ipert) in enumerate(ipert_res)
        candidates = setdiff(axes(ca, 2), @view ipert_res[1:(i-1)])
        pivot = candidates[argmax(abs.(@view ca[ipert, candidates, 1]))]
        for col in axes(ca, 2)
            col == pivot && continue
            factor = ca[ipert, col, 1] / ca[ipert, pivot, 1]
            @views ca[:, col, :] .-= factor .* ca[:, pivot, :]
            @views transform[:, col] .-= factor .* transform[:, pivot]
            ca[ipert, col, 1] = 0
        end
        if pivot != ipert
            swapped = ca[:, pivot, :]
            @views ca[:, pivot, :] .= ca[:, ipert, :]
            ca[:, ipert, :] .= swapped
            transform[:, [pivot, ipert]] .= transform[:, [ipert, pivot]]
        end
    end
    return transform
end

"""
    start_small_solutions!(ca, ipert_res)

Turn reduced left-side coefficients `ca` into right-side ones: the resonant coefficients vanish and column
`ipert_res[i]` becomes the small solution of resonance `i`.
"""
function start_small_solutions!(ca::Array{ComplexF64,3}, ipert_res)
    for ipert in ipert_res
        ca[ipert, :, :] .= 0
        ca[:, ipert, :] .= 0
    end
    for ipert in ipert_res
        ca[ipert, ipert, 2] = 1
    end
    return ca
end
