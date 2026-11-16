# Crossing an ideal singular surface through its asymptotic series: the parts the ODE integrators share.

"""
    asymptotic_bases(chunks, ctrl, equil, mats, intr) -> Dict{Int,NamedTuple}

For every crossed surface, keyed by its index in `intr.sing`: the distance `dpsi` from it at which its crossing chunk
ends, the asymptotic series on each side (`left_series`, `right_series`), and those series evaluated at `dpsi`
(`left`, `right`, each N × 2N × 2).
"""
function asymptotic_bases(chunks::Vector{IntegrationChunk}, ctrl::ForceFreeStatesControl,
    equil::Equilibrium.PlasmaEquilibrium, mats::MatrixSplines, intr::ForceFreeStatesInternal)
    bases = Dict{Int,
        @NamedTuple{dpsi::Float64, left::Array{ComplexF64,3}, right::Array{ComplexF64,3}, left_series::SingAsymptotics, right_series::SingAsymptotics}}()
    for chunk in chunks
        chunk.needs_crossing || continue
        singp = intr.sing[chunk.ising]
        dpsi = singp.psifac - chunk.psi_end
        left_series = compute_sing_asymptotics(singp, ctrl, equil, mats, intr; sig=-1.0)
        right_series = compute_sing_asymptotics(singp, ctrl, equil, mats, intr; sig=1.0)
        bases[chunk.ising] = (; dpsi, left=sing_get_ua(left_series, dpsi), right=sing_get_ua(right_series, dpsi), left_series, right_series)
    end
    return bases
end

"""
    state_from_basis!(u, basis, coefficients) -> u

State `u` (N × ncol × 2) of the solutions `basis · coefficients`, for a `basis` (N × 2N × 2) of 2N solutions and
`coefficients` (2N × ncol).
"""
function state_from_basis!(u::AbstractArray{ComplexF64,3}, basis::AbstractArray{ComplexF64,3}, coefficients::AbstractMatrix{ComplexF64})
    @views for half in 1:2
        mul!(u[:, :, half], basis[:, :, half], coefficients)
    end
    return u
end

"""
    eliminate_big_solutions!(ca, ipert_res) -> (pivots, factors)

Column-reduce the asymptotic coefficients `ca` (N × N × 2: basis solution, solution, big or small half of the
basis) so that one column, `pivots[i]`, carries the big solution of each resonance `ipert_res[i]`. `factors[i, j]`
is the multiple of that column added to column `j`.
"""
function eliminate_big_solutions!(ca::Array{ComplexF64,3}, ipert_res)
    N = size(ca, 2)
    pivots = Int[]
    factors = zeros(ComplexF64, length(ipert_res), N)
    free = trues(N)
    for (i, ipert) in enumerate(ipert_res)
        pivot = argmax(j -> free[j] ? abs(ca[ipert, j, 1]) : -Inf, 1:N)
        push!(pivots, pivot)
        free[pivot] = false
        for j in findall(free)
            factors[i, j] = -ca[ipert, j, 1] / ca[ipert, pivot, 1]
            @views ca[:, j, :] .+= factors[i, j] .* ca[:, pivot, :]
            ca[ipert, j, 1] = 0
        end
    end
    return pivots, factors
end

"""
    start_small_solutions!(ca, ipert_res, pivots)

Turn reduced left-side coefficients `ca` into right-side ones [Glasser & Kolemen 2018, Phys. Plasmas 25, 082502,
Eq. 36]: the resonant coefficients vanish and column `pivots[i]`, which carried the big solution of resonance `i`,
becomes its small solution.
"""
function start_small_solutions!(ca::Array{ComplexF64,3}, ipert_res, pivots)
    for (ipert, pivot) in zip(ipert_res, pivots)
        ca[ipert, :, :] .= 0
        ca[:, pivot, :] .= 0
    end
    for (ipert, pivot) in zip(ipert_res, pivots)
        ca[ipert, pivot, 2] = 1
    end
    return ca
end

"""
    crossing_fill_fractions(ctrl) -> Vector{Float64}

Distances from a crossed surface, as fractions of the crossing distance, of the states the asymptotic series adds:
`ctrl.crossing_fill_points` values, log-spaced down to `ctrl.crossing_fill_min / ctrl.singfac_min`.
"""
function crossing_fill_fractions(ctrl::ForceFreeStatesControl)
    n = ctrl.crossing_fill_points
    (n > 0 && 0 < ctrl.crossing_fill_min < ctrl.singfac_min) || return Float64[]
    return (ctrl.crossing_fill_min / ctrl.singfac_min) .^ ((1:n) ./ n)
end

"""
    crossing_fill(series, coefficients, psifac, sig, dpsi, fractions) -> Vector{Tuple{Float64,Array{ComplexF64,3}}}

ψ and state (N × ncol × 2), at the distances `fractions .* dpsi` from the surface at `psifac` on side `sig` (-1
left, +1 right), of the solutions with asymptotic `coefficients` (2N × ncol) in the `series` of that side
[Glasser 2016, Phys. Plasmas 23, 112506, Eq. 45].
"""
function crossing_fill(series::SingAsymptotics, coefficients::AbstractMatrix{ComplexF64}, psifac::Float64, sig::Float64,
    dpsi::Float64, fractions::Vector{Float64})
    N = size(series.vmat, 1)
    return [
        (psifac + sig * fraction * dpsi, state_from_basis!(zeros(ComplexF64, N, size(coefficients, 2), 2), sing_get_ua(series, fraction * dpsi), coefficients))
        for fraction in fractions
    ]
end

"""
    fill_crossings!(odet, bases, fractions, equil, intr)

Add states to the integrator's stored solution inside the crossing distance of every crossed surface, on both sides,
at `fractions` of that distance (`crossing_fill`). That solution is the ideal one, so its big-solution coefficient
is set to zero; solutions that carry a big solution are filled by `outer_solution_basis`. `crit_store` is
interpolated linearly across each surface.
"""
function fill_crossings!(odet::OdeState, bases::AbstractDict, fractions::Vector{Float64}, equil::Equilibrium.PlasmaEquilibrium,
    intr::ForceFreeStatesInternal)
    isempty(fractions) && return
    nstep = odet.step
    filled = Tuple{Float64,Array{ComplexF64,3}}[]
    crit_fill = Float64[]
    for (ising, basis) in bases
        ising <= intr.msing || continue
        singp = intr.sing[ising]
        left = searchsortedlast(@view(odet.psi_store[1:nstep]), singp.psifac)
        # Only surfaces the stored solution stops at on both sides.
        (1 <= left < nstep && isapprox(singp.psifac - odet.psi_store[left], basis.dpsi; rtol=1e-9) &&
         isapprox(odet.psi_store[left+1] - singp.psifac, basis.dpsi; rtol=1e-9)) || continue
        ipert_res = 1 .+ singp.m .- intr.mlow .+ (singp.n .- intr.nlow) .* intr.mpert
        for (sig, node, ua, series) in ((-1.0, left, basis.left, basis.left_series), (1.0, left + 1, basis.right, basis.right_series))
            ca = sing_get_ca(odet.u_store[:, :, :, node], ua, intr)
            ca[ipert_res, :, 1] .= 0
            points = crossing_fill(series, [ca[:, :, 1]; ca[:, :, 2]], singp.psifac, sig, basis.dpsi, fractions)
            append!(filled, points)
            for (psi, _) in points
                weight = (psi - odet.psi_store[left]) / (odet.psi_store[left+1] - odet.psi_store[left])
                push!(crit_fill, (1 - weight) * odet.crit_store[left] + weight * odet.crit_store[left+1])
            end
        end
    end
    isempty(filled) && return

    psi = [odet.psi_store[1:nstep]; first.(filled)]
    order = sortperm(psi)
    odet.psi_store = psi[order]
    odet.crit_store = [odet.crit_store[1:nstep]; crit_fill][order]
    odet.q_store = equil.profiles.q_spline.(odet.psi_store)
    odet.u_store = cat(odet.u_store[:, :, :, 1:nstep], last.(filled)...; dims=4)[:, :, :, order]
    odet.step = length(order)
end
