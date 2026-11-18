# STRIDE global boundary-value problem: Delta-prime matrix assembly, solve, PEST-3 decomposition.

# Relative difference between the two plasma response matrices above which `compute_delta_prime_matrix` warns.
const DELTA_PRIME_RESPONSE_MISMATCH = 1e-6
# Integration tolerance times the size of the big solution against the small one at the crossing distance, above which
# the small solution of a surface is not resolved.
const UNRESOLVED_SMALL_SOLUTION = 1.0

"""
    warn_unresolved_surfaces(bases, surfaces, rtol)

Warn for each crossed surface whose big asymptotic solution exceeds its small one, at the crossing distance, by more
than the integration tolerance `rtol` resolves, so that Δ' and the outer solutions there are unreliable. `bases` is
from `asymptotic_bases`; `surfaces` are the singular surfaces it indexes.
"""
function warn_unresolved_surfaces(bases::AbstractDict, surfaces::Vector{SingType}, rtol::Float64)
    for ising in sort!(collect(keys(bases)))
        ising <= length(surfaces) || continue
        alpha = maximum(real, bases[ising].left_series.alpha)
        ratio = bases[ising].dpsi^(-2 * alpha)
        rtol * ratio > UNRESOLVED_SMALL_SOLUTION &&
            @warn "Riccati: at the rational surface ψ = $(@sprintf("%.4f", surfaces[ising].psifac)) (m = $(join(surfaces[ising].m, ", "))) the big solution is " *
                  "$(@sprintf("%.1e", ratio)) times the small one at the crossing distance (Mercier exponent α = $(@sprintf("%.2f", alpha))), more than " *
                  "eulerlagrange_tolerance = $rtol resolves. Δ' and the outer solutions at this surface are unreliable."
    end
end

"""
    compute_delta_prime_matrix(intr, propagators, chunks, axis_state; wv=nothing, wp=nothing, psio=0.0,
        extended_precision=false, error_field_response=true) -> (delta_prime, coefficients)

Solve the STRIDE boundary-value problem for the inter-surface Δ' matrices
[Glasser & Kolemen 2018, Phys. Plasmas 25, 082502, Eq. 37]: with the vacuum at the edge, and with ξ(edge) = 0.

### Arguments

  - `intr`: Read for its mode space and singular surfaces.
  - `propagators`, `chunks`, `axis_state`: From `riccati_eulerlagrange_integration`.
  - `wv`: Singfac-scaled vacuum response for the edge condition `wp + wv = 0`. With `nothing` only the ξ(edge) = 0
    problem is solved, and no Δ_EF.
  - `wp`: The run's plasma response, used only as a check: a warning is issued when the edge momentum of the Δ_EF
    solve differs from it by more than `DELTA_PRIME_RESPONSE_MISMATCH`.
  - `psio`: Poloidal flux normalization.
  - `extended_precision`: Solve in `Complex{Double64}`.
  - `error_field_response`: Also solve for Δ_EF.

### Returns

  - `delta_prime`: A [`DeltaPrimeData`](@ref).
  - `coefficients`: The solution vectors of the ξ(edge) = 0 problem, one column per solution: the 2·msing Δ' solves,
    side-major, then the N of Δ_EF. Input of `outer_solution_basis`; `nothing` without Δ_EF.

Both are `nothing` when no surface is crossed or, with a warning, when a surface carries more than one resonant mode.
"""
function compute_delta_prime_matrix(
    intr::ForceFreeStatesInternal,
    propagators::Vector{ChunkPropagator},
    chunks::Vector{IntegrationChunk},
    axis_state::Matrix{ComplexF64};
    wv::Union{Nothing,Matrix{ComplexF64}}=nothing,
    wp::Union{Nothing,Matrix{ComplexF64}}=nothing,
    psio::Float64=0.0,
    extended_precision::Bool=false,
    error_field_response::Bool=true
)
    intr.msing == 0 && return nothing, nothing
    _has_unsupported_multi_resonance(intr) && return nothing, nothing

    sing, i_crossings, msing = _select_active_surfaces(intr, chunks)
    msing == 0 && return nothing, nothing
    N = intr.numpert_total
    ipert_all = [1 + sing[j].m[1] - intr.mlow + (sing[j].n[1] - intr.nlow) * intr.mpert for j in 1:msing]

    uShootR, uShootL = _shooting_matrices(propagators, chunks, i_crossings)
    uAxis = Matrix(qr(axis_state).Q)[:, 1:N]
    M, col_edge = _assemble_bvp(uShootR, uShootL, uAxis, ipert_all, msing, N, wv, psio)

    none = Matrix{ComplexF64}(undef, 0, 0)
    if wv === nothing
        # Assembled with the conducting-wall edge, so this is the ξ(edge) = 0 problem.
        raw_fixedboundary, edge_response, _ = _solve_bvp_raw(M, msing, N, ipert_all, col_edge, extended_precision)
        return DeltaPrimeData(none, raw_fixedboundary, none, edge_response), nothing
    end
    raw_freeboundary, _, _ = _solve_bvp_raw(M, msing, N, ipert_all, col_edge, extended_precision)
    Mc = _fixed_edge_bvp(M, msing, N, col_edge)
    raw_fixedboundary, edge_response, delta_prime_solutions = _solve_bvp_raw(Mc, msing, N, ipert_all, col_edge, extended_precision)
    error_field_response || return DeltaPrimeData(raw_freeboundary, raw_fixedboundary, none, edge_response), nothing

    delta_ef, delta_ef_solutions = _solve_bvp_delta_ef(Mc, msing, N, ipert_all)
    if wp !== nothing
        # Unit edge displacement, so the edge unknowns U₂(edge) are the plasma response times ψ₀².
        wp_bvp = delta_ef_solutions[col_edge, :] ./ psio^2
        mismatch = norm(wp_bvp - wp) / norm(wp)
        if mismatch > DELTA_PRIME_RESPONSE_MISMATCH
            @warn "Riccati: the plasma response from the Δ' system differs from the axis-to-edge one by $(@sprintf("%.1e", mismatch)) " *
                  "(relative), so the chunk propagators have lost accuracy. Raise [Riccati] nchunks (now $(length(chunks)) chunks); " *
                  "beyond about 150 poloidal harmonics also set extended_precision_bvp."
        end
    end
    return DeltaPrimeData(raw_freeboundary, raw_fixedboundary, delta_ef, edge_response), [delta_prime_solutions delta_ef_solutions]
end

# Column index helpers for the BVP matrix. j is the 1-based singular-surface index,
# N is numpert_total. Layout: c_axis(N), c_left[1](2N), c_right[1](2N), ..., c_edge(N).
_col_left(j::Int, N::Int) = (N+4N*(j-1)+1):(N+4N*(j-1)+2N)
_col_right(j::Int, N::Int) = (N+4N*(j-1)+2N+1):(N+4N*j)

# True (with a warning) if any surface carries more than one resonant (m, n), which the BVP does not handle.
function _has_unsupported_multi_resonance(intr::ForceFreeStatesInternal)
    msing = intr.msing
    n_res_per_surface = [length(intr.sing[j].m) for j in 1:msing]
    any(>(1), n_res_per_surface) || return false
    offenders = [(j, intr.sing[j].m, intr.sing[j].n) for j in 1:msing if n_res_per_surface[j] > 1]
    @warn "compute_delta_prime_matrix: skipping inter-surface Δ' BVP because some surfaces carry more than one resonant mode " *
          "(multi-n collision; generalization tracked as follow-up). " *
          "Per-surface Δ' is unaffected. Multi-resonance surfaces: $offenders"
    return true
end

# Map BVP surface index (1:msing_active) → intr.sing index using chunk.ising. Surfaces
# may be excluded at either end (below qlow or beyond psilim); each crossing chunk
# records its original surface index. Returns (sing alias, i_crossings, msing_active).
function _select_active_surfaces(intr::ForceFreeStatesInternal, chunks::Vector{IntegrationChunk})
    msing = intr.msing
    i_crossings = findall(c -> c.needs_crossing, chunks)
    sing_indices = [chunks[ic].ising for ic in i_crossings]
    msing_active = length(i_crossings)
    if msing_active < msing
        excluded = setdiff(1:msing, sing_indices)
        excluded_ms = [intr.sing[j].m for j in excluded]
        @debug "compute_delta_prime_matrix: $msing singular surfaces, $msing_active crossed (excluded: m=$excluded_ms)"
    end
    sing = [intr.sing[si] for si in sing_indices]
    return sing, i_crossings, msing_active
end

# Shooting matrices of the BVP [Glasser & Kolemen 2018, Phys. Plasmas 25, 082502, Eqs. 33-35]: `uShootR[j]` and
# `uShootL[j]` map the asymptotic coefficients right and left of surface j to the state at the matching point
# toward the next and the previous surface.
function _shooting_matrices(propagators::Vector{ChunkPropagator}, chunks::Vector{IntegrationChunk}, i_crossings::Vector{Int})
    msing = length(i_crossings)
    matrix(i) = [propagators[i].u[:, :, 1]; propagators[i].u[:, :, 2]]
    uShootR = Vector{Matrix{ComplexF64}}(undef, msing)
    uShootL = Vector{Matrix{ComplexF64}}(undef, msing)
    for j in 1:msing
        # The chunk after the surface starts from the right basis; the rest multiply on.
        range_R = _midpoint_shoot_range(chunks, i_crossings, j, msing; side=:right)
        uShootR[j] = foldl((X, i) -> matrix(i) * X, range_R[2:end]; init=matrix(first(range_R)))
        # The crossing chunk carries the left basis back to its start; forward chunks before it are inverted.
        range_L = j == 1 ? (1:0) : _midpoint_shoot_range(chunks, i_crossings, j, msing; side=:left)
        uShootL[j] = foldl((X, i) -> matrix(i) \ X, reverse(range_L); init=matrix(i_crossings[j]))
    end
    return uShootR, uShootL
end

# Chunk range from surface j to the ψ midpoint toward the next surface (`:right`; the edge for
# j == msing) or from the midpoint after surface j-1 to surface j (`:left`). The ψ midpoint, not the
# chunk-index midpoint, matches Fortran since chunks pack tighter near singularities.
function _midpoint_shoot_range(chunks::Vector{IntegrationChunk}, i_crossings::Vector{Int},
    j::Int, msing::Int; side::Symbol)
    if side === :right
        j == msing && return (i_crossings[msing]+1):length(chunks)
        chunk_start = i_crossings[j] + 1
        chunk_end = i_crossings[j+1] - 1
    else  # :left, j >= 2
        chunk_start = i_crossings[j-1] + 1
        chunk_end = i_crossings[j] - 1
    end
    psi_mid_target = (chunks[chunk_start].psi_start + chunks[chunk_end].psi_end) / 2
    i_mid_inter = chunk_start
    for ic in chunk_start:(chunk_end-1)
        if chunks[ic].psi_end >= psi_mid_target
            i_mid_inter = ic
            break
        end
        i_mid_inter = ic
    end
    return side === :right ? (chunk_start:i_mid_inter) : ((i_mid_inter+1):chunk_end)
end

# Assemble the BVP matrix M with the axis matched to the axis-regular solutions `uAxis` at the start of the first
# crossing chunk. Fortran-matched layout, nMat = (2 + 4·msing)·N. Returns (M, col_edge).
function _assemble_bvp(uShootR::Vector{Matrix{ComplexF64}},
    uShootL::Vector{Matrix{ComplexF64}},
    uAxis::Matrix{ComplexF64}, ipert_all::Vector{Int},
    msing::Int, N::Int,
    wv::Union{Nothing,Matrix{ComplexF64}}, psio::Float64)
    # STRIDE global BVP block structure [Glasser & Kolemen 2018, Phys. Plasmas 25, 082502, Eq. 37].
    nMat = (2 + 4 * msing) * N
    col_axis = 1:N
    col_edge = (nMat-N+1):nMat
    M = zeros(ComplexF64, nMat, nMat)

    # Axis matching: uShootL[1] · c_left[1] = uAxis · c_axis  (2N equations)
    M[1:2N, _col_left(1, N)] .= uShootL[1]
    M[1:2N, col_axis] .= -uAxis
    row_offset = 2N

    for j in 1:msing
        ipert_j = ipert_all[j]
        # Crossing: non-resonant modes continuity (asymptotic basis = identity)
        for i in 1:2N
            if i != ipert_j && i != ipert_j + N
                row_offset += 1
                M[row_offset, _col_left(j, N)[i]] = 1
                M[row_offset, _col_right(j, N)[i]] = -1
            end
        end

        junc_rows = (row_offset+1):(row_offset+2N)
        if j < msing
            # Midpoint matching between consecutive surfaces
            M[junc_rows, _col_right(j, N)] .= -uShootR[j]
            M[junc_rows, _col_left(j + 1, N)] .= uShootL[j+1]
        else
            # Edge junction
            M[junc_rows, _col_right(msing, N)] .= uShootR[msing]
            if wv !== nothing
                M[junc_rows[1:N], col_edge] .= -I(N)
                M[junc_rows[(N+1):end], col_edge] .= wv .* psio^2
            else
                M[junc_rows[(N+1):end], col_edge] .= -I(N)
            end
        end
        row_offset = last(junc_rows)
    end

    # Driving rows: set big-solution coefficient = 1 at each surface (asymptotic basis)
    for j in 1:msing
        ipert_j = ipert_all[j]
        row_offset += 1
        M[row_offset, _col_left(j, N)[ipert_j]] = 1
        row_offset += 1
        M[row_offset, _col_right(j, N)[ipert_j]] = 1
    end
    @assert row_offset == nMat "Row count mismatch: expected $nMat, got $row_offset"
    return M, col_edge
end

# The BVP matrix `M` with its Eq. (38) vacuum-edge rows [Glasser & Kolemen 2018, Phys. Plasmas 25, 082502] replaced by
# the conducting-wall edge of the `wv = nothing` assembly: U₁(edge) equals the right-hand side of the first N edge rows
# (0 for the fixed-boundary D') and the edge unknowns hold U₂(edge).
function _fixed_edge_bvp(M::Matrix{ComplexF64}, msing::Int, N::Int, col_edge)
    nMat = size(M, 1)
    edge_rows = (nMat-2msing-2N+1):(nMat-2msing)
    Mc = copy(M)
    Mc[edge_rows, col_edge] .= 0
    Mc[edge_rows[(N+1):end], col_edge] .= -I(N)
    return Mc
end

# Δ_EF (2·msing × N) from the fixed-edge BVP matrix `Mc`: the small-solution coefficient at each surface side for a
# unit edge displacement in each poloidal mode. Also returns the N solution vectors.
function _solve_bvp_delta_ef(Mc::Matrix{ComplexF64}, msing::Int, N::Int, ipert_all::Vector{Int})
    nMat = size(Mc, 1)
    u1_rows = (nMat-2msing-2N+1):(nMat-2msing-N)
    B = zeros(ComplexF64, nMat, N)
    B[u1_rows, :] .= I(N)                   # unit edge displacement, one column per edge poloidal mode
    X = lu(Mc) \ B
    delta_ef = zeros(ComplexF64, 2msing, N)
    for j in 1:msing
        row_left = _col_left(j, N)[ipert_all[j]+N]
        row_right = _col_right(j, N)[ipert_all[j]+N]
        @views delta_ef[2j-1, :] .= X[row_left, :]
        @views delta_ef[2j, :] .= X[row_right, :]
    end
    return delta_ef, X
end

# Solve the BVP for the raw side-major D' (2·msing × 2·msing), in Complex{Double64} when `extended_precision`. Also
# returns the edge unknowns of each solve (N × 2·msing) and the 2·msing solution vectors.
function _solve_bvp_raw(M::Matrix{ComplexF64}, msing::Int, N::Int, ipert_all::Vector{Int}, col_edge, extended_precision::Bool)
    nMat = size(M, 1)
    s2 = 2 * msing
    Tc = extended_precision ? Complex{Double64} : ComplexF64
    M_solve = Tc.(M)
    M_lu = lu(M_solve; check=false)
    use_lu = issuccess(M_lu)
    M_pinv = use_lu ? nothing : pinv(M_solve)
    if !use_lu
        @warn "Δ' BVP: LU factorization singular (rank $(rank(M))/$nMat), using pseudo-inverse fallback"
    end

    # One solve per surface side, driving that side's big-solution coefficient.
    dp_raw = zeros(Tc, s2, s2)
    solutions = zeros(ComplexF64, nMat, s2)
    b = zeros(Tc, nMat)
    for dRow in 1:s2
        fill!(b, 0)
        b[nMat-s2+dRow] = 1
        x = use_lu ? (M_lu \ b) : (M_pinv * b)
        solutions[:, dRow] .= x
        for ksing in 1:msing
            ipert_k = ipert_all[ksing]
            dp_raw[dRow, 2ksing-1] = x[_col_left(ksing, N)[ipert_k+N]]
            dp_raw[dRow, 2ksing] = x[_col_right(ksing, N)[ipert_k+N]]
        end
    end
    return ComplexF64.(dp_raw), solutions[col_edge, :], solutions
end
