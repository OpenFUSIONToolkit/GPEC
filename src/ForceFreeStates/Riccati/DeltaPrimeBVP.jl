# STRIDE global boundary-value problem: Delta-prime matrix assembly, solve, PEST-3 decomposition.

# Relative difference between the two plasma response matrices above which `compute_delta_prime_matrix` warns.
const DELTA_PRIME_RESPONSE_MISMATCH = 1e-6

"""
    compute_delta_prime_matrix(intr, propagators, chunks, axis_state; wv=nothing, wp=nothing, psio=0.0, extended_precision=false) -> Union{Nothing,DeltaPrimeData}

Solve the STRIDE boundary-value problem for the inter-surface Δ' matrix
[Glasser & Kolemen 2018, Phys. Plasmas 25, 082502, Eq. 37].

### Arguments

  - `intr`: Read for its mode space and singular surfaces.
  - `propagators`, `chunks`, `axis_state`: From `riccati_eulerlagrange_integration`.
  - `wv`: Singfac-scaled vacuum response for the edge condition `wp + wv = 0`; `nothing` puts a conducting wall there.
  - `wp`: Plasma response from `free_run`. A warning is issued when the BVP's own differs from it by more than
    `DELTA_PRIME_RESPONSE_MISMATCH`.
  - `psio`: Poloidal flux normalization.
  - `extended_precision`: Solve in `Complex{Double64}`.

### Returns

A [`DeltaPrimeData`](@ref) whose `coil` is Δ_EF (empty without `wv`), or `nothing` when no surface is crossed
or, with a warning, when a surface carries more than one resonant mode.
"""
function compute_delta_prime_matrix(
    intr::ForceFreeStatesInternal,
    propagators::Vector{ChunkPropagator},
    chunks::Vector{IntegrationChunk},
    axis_state::Matrix{ComplexF64};
    wv::Union{Nothing,Matrix{ComplexF64}}=nothing,
    wp::Union{Nothing,Matrix{ComplexF64}}=nothing,
    psio::Float64=0.0,
    extended_precision::Bool=false
)
    intr.msing == 0 && return nothing
    _has_unsupported_multi_resonance(intr) && return nothing

    sing, i_crossings, msing = _select_active_surfaces(intr, chunks)
    msing == 0 && return nothing
    N = intr.numpert_total
    ipert_all = [1 + sing[j].m[1] - intr.mlow + (sing[j].n[1] - intr.nlow) * intr.mpert for j in 1:msing]

    uShootR, uShootL = _shooting_matrices(propagators, chunks, i_crossings)
    uAxis = Matrix(qr(axis_state).Q)[:, 1:N]
    M, col_edge = _assemble_bvp(uShootR, uShootL, uAxis, ipert_all, msing, N, wv, psio)

    # Δ_EF: needs a vacuum edge in the assembled matrix.
    delta_ef = Matrix{ComplexF64}(undef, 0, 0)
    if wv !== nothing
        delta_ef, edge_coefficients = _solve_bvp_delta_ef(M, col_edge, msing, N, ipert_all)
        if wp !== nothing
            # Unit edge displacement, so the edge U₂ is the plasma response times ψ₀².
            wp_bvp = uShootR[msing][(N+1):(2N), :] * edge_coefficients ./ psio^2
            mismatch = norm(wp_bvp - wp) / norm(wp)
            if mismatch > DELTA_PRIME_RESPONSE_MISMATCH
                @warn "Riccati: the plasma response from the Δ' system differs from the axis-to-edge one by $(@sprintf("%.1e", mismatch)) " *
                      "(relative), so the chunk propagators have lost accuracy. Raise [Riccati] nchunks (now $(length(chunks)) chunks); " *
                      "beyond about 150 poloidal harmonics also set extended_precision_bvp."
            end
        end
    end
    # The raw side-major D' (Fortran gal_write_delta layout) feeds `pest3_decompose`; Riccati leaves the blocks unfilled.
    delta_prime, delta_prime_raw = _solve_bvp_and_combine_pest3(M, msing, N, ipert_all, extended_precision)
    return DeltaPrimeData(delta_prime, delta_prime_raw, delta_ef, nothing, nothing, nothing)
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

# Δ_EF (2·msing × N): the small-solution coefficient at each surface side per unit displacement of each edge mode,
# from the BVP with its edge fixed. Also returns the coefficients right of the last surface (2N × N).
function _solve_bvp_delta_ef(M::Matrix{ComplexF64}, col_edge, msing::Int, N::Int, ipert_all::Vector{Int})
    nMat = size(M, 1)
    bot = (nMat-2msing-N+1):(nMat-2msing)   # Eq. (38) bottom rows, carrying the W_V block
    Mc = copy(M)
    Mc[bot, :] .= 0
    Mc[bot, col_edge] .= I(N)               # Dirichlet edge: the edge coefficients equal the source
    B = zeros(ComplexF64, nMat, N)
    B[bot, :] .= I(N)                       # unit displacement, one column per edge poloidal mode
    X = lu(Mc) \ B
    delta_ef = zeros(ComplexF64, 2msing, N)
    for j in 1:msing
        row_left = _col_left(j, N)[ipert_all[j]+N]
        row_right = _col_right(j, N)[ipert_all[j]+N]
        @views delta_ef[2j-1, :] .= X[row_left, :]
        @views delta_ef[2j, :] .= X[row_right, :]
    end
    return delta_ef, X[_col_right(msing, N), :]
end

# Solve the BVP and apply the PEST3 four-term combination, in Complex{Double64} when `extended_precision`.
function _solve_bvp_and_combine_pest3(M::Matrix{ComplexF64}, msing::Int, N::Int, ipert_all::Vector{Int}, extended_precision::Bool)
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
    b = zeros(Tc, nMat)
    for dRow in 1:s2
        fill!(b, 0)
        b[nMat-s2+dRow] = 1
        x = use_lu ? (M_lu \ b) : (M_pinv * b)
        for ksing in 1:msing
            ipert_k = ipert_all[ksing]
            dp_raw[dRow, 2ksing-1] = x[_col_left(ksing, N)[ipert_k+N]]
            dp_raw[dRow, 2ksing] = x[_col_right(ksing, N)[ipert_k+N]]
        end
    end

    # PEST3 four-term combination [Chance PPPL-2527]: Δ'[i,j] = (NW − NE − SW + SE) on each 2×2 block of dp_raw.
    deltap = zeros(Tc, msing, msing)
    for i in 1:msing, j in 1:msing
        deltap[i, j] = dp_raw[2i, 2j] - dp_raw[2i, 2j-1] - dp_raw[2i-1, 2j] + dp_raw[2i-1, 2j-1]
    end
    return ComplexF64.(deltap), ComplexF64.(dp_raw)
end

"""
    pest3_decompose(dp_raw::AbstractMatrix) -> NamedTuple{(:A, :B, :Γ, :Δ)}

Rotate the raw 2m×2m outer-region matching matrix `dp_raw` (side-major
ordering `[L_s1, R_s1, L_s2, R_s2, …]`) into the Pletzer–Dewar 1991 parity
blocks. Given rows and columns paired by surface (odd index = left, even
index = right), the Fortran RDCON parity combination is

```
A'(i,j) = RR + RL + LR + LL    (even-i, even-j)   — interchange↔interchange
B'(i,j) = RR − RL + LR − LL    (even-i, odd-j)    — interchange↔tearing
Γ'(i,j) = RR + RL − LR − LL    (odd-i,  even-j)   — tearing↔interchange
Δ'(i,j) = RR − RL − LR + LL    (odd-i,  odd-j)    — tearing↔tearing
```

where `RR = dp_raw[2i, 2j]`, `RL = dp_raw[2i, 2j−1]`,
`LR = dp_raw[2i−1, 2j]`, `LL = dp_raw[2i−1, 2j−1]`. Each block is m×m.

No Pletzer–Dewar ½ prefactor, matching Fortran RDCON; it cancels in `det(D' − D(γ)) = 0`.

The Δ' block returned here equals `DeltaPrimeData.matrix`.

# Arguments

  - `dp_raw` — 2m×2m complex matrix (typically `DeltaPrimeData.raw`).

# Returns

Named tuple `(A, B, Γ, Δ)` of m×m matrices, the blocks of `D' = [[A' B'] [Γ' Δ']]` in the
`det(D' − D(γ)) = 0` problem (interchange upper-left, tearing lower-right).
"""
function pest3_decompose(dp_raw::AbstractMatrix)
    s2 = size(dp_raw, 1)
    size(dp_raw, 2) == s2 ||
        throw(ArgumentError("pest3_decompose: dp_raw must be square, got $(size(dp_raw))"))
    iseven(s2) ||
        throw(ArgumentError("pest3_decompose: dp_raw side must be 2m for integer m, got $s2"))
    m = s2 ÷ 2
    Tc = eltype(dp_raw)
    Ap = zeros(Tc, m, m)
    Bp = zeros(Tc, m, m)
    Gp = zeros(Tc, m, m)
    Dp = zeros(Tc, m, m)
    for i in 1:m, j in 1:m
        LL = dp_raw[2i-1, 2j-1]
        LR = dp_raw[2i-1, 2j]
        RL = dp_raw[2i, 2j-1]
        RR = dp_raw[2i, 2j]
        Ap[i, j] = RR + RL + LR + LL
        Bp[i, j] = RR - RL + LR - LL
        Gp[i, j] = RR + RL - LR - LL
        Dp[i, j] = RR - RL - LR + LL
    end
    return (A=Ap, B=Bp, Γ=Gp, Δ=Dp)
end
