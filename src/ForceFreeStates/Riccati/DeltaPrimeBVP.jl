# STRIDE global boundary-value problem: Delta-prime matrix assembly, solve, PEST-3 decomposition.

"""
    compute_delta_prime_matrix!(intr, propagators, chunks; wv, psio, debug, ctrl, equil, mats)

Compute the inter-surface tearing stability matrix (msing × msing) using the
STRIDE global BVP formulation [Glasser 2018 Phys. Plasmas 25, 032501, Sec. III.B].

The BVP encodes the full plasma response with unknowns at each surface boundary:
```
  x_axis      (N):  free IC parameters at the axis  (U₁ = 0 regular solutions)
  x_left[j]  (2N):  state at left inner-layer boundary of surface j
  x_right[j] (2N):  state at right inner-layer boundary of surface j
  x_edge      (N):  free IC parameters at the edge
  Total unknowns: nMat = (2 + 4·msing)·N
```

## Edge boundary condition

When `wv` is provided (the vacuum response matrix, singfac-scaled), the edge BC
follows the Fortran STRIDE convention:
```
  U₁ = c,  U₂ = -wv·ψ₀²·c
```
which is the free-boundary condition `wp + wv = 0` at the edge.
When `wv` is `nothing`, a conducting wall BC (`U₁ = 0`) is used.

## Gaussian reduction (conditioning)

Forward-propagated segment propagators (axis→surface, surface→surface) can be
extremely ill-conditioned (cond ~ 10²⁴) due to exponential growth of the big
solution. Following STRIDE's `ode_fixup`, Gaussian reduction is applied to each
assembled propagator's U₂ columns before inserting into the BVP matrix. This
keeps the BVP matrix full-rank and well-conditioned.

## Output: PEST3-convention Δ' (deltap)

The raw BVP solution is a 2·msing × 2·msing matrix `dp` with left/right
sub-indices at each surface. The PEST3-convention Δ' matrix is the linear
combination [Chance, PPPL-2527]:
```
  deltap(i,j) = dp(2i,2j) - dp(2i,2j-1) - dp(2i-1,2j) + dp(2i-1,2j-1)
```
stored in `intr.delta_prime_matrix` (msing × msing).

## Limitations

This routine currently assumes exactly one resonant mode per singular surface
(the standard single-`n` case).  When **any** surface carries more than one
resonant mode — i.e., a multi-`n` run where a single q value satisfies two
distinct `(m, n)` tuples (e.g. q = 2 with `(m=2, n=1)` AND `(m=4, n=2)`) —
the routine emits a warning and skips the inter-surface BVP rather than
crashing.  Generalizing the BVP to multi-resonance surfaces is tracked as a
follow-up: the matrix shape becomes `n_res_total × n_res_total` with
`n_res_total = sum(length(intr.sing[j].m))` and a `(surface, mode, side)`
↔ BVP-row map; see PR discussion.

Note: `intr.delta_prime_matrix` is the **only physically valid Δ'** produced
by this code. The per-surface ca-based stub `intr.sing[*].delta_prime` /
`delta_prime_col` (populated by `riccati_cross_ideal_singular_surf!`) is a
diagnostic placeholder for future intra-surface coupling work and is not
expected to agree with `delta_prime_matrix`.
"""
function compute_delta_prime_matrix!(
    intr::ForceFreeStatesInternal,
    propagators::Vector{ChunkPropagator},
    chunks::Vector{IntegrationChunk};
    wv::Union{Nothing,Matrix{ComplexF64}} = nothing,
    psio::Float64 = 0.0,
    debug::Bool = false,
    S_at_surface_left::Union{Nothing,Vector{Matrix{ComplexF64}}} = nothing,
    ctrl::Union{Nothing,ForceFreeStatesControl} = nothing,
    equil::Union{Nothing,Equilibrium.PlasmaEquilibrium} = nothing,
    mats::Union{Nothing,MatrixSplines} = nothing
)
    intr.msing == 0 && return
    _has_unsupported_multi_resonance(intr) && return

    sing, i_crossings, msing = _select_active_surfaces(intr, chunks)
    msing == 0 && return
    N = intr.numpert_total

    use_S_axis = S_at_surface_left !== nothing && length(S_at_surface_left) == msing

    # The FM-axis-BC fallback (use_S_axis=false) wires Phi_L_mats[j] as forward propagators
    # in the BVP matrix. Crossing chunks with direction=-1 (bidirectional parallel FM) hold
    # *backward* propagators, so applying them as forward would produce a silently wrong
    # Δ' BVP. Forbid that combination explicitly — the parallel path always supplies
    # S_at_surface_left (so use_S_axis=true) and any new caller hitting the FM-axis path
    # needs forward crossing chunks.
    if !use_S_axis
        for ic in i_crossings
            chunks[ic].direction == 1 ||
                error("compute_delta_prime_matrix!: FM-axis fallback (use_S_axis=false) requires forward crossing chunks; " *
                      "chunk $ic has direction=$(chunks[ic].direction). Either provide S_at_surface_left or use bidirectional=false.")
        end
    end

    Phi_L_mats, Phi_R_mats, Phi_R_halves = _assemble_segment_propagators(
        propagators, chunks, i_crossings, msing, N, use_S_axis)

    ipert_all = [1 + sing[j].m[1] - intr.mlow + (sing[j].n[1] - intr.nlow) * intr.mpert for j in 1:msing]
    has_ua = all(j -> !isempty(sing[j].ua_left), 1:msing)
    T_left_mats, T_right_mats, T_left_inv, T_right_inv =
        _build_asymptotic_basis_matrices(sing, has_ua, N, msing)

    debug && _log_bvp_setup(chunks, sing, S_at_surface_left, use_S_axis, has_ua,
                            Phi_L_mats, Phi_R_mats, Phi_R_halves, ipert_all, wv, psio, N, msing)

    if use_S_axis
        uShootR, uShootL, uAxis = _build_S_axis_shooting_propagators(
            propagators, chunks, i_crossings, sing, msing, N,
            T_left_mats, T_right_mats, has_ua, ctrl, equil, mats, intr, debug)
        debug && _log_S_axis_shooting_propagators(uShootR, uShootL, uAxis,
                                                  S_at_surface_left, T_left_mats,
                                                  ipert_all, has_ua, msing, N)
        M, nMat, col_edge = _assemble_bvp_S_axis(
            uShootR, uShootL, uAxis, ipert_all, msing, N, wv, psio)
    else
        M, nMat, col_edge = _assemble_bvp_FM_axis(
            Phi_L_mats, Phi_R_mats, ipert_all, msing, N,
            T_left_inv, T_right_inv, has_ua, wv, psio)
    end

    if debug
        @info "Δ' BVP: nMat=$nMat, rank(M)=$(rank(M)), cond(M)=$(@sprintf("%.2e", cond(M)))"
    end

    # rpec coil-response block: needs the S-axis row layout that `_solve_bvp_edge_coil` assumes
    # for its edge rows, and a vacuum edge in the assembled matrix (col_edge in junc_rows).
    if use_S_axis && wv !== nothing
        intr.delta_coil = _solve_bvp_edge_coil(M, col_edge, msing, N, ipert_all)
    end

    deltap, dp_raw_persisted = _solve_bvp_and_combine_pest3(
        M, msing, N, nMat, use_S_axis, ipert_all, col_edge, ctrl, debug)

    # Persist both the PEST3 tearing projection (msing × msing) and the raw 2msing × 2msing
    # D' matrix (side-major ordering, byte-compatible with Fortran rdcon/gal.f::gal_write_delta).
    # The raw matrix is consumed by `pest3_decompose` to recover (A', B', Γ', Δ') for the full
    # det(D' − D(γ)) = 0 eigenvalue problem; see the `delta_prime_raw` docstring in CoreTypes.jl.
    intr.delta_prime_matrix = deltap
    intr.delta_prime_raw    = dp_raw_persisted
end

# Column index helpers for the BVP matrix. j is the 1-based singular-surface index,
# N is numpert_total. Layout: c_axis(N), c_left[1](2N), c_right[1](2N), ..., c_edge(N).
_col_left(j::Int, N::Int)  = (N + 4N*(j-1) + 1):(N + 4N*(j-1) + 2N)
_col_right(j::Int, N::Int) = (N + 4N*(j-1) + 2N + 1):(N + 4N*j)

# Multi-resonance surfaces (one q value satisfying multiple (m,n) tuples in a multi-n run)
# are not yet handled by the inter-surface BVP. Returns true if any surface has >1 modes;
# emits a warning as a side effect. The stub per-surface delta_prime is unaffected.
function _has_unsupported_multi_resonance(intr::ForceFreeStatesInternal)
    msing = intr.msing
    n_res_per_surface = [length(intr.sing[j].m) for j in 1:msing]
    any(>(1), n_res_per_surface) || return false
    offenders = [(j, intr.sing[j].m, intr.sing[j].n) for j in 1:msing if n_res_per_surface[j] > 1]
    @warn "compute_delta_prime_matrix!: skipping inter-surface Δ' BVP because some surfaces carry more than one resonant mode " *
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
        @debug "compute_delta_prime_matrix!: $msing singular surfaces, $msing_active crossed (excluded: m=$excluded_ms)"
    end
    sing = [intr.sing[si] for si in sing_indices]
    return sing, i_crossings, msing_active
end

# Assemble all segment propagators: per-surface single-chunk FMs (Phi_L), inter-surface
# and edge multi-chunk FMs (Phi_R), and midpoint-split halves (Phi_R_halves) used by the
# diagnostic comparisons. Phi_R[1] is only built when use_S_axis=false (FM-axis fallback).
# Midpoint splitting halves each inter-surface span's condition number — STRIDE's trick:
# cond(full) = 10¹⁵ → cond(half) ≈ 10⁷·⁵, an 8-digit accuracy gain.
function _assemble_segment_propagators(propagators::Vector{ChunkPropagator},
                                       chunks::Vector{IntegrationChunk},
                                       i_crossings::Vector{Int}, msing::Int, N::Int,
                                       use_S_axis::Bool)
    Phi_L_mats = [assemble_fm_matrix(propagators, i_crossings[j]:i_crossings[j]) for j in 1:msing]
    Phi_R_mats = Vector{Matrix{ComplexF64}}(undef, msing + 1)
    if !use_S_axis
        Phi_R_mats[1] = assemble_fm_matrix(propagators, 1:i_crossings[1]-1; condition=true)
    end
    for j in 2:msing
        Phi_R_mats[j] = assemble_fm_matrix(propagators, i_crossings[j-1]+1:i_crossings[j]-1)
    end
    Phi_R_mats[msing+1] = assemble_fm_matrix(propagators, i_crossings[msing]+1:length(chunks))

    Phi_R_halves = Vector{Tuple{Matrix{ComplexF64},Matrix{ComplexF64}}}(undef, msing - 1)
    for j in 1:msing-1
        chunk_start = i_crossings[j] + 1
        chunk_end   = i_crossings[j+1] - 1
        n_chunks    = chunk_end - chunk_start + 1
        if n_chunks >= 2
            i_mid = chunk_start + div(n_chunks, 2) - 1
            Phi_left_half  = assemble_fm_matrix(propagators, chunk_start:i_mid)
            Phi_right_half = assemble_fm_matrix(propagators, i_mid+1:chunk_end)
            Phi_R_halves[j] = (Phi_left_half, Phi_right_half)
        else
            Phi_R_halves[j] = (Matrix{ComplexF64}(I, 2N, 2N), Phi_R_mats[j+1])
        end
    end
    return Phi_L_mats, Phi_R_mats, Phi_R_halves
end

# Asymptotic-basis transformation T = [ua[:,:,1]; ua[:,:,2]] maps (small/big) coefficients
# to raw (ξ,η) state. Column ordering of ua: 1:N = big solutions (z^{-α}, diverging),
# N+1:2N = small solutions (z^{+α}, bounded). Fortran STRIDE bakes T into the shooting
# propagators (uFM_sing_init); we multiply T into the BVP propagator blocks at each surface.
function _build_asymptotic_basis_matrices(sing::Vector{SingType}, has_ua::Bool, N::Int, msing::Int)
    T_left_mats  = Vector{Matrix{ComplexF64}}(undef, msing)
    T_right_mats = Vector{Matrix{ComplexF64}}(undef, msing)
    T_left_inv   = Vector{Matrix{ComplexF64}}(undef, msing)
    T_right_inv  = Vector{Matrix{ComplexF64}}(undef, msing)
    if has_ua
        for j in 1:msing
            sp = sing[j]
            T_left_mats[j]  = [sp.ua_left[:,:,1]; sp.ua_left[:,:,2]]
            T_right_mats[j] = [sp.ua_right[:,:,1]; sp.ua_right[:,:,2]]
            T_left_inv[j]   = inv(T_left_mats[j])
            T_right_inv[j]  = inv(T_right_mats[j])
        end
    end
    return T_left_mats, T_right_mats, T_left_inv, T_right_inv
end

# Build the S-axis shooting propagators uShootR (forward from surface j right → midpoint)
# and uShootL (backward from surface j left → midpoint), and the conditioned axis
# propagator uAxis. uShootL[1] is built specially using the QR-conditioned axis path
# (Fortran ode_fixup) so that surface 1 inherits the well-conditioned S axis BC instead
# of going through a catastrophically ill-conditioned full axis FM.
function _build_S_axis_shooting_propagators(
    propagators::Vector{ChunkPropagator}, chunks::Vector{IntegrationChunk},
    i_crossings::Vector{Int}, sing::Vector{SingType}, msing::Int, N::Int,
    T_left_mats::Vector{Matrix{ComplexF64}}, T_right_mats::Vector{Matrix{ComplexF64}},
    has_ua::Bool, ctrl, equil, mats, intr::ForceFreeStatesInternal, debug::Bool)

    can_reintegrate = has_ua && ctrl !== nothing && equil !== nothing && mats !== nothing
    uShootR = Vector{Matrix{ComplexF64}}(undef, msing)
    uShootL = Vector{Matrix{ComplexF64}}(undef, msing)   # uShootL[1] handled separately below

    for j in 1:msing
        shoot_range_R = _midpoint_shoot_range(chunks, i_crossings, j, msing; side=:right)
        if debug && !isempty(shoot_range_R)
            psi_surf_R = chunks[first(shoot_range_R)].psi_start
            psi_mid_R = chunks[last(shoot_range_R)].psi_end
            psi_ua_R = sing[j].psi_ua_right
            @info "    uShootR[$j]: shoot_range=$(shoot_range_R), psi_chunk=$(@sprintf("%.6f", psi_surf_R)), psi_ua=$(@sprintf("%.6f", psi_ua_R)), psi_mid=$(@sprintf("%.6f", psi_mid_R)), Δψ_fix=$(@sprintf("%.6e", psi_ua_R - psi_surf_R))"
        end
        if can_reintegrate && !isempty(shoot_range_R)
            uShootR[j] = integrate_fm_with_ua_ic(chunks, shoot_range_R, sing[j].ua_right,
                            ctrl, equil, mats, intr; backward=false, psi_ua=sing[j].psi_ua_right)
        else
            T_init = has_ua ? T_right_mats[j] : nothing
            uShootR[j] = assemble_fm_matrix(propagators, shoot_range_R; T_init=T_init)
        end

        # uShootL[j>=2]: backward from surface j left to midpoint. uShootL[1] handled below.
        j == 1 && continue
        shoot_range_L = _midpoint_shoot_range(chunks, i_crossings, j, msing; side=:left)
        if debug
            psi_mid = chunks[first(shoot_range_L)].psi_start
            psi_surf = chunks[last(shoot_range_L)].psi_end
            psi_ua_L = sing[j].psi_ua_left
            @info "    uShootL[$j]: shoot_range=$(shoot_range_L), psi_mid=$(@sprintf("%.6f", psi_mid)), psi_chunk=$(@sprintf("%.6f", psi_surf)), psi_ua=$(@sprintf("%.6f", psi_ua_L)), Δψ_fix=$(@sprintf("%.6e", psi_ua_L - psi_surf))"
        end
        if can_reintegrate && !isempty(shoot_range_L)
            uShootL[j] = integrate_fm_with_ua_ic(chunks, shoot_range_L, sing[j].ua_left,
                            ctrl, equil, mats, intr; backward=true, psi_ua=sing[j].psi_ua_left)
        else
            T_init = has_ua ? T_left_mats[j] : nothing
            uShootL[j] = assemble_fm_matrix(propagators, shoot_range_L; T_init=T_init)
        end
    end

    uAxis, i_axis_mid = _build_conditioned_axis_propagator(propagators, i_crossings, N)
    uShootL[1] = _build_uShootL_first(propagators, chunks, i_crossings, sing,
                                      T_left_mats, has_ua, can_reintegrate, i_axis_mid,
                                      ctrl, equil, mats, intr, N)
    if debug
        shoot_range_L1 = (i_axis_mid + 1):(i_crossings[1] - 1)
        @info "  Axis propagator: $(i_axis_mid) chunks, cond=$(@sprintf("%.2e", cond(uAxis)))"
        @info "  uShootL[1]: range=$(shoot_range_L1), cond=$(@sprintf("%.2e", cond(uShootL[1])))"
    end
    return uShootR, uShootL, uAxis
end

# Locate the chunk midpoint between two singular surfaces (or surface↔edge) in ψ space.
# Side `:right` returns the range from chunk(i_crossings[j]+1) to the ψ-midpoint chunk
# (or to the last chunk for j==msing). Side `:left` returns the range from the midpoint
# chunk+1 to chunk(i_crossings[j]-1). The ψ midpoint is used (not the chunk-index midpoint)
# because chunks near singularities are packed tighter in ψ — Fortran convention.
function _midpoint_shoot_range(chunks::Vector{IntegrationChunk}, i_crossings::Vector{Int},
                               j::Int, msing::Int; side::Symbol)
    if side === :right
        j == msing && return (i_crossings[msing] + 1):length(chunks)
        chunk_start = i_crossings[j] + 1
        chunk_end   = i_crossings[j+1] - 1
    else  # :left, j >= 2
        chunk_start = i_crossings[j-1] + 1
        chunk_end   = i_crossings[j] - 1
    end
    psi_mid_target = (chunks[chunk_start].psi_start + chunks[chunk_end].psi_end) / 2
    i_mid_inter = chunk_start
    for ic in chunk_start:chunk_end-1
        if chunks[ic].psi_end >= psi_mid_target
            i_mid_inter = ic
            break
        end
        i_mid_inter = ic
    end
    return side === :right ? (chunk_start:i_mid_inter) : ((i_mid_inter + 1):chunk_end)
end

# Build a well-conditioned axis propagator by forward-propagating [0; I] through the
# pre-first-crossing chunks with QR fixup after each chunk (Fortran ode_fixup). The axis
# midpoint is placed one chunk before the first surface so that uShootL[1] covers only the
# last chunk, keeping it well-conditioned.
function _build_conditioned_axis_propagator(propagators::Vector{ChunkPropagator},
                                            i_crossings::Vector{Int}, N::Int)
    n_pre_cross = i_crossings[1] - 1
    i_axis_mid = max(1, n_pre_cross - 1)
    uAxis = zeros(ComplexF64, 2N, N)
    for i in 1:N
        uAxis[N+i, i] = 1
    end
    for ic in 1:i_axis_mid
        prop = propagators[ic]
        upper_old = uAxis[1:N, :]
        lower_old = uAxis[N+1:2N, :]
        uAxis[1:N, :]    .= prop.block_upper_ic[:,:,1] * upper_old .+ prop.block_lower_ic[:,:,1] * lower_old
        uAxis[N+1:2N, :] .= prop.block_upper_ic[:,:,2] * upper_old .+ prop.block_lower_ic[:,:,2] * lower_old
        Q, _ = qr(uAxis)
        uAxis .= Matrix(Q)[:, 1:N]
    end
    for j in 1:N
        uAxis[:, j] ./= norm(@view uAxis[:, j])
    end
    return uAxis, i_axis_mid
end

# Build uShootL[1]: backward propagator from surface 1 left boundary to the axis midpoint.
# Falls back to T_left_mats[1] (or identity if no ua) when there's only 1 chunk before the
# first crossing.
function _build_uShootL_first(propagators::Vector{ChunkPropagator},
                              chunks::Vector{IntegrationChunk}, i_crossings::Vector{Int},
                              sing::Vector{SingType}, T_left_mats::Vector{Matrix{ComplexF64}},
                              has_ua::Bool, can_reintegrate::Bool, i_axis_mid::Int,
                              ctrl, equil, mats, intr::ForceFreeStatesInternal, N::Int)
    shoot_range_L1 = (i_axis_mid + 1):(i_crossings[1] - 1)
    if can_reintegrate && !isempty(shoot_range_L1)
        return integrate_fm_with_ua_ic(chunks, shoot_range_L1, sing[1].ua_left,
                                       ctrl, equil, mats, intr;
                                       backward=true, psi_ua=sing[1].psi_ua_left)
    elseif !isempty(shoot_range_L1)
        return assemble_fm_matrix(propagators, shoot_range_L1;
                                  T_init=has_ua ? T_left_mats[1] : nothing)
    else
        return has_ua ? T_left_mats[1] : Matrix{ComplexF64}(I, 2N, 2N)
    end
end

# Assemble the BVP matrix M with S-based axis BC. The Riccati S matrix at surface 1's left
# boundary encodes the axis BC (U₁ = S·U₂) in a well-conditioned form (cond ~ 10⁶), avoiding
# the catastrophically ill-conditioned axis FM. Fortran-matched structure with
# nMat = (2 + 4·msing)·N. Returns (M, nMat, col_edge).
function _assemble_bvp_S_axis(uShootR::Vector{Matrix{ComplexF64}},
                              uShootL::Vector{Matrix{ComplexF64}},
                              uAxis::Matrix{ComplexF64}, ipert_all::Vector{Int},
                              msing::Int, N::Int,
                              wv::Union{Nothing,Matrix{ComplexF64}}, psio::Float64)
    # STRIDE global BVP block structure [Glasser-Kolemen 2018 PoP 25, 032501 Eq. 37].
    nMat = (2 + 4 * msing) * N
    col_axis = 1:N
    col_edge = (nMat - N + 1):nMat
    M = zeros(ComplexF64, nMat, nMat)

    # Axis matching: uShootL[1] · c_left[1] = uAxis · c_axis  (2N equations)
    M[1:2N, _col_left(1, N)] .= uShootL[1]
    M[1:2N, col_axis]        .= -uAxis
    row_offset = 2N

    for j in 1:msing
        ipert_j = ipert_all[j]
        # Crossing: non-resonant modes continuity (asymptotic basis = identity)
        for i in 1:2N
            if i != ipert_j && i != ipert_j + N
                row_offset += 1
                M[row_offset, _col_left(j, N)[i]]  =  1
                M[row_offset, _col_right(j, N)[i]] = -1
            end
        end

        junc_rows = (row_offset + 1):(row_offset + 2N)
        if j < msing
            # Midpoint matching between consecutive surfaces
            M[junc_rows, _col_right(j, N)]   .= -uShootR[j]
            M[junc_rows, _col_left(j+1, N)]  .=  uShootL[j+1]
        else
            # Edge junction
            M[junc_rows, _col_right(msing, N)] .= uShootR[msing]
            if wv !== nothing
                M[junc_rows[1:N],     col_edge] .= -I(N)
                M[junc_rows[N+1:end], col_edge] .= wv .* psio^2
            else
                M[junc_rows[N+1:end], col_edge] .= -I(N)
            end
        end
        row_offset = last(junc_rows)
    end

    # Driving rows: set big-solution coefficient = 1 at each surface (asymptotic basis)
    for j in 1:msing
        ipert_j = ipert_all[j]
        row_offset += 1
        M[row_offset, _col_left(j, N)[ipert_j]]  = 1
        row_offset += 1
        M[row_offset, _col_right(j, N)[ipert_j]] = 1
    end
    @assert row_offset == nMat "Row count mismatch: expected $nMat, got $row_offset"
    return M, nMat, col_edge
end

# Coil-response block for the Eq. (37) edge [Glasser-Kolemen 2018 PoP 25, 032501]: impose the rpec edge
# boundary condition — identity edge plus a unit source per poloidal mode, matching RDCON's
# `gal_set_boundary` rpec branch — then read the small-solution (+N slot) coefficient at every surface.
# The BC is the same for every edge mode, so the matrix is factorized once and all N modes are solved
# together as columns of one right-hand side. Returns delta_coil (2·msing × N), rows = surface side.
function _solve_bvp_edge_coil(M::Matrix{ComplexF64}, col_edge, msing::Int, N::Int, ipert_all::Vector{Int})
    nMat = size(M, 1)
    bot = (nMat-2msing-N+1):(nMat-2msing)   # Eq. (38) bottom rows, carrying the W_V block
    Mc = copy(M)
    Mc[bot, :] .= 0
    Mc[bot, col_edge] .= I(N)               # Dirichlet edge: the edge coefficients equal the source
    B = zeros(ComplexF64, nMat, N)
    B[bot, :] .= I(N)                       # unit drive, one column per edge poloidal mode
    X = lu(Mc) \ B
    delta_coil = zeros(ComplexF64, 2msing, N)
    for j in 1:msing
        row_left = _col_left(j, N)[ipert_all[j]+N]
        row_right = _col_right(j, N)[ipert_all[j]+N]
        @views delta_coil[2j-1, :] .= X[row_left, :]
        @views delta_coil[2j, :] .= X[row_right, :]
    end
    return delta_coil
end

# Fallback BVP assembly with FM-based axis BC (used when no Riccati S matrices are available).
# Uses the conditioned axis propagator Phi_R[1][:,N+1:2N] in place of S-axis matching.
function _assemble_bvp_FM_axis(Phi_L_mats::Vector{Matrix{ComplexF64}},
                               Phi_R_mats::Vector{Matrix{ComplexF64}}, ipert_all::Vector{Int},
                               msing::Int, N::Int,
                               T_left_inv::Vector{Matrix{ComplexF64}},
                               T_right_inv::Vector{Matrix{ComplexF64}}, has_ua::Bool,
                               wv::Union{Nothing,Matrix{ComplexF64}}, psio::Float64)
    nMat = (2 + 4 * msing) * N
    col_axis = 1:N
    col_edge = (N + 4N*msing + 1):nMat
    M = zeros(ComplexF64, nMat, nMat)

    M[1:2N, (N+1):(N+2N)] .= Phi_L_mats[1]
    M[1:2N, col_axis]     .= -view(Phi_R_mats[1], :, N+1:2N)

    row_drive_base = 2N + (4N-2)*msing
    for j in 1:msing
        ipert_j = ipert_all[j]
        cl = _col_left(j, N)
        cr = _col_right(j, N)
        row_cont = 2N + (4N-2)*(j-1)
        for i in 1:2N
            if i != ipert_j && i != ipert_j + N
                row_cont += 1
                M[row_cont, cl[i]] =  1
                M[row_cont, cr[i]] = -1
            end
        end
        junc_rows = (row_cont + 1):(2N + (4N-2)*j)
        if j < msing
            M[junc_rows, cr]                .=  Phi_R_mats[j+1]
            M[junc_rows, _col_left(j+1, N)] .= -Phi_L_mats[j+1]
        else
            M[junc_rows, cr] .= Phi_R_mats[msing+1]
            if wv !== nothing
                M[junc_rows[1:N],     col_edge] .= -I(N)
                M[junc_rows[N+1:end], col_edge] .= wv .* psio^2
            else
                M[junc_rows[N+1:end], col_edge] .= -I(N)
            end
        end
        if has_ua
            M[row_drive_base + 2j-1, cl] .= T_left_inv[j][ipert_j, :]
            M[row_drive_base + 2j,   cr] .= T_right_inv[j][ipert_j, :]
        else
            M[row_drive_base + 2j-1, cl[ipert_j]] = 1
            M[row_drive_base + 2j,   cr[ipert_j]] = 1
        end
    end
    return M, nMat, col_edge
end

# Solve the BVP for each driving configuration and apply the PEST3 four-term combination.
# Promotes to Complex{Double64} if ctrl.extended_precision_bvp (default true) — the PEST3
# combination subtracts dp_raw entries up to ~3×10⁴ larger than the result, and Float64
# precision lets the imaginary part drift 2–5× on DIIID-class equilibria.
function _solve_bvp_and_combine_pest3(M::Matrix{ComplexF64}, msing::Int, N::Int, nMat::Int,
                                      use_S_axis::Bool, ipert_all::Vector{Int}, col_edge,
                                      ctrl, debug::Bool)
    s2 = 2 * msing
    Tc = (ctrl === nothing || ctrl.extended_precision_bvp) ? Complex{Double64} : ComplexF64
    M_solve = Tc.(M)

    M_lu = lu(M_solve; check=false)
    use_lu = issuccess(M_lu)
    M_pinv = use_lu ? nothing : pinv(M_solve)
    if !use_lu
        @warn "Δ' BVP: LU factorization singular (rank $(rank(M))/$nMat), using pseudo-inverse fallback"
    end

    dp_raw = zeros(Tc, s2, s2)
    b = zeros(Tc, nMat)
    for jsing in 1:msing, side in 1:2
        dRow = 2jsing - (2 - side)
        fill!(b, 0)
        drive_row = use_S_axis ? (nMat - s2 + dRow) : (2N + (4N-2)*msing + dRow)
        b[drive_row] = 1
        x = use_lu ? (M_lu \ b) : (M_pinv * b)

        debug && _log_bvp_solve(x, b, M_solve, jsing, side, dRow, msing, N,
                                ipert_all, col_edge, use_S_axis)

        for ksing in 1:msing
            ipert_k = ipert_all[ksing]
            dp_raw[dRow, 2ksing-1] = x[_col_left(ksing, N)[ipert_k+N]]
            dp_raw[dRow, 2ksing]   = x[_col_right(ksing, N)[ipert_k+N]]
        end
    end

    # PEST3 four-term combination [Chance PPPL-2527; Glasser-Kolemen 2018 PoP 25, 032501 Eq. 31].
    # Δ'[i,j] = (NW − NE − SW + SE) on each 2×2 block of dp_raw, in extended precision.
    deltap_ext = zeros(Tc, msing, msing)
    for i in 1:msing, j in 1:msing
        deltap_ext[i, j] = dp_raw[2i, 2j] - dp_raw[2i, 2j-1] - dp_raw[2i-1, 2j] + dp_raw[2i-1, 2j-1]
    end
    deltap = ComplexF64.(deltap_ext)

    debug && _log_bvp_pest3(dp_raw, deltap, s2, msing, Tc)
    # Return the PEST3-combined matrix AND the raw 2msing×2msing D' matrix (ComplexF64
    # for compatibility with downstream pest3_decompose / HDF5 writer).
    return deltap, ComplexF64.(dp_raw)
end

# Logging helpers for `compute_delta_prime_matrix!`. Called only when debug=true.
function _log_bvp_setup(chunks, sing, S_at_surface_left, use_S_axis, has_ua,
                        Phi_L_mats, Phi_R_mats, Phi_R_halves, ipert_all, wv, psio, N, msing)
    @info "Δ' BVP: $(length(chunks)) chunks, $msing surfaces, N=$N"
    @info "Δ' BVP: Axis BC: $(use_S_axis ? "S-based (Riccati)" : "FM-based (conditioned)")"
    @info "Δ' BVP: Asymptotic basis: $(has_ua ? "available" : "NOT available (raw basis driving)")"
    if use_S_axis
        for j in 1:msing
            @info "  S_left[$j]: max=$(@sprintf("%.2e", maximum(abs, S_at_surface_left[j]))), cond=$(@sprintf("%.2e", cond(S_at_surface_left[j])))"
        end
    end
    if has_ua
        for j in 1:msing
            sp = sing[j]
            T_l = [sp.ua_left[:,:,1]; sp.ua_left[:,:,2]]
            T_r = [sp.ua_right[:,:,1]; sp.ua_right[:,:,2]]
            @info "  Surface $j: cond(T_left)=$(@sprintf("%.2e", cond(T_l))), cond(T_right)=$(@sprintf("%.2e", cond(T_r)))"
            ipert_j = ipert_all[j]
            @info "  Surface $j ua_left (ipert=$ipert_j, psi_ua_left=$(@sprintf("%.8f", sp.psi_ua_left))):"
            for i in 1:min(5, N)
                @info "    ua($i,$ipert_j,1)=$(@sprintf("%16.8e %16.8e", real(sp.ua_left[i,ipert_j,1]), imag(sp.ua_left[i,ipert_j,1])))  ua($i,$ipert_j,2)=$(@sprintf("%16.8e %16.8e", real(sp.ua_left[i,ipert_j,2]), imag(sp.ua_left[i,ipert_j,2])))"
            end
            @info "    small: ua(1,$(ipert_j+N),1)=$(@sprintf("%16.8e %16.8e", real(sp.ua_left[1,ipert_j+N,1]), imag(sp.ua_left[1,ipert_j+N,1])))"
        end
    end
    for j in 1:msing-1
        Phi_L_h, Phi_R_h = Phi_R_halves[j]
        @info "  Inter-surface $j→$(j+1): half_L cond=$(@sprintf("%.2e",cond(Phi_L_h))), half_R cond=$(@sprintf("%.2e",cond(Phi_R_h))), full cond=$(@sprintf("%.2e",cond(Phi_R_mats[j+1])))"
    end
    @info "  Phi_R[$(msing+1)] (edge): cond=$(@sprintf("%.2e",cond(Phi_R_mats[msing+1])))"
    for j in 1:msing
        @info "  Surface $j (m=$(sing[j].m[1])): ipert=$(ipert_all[j]), cond(Phi_L)=$(@sprintf("%.2e", cond(Phi_L_mats[j])))"
    end
    @info "Δ' BVP: Vacuum BC $(wv === nothing ? "off (conducting wall)" : "on (psio=$psio)")"
    for j in 1:msing
        if !isempty(sing[j].delta_prime)
            @info "  Surface $j ca-based Δ' = $(@sprintf("%.6f%+.6fi", real(sing[j].delta_prime[1]), imag(sing[j].delta_prime[1])))"
        end
    end
end

function _log_S_axis_shooting_propagators(uShootR, uShootL, uAxis, S_at_surface_left,
                                          T_left_mats, ipert_all, has_ua, msing, N)
    @info "  Shooting propagators (S-based axis BC, no axis unknowns):"
    for j in 1:msing
        shoot_R_str = @sprintf("%.2e", cond(uShootR[j]))
        shoot_L_str = j >= 2 ? @sprintf("%.2e", cond(uShootL[j])) : "N/A (S axis BC)"
        @info "    uShootL[$j]: cond=$shoot_L_str, uShootR[$j]: cond=$shoot_R_str"
    end
    S1 = S_at_surface_left[1]
    if has_ua
        T1 = T_left_mats[1]
        axis_BC = T1[1:N, :] - S1 * T1[N+1:2N, :]
        @info "    S-axis BC matrix: cond=$(@sprintf("%.2e", cond(axis_BC)))"
    end
    for j in 1:msing
        ipert_j = ipert_all[j]
        col_norms_R = [norm(view(uShootR[j], :, k)) for k in 1:2N]
        @info "    uShootR[$j] column norms: min=$(@sprintf("%.2e", minimum(col_norms_R))), max=$(@sprintf("%.2e", maximum(col_norms_R)))"
        @info "    uShootR[$j] col ipert=$ipert_j norm=$(@sprintf("%.2e", col_norms_R[ipert_j])), col ipert+N=$(ipert_j+N) norm=$(@sprintf("%.2e", col_norms_R[ipert_j+N]))"
        if j >= 2
            col_norms_L = [norm(view(uShootL[j], :, k)) for k in 1:2N]
            @info "    uShootL[$j] column norms: min=$(@sprintf("%.2e", minimum(col_norms_L))), max=$(@sprintf("%.2e", maximum(col_norms_L)))"
            @info "    uShootL[$j] col ipert=$ipert_j norm=$(@sprintf("%.2e", col_norms_L[ipert_j])), col ipert+N=$(ipert_j+N) norm=$(@sprintf("%.2e", col_norms_L[ipert_j+N]))"
        end
    end
    for j in 1:msing-1
        mid_block = hcat(uShootR[j], -uShootL[j+1])
        @info "    Midpoint $j→$(j+1): cond([uShootR[$j] | -uShootL[$(j+1)]]) = $(@sprintf("%.2e", cond(mid_block)))"
        col_norms_Ljp1 = [norm(view(uShootL[j+1], :, k)) for k in 1:2N]
        @info "    uShootL[$(j+1)] all col norms: $([(@sprintf("%.2e", c)) for c in col_norms_Ljp1])"
    end
end

function _log_bvp_solve(x, b, M_solve, jsing, side, dRow, msing, N,
                        ipert_all, col_edge, use_S_axis)
    residual = norm(ComplexF64.(M_solve * x - b))
    side_str = side == 1 ? "left" : "right"
    @info "  BVP solve: jsing=$jsing side=$side_str (dRow=$dRow): ||Mx-b||=$(@sprintf("%.2e", residual)), ||x||=$(@sprintf("%.2e", Float64(norm(x))))"
    for ks in 1:msing
        ipert_ks = ipert_all[ks]
        cl = _col_left(ks, N)
        cr = _col_right(ks, N)
        xl_big   = ComplexF64(x[cl[ipert_ks]])
        xl_small = ComplexF64(x[cl[ipert_ks+N]])
        xr_big   = ComplexF64(x[cr[ipert_ks]])
        xr_small = ComplexF64(x[cr[ipert_ks+N]])
        @info "    surf $ks: x_left[big]=$(@sprintf("%+.4e%+.4ei", real(xl_big), imag(xl_big))), x_left[small]=$(@sprintf("%+.4e%+.4ei", real(xl_small), imag(xl_small)))"
        @info "    surf $ks: x_right[big]=$(@sprintf("%+.4e%+.4ei", real(xr_big), imag(xr_big))), x_right[small]=$(@sprintf("%+.4e%+.4ei", real(xr_small), imag(xr_small)))"
        @info "    surf $ks: ||x_left||=$(@sprintf("%.2e", Float64(norm(x[cl])))), ||x_right||=$(@sprintf("%.2e", Float64(norm(x[cr]))))"
    end
    if use_S_axis
        @info "    ||x_edge||=$(@sprintf("%.2e", Float64(norm(x[col_edge]))))"
    end
end

function _log_bvp_pest3(dp_raw, deltap, s2, msing, Tc)
    @info "Δ' BVP: Full dp_raw matrix ($(s2)×$(s2)) [$(Tc)]:"
    for i in 1:s2
        row_str = join([@sprintf("%+.6e", Float64(real(dp_raw[i,j]))) for j in 1:s2], "  ")
        @info "  dp_raw[$i,:] = $row_str"
    end
    @info "Δ' BVP: Raw dp diagonal = $([@sprintf("%.4f%+.4fi", Float64(real(dp_raw[i,i])), Float64(imag(dp_raw[i,i]))) for i in 1:s2])"
    @info "Δ' BVP: deltap diagonal = $([@sprintf("%.4f%+.4fi", real(deltap[i,i]), imag(deltap[i,i])) for i in 1:msing])"
end

"""
    pest3_decompose(dp_raw::AbstractMatrix) -> (A', B', Γ', Δ')

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

Matches Fortran exactly — no ½ prefactor (Pletzer–Dewar multiply by ½, but
the Fortran RDCON code leaves it commented out and our Julia port follows
Fortran to keep the benchmark bit-identical; the prefactor cancels in
`det(D' − D(γ)) = 0`).

The Δ' block returned here equals `intr.delta_prime_matrix` (the m×m PEST3
tearing projection computed inside `compute_delta_prime_matrix!`).

# Arguments

  - `dp_raw` — 2m×2m complex matrix (typically `intr.delta_prime_raw`).

# Returns

Named tuple `(A=A', B=B', Γ=Gp, Δ=Dp)` of four m×m complex matrices. In the
full `det(D' − D(γ)) = 0` eigenvalue problem, these fill the 2m×2m outer
matrix as `D' = [[A' B'] [Γ' Δ']]` with the interchange channel (Glasser
stabilization) in the upper-left block and the tearing channel in the
lower-right.
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
        RL = dp_raw[2i,   2j-1]
        RR = dp_raw[2i,   2j]
        Ap[i, j] = RR + RL + LR + LL
        Bp[i, j] = RR - RL + LR - LL
        Gp[i, j] = RR + RL - LR - LL
        Dp[i, j] = RR - RL - LR + LL
    end
    return (A=Ap, B=Bp, Γ=Gp, Δ=Dp)
end
