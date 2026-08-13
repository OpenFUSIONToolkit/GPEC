"""
    Riccati.jl - Dual Riccati reformulation of the Euler-Lagrange ODE

Implements the dual Riccati matrix S = U₁ · U₂⁻¹ = P⁻¹, which satisfies a bounded
ODE even near singular surfaces where U₁, U₂ grow exponentially. This reduced stiffness
leads to fewer ODE integration steps and faster wall-clock time.

Reference: Glasser (2018) Phys. Plasmas 25, 032507 — Eq. 19 (adapted for dual form S = P⁻¹)
where P = U₂ · U₁⁻¹ is the forward plasma response matrix.

## Dual Riccati ODE

Starting from the Euler-Lagrange system [Glasser 2016 eq. 24]:
  dU₁/dψ = A·U₁ + B·U₂        A = -Q·F̄⁻¹·K̄,  B = Q·F̄⁻¹·Q
  dU₂/dψ = C·U₁ + D·U₂        C = Ḡ - K̄†·F̄⁻¹·K̄,  D = K̄†·F̄⁻¹·Q

with S = U₁·U₂⁻¹, differentiating gives the Riccati ODE:
  dS/dψ = B + A·S - S·D - S·C·S

Setting w = Q - K̄·S (shape N×N) and v = F̄⁻¹·w (Cholesky solve), this simplifies to:
  dS/dψ = w†·v - S·Ḡ·S     [Glasser 2018 eq. 19, dual form]

## Integration Strategy

### Why not integrate the Riccati ODE directly?

`riccati_der!` evaluates the explicit Riccati RHS `dS/dψ = w†F̄⁻¹w − S·Ḡ·S` correctly,
but this ODE is **quadratic** in S. Near a rational surface, S grows large, so the quadratic
term `-SGS` dominates and the RHS grows as |S|². Explicit adaptive solvers (Vern9) use
*relative* error control: they accept a step when |Δu|/|u| < reltol. When |S| is large,
the absolute error |ΔS| can be enormous while the relative error stays within tolerance.
The solver takes large steps through what is effectively a near-blowup — no amount of
step-size adaptation saves it because the problem is the error *metric*, not the step size.
An implicit solver could handle this stiffness, but is deferred.

### Actual implementation: EL ODE + renormalization

Instead we integrate the standard EL ODE (`sing_der!`) in the (U₁, U₂) variables and
recover S = U₁·U₂⁻¹ by renormalization. This achieves the same Riccati trajectory with
**no accuracy loss**:

- `sing_der!` evaluates the exact EL RHS — no approximation.
- Vern9 integrates (U₁, U₂) to **9th-order accuracy** with the adaptive step-size
  controller enforcing the configured reltol at every accepted step.
- Renormalization `S = U₁·U₂⁻¹` is **exact** (a change of variables, not an approximation).
- The global error is the same as the standard EL path — controlled by the ODE solver
  reltol, not by the renormalization frequency.

This works because the EL ODE is **linear** in (U₁, U₂): the RHS does not grow with |S|,
so relative error control is faithful even when S is large. Renormalization triggered by
`renormalize_riccati_inplace!` in the callback (when max(|U₁|) or max(|U₂|) > ucrit) keeps
both matrices bounded, preventing overflow and maintaining a well-conditioned state for the
solver — exactly analogous to Gaussian reduction in the standard ODE.

### Consistency with the Riccati ODE (local analysis)

To verify the method is consistent with the Riccati ODE, consider a single step from (S, I):

  After one step: U₁_new = S + (A·S + B)·Δψ + O(Δψ²),  U₂_new = I + (C·S + D)·Δψ + O(Δψ²)
  Renorm:         S_new = U₁_new · U₂_new⁻¹ = S + (B + A·S − S·D − S·C·S)·Δψ + O(Δψ²) ✓

The leading term matches the Riccati ODE exactly. This is a local consistency check only —
it does not imply the integration is first-order. In practice Vern9 captures all higher-order
terms through its internal stages, achieving 9th-order global accuracy at the configured reltol.

## Storage Convention

During chunk integration (with sing_der! as ODE RHS):
  u[:,:,1] = U₁  (starts as S_prev, evolves toward new S)
  u[:,:,2] = U₂  (starts as I, evolves with EL dynamics)

After renormalization (at crossing or when norms exceed ucrit):
  u[:,:,1] = S = U₁ · U₂⁻¹
  u[:,:,2] = I

This is compatible with downstream code (which uses U₁/U₂ ratio):
  - Free.jl:     wp = u[:,:,2] / u[:,:,1] = I · S⁻¹ = P  ✓  (post-renorm)
  - FixedBoundaryStability.jl: crit = min_eigval(u[:,:,1] / u[:,:,2]) = min_eigval(S)  ✓
  - Axis init:   determined by `ctrl.fixed_axis`. When `true`, U₁=0, U₂=I → S(ψ₀)=0 (original
    Glasser fixed-axis BC). When `false` (default), Frobenius eigenvalue init [Glasser 2016 Eq. 51]
    sets U₂=I and U₁ to the regular Frobenius eigenvector per mode → S(ψ₀) = U₁_Frobenius is
    nonzero in general. Riccati S-evolution remains well-defined either way.

## Key Differences from Standard Integration

1. `sing_der!` is used as the ODE RHS (same as standard, NOT `riccati_der!`)
2. `riccati_integrator_callback!` replaces `integrator_callback!`: uses
   `renormalize_riccati_inplace!` instead of Gaussian reduction
3. `riccati_cross_ideal_singular_surf!` replaces `cross_ideal_singular_surf!`: skips Gaussian
   reduction and uses ipert_res directly for column zeroing, then renormalizes to (S_new, I)
4. `transform_u!` is skipped — S is already the true solution
"""

# Save-frequency thresholds for `riccati_integrator_callback!`. Near the right endpoint of
# a segment we save every step so that the crossing / chunk boundary captures fine detail;
# elsewhere we save every `ctrl.save_interval`-th step. The relative band catches normal-
# length chunks; the absolute floor catches short chunks where 5% of the span would be
# smaller than the typical ODE step.
const SAVE_NEAR_END_FRAC = 0.05
const SAVE_NEAR_END_PSI  = 1e-4

"""
    assemble_fm_matrix(propagators, idx_range; condition=false) -> Matrix{ComplexF64}

Assemble the 2N×2N fundamental matrix (propagator) by multiplying chunk propagators
in order for indices `idx_range`. Returns Φ_end * ... * Φ_start, so that the result
maps the IC at the start of `idx_range[1]` to the state at the end of `idx_range[end]`.

Each `ChunkPropagator` stores the 2N columns of Φ split into two N×N×2 blocks:
```
  block_upper_ic[:,:,1:2] ↔ Φ[:,1:N]     (result from IC=(I,0))
  block_lower_ic[:,:,1:2] ↔ Φ[:,N+1:2N]  (result from IC=(0,I))
```

When `condition=true`, applies Gaussian reduction (`condition_propagator!`) after each
multiplication step, following STRIDE's `ode_fixup` convention. This
prevents exponential growth of the accumulated product: without conditioning, products
of K chunk propagators can reach cond ~ (cond_per_chunk)^K, causing catastrophic
cancellation. With periodic conditioning, each step stays at O(cond_per_chunk) and
only the N well-conditioned U₂ columns (right half) survive.

Use `condition=true` for the axis→first-surface segment, where the axis BC (U₁=0)
means only U₂ ICs are needed. Do NOT use for inter-surface segments where both U₁
and U₂ components carry physical information.
"""
function assemble_fm_matrix(propagators::Vector{ChunkPropagator}, idx_range;
                            condition::Bool=false,
                            T_init::Union{Nothing,Matrix{ComplexF64}}=nothing)
    # Determine matrix size from T_init if provided (lets us handle empty idx_range and even
    # an empty propagators list, provided T_init carries the dimension). Otherwise fall back
    # to the first propagator that actually exists in idx_range, with a final fallback to
    # propagators[1] when both idx_range and T_init pin nothing down.
    N = if T_init !== nothing
        size(T_init, 1) ÷ 2
    elseif !isempty(idx_range)
        size(propagators[first(idx_range)].block_upper_ic, 1)
    else
        @assert !isempty(propagators) "assemble_fm_matrix: cannot infer N from empty propagators with no T_init"
        size(propagators[1].block_upper_ic, 1)
    end
    Phi = T_init !== nothing ? copy(T_init) : Matrix{ComplexF64}(I, 2N, 2N)
    isempty(idx_range) && return Phi
    for i in idx_range
        p = propagators[i]
        #! format: off
        Phi_i = [p.block_upper_ic[:,:,1]  p.block_lower_ic[:,:,1];
                 p.block_upper_ic[:,:,2]  p.block_lower_ic[:,:,2]]
        #! format: on
        Phi = Phi_i * Phi
        if condition
            condition_propagator!(Phi, N)
        end
    end
    return Phi
end

"""
    condition_propagator!(Phi, N)

Apply Gaussian reduction to the U₂-columns (columns N+1:2N) of a 2N×2N propagator
matrix in-place, following STRIDE's `ode_fixup` convention. Triangularizes the U₁
(upper N rows) subblock by pivoted elimination, improving the condition number so
the propagator can be used in a BVP without losing numerical rank.

After conditioning, only the U₂ columns carry meaningful information; the U₁ columns
(1:N) are zeroed.  The BVP axis block uses `Phi[:, N+1:2N]` (the conditioned half).
"""
function condition_propagator!(Phi::Matrix{ComplexF64}, N::Int)
    # Work on the right half: columns N+1:2N (U₂ initial conditions)
    cols = view(Phi, :, N+1:2N)

    # Sort columns by norm of the U₁ (upper N) block — largest first
    norms = [norm(view(cols, 1:N, k)) for k in 1:N]
    order = sortperm(norms; rev=true)

    mask_col = trues(N)   # which columns remain to process
    mask_row = trues(N)   # which pivot rows remain available

    for isol in 1:N
        kcol = order[isol]
        mask_col[kcol] = false

        # Find best pivot row (largest |element| among unmasked rows)
        best_row = 0
        best_val = 0.0
        for r in 1:N
            if mask_row[r] && abs(cols[r, kcol]) > best_val
                best_val = abs(cols[r, kcol])
                best_row = r
            end
        end
        if best_row == 0 || best_val == 0
            continue
        end
        mask_row[best_row] = false

        # Eliminate this pivot from all other unmasked columns
        pivot = cols[best_row, kcol]
        for jcol in 1:N
            if mask_col[jcol]
                factor = -cols[best_row, jcol] / pivot
                @views cols[:, jcol] .+= factor .* cols[:, kcol]
                cols[best_row, jcol] = 0  # exact zero
            end
        end
    end

    # Zero the U₁ columns (left half) — they are no longer meaningful
    Phi[:, 1:N] .= 0
    return Phi
end

"""
    compute_delta_prime_matrix!(intr, propagators, chunks; wv, psio, debug, ctrl, equil, ffit)

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
    ffit::Union{Nothing,FourFitVars} = nothing
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
            T_left_mats, T_right_mats, has_ua, ctrl, equil, ffit, intr, debug)
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
    # det(D' − D(γ)) = 0 eigenvalue problem; see ForceFreeStatesStructs.jl docstring.
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
    has_ua::Bool, ctrl, equil, ffit, intr::ForceFreeStatesInternal, debug::Bool)

    can_reintegrate = has_ua && ctrl !== nothing && equil !== nothing && ffit !== nothing
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
                            ctrl, equil, ffit, intr; backward=false, psi_ua=sing[j].psi_ua_right)
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
                            ctrl, equil, ffit, intr; backward=true, psi_ua=sing[j].psi_ua_left)
        else
            T_init = has_ua ? T_left_mats[j] : nothing
            uShootL[j] = assemble_fm_matrix(propagators, shoot_range_L; T_init=T_init)
        end
    end

    uAxis, i_axis_mid = _build_conditioned_axis_propagator(propagators, i_crossings, N)
    uShootL[1] = _build_uShootL_first(propagators, chunks, i_crossings, sing,
                                      T_left_mats, has_ua, can_reintegrate, i_axis_mid,
                                      ctrl, equil, ffit, intr, N)
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
                              ctrl, equil, ffit, intr::ForceFreeStatesInternal, N::Int)
    shoot_range_L1 = (i_axis_mid + 1):(i_crossings[1] - 1)
    if can_reintegrate && !isempty(shoot_range_L1)
        return integrate_fm_with_ua_ic(chunks, shoot_range_L1, sing[1].ua_left,
                                       ctrl, equil, ffit, intr;
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

"""
    riccati_der!(du, u, params, psieval)

Evaluate the explicit dual Riccati ODE right-hand side:
  dS/dψ = w†·F̄⁻¹·w - S·Ḡ·S,   w = Q - K̄·S

where Q = diag(1/(m - n·q)) is the diagonal singular factor matrix.
The identity slice u[:,:,2] = I does not evolve (du[:,:,2] = 0).

**REFERENCE IMPLEMENTATION — not called in production.** The explicit Riccati ODE is
numerically unstable for explicit solvers: the quadratic S·Ḡ·S term blows up when K̄·S ≫ Q.
The production path integrates `sing_der!` with periodic `renormalize_riccati_inplace!`
instead (see module docstring). Kept here for documentation of Eq. 19 in source form and
for future use with implicit solvers; exercised only by unit tests that verify the formula.

See: Glasser (2018) Phys. Plasmas 25, 032507 — Eq. 19 (dual Riccati form)
"""
@with_pool pool function riccati_der!(
    du::Array{ComplexF64,3},
    u::Array{ComplexF64,3},
    params::Tuple{ForceFreeStatesControl,Equilibrium.PlasmaEquilibrium,
        FourFitVars,ForceFreeStatesInternal,OdeState,IntegrationChunk},
    psieval::Float64
)

    _, equil, ffit, intr, odet, _ = params

    Npert = intr.numpert_total
    S  = @view u[:, :, 1]
    dS = @view du[:, :, 1]
    @view(du[:, :, 2]) .= 0  # identity does not evolve

    # Compute singfac = 1/(m - n·q) as column vector Q = diag(singfac_vec)
    # [Glasser 2016 eq. 24]
    singfac_vec = acquire!(pool, Float64, Npert)
    singfac_mat = reshape(singfac_vec, intr.mpert, intr.npert)
    odet.q = equil.profiles.q_spline(psieval; hint=odet.spline_hint)
    singfac_mat .= 1.0 ./ ((intr.mlow:intr.mhigh) .- odet.q .* (intr.nlow:intr.nhigh)')

    # Allocate temporaries from pool
    fmat_lower = acquire!(pool, ComplexF64, Npert, Npert)
    kmat = similar!(pool, fmat_lower)
    gmat = similar!(pool, fmat_lower)
    w    = similar!(pool, fmat_lower)  # w = Q - K̄·S
    v    = similar!(pool, fmat_lower)  # v = F̄⁻¹·w (then reused for S·Ḡ·S)
    tmp  = similar!(pool, fmat_lower)  # scratch

    # Evaluate F̄ (Cholesky factor), K̄, Ḡ splines at current ψ
    ffit.fmats_lower(vec(fmat_lower), psieval; hint=ffit._hint)
    ffit.kmats(vec(kmat), psieval; hint=ffit._hint)
    ffit.gmats(vec(gmat), psieval; hint=ffit._hint)

    # w = Q - K̄·S:  w[i,j] = singfac_vec[i]·δ_ij - (K̄·S)[i,j]
    # Q is DIAGONAL (singfac_vec[i] only on i==j), so we cannot broadcast singfac_vec
    # over all columns — that would give the wrong off-diagonal values.
    mul!(w, kmat, S)      # w = K̄·S
    @. w = -w             # w = -K̄·S
    for i in 1:Npert
        @inbounds w[i, i] += singfac_vec[i]  # add diagonal Q: w = Q - K̄·S
    end

    # v = F̄⁻¹·w  (in-place Cholesky solve with stored lower-triangular factor)
    v .= w
    ldiv!(LowerTriangular(fmat_lower), v)
    ldiv!(UpperTriangular(fmat_lower'), v)

    # dS = w†·v - S·Ḡ·S  [Glasser 2018 eq. 19, dual Riccati]
    mul!(dS, adjoint(w), v)   # dS = w†·v

    # Store du1/dψ = Q·v as a diagnostic before v is reused
    # Q·v = diag(singfac_vec)·v = Ξ'_Ψ (displacement gradient, with U₂ = I)
    @. odet.du[:, :, 1] = singfac_vec * v
    @view(odet.du[:, :, 2]) .= 0
    odet.xi_s .= 0

    # Subtract S·Ḡ·S (reuse v and tmp to avoid extra allocation)
    mul!(tmp, gmat, S)        # tmp = Ḡ·S
    mul!(v, S, tmp)           # v   = S·Ḡ·S
    dS .-= v
end

"""
    riccati_integrator_callback!(integrator)

Callback function for the Riccati ODE integrator. Handles tolerance updates,
renormalization, and storage at each step.

Uses `sing_der!` as the ODE RHS: u[:,:,1] = U₁ (starts as S), u[:,:,2] = U₂ (starts as I).
When max(|U₁|) or max(|U₂|) exceeds `ctrl.ucrit`, applies `renormalize_riccati_inplace!`
to compute S = U₁·U₂⁻¹ and reset U₂ = I. This is the Riccati analogue of Gaussian
reduction in the standard `integrator_callback!`, and keeps the ODE inputs bounded.
"""
function riccati_integrator_callback!(integrator)

    ctrl, _, _, intr, odet, chunk = integrator.p

    odet.total_steps += 1  # count every accepted solver step (saved or not), as segment_callback! does

    # Use unified tolerance (matches integrate_el_region! on develop)
    integrator.opts.reltol = ctrl.eulerlagrange_tolerance

    # Renormalize when norms exceed ucrit (analogous to Gaussian reduction in integrator_callback!)
    # During sing_der! integration: u[:,:,1]=U₁ (grows), u[:,:,2]=U₂ (grows).
    # Renorm computes S = U₁·U₂⁻¹ and resets U₂ = I, keeping inputs bounded.
    if maximum(abs, @view(integrator.u[:, :, 1])) > ctrl.ucrit ||
       maximum(abs, @view(integrator.u[:, :, 2])) > ctrl.ucrit
        renormalize_riccati_inplace!(integrator.u, intr.numpert_total)
    end

    # Determine if we should save this step. Always save the first 1-2 steps of a segment
    # and the last few steps near the right endpoint (relative band SAVE_NEAR_END_FRAC of the
    # span, or absolute floor SAVE_NEAR_END_PSI for very short chunks); save every save_interval-th
    # step in between.
    psi_range = abs(integrator.sol.prob.tspan[2] - integrator.sol.prob.tspan[1])
    psi_remaining = abs(integrator.sol.prob.tspan[2] - integrator.t)
    near_end = psi_remaining < SAVE_NEAR_END_FRAC * psi_range || psi_remaining < SAVE_NEAR_END_PSI
    steps_in_segment = length(integrator.sol.t)
    near_start = steps_in_segment <= 2
    should_save = near_start || near_end || (odet.step % ctrl.save_interval == 0)

    if should_save
        store_ode_data!(odet, integrator.t, integrator.u)
    end
end

"""
    riccati_integrate_chunk!(odet, ctrl, equil, ffit, intr, chunk)

Integrate the dual Riccati ODE from `chunk.psi_start` to `chunk.psi_end`.

Uses `sing_der!` as the ODE RHS with `riccati_integrator_callback!`, which applies
`renormalize_riccati_inplace!` (instead of Gaussian reduction) when norms exceed ucrit.
Starting state: u[:,:,1] = S_prev, u[:,:,2] = I (set by initialization or previous renorm).
Ending state: u[:,:,1] = U₁, u[:,:,2] = U₂ (ratio S = U₁·U₂⁻¹ is the updated Riccati matrix).
"""
function riccati_integrate_chunk!(
    odet::OdeState, ctrl::ForceFreeStatesControl, equil::Equilibrium.PlasmaEquilibrium,
    ffit::FourFitVars, intr::ForceFreeStatesInternal, chunk::IntegrationChunk
)
    cb = DiscreteCallback((u, t, integrator) -> true, riccati_integrator_callback!)
    rtol = ctrl.eulerlagrange_tolerance
    prob = ODEProblem(sing_der!, odet.u, (chunk.psi_start, chunk.psi_end),
                      (ctrl, equil, ffit, intr, odet, chunk))
    sol = solve(prob, Vern9(); reltol=rtol, callback=cb, save_everystep=false, save_end=true)
    odet.u .= sol.u[end]
    odet.psifac = sol.t[end]
    # Renormalize end state to (S, I) convention for the next chunk.
    # When a crossing follows (needs_crossing=true), skip renorm so that ca_l is computed
    # from the bounded (U₁, U₂) state in riccati_cross_ideal_singular_surf!: this gives
    # consistent normalization with ca_r (also from pre-renorm state), enabling correct Δ'.
    # The callback guarantees max(|U₁|), max(|U₂|) ≤ ucrit, so the state is bounded.
    if !chunk.needs_crossing
        renormalize_riccati_inplace!(odet.u, intr.numpert_total)
    end
end

"""
    renormalize_riccati!(odet, intr)

After a singular surface crossing, restore the canonical Riccati storage convention:
  u[:,:,1] = S_new = U₁_new · U₂_new⁻¹
  u[:,:,2] = I

`riccati_cross_ideal_singular_surf!` leaves u[:,:,1] = U₁_new and u[:,:,2] = U₂_new (not I),
so this step is required before continuing the Riccati integration.

The u_store entry from the crossing correctly has U₁_new and U₂_new (stored before this call),
so `compute_smallest_eigenvalue` still computes U₁_new/U₂_new = S_new correctly.
"""
function renormalize_riccati!(odet::OdeState, intr::ForceFreeStatesInternal)
    N = intr.numpert_total
    # S_new = U₁_new · U₂_new⁻¹  (in-place to avoid allocation)
    U2_copy = copy(@view odet.u[:, :, 2])
    rdiv!(@view(odet.u[:, :, 1]), lu!(U2_copy))
    # Reset U₂ = I
    fill!(@view(odet.u[:, :, 2]), 0)
    for i in 1:N
        odet.u[i, i, 2] = 1
    end
end

"""
    renormalize_riccati_inplace!(u, N)

In-place Riccati renormalization on an arbitrary N×N×2 array:
  u[:,:,1] = U₁ · U₂⁻¹  (new S)
  u[:,:,2] = I

Used in `riccati_integrator_callback!` to renormalize the integrator's live state
when column norms grow beyond `ctrl.ucrit`, analogous to Gaussian reduction in the
standard ODE. This keeps the inputs to `sing_der!` bounded, preventing the same
exponential growth that occurs in the standard (non-Riccati) ODE without Gaussian reduction.
"""
function renormalize_riccati_inplace!(u::Array{ComplexF64,3}, N::Int)
    U2_copy = copy(@view u[:, :, 2])
    rdiv!(@view(u[:, :, 1]), lu!(U2_copy))
    fill!(@view(u[:, :, 2]), 0)
    for i in 1:N
        u[i, i, 2] = 1
    end
end

"""
    riccati_cross_ideal_singular_surf!(odet, ctrl, equil, ffit, intr, ising)

Cross a singular surface for the Riccati formulation. Replaces `cross_ideal_singular_surf!`
for the Riccati integration path with two key differences:

1. **No Gaussian reduction**: `cross_ideal_singular_surf!` calls `compute_solution_norms!`
   which applies Gaussian reduction to (S, I). This divides by pivot elements of S, which
   can be near-zero (S = 0 at axis and grows slowly), producing NaN/Inf in U₂. For Riccati,
   S is bounded so Gaussian reduction is unnecessary.

2. **Direct column zeroing**: Instead of using the GR-sorted `odet.index` to identify the
   column to zero, we use `ipert_res` directly (the resonant mode index). This is valid since
   without GR there is no permutation applied to the columns of S.

**Δ' normalization**: This function expects `odet.u` in the bounded (U₁, U₂) form produced by
`riccati_integrate_chunk!` with `needs_crossing=true` (final renorm skipped). ca_l is computed
from (U₁, U₂) before the crossing, and ca_r from (U₁_new, U₂_new) before `renormalize_riccati!`.
Since column `ipert_res` of [U₁_new; U₂_new] equals the introduced asymptotic solution exactly,
ca_r[ipert_res,ipert_res,2] = 1 regardless of other column normalizations. This gives a
physically meaningful Δ' = ca_r - ca_l with consistent left/right normalization.

After the predictor step and asymptotic introduction, `renormalize_riccati!` is called
to restore the canonical (S_new, I) form before continuing integration.

The u_store entry at the crossing step correctly stores (U₁_new, U₂_new) so that
`evaluate_stability_criterion!` can compute U₁_new / U₂_new = S_new correctly.
"""
function riccati_cross_ideal_singular_surf!(
    odet::OdeState, ctrl::ForceFreeStatesControl, equil::Equilibrium.PlasmaEquilibrium,
    ffit::FourFitVars, intr::ForceFreeStatesInternal, ising::Int
)
    # Skip Gaussian reduction — S is bounded so no large-norm columns exist.
    singp = intr.sing[ising]
    dpsi = singp.psifac - odet.psifac  # ψ_res - ψ_current (positive)
    ipert_res = 1 .+ singp.m .- intr.mlow .+ (singp.n .- intr.nlow) .* intr.mpert

    sing_asymp_left, sing_asymp_right = _two_sided_singular_asymptotics(singp, ctrl, equil, ffit, intr)
    _log_riccati_crossing_diagnostics(odet, intr, ising, singp, dpsi, sing_asymp_left, sing_asymp_right)

    _capture_left_crossing_data!(odet, singp, sing_asymp_left, dpsi, intr, ising)
    _predict_across_singular_surface!(odet, ctrl, equil, ffit, intr, ising, ipert_res, dpsi, sing_asymp_right)
    _capture_right_crossing_data!(odet, singp, sing_asymp_right, dpsi, intr, ising, ipert_res, ctrl)

    _stash_per_surface_delta_prime_stub!(odet, intr, ising, ipert_res, sing_asymp_right, equil, ctrl)
    _store_crossing_step!(odet)

    # Restore canonical (S_new, I) form before continuing integration.
    renormalize_riccati!(odet, intr)
end

"""
    _two_sided_singular_asymptotics(singp, ctrl, equil, ffit, intr) -> (left, right)

Compute left- (`sig=-1`) and right- (`sig=+1`) side singular asymptotics matching
Fortran STRIDE's separate vmatl/vmatr (sing_vmat). Alpha is taken from the right
side and shared with the left.
"""
function _two_sided_singular_asymptotics(singp::SingType, ctrl::ForceFreeStatesControl,
                                         equil::Equilibrium.PlasmaEquilibrium, ffit::FourFitVars,
                                         intr::ForceFreeStatesInternal)
    sing_asymp_right = compute_sing_asymptotics(singp, ctrl, equil, ffit, intr; sig=1.0)
    sing_asymp_left  = compute_sing_asymptotics(singp, ctrl, equil, ffit, intr; sig=-1.0,
                                                alpha_override=sing_asymp_right.alpha)
    return sing_asymp_left, sing_asymp_right
end

# @debug-only per-crossing diagnostics. Enable via JULIA_DEBUG=GeneralizedPerturbedEquilibrium.
function _log_riccati_crossing_diagnostics(odet, intr, ising, singp, dpsi, sing_asymp_left, sing_asymp_right)
    @debug begin
        ipert_res_diag = 1 .+ singp.m .- intr.mlow .+ (singp.n .- intr.nlow) .* intr.mpert
        msg = "  ising=$ising: psi_sing=$(@sprintf("%.10f", singp.psifac)), psi_eval=$(@sprintf("%.10f", odet.psifac)), dpsi=$(@sprintf("%.10e", dpsi))\n"
        msg *= "  alpha_L = $(sing_asymp_left.alpha), alpha_R = $(sing_asymp_right.alpha)\n"
        for ip in ipert_res_diag
            msg *= "  vmatL[0] big: vmat[$ip,$ip,1,1]=$(@sprintf("%.8e", real(sing_asymp_left.vmat[ip,ip,1,1]))), vmat[$ip,$ip,2,1]=$(@sprintf("%.8e", real(sing_asymp_left.vmat[ip,ip,2,1])))\n"
            msg *= "  vmatR[0] big: vmat[$ip,$ip,1,1]=$(@sprintf("%.8e", real(sing_asymp_right.vmat[ip,ip,1,1]))), vmat[$ip,$ip,2,1]=$(@sprintf("%.8e", real(sing_asymp_right.vmat[ip,ip,2,1])))\n"
        end
        msg
    end
end

# Capture left-side asymptotic data into odet.ca_l and singp.ua_left/psi_ua_left.
function _capture_left_crossing_data!(odet::OdeState, singp::SingType, sing_asymp_left,
                                      dpsi::Float64, intr::ForceFreeStatesInternal, ising::Int)
    ua = sing_get_ua(sing_asymp_left, dpsi)
    singp.ua_left = copy(ua)
    singp.psi_ua_left = odet.psifac
    odet.ca_l[:, :, :, ising] .= sing_get_ca(odet.u, ua, intr)
end

# Trapezoidal predictor across the singular surface: zero the resonant columns,
# evaluate sing_der! on both sides, advance odet by (du1 + du2)·dpsi, and jump
# odet.psifac to the right side. The zeroed columns stay zero through the predictor
# since du[:, ipert_res, :] = 0 when u[:, ipert_res, :] = 0.
function _predict_across_singular_surface!(odet::OdeState, ctrl::ForceFreeStatesControl,
                                           equil::Equilibrium.PlasmaEquilibrium, ffit::FourFitVars,
                                           intr::ForceFreeStatesInternal, ising::Int,
                                           ipert_res, dpsi::Float64, sing_asymp_right)
    if ctrl.kinetic_factor == 0
        for i in eachindex(sing_asymp_right.r1)
            odet.u[:, ipert_res[i], :] .= 0
        end
    end
    params = (ctrl, equil, ffit, intr, odet, IntegrationChunk(0.0, 0.0, false, ising, 1))
    du1 = zeros(ComplexF64, intr.numpert_total, intr.numpert_total, 2)
    du2 = zeros(ComplexF64, intr.numpert_total, intr.numpert_total, 2)
    sing_der!(du1, odet.u, params, odet.psifac)
    odet.psifac += 2 * dpsi  # jump to other side of singular surface
    sing_der!(du2, odet.u, params, odet.psifac)
    odet.u .+= (du1 .+ du2) .* dpsi
end

# Inject the right-side small asymptotic into the resonant columns of (U₁_new, U₂_new),
# capture odet.ca_r, and save singp.ua_right / psi_ua_right.
# Column ipert_res of [U₁_new; U₂_new] = ua[:, ipert_res+N, :] (the introduced small asymptotic),
# so ca_r[ipert_res, ipert_res, 2] = 1 regardless of other columns' normalization.
function _capture_right_crossing_data!(odet::OdeState, singp::SingType, sing_asymp_right,
                                       dpsi::Float64, intr::ForceFreeStatesInternal, ising::Int,
                                       ipert_res, ctrl::ForceFreeStatesControl)
    ua = sing_get_ua(sing_asymp_right, dpsi)
    singp.ua_right = copy(ua)
    singp.psi_ua_right = odet.psifac
    if ctrl.kinetic_factor == 0
        for i in eachindex(sing_asymp_right.r1)
            odet.u[ipert_res[i], :, :] .= 0
            odet.u[:, ipert_res[i], :] .= ua[:, ipert_res[i]+intr.numpert_total, :]
        end
    end
    odet.ca_r[:, :, :, ising] .= sing_get_ca(odet.u, ua, intr)
end

# STUB: per-surface ca-based Δ' (not physically valid; see SingType.delta_prime docstring).
# The canonical Δ' is intr.delta_prime_matrix from compute_delta_prime_matrix!.
function _stash_per_surface_delta_prime_stub!(odet::OdeState, intr::ForceFreeStatesInternal,
                                              ising::Int, ipert_res, sing_asymp_right,
                                              equil::Equilibrium.PlasmaEquilibrium,
                                              ctrl::ForceFreeStatesControl)
    ctrl.kinetic_factor == 0 || return
    denom = (2π)^2 * equil.psio
    n_res = length(sing_asymp_right.r1)
    N = intr.numpert_total
    resize!(intr.sing[ising].delta_prime, n_res)
    intr.sing[ising].delta_prime_col = zeros(ComplexF64, N, n_res)
    for i in eachindex(sing_asymp_right.r1)
        Δca_col = (odet.ca_r[:, ipert_res[i], 2, ising] - odet.ca_l[:, ipert_res[i], 2, ising]) / denom
        intr.sing[ising].delta_prime_col[:, i] .= Δca_col
        intr.sing[ising].delta_prime[i] = Δca_col[ipert_res[i]]
    end
end

# Store (U₁_new, U₂_new) into u_store before renormalization so that
# evaluate_stability_criterion! can recover S_new = U₁_new / U₂_new via compute_smallest_eigenvalue.
function _store_crossing_step!(odet::OdeState)
    store_ode_data!(odet, odet.psifac, odet.u)
end

"""
    riccati_eulerlagrange_integration(ctrl, equil, ffit, intr) -> OdeState

Main driver for integrating the dual Riccati ODE across the plasma.
Functionally identical to `eulerlagrange_integration` except:

1. Uses `riccati_integrate_chunk!`: drives `sing_der!` with `riccati_integrator_callback!`
   which applies `renormalize_riccati_inplace!` (instead of Gaussian reduction) when
   column norms exceed ucrit
2. Uses `riccati_cross_ideal_singular_surf!` instead of `cross_ideal_singular_surf!`:
   skips Gaussian reduction (avoids near-zero pivot issues when S is small near axis)
   and renormalizes to (S_new, I) in one step
3. Skips `transform_u!` — S is already the true solution, no Gaussian-reduction undo needed

Enable via `use_riccati = true` in `[ForceFreeStates]` section of gpec.toml, or by
setting `ctrl.use_riccati = true` programmatically.
"""
function riccati_eulerlagrange_integration(
    ctrl::ForceFreeStatesControl, equil::Equilibrium.PlasmaEquilibrium,
    ffit::FourFitVars, intr::ForceFreeStatesInternal
)
    # Initialization — same as eulerlagrange_integration
    odet = OdeState(intr.numpert_total, ctrl.numsteps_init, ctrl.numunorms_init, intr.msing)
    if ctrl.sing_start <= 0
        initialize_el_at_axis!(odet, ctrl, ffit, equil.profiles, intr)
    elseif ctrl.sing_start <= intr.msing
        error("sing_start > 0 not implemented yet!")
    else
        error("Invalid value for sing_start: $(ctrl.sing_start) > msing = $(intr.msing)")
    end

    chunks = chunk_el_integration_bounds(odet, ctrl, intr)

    # Prime odet.new = false so that compute_solution_norms! (if called elsewhere)
    # does not skip Gaussian reduction on first invocation. Also initialize unorm0
    # to safe defaults since the Riccati callback never calls compute_solution_norms!.
    odet.new = false
    fill!(odet.unorm0, 1.0)

    if ctrl.verbose
        @info "   ψ = $((@sprintf "%.3f" odet.psifac)),  q = $((@sprintf "%.3f" equil.profiles.q_spline(odet.psifac)))"
    end

    for chunk in chunks
        # Integrate this chunk using the Riccati ODE (Riccati callback skips Gaussian reduction)
        riccati_integrate_chunk!(odet, ctrl, equil, ffit, intr, chunk)
        if ctrl.verbose
            @info "   ψ = $((@sprintf "%.3f" odet.psifac)),  q= $((@sprintf "%.3f" odet.q)),  max(S) = $((@sprintf "%.2e" maximum(abs, odet.u[:,:,1]))),  steps = $(odet.step-1)"
        end

        # Cross rational surface (Riccati crossing skips GR, uses ipert_res directly)
        if chunk.needs_crossing
            if ctrl.kinetic_factor > 0
                error("kinetic_factor > 0 not implemented yet in Riccati!")
            else
                riccati_cross_ideal_singular_surf!(odet, ctrl, equil, ffit, intr, chunk.ising)
                # renormalize_riccati! is called inside riccati_cross_ideal_singular_surf!
            end
        end
    end

    # Edge-dW scan over [psiedge, psilim] — populates odet.edge_scan for HDF5 output.
    # See EulerLagrange.jl counterpart and ForceFreeStatesControl docstring for the
    # diagnostic vs legacy-truncation semantics and reliability caveats on
    # truncate_at_dW_peak=true.
    odet.step -= 1
    trim_storage!(odet)
    if ctrl.psiedge < intr.psilim
        saved_psifac, saved_u = odet.psifac, copy(odet.u)
        peak_step = findmax_dW_edge!(odet, ctrl, equil, ffit, intr)
        if ctrl.truncate_at_dW_peak
            # Legacy: truncate integration data to dW peak (corrupts Δ' and δW).
            odet.step = peak_step
            trim_storage!(odet)
            intr.psilim = odet.psi_store[end]
            intr.qlim = odet.q_store[end]
            odet.u .= odet.u_store[:, :, :, end]
            if ctrl.verbose
                @info "Truncating integration at peak edge dW (LEGACY — Δ'/δW unreliable): ψ = $((@sprintf "%.2f" odet.psi_store[odet.step])),  q = $((@sprintf "%.2f" odet.q_store[odet.step]))"
            end
        else
            odet.psifac = saved_psifac
            odet.u .= saved_u
            if ctrl.verbose
                @info "Edge-dW peak (diagnostic): ψ = $((@sprintf "%.2f" odet.psi_store[peak_step])),  q = $((@sprintf "%.2f" odet.q_store[peak_step])); integration domain unchanged"
            end
        end
    end

    # Evaluate fixed-boundary stability criterion
    if ctrl.verbose
        @info "Evaluating fixed-boundary stability criterion"
    end
    odet.nzero = evaluate_stability_criterion!(odet, equil.profiles)

    # Note: transform_u! is intentionally skipped.
    # S is already the true solution (invariant under Gaussian reduction),
    # and u_store entries have u[:,:,1]=S, u[:,:,2]=I throughout integration.
    # At crossing steps, u_store has U₁_new/U₂_new which compute_smallest_eigenvalue
    # correctly resolves to S_new via rdiv. No transformation is needed.

    return odet
end

"""
    integrate_propagator_chunk!(prop, chunk, ctrl, equil, ffit, intr, odet_proxy)

Compute the fundamental matrix (propagator) for one integration chunk by solving the
EL ODE twice from identity-block initial conditions.

The first solve uses IC = (I_N, 0_N) (U₁=I, U₂=0) and stores the result in
`prop.block_upper_ic`. The second uses IC = (0_N, I_N) (U₁=0, U₂=I) and stores
the result in `prop.block_lower_ic`.

`odet_proxy` is a per-thread lightweight `OdeState` used to provide thread-local
storage for `sing_der!` side effects (`q`, `ud`, `spline_hint`). Multiple threads
may call this function concurrently using distinct `odet_proxy` objects.

No callback is used: the propagator integration proceeds without normalization or
storage steps, since the identity ICs ensure bounded solutions within each chunk.
"""
function integrate_propagator_chunk!(
    prop::ChunkPropagator,
    chunk::IntegrationChunk,
    ctrl::ForceFreeStatesControl,
    equil::Equilibrium.PlasmaEquilibrium,
    ffit::FourFitVars,
    intr::ForceFreeStatesInternal,
    odet_proxy::OdeState
)
    N = intr.numpert_total
    # Reverse tspan for backward chunks (direction=-1): OrdinaryDiffEq handles negative tspan
    # naturally. The resulting propagator maps state at psi_end → psi_start, which is
    # well-conditioned because exponentially growing solutions (forward) decay backward.
    tspan = chunk.direction == 1 ?
        (chunk.psi_start, chunk.psi_end) :
        (chunk.psi_end,   chunk.psi_start)
    rtol = ctrl.eulerlagrange_tolerance
    params = (ctrl, equil, ffit, intr, odet_proxy, chunk)

    # Upper block IC: U₁ = I, U₂ = 0
    u_upper = zeros(ComplexF64, N, N, 2)
    for i in 1:N
        u_upper[i, i, 1] = 1
    end
    odet_proxy.spline_hint[] = 1
    odet_proxy.ffit_hint[] = 1
    prob = ODEProblem(sing_der!, u_upper, tspan, params)
    sol = solve(prob, Vern9(); reltol=rtol, save_everystep=false, save_end=true)
    prop.block_upper_ic .= sol.u[end]
    odet_proxy.total_steps += sol.stats.naccept  # thread-local; summed into odet after the BVP barrier

    # Lower block IC: U₁ = 0, U₂ = I
    u_lower = zeros(ComplexF64, N, N, 2)
    for i in 1:N
        u_lower[i, i, 2] = 1
    end
    odet_proxy.spline_hint[] = 1
    odet_proxy.ffit_hint[] = 1
    prob = ODEProblem(sing_der!, u_lower, tspan, params)
    sol = solve(prob, Vern9(); reltol=rtol, save_everystep=false, save_end=true)
    prop.block_lower_ic .= sol.u[end]
    odet_proxy.total_steps += sol.stats.naccept
end

"""
    integrate_fm_with_ua_ic(chunks, chunk_range, ua, ctrl, equil, ffit, intr;
                            backward=false) -> Matrix{ComplexF64}

Re-integrate a span of chunks using ua (asymptotic solution) as initial conditions, matching
Fortran STRIDE's uFM_sing_init behavior. Returns a 2N×2N fundamental matrix
where column j is the ODE solution at the span endpoint with IC = column j of T = [ua[:,:,1]; ua[:,:,2]].

When `backward=false` (default): ua is the IC at psi_start, integrate forward to psi_end.
When `backward=true`: ua is the IC at psi_end, integrate backward to psi_start. The result
maps asymptotic coefficients at psi_end → state at psi_start.

This provides numerically accurate propagators near singular surfaces because the ODE integrator
maintains per-column relative accuracy even when columns span a 10^8+ dynamic range (big/small
solutions). In contrast, post-multiplying a pre-computed identity-IC propagator by T loses the
small-solution information to roundoff.
"""
function integrate_fm_with_ua_ic(
    chunks::Vector{IntegrationChunk},
    chunk_range::UnitRange{Int},
    ua::Array{ComplexF64,3},
    ctrl::ForceFreeStatesControl,
    equil::Equilibrium.PlasmaEquilibrium,
    ffit::FourFitVars,
    intr::ForceFreeStatesInternal;
    backward::Bool = false,
    psi_ua::Float64 = NaN
)
    N = intr.numpert_total
    psi_start = chunks[first(chunk_range)].psi_start
    psi_end   = chunks[last(chunk_range)].psi_end
    # Use stored ua ψ location if provided; otherwise fall back to chunk boundary.
    # The ua is evaluated at the inner-layer boundary (exact ψ from singular crossing),
    # which may differ slightly from the nearest chunk boundary.
    if backward && !isnan(psi_ua)
        psi_end = psi_ua  # ua lives at psi_ua, not at chunk boundary
    elseif !backward && !isnan(psi_ua)
        psi_start = psi_ua  # ua lives at psi_ua, not at chunk boundary
    end
    # For backward integration: start at psi_end (where ua lives), integrate to psi_start
    tspan = backward ? (psi_end, psi_start) : (psi_start, psi_end)
    rtol = ctrl.eulerlagrange_tolerance

    result = zeros(ComplexF64, 2N, 2N)
    odet_proxy = OdeState(N, 1, 1, 0)
    dummy_chunk = IntegrationChunk(psi_start, psi_end, false, 0, backward ? -1 : 1)
    params = (ctrl, equil, ffit, intr, odet_proxy, dummy_chunk)

    # Batch 1: columns 1:N of T (big solutions)
    u0 = zeros(ComplexF64, N, N, 2)
    u0[:, :, 1] .= ua[:, 1:N, 1]
    u0[:, :, 2] .= ua[:, 1:N, 2]
    odet_proxy.spline_hint[] = 1
    odet_proxy.ffit_hint[] = 1
    prob = ODEProblem(sing_der!, u0, tspan, params)
    sol = solve(prob, Vern9(); reltol=rtol, save_everystep=false, save_end=true)
    result[1:N, 1:N]     .= sol.u[end][:, :, 1]
    result[N+1:2N, 1:N]  .= sol.u[end][:, :, 2]

    # Batch 2: columns N+1:2N of T (small solutions)
    u0[:, :, 1] .= ua[:, N+1:2N, 1]
    u0[:, :, 2] .= ua[:, N+1:2N, 2]
    odet_proxy.spline_hint[] = 1
    odet_proxy.ffit_hint[] = 1
    prob = ODEProblem(sing_der!, u0, tspan, params)
    sol = solve(prob, Vern9(); reltol=rtol, save_everystep=false, save_end=true)
    result[1:N, N+1:2N]     .= sol.u[end][:, :, 1]
    result[N+1:2N, N+1:2N]  .= sol.u[end][:, :, 2]

    return result
end

"""
    apply_propagator!(odet, prop)

Apply the chunk propagator `prop` to the current state `odet.u` in-place.

The propagator acts as a linear map on the (U₁, U₂) pair:

  U₁_new = block_upper_ic[:,:,1] · U₁_prev + block_lower_ic[:,:,1] · U₂_prev
  U₂_new = block_upper_ic[:,:,2] · U₁_prev + block_lower_ic[:,:,2] · U₂_prev

This correctly propagates any state (not just the identity), including the
(S, I) form produced by Riccati-style crossings.

Implements the subpropagator composition Φ(ψ₂, ψ₀) = Φ(ψ₂, ψ₁) · Φ(ψ₁, ψ₀) of
Glasser-Kolemen (2018) Phys. Plasmas 25, 032501 Eq. 29.
"""
function apply_propagator!(odet::OdeState, prop::ChunkPropagator)
    U1_upper = @view prop.block_upper_ic[:, :, 1]
    U2_upper = @view prop.block_upper_ic[:, :, 2]
    U1_lower = @view prop.block_lower_ic[:, :, 1]
    U2_lower = @view prop.block_lower_ic[:, :, 2]

    u1_prev = copy(@view odet.u[:, :, 1])
    u2_prev = copy(@view odet.u[:, :, 2])
    tmp = similar(u1_prev)

    # U₁_new = U1_upper · u1_prev + U1_lower · u2_prev
    mul!(view(odet.u, :, :, 1), U1_upper, u1_prev)
    mul!(tmp, U1_lower, u2_prev)
    odet.u[:, :, 1] .+= tmp

    # U₂_new = U2_upper · u1_prev + U2_lower · u2_prev
    mul!(view(odet.u, :, :, 2), U2_upper, u1_prev)
    mul!(tmp, U2_lower, u2_prev)
    odet.u[:, :, 2] .+= tmp
end

"""
    apply_propagator_inverse!(odet, prop)

Apply the *inverse* of the chunk propagator `prop` to the current state `odet.u` in-place.

Used for backward chunks (direction=-1): the stored propagator Φ_bwd maps state at
`psi_end` → state at `psi_start` (well-conditioned because solutions that grow
exponentially forward decay backward). To advance the Riccati state from `psi_start`
to `psi_end`, we solve Φ_bwd · x = u_old, which gives x = Φ_bwd⁻¹ · u_old = Φ_fwd · u_old.

Since Φ_bwd is well-conditioned, the LU solve is accurate, giving the same result as
applying the (ill-conditioned) forward propagator Φ_fwd but with far better precision.

Implements the inverse subpropagator identity Φ(ψ₂, ψ₁) = Φ(ψ₁, ψ₂)⁻¹ of
Glasser-Kolemen (2018) Phys. Plasmas 25, 032501 Eq. 33.
"""
function apply_propagator_inverse!(odet::OdeState, prop::ChunkPropagator)
    N = size(odet.u, 1)
    # Assemble 2N×2N backward FM Φ_bwd
    #! format: off
    Φ = [prop.block_upper_ic[:,:,1]  prop.block_lower_ic[:,:,1];
         prop.block_upper_ic[:,:,2]  prop.block_lower_ic[:,:,2]]
    #! format: on
    # Φ_bwd maps state at psi_end → psi_start (well-conditioned).
    # We want Φ_fwd = Φ_bwd⁻¹ to advance state from psi_start → psi_end.
    # Solving Φ_bwd · x = [U₁_old; U₂_old] gives x = Φ_bwd⁻¹ · [U₁_old; U₂_old].
    u_old = [odet.u[:,:,1]; odet.u[:,:,2]]   # 2N × N
    u_new = Φ \ u_old                         # LU solve, 2N × N
    odet.u[:,:,1] .= u_new[1:N, :]
    odet.u[:,:,2] .= u_new[N+1:2N, :]
end

"""
    parallel_eulerlagrange_integration(ctrl, equil, ffit, intr) -> OdeState

Parallel fundamental matrix (propagator) driver for the EL integration.

Functionally equivalent to `eulerlagrange_integration`, integrating all bulk chunks
concurrently using `Threads.@threads`, then re-integrating the outer plasma serially:

1. **Chunk generation**: calls `chunk_el_integration_bounds`, then `balance_integration_chunks`
   to sub-divide chunks for load-balanced parallel execution.
2. **Parallel phase**: `integrate_propagator_chunk!` integrates each chunk independently
   from identity initial conditions (no accumulated state, no normalization/callback).
   Each thread uses a private `OdeState` proxy for `sing_der!` side effects.
3. **Serial assembly**: propagators are applied sequentially with `apply_propagator!`.
   Rational surface crossings use `riccati_cross_ideal_singular_surf!` (no Gaussian
   reduction) matching the Riccati path convention.
4. **Outer plasma re-integration**: after the last rational surface crossing, the outer
   plasma (from last ψ_s to psilim) is re-integrated using `riccati_integrate_chunk!`.
   FM propagation in this region is prone to precision loss for high N (exponential growth
   without renormalization); Riccati integration keeps matrices bounded and provides dense
   checkpoints for `findmax_dW_edge!`.

Enable via `use_parallel = true` in `[ForceFreeStates]` of gpec.toml, or by setting
`ctrl.use_parallel = true` programmatically. Requires `singfac_min != 0`.

**Key differences from standard integration:**
- No Gaussian reduction in the propagator BVP phase (crossings use the
  Riccati-style algorithm, parallel `odet.ifix` stays 0)
- `transform_u!` is called on the parallel odet but is a no-op (ifix=0)
- Outer plasma uses serial Riccati integration for numerical stability
- A serial Euler-Lagrange **dense pass** is appended at the end and
  replaces the parallel `odet` so that `u_store` / `du_store` are dense and
  in axis basis — the only convention the PerturbedEquilibrium downstream
  code consumes correctly.  Δ' (`singular/delta_prime_matrix`) is computed
  from the parallel BVP and is bit-identical with vs. without this pass.
  Toggle off with `ctrl.populate_dense_xi = false` if only Δ' / vacuum /
  energies are needed and the extra serial-EL cost is unwanted (HDF5
  `integration/xi_*` will then be sparse / zero).

**Bidirectional integration for large-N accuracy:**
The crossing chunk (nearest to each rational surface singL[j]) is integrated *backward*
(`direction=-1`, `tspan` reversed). Backward integration of a region where solutions grow
exponentially forward causes them to *decay*, so the resulting backward FM Φ_bwd is
well-conditioned. The accurate forward propagation is recovered as Φ_bwd⁻¹ via a stable
LU solve in `apply_propagator_inverse!`. This follows the same principle as STRIDE
(Glasser 2018 Phys. Plasmas 25, 032501). The all-forward path had ~10% energy error for
the DIIID-like example (N=26, n=1); bidirectional reduces this to within 2%.
"""
function parallel_eulerlagrange_integration(
    ctrl::ForceFreeStatesControl, equil::Equilibrium.PlasmaEquilibrium,
    ffit::FourFitVars, intr::ForceFreeStatesInternal
)
    odet = _initialize_parallel_odet(ctrl, equil, ffit, intr)
    chunks, propagators, odet_proxies = _setup_parallel_chunks_and_proxies(odet, ctrl, intr)
    bvp_threads = max(1, min(Threads.nthreads(), ctrl.parallel_threads))
    _log_parallel_start(ctrl, odet, equil, chunks, bvp_threads)

    _run_parallel_bvp_phase!(propagators, chunks, ctrl, equil, ffit, intr, odet_proxies, bvp_threads)

    # Harvest solver-step counts accumulated thread-locally in each proxy during the BVP phase.
    # The outer re-integration below uses riccati_integrate_chunk!, which counts via its callback.
    odet.total_steps += sum(p.total_steps for p in odet_proxies)

    S_at_surface_left, last_crossing_step =
        _assemble_propagators_serially!(odet, propagators, chunks, ctrl, equil, ffit, intr)

    _reintegrate_outer_plasma!(odet, last_crossing_step, ctrl, equil, ffit, intr)

    chunks, propagators = _handle_edge_dW_scan!(odet, chunks, propagators, ctrl, equil, ffit, intr)

    # compute_delta_prime_matrix! is called from the main pipeline (after free_run!) so
    # that vacuum response wv is available for the edge BC. With self-consistent truncation,
    # the propagators/chunks returned here match intr.psilim exactly, so Δ' is well-defined
    # for both truncate_at_dW_peak=false (full domain) and =true (peak).
    if ctrl.verbose
        @info "Evaluating fixed-boundary stability criterion"
    end
    odet.nzero = evaluate_stability_criterion!(odet, equil.profiles)
    transform_u!(odet, intr)  # no-op when ifix=0 (no Gaussian reduction)

    # Replace BVP `odet` with a dense serial-EL pass so HDF5 `integration/xi_*` carries
    # valid DCON ξ in axis basis for PerturbedEquilibrium. Skipped when force_termination=true.
    if ctrl.populate_dense_xi && !ctrl.force_termination
        odet = _populate_dense_xi_via_serial_el!(odet, ctrl, equil, ffit, intr)
    end
    return odet, propagators, chunks, S_at_surface_left
end

# Build odet and initialize at the magnetic axis. Same path as serial eulerlagrange_integration.
function _initialize_parallel_odet(ctrl::ForceFreeStatesControl,
                                   equil::Equilibrium.PlasmaEquilibrium,
                                   ffit::FourFitVars,
                                   intr::ForceFreeStatesInternal)
    odet = OdeState(intr.numpert_total, ctrl.numsteps_init, ctrl.numunorms_init, intr.msing)
    if ctrl.sing_start <= 0
        initialize_el_at_axis!(odet, ctrl, ffit, equil.profiles, intr)
    elseif ctrl.sing_start <= intr.msing
        error("sing_start > 0 not implemented yet!")
    else
        error("Invalid value for sing_start: $(ctrl.sing_start) > msing = $(intr.msing)")
    end
    # Prime odet.new = false (consistent with riccati path — no Gaussian reduction used).
    odet.new = false
    fill!(odet.unorm0, 1.0)
    return odet
end

# Build the (bidirectional) chunk list, allocate per-chunk propagators, and allocate
# per-thread proxy OdeStates sized by maxthreadid() (Julia 1.9+ may report threadid
# values above nthreads() due to the interactive thread pool).
function _setup_parallel_chunks_and_proxies(odet::OdeState, ctrl::ForceFreeStatesControl,
                                            intr::ForceFreeStatesInternal)
    # Bidirectional chunks: crossing chunks are assigned direction=-1 so they are
    # integrated backward. The resulting Φ_bwd is well-conditioned because growing EL
    # solutions decay backward; forward propagation is recovered via LU solve in
    # apply_propagator_inverse! during serial assembly.
    base_chunks = chunk_el_integration_bounds(odet, ctrl, intr; bidirectional=true)
    chunks = balance_integration_chunks(base_chunks, ctrl, intr)
    N = intr.numpert_total
    propagators = [ChunkPropagator(N) for _ in chunks]
    odet_proxies = [OdeState(N, 1, 1, 0) for _ in 1:Threads.maxthreadid()]
    return chunks, propagators, odet_proxies
end

function _log_parallel_start(ctrl::ForceFreeStatesControl, odet::OdeState,
                             equil::Equilibrium.PlasmaEquilibrium,
                             chunks::Vector{IntegrationChunk}, bvp_threads::Int)
    ctrl.verbose || return
    @info "   ψ = $((@sprintf "%.3f" odet.psifac)),  q = $((@sprintf "%.3f" equil.profiles.q_spline(odet.psifac)))"
    @info "   Parallel FM: $(length(chunks)) chunks, $bvp_threads BVP thread$(bvp_threads == 1 ? "" : "s") (julia_nthreads=$(Threads.nthreads()), ctrl.parallel_threads=$(ctrl.parallel_threads))"
end

# Integrate each chunk's FM propagator from identity IC. Serial when bvp_threads == 1
# (bit-deterministic; ~20% slower than 2-thread but immune to thread-
# schedule sensitivity). Parallel uses :static scheduler so Threads.threadid() returns a
# stable index into odet_proxies. If a parallel run ever diverges on a delicate equilibrium,
# drop to parallel_threads = 1 rather than use_parallel = false — the latter is silently wrong.
function _run_parallel_bvp_phase!(propagators::Vector{ChunkPropagator},
                                  chunks::Vector{IntegrationChunk},
                                  ctrl::ForceFreeStatesControl,
                                  equil::Equilibrium.PlasmaEquilibrium, ffit::FourFitVars,
                                  intr::ForceFreeStatesInternal,
                                  odet_proxies::Vector{OdeState}, bvp_threads::Int)
    if bvp_threads == 1
        for i in eachindex(chunks)
            integrate_propagator_chunk!(propagators[i], chunks[i], ctrl, equil, ffit, intr,
                                        odet_proxies[1])
        end
    else
        Threads.@threads :static for i in eachindex(chunks)
            integrate_propagator_chunk!(propagators[i], chunks[i], ctrl, equil, ffit, intr,
                                        odet_proxies[Threads.threadid()])
        end
    end
end

# Apply per-chunk propagators serially to odet, renormalizing to (S, I) after each.
# This is the Julia equivalent of STRIDE's ode_fixup: products of K chunk FMs can have
# cond ~ (cond_per_chunk)^K causing catastrophic cancellation for large N (≥20); periodic
# renorm keeps each step at O(cond_per_chunk). Backward (direction=-1) crossing chunks are
# applied via apply_propagator_inverse! (Φ_bwd⁻¹ from LU solve). S_at_surface_left records
# the well-conditioned Riccati S at each surface's left boundary for use as the Δ' BVP
# axis BC. Returns (S_at_surface_left, last_crossing_step).
function _assemble_propagators_serially!(odet::OdeState, propagators::Vector{ChunkPropagator},
                                         chunks::Vector{IntegrationChunk},
                                         ctrl::ForceFreeStatesControl,
                                         equil::Equilibrium.PlasmaEquilibrium,
                                         ffit::FourFitVars, intr::ForceFreeStatesInternal)
    N = intr.numpert_total
    S_at_surface_left = Matrix{ComplexF64}[]
    last_crossing_step = 1
    for (i, chunk) in enumerate(chunks)
        if chunk.direction == -1
            apply_propagator_inverse!(odet, propagators[i])
        else
            apply_propagator!(odet, propagators[i])
        end
        renormalize_riccati_inplace!(odet.u, N)
        odet.psifac = chunk.psi_end
        odet.q = equil.profiles.q_spline(odet.psifac)

        if ctrl.verbose
            @info "   ψ = $((@sprintf "%.3f" odet.psifac)),  q= $((@sprintf "%.3f" odet.q)),  max(S) = $((@sprintf "%.2e" maximum(abs, odet.u[:,:,1]))),  steps = $(odet.step-1)"
        end

        if chunk.needs_crossing
            ctrl.kinetic_factor > 0 && error("kinetic_factor > 0 not implemented yet in Riccati!")
            # State is (S, I) from the renorm above — well-conditioned at the surface's left boundary.
            push!(S_at_surface_left, copy(odet.u[:, :, 1]))
            riccati_cross_ideal_singular_surf!(odet, ctrl, equil, ffit, intr, chunk.ising)
            last_crossing_step = odet.step - 1
        else
            # Save non-crossing end-of-chunk state. du_store is not meaningful here — when
            # ctrl.populate_dense_xi=true the entire odet is replaced by a serial-EL pass
            # at the end of parallel_eulerlagrange_integration.
            if odet.step >= size(odet.u_store, 4)
                resize_storage!(odet)
            end
            odet.psi_store[odet.step] = odet.psifac
            odet.q_store[odet.step] = odet.q
            @views odet.u_store[:, :, :, odet.step] .= odet.u
            odet.step += 1
        end
    end
    return S_at_surface_left, last_crossing_step
end

# Re-integrate the outer plasma (last rational surface → psilim) with Riccati for numerical
# stability and dense checkpoint storage. FM propagation here is prone to precision loss at
# high N because the solution grows exponentially without renormalization; Riccati keeps
# matrices bounded. Dense checkpoints are also needed by findmax_dW_edge!. The u_store
# entry at last_crossing_step holds (U₁_new, U₂_new) from riccati_cross_ideal_singular_surf!
# before renormalization; we renorm here to (S_new, I) as the Riccati starting state.
function _reintegrate_outer_plasma!(odet::OdeState, last_crossing_step::Int,
                                    ctrl::ForceFreeStatesControl,
                                    equil::Equilibrium.PlasmaEquilibrium, ffit::FourFitVars,
                                    intr::ForceFreeStatesInternal)
    N = intr.numpert_total
    odet.u .= odet.u_store[:, :, :, last_crossing_step]
    odet.psifac = odet.psi_store[last_crossing_step]
    odet.q = odet.q_store[last_crossing_step]
    odet.step = last_crossing_step + 1
    renormalize_riccati_inplace!(odet.u, N)
    outer_chunk = IntegrationChunk(; psi_start=odet.psifac, psi_end=intr.psilim * (1 - eps),
                                   needs_crossing=false, ising=0)
    riccati_integrate_chunk!(odet, ctrl, equil, ffit, intr, outer_chunk)
    # Post: odet.u is in (S, I) form; odet.step points to next empty slot.
end

# Edge-dW scan over [psiedge, psilim] — populates odet.edge_scan for HDF5. By default
# (truncate_at_dW_peak=false) it's diagnostic-only: integration domain is unchanged.
# When truncate_at_dW_peak=true, the dW peak becomes the new physical edge: intr.psilim,
# odet, propagators, and chunks are made self-consistent (straddling chunk rebuilt with
# shorter psi_end; chunks past the new boundary dropped). Without that rebuild, the Δ' BVP
# would apply the edge BC at the truncated psilim to a propagator still extending to the
# original psilim — silently shifting the outermost rational's Δ' by tens of percent.
# Returns the (possibly truncated) chunks and propagators arrays.
function _handle_edge_dW_scan!(odet::OdeState, chunks::Vector{IntegrationChunk},
                               propagators::Vector{ChunkPropagator},
                               ctrl::ForceFreeStatesControl,
                               equil::Equilibrium.PlasmaEquilibrium, ffit::FourFitVars,
                               intr::ForceFreeStatesInternal)
    N = intr.numpert_total
    odet.step -= 1
    trim_storage!(odet)
    ctrl.psiedge < intr.psilim || return chunks, propagators

    saved_psifac, saved_u = odet.psifac, copy(odet.u)
    peak_step = findmax_dW_edge!(odet, ctrl, equil, ffit, intr)

    if !ctrl.truncate_at_dW_peak
        odet.psifac = saved_psifac
        odet.u .= saved_u
        if ctrl.verbose
            @info "Edge-dW peak (diagnostic): ψ = $((@sprintf "%.2f" odet.psi_store[peak_step])),  q = $((@sprintf "%.2f" odet.q_store[peak_step])); integration domain unchanged"
        end
        return chunks, propagators
    end

    # Truncate to dW peak: relocate intr.psilim and rebuild Δ' BVP self-consistently.
    n_chunks_before = length(chunks)
    odet.step = peak_step
    trim_storage!(odet)
    intr.psilim = odet.psi_store[end]
    intr.qlim = odet.q_store[end]
    odet.u .= odet.u_store[:, :, :, end]
    renormalize_riccati_inplace!(odet.u, N)  # stored snapshot may be pre-renorm

    peak_psi = odet.psi_store[end]
    last_chunk_idx = findlast(c -> c.psi_start < peak_psi, chunks)
    if last_chunk_idx === nothing
        error("truncate_at_dW_peak: peak ψ=$peak_psi lies before all chunk starts")
    end
    straddling = chunks[last_chunk_idx]
    if straddling.psi_end > peak_psi
        new_chunk = IntegrationChunk(
            psi_start = straddling.psi_start,
            psi_end   = peak_psi,
            needs_crossing = straddling.needs_crossing,
            ising     = straddling.ising,
            direction = straddling.direction,
        )
        chunks[last_chunk_idx] = new_chunk
        odet_proxy = OdeState(N, 1, 1, 0)
        integrate_propagator_chunk!(propagators[last_chunk_idx], new_chunk,
                                    ctrl, equil, ffit, intr, odet_proxy)
    end
    n_dropped = 0
    if last_chunk_idx < length(chunks)
        n_dropped = length(chunks) - last_chunk_idx
        chunks      = chunks[1:last_chunk_idx]
        propagators = propagators[1:last_chunk_idx]
    end
    if ctrl.verbose
        @info "Truncating integration at peak edge dW (self-consistent): ψ = $((@sprintf "%.4f" peak_psi)),  q = $((@sprintf "%.3f" odet.q_store[end])).  Rebuilt chunk $last_chunk_idx; dropped $n_dropped of $n_chunks_before outer chunks."
    end
    return chunks, propagators
end

"""
    _populate_dense_xi_via_serial_el!(odet, ctrl, equil, ffit, intr) -> fresh_odet

Replace the propagator-BVP's `odet` with a fresh serial-EL `odet` that has
dense `u_store` / `du_store` populated in axis basis (the PerturbedEquilibrium
convention).  The caller's `odet` is fully replaced by the fresh one because
`free_run!` downstream uses `odet.u[:,:,1,end]` to normalize `odet.u_store`,
so both must be in the same basis.  The parallel BVP results that survive
downstream are stored in `intr` (psilim/qlim, sing[*].delta_prime, …) and in
the externally-returned `propagators` / `chunks` / `S_at_surface_left` —
none of those live on `odet`, so replacing `odet` is safe.

The dense pass uses the **serial EL path** (`sing_der!` with standard
`integrator_callback!`, Gaussian reduction, and `transform_u!`) so that
`u_store` is in the axis basis — the only convention the PerturbedEquilibrium
/ FieldReconstruction downstream code is known to consume correctly.

We do save and restore the `intr.psilim` / `intr.qlim` / `intr.sing[*]` fields
that the parallel BVP populated, because the dense EL pass would otherwise
overwrite them (its standard `cross_ideal_singular_surf!` runs unconditionally
and does NOT populate `delta_prime`; we keep the parallel pass's values
which `compute_delta_prime_matrix!` uses).

Called from `parallel_eulerlagrange_integration` when
`ctrl.populate_dense_xi = true` (default).  Approximate cost: one serial
EL integration on top of the parallel BVP phase.  Required to make
`use_parallel = true` produce DCON eigenfunctions usable by the
PerturbedEquilibrium downstream pipeline.
"""
function _populate_dense_xi_via_serial_el!(
    odet::OdeState, ctrl::ForceFreeStatesControl,
    equil::Equilibrium.PlasmaEquilibrium, ffit::FourFitVars,
    intr::ForceFreeStatesInternal
)
    msing = intr.msing

    # Preserve parallel-BVP state on intr/odet that the serial-EL pass would otherwise
    # overwrite. PE downstream (SingularCoupling.jl) is calibrated against the (S, I)
    # Riccati gauge of `ca_l`/`ca_r`, so keeping the parallel-BVP values is critical.
    saved = (
        psilim    = intr.psilim,
        qlim      = intr.qlim,
        ca_l      = copy(odet.ca_l),
        ca_r      = copy(odet.ca_r),
        sing_state = [(
            delta_prime     = copy(intr.sing[s].delta_prime),
            delta_prime_col = copy(intr.sing[s].delta_prime_col),
            ua_left         = copy(intr.sing[s].ua_left),
            psi_ua_left     = intr.sing[s].psi_ua_left,
        ) for s in 1:msing],
    )

    # Temporarily switch dispatch flags so `eulerlagrange_integration`
    # follows the serial EL branch (axis-basis u_store) for this call.
    saved_use_parallel = ctrl.use_parallel
    saved_use_riccati  = ctrl.use_riccati
    saved_verbose      = ctrl.verbose
    ctrl.use_parallel = false
    ctrl.use_riccati  = false
    ctrl.verbose      = false  # suppress duplicate per-chunk logging

    if saved_verbose
        @info "   S → ξ: serial EL dense pass for HDF5 integration/xi_*"
    end

    local fresh_odet::OdeState
    try
        fresh_odet, _, _, _ = eulerlagrange_integration(ctrl, equil, ffit, intr)
    finally
        ctrl.use_parallel = saved_use_parallel
        ctrl.use_riccati  = saved_use_riccati
        ctrl.verbose      = saved_verbose
    end

    # Restore BVP-result fields on `intr`.
    intr.psilim = saved.psilim
    intr.qlim   = saved.qlim
    for s in 1:msing
        intr.sing[s].delta_prime     = saved.sing_state[s].delta_prime
        intr.sing[s].delta_prime_col = saved.sing_state[s].delta_prime_col
        intr.sing[s].ua_left         = saved.sing_state[s].ua_left
        intr.sing[s].psi_ua_left     = saved.sing_state[s].psi_ua_left
    end

    # Restore the parallel BVP's Riccati-gauge `ca_l` / `ca_r` onto the
    # fresh EL odet — these feed PE's `SingularCoupling.jl` which is
    # written against the (S, I) Riccati convention.
    fresh_odet.ca_l .= saved.ca_l
    fresh_odet.ca_r .= saved.ca_r

    # Return the fresh serial-EL odet (self-consistent for ξ-function
    # storage in axis basis; `ca_l`/`ca_r` carry the parallel-BVP
    # Riccati-gauge values needed by PE downstream).
    return fresh_odet
end
