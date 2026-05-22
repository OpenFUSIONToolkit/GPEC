# DeltaPrime.jl — STRIDE inter-surface Δ' boundary-value problem and the
# per-surface Δ' diagnostic, factored out of the chunked-Riccati integration.

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
    N = size(propagators[1].block_upper_ic, 1)
    Phi = T_init !== nothing ? copy(T_init) : Matrix{ComplexF64}(I, 2N, 2N)
    isempty(idx_range) && return Phi
    for i in idx_range
        p = propagators[i]
        Phi_i = [p.block_upper_ic[:,:,1]  p.block_lower_ic[:,:,1];
                 p.block_upper_ic[:,:,2]  p.block_lower_ic[:,:,2]]
        Phi = Phi_i * Phi
        if condition
            condition_propagator!(Phi, N)
        end
    end
    return Phi
end

"""
    integrate_backward_chunk_fms(chunks, chunk_range, ctrl, equil, ffit, intr; T_init)

Compute backward per-chunk FMs by integrating the ODE backward within each chunk,
then chain them with ua initialization. Maps from surface → midpoint.

Matches Fortran STRIDE's approach: each interval near the singular surface is integrated
backward (`psiDirs=-1`), producing a backward FM that maps from right → left boundary.
These are chained to form the complete backward propagator.

This is more numerically stable than a single long backward ODE solve because each
per-chunk backward FM spans a short ψ range with moderate condition number.
"""
function integrate_backward_chunk_fms(
    chunks::Vector{IntegrationChunk},
    chunk_range::UnitRange{Int},
    ctrl::ForceFreeStatesControl,
    equil::Equilibrium.PlasmaEquilibrium,
    ffit::FourFitVars,
    intr::ForceFreeStatesInternal;
    T_init::Union{Nothing,Matrix{ComplexF64}}=nothing
)
    N = intr.numpert_total
    isempty(chunk_range) && return (T_init !== nothing ? copy(T_init) : Matrix{ComplexF64}(I, 2N, 2N))

    rtol = ctrl.eulerlagrange_tolerance
    odet_proxy = OdeState(N, 1, 1, 0)

    # Compute backward FM for each chunk in the range
    backward_fms = Vector{Matrix{ComplexF64}}(undef, length(chunk_range))
    for (idx, ic) in enumerate(chunk_range)
        c = chunks[ic]
        # Backward: integrate from psi_end to psi_start
        tspan = (c.psi_end, c.psi_start)
        dummy_chunk = IntegrationChunk(c.psi_start, c.psi_end, false, 0, -1)
        params = (ctrl, equil, ffit, intr, odet_proxy, dummy_chunk)

        fm = zeros(ComplexF64, 2N, 2N)
        # Integrate from identity ICs at psi_end → state at psi_start
        u0 = zeros(ComplexF64, N, N, 2)
        # Batch 1: columns 1:N (upper block IC = I, lower block = 0)
        for i in 1:N; u0[i, i, 1] = 1; end
        odet_proxy.spline_hint[] = 1
        prob = ODEProblem(sing_der!, u0, tspan, params)
        sol = solve(prob, Vern9(); reltol=rtol, save_everystep=false, save_end=true)
        fm[1:N, 1:N]     .= sol.u[end][:, :, 1]
        fm[N+1:2N, 1:N]  .= sol.u[end][:, :, 2]

        # Batch 2: columns N+1:2N (upper block = 0, lower block IC = I)
        fill!(u0, 0)
        for i in 1:N; u0[i, i, 2] = 1; end
        odet_proxy.spline_hint[] = 1
        prob = ODEProblem(sing_der!, u0, tspan, params)
        sol = solve(prob, Vern9(); reltol=rtol, save_everystep=false, save_end=true)
        fm[1:N, N+1:2N]     .= sol.u[end][:, :, 1]
        fm[N+1:2N, N+1:2N]  .= sol.u[end][:, :, 2]

        backward_fms[idx] = fm
    end

    # Chain backward FMs from surface toward midpoint.
    # Backward FM[i] maps state at chunk i psi_end → state at chunk i psi_start.
    # Chain: FM[start] * FM[start+1] * ... * FM[end] maps from end's psi_end to start's psi_start.
    # Iterate from the last chunk (surface) to the first (midpoint), pre-multiplying.
    Phi = T_init !== nothing ? copy(T_init) : Matrix{ComplexF64}(I, 2N, 2N)
    for idx in length(backward_fms):-1:1
        Phi = backward_fms[idx] * Phi
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
crashing.  The per-surface scalar Δ' values in `intr.sing[*].delta_prime`
(computed inline by `riccati_cross_ideal_singular_surf!` during chunk
crossings) are still populated and written to HDF5 in that case; only
`intr.delta_prime_matrix` (and HDF5 `singular/delta_prime_matrix`) is
omitted.  Generalizing the BVP to multi-resonance surfaces is tracked as a
follow-up: the matrix shape becomes `n_res_total × n_res_total` with
`n_res_total = sum(length(intr.sing[j].m))` and a `(surface, mode, side)`
↔ BVP-row map; see PR discussion.
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
    msing = intr.msing
    msing == 0 && return
    N = intr.numpert_total

    # Multi-resonance surfaces (one q satisfying multiple (m, n) tuples in a
    # multi-n run) are not yet handled by the inter-surface BVP.  Skip with a
    # warning rather than crashing the pipeline; per-surface Δ' values are
    # still populated upstream by `riccati_cross_ideal_singular_surf!` and
    # written to HDF5 under `singular/delta_prime` / `delta_prime_col`.
    n_res_per_surface = [length(intr.sing[j].m) for j in 1:msing]
    if any(>(1), n_res_per_surface)
        offenders = [(j, intr.sing[j].m, intr.sing[j].n) for j in 1:msing if n_res_per_surface[j] > 1]
        @warn "compute_delta_prime_matrix!: skipping inter-surface Δ' BVP because some surfaces carry more than one resonant mode " *
              "(multi-n collision; generalization tracked as follow-up). " *
              "Per-surface Δ' is unaffected. Multi-resonance surfaces: $offenders"
        return
    end

    i_crossings = findall(c -> c.needs_crossing, chunks)
    # Map from BVP surface index (1:msing_active) to intr.sing index.
    # Surfaces may be excluded at either end: below qlow (inner) or beyond psilim (outer).
    # Each crossing chunk records its original surface index in chunk.ising.
    sing_indices = [chunks[ic].ising for ic in i_crossings]
    msing_active = length(i_crossings)
    if msing_active < msing
        excluded = setdiff(1:msing, sing_indices)
        excluded_ms = [intr.sing[j].m for j in excluded]
        @debug "compute_delta_prime_matrix!: $msing singular surfaces, $msing_active crossed (excluded: m=$excluded_ms)"
        msing = msing_active
    end
    msing == 0 && return

    # Build a view into intr.sing that contains only the crossed surfaces.
    # All subsequent code uses `sing[j]` (local alias) instead of `intr.sing[j]`.
    sing = [intr.sing[si] for si in sing_indices]

    # Use S-based axis BC when Riccati S matrices are available (parallel FM path).
    # The S matrix at each surface's left boundary is always well-conditioned (bounded,
    # typically O(1)–O(10⁴)), avoiding the catastrophically ill-conditioned axis FM
    # (cond ~ 10²⁴) that makes the FM-based axis block rank-deficient.
    use_S_axis = S_at_surface_left !== nothing && length(S_at_surface_left) == msing

    # Assemble segment propagators.
    # Crossing chunks: single-chunk FMs at each surface (well-conditioned, backward-integrated)
    # Inter-surface segments: raw (unconditioned) multi-chunk FMs
    # Edge segment: raw multi-chunk FM
    # Axis segment: only assembled if S-based BC is NOT available (fallback)
    Phi_L_mats = [assemble_fm_matrix(propagators, i_crossings[j]:i_crossings[j]) for j in 1:msing]
    Phi_R_mats = Vector{Matrix{ComplexF64}}(undef, msing + 1)
    if !use_S_axis
        Phi_R_mats[1] = assemble_fm_matrix(propagators, 1:i_crossings[1]-1; condition=true)
    end
    for j in 2:msing
        Phi_R_mats[j] = assemble_fm_matrix(propagators, i_crossings[j-1]+1:i_crossings[j]-1)
    end
    Phi_R_mats[msing+1] = assemble_fm_matrix(propagators, i_crossings[msing]+1:length(chunks))

    # Midpoint shooting for inter-surface segments: split each gap at a midpoint,
    # producing two half-span propagators with cond ≈ √(full span cond). This is the
    # key STRIDE trick — by introducing midpoint unknowns in the BVP, each shooting
    # matrix covers half the distance, dramatically improving conditioning.
    # E.g., cond(full span) = 10¹⁵ → cond(half span) ≈ 10⁷·⁵ — 8 digits of accuracy.
    Phi_R_halves = Vector{Tuple{Matrix{ComplexF64}, Matrix{ComplexF64}}}(undef, msing - 1)
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
            # Only 1 chunk — can't split, use identity for left half
            Phi_R_halves[j] = (Matrix{ComplexF64}(I, 2N, 2N), Phi_R_mats[j+1])
        end
    end

    # Resonant mode index (1:N) for each surface
    ipert_all = [begin
        sp = sing[j]
        1 + sp.m[1] - intr.mlow + (sp.n[1] - intr.nlow) * intr.mpert
    end for j in 1:msing]

    # Asymptotic basis transformation: T = [ua[:,:,1]; ua[:,:,2]] maps asymptotic
    # (small/big) coefficients → raw (ξ,η) state. Column ordering of ua:
    #   columns 1:N = big solutions (z^{-α}, diverging),
    #   columns N+1:2N = small solutions (z^{+α}, bounded).
    # In asymptotic basis: component ipert = big soln coeff, ipert+N = small soln coeff.
    # Fortran STRIDE bakes T into the shooting propagators (uFM_sing_init);
    # here we multiply T into the BVP propagator blocks at each surface boundary.
    has_ua = all(j -> !isempty(sing[j].ua_left), 1:msing)

    if debug
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
        # Print per-surface Δ' from ca coefficients (diagonal reference)
        for j in 1:msing
            if !isempty(sing[j].delta_prime)
                @info "  Surface $j ca-based Δ' = $(@sprintf("%.6f%+.6fi", real(sing[j].delta_prime[1]), imag(sing[j].delta_prime[1])))"
            end
        end
    end

    # BVP structure depends on axis BC type.
    #
    # S-based axis BC (use_S_axis=true):
    #   Eliminates x_axis unknowns. The axis BC is u₁ = S₁·u₂ at surface 1 left boundary.
    #   nMat = (1 + 4·msing)·N
    #   Unknowns: x_left[j](2N), x_right[j](2N) for j=1..msing, x_edge(N)
    #
    # FM-based axis BC (use_S_axis=false, fallback):
    #   Uses conditioned axis propagator Phi_R[1][:,N+1:2N].
    #   nMat = (2 + 4·msing)·N
    #   Unknowns: x_axis(N), x_left[j](2N), x_right[j](2N), x_edge(N)
    s2 = 2 * msing

    # Column index helpers (used by both BVP paths and dp_raw extraction)
    col_left(j)  = N + 4N*(j-1) + 1 : N + 4N*(j-1) + 2N
    col_right(j) = N + 4N*(j-1) + 2N + 1 : N + 4N*j

    # Pre-compute T matrices: T = [ua[:,:,1]; ua[:,:,2]] maps asymptotic → raw.
    # Used by both S-based and FM-based BVP paths.
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

    if use_S_axis
        # STRIDE-style BVP with S-based axis BC.
        #
        # The Riccati S matrix at surface 1 left boundary encodes the axis BC
        # (U₁ = S·U₂) in a well-conditioned form (cond ~ 10⁶), eliminating the
        # catastrophically ill-conditioned axis propagator (cond ~ 10¹⁷+).
        #
        # Axis BC: T_left[1] maps asymptotic coefficients → raw (ξ,η) state.
        #   [ξ; η] = T·c  →  ξ = T₁·c,  η = T₂·c
        #   Axis regularity: ξ = S·η  →  (T₁ - S·T₂)·c = 0  (N equations)
        #
        # NOTE: The S-based BVP (nMat = (4*msing+1)*N = 288) has been replaced by
        # the Fortran-matched nMat = (2+4*msing)*N = 320 BVP below. The shooting
        # propagators (uShootR, uShootL, uAxis) built in this block are reused.

        # Build shooting propagators for inter-surface and edge segments.
        # Re-integrate with ua ICs for per-column accuracy (Fortran uFM_sing_init approach).
        can_reintegrate = has_ua && ctrl !== nothing && equil !== nothing && ffit !== nothing

        # Inter-surface shooting propagators meet at midpoints.
        # uShootR[j]: forward from surface j right → midpoint (ua_right IC at surface)
        # uShootL[j]: backward from surface j left → midpoint (ua_left IC at surface)
        # Only needed for j >= 2 (surface 1 uses S-based axis BC instead of uShootL).
        uShootR = Vector{Matrix{ComplexF64}}(undef, msing)
        uShootL = Vector{Matrix{ComplexF64}}(undef, msing)  # uShootL[1] unused with S axis BC

        for j in 1:msing
            # uShootR[j]: forward from surface j right
            if j < msing
                chunk_start = i_crossings[j] + 1
                chunk_end   = i_crossings[j+1] - 1
                n_inter = chunk_end - chunk_start + 1
                # Place midpoint at the ψ midpoint between surfaces (Fortran convention),
                # not at the chunk-index midpoint. Chunks near singularities are packed
                # tighter in ψ, so the index midpoint falls too close to the first surface.
                psi_mid_target = (chunks[chunk_start].psi_start + chunks[chunk_end].psi_end) / 2
                i_mid_inter = chunk_start
                for ic in chunk_start:chunk_end-1
                    if chunks[ic].psi_end >= psi_mid_target
                        i_mid_inter = ic
                        break
                    end
                    i_mid_inter = ic
                end
                shoot_range_R = chunk_start : i_mid_inter
            else
                shoot_range_R = i_crossings[msing]+1 : length(chunks)
            end
            if debug && !isempty(shoot_range_R)
                psi_surf_R = chunks[first(shoot_range_R)].psi_start
                psi_mid_R = chunks[last(shoot_range_R)].psi_end
                psi_ua_R = sing[j].psi_ua_right
                @info "    uShootR[$j]: shoot_range=$(shoot_range_R), psi_chunk=$(@sprintf("%.6f", psi_surf_R)), psi_ua=$(@sprintf("%.6f", psi_ua_R)), psi_mid=$(@sprintf("%.6f", psi_mid_R)), Δψ_fix=$(@sprintf("%.6e", psi_ua_R - psi_surf_R))"
            end
            if can_reintegrate && !isempty(shoot_range_R)
                uShootR[j] = integrate_fm_with_ua_ic(chunks, shoot_range_R,
                                sing[j].ua_right, ctrl, equil, ffit, intr;
                                backward=false, psi_ua=sing[j].psi_ua_right)
            else
                T_init = has_ua ? T_right_mats[j] : nothing
                uShootR[j] = assemble_fm_matrix(propagators, shoot_range_R; T_init=T_init)
            end

            # uShootL[j]: backward from surface j left (only needed for j >= 2)
            if j >= 2
                chunk_start = i_crossings[j-1] + 1
                chunk_end   = i_crossings[j] - 1
                n_inter = chunk_end - chunk_start + 1
                # Same ψ-midpoint logic as uShootR above
                psi_mid_target = (chunks[chunk_start].psi_start + chunks[chunk_end].psi_end) / 2
                i_mid_inter = chunk_start
                for ic in chunk_start:chunk_end-1
                    if chunks[ic].psi_end >= psi_mid_target
                        i_mid_inter = ic
                        break
                    end
                    i_mid_inter = ic
                end
                shoot_range_L = i_mid_inter+1 : chunk_end
                if debug
                    psi_mid = chunks[first(shoot_range_L)].psi_start
                    psi_surf = chunks[last(shoot_range_L)].psi_end
                    psi_ua_L = sing[j].psi_ua_left
                    @info "    uShootL[$j]: shoot_range=$(shoot_range_L), psi_mid=$(@sprintf("%.6f", psi_mid)), psi_chunk=$(@sprintf("%.6f", psi_surf)), psi_ua=$(@sprintf("%.6f", psi_ua_L)), Δψ_fix=$(@sprintf("%.6e", psi_ua_L - psi_surf))"
                end
                if can_reintegrate && !isempty(shoot_range_L)
                    uShootL[j] = integrate_fm_with_ua_ic(chunks, shoot_range_L,
                                    sing[j].ua_left, ctrl, equil, ffit, intr;
                                    backward=true, psi_ua=sing[j].psi_ua_left)
                else
                    T_init = has_ua ? T_left_mats[j] : nothing
                    uShootL[j] = assemble_fm_matrix(propagators, shoot_range_L; T_init=T_init)
                end
            end
        end

        if debug
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

            # Diagnostic: column norms of each shooting propagator
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

            # Diagnostic: midpoint matching submatrix conditioning
            for j in 1:msing-1
                # The midpoint block is [uShootR[j] | -uShootL[j+1]]
                mid_block = hcat(uShootR[j], -uShootL[j+1])
                @info "    Midpoint $j→$(j+1): cond([uShootR[$j] | -uShootL[$(j+1)]]) = $(@sprintf("%.2e", cond(mid_block)))"
                # Also show uShootL[j+1] column norms individually
                ipert_jp1 = ipert_all[j+1]
                col_norms_Ljp1 = [norm(view(uShootL[j+1], :, k)) for k in 1:2N]
                @info "    uShootL[$(j+1)] all col norms: $([(@sprintf("%.2e", c)) for c in col_norms_Ljp1])"
            end
        end

        # Build conditioned axis propagator (Fortran ode_fixup approach).
        # Start with lower-IC at axis: [0; I] (N regular solutions).
        # Forward-propagate through chunks 1..axis_mid, with QR fixup after each chunk.
        n_pre_cross = i_crossings[1] - 1  # chunks before first crossing
        # Place midpoint 1 chunk before the surface (Fortran: singMidPt = singIntervalL - 1).
        # The conditioned axis propagator covers most of the range; uShootL[1] covers
        # only the last chunk, keeping it well-conditioned.
        i_axis_mid = max(1, n_pre_cross - 1)
        uAxis = zeros(ComplexF64, 2N, N)
        for i in 1:N
            uAxis[N+i, i] = 1  # lower block = I (Fortran: q=0 at axis)
        end
        for ic in 1:i_axis_mid
            prop = propagators[ic]
            upper_old = uAxis[1:N, :]
            lower_old = uAxis[N+1:2N, :]
            uAxis[1:N, :]    .= prop.block_upper_ic[:,:,1] * upper_old .+ prop.block_lower_ic[:,:,1] * lower_old
            uAxis[N+1:2N, :] .= prop.block_upper_ic[:,:,2] * upper_old .+ prop.block_lower_ic[:,:,2] * lower_old
            # QR fixup: maintain orthogonal columns (Fortran: ode_fixup triangularization)
            Q, _ = qr(uAxis)
            uAxis .= Matrix(Q)[:, 1:N]
        end
        # Normalize columns
        for j in 1:N
            uAxis[:, j] ./= norm(@view uAxis[:, j])
        end

        # Build uShootL[1]: backward from surface 1 left to axis midpoint
        shoot_range_L1 = i_axis_mid+1 : i_crossings[1]-1
        if can_reintegrate && !isempty(shoot_range_L1)
            uShootL[1] = integrate_fm_with_ua_ic(chunks, shoot_range_L1,
                            sing[1].ua_left, ctrl, equil, ffit, intr;
                            backward=true, psi_ua=sing[1].psi_ua_left)
        elseif !isempty(shoot_range_L1)
            uShootL[1] = assemble_fm_matrix(propagators, shoot_range_L1;
                            T_init=has_ua ? T_left_mats[1] : nothing)
        else
            # Only 1 chunk before crossing, uShootL[1] = T (identity in asymptotic basis)
            uShootL[1] = has_ua ? T_left_mats[1] : Matrix{ComplexF64}(I, 2N, 2N)
        end

        if debug
            @info "  Axis propagator: $(i_axis_mid) chunks, cond=$(@sprintf("%.2e", cond(uAxis)))"
            @info "  uShootL[1]: range=$(shoot_range_L1), cond=$(@sprintf("%.2e", cond(uShootL[1])))"
        end

        # BVP assembly — Fortran-matched structure with nMat = (2 + 4*msing)*N = 320
        # Column layout: c_axis(N), c_left[1](2N), c_right[1](2N), ..., c_left[msing](2N), c_right[msing](2N), c_edge(N)
        nMat = (2 + 4 * msing) * N
        col_axis  = 1:N
        col_edge  = nMat - N + 1 : nMat
        M = zeros(ComplexF64, nMat, nMat)

        row_offset = 0

        # Axis matching: uShootL[1]*c_left[1] = uAxis*c_axis  (2N equations)
        # → uShootL[1]*c_left[1] - uAxis*c_axis = 0
        M[1:2N, col_left(1)] .= uShootL[1]
        M[1:2N, col_axis]    .= -uAxis
        row_offset = 2N

        for j in 1:msing
            ipert_j = ipert_all[j]

            # Crossing: non-resonant modes continuity (asymptotic basis = identity)
            for i in 1:2N
                if i != ipert_j && i != ipert_j + N
                    row_offset += 1
                    M[row_offset, col_left(j)[i]]  =  1
                    M[row_offset, col_right(j)[i]] = -1
                end
            end

            # Inter-surface or edge junction
            junc_start = row_offset + 1
            junc_end   = junc_start + 2N - 1
            junc_rows  = junc_start:junc_end
            if j < msing
                # Midpoint matching: uShootR[j] * x_right[j] = uShootL[j+1] * x_left[j+1]
                M[junc_rows, col_right(j)]  .= -uShootR[j]
                M[junc_rows, col_left(j+1)] .=  uShootL[j+1]
            else
                # Edge: uShootR[msing] * x_right = edge BC * x_edge
                M[junc_rows, col_right(msing)] .= uShootR[msing]
                if wv !== nothing
                    M[junc_rows[1:N],     col_edge] .= -I(N)
                    M[junc_rows[N+1:end], col_edge] .= wv .* psio^2
                else
                    M[junc_rows[N+1:end], col_edge] .= -I(N)
                end
            end
            row_offset = junc_end
        end

        # Driving: set big solution coefficient = 1 at each surface (asymptotic basis).
        for j in 1:msing
            ipert_j = ipert_all[j]
            row_offset += 1
            M[row_offset, col_left(j)[ipert_j]]  = 1
            row_offset += 1
            M[row_offset, col_right(j)[ipert_j]] = 1
        end

        @assert row_offset == nMat "Row count mismatch: expected $nMat, got $row_offset"

    else
        # Fallback: FM-based axis BC (original structure, rarely used)
        nMat = (2 + 4 * msing) * N
        col_axis = 1:N
        # Inline index calculations to avoid closure name collision with S-based branch
        M = zeros(ComplexF64, nMat, nMat)

        M[1:2N, (N+1):(N+2N)] .= Phi_L_mats[1]
        M[1:2N, col_axis]     .= -view(Phi_R_mats[1], :, N+1:2N)

        row_drive_base = 2N + (4N-2)*msing
        for j in 1:msing
            ipert_j = ipert_all[j]
            cl = (N + 4N*(j-1)+1) : (N + 4N*(j-1)+2N)   # col_left(j) inline
            cr = (N + 4N*(j-1)+2N+1) : (N + 4N*j)        # col_right(j) inline
            row_cont = 2N + (4N-2)*(j-1)
            for i in 1:2N
                if i != ipert_j && i != ipert_j + N
                    row_cont += 1
                    M[row_cont, cl[i]]  =  1
                    M[row_cont, cr[i]] = -1
                end
            end
            junc_rows = (row_cont+1) : (2N + (4N-2)*j)
            if j < msing
                cl_next = (N + 4N*j+1) : (N + 4N*j+2N)
                M[junc_rows, cr]     .= Phi_R_mats[j+1]
                M[junc_rows, cl_next] .= -Phi_L_mats[j+1]
            else
                ce = (N + 4N*msing+1) : nMat  # col_edge inline
                M[junc_rows, cr] .= Phi_R_mats[msing+1]
                if wv !== nothing
                    M[junc_rows[1:N],     ce] .= -I(N)
                    M[junc_rows[N+1:end], ce] .= wv .* psio^2
                else
                    M[junc_rows[N+1:end], ce] .= -I(N)
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
    end

    if debug
        @info "Δ' BVP: nMat=$nMat, rank(M)=$(rank(M)), cond(M)=$(@sprintf("%.2e", cond(M)))"
    end

    # Promote BVP matrix to Double64 for extended precision during the solve and
    # PEST3 combination. The PEST3 formula subtracts dp_raw entries that can be
    # 10,000-30,000× larger than the result; Double64 (~31 digits) preserves ~15
    # extra digits through this cancellation vs Float64 (~16 digits).
    use_d64 = ctrl !== nothing && ctrl.use_double64_bvp
    Tc = use_d64 ? Complex{Double64} : ComplexF64
    M_solve = use_d64 ? Tc.(M) : M

    # Solve the BVP for each driving configuration.
    M_lu = lu(M_solve; check=false)
    use_lu = issuccess(M_lu)
    M_pinv = use_lu ? nothing : pinv(M_solve)
    if !use_lu
        @warn "Δ' BVP: LU factorization singular (rank $(rank(M))/$nMat), using pseudo-inverse fallback"
    end
    dp_raw = zeros(Tc, s2, s2)
    b = zeros(Tc, nMat)

    for jsing in 1:msing
        for side in 1:2
            dRow = 2jsing - (2 - side)
            fill!(b, 0)
            if use_S_axis
                drive_row = nMat - s2 + dRow
            else
                drive_row = 2N + (4N-2)*msing + dRow
            end
            b[drive_row] = 1
            x = use_lu ? (M_lu \ b) : (M_pinv * b)

            if debug
                residual = norm(ComplexF64.(M_solve * x - b))
                side_str = side == 1 ? "left" : "right"
                @info "  BVP solve: jsing=$jsing side=$side_str (dRow=$dRow): ||Mx-b||=$(@sprintf("%.2e", residual)), ||x||=$(@sprintf("%.2e", Float64(norm(x))))"
                for ks in 1:msing
                    ipert_ks = ipert_all[ks]
                    xl_big   = ComplexF64(x[col_left(ks)[ipert_ks]])
                    xl_small = ComplexF64(x[col_left(ks)[ipert_ks+N]])
                    xr_big   = ComplexF64(x[col_right(ks)[ipert_ks]])
                    xr_small = ComplexF64(x[col_right(ks)[ipert_ks+N]])
                    @info "    surf $ks: x_left[big]=$(@sprintf("%+.4e%+.4ei", real(xl_big), imag(xl_big))), x_left[small]=$(@sprintf("%+.4e%+.4ei", real(xl_small), imag(xl_small)))"
                    @info "    surf $ks: x_right[big]=$(@sprintf("%+.4e%+.4ei", real(xr_big), imag(xr_big))), x_right[small]=$(@sprintf("%+.4e%+.4ei", real(xr_small), imag(xr_small)))"
                    @info "    surf $ks: ||x_left||=$(@sprintf("%.2e", Float64(norm(x[col_left(ks)])))), ||x_right||=$(@sprintf("%.2e", Float64(norm(x[col_right(ks)]))))"
                end
                if use_S_axis
                    @info "    ||x_edge||=$(@sprintf("%.2e", Float64(norm(x[col_edge]))))"
                end
            end

            for ksing in 1:msing
                ipert_k = ipert_all[ksing]
                dp_raw[dRow, 2ksing-1] = x[col_left(ksing)[ipert_k+N]]
                dp_raw[dRow, 2ksing]   = x[col_right(ksing)[ipert_k+N]]
            end
        end
    end

    # PEST3-convention Δ' in extended precision, then convert back to Float64
    deltap_ext = zeros(Tc, msing, msing)
    for i in 1:msing, j in 1:msing
        deltap_ext[i, j] = dp_raw[2i, 2j] - dp_raw[2i, 2j-1] - dp_raw[2i-1, 2j] + dp_raw[2i-1, 2j-1]
    end
    deltap = ComplexF64.(deltap_ext)

    if debug
        @info "Δ' BVP: Full dp_raw matrix ($(s2)×$(s2))$(use_d64 ? " [Double64]" : ""):"
        for i in 1:s2
            row_str = join([@sprintf("%+.6e", Float64(real(dp_raw[i,j]))) for j in 1:s2], "  ")
            @info "  dp_raw[$i,:] = $row_str"
        end
        @info "Δ' BVP: Raw dp diagonal = $([@sprintf("%.4f%+.4fi", Float64(real(dp_raw[i,i])), Float64(imag(dp_raw[i,i]))) for i in 1:s2])"
        @info "Δ' BVP: deltap diagonal = $([@sprintf("%.4f%+.4fi", real(deltap[i,i]), imag(deltap[i,i])) for i in 1:msing])"
    end

    intr.delta_prime_matrix = deltap
end

"""
    compute_delta_prime_from_ca!(odet, intr, equil)

Compute the tearing stability parameter Δ' for each singular surface from the
asymptotic coefficients `ca_l` and `ca_r` accumulated during integration.

Uses the diagonal formula Δ'[i] = (ca_r[i,i,2,s] - ca_l[i,i,2,s]) / (4π² · psio),
which is correct when the small asymptotic was introduced in column `ipert_res` directly
(no GR permutation).

**Note**: This function is no longer called from any integration driver. Δ' is now computed
inline inside each crossing function where the correct column index is known:
- `cross_ideal_singular_surf!` uses `perm_col` (GR-permuted column)
- `riccati_cross_ideal_singular_surf!` uses the diagonal `ipert_res` (no GR permutation)

Retained for reference and potential use in testing.

This matches the formula in `PerturbedEquilibrium/SingularCoupling.jl` (lines ~197):
  `delta_prime_val = (rbwp1 - lbwp1) / (twopi * chi1)`
with `chi1 = 2π·psio`, so the denominators are identical.
"""
function compute_delta_prime_from_ca!(odet::OdeState, intr::ForceFreeStatesInternal, equil::Equilibrium.PlasmaEquilibrium)
    denom = (2π)^2 * equil.psio  # = twopi * chi1 in SingularCoupling.jl
    for s in 1:intr.msing
        sing = intr.sing[s]
        n_modes = length(sing.m)
        resize!(intr.sing[s].delta_prime, n_modes)
        for i in 1:n_modes
            ipert_res = 1 + sing.m[i] - intr.mlow + (sing.n[i] - intr.nlow) * intr.mpert
            if 1 <= ipert_res <= intr.numpert_total
                Δca = odet.ca_r[ipert_res, ipert_res, 2, s] - odet.ca_l[ipert_res, ipert_res, 2, s]
                intr.sing[s].delta_prime[i] = Δca / denom
            else
                intr.sing[s].delta_prime[i] = 0.0 + 0.0im
            end
        end
    end
end

