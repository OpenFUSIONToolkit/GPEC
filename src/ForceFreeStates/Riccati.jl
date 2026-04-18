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
term `-SGS` dominates and the RHS grows as |S|². Explicit adaptive solvers (Tsit5) use
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
- Tsit5 integrates (U₁, U₂) to **5th-order accuracy** with the adaptive step-size
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
it does not imply the integration is first-order. In practice Tsit5 captures all higher-order
terms through its internal stages, achieving 5th-order global accuracy at the configured reltol.

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
  - Axis init:   S(ψ₀) = 0  (initialize_el_at_axis! sets u[:,:,1]=0, u[:,:,2]=I)  ✓

## Key Differences from Standard Integration

1. `sing_der!` is used as the ODE RHS (same as standard, NOT `riccati_der!`)
2. `riccati_integrator_callback!` replaces `integrator_callback!`: uses
   `renormalize_riccati_inplace!` instead of Gaussian reduction
3. `riccati_cross_ideal_singular_surf!` replaces `cross_ideal_singular_surf!`: skips Gaussian
   reduction and uses ipert_res directly for column zeroing, then renormalizes to (S_new, I)
4. `transform_u!` is skipped — S is already the true solution
"""

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
multiplication step, following STRIDE's `ode_fixup` convention [ode.F:800-808]. This
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
- Assumes exactly one resonant mode per singular surface (standard single-n case).
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

    @assert all(j -> length(intr.sing[j].m) == 1, 1:msing) "compute_delta_prime_matrix! only supports single-resonance surfaces"

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
    riccati_der!(du, u, params, psieval)

Evaluate the explicit dual Riccati ODE right-hand side:
  dS/dψ = w†·F̄⁻¹·w - S·Ḡ·S,   w = Q - K̄·S

where Q = diag(1/(m - n·q)) is the diagonal singular factor matrix.
The identity slice u[:,:,2] = I does not evolve (du[:,:,2] = 0).

**NOTE**: This function is NOT used as the ODE RHS in `riccati_integrate_chunk!`.
The explicit Riccati ODE is numerically unstable for explicit solvers: the quadratic
term S·Ḡ·S causes finite-time blowup when K̄·S >> Q. Instead, `sing_der!` is used
with periodic renormalization via `renormalize_riccati_inplace!`. This function is
retained for reference and potential use with implicit solvers.

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

    # Store du1/dψ = Q·v for ud diagnostic before v is reused
    # Q·v = diag(singfac_vec)·v = Ξ'_Ψ (displacement gradient, with U₂ = I)
    @. odet.ud[:, :, 1] = singfac_vec * v
    @view(odet.ud[:, :, 2]) .= 0

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

    # Use unified tolerance (matches integrate_el_region! on develop)
    integrator.opts.reltol = ctrl.eulerlagrange_tolerance

    # Renormalize when norms exceed ucrit (analogous to Gaussian reduction in integrator_callback!)
    # During sing_der! integration: u[:,:,1]=U₁ (grows), u[:,:,2]=U₂ (grows).
    # Renorm computes S = U₁·U₂⁻¹ and resets U₂ = I, keeping inputs bounded.
    if maximum(abs, @view(integrator.u[:, :, 1])) > ctrl.ucrit ||
       maximum(abs, @view(integrator.u[:, :, 2])) > ctrl.ucrit
        renormalize_riccati_inplace!(integrator.u, intr.numpert_total)
    end

    # Determine if we should save this step
    psi_range = abs(integrator.sol.prob.tspan[2] - integrator.sol.prob.tspan[1])
    psi_remaining = abs(integrator.sol.prob.tspan[2] - integrator.t)
    near_end = psi_remaining < 0.05 * psi_range || psi_remaining < 1e-4
    steps_in_segment = length(integrator.sol.t)
    near_start = steps_in_segment <= 2
    should_save = near_start || near_end || (odet.step % ctrl.save_interval == 0)

    if should_save
        if odet.step >= size(odet.u_store, 4)
            resize_storage!(odet)
        end
        odet.psi_store[odet.step] = integrator.t
        @views odet.u_store[:, :, :, odet.step] .= integrator.u
        odet.q_store[odet.step] = odet.q
        @views odet.ud_store[:, :, :, odet.step] .= odet.ud
        odet.step += 1
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
    # Skip Gaussian reduction — S is bounded so no large-norm columns exist

    singp = intr.sing[ising]
    dpsi = singp.psifac - odet.psifac  # ψ_res - ψ_current (positive)

    # Compute separate left-side (sig=-1) and right-side (sig=+1) asymptotics,
    # matching Fortran's separate vmatl/vmatr [sing.F: sing_vmat].
    # Alpha is computed from the right-side m0mat and shared with the left side.
    sing_asymp_right = compute_sing_asymptotics(singp, ctrl, equil, ffit, intr; sig=1.0)
    sing_asymp_left = compute_sing_asymptotics(singp, ctrl, equil, ffit, intr; sig=-1.0, alpha_override=sing_asymp_right.alpha)

    # Asymptotic-quantity diagnostics (gated behind ctrl.verbose so they don't
    # fire on every crossing).
    if ctrl.verbose
        ipert_res_diag = 1 .+ singp.m .- intr.mlow .+ (singp.n .- intr.nlow) .* intr.mpert
        @info "  ising=$ising: psi_sing=$(@sprintf("%.10f", singp.psifac)), psi_eval=$(@sprintf("%.10f", odet.psifac)), dpsi=$(@sprintf("%.10e", dpsi))"
        @info "  alpha_L = $(sing_asymp_left.alpha), alpha_R = $(sing_asymp_right.alpha)"
        for ip in ipert_res_diag
            @info "  vmatL[0] big: vmat[$ip,$ip,1,1]=$(@sprintf("%.8e", real(sing_asymp_left.vmat[ip,ip,1,1]))), vmat[$ip,$ip,2,1]=$(@sprintf("%.8e", real(sing_asymp_left.vmat[ip,ip,2,1])))"
            @info "  vmatR[0] big: vmat[$ip,$ip,1,1]=$(@sprintf("%.8e", real(sing_asymp_right.vmat[ip,ip,1,1]))), vmat[$ip,$ip,2,1]=$(@sprintf("%.8e", real(sing_asymp_right.vmat[ip,ip,2,1])))"
        end
    end

    # Get asymptotic coefficients before crossing (LEFT side); save ua for Δ' BVP
    # sing_get_ua now takes positive dpsi and uses the direction-specific asymptotics
    ua = sing_get_ua(sing_asymp_left, dpsi)
    singp.ua_left = copy(ua)
    singp.psi_ua_left = odet.psifac
    odet.ca_l[:, :, :, ising] .= sing_get_ca(odet.u, ua, intr)

    # Resonant perturbation indices (same formula as in cross_ideal_singular_surf!)
    ipert_res = 1 .+ singp.m .- intr.mlow .+ (singp.n .- intr.nlow) .* intr.mpert

    if !ctrl.con_flag
        # Zero the resonant column of (S, I) using ipert_res directly (no GR sorting needed).
        # The zeroed column stays zero through the predictor step since both slices are zero.
        for i in eachindex(sing_asymp_right.r1)
            odet.u[:, ipert_res[i], :] .= 0
        end
    end

    # Predictor: approximate solution on the other side of the singular surface.
    # sing_der! works on any (U1, U2) state — the zeroed column remains zero since
    # du1[:, ipert_res] = 0 and du2[:, ipert_res] = 0 when u[:, ipert_res, :] = 0.
    params = (ctrl, equil, ffit, intr, odet, IntegrationChunk(0.0, 0.0, false, ising, 1))
    du1 = zeros(ComplexF64, intr.numpert_total, intr.numpert_total, 2)
    du2 = zeros(ComplexF64, intr.numpert_total, intr.numpert_total, 2)
    sing_der!(du1, odet.u, params, odet.psifac)
    odet.psifac += 2 * dpsi  # jump to other side of singular surface
    sing_der!(du2, odet.u, params, odet.psifac)
    odet.u .+= (du1 .+ du2) .* dpsi

    # Apply asymptotic solution on other side of singular surface; save ua for Δ' BVP
    ua = sing_get_ua(sing_asymp_right, dpsi)
    singp.ua_right = copy(ua)
    singp.psi_ua_right = odet.psifac  # ψ where ua_right is evaluated (right inner-layer boundary)
    if !ctrl.con_flag
        for i in eachindex(sing_asymp_right.r1)
            # Zero the resonant row (removes large components at the resonant mode)
            odet.u[ipert_res[i], :, :] .= 0
            # Introduce the small asymptotic resonant solution in the zeroed column.
            # ua[:, ipert_res[i]+numpert_total, :] is the "lower" (small) solution for mode ipert_res[i].
            # After this, u[:,:,2] = U₂_new ≠ I (has asymptotic in column ipert_res[i]);
            # renormalize_riccati! will compute S_new = U₁_new · U₂_new⁻¹ and reset U₂ = I.
            odet.u[:, ipert_res[i], :] .= ua[:, ipert_res[i]+intr.numpert_total, :]
        end
    end
    # Compute ca_r from (U₁_new, U₂_new) before renormalization.
    # Column ipert_res of [U₁_new; U₂_new] = ua[:,ipert_res+N,:] (the introduced small asymptotic),
    # so ca_r[:,ipert_res] = e_{ipert_res+N} and ca_r[ipert_res,ipert_res,2] = 1 regardless of
    # the normalization of the other columns. This gives Δ' = 1 - ca_l[ipert_res,ipert_res,2].
    odet.ca_r[:, :, :, ising] .= sing_get_ca(odet.u, ua, intr)

    # Compute Δ' using ipert_res directly (no GR → perm_col = ipert_res, ca_r diagonal = 1).
    # Also compute the full column Δ' (all N modes) for the off-diagonal coupling.
    if !ctrl.con_flag
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

    # Store (U₁_new, U₂_new) before renormalization so evaluate_stability_criterion!
    # can recover S_new = U₁_new / U₂_new correctly via compute_smallest_eigenvalue
    odet.psi_store[odet.step] = odet.psifac
    odet.q_store[odet.step] = odet.q
    odet.u_store[:, :, :, odet.step] = odet.u
    odet.ud_store[:, :, :, odet.step] = odet.ud
    odet.step += 1

    # Renormalize to Riccati convention: S_new = U₁_new · U₂_new⁻¹, reset U₂ = I
    renormalize_riccati!(odet, intr)
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
        initialize_el_at_axis!(odet, ctrl, equil.profiles, intr)
        # axis init sets u[:,:,1]=0, u[:,:,2]=I → S=0 at axis ✓
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
            if ctrl.kin_flag
                error("kin_flag = true not implemented yet!")
            else
                riccati_cross_ideal_singular_surf!(odet, ctrl, equil, ffit, intr, chunk.ising)
                # renormalize_riccati! is called inside riccati_cross_ideal_singular_surf!
            end
        end
    end

    # Find peak dW in edge region if applicable (uses free_compute_total which reads wp = I/S = P)
    if ctrl.psiedge < intr.psilim
        odet.step = findmax_dW_edge!(odet, ctrl, equil, ffit, intr)
        trim_storage!(odet)
        if ctrl.verbose
            @info "Truncating integration at peak edge dW: ψ = $((@sprintf "%.2f" odet.psi_store[odet.step])),  q = $((@sprintf "%.2f" odet.q_store[odet.step]))"
        end
        intr.psilim = odet.psi_store[end]
        intr.qlim = odet.q_store[end]
        odet.u .= odet.u_store[:, :, :, end]
    else
        odet.step -= 1
        trim_storage!(odet)
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
    prob = ODEProblem(sing_der!, u_upper, tspan, params)
    sol = solve(prob, Vern9(); reltol=rtol, save_everystep=false, save_end=true)
    prop.block_upper_ic .= sol.u[end]

    # Lower block IC: U₁ = 0, U₂ = I
    u_lower = zeros(ComplexF64, N, N, 2)
    for i in 1:N
        u_lower[i, i, 2] = 1
    end
    odet_proxy.spline_hint[] = 1
    prob = ODEProblem(sing_der!, u_lower, tspan, params)
    sol = solve(prob, Vern9(); reltol=rtol, save_everystep=false, save_end=true)
    prop.block_lower_ic .= sol.u[end]
end

"""
    integrate_fm_with_ua_ic(chunks, chunk_range, ua, ctrl, equil, ffit, intr;
                            backward=false) -> Matrix{ComplexF64}

Re-integrate a span of chunks using ua (asymptotic solution) as initial conditions, matching
Fortran STRIDE's uFM_sing_init behavior [ode.F:374-402]. Returns a 2N×2N fundamental matrix
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
    prob = ODEProblem(sing_der!, u0, tspan, params)
    sol = solve(prob, Vern9(); reltol=rtol, save_everystep=false, save_end=true)
    result[1:N, 1:N]     .= sol.u[end][:, :, 1]
    result[N+1:2N, 1:N]  .= sol.u[end][:, :, 2]

    # Batch 2: columns N+1:2N of T (small solutions)
    u0[:, :, 1] .= ua[:, N+1:2N, 1]
    u0[:, :, 2] .= ua[:, N+1:2N, 2]
    odet_proxy.spline_hint[] = 1
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
"""
function apply_propagator_inverse!(odet::OdeState, prop::ChunkPropagator)
    N = size(odet.u, 1)
    # Assemble 2N×2N backward FM Φ_bwd
    Φ = [prop.block_upper_ic[:,:,1] prop.block_lower_ic[:,:,1];
         prop.block_upper_ic[:,:,2] prop.block_lower_ic[:,:,2]]
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
- No Gaussian reduction (crossings use riccati-style, odet.ifix stays 0)
- `transform_u!` is called but is a no-op (identity transform, ifix=0)
- `ud_store` is approximate (set to zeros for FM chunks; does not affect energies or Δ')
- Outer plasma uses serial Riccati integration for numerical stability

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
    # Initialization — same as eulerlagrange_integration
    odet = OdeState(intr.numpert_total, ctrl.numsteps_init, ctrl.numunorms_init, intr.msing)
    if ctrl.sing_start <= 0
        initialize_el_at_axis!(odet, ctrl, equil.profiles, intr)
    elseif ctrl.sing_start <= intr.msing
        error("sing_start > 0 not implemented yet!")
    else
        error("Invalid value for sing_start: $(ctrl.sing_start) > msing = $(intr.msing)")
    end

    # Prime odet.new = false (consistent with riccati path — no Gaussian reduction used)
    odet.new = false
    fill!(odet.unorm0, 1.0)

    # Build chunks and sub-divide for load-balanced parallel execution.
    # bidirectional=true: crossing chunks (nearest to each rational surface) are assigned
    # direction=-1, so they are integrated backward. The resulting backward propagator
    # Φ_bwd is well-conditioned because growing EL solutions decay backward. The forward
    # propagation is recovered as Φ_bwd⁻¹ via LU solve in apply_propagator_inverse!.
    base_chunks = chunk_el_integration_bounds(odet, ctrl, intr; bidirectional=true)
    chunks = balance_integration_chunks(base_chunks, ctrl, intr)

    N = intr.numpert_total
    propagators = [ChunkPropagator(N) for _ in chunks]

    # Per-thread lightweight proxy OdeState for sing_der! side effects
    nthreads = Threads.nthreads()
    odet_proxies = [OdeState(N, 1, 1, 0) for _ in 1:nthreads]

    if ctrl.verbose
        @info "   ψ = $((@sprintf "%.3f" odet.psifac)),  q = $((@sprintf "%.3f" equil.profiles.q_spline(odet.psifac)))"
        @info "   Parallel FM: $(length(chunks)) chunks, $nthreads threads"
    end

    # PARALLEL phase: integrate all chunks independently from identity IC.
    # :static scheduler pins each task to one OS thread for its lifetime, so
    # Threads.threadid() returns a stable index into odet_proxies.
    # Without :static, Julia's task scheduler can migrate tasks between threads,
    # making threadid() unreliable (Julia 1.7+).
    Threads.@threads :static for i in eachindex(chunks)
        integrate_propagator_chunk!(propagators[i], chunks[i], ctrl, equil, ffit, intr,
                                    odet_proxies[Threads.threadid()])
    end

    # SERIAL assembly: apply propagators and handle crossings in order.
    # After each apply_propagator!, renormalize to (S, I) form. This is the Julia
    # equivalent of STRIDE's ode_fixup: it prevents exponential growth of the
    # accumulated state between crossings. Without this renorm, products of N chunk
    # FMs can have condition numbers up to (cond_per_chunk)^N, causing catastrophic
    # cancellation for large N (N ≳ 20). With renorm, each chunk is applied as a
    # Möbius transformation on the bounded S matrix, keeping errors at O(eps × cond_chunk)
    # rather than O(eps × cond_chunk^N). [STRIDE ode.F: ode_fixup called after each uAxis step]
    #
    # S_at_surface_left: save the Riccati matrix S = U₁·U₂⁻¹ at the left boundary
    # of each singular surface (just before crossing). These well-conditioned matrices
    # (bounded, typically O(1)-O(10⁴)) encode the axis BC for the Δ' BVP without
    # needing the catastrophically ill-conditioned axis fundamental matrix.
    #
    # last_crossing_step tracks the u_store index of the most recent crossing so that
    # the outer plasma (from last rational surface to psilim) can be re-integrated.
    S_at_surface_left = Matrix{ComplexF64}[]
    last_crossing_step = 1
    for (i, chunk) in enumerate(chunks)
        # Forward chunks: apply propagator directly (Φ_fwd maps psi_start → psi_end).
        # Backward chunks (crossing chunks with direction=-1): apply inverse of the
        # backward propagator. Φ_bwd maps psi_end → psi_start and is well-conditioned;
        # its inverse Φ_fwd = Φ_bwd⁻¹ gives accurate forward propagation via LU solve.
        if chunk.direction == -1
            apply_propagator_inverse!(odet, propagators[i])
        else
            apply_propagator!(odet, propagators[i])
        end
        # Renorm to (S, I) after every chunk — equivalent to STRIDE's ode_fixup.
        # The state entering each crossing is already in (S, I) form.
        renormalize_riccati_inplace!(odet.u, N)
        odet.psifac = chunk.psi_end
        odet.q = equil.profiles.q_spline(odet.psifac)

        if ctrl.verbose
            @info "   ψ = $((@sprintf "%.3f" odet.psifac)),  q= $((@sprintf "%.3f" odet.q)),  max(S) = $((@sprintf "%.2e" maximum(abs, odet.u[:,:,1]))),  steps = $(odet.step-1)"
        end

        if chunk.needs_crossing
            if ctrl.kin_flag
                error("kin_flag = true not implemented yet!")
            else
                # Save S at left boundary of this surface (before crossing).
                # State is (S, I) from the renorm above; S is well-conditioned.
                push!(S_at_surface_left, copy(odet.u[:, :, 1]))

                # riccati_cross_ideal_singular_surf! zeros column ipert_res directly
                # (the resonant mode, no GR permutation needed in Riccati form).
                riccati_cross_ideal_singular_surf!(odet, ctrl, equil, ffit, intr, chunk.ising)
                last_crossing_step = odet.step - 1  # u_store index of the crossing state
            end
        else
            # Save non-crossing end-of-chunk state (now always in (S, I) form)
            if odet.step >= size(odet.u_store, 4)
                resize_storage!(odet)
            end
            odet.psi_store[odet.step] = odet.psifac
            odet.q_store[odet.step] = odet.q
            @views odet.u_store[:, :, :, odet.step] .= odet.u
            # ud not available from propagator integration — left as zeros
            odet.step += 1
        end
    end

    # Re-integrate the outer plasma (from last rational surface crossing to psilim) using
    # Riccati for numerical stability and dense checkpoint storage.
    #
    # FM propagation in the outer plasma (no rational surfaces) is prone to precision loss
    # for high N: the solution grows exponentially without renormalization, causing matrix
    # condition numbers to grow and wp = U₂·U₁⁻¹ to lose accuracy. Riccati integration
    # keeps matrices bounded via periodic renormalization.
    #
    # Dense checkpoints from this re-integration are also required for findmax_dW_edge! to
    # accurately locate the peak dW in the edge region (psiedge < psilim case).
    #
    # The u_store entry at last_crossing_step contains (U₁_new, U₂_new) stored by
    # riccati_cross_ideal_singular_surf! before renormalization; renormalizing here gives
    # (S_new, I) as the correct Riccati starting state for the re-integration.
    odet.u .= odet.u_store[:, :, :, last_crossing_step]
    odet.psifac = odet.psi_store[last_crossing_step]
    odet.q = odet.q_store[last_crossing_step]
    odet.step = last_crossing_step + 1
    renormalize_riccati_inplace!(odet.u, N)
    outer_chunk = IntegrationChunk(; psi_start=odet.psifac, psi_end=intr.psilim * (1 - eps),
                                     needs_crossing=false, ising=0)
    riccati_integrate_chunk!(odet, ctrl, equil, ffit, intr, outer_chunk)
    # After riccati_integrate_chunk! with needs_crossing=false:
    #   odet.u is in (S, I) form (renorm'd at end of integration)
    #   odet.step points to next empty slot; dense checkpoints stored for outer region

    # Find peak dW in edge region (same as standard/Riccati path)
    if ctrl.psiedge < intr.psilim
        odet.step = findmax_dW_edge!(odet, ctrl, equil, ffit, intr)
        trim_storage!(odet)
        if ctrl.verbose
            @info "Truncating integration at peak edge dW: ψ = $((@sprintf "%.2f" odet.psi_store[odet.step])),  q = $((@sprintf "%.2f" odet.q_store[odet.step]))"
        end
        intr.psilim = odet.psi_store[end]
        intr.qlim = odet.q_store[end]
        odet.u .= odet.u_store[:, :, :, end]
        # The stored state may be a pre-renorm callback snapshot; renorm to (S, I) for free_run!
        renormalize_riccati_inplace!(odet.u, N)
    else
        odet.step -= 1
        trim_storage!(odet)
        # odet.u is already in (S, I) from riccati_integrate_chunk! above
    end

    # NOTE: compute_delta_prime_matrix! is called from the main pipeline (after free_run!)
    # so that vacuum response wv is available for the edge BC. The propagators and chunks
    # are returned alongside odet for this purpose.

    # Evaluate fixed-boundary stability criterion
    if ctrl.verbose
        @info "Evaluating fixed-boundary stability criterion"
    end
    odet.nzero = evaluate_stability_criterion!(odet, equil.profiles)

    # transform_u! is called for consistency but is a no-op (ifix=0, no Gaussian reduction)
    transform_u!(odet, intr)

    return odet, propagators, chunks, S_at_surface_left
end
