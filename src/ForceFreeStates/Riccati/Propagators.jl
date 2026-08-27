# Chunk-propagator integration: EL-ODE fundamental matrices, Riccati renormalization, assembly.

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
        MatrixSplines,ForceFreeStatesInternal,OdeState,IntegrationChunk},
    psieval::Float64
)

    _, equil, mats, intr, odet, _ = params

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
    mats.ideal.F_spline_lower(vec(fmat_lower), psieval; hint=mats._hint)
    mats.ideal.K_spline(vec(kmat), psieval; hint=mats._hint)
    mats.ideal.G_spline(vec(gmat), psieval; hint=mats._hint)

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
    riccati_integrate_chunk!(odet, ctrl, equil, mats, intr, chunk)

Integrate the dual Riccati ODE from `chunk.psi_start` to `chunk.psi_end`.

Uses `sing_der!` as the ODE RHS with `riccati_integrator_callback!`, which applies
`renormalize_riccati_inplace!` (instead of Gaussian reduction) when norms exceed ucrit.
Starting state: u[:,:,1] = S_prev, u[:,:,2] = I (set by initialization or previous renorm).
Ending state: u[:,:,1] = U₁, u[:,:,2] = U₂ (ratio S = U₁·U₂⁻¹ is the updated Riccati matrix).
"""
function riccati_integrate_chunk!(
    odet::OdeState, ctrl::ForceFreeStatesControl, equil::Equilibrium.PlasmaEquilibrium,
    mats::MatrixSplines, intr::ForceFreeStatesInternal, chunk::IntegrationChunk
)
    cb = DiscreteCallback((u, t, integrator) -> true, riccati_integrator_callback!)
    rtol = ctrl.eulerlagrange_tolerance
    prob = ODEProblem(sing_der!, odet.u, (chunk.psi_start, chunk.psi_end),
                      (ctrl, equil, mats, intr, odet, chunk))
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
    integrate_propagator_chunk!(prop, chunk, ctrl, equil, mats, intr, odet_proxy)

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
    mats::MatrixSplines,
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
    params = (ctrl, equil, mats, intr, odet_proxy, chunk)

    # Upper block IC: U₁ = I, U₂ = 0
    u_upper = zeros(ComplexF64, N, N, 2)
    for i in 1:N
        u_upper[i, i, 1] = 1
    end
    odet_proxy.spline_hint[] = 1
    odet_proxy.mats_hint[] = 1
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
    odet_proxy.mats_hint[] = 1
    prob = ODEProblem(sing_der!, u_lower, tspan, params)
    sol = solve(prob, Vern9(); reltol=rtol, save_everystep=false, save_end=true)
    prop.block_lower_ic .= sol.u[end]
    odet_proxy.total_steps += sol.stats.naccept
end

"""
    integrate_fm_with_ua_ic(chunks, chunk_range, ua, ctrl, equil, mats, intr;
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
    mats::MatrixSplines,
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
    params = (ctrl, equil, mats, intr, odet_proxy, dummy_chunk)

    # Batch 1: columns 1:N of T (big solutions)
    u0 = zeros(ComplexF64, N, N, 2)
    u0[:, :, 1] .= ua[:, 1:N, 1]
    u0[:, :, 2] .= ua[:, 1:N, 2]
    # Per-column absolute tolerance so a batch's largest column cannot set the error floor of its smallest:
    # otherwise the resonant small-solution column inherits an absolute error set by the big solution's magnitude.
    abstol_arr = similar(u0, Float64)
    for j in 1:N
        abstol_arr[:, j, :] .= max(maximum(abs, @view u0[:, j, :]), 1e-30) * rtol
    end
    odet_proxy.spline_hint[] = 1
    odet_proxy.mats_hint[] = 1
    prob = ODEProblem(sing_der!, u0, tspan, params)
    sol = solve(prob, Vern9(); reltol=rtol, abstol=abstol_arr, save_everystep=false, save_end=true)
    result[1:N, 1:N]     .= sol.u[end][:, :, 1]
    result[N+1:2N, 1:N]  .= sol.u[end][:, :, 2]

    # Batch 2: columns N+1:2N of T (small solutions)
    u0[:, :, 1] .= ua[:, N+1:2N, 1]
    u0[:, :, 2] .= ua[:, N+1:2N, 2]
    for j in 1:N
        abstol_arr[:, j, :] .= max(maximum(abs, @view u0[:, j, :]), 1e-30) * rtol
    end
    odet_proxy.spline_hint[] = 1
    odet_proxy.mats_hint[] = 1
    prob = ODEProblem(sing_der!, u0, tspan, params)
    sol = solve(prob, Vern9(); reltol=rtol, abstol=abstol_arr, save_everystep=false, save_end=true)
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
