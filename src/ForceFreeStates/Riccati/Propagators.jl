# Chunk-propagator integration: EL-ODE fundamental matrices, Riccati renormalization, assembly.

# `riccati_integrator_callback!` saves every step within this fraction of the span (or absolute ψ
# floor, for short chunks) of a segment's right end; elsewhere every `save_interval`-th step.
const SAVE_NEAR_END_FRAC = 0.05
const SAVE_NEAR_END_PSI  = 1e-4

"""
    assemble_fm_matrix(propagators, idx_range; condition=false, T_init=nothing) -> Matrix{ComplexF64}

Assemble the 2N×2N fundamental matrix Φ_end * ... * Φ_start * T_init over chunks `idx_range`,
mapping the IC at the start of `idx_range[1]` to the state at the end of `idx_range[end]`
(`T_init` defaults to the identity).

Each `ChunkPropagator` stores the 2N columns of Φ split into two N×N×2 blocks:
```
  block_upper_ic[:,:,1:2] ↔ Φ[:,1:N]     (result from IC=(I,0))
  block_lower_ic[:,:,1:2] ↔ Φ[:,N+1:2N]  (result from IC=(0,I))
```

`condition=true` applies `condition_propagator!` after each product (STRIDE's `ode_fixup`), so only
the N U₂ columns survive. Use it only for the axis→first-surface segment, where the axis BC (U₁ = 0)
needs only U₂ ICs; never for inter-surface segments, where both halves carry information.
"""
function assemble_fm_matrix(propagators::Vector{ChunkPropagator}, idx_range;
                            condition::Bool=false,
                            T_init::Union{Nothing,Matrix{ComplexF64}}=nothing)
    # Take N from T_init when given, so an empty idx_range (or propagator list) still works.
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

Reference implementation, not called in production: the quadratic S·Ḡ·S term makes the explicit
Riccati ODE unstable for explicit solvers, so the driver integrates `sing_der!` with
`renormalize_riccati_inplace!` instead. Exercised only by unit tests.

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

    # w = Q - K̄·S; Q is diagonal, so singfac_vec is added on the diagonal only.
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

Per-step callback for `riccati_integrate_chunk!`: counts the step, applies
`renormalize_riccati_inplace!` when max(|U₁|) or max(|U₂|) exceeds `ctrl.algorithm.renorm_threshold`
(the Riccati analogue of Gaussian reduction), refreshes the per-column abstol, and stores selected steps.
"""
function riccati_integrator_callback!(integrator)

    ctrl, _, _, intr, odet, chunk = integrator.p

    odet.total_steps += 1  # count every accepted solver step (saved or not), as segment_callback! does

    # Renormalize to (S, I) when either block exceeds renorm_threshold.
    if maximum(abs, @view(integrator.u[:, :, 1])) > ctrl.algorithm.renorm_threshold ||
       maximum(abs, @view(integrator.u[:, :, 2])) > ctrl.algorithm.renorm_threshold
        renormalize_riccati_inplace!(integrator.u, intr.numpert_total)
    end

    # Refresh after any renorm so the tolerance matches the current state.
    column_abstol!(integrator.opts.abstol, integrator.u, ctrl.algorithm.eulerlagrange_tolerance)

    # Save the first two steps, every step near the right end, and every save_interval-th step otherwise.
    psi_range = abs(integrator.sol.prob.tspan[2] - integrator.sol.prob.tspan[1])
    psi_remaining = abs(integrator.sol.prob.tspan[2] - integrator.t)
    near_end = psi_remaining < SAVE_NEAR_END_FRAC * psi_range || psi_remaining < SAVE_NEAR_END_PSI
    steps_in_segment = length(integrator.sol.t)
    near_start = steps_in_segment <= 2
    should_save = near_start || near_end || (odet.step % ctrl.algorithm.save_interval == 0)

    if should_save
        store_ode_data!(odet, integrator.t, integrator.u)
    end
end

"""
    riccati_integrate_chunk!(odet, ctrl, equil, mats, intr, chunk)

Integrate the Riccati state from `chunk.psi_start` to `chunk.psi_end` with `sing_der!` and
`riccati_integrator_callback!`, starting from u = (S, I). The end state is renormalized to (S, I)
unless `chunk.needs_crossing`, in which case the bounded (U₁, U₂) is left for the crossing.
"""
function riccati_integrate_chunk!(
    odet::OdeState, ctrl::ForceFreeStatesControl, equil::Equilibrium.PlasmaEquilibrium,
    mats::MatrixSplines, intr::ForceFreeStatesInternal, chunk::IntegrationChunk
)
    cb = DiscreteCallback((u, t, integrator) -> true, riccati_integrator_callback!)
    rtol = ctrl.algorithm.eulerlagrange_tolerance
    prob = ODEProblem(sing_der!, odet.u, (chunk.psi_start, chunk.psi_end),
                      (ctrl, equil, mats, intr, odet, chunk))
    sol = solve_el(prob, rtol; callback=cb)
    odet.u .= sol.u[end]
    odet.psifac = sol.t[end]
    # Before a crossing, skip the renorm so ca_l and ca_r share the pre-renorm normalization (needed for Δ').
    if !chunk.needs_crossing
        renormalize_riccati_inplace!(odet.u, intr.numpert_total)
    end
end

"""
    renormalize_riccati_inplace!(u, N)

In-place Riccati renormalization on an arbitrary N×N×2 array:
  u[:,:,1] = U₁ · U₂⁻¹  (new S)
  u[:,:,2] = I

The Riccati analogue of Gaussian reduction: keeps the inputs to `sing_der!` bounded.
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
    renormalize_riccati!(odet, intr)

[`renormalize_riccati_inplace!`](@ref) on `odet.u`. After a singular surface crossing, which leaves
u[:,:,2] = U₂_new ≠ I, it restores the canonical storage u[:,:,1] = S_new = U₁_new·U₂_new⁻¹, u[:,:,2] = I
before the integration continues. The crossing's u_store entry keeps U₁_new and U₂_new, so
`compute_smallest_eigenvalue` is unaffected.
"""
renormalize_riccati!(odet::OdeState, intr::ForceFreeStatesInternal) = renormalize_riccati_inplace!(odet.u, intr.numpert_total)

"""
    integrate_propagator_chunk!(prop, chunk, ctrl, equil, mats, intr, odet_proxy)

Fill `prop` with the chunk's fundamental matrix by integrating the EL ODE from IC (I, 0) into
`prop.block_upper_ic` and from (0, I) into `prop.block_lower_ic`, without renormalization or storage.
`odet_proxy` holds the `sing_der!` side effects; concurrent calls need distinct proxies.
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
    # Backward chunks integrate psi_end → psi_start, where forward-growing solutions decay.
    tspan = chunk.direction == 1 ?
        (chunk.psi_start, chunk.psi_end) :
        (chunk.psi_end,   chunk.psi_start)
    rtol = ctrl.algorithm.eulerlagrange_tolerance
    params = (ctrl, equil, mats, intr, odet_proxy, chunk)

    # Upper block IC: U₁ = I, U₂ = 0
    u_upper = zeros(ComplexF64, N, N, 2)
    for i in 1:N
        u_upper[i, i, 1] = 1
    end
    odet_proxy.spline_hint[] = 1
    odet_proxy.mats_hint[] = 1
    prob = ODEProblem(sing_der!, u_upper, tspan, params)
    sol = solve_el(prob, rtol)
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
    sol = solve_el(prob, rtol)
    prop.block_lower_ic .= sol.u[end]
    odet_proxy.total_steps += sol.stats.naccept
end

"""
    integrate_fm_with_ua_ic(chunks, chunk_range, ua, ctrl, equil, mats, intr;
                            backward=false, psi_ua=NaN) -> Matrix{ComplexF64}

Re-integrate a span of chunks from the asymptotic solutions `ua` as initial conditions (Fortran
STRIDE `uFM_sing_init`). Column j of the 2N×2N result is the solution at the far end with IC
column j of T = [ua[:,:,1]; ua[:,:,2]]. With `backward=true` the IC sits at the span's right end
and the integration runs toward its start. `psi_ua`, when given, replaces the chunk boundary as the
IC location. Integrating from `ua` keeps the small solutions that multiplying an identity-IC
propagator by T loses to roundoff.
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
    # ua lives at the inner-layer boundary psi_ua, which may differ slightly from the chunk boundary.
    if backward && !isnan(psi_ua)
        psi_end = psi_ua
    elseif !backward && !isnan(psi_ua)
        psi_start = psi_ua
    end
    tspan = backward ? (psi_end, psi_start) : (psi_start, psi_end)
    rtol = ctrl.algorithm.eulerlagrange_tolerance

    result = zeros(ComplexF64, 2N, 2N)
    odet_proxy = OdeState(N, 1, 1, 0)
    dummy_chunk = IntegrationChunk(psi_start, psi_end, false, 0, backward ? -1 : 1)
    params = (ctrl, equil, mats, intr, odet_proxy, dummy_chunk)

    # Batch 1: columns 1:N of T (big solutions)
    u0 = zeros(ComplexF64, N, N, 2)
    u0[:, :, 1] .= ua[:, 1:N, 1]
    u0[:, :, 2] .= ua[:, 1:N, 2]
    odet_proxy.spline_hint[] = 1
    odet_proxy.mats_hint[] = 1
    prob = ODEProblem(sing_der!, u0, tspan, params)
    sol = solve_el(prob, rtol)
    result[1:N, 1:N]     .= sol.u[end][:, :, 1]
    result[N+1:2N, 1:N]  .= sol.u[end][:, :, 2]

    # Batch 2: columns N+1:2N of T (small solutions)
    u0[:, :, 1] .= ua[:, N+1:2N, 1]
    u0[:, :, 2] .= ua[:, N+1:2N, 2]
    odet_proxy.spline_hint[] = 1
    odet_proxy.mats_hint[] = 1
    prob = ODEProblem(sing_der!, u0, tspan, params)
    sol = solve_el(prob, rtol)
    result[1:N, N+1:2N]     .= sol.u[end][:, :, 1]
    result[N+1:2N, N+1:2N]  .= sol.u[end][:, :, 2]

    return result
end

"""
    apply_propagator!(odet, prop)

Apply the chunk propagator `prop` to any state `odet.u` (including the Riccati (S, I) form) in-place.
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

Used for backward chunks: the well-conditioned Φ_bwd maps `psi_end` → `psi_start`, so solving
Φ_bwd · x = u_old by LU advances the state `psi_start` → `psi_end` far more accurately than Φ_fwd.

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
    # x = Φ_bwd⁻¹ · [U₁_old; U₂_old] advances psi_start → psi_end.
    u_old = [odet.u[:,:,1]; odet.u[:,:,2]]   # 2N × N
    u_new = Φ \ u_old                         # LU solve, 2N × N
    odet.u[:,:,1] .= u_new[1:N, :]
    odet.u[:,:,2] .= u_new[N+1:2N, :]
end
