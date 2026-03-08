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

The explicit Riccati ODE (`riccati_der!`) is mathematically correct but numerically unstable
for explicit solvers: the RHS is quadratic in S, so if S grows large (K̄·S >> Q), the
quadratic term (K̄·S)²/F̄ causes finite-time blowup that the adaptive step-size controller
cannot prevent (relative error control allows large absolute errors when |S| is large).

Instead, the Riccati integration uses `sing_der!` (the standard EL ODE) with periodic
renormalization. Starting each chunk with U₁ = S_prev, U₂ = I:

  After a step Δψ: U₁_new ≈ S + (A·S + B)·Δψ,  U₂_new ≈ I + (C·S + D)·Δψ
  Renorm: S_new = U₁_new · U₂_new⁻¹ ≈ S + (B + A·S - S·D - S·C·S)·Δψ  ✓

This is numerically stable because U₁ and U₂ track each other — their ratio stays bounded
even as each individually grows large. Renormalization is triggered by
`renormalize_riccati_inplace!` in the callback when max(|U₁|) or max(|U₂|) exceeds ucrit,
exactly analogous to Gaussian reduction in the standard ODE.

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
    assemble_fm_matrix(propagators, idx_range) -> Matrix{ComplexF64}

Assemble the 2N×2N fundamental matrix (propagator) by multiplying chunk propagators
in order for indices `idx_range`. Returns Φ_end * ... * Φ_start, so that the result
maps the IC at the start of `idx_range[1]` to the state at the end of `idx_range[end]`.

Each `ChunkPropagator` stores the 2N columns of Φ split into two N×N×2 blocks:
  block_upper_ic[:,:,1:2] ↔ Φ[:,1:N]   (result from IC=(I,0))
  block_lower_ic[:,:,1:2] ↔ Φ[:,N+1:2N]  (result from IC=(0,I))
"""
function assemble_fm_matrix(propagators::Vector{ChunkPropagator}, idx_range)
    N = size(propagators[1].block_upper_ic, 1)
    Phi = Matrix{ComplexF64}(I, 2N, 2N)
    isempty(idx_range) && return Phi
    for i in idx_range
        p = propagators[i]
        Phi_i = [p.block_upper_ic[:,:,1]  p.block_lower_ic[:,:,1];
                 p.block_upper_ic[:,:,2]  p.block_lower_ic[:,:,2]]
        Phi = Phi_i * Phi
    end
    return Phi
end

"""
    compute_delta_prime_matrix!(intr, propagators, chunks)

Compute the inter-surface tearing stability matrix (2·msing × 2·msing) using the
STRIDE global BVP formulation [Glasser 2018 Phys. Plasmas 25, 032501, Sec. III.B].

The BVP encodes the full plasma response with unknowns at each surface boundary:
  x_axis   (N):    free IC parameters at the axis  (U₁ = 0 regular solutions)
  x_left[j]  (2N): state at left inner-layer boundary of surface j
  x_right[j] (2N): state at right inner-layer boundary of surface j
  x_edge   (N):    free IC parameters at the edge  (conducting wall, U₁ = 0)
Total unknowns: nMat = (2 + 4·msing)·N.

The BVP matrix M is assembled from segment propagators, inner-layer continuity
equations (non-resonant modes are continuous through each surface), and driving
terms (unit U₂[ipert_res] amplitude at each surface side). Each of the 2·msing
driving configurations is solved independently by LU back-substitution.

## Well-conditioned BVP via bidirectional propagators

For each inter-surface segment j (from singR[j-1] to singL[j]), the crossing chunk
(direction=-1) was integrated backward, giving a well-conditioned backward FM:
  Phi_L[j] = propagators[i_crossings[j]]: maps state at singL[j] → state at psi_m[j]

The forward chunks (direction=+1) between singR[j-1] and psi_m[j] give:
  Phi_R[j] = product of forward propagators: maps state at singR[j-1] → state at psi_m[j]

Continuity at the junction psi_m[j]:
  Phi_R[j] · x_right[j-1] = Phi_L[j] · x_left[j]
  → Phi_R[j] · x_right[j-1] - Phi_L[j] · x_left[j] = 0

This replaces the ill-conditioned monolithic Phi_segs[j] = Phi_L[j]⁻¹ · Phi_R[j]
with a split formulation where each factor is well-conditioned.

Element delta_prime_matrix[dRow, 2k-1] = U₂[ipert_k] component at the left side
of surface k when driving term dRow is active. dRow = 2j-1 (left of surface j) or
2j (right of surface j). This is the raw BVP coefficient; it differs from `delta_prime`
(which uses the asymptotic normalization from sing_get_ca).

Only called from `parallel_eulerlagrange_integration` (requires FM propagators).
The result is stored in `intr.delta_prime_matrix`.

## Limitations
- Assumes exactly one resonant mode per singular surface (standard single-n case).
- Uses a conducting wall edge BC (U₁ = 0). Vacuum BC is deferred.
"""
function compute_delta_prime_matrix!(
    intr::ForceFreeStatesInternal,
    propagators::Vector{ChunkPropagator},
    chunks::Vector{IntegrationChunk}
)
    msing = intr.msing
    msing == 0 && return
    N = intr.numpert_total

    # Single-resonance assumption: each surface has exactly one resonant mode.
    # Multi-resonance surfaces would require coupling all resonant modes simultaneously;
    # only the first (sp.m[1], sp.n[1]) is used below.
    @assert all(j -> length(intr.sing[j].m) == 1, 1:msing) "compute_delta_prime_matrix! only supports single-resonance surfaces"

    # Find the index of the crossing chunk for each surface (direction=-1 in bidirectional mode)
    i_crossings = findall(c -> c.needs_crossing, chunks)
    @assert length(i_crossings) == msing

    # Build Phi_L[j] (backward crossing chunk FM) and Phi_R[j] (product of forward
    # chunks before the junction psi_m[j]) for each inter-surface segment j.
    #
    # Phi_L[j]: single backward chunk propagator at i_crossings[j]
    #   Maps state at psi_end (≈ singL[j]) → psi_start (= psi_m[j], away from singularity)
    #   Well-conditioned because growing EL solutions decay when integrated backward.
    #
    # Phi_R[j]: product of forward chunk propagators from singR[j-1] to psi_m[j]
    #   Maps state at singR[j-1] → psi_m[j]
    #   Phi_R[msing+1]: forward chunks from singR[msing] to edge (for edge BC)
    Phi_L_mats = [assemble_fm_matrix(propagators, i_crossings[j]:i_crossings[j]) for j in 1:msing]
    Phi_R_mats = Vector{Matrix{ComplexF64}}(undef, msing + 1)
    Phi_R_mats[1] = assemble_fm_matrix(propagators, 1:i_crossings[1]-1)
    for j in 2:msing
        Phi_R_mats[j] = assemble_fm_matrix(propagators, i_crossings[j-1]+1:i_crossings[j]-1)
    end
    Phi_R_mats[msing+1] = assemble_fm_matrix(propagators, i_crossings[msing]+1:length(chunks))

    # Resonant mode index (1:N) for each surface (single-resonance case)
    ipert_all = [begin
        sp = intr.sing[j]
        idx = 1 + sp.m[1] - intr.mlow + (sp.n[1] - intr.nlow) * intr.mpert
        @assert 1 <= idx <= N "Resonant mode index out of range"
        idx
    end for j in 1:msing]

    # BVP dimensions
    nMat = (2 + 4 * msing) * N
    s2   = 2 * msing

    # Column layout (1-indexed):
    #   x_axis:     1:N
    #   x_left[j]:  N + 4N*(j-1)+1 : N + 4N*(j-1)+2N
    #   x_right[j]: N + 4N*(j-1)+2N+1 : N + 4N*j
    #   x_edge:     N + 4N*msing+1 : nMat
    col_axis     = 1:N
    col_left(j)  = (N + 4N*(j-1)+1) : (N + 4N*(j-1)+2N)
    col_right(j) = (N + 4N*(j-1)+2N+1) : (N + 4N*j)
    col_edge     = (N + 4N*msing+1) : nMat

    # Row layout:
    #   Axis-to-surface 1 junction:  1:2N   (2N rows)
    #   For each surface j:
    #     Continuity:      2N + (4N-2)*(j-1)+1 : 2N + (4N-2)*(j-1)+(2N-2)  (2N-2 rows)
    #     Junction/edge:   2N + (4N-2)*(j-1)+(2N-2)+1 : 2N + (4N-2)*j      (2N rows)
    #   Driving terms:     2N + (4N-2)*msing+1 : nMat                        (2·msing rows)
    row_drive_base = 2N + (4N-2)*msing

    M = zeros(ComplexF64, nMat, nMat)

    # Axis-to-surface 1 junction at psi_m[1]:
    # Phi_R[1][:,N+1:2N]·x_axis = Phi_L[1]·x_left[1]
    # → Phi_L[1]·x_left[1] - Phi_R[1][:,N+1:2N]·x_axis = 0
    # (Phi_R[1][:,N+1:2N] selects the N regular-solution columns from the axis IC U₂=I)
    M[1:2N, col_left(1)] .= Phi_L_mats[1]
    M[1:2N, col_axis]    .= -view(Phi_R_mats[1], :, N+1:2N)

    for j in 1:msing
        ipert_j = ipert_all[j]

        # Continuity at surface j: x_left[j][i] = x_right[j][i] for non-resonant i
        # (skip i = ipert_j and i = ipert_j+N, the two resonant-mode rows)
        row_cont = 2N + (4N-2)*(j-1)
        for i in 1:2N
            if i != ipert_j && i != ipert_j + N
                row_cont += 1
                M[row_cont, col_left(j)[i]]  =  1
                M[row_cont, col_right(j)[i]] = -1
            end
        end

        # Junction / edge matching (2N rows starting at row_cont+1)
        junc_rows = (row_cont+1) : (2N + (4N-2)*j)
        if j < msing
            # Junction at psi_m[j+1]:
            # Phi_R[j+1]·x_right[j] = Phi_L[j+1]·x_left[j+1]
            # → Phi_R[j+1]·x_right[j] - Phi_L[j+1]·x_left[j+1] = 0
            M[junc_rows, col_right(j)]   .=  Phi_R_mats[j+1]
            M[junc_rows, col_left(j+1)]  .= -Phi_L_mats[j+1]
        else
            # Conducting wall: Phi_R[msing+1]·x_right[msing] = [0; I_N]·x_edge
            # Upper N rows: U₁ = 0  (no x_edge contribution)
            # Lower N rows: U₂ = x_edge  (contribution from -I·x_edge)
            # (Phi_R[msing+1] is all forward chunks → same as old Phi_segs[msing+1])
            M[junc_rows, col_right(msing)] .= Phi_R_mats[msing+1]
            M[junc_rows[N+1:end], col_edge] .= -I(N)
        end

        # Driving terms: unit U₂[ipert_j] amplitude at left and right of surface j
        M[row_drive_base + 2j-1, col_left(j)[ipert_j+N]]  = 1
        M[row_drive_base + 2j,   col_right(j)[ipert_j+N]] = 1
    end

    M_lu = lu(M)
    delta_mat = zeros(ComplexF64, s2, s2)
    b = zeros(ComplexF64, nMat)

    for jsing in 1:msing
        for side in 1:2   # side=1: left drive; side=2: right drive
            dRow = 2jsing - (2 - side)   # 2j-1 for left, 2j for right
            fill!(b, 0)
            b[row_drive_base + dRow] = 1
            x = M_lu \ b

            for ksing in 1:msing
                ipert_k = ipert_all[ksing]
                # Extract U₂[ipert_k] at left and right boundaries of surface ksing
                delta_mat[dRow, 2ksing-1] = x[col_left(ksing)[ipert_k+N]]
                delta_mat[dRow, 2ksing]   = x[col_right(ksing)[ipert_k+N]]
            end
        end
    end

    intr.delta_prime_matrix = delta_mat
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

    # Update integration tolerances (same logic as integrator_callback!)
    integrator.opts.reltol = compute_tols(ctrl, intr, odet, chunk.ising)

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
    rtol = compute_tols(ctrl, intr, odet, chunk.ising)
    prob = ODEProblem(sing_der!, odet.u, (chunk.psi_start, chunk.psi_end),
                      (ctrl, equil, ffit, intr, odet, chunk))
    sol = solve(prob, BS5(); reltol=rtol, callback=cb, save_everystep=false, save_end=true)
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
    sing_asymp = compute_sing_asymptotics(singp, ctrl, equil, ffit, intr)
    dpsi = singp.psifac - odet.psifac  # ψ_res - ψ_current (positive)

    # Get asymptotic coefficients before crossing
    ua = sing_get_ua(sing_asymp, -dpsi)
    odet.ca_l[:, :, :, ising] .= sing_get_ca(odet.u, ua, intr)

    # Resonant perturbation indices (same formula as in cross_ideal_singular_surf!)
    ipert_res = 1 .+ singp.m .- intr.mlow .+ (singp.n .- intr.nlow) .* intr.mpert

    if !ctrl.con_flag
        # Zero the resonant column of (S, I) using ipert_res directly (no GR sorting needed).
        # The zeroed column stays zero through the predictor step since both slices are zero.
        for i in eachindex(sing_asymp.r1)
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

    # Apply asymptotic solution on other side of singular surface
    ua = sing_get_ua(sing_asymp, dpsi)
    if !ctrl.con_flag
        for i in eachindex(sing_asymp.r1)
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
        n_res = length(sing_asymp.r1)
        N = intr.numpert_total
        resize!(intr.sing[ising].delta_prime, n_res)
        intr.sing[ising].delta_prime_col = zeros(ComplexF64, N, n_res)
        for i in eachindex(sing_asymp.r1)
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

Enable via `use_riccati = true` in `[ForceFreeStates]` section of jpec.toml, or by
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
        println("   ψ = $((@sprintf "%.3f" odet.psifac)),  q = $((@sprintf "%.3f" equil.profiles.q_spline(odet.psifac)))")
    end

    for chunk in chunks
        # Integrate this chunk using the Riccati ODE (Riccati callback skips Gaussian reduction)
        riccati_integrate_chunk!(odet, ctrl, equil, ffit, intr, chunk)
        if ctrl.verbose
            println("   ψ = $((@sprintf "%.3f" odet.psifac)),  q= $((@sprintf "%.3f" odet.q)),  max(S) = $((@sprintf "%.2e" maximum(abs, odet.u[:,:,1]))),  steps = $(odet.step-1)")
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
            println("Truncating integration at peak edge dW: ψ = $((@sprintf "%.2f" odet.psi_store[odet.step])),  q = $((@sprintf "%.2f" odet.q_store[odet.step]))")
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
        println("Evaluating fixed-boundary stability criterion")
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
    rtol = chunk.ising > 0 ? ctrl.tol_r : ctrl.tol_nr
    params = (ctrl, equil, ffit, intr, odet_proxy, chunk)

    # Upper block IC: U₁ = I, U₂ = 0
    u_upper = zeros(ComplexF64, N, N, 2)
    for i in 1:N
        u_upper[i, i, 1] = 1
    end
    odet_proxy.spline_hint[] = 1
    prob = ODEProblem(sing_der!, u_upper, tspan, params)
    sol = solve(prob, BS5(); reltol=rtol, save_everystep=false, save_end=true)
    prop.block_upper_ic .= sol.u[end]

    # Lower block IC: U₁ = 0, U₂ = I
    u_lower = zeros(ComplexF64, N, N, 2)
    for i in 1:N
        u_lower[i, i, 2] = 1
    end
    odet_proxy.spline_hint[] = 1
    prob = ODEProblem(sing_der!, u_lower, tspan, params)
    sol = solve(prob, BS5(); reltol=rtol, save_everystep=false, save_end=true)
    prop.block_lower_ic .= sol.u[end]
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

Enable via `use_parallel = true` in `[ForceFreeStates]` of jpec.toml, or by setting
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
        println("   ψ = $((@sprintf "%.3f" odet.psifac)),  q = $((@sprintf "%.3f" equil.profiles.q_spline(odet.psifac)))")
        println("   Parallel FM: $(length(chunks)) chunks, $nthreads threads")
    end

    # PARALLEL phase: integrate all chunks independently from identity IC
    Threads.@threads for i in eachindex(chunks)
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
    # last_crossing_step tracks the u_store index of the most recent crossing so that
    # the outer plasma (from last rational surface to psilim) can be re-integrated.
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
            println("   ψ = $((@sprintf "%.3f" odet.psifac)),  q= $((@sprintf "%.3f" odet.q)),  max(S) = $((@sprintf "%.2e" maximum(abs, odet.u[:,:,1]))),  steps = $(odet.step-1)")
        end

        if chunk.needs_crossing
            if ctrl.kin_flag
                error("kin_flag = true not implemented yet!")
            else
                # State is already (S, I) from the renorm above.
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
    outer_chunk = IntegrationChunk(; psi_start=odet.psifac, psi_end=intr.psilim,
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
            println("Truncating integration at peak edge dW: ψ = $((@sprintf "%.2f" odet.psi_store[odet.step])),  q = $((@sprintf "%.2f" odet.q_store[odet.step]))")
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

    # Compute inter-surface Δ' matrix using the STRIDE global BVP.
    # Uses the chunk propagators from the parallel phase (all chunks, including outer plasma).
    # Only called when there are singular surfaces to couple.
    if !ctrl.con_flag && intr.msing > 0
        compute_delta_prime_matrix!(intr, propagators, chunks)
    end

    # Evaluate fixed-boundary stability criterion
    if ctrl.verbose
        println("Evaluating fixed-boundary stability criterion")
    end
    odet.nzero = evaluate_stability_criterion!(odet, equil.profiles)

    # transform_u! is called for consistency but is a no-op (ifix=0, no Gaussian reduction)
    transform_u!(odet, intr)

    return odet
end
