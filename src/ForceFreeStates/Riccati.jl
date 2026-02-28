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
    # Renormalize end state to (S, I) convention for the next chunk or crossing
    renormalize_riccati_inplace!(odet.u, intr.numpert_total)
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
    params = (ctrl, equil, ffit, intr, odet, IntegrationChunk(0.0, 0.0, false, ising))
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
    odet.ca_r[:, :, :, ising] .= sing_get_ca(odet.u, ua, intr)

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
