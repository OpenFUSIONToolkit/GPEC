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
    # matching Fortran STRIDE's separate vmatl/vmatr (sing_vmat).
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

    if ctrl.kinetic_factor == 0
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
    if ctrl.kinetic_factor == 0
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
    if ctrl.kinetic_factor == 0
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

Serial single-chunk Riccati driver, retained as a reference/unit-test path for the
Riccati renormalization. The production chunked-Riccati path is
`parallel_eulerlagrange_integration`, reached via `forcefreestates_integration`.
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
    subsample_chunk_steps(nsteps, save_interval) -> Vector{Int}

Pick which of a chunk's `nsteps` adaptive ODE steps to retain for dense `u_store`
reconstruction. Always keeps the first two and last two steps (chunk endpoints and
the rapid variation near singular surfaces) plus every `save_interval`-th step,
mirroring the density policy of `riccati_integrator_callback!`.
"""
function subsample_chunk_steps(nsteps::Int, save_interval::Int)
    nsteps <= 4 && return collect(1:nsteps)
    keep = Set{Int}([1, 2, nsteps - 1, nsteps])
    for k in 1:max(1, save_interval):nsteps
        push!(keep, k)
    end
    return sort!(collect(keep))
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

Both solves save every accepted step; a subsampled common ψ grid (see
`subsample_chunk_steps`) is stored in `prop.psi_history` / `upper_history` /
`lower_history` so the serial assembly can reconstruct a dense `u_store` directly.
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

    # Upper block IC: U₁ = I, U₂ = 0. Save every accepted step so the per-ψ
    # chunk history is available for direct dense u_store reconstruction.
    u_upper = zeros(ComplexF64, N, N, 2)
    for i in 1:N
        u_upper[i, i, 1] = 1
    end
    odet_proxy.spline_hint[] = 1
    prob = ODEProblem(sing_der!, u_upper, tspan, params)
    sol_upper = solve(prob, Vern9(); reltol=rtol)
    prop.block_upper_ic .= sol_upper.u[end]

    # Common ψ grid for the chunk history: a subsample of the upper-IC step grid.
    keep = subsample_chunk_steps(length(sol_upper.t), ctrl.save_interval)
    psi_grid = sol_upper.t[keep]

    # Lower block IC: U₁ = 0, U₂ = I. Vern9's dense interpolant is evaluated at the
    # shared psi_grid so upper/lower history align on identical ψ.
    u_lower = zeros(ComplexF64, N, N, 2)
    for i in 1:N
        u_lower[i, i, 2] = 1
    end
    odet_proxy.spline_hint[] = 1
    prob = ODEProblem(sing_der!, u_lower, tspan, params)
    sol_lower = solve(prob, Vern9(); reltol=rtol)
    prop.block_lower_ic .= sol_lower.u[end]

    # Store per-ψ history, ordered monotonically increasing in ψ for both chunk
    # directions (backward chunks integrate psi_end → psi_start, so reverse).
    empty!(prop.psi_history)
    empty!(prop.upper_history)
    empty!(prop.lower_history)
    order = chunk.direction == 1 ? (1:length(keep)) : reverse(1:length(keep))
    for k in order
        push!(prop.psi_history, psi_grid[k])
        push!(prop.upper_history, copy(sol_upper.u[keep[k]]))
        push!(prop.lower_history, sol_lower(psi_grid[k]))
    end
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
    propagate_chunk_state(upper, lower, anchor) -> Array{ComplexF64,3}

Apply a chunk's fundamental matrix — its (upper, lower) identity-IC solution blocks
captured at one ψ — to an `anchor` state. Same linear map as `apply_propagator!`,
but pure (returns a fresh (N,N,2) array) for dense `u_store` reconstruction.
"""
function propagate_chunk_state(upper::Array{ComplexF64,3}, lower::Array{ComplexF64,3},
                               anchor::Array{ComplexF64,3})
    out = similar(anchor)
    a1 = @view anchor[:, :, 1]
    a2 = @view anchor[:, :, 2]
    @views out[:, :, 1] .= upper[:, :, 1] * a1 .+ lower[:, :, 1] * a2
    @views out[:, :, 2] .= upper[:, :, 2] * a1 .+ lower[:, :, 2] * a2
    return out
end

"""
    assemble_riccati_s_gauge!(odet, chunks, propagators, ctrl, equil, ffit, intr) -> S_at_surface_left

S-gauge assembly pass of the parallel-FM solution. Applies each chunk propagator in order,
renormalizing to the (S, I) Riccati form after each chunk (STRIDE `ode_fixup`), and crosses
singular surfaces with `riccati_cross_ideal_singular_surf!`.

Its purpose is the Riccati-gauge diagnostics: it returns `S_at_surface_left` (the Riccati
matrix S at each singular surface's left boundary, the axis BC for the Δ' BVP) and
populates `odet.ca_l`/`odet.ca_r` and `intr.sing[*]` (Δ', `ua_left`, …) in the (S, I) gauge
that `PerturbedEquilibrium`'s `SingularCoupling.jl` and `compute_delta_prime_matrix!`
require. It does NOT produce the canonical `u_store` — that comes from
`standard_eulerlagrange_pass`. Run on a scratch `OdeState` so its axis-gauge bookkeeping
does not collide with the canonical odet from the standard pass.
"""
function assemble_riccati_s_gauge!(
    odet::OdeState, chunks::Vector{IntegrationChunk}, propagators::Vector{ChunkPropagator},
    ctrl::ForceFreeStatesControl, equil::Equilibrium.PlasmaEquilibrium,
    ffit::FourFitVars, intr::ForceFreeStatesInternal
)
    N = intr.numpert_total
    initialize_el_at_axis!(odet, ctrl, equil.profiles, intr)
    odet.new = false
    fill!(odet.unorm0, 1.0)
    odet.step = 1

    S_at_surface_left = Matrix{ComplexF64}[]

    for (i, chunk) in enumerate(chunks)
        # Forward chunks: apply Φ_fwd directly. Backward (crossing) chunks: apply the
        # inverse of the well-conditioned backward propagator via LU solve.
        if chunk.direction == -1
            apply_propagator_inverse!(odet, propagators[i])
        else
            apply_propagator!(odet, propagators[i])
        end

        # Renorm to (S, I) after every chunk — equivalent to STRIDE's ode_fixup.
        renormalize_riccati_inplace!(odet.u, N)
        odet.psifac = chunk.psi_end
        odet.q = equil.profiles.q_spline(odet.psifac)

        if chunk.needs_crossing
            if ctrl.kinetic_factor > 0
                error("kinetic_factor > 0 not implemented yet in Riccati!")
            else
                # Save S at the left boundary of this surface (state is (S, I)).
                push!(S_at_surface_left, copy(odet.u[:, :, 1]))

                # riccati_cross_ideal_singular_surf! computes ca_l/ca_r and intr.sing[*]
                # in the (S, I) Riccati gauge and zeros column ipert_res directly.
                riccati_cross_ideal_singular_surf!(odet, ctrl, equil, ffit, intr, chunk.ising)
            end
        end
    end

    return S_at_surface_left
end

"""
    parallel_eulerlagrange_integration(ctrl, equil, ffit, intr)
        -> (odet, propagators, chunks, S_at_surface_left)

Parallel fundamental-matrix (propagator) driver for the EL integration. The chunk
propagators are integrated once (in parallel) and used only for the deferred Δ' BVP and
S-gauge outputs — `u_store` is built by the standard EL pass so all three integration
paths produce a bit-identical `u_store`.

1. **u_store pass**: `standard_eulerlagrange_pass` runs the standard forward EL sweep on
   natural chunks (`chunk_el_integration_bounds`), with `integrate_el_region!`'s
   partition-invariant ContinuousCallback GR firing. Output is the canonical
   ξ_ψ in the axis basis that `PerturbedEquilibrium` consumes.
2. **Chunk generation**: `chunk_el_integration_bounds` (bidirectional) + `balance_integration_chunks`.
3. **Parallel phase**: `integrate_propagator_chunk!` integrates each chunk's fundamental
   matrix from identity ICs, capturing per-ψ history. `Threads.@threads :static`.
4. **S-gauge pass** (`assemble_riccati_s_gauge!`, on a scratch `OdeState`): produces the
   (S, I) Riccati-gauge `ca_l`/`ca_r`, `intr.sing[*]`, and `S_at_surface_left` that the Δ'
   BVP (`compute_delta_prime_matrix!`) and `SingularCoupling.jl` require.

This is the `integration_method = "ChunkedRiccati"` path, reached via
`forcefreestates_integration`. Requires `singfac_min != 0`. `compute_delta_prime_matrix!`
is called from the main pipeline (after `free_run!`, when the vacuum edge BC is
available) using the returned propagators/chunks.

**Bidirectional integration for large-N accuracy:** the crossing chunk nearest each
rational surface is integrated *backward* (`direction=-1`); the backward propagator Φ_bwd
is well-conditioned, and forward propagation is recovered via a stable LU solve in
`apply_propagator_inverse!` (STRIDE; Glasser 2018 Phys. Plasmas 25, 032501).
"""
function parallel_eulerlagrange_integration(
    ctrl::ForceFreeStatesControl, equil::Equilibrium.PlasmaEquilibrium,
    ffit::FourFitVars, intr::ForceFreeStatesInternal
)
    N = intr.numpert_total

    # u_store is built by the standard EL pass on natural (un-balanced) chunks. With the
    # Phase C ContinuousCallback fix in integrate_el_region!, GR fires at the exact
    # uratio = ucrit crossing inside each chunk, so the resulting u_store is what the
    # standard / Riccati / parallel paths all consume from PerturbedEquilibrium.
    # Delegating to the standard pass here unifies the three paths' u_store basis
    # without bisecting through balanced sub-chunk propagator histories.
    odet = standard_eulerlagrange_pass(ctrl, equil, ffit, intr)

    # Build balanced sub-chunks for the parallel-FM propagator pass. These feed the Δ'
    # BVP (compute_delta_prime_matrix!) and the S-gauge ca_l/ca_r outputs, both of
    # which benefit from load-balanced parallel execution. We use a *throwaway* OdeState
    # primed at the axis only so chunk_el_integration_bounds reads psifac/ising_start
    # correctly — odet has already advanced past the edge.
    chunk_odet = OdeState(N, 1, 1, intr.msing)
    if ctrl.sing_start <= 0
        initialize_el_at_axis!(chunk_odet, ctrl, equil.profiles, intr)
    end
    base_chunks = chunk_el_integration_bounds(chunk_odet, ctrl, intr; bidirectional=true)
    chunks = balance_integration_chunks(base_chunks, ctrl, intr)
    propagators = [ChunkPropagator(N) for _ in chunks]

    julia_nthreads = Threads.nthreads()
    odet_proxies = [OdeState(N, 1, 1, 0) for _ in 1:Threads.maxthreadid()]
    # integration_threads = 0 → use all available threads; otherwise cap at nthreads().
    requested_threads = ctrl.integration_threads == 0 ? julia_nthreads : ctrl.integration_threads
    bvp_threads = max(1, min(julia_nthreads, requested_threads))

    if ctrl.verbose
        @info "   Parallel FM: $(length(chunks)) chunks, $bvp_threads thread$(bvp_threads == 1 ? "" : "s")"
    end

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

    psilim_before = intr.psilim
    finalize_canonical_u_store!(odet, ctrl, equil, ffit, intr)
    # free_run! derives `coeffs` from odet.u and right-multiplies u_store by it — keep them
    # in the same gauge by anchoring odet.u to the canonical edge state.
    odet.u .= odet.u_store[:, :, :, end]

    # truncate_at_dW_peak self-consistency for the Δ' BVP: rebuild the straddling chunk's
    # propagator and drop chunks past the relocated psilim, so compute_delta_prime_matrix!
    # applies the edge BC at the truncated psilim to a matching propagator set.
    if ctrl.truncate_at_dW_peak && intr.psilim < psilim_before
        peak_psi = intr.psilim
        last_chunk_idx = findlast(c -> c.psi_start < peak_psi, chunks)
        last_chunk_idx === nothing &&
            error("truncate_at_dW_peak: peak ψ=$peak_psi lies before all chunk starts")
        straddling = chunks[last_chunk_idx]
        if straddling.psi_end > peak_psi
            chunks[last_chunk_idx] = IntegrationChunk(
                psi_start=straddling.psi_start, psi_end=peak_psi,
                needs_crossing=straddling.needs_crossing, ising=straddling.ising,
                direction=straddling.direction)
            integrate_propagator_chunk!(propagators[last_chunk_idx], chunks[last_chunk_idx],
                                        ctrl, equil, ffit, intr, OdeState(N, 1, 1, 0))
        end
        if last_chunk_idx < length(chunks)
            chunks = chunks[1:last_chunk_idx]
            propagators = propagators[1:last_chunk_idx]
        end
    end

    # S-gauge pass: Riccati-gauge ca_l/ca_r, intr.sing[*], and S_at_surface_left for the Δ'
    # BVP, on a scratch OdeState so its axis-gauge bookkeeping does not disturb the
    # canonical GR-gauge odet. The standard crossing in the GR pass wrote axis-gauge
    # ca_l/ca_r into odet; overwrite them with the (S, I) values SingularCoupling.jl needs.
    odet_s = OdeState(N, ctrl.numsteps_init, ctrl.numunorms_init, intr.msing)
    S_at_surface_left = assemble_riccati_s_gauge!(odet_s, chunks, propagators, ctrl, equil,
                                                  ffit, intr)
    odet.ca_l .= odet_s.ca_l
    odet.ca_r .= odet_s.ca_r

    return odet, propagators, chunks, S_at_surface_left
end
