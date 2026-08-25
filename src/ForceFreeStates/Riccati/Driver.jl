"""
    Riccati/ - Dual Riccati reformulation of the Euler-Lagrange ODE

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

"""
    riccati_eulerlagrange_integration(ctrl, equil, ffit, intr) -> (odet, propagators, chunks, S_left)

The Riccati/STRIDE integrator: a chunked fundamental matrix (propagator) driver for the EL
integration. The trailing three return values feed the Δ' BVP in `compute_delta_prime_matrix!`;
this is the only branch that produces them.

Solves the same system as [`forward_eulerlagrange_integration`](@ref), but integrates all bulk
chunks concurrently using `Threads.@threads`, then re-integrates the outer plasma serially:

1. **Chunk generation**: calls `chunk_el_integration_bounds`, then `balance_integration_chunks`
   to sub-divide chunks for load balancing. The chunk count depends only on `intr.msing` and
   `ctrl.nchunks`, never on the thread count, so results are thread-independent.
2. **Propagator phase**: `integrate_propagator_chunk!` integrates each chunk independently
   from identity initial conditions (no accumulated state, no normalization/callback).
   Each thread uses a private `OdeState` proxy for `sing_der!` side effects.
3. **Serial assembly**: propagators are applied sequentially with `apply_propagator!`.
   Rational surface crossings use `riccati_cross_ideal_singular_surf!` (no Gaussian
   reduction).
4. **Outer plasma re-integration**: after the last rational surface crossing, the outer
   plasma (from last ψ_s to psilim) is re-integrated using `riccati_integrate_chunk!`.
   FM propagation in this region is prone to precision loss for high N (exponential growth
   without renormalization); Riccati integration keeps matrices bounded and provides dense
   checkpoints for `findmax_dW_edge!`.

Select via `integrator = "riccati"` in `[ForceFreeStates]` of gpec.toml. Requires
`singfac_min != 0`. Uses whatever threads `julia -t` provides; `ctrl.nchunks` is the only
tunable.

**Key differences from the forward integrator:**
- No Gaussian reduction in the propagator BVP phase (crossings use the
  Riccati-style algorithm, `odet.ifix` stays 0)
- `transform_u!` is called on the odet but is a no-op (ifix=0)
- Outer plasma uses serial Riccati integration for numerical stability
- `odet.u_store` holds chunk-endpoint Riccati states, not dense Euler-Lagrange ξ, and
  `odet.u_store_el_basis` stays `false`: this integrator never claims the EL basis, so
  PerturbedEquilibrium and the HDF5 forward-integration ξ datasets require the forward path.

**Bidirectional integration for large-N accuracy:**
The crossing chunk (nearest to each rational surface singL[j]) is integrated *backward*
(`direction=-1`, `tspan` reversed). Backward integration of a region where solutions grow
exponentially forward causes them to *decay*, so the resulting backward FM Φ_bwd is
well-conditioned. The accurate forward propagation is recovered as Φ_bwd⁻¹ via a stable
LU solve in `apply_propagator_inverse!`. This follows the same principle as STRIDE
(Glasser 2018 Phys. Plasmas 25, 032501). The all-forward path had ~10% energy error for
the DIIID-like example (N=26, n=1); bidirectional reduces this to within 2%.
"""
function riccati_eulerlagrange_integration(
    ctrl::ForceFreeStatesControl, equil::Equilibrium.PlasmaEquilibrium,
    ffit::FourFitVars, intr::ForceFreeStatesInternal
)
    odet = _initialize_parallel_odet(ctrl, equil, ffit, intr)
    chunks, propagators, odet_proxies = _setup_parallel_chunks_and_proxies(odet, ctrl, intr)
    _log_parallel_start(ctrl, odet, equil, chunks)

    _run_parallel_bvp_phase!(propagators, chunks, ctrl, equil, ffit, intr, odet_proxies)

    # Harvest solver-step counts accumulated thread-locally in each proxy during the BVP phase.
    # The outer re-integration below uses riccati_integrate_chunk!, which counts via its callback.
    odet.total_steps += sum(p.total_steps for p in odet_proxies)

    S_at_surface_left, last_crossing_step =
        _assemble_propagators_serially!(odet, propagators, chunks, ctrl, equil, ffit, intr)

    _reintegrate_outer_plasma!(odet, last_crossing_step, ctrl, equil, ffit, intr)

    chunks, propagators = _handle_edge_dW_scan!(odet, chunks, propagators, ctrl, equil, ffit, intr)

    # compute_delta_prime_matrix! is called from the main pipeline (after free_run) so
    # that vacuum response wv is available for the edge BC. With self-consistent truncation,
    # the propagators/chunks returned here match intr.psilim exactly, so Δ' is well-defined
    # for both truncate_at_dW_peak=false (full domain) and =true (peak).
    if ctrl.verbose
        @info "Evaluating fixed-boundary stability criterion"
    end
    odet.nzero = evaluate_stability_criterion!(odet, equil.profiles)
    transform_u!(odet, intr)  # no-op when ifix=0 (no Gaussian reduction)

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
                             chunks::Vector{IntegrationChunk})
    ctrl.verbose || return
    @info "   ψ = $((@sprintf "%.3f" odet.psifac)),  q = $((@sprintf "%.3f" equil.profiles.q_spline(odet.psifac)))"
    @info "   Riccati FM: $(length(chunks)) chunks over $(Threads.nthreads()) thread$(Threads.nthreads() == 1 ? "" : "s")"
end

# Integrate each chunk's FM propagator from identity IC across whatever threads `julia -t`
# provides. The :static scheduler makes Threads.threadid() a stable index into odet_proxies.
# Each chunk is independent (identity IC, no accumulated state), so the result does not
# depend on how chunks are distributed across threads.
function _run_parallel_bvp_phase!(propagators::Vector{ChunkPropagator},
                                  chunks::Vector{IntegrationChunk},
                                  ctrl::ForceFreeStatesControl,
                                  equil::Equilibrium.PlasmaEquilibrium, ffit::FourFitVars,
                                  intr::ForceFreeStatesInternal,
                                  odet_proxies::Vector{OdeState})
    Threads.@threads :static for i in eachindex(chunks)
        integrate_propagator_chunk!(propagators[i], chunks[i], ctrl, equil, ffit, intr,
                                    odet_proxies[Threads.threadid()])
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
            # Save non-crossing end-of-chunk state. These columns are FM/Riccati chunk
            # endpoints, not the Euler-Lagrange state, so the odet never claims the EL basis.
            odet.u_store_el_basis = false
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
