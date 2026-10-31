"""
    Riccati/ - Dual Riccati reformulation of the Euler-Lagrange ODE

Tracks the dual Riccati matrix S = U₁·U₂⁻¹ = P⁻¹ (P = U₂·U₁⁻¹ the plasma response matrix), which stays
bounded near singular surfaces where U₁ and U₂ grow exponentially. Reference: Glasser (2018) Phys. Plasmas
25, 032507 — Eq. 19 (dual form), dS/dψ = w†·F̄⁻¹·w − S·Ḡ·S with w = Q − K̄·S.

The explicit Riccati ODE is not integrated directly: its quadratic term defeats relative error control when
|S| is large. Instead the linear EL ODE (`sing_der!`) is integrated in (U₁, U₂), and the exact change of
variables `renormalize_riccati_inplace!` resets the state to (S, I) whenever max(|U₁|) or max(|U₂|) exceeds
`renorm_threshold`, the Riccati analogue of Gaussian reduction.

## Storage convention

  - During a chunk: u[:,:,1] = U₁ (starts as S), u[:,:,2] = U₂ (starts as I).
  - After renormalization: u[:,:,1] = S, u[:,:,2] = I, so downstream U₁/U₂ ratios (wp in Free.jl, the
    fixed-boundary criterion) read S directly.
  - Axis: S(ψ₀) = 0 (U₁ = 0, U₂ = I), unless a positive `ctrl.frobenius_psi_max` at or above the start
    opts into the diagonal Frobenius init (U₂ = I, per-mode U₁).

Unlike the forward path, `riccati_integrator_callback!` renormalizes instead of Gaussian reduction,
`riccati_cross_ideal_singular_surf!` crosses without Gaussian reduction, and `transform_u!` is a no-op.
"""

"""
    riccati_eulerlagrange_integration(ctrl, equil, mats, intr) -> (odet, propagators, chunks, S_left)

The Riccati/STRIDE integrator: solves the same system as [`forward_eulerlagrange_integration`](@ref)
by integrating chunk fundamental matrices (propagators) concurrently with `Threads.@threads`,
assembling them serially with Riccati crossings (no Gaussian reduction), and re-integrating the
outer plasma with `riccati_integrate_chunk!` for dense, bounded checkpoints. Crossing chunks are
integrated backward and inverted by LU, which keeps them well-conditioned at large N (STRIDE,
Glasser 2018 Phys. Plasmas 25, 032501). The trailing three return values feed the Δ' BVP in
`compute_delta_prime_matrix!`; this is the only branch that produces them.

Select with `Riccati()`, or `integrator = "Riccati"` in a deck's `[ForceFreeStates]`. Requires
`singfac_min != 0`. Results do not depend on the thread count; `ctrl.algorithm.nchunks` is the only
tunable. `odet.u_store` holds chunk-endpoint Riccati states, not the EL basis
(`odet.u_store_el_basis == false`), so PerturbedEquilibrium and the HDF5 ξ datasets require `Forward()`.
"""
function riccati_eulerlagrange_integration(
    ctrl::ForceFreeStatesControl, equil::Equilibrium.PlasmaEquilibrium,
    mats::MatrixSplines, intr::ForceFreeStatesInternal
)
    odet = _initialize_parallel_odet(ctrl, equil, mats, intr)
    chunks, propagators, odet_proxies = _setup_parallel_chunks_and_proxies(odet, ctrl, intr)
    intr.verbose && _log_parallel_start(odet, equil, chunks)

    _run_parallel_bvp_phase!(propagators, chunks, ctrl, equil, mats, intr, odet_proxies)

    # Harvest thread-local solver-step counts; the outer re-integration counts via its own callback.
    odet.total_steps += sum(p.total_steps for p in odet_proxies)

    S_at_surface_left, last_crossing_step =
        _assemble_propagators_serially!(odet, propagators, chunks, ctrl, equil, mats, intr)

    _reintegrate_outer_plasma!(odet, last_crossing_step, ctrl, equil, mats, intr)

    chunks, propagators = _handle_edge_dW_scan!(odet, chunks, propagators, ctrl, equil, mats, intr)
    resize!(S_at_surface_left, intr.msing)  # one per surviving surface, as the Δ' BVP requires

    # compute_delta_prime_matrix! runs later, after free_run supplies the vacuum edge response wv.
    if intr.verbose
        @info "Evaluating fixed-boundary stability criterion"
    end
    odet.nzero = evaluate_stability_criterion!(odet, equil.profiles)
    transform_u!(odet, intr)  # no-op when ifix=0 (no Gaussian reduction)

    return odet, propagators, chunks, S_at_surface_left
end

# Build odet and initialize at the magnetic axis. Same path as serial eulerlagrange_integration.
function _initialize_parallel_odet(ctrl::ForceFreeStatesControl,
    equil::Equilibrium.PlasmaEquilibrium,
    mats::MatrixSplines,
    intr::ForceFreeStatesInternal)
    odet = OdeState(intr.numpert_total, ctrl.algorithm.numsteps_init, ctrl.algorithm.numunorms_init, intr.msing)
    initialize_el_at_axis!(odet, ctrl, mats, equil.profiles, intr)
    # No Gaussian reduction on the Riccati path.
    odet.new = false
    fill!(odet.unorm0, 1.0)
    return odet
end

# Build the bidirectional chunk list, per-chunk propagators, and per-thread proxy OdeStates
# (sized by maxthreadid(), since threadid() can exceed nthreads() with an interactive pool).
function _setup_parallel_chunks_and_proxies(odet::OdeState, ctrl::ForceFreeStatesControl,
    intr::ForceFreeStatesInternal)
    # Crossing chunks get direction=-1: growing EL solutions decay backward, so Φ_bwd is well-conditioned.
    base_chunks = chunk_el_integration_bounds(odet, ctrl, intr; bidirectional=true)
    chunks = balance_integration_chunks(base_chunks, ctrl, intr)
    N = intr.numpert_total
    propagators = [ChunkPropagator(N) for _ in chunks]
    odet_proxies = [OdeState(N, 1, 1, 0) for _ in 1:Threads.maxthreadid()]
    return chunks, propagators, odet_proxies
end

function _log_parallel_start(odet::OdeState, equil::Equilibrium.PlasmaEquilibrium, chunks::Vector{IntegrationChunk})
    @info "   ψ = $((@sprintf "%.3f" odet.psifac)),  q = $((@sprintf "%.3f" equil.profiles.q_spline(odet.psifac)))"
    @info "   Riccati FM: $(length(chunks)) chunks over $(Threads.nthreads()) thread$(Threads.nthreads() == 1 ? "" : "s")"
end

# Integrate each chunk's propagator from identity IC in parallel; chunks are independent, so the
# result does not depend on the thread layout. :static makes threadid() a stable proxy index.
function _run_parallel_bvp_phase!(propagators::Vector{ChunkPropagator},
    chunks::Vector{IntegrationChunk},
    ctrl::ForceFreeStatesControl,
    equil::Equilibrium.PlasmaEquilibrium, mats::MatrixSplines,
    intr::ForceFreeStatesInternal,
    odet_proxies::Vector{OdeState})
    Threads.@threads :static for i in eachindex(chunks)
        integrate_propagator_chunk!(propagators[i], chunks[i], ctrl, equil, mats, intr,
            odet_proxies[Threads.threadid()])
    end
end

# Apply chunk propagators serially, renormalizing to (S, I) after each (STRIDE's ode_fixup role:
# keeps conditioning at O(cond_per_chunk) instead of its K-th power). Backward chunks are applied
# via apply_propagator_inverse!. S_at_surface_left is the Riccati S at each surface's left boundary,
# the Δ' BVP axis BC. Returns (S_at_surface_left, last_crossing_step).
function _assemble_propagators_serially!(odet::OdeState, propagators::Vector{ChunkPropagator},
    chunks::Vector{IntegrationChunk},
    ctrl::ForceFreeStatesControl,
    equil::Equilibrium.PlasmaEquilibrium,
    mats::MatrixSplines, intr::ForceFreeStatesInternal)
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

        if intr.verbose
            @info "   ψ = $((@sprintf "%.3f" odet.psifac)),  q= $((@sprintf "%.3f" odet.q)),  max(S) = $((@sprintf "%.2e" maximum(abs, odet.u[:,:,1]))),  steps = $(odet.step-1)"
        end

        if chunk.needs_crossing
            ctrl.kinetic !== nothing && error("Riccati() does not support kinetic runs; use Forward().")
            # State is (S, I) from the renorm above — well-conditioned at the surface's left boundary.
            push!(S_at_surface_left, copy(odet.u[:, :, 1]))
            riccati_cross_ideal_singular_surf!(odet, ctrl, equil, mats, intr, chunk.ising)
            last_crossing_step = odet.step - 1
        else
            # Store the chunk-endpoint state; these are Riccati states, not the EL basis.
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

# Re-integrate last rational surface → psilim with Riccati: FM propagation loses precision there at
# high N, and findmax_dW_edge! needs dense checkpoints. Starts from the pre-renorm crossing state.
function _reintegrate_outer_plasma!(odet::OdeState, last_crossing_step::Int,
    ctrl::ForceFreeStatesControl,
    equil::Equilibrium.PlasmaEquilibrium, mats::MatrixSplines,
    intr::ForceFreeStatesInternal)
    N = intr.numpert_total
    odet.u .= odet.u_store[:, :, :, last_crossing_step]
    odet.psifac = odet.psi_store[last_crossing_step]
    odet.q = odet.q_store[last_crossing_step]
    odet.step = last_crossing_step + 1
    renormalize_riccati_inplace!(odet.u, N)
    outer_chunk = IntegrationChunk(; psi_start=odet.psifac, psi_end=intr.psilim * (1 - eps),
        needs_crossing=false, ising=0)
    riccati_integrate_chunk!(odet, ctrl, equil, mats, intr, outer_chunk)
    # Post: odet.u is in (S, I) form; odet.step points to next empty slot.
end

# Edge-dW scan; on truncation, also cut the Δ' chunks and propagators back to the new edge.
function _handle_edge_dW_scan!(odet::OdeState, chunks::Vector{IntegrationChunk},
    propagators::Vector{ChunkPropagator},
    ctrl::ForceFreeStatesControl,
    equil::Equilibrium.PlasmaEquilibrium, mats::MatrixSplines,
    intr::ForceFreeStatesInternal)
    N = intr.numpert_total
    odet.step -= 1
    trim_storage!(odet)
    scan_edge_dW!(odet, ctrl, equil, mats, intr) || return chunks, propagators

    renormalize_riccati_inplace!(odet.u, N)  # stored snapshot may be pre-renorm
    rebuild = truncate_chunks!(chunks, intr.psilim, intr.msing)
    resize!(propagators, length(chunks))
    if rebuild !== nothing
        integrate_propagator_chunk!(propagators[rebuild], chunks[rebuild], ctrl, equil, mats, intr, OdeState(N, 1, 1, 0))
    end
    return chunks, propagators
end

"""
    truncate_chunks!(chunks, psi_edge, msing) -> Union{Int,Nothing}

Cut `chunks` back to an edge at `psi_edge`: drop those past it, end the straddling one at it, and
clear any crossing onto a surface no longer in the plasma. Returns the index of a chunk that was
shortened and so must be re-integrated.
"""
function truncate_chunks!(chunks::Vector{IntegrationChunk}, psi_edge::Real, msing::Int)
    last = findlast(c -> c.psi_start < psi_edge, chunks)
    last === nothing && error("truncate_chunks!: edge ψ=$psi_edge lies before all chunk starts")
    resize!(chunks, last)
    c = chunks[last]
    shortened = c.psi_end > psi_edge
    # ising <= msing means the surface survived: edge truncation drops surfaces from the outside only.
    crosses = c.needs_crossing && c.ising <= msing
    if shortened || crosses != c.needs_crossing
        chunks[last] = IntegrationChunk(; psi_start=c.psi_start, psi_end=min(c.psi_end, psi_edge),
            needs_crossing=crosses, ising=crosses ? c.ising : 0, direction=c.direction)
    end
    return shortened ? last : nothing
end
