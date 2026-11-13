"""
    riccati_eulerlagrange_integration(ctrl, equil, mats, intr) -> (odet, riccati)

Integrate the Euler-Lagrange equations from the axis to `intr.psilim` with parallel chunk propagators
[Glasser, Kolemen & Glasser 2018, Phys. Plasmas 25, 032507; Glasser & Kolemen 2018, Phys. Plasmas 25, 082502].

Returns `odet`, whose stores hold the state as (S, I), S = U₁U₂⁻¹, at every stored solver step, and `riccati`,
the inputs of `compute_delta_prime_matrix`: `(; propagators, chunks, bases, axis_state)`, with `axis_state`
(2N × N) the state entering the first crossing chunk.
"""
function riccati_eulerlagrange_integration(
    ctrl::ForceFreeStatesControl, equil::Equilibrium.PlasmaEquilibrium,
    mats::MatrixSplines, intr::ForceFreeStatesInternal
)
    odet = _initialize_parallel_odet(ctrl, equil, mats, intr)
    chunks = balance_integration_chunks(chunk_el_integration_bounds(odet, ctrl, intr; bidirectional=true), ctrl, intr)
    bases = asymptotic_bases(chunks, ctrl, equil, mats, intr)
    if intr.verbose
        @info "   ψ = $((@sprintf "%.3f" odet.psifac)),  q = $((@sprintf "%.3f" equil.profiles.q_spline(odet.psifac)))"
        @info "   Riccati FM: $(length(chunks)) chunks over $(Threads.nthreads()) thread$(Threads.nthreads() == 1 ? "" : "s")"
    end

    propagators = chunk_propagators(chunks, bases, ctrl.algorithm.eulerlagrange_tolerance, equil, mats, ModeIndices(intr))
    odet.total_steps += sum(p -> length(p.psi) - 1, propagators)
    axis_state = _assemble_propagators_serially!(odet, propagators, chunks, ctrl, equil, intr)
    _handle_edge_dW_scan!(odet, chunks, propagators, bases, ctrl, equil, mats, intr)
    compute_delta_prime_from_ca!(odet, intr, equil)

    if intr.verbose
        @info "Evaluating fixed-boundary stability criterion"
    end
    odet.nzero = evaluate_stability_criterion!(odet, equil.profiles)
    transform_u!(odet, intr)  # no-op when ifix=0 (no Gaussian reduction)

    return odet, (; propagators, chunks, bases, axis_state)
end

# OdeState at the axis start, with empty stores.
function _initialize_parallel_odet(ctrl::ForceFreeStatesControl,
    equil::Equilibrium.PlasmaEquilibrium,
    mats::MatrixSplines,
    intr::ForceFreeStatesInternal)
    odet = OdeState(intr.numpert_total, 1, 1, intr.msing)
    initialize_el_at_axis!(odet, ctrl, mats, equil.profiles, intr)
    odet.new = false
    odet.u_store_el_basis = false
    fill!(odet.unorm0, 1.0)
    return odet
end

# Every chunk's propagator, integrated across threads from inputs none of them can modify.
function chunk_propagators(chunks::Vector{IntegrationChunk}, bases::AbstractDict, rtol::Float64,
    equil::Equilibrium.PlasmaEquilibrium, mats::MatrixSplines, modes::ModeIndices)
    propagators = Vector{ChunkPropagator}(undef, length(chunks))
    nblas = BLAS.get_num_threads()
    BLAS.set_num_threads(1)
    try
        Threads.@threads for i in eachindex(chunks)
            propagators[i] = chunk_propagator(chunks[i], propagator_start(chunks, i, bases, modes.numpert_total), rtol, equil, mats, modes)
        end
    finally
        BLAS.set_num_threads(nblas)
    end
    return propagators
end

# Solver steps of a chunk whose states are stored, in increasing ψ: every `save_interval`-th, all from
# `dW_edge_scan_start` on, and the upper end. `after_crossing` adds the first, the state right of the surface.
function stored_step_indices(prop::ChunkPropagator, chunk::IntegrationChunk, after_crossing::Bool, ctrl::ForceFreeStatesControl)
    nstep = length(prop.psi)
    order = chunk.direction == 1 ? ((after_crossing ? 1 : 2):nstep) : ((nstep-1):-1:1)
    return [k for (i, k) in enumerate(order) if
            i == length(order) || i % ctrl.algorithm.save_interval == 0 || prop.psi[k] >= ctrl.dW_edge_scan_start]
end

# Multiply the chunk propagators from the axis outward, crossing each singular surface, and fill `odet`'s stores
# with the (S, I) state at every stored step. Returns the state entering the first crossing chunk (2N × N).
function _assemble_propagators_serially!(odet::OdeState, propagators::Vector{ChunkPropagator},
    chunks::Vector{IntegrationChunk}, ctrl::ForceFreeStatesControl,
    equil::Equilibrium.PlasmaEquilibrium, intr::ForceFreeStatesInternal)
    N = intr.numpert_total
    stored = [stored_step_indices(propagators[i], chunks[i], i > 1 && chunks[i-1].needs_crossing, ctrl) for i in eachindex(chunks)]
    slots = cumsum([0; length.(stored)])  # store slot before each chunk's first
    odet.psi_store = Vector{Float64}(undef, slots[end])
    odet.q_store = Vector{Float64}(undef, slots[end])
    odet.u_store = Array{ComplexF64}(undef, N, N, 2, slots[end])

    chunk_starts = Vector{Matrix{ComplexF64}}(undef, length(chunks))
    axis_state = Matrix{ComplexF64}(undef, 0, 0)
    for (i, chunk) in enumerate(chunks)
        if chunk.needs_crossing
            isempty(axis_state) && (axis_state = [odet.u[:, :, 1]; odet.u[:, :, 2]])
            apply_propagator_inverse!(odet, propagators[i])
            chunk_starts[i] = [odet.u[:, :, 1]; odet.u[:, :, 2]]
            cross_singular_surface!(odet, intr, chunk.ising)
            odet.psifac = chunks[i+1].psi_start
        else
            chunk_starts[i] = [odet.u[:, :, 1]; odet.u[:, :, 2]]
            apply_propagator!(odet, propagators[i])
            renormalize_riccati_inplace!(odet.u, N)
            odet.psifac = chunk.psi_end
        end
        if intr.verbose
            @info "   ψ = $((@sprintf "%.3f" odet.psifac)),  q= $((@sprintf "%.3f" equil.profiles.q_spline(odet.psifac))),  steps = $(slots[i+1])"
        end
    end
    odet.q = equil.profiles.q_spline(odet.psifac)

    nblas = BLAS.get_num_threads()
    BLAS.set_num_threads(1)
    try
        Threads.@threads for i in eachindex(chunks)
            prop = propagators[i]
            u = zeros(ComplexF64, N, N, 2)
            for (j, k) in enumerate(stored[i])
                @views for half in 1:2
                    mul!(u[:, :, half], prop.steps[k][:, :, half], chunk_starts[i])
                end
                renormalize_riccati_inplace!(u, N)
                slot = slots[i] + j
                odet.psi_store[slot] = prop.psi[k]
                odet.q_store[slot] = equil.profiles.q_spline(prop.psi[k])
                @views odet.u_store[:, :, :, slot] .= u
            end
            empty!(prop.psi)
            empty!(prop.steps)
        end
    finally
        BLAS.set_num_threads(nblas)
    end
    odet.step = slots[end]
    return axis_state
end

# Edge-dW scan; on truncation, also cut `chunks` and `propagators` back to the new edge.
function _handle_edge_dW_scan!(odet::OdeState, chunks::Vector{IntegrationChunk},
    propagators::Vector{ChunkPropagator}, bases::AbstractDict,
    ctrl::ForceFreeStatesControl,
    equil::Equilibrium.PlasmaEquilibrium, mats::MatrixSplines,
    intr::ForceFreeStatesInternal)
    scan_edge_dW!(odet, ctrl, equil, mats, intr) || return

    rebuild = truncate_chunks!(chunks, intr.psilim, intr.msing)
    resize!(propagators, length(chunks))
    if rebuild !== nothing
        prop = chunk_propagator(chunks[rebuild], propagator_start(chunks, rebuild, bases, intr.numpert_total),
            ctrl.algorithm.eulerlagrange_tolerance, equil, mats, ModeIndices(intr))
        propagators[rebuild] = ChunkPropagator(prop.u, Float64[], Array{ComplexF64,3}[])
    end
    return
end

"""
    truncate_chunks!(chunks, psi_edge, msing) -> Union{Int,Nothing}

Cut `chunks` back to an edge at `psi_edge`: drop those past it, end the straddling one at it, and turn a
crossing chunk whose surface is no longer in the plasma into a forward chunk. Returns the index of a chunk
that changed and so must be re-integrated.
"""
function truncate_chunks!(chunks::Vector{IntegrationChunk}, psi_edge::Real, msing::Int)
    last = findlast(c -> c.psi_start < psi_edge, chunks)
    last === nothing && error("truncate_chunks!: edge ψ=$psi_edge lies before all chunk starts")
    resize!(chunks, last)
    c = chunks[last]
    # Truncation drops surfaces from the outside, so ising <= msing means the surface survived.
    crosses = c.needs_crossing && c.ising <= msing
    (c.psi_end > psi_edge || crosses != c.needs_crossing) || return nothing
    chunks[last] = IntegrationChunk(; psi_start=c.psi_start, psi_end=min(c.psi_end, psi_edge),
        needs_crossing=crosses, ising=crosses ? c.ising : 0, direction=crosses ? c.direction : 1)
    return last
end
