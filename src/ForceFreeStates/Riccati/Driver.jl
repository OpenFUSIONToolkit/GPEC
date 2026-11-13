"""
    riccati_eulerlagrange_integration(ctrl, equil, mats, intr) -> (odet, riccati)

Integrate the Euler-Lagrange equations from the axis to `intr.psilim` with parallel chunk propagators
[Glasser, Kolemen & Glasser 2018, Phys. Plasmas 25, 032507; Glasser & Kolemen 2018, Phys. Plasmas 25, 082502].

Returns `odet`, whose stores hold the solution at every stored solver step in columns that are continuous in ψ
and equal (S, I), S = U₁U₂⁻¹, at the edge, and `riccati`, the inputs of `compute_delta_prime_matrix`:
`(; propagators, chunks, columns)`, with `columns` from `_assemble_propagators_serially!`.
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

    rtol = ctrl.algorithm.eulerlagrange_tolerance
    propagators = chunk_propagators(chunks, bases, rtol, equil, mats, ModeIndices(intr))
    odet.total_steps += sum(p -> length(p.psi) - 1, propagators)
    columns = _assemble_propagators_serially!(odet, propagators, chunks, ctrl, equil, intr)

    if scan_edge_dW!(odet, ctrl, equil, mats, intr)
        # The edge moved in: shorten the chunk list and multiply the propagators again over the new domain.
        rebuild = truncate_chunks!(chunks, intr.psilim, intr.msing)
        resize!(propagators, length(chunks))
        if rebuild !== nothing
            start = propagator_start(chunks, rebuild, bases, intr.numpert_total)
            propagators[rebuild] = chunk_propagator(chunks[rebuild], start, rtol, equil, mats, ModeIndices(intr))
        end
        fill!(odet.u, 0)
        initialize_el_at_axis!(odet, ctrl, mats, equil.profiles, intr)
        columns = _assemble_propagators_serially!(odet, propagators, chunks, ctrl, equil, intr)
    end
    compute_delta_prime_from_ca!(odet, intr, equil)

    if intr.verbose
        @info "Evaluating fixed-boundary stability criterion"
    end
    odet.nzero = evaluate_stability_criterion!(odet, equil.profiles)
    restore_solution_columns!(odet, propagators, chunks, columns)

    return odet, (; propagators, chunks, columns)
end

# OdeState at the axis start, with empty stores.
function _initialize_parallel_odet(ctrl::ForceFreeStatesControl,
    equil::Equilibrium.PlasmaEquilibrium,
    mats::MatrixSplines,
    intr::ForceFreeStatesInternal)
    odet = OdeState(intr.numpert_total, 1, 1, intr.msing)
    initialize_el_at_axis!(odet, ctrl, mats, equil.profiles, intr)
    odet.new = false
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
# with the (S, I) state at every stored step. Returns what later stages need to rebuild solutions from the
# propagators: `axis_state` (2N × N, the state entering the first crossing chunk), `starts` (per chunk, the
# 2N × N coefficients its propagator multiplies), `stored` and `slots` (per chunk, the stored solver steps and the
# store slot before its first), and `continuations` (per crossing chunk, from `cross_singular_surface!`).
function _assemble_propagators_serially!(odet::OdeState, propagators::Vector{ChunkPropagator},
    chunks::Vector{IntegrationChunk}, ctrl::ForceFreeStatesControl,
    equil::Equilibrium.PlasmaEquilibrium, intr::ForceFreeStatesInternal)
    N = intr.numpert_total
    stored = [stored_step_indices(propagators[i], chunks[i], i > 1 && chunks[i-1].needs_crossing, ctrl) for i in eachindex(chunks)]
    slots = cumsum([0; length.(stored)])
    odet.step = slots[end]
    odet.psi_store = Vector{Float64}(undef, odet.step)
    odet.q_store = Vector{Float64}(undef, odet.step)
    odet.u_store = Array{ComplexF64}(undef, N, N, 2, odet.step)

    starts = Vector{Matrix{ComplexF64}}(undef, length(chunks))
    continuations = Dict{Int,Matrix{ComplexF64}}()
    axis_state = Matrix{ComplexF64}(undef, 0, 0)
    for (i, chunk) in enumerate(chunks)
        if chunk.needs_crossing
            isempty(axis_state) && (axis_state = [odet.u[:, :, 1]; odet.u[:, :, 2]])
            apply_propagator_inverse!(odet, propagators[i])
            starts[i] = [odet.u[:, :, 1]; odet.u[:, :, 2]]
            continuations[i] = cross_singular_surface!(odet, intr, chunk.ising)
            odet.psifac = chunks[i+1].psi_start
        else
            starts[i] = [odet.u[:, :, 1]; odet.u[:, :, 2]]
            apply_propagator!(odet, propagators[i])
            renormalize_riccati_inplace!(odet.u, N)
            odet.psifac = chunk.psi_end
        end
        if intr.verbose
            @info "   ψ = $((@sprintf "%.3f" odet.psifac)),  q= $((@sprintf "%.3f" equil.profiles.q_spline(odet.psifac))),  steps = $(slots[i+1])"
        end
    end
    odet.q = equil.profiles.q_spline(odet.psifac)
    for (i, prop) in enumerate(propagators), (j, k) in enumerate(stored[i])
        odet.psi_store[slots[i]+j] = prop.psi[k]
        odet.q_store[slots[i]+j] = equil.profiles.q_spline(prop.psi[k])
    end
    store_propagated!(odet.u_store, propagators, starts, stored, slots; reduce=true)
    return (; axis_state, starts, stored, slots, continuations)
end

# Fill `u_store` with each chunk's propagator at its stored steps times that chunk's `coefficients` (2N × N),
# reduced to (S, I) when `reduce`.
function store_propagated!(u_store::Array{ComplexF64,4}, propagators::Vector{ChunkPropagator},
    coefficients::Vector{Matrix{ComplexF64}}, stored::Vector{Vector{Int}}, slots::Vector{Int}; reduce::Bool)
    N = size(u_store, 1)
    nblas = BLAS.get_num_threads()
    BLAS.set_num_threads(1)
    try
        Threads.@threads for i in eachindex(propagators)
            u = zeros(ComplexF64, N, N, 2)
            for (j, k) in enumerate(stored[i])
                @views for half in 1:2
                    mul!(u[:, :, half], propagators[i].steps[k][:, :, half], coefficients[i])
                end
                reduce && renormalize_riccati_inplace!(u, N)
                @views u_store[:, :, :, slots[i]+j] .= u
            end
        end
    finally
        BLAS.set_num_threads(nblas)
    end
end

# Rewrite `odet.u_store` from (S, I) into solution columns that are continuous in ψ and equal (S, I) at the edge:
# what `transform_u!` leaves for the forward integrator.
function restore_solution_columns!(odet::OdeState, propagators::Vector{ChunkPropagator},
    chunks::Vector{IntegrationChunk}, columns)
    (; starts, stored, slots, continuations) = columns
    N = size(odet.u, 1)
    edge_columns = Matrix{ComplexF64}(I, N, N)   # each chunk's start columns in terms of the edge columns
    coefficients = similar(starts)
    for i in reverse(eachindex(chunks))
        if chunks[i].needs_crossing
            edge_columns = continuations[i] * edge_columns
        else
            chunk_end = [propagators[i].u[:, :, 1]; propagators[i].u[:, :, 2]] * starts[i]
            edge_columns = chunk_end[(N+1):(2N), :] \ edge_columns
        end
        coefficients[i] = starts[i] * edge_columns
    end
    store_propagated!(odet.u_store, propagators, coefficients, stored, slots; reduce=false)
end

"""
    truncate_chunks!(chunks, psi_edge, msing) -> Union{Int,Nothing}

Cut `chunks` back to an edge at `psi_edge`: drop those past it, end the straddling one at it, and turn a
crossing chunk whose surface is no longer in the plasma into a forward chunk. An edge at the right boundary of
a surface keeps the chunk after its crossing, at zero width. Returns the index of a chunk that changed and so
must be re-integrated.
"""
function truncate_chunks!(chunks::Vector{IntegrationChunk}, psi_edge::Real, msing::Int)
    last = findlast(i -> chunks[i].psi_start < psi_edge || (i > 1 && chunks[i-1].needs_crossing && chunks[i].psi_start == psi_edge),
        eachindex(chunks))
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
