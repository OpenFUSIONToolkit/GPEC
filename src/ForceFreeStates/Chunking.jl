# Chunking.jl — domain partitioning at rational surfaces and load-balanced
# sub-division for the chunked-Riccati and legacy Euler-Lagrange paths.

"""
    ode_itime_cost(psi1, psi2, intr) -> Float64

Estimate the relative ODE integration cost for the interval [ψ₁, ψ₂] using the
empirical log-divergent cost model from STRIDE (Glasser 2018).

The cost is a sum of logarithmic contributions from reference points:
  - Magnetic axis (ψ_ref = 0): steep divergence, (a,b) = (39695, 212830)
  - Each rational surface (ψ_ref = ψ_s): moderate divergence, (a,b) = (17147, 470710)
  - Edge (ψ_ref = ψ_lim): mild divergence, (a,b) = (1646, 4683)

For each reference: cost += (a/b) * |log(1 + b|ψ₂-ref|) - log(1 + b|ψ₁-ref|)|

The cost model is additive for sub-intervals not containing rational surfaces,
which makes it suitable for equal-cost splitting via bisection.
"""
function ode_itime_cost(psi1::Float64, psi2::Float64, intr::ForceFreeStatesInternal)
    a_ax, b_ax = 39695.0, 212830.0
    a_rat, b_rat = 17147.0, 470710.0
    a_edge, b_edge = 1646.0, 4683.0

    cost = (a_ax / b_ax) * abs(log(1.0 + b_ax * abs(psi2)) - log(1.0 + b_ax * abs(psi1)))

    for sing in intr.sing
        ref = sing.psifac
        cost += (a_rat / b_rat) * abs(log(1.0 + b_rat * abs(psi2 - ref)) - log(1.0 + b_rat * abs(psi1 - ref)))
    end

    ref_edge = intr.psilim
    cost += (a_edge / b_edge) * abs(log(1.0 + b_edge * abs(psi2 - ref_edge)) - log(1.0 + b_edge * abs(psi1 - ref_edge)))

    return cost
end

"""
    balance_integration_chunks(chunks, ctrl, intr) -> Vector{IntegrationChunk}

Sub-divide integration chunks to produce a load-balanced set for parallel execution.
Starts from the output of `chunk_el_integration_bounds` and iteratively splits the
highest-cost chunk (by `ode_itime_cost`) until the total chunk count reaches
`max(2*msing + 3, 4 * Threads.nthreads())`.

Each split finds the equal-cost midpoint ψ_mid via bisection:
  ode_itime_cost(psi_start, psi_mid) ≈ ode_itime_cost(psi_start, psi_end) / 2

Sub-chunks inherit `needs_crossing=false` and `ising=0`. Only the LAST sub-chunk of
each original chunk retains `needs_crossing=true` and the original `ising`, so the
rational surface crossing still fires at the correct ψ in the serial assembly phase.
"""
function balance_integration_chunks(chunks::Vector{IntegrationChunk}, ctrl::ForceFreeStatesControl, intr::ForceFreeStatesInternal)
    min_chunks = 2 * intr.msing + 3
    # Ensure enough sub-chunks for BVP propagator conditioning: at least 5 non-crossing
    # sub-chunks per segment (axis→surf₁, surfᵢ→surfᵢ₊₁, surfₙ→edge), plus crossing
    # chunks. STRIDE uses 33 intervals for comparable problems. Without enough sub-chunks,
    # assemble_fm_matrix(condition=true) can't keep accumulated products well-conditioned
    # because single long-span propagators may already have cond ~ 10²⁴.
    min_bvp_intervals = 8 * (intr.msing + 1) + intr.msing
    target_n = max(min_chunks, 4 * Threads.nthreads(), min_bvp_intervals)

    result = collect(chunks)

    while length(result) < target_n
        # Find the highest-cost splittable chunk
        best_idx = 0
        best_cost = -Inf
        for (i, chunk) in enumerate(result)
            width = chunk.psi_end - chunk.psi_start
            if width > 1e-8
                c = ode_itime_cost(chunk.psi_start, chunk.psi_end, intr)
                if c > best_cost
                    best_cost = c
                    best_idx = i
                end
            end
        end

        best_idx == 0 && break  # No more splittable chunks

        chunk = result[best_idx]
        total_cost = best_cost
        target_cost = total_cost / 2.0

        # Bisect to find ψ_mid where cost(psi_start, ψ_mid) ≈ target_cost
        lo, hi = chunk.psi_start, chunk.psi_end
        for _ in 1:50
            mid = (lo + hi) / 2.0
            if ode_itime_cost(chunk.psi_start, mid, intr) < target_cost
                lo = mid
            else
                hi = mid
            end
        end
        psi_mid = (lo + hi) / 2.0

        left = IntegrationChunk(; psi_start=chunk.psi_start, psi_end=psi_mid,
                                  needs_crossing=false, ising=0, direction=1)
        right = IntegrationChunk(; psi_start=psi_mid, psi_end=chunk.psi_end,
                                   needs_crossing=chunk.needs_crossing, ising=chunk.ising,
                                   direction=chunk.direction)
        splice!(result, best_idx, [left, right])
    end

    return result
end

"""
    chunk_el_integration_bounds(odet::OdeState, ctrl::ForceFreeStatesControl, intr::ForceFreeStatesInternal)

Pre-compute all integration chunks from the current position to the edge.
Returns a vector of `IntegrationChunk` objects, each representing a region to integrate
and whether it needs a rational surface crossing beforehand.

This function replaces the iterative while-loop logic with a single upfront computation,
making the integration flow more predictable and easier to parallelize (e.g., for STRIDE).

### Arguments

  - `odet::OdeState` - ODE state struct (starting position and singular surface index)
  - `ctrl::ForceFreeStatesControl` - Control parameters
  - `intr::ForceFreeStatesInternal` - Internal data (singular surfaces, limits)

### Returns

  - `Vector{IntegrationChunk}` - Array of integration chunks to process
"""
function chunk_el_integration_bounds(odet::OdeState, ctrl::ForceFreeStatesControl, intr::ForceFreeStatesInternal; bidirectional::Bool=false)
    chunks = IntegrationChunk[]

    # Start from current position
    psi_current = odet.psifac
    ising_current = odet.ising_start

    # Wrapper to find next singular surface to integrate toward that is resonant within integration limits
    function find_next_resonant_surface!(ising::Int, intr::ForceFreeStatesInternal)
        ising += 1
        while ising <= intr.msing
            if intr.psilim < intr.sing[ising].psifac ||
               any(m -> intr.mlow <= m <= intr.mhigh, intr.sing[ising].m)
                break
            end
            ising += 1
        end
        return ising
    end

    # -------------------- Create chunks ------------------------
    if ctrl.kinetic_factor > 0
        # Single chunk from axis to edge. Kinetic contributions regularize the
        # F-matrix singularity at rational surfaces (Logan 2015 Eq 7.46), making
        # them integrable. The adaptive ODE solver handles stiffness automatically.
        push!(chunks, IntegrationChunk(;
            psi_start=psi_current,
            psi_end=(intr.psilim * (1 - eps)),
            needs_crossing=false,
            ising=0
        ))
    else
        # Loop through singular surfaces to cross until edge is reached
        ising_current = find_next_resonant_surface!(ising_current, intr)
        while ising_current <= intr.msing && intr.psilim >= intr.sing[ising_current].psifac && ctrl.singfac_min != 0
            # Set integration limit to just before the next singular surface
            psi_end = intr.sing[ising_current].psifac - ctrl.singfac_min /
                                                        abs(minimum(intr.sing[ising_current].n) * intr.sing[ising_current].q1)

            # Validate chunk bounds
            @assert psi_current < psi_end "Invalid chunk bounds: psi_start=$psi_current >= psi_end=$psi_end"
            @assert isempty(chunks) || psi_current >= chunks[end].psi_end "Overlapping chunks detected"

            push!(chunks, IntegrationChunk(;
                psi_start=psi_current,
                psi_end=psi_end,
                needs_crossing=true,
                ising=ising_current,
                direction = bidirectional ? -1 : 1
            ))

            # After crossing, we jump to the other side of the singular surface
            dpsi = intr.sing[ising_current].psifac - psi_end
            psi_current = psi_end + 2 * dpsi

            # Move to next singular surface that is either resonant or beyond integration limits
            ising_current = find_next_resonant_surface!(ising_current, intr)
        end

        # No more singular surfaces to cross, set integration limit to edge
        @assert psi_current < intr.psilim * (1 - eps) "Final chunk has invalid bounds"
        @assert isempty(chunks) || psi_current >= chunks[end].psi_end "Final chunk overlaps with previous chunk"

        push!(chunks, IntegrationChunk(;
            psi_start=psi_current,
            psi_end=(intr.psilim * (1 - eps)),
            needs_crossing=false,
            ising=0
        ))
    end

    return chunks
end

