"""
    compute_delta_prime_from_ca!(odet, intr, equil)

Compute the tearing stability parameter Δ' for each singular surface from the
asymptotic coefficients `ca_l` and `ca_r` accumulated during integration.

Uses the diagonal formula Δ'[i] = (ca_r[i,i,2,s] - ca_l[i,i,2,s]) / (4π² · psio),
which is correct when the small asymptotic was introduced in column `ipert_res` directly
(no GR permutation).

**Note**: This function is no longer called from any integration driver. Δ' is now computed
inline inside each crossing function where the correct column index is known:
- `cross_ideal_singular_surf!` uses `perm_col` (GR-permuted column)
- `riccati_cross_ideal_singular_surf!` uses the diagonal `ipert_res` (no GR permutation)

Retained for reference and potential use in testing.

This matches the formula in `PerturbedEquilibrium/SingularCoupling.jl` (lines ~197):
  `delta_prime_val = (rbwp1 - lbwp1) / (twopi * chi1)`
with `chi1 = 2π·psio`, so the denominators are identical.
"""
function compute_delta_prime_from_ca!(odet::OdeState, intr::ForceFreeStatesInternal, equil::Equilibrium.PlasmaEquilibrium)
    denom = (2π)^2 * equil.psio  # = twopi * chi1 in SingularCoupling.jl
    for s in 1:intr.msing
        sing = intr.sing[s]
        n_modes = length(sing.m)
        resize!(intr.sing[s].delta_prime, n_modes)
        for i in 1:n_modes
            ipert_res = 1 + sing.m[i] - intr.mlow + (sing.n[i] - intr.nlow) * intr.mpert
            if 1 <= ipert_res <= intr.numpert_total
                Δca = odet.ca_r[ipert_res, ipert_res, 2, s] - odet.ca_l[ipert_res, ipert_res, 2, s]
                intr.sing[s].delta_prime[i] = Δca / denom
            else
                intr.sing[s].delta_prime[i] = 0.0 + 0.0im
            end
        end
    end
end

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
    eulerlagrange_integration(ctrl::ForceFreeStatesControl, equil::Equilibrium.PlasmaEquilibrium, ffit::FourFitVars, intr::ForceFreeStatesInternal)

Main driver for integrating the Euler-Lagrange equations across the plasma and detecting singular surfaces.
Formerly `ode_run`. Has the same functionality as `ode_run` in the Fortran code, with the addition of
a single dump to the `euler.h5` file at the end of integration instead of multiple dumps
to `euler.bin` throughout the integration. We have made the control logic more clear
by pre-computing all integration chunks upfront and using a for loop to iterate through them,
eliminating the while-loop logic and making integration bounds explicit at each step.
We now perform significant post-processing after integration including finding the peak dW
in the edge region and evaluating the stability criterion over the entire integration,
which were previously done during integration in the Fortran code.

### TODOs

restype functionality if we decide to do this

### Returns

An OdeState struct containing the final state of the ODE solver after integration is complete.
"""
function eulerlagrange_integration(ctrl::ForceFreeStatesControl, equil::Equilibrium.PlasmaEquilibrium, ffit::FourFitVars, intr::ForceFreeStatesInternal)

    # Dispatch. Both the parallel-FM and serial-Riccati paths are served by
    # parallel_eulerlagrange_integration — it returns the canonical GR-axis-basis u_store
    # that PerturbedEquilibrium consumes, plus (propagators, chunks, S_at_surface_left) for
    # the deferred Δ' BVP. use_riccati routes through the same machinery with the chunk
    # integration forced single-threaded (parallel_threads = 1).
    if ctrl.use_parallel
        return parallel_eulerlagrange_integration(ctrl, equil, ffit, intr)
    elseif ctrl.use_riccati
        saved_threads = ctrl.parallel_threads
        ctrl.parallel_threads = 1
        try
            return parallel_eulerlagrange_integration(ctrl, equil, ffit, intr)
        finally
            ctrl.parallel_threads = saved_threads
        end
    end

    # Initialization
    odet = OdeState(intr.numpert_total, ctrl.numsteps_init, ctrl.numunorms_init, intr.msing)
    if ctrl.sing_start <= 0
        initialize_el_at_axis!(odet, ctrl, equil.profiles, intr)
    elseif ctrl.sing_start <= intr.msing
        error("sing_start > 0 not implemented yet!")
        # initialize_el_at_singular_surf!(ctrl, equil, intr, odet)
    else
        error("Invalid value for sing_start: $(ctrl.sing_start) > msing = $(intr.msing)")
    end

    # Pre-compute all integration chunks
    chunks = chunk_el_integration_bounds(odet, ctrl, intr)

    # Print initial integration condition
    if ctrl.verbose
        @info "   ψ = $((@sprintf "%.3f" odet.psifac)),  q = $((@sprintf "%.3f" equil.profiles.q_spline(odet.psifac)))"
    end

    # Iterate through each integration chunk
    for chunk in chunks
        # Integrate this region and display progress
        integrate_el_region!(odet, ctrl, equil, ffit, intr, chunk)
        if ctrl.verbose
            @info "   ψ = $((@sprintf "%.3f" odet.psifac)),  q = $((@sprintf "%.3f" odet.q)),  steps = $(odet.total_steps)"
        end

        # Cross a rational surface after integration if this chunk requires it
        if chunk.needs_crossing
            # Ideal surface crossings apply only in the ideal (non-kinetic) path.
            cross_ideal_singular_surf!(odet, ctrl, equil, ffit, intr, chunk.ising)
        end
    end

    finalize_canonical_u_store!(odet, ctrl, equil, ffit, intr)

    return (odet, nothing, nothing, nothing)
end

"""
    finalize_canonical_u_store!(odet, ctrl, equil, ffit, intr)

Shared post-integration finalization that produces the canonical `u_store` — the eigenmode
radial-displacement fundamental matrix ξ_ψ in the Gaussian-reduction axis basis that
`PerturbedEquilibrium` consumes. Called by every integration path (standard EL directly;
the parallel-FM path after its GR-based dense reconstruction) so all paths converge on the
identical representation.

Performs, in order: trim unused storage; the edge-dW scan over `[psiedge, psilim]`
(`findmax_dW_edge!`, with `truncate_at_dW_peak` handling); the fixed-boundary stability
criterion; and `transform_u!`, which composes the recorded Gaussian-reduction transforms
to rotate `u_store`/`ud_store` into the axis basis. On entry `odet.step` is one past the
last filled index (the integration/reconstruction convention).
"""
function finalize_canonical_u_store!(
    odet::OdeState, ctrl::ForceFreeStatesControl, equil::Equilibrium.PlasmaEquilibrium,
    ffit::FourFitVars, intr::ForceFreeStatesInternal
)
    # `odet.step` was incremented one past the last filled index; point it at the last.
    odet.step -= 1
    trim_storage!(odet)

    # Edge-dW scan over [psiedge, psilim] — populates odet.edge_scan for HDF5 output.
    # The scan mutates odet.psifac and odet.u internally; save/restore them around the call.
    #
    # Default (ctrl.truncate_at_dW_peak = false): diagnostic-only. Integration domain is
    # determined solely by qhigh / psihigh / dmlim so Δ' and δW are independent of peak
    # location. Legacy path (true) reproduces the ode_record_edge heuristic from Fortran
    # STRIDE — psilim/qlim/u are pulled back to the dW peak. Preserved for experimental
    # work; see docstring in ForceFreeStatesStructs.jl for the reliability caveats.
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
                @info "Truncating integration at peak edge dW (LEGACY — Δ'/δW unreliable): ψ = $((@sprintf "%.3f" odet.psi_store[odet.step])),  q = $((@sprintf "%.3f" odet.q_store[odet.step]))"
            end
        else
            odet.psifac = saved_psifac
            odet.u .= saved_u
            if ctrl.verbose
                @info "Edge-dW peak (diagnostic): ψ = $((@sprintf "%.3f" odet.psi_store[peak_step])),  q = $((@sprintf "%.3f" odet.q_store[peak_step])); integration domain unchanged"
            end
        end
    end

    # Evaluate stability criterion (critical determinant) of saved solutions
    if ctrl.verbose
        @info "Evaluating fixed-boundary stability criterion"
    end
    odet.nzero = evaluate_stability_criterion!(odet, equil.profiles)

    # Undo Gaussian reduction to get true solution vectors (for free_run! eigenvector use)
    transform_u!(odet, intr)
    return odet
end

"""
    initialize_el_at_axis!(odet::OdeState, ctrl::ForceFreeStatesControl, profiles::Equilibrium.ProfileSplines, intr::ForceFreeStatesInternal)

Initialize the OdeState struct for the case of sing_start = 0 (axis initialization).
Formerly `ode_axis_init!`. This now only initializes `psifac`, `ising_start`, and `u`.

### TODOs

Move ising_start logic to chunk_el_integration_bounds?
"""
function initialize_el_at_axis!(odet::OdeState, ctrl::ForceFreeStatesControl, profiles::Equilibrium.ProfileSplines, intr::ForceFreeStatesInternal)

    # Default psifac to minimum equilibrium psi value
    odet.psifac = profiles.xs[1]

    # Use Newton iteration to find starting psi if qlow is above q0
    if ctrl.qlow > profiles.q_spline.y[1]
        # Find last index where q < qlow
        idx = findlast(jpsi -> profiles.q_spline.y[jpsi-1] < ctrl.qlow, 2:profiles.npts)
        if idx !== nothing
            odet.psifac = profiles.xs[idx]
        end
        odet.psifac = find_zero(
            (psi -> profiles.q_spline(psi) - ctrl.qlow,
                psi -> profiles.q_deriv(psi)),
            odet.psifac, Roots.Newton()
        )
    end

    # Find starting singular surface (where sing.psifac > psi(qlow/q0))
    # Note: This logic is kept in initialize_el_at_axis! rather than chunk_el_integration_bounds
    # because it depends on the starting psifac which is set here. The logic for sing_start != 0
    # and kinetic mode would also live here when implemented.
    if ctrl.kinetic_factor > 0
        # No singular surface tracking needed — kinetic terms regularize singularities
        odet.ising_start = 0
    else
        odet.ising_start = searchsortedfirst(getfield.(intr.sing, :psifac), odet.psifac) - 1
    end

    # Initialize solutions with the identity matrix for U_22 [Glasser Phys. Plasmas 2016 112506 Section VI]
    for ipert in 1:intr.numpert_total
        odet.u[ipert, ipert, 2] = 1
    end
end

# TODO: NOT IMPLEMENTED YET! (low priority, just make sure sing_start = 0 in toml)
function initialize_el_at_singular_surf()
    return
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

"""
    cross_ideal_singular_surf!(odet::OdeState, ctrl::ForceFreeStatesControl, equil::Equilibrium.PlasmaEquilibrium, ffit::FourFitVars, intr::ForceFreeStatesInternal)

Handle the crossing of a rational surface during integration if kinetic mode is disabled.
Formerly `ode_ideal_cross!`. Performs the same function as `ode_ideal_cross` in the Fortran code.
Differences mainly in integration data storage logic, but otherwise identical. It normalizes and
reinitializes the solution vector at the singularity, and updates relevant state variables.

Asymptotics are now computed on-demand here instead of being pre-computed, making it clear
that asymptotic calculations are specific to ideal ForceFreeStates and not inherent to the singular surface.

### Arguments

  - `ising::Int` - Index of the singular surface being crossed
"""
function cross_ideal_singular_surf!(
    odet::OdeState,
    ctrl::ForceFreeStatesControl,
    equil::Equilibrium.PlasmaEquilibrium,
    ffit::FourFitVars,
    intr::ForceFreeStatesInternal,
    ising::Int
)

    # Fixup solution at singular surface
    compute_solution_norms!(odet.u, odet, ctrl, intr, true)

    # Compute direction-specific asymptotic power series for this singular surface
    singp = intr.sing[ising]
    sing_asymp_right = compute_sing_asymptotics(singp, ctrl, equil, ffit, intr; sig=1.0)
    sing_asymp_left = compute_sing_asymptotics(singp, ctrl, equil, ffit, intr; sig=-1.0, alpha_override=sing_asymp_right.alpha)
    dpsi = singp.psifac - odet.psifac # ψ_res - ψ (positive)

    # Get asymptotic coefficients before crossing (left side)
    ua = sing_get_ua(sing_asymp_left, dpsi)
    odet.ca_l[:, :, :, ising] .= sing_get_ca(odet.u, ua, intr)

    # Single n: remove largest solution and sub in asymptotics on the other side
    # Multi-n: if we remove the N largest modes in arbitrary order, we can mess up the
    # diagonal structure of the matrix and later calculations. zeroed_idx let's us make sure
    # the solution vector we're zeroing corresponds to the same block as the resonant mode we
    # introduce. It is also needed when transforming u back to the full solution after integration.
    ipert_res = 1 .+ singp.m .- intr.mlow .+ (singp.n .- intr.nlow) .* intr.mpert
    if ctrl.kinetic_factor == 0
        # Eliminate the solution with the largest norm (in the same block) for each resonance
        odet.zeroed_idx[odet.ifix] = Int[]
        for i in eachindex(sing_asymp_right.r1)
            push!(odet.zeroed_idx[odet.ifix], findfirst(j -> (ipert_res[i] - 1) ÷ intr.mpert == (odet.index[j, odet.ifix] - 1) ÷ intr.mpert, 1:intr.numpert_total))
            odet.u[:, odet.index[odet.zeroed_idx[odet.ifix][i], odet.ifix], :] .= 0
        end
    end

    # Re-initialize on opposite side of rational surface by approximating solution
    params = (ctrl, equil, ffit, intr, odet, IntegrationChunk(0.0, 0.0, false, ising, 1))
    du1 = zeros(ComplexF64, intr.numpert_total, intr.numpert_total, 2)
    du2 = zeros(ComplexF64, intr.numpert_total, intr.numpert_total, 2)
    sing_der!(du1, odet.u, params, odet.psifac)
    odet.psifac += 2 * dpsi # jump to other side of singular surface
    sing_der!(du2, odet.u, params, odet.psifac)
    odet.u .+= (du1 .+ du2) .* dpsi

    # Apply asymptotic solution on other side of singular surface (right side)
    ua = sing_get_ua(sing_asymp_right, dpsi)
    if ctrl.kinetic_factor == 0
        for i in eachindex(sing_asymp_right.r1)
            # Zero out the resonant components
            odet.u[ipert_res[i], :, :] .= 0
            # Introduce the small asymptotic resonant solution on the other side of the singular surface
            odet.u[:, odet.index[odet.zeroed_idx[odet.ifix][i], odet.ifix], :] .= ua[:, ipert_res[i]+intr.numpert_total, :]
        end
    end
    # Get asymptotic coefficients after crossing rational surface
    odet.ca_r[:, :, :, ising] .= sing_get_ca(odet.u, ua, intr)

    # Δ' is NOT computed here in the standard path. Δ' is normalization-convention-dependent
    # and requires the Riccati gauge (U₂=I); the standard path lacks that gauge. Δ' is computed
    # in riccati_cross_ideal_singular_surf! for the Riccati and parallel-FM paths instead.

    # Recompute ud from the final post-crossing u so ud_store is consistent with u_store.
    # The earlier sing_der! calls operated on the pre-trapezoidal / pre-asymptotic u and left
    # odet.ud stale; without this, ud_store diverges from u_store after rational crossings.
    sing_der!(du1, odet.u, params, odet.psifac)

    # Store values after crossing step and advance
    odet.psi_store[odet.step] = odet.psifac
    odet.q_store[odet.step] = odet.q
    odet.u_store[:, :, :, odet.step] = odet.u
    odet.ud_store[:, :, :, odet.step] = odet.ud
    odet.step += 1
end

"""
    integrate_el_region!(odet::OdeState, ctrl::ForceFreeStatesControl, equil::Equilibrium.PlasmaEquilibrium, ffit::FourFitVars, intr::ForceFreeStatesInternal, chunk::IntegrationChunk)

Integrate the Euler-Lagrange equations from `psi_start` to `psi_end`.
Formerly `ode_step!`. Performs the same function as `ode_step` in the Fortran code, with the addition of
a callback function to handle tolerances, normalization, and storage at each
step of the integration. In Fortran, this was performed by running LSODE in one-step
mode (so ode_step was called hundreds of times) and calling the relevant functions in
a DO loop. Here, we use the DifferentialEquations.jl interface to achieve the same
functionality in a more Julian way. The integration bounds are now explicit arguments,
making it clear what region is being integrated.

### Arguments

  - `odet::OdeState` - ODE state struct (modified in-place)
  - `ctrl::ForceFreeStatesControl` - Control parameters
  - `equil::Equilibrium.PlasmaEquilibrium` - Plasma equilibrium
  - `ffit::FourFitVars` - Fourier fit variables
  - `intr::ForceFreeStatesInternal` - Internal data
  - `chunk::IntegrationChunk` - Integration chunk containing start and end ψ for integration

### TODOs

Check sensitivity of results to tolerances, currently using same logic as Fortran
Check absolute tolerances, currently only relative tolerances are updated
"""
function integrate_el_region!(
    odet::OdeState,
    ctrl::ForceFreeStatesControl,
    equil::Equilibrium.PlasmaEquilibrium,
    ffit::FourFitVars,
    intr::ForceFreeStatesInternal,
    chunk::IntegrationChunk
)

    # Fraction of the q-range defining "near boundary" dense-save zones at each end of
    # a segment. TODO: expose as a ctrl field when a good default is validated.
    near_q_frac = 0.05

    # q at segment boundaries — used for symmetric near-boundary heuristic.
    # odet.q is updated at every step inside sing_der!, so we compare against these
    # fixed endpoints in the callback rather than using psi-based distances.
    q_start = equil.profiles.q_spline(chunk.psi_start)
    q_end = equil.profiles.q_spline(chunk.psi_end)
    q_range = abs(q_end - q_start)

    steps_in_segment = Ref(0)
    du_buffer = zeros(ComplexF64, intr.numpert_total, intr.numpert_total, 2)

    # save_callback fires after every accepted step. It does NOT fire Gaussian reduction
    # any more — that is delegated to gr_continuous_callback below, which catches the
    # exact ψ where `uratio - ucrit` crosses zero (partition-invariant by construction).
    # save_callback only:
    #   - bumps step counters
    #   - initializes unorm0 the first time after a GR fixup (or chunk start)
    #   - writes saved snapshots to u_store / ud_store at the existing heuristic
    function save_callback!(integrator)
        ctrl, _, _, intr, odet, chunk = integrator.p
        odet.total_steps += 1
        steps_in_segment[] += 1

        # First sample of a new GR-interval: capture the reference norms used by the
        # ContinuousCallback condition. Without this we would compare against stale
        # unorm0 from before the most recent fixup.
        if odet.new
            odet.new = false
            @views odet.unorm0 .= norm.(eachcol(integrator.u[:, :, 1]))
        end

        # Save near segment boundaries (symmetric, in q not psi) and every Nth step.
        # The step-count fallback (== 1) guarantees the first step is always saved
        # even for near-degenerate segments where q_range ≈ 0.
        near_start = abs(odet.q - q_start) < near_q_frac * q_range || steps_in_segment[] == 1
        near_end = abs(odet.q - q_end) < near_q_frac * q_range
        # Always save in the edge scan region so findmax_dW_edge! has dense q coverage.
        in_edge_scan = ctrl.psiedge < intr.psilim && integrator.t >= ctrl.psiedge

        if near_start || near_end || (odet.total_steps % ctrl.save_interval == 0) || in_edge_scan
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

    # ContinuousCallback condition: `max(unorm) / min(unorm) - ucrit`, evaluated against
    # the (frozen) unorm0 set by save_callback at the start of the current GR-interval.
    # Returns a large negative sentinel while odet.new=true (i.e., unorm0 not yet initialized)
    # so the bisection root-find never trips in that state.
    function gr_condition(u, t, integrator)
        ctrl, _, _, intr, odet, _ = integrator.p
        odet.new && return -1.0
        norms = norm.(eachcol(u[:, :, 1]))
        mn = minimum(norms)
        mn == 0 && return -1.0
        ratios = norms ./ odet.unorm0
        return maximum(ratios) / minimum(ratios) - ctrl.ucrit
    end

    # ContinuousCallback affect: fires GR at the exact ψ where gr_condition == 0.
    # OrdinaryDiffEq bisects the dense interpolant to find this root, so the firing
    # ψ depends only on the (continuous) solution u(ψ), not on the adaptive step grid
    # or on how the integration domain was partitioned. After the fixup, the post-GR
    # column norms are written *immediately* to unorm0 (and odet.new is kept false),
    # so the next gr_condition evaluation compares against a reference set at the
    # exact crossing ψ — also partition-invariant.
    function gr_affect!(integrator)
        ctrl, _, _, intr, odet, _ = integrator.p
        odet.unorm .= norm.(eachcol(integrator.u[:, :, 1]))
        odet.unorm ./= odet.unorm0
        if odet.ifix < ctrl.numunorms_init
            odet.ifix += 1
        else
            @warn "unorm storage reached, no longer saving fixfac data. Stability outputs and unorming will be correct, but cannot reconstruct `u`. \n
            Increase `numunorms_init` if needed."
        end
        apply_gaussian_reduction!(integrator.u, odet, intr, false)
        # Anchor the next GR-interval's reference norms to the post-GR state at this
        # exact ψ. Replace zero post-GR norms (apply_gaussian_reduction! zeroes some
        # pivots) with the geometric mean of the survivors, so the next gr_condition
        # eval stays finite.
        odet.unorm0 .= norm.(eachcol(@view integrator.u[:, :, 1]))
        nonzero = odet.unorm0 .> 0
        if any(nonzero) && !all(nonzero)
            ref = exp(sum(log, odet.unorm0[nonzero]) / count(nonzero))
            for j in eachindex(odet.unorm0)
                odet.unorm0[j] == 0 && (odet.unorm0[j] = ref)
            end
        end
        odet.new = false
        # Recompute ud after Gaussian reduction so ud_store stays consistent with u_store.
        sing_der!(du_buffer, integrator.u, integrator.p, integrator.t)
    end

    cb_save = DiscreteCallback((u, t, integrator) -> true, save_callback!)
    # `interp_points=10` and a tight `abstol` give a sharp root, comfortably below the
    # ucrit threshold's relative scale (default ucrit = 1e4).
    cb_gr = ContinuousCallback(gr_condition, gr_affect!; interp_points=10, abstol=1e-6 * ctrl.ucrit)
    cb = CallbackSet(cb_gr, cb_save)
    prob = ODEProblem(sing_der!, odet.u, (chunk.psi_start, chunk.psi_end), (ctrl, equil, ffit, intr, odet, chunk))
    sol = solve(prob, Vern9(); reltol=ctrl.eulerlagrange_tolerance, callback=cb, save_everystep=false, save_end=true)

    # Unconditionally save the final step if the callback did not already capture it.
    # Guarantees the pre-crossing (or pre-edge) state is always stored in u_store,
    # regardless of where the last accepted step landed relative to the near_end band.
    if odet.step == 1 || odet.psi_store[odet.step-1] != sol.t[end]
        if odet.step >= size(odet.u_store, 4)
            resize_storage!(odet)
        end
        odet.psi_store[odet.step] = sol.t[end]
        @views odet.u_store[:, :, :, odet.step] .= sol.u[end]
        odet.q_store[odet.step] = odet.q
        @views odet.ud_store[:, :, :, odet.step] .= odet.ud
        odet.step += 1
    end

    odet.u .= sol.u[end]
    odet.psifac = sol.t[end]
end

"""
    compute_solution_norms!(u::Array{ComplexF64,3}, odet::OdeState, ctrl::ForceFreeStatesControl, intr::ForceFreeStatesInternal, sing_flag::Bool)

Computes norms of the solution vectors of the array `u` and normalizes them
if this is not the first call after a fixup. Formerly `ode_unorm!`.
Throws an error if any vector norm is zero. It then compares the variation in norms
relative to initial values after a fixup, and applies the Gaussian reduction via
`apply_gaussian_reduction!` if the variation exceeds `ctrl.ucrit` or if `sing_flag` is true.
Performs the same function as `ode_unorm` in the Fortran code, with minor differences in indexing
and array handling.

Note that we pass `u` as an argument to reduce the number of copies that are
necessary in the integration callback, since otherwise we'd need to copy from
the integrator state to `odet.u` and then back again. This way, we can just
operate on `u` directly without extra copies.

### Arguments

  - u: Current solution vector array, updated in-place if fixfac is called
  - sing_flag: Indicates if normalization is occuring at a singular surface or not

### TODOs

Add resizing logic for unorm arrays when ifix exceeds allocated size
"""
function compute_solution_norms!(u::Array{ComplexF64,3}, odet::OdeState, ctrl::ForceFreeStatesControl, intr::ForceFreeStatesInternal, sing_flag::Bool)

    # Compute norms of first solution vectors, abort if any are zero
    odet.unorm .= norm.(eachcol(u[:, :, 1]))
    if minimum(odet.unorm) == 0
        jmax = argmin(odet.unorm)
        error("One of the first solution vector norms unorm(1,$jmax) = 0")
    end

    # Normalize unorm and perform Gaussian reduction if required
    if odet.new
        odet.new = false
        odet.unorm0 .= odet.unorm
    else
        odet.unorm ./= odet.unorm0
        uratio = maximum(odet.unorm) / minimum(odet.unorm)
        if uratio > ctrl.ucrit || sing_flag
            # TODO: add resizing logic here as well
            if odet.ifix < ctrl.numunorms_init
                odet.ifix += 1
            else
                @warn "unorm storage reached, no longer saving fixfac data. Stability outputs and unorming will be correct, but cannot reconstruct `u`. \n
                Increase `numunorms_init` if needed. Automatic resizing will be added in a future version."
            end
            apply_gaussian_reduction!(u, odet, intr, sing_flag)
            odet.new = true
        end
    end
end

"""
    apply_gaussian_reduction!(u::Array{ComplexF64,3}, odet::OdeState, intr::ForceFreeStatesInternal, sing_flag::Bool)

Applies Gaussian reduction to orthogonalize solution vectors in `u`.
Formerly `ode_fixup!`. Performs the same function as `ode_fixup` in the Fortran code,
except now relevant `fixfac` data are stored in memory instead of dumped to `euler.bin`.
Used when the spread in norms exceeds a threshold or when a rational surface is reached.
This will update both `u` and relevant fields in `odet` in-place. See the
description of `compute_solution_norms!` for more details on the benefits of in-place `u`
updates.
"""
function apply_gaussian_reduction!(u::Array{ComplexF64,3}, odet::OdeState, intr::ForceFreeStatesInternal, sing_flag::Bool)

    # Store data for the current fixup
    ifix = odet.ifix
    odet.sing_flag[ifix] = sing_flag
    # Since the current step has been fixed-up, we denote the end of the previous
    # fixup region as the previous step (this just avoids a -1 index later)
    odet.fixstep[ifix] = odet.step - 1

    # Initialize fixfac
    for isol in 1:intr.numpert_total
        odet.fixfac[isol, isol, ifix] = 1
    end
    # Sort unorm in descending order (since we triangularize from largest to smallest)
    odet.index[:, ifix] = sortperm(odet.unorm; rev=true)

    # Triangularize primary solutions
    mask = trues(2, intr.numpert_total)
    for isol in 1:intr.numpert_total
        ksol = odet.index[isol, ifix]
        mask[2, ksol] = false
        # Set pivot row based on max location
        @views kpert = argmax(abs.(u[:, ksol, 1]) .* mask[1, :])
        mask[1, kpert] = false
        # Eliminate other solution vectors below the pivot
        for jsol in 1:intr.numpert_total
            if mask[2, jsol]
                odet.fixfac[ksol, jsol, ifix] = -u[kpert, jsol, 1] / u[kpert, ksol, 1]
                @. @views u[:, jsol, :] .= u[:, jsol, :] .+ u[:, ksol, :] .* odet.fixfac[ksol, jsol, ifix]
                u[kpert, jsol, 1] = 0
            end
        end
    end
end

"""
    findmax_dW_edge!(odet::OdeState, ctrl::ForceFreeStatesControl, equil::Equilibrium.PlasmaEquilibrium, ffit::FourFitVars, intr::ForceFreeStatesInternal)

Records the total dW in the integration region between `ctrl.psiedge` and
`ctrl.psilim`. This performs the same function as `ode_record_edge` in the
Fortran, but everything is now done post-integration which cleans up the logic,
i.e. no "_edge" arrays.

The dW is stored at that step index in `odet.dW_edge`; because we initialize
dW_edge to -Inf, we can just take the max value after integration to get the
total dW at the edge and avoid unphysical kink modes that might occur just
inside rational surfaces.

We have also separated the computation of the wv matrix spline and the total dW
calculation into `free_compute_wv_spline` and `free_compute_total` respectively
for clarity. We create the wv matrix spline once prior to the loop.
"""
function findmax_dW_edge!(odet::OdeState, ctrl::ForceFreeStatesControl, equil::Equilibrium.PlasmaEquilibrium, ffit::FourFitVars, intr::ForceFreeStatesInternal)

    # Find the first ODE step at or past psiedge; all subsequent steps are contiguous edge steps
    edge_start = findfirst(i -> odet.psi_store[i] >= ctrl.psiedge, 1:odet.step)
    N_edge = odet.step - edge_start + 1

    # Initialize EdgeScanState sized exactly to the number of edge steps
    odet.edge_scan = EdgeScanState(intr.numpert_total, N_edge)
    es = odet.edge_scan

    es.psi .= odet.psi_store[edge_start:odet.step]
    es.q .= odet.q_store[edge_start:odet.step]

    # Create a rough spline for wv matrix between psiedge -> psilim so we can approximate dW
    es.wvmat = free_compute_wv_spline(ctrl, equil, intr)

    # Loop with compact index j into EdgeScanState; ODE index is edge_start + j - 1.
    # Steps where free_compute_total hits a singular wp solve are left as NaN per the EdgeScanState contract.
    for j in 1:N_edge
        istep = edge_start + j - 1
        odet.psifac = odet.psi_store[istep]
        odet.u .= odet.u_store[:, :, :, istep]
        try
            result = free_compute_total(equil, ffit, intr, odet)
            es.total_eigenvalue[j] = result.total_eigenvalue
            es.plasma_energy[j] = result.plasma_energy
            es.vacuum_energy[j] = result.vacuum_energy
            es.vacuum_eigenvalue[j] = result.vacuum_eigenvalue
        catch e
            e isa LinearAlgebra.SingularException || rethrow()
        end
    end

    # Return the ODE step index at peak total_eigenvalue (NaN-safe; failed steps ignored)
    peak_j = argmax(j -> isnan(real(es.total_eigenvalue[j])) ? typemin(Float64) : real(es.total_eigenvalue[j]), 1:N_edge)
    return edge_start + peak_j - 1
end

"""
    transform_u!(odet::OdeState, intr::ForceFreeStatesInternal)

Constructs the transformation matrices to form the true solution vectors. Effectively
"undoes" the Gaussian reduction applied during fixups throughout the integration, such that
we have the true solution vectors for use in GPEC. Modifies the store arrays in `odet` in-place.
Performs a similar function as `idcon_transform` + `idcon_build` in the Fortran code,
except we separate the building of the transformation matrices and determining the coefficients
for a chosen force-free solution, which can be done in postprocessing.
"""
function transform_u!(odet::OdeState, intr::ForceFreeStatesInternal)

    # Gaussian reduction matrices for each fixup
    gauss = Array{ComplexF64,3}(undef, intr.numpert_total, intr.numpert_total, odet.ifix)
    # Transformation matrices for each region between fixups (ifix + 1 regions)
    transforms = Array{ComplexF64,3}(undef, intr.numpert_total, intr.numpert_total, odet.ifix + 1)
    # Temporary workspace matrices
    gauss_buffer = Matrix{ComplexF64}(undef, intr.numpert_total, intr.numpert_total)
    identity = Matrix{ComplexF64}(I, intr.numpert_total, intr.numpert_total)
    temp = Matrix{ComplexF64}(undef, intr.numpert_total, intr.numpert_total)
    mask = trues(intr.numpert_total)

    # Construct gaussian reduction matrices for each fixup
    for ifix in 1:odet.ifix
        gauss[:, :, ifix] .= identity
        mask .= true
        for isol in 1:intr.numpert_total
            ksol = odet.index[isol, ifix]
            mask[ksol] = false
            temp .= identity
            for jsol in 1:intr.numpert_total
                if mask[jsol]
                    temp[ksol, jsol] = odet.fixfac[ksol, jsol, ifix]
                end
            end
            mul!(gauss_buffer, view(gauss,:,:,ifix), temp)
            gauss[:, :, ifix] .= gauss_buffer
        end
        # Account for zeroed indices at singular surfaces in `ode_ideal_cross`
        if odet.sing_flag[ifix]
            for zeroed_idx in odet.zeroed_idx[ifix]
                gauss[:, odet.index[zeroed_idx, ifix], ifix] .= 0.0
            end
        end
    end

    # Concatenate gaussian reduction matrix to form transform matrix for each region
    # Here, the i'th region is between the (i-1)'th and i'th fixup e.g. transforms[:, :, 1]
    # is the transform matrix for the region between init and first fixup
    # and mfix + 1 is the for the region after the last fixup and before the edge
    transforms[:, :, end] .= identity
    for ifix in odet.ifix:-1:1
        mul!(view(transforms,:,:,ifix), view(gauss,:,:,ifix), view(transforms,:,:,(ifix+1)))
    end

    # Now that we have the transform matrices, we can apply them to the solution vectors
    # "undoing" the Gaussian reductions to get the true solution vectors
    jfix = 1
    for ifix in 1:(odet.ifix+1)
        # If after the last fixup, go to the end of integration.
        # Cap kfix at odet.step: fixstep entries from fixups AFTER the peak (set during integration
        # before trim_storage!) can exceed the trimmed storage size and must be clamped.
        kfix = ifix != odet.ifix + 1 ? min(odet.fixstep[ifix], odet.step) : odet.step
        jfix > odet.step && break
        @views for istep in jfix:kfix
            # This is u1->u4 in Fortran
            mul!(gauss_buffer, odet.u_store[:, :, 1, istep], transforms[:, :, ifix])
            odet.u_store[:, :, 1, istep] .= gauss_buffer
            mul!(gauss_buffer, odet.u_store[:, :, 2, istep], transforms[:, :, ifix])
            odet.u_store[:, :, 2, istep] .= gauss_buffer
            mul!(gauss_buffer, odet.ud_store[:, :, 1, istep], transforms[:, :, ifix])
            odet.ud_store[:, :, 1, istep] .= gauss_buffer
            mul!(gauss_buffer, odet.ud_store[:, :, 2, istep], transforms[:, :, ifix])
            odet.ud_store[:, :, 2, istep] .= gauss_buffer
        end
        jfix = kfix + 1
    end
end
