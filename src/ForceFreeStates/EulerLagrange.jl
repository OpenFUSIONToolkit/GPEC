"""
    eulerlagrange_integration(ctrl::ForceFreeStatesControl, equil::Equilibrium.PlasmaEquilibrium, ffit::FourFitVars, intr::ForceFreeStatesInternal) -> OdeState

Legacy Euler-Lagrange integration driver — the standard forward sweep over
`chunk_el_integration_bounds` with Gaussian-reduction normalization. This is the
`integration_method = "LegacyEulerLagrange"` path, kept for benchmarking against the
chunked-Riccati default. It does not produce a Δ' matrix.

Pre-computes all integration chunks upfront and iterates them, then post-processes
(edge-dW scan, fixed-boundary stability criterion, Gaussian-reduction undo). Returns the
final `OdeState`; callers reach it via [`forcefreestates_integration`](@ref).
"""
function eulerlagrange_integration(ctrl::ForceFreeStatesControl, equil::Equilibrium.PlasmaEquilibrium, ffit::FourFitVars, intr::ForceFreeStatesInternal)
    odet = standard_eulerlagrange_pass(ctrl, equil, ffit, intr)
    finalize_canonical_u_store!(odet, ctrl, equil, ffit, intr)
    return odet
end

"""
    standard_eulerlagrange_pass(ctrl, equil, ffit, intr) -> OdeState

Build the canonical `u_store` via the standard Euler-Lagrange path: forward
integration over `chunk_el_integration_bounds`, with `cross_ideal_singular_surf!`
at each rational surface. Caller is responsible for calling
`finalize_canonical_u_store!` afterwards.

This is the partition-invariant reference build (GR firing is bisected on the
`uratio − ucrit` zero crossing inside `integrate_el_region!`, see Phase C).
`parallel_eulerlagrange_integration` calls this to obtain `u_store` that matches
the standard path byte-for-byte, then runs its parallel propagator phase only for
the deferred Δ' BVP / S-gauge outputs.
"""
function standard_eulerlagrange_pass(
    ctrl::ForceFreeStatesControl, equil::Equilibrium.PlasmaEquilibrium,
    ffit::FourFitVars, intr::ForceFreeStatesInternal
)
    odet = OdeState(intr.numpert_total, ctrl.numsteps_init, ctrl.numunorms_init, intr.msing)
    if ctrl.sing_start <= 0
        initialize_el_at_axis!(odet, ctrl, equil.profiles, intr)
    elseif ctrl.sing_start <= intr.msing
        error("sing_start > 0 not implemented yet!")
    else
        error("Invalid value for sing_start: $(ctrl.sing_start) > msing = $(intr.msing)")
    end

    chunks = chunk_el_integration_bounds(odet, ctrl, intr)

    if ctrl.verbose
        @info "   ψ = $((@sprintf "%.3f" odet.psifac)),  q = $((@sprintf "%.3f" equil.profiles.q_spline(odet.psifac)))"
    end

    for chunk in chunks
        integrate_el_region!(odet, ctrl, equil, ffit, intr, chunk)
        if ctrl.verbose
            @info "   ψ = $((@sprintf "%.3f" odet.psifac)),  q = $((@sprintf "%.3f" odet.q)),  steps = $(odet.total_steps)"
        end
        if chunk.needs_crossing
            cross_ideal_singular_surf!(odet, ctrl, equil, ffit, intr, chunk.ising)
        end
    end

    return odet
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
