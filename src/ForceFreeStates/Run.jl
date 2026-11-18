# Run.jl: the force-free-states solve on a prepared equilibrium, from the toroidal mode range to the published result.

"""
    _mode_range_label(intr) -> String

Compact label for the resolved toroidal mode range, `"1"` for a single `n` and `"1:3"` for a range.
"""
_mode_range_label(intr::ForceFreeStatesInternal) = intr.npert == 1 ? "$(intr.nlow)" : "$(intr.nlow):$(intr.nhigh)"

"""
    resolve_mode_space!(intr, ctrl) -> intr

Resolve the requested toroidal mode range onto `intr`, filling an unspecified bound from the
other one and rejecting ranges that contain no supported `n >= 1` mode.
"""
function resolve_mode_space!(intr::ForceFreeStatesInternal, ctrl::ForceFreeStatesControl)
    # Determine toroidal mode numbers (n >= 1 required; 0 means "not specified")
    intr.nlow, intr.nhigh = ctrl.nn_low, ctrl.nn_high
    if intr.nlow == 0 && intr.nhigh == 0
        error("the toroidal mode number nn must be set")
    elseif intr.nlow == 0
        intr.nlow = intr.nhigh
    elseif intr.nhigh == 0
        intr.nhigh = intr.nlow
    end
    if intr.nlow > intr.nhigh
        error("the toroidal mode range $(intr.nlow):$(intr.nhigh) is empty")
    end
    if intr.nhigh < 1
        error("All requested toroidal modes (n=$(intr.nlow):$(intr.nhigh)) are below 1; " *
              "n < 1 modes are not supported")
    end
    if intr.nlow < 1
        @warn "Clamping the lowest toroidal mode from $(intr.nlow) to 1; n < 1 modes are not supported"
        intr.nlow = 1
    end
    intr.npert = intr.nhigh - intr.nlow + 1
    return intr
end

"""
    run_force_free_states(ctrl, equil, mats, intr, metric; cut_solution=false) -> ForceFreeStatesResult

Run the formalism selected by `ctrl.algorithm` — the standalone Galerkin solve, or the
Euler-Lagrange sweep with its free-boundary energies and Δ′ BVP — and publish its products as a
`ForceFreeStatesResult`. `cut_solution` is the scripting-only `Galerkin(; cut_solution)` option.
"""
function run_force_free_states(
    ctrl::ForceFreeStatesControl,
    equil::Equilibrium.PlasmaEquilibrium,
    mats,
    intr::ForceFreeStatesInternal,
    metric;
    cut_solution::Bool=false
)
    nstring = _mode_range_label(intr)

    # The three formalisms are exclusive. Galerkin solves the same Euler-Lagrange system
    # variationally rather than by radial ODE integration, so it replaces both the integration
    # and the free-boundary energies, and supplies its own vacuum response at psilim.
    odet = nothing
    free_energies = nothing
    gal_data = nothing
    gal_dp = nothing
    if ctrl.algorithm isa Galerkin
        ctrl.kinetic === nothing ||
            error("Galerkin() does not support kinetic runs; use Forward().")
        gal_start = time()
        wv = ctrl.vac_flag ? first(compute_scaled_wv(ctrl, equil, intr)) : nothing
        gal_data, gal_dp = galerkin_solve(ctrl, equil, mats, intr; wv=wv, cut_solution=cut_solution)
        gal_dt = time() - gal_start
        @info "Galerkin solve completed in $(@sprintf("%.3f", gal_dt)) s"
    else
        if intr.verbose
            @info "Integrating Euler-Lagrange equation"
        end
        odet, fm_propagators, fm_chunks, fm_S_left = eulerlagrange_integration(ctrl, equil, mats, intr)
        if odet.nzero > 0 && intr.verbose
            @warn "Fixed-boundary mode unstable for n = $nstring"
        end

        if ctrl.vac_flag
            if intr.verbose
                wall_desc = intr.wall_settings === nothing ? "no wall" : nameof(typeof(intr.wall_settings))
                @info "Computing free boundary energies ($wall_desc)"
            end
            free_energies = free_run(odet, ctrl, equil, mats, intr)
            normalize_eigenfunctions!(odet, free_energies.wt, equil.psio)
            if real(free_energies.et[1]) < 0
                if intr.verbose
                    @warn "Free-boundary mode unstable for n = $nstring"
                end
            else
                if intr.verbose
                    @info "All free-boundary modes stable for n = $nstring"
                end
            end

            # Inter-surface Δ' (STRIDE BVP): needs Riccati() propagators and wv from free_run.
            if ctrl.kinetic === nothing && intr.msing > 0 && fm_propagators !== nothing
                if intr.verbose
                    @info "Computing Δ' matrix (STRIDE BVP with vacuum coupling)"
                end
                compute_delta_prime_matrix!(intr, fm_propagators, fm_chunks;
                    wv=free_energies.wv, psio=equil.psio, debug=intr.verbose,
                    S_at_surface_left=fm_S_left,
                    ctrl=ctrl, equil=equil, mats=mats)
            end
        end
    end

    # Publish the solve: from here on the downstream stages read the result, never `intr`.
    result = build_result(_integrator_symbol(ctrl.algorithm), ctrl, equil, intr, metric, mats, odet, free_energies, gal_data, gal_dp)

    return result
end
