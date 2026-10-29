"""
    make_optimal_psi_grid(psilow, psihigh, N_core, N_mid, N_edge; psi_split_core, psi_split_edge)

Build a three-region ψ grid with (N_core + N_mid + N_edge)+1 knots, using the counts
directly as provided — core and edge are geometric in log(ψ) and log(1−ψ) respectively;
middle is uniform in ψ.
"""
function make_optimal_psi_grid(psilow, psihigh, N_core, N_mid, N_edge;
    psi_split_core=0.03, psi_split_edge=0.98)
    # Core: [psilow, psi_split_core], geometric in log(ψ)
    core_pts = [psilow * (psi_split_core / psilow)^(i / N_core) for i in 0:N_core]
    # Middle: [psi_split_core, psi_split_edge], uniform (skip first to avoid duplicate)
    mid_pts = [psi_split_core + (psi_split_edge - psi_split_core) * i / N_mid for i in 1:N_mid]
    # Edge: [psi_split_edge, psihigh], geometric in log(1−ψ) (skip first to avoid duplicate)
    edge_pts = [1.0 - (1.0 - psi_split_edge) * ((1.0 - psihigh) / (1.0 - psi_split_edge))^(i / N_edge) for i in 1:N_edge]

    return vcat(core_pts, mid_pts, edge_pts)
end

"""
    _build_psi_grid(equil_params, psilow, psihigh)

Resolve `mpsi` and build `psi_nodes` for any supported `grid_type`.

For `"auto"` with `mpsi=0`, this is the coarse pass-1 layout of
the two-pass refinement — the main driver measures the formed equilibrium and re-forms on
the `refined_psi_grid` result. Shared by `direct_fieldline_int` and `efit_by_inversion`
solvers.
"""
function _build_psi_grid(equil_params, psilow, psihigh)
    mpsi = equil_params.mpsi
    if equil_params.grid_type == "auto" && mpsi == 0 && equil_params.psi_accuracy > 0
        # Two-pass auto grid: this is the pass-1 layout; the driver measures the formed
        # equilibrium's curvature and re-forms on a refined grid (GridRefinement.jl). Sized to
        # resolve the gradient structure accurately so pass 2 is well-provisioned (PASS1_INTERVALS).
        mpsi = PASS1_INTERVALS
        @info "Auto psi grid: forming pass-1 equilibrium on $(mpsi)-interval log_asymptotic grid pending curvature-based refinement"
    elseif mpsi == 0
        mpsi = 128
    end

    psi_nodes = if equil_params.grid_type == "auto"
        psihigh > 0.98 ||
            @warn "grid_type = \"$(equil_params.grid_type)\" needs psihigh > 0.98 (its grid has a fixed edge region on [0.98, psihigh]); " *
                  "psihigh = $psihigh gives a non-monotonic ψ grid and the equilibrium will fail. Use grid_type = \"core_edge_packed\" instead."
        # Distribute mpsi across the three regions by log-weights
        log_core = log(0.03 / psilow)
        log_mid = log(0.98 / 0.03)
        log_edge = log((1.0 - 0.98) / (1.0 - psihigh))
        log_total = log_core + log_mid + log_edge
        N_edge = clamp(round(Int, mpsi * log_edge / log_total), 2, mpsi ÷ 2)
        N_core = round(Int, mpsi * log_core / log_total)
        N_mid = mpsi - N_edge - N_core
        make_optimal_psi_grid(psilow, psihigh, N_core, N_mid, N_edge)
    elseif equil_params.grid_type == "core_edge_packed"
        [psilow + (psihigh - psilow) * sin((ipsi / mpsi) * (π / 2))^2 for ipsi in 0:mpsi]
    elseif equil_params.grid_type == "pow1"
        # Fortran powspace(psilow, psihigh, 1, mpsi+1, "upper") — edge-packed grid (equil/grid.f90:92-195)
        [psilow + (psihigh - psilow) * (3(ipsi / mpsi) - (ipsi / mpsi)^3) / 2 for ipsi in 0:mpsi]
    elseif equil_params.grid_type == "uniform"
        # Evenly spaced surfaces, e.g. for profile diagnostics at constant resolution
        collect(range(psilow, psihigh; length=mpsi + 1))
    else
        error("Unsupported grid_type: $(equil_params.grid_type)")
    end
    # Floor node spacing on the fixed grids: core_edge_packed/pow1 pack the edge as ~(π/2·mpsi)⁻² and at high
    # mpsi drive it below the integration-noise scale (garbage curvature). The auto grid floors its
    # refined pass-2 grid in refined_psi_grid, so it is left untouched here.
    return equil_params.grid_type == "auto" ? psi_nodes : enforce_min_spacing(psi_nodes, MIN_KNOT_SPACING)
end
