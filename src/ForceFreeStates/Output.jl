import ..Utilities: save_to_hdf5
import ..LocalStability
import ..ForcingTerms
import ..Utilities: apply_main_h5_metadata!

"""
    save_to_hdf5(target, result::ForceFreeStatesResult; git_version="unknown", local_stability=nothing)

Save the FFS solutions, operators, singular-surface diagnostics, and vacuum products, and record the deck that
reproduces the solve (equilibrium, force balance and any match) in `Input/`. The associated equilibrium and optional
local-stability result use their own save methods. Unavailable numerical products use empty datasets.
"""
function save_to_hdf5(
    out_h5::Union{HDF5.File,HDF5.Group},
    result::ForceFreeStatesResult;
    git_version::String="unknown",
    local_stability::Union{LocalStability.LocalStabilityResult,Nothing}=nothing
)

    ctrl = result.control
    equil = result.equil
    mats = result.mats
    free_energies = result.free_boundary
    gal_data = result.galerkin
    diag = result.diagnostics
    msing = length(result.surfaces)
    # Closed ξ profiles are written into the producing formalism's Solutions group with the
    # same dataset names and (mode, solution, psi) axis order: ForwardIntegration for the
    # axis-basis sweep, GalerkinIntegration for the matched gal solution (its own grid).
    xi_solution = (result.solution !== nothing && result.solution.basis === :el_axis) ? result.solution : nothing
    gal_solution = (result.solution !== nothing && result.solution.basis === :gal_native) ? result.solution : nothing


    # File-level metadata contract (schema_version, Conventions, title, date).
    Utilities.HDF5Annotations.write_root_attrs!(out_h5; title="GPEC output: $(basename(dirname(abspath(HDF5.filename(out_h5)))))")

    # Store git version for reproducibility
    out_h5["Info/git_version"] = git_version

    # Outer-region Galerkin solver outputs (RDCON), if it ran
    if gal_data !== nothing
        write_galerkin!(out_h5, gal_data; basis_output=result.control.algorithm.basis_output)
    end

    if gal_solution !== nothing
        gal = "ForceFreeStates/Solutions/GalerkinIntegration"
        out_h5["$gal/psi"] = gal_solution.psi_store
        out_h5["$gal/q"] = gal_solution.q_store
        out_h5["$gal/xi_psi"] = gal_solution.u_store[:, :, 1, :]
        out_h5["$gal/dxi_psidpsi"] = gal_solution.du_store
        out_h5["$gal/xi_s"] = gal_solution.xi_s_store
    end

    save_equilibrium_ingest!(out_h5, equil)
    Utilities.record_inputs!(out_h5, result)

    # Write derived run parameters
    out_h5["Info/mpert"] = result.mpert
    out_h5["Info/mlow"] = result.mlow
    out_h5["Info/mhigh"] = result.mhigh
    out_h5["Info/npert"] = result.npert
    out_h5["Info/nlow"] = result.nlow
    out_h5["Info/nhigh"] = result.nhigh
    m = [(i - 1) % result.mpert + result.mlow for i in 1:(result.numpert_total)]
    n = [(i - 1) ÷ result.mpert + result.nlow for i in 1:(result.numpert_total)]
    out_h5["Info/mn_index"] = hcat(m, n)   # (N, 2) matrix
    out_h5["Info/psilim"] = result.psilim
    out_h5["Info/qlim"] = result.qlim
    out_h5["Info/dqdpsi_lim"] = result.q1lim

    save_to_hdf5(out_h5, equil)

    ls = local_stability
    if ls !== nothing
        save_to_hdf5(out_h5, ls)
    else
        for name in (:psi, :D_I, :D_R, :ballooning_Delta_prime, :ballooning_psi, :alpha, :alpha_critical)
            out_h5["LocalStability/$name"] = Float64[]
        end
    end
    out_h5["SingularSurfaces/D_I"] = (ls !== nothing && !isempty(result.surfaces)) ?
                                     [LocalStability.mercier_D_I(ls, sing.psifac) for sing in result.surfaces] : Float64[]

    # Resistive-layer overlap scan, present when psilim_from_layer_overlap capped the domain.
    if result.layer_overlap !== nothing
        lo = result.layer_overlap
        out_h5["ForceFreeStates/LayerOverlap/m"] = lo.m
        out_h5["ForceFreeStates/LayerOverlap/n"] = lo.n
        out_h5["ForceFreeStates/LayerOverlap/psi"] = lo.psi
        out_h5["ForceFreeStates/LayerOverlap/r_s"] = lo.rs
        out_h5["ForceFreeStates/LayerOverlap/delta_s_abs"] = lo.delta_s_m
        out_h5["ForceFreeStates/LayerOverlap/width_delta_s"] = lo.width_delta_s
        out_h5["ForceFreeStates/LayerOverlap/width_visco"] = lo.width_visco
        out_h5["ForceFreeStates/LayerOverlap/width_dr"] = lo.width_dr
        out_h5["ForceFreeStates/LayerOverlap/extrapolated"] = Int.(lo.extrapolated)
        out_h5["ForceFreeStates/LayerOverlap/psilim_overlap"] =
            lo.psihigh === nothing ? NaN : lo.psihigh
        out_h5["ForceFreeStates/LayerOverlap/first_overlap_index"] =
            lo.first_overlap === nothing ? -1 : lo.first_overlap
        # "applied" records that the bound was active going into sing_lim!, not that it set
        # the final psilim -- rational_offset_fraction/qhigh may truncate deeper inside it.
        out_h5["ForceFreeStates/LayerOverlap/applied"] =
            Int(lo.psihigh !== nothing &&
                lo.psihigh < equil.params.psihigh_resolved)
    end

    # Write integration data: the ψ trace and integrator diagnostics from the raw ODE state,
    # the ξ profiles from the solution. Either may be absent (Galerkin has no ODE state;
    # Riccati has no ξ solution), in which case the datasets are written empty.
    fwd = "ForceFreeStates/Solutions/ForwardIntegration"
    out_h5["$fwd/nstep"] = diag !== nothing ? diag.step : 0            # Number of saved solution snapshots
    out_h5["$fwd/nstep_total"] = diag !== nothing ? diag.total_steps : 0  # Total ODE solver steps taken
    out_h5["$fwd/psi"] = diag !== nothing ? diag.psi_store : Float64[]
    out_h5["$fwd/q"] = diag !== nothing ? diag.q_store : Float64[]
    out_h5["$fwd/xi_psi"] = xi_solution !== nothing ? xi_solution.u_store[:, :, 1, :] : ComplexF64[]
    out_h5["$fwd/u2"] = xi_solution !== nothing ? xi_solution.u_store[:, :, 2, :] : ComplexF64[] # conjugate momenta of ξ_ψ
    out_h5["$fwd/dxi_psidpsi"] = xi_solution !== nothing ? xi_solution.du_store : ComplexF64[]
    out_h5["$fwd/xi_s"] = xi_solution !== nothing ? xi_solution.xi_s_store : ComplexF64[]
    out_h5["$fwd/crit"] = diag !== nothing ? diag.crit_store : Float64[]

    # Write edge stability scan data (only present when dW_edge_scan_start < psilim).
    # Generalized (W, N) pencil energies — power-normalized, Jacobian-invariant; these are
    # the values findmax_dW_edge! uses to choose the truncation point.
    if diag !== nothing && !isempty(diag.edge_scan.psi)
        es = diag.edge_scan
        out_h5["ForceFreeStates/EdgeScan/psi"] = es.psi
        out_h5["ForceFreeStates/EdgeScan/q"] = es.q
        out_h5["ForceFreeStates/EdgeScan/total_energy"] = es.total_eigenvalue
        out_h5["ForceFreeStates/EdgeScan/plasma_energy"] = es.plasma_energy
        out_h5["ForceFreeStates/EdgeScan/vacuum_energy"] = es.vacuum_energy
        out_h5["ForceFreeStates/EdgeScan/vacuum_eigenvalue"] = es.vacuum_eigenvalue
    end

    # Write singular surface data
    out_h5["SingularSurfaces/rational_count"] = msing
    out_h5["SingularSurfaces/rational_psi"] = [sing.psifac for sing in result.surfaces]
    out_h5["SingularSurfaces/rational_q"] = [sing.q for sing in result.surfaces]
    out_h5["SingularSurfaces/dqdpsi"] = [sing.q1 for sing in result.surfaces]
    # Kinetic and galerkin-matched runs never populate ca_l/ca_r (only ideal surface
    # crossings do); emit zero-extent sentinels instead of unpopulated arrays.
    out_h5["SingularSurfaces/ca_left"] = (diag !== nothing && diag.ca_populated) ? diag.ca_l : zeros(ComplexF64, 0, 0, 0, 0)
    out_h5["SingularSurfaces/ca_right"] = (diag !== nothing && diag.ca_populated) ? diag.ca_r : zeros(ComplexF64, 0, 0, 0, 0)

    if msing > 0
        # Mode numbers at each surface (jagged — pad with 0 to max_modes width)
        max_modes = maximum(s -> length(s.m), result.surfaces)
        m_matrix = zeros(Int, msing, max_modes)
        n_matrix = zeros(Int, msing, max_modes)
        for (s, sing) in enumerate(result.surfaces)
            for i in 1:length(sing.m)
                m_matrix[s, i] = sing.m[i]
                n_matrix[s, i] = sing.n[i]
            end
        end
        out_h5["SingularSurfaces/rational_m"] = m_matrix
        out_h5["SingularSurfaces/rational_n"] = n_matrix

        # GGJ coefficients and geometric surface averages, enough for ggj_parameters to rebuild τ_A / τ_R from any kinetic profiles.
        if all(s -> s.restype !== nothing, result.surfaces)
            out_h5["SingularSurfaces/E"] = [s.restype.E for s in result.surfaces]
            out_h5["SingularSurfaces/F"] = [s.restype.F for s in result.surfaces]
            out_h5["SingularSurfaces/G"] = [s.restype.G for s in result.surfaces]
            out_h5["SingularSurfaces/H"] = [s.restype.H for s in result.surfaces]
            out_h5["SingularSurfaces/K"] = [s.restype.K for s in result.surfaces]
            out_h5["SingularSurfaces/M"] = [s.restype.M for s in result.surfaces]
            out_h5["SingularSurfaces/avg_bsq_over_dpsisq"] = [s.restype.avg_bsq_over_dpsisq for s in result.surfaces]
            out_h5["SingularSurfaces/avg_bsq"] = [s.restype.avg_bsq for s in result.surfaces]
            out_h5["SingularSurfaces/mu0p"] = [s.restype.p_local for s in result.surfaces]
            out_h5["SingularSurfaces/dmu0pdpsi"] = [s.restype.p1_local for s in result.surfaces]
            out_h5["SingularSurfaces/dVdpsi"] = [s.restype.v1_local for s in result.surfaces]
        end
    end

    # Per-surface ca-based Δ' (`sing.delta_prime`) is a stub; only the BVP matrix is emitted (see SingType.delta_prime docstring).

    # Write the Δ' payload on one set of canonical paths, whichever formalism produced it
    # (Riccati BVP or Galerkin): both compute the same quantities in the same PEST-3
    # convention. Surface indexing follows the producing formalism's surface list, which for
    # Galerkin is the in-domain in-band subset of `SingularSurfaces/`.
    dp = result.delta_prime
    if msing > 0 && dp !== nothing
        # Inter-surface Δ' matrix, shape [msing × msing] — PEST3-convention deltap.
        out_h5["SingularSurfaces/Delta_prime_matrix"] = dp.matrix

        # Edge coil-response matrix, stored (numpert_total × 2msing) = (edge mode, surface-side)
        # so H5Web heatmaps read x = edge mode, y = surface-side. The carried matrix stays
        # (2msing × numpert_total); transpose only at write.
        isempty(dp.coil) || (out_h5["SingularSurfaces/Delta_coil"] = permutedims(dp.coil))

        # Raw 2msing×2msing outer-region D' matrix, side-major [L_s1, R_s1, L_s2, R_s2, …]; pest3_decompose recovers (A', B', Γ', Δ').
        isempty(dp.raw) || (out_h5["SingularSurfaces/Delta_prime_raw"] = dp.raw)

        # Remaining PEST-3 parity blocks, when the formalism persisted them (Galerkin);
        # Riccati recovers them from Delta_prime_raw via pest3_decompose.
        dp.A === nothing || (out_h5["SingularSurfaces/pest3_A"] = dp.A)
        dp.B === nothing || (out_h5["SingularSurfaces/pest3_B"] = dp.B)
        dp.Gamma === nothing || (out_h5["SingularSurfaces/pest3_Gamma"] = dp.Gamma)
    end
    result.match === nothing || save_to_hdf5(out_h5, result.match)

    # Write kinetic singular surface data (det(F̄) near-zeros) and the cond(F̄) scan
    # used to find them. Populated only when kinetic crossings were searched for.
    out_h5["SingularSurfaces/Kinetic/rational_count"] = result.kinetic.kmsing
    out_h5["SingularSurfaces/Kinetic/rational_psi"] = [s.psifac for s in result.kinetic.kinsing]
    out_h5["SingularSurfaces/Kinetic/rational_q"] = [s.q for s in result.kinetic.kinsing]
    out_h5["SingularSurfaces/Kinetic/dqdpsi"] = [s.q1 for s in result.kinetic.kinsing]
    out_h5["SingularSurfaces/Kinetic/scan_psi"] = result.kinetic.scan_psi
    out_h5["SingularSurfaces/Kinetic/scan_cond"] = result.kinetic.scan_cond
    out_h5["SingularSurfaces/Kinetic/scan_threshold"] = result.kinetic.scan_threshold

    # Free-boundary stability: eigenmode energies are generalized eigenvalues of the (W, N) pencil (power-normalized,
    # Jacobian-invariant); eigenmode columns are sorted most-unstable first, unit power norm, largest entry real-positive.
    fbs = "ForceFreeStates/FreeBoundaryStability"
    out_h5["$fbs/W_freeboundary"] = free_energies !== nothing ? free_energies.wt0 : ComplexF64[]
    out_h5["$fbs/W_plasma"] = free_energies !== nothing ? free_energies.wp : ComplexF64[]
    out_h5["$fbs/W_vacuum"] = free_energies !== nothing ? free_energies.wv : ComplexF64[]
    out_h5["$fbs/W_freeboundary_eigenmodes"] = free_energies !== nothing ? free_energies.wt : ComplexF64[]
    out_h5["$fbs/eigenmode_energies"] = free_energies !== nothing ? free_energies.et : ComplexF64[]
    out_h5["$fbs/eigenmode_plasma_energies"] = free_energies !== nothing ? free_energies.ep : ComplexF64[]
    out_h5["$fbs/eigenmode_vacuum_energies"] = free_energies !== nothing ? free_energies.ev : ComplexF64[]
    out_h5["$fbs/vacuum_eigenvalue"] = free_energies !== nothing ? free_energies.vacuum_eigenvalue : NaN

    # Cartesian surface point clouds for visualization.
    out_h5["SurfaceGeometries/Plasma/x"] = free_energies !== nothing ? free_energies.plasma_pts[:, 1] : Float64[]
    out_h5["SurfaceGeometries/Plasma/y"] = free_energies !== nothing ? free_energies.plasma_pts[:, 2] : Float64[]
    out_h5["SurfaceGeometries/Plasma/z"] = free_energies !== nothing ? free_energies.plasma_pts[:, 3] : Float64[]
    out_h5["SurfaceGeometries/Wall/x"] = free_energies !== nothing ? free_energies.wall_pts[:, 1] : Float64[]
    out_h5["SurfaceGeometries/Wall/y"] = free_energies !== nothing ? free_energies.wall_pts[:, 2] : Float64[]
    out_h5["SurfaceGeometries/Wall/z"] = free_energies !== nothing ? free_energies.wall_pts[:, 3] : Float64[]

    if ctrl.write_el_matrices
        # Write fundamental matrices on the ψ grid
        xs = equil.rzphi_xs
        npsi = length(xs)
        np = result.numpert_total

        # Helper: evaluate a matrix spline on the psi grid → (npsi, np, np) array
        function _eval_mat_spline(spline)
            arr = zeros(ComplexF64, npsi, np, np)
            hint = Ref(1)
            for i in 1:npsi
                arr[i, :, :] .= reshape(spline(xs[i]; hint=hint), np, np)
            end
            return arr
        end

        elm = "ForceFreeStates/EulerLagrangeMatrices"
        out_h5["$elm/psi"] = xs
        # Ideal primitive matrices (A, B, C, D, E, H)
        out_h5["$elm/Ideal/A"] = _eval_mat_spline(mats.ideal.A_spline)
        out_h5["$elm/Ideal/B"] = _eval_mat_spline(mats.ideal.B_spline)
        out_h5["$elm/Ideal/C"] = _eval_mat_spline(mats.ideal.C_spline)
        out_h5["$elm/Ideal/D"] = _eval_mat_spline(mats.ideal.D_spline_prim)
        out_h5["$elm/Ideal/E"] = _eval_mat_spline(mats.ideal.E_spline_prim)
        out_h5["$elm/Ideal/H"] = _eval_mat_spline(mats.ideal.H_spline)

        # Ideal derived matrices (F, K, G)
        out_h5["$elm/Ideal/F"] = _eval_mat_spline(mats.ideal.F_spline_lower)
        out_h5["$elm/Ideal/K"] = _eval_mat_spline(mats.ideal.K_spline)
        out_h5["$elm/Ideal/G"] = _eval_mat_spline(mats.ideal.G_spline)

        # Kinetic-modified matrices
        kin = mats.kinetic
        if kin !== nothing
            out_h5["$elm/Kinetic/A"] = _eval_mat_spline(kin.A_spline)
            out_h5["$elm/Kinetic/B"] = _eval_mat_spline(kin.B_spline)
            out_h5["$elm/Kinetic/C"] = _eval_mat_spline(kin.C_spline)
            out_h5["$elm/Kinetic/f0"] = _eval_mat_spline(kin.F0_spline)
            out_h5["$elm/Kinetic/K"] = _eval_mat_spline(kin.Kk_spline)
            out_h5["$elm/Kinetic/G"] = _eval_mat_spline(kin.G_spline_adj)
        end
    end

    # Self-describing metadata pass (long_name/units/dims + dimension scales).
    apply_main_h5_metadata!(out_h5)
    write_kinetic_validity!(out_h5, ctrl.kinetic, result)
    return nothing
end

"""
    write_kinetic_validity!(target, kinetic, result)

Write the validity diagnostics of the force operator's kinetic source `kinetic`. A no-op unless the module that
owns the source adds a method (KineticForces does, for `KineticForcesControl`).
"""
write_kinetic_validity!(target, kinetic, result) = nothing
write_kinetic_validity!(target, km::KineticMatrices, result) = write_kinetic_validity!(target, km.source, result)
