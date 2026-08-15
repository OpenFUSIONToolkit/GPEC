# Metadata tables for the datasets written by write_outputs_to_HDF5 (the main gpec.h5
# writer), applied post-write by Utilities.HDF5Annotations.annotate!. Sub-writers
# (Galerkin, PerturbedEquilibrium, KineticForces, Tearing) keep their tables next to
# their own writers. Paths absent from a given run are skipped automatically.
#
# Conventions (docs/development/hdf5-conventions.md): units are SI strings, "1" for
# dimensionless; ψ always means the normalized poloidal flux ψ_N ∈ [0, 1]; stability
# energies are power-normalized (per unit surface-averaged |ξ|², not joules); `dims`
# lists axis names in Julia (column-major) order, axis 1 first.

const MAIN_H5_ANNOTATIONS = [
    # --- Info/ ---
    "Info/git_version" => (; long_name="GPEC git version that produced this file"),
    "Info/mpert" => (; long_name="number of poloidal harmonics per toroidal mode"),
    "Info/mlow" => (; long_name="lowest poloidal mode number m"),
    "Info/mhigh" => (; long_name="highest poloidal mode number m"),
    "Info/npert" => (; long_name="number of toroidal mode numbers"),
    "Info/nlow" => (; long_name="lowest toroidal mode number n"),
    "Info/nhigh" => (; long_name="highest toroidal mode number n"),
    "Info/mn_index" => (; long_name="(m, n) mode numbers for each perturbation index", dims=("mode_index", "m_or_n")),
    "Info/psilim" => (; long_name="normalized poloidal flux at the integration boundary"),
    "Info/qlim" => (; long_name="safety factor q at the integration boundary"),
    "Info/dqdpsi_lim" => (; long_name="dq/dψ_N at the integration boundary"),
    # --- Equilibrium/ scalars (written per-field when set; superset listed) ---
    "Equilibrium/ro" => (; long_name="R-coordinate of the magnetic axis", units="m"),
    "Equilibrium/zo" => (; long_name="Z-coordinate of the magnetic axis", units="m"),
    "Equilibrium/psio" => (; long_name="total poloidal flux difference |ψ_axis - ψ_boundary|", units="Wb/rad"),
    "Equilibrium/rsep" => (; long_name="R-coordinates of the plasma boundary", units="m"),
    "Equilibrium/zsep" => (; long_name="Z-coordinates of the plasma boundary", units="m"),
    "Equilibrium/rext" => (; long_name="R-coordinates of the plasma edge", units="m"),
    "Equilibrium/zext" => (; long_name="Z-coordinates of the plasma edge", units="m"),
    "Equilibrium/psi0" => (; long_name="normalized poloidal flux at reference location"),
    "Equilibrium/b0" => (; long_name="total magnetic field strength at the axis", units="T"),
    "Equilibrium/q0" => (; long_name="safety factor at the magnetic axis"),
    "Equilibrium/qmin" => (; long_name="minimum safety factor in the plasma"),
    "Equilibrium/qmax" => (; long_name="maximum safety factor in the plasma"),
    "Equilibrium/qa" => (; long_name="safety factor at the plasma edge"),
    "Equilibrium/q95" => (; long_name="safety factor at the 95% flux surface"),
    "Equilibrium/qextrema_psi" => (; long_name="normalized poloidal flux at q-profile extrema"),
    "Equilibrium/qextrema_q" => (; long_name="safety factor at q-profile extrema"),
    "Equilibrium/mextrema" => (; long_name="number of extrema in the q-profile"),
    "Equilibrium/rmean" => (; long_name="mean major radius of the plasma", units="m"),
    "Equilibrium/amean" => (; long_name="mean minor radius of the plasma", units="m"),
    "Equilibrium/aratio" => (; long_name="aspect ratio R0/a"),
    "Equilibrium/kappa" => (; long_name="plasma elongation"),
    "Equilibrium/delta1" => (; long_name="upper triangularity"),
    "Equilibrium/delta2" => (; long_name="lower triangularity"),
    "Equilibrium/bt0" => (; long_name="toroidal field at the axis", units="T"),
    "Equilibrium/crnt" => (; long_name="plasma current", units="A"),
    "Equilibrium/bwall" => (; long_name="toroidal field at the wall", units="T"),
    "Equilibrium/betat" => (; long_name="toroidal beta"),
    "Equilibrium/betan" => (; long_name="normalized beta β_N"),
    "Equilibrium/betap1" => (; long_name="poloidal beta (definition 1)"),
    "Equilibrium/betap2" => (; long_name="poloidal beta (definition 2)"),
    "Equilibrium/betap3" => (; long_name="poloidal beta (definition 3)"),
    "Equilibrium/betaj" => (; long_name="current-weighted beta"),
    "Equilibrium/li1" => (; long_name="internal inductance (definition 1)"),
    "Equilibrium/li2" => (; long_name="internal inductance (definition 2)"),
    "Equilibrium/li3" => (; long_name="internal inductance (definition 3)"),
    "Equilibrium/volume" => (; long_name="plasma volume", units="m^3"),
    "Equilibrium/bt_sign" => (; long_name="sign of the toroidal field"),
    "Equilibrium/psi_norm" => (; long_name="normalized poloidal flux at the axis"),
    "Equilibrium/b_norm" => (; long_name="normalized total field strength at the axis"),
    "Equilibrium/psi_axis" => (; long_name="poloidal flux at the magnetic axis", units="Wb/rad"),
    "Equilibrium/psi_boundary" => (; long_name="poloidal flux at the plasma boundary", units="Wb/rad"),
    "Equilibrium/psi_axis_norm" => (; long_name="normalized poloidal flux at the axis"),
    "Equilibrium/psi_boundary_norm" => (; long_name="normalized poloidal flux at the boundary"),
    "Equilibrium/psi_axis_offset" => (; long_name="offset applied to the axis poloidal flux", units="Wb/rad"),
    "Equilibrium/psi_boundary_offset" => (; long_name="offset applied to the boundary poloidal flux", units="Wb/rad"),
    "Equilibrium/psi_axis_sign" => (; long_name="sign of the axis poloidal flux"),
    "Equilibrium/psi_boundary_sign" => (; long_name="sign of the boundary poloidal flux"),
    "Equilibrium/psi_boundary_zero" => (; long_name="flag: boundary poloidal flux is zero"),
    "Equilibrium/verbose" => (; long_name="flag: equilibrium setup ran with verbose output (diagnostic echo)"),
    "Equilibrium/diagnose_src" => (; long_name="flag: equilibrium source-data diagnostics were enabled (diagnostic echo)"),
    "Equilibrium/diagnose_maxima" => (; long_name="flag: equilibrium extrema diagnostics were enabled (diagnostic echo)"),
    # --- Equilibrium/Profiles/ (1-D profiles on the ψ_N grid xs) ---
    "Equilibrium/Profiles/xs" => (; long_name="normalized poloidal flux ψ_N profile grid", scale="psi"),
    "Equilibrium/Profiles/2piF" => (; long_name="2π F with F = R B_φ the toroidal field function", units="T*m", dims=("psi",), attach=(1 => "Equilibrium/Profiles/xs",)),
    "Equilibrium/Profiles/mu0p" => (; long_name="μ0 × plasma pressure", units="T^2", dims=("psi",), attach=(1 => "Equilibrium/Profiles/xs",)),
    "Equilibrium/Profiles/dVdpsi" => (; long_name="flux-surface volume derivative dV/dψ_N", units="m^3", dims=("psi",), attach=(1 => "Equilibrium/Profiles/xs",)),
    "Equilibrium/Profiles/q" => (; long_name="safety factor profile", dims=("psi",), attach=(1 => "Equilibrium/Profiles/xs",)),
    # --- Equilibrium/Geometry/ (2-D flux-coordinate maps on (xs, ys) = (ψ_N, θ/2π)) ---
    "Equilibrium/Geometry/xs" => (; long_name="normalized poloidal flux ψ_N geometry grid", scale="psi"),
    "Equilibrium/Geometry/ys" => (; long_name="normalized poloidal angle θ/2π geometry grid", scale="theta"),
    "Equilibrium/Geometry/rcoords" =>
        (; long_name="squared minor-radius coordinate r² of the working coordinate map", units="m^2", dims=("psi", "theta"),
            attach=(1 => "Equilibrium/Geometry/xs", 2 => "Equilibrium/Geometry/ys")),
    "Equilibrium/Geometry/offset" =>
        (; long_name="poloidal-angle offset of the working coordinate map (fraction of 2π)", dims=("psi", "theta"),
            attach=(1 => "Equilibrium/Geometry/xs", 2 => "Equilibrium/Geometry/ys")),
    "Equilibrium/Geometry/nu" =>
        (; long_name="toroidal-angle offset ν = φ − 2πζ of the working coordinate map", units="rad", dims=("psi", "theta"),
            attach=(1 => "Equilibrium/Geometry/xs", 2 => "Equilibrium/Geometry/ys")),
    "Equilibrium/Geometry/jac" =>
        (; long_name="Jacobian of the (ψ_N, θ, ζ) working coordinates", units="m^3", dims=("psi", "theta"),
            attach=(1 => "Equilibrium/Geometry/xs", 2 => "Equilibrium/Geometry/ys")),
    # --- LocalStability/ ---
    "LocalStability/D_I" => (; long_name="Mercier ideal interchange criterion D_I", dims=("psi",)),
    "LocalStability/D_R" => (; long_name="Glasser-Greene-Johnson resistive interchange criterion D_R", dims=("psi",)),
    "LocalStability/ballooning_Delta_prime" => (; long_name="high-n ballooning Δ' (distinct from the tearing Δ')", dims=("psi",)),
    "LocalStability/psi" => (; long_name="normalized poloidal flux ψ_N of the ballooning α boundary scan"),
    "LocalStability/alpha" => (; long_name="experimental normalized pressure gradient α", dims=("psi",)),
    "LocalStability/alpha_critical" => (; long_name="critical normalized pressure gradient α for first ballooning stability", dims=("psi",)),
    # --- ForceFreeStates/Solutions/ForwardIntegration/ ---
    "ForceFreeStates/Solutions/ForwardIntegration/nstep" => (; long_name="number of saved solution snapshots"),
    "ForceFreeStates/Solutions/ForwardIntegration/nstep_total" => (; long_name="total ODE solver steps taken"),
    "ForceFreeStates/Solutions/ForwardIntegration/psi" => (; long_name="normalized poloidal flux ψ_N at saved solution snapshots", scale="psi"),
    "ForceFreeStates/Solutions/ForwardIntegration/q" =>
        (; long_name="safety factor at saved solution snapshots", dims=("psi",), attach=(1 => "ForceFreeStates/Solutions/ForwardIntegration/psi",)),
    "ForceFreeStates/Solutions/ForwardIntegration/xi_psi" =>
        (; long_name="fundamental-matrix solutions ξ^ψ (arbitrary amplitude)", dims=("mode", "solution", "psi"), attach=(3 => "ForceFreeStates/Solutions/ForwardIntegration/psi",)),
    "ForceFreeStates/Solutions/ForwardIntegration/u2" =>
        (; long_name="conjugate momenta of the fundamental-matrix solutions (arbitrary amplitude)", dims=("mode", "solution", "psi")),
    "ForceFreeStates/Solutions/ForwardIntegration/dxi_psidpsi" =>
        (; long_name="ψ_N derivative of the fundamental-matrix solutions ξ^ψ (arbitrary amplitude)", dims=("mode", "solution", "psi")),
    "ForceFreeStates/Solutions/ForwardIntegration/xi_s" =>
        (; long_name="Clebsch surface-displacement solutions Ξ_s (arbitrary amplitude)", dims=("mode", "solution", "psi"),
            attach=(3 => "ForceFreeStates/Solutions/ForwardIntegration/psi",)),
    "ForceFreeStates/Solutions/ForwardIntegration/crit" =>
        (; long_name="DCON zero-crossing criterion at saved snapshots", dims=("psi",), attach=(1 => "ForceFreeStates/Solutions/ForwardIntegration/psi",)),
    # --- ForceFreeStates/EdgeScan/ (power-normalized (W, N) pencil energies) ---
    "ForceFreeStates/EdgeScan/psi" => (; long_name="normalized poloidal flux ψ_N of the edge truncation scan", scale="psi"),
    "ForceFreeStates/EdgeScan/q" => (; long_name="safety factor at scan points", dims=("psi",), attach=(1 => "ForceFreeStates/EdgeScan/psi",)),
    "ForceFreeStates/EdgeScan/total_energy" =>
        (; long_name="power-normalized total energy of the least-stable free-boundary mode (per unit ⟨|ξ|²⟩)", dims=("psi",), attach=(1 => "ForceFreeStates/EdgeScan/psi",)),
    "ForceFreeStates/EdgeScan/plasma_energy" =>
        (; long_name="power-normalized plasma energy of the least-stable mode (per unit ⟨|ξ|²⟩)", dims=("psi",), attach=(1 => "ForceFreeStates/EdgeScan/psi",)),
    "ForceFreeStates/EdgeScan/vacuum_energy" =>
        (; long_name="power-normalized vacuum energy of the least-stable mode (per unit ⟨|ξ|²⟩)", dims=("psi",), attach=(1 => "ForceFreeStates/EdgeScan/psi",)),
    "ForceFreeStates/EdgeScan/vacuum_eigenvalue" =>
        (; long_name="least vacuum eigenvalue of the (W, N) pencil at scan points", dims=("psi",), attach=(1 => "ForceFreeStates/EdgeScan/psi",)),
    # --- SingularSurfaces/ ---
    "SingularSurfaces/rational_count" => (; long_name="number of rational (singular) surfaces in the domain"),
    "SingularSurfaces/rational_psi" => (; long_name="normalized poloidal flux ψ_N of each rational surface", scale="psi_rational"),
    "SingularSurfaces/rational_q" => (; long_name="safety factor q = m/n at each rational surface", dims=("surface",), attach=(1 => "SingularSurfaces/rational_psi",)),
    "SingularSurfaces/dqdpsi" => (; long_name="dq/dψ_N at each rational surface", dims=("surface",), attach=(1 => "SingularSurfaces/rational_psi",)),
    "SingularSurfaces/rational_m" => (; long_name="resonant poloidal mode numbers per surface (0-padded)", dims=("surface", "mode")),
    "SingularSurfaces/rational_n" => (; long_name="resonant toroidal mode numbers per surface (0-padded)", dims=("surface", "mode")),
    "SingularSurfaces/D_I" => (; long_name="Mercier D_I evaluated at each rational surface", dims=("surface",), attach=(1 => "SingularSurfaces/rational_psi",)),
    "SingularSurfaces/ca_left" =>
        (; long_name="asymptotic large/small-solution coefficient matrices just left of each surface (zero-extent when not computed — ideal crossings only)",
            dims=("mode", "solution", "large_small", "surface")),
    "SingularSurfaces/ca_right" =>
        (; long_name="asymptotic large/small-solution coefficient matrices just right of each surface (zero-extent when not computed — ideal crossings only)",
            dims=("mode", "solution", "large_small", "surface")),
    "SingularSurfaces/E" => (; long_name="Glasser-Greene-Johnson coefficient E per surface", dims=("surface",), attach=(1 => "SingularSurfaces/rational_psi",)),
    "SingularSurfaces/F" => (; long_name="Glasser-Greene-Johnson coefficient F per surface", dims=("surface",), attach=(1 => "SingularSurfaces/rational_psi",)),
    "SingularSurfaces/G" => (; long_name="Glasser-Greene-Johnson coefficient G per surface", dims=("surface",), attach=(1 => "SingularSurfaces/rational_psi",)),
    "SingularSurfaces/H" => (; long_name="Glasser-Greene-Johnson coefficient H per surface", dims=("surface",), attach=(1 => "SingularSurfaces/rational_psi",)),
    "SingularSurfaces/K" => (; long_name="Glasser-Greene-Johnson coefficient K per surface", dims=("surface",), attach=(1 => "SingularSurfaces/rational_psi",)),
    "SingularSurfaces/M" => (; long_name="Glasser-Greene-Johnson coefficient M per surface", dims=("surface",), attach=(1 => "SingularSurfaces/rational_psi",)),
    "SingularSurfaces/avg_bsq_over_dpsisq" =>
        (; long_name="flux-surface average ⟨B²/|∇ψ_N|²⟩ per surface", units="T^2*m^2", dims=("surface",), attach=(1 => "SingularSurfaces/rational_psi",)),
    "SingularSurfaces/avg_bsq" => (; long_name="flux-surface average ⟨B²⟩ per surface", units="T^2", dims=("surface",), attach=(1 => "SingularSurfaces/rational_psi",)),
    "SingularSurfaces/mu0p" => (; long_name="μ0 × local pressure at each surface", units="T^2", dims=("surface",), attach=(1 => "SingularSurfaces/rational_psi",)),
    "SingularSurfaces/dmu0pdpsi" => (; long_name="μ0 × dp/dψ_N at each surface", units="T^2", dims=("surface",), attach=(1 => "SingularSurfaces/rational_psi",)),
    "SingularSurfaces/dVdpsi" => (; long_name="dV/dψ_N at each surface", units="m^3", dims=("surface",), attach=(1 => "SingularSurfaces/rational_psi",)),
    "SingularSurfaces/Delta_prime_matrix" => (; long_name="inter-surface Δ' matrix (PEST3 convention, STRIDE BVP with vacuum coupling)", dims=("surface_row", "surface_col")),
    "SingularSurfaces/Delta_prime_raw" =>
        (; long_name="raw 2msing×2msing outer-region D' matrix, side-major ordering [L_s1, R_s1, ...]", dims=("surface_side_row", "surface_side_col")),
    "SingularSurfaces/Delta_coil" => (; long_name="edge coil-response matrix (edge mode × surface-side)", dims=("mode", "surface_side")),
    # --- SingularSurfaces/Kinetic/ ---
    "SingularSurfaces/Kinetic/rational_count" => (; long_name="number of kinetic singular surfaces (det(F̄) near-zeros)"),
    "SingularSurfaces/Kinetic/rational_psi" => (; long_name="normalized poloidal flux ψ_N of kinetic singular surfaces"),
    "SingularSurfaces/Kinetic/rational_q" => (; long_name="safety factor at kinetic singular surfaces"),
    "SingularSurfaces/Kinetic/dqdpsi" => (; long_name="dq/dψ_N at kinetic singular surfaces"),
    "SingularSurfaces/Kinetic/scan_psi" => (; long_name="ψ_N grid of the cond(F̄) scan"),
    "SingularSurfaces/Kinetic/scan_cond" => (; long_name="condition number of F̄ along the scan"),
    "SingularSurfaces/Kinetic/scan_threshold" => (; long_name="cond(F̄) threshold used to flag kinetic singular surfaces"),
    # --- ForceFreeStates/FreeBoundaryStability/ (power-normalized (W, N) pencil) ---
    "ForceFreeStates/FreeBoundaryStability/W_freeboundary" => (; long_name="power-normalized free-boundary energy matrix W (per unit ⟨|ξ|²⟩)", dims=("mode_row", "mode_col")),
    "ForceFreeStates/FreeBoundaryStability/W_plasma" => (; long_name="power-normalized plasma energy matrix (per unit ⟨|ξ|²⟩)", dims=("mode_row", "mode_col")),
    "ForceFreeStates/FreeBoundaryStability/W_vacuum" => (; long_name="power-normalized vacuum energy matrix (per unit ⟨|ξ|²⟩)", dims=("mode_row", "mode_col")),
    "ForceFreeStates/FreeBoundaryStability/W_freeboundary_eigenmodes" =>
        (; long_name="generalized eigenvectors of the (W, N) pencil, columns sorted most-unstable first, unit power norm", dims=("mode", "eigenmode")),
    "ForceFreeStates/FreeBoundaryStability/eigenmode_energies" =>
        (; long_name="generalized eigenvalues of the (W, N) pencil: total energy per unit ⟨|ξ|²⟩, coordinate-invariant", dims=("eigenmode",)),
    "ForceFreeStates/FreeBoundaryStability/eigenmode_plasma_energies" => (; long_name="plasma contribution to the power-normalized eigenmode energies", dims=("eigenmode",)),
    "ForceFreeStates/FreeBoundaryStability/eigenmode_vacuum_energies" => (; long_name="vacuum contribution to the power-normalized eigenmode energies", dims=("eigenmode",)),
    "ForceFreeStates/FreeBoundaryStability/vacuum_eigenvalue" => (; long_name="least eigenvalue of the vacuum energy matrix"),
    # --- SurfaceGeometries/ ---
    "SurfaceGeometries/Plasma/x" => (; long_name="Cartesian x of plasma-surface point cloud", units="m"),
    "SurfaceGeometries/Plasma/y" => (; long_name="Cartesian y of plasma-surface point cloud", units="m"),
    "SurfaceGeometries/Plasma/z" => (; long_name="Cartesian z of plasma-surface point cloud", units="m"),
    "SurfaceGeometries/Wall/x" => (; long_name="Cartesian x of wall point cloud", units="m"),
    "SurfaceGeometries/Wall/y" => (; long_name="Cartesian y of wall point cloud", units="m"),
    "SurfaceGeometries/Wall/z" => (; long_name="Cartesian z of wall point cloud", units="m"),
]

# Euler-Lagrange operator matrices: same wording per letter, Ideal/ and Kinetic/ variants.
const _ELM_IDEAL_LETTERS = [
    ("A", "Euler-Lagrange primitive coefficient matrix A"),
    ("B", "Euler-Lagrange primitive coefficient matrix B"),
    ("C", "Euler-Lagrange primitive coefficient matrix C"),
    ("D", "Euler-Lagrange primitive coefficient matrix D"),
    ("E", "Euler-Lagrange primitive coefficient matrix E"),
    ("H", "Euler-Lagrange primitive coefficient matrix H"),
    ("F", "Euler-Lagrange derived coefficient matrix F"),
    ("K", "Euler-Lagrange derived coefficient matrix K"),
    ("G", "Euler-Lagrange derived coefficient matrix G"),
]
const _ELM_KINETIC_LETTERS = vcat(_ELM_IDEAL_LETTERS, [("f0", "raw kinetic component matrix f0")])
const ELM_H5_ANNOTATIONS = vcat(
    ["ForceFreeStates/EulerLagrangeMatrices/psi" => (; long_name="normalized poloidal flux ψ_N grid of the operator matrices", scale="psi")],
    ["ForceFreeStates/EulerLagrangeMatrices/Ideal/$l" =>
        (; long_name="ideal " * d, dims=("psi", "mode_row", "mode_col"), attach=(1 => "ForceFreeStates/EulerLagrangeMatrices/psi",)) for (l, d) in _ELM_IDEAL_LETTERS],
    ["ForceFreeStates/EulerLagrangeMatrices/Kinetic/$l" =>
        (; long_name="kinetic-modified " * d, dims=("psi", "mode_row", "mode_col"), attach=(1 => "ForceFreeStates/EulerLagrangeMatrices/psi",)) for (l, d) in _ELM_KINETIC_LETTERS]
)

"""
    apply_main_h5_metadata!(out_h5, intr)

Apply the self-describing metadata contract to the datasets written by
`write_outputs_to_HDF5`: `long_name`/`units`/`dims` attributes plus HDF5 Dimension
Scales for the shared coordinate datasets (ψ_N grids, rational-surface ψ).
"""
function apply_main_h5_metadata!(out_h5, intr)
    # Scales and attachments are declared in-table via the scale/attach entry fields.
    Utilities.HDF5Annotations.annotate!(out_h5, MAIN_H5_ANNOTATIONS)
    Utilities.HDF5Annotations.annotate!(out_h5, ELM_H5_ANNOTATIONS)
    return out_h5
end
