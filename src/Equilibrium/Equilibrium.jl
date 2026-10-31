"""
    Equilibrium

Reads or solves an MHD equilibrium and maps it to straight-field-line flux coordinates as a
[`PlasmaEquilibrium`](@ref).

## Where code goes

  - `Config.jl`: the run configuration (`EquilibriumConfig`) and global parameters (`EquilibriumParameters`).
  - `PlasmaEquilibrium.jl`: the result type and its spline containers.
  - `Construction.jl`: `run_input` for each source type, the `PlasmaEquilibrium` constructors, and `reform`.
  - `Sources/`: one file per input format, each building a `DirectRunInput` or `InverseRunInput`
    (`Ingest.jl` holds those and the ingest types). A new analytic model is one file in
    `Sources/Analytic/` holding its config and solver, plus one `ANALYTIC_EQ` row.
  - `Grid/`: radial ψ grids, first-pass and refined.
  - `Axisymmetric/`: code that assumes axisymmetry (the 2D solvers, the Grad-Shafranov residual,
    separatrix finding). A non-axisymmetric equilibrium gets a sibling folder.
  - `Derived/`: quantities computed from a formed `PlasmaEquilibrium`.
"""
module Equilibrium

# --- Module-level Dependencies ---

using Printf, OrdinaryDiffEq, DiffEqCallbacks, LinearAlgebra, HDF5
using Statistics: median
using Roots
using TOML
using FastInterpolations
using AdaptiveArrayPools
import StaticArrays: @MMatrix, SVector
import ..Utilities
import IMASdd

# --- Constants ---
const mu0 = 4π * 1e-7

# efit-family equilibrium kinds that share the separatrix-clamp / inversion handling.
const EFIT_KINDS = ("efit", "efit_arclength", "efit_by_inversion")

# --- Internal Module Structure ---
include("Config.jl")
include("Sources/Ingest.jl")
include("Sources/Analytic/Analytic.jl")
include("PlasmaEquilibrium.jl")
include("Axisymmetric/SolveTolerances.jl")
include("Grid/GridRefinement.jl")
include("Grid/PsiGrid.jl")
include("Axisymmetric/Separatrix.jl")
include("Derived/FluxSurfaceMetrics.jl")
include("Derived/CoordinateInvariant.jl")
include("Sources/GFile.jl")
include("Sources/Chease.jl")
include("Sources/IFile.jl")
include("Sources/IMAS.jl")
include("Axisymmetric/Direct.jl")
include("Axisymmetric/DirectArcLength.jl")
include("Axisymmetric/DirectByInversion.jl")
include("Axisymmetric/Inverse.jl")
include("Derived/GeometryProfiles.jl")
include("KineticProfiles.jl")
include("Construction.jl")
include("Derived/GlobalParameters.jl")
include("Axisymmetric/GradShafranovResidual.jl")

# --- Expose types and functions to the user ---
export reform, EquilibriumConfig, PlasmaEquilibrium, EquilibriumParameters, AnalyticEquilibriumConfig,
    ProfileSplines, GeometryProfileSplines, compute_geometry_profiles,
    KineticProfileSplines, kinetic_profile_splines, shift_exb_rotation,
    KineticProfileData, IonSpecies, read_kinetic_file, write_kinetic_h5, attach_kinetic_profiles!
export flux_surface_metric, flux_surface_area
export wants_two_pass, refined_psi_grid, merge_mandatory_nodes, bracket_mandatory_nodes, enforce_min_spacing, implied_knot_count
export compute_jac_delpsi, compute_sqrt_jac_delpsi, compute_sqrtamat, rootarea_to_area_weight, area_to_rootarea_weight

include("Output.jl")

end # module Equilibrium
