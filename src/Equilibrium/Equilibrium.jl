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
include("EquilibriumTypes.jl")
include("SolveTolerances.jl")
include("GridRefinement.jl")
include("SeparatrixQLaw.jl")
include("FluxSurfaceMetrics.jl")
include("CoordinateInvariant.jl")
include("ReadEquilibrium.jl")
include("DirectEquilibrium.jl")
include("DirectEquilibriumArcLength.jl")
include("DirectEquilibriumByInversion.jl")
include("InverseEquilibrium.jl")
include("AnalyticEquilibrium.jl")
include("GeometryProfiles.jl")
include("KineticProfiles.jl")
include("Construction.jl")
include("PostProcessing.jl")

# --- Expose types and functions to the user ---
export setup_equilibrium, reform, EquilibriumConfig, PlasmaEquilibrium, EquilibriumParameters, AnalyticEquilibriumConfig,
    ProfileSplines, GeometryProfileSplines, compute_geometry_profiles,
    KineticProfileSplines, load_kinetic_profiles, shift_exb_rotation,
    KineticProfileData, read_kinetic_file, write_kinetic_h5
export flux_surface_metric, flux_surface_area
export wants_two_pass, refined_psi_grid, merge_mandatory_nodes, bracket_mandatory_nodes, enforce_min_spacing, implied_knot_count
export compute_sqrt_jac_delpsi, compute_sqrtamat, rootarea_to_area_weight, area_to_rootarea_weight

end # module Equilibrium
