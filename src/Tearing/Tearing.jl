# Tearing.jl
#
# Tearing-mode stability of a finished force-balance solve:
#
#   Dispersion  -- physics-agnostic scan + contour-intersection root
#                  extraction (consumes any InnerLayerModel)
#   CriticalResonantField -- real-Q torque-balance scan for the critical
#                  resonant field of error-field penetration
#   TearingProblem / TearingConfig / TearingResult -- the public solve,
#                  `solve(TearingProblem(ffs; ...), model; config)`, with one
#                  method per inner-layer model (SLAYERModel, GGJModel)
#
# `InnerLayer` itself lives at the top level (`src/InnerLayer/`) and is loaded
# before `ForceFreeStates`, which depends on it for the matched-Δ′ Galerkin
# solve. Tearing re-binds it here so `Dispersion` reaches it via `..InnerLayer`.

module Tearing

using LinearAlgebra
using Statistics: mean, median
using HDF5

import CommonSolve
using FastInterpolations: cubic_interp
using ..Utilities
using ..Utilities: KineticProfiles
using ..Equilibrium: KineticProfileData
using ..ForceFreeStates: ForceFreeStatesResult, ggj_parameters

import ..InnerLayer as InnerLayer
using .InnerLayer: InnerLayerParameters, InnerLayerResponse, solve_inner,
    SLAYERModel, SLAYERParameters, build_slayer_inputs,
    GGJModel, GGJParameters,
    LayerWidths, slayer_layer_thickness

include("LayerOverlap.jl")
include("Dispersion/Dispersion.jl")
include("CriticalResonantField/CriticalResonantField.jl")

import .Dispersion as Dispersion
import .CriticalResonantField as CriticalResonantField
using .Dispersion: SurfaceCoupling, surface_coupling,
    MultiSurfaceCoupling, multi_surface_coupling,
    ScanResult, brute_force_scan,
    AMRResult, amr_scan,
    MultiBoxAMRResult, multi_box_amr_scan, as_amr_result,
    GrowthRateResult, find_growth_rates
using .CriticalResonantField: TorqueBalance, torque_balance_scan

include("Config.jl")
include("Result.jl")
include("TearingProblem.jl")
include("SLAYERTearing.jl")
include("GGJTearing.jl")
include("HDF5Output.jl")

export InnerLayer, Dispersion, CriticalResonantField
export resistive_layer_overlap, LayerOverlapScan
export TearingProblem, TearingConfig, TearingResult
export run_slayer_from_inputs, ggj_inner_deltas
export save_to_hdf5
export CriticalResonantFieldControl, CriticalResonantFieldScan, CriticalResonantFieldResult
export run_critical_resonant_field

end # module Tearing
