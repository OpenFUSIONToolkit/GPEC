# Runner.jl
#
# Top-level orchestration module that ties together the building blocks
# from InnerLayer, Dispersion, and Utilities into the user-facing SLAYER
# tearing-mode analysis pipeline.
#
#   gpec.toml  [SLAYER]  →  SLAYERControl
#                            │
#   equilibrium + Δ'         │
#          +  profiles   →   build_slayer_inputs   →   SLAYERParameters[]
#                            │
#                            ▼
#              SurfaceCoupling[] / MultiSurfaceCoupling
#                            │
#                            ▼
#               brute_force_scan / amr_scan
#                            │
#                            ▼
#                   find_growth_rates
#                            │
#                            ▼
#                      SLAYERResult  →  HDF5 (`Tearing/` group)

module Runner

using LinearAlgebra
using Statistics: mean, median
using HDF5

using FastInterpolations: cubic_interp
using ..Utilities
using ..Utilities: KineticProfiles
using ...Equilibrium: read_kinetic_file, KineticProfileData
using ..InnerLayer
using ..InnerLayer:
    InnerLayerParameters, InnerLayerResponse, solve_inner,
    SLAYERModel, SLAYERParameters, build_slayer_inputs,
    GGJModel, GGJParameters,
    LayerWidths, slayer_layer_thickness
import ..build_ggj_inputs   # defined at the Tearing level (needs ForceFreeStates)
import ...Equilibrium as Equilibrium
using ...Equilibrium: read_kinetic_file, KineticProfileData, load_kinetic_profiles
using ..Dispersion
using ..Dispersion: SurfaceCoupling, surface_coupling,
    MultiSurfaceCoupling, multi_surface_coupling,
    ScanResult, brute_force_scan,
    AMRResult, amr_scan,
    MultiBoxAMRResult, multi_box_amr_scan, as_amr_result,
    GrowthRateResult, find_growth_rates

include("Control.jl")
include("Result.jl")
include("run_slayer.jl")
include("HDF5Output.jl")

export SLAYERControl, slayer_control_from_toml, validate
export SLAYERResult, empty_slayer_result
export run_slayer, run_slayer_from_inputs, ggj_inner_deltas
export write_slayer_hdf5!
export CriticalResonantFieldControl, critical_resonant_field_control_from_toml
export CriticalResonantFieldResult, empty_critical_resonant_field_result
export run_critical_resonant_field
export write_critical_resonant_field_hdf5!

end # module Runner
