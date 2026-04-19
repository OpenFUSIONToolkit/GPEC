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
#                      SLAYERResult  →  HDF5 (`slayer/` group)

module Runner

using LinearAlgebra
using HDF5

using ..Utilities
using ..Utilities: KineticProfiles, kinetic_profiles_from_toml,
                    kinetic_profiles_from_h5
using ..InnerLayer
using ..InnerLayer: SLAYERModel, SLAYERParameters, GGJModel, build_slayer_inputs
using ..Dispersion
using ..Dispersion: SurfaceCoupling, surface_coupling,
                     MultiSurfaceCoupling, multi_surface_coupling,
                     ScanResult, brute_force_scan,
                     AMRResult, amr_scan,
                     GrowthRateResult, find_growth_rates

include("Control.jl")
include("Result.jl")
include("run_slayer.jl")
include("HDF5Output.jl")

export SLAYERControl, slayer_control_from_toml, validate
export SLAYERResult, empty_slayer_result
export run_slayer, run_slayer_from_inputs
export write_slayer_hdf5!

end # module Runner
