# SLAYER.jl
#
# SLAYER (Slab Layer) drift-MHD inner-layer model. Port of the Fortran
# SLAYER code by J.K. Park (2023) at GPEC/slayer/, branch
# `slayer_growthrate`. Implements the Fitzpatrick (riccati_f)
# formulation: P_perp / P_tor transport, c_beta compressibility, D_norm
# normalized ion-skin scale, two-fluid drift coupling via Q_e, Q_i,
# iota_e. The standard `riccati()` and `riccati_del_s()` Fortran variants
# are intentionally not ported (use this Fitzpatrick path only).
#
# Type-parameter `S` of `SLAYERModel{S}` selects the Riccati formulation;
# only `:fitzpatrick` is implemented at present.
#
# `Q = ω + iγ` is passed directly to `solve_inner` rather than stored on
# the parameter struct.

module SLAYER

using LinearAlgebra
using StaticArrays

import ..InnerLayerModel, ..solve_inner
using ...Utilities.PhysicalConstants

"""
    SLAYERModel{S} <: InnerLayerModel

SLAYER inner-layer model selector. The type parameter `S` selects the
Riccati formulation:

  - `:fitzpatrick` -- P_perp/P_tor Fitzpatrick formulation (default,
    mirrors Fortran `riccati_f` in `delta.f:323-438`)

Future variants (e.g. `:standard`, `:del_s`) may be added but are not
currently implemented.
"""
struct SLAYERModel{S} <: InnerLayerModel end

SLAYERModel(; variant::Symbol=:fitzpatrick) = SLAYERModel{variant}()

include("LayerParameters.jl")
include("Riccati.jl")
include("LayerInputs.jl")

export SLAYERModel, SLAYERParameters, slayer_parameters
export r_based_shear
export surface_minor_radius, surface_da_dpsi, build_slayer_inputs

end # module SLAYER
