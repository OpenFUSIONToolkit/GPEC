# SLAYER.jl
#
# SLAYER (Slab Layer) drift-MHD inner-layer model. Port of the Fortran
# SLAYER code by J.K. Park (2023) at GPEC/slayer/, branch
# `slayer_growthrate`. Implements the Fitzpatrick (riccati_f)
# formulation: P_perp / P_tor transport, c_beta compressibility, D_norm
# normalized ion-skin scale, two-fluid drift coupling via Q_e, Q_i,
# iota_e. The standard `riccati()` growth-rate Fortran variant is not
# ported (use this Fitzpatrick path for the dispersion relation).
#
# The `riccati_del_s` Fortran variant IS ported, but as a standalone
# layer-thickness diagnostic (`slayer_layer_thickness` in
# `LayerThickness.jl`) rather than a `solve_inner` dispersion path: it
# returns the resistive layer thickness in meters at each rational
# surface, not an alternate growth rate.
#
# Type-parameter `S` of `SLAYERModel{S}` selects the Riccati formulation
# used for the dispersion relation; only `:fitzpatrick` is implemented.
#
# `Q = ω + iγ` is passed directly to `solve_inner` rather than stored on
# the parameter struct.

module SLAYER

using LinearAlgebra
using StaticArrays

import ..InnerLayerModel, ..InnerLayerResponse, ..solve_inner
using ...Utilities.PhysicalConstants
using ...Utilities.NeoclassicalResistivity
using ...Utilities.NeoclassicalResistivity: NeoResistivityModel, SpitzerModel,
    SauterNeoModel, RedlNeoModel,
    coulomb_log_e, eta_spitzer, trapped_fraction_eps, nu_star_e,
    eta_neoclassical

"""
    SLAYERModel{S} <: InnerLayerModel

SLAYER inner-layer model selector. The type parameter `S` selects the
Riccati formulation:

  - `:fitzpatrick` -- P_perp/P_tor Fitzpatrick formulation (default,
    mirrors Fortran `riccati_f` in `delta.f:323-438`)

Future dispersion variants (e.g. `:standard`) may be added but are not
currently implemented. The `del_s` formulation is exposed separately as
the layer-thickness diagnostic [`slayer_layer_thickness`](@ref), not as
a dispersion `solve_inner` path.
"""
struct SLAYERModel{S} <: InnerLayerModel end

SLAYERModel(; variant::Symbol=:fitzpatrick) = SLAYERModel{variant}()

include("LayerParameters.jl")
include("Riccati.jl")
include("LayerThickness.jl")
include("LayerInputs.jl")

export SLAYERModel, SLAYERParameters, slayer_parameters
export r_based_shear
export riccati_del_s, slayer_layer_thickness, LayerWidths
export surface_minor_radius, surface_da_dpsi, build_slayer_inputs
export NeoResistivityModel, SpitzerModel, SauterNeoModel, RedlNeoModel

end # module SLAYER
