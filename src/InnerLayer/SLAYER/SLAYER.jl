# SLAYER.jl
#
# SLAYER (Slab Layer) two-fluid drift-MHD inner-layer model (Fitzpatrick,
# Tearing Mode Dynamics in Tokamak Plasmas, IOP 2023; Park et al. 2022,
# Phys. Plasmas 29, 122505; Burgess et al. 2026). The dispersion path uses
# the Fitzpatrick layer formulation — P_perp / P_tor transport, c_beta
# compressibility, D_norm normalized ion-skin scale, two-fluid drift
# coupling via Q_e, Q_i, iota_e (see `Riccati.jl`).
#
# A separate `del_s` layer-thickness diagnostic (`slayer_layer_thickness` in
# `LayerThickness.jl`) returns the resistive layer width in meters at each
# rational surface.
#
# Type-parameter `S` of `SLAYERModel{S}` selects the Riccati formulation
# used for the dispersion relation; only `:fitzpatrick` is implemented.
#
# `Q = ω + iγ` is passed directly to `solve_inner` rather than stored on
# the parameter struct.

module SLAYER

using LinearAlgebra
using StaticArrays

import ..InnerLayerModel, ..InnerLayerResponse, ..solve_inner, ..InnerLayerParameters
using ...Utilities.PhysicalConstants
using ...Utilities.NeoclassicalResistivity
using ...Utilities.NeoclassicalResistivity: NeoResistivityModel, SpitzerModel,
    SpitzerHarmModel, SauterNeoModel, RedlNeoModel,
    coulomb_log_e, eta_spitzer, tau_ee_spitzer_harm, eta_spitzer_harm,
    trapped_fraction_eps, nu_star_e, eta_neoclassical

"""
    SLAYERModel{S} <: InnerLayerModel

SLAYER inner-layer model selector. The type parameter `S` selects the
Riccati formulation:

  - `:fitzpatrick` -- P_perp/P_tor Fitzpatrick formulation (default;
    Fitzpatrick 2023 two-fluid drift-MHD layer)

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
export surface_minor_radius, surface_da_dpsi, radial_label, build_slayer_inputs, toroidal_dgeo
export NeoResistivityModel, SpitzerModel, SpitzerHarmModel, SauterNeoModel, RedlNeoModel

end # module SLAYER
