# Tearing.jl
#
# Umbrella module grouping the tearing-mode analysis stack into a single
# layered hierarchy:
#
#   InnerLayer  -- pure physics: Δ_inner(Q) for GGJ or SLAYER models
#   Dispersion  -- physics-agnostic scan + contour-intersection root
#                  extraction (consumes any InnerLayerModel)
#   Runner      -- user-facing orchestration: TOML config, profile
#                  loading, HDF5 output, workflow hooks
#
# `InnerLayer` itself lives at the top level (`src/InnerLayer/`) and is loaded
# before `ForceFreeStates`, which depends on it for the matched-Δ′ Galerkin
# solve. Tearing re-binds it here so `Dispersion` and `Runner` reach it via
# `..InnerLayer`.

module Tearing

using ..Utilities

import ..InnerLayer as InnerLayer

include("Dispersion/Dispersion.jl")
include("Runner/Runner.jl")

import .Dispersion as Dispersion
import .Runner as Runner

export InnerLayer, Dispersion, Runner

end # module Tearing
