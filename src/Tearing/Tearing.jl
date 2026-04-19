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
# Relative-import dot counts inside this umbrella are simplified by
# re-binding `Utilities` at the Tearing level: all submodules reach
# Utilities via `..Utilities` (or `...Utilities` from sub-sub-modules)
# regardless of their depth in the original layout.

module Tearing

using ..Utilities

include("InnerLayer/InnerLayer.jl")
include("Dispersion/Dispersion.jl")
include("Runner/Runner.jl")

import .InnerLayer as InnerLayer
import .Dispersion as Dispersion
import .Runner as Runner

export InnerLayer, Dispersion, Runner

end # module Tearing
