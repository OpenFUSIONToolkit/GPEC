# Callen's reduced modified Rutherford equation for coil-driven islands: the critical vacuum island width, the
# critical resonant field and the steady half-widths per resonant row of a single-n perturbed equilibrium.

module Callen

using HDF5
import CommonSolve

import ...Utilities
import ...Utilities: save_to_hdf5
using ...Utilities.HDF5Annotations: replace_group!
using ...InnerLayer: layer_plasma
using ..ResonantForcing: resonant_forcing, callen_polarization_width

include("Problem.jl")
include("HDF5Output.jl")

export CallenProblem, CallenResult, callen_cubic_roots

end # module Callen
