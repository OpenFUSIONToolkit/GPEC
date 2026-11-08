# Callen's reduced modified Rutherford equation for coil-driven islands: the critical vacuum island width, the
# critical resonant field and the steady half-widths per resonant row of a single-n perturbed equilibrium.

module Callen

using HDF5
import CommonSolve
import FastInterpolations

import ...Utilities
import ...ForcingTerms
import ...Utilities: save_to_hdf5
using ...Utilities.HDF5Annotations: replace_group!
using ...Utilities.PhysicalConstants: M_P, E_CHG
using ...InnerLayer.SLAYER: _slayer_profiles

include("Problem.jl")
include("HDF5Output.jl")

export CallenProblem, CallenResult, callen_cubic_roots

end # module Callen
