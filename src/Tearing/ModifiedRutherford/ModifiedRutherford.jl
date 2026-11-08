# The steady modified Rutherford equation: switchable terms and their sum F(w) per rational surface, from the
# force-balance Δ′, the resistive-layer geometry and the equilibrium's kinetic data.

module ModifiedRutherford

using HDF5
import CommonSolve
import FastInterpolations
using FastInterpolations: cubic_interp, Series, PeriodicBC, DerivOp

import ...Utilities
import ...Utilities: save_to_hdf5
using ...Utilities.HDF5Annotations: replace_group!
using ...Utilities.PhysicalConstants: M_E, M_P, E_CHG, MU_0
using ...InnerLayer.SLAYER: _slayer_profiles, radial_label, build_slayer_inputs, _layer_zeff, _layer_ion_mass, _layer_mass_per_electron,
    _build_resistivity_model
using Roots: find_zero, Brent

include("Geometry.jl")
include("Terms.jl")
include("Problem.jl")
include("HDF5Output.jl")

export MRESurfaceGeometry, mre_geometry, current_density_profile, sauter2002_trapped_fraction, bootstrap_coefficient
export classical_term, saturation_coefficient, saturation_term, bootstrap_term, curvature_term, lutjens_curvature_term, layer_transport_width, polarization_term,
    reduced_polarization_term,
    transport_width
export mre_roots, ModifiedRutherfordProblem, ModifiedRutherfordResult

end # module ModifiedRutherford
