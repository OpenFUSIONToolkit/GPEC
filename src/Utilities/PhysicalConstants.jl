"""
    PhysicalConstants

Shared physical constants used across GPEC modules. Values match the
Fortran GPEC/SLAYER conventions (sglobal_mod) so numerical results can
be directly compared.

All quantities in SI units.
"""
module PhysicalConstants

# Match the Fortran GPEC/SLAYER sglobal_mod values exactly so cross-code numerical comparison is meaningful.
const MU_0  = 4.0e-7 * π            # vacuum permeability         [H/m]
const M_E   = 9.1094e-31            # electron mass               [kg]
const M_P   = 1.6726e-27            # proton mass                 [kg]
const E_CHG = 1.6021917e-19         # elementary charge           [C]
const K_B   = 1.3807e-23            # Boltzmann constant          [J/K]
const EPS_0 = 8.8542e-12            # vacuum permittivity         [F/m]

export MU_0, M_E, M_P, E_CHG, K_B, EPS_0

end # module PhysicalConstants
