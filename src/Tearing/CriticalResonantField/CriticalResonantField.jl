# CriticalResonantField.jl
#
# Uses the inner-layer delta from InnerLayer.jl to find the critical resonant field
# required for mode penetration at each rational surface. This can then be compared
# to the actual resonant field from the perturbed equilibrium to determine which modes
# are predicted to penetrate.

module CriticalResonantField

using Printf

using ..InnerLayer: InnerLayerModel, solve_inner

include("TorqueBalance.jl")

export TorqueBalance, cole_delta, torque_balance_value, torque_balance_window, torque_balance_scan

end # module CriticalResonantField
