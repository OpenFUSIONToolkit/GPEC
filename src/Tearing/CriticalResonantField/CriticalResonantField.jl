# Critical resonant field for error-field penetration at each rational surface, from the
# inner-layer Δ via torque balance; compare against the perturbed-equilibrium resonant field.

module CriticalResonantField

using Printf

using ..InnerLayer: InnerLayerModel, layer_response

include("TorqueBalance.jl")

export TorqueBalance, cole_delta, torque_balance_value, fold_search_range, torque_balance_scan

end # module CriticalResonantField
