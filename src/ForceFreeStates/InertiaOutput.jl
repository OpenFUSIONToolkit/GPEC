# Writer for an `IdealGrowthRateResult`. Layout under the caller's parent group (inputs live only under Input/):
#
#   ForceFreeStates/GrowthRate/
#   ├── gamma, gamma_sq, omega_star_pi -- growth rate, its square and the EPED1 ω_*pi reference
#   ├── psi, rho                       -- inertia density on the marginal solve's ψ_N grid
#   └── History/                       -- every (γ², λ_min, nzero) evaluated by the root search

const GROWTH_RATE_H5_ANNOTATIONS = [
    "gamma" => (; long_name="growth rate γ of the least-stable free-boundary mode with perpendicular inertia (0 when stable)", units="s^-1"),
    "gamma_sq" => (; long_name="γ² at which the least-stable free-boundary energy eigenvalue crosses zero (0 when stable)", units="s^-2"),
    "omega_star_pi" => (; long_name="peak ion diamagnetic frequency |ω_*pi| over the integration domain at the lowest n (NaN without kinetic T_i)",
        units="rad/s"),
    "psi" => (; long_name="normalized poloidal flux ψ_N grid of the marginal solve", dims=("psi",), scale="psi"),
    "rho" => (; long_name="perpendicular-inertia mass density ρ", units="kg/m^3", dims=("psi",), attach=(1 => "psi",)),
    "History/gamma_sq" => (; long_name="γ² evaluated by the growth-rate root search, in order", units="s^-2", dims=("iteration",)),
    "History/lambda_min" => (; long_name="power-normalized least-stable free-boundary energy at each evaluated γ² (per unit ⟨|ξ|²⟩)", dims=("iteration",)),
    "History/nzero" => (; long_name="number of zero crossings of the fixed-boundary critical eigenvalue at each evaluated γ²", dims=("iteration",))
]

"""
    save_to_hdf5(parent::Union{HDF5.File,HDF5.Group}, result::IdealGrowthRateResult)

Write `result` into a `ForceFreeStates/GrowthRate/` subgroup of `parent`, replacing any existing one, and record its
inputs in `Input/`. The growth rate, its square and the peak ω_*pi are scalars; the inertia density is sampled on
the marginal solve's ψ_N grid; `History/` holds every (γ², λ_min, nzero) the root search evaluated.
"""
function save_to_hdf5(parent::Union{HDF5.File,HDF5.Group}, result::IdealGrowthRateResult)
    ffs = haskey(parent, "ForceFreeStates") ? parent["ForceFreeStates"] : create_group(parent, "ForceFreeStates")
    g = Utilities.HDF5Annotations.replace_group!(ffs, "GrowthRate")
    g["gamma"] = result.gamma
    g["gamma_sq"] = result.gamma_sq
    g["omega_star_pi"] = result.omega_star_pi
    psi = collect(Float64, result.marginal.metric.xs)
    g["psi"] = psi
    g["rho"] = [_rho_at(result.inertia, x) for x in psi]
    g["History/gamma_sq"] = [h.gamma_sq for h in result.history]
    g["History/lambda_min"] = [h.lambda for h in result.history]
    g["History/nzero"] = [h.nzero for h in result.history]
    Utilities.HDF5Annotations.annotate!(g, GROWTH_RATE_H5_ANNOTATIONS)
    Utilities.record_inputs!(parent, result)
    return g
end
