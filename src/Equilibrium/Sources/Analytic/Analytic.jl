"""
    AnalyticEquilibriumConfig

Supertype of the parameter structs of the analytic equilibrium kinds. Each subtype is the
`config_type` of one `ANALYTIC_EQ` row, which pairs it with an `eq_type`, its deck section,
and its solver.
"""
abstract type AnalyticEquilibriumConfig end

include("Solovev.jl")
include("LargeAspectRatio.jl")
include("TJAnalytic.jl")

"""
    AnalyticEqSpec

Single-source-of-truth entry describing one analytic-equilibrium kind: the `gpec.toml`
section that carries its parameters, the `*Config` type built from that section, and the
run function that turns the config into solver input.

## Fields

  - `section::String` — `gpec.toml` section, the config type's name without `Config` (e.g. `"Solovev"`)
  - `config_type::DataType` — `*Config` type, built from the section's keys as keywords
  - `run_fn::Function` — `(EquilibriumConfig, config) -> eq_input` solver entry point
"""
struct AnalyticEqSpec
    section::String
    config_type::DataType
    run_fn::Function
end

# Registry of analytic equilibrium kinds, keyed by the internal eq_type each config implies. `run_input` and the
# deck parser both dispatch off this table, so adding a new analytic kind is one new row.
const ANALYTIC_EQ = Dict(
    "sol" => AnalyticEqSpec("Solovev", SolovevConfig, sol_run),
    "lar" => AnalyticEqSpec("LargeAspectRatio", LargeAspectRatioConfig, lar_run),
    "tj_analytic" => AnalyticEqSpec("TJAnalytic", TJAnalyticConfig, tj_analytic_run)
)
