"""
    AnalyticEquilibriumConfig

Supertype of the parameter structs of the analytic equilibrium kinds. Each subtype is the
`config_type` of one or more `ANALYTIC_EQ` rows, which pair it with an `eq_type`, its TOML
section, and its solver.
"""
abstract type AnalyticEquilibriumConfig end

include("Solovev.jl")
include("LargeAspectRatio.jl")
include("TJAnalytic.jl")

"""
    AnalyticEqSpec

Single-source-of-truth entry describing one analytic-equilibrium kind: the `gpec.toml`
section that carries its parameters, the `*Config` type built from that section, and the
run function that turns the config into solver input. `tj_analytic` and `tj_analytic_direct`
share `TJAnalyticConfig` but bind different run functions, so `run_fn` is per-entry rather
than derived from `config_type`.

## Fields

  - `section::String` — `gpec.toml` section key (e.g. `"SOL_INPUT"`)
  - `config_type::DataType` — `*Config` type, constructed via its `(::Dict)` / `(::String)` ctor
  - `run_fn::Function` — `(EquilibriumConfig, config) -> eq_input` solver entry point
"""
struct AnalyticEqSpec
    section::String
    config_type::DataType
    run_fn::Function
end

# Registry of analytic equilibrium kinds. Both `run_input` (fresh runs) and the rerun input
# builder dispatch off this table, so adding a new analytic kind is one new row.
const ANALYTIC_EQ = Dict(
    "sol" => AnalyticEqSpec("SOL_INPUT", SolovevConfig, sol_run),
    "lar" => AnalyticEqSpec("LAR_INPUT", LargeAspectRatioConfig, lar_run),
    "tj_analytic" => AnalyticEqSpec("TJ_ANALYTIC_INPUT", TJAnalyticConfig, tj_analytic_run),
    "tj_analytic_direct" => AnalyticEqSpec("TJ_ANALYTIC_INPUT", TJAnalyticConfig, tj_analytic_run_direct)
)
