# Construction.jl
#
# How a PlasmaEquilibrium is formed: the solver input for each source type, the constructors,
# and re-forming on a new grid.

"""
    run_input(eq_config::EquilibriumConfig, source) -> DirectRunInput or InverseRunInput

Build the solver input for `eq_config` from `source`: `nothing` reads `eq_config.eq_filename`, an
[`AnalyticEquilibriumConfig`](@ref) runs its analytic solver, an `IMASdd.dd` is read by
[`read_imas`](@ref), and a run input rebuilt from a stored ingest is rebound onto `eq_config`.
"""
function run_input(eq_config::EquilibriumConfig, ::Nothing)
    eq_type = eq_config.eq_type
    eq_type in EFIT_KINDS && return resolve_psihigh!(read_efit(eq_config))
    eq_type in ("chease2", "chease_ascii") && return read_chease_ascii(eq_config)
    eq_type in ("chease", "chease_binary") && return read_chease_binary(eq_config)
    eq_type == "ifile" && return resolve_psihigh!(read_ifile(eq_config))
    if haskey(ANALYTIC_EQ, eq_type)
        spec = ANALYTIC_EQ[eq_type]
        error("analytic eq_type=\"$eq_type\" needs its $(spec.config_type) as the source (built from the embedded [$(spec.section)] section)")
    end
    eq_type == "imas" && error("eq_type=\"imas\" needs an IMASdd.dd as the source")
    return error("Equilibrium type $(eq_type) is not implemented")
end

function run_input(eq_config::EquilibriumConfig, cfg::AnalyticEquilibriumConfig)
    spec = get(ANALYTIC_EQ, eq_config.eq_type, nothing)
    spec !== nothing && cfg isa spec.config_type ||
        error("eq_type=\"$(eq_config.eq_type)\" is not an analytic kind solved from a $(typeof(cfg))")
    return spec.run_fn(eq_config, cfg)
end

function run_input(eq_config::EquilibriumConfig, dd::IMASdd.dd)
    eq_config.eq_type == "imas" || error("an IMASdd.dd source needs eq_type=\"imas\", not \"$(eq_config.eq_type)\"")
    return read_imas(eq_config, dd)
end

# A run input rebuilt from an ingest is rebound onto the current config, resetting psihigh so a
# pass-1 clamped value does not shadow a psihigh overridden in the rerun TOML.
function run_input(eq_config::EquilibriumConfig, input::DirectRunInput)
    input.config = eq_config
    input.psihigh_resolved = eq_config.psihigh
    return eq_config.eq_type in EFIT_KINDS ? resolve_psihigh!(input) : input
end

function run_input(eq_config::EquilibriumConfig, input::InverseRunInput)
    input.config = eq_config
    input.psihigh_resolved = eq_config.psihigh
    return resolve_psihigh!(input)
end

"""
    PlasmaEquilibrium(eq_config::EquilibriumConfig, source=nothing; override_psi_nodes=nothing) -> PlasmaEquilibrium

Form the equilibrium `eq_config` describes from `source` (see [`run_input`](@ref)), run the
solver, and return it with global parameters, q-profile, and GSE diagnostics. Every other
constructor ends here.

`override_psi_nodes` bypasses the config-driven radial grid (`grid_type`/`mpsi`) and forms
the equilibrium on the given strictly increasing ψ_N node vector, whose endpoints must match
`psilow`/`psihigh`. Used by the two-pass auto-grid refinement through [`reform`](@ref); standalone
callers with `mpsi=0` get the coarse first-pass grid unless they refine and re-form.
"""
function PlasmaEquilibrium(eq_config::EquilibriumConfig, source=nothing; override_psi_nodes::Union{Nothing,Vector{Float64}}=nothing)
    eq_input = run_input(eq_config, source)

    plasma_equilibrium = if eq_config.eq_type == "efit_by_inversion"
        equilibrium_solver_by_inversion(eq_input; override_psi_nodes)
    elseif eq_config.eq_type == "efit_arclength"
        equilibrium_solver(eq_input, arclength_fieldline_int; override_psi_nodes)
    else
        equilibrium_solver(eq_input; override_psi_nodes)
    end

    # Forward the captured ingest so the gpec.h5 writer can snapshot it (nothing for analytic).
    plasma_equilibrium.ingest = eq_input.ingest
    # Analytic configs are still mutable; copy so later edits by the caller cannot change a re-form.
    source isa AnalyticEquilibriumConfig && (plasma_equilibrium.analytic = deepcopy(source))

    equilibrium_global_parameters!(plasma_equilibrium)
    equilibrium_qfind!(plasma_equilibrium)
    equilibrium_gse!(plasma_equilibrium)

    return plasma_equilibrium
end

"""
    reform(equil::PlasmaEquilibrium, psi_nodes::Vector{Float64}) -> PlasmaEquilibrium

Re-form `equil` on the ψ_N nodes `psi_nodes` from the source it remembers: the stored
`analytic` config, else its raw `ingest`, so no file is re-read. Only an equilibrium carrying
neither (one built with the positional constructor, or by a reader that captures no ingest)
falls back to re-reading `config.eq_filename`. The node endpoints must match `psilow`/`psihigh`.
"""
function reform(equil::PlasmaEquilibrium, psi_nodes::Vector{Float64})
    source = if equil.analytic !== nothing
        equil.analytic
    elseif equil.ingest isa DirectIngest
        build_direct_from_ingest(equil.config, equil.ingest)
    elseif equil.ingest isa InverseIngest
        build_inverse_from_ingest(equil.config, equil.ingest)
    else
        nothing
    end
    return PlasmaEquilibrium(equil.config, source; override_psi_nodes=psi_nodes)
end

"""
    PlasmaEquilibrium(path::AbstractString; eq_type="efit", kwargs...) -> PlasmaEquilibrium

Read the equilibrium file at `path` and return the processed equilibrium. Convenience entry
point of the scripting API: `kwargs` are [`EquilibriumConfig`](@ref) fields, so
`PlasmaEquilibrium("g000001.00001"; jac_type="hamada", mpsi=128)` is the whole setup.

Only file-based equilibria go through this constructor. Analytic and IMAS equilibria have
their own `PlasmaEquilibrium(::AnalyticEquilibriumConfig)` and `PlasmaEquilibrium(::IMASdd.dd)` forms.

```julia
eq = PlasmaEquilibrium("input.geqdsk"; jac_type="hamada")
```
"""
function PlasmaEquilibrium(path::AbstractString; eq_type::String="efit", kwargs...)
    haskey(ANALYTIC_EQ, eq_type) &&
        error("$eq_type is an analytic equilibrium: build it with PlasmaEquilibrium($(ANALYTIC_EQ[eq_type].config_type)(...)) instead")
    return PlasmaEquilibrium(EquilibriumConfig(; eq_type, eq_filename=abspath(path), kwargs...))
end

# Default analytic eq_type for a config: the first registry kind, in sorted order, that takes it.
_analytic_eq_type(cfg::AnalyticEquilibriumConfig) = first(k for k in sort!(collect(keys(ANALYTIC_EQ))) if ANALYTIC_EQ[k].config_type === typeof(cfg))

"""
    PlasmaEquilibrium(cfg::AnalyticEquilibriumConfig; eq_type=<from ANALYTIC_EQ>, kwargs...) -> PlasmaEquilibrium

Solve the analytic equilibrium described by `cfg` and return the processed equilibrium. The
`eq_type` is implied by the config type; `TJAnalyticConfig` defaults to `tj_analytic` and takes
`eq_type="tj_analytic_direct"` for the direct solve. `kwargs` are [`EquilibriumConfig`](@ref) fields.

```julia
eq = PlasmaEquilibrium(Equilibrium.SolovevConfig(; elongation=1.5); mpsi=0, psi_accuracy=1e-3)
```
"""
function PlasmaEquilibrium(cfg::AnalyticEquilibriumConfig; eq_type::String=_analytic_eq_type(cfg), kwargs...)
    return PlasmaEquilibrium(EquilibriumConfig(; eq_type, kwargs...), cfg)
end

"""
    PlasmaEquilibrium(dd::IMASdd.dd; kwargs...) -> PlasmaEquilibrium

Read the active equilibrium time slice of the IMAS data dictionary `dd` and return the processed
equilibrium. `kwargs` are [`EquilibriumConfig`](@ref) fields; `imas_cocos` sets the ψ convention.

```julia
eq = PlasmaEquilibrium(dd; imas_cocos=11, jac_type="hamada")
```
"""
function PlasmaEquilibrium(dd::IMASdd.dd; kwargs...)
    return PlasmaEquilibrium(EquilibriumConfig(; eq_type="imas", kwargs...), dd)
end
