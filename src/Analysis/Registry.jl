# Registry of the Analysis plots: which data products each one draws, which sources (a result in memory or a
# gpec.h5 path) supply them, and what a source must have computed for the plot to draw. Each plotting submodule
# registers its own plots at load time; `capabilities` is the single record of what each source holds.

using HDF5

import ..Equilibrium: PlasmaEquilibrium
import ..LocalStability: LocalStabilityResult
import ..ForceFreeStates: ForceFreeStatesResult
import ..PerturbedEquilibrium: PerturbedEquilibriumState
import ..ErrorFields as EF

"""
    PlotEntry

A registered plot.

## Fields

  - `fn::Function` - The plotting function.
  - `sources::Tuple` - The types it accepts as its first argument; `AbstractString` means a `gpec.h5` path.
  - `needs::Tuple{Vararg{Symbol}}` - The capabilities (see [`capabilities`](@ref)) a source must have for it to draw.
  - `description::String` - One line on what it draws.
"""
struct PlotEntry
    fn::Function
    sources::Tuple
    needs::Tuple{Vararg{Symbol}}
    description::String
end

const PLOTS = PlotEntry[]

# Add a plot to the registry; called by each plotting submodule as it loads.
register!(fn, sources, needs, description) = (push!(PLOTS, PlotEntry(fn, Tuple(sources), Tuple(needs), description)); fn)

"""
What each capability needs, worded as what to change in the run to get it.
"""
const REASONS = Dict{Symbol,String}(
    :equilibrium => "needs an equilibrium",
    :singular_surfaces => "needs rational surfaces in the integration domain",
    :solution_profiles => "needs stored ξ profiles: use the Forward integrator or a matched Galerkin solve (Riccati and unmatched Galerkin store none)",
    :fixed_boundary_criterion => "needs the critical determinant scan: use the Forward or Riccati integrator (Galerkin computes none)",
    :free_boundary => "needs the free-boundary energies: run with vac_flag = true and the Forward or Riccati integrator",
    :edge_scan => "needs the edge stability scan: set dW_edge_scan_start below psilim",
    :delta_prime => "needs the Δ′ matrix: use the Riccati or Galerkin integrator (Forward computes none)",
    :kinetic_scan => "needs the kinetic singular-surface scan: run with a kinetic source and singfac_min > 0",
    :ballooning_scan => "needs the ballooning scan: add a [LocalStability] section",
    :gse_files => "needs gse.h5 next to the file: set diagnose_src = true in [Equilibrium]",
    :singular_coupling => "needs the singular coupling: run PerturbedEquilibrium with compute_singular_coupling = true",
    :response_xi => "needs the reconstructed displacement ξ: run PerturbedEquilibrium on a Forward free-boundary result",
    :response_b => "needs the reconstructed perturbed field b: run PerturbedEquilibrium on a Forward free-boundary result",
    :coil_sets => "needs coil forcing: give ForcingTerms coil sets (forcing_data_format = \"coil\")",
    :forcing_modes => "needs forcing modes: give ForcingTerms a forcing",
    :ef_sensitivities => "needs the coil sensitivities: add an [ErrorFields] section",
    :dominant_mode => "needs the dominant resonant-coupling mode: run PerturbedEquilibrium with compute_singular_coupling = true",
    :ef_monte_carlo => "needs the tolerance Monte Carlo: give ErrorFields a tolerance file",
    :ef_risk => "needs the locking risk: give ErrorFields a tolerance file and a scenario",
    :ef_scan => "needs the tolerance scan: set RiskControl scan_scales"
)

"""
    capabilities(src) -> Set{Symbol}

The data products `src` holds, by the names plots declare in their `needs`: a result in memory, or a
`gpec.h5` path (a dataset counts when present and non-empty). See [`REASONS`](@ref) for what each name means.
"""
function capabilities end

# A type no plot takes holds nothing a plot reads.
capabilities(_) = Set{Symbol}()

capabilities(::PlasmaEquilibrium) = Set([:equilibrium])

function capabilities(r::ForceFreeStatesResult)
    caps = Set([:equilibrium])
    isempty(r.surfaces) || push!(caps, :singular_surfaces)
    r.solution === nothing || push!(caps, :solution_profiles)
    diag = r.diagnostics
    diag === nothing || isempty(diag.crit_store) || push!(caps, :fixed_boundary_criterion)
    diag === nothing || isempty(diag.edge_scan.psi) || push!(caps, :edge_scan)
    r.free_boundary === nothing || isempty(r.free_boundary.et) || push!(caps, :free_boundary)
    r.delta_prime === nothing || isempty(r.delta_prime.matrix) || push!(caps, :delta_prime)
    isempty(r.kinetic.scan_psi) || push!(caps, :kinetic_scan)
    return caps
end

capabilities(r::LocalStabilityResult) = isempty(r.alpha) ? Set{Symbol}() : Set([:ballooning_scan])

function capabilities(s::PerturbedEquilibriumState)
    caps = Set{Symbol}()
    isempty(s.resonant_area_weighted_field) || push!(caps, :singular_coupling)
    s.xi_modes === nothing || push!(caps, :response_xi)
    s.b_modes === nothing || push!(caps, :response_b)
    isempty(s.coil_sets) || push!(caps, :coil_sets)
    isempty(s.forcing_modes) || push!(caps, :forcing_modes)
    return caps
end

function capabilities(r::EF.ErrorFieldsResult)
    caps = Set([:ef_sensitivities])
    isempty(r.response.dominant_singular_values) || push!(caps, :dominant_mode)
    r.monte_carlo === nothing || push!(caps, :ef_monte_carlo)
    r.risk === nothing || push!(caps, :ef_risk)
    r.scan === nothing || push!(caps, :ef_scan)
    return caps
end

const _H5_CAPABILITIES = (
    :equilibrium => "Equilibrium/Profiles/psi",
    :singular_surfaces => "SingularSurfaces/rational_psi",
    :solution_profiles => ("ForceFreeStates/Solutions/ForwardIntegration/xi_psi", "ForceFreeStates/Solutions/GalerkinIntegration/xi_psi"),
    :fixed_boundary_criterion => "ForceFreeStates/Solutions/ForwardIntegration/crit",
    :edge_scan => "ForceFreeStates/EdgeScan/psi",
    :free_boundary => "ForceFreeStates/FreeBoundaryStability/eigenmode_energies",
    :delta_prime => "SingularSurfaces/Delta_prime_matrix",
    :kinetic_scan => "SingularSurfaces/Kinetic/scan_psi",
    :ballooning_scan => "LocalStability/alpha",
    :singular_coupling => "PerturbedEquilibrium/SingularCoupling/resonant_area_weighted_field",
    :response_xi => "PerturbedEquilibrium/Response/xi_psi",
    :response_b => "PerturbedEquilibrium/Response/b_psi_area_weighted",
    :ef_sensitivities => EF._H5_GROUP,
    :dominant_mode => "PerturbedEquilibrium/SingularCoupling/DominantMode",
    :ef_monte_carlo => EF._MC_GROUP,
    :ef_risk => EF._RISK_GROUP,
    :ef_scan => "$(EF._RISK_GROUP)/ToleranceScan"
)

function capabilities(h5path::AbstractString)
    isfile(h5path) || throw(ArgumentError("no such file: $h5path"))
    caps = Set{Symbol}()
    h5open(h5path, "r") do f
        # A group counts when present; a dataset when it also holds data.
        has(k) = haskey(f, k) && (f[k] isa HDF5.Group || !isempty(f[k]))
        for (cap, keys) in _H5_CAPABILITIES
            any(has, keys isa Tuple ? keys : (keys,)) && push!(caps, cap)
        end
    end
    isfile(joinpath(dirname(h5path), "gse.h5")) && push!(caps, :gse_files)
    return caps
end

# The registered entry of `fn`.
function _entry(fn)
    i = findfirst(e -> e.fn === fn, PLOTS)
    i === nothing && error("$(nameof(fn)) is not a registered plot")
    return PLOTS[i]
end

_accepts(e::PlotEntry, src) = any(T -> src isa T, e.sources)
_missing(e::PlotEntry, caps) = [k for k in e.needs if k ∉ caps]
_why(missing) = join((REASONS[k] for k in missing), "; ")

"""
    _require(fn, src)

Throw an `ArgumentError` naming what the run must change when `src` lacks a capability that plot `fn` needs.
"""
function _require(fn, src)
    e = _entry(fn)
    missing = _missing(e, capabilities(src))
    isempty(missing) || throw(ArgumentError("$(nameof(fn)): $(_why(missing))"))
    return nothing
end

# Whether `src` has every capability plot `fn` needs; composites use it to skip the panels they cannot draw.
_can(fn, caps) = isempty(_missing(_entry(fn), caps))

"""
    PlotCatalog

The plots registered for a source, from [`available_plots`](@ref): one section per result, each listing the plots
that accept it, paired with the capabilities it lacks for them (none when it can draw them).
"""
struct PlotCatalog
    sections::Vector{Tuple{String,Vector{Tuple{PlotEntry,Vector{Symbol}}}}}
end

_source_label(h5path::AbstractString) = h5path
_source_label(src) = string(nameof(typeof(src)))
_source_label(r::ForceFreeStatesResult) = "ForceFreeStatesResult ($(r.integrator))"

function _section(src)
    caps = capabilities(src)
    rows = [(e, _missing(e, caps)) for e in PLOTS if _accepts(e, src)]
    return (_source_label(src), rows)
end

"""
    available_plots(src) -> PlotCatalog

The registered plots for `src` with whether each can draw from it, and the reason when it cannot. `src` is a
result in memory, a `gpec.h5` path, or the named tuple `main` returns (one section per stage result). Every
listed function takes `src` as its first argument.
"""
available_plots(src) = PlotCatalog([_section(src)])
available_plots(results::NamedTuple) =
    PlotCatalog([_section(x) for x in unique(values(results)) if x !== nothing && any(e -> _accepts(e, x), PLOTS)])

"""
    plot_counts(src) -> (available, total)

How many registered plots accept `src`, and how many of those it can draw.
"""
function plot_counts(src)
    _, rows = _section(src)
    return count(r -> isempty(r[2]), rows), length(rows)
end

Base.length(c::PlotCatalog) = sum(length(rows) for (_, rows) in c.sections; init=0)

_qualified(e::PlotEntry) = string(nameof(parentmodule(e.fn)), ".", nameof(e.fn))

function Base.show(io::IO, ::MIME"text/plain", c::PlotCatalog)
    isempty(c.sections) && return print(io, "PlotCatalog: no registered plots accept this source")
    for (i, (label, rows)) in enumerate(c.sections)
        i > 1 && print(io, "\n")
        ok = count(r -> isempty(r[2]), rows)
        print(io, "Plots for ", label, ": ", ok, " of ", length(rows), " available")
        width = maximum((length(_qualified(e)) for (e, _) in rows); init=0)
        for (e, missing) in rows
            mark = isempty(missing) ? "✓" : "✗"
            note = isempty(missing) ? e.description : _why(missing)
            print(io, "\n  ", mark, " ", rpad(_qualified(e), width), "  ", note)
        end
    end
    return nothing
end

Base.show(io::IO, c::PlotCatalog) = print(io, "PlotCatalog(", length(c), " plots)")
