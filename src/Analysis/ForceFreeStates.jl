"""
    ForceFreeStates

Post-processing and visualization functions for ForceFreeStates (DCON-style ideal MHD stability) results. Each plot
takes the `ForceFreeStatesResult` in memory or the path of a `gpec.h5` it was saved to, and throws an `ArgumentError`
naming what to change in the run when the source lacks the data it draws (see `Analysis.available_plots`).
"""
module ForceFreeStates

using HDF5

using ..Analysis: register!, _require, _can, _missing, _entry, _why, capabilities, _draw
import ...ForceFreeStates: ForceFreeStatesResult
import ...LocalStability: LocalStabilityResult

const Source = Union{AbstractString,ForceFreeStatesResult}

const _FWD = "ForceFreeStates/Solutions/ForwardIntegration"
const _GAL = "ForceFreeStates/Solutions/GalerkinIntegration"
const _FBS = "ForceFreeStates/FreeBoundaryStability"

# Least-stable total energy, or `nothing` without free-boundary energies.
_first_energy(et) = isempty(et) ? nothing : et[1]

# The rational surfaces as (ψ_N, "m/n" labels).
_rationals(r::ForceFreeStatesResult) = (; psi=[s.psifac for s in r.surfaces], labels=["$(s.m[1])/$(s.n[1])" for s in r.surfaces])
function _rationals(h5path::AbstractString)
    h5open(h5path, "r") do fid
        haskey(fid, "SingularSurfaces/rational_psi") || return (; psi=Float64[], labels=String[])
        m, n = vec(read(fid["SingularSurfaces/rational_m"])), vec(read(fid["SingularSurfaces/rational_n"]))
        (; psi=vec(read(fid["SingularSurfaces/rational_psi"])), labels=["$mi/$ni" for (mi, ni) in zip(m, n)])
    end
end

function _mode_displacement_data(r::ForceFreeStatesResult)
    sol = r.solution
    return (; mlow=r.mlow, psi=sol.psi_store, xi=sol.u_store[:, 1, 1, :], dW=r.free_boundary === nothing ? nothing : _first_energy(r.free_boundary.et))
end

function _mode_displacement_data(h5path::AbstractString)
    h5open(h5path, "r") do fid
        # Forward integration stores its profiles in ForwardIntegration, a matched Galerkin solve in GalerkinIntegration.
        group = isempty(fid["$_FWD/xi_psi"]) ? _GAL : _FWD
        (; mlow=read(fid["Info/mlow"]), psi=read(fid["$group/psi"]), xi=read(fid["$group/xi_psi"])[:, 1, :],
            dW=_first_energy(read(fid["$_FBS/eigenmode_energies"])))
    end
end

_criterion_data(r::ForceFreeStatesResult) = (; psi=r.diagnostics.psi_store, crit=r.diagnostics.crit_store)
_criterion_data(h5path::AbstractString) = h5open(fid -> (; psi=read(fid["$_FWD/psi"]), crit=read(fid["$_FWD/crit"])), h5path, "r")

_energy_eigenvector_data(r::ForceFreeStatesResult) = (; wt=r.free_boundary.wt, mlow=r.mlow)
_energy_eigenvector_data(h5path::AbstractString) = h5open(fid -> (; wt=read(fid["$_FBS/W_freeboundary_eigenmodes"]), mlow=read(fid["Info/mlow"])), h5path, "r")

_energy_data(r::ForceFreeStatesResult) = (; et=r.free_boundary.et, ep=r.free_boundary.ep, ev=r.free_boundary.ev)
function _energy_data(h5path::AbstractString)
    h5open(h5path, "r") do fid
        (; et=read(fid["$_FBS/eigenmode_energies"]), ep=read(fid["$_FBS/eigenmode_plasma_energies"]), ev=read(fid["$_FBS/eigenmode_vacuum_energies"]))
    end
end

function _edge_scan_data(r::ForceFreeStatesResult)
    es = r.diagnostics.edge_scan
    return (; q=es.q, et=es.total_eigenvalue, ep=es.plasma_energy, ev=es.vacuum_energy, qlim=r.qlim)
end

function _edge_scan_data(h5path::AbstractString)
    h5open(h5path, "r") do fid
        (; q=read(fid["ForceFreeStates/EdgeScan/q"]), et=read(fid["ForceFreeStates/EdgeScan/total_energy"]),
            ep=read(fid["ForceFreeStates/EdgeScan/plasma_energy"]), ev=read(fid["ForceFreeStates/EdgeScan/vacuum_energy"]), qlim=read(fid["Info/qlim"]))
    end
end

# The Δ′ family with "m/n" labels for the surfaces it spans: Galerkin solves across its in-band subset of the rational surfaces.
# The free-boundary Δ′ and raw D′, or the fixed-boundary ones when the run has no vacuum. `ef` is Δ_EF as
# (edge harmonic × surface side), empty when the run computed none.
function _delta_prime_data(r::ForceFreeStatesResult)
    dp = r.delta_prime
    blocks, raw = dp.freeboundary === nothing ? (dp.fixedboundary, dp.raw_fixedboundary) : (dp.freeboundary, dp.raw_freeboundary)
    k = size(blocks.Delta, 1)
    g = r.galerkin
    m, n = g !== nothing && length(g.sing_m) == k ? (g.sing_m, g.sing_n) : ([s.m[1] for s in r.surfaces[1:k]], [s.n[1] for s in r.surfaces[1:k]])
    return (; matrix=blocks.Delta, raw, ef=permutedims(dp.ef), labels=["$mi/$ni" for (mi, ni) in zip(m, n)], mlow=r.mlow)
end

function _delta_prime_data(h5path::AbstractString)
    h5open(h5path, "r") do fid
        suffix = haskey(fid, "SingularSurfaces/Delta_prime_matrix") ? "" : "_fixedboundary"
        matrix = read(fid["SingularSurfaces/Delta_prime_matrix$suffix"])
        k = size(matrix, 1)
        group = haskey(fid, "$_GAL/rational_m") && length(fid["$_GAL/rational_m"]) == k ? _GAL : "SingularSurfaces"
        m, n = vec(read(fid["$group/rational_m"]))[1:k], vec(read(fid["$group/rational_n"]))[1:k]
        ef = haskey(fid, "SingularSurfaces/Delta_EF") ? read(fid["SingularSurfaces/Delta_EF"]) : zeros(ComplexF64, 0, 2k)
        raw = read(fid["SingularSurfaces/Delta_prime_raw_$(isempty(suffix) ? "freeboundary" : "fixedboundary")"])
        (; matrix, raw, ef, labels=["$mi/$ni" for (mi, ni) in zip(m, n)], mlow=read(fid["Info/mlow"]))
    end
end

_ballooning_data(r::LocalStabilityResult) = (; psi=r.ballooning_psi, alpha=r.alpha, alpha_crit=r.alpha_critical)
_ballooning_data(h5path::AbstractString) =
    h5open(fid -> (; psi=read(fid["LocalStability/ballooning_psi"]), alpha=read(fid["LocalStability/alpha"]), alpha_crit=read(fid["LocalStability/alpha_critical"])), h5path, "r")

function _cond_fbar_data(r::ForceFreeStatesResult)
    k = r.kinetic
    return (; scan_psi=k.scan_psi, scan_cond=k.scan_cond, thr=k.scan_threshold, k_psi=[s.psifac for s in k.kinsing], kmsing=k.kmsing)
end

function _cond_fbar_data(h5path::AbstractString)
    h5open(h5path, "r") do fid
        kg = fid["SingularSurfaces/Kinetic"]
        (; scan_psi=read(kg["scan_psi"]), scan_cond=read(kg["scan_cond"]), thr=read(kg["scan_threshold"]), k_psi=read(kg["rational_psi"]),
            kmsing=read(kg["rational_count"]))
    end
end

# Drawing functions, defined by the GPECMakieExt extension.
function _plot_mode_displacement end
function _plot_critical_determinant end
function _plot_energy_eigenvectors end
function _plot_eigenvalues end
function _plot_edge_stability_scan end
function _plot_delta_prime end
function _plot_delta_prime_raw end
function _plot_delta_ef end
function _plot_cond_fbar end
function _plot_ballooning_alpha_boundary end
function _plot_ballooning_alpha_boundaries end
function _plot_ffs_summary end

"""
    plot_mode_displacement(src; modes=nothing, save_path=nothing)

Plot |ξ^ψ_m| against ψ_N for the least stable free-boundary eigenmode, one curve per harmonic, with the rational
surfaces labelled along the top.

### Arguments

  - `src`: A `ForceFreeStatesResult` with stored ξ profiles, or the path of a `gpec.h5` it was saved to

### Keyword arguments

  - `modes`: The m values to draw (default: `nothing`, the six largest)
  - `save_path`: If given, save the figure to this path (default: `nothing`)

### Returns

A Makie `Figure`.
"""
function plot_mode_displacement(src::Source; modes=nothing, save_path=nothing)
    _require(plot_mode_displacement, src)
    return _draw(_plot_mode_displacement, _mode_displacement_data(src), _rationals(src); modes, save_path)
end

"""
    plot_critical_determinant(src; save_path=nothing)

Plot the critical determinant against ψ_N on a signed pseudo-log axis, titled with its zero crossings (none means
fixed-boundary stable).

### Arguments

  - `src`: A Forward or Riccati `ForceFreeStatesResult`, or the path of a `gpec.h5` it was saved to

### Keyword arguments

  - `save_path`: If given, save the figure to this path (default: `nothing`)

### Returns

A Makie `Figure`.
"""
function plot_critical_determinant(src::Source; save_path=nothing)
    _require(plot_critical_determinant, src)
    return _draw(_plot_critical_determinant, _criterion_data(src), _rationals(src); save_path)
end

"""
    plot_energy_eigenvectors(src; nmodes=12, save_path=nothing)

Plot |v_m| of the free-boundary energy eigenvectors (unit power norm) against poloidal harmonic m, least stable at
the top.

### Arguments

  - `src`: A `ForceFreeStatesResult` with free-boundary energies (`vac_flag = true`), or the path of a `gpec.h5` it was saved to

### Keyword arguments

  - `nmodes`: Number of eigenmodes drawn (default: 12)
  - `save_path`: If given, save the figure to this path (default: `nothing`)

### Returns

A Makie `Figure`.
"""
function plot_energy_eigenvectors(src::Source; nmodes=12, save_path=nothing)
    _require(plot_energy_eigenvectors, src)
    return _draw(_plot_energy_eigenvectors, _energy_eigenvector_data(src); nmodes, save_path)
end

const _ENERGIES = (:total, :plasma, :vacuum)

"""
    plot_eigenvalues(src; matrix_type=:all, save_path=nothing)

Plot the free-boundary energy eigenvalues (the power-normalized pencil W·v = λN·v), most unstable first: the total
δW and its plasma and vacuum parts, on a signed pseudo-log axis with the unstable sign shaded.

### Arguments

  - `src`: A `ForceFreeStatesResult` with free-boundary energies (`vac_flag = true`), or the path of a `gpec.h5` it was saved to

### Keyword arguments

  - `matrix_type`: `:all` (default), or one of `:total`, `:plasma`, `:vacuum`
  - `save_path`: If given, save the figure to this path (default: `nothing`)

### Returns

A Makie `Figure`.
"""
function plot_eigenvalues(src::Source; matrix_type=:all, save_path=nothing)
    matrix_type === :all || matrix_type in _ENERGIES || throw(ArgumentError("matrix_type must be :all, :total, :plasma or :vacuum"))
    _require(plot_eigenvalues, src)
    return _draw(_plot_eigenvalues, _energy_data(src); which=matrix_type === :all ? _ENERGIES : (matrix_type,), save_path)
end

"""
    plot_edge_stability_scan(src; save_path=nothing)

Plot the total, plasma and vacuum free-boundary energies against the edge truncation q, with the chosen truncation
marked.

### Arguments

  - `src`: A `ForceFreeStatesResult` produced with `dW_edge_scan_start < psilim`, or the path of a `gpec.h5` it was saved to

### Keyword arguments

  - `save_path`: If given, save the figure to this path (default: `nothing`)

### Returns

A Makie `Figure`.
"""
function plot_edge_stability_scan(src::Source; save_path=nothing)
    _require(plot_edge_stability_scan, src)
    return _draw(_plot_edge_stability_scan, _edge_scan_data(src); save_path)
end

const _PARTS = (:real, :imag, :abs)
_check_part(part) = part in _PARTS || throw(ArgumentError("part must be :real, :imag or :abs"))

"""
    plot_delta_prime(src; part=:real, save_path=nothing)

Plot the inter-surface Δ′ matrix (PEST-3 convention, ψ_N-referenced) as a heatmap with each entry written in,
rows and columns labelled by rational surface.

### Arguments

  - `src`: A Riccati or Galerkin `ForceFreeStatesResult`, or the path of a `gpec.h5` it was saved to

### Keyword arguments

  - `part`: `:real` (default, diverging symmetric-log colours), `:imag`, or `:abs` (log colours)
  - `save_path`: If given, save the figure to this path (default: `nothing`)

### Returns

A Makie `Figure`.
"""
function plot_delta_prime(src::Source; part=:real, save_path=nothing)
    _check_part(part)
    _require(plot_delta_prime, src)
    return _draw(_plot_delta_prime, _delta_prime_data(src); part, save_path)
end

"""
    plot_delta_prime_raw(src; part=:real, save_path=nothing)

Plot the raw outer-region matching matrix D′ as a heatmap, with the left and right sides of each rational surface
grouped under it on both axes.

### Arguments

  - `src`: A Riccati or Galerkin `ForceFreeStatesResult`, or the path of a `gpec.h5` it was saved to

### Keyword arguments

  - `part`: `:real` (default, diverging symmetric-log colours), `:imag`, or `:abs` (log colours)
  - `save_path`: If given, save the figure to this path (default: `nothing`)

### Returns

A Makie `Figure`.
"""
function plot_delta_prime_raw(src::Source; part=:real, save_path=nothing)
    _check_part(part)
    _require(plot_delta_prime_raw, src)
    return _draw(_plot_delta_prime_raw, _delta_prime_data(src); part, save_path)
end

"""
    plot_delta_ef(src; part=:real, save_path=nothing)

Plot Δ_EF, the small-solution coefficient at each side of each rational surface for a unit edge displacement in each
edge poloidal harmonic, as a heatmap.

### Arguments

  - `src`: A Riccati or Galerkin `ForceFreeStatesResult` solved with `error_field_response = true`, or the path of a `gpec.h5` it was saved to

### Keyword arguments

  - `part`: `:real` (default, diverging symmetric-log colours), `:imag`, or `:abs` (log colours)
  - `save_path`: If given, save the figure to this path (default: `nothing`)

### Returns

A Makie `Figure`.
"""
function plot_delta_ef(src::Source; part=:real, save_path=nothing)
    _check_part(part)
    _require(plot_delta_ef, src)
    return _draw(_plot_delta_ef, _delta_prime_data(src); part, save_path)
end

"""
    plot_cond_fbar(src; save_path=nothing)

Plot cond(F̄) against ψ_N from the kinetic singular-surface scan, with the threshold, the rational surfaces and any
accepted kinetic singular surfaces marked.

### Arguments

  - `src`: A kinetic `ForceFreeStatesResult`, or the path of a `gpec.h5` it was saved to

### Keyword arguments

  - `save_path`: If given, save the figure to this path (default: `nothing`)

### Returns

A Makie `Figure`.
"""
function plot_cond_fbar(src::Source; save_path=nothing)
    _require(plot_cond_fbar, src)
    return _draw(_plot_cond_fbar, _cond_fbar_data(src), _rationals(src); save_path)
end

"""
    plot_ballooning_alpha_boundary(src; psi_min=0.0, save_path=nothing)

Plot the infinite-n ballooning diagram: the equilibrium pressure gradient α and its first stability boundary against
ψ_N, with the unstable surfaces shaded.

### Arguments

  - `src`: A `LocalStabilityResult`, or the path of a `gpec.h5` written with a `[LocalStability]` section

### Keyword arguments

  - `psi_min`: Lower ψ_N limit, e.g. `0.9` for the pedestal (default: `0.0`)
  - `save_path`: If given, save the figure to this path (default: `nothing`)

### Returns

A Makie `Figure`.
"""
function plot_ballooning_alpha_boundary(src::Union{AbstractString,LocalStabilityResult}; psi_min=0.0, save_path=nothing)
    _require(plot_ballooning_alpha_boundary, src)
    return _draw(_plot_ballooning_alpha_boundary, _ballooning_data(src); psi_min, save_path)
end

"""
    plot_ballooning_alpha_boundaries(bnd; psi_min=0.0, save_path=nothing)
    plot_ballooning_alpha_boundaries(bnd, dpmap; psi_min=0.0, save_path=nothing)

Plot the infinite-n ballooning diagram with the first and second stability boundaries; with `dpmap`, over a map of
Δ′ signed so that positive is stable on every surface, with its zero contour.

### Arguments

  - `bnd`: The named tuple `LocalStability.ballooning_alpha_boundaries` returns
  - `dpmap`: The named tuple `LocalStability.ballooning_delta_prime_map` returns

### Keyword arguments

  - `psi_min`: Lower ψ_N limit (default: `0.0`)
  - `save_path`: If given, save the figure to this path (default: `nothing`)

### Returns

A Makie `Figure`.
"""
plot_ballooning_alpha_boundaries(bnd; psi_min=0.0, save_path=nothing) = _draw(_plot_ballooning_alpha_boundaries, bnd, nothing; psi_min, save_path)
plot_ballooning_alpha_boundaries(bnd, dpmap; psi_min=0.0, save_path=nothing) = _draw(_plot_ballooning_alpha_boundaries, bnd, dpmap; psi_min, save_path)

# The summary's panels in drawing order, each with the plot whose capabilities it needs and the data it draws.
_ffs_panels(src) = (
    plot_critical_determinant => () -> (:critical_determinant, _criterion_data(src), _rationals(src)),
    plot_eigenvalues => () -> (:energies, _energy_data(src)),
    plot_mode_displacement => () -> (:mode_displacement, _mode_displacement_data(src), _rationals(src)),
    plot_delta_prime => () -> (:delta_prime, _delta_prime_data(src)),
    plot_energy_eigenvectors => () -> (:eigenvectors, _energy_eigenvector_data(src))
)

"""
    plot_ffs_summary(src; save_path=nothing)

Plot the stability panels the source supplies, two per row: the critical determinant, the free-boundary energies,
the least stable mode's displacement, Δ′_ii at each rational surface, and the energy eigenvectors.

### Arguments

  - `src`: A `ForceFreeStatesResult`, or the path of a `gpec.h5` it was saved to

### Keyword arguments

  - `save_path`: If given, save the figure to this path (default: `nothing`)

### Returns

A Makie `Figure`. Throws an `ArgumentError` naming every missing capability when no panel can be drawn.
"""
function plot_ffs_summary(src::Source; save_path=nothing)
    _require(plot_ffs_summary, src)
    caps = capabilities(src)
    panels = [data() for (panel_fn, data) in _ffs_panels(src) if _can(panel_fn, caps)]
    return _draw(_plot_ffs_summary, panels; save_path)
end

register!(plot_ffs_summary, (ForceFreeStatesResult, AbstractString), (), "the stability panels the run supplies";
    any_of=((:fixed_boundary_criterion,), (:free_boundary,), (:solution_profiles,), (:delta_prime,)))
register!(plot_critical_determinant, (ForceFreeStatesResult, AbstractString), (:fixed_boundary_criterion,), "critical determinant against ψ_N")
register!(plot_eigenvalues, (ForceFreeStatesResult, AbstractString), (:free_boundary,), "free-boundary energy eigenvalues")
register!(plot_mode_displacement, (ForceFreeStatesResult, AbstractString), (:solution_profiles,), "|ξ^ψ_m| of the least stable mode")
register!(plot_energy_eigenvectors, (ForceFreeStatesResult, AbstractString), (:free_boundary,), "free-boundary energy eigenvectors per m")
register!(plot_edge_stability_scan, (ForceFreeStatesResult, AbstractString), (:edge_scan,), "edge energies against the truncation q")
register!(plot_delta_prime, (ForceFreeStatesResult, AbstractString), (:delta_prime,), "Δ′ matrix between rational surfaces")
register!(plot_delta_prime_raw, (ForceFreeStatesResult, AbstractString), (:delta_prime,), "raw outer-region matching matrix D′")
register!(plot_delta_ef, (ForceFreeStatesResult, AbstractString), (:delta_ef,), "Δ_EF: surface response to each edge harmonic")
register!(plot_cond_fbar, (ForceFreeStatesResult, AbstractString), (:kinetic_scan,), "cond(F̄) scan for kinetic singular surfaces")
register!(plot_ballooning_alpha_boundary, (LocalStabilityResult, AbstractString), (:ballooning_scan,), "ballooning α against its first stability boundary")

end # module ForceFreeStates
