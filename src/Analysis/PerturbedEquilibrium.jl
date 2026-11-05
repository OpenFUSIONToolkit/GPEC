"""
    PerturbedEquilibrium

Post-processing and visualization functions for GPEC perturbed equilibrium results. Each plot takes the
`PerturbedEquilibriumState` in memory or the path of a `gpec.h5` it was saved to, and throws an `ArgumentError`
naming what to change in the run when the source lacks the data it draws (see `Analysis.available_plots`).
"""
module PerturbedEquilibrium

using HDF5

using ..Analysis: register!, _require, _can, capabilities, REASONS, _draw
import ...PerturbedEquilibrium: PerturbedEquilibriumState

const Source = Union{AbstractString,PerturbedEquilibriumState}

const _COUPLING = "PerturbedEquilibrium/SingularCoupling/"
const _RESPONSE = "PerturbedEquilibrium/Response/"

function _coupling_data(s::PerturbedEquilibriumState)
    r = s.response
    return (; field=s.resonant_area_weighted_field, current=s.resonant_current, island_hw=s.island_half_width, chirikov=s.chirikov_parameter,
        delta_prime=s.delta_prime, psi=r.rational_psi, q=r.rational_q, m=r.rational_m_res, n=r.rational_n)
end

function _coupling_data(h5path::AbstractString)
    h5open(h5path, "r") do fid
        rd(name) = read(fid[_COUPLING*name])
        (; field=rd("resonant_area_weighted_field"), current=rd("resonant_current"), island_hw=rd("island_half_width"), chirikov=rd("chirikov_parameter"),
            delta_prime=rd("Delta_prime"), psi=rd("rational_psi"), q=rd("rational_q"), m=rd("rational_m"), n=rd("rational_n"))
    end
end

const _COMPONENTS = Dict(
    :xi_psi => (dataset="xi_psi", field=:psi, modes=:xi_modes, needs=:response_xi),
    :b_psi => (dataset="b_psi_area_weighted", field=:b_psi_area_weighted, modes=:b_modes, needs=:response_b),  # area-normalized b^ψ
    :Jb_theta => (dataset="Jb_theta", field=:theta, modes=:b_modes, needs=:response_b),
    :Jb_zeta => (dataset="Jb_zeta", field=:zeta, modes=:b_modes, needs=:response_b)
)

# A response component with q on the equilibrium's profile nodes and the rational surfaces the coupling resolved.
function _response_data(s::PerturbedEquilibriumState, component)
    c = _COMPONENTS[component]
    ffs = s.response.ffs
    prof = ffs.equil.profiles
    return (; data=getfield(getfield(s, c.modes), c.field), psi=s.psi_grid, mlow=ffs.mlow, mhigh=ffs.mhigh, nhigh=ffs.nhigh,
        q_psi=prof.xs, q=prof.q_spline.y, rational_psi=s.response.rational_psi, rational_m=s.response.rational_m_res)
end

function _response_data(h5path::AbstractString, component)
    h5open(h5path, "r") do fid
        coupled = haskey(fid, _COUPLING * "rational_psi")
        (; data=read(fid[_RESPONSE*_COMPONENTS[component].dataset]), psi=read(fid[_RESPONSE*"psi"]), mlow=read(fid["Info/mlow"]),
            mhigh=read(fid["Info/mhigh"]), nhigh=read(fid["Info/nhigh"]), q_psi=read(fid["Equilibrium/Profiles/psi"]),
            q=read(fid["Equilibrium/Profiles/q"]), rational_psi=coupled ? read(fid[_COUPLING*"rational_psi"]) : Float64[],
            rational_m=coupled ? read(fid[_COUPLING*"rational_m"]) : Int[])
    end
end

# Check `component` names a response field `src` holds; `_require` covers ξ, which every component's run also has.
function _require_component(fn, src, component)
    haskey(_COMPONENTS, component) || throw(ArgumentError("component must be one of :xi_psi, :b_psi, :Jb_theta, :Jb_zeta"))
    _require(fn, src)
    need = _COMPONENTS[component].needs
    need in capabilities(src) || throw(ArgumentError("$(nameof(fn)): $(REASONS[need])"))
    return nothing
end

# Drawing functions, defined by the GPECMakieExt extension.
function _plot_resonant_coupling end
function _plot_mode_spectrogram end
function _plot_perturbed_equilibrium_summary end

const _COUPLING_ROWS = (:field, :current, :island_hw, :chirikov, :delta_prime)

"""
    plot_resonant_area_weighted_field(src; save_path=nothing)

Plot the resonant coupling at each rational surface in rows sharing a q = m/n axis: the resonant area-weighted field
|b̄_res|, the resonant current |I_res|, the island half-width w/2, the effective island-overlap Chirikov parameter
and the forcing-driven Re Δ′. Toroidal mode numbers are distinguished by colour; the title counts the surfaces with
overlapping islands.

### Arguments

  - `src`: A `PerturbedEquilibriumState`, or the path of a `gpec.h5` it was saved to

### Keyword arguments

  - `save_path`: If given, save the figure to this path (default: `nothing`)

### Returns

A Makie `Figure`.
"""
function plot_resonant_area_weighted_field(src::Source; save_path=nothing)
    _require(plot_resonant_area_weighted_field, src)
    return _draw(_plot_resonant_coupling, _coupling_data(src), _COUPLING_ROWS; save_path)
end

"""
    plot_resonant_area_weighted_field_amplitude(src; save_path=nothing)

Plot the resonant area-weighted field |b̄_res| = |Φ_res|/A at each rational surface [Pharr 2026], in tesla.

### Arguments

  - `src`: A `PerturbedEquilibriumState`, or the path of a `gpec.h5` it was saved to

### Keyword arguments

  - `save_path`: If given, save the figure to this path (default: `nothing`)

### Returns

A Makie `Figure`.
"""
function plot_resonant_area_weighted_field_amplitude(src::Source; save_path=nothing)
    _require(plot_resonant_area_weighted_field_amplitude, src)
    return _draw(_plot_resonant_coupling, _coupling_data(src), (:field,); save_path)
end

"""
    plot_island_widths(src; save_path=nothing)

Plot the island half-width w/2 at each rational surface, in ψ_N.

### Arguments

  - `src`: A `PerturbedEquilibriumState`, or the path of a `gpec.h5` it was saved to

### Keyword arguments

  - `save_path`: If given, save the figure to this path (default: `nothing`)

### Returns

A Makie `Figure`.
"""
function plot_island_widths(src::Source; save_path=nothing)
    _require(plot_island_widths, src)
    return _draw(_plot_resonant_coupling, _coupling_data(src), (:island_hw,); save_path)
end

"""
    plot_chirikov_parameter(src; save_path=nothing)

Plot the effective island-overlap Chirikov parameter at each rational surface, with the overlap threshold 1 marked.

### Arguments

  - `src`: A `PerturbedEquilibriumState`, or the path of a `gpec.h5` it was saved to

### Keyword arguments

  - `save_path`: If given, save the figure to this path (default: `nothing`)

### Returns

A Makie `Figure`.
"""
function plot_chirikov_parameter(src::Source; save_path=nothing)
    _require(plot_chirikov_parameter, src)
    return _draw(_plot_resonant_coupling, _coupling_data(src), (:chirikov,); save_path)
end

"""
    plot_driven_delta_prime(src; save_path=nothing)

Plot the forcing-driven Re Δ′ at each rational surface, the response to the applied forcing; for the
equilibrium's own Δ′ of a Riccati or Galerkin solve, see `ForceFreeStates.plot_delta_prime`.

### Arguments

  - `src`: A `PerturbedEquilibriumState`, or the path of a `gpec.h5` it was saved to

### Keyword arguments

  - `save_path`: If given, save the figure to this path (default: `nothing`)

### Returns

A Makie `Figure`.
"""
function plot_driven_delta_prime(src::Source; save_path=nothing)
    _require(plot_driven_delta_prime, src)
    return _draw(_plot_resonant_coupling, _coupling_data(src), (:delta_prime,); save_path)
end

const _STYLES = (:log_heatmap, :linear_heatmap, :lines)

"""
    plot_mode_spectrogram(src; component=:xi_psi, style=:log_heatmap, save_path=nothing)

Plot a response component's poloidal harmonics against ψ_N, with the rational surfaces and the resonance m = n q(ψ)
marked.

### Arguments

  - `src`: A `PerturbedEquilibriumState` with the reconstructed response, or the path of a `gpec.h5` it was saved to

### Keyword arguments

  - `component`: `:xi_psi` (ξ^ψ, default), `:b_psi` (the area-weighted b̄^ψ, in T), `:Jb_theta` or `:Jb_zeta` (J·b^θ and
    J·b^ζ, in T m²)
  - `style`: `:log_heatmap` (default), `:linear_heatmap`, or `:lines` (the eight largest harmonics)
  - `save_path`: If given, save the figure to this path (default: `nothing`)

### Returns

A Makie `Figure`.
"""
function plot_mode_spectrogram(src::Source; component=:xi_psi, style=:log_heatmap, save_path=nothing)
    style in _STYLES || throw(ArgumentError("style must be :log_heatmap, :linear_heatmap or :lines"))
    _require_component(plot_mode_spectrogram, src, component)
    return _draw(_plot_mode_spectrogram, _response_data(src, component), component; style, save_path)
end

"""
    plot_perturbed_equilibrium_summary(src; save_path=nothing)

Plot the panels the source supplies: the resonant field at each rational surface, the edge b̄^ψ spectrum, and the
ξ^ψ spectrogram.

### Arguments

  - `src`: A `PerturbedEquilibriumState`, or the path of a `gpec.h5` it was saved to

### Keyword arguments

  - `save_path`: If given, save the figure to this path (default: `nothing`)

### Returns

A Makie `Figure`. Throws an `ArgumentError` when no panel can be drawn.
"""
function plot_perturbed_equilibrium_summary(src::Source; save_path=nothing)
    caps = capabilities(src)
    coupling = _can(plot_resonant_area_weighted_field_amplitude, caps) ? _coupling_data(src) : nothing
    edge = :response_b in caps ? _response_data(src, :b_psi) : nothing
    xi = _can(plot_mode_spectrogram, caps) ? _response_data(src, :xi_psi) : nothing
    coupling === nothing && edge === nothing && xi === nothing &&
        throw(ArgumentError("plot_perturbed_equilibrium_summary: no panel can be drawn: $(REASONS[:singular_coupling]); $(REASONS[:response_xi])"))
    return _draw(_plot_perturbed_equilibrium_summary, coupling, edge, xi; save_path)
end

const _PE_SOURCES = (PerturbedEquilibriumState, AbstractString)
register!(plot_perturbed_equilibrium_summary, _PE_SOURCES, (), "resonant field, edge b̄^ψ spectrum and ξ^ψ spectrogram")
register!(plot_resonant_area_weighted_field, _PE_SOURCES, (:singular_coupling,), "resonant coupling rows per rational surface")
register!(plot_resonant_area_weighted_field_amplitude, _PE_SOURCES, (:singular_coupling,), "|b̄_res| per rational surface")
register!(plot_island_widths, _PE_SOURCES, (:singular_coupling,), "island half-width per rational surface")
register!(plot_chirikov_parameter, _PE_SOURCES, (:singular_coupling,), "effective island-overlap Chirikov parameter per rational surface")
register!(plot_driven_delta_prime, _PE_SOURCES, (:singular_coupling,), "forcing-driven Δ′ per rational surface")
register!(plot_mode_spectrogram, _PE_SOURCES, (:response_xi,), "response harmonics against ψ_N (component = :xi_psi, :b_psi, :Jb_theta, :Jb_zeta)")

end # module PerturbedEquilibrium
