"""
    PerturbedEquilibrium

Post-processing and visualization functions for GPEC perturbed equilibrium results. Each plot takes the
`PerturbedEquilibriumState` in memory or the path of a `gpec.h5` it was saved to, and throws an `ArgumentError`
naming what to change in the run when the source lacks the data it draws (see `Analysis.available_plots`).
"""
module PerturbedEquilibrium

using HDF5
using LaTeXStrings
using Plots

using ..Analysis: register!, _require, _can, capabilities, REASONS
import ...PerturbedEquilibrium: PerturbedEquilibriumState

const Source = Union{AbstractString,PerturbedEquilibriumState}

const _COUPLING = "PerturbedEquilibrium/SingularCoupling/"
const _RESPONSE = "PerturbedEquilibrium/Response/"

function _coupling_data(s::PerturbedEquilibriumState)
    r = s.response
    return (; field=s.resonant_area_weighted_field, current=s.resonant_current, island_hw=s.island_half_width, chirikov=s.chirikov_parameter,
        delta_prime=s.delta_prime, psi=r.rational_psi, q=r.rational_q, n=r.rational_n)
end

function _coupling_data(h5path::AbstractString)
    h5open(h5path, "r") do fid
        rd(name) = read(fid[_COUPLING*name])
        (; field=rd("resonant_area_weighted_field"), current=rd("resonant_current"), island_hw=rd("island_half_width"), chirikov=rd("chirikov_parameter"),
            delta_prime=rd("Delta_prime"), psi=rd("rational_psi"), q=rd("rational_q"), n=rd("rational_n"))
    end
end

"""
    plot_resonant_area_weighted_field_amplitude(src; save_path=nothing)

Scatter plot of the resonant area-weighted field `|b^r|` per rational surface vs ψ_N. One marker series
per toroidal mode n. Integer-valued q rational surfaces are annotated. The resonant area-weighted field
`b^r = Φ^r/A^r` is the resonant flux normalized by the scalar surface area, in tesla [Pharr 2026].

Requires the perturbed equilibrium singular coupling.

### Arguments

  - `src`: A `PerturbedEquilibriumState`, or the path of a `gpec.h5` it was saved to

### Keyword arguments

  - `save_path`: If provided, save the figure to this path (default: `nothing`)

### Returns

A `Plots.jl` plot object.
"""
function plot_resonant_area_weighted_field_amplitude(src::Source; save_path=nothing)
    _require(plot_resonant_area_weighted_field_amplitude, src)
    p = _plot_resonant_area_weighted_field_amplitude(_coupling_data(src))
    isnothing(save_path) || savefig(p, save_path)
    return p
end

function _plot_resonant_area_weighted_field_amplitude(d)
    resonant_area_weighted_field, rational_psi, rational_q, rational_n = d.field, d.psi, d.q, d.n

    p = plot(; xlabel="Norm. Poloidal Flux", ylabel="|b^r| [T]",
        title="Resonant area-weighted field |b^r| per surface", legend=:outertopright,
        left_margin=10Plots.mm, bottom_margin=5Plots.mm)

    for nn in unique(rational_n)
        mask = rational_n .== nn
        scatter!(p, rational_psi[mask], abs.(resonant_area_weighted_field[mask]);
            label="n=$nn", markersize=7, markerstrokewidth=0)
        for k in findall(mask)
            abs(rational_q[k] - round(rational_q[k])) < 0.05 || continue
            annotate!(p, rational_psi[k], abs(resonant_area_weighted_field[k]),
                text("  q=$(round(Int, rational_q[k]))", 8, :left, :black))
        end
    end
    return p
end

"""
    plot_island_widths(src; save_path=nothing)

Scatter plot of island half-width `w/2` per rational surface vs ψ_N.
Integer-valued q rational surfaces are annotated.

Requires the perturbed equilibrium singular coupling.

### Arguments

  - `src`: A `PerturbedEquilibriumState`, or the path of a `gpec.h5` it was saved to

### Keyword arguments

  - `save_path`: If provided, save the figure to this path (default: `nothing`)

### Returns

A `Plots.jl` plot object.
"""
function plot_island_widths(src::Source; save_path=nothing)
    _require(plot_island_widths, src)
    p = _plot_island_widths(_coupling_data(src))
    isnothing(save_path) || savefig(p, save_path)
    return p
end

function _plot_island_widths(d)
    island_hw, rational_psi, rational_q = d.island_hw, d.psi, d.q

    p = scatter(
        rational_psi, island_hw;
        xlabel="Norm. Poloidal Flux",
        ylabel="w/2",
        title="Island half-widths",
        legend=false,
        color=:steelblue,
        markersize=7,
        markerstrokewidth=0,
        left_margin=10Plots.mm,
        bottom_margin=5Plots.mm
    )
    for k in eachindex(rational_psi)
        abs(rational_q[k] - round(rational_q[k])) < 0.05 || continue
        annotate!(p, rational_psi[k], island_hw[k],
            text("  q=$(round(Int, rational_q[k]))", 8, :left, :black))
    end
    return p
end

"""
    plot_chirikov_parameter(src; save_path=nothing)

Scatter plot of the Chirikov overlap parameter per rational surface vs ψ_N, with a
horizontal reference line at K = 1 (island overlap threshold). Points are colored red
when K > 1. Integer-valued q rational surfaces are annotated.

Requires the perturbed equilibrium singular coupling.

### Arguments

  - `src`: A `PerturbedEquilibriumState`, or the path of a `gpec.h5` it was saved to

### Keyword arguments

  - `save_path`: If provided, save the figure to this path (default: `nothing`)

### Returns

A `Plots.jl` plot object.
"""
function plot_chirikov_parameter(src::Source; save_path=nothing)
    _require(plot_chirikov_parameter, src)
    p = _plot_chirikov_parameter(_coupling_data(src))
    isnothing(save_path) || savefig(p, save_path)
    return p
end

function _plot_chirikov_parameter(d)
    chirikov, rational_psi, rational_q = d.chirikov, d.psi, d.q

    colors = [k > 1.0 ? :red : :steelblue for k in chirikov]

    p = scatter(
        rational_psi, chirikov;
        xlabel="Norm. Poloidal Flux",
        ylabel="K (Chirikov)",
        title="Chirikov overlap parameter",
        legend=false,
        color=colors,
        markersize=7,
        markerstrokewidth=0,
        left_margin=10Plots.mm,
        bottom_margin=5Plots.mm
    )
    hline!(p, [1.0]; linestyle=:dash, color=:black, label=nothing)
    for k in eachindex(rational_psi)
        abs(rational_q[k] - round(rational_q[k])) < 0.05 || continue
        annotate!(p, rational_psi[k], chirikov[k],
            text("  q=$(round(Int, rational_q[k]))", 8, :left, :black))
    end
    return p
end

"""
    plot_driven_delta_prime(src; save_path=nothing)

Scatter plot of `Re(Δ')` per rational surface vs ψ_N, computed by the perturbed
equilibrium singular coupling.
One marker series per toroidal mode n. Integer-valued q rational surfaces are
annotated.

This is the forcing-driven Δ' (response to the applied perturbation amplitudes
in `forcing_modes`); for the equilibrium-intrinsic Δ' of a Riccati or Galerkin solve,
see `ForceFreeStates.plot_delta_prime`.

### Arguments

  - `src`: A `PerturbedEquilibriumState`, or the path of a `gpec.h5` it was saved to

### Keyword arguments

  - `save_path`: If provided, save the figure to this path (default: `nothing`)

### Returns

A `Plots.jl` plot object.
"""
function plot_driven_delta_prime(src::Source; save_path=nothing)
    _require(plot_driven_delta_prime, src)
    p = _plot_driven_delta_prime(_coupling_data(src))
    isnothing(save_path) || savefig(p, save_path)
    return p
end

function _plot_driven_delta_prime(d)
    delta_prime, rational_psi, rational_q, rational_n = d.delta_prime, d.psi, d.q, d.n

    p = plot(; xlabel="Norm. Poloidal Flux", ylabel="Re(Δ')",
        title="Tearing stability Δ' (driven, perturbed equilibrium)", legend=:outertopright,
        left_margin=10Plots.mm, bottom_margin=5Plots.mm)
    hline!(p, [0.0]; linestyle=:dash, color=:black, label=nothing)

    for nn in unique(rational_n)
        mask = rational_n .== nn
        dp_n = real.(delta_prime[mask])
        colors = [v > 0 ? :red : :steelblue for v in dp_n]
        scatter!(p, rational_psi[mask], dp_n; label="n=$nn", color=colors,
            markersize=7, markerstrokewidth=0)
        for (k, idx) in enumerate(findall(mask))
            abs(rational_q[idx] - round(rational_q[idx])) < 0.05 || continue
            annotate!(p, rational_psi[idx], dp_n[k],
                text("  q=$(round(Int, rational_q[idx]))", 8, :left, :black))
        end
    end
    return p
end

"""
    plot_resonant_area_weighted_field(src; save_path=nothing)

Five-panel summary of resonant coupling quantities at each rational surface vs ψ_N:

  - `|b^r|`: resonant area-weighted field (`plot_resonant_area_weighted_field_amplitude`)
  - `Re(Δ')`: tearing stability parameter (`plot_driven_delta_prime`)
  - `|I_res|`: resonant current
  - `w/2`: island half-width (`plot_island_widths`)
  - `K`: Chirikov overlap parameter (`plot_chirikov_parameter`)

Inspired by `plot_resonant_flux.py` from OMFIT GPEC.

### Arguments

  - `src`: A `PerturbedEquilibriumState`, or the path of a `gpec.h5` it was saved to

### Keyword arguments

  - `save_path`: If provided, save the figure to this path (default: `nothing`)

### Returns

A `Plots.jl` plot object.
"""
function plot_resonant_area_weighted_field(src::Source; save_path=nothing)
    _require(plot_resonant_area_weighted_field, src)
    d = _coupling_data(src)
    p1 = _plot_resonant_area_weighted_field_amplitude(d)
    p2 = _plot_driven_delta_prime(d)
    p3 = _plot_resonant_current(d)
    p4 = _plot_island_widths(d)
    p5 = _plot_chirikov_parameter(d)

    # Move outer legends inside so all panels have the same right margin
    for p in (p1, p2, p3)
        plot!(p; title="", legend=:topright)
    end
    for p in (p4, p5)
        plot!(p; title="")
    end

    hide_xaxis!(p) = plot!(p; xlabel="", xformatter=_ -> "", bottom_margin=1Plots.mm)
    for p in (p1, p2, p3, p4)
        hide_xaxis!(p)
    end

    p = plot(p1, p2, p3, p4, p5; layout=(5, 1), size=(900, 1600))

    isnothing(save_path) || savefig(p, save_path)
    return p
end

# Internal helper — resonant current scatter plot
function _plot_resonant_current(d)
    resonant_current, rational_psi, rational_q, rational_n = d.current, d.psi, d.q, d.n

    p = plot(; xlabel="Norm. Poloidal Flux", ylabel="|I_res|",
        title="Resonant current |I_res| per surface", legend=:outertopright,
        left_margin=10Plots.mm, bottom_margin=5Plots.mm)

    for nn in unique(rational_n)
        mask = rational_n .== nn
        rc_n = abs.(resonant_current[mask])
        scatter!(p, rational_psi[mask], rc_n;
            label="n=$nn", markersize=7, markerstrokewidth=0)
        for (k, idx) in enumerate(findall(mask))
            abs(rational_q[idx] - round(rational_q[idx])) < 0.05 || continue
            annotate!(p, rational_psi[idx], rc_n[k],
                text("  q=$(round(Int, rational_q[idx]))", 8, :left, :black))
        end
    end

    return p
end

const _COMPONENTS = Dict(
    :xi_psi => (dataset="xi_psi", field=:psi, modes=:xi_modes, needs=:response_xi),
    :b_psi => (dataset="b_psi_area_weighted", field=:b_psi_area_weighted, modes=:b_modes, needs=:response_b),  # area-normalized b^ψ
    :Jb_theta => (dataset="Jb_theta", field=:theta, modes=:b_modes, needs=:response_b),
    :Jb_zeta => (dataset="Jb_zeta", field=:zeta, modes=:b_modes, needs=:response_b)
)

function _response_data(s::PerturbedEquilibriumState, component)
    c = _COMPONENTS[component]
    ffs = s.response.ffs
    return (; data=getfield(getfield(s, c.modes), c.field), psi=s.psi_grid, mlow=ffs.mlow, mhigh=ffs.mhigh, nhigh=ffs.nhigh,
        q95=ffs.equil.params.q95, rational_psi=s.response.rational_psi)
end

function _response_data(h5path::AbstractString, component)
    h5open(h5path, "r") do fid
        (; data=read(fid[_RESPONSE*_COMPONENTS[component].dataset]), psi=read(fid[_RESPONSE*"psi"]), mlow=read(fid["Info/mlow"]),
            mhigh=read(fid["Info/mhigh"]), nhigh=read(fid["Info/nhigh"]), q95=read(fid["Equilibrium/q_95"]),
            rational_psi=haskey(fid, _COUPLING * "rational_psi") ? read(fid[_COUPLING*"rational_psi"]) : Float64[])
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

"""
    plot_mode_spectrogram(src; component=:xi_psi, save_path=nothing)

Two-panel spectrogram of a perturbed equilibrium response field component:

  - Top: `|component|` vs ψ_N, one curve per poloidal mode m. Only resonant modes
    (m ∈ [0, nhigh·q95)) are labeled to keep the legend readable.
  - Bottom: Heatmap of `|component|` in (m, ψ_N) space (psi on vertical axis), with
    white dashed lines at rational surface locations.

Inspired by `plot_spectrograms.py` from OMFIT GPEC.

### Arguments

  - `src`: A `PerturbedEquilibriumState` with the reconstructed response, or the path of a `gpec.h5` it was saved to

### Keyword arguments

  - `component`: Response field component to plot; one of `:xi_psi`, `:b_psi`,
    `:Jb_theta`, `:Jb_zeta` (default: `:xi_psi`). `:b_psi` is the area-normalized
    `b_psi_area_weighted` field; `:Jb_theta` and `:Jb_zeta` are the Jacobian-weighted
    contravariant field components.
  - `save_path`: If provided, save the figure to this path (default: `nothing`)

### Returns

A `Plots.jl` plot object.
"""
function plot_mode_spectrogram(src::Source; component=:xi_psi, save_path=nothing)
    _require_component(plot_mode_spectrogram, src, component)
    p = _plot_mode_spectrogram(_response_data(src, component), component)
    isnothing(save_path) || savefig(p, save_path)
    return p
end

function _plot_mode_spectrogram(d, component)
    mpert = d.mhigh - d.mlow + 1
    m_vals = d.mlow:d.mhigh
    data_mn = d.data[:, 1:mpert]  # (npsi, mpert): first mpert columns = first n's modes

    # Top panel: line plot per mode — only label resonant range m ∈ [0, nhigh·q95)
    m_max_legend = d.nhigh * d.q95
    p1 = plot(;
        xlabel="Norm. Poloidal Flux",
        ylabel="|$(component)|",
        title="Mode spectrogram: $(component)",
        legend=:outertopright,
        left_margin=10Plots.mm,
        bottom_margin=5Plots.mm
    )
    cmap = cgrad(:roma, mpert; categorical=true)
    for (i, m) in enumerate(m_vals)
        show_label = 0 <= m < m_max_legend
        plot!(p1, d.psi, abs.(data_mn[:, i]);
            label=(show_label ? "m=$m" : nothing), color=cmap[i], linewidth=1.5)
    end

    # Bottom panel: heatmap in (m, ψ_N) space — psi on vertical axis, m on horizontal
    p2 = heatmap(
        collect(m_vals), d.psi, abs.(data_mn);
        xlabel="m",
        ylabel="Norm. Poloidal Flux",
        title="",
        colorbar_title="|$(component)|",
        left_margin=10Plots.mm,
        right_margin=20Plots.mm,
        bottom_margin=5Plots.mm
    )
    # Overlay rational surface locations as white dashed lines
    for psi_s in d.rational_psi
        hline!(p2, [psi_s]; linestyle=:dash, color=:white, linewidth=1.5, label=nothing)
    end

    return plot(p1, p2; layout=(2, 1), size=(950, 750))
end

"""
    plot_perturbed_equilibrium_summary(src; save_path=nothing)

Composite summary of perturbed equilibrium results, with each panel the source can supply:

  - Top-left: Resonant area-weighted field per surface (`plot_resonant_area_weighted_field_amplitude`)
  - Top-right: Edge |b_ψ| spectrum
  - Bottom: ξ_ψ mode spectrogram (`plot_mode_spectrogram`)

Panels the source cannot draw are left out; it throws an `ArgumentError` only when none can be drawn.

### Arguments

  - `src`: A `PerturbedEquilibriumState`, or the path of a `gpec.h5` it was saved to

### Keyword arguments

  - `save_path`: If provided, save the figure to this path (default: `nothing`)

### Returns

A `Plots.jl` plot object.
"""
function plot_perturbed_equilibrium_summary(src::Source; save_path=nothing)
    caps = capabilities(src)
    top = Plots.Plot[]
    _can(plot_resonant_area_weighted_field_amplitude, caps) && push!(top, _plot_resonant_area_weighted_field_amplitude(_coupling_data(src)))
    :response_b in caps && push!(top, _plot_bpsi_edge_spectrum(_response_data(src, :b_psi)))
    has_spectro = _can(plot_mode_spectrogram, caps)
    if !has_spectro && isempty(top)
        throw(ArgumentError("plot_perturbed_equilibrium_summary: no panel can be drawn: $(REASONS[:singular_coupling]); $(REASONS[:response_xi])"))
    end
    p = if !has_spectro
        plot(top...; layout=(1, length(top)), size=(550 * length(top), 450))
    else
        p_spectro = _plot_mode_spectrogram(_response_data(src, :xi_psi), :xi_psi)
        isempty(top) ? p_spectro : plot(top..., p_spectro; layout=@layout([grid(1, length(top)){0.35h}; b]), size=(1100, 1100))
    end

    isnothing(save_path) || savefig(p, save_path)
    return p
end

# Internal helper — |b_psi(m)| spectrum at the outermost psi surface
function _plot_bpsi_edge_spectrum(d)
    mpert = d.mhigh - d.mlow + 1
    m_vals = d.mlow:d.mhigh
    bpsi_edge = abs.(d.data[end, 1:mpert])

    p = bar(
        collect(m_vals), bpsi_edge;
        xlabel="m",
        ylabel="|b_ψ|",
        title="b_ψ spectrum at ψ edge",
        legend=false,
        color=:steelblue,
        linewidth=0,
        left_margin=10Plots.mm,
        bottom_margin=5Plots.mm
    )

    return p
end

const _PE_SOURCES = (PerturbedEquilibriumState, AbstractString)
register!(plot_perturbed_equilibrium_summary, _PE_SOURCES, (), "resonant field, edge b_ψ spectrum and ξ spectrogram")
register!(plot_resonant_area_weighted_field, _PE_SOURCES, (:singular_coupling,), "five resonant-coupling panels per rational surface")
register!(plot_resonant_area_weighted_field_amplitude, _PE_SOURCES, (:singular_coupling,), "|b̄_res| per rational surface")
register!(plot_island_widths, _PE_SOURCES, (:singular_coupling,), "island half-width per rational surface")
register!(plot_chirikov_parameter, _PE_SOURCES, (:singular_coupling,), "Chirikov parameter per rational surface")
register!(plot_driven_delta_prime, _PE_SOURCES, (:singular_coupling,), "forcing-driven Δ′ per rational surface")
register!(plot_mode_spectrogram, _PE_SOURCES, (:response_xi,), "|response| per m against ψ_N (component = :xi_psi, :b_psi, :Jb_theta, :Jb_zeta)")

end # module PerturbedEquilibrium
