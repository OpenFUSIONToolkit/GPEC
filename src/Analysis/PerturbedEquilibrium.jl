"""
    PerturbedEquilibrium

Post-processing and visualization functions for GPEC perturbed equilibrium results stored
in the `perturbed_equilibrium/` group of a GPEC HDF5 output file.
"""
module PerturbedEquilibrium

using HDF5
using Plots

# Check that a PE dataset exists and is non-empty.
function _has_pe_data(h5path, key)
    h5open(h5path, "r") do fid
        haskey(fid, key) && !isempty(read(fid[key]))
    end
end

"""
    plot_resonant_flux(h5path; save_path=nothing)

Bar chart of `|Φ_res|` (normalized resonant flux) per singular surface. One bar series per
toroidal mode n, labeled by n value.

Requires the perturbed equilibrium module to have been run and `singular_coupling/resonant_flux`
to be present in the HDF5 file.

### Arguments

  - `h5path`: Path to a GPEC HDF5 output file with perturbed equilibrium output

### Keyword arguments

  - `save_path`: If provided, save the figure to this path (default: `nothing`)

### Returns

A `Plots.jl` plot object.
"""
function plot_resonant_flux(h5path; save_path=nothing)
    key = "perturbed_equilibrium/singular_coupling/resonant_flux"
    _has_pe_data(h5path, key) ||
        return plot(; title="No resonant flux data — run with perturbed equilibrium enabled", legend=false)

    resonant_flux, q_sing, msing, mn_index = h5open(h5path, "r") do fid
        read(fid[key]), read(fid["singular/q"]),
        read(fid["singular/msing"]), read(fid["info/mn_index"])
    end

    labels = ["q=$(round(q_sing[s], digits=3))" for s in 1:msing]
    xticks_arg = (1:msing, labels)

    p = plot(; xlabel="rational surface", ylabel="|Φ_res|",
        title="Resonant flux |Φ_res| per surface", legend=:outertopright)

    n_vals = unique(mn_index[:, 2])
    for nn in n_vals
        n_rows = findall(j -> mn_index[j, 2] == nn, 1:size(mn_index, 1))
        # Sum over poloidal modes for this n
        rf_n = [sum(abs.(resonant_flux[n_rows, s])) for s in 1:msing]
        bar!(p, 1:msing, rf_n; label="n=$nn", alpha=0.7)
    end
    plot!(p; xticks=xticks_arg, xrotation=30)

    isnothing(save_path) || savefig(p, save_path)
    return p
end

"""
    plot_island_widths(h5path; save_path=nothing)

Bar chart of island half-width `w/2` per singular surface.

Requires `singular_coupling/island_half_width` in the HDF5 file.

### Arguments

  - `h5path`: Path to a GPEC HDF5 output file with perturbed equilibrium output

### Keyword arguments

  - `save_path`: If provided, save the figure to this path (default: `nothing`)

### Returns

A `Plots.jl` plot object.
"""
function plot_island_widths(h5path; save_path=nothing)
    key = "perturbed_equilibrium/singular_coupling/island_half_width"
    _has_pe_data(h5path, key) ||
        return plot(; title="No island width data — run with perturbed equilibrium enabled", legend=false)

    island_hw, q_sing, msing = h5open(h5path, "r") do fid
        read(fid[key]), read(fid["singular/q"]), read(fid["singular/msing"])
    end

    labels = ["q=$(round(q_sing[s], digits=3))" for s in 1:msing]

    p = bar(
        1:msing, island_hw;
        xticks=(1:msing, labels),
        xlabel="rational surface",
        ylabel="w/2",
        title="Island half-widths",
        legend=false,
        color=:steelblue,
        xrotation=30
    )

    isnothing(save_path) || savefig(p, save_path)
    return p
end

"""
    plot_chirikov_parameter(h5path; save_path=nothing)

Bar chart of the Chirikov overlap parameter per singular surface, with a horizontal reference
line at K = 1 (island overlap threshold). Bars are colored red when K > 1.

Requires `singular_coupling/chirikov_parameter` in the HDF5 file.

### Arguments

  - `h5path`: Path to a GPEC HDF5 output file with perturbed equilibrium output

### Keyword arguments

  - `save_path`: If provided, save the figure to this path (default: `nothing`)

### Returns

A `Plots.jl` plot object.
"""
function plot_chirikov_parameter(h5path; save_path=nothing)
    key = "perturbed_equilibrium/singular_coupling/chirikov_parameter"
    _has_pe_data(h5path, key) ||
        return plot(; title="No Chirikov data — run with perturbed equilibrium enabled", legend=false)

    chirikov, q_sing, msing = h5open(h5path, "r") do fid
        read(fid[key]), read(fid["singular/q"]), read(fid["singular/msing"])
    end

    labels = ["q=$(round(q_sing[s], digits=3))" for s in 1:msing]
    colors = [k > 1.0 ? :red : :steelblue for k in chirikov]

    p = bar(
        1:msing, chirikov;
        xticks=(1:msing, labels),
        xlabel="rational surface",
        ylabel="K (Chirikov)",
        title="Chirikov overlap parameter",
        legend=false,
        color=colors,
        xrotation=30
    )
    hline!(p, [1.0]; linestyle=:dash, color=:black, label=nothing)

    isnothing(save_path) || savefig(p, save_path)
    return p
end

"""
    plot_pe_delta_prime(h5path; save_path=nothing)

Bar chart of `|Δ'|` per singular surface computed by the perturbed equilibrium module
(from `singular_coupling/delta_prime`). One bar series per toroidal mode n.

This is complementary to `Analysis.ForceFreeStates.plot_delta_prime`, which uses the FFS
asymptotic coefficients. The PE result includes the vacuum Green's function contribution.

Requires `singular_coupling/delta_prime` in the HDF5 file.

### Arguments

  - `h5path`: Path to a GPEC HDF5 output file with perturbed equilibrium output

### Keyword arguments

  - `save_path`: If provided, save the figure to this path (default: `nothing`)

### Returns

A `Plots.jl` plot object.
"""
function plot_pe_delta_prime(h5path; save_path=nothing)
    key = "perturbed_equilibrium/singular_coupling/delta_prime"
    _has_pe_data(h5path, key) ||
        return plot(; title="No PE Δ' data — run with perturbed equilibrium enabled", legend=false)

    delta_prime, q_sing, msing, mn_index = h5open(h5path, "r") do fid
        read(fid[key]), read(fid["singular/q"]), read(fid["singular/msing"]),
        read(fid["info/mn_index"])
    end

    labels = ["q=$(round(q_sing[s], digits=3))" for s in 1:msing]

    p = plot(; xlabel="rational surface", ylabel="|Δ'|",
        title="Tearing stability Δ' (PE)", legend=:outertopright)

    n_vals = unique(mn_index[:, 2])
    for nn in n_vals
        n_rows = findall(j -> mn_index[j, 2] == nn, 1:size(mn_index, 1))
        dp_n = [maximum(abs.(delta_prime[n_rows, s])) for s in 1:msing]
        bar!(p, 1:msing, dp_n; label="n=$nn", alpha=0.7)
    end
    plot!(p; xticks=(1:msing, labels), xrotation=30)

    isnothing(save_path) || savefig(p, save_path)
    return p
end

"""
    plot_resonant_field(h5path; save_path=nothing)

Five-panel summary of resonant coupling quantities at each singular surface:

  - `|Φ_res|`: resonant flux (`plot_resonant_flux`)
  - `|Δ'|`: tearing stability parameter (`plot_pe_delta_prime`)
  - `|I_res|`: resonant current
  - `w/2`: island half-width (`plot_island_widths`)
  - `K`: Chirikov overlap parameter (`plot_chirikov_parameter`)

Inspired by `plot_resonant_field.py` from OMFIT GPEC.

### Arguments

  - `h5path`: Path to a GPEC HDF5 output file with perturbed equilibrium output

### Keyword arguments

  - `save_path`: If provided, save the figure to this path (default: `nothing`)

### Returns

A `Plots.jl` plot object.
"""
function plot_resonant_field(h5path; save_path=nothing)
    p1 = plot_resonant_flux(h5path)
    p2 = plot_pe_delta_prime(h5path)
    p3 = _plot_resonant_current(h5path)
    p4 = plot_island_widths(h5path)
    p5 = plot_chirikov_parameter(h5path)

    p = plot(p1, p2, p3, p4, p5; layout=(5, 1), size=(800, 1400))

    isnothing(save_path) || savefig(p, save_path)
    return p
end

# Internal helper — resonant current bar chart
function _plot_resonant_current(h5path)
    key = "perturbed_equilibrium/singular_coupling/resonant_current"
    _has_pe_data(h5path, key) ||
        return plot(; title="No resonant current data", legend=false)

    resonant_current, q_sing, msing, mn_index = h5open(h5path, "r") do fid
        read(fid[key]), read(fid["singular/q"]), read(fid["singular/msing"]),
        read(fid["info/mn_index"])
    end

    labels = ["q=$(round(q_sing[s], digits=3))" for s in 1:msing]

    p = plot(; xlabel="rational surface", ylabel="|I_res|",
        title="Resonant current |I_res| per surface", legend=:outertopright)

    n_vals = unique(mn_index[:, 2])
    for nn in n_vals
        n_rows = findall(j -> mn_index[j, 2] == nn, 1:size(mn_index, 1))
        rc_n = [sum(abs.(resonant_current[n_rows, s])) for s in 1:msing]
        bar!(p, 1:msing, rc_n; label="n=$nn", alpha=0.7)
    end
    plot!(p; xticks=(1:msing, labels), xrotation=30)

    return p
end

"""
    plot_mode_spectrogram(h5path; component=:xi_psi, save_path=nothing)

Two-panel spectrogram of a perturbed equilibrium response field component:

  - Top: `|component|` vs ψ_N, one curve per poloidal mode m (colored by m)
  - Bottom: Heatmap of `|component|` in (ψ_N × m) space, with white dashed lines at
    rational surface locations

Inspired by `plot_spectrograms.py` from OMFIT GPEC.

### Arguments

  - `h5path`: Path to a GPEC HDF5 output file with perturbed equilibrium response output

### Keyword arguments

  - `component`: Response field component to plot; one of `:xi_psi`, `:b_psi`,
    `:b_theta`, `:b_zeta` (default: `:xi_psi`)
  - `save_path`: If provided, save the figure to this path (default: `nothing`)

### Returns

A `Plots.jl` plot object.
"""
function plot_mode_spectrogram(h5path; component=:xi_psi, save_path=nothing)
    comp_map = Dict(
        :xi_psi  => ("xi_psi_real",  "xi_psi_imag"),
        :b_psi   => ("b_psi_real",   "b_psi_imag"),
        :b_theta => ("b_theta_real", "b_theta_imag"),
        :b_zeta  => ("b_zeta_real",  "b_zeta_imag"),
    )
    haskey(comp_map, component) ||
        error("component must be one of :xi_psi, :b_psi, :b_theta, :b_zeta")

    rkey, ikey = comp_map[component]
    base = "perturbed_equilibrium/response/"

    _has_pe_data(h5path, base * rkey) ||
        return plot(; title="No response data — run with perturbed equilibrium enabled", legend=false)

    data_r, data_i, xs, mlow, mhigh, msing, psi_sing = h5open(h5path, "r") do fid
        read(fid[base * rkey]), read(fid[base * ikey]),
        read(fid["splines/profiles/xs"]),
        read(fid["info/mlow"]), read(fid["info/mhigh"]),
        read(fid["singular/msing"]), read(fid["singular/psi"])
    end

    data = complex.(data_r, data_i)  # shape: (numpert_total, npsi)
    mpert = mhigh - mlow + 1

    # Use only the first n's modes for a clean spectrogram (single-n assumption for display)
    m_vals = mlow:mhigh
    data_mn = data[1:mpert, :]  # first mpert rows correspond to first n

    # Top panel: line plot per mode
    p1 = plot(;
        xlabel="ψ_N",
        ylabel="|$(component)|",
        title="Mode spectrogram: $(component)",
        legend=:outertopright
    )
    cmap = cgrad(:roma, mpert; categorical=true)
    for (i, m) in enumerate(m_vals)
        plot!(p1, xs, abs.(data_mn[i, :]); label="m=$m", color=cmap[i], linewidth=1.5)
    end

    # Bottom panel: heatmap
    p2 = heatmap(
        xs, collect(m_vals), abs.(data_mn);
        xlabel="ψ_N",
        ylabel="m",
        title="",
        colorbar_title="|$(component)|"
    )
    # Overlay rational surface locations as white dashed lines
    for s in 1:msing
        vline!(p2, [psi_sing[s]]; linestyle=:dash, color=:white, linewidth=1.5, label=nothing)
    end

    p = plot(p1, p2; layout=(2, 1), size=(900, 700))

    isnothing(save_path) || savefig(p, save_path)
    return p
end

"""
    plot_perturbed_equilibrium_summary(h5path; save_path=nothing)

Three-panel composite summary of perturbed equilibrium results:

  - Top-left: Island half-widths and Chirikov parameter overlay (`plot_island_widths` +
    `plot_chirikov_parameter` on shared axes)
  - Top-right: Energy breakdown — plasma, vacuum, and total energies
  - Bottom: ξ_ψ mode spectrogram (`plot_mode_spectrogram`)

### Arguments

  - `h5path`: Path to a GPEC HDF5 output file with perturbed equilibrium output

### Keyword arguments

  - `save_path`: If provided, save the figure to this path (default: `nothing`)

### Returns

A `Plots.jl` plot object.
"""
function plot_perturbed_equilibrium_summary(h5path; save_path=nothing)
    p_islands  = plot_island_widths(h5path)
    p_energies = _plot_energies(h5path)
    p_spectro  = plot_mode_spectrogram(h5path; component=:xi_psi)

    l = @layout [grid(1, 2){0.35h}; b]
    p = plot(p_islands, p_energies, p_spectro; layout=l, size=(1000, 1000))

    isnothing(save_path) || savefig(p, save_path)
    return p
end

# Internal helper — energy breakdown bar chart
function _plot_energies(h5path)
    key = "perturbed_equilibrium/energies/plasma_energy"
    _has_pe_data(h5path, key) ||
        return plot(; title="No energy data", legend=false)

    ep, ev, et = h5open(h5path, "r") do fid
        read(fid["perturbed_equilibrium/energies/plasma_energy"]),
        read(fid["perturbed_equilibrium/energies/vacuum_energy"]),
        read(fid["perturbed_equilibrium/energies/total_energy"])
    end

    vals  = [real(ep), real(ev), real(et)]
    names = ["Plasma", "Vacuum", "Total"]
    colors = [:steelblue, :darkorange, :green]

    p = bar(
        names, vals;
        ylabel="Energy",
        title="Energy breakdown",
        legend=false,
        color=colors
    )
    hline!(p, [0]; linestyle=:dash, color=:black, label=nothing)

    return p
end

end # module PerturbedEquilibrium
