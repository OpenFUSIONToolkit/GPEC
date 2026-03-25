"""
    PerturbedEquilibrium

Post-processing and visualization functions for GPEC perturbed equilibrium results stored
in the `perturbed_equilibrium/` group of a GPEC HDF5 output file.
"""
module PerturbedEquilibrium

using HDF5
using LaTeXStrings
using Plots

# Check that a PE dataset exists and is non-empty.
function _has_pe_data(h5path, key)
    h5open(h5path, "r") do fid
        haskey(fid, key) && !isempty(read(fid[key]))
    end
end

"""
    plot_resonant_flux(h5path; save_path=nothing)

Scatter plot of `|Φ_res|` (normalized resonant flux) per singular surface vs ψ_N. One
marker series per toroidal mode n. Integer-valued q rational surfaces are annotated.

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

    resonant_flux, psi_sing, q_sing, msing, pe_n = h5open(h5path, "r") do fid
        read(fid[key]), read(fid["singular/psi"]), read(fid["singular/q"]),
        read(fid["singular/msing"]),
        read(fid["perturbed_equilibrium/forcing_modes/n"])
    end

    p = plot(; xlabel="ψₙ", ylabel="|Φ_res|",
        title="Resonant flux |Φ_res| per surface", legend=:outertopright,
        left_margin=5Plots.mm, bottom_margin=5Plots.mm)

    n_vals = unique(pe_n)
    for nn in n_vals
        n_rows = findall(==(nn), pe_n)
        rf_n = [sum(abs.(resonant_flux[n_rows, s])) for s in 1:msing]
        scatter!(p, psi_sing, rf_n; label="n=$nn", markersize=7, markerstrokewidth=0)
        # Annotate integer-q surfaces only (avoids overcrowding for n>1)
        for s in 1:msing
            abs(q_sing[s] - round(q_sing[s])) < 0.05 || continue
            annotate!(p, psi_sing[s], rf_n[s],
                text("  q=$(round(Int, q_sing[s]))", 8, :left, :black))
        end
    end

    isnothing(save_path) || savefig(p, save_path)
    return p
end

"""
    plot_island_widths(h5path; save_path=nothing)

Scatter plot of island half-width `w/2` per singular surface vs ψ_N.
Integer-valued q rational surfaces are annotated.

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

    island_hw, psi_sing, q_sing, msing = h5open(h5path, "r") do fid
        read(fid[key]), read(fid["singular/psi"]), read(fid["singular/q"]),
        read(fid["singular/msing"])
    end

    p = scatter(
        psi_sing, island_hw;
        xlabel="ψₙ",
        ylabel="w/2",
        title="Island half-widths",
        legend=false,
        color=:steelblue,
        markersize=7,
        markerstrokewidth=0,
        left_margin=5Plots.mm,
        bottom_margin=5Plots.mm
    )
    for s in 1:msing
        abs(q_sing[s] - round(q_sing[s])) < 0.05 || continue
        annotate!(p, psi_sing[s], island_hw[s],
            text("  q=$(round(Int, q_sing[s]))", 8, :left, :black))
    end

    isnothing(save_path) || savefig(p, save_path)
    return p
end

"""
    plot_chirikov_parameter(h5path; save_path=nothing)

Scatter plot of the Chirikov overlap parameter per singular surface vs ψ_N, with a horizontal
reference line at K = 1 (island overlap threshold). Points are colored red when K > 1.
Integer-valued q rational surfaces are annotated.

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

    chirikov, psi_sing, q_sing, msing = h5open(h5path, "r") do fid
        read(fid[key]), read(fid["singular/psi"]), read(fid["singular/q"]),
        read(fid["singular/msing"])
    end

    colors = [k > 1.0 ? :red : :steelblue for k in chirikov]

    p = scatter(
        psi_sing, chirikov;
        xlabel="ψₙ",
        ylabel="K (Chirikov)",
        title="Chirikov overlap parameter",
        legend=false,
        color=colors,
        markersize=7,
        markerstrokewidth=0,
        left_margin=5Plots.mm,
        bottom_margin=5Plots.mm
    )
    hline!(p, [1.0]; linestyle=:dash, color=:black, label=nothing)
    for s in 1:msing
        abs(q_sing[s] - round(q_sing[s])) < 0.05 || continue
        annotate!(p, psi_sing[s], chirikov[s],
            text("  q=$(round(Int, q_sing[s]))", 8, :left, :black))
    end

    isnothing(save_path) || savefig(p, save_path)
    return p
end

"""
    plot_pe_delta_prime(h5path; save_path=nothing)

Scatter plot of `Re(Δ')` per singular surface vs ψ_N, computed by the perturbed equilibrium
module (from `singular_coupling/delta_prime`). One marker series per toroidal mode n.
Integer-valued q rational surfaces are annotated.

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

    delta_prime, psi_sing, q_sing, msing, pe_n = h5open(h5path, "r") do fid
        read(fid[key]), read(fid["singular/psi"]), read(fid["singular/q"]),
        read(fid["singular/msing"]),
        read(fid["perturbed_equilibrium/forcing_modes/n"])
    end

    p = plot(; xlabel="ψₙ", ylabel="Re(Δ')",
        title="Tearing stability Δ' (PE)", legend=:outertopright,
        left_margin=5Plots.mm, bottom_margin=5Plots.mm)
    hline!(p, [0.0]; linestyle=:dash, color=:black, label=nothing)

    n_vals = unique(pe_n)
    for nn in n_vals
        n_rows = findall(==(nn), pe_n)
        dp_n = [real(delta_prime[n_rows[1], s]) for s in 1:msing]
        colors = [v > 0 ? :red : :steelblue for v in dp_n]
        scatter!(p, psi_sing, dp_n; label="n=$nn", color=colors,
            markersize=7, markerstrokewidth=0)
        for s in 1:msing
            abs(q_sing[s] - round(q_sing[s])) < 0.05 || continue
            annotate!(p, psi_sing[s], dp_n[s],
                text("  q=$(round(Int, q_sing[s]))", 8, :left, :black))
        end
    end

    isnothing(save_path) || savefig(p, save_path)
    return p
end

"""
    plot_resonant_field(h5path; save_path=nothing)

Five-panel summary of resonant coupling quantities at each singular surface vs ψ_N:

  - `|Φ_res|`: resonant flux (`plot_resonant_flux`)
  - `Re(Δ')`: tearing stability parameter (`plot_pe_delta_prime`)
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

    # Move outer legends inside so all panels have the same right margin
    for p in (p1, p2, p3)
        plot!(p; title="", legend=:topright)
    end
    for p in (p4, p5)
        plot!(p; title="")
    end

    hide_xaxis!(p) = plot!(p; xlabel="", xformatter=_->"", bottom_margin=1Plots.mm)
    for p in (p1, p2, p3, p4)
        hide_xaxis!(p)
    end

    p = plot(p1, p2, p3, p4, p5; layout=(5, 1), size=(900, 1600))

    isnothing(save_path) || savefig(p, save_path)
    return p
end

# Internal helper — resonant current scatter plot
function _plot_resonant_current(h5path)
    key = "perturbed_equilibrium/singular_coupling/resonant_current"
    _has_pe_data(h5path, key) ||
        return plot(; title="No resonant current data", legend=false)

    resonant_current, psi_sing, q_sing, msing, pe_n = h5open(h5path, "r") do fid
        read(fid[key]), read(fid["singular/psi"]), read(fid["singular/q"]),
        read(fid["singular/msing"]),
        read(fid["perturbed_equilibrium/forcing_modes/n"])
    end

    p = plot(; xlabel="ψₙ", ylabel="|I_res|",
        title="Resonant current |I_res| per surface", legend=:outertopright,
        left_margin=5Plots.mm, bottom_margin=5Plots.mm)

    n_vals = unique(pe_n)
    for nn in n_vals
        n_rows = findall(==(nn), pe_n)
        rc_n = [sum(abs.(resonant_current[n_rows, s])) for s in 1:msing]
        scatter!(p, psi_sing, rc_n; label="n=$nn", markersize=7, markerstrokewidth=0)
        for s in 1:msing
            abs(q_sing[s] - round(q_sing[s])) < 0.05 || continue
            annotate!(p, psi_sing[s], rc_n[s],
                text("  q=$(round(Int, q_sing[s]))", 8, :left, :black))
        end
    end

    return p
end

"""
    plot_mode_spectrogram(h5path; component=:xi_psi, save_path=nothing)

Two-panel spectrogram of a perturbed equilibrium response field component:

  - Top: `|component|` vs ψ_N, one curve per poloidal mode m. Only resonant modes
    (m ∈ [0, nhigh·q95)) are labeled to keep the legend readable.
  - Bottom: Heatmap of `|component|` in (m, ψ_N) space (psi on vertical axis), with
    white dashed lines at rational surface locations.

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

    data_r, data_i, psi_response, mlow, mhigh, nhigh, q95, msing, psi_sing = h5open(h5path, "r") do fid
        read(fid[base * rkey]), read(fid[base * ikey]),
        read(fid["integration/psi"]),
        read(fid["info/mlow"]), read(fid["info/mhigh"]), read(fid["info/nhigh"]),
        read(fid["equil/q95"]),
        read(fid["singular/msing"]), read(fid["singular/psi"])
    end

    data = complex.(data_r, data_i)  # shape: (npsi, numpert_total)
    mpert = mhigh - mlow + 1

    # Use only the first n's modes for a clean spectrogram (single-n assumption for display)
    m_vals = mlow:mhigh
    data_mn = data[:, 1:mpert]  # (npsi, mpert): first mpert columns = first n's modes

    # Top panel: line plot per mode — only label resonant range m ∈ [0, nhigh·q95)
    m_max_legend = nhigh * q95
    p1 = plot(;
        xlabel="ψₙ",
        ylabel="|$(component)|",
        title="Mode spectrogram: $(component)",
        legend=:outertopright,
        left_margin=5Plots.mm,
        bottom_margin=5Plots.mm
    )
    cmap = cgrad(:roma, mpert; categorical=true)
    for (i, m) in enumerate(m_vals)
        show_label = 0 <= m < m_max_legend
        plot!(p1, psi_response, abs.(data_mn[:, i]);
            label=(show_label ? "m=$m" : nothing), color=cmap[i], linewidth=1.5)
    end

    # Bottom panel: heatmap in (m, ψ_N) space — psi on vertical axis, m on horizontal
    # z must be (n_psi, n_m) = (npsi, mpert) so that rows = psi, cols = m
    p2 = heatmap(
        collect(m_vals), psi_response, abs.(data_mn);
        xlabel="m",
        ylabel="ψₙ",
        title="",
        colorbar_title="|$(component)|",
        left_margin=5Plots.mm,
        right_margin=10Plots.mm,
        bottom_margin=5Plots.mm
    )
    # Overlay rational surface locations as white dashed lines
    for s in 1:msing
        hline!(p2, [psi_sing[s]]; linestyle=:dash, color=:white, linewidth=1.5, label=nothing)
    end

    p = plot(p1, p2; layout=(2, 1), size=(950, 750))

    isnothing(save_path) || savefig(p, save_path)
    return p
end

"""
    plot_perturbed_equilibrium_summary(h5path; save_path=nothing)

Three-panel composite summary of perturbed equilibrium results:

  - Top-left: Island half-widths (`plot_island_widths`)
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
    p = plot(p_islands, p_energies, p_spectro; layout=l, size=(1100, 1100))

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

    vals  = [real(ep), real(ev), real(et)]  # real() in case energies are stored as complex
    names = ["Plasma", "Vacuum", "Total"]
    colors = [:steelblue, :darkorange, :green]

    p = bar(
        names, vals;
        ylabel="Energy",
        title="Energy breakdown",
        legend=false,
        color=colors,
        left_margin=5Plots.mm,
        bottom_margin=5Plots.mm
    )
    hline!(p, [0]; linestyle=:dash, color=:black, label=nothing)

    return p
end

end # module PerturbedEquilibrium
