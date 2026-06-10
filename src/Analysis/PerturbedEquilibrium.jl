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

Scatter plot of `|Φ_res|` per rational surface vs ψ_N. One marker series per toroidal
mode n. Integer-valued q rational surfaces are annotated.

Requires the perturbed equilibrium module to have been run and
`singular_coupling/resonant_flux` to be present in the HDF5 file.

### Arguments

  - `h5path`: Path to a GPEC HDF5 output file with perturbed equilibrium output

### Keyword arguments

  - `save_path`: If provided, save the figure to this path (default: `nothing`)

### Returns

A `Plots.jl` plot object.
"""
function plot_resonant_flux(h5path; save_path=nothing)
    base = "perturbed_equilibrium/singular_coupling/"
    _has_pe_data(h5path, base * "resonant_flux") ||
        return plot(; title="No resonant flux data — run with perturbed equilibrium enabled", legend=false)

    resonant_flux, rational_psi, rational_q, rational_n = h5open(h5path, "r") do fid
        read(fid[base * "resonant_flux"]),
        read(fid[base * "rational_psi"]),
        read(fid[base * "rational_q"]),
        read(fid[base * "rational_n"])
    end

    p = plot(; xlabel="Norm. Poloidal Flux", ylabel="|Φ_res|",
        title="Resonant flux |Φ_res| per surface", legend=:outertopright,
        left_margin=10Plots.mm, bottom_margin=5Plots.mm)

    for nn in unique(rational_n)
        mask = rational_n .== nn
        scatter!(p, rational_psi[mask], abs.(resonant_flux[mask]);
            label="n=$nn", markersize=7, markerstrokewidth=0)
        for k in findall(mask)
            abs(rational_q[k] - round(rational_q[k])) < 0.05 || continue
            annotate!(p, rational_psi[k], abs(resonant_flux[k]),
                text("  q=$(round(Int, rational_q[k]))", 8, :left, :black))
        end
    end

    isnothing(save_path) || savefig(p, save_path)
    return p
end

"""
    plot_island_widths(h5path; save_path=nothing)

Scatter plot of island half-width `w/2` per rational surface vs ψ_N.
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
    base = "perturbed_equilibrium/singular_coupling/"
    _has_pe_data(h5path, base * "island_half_width") ||
        return plot(; title="No island width data — run with perturbed equilibrium enabled", legend=false)

    island_hw, rational_psi, rational_q = h5open(h5path, "r") do fid
        read(fid[base * "island_half_width"]),
        read(fid[base * "rational_psi"]),
        read(fid[base * "rational_q"])
    end

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

    isnothing(save_path) || savefig(p, save_path)
    return p
end

"""
    plot_chirikov_parameter(h5path; save_path=nothing)

Scatter plot of the Chirikov overlap parameter per rational surface vs ψ_N, with a
horizontal reference line at K = 1 (island overlap threshold). Points are colored red
when K > 1. Integer-valued q rational surfaces are annotated.

Requires `singular_coupling/chirikov_parameter` in the HDF5 file.

### Arguments

  - `h5path`: Path to a GPEC HDF5 output file with perturbed equilibrium output

### Keyword arguments

  - `save_path`: If provided, save the figure to this path (default: `nothing`)

### Returns

A `Plots.jl` plot object.
"""
function plot_chirikov_parameter(h5path; save_path=nothing)
    base = "perturbed_equilibrium/singular_coupling/"
    _has_pe_data(h5path, base * "chirikov_parameter") ||
        return plot(; title="No Chirikov data — run with perturbed equilibrium enabled", legend=false)

    chirikov, rational_psi, rational_q = h5open(h5path, "r") do fid
        read(fid[base * "chirikov_parameter"]),
        read(fid[base * "rational_psi"]),
        read(fid[base * "rational_q"])
    end

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

    isnothing(save_path) || savefig(p, save_path)
    return p
end

"""
    plot_driven_delta_prime(h5path; save_path=nothing)

Scatter plot of `Re(Δ')` per rational surface vs ψ_N, computed by the perturbed
equilibrium module (from `perturbed_equilibrium/singular_coupling/delta_prime`).
One marker series per toroidal mode n. Integer-valued q rational surfaces are
annotated.

This is the forcing-driven Δ' (response to the applied perturbation amplitudes
in `intr.forcing_modes`); for the equilibrium-intrinsic Δ' from the STRIDE BVP,
read `singular/delta_prime_matrix` from the HDF5 directly.

Requires `perturbed_equilibrium/singular_coupling/delta_prime` in the HDF5 file.

### Arguments

  - `h5path`: Path to a GPEC HDF5 output file

### Keyword arguments

  - `save_path`: If provided, save the figure to this path (default: `nothing`)

### Returns

A `Plots.jl` plot object.
"""
function plot_driven_delta_prime(h5path; save_path=nothing)
    base = "perturbed_equilibrium/singular_coupling/"
    _has_pe_data(h5path, base * "delta_prime") ||
        return plot(; title="No PE Δ' data — run with perturbed equilibrium enabled", legend=false)

    delta_prime, rational_psi, rational_q, rational_n = h5open(h5path, "r") do fid
        read(fid[base * "delta_prime"]),
        read(fid[base * "rational_psi"]),
        read(fid[base * "rational_q"]),
        read(fid[base * "rational_n"])
    end

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

    isnothing(save_path) || savefig(p, save_path)
    return p
end

"""
    plot_resonant_field(h5path; save_path=nothing)

Five-panel summary of resonant coupling quantities at each rational surface vs ψ_N:

  - `|Φ_res|`: resonant flux (`plot_resonant_flux`)
  - `Re(Δ')`: tearing stability parameter (`plot_driven_delta_prime`)
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
    p2 = plot_driven_delta_prime(h5path)
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
    base = "perturbed_equilibrium/singular_coupling/"
    _has_pe_data(h5path, base * "resonant_current") ||
        return plot(; title="No resonant current data", legend=false)

    resonant_current, rational_psi, rational_q, rational_n = h5open(h5path, "r") do fid
        read(fid[base * "resonant_current"]),
        read(fid[base * "rational_psi"]),
        read(fid[base * "rational_q"]),
        read(fid[base * "rational_n"])
    end

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
    `:b_theta`, `:b_zeta` (default: `:xi_psi`). `:b_psi` reads the area-normalized
    `psi_area` dataset.
  - `save_path`: If provided, save the figure to this path (default: `nothing`)

### Returns

A `Plots.jl` plot object.
"""
function plot_mode_spectrogram(h5path; component=:xi_psi, save_path=nothing)
    comp_map = Dict(
        :xi_psi  => "xi_psi",
        :b_psi   => "psi_area",  # area-normalized b^ψ
        :b_theta => "b_theta",
        :b_zeta  => "b_zeta",
    )
    haskey(comp_map, component) ||
        error("component must be one of :xi_psi, :b_psi, :b_theta, :b_zeta")

    base = "perturbed_equilibrium/response/"
    dataset_path = base * comp_map[component]

    _has_pe_data(h5path, dataset_path) ||
        return plot(; title="No response data — run with perturbed equilibrium enabled", legend=false)

    data, psi_response, mlow, mhigh, nhigh, q95, rational_psi = h5open(h5path, "r") do fid
        read(fid[dataset_path]),
        read(fid["integration/psi"]),
        read(fid["info/mlow"]), read(fid["info/mhigh"]), read(fid["info/nhigh"]),
        read(fid["equil/q95"]),
        read(fid["perturbed_equilibrium/singular_coupling/rational_psi"])
    end

    mpert = mhigh - mlow + 1
    m_vals = mlow:mhigh
    data_mn = data[:, 1:mpert]  # (npsi, mpert): first mpert columns = first n's modes

    # Top panel: line plot per mode — only label resonant range m ∈ [0, nhigh·q95)
    m_max_legend = nhigh * q95
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
        plot!(p1, psi_response, abs.(data_mn[:, i]);
            label=(show_label ? "m=$m" : nothing), color=cmap[i], linewidth=1.5)
    end

    # Bottom panel: heatmap in (m, ψ_N) space — psi on vertical axis, m on horizontal
    p2 = heatmap(
        collect(m_vals), psi_response, abs.(data_mn);
        xlabel="m",
        ylabel="Norm. Poloidal Flux",
        title="",
        colorbar_title="|$(component)|",
        left_margin=10Plots.mm,
        right_margin=20Plots.mm,
        bottom_margin=5Plots.mm
    )
    # Overlay rational surface locations as white dashed lines
    for psi_s in rational_psi
        hline!(p2, [psi_s]; linestyle=:dash, color=:white, linewidth=1.5, label=nothing)
    end

    p = plot(p1, p2; layout=(2, 1), size=(950, 750))

    isnothing(save_path) || savefig(p, save_path)
    return p
end

"""
    plot_perturbed_equilibrium_summary(h5path; save_path=nothing)

Three-panel composite summary of perturbed equilibrium results:

  - Top-left: Resonant flux per surface (`plot_resonant_flux`)
  - Top-right: Edge |b_ψ| spectrum
  - Bottom: ξ_ψ mode spectrogram (`plot_mode_spectrogram`)

### Arguments

  - `h5path`: Path to a GPEC HDF5 output file with perturbed equilibrium output

### Keyword arguments

  - `save_path`: If provided, save the figure to this path (default: `nothing`)

### Returns

A `Plots.jl` plot object.
"""
function plot_perturbed_equilibrium_summary(h5path; save_path=nothing)
    p_islands  = plot_resonant_flux(h5path)
    p_bpsi     = _plot_bpsi_edge_spectrum(h5path)
    p_spectro  = plot_mode_spectrogram(h5path; component=:xi_psi)

    l = @layout [grid(1, 2){0.35h}; b]
    p = plot(p_islands, p_bpsi, p_spectro; layout=l, size=(1100, 1100))

    isnothing(save_path) || savefig(p, save_path)
    return p
end

# Internal helper — |b_psi(m)| spectrum at the outermost psi surface
function _plot_bpsi_edge_spectrum(h5path)
    base = "perturbed_equilibrium/response/"
    _has_pe_data(h5path, base * "psi_area") ||
        return plot(; title="No b_psi data — run with perturbed equilibrium enabled", legend=false)

    data, mlow, mhigh = h5open(h5path, "r") do fid
        read(fid[base * "psi_area"]),
        read(fid["info/mlow"]), read(fid["info/mhigh"])
    end

    mpert  = mhigh - mlow + 1
    m_vals = mlow:mhigh
    bpsi_edge = abs.(data[end, 1:mpert])

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

end # module PerturbedEquilibrium
