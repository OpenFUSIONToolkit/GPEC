"""
    FieldLineTracing

Post-processing and visualization for GPEC field-line-tracing results stored in the
`field_line_tracing/` group of a GPEC HDF5 output file: Poincaré puncture plots (magnetic
islands and stochasticity), MAFOT-style connection-length maps, divertor footprints, and
island-width/Chirikov summaries.
"""
module FieldLineTracing

using HDF5
using LaTeXStrings
using Plots

# Check that a field_line_tracing dataset exists and is non-empty.
function _has_flt_data(h5path, key)
    h5open(h5path, "r") do fid
        haskey(fid, key) && !isempty(read(fid[key]))
    end
end

_no_data_plot(msg) = plot(; title=msg, legend=false)

"""
    plot_poincare(h5path; save_path=nothing, plane=nothing)

Poincaré puncture plot in `(R,Z)`. Each launched field line is a distinct colour; island
chains appear as nested loops and stochastic regions as scattered clouds. If `plane` is
given, only punctures at that toroidal section angle [rad] are shown.

Returns a `Plots.jl` object.
"""
function plot_poincare(h5path; save_path=nothing, plane=nothing)
    base = "field_line_tracing/punctures/"
    _has_flt_data(h5path, base * "R") ||
        return _no_data_plot("No field-line-tracing data — enable [FieldLineTracing]")

    R, Z, line, phi = h5open(h5path, "r") do fid
        read(fid[base * "R"]), read(fid[base * "Z"]),
        read(fid[base * "line_id"]), read(fid[base * "phi"])
    end

    keep = plane === nothing ? trues(length(R)) : isapprox.(phi, plane; atol=1e-6)
    p = scatter(; xlabel="R [m]", ylabel="Z [m]", title="Poincaré section",
                legend=false, aspect_ratio=:equal, markersize=0.8, markerstrokewidth=0,
                left_margin=10Plots.mm, bottom_margin=5Plots.mm)
    for lid in unique(line[keep])
        m = keep .& (line .== lid)
        scatter!(p, R[m], Z[m]; markersize=0.8, markerstrokewidth=0)
    end
    save_path !== nothing && savefig(p, save_path)
    return p
end

"""
    plot_poincare_flux(h5path; save_path=nothing)

Poincaré plot in flux coordinates `(θ, ψ_N)` — the natural view of island chains and the
`n·q = m` resonance structure. Returns a `Plots.jl` object.
"""
function plot_poincare_flux(h5path; save_path=nothing)
    base = "field_line_tracing/punctures/"
    _has_flt_data(h5path, base * "psi") ||
        return _no_data_plot("No field-line-tracing data — enable [FieldLineTracing]")

    theta, psi, line = h5open(h5path, "r") do fid
        read(fid[base * "theta"]), read(fid[base * "psi"]), read(fid[base * "line_id"])
    end
    keep = .!isnan.(psi) .& .!isnan.(theta)
    p = scatter(; xlabel=L"\theta", ylabel=L"\psi_N", title="Poincaré section (flux coords)",
                legend=false, markersize=0.8, markerstrokewidth=0,
                left_margin=10Plots.mm, bottom_margin=5Plots.mm)
    for lid in unique(line[keep])
        m = keep .& (line .== lid)
        scatter!(p, theta[m], psi[m]; markersize=0.8, markerstrokewidth=0)
    end
    save_path !== nothing && savefig(p, save_path)
    return p
end

"""
    plot_connection_length(h5path; save_path=nothing)

Connection-length / laminar map: length (toroidal transits or metres) vs launch `ψ_N`,
coloured by topology class (confined / wall-terminated). Returns a `Plots.jl` object.
"""
function plot_connection_length(h5path; save_path=nothing)
    base = "field_line_tracing/connection_length/"
    _has_flt_data(h5path, base * "length") ||
        return _no_data_plot("No connection-length data")

    psi, len, cls = h5open(h5path, "r") do fid
        read(fid[base * "psi"]), read(fid[base * "length"]), read(fid[base * "class"])
    end
    p = scatter(psi, len; group=cls, xlabel=L"\psi_N", ylabel="connection length",
                title="Connection length / laminar map", legend=:outertopright,
                markerstrokewidth=0, left_margin=10Plots.mm, bottom_margin=5Plots.mm)
    save_path !== nothing && savefig(p, save_path)
    return p
end

"""
    plot_footprints(h5path; save_path=nothing)

Divertor/wall strike-point footprints in `(R,Z)`. Returns a `Plots.jl` object.
"""
function plot_footprints(h5path; save_path=nothing)
    base = "field_line_tracing/footprints/"
    _has_flt_data(h5path, base * "R") ||
        return _no_data_plot("No footprint data (real-space tracing with a wall)")

    R, Z, phi = h5open(h5path, "r") do fid
        read(fid[base * "R"]), read(fid[base * "Z"]), read(fid[base * "phi"])
    end
    p = scatter(R, Z; zcolor=phi, xlabel="R [m]", ylabel="Z [m]",
                title="Wall footprints", colorbar_title="φ [rad]", legend=false,
                markerstrokewidth=0, aspect_ratio=:equal,
                left_margin=10Plots.mm, bottom_margin=5Plots.mm)
    save_path !== nothing && savefig(p, save_path)
    return p
end

"""
    plot_island_widths(h5path; save_path=nothing)

Island half-width and Chirikov overlap vs rational-surface `ψ_N`. Returns a `Plots.jl` object.
"""
function plot_island_widths(h5path; save_path=nothing)
    base = "field_line_tracing/islands/"
    _has_flt_data(h5path, base * "psi") ||
        return _no_data_plot("No island data")

    psi, w, chi, m = h5open(h5path, "r") do fid
        read(fid[base * "psi"]), read(fid[base * "half_width"]),
        read(fid[base * "chirikov"]), read(fid[base * "m"])
    end
    wmax = isempty(w) ? 1.0 : maximum(w)
    p = bar(psi, w; xlabel=L"\psi_N", ylabel="island half-width [ψ_N]", bar_width=0.008,
            title="Island widths & Chirikov overlap", label="half-width", color=:steelblue,
            ylims=(0, 1.3 * wmax), left_margin=10Plots.mm, bottom_margin=5Plots.mm)
    for i in eachindex(psi)
        annotate!(p, psi[i], w[i], text("m=$(m[i])", 7, :bottom))
    end
    chi_finite = replace(chi, NaN => 0.0)
    pr = twinx(p)
    plot!(pr, psi, chi_finite; seriestype=:sticks, marker=:circle, color=:red,
          label="Chirikov σ", ylabel="Chirikov σ", ylims=(0, 1.2), legend=:topright)
    hline!(pr, [1.0]; color=:red, linestyle=:dash, label="σ=1 (overlap)")
    save_path !== nothing && savefig(p, save_path)
    return p
end

"""
    plot_field_line_tracing_summary(h5path; save_path=nothing)

Composite of the Poincaré section, flux-coordinate Poincaré, connection length, and island
widths. Returns a `Plots.jl` object.
"""
function plot_field_line_tracing_summary(h5path; save_path=nothing)
    p1 = plot_poincare(h5path)
    p2 = plot_poincare_flux(h5path)
    p3 = plot_connection_length(h5path)
    p4 = plot_island_widths(h5path)
    p = plot(p1, p2, p3, p4; layout=(2, 2), size=(1200, 900))
    save_path !== nothing && savefig(p, save_path)
    return p
end

end # module FieldLineTracing
