# plot_3d.jl — 3D visualisations of synthetic Hall-probe fields with the
# coil geometry overlaid. Uses GLMakie for all 3D plots (interactive
# rotation, true 3D arrows, dense scatter that renders as solid planes).
#
# Generates a handful of PNGs into plots/, then walks through the figures
# in interactive GLMakie windows; press ENTER to advance, or q+ENTER to
# quit early.

include(joinpath(@__DIR__, "synth_hall_cyl.jl"))

using GLMakie
GLMakie.activate!()

using Printf
using Statistics

# ───────────────────────────────────────────────────────────────────────────
# Coil-geometry helpers
# ───────────────────────────────────────────────────────────────────────────

"""
    coil_xyz(coil_file; pose=(0,0,0,0,0), current_A=100.0, stride=20)
        -> (xs, ys, zs)

Read `coil_file`, apply the 5- or 6-DOF rigid pose, and return the
(possibly sub-sampled) segment-endpoint coordinates as three flat
`Vector{Float64}` suitable for `lines!`.
"""
function coil_xyz(coil_file::AbstractString;
        pose=(0.0, 0.0, 0.0, 0.0, 0.0), current_A=1000.0, stride::Int=20)
    cs = read_coil_dat(coil_file)
    cs.currents .= current_A
    coils = [cs]
    tz = length(pose) >= 6 ? pose[6] : 0.0
    place_coil!(coils, pose[1], pose[2], pose[3];
                tx_deg=pose[4], ty_deg=pose[5], tz_deg=tz)
    cs2 = coils[1]
    xs = Float64[]; ys = Float64[]; zs = Float64[]
    @inbounds for j in 1:cs2.ncoil, k in 1:cs2.s, l in 1:stride:cs2.nsec
        push!(xs, cs2.x[j, k, l])
        push!(ys, cs2.y[j, k, l])
        push!(zs, cs2.z[j, k, l])
    end
    return xs, ys, zs
end

function plot_coil_only!(ax, coil_file;
        pose=(0.0, 0.0, 0.0, 0.0, 0.0), current_A=100.0,
        color=:black, linewidth=2)
    xs, ys, zs = coil_xyz(coil_file; pose=pose, current_A=current_A)
    lines!(ax, xs, ys, zs; color=color, linewidth=linewidth)
end

# ───────────────────────────────────────────────────────────────────────────
# Shared helpers
# ───────────────────────────────────────────────────────────────────────────

function _xy_from_cyl(hall)
    x = hall.R .* cos.(hall.phi)
    y = hall.R .* sin.(hall.phi)
    return x, y
end

function _component_data(hall, component::Symbol)
    if component == :Bmag
        return sqrt.(hall.B_R.^2 .+ hall.B_phi.^2 .+ hall.B_Z.^2), "|B| [T]"
    elseif component == :B_R
        return hall.B_R, "B_R [T]"
    elseif component == :B_Z
        return hall.B_Z, "B_Z [T]"
    elseif component == :B_phi
        return hall.B_phi, "B_phi [T]"
    else
        error("unknown component $component")
    end
end

# Zero-is-white colormap: signed components get diverging blue-white-red;
# |B| gets sequential white-to-blue. `clip_quantile` ∈ (0, 1] caps the
# color range at the requested quantile of |data|, saturating outliers and
# letting the bulk of the data show in stronger mid-range tones rather
# than washed-out near-white. `clip_quantile=1.0` reproduces the full range.
function _cmap_clims(component::Symbol, data; clip_quantile::Real=1.0)
    abs_data = abs.(data)
    vmax = clip_quantile >= 1 ? maximum(abs_data) :
                                quantile(abs_data, clip_quantile)
    vmax = max(vmax, eps())  # avoid degenerate clims when all data ≈ 0
    if component == :Bmag
        return :Blues, (0.0, vmax)
    else
        return Reverse(:RdBu), (-vmax, vmax)
    end
end

function _setup_axis(title::AbstractString; figsize=(900, 720))
    fig = Figure(size=figsize)
    ax = Axis3(fig[1, 1]; xlabel="x [m]", ylabel="y [m]", zlabel="z [m]",
               title=title, aspect=:data,
               azimuth=π/5, elevation=π/8)
    return fig, ax
end

# ───────────────────────────────────────────────────────────────────────────
# Plot makers
# ───────────────────────────────────────────────────────────────────────────

"""
    plot_probes_colored(hall; component, coil_file, pose, current_A, title, outpath)

3D scatter (GLMakie) of probes coloured by the requested field component.
Coil geometry overlaid. `component` ∈ `(:Bmag, :B_R, :B_phi, :B_Z)`.
"""
function plot_probes_colored(hall;
        component::Symbol=:Bmag, coil_file::AbstractString,
        pose=(0.0, 0.0, 0.0, 0.0, 0.0), current_A=100.0,
        clip_quantile::Real=0.95,
        title::AbstractString="", outpath::AbstractString="")
    data, clabel = _component_data(hall, component)
    cmap, clims = _cmap_clims(component, data; clip_quantile=clip_quantile)
    x, y = _xy_from_cyl(hall)
    ttl = isempty(title) ? "Probes coloured by $clabel" : title
    fig, ax = _setup_axis(ttl)
    plot_coil_only!(ax, coil_file; pose=pose, current_A=current_A)
    sc = scatter!(ax, x, y, hall.Z;
                  color=data, colormap=cmap, colorrange=clims,
                  markersize=12)
    Colorbar(fig[1, 2], sc; label=clabel)
    if !isempty(outpath)
        mkpath(dirname(abspath(outpath)))
        save(outpath, fig); println("  wrote ", outpath)
    end
    return fig
end

"""
    plot_field_slices(coil_file; pose, current_A,
                      phi_values, R_min, R_max, nR, Z_min, Z_max, nZ,
                      component, title, outpath)

Dense 3D scatter (GLMakie) on flat (R, Z) slices at the requested toroidal
angles — appears as solid intersecting "blades" in 3D. Coil overlaid.
"""
function plot_field_slices(coil_file::AbstractString;
        pose=(0.0, 0.0, 0.0, 0.0, 0.0), current_A=100.0,
        phi_values::AbstractVector=collect((0.0, π/2, π, 3π/2)),
        R_min::Real=0.05, R_max::Real=1.5, nR::Int=80,
        Z_min::Real=1.6, Z_max::Real=3.1, nZ::Int=100,
        component::Symbol=:Bmag,
        clip_quantile::Real=0.95,
        title::AbstractString="", outpath::AbstractString="")
    R_grid = collect(range(R_min, R_max; length=nR))
    Z_grid = collect(range(Z_min, Z_max; length=nZ))
    R_all = Float64[]; phi_all = Float64[]; Z_all = Float64[]
    for ph in phi_values, r in R_grid, z in Z_grid
        push!(R_all, r); push!(phi_all, ph); push!(Z_all, z)
    end
    tz = length(pose) >= 6 ? pose[6] : 0.0
    hall = synth_hall_cyl(coil_file,
        pose[1], pose[2], pose[3];
        tx_deg=pose[4], ty_deg=pose[5], tz_deg=tz,
        current_A=current_A,
        R_probes=R_all, phi_probes=phi_all, Z_probes=Z_all)

    data, clabel = _component_data(hall, component)
    cmap, clims = _cmap_clims(component, data; clip_quantile=clip_quantile)
    x = R_all .* cos.(phi_all); y = R_all .* sin.(phi_all); z = Z_all

    phi_label = join((@sprintf("%.0f°", rad2deg(ph)) for ph in phi_values), ", ")
    ttl = isempty(title) ?
        "Field slice at φ = $(phi_label), component $clabel" : title
    fig, ax = _setup_axis(ttl)
    plot_coil_only!(ax, coil_file; pose=pose, current_A=current_A)
    sc = scatter!(ax, x, y, z;
                  color=data, colormap=cmap, colorrange=clims,
                  markersize=6)
    Colorbar(fig[1, 2], sc; label=clabel)
    if !isempty(outpath)
        mkpath(dirname(abspath(outpath)))
        save(outpath, fig); println("  wrote ", outpath)
    end
    return fig
end

"""
    plot_field_vectors(hall; coil_file, pose, current_A, scale, outpath)

True 3D arrows for the B-field at every probe (GLMakie `arrows!`).
Arrow length and colour scale with `|B|`. Coil geometry overlaid.
"""
function plot_field_vectors(hall;
        coil_file::AbstractString,
        pose=(0.0, 0.0, 0.0, 0.0, 0.0), current_A=100.0,
        scale::Real=0.05, clip_quantile::Real=0.95,
        outpath::AbstractString="")
    x, y = _xy_from_cyl(hall); z = hall.Z
    Bx = hall.B_R .* cos.(hall.phi) .- hall.B_phi .* sin.(hall.phi)
    By = hall.B_R .* sin.(hall.phi) .+ hall.B_phi .* cos.(hall.phi)
    Bz = hall.B_Z
    Bmag = sqrt.(Bx.^2 .+ By.^2 .+ Bz.^2)
    Bmax_full = maximum(Bmag)            # arrow length scale (true max)
    Bmax_cap = clip_quantile >= 1 ? Bmax_full : quantile(Bmag, clip_quantile)
    Bmax_cap = max(Bmax_cap, eps())
    fig, ax = _setup_axis("B-field vectors at probes (length ∝ |B|)")
    plot_coil_only!(ax, coil_file; pose=pose, current_A=current_A)
    ps   = [Point3f(x[i], y[i], z[i])    for i in eachindex(x)]
    dirs = [Vec3f(Bx[i], By[i], Bz[i])   for i in eachindex(x)]
    arrows!(ax, ps, dirs;
            lengthscale=scale / Bmax_full,
            arrowsize=Vec3f(0.008, 0.008, 0.014),
            linewidth=0.005,
            color=Bmag, colormap=:Blues, colorrange=(0.0, Bmax_cap))
    Colorbar(fig[1, 2]; colormap=:Blues, colorrange=(0.0, Bmax_cap),
             label="|B| [T]")
    if !isempty(outpath)
        mkpath(dirname(abspath(outpath)))
        save(outpath, fig); println("  wrote ", outpath)
    end
    return fig
end

# ───────────────────────────────────────────────────────────────────────────
# Interactive walk
# ───────────────────────────────────────────────────────────────────────────

function _wait_or_skip(fig, name)
    display(fig)
    print("    showing ", name, " — press ENTER to continue (or q+ENTER to quit) > ")
    flush(stdout)
    line = strip(readline())
    return lowercase(line) == "q"
end

# ───────────────────────────────────────────────────────────────────────────
# Demo
# ───────────────────────────────────────────────────────────────────────────

if abspath(PROGRAM_FILE) == @__FILE__
    coil_file = joinpath(@__DIR__, "..", "examples", "Br_3D_example", "sparc_pf1u.dat")
    plot_dir  = joinpath(@__DIR__, "plots")
    mkpath(plot_dir)

    pose = (0.005, -0.002, 0.001, 0.30, -0.10, 0.0)
    current_A = 100.0

    # Toggle plot categories. Default: only toroidal slices.
    show_probes   = false   # 4 dual-shell scatter plots (|B|, B_R, B_Z, B_phi)
    show_vectors  = false   # 3D arrows quiver
    show_slices   = true    # 4 toroidal slice "blade" plots
    # Color saturation: 1.0 → full range (weak mid-tones),
    # lower → stronger / brighter colors by saturating outliers.
    clip_quantile = 0.95

    plots_to_show = Tuple{String,Any}[]

    # The dual-shell scatter / vector plots need the dual-shell Hall data.
    if show_probes || show_vectors
        R, phi, Z = dual_shell_grid(0.50, 0.80, 2.31, 0.6, 0.4;
            nphi_inner=24, nz_inner=14, nphi_outer=16, nz_outer=8)
        hall = synth_hall_cyl(coil_file,
            pose[1], pose[2], pose[3];
            tx_deg=pose[4], ty_deg=pose[5], tz_deg=pose[6],
            current_A=current_A,
            R_probes=R, phi_probes=phi, Z_probes=Z)
    end

    if show_probes
        push!(plots_to_show, ("|B| on dual-shell probes",
            plot_probes_colored(hall; component=:Bmag, coil_file=coil_file, pose=pose,
                current_A=current_A, clip_quantile=clip_quantile,
                title="Coil + probes coloured by |B|",
                outpath=joinpath(plot_dir, "3d_probes_Bmag.png"))))
        push!(plots_to_show, ("B_R on dual-shell probes",
            plot_probes_colored(hall; component=:B_R, coil_file=coil_file, pose=pose,
                current_A=current_A, clip_quantile=clip_quantile,
                title="Coil + probes coloured by B_R",
                outpath=joinpath(plot_dir, "3d_probes_BR.png"))))
        push!(plots_to_show, ("B_Z on dual-shell probes",
            plot_probes_colored(hall; component=:B_Z, coil_file=coil_file, pose=pose,
                current_A=current_A, clip_quantile=clip_quantile,
                title="Coil + probes coloured by B_Z",
                outpath=joinpath(plot_dir, "3d_probes_BZ.png"))))
        push!(plots_to_show, ("B_phi on dual-shell probes",
            plot_probes_colored(hall; component=:B_phi, coil_file=coil_file, pose=pose,
                current_A=current_A, clip_quantile=clip_quantile,
                title="Coil + probes coloured by B_phi (small for near-axisymmetric coil)",
                outpath=joinpath(plot_dir, "3d_probes_Bphi.png"))))
    end

    if show_vectors
        push!(plots_to_show, ("B-field vectors at probes",
            plot_field_vectors(hall; coil_file=coil_file, pose=pose,
                current_A=current_A, scale=0.10, clip_quantile=clip_quantile,
                outpath=joinpath(plot_dir, "3d_field_vectors.png"))))
    end

    if show_slices
        push!(plots_to_show, ("Toroidal |B| slices",
            plot_field_slices(coil_file; pose=pose, current_A=current_A,
                component=:Bmag, clip_quantile=clip_quantile,
                title="Toroidal slices at φ=0,π/2,π,3π/2 — |B|",
                outpath=joinpath(plot_dir, "3d_slices_Bmag.png"))))
        push!(plots_to_show, ("Toroidal B_R slices",
            plot_field_slices(coil_file; pose=pose, current_A=current_A,
                component=:B_R, clip_quantile=clip_quantile,
                title="Toroidal slices at φ=0,π/2,π,3π/2 — B_R",
                outpath=joinpath(plot_dir, "3d_slices_BR.png"))))
        push!(plots_to_show, ("Toroidal B_Z slices",
            plot_field_slices(coil_file; pose=pose, current_A=current_A,
                component=:B_Z, clip_quantile=clip_quantile,
                title="Toroidal slices at φ=0,π/2,π,3π/2 — B_Z",
                outpath=joinpath(plot_dir, "3d_slices_BZ.png"))))
        push!(plots_to_show, ("Eight-blade toroidal |B|",
            plot_field_slices(coil_file; pose=pose, current_A=current_A,
                phi_values=collect(range(0, 2π; length=9)[1:end-1]),
                component=:Bmag, clip_quantile=clip_quantile,
                title="8 toroidal slices — |B| (highlights near-axisymmetry)",
                outpath=joinpath(plot_dir, "3d_slices_eight_blades.png"))))
    end

    println("\nWalking through ", length(plots_to_show),
            " GLMakie figures — press ENTER to advance.")
    for (name, fig) in plots_to_show
        if _wait_or_skip(fig, name); println("    user quit"); break; end
    end

    println("\nDone. Plots in: ", plot_dir)
end
