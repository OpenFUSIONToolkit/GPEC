"""
    CoilForcing

Visualization functions for coil geometry and the resulting normal field
perturbations on the plasma boundary.

Two 3D entry points cover the common cases: [`plot_coil_geometry_3d`](@ref) draws the coils
alone, and [`plot_surface_3d`](@ref) draws the plasma boundary surface — optionally coloured by
any control-surface field (via [`control_surface_scalar`](@ref)) and/or with coils overlaid —
the Julia analog of GPEC's `plot_control_3d`. The `:RdBu` colour scale is shared, so a blue
(positive-current) coil and a blue (positive) field read identically.

All functions return `Plots.jl` plot objects and accept an optional `save_path`
keyword argument to write the figure to disk.
"""
module CoilForcing

using Plots
import ...Utilities.FourierTransforms: FourierTransform, inverse

"""
    plot_coil_geometry_3d(coil_sets; save_path=nothing, kwargs...)

Plot the 3D wire geometry of all coil sets.

Each coil set is shown with a distinct color; individual conductors are plotted
as lines in Cartesian (X, Y, Z) space in meters.

### Arguments

  - `coil_sets`: `Vector{CoilSet}` from `ForcingTerms.load_coil_sets`

### Keyword arguments

  - `save_path`: file path to save the figure (default: `nothing`)
  - Any extra kwargs are forwarded to `Plots.plot`

### Returns

A `Plots.jl` plot object.
"""
function plot_coil_geometry_3d(coil_sets; save_path=nothing, kwargs...)
    clim = _current_clim(coil_sets)

    p = plot(; xlabel="X [m]", ylabel="Y [m]", zlabel="Z [m]",
        title="Coil geometry (3D)", legend=false,
        camera=(30, 20), kwargs...)

    _overlay_coils!(p, coil_sets; clim=clim)

    # Invisible scatter spanning the current range to force the colorbar to appear
    scatter!(p, [NaN], [NaN], [NaN];
        zcolor=[0.0], clims=(-clim, clim), color=:RdBu,
        colorbar_title="I [A]", markersize=0)

    isnothing(save_path) || savefig(p, save_path)
    return p
end

# Symmetric current colour-limit (centred at zero) for a set of coils.
function _current_clim(coil_sets)
    all_currents = [I for cs in coil_sets for I in cs.currents]
    return isempty(all_currents) ? 1.0 : max(maximum(abs, all_currents), eps())
end

"""
    _overlay_coils!(p, coil_sets; clim=nothing)

Draw each conductor of every coil set onto plot `p` as a 3D `path3d` curve, coloured on a
symmetric `:RdBu` scale by its current (positive = blue, negative = red — the same mapping
used for surface fields, so a blue coil and a blue field read identically).
"""
function _overlay_coils!(p, coil_sets; clim=nothing)
    cl = isnothing(clim) ? _current_clim(coil_sets) : clim
    for cs in coil_sets
        for j in 1:cs.ncoil
            t = clamp((cs.currents[j] + cl) / (2cl), 0.0, 1.0)
            col = get(cgrad(:RdBu), t)
            for k in 1:cs.s
                plot!(p, view(cs.x, j, k, :), view(cs.y, j, k, :), view(cs.z, j, k, :);
                    seriestype=:path3d, color=col, lw=2, linestyle=:solid, label="")
            end
        end
    end
    return p
end

"""
    plot_coil_geometry_rz(coil_sets; equil=nothing, psi=nothing, save_path=nothing, kwargs...)

Plot coil cross-sections in the (R, Z) meridional plane.

Each coil strand is shown as a closed curve in `(R, Z) = (√(X²+Y²), Z)` space.
If `equil` is provided, the plasma boundary is overplotted for reference.

### Arguments

  - `coil_sets`: `Vector{CoilSet}` from `ForcingTerms.load_coil_sets`

### Keyword arguments

  - `equil`: optional `PlasmaEquilibrium`; if given, overlays the plasma boundary
  - `psi`: flux surface to trace (default: `equil.rzphi_xs[end]`, the outermost grid point)
  - `save_path`: file path to save the figure (default: `nothing`)
  - Any extra kwargs are forwarded to `Plots.plot`

### Returns

A `Plots.jl` plot object.
"""
function plot_coil_geometry_rz(coil_sets; equil=nothing, psi=nothing, save_path=nothing, kwargs...)
    p = plot(; xlabel="R [m]", ylabel="Z [m]",
        title="Coil cross-sections (R, Z)", legend=:topright,
        aspect_ratio=:equal, kwargs...)

    colors = [:blue, :red, :green, :orange, :purple, :brown, :cyan, :magenta]
    for (ci, cs) in enumerate(coil_sets)
        col = colors[mod1(ci, length(colors))]
        for j in 1:cs.ncoil
            for k in 1:cs.s
                R_strand = sqrt.(view(cs.x, j, k, :) .^ 2 .+ view(cs.y, j, k, :) .^ 2)
                Z_strand = view(cs.z, j, k, :)
                label = (j == 1 && k == 1) ? cs.name : ""
                plot!(p, R_strand, Z_strand; label=label, color=col, lw=1.5)
            end
        end
    end

    if !isnothing(equil)
        psi_bnd = isnothing(psi) ? equil.rzphi_xs[end] : Float64(psi)
        mtheta_boundary = length(equil.rzphi_ys)
        hint2d = (Ref(1), Ref(1))
        R_bnd = zeros(mtheta_boundary)
        Z_bnd = zeros(mtheta_boundary)
        for (i, θ_sfl) in enumerate(equil.rzphi_ys)
            r_minor = sqrt(equil.rzphi_rsquared((psi_bnd, θ_sfl); hint=hint2d))
            θ_cyl = 2π * (θ_sfl + equil.rzphi_offset((psi_bnd, θ_sfl); hint=hint2d))
            R_bnd[i] = equil.ro + r_minor * cos(θ_cyl)
            Z_bnd[i] = equil.zo + r_minor * sin(θ_cyl)
        end
        plot!(p, [R_bnd; R_bnd[1]], [Z_bnd; Z_bnd[1]];
            label="plasma boundary (ψ=$(round(psi_bnd; digits=3)))", color=:black, lw=2)
    end

    isnothing(save_path) || savefig(p, save_path)
    return p
end

"""
    plot_bn_contour(bn, mtheta, nzeta; n=nothing, save_path=nothing, kwargs...)

Plot the normal magnetic field `bn(θ, ζ)` on the plasma boundary as a 2D heatmap.

Expects the raw Biot-Savart B·n̂ field in Tesla (before `project_normal_flux!`
converts to flux). If plotting post-projection data, note units are T·m² not T.

### Arguments

  - `bn`: normal field matrix `[mtheta, nzeta]` — typically B·n̂ in Tesla
  - `mtheta`: number of poloidal grid points
  - `nzeta`: number of toroidal grid points

### Keyword arguments

  - `n`: toroidal mode number (used for axis label only)
  - `save_path`: file path to save the figure (default: `nothing`)
  - Any extra kwargs are forwarded to `Plots.heatmap`

### Returns

A `Plots.jl` plot object.
"""
function plot_bn_contour(bn::Matrix{Float64}, mtheta::Int, nzeta::Int;
    n=nothing, save_path=nothing, kwargs...)
    # Shift rows so θ=0 (outboard midplane) is centred on the y-axis at 0°
    bn_wrapped = circshift(bn, (mtheta ÷ 2, 0))
    theta_deg = range(-180; step=360.0 / mtheta, length=mtheta)
    zeta_deg = range(0; step=360.0 / nzeta, length=nzeta)

    n_label = isnothing(n) ? "" : " (n=$n)"
    p = heatmap(zeta_deg, theta_deg, bn_wrapped;
        xlabel="zeta [deg]", ylabel="theta [deg]",
        ylims=(-180, 180),
        title="Normal field bₙ(θ, ζ)$n_label [T]",
        color=:RdBu, kwargs...)

    isnothing(save_path) || savefig(p, save_path)
    return p
end

"""
    plot_mode_spectrum(forcing_modes; mlow=nothing, mhigh=nothing, save_path=nothing, kwargs...)

Plot the Fourier mode spectrum `|bmn|` vs poloidal mode number m as a step line, grouped by n.

### Arguments

  - `forcing_modes`: `Vector{ForcingMode}` from `compute_coil_forcing_modes!`

### Keyword arguments

  - `mlow`, `mhigh`: poloidal mode range for the x-axis (e.g. from ForceFreeStates config);
    if provided, sets xlims to span the full stability range even if some modes are zero
  - `save_path`: file path to save the figure (default: `nothing`)
  - Any extra kwargs are forwarded to `Plots.plot`

### Returns

A `Plots.jl` plot object.
"""
function plot_mode_spectrum(forcing_modes; mlow=nothing, mhigh=nothing, save_path=nothing, kwargs...)
    n_vals = sort(unique(m.n for m in forcing_modes))
    colors = [:blue, :red, :green, :orange, :purple, :brown]

    p = plot(; xlabel="Poloidal mode m", ylabel="|Φ_x(m,n)| [unit-norm]",
        title="Coil forcing mode spectrum",
        legend=:topright, kwargs...)

    for (ni, n) in enumerate(n_vals)
        modes_n = filter(m -> m.n == n, forcing_modes)
        sort!(modes_n; by=m -> m.m)
        m_vals = [m.m for m in modes_n]
        amps = abs.([m.amplitude for m in modes_n])
        col = colors[mod1(ni, length(colors))]
        m_ext = [m_vals[1] - 1; m_vals; m_vals[end] + 1]
        amp_ext = [0.0; amps; 0.0]
        plot!(p, m_ext, amp_ext; label="n=$n", color=col, lw=2, seriestype=:steppre)
    end

    ylims!(p, (0, Inf))
    if !isnothing(mlow) && !isnothing(mhigh)
        # zero endpoints are at mlow-1 and mhigh+1; add ±1 so they're visible with whitespace
        xlims!(p, mlow - 2, mhigh + 2)
    end

    isnothing(save_path) || savefig(p, save_path)
    return p
end

# Helicity = sign(Bt)·sign(Ip); the SFL→machine toroidal-angle handedness.
function _equil_helicity(equil)
    bt = isnothing(equil.params.bt_sign) ? 1 : Int(sign(equil.params.bt_sign))
    ip = isnothing(equil.params.crnt) ? 1 : Int(sign(equil.params.crnt))
    return bt * ip
end

# Unpack `modes` (a Vector of ForcingMode-like objects filtered to toroidal mode `n`,
# or a `(m_vals, amplitudes)` tuple) into parallel m and complex-amplitude vectors.
function _unpack_modes(modes, n::Int)
    if modes isa Tuple
        return collect(Int, modes[1]), collect(ComplexF64, modes[2])
    end
    sel = [md for md in modes if md.n == n]
    isempty(sel) && error("control_surface_scalar: no modes found with n=$n")
    return Int[md.m for md in sel], ComplexF64[md.amplitude for md in sel]
end

"""
    control_surface_scalar(equil, modes, n; mtheta=180, nphi=180, psi=nothing,
                           helicity=nothing) -> Matrix{Float64}

Inverse-Fourier-transform a spectral control-surface quantity into a real-space `[mtheta, nphi]`
field on the plasma boundary, ready to colour [`plot_surface_3d`](@ref).

`modes` are complex Fourier amplitudes indexed by poloidal mode `m` for a single toroidal mode
`n` — a `Vector{ForcingMode}` (filtered to `n`) or a `(m_vals, amplitudes)` tuple. The quantity is
arbitrary (e.g. applied normal field `b_n_x`, total `b_n`, displacement `xi_n`); the caller
supplies whichever modes they want to view.

The transform inverts the SFL Fourier decomposition used by `ForcingTerms.fourier_decompose_bn`:
`f(θ) = Σ_m amp_m e^{i m θ}`, the per-θ correction `e^{i n ν(ψ,θ)}` from `equil.rzphi_nu`, then the
toroidal extension in the **machine angle** `c(θ,φ) = -Re[f(θ) e^{i n·hel·φ}]`. The `e^{i n·hel·φ}`
follows from the SFL→machine map `φ = -hel(2π ζ + ν)` (`ζ` the SFL toroidal angle the modes were
decomposed against); substituting `2π ζ = -hel·φ - ν` is what turns the decomposition's `ζ`
exponential into the `ν` correction times `e^{i n·hel·φ}`.

!!! note "Sign / helicity convention"

    The reconstruction tracks the source coil **toroidally** (the field lobe sits at the coil's φ),
    carried by `helicity = sign(Bt)·sign(Ip)` in `e^{i n·hel·φ}` (`equil.rzphi_nu`/`equil.params`).
    The leading `-` is a display orientation so that **a positive (blue) coil current reads as a
    positive (blue) `b_n_x` lobe beneath it** — GPEC's outward-normal flux is intrinsically negative
    there, so the overall sign is the blue-coil↔blue-surface convention while the spatial structure
    and magnitude are physical. Pass `helicity` to override `sign(Bt)·sign(Ip)` if the toroidal
    tracking is wrong for your machine.

!!! note "Band-limited reconstruction"

    The field is reconstructed only from the supplied `m` modes, so it is band-limited to that
    poloidal range. A source placed within ~one grid cell of the surface makes the field
    near-singular and the truncated series rings (Gibbs oscillation — e.g. a spurious dip where a
    single lobe is expected). Keep coils at a realistic standoff from the boundary, or widen the
    `m` range, for a clean reconstruction.

### Keyword arguments

  - `mtheta`, `nphi`: poloidal/toroidal grid resolution
  - `psi`: flux surface (default `equil.rzphi_xs[end]`, the boundary)
  - `helicity`: override `sign(Bt)·sign(Ip)`
"""
function control_surface_scalar(equil, modes, n::Int; mtheta::Int=180, nphi::Int=180,
    psi=nothing, helicity=nothing)
    psi_b = isnothing(psi) ? equil.rzphi_xs[end] : Float64(psi)
    m_vals, amps = _unpack_modes(modes, n)
    m_low, m_high = extrema(m_vals)
    mpert = m_high - m_low + 1

    vec = zeros(ComplexF64, mpert)
    for (m, a) in zip(m_vals, amps)
        vec[m-m_low+1] = a
    end

    # m→θ inverse transform; f[i] is at normalized θ = (i-1)/mtheta
    ft = FourierTransform(mtheta, mpert, m_low)
    f = inverse(ft, vec)

    # SFL→machine toroidal-angle correction e^{i n ν(ψ,θ)}. The modes are decomposed on the SFL
    # grid against e^{-i(m θ - n ζ)}, with the machine angle φ = -hel(2π ζ + ν) (see CoilFourier
    # `sample_boundary_grid`). Substituting 2π ζ = -hel φ - ν into the inverse turns the ζ
    # exponential into e^{i n ν(θ)} e^{i n hel φ}: the per-θ ν enters the field, hel sets the
    # toroidal handedness, and the real field reconstructs faithfully — so a positive (blue) coil
    # current yields positive (blue) b_n_x directly beneath the coil with no extra sign flip.
    hel = isnothing(helicity) ? _equil_helicity(equil) : helicity
    hint = (Ref(1), Ref(1))
    for i in 1:mtheta
        θ = (i - 1) / mtheta
        ν = equil.rzphi_nu((psi_b, θ); hint=hint)
        f[i] *= exp(im * n * ν)
    end

    # θ→φ extension in the machine angle: c(θ,φ) = -Re[ f(θ) e^{i n hel φ} ]. The reconstruction
    # tracks the source coil toroidally (the lobe sits at the coil's φ); the leading − is a display
    # orientation that makes a positive (blue) coil current read as a positive (blue) b_n_x lobe
    # beneath it. GPEC's outward-normal flux is intrinsically negative there, so this overall sign
    # is the blue-coil↔blue-surface convention — the spatial structure and magnitude are physical.
    # It is a constant flip (not helicity-dependent): the normal-projection sign is set by geometry,
    # while hel only carries the toroidal handedness, already in the exponential.
    c = zeros(Float64, mtheta, nphi)
    for k in 1:nphi
        ph = exp(im * n * hel * (k - 1) * 2π / nphi)
        for i in 1:mtheta
            c[i, k] = -real(f[i] * ph)
        end
    end
    return c
end

# Build the closed (θ,φ)→(x,y,z) boundary surface from the equilibrium splines.
# Returns X, Y, Z each [mtheta+1, nphi+1] (wrapped so the torus closes seamlessly).
function _boundary_surface_xyz(equil, mtheta::Int, nphi::Int, psi)
    psi_b = isnothing(psi) ? equil.rzphi_xs[end] : Float64(psi)
    hint = (Ref(1), Ref(1))
    R = zeros(mtheta)
    Z = zeros(mtheta)
    for i in 1:mtheta
        θ = (i - 1) / mtheta
        r_minor = sqrt(equil.rzphi_rsquared((psi_b, θ); hint=hint))
        θ_cyl = 2π * (θ + equil.rzphi_offset((psi_b, θ); hint=hint))
        R[i] = equil.ro + r_minor * cos(θ_cyl)
        Z[i] = equil.zo + r_minor * sin(θ_cyl)
    end

    X = zeros(mtheta + 1, nphi + 1)
    Y = zeros(mtheta + 1, nphi + 1)
    Zs = zeros(mtheta + 1, nphi + 1)
    for k in 1:(nphi+1)
        φ = (k - 1) * 2π / nphi
        cφ, sφ = cos(φ), sin(φ)
        for i in 1:(mtheta+1)
            ii = i <= mtheta ? i : 1
            X[i, k] = R[ii] * cφ
            Y[i, k] = R[ii] * sφ
            Zs[i, k] = Z[ii]
        end
    end
    return X, Y, Zs
end

# Wrap a [mtheta, nphi] field to [mtheta+1, nphi+1] to match the closed surface mesh.
_wrap_field(c) = [c[i <= size(c, 1) ? i : 1, k <= size(c, 2) ? k : 1]
                  for i in 1:(size(c, 1)+1), k in 1:(size(c, 2)+1)]

"""
    plot_surface_3d(equil; color_by=nothing, coil_sets=nothing, mtheta=180, nphi=180,
                    psi=nothing, clim=nothing, colorbar_title="", title="Plasma boundary",
                    camera=(35, 25), save_path=nothing, kwargs...)

Plot the 3D plasma boundary surface, optionally coloured by a control-surface field and/or with
coils overlaid — the Julia analog of GPEC's `plot_control_3d`.

!!! note "Backend for coloured surfaces"

    Scalar colouring of the closed surface requires a PlotlyJS-style backend (`plotlyjs()`,
    already a GPEC dependency); the default GR backend cannot colour a parametric surface by an
    independent array and will draw an **uncoloured** surface with a warning. Call `plotlyjs()`
    before requesting `color_by`. Coil overlay and uncoloured surfaces work on any backend.

### Keyword arguments

  - `color_by`: optional real-space `[mtheta, nphi]` scalar (any control-surface output — e.g. from
    [`control_surface_scalar`](@ref)). Coloured on a symmetric `:RdBu` scale centred at zero
    (positive = blue), matching the coil-current colours. `nothing` draws an uncoloured surface.
  - `coil_sets`: optional `Vector{CoilSet}` overlaid as 3D curves (see [`plot_coil_geometry_3d`](@ref)).
  - `mtheta`, `nphi`, `psi`: surface grid resolution and flux surface (must match `color_by`).
  - `clim`: symmetric colour limit (default `maximum(abs, color_by)`).
  - `save_path`: file path to save the figure; extra `kwargs` forward to `Plots.surface`.

### Returns

A `Plots.jl` plot object.

    plot_surface_3d(equil, modes, n; coil_sets=nothing, colorbar_title="", kwargs...)

Convenience method: colour the surface by [`control_surface_scalar(equil, modes, n; …)`](@ref).
"""
function plot_surface_3d(equil; color_by=nothing, coil_sets=nothing, mtheta::Int=180,
    nphi::Int=180, psi=nothing, clim=nothing, colorbar_title="",
    title="Plasma boundary", camera=(35, 25), save_path=nothing, kwargs...)
    X, Y, Z = _boundary_surface_xyz(equil, mtheta, nphi, psi)

    common = (; xlabel="X [m]", ylabel="Y [m]", zlabel="Z [m]", title=title,
        legend=false, aspect_ratio=:equal, camera=camera)

    want_color = !isnothing(color_by)
    if want_color
        size(color_by) == (mtheta, nphi) ||
            error("color_by must be [mtheta, nphi] = ($mtheta, $nphi), got $(size(color_by))")
        # Only PlotlyJS-style backends honour `fill_z` on a closed/parametric surface; GR silently
        # colours by Z instead, so warn and degrade rather than show a misleading figure.
        if !(backend_name() in (:plotlyjs, :pythonplot, :pyplot))
            @warn "plot_surface_3d: the active Plots backend ($(backend_name())) cannot colour a " *
                  "parametric 3D surface by a scalar — call `plotlyjs()` first for field-coloured " *
                  "surfaces. Drawing an uncoloured surface." maxlog = 1
            want_color = false
        end
    end

    if want_color
        cl = isnothing(clim) ? max(maximum(abs, color_by), eps()) : clim
        cw = _wrap_field(color_by)
        p = surface(X, Y, Z; fill_z=cw, color=:RdBu, clims=(-cl, cl),
            colorbar_title=colorbar_title, common..., kwargs...)
    else
        p = surface(X, Y, Z; color=:grays, alpha=0.5, colorbar=false, common..., kwargs...)
    end

    isnothing(coil_sets) || _overlay_coils!(p, coil_sets)
    isnothing(save_path) || savefig(p, save_path)
    return p
end

function plot_surface_3d(equil, modes, n::Int; coil_sets=nothing, mtheta::Int=180,
    nphi::Int=180, psi=nothing, helicity=nothing,
    colorbar_title="", save_path=nothing, kwargs...)
    c = control_surface_scalar(equil, modes, n; mtheta=mtheta, nphi=nphi, psi=psi, helicity=helicity)
    return plot_surface_3d(equil; color_by=c, coil_sets=coil_sets, mtheta=mtheta, nphi=nphi,
        psi=psi, colorbar_title=colorbar_title, save_path=save_path, kwargs...)
end

export plot_coil_geometry_3d, plot_coil_geometry_rz
export plot_bn_contour, plot_mode_spectrum
export plot_surface_3d, control_surface_scalar

end # module CoilForcing
