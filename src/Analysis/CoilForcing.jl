"""
    CoilForcing

Coil geometry and fields on the plasma control surface: the coils in 3D and in (R, Z), the forcing spectrum, normal
field maps in (φ, θ), and the 3D boundary coloured by any control-surface field. Coils are coloured on the same
diverging map as the normal field they drive, so a positive current reads as positive b_n beneath the coil.
"""
module CoilForcing

import ...Utilities.FourierTransforms: FourierTransform, inverse
import ...PerturbedEquilibrium: PerturbedEquilibriumState, PlasmaResponse
import ...ForcingTerms as FT
using ..Analysis: register!, _require, _draw

# Helicity = sign(Bt)·sign(Ip); the SFL→machine toroidal-angle handedness.
_equil_helicity(equil) = equil.params.bt_sign * equil.params.ip_sign

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
    The leading `-` is a display orientation so that **a positive coil current reads as a
    positive `b_n_x` lobe beneath it** — GPEC's outward-normal flux is intrinsically negative
    there, so the overall sign is the coil↔surface colour convention while the spatial structure
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

    # SFL→machine toroidal-angle correction e^{i n ν(ψ,θ)}: substituting 2π ζ = -hel φ - ν into the inverse turns the
    # ζ exponential into e^{i n ν(θ)} e^{i n hel φ}.
    hel = isnothing(helicity) ? _equil_helicity(equil) : helicity
    hint = (Ref(1), Ref(1))
    for i in 1:mtheta
        θ = (i - 1) / mtheta
        ν = equil.rzphi_nu((psi_b, θ); hint=hint)
        f[i] *= exp(im * n * ν)
    end

    # θ→φ extension in the machine angle; the leading − makes a positive coil current read as positive b_n beneath it.
    c = zeros(Float64, mtheta, nphi)
    for k in 1:nphi
        ph = exp(im * n * hel * (k - 1) * 2π / nphi)
        for i in 1:mtheta
            c[i, k] = -real(f[i] * ph)
        end
    end
    return c
end

# The boundary (or flux surface `psi`) in (R, Z) at `mtheta` SFL angles θ = (i-1)/mtheta.
function _surface_rz(equil, mtheta::Int, psi)
    psi_b = isnothing(psi) ? equil.rzphi_xs[end] : Float64(psi)
    hint = (Ref(1), Ref(1))
    R, Z = zeros(mtheta), zeros(mtheta)
    for i in 1:mtheta
        θ = (i - 1) / mtheta
        r = sqrt(max(0.0, equil.rzphi_rsquared((psi_b, θ); hint=hint)))
        η = 2π * (θ + equil.rzphi_offset((psi_b, θ); hint=hint))
        R[i] = equil.ro + r * cos(η)
        Z[i] = equil.zo + r * sin(η)
    end
    return R, Z
end

# Build the closed (θ,φ)→(x,y,z) boundary surface from the equilibrium splines.
# Returns X, Y, Z each [mtheta+1, nphi+1] (wrapped so the torus closes seamlessly).
function _boundary_surface_xyz(equil, mtheta::Int, nphi::Int, psi)
    R, Z = _surface_rz(equil, mtheta, psi)
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
_wrap_field(c) = [c[i <= size(c, 1) ? i : 1, k <= size(c, 2) ? k : 1] for i in 1:(size(c, 1)+1), k in 1:(size(c, 2)+1)]

# A real control-surface field with θ shifted so the outboard midplane is at 0: φ, θ in degrees and the field [θ, φ].
function _surface_map(c)
    mtheta, nphi = size(c)
    θ = collect(range(-180, 180; length=mtheta + 1)[1:mtheta])
    φ = collect(range(0, 360; length=nphi + 1)[1:nphi])
    return (; phi=φ, theta=θ, field=circshift(c, (mtheta ÷ 2, 0)))
end

# Each conductor of `coil_sets` projected onto the control-surface map: machine φ = atan(y, x), and the boundary θ (SFL,
# outboard midplane at 0) whose point has the same geometric poloidal angle about the magnetic axis. Returns
# (φ, θ, current) per conductor, in degrees.
function _coils_on_surface(equil, coil_sets; ntheta=2048)
    psi_b = equil.rzphi_xs[end]
    hint = (Ref(1), Ref(1))
    θs = range(0, 1; length=ntheta + 1)[1:ntheta]
    geometric = [mod(2π * (θ + equil.rzphi_offset((psi_b, θ); hint=hint)), 2π) for θ in θs]
    order = sortperm(geometric)
    sorted = geometric[order]
    sfl(η) = θs[order[clamp(searchsortedfirst(sorted, mod(η, 2π)), 1, ntheta)]]
    lines = Tuple{Vector{Float64},Vector{Float64},Float64}[]
    for cs in coil_sets, j in 1:cs.ncoil, k in 1:cs.s
        x, y, z = cs.x[j, k, :], cs.y[j, k, :], cs.z[j, k, :]
        φ = mod.(rad2deg.(atan.(y, x)), 360)
        θ = [mod(360 * sfl(atan(zz - equil.zo, hypot(xx, yy) - equil.ro)) + 180, 360) - 180 for (xx, yy, zz) in zip(x, y, z)]
        push!(lines, (φ, θ, cs.currents[j]))
    end
    return lines
end

const _WEIGHTINGS = (:root_area, :area, :bare)

# A control-surface spectrum in any weighting, converted to the bare normal field b (T): b = (S·√A)⁻¹·b̃, b̃ = S⁻¹·b̄.
function _to_bare(resp, v, weighting)
    weighting in _WEIGHTINGS || throw(ArgumentError("weighting must be :root_area, :area or :bare"))
    S, A = resp.rootarea_to_area_weight, resp.surface_area
    btilde = weighting === :root_area ? v : weighting === :area ? S \ v : (S .* sqrt(A)) * v
    return (S .* sqrt(A)) \ btilde
end

# The bare normal field of spectrum `v` on the control surface for toroidal mode `n`, mapped to (φ, θ), with the
# conductors of `coil_sets` projected onto it when given.
function _control_surface_data(resp::PlasmaResponse, v; weighting=:root_area, n=nothing, coil_sets=nothing, mtheta=256, nphi=180)
    equil = resp.ffs.equil
    nn = isnothing(n) ? first(resp.n_modes) : n
    sel = resp.n_modes .== nn
    any(sel) || throw(ArgumentError("plot_control_surface: no harmonics with n = $nn"))
    b = _to_bare(resp, v, weighting)
    c = control_surface_scalar(equil, (resp.m_modes[sel], b[sel]), nn; mtheta, nphi)
    return (; _surface_map(c)..., n=nn, coils=coil_sets === nothing ? nothing : _coils_on_surface(equil, coil_sets))
end

# Drawing functions, defined by the GPECMakieExt extension.
function _plot_coils_3d end
function _plot_coils_rz end
function _plot_surface_map end
function _plot_forcing_spectrum end

"""
    plot_coil_geometry_3d(coil_sets; equil=nothing, save_path=nothing)
    plot_coil_geometry_3d(pe; save_path=nothing)

Plot the coil conductors in 3D at equal data aspect, coloured by current, with the plasma boundary as an opaque surface
when an equilibrium is given. Use GLMakie to rotate it.

### Arguments

  - `coil_sets`: `Vector{CoilSet}` from `ForcingTerms.load_coil_sets`
  - `pe`: A `PerturbedEquilibriumState` forced by coils, drawn with its boundary

### Keyword arguments

  - `equil`: A `PlasmaEquilibrium` whose boundary is drawn (default: `nothing`)
  - `save_path`: If given, save the figure to this path (default: `nothing`)

### Returns

A Makie `Figure`.
"""
function plot_coil_geometry_3d(coil_sets; equil=nothing, save_path=nothing)
    surface = isnothing(equil) ? nothing : _boundary_surface_xyz(equil, 120, 160, nothing)
    return _draw(_plot_coils_3d, coil_sets, surface, nothing; save_path)
end

function plot_coil_geometry_3d(pe::PerturbedEquilibriumState; save_path=nothing)
    _require(plot_coil_geometry_3d, pe)
    return plot_coil_geometry_3d(pe.coil_sets; equil=pe.response.ffs.equil, save_path)
end

# The coil sets of every coil term of `field`, each set's currents scaled by its term's weight (by the weight's magnitude
# when it is complex, which shifts the toroidal phase), with the number of terms that have no geometry.
function _field_coil_sets(field::FT.ForcingField; n=1)
    sets, skipped, complex_weight = FT.CoilSet[], 0, false
    for t in FT._field_terms(field)
        if !(t isa FT.ForcingSource && t.ctrl.forcing_data_format == "coil")
            skipped += 1
            continue
        end
        complex_weight |= !isreal(t.scale)
        w = isreal(t.scale) ? real(t.scale) : abs(t.scale)
        for cs in FT.load_coil_sets(FT.CoilConfig(t.ctrl), n)
            push!(sets, FT.CoilSet(cs.name, cs.ncoil, cs.s, cs.nw, cs.nsec, cs.x, cs.y, cs.z, w .* cs.currents))
        end
    end
    return (; sets, skipped, complex_weight)
end

"""
    plot_sources(field; n=1, color=:current, save_path=nothing)

Plot the coils of a `ForcingField` in 3D at equal data aspect, with no plasma: every coil term of a sum together, each
term's currents scaled by its weight. Mode-table and spectrum terms have no geometry and are counted in the title. Use
GLMakie to rotate it.

### Arguments

  - `field`: A `ForcingField` with at least one coil term

### Keyword arguments

  - `n`: Toroidal mode number that sets the n-modulated tilts (default: `1`)
  - `color`: `:current` colours each conductor by its current in kA, `:set` gives each coil set its own colour (default: `:current`)
  - `save_path`: If given, save the figure to this path (default: `nothing`)

### Returns

A Makie `Figure`.
"""
function plot_sources(field::FT.ForcingField; n::Int=1, color::Symbol=:current, save_path=nothing)
    _require(plot_sources, field)
    color in (:current, :set) || throw(ArgumentError("plot_sources: color must be :current or :set"))
    d = _field_coil_sets(field; n)
    notes = String[]
    d.skipped > 0 && push!(notes, "$(d.skipped) mode-table term$(d.skipped == 1 ? "" : "s") not drawn")
    d.complex_weight && push!(notes, "complex weights drawn at their magnitude")
    nset, ncond = length(d.sets), sum(cs.ncoil for cs in d.sets)
    title = "Coil sources: $nset set$(nset == 1 ? "" : "s"), $ncond conductor$(ncond == 1 ? "" : "s")" * (isempty(notes) ? "" : " ($(join(notes, "; ")))")
    return _draw(_plot_coils_3d, d.sets, nothing, nothing; title, by_set=color === :set, save_path)
end

"""
    plot_coil_geometry_rz(coil_sets; equil=nothing, psi=nothing, save_path=nothing)
    plot_coil_geometry_rz(pe; save_path=nothing)

Plot the coils in the (R, Z) plane, each set labelled beside itself, over the plasma boundary when an equilibrium is
given. A single toroidal loop is a point in (R, Z) and is drawn as a marker.

### Arguments

  - `coil_sets`: `Vector{CoilSet}` from `ForcingTerms.load_coil_sets`
  - `pe`: A `PerturbedEquilibriumState` forced by coils, drawn with its boundary

### Keyword arguments

  - `equil`: A `PlasmaEquilibrium` whose boundary is drawn (default: `nothing`)
  - `psi`: The flux surface drawn as the boundary (default: the outermost grid surface)
  - `save_path`: If given, save the figure to this path (default: `nothing`)

### Returns

A Makie `Figure`.
"""
function plot_coil_geometry_rz(coil_sets; equil=nothing, psi=nothing, save_path=nothing)
    boundary = isnothing(equil) ? nothing : _surface_rz(equil, 256, psi)
    return _draw(_plot_coils_rz, coil_sets, boundary; save_path)
end

function plot_coil_geometry_rz(pe::PerturbedEquilibriumState; save_path=nothing)
    _require(plot_coil_geometry_rz, pe)
    return plot_coil_geometry_rz(pe.coil_sets; equil=pe.response.ffs.equil, save_path)
end

"""
    plot_mode_spectrum(forcing_modes; mlow=nothing, mhigh=nothing, rational_m=Int[], save_path=nothing)
    plot_mode_spectrum(pe; save_path=nothing)

Plot the forcing spectrum |Φ_x(m, n)| against poloidal harmonic m as step lines, one per toroidal mode number.

### Arguments

  - `forcing_modes`: `Vector{ForcingMode}` from `compute_coil_forcing_modes!`
  - `pe`: A `PerturbedEquilibriumState`, drawn across its poloidal range with its resonant harmonics marked

### Keyword arguments

  - `mlow`, `mhigh`: Poloidal range of the x axis (default: the modes' own range)
  - `rational_m`: Resonant harmonics to mark (default: none)
  - `save_path`: If given, save the figure to this path (default: `nothing`)

### Returns

A Makie `Figure`.
"""
function plot_mode_spectrum(forcing_modes; mlow=nothing, mhigh=nothing, rational_m=Int[], save_path=nothing)
    return _draw(_plot_forcing_spectrum, forcing_modes, mlow, mhigh, rational_m; save_path)
end

function plot_mode_spectrum(pe::PerturbedEquilibriumState; save_path=nothing)
    _require(plot_mode_spectrum, pe)
    ffs = pe.response.ffs
    return plot_mode_spectrum(pe.forcing_modes; mlow=ffs.mlow, mhigh=ffs.mhigh, rational_m=pe.response.rational_m_res, save_path)
end

"""
    plot_control_surface(src, v; weighting=:root_area, n=nothing, label="the given spectrum", arb=false, coils=false,
                         save_path=nothing)

Plot any control-surface spectrum, such as a forcing, a response or an eigenvector, as the bare normal field b_n on the
control surface: toroidal angle φ against poloidal angle θ (0 at the outboard midplane).

### Arguments

  - `src`: A `PlasmaResponse` supplying the control-surface operators and geometry, or a `PerturbedEquilibriumState`,
    whose coils can also be drawn
  - `v`: The spectrum, one entry per control-surface harmonic

### Keyword arguments

  - `weighting`: How `v` is stored: `:root_area` (b̃, default), `:area` (b̄) or `:bare` (b)
  - `n`: Toroidal mode number drawn (default: the first)
  - `label`: The source named in the title, e.g. `"the applied forcing"`
  - `arb`: Label the colour scale in arbitrary units, for a normalized vector (default: `false`, tesla)
  - `coils`: Draw the coil conductors projected onto the map (default: `false`)
  - `save_path`: If given, save the figure to this path (default: `nothing`)

### Returns

A Makie `Figure`.
"""
function plot_control_surface(resp::PlasmaResponse, v::AbstractVector; weighting=:root_area, n=nothing, label="the given spectrum", arb=false,
    coils=false, save_path=nothing)
    coils && throw(ArgumentError("plot_control_surface: coils = true needs the coil-forced PerturbedEquilibriumState, not its PlasmaResponse"))
    d = _control_surface_data(resp, v; weighting, n)
    return _draw(_plot_surface_map, d; title="Normal field on the control surface from $label", arb, save_path)
end

function plot_control_surface(pe::PerturbedEquilibriumState, v::AbstractVector; weighting=:root_area, n=nothing, label="the given spectrum",
    arb=false, coils=false, save_path=nothing)
    coils && isempty(pe.coil_sets) && throw(ArgumentError("plot_control_surface: coils = true needs a coil-forced run"))
    d = _control_surface_data(pe.response, v; weighting, n, coil_sets=coils ? pe.coil_sets : nothing)
    return _draw(_plot_surface_map, d; title="Normal field on the control surface from $label", arb, save_path)
end


"""
    plot_bn_contour(pe; coils=false, save_path=nothing)
    plot_bn_contour(bn, mtheta, nzeta; n=nothing, save_path=nothing)

Plot the applied normal field b_n on the control surface, toroidal angle φ against poloidal angle θ (0 at the outboard
midplane), optionally with the coil conductors projected onto it; or a raw Biot–Savart B·n̂ grid in tesla.

### Arguments

  - `pe`: A `PerturbedEquilibriumState` with a forcing
  - `bn`: Normal field `[mtheta, nzeta]` on the boundary grid

### Keyword arguments

  - `coils`: Draw the coil conductors projected onto the map (default: `false`)
  - `n`: Toroidal mode number named in the title (default: `nothing`)
  - `save_path`: If given, save the figure to this path (default: `nothing`)

### Returns

A Makie `Figure`.
"""
function plot_bn_contour(pe::PerturbedEquilibriumState; coils=false, save_path=nothing)
    _require(plot_bn_contour, pe)
    return plot_control_surface(pe, pe.forcing_b; weighting=:bare, label="the applied forcing", coils, save_path)
end

function plot_bn_contour(bn::Matrix{Float64}, mtheta::Int, nzeta::Int; n=nothing, save_path=nothing)
    size(bn) == (mtheta, nzeta) || throw(ArgumentError("bn must be [mtheta, nzeta] = ($mtheta, $nzeta), got $(size(bn))"))
    d = (; _surface_map(bn)..., n, coils=nothing)
    title = "Normal field B·n̂ on the boundary" * (isnothing(n) ? "" : ", n = $n")
    return _draw(_plot_surface_map, d; title, arb=false, save_path)
end

"""
    plot_surface_3d(equil; color_by=nothing, coil_sets=nothing, mtheta=180, nphi=180, psi=nothing, clim=nothing,
                    label="", title="Plasma boundary", save_path=nothing)
    plot_surface_3d(equil, modes, n; coil_sets=nothing, mtheta=180, nphi=180, psi=nothing, helicity=nothing,
                    label="", save_path=nothing)

Plot the 3D plasma boundary at equal data aspect, unlit and coloured by a control-surface field on a diverging map,
with coils coloured by current on the same map. Use GLMakie to rotate it. The second form colours the surface by
[`control_surface_scalar`](@ref).

### Keyword arguments

  - `color_by`: A real `[mtheta, nphi]` field on the surface (default: `nothing`, an uncoloured surface)
  - `coil_sets`: Coils drawn with the surface (default: `nothing`)
  - `mtheta`, `nphi`, `psi`: Surface grid and flux surface; must match `color_by`
  - `clim`: Symmetric colour limit (default: the field's largest magnitude)
  - `label`: Colour-bar label of the field
  - `save_path`: If given, save the figure to this path (default: `nothing`)

### Returns

A Makie `Figure`.
"""
function plot_surface_3d(equil; color_by=nothing, coil_sets=nothing, mtheta::Int=180, nphi::Int=180, psi=nothing, clim=nothing, label="",
    title="Plasma boundary", save_path=nothing)
    isnothing(color_by) || size(color_by) == (mtheta, nphi) ||
        throw(ArgumentError("color_by must be [mtheta, nphi] = ($mtheta, $nphi), got $(size(color_by))"))
    surface = _boundary_surface_xyz(equil, mtheta, nphi, psi)
    field = isnothing(color_by) ? nothing : (; values=_wrap_field(color_by), clim, label)
    return _draw(_plot_coils_3d, isnothing(coil_sets) ? [] : coil_sets, surface, field; title, save_path)
end

function plot_surface_3d(equil, modes, n::Int; coil_sets=nothing, mtheta::Int=180, nphi::Int=180, psi=nothing, helicity=nothing, label="",
    save_path=nothing)
    c = control_surface_scalar(equil, modes, n; mtheta, nphi, psi, helicity)
    return plot_surface_3d(equil; color_by=c, coil_sets, mtheta, nphi, psi, label, save_path)
end

register!(plot_bn_contour, (PerturbedEquilibriumState,), (:forcing_modes,), "applied b_n on the control surface (coils = true for the coils)")
register!(plot_coil_geometry_3d, (PerturbedEquilibriumState,), (:coil_sets,), "3D coils and plasma boundary")
register!(plot_coil_geometry_rz, (PerturbedEquilibriumState,), (:coil_sets,), "coils over the plasma boundary in (R, Z)")
register!(plot_mode_spectrum, (PerturbedEquilibriumState,), (:forcing_modes,), "|Φ_x(m, n)| forcing spectrum")
register!(plot_sources, (FT.ForcingField,), (:coil_source,), "the coil sources in 3D, coloured by current or by set")

export plot_coil_geometry_3d, plot_coil_geometry_rz
export plot_bn_contour, plot_mode_spectrum, plot_control_surface, plot_sources
export plot_surface_3d, control_surface_scalar

end # module CoilForcing
