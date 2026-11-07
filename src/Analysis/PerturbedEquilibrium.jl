"""
    PerturbedEquilibrium

Post-processing and visualization functions for GPEC perturbed equilibrium results. Each plot takes the
`PerturbedEquilibriumState` in memory or the path of a `gpec.h5` it was saved to, and throws an `ArgumentError`
naming what to change in the run when the source lacks the data it draws (see `Analysis.available_plots`).
"""
module PerturbedEquilibrium

using HDF5

using ..Analysis: register!, _require, _can, capabilities, REASONS, _draw
import ...PerturbedEquilibrium: PerturbedEquilibriumState, PlasmaResponse
import ..CoilForcing: plot_control_surface

const Source = Union{AbstractString,PerturbedEquilibriumState}

const _COUPLING = "PerturbedEquilibrium/SingularCoupling/"
const _RESPONSE = "PerturbedEquilibrium/Response/"

# The penetrated field when the run carries a nonzero one (matched runs), otherwise `nothing`.
_penetrated(v) = isempty(v) || all(iszero, v) ? nothing : v

function _coupling_data(s::PerturbedEquilibriumState)
    r = s.response
    return (; field=s.resonant_area_weighted_field, penetrated=_penetrated(s.penetrated_area_weighted_field), current=s.resonant_current, island_hw=s.island_half_width,
        chirikov=s.chirikov_parameter,
        delta_prime=s.delta_prime, psi=r.rational_psi, q=r.rational_q, m=r.rational_m_res, n=r.rational_n)
end

function _coupling_data(h5path::AbstractString)
    h5open(h5path, "r") do fid
        rd(name) = read(fid[_COUPLING*name])
        pen = haskey(fid, _COUPLING * "penetrated_area_weighted_field") ? _penetrated(rd("penetrated_area_weighted_field")) : nothing
        (; field=rd("resonant_area_weighted_field"), penetrated=pen, current=rd("resonant_current"), island_hw=rd("island_half_width"), chirikov=rd("chirikov_parameter"),
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

Plot the resonant area-weighted field |b̄_res| = |Φ_res|/A at each rational surface [Pharr 2026], in tesla, beside the
penetrated field |b̄_pen| when the run carries one.

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

Plot the panels the source supplies: the resonant field at each rational surface, the edge b̄^ψ spectrum, the
ξ^ψ spectrogram and, from a state in memory, b_n on the poloidal plane at φ = 0.

### Arguments

  - `src`: A `PerturbedEquilibriumState`, or the path of a `gpec.h5` it was saved to

### Keyword arguments

  - `save_path`: If given, save the figure to this path (default: `nothing`)

### Returns

A Makie `Figure`. Throws an `ArgumentError` when no panel can be drawn.
"""
function plot_perturbed_equilibrium_summary(src::Source; save_path=nothing)
    _require(plot_perturbed_equilibrium_summary, src)
    caps = capabilities(src)
    coupling = _can(plot_resonant_area_weighted_field_amplitude, caps) ? _coupling_data(src) : nothing
    edge = :response_b in caps ? _response_data(src, :b_psi) : nothing
    xi = _can(plot_mode_spectrogram, caps) ? _response_data(src, :xi_psi) : nothing
    rz = src isa PerturbedEquilibriumState && _can(plot_perturbed_field_rz, caps) ? _poloidal_data(src, :b_n) : nothing
    return _draw(_plot_perturbed_equilibrium_summary, coupling, edge, xi, rz; save_path)
end

# (R, Z) of the flux surface ψ at SFL angle θ ∈ [0, 1).
function _rz(equil, psi, theta, hint)
    r = sqrt(max(0.0, equil.rzphi_rsquared((psi, theta); hint)))
    eta = 2π * (theta + equil.rzphi_offset((psi, theta); hint))
    return equil.ro + r * cos(eta), equil.zo + r * sin(eta)
end

# A closed flux-surface outline in (R, Z).
function _outline(equil, psi; ntheta=256)
    hint = (Ref(1), Ref(1))
    pts = [_rz(equil, psi, θ, hint) for θ in range(0, 1; length=ntheta + 1)]
    return first.(pts), last.(pts)
end

const _NORMAL = Dict(:b_n => :b_n_modes, :xi_n => :xi_n_modes)

# The stored normal field or displacement summed over harmonics into real space on the poloidal plane at toroidal angle
# φ, on the flux-surface grid (every `every`-th ψ), with outlines of the rational surfaces and the boundary.
function _poloidal_data(pe::PerturbedEquilibriumState, component; phi=0.0, ntheta=256, every=4)
    haskey(_NORMAL, component) || throw(ArgumentError("component must be :b_n or :xi_n"))
    X = getfield(pe, _NORMAL[component])
    resp = pe.response
    equil = resp.ffs.equil
    rows = 1:every:length(pe.psi_grid)
    θs = range(0, 1; length=ntheta + 1)
    R, Z, V = (zeros(length(rows), length(θs)) for _ in 1:3)
    hint = (Ref(1), Ref(1))
    for (a, i) in enumerate(rows), (b, θ) in enumerate(θs)
        psi = pe.psi_grid[i]
        R[a, b], Z[a, b] = _rz(equil, psi, θ, hint)
        ν = equil.rzphi_nu((psi, θ); hint)
        V[a, b] = real(sum(X[i, k] * cis(2π * resp.m_modes[k] * θ + resp.n_modes[k] * (ν - phi)) for k in axes(X, 2)))
    end
    outlines = [_outline(equil, psi) for psi in vcat(resp.rational_psi, pe.psi_grid[end])]
    return (; R, Z, values=V, outlines, phi)
end

# The boundary in (R, Z) at toroidal angle φ with its normal displacement ξ_n there.
function _boundary_displacement(pe::PerturbedEquilibriumState; phi=0.0, ntheta=720)
    resp = pe.response
    equil = resp.ffs.equil
    psi = pe.psi_grid[end]
    X = pe.xi_n_modes[end, :]
    hint = (Ref(1), Ref(1))
    θs = range(0, 1; length=ntheta + 1)
    pts = [_rz(equil, psi, θ, hint) for θ in θs]
    xi = [real(sum(X[k] * cis(2π * resp.m_modes[k] * θ + resp.n_modes[k] * (equil.rzphi_nu((psi, θ); hint) - phi)) for k in eachindex(X))) for θ in θs]
    return (; R=first.(pts), Z=last.(pts), xi)
end

# Island chains at toroidal angle φ between ψ_s ∓ (w/2)|cos(ξ/2)|, ξ = 2π m θ + n (ν − φ) + α with α the phase of the
# surface's resonant field, in (θ, ψ_N) and as (R, Z) polygons; ψ is clipped to the last flux surface.
function _island_chains(pe::PerturbedEquilibriumState, halfwidth; phi=0.0, ntheta=1440)
    resp = pe.response
    equil = resp.ffs.equil
    psimax = pe.psi_grid[end]
    hint = (Ref(1), Ref(1))
    θs = collect(range(0, 1; length=ntheta + 1))
    chains = NamedTuple[]
    for row in eachindex(resp.rational_psi)
        psis, m, n, w = resp.rational_psi[row], resp.rational_m_res[row], resp.rational_n[row], halfwidth[row]
        w > 0 || continue
        α = angle(pe.resonant_area_weighted_field[row])
        half = [w * abs(cos((2π * m * θ + n * (equil.rzphi_nu((psis, θ); hint) - phi) + α) / 2)) for θ in θs]
        lo, hi = clamp.(psis .- half, 0.0, psimax), clamp.(psis .+ half, 0.0, psimax)
        outer = [_rz(equil, p, θ, hint) for (p, θ) in zip(hi, θs)]
        inner = [_rz(equil, p, θ, hint) for (p, θ) in zip(lo, θs)]
        push!(chains, (; label="$m/$n", psis, w, theta=360 .* θs .- 180, lo, hi, polygon=vcat(outer, reverse(inner))))
    end
    return (; chains, phi, surfaces=[_outline(equil, p * psimax) for p in range(0.1, 1; length=10)])
end

# Drawing functions, defined by the GPECMakieExt extension.
function _plot_perturbed_field_rz end
function _plot_islands end

"""
    plot_perturbed_field_rz(pe; component=:b_n, phi=0.0, deform=false, clip=0.9, save_path=nothing)

Plot the perturbed normal field b_n or normal displacement ξ_n on the poloidal plane at toroidal angle φ, drawn on the
flux-surface grid with the rational surfaces outlined; with `deform`, also the boundary displaced along its normal by
a scaled ξ_n over the equilibrium outline.

### Arguments

  - `pe`: A `PerturbedEquilibriumState` with the stored normal field and displacement

### Keyword arguments

  - `component`: `:b_n` (T, default) or `:xi_n` (m)
  - `phi`: Toroidal angle in radians (default: `0.0`)
  - `deform`: Draw the displaced boundary, scaled so its largest displacement is 5 % of the minor radius (default: `false`)
  - `clip`: Quantile of |value| that sets the symmetric colour range; larger values saturate. `1` shows the full range (default: `0.9`)
  - `save_path`: If given, save the figure to this path (default: `nothing`)

### Returns

A Makie `Figure`.
"""
function plot_perturbed_field_rz(pe::PerturbedEquilibriumState; component=:b_n, phi=0.0, deform=false, clip=0.9, save_path=nothing)
    _require(plot_perturbed_field_rz, pe)
    0 < clip <= 1 || throw(ArgumentError("plot_perturbed_field_rz: clip must be in (0, 1], got $clip"))
    d = _poloidal_data(pe, component; phi)
    boundary = deform ? _boundary_displacement(pe; phi) : nothing
    return _draw(_plot_perturbed_field_rz, d, component, boundary; clip, save_path)
end

"""
    plot_effective_islands(pe; phi=0.0, save_path=nothing)

Plot the island chains the effective resonant flux would open, in (θ, ψ_N) and on the poloidal plane at toroidal angle
φ, with each chain's half-width w/2 (in ψ_N) listed.

### Arguments

  - `pe`: A `PerturbedEquilibriumState` with the singular coupling

### Keyword arguments

  - `phi`: Toroidal angle in radians (default: `0.0`)
  - `save_path`: If given, save the figure to this path (default: `nothing`)

### Returns

A Makie `Figure`.
"""
function plot_effective_islands(pe::PerturbedEquilibriumState; phi=0.0, save_path=nothing)
    _require(plot_effective_islands, pe)
    return _draw(_plot_islands, _island_chains(pe, pe.island_half_width; phi), "Effective islands"; save_path)
end

"""
    plot_islands(pe; phi=0.0, save_path=nothing)

Plot the island chains of the penetrated field in (θ, ψ_N) and on the poloidal plane at toroidal angle φ. The
half-width scales from the effective one as w² ∝ |b̄|: w_pen = w_eff √(|b̄_pen| / |b̄_res|).

### Arguments

  - `pe`: A `PerturbedEquilibriumState` from a matched run, which carries a penetrated field

### Keyword arguments

  - `phi`: Toroidal angle in radians (default: `0.0`)
  - `save_path`: If given, save the figure to this path (default: `nothing`)

### Returns

A Makie `Figure`.
"""
function plot_islands(pe::PerturbedEquilibriumState; phi=0.0, save_path=nothing)
    _require(plot_islands, pe)
    ratio = abs.(pe.penetrated_area_weighted_field) ./ max.(abs.(pe.resonant_area_weighted_field), floatmin())
    return _draw(_plot_islands, _island_chains(pe, pe.island_half_width .* sqrt.(ratio); phi), "Penetrated islands"; save_path)
end

"""
    plot_dominant_mode_surface(src; mode=1, n=nothing, save_path=nothing)

Plot dominant resonant-coupling mode `mode`, the right singular vector of the resonant coupling matrix, as the normal
field it represents on the control surface: toroidal angle φ against poloidal angle θ, in arbitrary units (the vector
has unit norm).

### Arguments

  - `src`: A `PlasmaResponse`, or a `PerturbedEquilibriumState` (its response's modes)

### Keyword arguments

  - `mode`: Which dominant mode, 1 the strongest (default: `1`)
  - `n`: Toroidal mode number drawn (default: the first)
  - `save_path`: If given, save the figure to this path (default: `nothing`)

### Returns

A Makie `Figure`.
"""
function plot_dominant_mode_surface(src::Union{PlasmaResponse,PerturbedEquilibriumState}; mode::Int=1, n=nothing, save_path=nothing)
    _require(plot_dominant_mode_surface, src)
    resp = src isa PlasmaResponse ? src : src.response
    1 <= mode <= length(resp.dominant_singular_values) ||
        throw(ArgumentError("plot_dominant_mode_surface: mode must be 1 to $(length(resp.dominant_singular_values))"))
    σ = resp.dominant_singular_values[mode]
    return plot_control_surface(resp, resp.dominant_right_singular_vectors[:, mode]; n, arb=true, label="dominant mode $mode (σ = $(round(σ; sigdigits=3)))",
        save_path)
end

const _PE_SOURCES = (PerturbedEquilibriumState, AbstractString)
register!(plot_perturbed_equilibrium_summary, _PE_SOURCES, (), "resonant field, edge b̄^ψ spectrum, ξ^ψ spectrogram and b_n on the poloidal plane";
    any_of=((:singular_coupling,), (:response_b,), (:response_xi,), (:response_n,)))
register!(plot_resonant_area_weighted_field, _PE_SOURCES, (:singular_coupling,), "resonant coupling rows per rational surface")
register!(plot_resonant_area_weighted_field_amplitude, _PE_SOURCES, (:singular_coupling,), "|b̄_res| per rational surface")
register!(plot_island_widths, _PE_SOURCES, (:singular_coupling,), "island half-width per rational surface")
register!(plot_chirikov_parameter, _PE_SOURCES, (:singular_coupling,), "effective island-overlap Chirikov parameter per rational surface")
register!(plot_driven_delta_prime, _PE_SOURCES, (:singular_coupling,), "forcing-driven Δ′ per rational surface")
register!(plot_mode_spectrogram, _PE_SOURCES, (:response_xi,), "response harmonics against ψ_N (component = :xi_psi, :b_psi, :Jb_theta, :Jb_zeta)")
register!(plot_dominant_mode_surface, (PlasmaResponse, PerturbedEquilibriumState), (:dominant_mode,),
    "a dominant resonant-coupling mode as b_n on the control surface")
register!(plot_perturbed_field_rz, (PerturbedEquilibriumState,), (:response_n,), "b_n or ξ_n on the poloidal plane (deform = true for the displaced boundary)")
register!(plot_effective_islands, (PerturbedEquilibriumState,), (:singular_coupling,), "island chains from the effective resonant flux")
register!(plot_islands, (PerturbedEquilibriumState,), (:penetrated_field,), "island chains from the penetrated field")

end # module PerturbedEquilibrium
