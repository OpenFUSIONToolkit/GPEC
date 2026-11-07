"""
    Equilibrium

Post-processing and visualization functions for GPEC equilibria. Each plot takes a `PlasmaEquilibrium`, a
`ForceFreeStatesResult` (which adds its rational surfaces as markers), or the path of a `gpec.h5`.
"""
module Equilibrium

using HDF5

using ..Analysis: register!, _require, _draw
import ...Equilibrium: PlasmaEquilibrium
import ...ForceFreeStates: ForceFreeStatesResult

const Source = Union{AbstractString,PlasmaEquilibrium,ForceFreeStatesResult}

# Rational surfaces (ψ_N, q) to mark: none for a bare equilibrium.
_surfaces(::PlasmaEquilibrium) = (; sing_psi=Float64[], sing_q=Float64[])
_surfaces(r::ForceFreeStatesResult) = (; sing_psi=[s.psifac for s in r.surfaces], sing_q=[s.q for s in r.surfaces])
function _surfaces(fid::HDF5.File)
    haskey(fid, "SingularSurfaces/rational_psi") || return _surfaces(nothing)
    return (; sing_psi=read(fid["SingularSurfaces/rational_psi"]), sing_q=read(fid["SingularSurfaces/rational_q"]))
end
_surfaces(::Nothing) = (; sing_psi=Float64[], sing_q=Float64[])

_equil(eq::PlasmaEquilibrium) = eq
_equil(r::ForceFreeStatesResult) = r.equil

function _profile_data(src::Union{PlasmaEquilibrium,ForceFreeStatesResult})
    eq = _equil(src)
    prof = eq.profiles
    return (; psi=prof.xs, q=prof.q_spline.y, mu0p=prof.P_spline.y, twopif=prof.F_spline.y, q0=eq.params.q0, q95=eq.params.q95, _surfaces(src)...)
end

function _profile_data(h5path::AbstractString)
    h5open(h5path, "r") do fid
        (; psi=read(fid["Equilibrium/Profiles/psi"]), q=read(fid["Equilibrium/Profiles/q"]), mu0p=read(fid["Equilibrium/Profiles/mu0p"]),
            twopif=read(fid["Equilibrium/Profiles/2piF"]), q0=read(fid["Equilibrium/q_axis"]), q95=read(fid["Equilibrium/q_95"]), _surfaces(fid)...)
    end
end

function _geometry_data(src::Union{PlasmaEquilibrium,ForceFreeStatesResult})
    eq = _equil(src)
    return (; rcoords=eq.rzphi_rsquared.nodal_derivs.partials[1, :, :], offset=eq.rzphi_offset.nodal_derivs.partials[1, :, :],
        psi=eq.rzphi_xs, theta=eq.rzphi_ys, ro=eq.ro, zo=eq.zo, _surfaces(src)...)
end

function _geometry_data(h5path::AbstractString)
    h5open(h5path, "r") do fid
        (; rcoords=read(fid["Equilibrium/Geometry/rcoords"]), offset=read(fid["Equilibrium/Geometry/offset"]), psi=read(fid["Equilibrium/Geometry/psi"]),
            theta=read(fid["Equilibrium/Geometry/theta"]), ro=read(fid["Equilibrium/R_axis"]), zo=read(fid["Equilibrium/Z_axis"]), _surfaces(fid)...)
    end
end

# R, Z on the full nodal (ψ, θ) grid of the geometry data.
function _rz_grid(d)
    R = Matrix{Float64}(undef, length(d.psi), length(d.theta))
    Z = similar(R)
    for ipsi in eachindex(d.psi), itheta in eachindex(d.theta)
        rfac = sqrt(max(0.0, d.rcoords[ipsi, itheta]))
        eta = 2π * (d.theta[itheta] + d.offset[ipsi, itheta])
        R[ipsi, itheta] = d.ro + rfac * cos(eta)
        Z[ipsi, itheta] = d.zo + rfac * sin(eta)
    end
    return R, Z
end

# Headline scalars; the plasma current is in MA.
function _global_data(src::Union{PlasmaEquilibrium,ForceFreeStatesResult})
    p = _equil(src).params
    return (; q0=p.q0, q95=p.q95, betan=p.betan, kappa=p.kappa, ip=p.crnt, bt=p.bt0)
end

function _global_data(h5path::AbstractString)
    h5open(h5path, "r") do fid
        (; q0=read(fid["Equilibrium/q_axis"]), q95=read(fid["Equilibrium/q_95"]), betan=read(fid["Equilibrium/beta_N"]),
            kappa=read(fid["Equilibrium/kappa"]), ip=read(fid["Equilibrium/I_p"]), bt=read(fid["Equilibrium/B_T_axis"]))
    end
end

# Drawing functions, defined by the GPECMakieExt extension.
function _plot_qprofile end
function _plot_pressure_profile end
function _plot_f_profile end
function _plot_flux_surfaces end
function _plot_equilibrium_summary end
function _plot_gse end

"""
    plot_qprofile(src; show_singular=true, save_path=nothing)

Plot the safety factor q against ψ_N, with q₀ and q₉₅ marked and the rational surfaces labelled along the top.

### Arguments

  - `src`: A `PlasmaEquilibrium`, a `ForceFreeStatesResult`, or the path of a `gpec.h5`

### Keyword arguments

  - `show_singular`: Mark the rational surfaces (default: `true`)
  - `save_path`: If given, save the figure to this path (default: `nothing`)

### Returns

A Makie `Figure`.
"""
function plot_qprofile(src::Source; show_singular=true, save_path=nothing)
    _require(plot_qprofile, src)
    return _draw(_plot_qprofile, _profile_data(src); show_singular, save_path)
end

"""
    plot_pressure_profile(src; save_path=nothing)

Plot μ₀p against ψ_N, with the rational surfaces labelled along the top.

### Arguments

  - `src`: A `PlasmaEquilibrium`, a `ForceFreeStatesResult`, or the path of a `gpec.h5`

### Keyword arguments

  - `save_path`: If given, save the figure to this path (default: `nothing`)

### Returns

A Makie `Figure`.
"""
function plot_pressure_profile(src::Source; save_path=nothing)
    _require(plot_pressure_profile, src)
    return _draw(_plot_pressure_profile, _profile_data(src); save_path)
end

"""
    plot_f_profile(src; save_path=nothing)

Plot 2πF = RB_φ against ψ_N, with the rational surfaces labelled along the top.

### Arguments

  - `src`: A `PlasmaEquilibrium`, a `ForceFreeStatesResult`, or the path of a `gpec.h5`

### Keyword arguments

  - `save_path`: If given, save the figure to this path (default: `nothing`)

### Returns

A Makie `Figure`.
"""
function plot_f_profile(src::Source; save_path=nothing)
    _require(plot_f_profile, src)
    return _draw(_plot_f_profile, _profile_data(src); save_path)
end

"""
    plot_flux_surfaces(src; n_psi=11, n_theta=18, save_path=nothing)

Plot flux surfaces and constant-θ spokes in (R, Z) at equal aspect, with the rational surfaces in colour.

### Arguments

  - `src`: A `PlasmaEquilibrium`, a `ForceFreeStatesResult`, or the path of a `gpec.h5`

### Keyword arguments

  - `n_psi`: Number of flux surfaces drawn (default: 11)
  - `n_theta`: Number of constant-θ spokes drawn (default: 18)
  - `save_path`: If given, save the figure to this path (default: `nothing`)

### Returns

A Makie `Figure`.
"""
function plot_flux_surfaces(src::Source; n_psi=11, n_theta=18, save_path=nothing)
    _require(plot_flux_surfaces, src)
    return _draw(_plot_flux_surfaces, _geometry_data(src); n_psi, n_theta, save_path)
end

"""
    plot_equilibrium_summary(src; save_path=nothing)

Plot the equilibrium's headline scalars, its flux surfaces in (R, Z), and q, μ₀p and 2πF against a shared ψ_N axis.

### Arguments

  - `src`: A `PlasmaEquilibrium`, a `ForceFreeStatesResult`, or the path of a `gpec.h5`

### Keyword arguments

  - `save_path`: If given, save the figure to this path (default: `nothing`)

### Returns

A Makie `Figure`.
"""
function plot_equilibrium_summary(src::Source; save_path=nothing)
    _require(plot_equilibrium_summary, src)
    return _draw(_plot_equilibrium_summary, _global_data(src), _profile_data(src), _geometry_data(src); save_path)
end

"""
    plot_gse(gse; n_theta_lines=8, save_path=nothing)

Plot the Grad-Shafranov residual against ψ_N on several θ slices, with the θ-integrated residual as a thick line.
Not registered: the equilibrium does not store its residual yet.

### Arguments

  - `gse`: The named tuple `Equilibrium.equilibrium_gse!` returns

### Keyword arguments

  - `n_theta_lines`: Number of θ slices drawn (default: 8)
  - `save_path`: If given, save the figure to this path (default: `nothing`)

### Returns

A Makie `Figure`.
"""
plot_gse(gse::NamedTuple; n_theta_lines=8, save_path=nothing) = _draw(_plot_gse, gse; n_theta_lines, save_path)

register!(plot_equilibrium_summary, (PlasmaEquilibrium, ForceFreeStatesResult, AbstractString), (:equilibrium,), "flux surfaces and profiles")
register!(plot_qprofile, (PlasmaEquilibrium, ForceFreeStatesResult, AbstractString), (:equilibrium,), "q against ψ_N with the rational surfaces")
register!(plot_pressure_profile, (PlasmaEquilibrium, ForceFreeStatesResult, AbstractString), (:equilibrium,), "μ₀p against ψ_N")
register!(plot_f_profile, (PlasmaEquilibrium, ForceFreeStatesResult, AbstractString), (:equilibrium,), "2πF against ψ_N")
register!(plot_flux_surfaces, (PlasmaEquilibrium, ForceFreeStatesResult, AbstractString), (:equilibrium,), "flux surfaces and θ spokes in (R, Z)")

end # module Equilibrium
