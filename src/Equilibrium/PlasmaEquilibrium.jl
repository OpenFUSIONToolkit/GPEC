"""
    ProfileSplines

Named 1D cubic spline interpolants for equilibrium profiles.
Each profile is stored as a separate spline for code clarity.

# Fields

  - `xs::Vector{Float64}`: Shared x-axis (normalized psi)
  - `F_spline`: 2π*F (toroidal flux function, where F = R * B_toroidal)
  - `P_spline`: μ₀*P (plasma pressure × μ₀)
  - `dVdpsi_spline`: dV/dψ (volume derivative)
  - `q_spline`: q (safety factor)

# Derivative Interpolants (for continuous derivative evaluation)

  - `F_deriv`, `P_deriv`, `dVdpsi_deriv`, `q_deriv`: First derivative interpolants

# Notes

  - Node values at grid points: Access via `spline.y[i]`
  - Derivative at any point: Call `deriv(x)` (derivative views are callable)
  - Grid: Access via `xs` field or `spline.cache.x`
"""
struct ProfileSplines{S,D}
    xs::Vector{Float64}
    npts::Int          # length(xs), avoids redundant length() calls
    npts_minus_1::Int  # npts - 1, for hint at last interval
    # Value interpolants
    F_spline::S
    P_spline::S
    dVdpsi_spline::S
    q_spline::S
    # Derivative views (callable, share data with value interpolants)
    F_deriv::D
    P_deriv::D
    dVdpsi_deriv::D
    q_deriv::D
end

"""
    ProfileSplines(xs, F_vals, P_vals, dVdpsi_vals, q_vals; extrap=ExtendExtrap())

Create ProfileSplines from arrays of profile values.
Uses CubicFit boundary conditions with extension extrapolation.
"""
function ProfileSplines(xs::Vector{Float64},
    F_vals::Vector{Float64},
    P_vals::Vector{Float64},
    dVdpsi_vals::Vector{Float64},
    q_vals::Vector{Float64};
    extrap::AbstractExtrap=ExtendExtrap())
    npts = length(xs)
    npts_minus_1 = npts - 1
    @assert length(F_vals) == npts
    @assert length(P_vals) == npts
    @assert length(dVdpsi_vals) == npts
    @assert length(q_vals) == npts

    # Create value interpolants with CubicFit BC (default) for sequential psi access
    F_spline = cubic_interp(xs, F_vals; extrap=extrap)
    P_spline = cubic_interp(xs, P_vals; extrap=extrap)
    dVdpsi_spline = cubic_interp(xs, dVdpsi_vals; extrap=extrap)
    q_spline = cubic_interp(xs, q_vals; extrap=extrap)

    # Create derivative views (these share data with value interpolants, no extra storage)
    F_deriv = deriv1(F_spline)
    P_deriv = deriv1(P_spline)
    dVdpsi_deriv = deriv1(dVdpsi_spline)
    q_deriv = deriv1(q_spline)

    ProfileSplines{typeof(F_spline),typeof(F_deriv)}(
        xs, npts, npts_minus_1,
        F_spline, P_spline, dVdpsi_spline, q_spline,
        F_deriv, P_deriv, dVdpsi_deriv, q_deriv
    )
end

"""
    GeometryProfileSplines

Named 1D cubic spline interpolants for flux-surface-averaged geometric quantities.
Built once during equilibrium construction so they are available to every downstream
module without per-caller recomputation.

# Fields

  - `xs::Vector{Float64}`: Shared ψ axis (matches `rzphi_xs`)
  - `area_spline`: Flux-surface area ∫ dA(ψ) [m²]
  - `avg_r_spline`: Surface-average minor radius ⟨r⟩(ψ) [m]
  - `avg_R_spline`: Surface-average major radius ⟨R⟩(ψ) [m]
"""
struct GeometryProfileSplines{S}
    xs::Vector{Float64}
    npts::Int
    npts_minus_1::Int
    area_spline::S
    avg_r_spline::S
    avg_R_spline::S
end

"""
    GeometryProfileSplines(xs, area_vals, avg_r_vals, avg_R_vals; extrap=ExtendExtrap())

Construct `GeometryProfileSplines` from arrays of surface-averaged values defined
on the shared ψ grid `xs`. Uses CubicFit boundary conditions to match `ProfileSplines`.
"""
function GeometryProfileSplines(xs::Vector{Float64},
    area_vals::Vector{Float64},
    avg_r_vals::Vector{Float64},
    avg_R_vals::Vector{Float64};
    extrap::AbstractExtrap=ExtendExtrap())
    npts = length(xs)
    @assert length(area_vals) == npts
    @assert length(avg_r_vals) == npts
    @assert length(avg_R_vals) == npts

    area_spline = cubic_interp(xs, area_vals; extrap=extrap)
    avg_r_spline = cubic_interp(xs, avg_r_vals; extrap=extrap)
    avg_R_spline = cubic_interp(xs, avg_R_vals; extrap=extrap)

    GeometryProfileSplines{typeof(area_spline)}(
        xs, npts, npts - 1,
        area_spline, avg_r_spline, avg_R_spline
    )
end

"""
    KineticProfileData

Raw kinetic-profile columns read from a kinetic file (HDF5 or ASCII), before
resampling and spline construction. `psi` (normalized poloidal flux) is always
present; every profile field is `nothing` when the source omits it, so each
consumer validates only the fields it needs. Units: densities m⁻³,
temperatures eV, frequencies rad/s, diffusivities m²/s.

| field               | meaning                                                                                                           |
|:------------------- |:----------------------------------------------------------------------------------------------------------------- |
| `n_i`               | main-ion density                                                                                                  |
| `n_e`               | electron density                                                                                                  |
| `T_i`/`T_e`         | ion / electron temperature                                                                                        |
| `omega_E`           | ExB rotation                                                                                                      |
| `omega_tor`         | toroidal rotation (optional)                                                                                      |
| `chi_e`             | perpendicular heat diffusivity χ⊥ (shipped data: electron power-balance χ_e; see the stability docs)              |
| `chi_phi`           | toroidal momentum diffusivity χ_φ                                                                                 |
| `zeff`              | effective charge Z_eff(ψ) supplied directly (optional); the resistive-layer analysis prefers it to its closure    |
| `species_densities` | named per-species densities (e.g. `"n_D"`, `"n_T"`) for explicit multi-ion input; `nothing` for ASCII / no extras |
| `provenance`        | short string recording the source file/format                                                                     |
"""
struct KineticProfileData
    psi::Vector{Float64}
    n_i::Union{Nothing,Vector{Float64}}
    n_e::Union{Nothing,Vector{Float64}}
    T_i::Union{Nothing,Vector{Float64}}
    T_e::Union{Nothing,Vector{Float64}}
    omega_E::Union{Nothing,Vector{Float64}}
    omega_tor::Union{Nothing,Vector{Float64}}
    chi_e::Union{Nothing,Vector{Float64}}
    chi_phi::Union{Nothing,Vector{Float64}}
    zeff::Union{Nothing,Vector{Float64}}
    species_densities::Union{Nothing,Dict{String,Vector{Float64}}}
    provenance::String
end

function KineticProfileData(; psi, n_i=nothing, n_e=nothing, T_i=nothing, T_e=nothing,
    omega_E=nothing, omega_tor=nothing, chi_e=nothing, chi_phi=nothing, zeff=nothing,
    species_densities=nothing, provenance="")
    _v(x) = x === nothing ? nothing : Float64.(collect(x))
    return KineticProfileData(Float64.(collect(psi)), _v(n_i), _v(n_e), _v(T_i), _v(T_e),
        _v(omega_E), _v(omega_tor), _v(chi_e), _v(chi_phi), _v(zeff), species_densities, String(provenance))
end

"""
    KineticProfileSplines

Named 1D cubic spline interpolants for kinetic profiles (densities, temperatures,
ExB rotation, collisional diagnostics) loaded from an external `kinetic.dat`
file. Mirrors the `ProfileSplines` pattern: each quantity is its own named
spline plus a paired derivative view for the species the NTV kernel needs
(densities and temperatures, used by `wdian`/`wdiat` in `KineticForces/Torque.jl`).

# Fields

  - `xs::Vector{Float64}`: Shared ψ axis (the regular kinetic grid)
  - `ni_spline`, `ne_spline`: Ion / electron number densities [m⁻³]
  - `Ti_spline`, `Te_spline`: Ion / electron temperatures [J]
  - `omegaE_spline`: ExB rotation ω_E [rad/s]
  - `loglam_spline`: Coulomb logarithm
  - `nui_spline`, `nue_spline`: Krook collision frequencies [s⁻¹]
  - `zeff_spline`: Effective charge Z_eff
  - `ni_deriv`, `ne_deriv`, `Ti_deriv`, `Te_deriv`: Derivative views for ψ-derivative access
"""
struct KineticProfileSplines{S,D}
    xs::Vector{Float64}
    npts::Int
    npts_minus_1::Int
    ni_spline::S
    ne_spline::S
    Ti_spline::S
    Te_spline::S
    omegaE_spline::S
    loglam_spline::S
    nui_spline::S
    nue_spline::S
    zeff_spline::S
    ni_deriv::D
    ne_deriv::D
    Ti_deriv::D
    Te_deriv::D
end

"""
    KineticProfileSplines(xs, ni, ne, Ti, Te, omegaE, loglam, nui, nue, zeff;
                          extrap=ExtendExtrap())

Build the kinetic profile splines from arrays of values defined on the shared
ψ grid `xs`. Uses CubicFit boundary conditions to match `ProfileSplines`.
Temperatures must already be in Joules.
"""
function KineticProfileSplines(xs::Vector{Float64},
    ni::Vector{Float64}, ne::Vector{Float64},
    Ti::Vector{Float64}, Te::Vector{Float64},
    omegaE::Vector{Float64}, loglam::Vector{Float64},
    nui::Vector{Float64}, nue::Vector{Float64}, zeff::Vector{Float64};
    extrap::AbstractExtrap=ExtendExtrap())
    npts = length(xs)
    @assert all(length(v) == npts for v in (ni, ne, Ti, Te, omegaE, loglam, nui, nue, zeff))

    ni_spline = cubic_interp(xs, ni; extrap=extrap)
    ne_spline = cubic_interp(xs, ne; extrap=extrap)
    Ti_spline = cubic_interp(xs, Ti; extrap=extrap)
    Te_spline = cubic_interp(xs, Te; extrap=extrap)
    omegaE_spline = cubic_interp(xs, omegaE; extrap=extrap)
    loglam_spline = cubic_interp(xs, loglam; extrap=extrap)
    nui_spline = cubic_interp(xs, nui; extrap=extrap)
    nue_spline = cubic_interp(xs, nue; extrap=extrap)
    zeff_spline = cubic_interp(xs, zeff; extrap=extrap)

    ni_deriv = deriv1(ni_spline)
    ne_deriv = deriv1(ne_spline)
    Ti_deriv = deriv1(Ti_spline)
    Te_deriv = deriv1(Te_spline)

    KineticProfileSplines{typeof(ni_spline),typeof(ni_deriv)}(
        xs, npts, npts - 1,
        ni_spline, ne_spline, Ti_spline, Te_spline,
        omegaE_spline, loglam_spline, nui_spline, nue_spline, zeff_spline,
        ni_deriv, ne_deriv, Ti_deriv, Te_deriv
    )
end

"""
    PlasmaEquilibrium(...)

The final, self-contained result of the equilibrium reconstruction.
This object provides a complete representation of the processed plasma equilibrium in flux coordinates.

# Fields

  - `config::EquilibriumConfig`:
    The equilibrium configuration object used for the reconstruction.

  - `params::EquilibriumParameters`:
    Computed equilibrium parameters and diagnostics.
  - `profiles::ProfileSplines`:
    Named 1D profile splines (F, P, dV/dψ, q) on normalized psi grid.
    Access values at grid points via `profiles.F_spline.y[i]`, etc.
    Access derivatives via `profiles.F_deriv.y[i]` or `profiles.F_deriv(psi)`.
  - `geometry::GeometryProfileSplines`:
    Named 1D splines for flux-surface-averaged geometry (area, ⟨r⟩, ⟨R⟩),
    populated automatically by `compute_geometry_profiles` during construction.
  - **Grid coordinates (shared by all rzphi/eqfun interpolants):**

      + `rzphi_xs::Vector{Float64}`: ψ coordinates (length mpsi+1)
      + `rzphi_ys::Vector{Float64}`: θ coordinates (length mtheta+1)
  - **Geometric quantities (rzphi, 4 interpolants):**
    2D cubic interpolants for flux-coordinate mapping with periodic BC in theta.

      + **x value:** normalized ψ
      + **y value:** SFL poloidal angle ∈ [0, 1]
      + `rzphi_rsquared::CubicInterpolantND`: r_coord² = (R - ro)² + (Z - zo)²
      + `rzphi_offset::CubicInterpolantND`: η/(2π) - θₙₑw (angle offset)
      + `rzphi_nu::CubicInterpolantND`: ν in ϕ = 2πζ + ν(ψ, θ)
      + `rzphi_jac::CubicInterpolantND`: Jacobian
  - **Physics quantities (eqfun, 3 interpolants):**
    2D cubic interpolants storing local physics and geometric quantities.

      + **x value:** normalized ψ
      + **y value:** SFL poloidal angle θₙₑw
      + `eqfun_B::CubicInterpolantND`: Total magnetic field strength [T]
      + `eqfun_metric1::CubicInterpolantND`: (e₁⋅e₂ + q⋅e₃⋅e₁)/(J⋅B²)
      + `eqfun_metric2::CubicInterpolantND`: (e₂⋅e₃ + q⋅e₃⋅e₃)/(J⋅B²)
  - `ro::Float64`: R-coordinate of the magnetic axis [m]
  - `zo::Float64`: Z-coordinate of the magnetic axis [m]
  - `psio::Float64`: Total flux difference |Ψ_axis - Ψ_boundary| [Weber/radian]
  - `ingest::EquilibriumIngest`: raw arrays forwarded from the equilibrium input for the
    `gpec.h5` rerun snapshot — a [`DirectIngest`](@ref)/[`InverseIngest`](@ref) for file-based
    equilibria, or `nothing` for analytic ones (regenerated from their TOML section on replay)
  - `analytic::Union{Nothing,AnalyticEquilibriumConfig}`: the parameters an analytic kind
    (`sol`, `lar`, `tj_analytic`, `tj_analytic_direct`) was solved from, so [`reform`](@ref) can
    re-solve it on a new grid; `nothing` for file-based and IMAS equilibria. A defensive copy, so
    later edits to the caller's config do not reach it.
  - `kinetic::Union{Nothing,KineticProfileData}`: the raw kinetic profiles of `config.kinetic_file` (or of
    [`attach_kinetic_profiles!`](@ref)), on the file's own ψ grid; `nothing` when none are attached
"""
mutable struct PlasmaEquilibrium{P<:ProfileSplines,G<:GeometryProfileSplines,I2D<:FastInterpolations.CubicInterpolantND}
    config::EquilibriumConfig
    params::EquilibriumParameters
    profiles::P
    geometry::G

    # Grid coordinates (shared by all 2D interpolants)
    rzphi_xs::Vector{Float64}
    rzphi_ys::Vector{Float64}

    # Geometric quantities (4 interpolants)
    rzphi_rsquared::I2D
    rzphi_offset::I2D
    rzphi_nu::I2D
    rzphi_jac::I2D

    # Physics quantities (3 interpolants)
    eqfun_B::I2D
    eqfun_metric1::I2D
    eqfun_metric2::I2D

    ro::Float64
    zo::Float64
    psio::Float64

    ingest::EquilibriumIngest
    analytic::Union{Nothing,AnalyticEquilibriumConfig}
    kinetic::Union{Nothing,KineticProfileData}
end

# Solvers build the equilibrium before PlasmaEquilibrium records its source and attaches the kinetic
# profiles, so allow construction without them; ingest, analytic and kinetic default to nothing.
PlasmaEquilibrium(config, params, profiles, geometry, rzphi_xs, rzphi_ys,
    rzphi_rsquared, rzphi_offset, rzphi_nu, rzphi_jac,
    eqfun_B, eqfun_metric1, eqfun_metric2, ro, zo, psio) =
    PlasmaEquilibrium(config, params, profiles, geometry, rzphi_xs, rzphi_ys,
        rzphi_rsquared, rzphi_offset, rzphi_nu, rzphi_jac,
        eqfun_B, eqfun_metric1, eqfun_metric2, ro, zo, psio, nothing, nothing, nothing)
