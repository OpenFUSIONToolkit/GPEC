function symbolize_keys(dict::Dict{String,Any})
    return Dict(Symbol(k) => v for (k, v) in dict)
end

"""
    EquilibriumConfig(...)

A mutable struct containing configuration parameters for equilibrium reconstruction.
Bundles all necessary settings originally specified in the equil fortran namelists.

## Fields

  - `eq_type::String` - Type of equilibrium file ("efit", "solovev", "lar", etc.)
  - `eq_filename::String` - Path to equilibrium input file
  - `jac_type::String` - Jacobian coordinate type ("hamada", "pest", "equal_arc", "boozer", "park", "other")
  - `power_bp::Int` - Poloidal field power exponent for Jacobian
  - `power_b::Int` - Toroidal field power exponent for Jacobian
  - `power_r::Int` - Major radius power exponent for Jacobian
  - `r0exp::Float64` - Major radius normalization for CHEASE/EQDSK [m]
  - `b0exp::Float64` - On-axis toroidal field normalization for CHEASE/EQDSK [T]
  - `grid_type::String` - Grid type for flux surface discretization ("ldp", etc.)
  - `psilow::Float64` - Lower limit of normalized flux coordinate
  - `psihigh::Float64` - Upper limit of normalized flux coordinate
  - `mpsi::Int` - Number of radial grid points
  - `mtheta::Int` - Number of poloidal grid points
  - `newq0::Int` - Override for on-axis safety factor (0 = use input value)
  - `etol::Float64` - Error tolerance for equilibrium solver
  - `force_termination::Bool` - Terminate after equilibrium setup (skip stability calculations)
  - `use_galgrid::Bool` - Use the same grid as galerkin method
"""
@kwdef mutable struct EquilibriumConfig
    eq_type::String = "efit"
    eq_filename::String = "mypath"
    r0exp::Float64 = 1.0
    b0exp::Float64 = 1.0

    jac_type::String = "hamada"
    power_bp::Int = 0
    power_b::Int = 0
    power_r::Int = 0

    grid_type::String = "ldp"
    psilow::Float64 = 1e-2
    psihigh::Float64 = 0.994
    mpsi::Int = 128
    mtheta::Int = 256

    newq0::Int = 0
    etol::Float64 = 1e-7

    force_termination::Bool = false
    use_galgrid::Bool = true

    """
    Modified internal constructor that enforces self consistency within the inputs
    """
    function EquilibriumConfig(eq_type, eq_filename, r0exp, b0exp, jac_type, power_bp, power_b, power_r,
        grid_type, psilow, psihigh, mpsi, mtheta, newq0, etol,
        force_termination, use_galgrid)
        if jac_type == "hamada"
            @info "Forcing hamada coordinate jacobian exponents: power_*"
            power_b = 0
            power_bp = 0
            power_r = 0
        elseif jac_type == "pest"
            @info "Forcing pest coordinate jacobian exponents: power_*"
            power_b = 0
            power_bp = 0
            power_r = 2
        elseif jac_type == "equal_arc"
            @info "Forcing equal_arc coordinate jacobian exponents: power_*"
            power_b = 0
            power_bp = 1
            power_r = 0
        elseif jac_type == "boozer"
            @info "Forcing boozer coordinate jacobian exponents: power_*"
            power_b = 2
            power_bp = 0
            power_r = 0
        elseif jac_type == "park"
            @info "Forcing park coordinate jacobian exponents: power_*"
            power_b = 1
            power_bp = 0
            power_r = 0
        elseif jac_type == "other"
            @info "Using manual jacobian exponents: power b, bp, r = $(power_b), $(power_bp), $(power_r)"
        elseif jac_type != "other"
            error("Cannot recognize jac_type = $(jac_type)")
        end
        return new(eq_type, eq_filename, r0exp, b0exp, jac_type, power_bp, power_b, power_r,
            grid_type, psilow, psihigh, mpsi, mtheta, newq0, etol,
            force_termination, use_galgrid)
    end
end

"""
Outer constructor for EquilibriumConfig from a parsed TOML dictionary
"""
function EquilibriumConfig(equil_dict::Dict{String,Any}, base_path::String="./")
    # Check for required fields
    required_keys = ("eq_filename", "eq_type")
    missingkeys = filter(k -> !haskey(equil_dict, k), required_keys)

    if !isempty(missingkeys)
        error("Missing required key(s) in [Equilibrium]: $(join(missingkeys, ", "))")
    end

    # Filter to only known parameters
    config_fields = Set(String.(fieldnames(EquilibriumConfig)))
    config_data = Dict{String,Any}()

    for (k, v) in equil_dict
        if k in config_fields
            config_data[k] = v
        else
            @warn "Unknown equilibrium parameter: $k"
        end
    end

    # Construct validated struct
    config = EquilibriumConfig(; symbolize_keys(config_data)...)
    if !isabspath(config.eq_filename)
        config.eq_filename = normpath(joinpath(base_path, config.eq_filename))
    end

    return config
end

"""
Outer constructor for EquilibriumConfig that enables a toml file
interface for specifying the configuration settings

DEPRECATED: Use [Equilibrium] section in jpec.toml instead
"""
function EquilibriumConfig(path::String)
    raw = TOML.parsefile(path)

    # Extract EQUIL_CONTROL with default fallback
    config_data = get(raw, "EQUIL_CONTROL", Dict())

    # Check for required fields
    required_keys = ("eq_filename", "eq_type")
    missingkeys = filter(k -> !haskey(config_data, k), required_keys)

    if !isempty(missingkeys)
        error("Missing required key(s) in [EQUIL_CONTROL]: $(join(missingkeys, ", "))")
    end

    # Construct validated struct
    config = EquilibriumConfig(; symbolize_keys(config_data)...)
    if !isabspath(config.eq_filename)
        config.eq_filename = normpath(joinpath(dirname(path), config.eq_filename))
    end

    return config
end

"""
    LargeAspectRatioConfig(...)

A mutable struct holding parameters for the Large Aspect Ratio (LAR) plasma equilibrium model.

## Fields:

  - `lar_r0`: The major radius of the plasma [m].
  - `lar_a`: The minor radius of the plasma [m].
  - `beta0`: The beta value on axis (normalized pressure).
  - `q0`: The safety factor on axis.
  - `p_pres`: The exponent for the pressure profile, defined as `p00 * (1 - (r / a)^2)^p_pres`.
  - `p_sig`: The exponent that determines the shape of the current-related function profile.
  - `sigma_type`: The type of sigma profile, can be "default" or "wesson". If "wesson", the sigma profile is defined as `sigma0 * (1 - (r / a)^2)^p_sig`.
  - `mtau`: The number of grid points in the poloidal direction.
  - `ma`: The number of grid points in the radial direction.
  - `zeroth`: If set to true, it neglects the Shafranov shift
"""
@kwdef mutable struct LargeAspectRatioConfig
    lar_r0::Float64 = 10.0
    lar_a::Float64 = 1.0
    beta0::Float64 = 1e-3
    q0::Float64 = 1.5
    p_pres::Float64 = 2.0
    p_sig::Float64 = 1.0
    sigma_type::String = "default"
    mtau::Int = 128
    ma::Int = 128
    zeroth::Bool = false
end

"""
Outer constructor for LargeAspectRatioConfig that enables a toml file
interface for specifying the configuration settings
"""
function LargeAspectRatioConfig(path::String)
    raw = TOML.parsefile(path)
    input_data = get(raw, "LAR_INPUT", Dict())
    return LargeAspectRatioConfig(; symbolize_keys(input_data)...)
end

"""
    SolovevConfig(...)

A mutable struct holding parameters for the Solev'ev (SOL) plasma equilibrium model.

## Fields:

  - `mr`: number of radial grid zones
  - `mz`: number of axial grid zones
  - `ma`: number of flux grid zones
  - `e`:  elongation
  - `a`: minor radius
  - `r0`: major radius
  - `q0`: safety factor at the o-point
  - `p0fac`: scale on-axis pressure (P-> P+P0*p0fac. beta changes. Phi,q constant)
  - `b0fac`: scale toroidal field at constant beta (s*Phi,s*f,s^2*P. bt changes. Shape,beta constant)
  - `f0fac`: scale toroidal field at constant pressure (s*f. beta,q changes. Phi,p,bp constant)
"""
@kwdef mutable struct SolovevConfig
    mr::Int = 128      # number of radial grid zones
    mz::Int = 128      # number of axial grid zones
    ma::Int = 128      # number of flux grid zones
    e::Float64 = 1.6       # elongation
    a::Float64 = 0.33      # minor radius
    r0::Float64 = 1.0      # major radius
    q0::Float64 = 1.9      # safety factor at the o-point
    p0fac::Float64 = 1       # scale on-axis pressure (P-> P+P0*p0fac. beta changes. Phi,q constant)
    b0fac::Float64 = 1       # scale toroidal field at constant beta (s*Phi,s*f,s^2*P. bt changes. Shape,beta constant)
    f0fac::Float64 = 1       # scale toroidal field at constant pressure (s*f. beta,q changes. Phi,p,bp constant)
end

"""
Outer constructor for SolovevConfig that enables a toml file
interface for specifying the configuration settings
"""
function SolovevConfig(path::String) # if we use @kwdef, it generates SolovevConfig() so it conflicts with this line.
    raw = TOML.parsefile(path)
    input_data = get(raw, "SOL_INPUT", Dict())
    return SolovevConfig(; symbolize_keys(input_data)...)
end

"""
    DirectRunInput(...)

A container struct that bundles all necessary inputs for the `direct_run` function.
It is created by one of the equilibrium file read-in functions after processing the
raw equilibrium data and preparing the initial splines.

## Fields

  - `config::EquilibriumConfig`
    The equilibrium configuration object.

  - `sq_in`
    1D spline data versus normalized poloidal flux `psin`.
    Quantities:

     1. `F = R * B_t` — toroidal flux function [m·T]
     2. `μ₀ * Pressure` — plasma pressure (non-negative) [T²]
     3. `q` — safety factor profile
     4. `√ψ_norm` — square root of normalized flux
  - `psi_in`
    2D cubic interpolant on the (R, Z) grid [m].
    The values correspond to the **poloidal flux** adjusted to be zero at the boundary [Wb/rad].
    Definitions:

     1. `ψ(R, Z) = ψ_boundary - ψ(R, Z)`

     2. `ψ = ψ * sign(ψ(centerR, centerZ))`

          * 1D profiles are represented by `CubicInterpolant` or `CubicSeriesInterpolant`
          * 2D flux surfaces by `CubicInterpolantND`
  - `psi_in_xs::Vector{Float64}` — R coordinate grid for psi_in [m]
  - `psi_in_ys::Vector{Float64}` — Z coordinate grid for psi_in [m]
  - `rmin::Float64` — Minimum R-coordinate of the computational grid [m]
  - `rmax::Float64` — Maximum R-coordinate of the computational grid [m]
  - `zmin::Float64` — Minimum Z-coordinate of the computational grid [m]
  - `zmax::Float64` — Maximum Z-coordinate of the computational grid [m]
  - `psio::Float64` — Total flux difference `|ψ_axis - ψ_boundary|` [Wb/rad]
"""
mutable struct DirectRunInput{S<:FastInterpolations.CubicSeriesInterpolant,I2D<:FastInterpolations.CubicInterpolantND}
    config::EquilibriumConfig
    sq_in::S       # 1D profile spline for F, P, q
    psi_in::I2D    # 2D flux interpolant (CubicInterpolantND)
    psi_in_xs::Vector{Float64}  # R coordinates
    psi_in_ys::Vector{Float64}  # Z coordinates
    rmin::Float64    # Minimum R-coordinate of the computational grid [m].
    rmax::Float64    # Maximum R-coordinate of the computational grid [m].
    zmin::Float64    # Minimum Z-coordinate of the computational grid [m].
    zmax::Float64    # Maximum Z-coordinate of the computational grid [m].
    psio::Float64    # The total flux difference |ψ_axis - ψ_boundary| [Weber / radian].
end

"""
    InverseRunInput(...)

A container struct for inputs to the `inverse_run` function.

## Fields

  - `config::EquilibriumConfig` - The equilibrium configuration object
  - `sq_in::CubicSeriesInterpolant` - 1D profile spline for F, P, q
  - `rz_in_xs::Vector{Float64}` - ψ coordinate grid for rz_in
  - `rz_in_ys::Vector{Float64}` - θ coordinate grid for rz_in
  - `rz_in_R::CubicInterpolantND` - R coordinate interpolant [m]
  - `rz_in_Z::CubicInterpolantND` - Z coordinate interpolant [m]
  - `ro::Float64` - R-coordinate of magnetic axis [m]
  - `zo::Float64` - Z-coordinate of magnetic axis [m]
  - `psio::Float64` - Total flux difference |ψ_axis - ψ_boundary| [Wb/rad]
"""
mutable struct InverseRunInput{S<:FastInterpolations.CubicSeriesInterpolant,I2D<:FastInterpolations.CubicInterpolantND}
    config::EquilibriumConfig
    sq_in::S   # 1D profile spline for F, P, q
    rz_in_xs::Vector{Float64}   # ψ coordinates
    rz_in_ys::Vector{Float64}   # θ coordinates
    rz_in_R::I2D                # R coordinate interpolant
    rz_in_Z::I2D                # Z coordinate interpolant
    ro::Float64                 # R axis location
    zo::Float64                 # Z axis location
    psio::Float64               # Total flux difference |psi_axis - psi_boundary|
end

"""
    EquilibriumParameters

A mutable struct containing computed equilibrium parameters and diagnostic flags.

## Fields

  - `ro::Union{Nothing,Float64}` - R-coordinate of the magnetic axis [m]
  - `zo::Union{Nothing,Float64}` - Z-coordinate of the magnetic axis [m]
  - `psio::Union{Nothing,Float64}` - Total flux difference |ψ_axis - ψ_boundary| [Wb/rad]
  - `rsep::Union{Nothing,Vector{Float64}}` - R-coordinates of the plasma boundary [m]
  - `zsep::Union{Nothing,Vector{Float64}}` - Z-coordinates of the plasma boundary [m]
  - `rext::Union{Nothing,Vector{Float64}}` - R-coordinates of the plasma edge [m]
  - `zext::Union{Nothing,Vector{Float64}}` - Z-coordinates of the plasma edge [m]
  - `psi0::Union{Nothing,Float64}` - Normalized poloidal flux at reference location
  - `b0::Union{Nothing,Float64}` - Total magnetic field strength at the axis [T]
  - `q0::Union{Nothing,Float64}` - Safety factor at the axis
  - `qmin::Union{Nothing,Float64}` - Minimum safety factor in the plasma
  - `qmax::Union{Nothing,Float64}` - Maximum safety factor in the plasma
  - `qa::Union{Nothing,Float64}` - Safety factor at the plasma edge
  - `q95::Union{Nothing,Float64}` - Safety factor at 95% flux surface
  - `qextrema_psi::Union{Nothing,Vector{Float64}}` - Normalized flux values at q extrema
  - `qextrema_q::Union{Nothing,Vector{Float64}}` - Safety factor values at extrema
  - `mextrema::Union{Nothing,Int}` - Number of extrema in q-profile
  - `psi_norm::Union{Nothing,Float64}` - Normalized poloidal flux
  - `b_norm::Union{Nothing,Float64}` - Normalized magnetic field strength
  - `psi_axis::Union{Nothing,Float64}` - Poloidal flux at the axis
  - `psi_boundary::Union{Nothing,Float64}` - Poloidal flux at the boundary
  - `psi_boundary_norm::Union{Nothing,Float64}` - Normalized boundary flux
  - `psi_axis_norm::Union{Nothing,Float64}` - Normalized axis flux
  - `psi_boundary_offset::Union{Nothing,Float64}` - Boundary flux offset
  - `psi_axis_offset::Union{Nothing,Float64}` - Axis flux offset
  - `psi_boundary_sign::Union{Nothing,Int}` - Sign of boundary flux
  - `psi_axis_sign::Union{Nothing,Int}` - Sign of axis flux
  - `psi_boundary_zero::Union{Nothing,Bool}` - Whether boundary flux is zero
  - `rmean::Union{Nothing,Float64}` - Mean major radius [m]
  - `amean::Union{Nothing,Float64}` - Mean minor radius [m]
  - `aratio::Union{Nothing,Float64}` - Aspect ratio (R0/a)
  - `kappa::Union{Nothing,Float64}` - Plasma elongation
  - `delta1::Union{Nothing,Float64}` - Upper triangularity
  - `delta2::Union{Nothing,Float64}` - Lower triangularity
  - `bt0::Union{Nothing,Float64}` - Toroidal field at axis [T]
  - `crnt::Union{Nothing,Float64}` - Plasma current [A]
  - `bwall::Union{Nothing,Float64}` - Toroidal field at wall [T]
  - `verbose::Bool` - Enable verbose output
  - `diagnose_src::Bool` - Enable source data diagnostics
  - `diagnose_maxima::Bool` - Enable extrema diagnostics
  - `volume::Union{Nothing,Float64}` - Plasma volume [m³]
  - `betat::Union{Nothing,Float64}` - Toroidal beta
  - `betan::Union{Nothing,Float64}` - Normalized beta
  - `betaj::Union{Nothing,Float64}` - Total beta
  - `betap1::Union{Nothing,Float64}` - Poloidal beta (definition 1)
  - `betap2::Union{Nothing,Float64}` - Poloidal beta (definition 2)
  - `betap3::Union{Nothing,Float64}` - Poloidal beta (definition 3)
  - `li1::Union{Nothing,Float64}` - Internal inductance (definition 1)
  - `li2::Union{Nothing,Float64}` - Internal inductance (definition 2)
  - `li3::Union{Nothing,Float64}` - Internal inductance (definition 3)
"""
@kwdef mutable struct EquilibriumParameters
    ro::Union{Nothing,Float64} = nothing # R-coordinate of the magnetic axis [m]
    zo::Union{Nothing,Float64} = nothing # Z-coordinate of the magnetic axis [m]
    psio::Union{Nothing,Float64} = nothing # Total flux difference |ψ_axis - ψ_boundary| [Wb/rad]
    rsep::Union{Nothing,Vector{Float64}} = nothing # R-coordinates of the plasma boundary [m]
    zsep::Union{Nothing,Vector{Float64}} = nothing # Z-coordinates of the plasma boundary [m]
    rext::Union{Nothing,Vector{Float64}} = nothing # R-coordinates of the plasma edge [m]
    zext::Union{Nothing,Vector{Float64}} = nothing # Z-coordinates of the plasma edge [m]
    psi0::Union{Nothing,Float64} = nothing # Normalized poloidal flux
    b0::Union{Nothing,Float64} = nothing # Total magnetic field strength at the axis [T]
    q0::Union{Nothing,Float64} = nothing # Safety factor at the axis
    qmin::Union{Nothing,Float64} = nothing # Minimum safety factor in the plasma
    qmax::Union{Nothing,Float64} = nothing # Maximum safety factor in the plasma
    qa::Union{Nothing,Float64} = nothing # Safety factor at the plasma edge
    q95::Union{Nothing,Float64} = nothing # Safety factor at 95% flux surface
    qextrema_psi::Union{Nothing,Vector{Float64}} = nothing # Normalized poloidal flux values where q has extrema
    qextrema_q::Union{Nothing,Vector{Float64}} = nothing # Safety factor values at the extrema points
    mextrema::Union{Nothing,Int} = nothing # Number of extrema points in the q-profile
    psi_norm::Union{Nothing,Float64} = nothing # Normalized poloidal flux at the axis
    b_norm::Union{Nothing,Float64} = nothing # Normalized total magnetic field strength at the axis
    psi_axis::Union{Nothing,Float64} = nothing # Normalized poloidal flux at the axis
    psi_boundary::Union{Nothing,Float64} = nothing # Poloidal flux at the boundary
    psi_boundary_norm::Union{Nothing,Float64} = nothing # Normalized poloidal flux at the boundary
    psi_axis_norm::Union{Nothing,Float64} = nothing # Normalized poloidal flux at the axis
    psi_boundary_offset::Union{Nothing,Float64} = nothing # Offset for the boundary poloidal flux
    psi_axis_offset::Union{Nothing,Float64} = nothing  # Offset for the axis poloidal flux
    psi_boundary_sign::Union{Nothing,Int} = nothing # Sign of the boundary poloidal flux
    psi_axis_sign::Union{Nothing,Int} = nothing # Sign of the axis poloidal flux
    psi_boundary_zero::Union{Nothing,Bool} = nothing # Whether the boundary poloidal flux is zero
    rmean::Union{Nothing,Float64} = nothing # Mean R-coordinate of the plasma [m]
    amean::Union{Nothing,Float64} = nothing # Mean minor radius of the plasma [m]
    aratio::Union{Nothing,Float64} = nothing # Aspect ratio of the plasma (R0/a)
    kappa::Union{Nothing,Float64} = nothing # Elongation of the plasma cross-section
    delta1::Union{Nothing,Float64} = nothing # Triangularity of the plasma cross-section (upper triangularity)
    delta2::Union{Nothing,Float64} = nothing # Triangularity of the plasma cross-section (lower triangularity)
    bt0::Union{Nothing,Float64} = nothing # Toroidal magnetic field at the axis [T]
    crnt::Union{Nothing,Float64} = nothing # Plasma current at the axis [A]
    bwall::Union{Nothing,Float64} = nothing # Toroidal magnetic field at the wall [T]
    verbose::Bool = false # Whether to print verbose output
    diagnose_src::Bool = false # Whether to diagnose source data
    diagnose_maxima::Bool = false # Whether to diagnose maxima in the equilibrium
    volume::Union{Nothing,Float64} = nothing # Plasma volume [m³]
    betat::Union{Nothing,Float64} = nothing # Toroidal beta (normalized pressure) at the axis
    betan::Union{Nothing,Float64} = nothing # Normalized beta at the axis
    betaj::Union{Nothing,Float64} = nothing # Total beta at the axis
    betap1::Union{Nothing,Float64} = nothing # Pressure beta at the axis
    betap2::Union{Nothing,Float64} = nothing # Toroidal beta at the axis
    betap3::Union{Nothing,Float64} = nothing # Poloidal beta at the axis
    li1::Union{Nothing,Float64} = nothing  # Internal inductance at the axis
    li2::Union{Nothing,Float64} = nothing  # External inductance at the axis
    li3::Union{Nothing,Float64} = nothing  # Total inductance at the axis
end

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

    # Create value interpolants with CubicFit BC and LinearBinary search for sequential psi access
    F_spline = cubic_interp(xs, F_vals; bc=CubicFit(), extrap=extrap, search=LinearBinarySearch())
    P_spline = cubic_interp(xs, P_vals; bc=CubicFit(), extrap=extrap, search=LinearBinarySearch())
    dVdpsi_spline = cubic_interp(xs, dVdpsi_vals; bc=CubicFit(), extrap=extrap, search=LinearBinarySearch())
    q_spline = cubic_interp(xs, q_vals; bc=CubicFit(), extrap=extrap, search=LinearBinarySearch())

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
"""
mutable struct PlasmaEquilibrium{P<:ProfileSplines,I2D<:FastInterpolations.CubicInterpolantND}
    config::EquilibriumConfig
    params::EquilibriumParameters
    profiles::P

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
end
