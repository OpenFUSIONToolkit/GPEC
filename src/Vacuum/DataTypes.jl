"""
Vacuum structures and initialization functions.
"""

"""
    VacuumInput

Struct holding plasma boundary and mode data as provided from ForceFreeStates namelist and computed quantities.

# Fields

  - `r::Vector{Float64}`: Plasma boundary R-coordinate as a function of poloidal angle
  - `z::Vector{Float64}`: Plasma boundary Z-coordinate as a function of poloidal angle
  - `ν::Vector{Float64}`: Free parameter in specifying toroidal angle, ϕ = 2πζ + ν(ψ, θ), on theta grid
  - `mlow::Int`: Lower poloidal mode number for spectral representation
  - `mhigh::Int`: Upper poloidal mode number (mhigh = mlow + mpert - 1)
  - `mpert::Int`: Number of poloidal modes (mhigh - mlow + 1)
  - `n::Int`: Toroidal mode number
  - `mtheta::Int`: Number of vacuum calculation poloidal grid points
  - `force_wv_symmetry::Bool`: Boolean flag to enforce symmetry in the vacuum response matrix
"""
@kwdef struct VacuumInput
    r::Vector{Float64} = Float64[]
    z::Vector{Float64} = Float64[]
    ν::Vector{Float64} = Float64[]
    mlow::Int = 0
    mhigh::Int = 0
    mpert::Int = 0
    n::Int = 0
    mtheta::Int = 1
    force_wv_symmetry::Bool = true
end

"""
    VacuumInput(
        equil::Equilibrium.PlasmaEquilibrium,
        ψ::Float64,
        mtheta::Int,
        mpert::Int,
        mlow::Int,
        n::Int,
        force_wv_symmetry::Bool = true
    ) -> VacuumInput

Constructor to create a VacuumInput struct for computing Green's functions at arbitrary flux surface.
Extracts plasma geometry from equilibrium at the given flux surface and packages it into VacuumInput format.

## Arguments

  - `equil`: Equilibrium solution
  - `ψ`: Normalized flux coordinate
  - `mtheta`: Number of vacuum calculation poloidal points
  - `mpert`: Number of perturbing poloidal modes
  - `mlow`: Lowest poloidal mode number
  - `n`: Toroidal mode number
  - `force_wv_symmetry::Bool`: Boolean flag to enforce symmetry in the vacuum response matrix (default: true)

## Returns

VacuumInput structure ready for compute_vacuum_response()

## Usage

This is used to compute Green's functions at singular surfaces:

```julia
vac_input = VacuumInput(equil, sing_surf.psifac, mtheta, mpert, mlow, n; force_wv_symmetry=true)
```
"""
function VacuumInput(
    equil::Equilibrium.PlasmaEquilibrium,
    ψ::Float64,
    mtheta::Int,
    mpert::Int,
    mlow::Int,
    n::Int;
    force_wv_symmetry::Bool=true
)
    # Extract plasma surface geometry at this psi
    r, z, ν = extract_plasma_surface_at_psi(equil, ψ)

    return VacuumInput(;
        r=reverse(r),
        z=reverse(z),
        ν=reverse(ν),
        mlow=mlow,
        mhigh=mlow + mpert - 1,
        mpert=mpert,
        n=n,
        mtheta=mtheta,
        force_wv_symmetry=force_wv_symmetry
    )
end

"""
    PlasmaGeometry

Struct holding plasma geometry data on the mtheta grid for vacuum calculations. Arrays are
of length `mtheta`, where `mtheta` is the number of poloidal grid points and θ ∈ [0, 1).

# Fields

  - `x::Vector{Float64}`: Plasma surface R-coordinate on VACUUM theta grid
  - `z::Vector{Float64}`: Plasma surface Z-coordinate on VACUUM theta grid
  - `ν::Vector{Float64}`: Magnetic toroidal angle offset from geometric toroidal angle
"""
struct PlasmaGeometry
    x::Vector{Float64}
    z::Vector{Float64}
    ν::Vector{Float64}
end

"""
    WallGeometry

Struct holding wall geometry data for vacuum calculations. Arrays are of length
`mtheta`, where `mtheta` is the number of poloidal grid points and θ ∈ [0, 1).

# Fields

  - `nowall::Bool`: Boolean flag indicating if there is no wall
  - `x::Vector{Float64}`: Wall R-coordinates
  - `z::Vector{Float64}`: Wall Z-coordinates
"""
struct WallGeometry
    nowall::Bool
    x::Vector{Float64}
    z::Vector{Float64}
end

"""
    WallShapeSettings

Struct containing input settings for vacuum wall geometry.

# Fields

  - `shape::String`: String selecting wall shape. Options are:

        + `"nowall"`: No wall
        + `"conformal"`: Wall conformal to plasma surface at distance `a`
        + `"elliptical"`: Elliptical wall
        + `"dee"`: Dee-shaped wall
        + `"mod_dee"`: Modified Dee-shaped wall
        + `"filepath"`: Custom wall shape from the file you specify

  - `a::Float64`: Distance of wall from plasma in units of major radius (conformal), or shape parameter (others)
  - `aw::Float64`: Half-thickness parameter for Dee-shaped walls
  - `bw::Float64`: Elongation parameter for wall shapes
  - `cw::Float64`: Offset of the center of the wall from the major radius
  - `dw::Float64`: Triangularity parameter for wall shapes
  - `tw::Float64`: Sharpness of the corners of the wall (try 0.05 as initial value)
  - `equal_arc_wall::Bool`: Flag to enforce equal arc length distribution of nodes on the wall
    (recommended unless wall is very close to plasma)
"""
@kwdef struct WallShapeSettings

    # Core shape selection
    shape::String = "nowall"

    # Standard geometric parameters for Dee/Mod-Dee
    a::Float64 = 0.3
    aw::Float64 = 0.05
    bw::Float64 = 1.5
    cw::Float64 = 0.0
    dw::Float64 = 0.5
    tw::Float64 = 0.05

    # Algorithmic options
    equal_arc_wall::Bool = true
end

"""
    initialize_plasma_surface(inputs::VacuumInput) -> PlasmaGeometry

Initialize the plasma surface geometry based on the provided vacuum inputs.
We interpolate the input plasma boundary arrays from the inputs struct onto the mtheta grid.

# Arguments

  - `inputs::VacuumInput`: Struct containing plasma boundary data

# Returns

  - `PlasmaGeometry`: Struct containing plasma surface coordinates, derivatives, and basis functions
"""
function PlasmaGeometry(inputs::VacuumInput)

    # Interpolate arrays from input onto mtheta grid
    θ_in = range(0.0, 2π; length=length(inputs.r)) # VacuumInput uses [0, 2π] grid
    θ_out = range(; start=0, length=inputs.mtheta, step=2π/inputs.mtheta) # VACUUM uses [0, 2π) grid

    # Use one-shot API with PeriodicBC
    x = cubic_interp(θ_in, inputs.r, θ_out; bc=PeriodicBC()) # no endpoint handling needed!
    z = cubic_interp(θ_in, inputs.z, θ_out; bc=PeriodicBC())
    ν = cubic_interp(θ_in, inputs.ν, θ_out; bc=PeriodicBC())

    return PlasmaGeometry(x, z, ν)
end

"""
    WallGeometry(inputs::VacuumInput, plasma_surf::PlasmaGeometry, wall_settings::WallShapeSettings) -> WallGeometry

Constructor to initialize the wall geometry based on the provided vacuum inputs and wall shape settings.

This performs functionality similar to portions of the `arrays` function in the original
Fortran VACUUM code. It returns a `WallGeometry` struct containing the necessary wall
surface data for vacuum calculations.

# Arguments

  - `inputs::VacuumInput`: Struct containing vacuum calculation parameters
  - `plasma_surf::PlasmaGeometry`: Struct with plasma surface geometry (used for reference)
  - `wall_settings::WallShapeSettings`: Struct specifying wall shape and parameters

# Returns

  - `WallGeometry`: Struct containing wall surface coordinates and derivatives

# Notes

  - Supports multiple wall shapes: nowall, conformal, elliptical, dee, mod_dee, from_file
  - Optionally redistributes wall points to equal arc length spacing if `equal_arc_wall=true`
"""
function WallGeometry(inputs::VacuumInput, plasma_surf::PlasmaGeometry, wall_settings::WallShapeSettings)

    # Output wall coordinate arrays
    mtheta = inputs.mtheta
    x_wall = zeros(mtheta)
    z_wall = zeros(mtheta)

    if wall_settings.shape == "nowall"
        return WallGeometry(
            true,
            x_wall,
            z_wall
        )
    end

    # Compute plasma surface quantities
    x_plasma = plasma_surf.x
    z_plasma = plasma_surf.z

    # Output wall coordinate arrays
    x_wall = zeros(Float64, mtheta)
    z_wall = zeros(Float64, mtheta)

    # Common geometric parameters
    xmin = minimum(x_plasma)
    xmax = maximum(x_plasma)
    zmin = minimum(z_plasma)
    zmax = maximum(z_plasma)

    r_minor = 0.5 * (xmax - xmin)
    r_major = 0.5 * (xmax + xmin)

    # Destructuring settings for readability
    (; aw, bw, cw, dw, tw, a) = wall_settings
    wcentr = 0.0 # Initialize

    if wall_settings.shape == "conformal"
        dx = a * r_minor
        @info "Calculating conformal wall shape $((@sprintf "%.2e" dx)) m from plasma surface."
        wcentr = r_major
        centerstack_min = min(0.1, 0.1 * minimum(x_plasma))  # Avoid wall crossing R=0 axis
        for i in 1:mtheta
            j = mod1(i - 1, mtheta)
            k = mod1(i + 1, mtheta)
            # Normal vector calculation
            alph = atan(x_plasma[k] - x_plasma[j], z_plasma[j] - z_plasma[k])
            x_wall[i] = max(centerstack_min, x_plasma[i] + a * r_minor * cos(alph))
            z_wall[i] = z_plasma[i] + a * r_minor * sin(alph)
        end

        if any(x_wall .<= centerstack_min + eps(Float64))
            @warn "Conformal wall with a=$a would cross R=0 axis; forcing minimum wall R to $(@sprintf "%.2e" centerstack_min) m to avoid unphysical geometry."
        end

    elseif wall_settings.shape == "elliptical"
        @info "Calculating elliptical wall shape with a = $((@sprintf "%.2e" a)) m."
        wcentr = r_major

        zrad = 0.5 * (zmax - zmin)
        zh = sqrt(abs(zrad^2 - r_minor^2))
        zmuw = log((a/zh) + sqrt((a/zh)^2 + 1))
        bw_eff = (zh * cosh(zmuw)) / a

        for i in 1:mtheta
            the = (i - 1) * (2π / mtheta)
            x_wall[i] = r_major + a * cos(the)
            z_wall[i] = -bw_eff * a * sin(the)
        end

    elseif wall_settings.shape == "dee"
        wcentr = r_major + cw * r_minor
        @info "Calculating dee-shaped wall with R = $((@sprintf "%.2e" wcentr)) + $((@sprintf "%.2e" r_minor)) * (1.0 + $((@sprintf "%.2e" a)) - $((@sprintf "%.2e" cw))) * cos(θ + $((@sprintf "%.2e" dw)) * sin(θ)), Z = -$((@sprintf "%.2e" bw)) * $((@sprintf "%.2e" r_minor)) * (1.0 + $((@sprintf "%.2e" a)) - $((@sprintf "%.2e" cw))) * sin(θ + $((@sprintf "%.2e" tw)) * sin(2θ)) - $((@sprintf "%.2e" aw)) * $((@sprintf "%.2e" r_minor)) * sin(2θ)."
        for i in 1:mtheta
            the = (i - 1) * (2π / mtheta)
            x_wall[i] = wcentr + r_minor * (1.0 + a - cw) * cos(the + dw * sin(the))
            z_wall[i] = -bw * r_minor * (1.0 + a - cw) * sin(the + tw * sin(2.0*the)) - aw * r_minor * sin(2.0*the)
        end

    elseif wall_settings.shape == "mod_dee"
        @info "Calculating modified dee-shaped wall with R = $((@sprintf "%.2e" cw)) + $((@sprintf "%.2e" a)) * cos(θ + $((@sprintf "%.2e" dw)) * sin(θ)), Z = -$((@sprintf "%.2e" bw)) * $((@sprintf "%.2e" a)) * sin(θ + $((@sprintf "%.2e" tw)) * sin(2θ)) - $((@sprintf "%.2e" aw)) * sin(2θ)."
        wcentr = cw
        for i in 1:mtheta
            the = (i - 1) * (2π / mtheta)
            x_wall[i] = cw + a * cos(the + dw * sin(the))
            z_wall[i] = -bw * a * sin(the + tw * sin(2.0*the)) - aw * sin(2.0*the)
        end

    else
        filepath = wall_settings.shape
        if !isfile(filepath)
            error(
                "Wall geometry file $filepath does not exist. " *
                "Please set the wall shape parameter to a valid file path or a built-in shape " *
                "(nowall, conformal, elliptical, dee, mod_dee)."
            )
        end

        wcentr = 0.0
        open(wall_settings.shape, "r") do io
            npots0 = parse(Int, readline(io))  # Number of points in file
            wcentr = parse(Float64, readline(io))
            readline(io) # Skip header/comment line

            npots0 < mtheta && error("Wall geometry file $filepath contains fewer points ($npots0) than mtheta ($mtheta).")

            for i in 1:mtheta
                line = split(readline(io))
                # Assumes file format: [index  R_coord  Z_coord]
                x_wall[i] = parse(Float64, line[2])
                z_wall[i] = parse(Float64, line[3])
            end
        end
    end

    # Optional: Re-parameterization for equal arc length spacing of wall points
    if wall_settings.equal_arc_wall
        @info "Re-distributing wall points to equal arc length spacing (assumes closed, toroidal wall)."
        x_wall, z_wall = distribute_to_equal_arc_grid(x_wall, z_wall)
    end

    # To add support for x<0 walls, be sure to carefully replicate Chance's fortran code x<0 handling in the kernel function to account for the additional singularities associated with this
    any(x_wall .<= 0.0) && error("Wall R-coordinates contain non-physical values (R <= 0). Check wall geometry.")

    return WallGeometry(false, x_wall, z_wall)
end
