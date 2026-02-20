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

    # TODO: this was in Free.jl - is this general for GPEC too?
    # Invert values for n < 0
    if n < 0
        ν .= -ν
        n = -n
    end

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

@kwdef struct VacuumInput3D
    x::Vector{Float64} = Float64[]
    z::Vector{Float64} = Float64[]
    ν::Vector{Float64} = Float64[]
    mlow::Int = 0
    mpert::Int = 0
    nlow::Int = 0
    npert::Int = 0
    mtheta::Int = 1
    nzeta::Int = 1
    force_wv_symmetry::Bool = true
end

"""
    VacuumInput3D(inputs_2D::VacuumInput, nzeta::Int, nlow::Int, npert::Int)

Convenience constructor for a 3D VacuumInput3D struct from a 2D VacuumInput with the additional
required parameters for the toroidal grid/modes.
"""
function VacuumInput3D(inputs_2D::VacuumInput, nzeta::Int, nlow::Int, npert::Int)
    return VacuumInput3D(;
        x=inputs_2D.r,
        z=inputs_2D.z,
        ν=inputs_2D.ν,
        mlow=inputs_2D.mlow,
        mpert=inputs_2D.mpert,
        nlow=nlow,
        npert=npert,
        mtheta=inputs_2D.mtheta,
        nzeta=nzeta,
        force_wv_symmetry=inputs_2D.force_wv_symmetry
    )
end

"""
    KernelParams2D(n::Int)

Parameter struct for 2D vacuum kernel dispatch. Holds the toroidal mode number `n`.
"""
struct KernelParams2D
    n::Int
end

"""
    KernelParams3D(PATCH_RAD::Int, RAD_DIM::Int, INTERP_ORDER::Int)

Parameter struct for 3D vacuum kernel dispatch. Holds singular quadrature parameters.
"""
struct KernelParams3D
    PATCH_RAD::Int
    RAD_DIM::Int
    INTERP_ORDER::Int
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
    x = cubic_interp(θ_in, inputs.r; bc=PeriodicBC()).(θ_out) # no endpoint handling needed!
    z = cubic_interp(θ_in, inputs.z; bc=PeriodicBC()).(θ_out)
    ν = cubic_interp(θ_in, inputs.ν; bc=PeriodicBC()).(θ_out)

    return PlasmaGeometry(x, z, ν)
end

"""
    PlasmaGeometry3D

3D toroidal surface geometry for vacuum boundary integral calculations.

Built by toroidally extruding a 2D poloidal contour (`PlasmaGeometry`) and computing
Cartesian coordinates, tangent vectors, normals, and differential area elements. Note
that the gradient/area elements are scaled by dθ and dζ.

# Fields

  - `mtheta::Int`: Number of poloidal grid points
  - `nzeta::Int`: Number of toroidal grid points
  - `num_gridpoints::Int`: Total number of surface grid points (mtheta * nzeta)
  - `r::Matrix{Float64}`: Surface points in Cartesian (X,Y,Z), shape (num_gridpoints, 3)
  - `dr_dθ::Matrix{Float64}`: Poloidal tangent vector ∂r/∂θ × dθ, shape (num_gridpoints, 3)
  - `dr_dζ::Matrix{Float64}`: Toroidal tangent vector ∂r/∂ζ × dζ, shape (num_gridpoints, 3)
  - `normal::Matrix{Float64}`: Oriented normal vectors, shape (num_gridpoints, 3)
  - `normal_orient::Int`: Forces normals to face out from vacuum region (+1 or -1)
"""
@kwdef struct PlasmaGeometry3D
    mtheta::Int = 1
    nzeta::Int = 1
    r::Matrix{Float64} = zeros(1, 3)
    ν::Vector{Float64} = Float64[]
    dr_dθ::Matrix{Float64} = zeros(1, 3)
    dr_dζ::Matrix{Float64} = zeros(1, 3)
    normal::Matrix{Float64} = zeros(1, 3)
    normal_orient::Int = 1
end

"""
    PlasmaGeometry3D(plasma_2d::PlasmaGeometry, nzeta::Int)

Construct a 3D axisymmetric toroidal surface from a 2D poloidal contour.

# Algorithm

 0. Interpolate 2D arrays onto mtheta grid
 1. Map 2D (R, Z, ν) to 3D Cartesian: X = R cos(ϕ+ν), Y = R sin(ϕ+ν), Z = Z
 2. Fit periodic bicubic splines to (X, Y, Z) on (θ, ϕ) grid
 3. Compute tangent vectors via spline gradients
 4. Compute normals via cross product: n = ∂r/∂θ × ∂r/∂ζ

# Arguments

  - `plasma_2d`: 2D poloidal plasma geometry
  - `nzeta`: Number of toroidal grid points

# Returns

  - `PlasmaGeometry3D`: Complete 3D surface description
"""
function PlasmaGeometry3D(inputs::VacuumInput3D)

    # Extract 2D poloidal data
    (; mtheta, nzeta, x, z, ν) = inputs
    num_points = mtheta * nzeta
    dθ = 2π / mtheta
    dζ = 2π / nzeta
    θ_grid = range(; start=0, length=mtheta, step=dθ)
    ϕ_grid = range(; start=0, length=nzeta, step=dζ)

    # Allocate output arrays
    r = zeros(num_points, 3)
    dr_dθ = zeros(num_points, 3)
    dr_dζ = zeros(num_points, 3)
    normal = zeros(num_points, 3)

    # Interpolate arrays from input onto mtheta grid (same as 2D)
    θ_in = range(0.0, 2π; length=length(inputs.x)) # VacuumInput uses [0, 2π] grid
    x = cubic_interp(θ_in, inputs.x; bc=PeriodicBC()).(θ_grid) # no endpoint handling needed!
    z = cubic_interp(θ_in, inputs.z; bc=PeriodicBC()).(θ_grid)
    ν = cubic_interp(θ_in, inputs.ν; bc=PeriodicBC()).(θ_grid)

    # Build 3D surface point-by-point from 2D contour
    for i in 1:mtheta, (j, ϕ) in enumerate(ϕ_grid)
        r[i+mtheta*(j-1), :] .= [x[i] * cos(ϕ), x[i] * sin(ϕ), z[i]]
    end

    # Compute tangent vectors and normal vectors via periodic bicubic splines
    itps = [cubic_interp((θ_grid, ϕ_grid), reshape(r[:, k], mtheta, nzeta); bc=(PeriodicBC(; endpoint=:exclusive), PeriodicBC(; endpoint=:exclusive))) for k in 1:3]
    grad = zeros(4)
    for i in 1:mtheta, j in 1:nzeta
        idx = i + (j - 1) * mtheta
        for k in 1:3
            # Grad stores f, fx, fy, fxy
            grad .= itps[k].nodal_derivs.partials[:, i, j]
            dr_dθ[idx, k] = grad[2]
            dr_dζ[idx, k] = grad[3]
        end
        normal[idx, :] = cross(dr_dθ[idx, :], dr_dζ[idx, :])
    end

    # Determine normal orientation (inward for plasma) and enforce it
    idx = argmax(view(r, :, 1)) # outboard midplane
    normal_orient = normal[idx, 1] < 0 ? 1 : -1
    normal .*= normal_orient

    # Warn if grid spacing is highly anisotropic
    spacing_θ = sqrt(sum(abs2, dr_dθ) / size(dr_dθ, 1)) * dθ
    spacing_ζ = sqrt(sum(abs2, dr_dζ) / size(dr_dζ, 1)) * dζ
    aspect_ratio = spacing_ζ / spacing_θ
    @info "Average grid spacing [m]: dθ=$(round(spacing_θ, digits=4)), dζ=$(round(spacing_ζ, digits=4)), aspect ratio=$(round(aspect_ratio, digits=2))"
    aspect_ratio > 10.0 && @warn "Grid aspect ratio is highly anisotropic, which may degrade quadrature accuracy"

    return PlasmaGeometry3D(;
        mtheta=mtheta,
        nzeta=nzeta,
        r=r,
        ν=ν,
        dr_dθ=dr_dθ,
        dr_dζ=dr_dζ,
        normal=normal,
        normal_orient=normal_orient
    )
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
        @info "Using no wall"
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

"""
    WallGeometry3D

Struct holding wall geometry data for vacuum calculations. Arrays are of length
`mtheta`, where `mtheta` is the number of poloidal grid points and θ ∈ [0, 1).

# Fields

  - `nowall::Bool`: Boolean flag indicating if there is no wall
  - `is_closed_toroidal::Bool`: Boolean flag indicating if the wall is a closed toroidal surface
  - `mtheta::Int`: Number of poloidal grid points
  - `nzeta::Int`: Number of toroidal grid points
  - `r::Matrix{Float64}`: (x, y, z) wall coordinates at each grid point
  - `dr_dθ::Matrix{Float64}`: Derivative dR/dθ at wall
  - `dr_dζ::Matrix{Float64}`: Derivative dR/dζ at wall
  - `normal::Matrix{Float64}`: Outward normal vectors at wall
"""
@kwdef struct WallGeometry3D
    nowall::Bool = true
    is_closed_toroidal::Bool = true
    mtheta::Int = 1
    nzeta::Int = 1
    r::Matrix{Float64} = zeros(1, 3)
    dr_dθ::Matrix{Float64} = zeros(1, 3)
    dr_dζ::Matrix{Float64} = zeros(1, 3)
    normal::Matrix{Float64} = zeros(1, 3)
    normal_orient::Int = 1
end

"""
    WallGeometry3D(inputs::VacuumInput3D, plasma_surf::PlasmaGeometry3D, wall_settings::WallShapeSettings)

Contructor to initialize the 3D wall geometry based on the provided vacuum inputs and wall shape settings.
Currently only works for axisymmetric walls generated by toroidal extrusion of 2D poloidal contours.

# Arguments

  - `inputs::VacuumInput3D`: Struct containing vacuum calculation parameters
  - `plasma_surf::PlasmaGeometry3D`: Struct with plasma surface geometry (used for reference)
  - `wall_settings::WallShapeSettings`: Struct specifying wall shape and parameters

# Returns

  - `WallGeometry`: Struct containing wall surface coordinates and derivatives

# Notes

  - Supports multiple wall shapes: nowall, conformal, elliptical, dee, mod_dee, from_file
  - Optionally redistributes wall points to equal arc length spacing if `equal_arc_wall=true`
"""
function WallGeometry3D(inputs::VacuumInput3D, plasma_surf::PlasmaGeometry3D, wall_settings::WallShapeSettings)

    # Basic wall flags
    nowall = wall_settings.shape == "nowall"
    is_closed_toroidal = true

    (; mtheta, nzeta) = inputs
    dθ = 2π / mtheta
    dζ = 2π / nzeta
    θ_grid = range(; start=0, length=mtheta, step=dθ)
    ϕ_grid = range(; start=0, length=nzeta, step=dζ)
    num_points = mtheta * nzeta

    # Output wall coordinate arrays
    r = zeros(num_points, 3)
    normal = zeros(num_points, 3)
    dr_dθ = zeros(num_points, 3)
    dr_dζ = zeros(num_points, 3)
    normal_orient = 1

    if nowall
        @info "Using no wall"
        return WallGeometry3D(
            nowall,
            is_closed_toroidal,
            mtheta,
            nzeta,
            r,
            dr_dθ,
            dr_dζ,
            normal,
            normal_orient
        )
    end

    # Plasma surface coordinates (2D)
    x_plasma = plasma_surf.r[1:plasma_surf.mtheta, 1]
    z_plasma = plasma_surf.r[1:plasma_surf.mtheta, 3]
    xmin = minimum(x_plasma)
    xmax = maximum(x_plasma)
    zmin = minimum(z_plasma)
    zmax = maximum(z_plasma)
    r_minor = 0.5 * (xmax - xmin)
    r_major = 0.5 * (xmax + xmin)

    # Destructuring settings for readability
    (; aw, bw, cw, dw, tw, a) = wall_settings

    if wall_settings.shape == "conformal"
        dx = a * r_minor
        @info "Calculating conformal wall shape $((@sprintf "%.2e" dx)) m from plasma surface."
        centerstack_min = min(0.1, 0.1 * minimum(x_plasma))
        for (j, ϕ) in enumerate(ϕ_grid), i in 1:mtheta
            idx = i + (j - 1) * mtheta
            prev = mod1(i - 1, mtheta)
            next = mod1(i + 1, mtheta)
            # Approximate local tangent t = (dx, dz) using centered finite differences, t ≈ (dx, dz)
            # Then, extend in normal direction, n = (-dz, dx)
            alph = atan(x_plasma[next] - x_plasma[prev], z_plasma[prev] - z_plasma[next])
            R_wall = max(centerstack_min, x_plasma[i] + a * r_minor * cos(alph))
            # Map to Cartesian (X, Y, Z)
            r[idx, :] .= [R_wall * cos(ϕ), R_wall * sin(ϕ), z_plasma[i] + a * r_minor * sin(alph)]
        end

        if any(sqrt.(r[:, 1] .^ 2 .+ r[:, 2] .^ 2) .<= centerstack_min + eps(Float64))
            @warn "Conformal wall with a=$a would cross R=0 axis; forcing minimum wall R to $(@sprintf "%.2e" centerstack_min) m to avoid unphysical geometry."
        end
    elseif wall_settings.shape == "elliptical"
        @info "Calculating elliptical wall shape with a = $((@sprintf "%.2e" a)) m."
        zrad = 0.5 * (zmax - zmin)
        zh = sqrt(abs(zrad^2 - r_minor^2))
        zmuw = log((a/zh) + sqrt((a/zh)^2 + 1))
        bw_eff = (zh * cosh(zmuw)) / a
        for (j, ϕ) in enumerate(ϕ_grid), (i, θ) in enumerate(θ_grid)
            idx = i + (j - 1) * mtheta
            r[idx, :] .= [(r_major + a * cos(θ)) * cos(ϕ), (r_major + a * cos(θ)) * sin(ϕ), -bw_eff * a * sin(θ)]
        end
    elseif wall_settings.shape == "dee"
        error("Dee-shaped walls not yet implemented for 3D walls.")
    elseif wall_settings.shape == "mod_dee"
        error("Modified Dee-shaped walls not yet implemented for 3D walls.")
    else
        filepath = wall_settings.shape
        !isfile(filepath) && error("ERROR: Wall geometry file $filepath does not exist.
            Please set the wall shape parameter to a valid file path or a built-in shape (nowall, conformal, elliptical, dee, mod_dee).")

        open(filepath, "r") do io
            npots0 = parse(Int, readline(io))
            (npots0 != num_points) && error("ERROR: $filepath contains different points ($npots0) than mtheta * nzeta ($num_points).")
            # TODO: add an interpolation here for if they're different?
            for i in 1:num_points
                line = split(readline(io))
                r[i, 1] = parse(Float64, line[1])
                r[i, 2] = parse(Float64, line[2])
                r[i, 3] = parse(Float64, line[3])
            end
        end
    end

    # Optional: Re-parameterization for equal arc length spacing of wall points
    # Note: we still use θ_grid for the derivatives below because we are effectively defining
    # a new θ angle to parametrize the wall over, but maintain the equal spacing
    if wall_settings.equal_arc_wall && (wall_settings.shape != "nowall")
        @info "Re-distributing wall points to equal arc length spacing"
        !is_closed_toroidal && @error "Wall is not closed toroidally; equal arc length distribution assumes periodicity as cannot be safely used."
        x_wall, z_wall = distribute_to_equal_arc_grid(r[1:mtheta, 1], r[1:mtheta, 3])
        for (j, ϕ) in enumerate(ϕ_grid), i in 1:mtheta
            idx = i + (j - 1) * mtheta
            r[idx, :] .= [x_wall[i] * cos(ϕ), x_wall[i] * sin(ϕ), z_wall[i]]
        end
    end

    # Compute tangent vectors and normal vectors via periodic bicubic splines
    itps = [cubic_interp((θ_grid, ϕ_grid), reshape(r[:, k], mtheta, nzeta); bc=(PeriodicBC(; endpoint=:exclusive), PeriodicBC(; endpoint=:exclusive))) for k in 1:3]
    grad = zeros(4)
    for i in 1:mtheta, j in 1:nzeta
        idx = i + (j - 1) * mtheta
        for k in 1:3
            # Grad stores f, fx, fy, fxy
            grad .= itps[k].nodal_derivs.partials[:, i, j]
            dr_dθ[idx, k] = grad[2]
            dr_dζ[idx, k] = grad[3]
        end
        normal[idx, :] = cross(dr_dθ[idx, :], dr_dζ[idx, :])
    end

    # Determine normal orientation (outward for wall) and enforce it
    idx = argmax(view(r, :, 1)) # outboard midplane
    normal_orient = normal[idx, 1] > 0 ? 1 : -1
    @views normal .*= normal_orient

    return WallGeometry3D(;
        nowall=nowall,
        is_closed_toroidal=is_closed_toroidal,
        mtheta=mtheta,
        nzeta=nzeta,
        r=r,
        dr_dθ=dr_dθ,
        dr_dζ=dr_dζ,
        normal=normal,
        normal_orient=normal_orient
    )
end
