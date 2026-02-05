"""
    VacuumInput

Struct holding plasma boundary and mode data as provided from DCON namelist and computed quantities.

# Fields

  - `r::Vector{Float64}`: Plasma boundary R-coordinate on DCON theta grid
  - `z::Vector{Float64}`: Plasma boundary Z-coordinate on DCON theta grid
  - `ν::Vector{Float64}`: Free parameter in specifying toroidal angle, ϕ = 2πζ + ν(ψ, θ), on DCON theta grid
  - `mlow::Int`: Lower poloidal mode number
  - `mpert::Int`: Number of poloidal modes
  - `n::Int`: Toroidal mode number
  - `mtheta::Int`: Number of poloidal grid points for vacuum calculations
  - `kernelsign::Float64`: Sign for kernel; +1 or -1, only ≠ 1 for mutual inductance calculations
  - `force_wv_symmetry::Bool`: Boolean flag to enforce symmetry in the vacuum response matrix (set in dcon.toml)
"""
@kwdef struct VacuumInput
    r::Vector{Float64} = Float64[]
    z::Vector{Float64} = Float64[]
    ν::Vector{Float64} = Float64[]
    mlow::Int = 0
    mpert::Int = 0
    n::Int = 0
    mtheta::Int = 1
    kernelsign::Float64 = 1.0
    force_wv_symmetry::Bool = true
end

"""
    VacuumInput3D

Struct holding 2D plasma boundary for a 3D VACUUM run `and mode data as provided from DCON namelist and computed quantities.
2D contour becomes a 3D axisymmetric surface by toroidal extrusion.

# Fields

  - `x::Vector{Float64}`: Plasma boundary X-coordinate on DCON theta grid
  - `z::Vector{Float64}`: Plasma boundary Z-coordinate on DCON theta grid
  - `ν::Vector{Float64}`: Free parameter in specifying toroidal angle, ϕ = 2πζ + ν(ψ, θ), on DCON theta grid
  - `mlow::Int`: Lower poloidal mode number
  - `mpert::Int`: Number of poloidal modes
  - `nlow::Int`: Lower toroidal mode number
  - `npert::Int`: Number of toroidal modes
  - `mtheta::Int`: Number of poloidal collocation points
  - `nzeta::Int`: Number of toroidal collocation points
  - `kernelsign::Float64`: Sign for kernel; +1 or -1, only ≠ 1 for mutual inductance calculations
  - `force_wv_symmetry::Bool`: Boolean flag to enforce symmetry in the vacuum response matrix (set in dcon.toml)
"""
@kwdef struct VacuumInput3D
    x::Vector{Float64} = Float64[]
    z::Vector{Float64} = Float64[]
    ν::Vector{Float64} = Float64[]
    mlow::Int = 0
    mpert::Int = 0
    nlow::Int = 0
    npert::Int = 0
    n::Int = 0
    mtheta::Int = 1
    nzeta::Int = 1
    kernelsign::Float64 = 1.0
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
        kernelsign=inputs_2D.kernelsign,
        force_wv_symmetry=inputs_2D.force_wv_symmetry
    )
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
    PlasmaGeometry

Struct holding plasma geometry data on the mtheta grid for vacuum calculations. Arrays are
of length `mtheta`, where `mtheta` is the number of poloidal grid points and θ ∈ [0, 1).
It also precomputes trigonometric basis functions needed for Fourier calculations into matrices
of size (mtheta, mpert), where `mpert` is the number of poloidal modes.

# Fields

  - `x::Vector{Float64}`: Plasma surface R-coordinate on VACUUM theta grid
  - `z::Vector{Float64}`: Plasma surface Z-coordinate on VACUUM theta grid
  - `dx_dtheta::Vector{Float64}`: Derivative dR/dθ at plasma surface
  - `dz_dtheta::Vector{Float64}`: Derivative dZ/dθ at plasma surface
  - `sin_mn_basis::Matrix{Float64}`: sin(mθ - nν) basis functions for poloidal modes at plasma surface
  - `cos_mn_basis::Matrix{Float64}`: cos(mθ - nν) basis functions for poloidal modes at plasma surface
"""
@kwdef struct PlasmaGeometry
    x::Vector{Float64} = Float64[]
    z::Vector{Float64} = Float64[]
    ν::Vector{Float64} = Float64[]
    dx_dtheta::Vector{Float64} = Float64[]
    dz_dtheta::Vector{Float64} = Float64[]
    sin_mn_basis::Matrix{Float64} = zeros(1, 1)
    cos_mn_basis::Matrix{Float64} = zeros(1, 1)
end

"""
    PlasmaGeometry(inputs::VacuumInput)

Contructor to initialize the plasma surface geometry based on the provided vacuum inputs.

This function performs functionality from `readahg`, `arrays`, and `funint` in the
original Fortran VACUUM code. It returns a `PlasmaGeometry` struct containing
the necessary plasma surface data for vacuum calculations.

# Process

 1. Interpolate the input plasma boundary arrays onto the mtheta grid
 2. Compute derivatives of the plasma boundary with respect to poloidal angle θ
    using periodic cubic spline differentiation
 3. Compute trigonometric basis functions needed for Fourier calculations

# Arguments

  - `inputs::VacuumInput`: Struct containing plasma boundary data and calculation parameters

# Returns

  - `PlasmaGeometry`: Struct containing plasma surface coordinates, derivatives, and basis functions
"""
function PlasmaGeometry(inputs::VacuumInput)

    (; mtheta, mpert, mlow, ν, r, z, n) = inputs
    # Interpolate arrays from input onto mtheta grid
    R = interp_to_new_grid(r, mtheta)
    Z = interp_to_new_grid(z, mtheta)
    ν = interp_to_new_grid(ν, mtheta)

    # Plasma boundary theta derivative: for splines, need to add periodic point
    θ_grid = range(; start=0, length=mtheta, step=2π/mtheta)
    θ_grid_periodic = range(; start=0, length=mtheta+1, step=2π/mtheta)
    R_periodic = vcat(R, R[1])
    Z_periodic = vcat(Z, Z[1])
    dx_dtheta = periodic_cubic_deriv(θ_grid_periodic, R_periodic)
    dz_dtheta = periodic_cubic_deriv(θ_grid_periodic, Z_periodic)

    # Precompute Fourier transform terms, sin(mθ - nν) and cos(mθ - nν)
    sin_mn_basis = sin.((mlow .+ (0:(mpert-1))') .* θ_grid .- n .* ν)
    cos_mn_basis = cos.((mlow .+ (0:(mpert-1))') .* θ_grid .- n .* ν)

    return PlasmaGeometry(
        R,
        Z,
        ν,
        dx_dtheta,
        dz_dtheta,
        sin_mn_basis,
        cos_mn_basis
    )
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
  - `sin_mn_basis3D::Matrix{Float64}`: sin(mθ - nν - nϕ) basis functions at plasma surface
  - `cos_mn_basis3D::Matrix{Float64}`: cos(mθ - nν - nϕ) basis functions at plasma surface
  - `aspect_ratio::Float64`: Ratio of max to min grid spacing for anisotropy analysis
  - `normal_orient::Int`: Forces normals to face out from vacuum region (+1 or -1)
"""
@kwdef struct PlasmaGeometry3D
    mtheta::Int = 1
    nzeta::Int = 1
    r::Matrix{Float64} = zeros(1, 3)
    dr_dθ::Matrix{Float64} = zeros(1, 3)
    dr_dζ::Matrix{Float64} = zeros(1, 3)
    normal::Matrix{Float64} = zeros(1, 3)
    sin_mn_basis3D::Matrix{Float64} = zeros(1, 1)
    cos_mn_basis3D::Matrix{Float64} = zeros(1, 1)
    aspect_ratio::Float64 = 1.0
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
    (; mtheta, nzeta, npert, nlow, mlow, mpert) = inputs
    num_points = mtheta * nzeta
    dθ = 2π / mtheta
    dζ = 2π / nzeta
    θ_grid = range(; start=0, length=mtheta, step=dθ)
    ϕ_grid = range(; start=0, length=nzeta, step=dζ)

    # Allocate output arrays
    r = zeros(num_points, 3)
    normal = zeros(num_points, 3)
    dr_dθ = zeros(num_points, 3)
    dr_dζ = zeros(num_points, 3)

    # Interpolate arrays from input onto mtheta grid (same as 2D)
    x = interp_to_new_grid(inputs.x, mtheta)
    z = interp_to_new_grid(inputs.z, mtheta)
    ν = interp_to_new_grid(inputs.ν, mtheta)

    # Build 3D surface point-by-point from 2D contour
    for (i, θ) in enumerate(θ_grid), (j, ϕ) in enumerate(ϕ_grid)
        idx = i + (j - 1) * mtheta
        r[idx, :] .= [x[i] * cos(ϕ), x[i] * sin(ϕ), z[i]]
    end

    # Create splines for each Cartesian component (X, Y, Z) with periodic boundary conditions
    r_grid = reshape(r, mtheta, nzeta, 3)
    itps = [cubic_spline_interpolation((θ_grid, ϕ_grid), r_grid[:, :, k]; bc=Periodic(OnGrid())) for k in 1:3]

    # Compute tangent vectors, unit normals, and differential area elements via spline interpolation
    for (i, θ) in enumerate(θ_grid), (j, ϕ) in enumerate(ϕ_grid)
        idx = i + (j - 1) * mtheta
        # Compute gradients directly to avoid list comprehension allocation
        for k in 1:3
            g = Interpolations.gradient(itps[k], θ, ϕ)
            dr_dθ[idx, k] = g[1]
            dr_dζ[idx, k] = g[2]
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

    # Precompute Fourier transform terms, sin(lθ - nν(θ) - nϕ) and cos(lθ - nν(θ) - nϕ)
    sin_mn_basis3D = zeros(num_points, mpert*npert)
    cos_mn_basis3D = zeros(num_points, mpert*npert)
    for idx_n in 1:npert, idx_m in 1:mpert
        n = nlow + idx_n - 1
        m = mlow + idx_m - 1
        for (j, ϕ) in enumerate(ϕ_grid), (i, θ) in enumerate(θ_grid)
            cos_mn_basis3D[i+(j-1)*mtheta, idx_m+(idx_n-1)*mpert] = cos(m * θ - n * (ν[i] + ϕ))
            sin_mn_basis3D[i+(j-1)*mtheta, idx_m+(idx_n-1)*mpert] = sin(m * θ - n * (ν[i] + ϕ))
        end
    end

    return PlasmaGeometry3D(;
        mtheta,
        nzeta,
        r,
        dr_dθ,
        dr_dζ,
        normal,
        sin_mn_basis3D,
        cos_mn_basis3D,
        aspect_ratio,
        normal_orient
    )
end

"""
    WallGeometry

Struct holding wall geometry data for vacuum calculations. Arrays are of length
`mtheta`, where `mtheta` is the number of poloidal grid points and θ ∈ [0, 1).

# Fields

  - `nowall::Bool`: Boolean flag indicating if there is no wall
  - `is_closed_toroidal::Bool`: Boolean flag indicating if the wall is a closed toroidal surface
  - `x::Vector{Float64}`: Wall R-coordinates
  - `z::Vector{Float64}`: Wall Z-coordinates
  - `dx_dtheta::Vector{Float64}`: Derivative dR/dθ at wall
  - `dz_dtheta::Vector{Float64}`: Derivative dZ/dθ at wall
"""
@kwdef struct WallGeometry
    nowall::Bool = true
    is_closed_toroidal::Bool = true
    x::Vector{Float64} = Float64[]
    z::Vector{Float64} = Float64[]
    dx_dtheta::Vector{Float64} = Float64[]
    dz_dtheta::Vector{Float64} = Float64[]
end

"""
    WallGeometry(inputs::VacuumInput, plasma_surf::PlasmaGeometry, wall_settings::WallShapeSettings)

Contructor to initialize the wall geometry based on the provided vacuum inputs and wall shape settings.

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

    # Basic wall flags
    nowall = wall_settings.shape == "nowall"
    is_closed_toroidal = true

    # Output wall coordinate arrays
    mtheta = inputs.mtheta
    x_wall = zeros(mtheta)
    z_wall = zeros(mtheta)
    dx_dtheta = zeros(mtheta)
    dz_dtheta = zeros(mtheta)
    θ_grid = range(; start=0, length=mtheta, step=2π/mtheta)

    if nowall
        @info "Using no wall"
        return WallGeometry(;
            nowall=nowall,
            is_closed_toroidal=is_closed_toroidal,
            x=x_wall,
            z=z_wall,
            dx_dtheta=dx_dtheta,
            dz_dtheta=dz_dtheta
        )
    end

    # Compute plasma surface quantities
    x_plasma = plasma_surf.x
    z_plasma = plasma_surf.z
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
        centerstack_min = min(0.1, 0.1 * minimum(x_plasma))  # Avoid wall crossing R=0 axis
        for i in 1:mtheta
            prev = mod1(i - 1, mtheta)
            next = mod1(i + 1, mtheta)
            # Approximate local tangent t = (dx, dz) using centered finite differences, t ≈ (dx, dz)
            # Then, extend in normal direction, n = (-dz, dx)
            alph = -atan(x_plasma[next] - x_plasma[prev], z_plasma[next] - z_plasma[prev])
            x_wall[i] = max(centerstack_min, x_plasma[i] + a * r_minor * cos(alph))
            z_wall[i] = z_plasma[i] + a * r_minor * sin(alph)
        end
        if any(x_wall .<= centerstack_min + eps(Float64))
            @warn "Conformal wall with a=$a would cross R=0 axis; forcing minimum wall R to $(@sprintf "%.2e" centerstack_min) m to avoid unphysical geometry."
        end
    elseif wall_settings.shape == "elliptical"
        # TODO: need to verify I fixed these walls shapes for CCW correctly
        @info "Calculating elliptical wall shape with a = $((@sprintf "%.2e" a)) m."
        zrad = 0.5 * (zmax - zmin)
        zh = sqrt(abs(zrad^2 - r_minor^2))
        zmuw = log((a/zh) + sqrt((a/zh)^2 + 1))
        bw_eff = (zh * cosh(zmuw)) / a
        for (i, θ) in enumerate(θ_grid)
            x_wall[i] = r_major + a * cos(θ)
            z_wall[i] = bw_eff * a * sin(θ)
        end
    elseif wall_settings.shape == "dee"
        wcentr = r_major + cw * r_minor
        @info "Calculating dee-shaped wall with R = $((@sprintf "%.2e" wcentr)) + $((@sprintf "%.2e" r_minor)) * (1.0 + $((@sprintf "%.2e" a)) - $((@sprintf "%.2e" cw))) * cos(θ + $((@sprintf "%.2e" dw)) * sin(θ)), Z = -$((@sprintf "%.2e" bw)) * $((@sprintf "%.2e" r_minor)) * (1.0 + $((@sprintf "%.2e" a)) - $((@sprintf "%.2e" cw))) * sin(θ + $((@sprintf "%.2e" tw)) * sin(2θ)) - $((@sprintf "%.2e" aw)) * $((@sprintf "%.2e" r_minor)) * sin(2θ)."
        for (i, θ) in enumerate(θ_grid)
            x_wall[i] = wcentr + r_minor * (1.0 + a - cw) * cos(θ + dw * sin(θ))
            z_wall[i] = bw * r_minor * (1 + a - cw) * sin(θ + tw * sin(2 * θ)) - aw * r_minor * sin(2 * θ)
        end
    elseif wall_settings.shape == "mod_dee"
        @info "Calculating modified dee-shaped wall with R = $((@sprintf "%.2e" cw)) + $((@sprintf "%.2e" a)) * cos(θ + $((@sprintf "%.2e" dw)) * sin(θ)), Z = -$((@sprintf "%.2e" bw)) * $((@sprintf "%.2e" a)) * sin(θ + $((@sprintf "%.2e" tw)) * sin(2θ)) - $((@sprintf "%.2e" aw)) * sin(2θ)."
        for (i, θ) in enumerate(θ_grid)
            x_wall[i] = cw + a * cos(θ + dw * sin(θ))
            z_wall[i] = bw * a * sin(θ + tw * sin(2 * θ)) - aw * sin(2 * θ)
        end
    else
        filepath = wall_settings.shape
        !isfile(filepath) && @error "ERROR: Wall geometry file $filepath does not exist.
            Please set the wall shape parameter to a valid file path or a built-in shape (nowall, conformal, elliptical, dee, mod_dee)."
        open(wall_settings.shape, "r") do io
            npots0 = parse(Int, readline(io))  # Number of points in file
            readline(io) # Skip wcentr line
            readline(io) # Skip header/comment line
            (npots0 < mtheta) && @error "ERROR: $filename contains fewer points ($npots0) than mtheta ($mtheta)."
            for i in 1:mtheta
                line = split(readline(io))
                # Assumes file format: [index  R_coord  Z_coord]
                x_wall[i] = parse(line[2])
                z_wall[i] = parse(line[3])
            end
        end
    end

    # Optional: Re-parameterization
    if wall_settings.equal_arc_wall && (wall_settings.shape != "nowall")
        @info "Re-distributing wall points to equal arc length spacing"
        !is_closed_toroidal && error("Wall is not closed toroidally; equal arc length distribution assumes periodicity as cannot be safely used.")
        x_wall, z_wall, _, theta_grid, _ = distribute_to_equal_arc_grid(x_wall, z_wall, mtheta)
        theta_grid .= theta_grid .* (2π)  # Scale to [0, 2π) - irregular spacing
        fx_of_theta = interpolate((theta_grid,), x_wall, Gridded(Linear()))
        dx_dtheta = only.(Interpolations.gradient.(Ref(fx_of_theta), theta_grid))
        fz_of_theta = interpolate((theta_grid,), z_wall, Gridded(Linear()))
        dz_dtheta = only.(Interpolations.gradient.(Ref(fz_of_theta), theta_grid))
    else
        # used regular theta grid spacing to build wall
        dx_dtheta = periodic_cubic_deriv(θ_grid, x_wall)
        dz_dtheta = periodic_cubic_deriv(θ_grid, z_wall)
    end

    # to add support for x<0 walls, be sure to carefully replicate Chance's fortran code x<0 handling in the kernel function to account for the additional singularities associated with this
    any(x_wall .<= 0.0) && error("Wall R-coordinates contain non-physical values (R <= 0). Check wall geometry.")

    return WallGeometry(;
        nowall=nowall,
        is_closed_toroidal=is_closed_toroidal,
        x=x_wall,
        z=z_wall,
        dx_dtheta=dx_dtheta,
        dz_dtheta=dz_dtheta
    )
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
            k_prev = mod1(i - 1, mtheta)
            k_next = mod1(i + 1, mtheta)
            # Compute normal direction in poloidal plane
            alph = atan(x_plasma[k_next] - x_plasma[k_prev], z_plasma[k_prev] - z_plasma[k_next])
            # Wall radius in cylindrical coordinates
            R_wall = max(centerstack_min, x_plasma[i] + dx * cos(alph))
            Z_wall = z_plasma[i] + dx * sin(alph)
            # Map to Cartesian (X, Y, Z)
            r[idx, :] .= [R_wall * cos(ϕ), R_wall * sin(ϕ), Z_wall]
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
            # TODO: add an interpolation here for if they're different
            for i in 1:num_points
                line = split(readline(io))
                r[i, 1] = parse(Float64, line[1])
                r[i, 2] = parse(Float64, line[2])
                r[i, 3] = parse(Float64, line[3])
            end
        end
    end

    # Optional: Re-parameterization
    if wall_settings.equal_arc_wall && (wall_settings.shape != "nowall")
        error("Re-distributing wall points to equal arc length spacing not implemented for 3D walls yet.")
    end

    # Create splines for each Cartesian component (X, Y, Z) with periodic boundary conditions
    r_grid = reshape(r, mtheta, nzeta, 3)
    itps = [cubic_spline_interpolation((θ_grid, ϕ_grid), r_grid[:, :, k]; bc=Periodic(OnGrid())) for k in 1:3]

    # Compute tangent vectors, normals, and differential area elements
    for (i, θ) in enumerate(θ_grid), (j, ϕ) in enumerate(ϕ_grid)
        idx = i + (j - 1) * mtheta
        for k in 1:3
            g = Interpolations.gradient(itps[k], θ, ϕ)
            dr_dθ[idx, k] = g[1]
            dr_dζ[idx, k] = g[2]
        end
        normal[idx, :] = cross(dr_dθ[idx, :], dr_dζ[idx, :])
    end

    # Determine normal orientation (outward for wall) and enforce it
    idx = argmax(view(r, :, 1)) # outboard midplane
    normal_orient = normal[idx, 1] > 0 ? 1 : -1
    @views normal .*= normal_orient

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
