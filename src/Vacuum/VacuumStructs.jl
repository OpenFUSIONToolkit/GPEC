"""
    VacuumInput

Struct holding plasma boundary and mode data as provided from DCON namelist and computed quantities.

# Fields

- `r::Vector{Float64}`: Plasma boundary R-coordinate as a function of poloidal angle
- `z::Vector{Float64}`: Plasma boundary Z-coordinate as a function of poloidal angle
- `delta::Vector{Float64}`: Toroidal angle offset: -dφ/qa; 0 for coordinate systems using machine angle (e.g., PEST basis)
- `mlow::Int`: Lower poloidal mode number for spectral representation
- `mhigh::Int`: Upper poloidal mode number for spectral representation
- `mpert::Int`: Number of perturbation modes (mhigh - mlow + 1)
- `n::Int`: The toroidal mode number
- `qa::Float64`: Safety factor at the plasma boundary
- `mtheta_eq::Int`: Number of poloidal angles in the input equilibrium boundary arrays
- `mtheta::Int`: Number of poloidal grid points for vacuum calculations
- `kernelsign::Float64`: Sign for kernel; +1 or -1, only ≠ 1 for mutual inductance calculations
- `force_wv_symmetry::Bool`: Boolean flag to enforce symmetry in the vacuum response matrix (set in dcon.toml)
"""
@kwdef mutable struct VacuumInput
    r::Vector{Float64} = Float64[]
    z::Vector{Float64} = Float64[]
    delta::Vector{Float64} = Float64[]
    mlow::Int = 0
    mhigh::Int = 0
    mpert::Int = 0
    n::Int = 0
    qa::Float64 = 0.0
    mtheta_eq::Int = 1
    mtheta::Int = 1
    kernelsign::Float64 = 1.0
    force_wv_symmetry::Bool = true
end

"""
    PlasmaGeometry

Struct holding plasma geometry data on the mtheta grid for vacuum calculations. 

Arrays are of length `mtheta`, where `mtheta` is the number of poloidal grid points and θ ∈ [0, 1).

# Fields

- `x::Vector{Float64}`: Plasma surface R-coordinate
- `z::Vector{Float64}`: Plasma surface Z-coordinate
- `delta::Vector{Float64}`: Toroidal angle offset divided by qa (i.e., -ν/qa where ϕ = 2πζ + ν(ψ, θ)) at plasma surface
- `dx_dtheta::Vector{Float64}`: Derivative dR/dθ at plasma surface
- `dz_dtheta::Vector{Float64}`: Derivative dZ/dθ at plasma surface
- `cnqd::Vector{Float64}`: cos(n * qa * delta) at plasma surface
- `snqd::Vector{Float64}`: sin(n * qa * delta) at plasma surface
- `sinlt::Matrix{Float64}`: sin(l * θ) basis functions for poloidal modes at plasma surface
- `coslt::Matrix{Float64}`: cos(l * θ) basis functions for poloidal modes at plasma surface
- `snlth::Matrix{Float64}`: sin(l * θ + n * qa * delta) basis functions for poloidal modes at plasma surface
- `cslth::Matrix{Float64}`: cos(l * θ + n * qa * delta) basis functions for poloidal modes at plasma surface
"""
struct PlasmaGeometry
    x::Vector{Float64}
    z::Vector{Float64}
    delta::Vector{Float64}
    dx_dtheta::Vector{Float64}
    dz_dtheta::Vector{Float64}
    cnqd::Vector{Float64}
    snqd::Vector{Float64}
    sinlt::Matrix{Float64}
    coslt::Matrix{Float64}
    snlth::Matrix{Float64}
    cslth::Matrix{Float64}
end

"""
    WallGeometry

Struct holding wall geometry data for vacuum calculations. 

Arrays are of length `mtheta`, where `mtheta` is the number of poloidal grid points and θ ∈ [0, 1).

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
    WallShapeSettings

Struct containing input settings for vacuum wall geometry.

# Fields

- `shape::String`: String selecting wall shape. Options are:
  - `"nowall"`: No wall
  - `"conformal"`: Wall conformal to plasma surface at distance `a`
  - `"elliptical"`: Elliptical wall
  - `"dee"`: Dee-shaped wall
  - `"mod_dee"`: Modified Dee-shaped wall
  - `"from_file"`: Custom wall shape from wall_geo.dat file
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
function initialize_plasma_surface(inputs::VacuumInput)

    # Interpolate arrays from input onto mtheta grid (from readahg in the Fortran)
    mtheta = inputs.mtheta
    x_plasma = interp_to_new_grid(inputs.r, mtheta)
    z_plasma = interp_to_new_grid(inputs.z, mtheta)
    delta = interp_to_new_grid(inputs.delta, mtheta)
    # Plasma boundary theta derivative (this is semi-working)
    # All of these arrays are of length mth with θ = [0, 1)
    theta_grid = range(0, stop=2π, length=mtheta + 1)[1:end-1] # length mtheta without endpoint
    dx_plasma_dtheta = periodic_cubic_deriv(theta_grid, x_plasma)
    dz_plasma_dtheta = periodic_cubic_deriv(theta_grid, z_plasma)
    # Trigonometric basis arrays
    # Compute n*q*δ phase term for each poloidal angle
    nqdelta = inputs.n .* inputs.qa .* delta
    
    # Basis functions for the phase factor
    cos_nqdelta = cos.(nqdelta)
    sin_nqdelta = sin.(nqdelta)
    
    # Mode numbers: m = mlow, mlow+1, ..., mlow+mpert-1
    mode_numbers = (inputs.mlow-1) .+ (1:inputs.mpert)'  # Row vector for broadcasting
    
    # Outer product: theta_grid (column) × mode_numbers (row) → (mtheta × mpert) matrix
    mitheta = theta_grid * mode_numbers  # Broadcasting: m*θ for all combinations
    
    # Compute basis functions without phase (pure harmonics)
    sin_mstheta = sin.(mitheta)
    cos_mstheta = cos.(mitheta)
    
    # Add phase factor: m*θ + n*q*δ (broadcast nqdelta column-wise across modes)
    mitheta_arg = mitheta .+ nqdelta
    sin_mstheta_arg = sin.(mitheta_arg)
    cos_mstheta_arg = cos.(mitheta_arg)

    return PlasmaGeometry(
        x_plasma,
        z_plasma,
        delta,
        dx_plasma_dtheta,
        dz_plasma_dtheta,
        cos_nqdelta,
        sin_nqdelta,
        sin_mstheta,
        cos_mstheta,
        sin_mstheta_arg,
        cos_mstheta_arg
    )
end

#-----------------------------------------#
# Functions to initialize data structures #
#-----------------------------------------#
"""
    initialize_wall(inputs::VacuumInput, plasma_surf::PlasmaGeometry, wall_settings::WallShapeSettings) -> WallGeometry

Initialize the wall geometry based on the provided vacuum inputs and wall shape settings. 

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
function initialize_wall(inputs::VacuumInput, plasma_surf::PlasmaGeometry, wall_settings::WallShapeSettings)

    @assert(wall_settings.shape in ["nowall", "conformal", "elliptical", "dee", "mod_dee", "from_file"],
        "Invalid wall shape: $(wall_settings.shape). Must be one of: nowall, conformal, elliptical, dee, mod_dee, from_file")
    
    # Basic wall flags
    nowall = wall_settings.shape == "nowall"
    is_closed_toroidal = true

   # All of these arrays are of length mtheta with θ = [0, 1)
    mtheta = inputs.mtheta
    theta_grid = range(0, stop=2π, length=mtheta + 1)[1:end-1] 
    
    # Get wall shape from form_wall
    # Plasma surface coordinates
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

    if wall_settings.shape == "nowall"
        @info "Using no wall"
    elseif wall_settings.shape == "conformal"
        dx = a * r_minor
        @info "Calculating conformal wall shape $dx m from plasma surface." 
        wcentr = r_major
        csmin = min(0.1, 0.1 * minimum(x_plasma))
        for i in 1:mtheta
            j = (i == 1) ? mtheta : i - 1
            k = (i == mtheta) ? 1 : i + 1
            # Normal vector calculation
            alph = atan(x_plasma[k] - x_plasma[j], z_plasma[j] - z_plasma[k])
            x_wall[i] = max(csmin, x_plasma[i] + a * r_minor * cos(alph))
            z_wall[i] = z_plasma[i] + a * r_minor * sin(alph)
        end

    elseif wall_settings.shape == "elliptical"
        @info "Calculating elliptical wall shape with a = $a."
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
        @info "Calculating dee-shaped wall with R = $wcentr + $r_minor * (1.0 + $a - $cw) * cos(θ + $dw * sin(θ)), Z = -$bw * $r_minor * (1.0 + $a - $cw) * sin(θ + $tw * sin(2θ)) - $aw * $r_minor * sin(2θ)."
        wcentr = r_major + cw * r_minor
        for i in 1:mtheta
            the = (i - 1) * (2π / mtheta)
            x_wall[i] = wcentr + r_minor * (1.0 + a - cw) * cos(the + dw * sin(the))
            z_wall[i] = -bw * r_minor * (1.0 + a - cw) * sin(the + tw * sin(2.0*the)) - aw * r_minor * sin(2.0*the)
        end

    elseif wall_settings.shape == "mod_dee"
        @info "Calculating modified dee-shaped wall with R = $cw + $a * cos(θ + $dw * sin(θ)), Z = -$bw * $a * sin(θ + $tw * sin(2θ)) - $aw * sin(2θ)."
        wcentr = cw
        for i in 1:mtheta
            the = (i - 1) * (2π / mtheta)
            x_wall[i] = cw + a * cos(the + dw * sin(the))
            z_wall[i] = -bw * a * sin(the + tw * sin(2.0*the)) - aw * sin(2.0*the)
        end

    elseif wall_settings.shape == :from_file
        @info "Loading wall shape from external file 'wall_geo.dat' (**OPTION IS UNDER CONSTRUCTION**)."
        # Load wall geometry from external file "wall_geo.dat"
        wcentr = 0.0
        open("wall_geo.dat", "r") do io
            npots0 = parse(Int, readline(io))  # Number of points in file
            wcentr = parse(Float64, readline(io)) 
            readline(io) # Skip header/comment line

            if npots0 < mtheta
                @error "ERROR: wall_geo.dat contains fewer points ($npots0) than mtheta ($mtheta)."
                error("Wall geometry file size mismatch")
            end

            for i in 1:mtheta
                line = split(readline(io))
                # Assumes file format: [index  R_coord  Z_coord]
                x_wall[i] = parse(Float64, line[2])
                z_wall[i] = parse(Float64, line[3])
            end
        end
    else
        error("Wall shape '$(wall_settings.shape)' is not implemented.")
    end

    # Optional: Re-parameterization
    if wall_settings.equal_arc_wall
        x_wall, z_wall, _, _, _ = distribute_to_equal_arc_grid(x_wall, z_wall, mtheta)
    end

    # Compute wall derivatives
    theta_grid = range(0, stop=2π, length=mtheta + 1)[1:end-1] # length mtheta without endpoint
    dx_dtheta = periodic_cubic_deriv(theta_grid, x_wall)
    dz_dtheta = periodic_cubic_deriv(theta_grid, z_wall)

    # Trigonometric basis arrays
    return WallGeometry(
        nowall,
        is_closed_toroidal,
        x_wall,
        z_wall,
        dx_dtheta,
        dz_dtheta
    )
end

"""
    distribute_to_equal_arc_grid(xin, zin, mw1)

Perform arc length re-parameterization of a 2D curve. 

Takes an input curve defined by `(xin, zin)` coordinates and re-samples it such that
the new points `(xout, zout)` are equally spaced in arc length along the curve.

# Arguments

- `xin::Vector{Float64}`: Array of x-coordinates of the input curve
- `zin::Vector{Float64}`: Array of z-coordinates of the input curve
- `mw1::Int`: Number of points in the input and output curves

# Returns

- `xout::Vector{Float64}`: Array of x-coordinates of the arc-length re-parameterized curve
- `zout::Vector{Float64}`: Array of z-coordinates of the arc-length re-parameterized curve
- `ell::Vector{Float64}`: Array of cumulative arc lengths for the input curve
- `thgr::Vector{Float64}`: Array of re-parameterized 'theta' values corresponding to equal arc lengths
- `thlag::Vector{Float64}`: Array of normalized 'theta' values for the input curve (0 to 1)

# Notes

- Uses Lagrange interpolation for calculating arc length and resampling
- Ensures uniform spacing in arc length for improved numerical stability
"""
function distribute_to_equal_arc_grid(xin::Vector{Float64}, zin::Vector{Float64}, mw1::Int)
    # Temporary arrays for interpolation and arc-length calculation
    thlag = zeros(Float64, mw1) # Normalized input parameter [0, 1]
    ell   = zeros(Float64, mw1) # Cumulative arc length
    thgr  = zeros(Float64, mw1) # New parameter distribution for equal spacing
    xout  = zeros(Float64, mw1) # Uniformly spaced R-coordinates
    zout  = zeros(Float64, mw1) # Uniformly spaced Z-coordinates

    # Define initial normalized parameter thlag
    dt = 1.0 / (mw1 - 1)
    for iw in 1:mw1
        thlag[iw] = dt * (iw - 1)
    end

    # Calculate cumulative arc length using numerical integration
    # We use a mid-point derivative approximation to find the path length
    ell[1] = 0.0 
    for iw in 2:mw1
        # Evaluate derivative at the midpoint of the interval
        thet = (thlag[iw] + thlag[iw - 1]) / 2.0
        
        # Calculate dx/dt and dz/dt using Lagrange interpolation (order 3)
        _, d_xin = lagrange1d(thlag, xin, mw1, 3, thet, 1)
        _, d_zin = lagrange1d(thlag, zin, mw1, 3, thet, 1)
        
        # Instantaneous speed (ds/dt)
        ds_dt = sqrt(d_xin^2 + d_zin^2)
        
        # Accumulate length: ds = (ds/dt) * dt
        ell[iw] = ell[iw - 1] + ds_dt * dt
    end

    # Re-parameterize based on equal arc-length segments
    total_length = ell[mw1]
    ds_uniform = total_length / (mw1 - 1)
    
    for i in 1:mw1
        target_s = ds_uniform * (i - 1)
        # Find the value of 'thlag' that corresponds to the target arc length 's'
        f_th, _ = lagrange1d(ell, thlag, mw1, 3, target_s, 0)
        thgr[i] = f_th
    end

    # Compute final output coordinates (xout, zout)
    for i in 1:mw1
        t_target = thgr[i]
        
        # Interpolate the original (x,z) data at the new parameter points
        f_x, _ = lagrange1d(thlag, xin, mw1, 3, t_target, 0)
        f_z, _ = lagrange1d(thlag, zin, mw1, 3, t_target, 0)
        
        xout[i] = f_x
        zout[i] = f_z
    end
    
    return xout, zout, ell, thgr, thlag
end