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
  - `"filepath"`: Custom wall shape from the file you specify
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
    theta_grid = range(start=0, length=mtheta, step=2π/mtheta)
    dx_plasma_dtheta = periodic_cubic_deriv(theta_grid, x_plasma)
    dz_plasma_dtheta = periodic_cubic_deriv(theta_grid, z_plasma)
    # Trigonometric basis arrays
    # Pre-allocate output arrays
    cos_nqdelta = zeros(mtheta)
    sin_nqdelta = zeros(mtheta)
    sin_mstheta = zeros(mtheta, inputs.mpert)
    cos_mstheta = zeros(mtheta, inputs.mpert)
    sin_mstheta_arg = zeros(mtheta, inputs.mpert)
    cos_mstheta_arg = zeros(mtheta, inputs.mpert)

    # Calculate n*q*delta phase term
    nqdelta = inputs.n .* inputs.qa .* delta
    cos_nqdelta .= cos.(nqdelta)
    sin_nqdelta .= sin.(nqdelta)

    # Fuse loop for trigonometric basis functions to improve cache efficiency
    # and avoid intermediate array allocations.
    mlow = inputs.mlow
    mpert = inputs.mpert
    for l in 1:mpert
        mode_val = mlow + l - 1
        for i in 1:mtheta
            m_theta = theta_grid[i] * mode_val
            nqdelta_val = nqdelta[i]

            cos_mstheta[i, l] = cos(m_theta)
            sin_mstheta[i, l] = sin(m_theta)
            cos_mstheta_arg[i, l] = cos(m_theta + nqdelta_val)
            sin_mstheta_arg[i, l] = sin(m_theta + nqdelta_val)
        end
    end

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
    
    # Basic wall flags
    nowall = wall_settings.shape == "nowall"
    is_closed_toroidal = true

   # All of these arrays are of length mtheta with θ = [0, 1)
    mtheta = inputs.mtheta
    
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
        @info "Calculating conformal wall shape $((@sprintf "%.2e" dx)) m from plasma surface." 
        wcentr = r_major
        centerstack_min  = min(0.1, 0.1 * minimum(x_plasma))  # Avoid wall crossing R=0 axis
        for i in 1:mtheta
            j = mod1(i - 1, mtheta)
            k = mod1(i + 1, mtheta)
            # Normal vector calculation
            alph = atan(x_plasma[k] - x_plasma[j], z_plasma[j] - z_plasma[k])
            x_wall[i] = max(centerstack_min , x_plasma[i] + a * r_minor * cos(alph))
            z_wall[i] = z_plasma[i] + a * r_minor * sin(alph)
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
        !isfile(filepath) && @error "ERROR: Wall geometry file $filepath does not exist.
            Please set the wall shape parameter to a valid file path or a built-in shape (nowall, conformal, elliptical, dee, mod_dee)."

        wcentr = 0.0
        open(wall_settings.shape, "r") do io
            npots0 = parse(Int, readline(io))  # Number of points in file
            wcentr = parse(Float64, readline(io))
            readline(io) # Skip header/comment line

            (npots0 < mtheta) && @error "ERROR: $filename contains fewer points ($npots0) than mtheta ($mtheta)."

            for i in 1:mtheta
                line = split(readline(io))
                # Assumes file format: [index  R_coord  Z_coord]
                x_wall[i] = parse(Float64, line[2])
                z_wall[i] = parse(Float64, line[3])
            end
        end
    end

    # Optional: Re-parameterization
    if wall_settings.equal_arc_wall && (wall_settings.shape != "nowall")
        @info "Re-distributing wall points to equal arc length spacing"
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
function distribute_to_equal_arc_grid(xin::Vector{Float64}, zin::Vector{Float64}, mtheta::Int)

    # Temporary arrays for interpolation and arc-length calculation
    theta_in = zeros(Float64, mtheta) # Normalized input parameter [0, 1)
    theta_out  = zeros(Float64, mtheta) # New parameter distribution for equal spacing
    xout  = zeros(Float64, mtheta) # Uniformly spaced R-coordinates
    zout  = zeros(Float64, mtheta) # Uniformly spaced Z-coordinates

    # Define initial normalized parameter theta_in
    dt = 1.0 / mtheta
    theta_in .= range(start=0, length=mtheta, step=dt) # θ ∈ [0, 1)
    # we need a closed loop for arc length calculation
    mtheta2 = mtheta + 2
    xin2 = vcat(xin, xin[1:2])
    zin2 = vcat(zin, zin[1:2])
    theta_in2 = vcat(theta_in, [1.0, 1.0 + dt])
    ell   = zeros(Float64, mtheta+1) # Cumulative arc length of closed loop


    # Calculate cumulative arc length using numerical integration
    # We use a mid-point derivative approximation to find the path length
    for iw in 2:mtheta+1
        # Evaluate derivative at the midpoint of the interval
        theta = (theta_in2[iw] + theta_in2[iw - 1]) / 2.0
        
        # Calculate dx/dt and dz/dt using Lagrange interpolation (order 3)
        _, d_xin = lagrange1d(theta_in2, xin2, mtheta2, 3, theta, 1)
        _, d_zin = lagrange1d(theta_in2, zin2, mtheta2, 3, theta, 1)

        # Instantaneous speed (ds/dt)
        ds_dt = sqrt(d_xin^2 + d_zin^2)
        
        # Accumulate length: ds = (ds/dt) * dt
        ell[iw] = ell[iw - 1] + ds_dt * dt
    end

    # Re-parameterize based on equal arc-length segments
    ell_targets = range(0, step=ell[end]/mtheta, length=mtheta) # [0, Length) for open loop result
    for i in 2:mtheta
        # Find the value of 'theta_in' that corresponds to the target arc length 's'
        f_th, _ = lagrange1d(ell, theta_in, mtheta, 3, ell_targets[i], 0)
        theta_out[i] = f_th
    end

    # Interpolate the original (x,z) data at the new parameter points to get (xout, zout)
    for i in 1:mtheta
        f_x, _ = lagrange1d(theta_in, xin, mtheta, 3, theta_out[i], 0)
        f_z, _ = lagrange1d(theta_in, zin, mtheta, 3, theta_out[i], 0)
        
        xout[i] = f_x
        zout[i] = f_z
    end
    
    return xout, zout, ell, theta_out, theta_in
end