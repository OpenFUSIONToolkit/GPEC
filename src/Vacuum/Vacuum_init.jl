"""
    initialize_plasma_surface(inputs::VacuumInput) -> PlasmaGeometry

Initialize the plasma surface geometry based on the provided vacuum inputs. This
function performs some of the functionality within `readahg`, `arrays`, and `funint`
in the original Fortran VACUUM code. It returns a `PlasmaGeometry` struct containing
the necessary plasma surface data for vacuum calculations.

First, we interpolate the input plasma boundary arrays onto the mthvac grid. Then, we compute
the derivatives of the plasma boundary with respect to the poloidal angle θ using
periodic cubic spline differentiation. Finally, we compute the trigonometric basis functions
needed for the fourier calculations later in the code.
"""
function initialize_plasma_surface(inputs::VacuumInput)

    # Interpolate arrays from input onto mthvac grid (in readahg in the Fortran)
    mtheta = inputs.mtheta
    x_plasma = interp_to_new_grid(inputs.r, mtheta)
    z_plasma = interp_to_new_grid(inputs.z, mtheta)
    delta = interp_to_new_grid(inputs.delta, mtheta)
    # input_theta = range(0, stop=2π, length=length(inputs.r) + 1)[1:end-1] # length of input arrays without endpoint
    
    # x_plasma_spl = cubic_spline_interpolation(input_theta, inputs.r, extrapolation_bc=Interpolations.Periodic())
    # z_plasma_spl = cubic_spline_interpolation(input_theta, inputs.z, extrapolation_bc=Interpolations.Periodic())

    
    # x_plasma = x_plasma_spl(range(0, stop=2π, length=mtheta + 1)[1:end-1])
    # z_plasma = z_plasma_spl(range(0, stop=2π, length=mtheta + 1)[1:end-1])
    # delta = cubic_spline_interpolation(input_theta, inputs.delta, extrapolation_bc=Interpolations.Periodic())(range(0, stop=2π, length=mtheta + 1)[1:end-1])
    
    # Plasma boundary theta derivative (this is semi-working)
    # All of these arrays are of length mth with θ = [0, 1)
    theta_grid = range(0, stop=2π, length=mtheta + 1)[1:end-1] # length mtheta without endpoint
    dx_plasma_dtheta = periodic_cubic_deriv(theta_grid, x_plasma)
    dz_plasma_dtheta = periodic_cubic_deriv(theta_grid, z_plasma)
    # dx_plasma_dtheta = (t -> Interpolations.gradient(cubic_spline_interpolation(theta_grid, x_plasma, extrapolation_bc=Interpolations.Periodic()), t)).(theta_grid)
    # dz_plasma_dtheta = (t -> Interpolations.gradient(cubic_spline_interpolation(theta_grid, z_plasma, extrapolation_bc=Interpolations.Periodic()), t)).(theta_grid)
    # dx_plasma_dtheta = first.(Interpolations.gradient.(Ref(x_plasma_spl), theta_grid))
    # dz_plasma_dtheta = first.(Interpolations.gradient.(Ref(z_plasma_spl), theta_grid))
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

"""
    initialize_wall(inputs::VacuumInput, plasma_surf::PlasmaGeometry, wall_settings::WallShapeSettings) -> WallGeometry

Initialize the wall geometry based on the provided vacuum inputs and wall shape settings. This performs a similar
functionality to portions of the `arrays` function in the original Fortran VACUUM code. It returns a `WallGeometry`
struct containing the necessary wall surface data for vacuum calculations.
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
        @info "Loading wall shape from external file 'wall_geo.in'."
        # Load wall geometry from external file "wall_geo.in"
        wcentr = 0.0
        open("wall_geo.in", "r") do io
            npots0 = parse(Int, readline(io))  # Number of points in file
            wcentr = parse(Float64, readline(io)) 
            readline(io) # Skip header/comment line

            if npots0 < mtheta
                @error "ERROR: wall_geo.in contains fewer points ($npots0) than mtheta ($mtheta)."
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
    if wall_settings.ldistribute_to_equal_arc_grid == 1
        x_wall, z_wall, _, _, _ = distribute_to_equal_arc_grid(x_wall, z_wall, mtheta)
    end

    # Save results
    @info "Saving wall shape to 'wall_shape.h5'."
    h5open("wall_shape.h5", "w") do file
        attributes(file)["x_center"] = wcentr
        write(file, "x", x_wall)
        write(file, "z", z_wall)
    end

    # Compute wall derivatives
    input_theta = range(0, stop=2π, length=length(x_wall) + 1)[1:end-1] # length of input arrays without endpoint

    x_wall_spl = cubic_spline_interpolation(input_theta, x_wall[1:inputs.mtheta], extrapolation_bc=Interpolations.Periodic())
    z_wall_spl = cubic_spline_interpolation(input_theta, z_wall[1:inputs.mtheta], extrapolation_bc=Interpolations.Periodic())
    dx_dtheta = first.(Interpolations.gradient.(Ref(x_wall_spl), theta_grid))
    dz_dtheta = first.(Interpolations.gradient.(Ref(z_wall_spl), theta_grid))


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

This function performs arc length re-parameterization of a 2D curve. It takes an
input curve defined by `(xin, zin)` coordinates and re-samples it such that
the new points `(xout, zout)` are equally spaced in arc length.

# Arguments
- `xin::AbstractVector{Float64}`: Array of x-coordinates of the input curve.
- `zin::AbstractVector{Float64}`: Array of z-coordinates of the input curve.
- `mw1::Int`: Number of points in the input and output curves.

# Returns
- `xout::Vector{Float64}`: Array of x-coordinates of the arc-length re-parameterized curve.
- `zout::Vector{Float64}`: Array of z-coordinates of the arc-length re-parameterized curve.
- `ell::Vector{Float64}`: Array of cumulative arc lengths for the input curve.
- `thgr::Vector{Float64}`: Array of re-parameterized 'theta' values corresponding to equal arc lengths.
- `thlag::Vector{Float64}`: Array of normalized 'theta' values for the input curve (0 to 1).
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
