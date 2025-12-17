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
    cos_nqdelta = zeros(mtheta)
    sin_nqdelta = zeros(mtheta)
    sin_mstheta = zeros(mtheta, inputs.mpert)
    cos_mstheta = zeros(mtheta, inputs.mpert)
    sin_mstheta_arg = zeros(mtheta, inputs.mpert)
    cos_mstheta_arg = zeros(mtheta, inputs.mpert)
    for is in 1:mtheta
        theta = (is-1) * 2π / mtheta
        nqdelta = inputs.n * inputs.qa * delta[is]
        cos_nqdelta[is] = cos(nqdelta)
        sin_nqdelta[is] = sin(nqdelta)
        for l1 in 1:inputs.mpert
            mi = inputs.mlow - 1 + l1
            mitheta = mi * theta
            mitheta_arg = mi * theta + nqdelta
            sin_mstheta[is,l1] = sin(mitheta)
            cos_mstheta[is,l1] = cos(mitheta)
            sin_mstheta_arg[is,l1] = sin(mitheta_arg)
            cos_mstheta_arg[is,l1] = cos(mitheta_arg)
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

"""
    initialize_wall(inputs::VacuumInput, plasma_surf::PlasmaGeometry, wall_settings::WallShapeSettings) -> WallGeometry

Initialize the wall geometry based on the provided vacuum inputs and wall shape settings. This performs a similar
functionality to portions of the `arrays` function in the original Fortran VACUUM code. It returns a `WallGeometry`
struct containing the necessary wall surface data for vacuum calculations.
"""
function initialize_wall(inputs::VacuumInput, plasma_surf::PlasmaGeometry, wall_settings::WallShapeSettings)

   # All of these arrays are of length mtheta with θ = [0, 1)
    mtheta = inputs.mtheta
    theta_grid = range(0, stop=2π, length=mtheta + 1)[1:end-1] 
    
    # Get wall shape from wwall (TODO: this needs to be updated for size mtheta arrays)
    x_wall, z_wall, is_closed_toroidal = wwall(inputs, wall_settings, plasma_surf)

    # We need [1:mth1] below because these arrays are of size mth + 2 (for periodic finite differencing?) - try to remove this later
    # Wall boundary theta derivative
    # dx_dtheta = periodic_cubic_deriv(theta_grid, x_wall[1:inputs.mtheta])
    # dz_dtheta = periodic_cubic_deriv(theta_grid, z_wall[1:inputs.mtheta])

    input_theta = range(0, stop=2π, length=length(x_wall) + 1)[1:end-1] # length of input arrays without endpoint

    x_wall_spl = cubic_spline_interpolation(input_theta, x_wall[1:inputs.mtheta], extrapolation_bc=Interpolations.Periodic())
    z_wall_spl = cubic_spline_interpolation(input_theta, z_wall[1:inputs.mtheta], extrapolation_bc=Interpolations.Periodic())
    dx_dtheta = first.(Interpolations.gradient.(Ref(x_wall_spl), theta_grid))
    dz_dtheta = first.(Interpolations.gradient.(Ref(z_wall_spl), theta_grid))


    return WallGeometry(
        is_closed_toroidal,
        x_wall,
        z_wall,
        dx_dtheta,
        dz_dtheta
    )
end