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
    
    # Plasma boundary theta derivative (this is semi-working)
    # All of these arrays are of length mth with θ = [0, 1)
    theta_grid = range(0, stop=2π, length=mtheta + 1)[1:end-1] # length mtheta without endpoint
    xplap = periodic_cubic_deriv(theta_grid, x_plasma)
    zplap = periodic_cubic_deriv(theta_grid, z_plasma)

    # Trigonometric basis arrays
    cnqd = zeros(mtheta)
    snqd = zeros(mtheta)
    sinlt = zeros(mtheta, inputs.mpert)
    coslt = zeros(mtheta, inputs.mpert)
    snlth = zeros(mtheta, inputs.mpert)
    cslth = zeros(mtheta, inputs.mpert)
    for is in 1:mtheta
        theta = (is-1) * 2π / mtheta
        znqd = inputs.n * inputs.qa * delta[is]
        cnqd[is] = cos(znqd)
        snqd[is] = sin(znqd)
        for l1 in 1:inputs.mpert
            ll = inputs.mlow - 1 + l1
            elth = ll * theta
            elthnq = ll * theta + znqd
            sinlt[is,l1] = sin(elth)
            coslt[is,l1] = cos(elth)
            snlth[is,l1] = sin(elthnq)
            cslth[is,l1] = cos(elthnq)
        end
    end

    return PlasmaGeometry(
        x_plasma,
        z_plasma,
        delta,
        xplap,
        zplap,
        cnqd,
        snqd,
        sinlt,
        coslt,
        snlth,
        cslth
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
    dx_dtheta = periodic_cubic_deriv(theta_grid, x_wall[1:inputs.mtheta])
    dz_dtheta = periodic_cubic_deriv(theta_grid, z_wall[1:inputs.mtheta])

    return WallGeometry(
        is_closed_toroidal,
        x_wall,
        z_wall,
        dx_dtheta,
        dz_dtheta
    )
end