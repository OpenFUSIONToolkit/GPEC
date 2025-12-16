# """
#     build_vacuum_globals(
#         mthvac::Int,                # Number of poloidal grid points (like nths0 in Fortran)
#         mpert::Int,                 # Number of Fourier harmonics (like nfm)
#         wall::Bool,                 # Wall flag
#         farwal::Bool,               # Far wall flag
#         kernelsign::Float64,        # Kernel sign
#         wall_settings::WallShapeSettings, # Settings parsed from toml
#         input::VacuumInputType      # Vacuum input data from DCON
#     ) -> VacuumGlobalsType

# Constructs a VacuumGlobalsType, mimicking the Fortran workflow.
# Includes functionality from defglo, cardmo, dskmld, and readahg.

# This might not need to be a separate function?
# """
# function build_vacuum_globals(
#     mthvac::Int,
#     mpert::Int,
#     wall::Bool,
#     farwal::Bool,
#     kernelsign::Float64,
#     wall_settings::WallShapeSettings,
#     input::VacuumInputType,
# )
#     # Interpolate arrays from input onto mthvac grid (in readahg in the Fortran)
#     xinf = interp_to_new_grid(input.r, mthvac)
#     zinf = interp_to_new_grid(input.z, mthvac)
#     delta = interp_to_new_grid(input.delta, mthvac)

#     open("xpla_zpla_julia.out", "w") do io
#         println(io, "# index\t xpla\t zpla")
#         n = max(length(xinf), length(zinf))
#         for i in 1:n
#             xv = i <= length(xinf) ? xinf[i] : NaN
#             zv = i <= length(zinf) ? zinf[i] : NaN
#             println(io, "$(i)\t$(xv)\t$(zv)")
#         end
#     end

#     farwal = farwal || (wall_settings.a >= 10.)

#     return VacuumGlobalsType(
#         n=input.n,
#         mth=mthvac,
#         mth1=mthvac + 1,
#         mth2=mthvac + 2,
#         nfm=mpert,
#         mtot=mpert,
#         lmin=[input.mlow],
#         lmax=[input.mhigh],
#         xpla=xinf,
#         zpla=zinf,
#         delta=delta,
#         qa1=input.qa,
#         ga1=1.0,  # Placeholder, fill with correct logic as needed
#         fa1=1.0,  # Placeholder, fill with correct logic as needed
#         dth=2π / mthvac,
#         wall=wall,
#         farwal=farwal,
#         kernelsign=kernelsign
#     )
# end

"""
    setuparrays!(globals::VacuumGlobalsType, wall_settings::WallShapeSettings)

Julia implementation of the Fortran `arrays`` function. Sets up geometric arrays and updates the `globals` struct in-place.
It computes the wall shape and its derivatives, as well as the plasma boundary derivatives.
Returns delx, delz, cnqd, snqd, sinlt, coslt, snlth, cslth.

This function will need checking against the Fortran version to ensure correctness. There are many complex indexing
considerations that might be able to be avoided by just using periodic splines in Julia.
"""
# function setuparrays!(globals::VacuumGlobalsType, wall_settings::WallShapeSettings)
    
#     # Sizes
#     jmax1 = globals.lmax[1] - globals.lmin[1] + 1 # this is just mpert from DCON, yeah? rename
#     nq = globals.n * globals.qa1
    
#     # Compute geometric quantities
#     plrad = 0.5 * (maximum(globals.xpla) - minimum(globals.xpla)) # plasma radius (rename)
#     delx = plrad * wall_settings.delfac # not used yet?
#     delz = plrad * wall_settings.delfac

#     # TODO: Wall arrays lengths are not debugged - these should become length mth like the plasma
#     theta_grid = range(0, stop=2π, length=globals.mth1)
#     # Get wall shape from wwall (these are of length mth + 2)
#     globals.xwal, globals.zwal = wwall(wall_settings, globals)
#     # We need [1:mth1] below because these arrays are of size mth + 2 (for periodic finite differencing?) - try to remove this later
#     # Wall boundary theta derivative
#     globals.xwalp = periodic_cubic_deriv(theta_grid, globals.xwal[1:globals.mth1])
#     globals.zwalp = periodic_cubic_deriv(theta_grid, globals.zwal[1:globals.mth1])
#     globals.xwalp[globals.mth1] = globals.xwalp[1] # enforce periodicity?
#     globals.zwalp[globals.mth1] = globals.zwalp[1]

#     # Plasma boundary theta derivative (this is semi-working)
#     # All of these arrays are of length mth with θ = [0, 1)
#     theta_grid = range(0, stop=2π, length=globals.mth1)[1:end-1] # length mth
#     globals.xplap = periodic_cubic_deriv(theta_grid, globals.xpla)
#     globals.zplap = periodic_cubic_deriv(theta_grid, globals.zpla)

#     open("xplap_zplap_julia.out", "w") do io
#         println(io, "# index\t xplap\t zplap")
#         n = max(length(globals.xplap), length(globals.zplap))
#         for i in 1:n
#             xv = i <= length(globals.xplap) ? globals.xplap[i] : NaN
#             zv = i <= length(globals.zplap) ? globals.zplap[i] : NaN
#             println(io, "$(i)\t$(xv)\t$(zv)")
#         end
#     end

#     # Allocate arrays
#     globals.cnqd = zeros(globals.mth1)
#     globals.snqd = zeros(globals.mth1)
#     globals.sinlt = zeros(globals.mth1, jmax1)
#     globals.coslt = zeros(globals.mth1, jmax1)
#     globals.snlth = zeros(globals.mth1, jmax1)
#     globals.cslth = zeros(globals.mth1, jmax1)

#     # TODO: add cplar/cwallr loop here if needed

#     # Trigonometric basis arrays
#     for is in 1:globals.mth
#         theta = (is-1) * globals.dth # 2π / mtheta
#         znqd = nq * globals.delta[is]
#         globals.cnqd[is] = cos(znqd)
#         globals.snqd[is] = sin(znqd)
#         for l1 in 1:jmax1
#             ll = globals.lmin[1] - 1 + l1
#             elth = ll * theta
#             elthnq = ll * theta + znqd
#             globals.sinlt[is,l1] = sin(elth)
#             globals.coslt[is,l1] = cos(elth)
#             globals.snlth[is,l1] = sin(elthnq)
#             globals.cslth[is,l1] = cos(elthnq)
#         end
#     end
# end

function initialize_plasma_surface(inputs::VacuumInputType)

    mtheta = inputs.mtheta

    # Interpolate arrays from input onto mthvac grid (in readahg in the Fortran)
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

function initialize_wall(inputs::VacuumInputType, wall_settings::WallShapeSettings, plasma_surf::PlasmaGeometry)

    mtheta = inputs.mtheta

   # All of these arrays are of length mtheta with θ = [0, 1)
    theta_grid = range(0, stop=2π, length=mtheta + 1)[1:end-1] 
    
    # Get wall shape from wwall (these are of length mth + 2)
    x_wall, z_wall, is_closed_toroidal = wwall(inputs, wall_settings, plasma_surf)

    # We need [1:mth1] below because these arrays are of size mth + 2 (for periodic finite differencing?) - try to remove this later
    # Wall boundary theta derivative
    dx_dtheta = periodic_cubic_deriv(theta_grid, x_wall[1:inputs.mtheta])
    dz_dtheta = periodic_cubic_deriv(theta_grid, z_wall[1:inputs.mtheta])

    return WallGeometry(
        is_closed_toroidal
        x_wall,
        z_wall,
        dx_dtheta,
        dz_dtheta
    )
end