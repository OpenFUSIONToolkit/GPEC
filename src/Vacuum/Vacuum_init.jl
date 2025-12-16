"""
    build_vacuum_globals(
        mthvac::Int,                # Number of poloidal grid points (like nths0 in Fortran)
        mpert::Int,                 # Number of Fourier harmonics (like nfm)
        wall::Bool,                 # Wall flag
        farwal::Bool,               # Far wall flag
        kernelsign::Float64,        # Kernel sign
        settings::VacuumSettingsType, # Settings parsed from vac.toml
        input::VacuumInputType      # Vacuum input data from DCON
    ) -> VacuumGlobalsType

Constructs a VacuumGlobalsType, mimicking the Fortran workflow.
Includes functionality from defglo, cardmo, dskmld, and readahg.

This might not need to be a separate function?
"""
function build_vacuum_globals(
    mthvac::Int,
    mpert::Int,
    wall::Bool,
    farwal::Bool,
    kernelsign::Float64,
    settings::VacuumSettingsType,
    input::VacuumInputType,
)
    # Interpolate arrays from input onto mthvac grid (in readahg in the Fortran)
    xinf = interp_to_new_grid(input.r, mthvac)
    zinf = interp_to_new_grid(input.z, mthvac)
    delta = interp_to_new_grid(input.delta, mthvac)

    farwal = farwal || (settings.wall.a >= 10.)

    return VacuumGlobalsType(
        n=input.n,
        mth=mthvac,
        mth1=mthvac + 1,
        mth2=mthvac + 2,
        nfm=mpert,
        mtot=mpert,
        lmin=[input.mlow],
        lmax=[input.mhigh],
        xpla=xinf,
        zpla=zinf,
        delta=delta,
        qa1=input.qa,
        ga1=1.0,  # Placeholder, fill with correct logic as needed
        fa1=1.0,  # Placeholder, fill with correct logic as needed
        dth=2π / mthvac,
        wall=wall,
        farwal=farwal,
        kernelsign=kernelsign
    )
end

"""
    setuparrays!(globals::VacuumGlobalsType, settings::VacuumSettingsType)

Julia implementation of the Fortran `arrays`` function. Sets up geometric arrays and updates the `globals` struct in-place.
It computes the wall shape and its derivatives, as well as the plasma boundary derivatives.
Returns delx, delz, cnqd, snqd, sinlt, coslt, snlth, cslth.

This function will need checking against the Fortran version to ensure correctness. There are many complex indexing
considerations that might be able to be avoided by just using periodic splines in Julia.
"""
function setuparrays!(globals::VacuumGlobalsType, settings::VacuumSettingsType)
    
    # Sizes
    theta_grid = range(0, stop=2π, length=globals.mth1)
    jmax1 = globals.lmax[1] - globals.lmin[1] + 1 # this is just mpert from DCON, yeah? rename
    nq = globals.n * globals.qa1
    
    # Compute geometric quantities
    plrad = 0.5 * (maximum(globals.xpla) - minimum(globals.xpla)) # plasma radius (rename)
    delx = plrad * settings.vacdat.delfac # not used yet?
    delz = plrad * settings.vacdat.delfac

    # Get wall shape from wwall (these are of length mth + 2)
    globals.xwal, globals.zwal = wwall(settings, globals)

    # We need [1:mth1] below because these arrays are of size mth + 2 (for periodic finite differencing?) - try to remove this later
    # Wall boundary theta derivative
    globals.xwalp = periodic_cubic_deriv(theta_grid, globals.xwal[1:globals.mth1])
    globals.zwalp = periodic_cubic_deriv(theta_grid, globals.zwal[1:globals.mth1])
    globals.xwalp[globals.mth1] = globals.xwalp[1] # enforce periodicity?
    globals.zwalp[globals.mth1] = globals.zwalp[1]

    # Plasma boundary theta derivative
    globals.xplap = periodic_cubic_deriv(theta_grid, globals.xpla[1:globals.mth1])
    globals.zplap = periodic_cubic_deriv(theta_grid, globals.zpla[1:globals.mth1])

    # Allocate arrays
    globals.cnqd = zeros(globals.mth1)
    globals.snqd = zeros(globals.mth1)
    globals.sinlt = zeros(globals.mth1, jmax1)
    globals.coslt = zeros(globals.mth1, jmax1)
    globals.snlth = zeros(globals.mth1, jmax1)
    globals.cslth = zeros(globals.mth1, jmax1)

    # TODO: add cplar/cwallr loop here if needed

    # Trigonometric basis arrays
    for is in 1:globals.mth1
        theta = (is-1) * globals.dth
        znqd = nq * globals.delta[is]
        globals.cnqd[is] = cos(znqd)
        globals.snqd[is] = sin(znqd)
        for l1 in 1:jmax1
            ll = globals.lmin[1] - 1 + l1
            elth = ll * theta
            elthnq = ll * theta + znqd
            globals.sinlt[is,l1] = sin(elth)
            globals.coslt[is,l1] = cos(elth)
            globals.snlth[is,l1] = sin(elthnq)
            globals.cslth[is,l1] = cos(elthnq)
        end
    end
end