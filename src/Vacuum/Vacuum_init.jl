"""
    build_vacuum_globals(
        mthvac::Int,                # Number of poloidal grid points (like nths0 in Fortran)
        mpert::Int,                 # Number of Fourier harmonics (like nfm)
        settings::VacuumSettingsType,
        input::VacuumInputType
    ) -> VacuumGlobalsType

Constructs a VacuumGlobalsType, mimicking the Fortran workflow.

WIP WIP WIP WIP WIP WIP WIP WIP WIP WIP WIP WIP WIP WIP WIP WIP 

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

    #-------
    # Fortran's defglo and cardmo
    #-------
    # These are the canonical Fortran variables:
    mth = mthvac
    mth1 = mth + 1
    mth2 = mth1 + 1
    nfm = mpert
    mtot = mpert
    dth = 2π / mth


    # 
    # Mode index arrays (lmin/lmax): get from input or compute

    lmin = [input.mlow]
    lmax = [input.mhigh]
    n = input.n

    # Arrays from input
    xinf = copy(input.r)
    zinf = copy(input.z)
    delta = copy(input.delta)
    
    # TODO: need to add trans (interpolate grids from mtheta -> mthvac) in here, see readahg in Fortran

    # Physical derived parameters
    qa1 = input.qa
    ga1 = 1.0  # Placeholder, fill with correct logic as needed
    fa1 = 1.0  # Placeholder, fill with correct logic as needed

    # Wall arrays
    xwal = zeros(mth1)
    zwal = zeros(mth1)
    xwalp = zeros(mth1)
    zwalp = zeros(mth1)

    farwal = farwal || (settings.shape.a >= 10.)

    delx, delz, xwal, zwal, xwalp, zwalp,
        cnqd, snqd, sinlt, coslt, snlth, 
        cslth, xplap, zplap = setuparrays!(
            mth, dth, lmin, lmax,
            qa1, n, settings.vacdat.delfac,
            delta, input.r, input.z
        )


    
    

    return VacuumGlobalsType(
        n=n,
        mth=mth,
        mth1=mth1,
        mth2=mth2,
        nfm=nfm,
        mtot=mtot,
        lmin=lmin,
        lmax=lmax,
        xinf=xinf,
        zinf=zinf,
        delta=delta,
        xplap=xplap,
        zplap=zplap,
        qa1=qa1,
        ga1=ga1,
        fa1=fa1,
        # delx=delx,
        # delz=delz,
        xwal=xwal,
        zwal=zwal,
        xwalp=xwalp,
        zwalp=zwalp,
        dth=dth,
        wall=wall,
        farwal=farwal,
        kernelsign=kernelsign
    )
end

function setuparrays!(
    mth::Int,
    dth::Float64,
    lmin::Vector{Int},
    lmax::Vector{Int},
    q::Float64,
    n::Int,
    delfac::Float64,
    delta::Vector{Float64},
    xpla::Vector{Float64},
    zpla::Vector{Float64},
)
    # Sizes
    mth1 = mth + 1
    thgr = range(0, stop=2π, length=mth1)
    jmax1 = lmax[1] - lmin[1] + 1
    nq = n * q

    # Compute geometry bounds
    xmin = minimum(xpla)
    xmax = maximum(xpla)
    # zmin = minimum(zpla)
    # zmax = maximum(zpla)

    plrad = 0.5 * (xmax - xmin)
    delx = plrad * delfac
    delz = plrad * delfac

    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # Wall shape (placeholder: straight line wall matching plasma)
    # This is a placeholder for the wall shape, which should be replaced
    # with the actual wall shape logic in wwall.
    xwal = rand(mth1)
    zwal = rand(mth1)
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    # Derivatives (simple finite difference instead of spline)
    # Will convert to spline once equil branch is merged
    # xwalp = [ (xwal[mod1(i+1,mth)] - xwal[i]) / dth for i in 1:mth ]
    # zwalp = [ (zwal[mod1(i+1,mth)] - zwal[i]) / dth for i in 1:mth ]

    function periodic_cubic(th, vals)
        raw = interpolate(vals, BSpline(Cubic(Periodic(OnGrid()))))
        return scale(raw, th)
    end

    # Wall
    itp_x = periodic_cubic(thgr, xwal)
    itp_z = periodic_cubic(thgr, zwal)

    xwalp = first.(Interpolations.gradient.(Ref(itp_x), thgr))
    zwalp = first.(Interpolations.gradient.(Ref(itp_z), thgr))

    # Plasma boundary
    # TEMP - need to get sizes corrected between mtheta and mthvac (please rename these to mtheta_plas/mtheta_vac or something)
    thgr_plas = range(0, stop=2π, length=length(xpla))
    itp_x = periodic_cubic(thgr_plas, xpla)
    itp_z = periodic_cubic(thgr_plas, zpla)

    xplap = first.(Interpolations.gradient.(Ref(itp_x), thgr_plas))
    zplap = first.(Interpolations.gradient.(Ref(itp_z), thgr_plas))

    push!(xplap, xwalp[1])
    push!(zplap, zwalp[1])

    # Allocate arrays
    cnqd = zeros(mth1)
    snqd = zeros(mth1)
    sinlt = zeros(mth1, jmax1)
    coslt = zeros(mth1, jmax1)
    snlth = zeros(mth1, jmax1)
    cslth = zeros(mth1, jmax1)

    # Trigonometric basis arrays
    # UNCOMMENT ONCE WE HAVE THE RIGHT SIZES
    # for is in 1:mth1
    #     theta = (is-1) * dth
    #     znqd = nq * delta[is]
    #     cnqd[is] = cos(znqd)
    #     snqd[is] = sin(znqd)
    #     for l1 in 1:jmax1
    #         ll = lmin[1] - 1 + l1
    #         elth = ll * theta
    #         elthnq = ll * theta + znqd
    #         sinlt[is,l1] = sin(elth)
    #         coslt[is,l1] = cos(elth)
    #         snlth[is,l1] = sin(elthnq)
    #         cslth[is,l1] = cos(elthnq)
    #     end
    # end

    return (
        delx, delz, xwal, zwal, xwalp, zwalp,
        cnqd, snqd, sinlt, coslt, snlth, cslth, xplap, zplap
    )
end
