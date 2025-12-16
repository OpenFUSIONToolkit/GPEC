# VERY MUCH A WORK IN PROGRESS, NOTHING HERE WILL WORK YET
# I just pulled this relevant stuff from vac_vaccal.ipynb and threw it into this file
# and commented out the wall/debug stuff for now

# Gaussian quadrature weights and points for 8-point integration
const WGAUS = [0.101228536290376, 0.222381034453374, 0.313706645877887, 0.362683783378362,
               0.362683783378362, 0.313706645877887, 0.222381034453374, 0.101228536290376]

const XGAUS = [-0.960289856497536, -0.796666477413627, -0.525532409916329, -0.183434642495650,
                0.183434642495650,  0.525532409916329,  0.796666477413627,  0.960289856497536]


function vaccal!(inputs::VacuumInputType, plasma_surf::PlasmaGeometry, wall::WallGeometry, wall_settings::WallShapeSettings)

    # Initialization
    factpi  = 2π
    jmax    = 2*inputs.mhigh + 1
    jmax1   = inputs.mpert
    lmax1   = inputs.mhigh + 1
    ln      = inputs.mlow
    lx      = inputs.mhigh
    jdel    = 8
    nq      = inputs.n * inputs.qa
    tmth    = 2 * inputs.mtheta
    mthsq   = tmth * tmth
    lmth    = tmth * 2 * jmax1
    j1v     = inputs.mpert
    j2v     = inputs.mpert

    grri = zeros(Float64, 2 * (inputs.mtheta + 5), 2 * inputs.mpert)
    grdgre = zeros(Float64, 2 * (inputs.mtheta + 5), 2 * (inputs.mtheta + 5))
    grpp = zeros(Float64, 2 * (inputs.mtheta + 5), 2 * (inputs.mtheta + 5))

    # ----------------------------------------------------------
    # Apply wall boundary conditions
    # ----------------------------------------------------------
    # TODO: does this need to get called more than once? Currently calling it three separate times
    # globals.xwal, globals.zwal = wwall(wall_settings, globals)

    # ----------------------------------------------------------
    # Plasma–Plasma block
    # ----------------------------------------------------------
    j1, j2 = 1, 1
    ksgn = 2*j2 - 3
    kernel!(grdgre, grpp, plasma_surf.x, plasma_surf.z, plasma_surf.x, plasma_surf.z, j1, j2, ksgn, 1, 1, false, inputs, wall)

    # Enforce periodic boundary conditions
    for i = 1:inputs.mtheta + 2
        grpp[i,inputs.mtheta + 1] = grpp[i,1]
        grpp[i,inputs.mtheta + 2] = grpp[i,2]
    end

    # Fourier transform plasma-plasma block
    fourier_transform!(grri, grpp, plasma_surf.cslth, 0, 0, inputs.mlow, inputs.mhigh, inputs.mtheta)
    fourier_transform!(grri, grpp, plasma_surf.snlth, 0, inputs.mpert, inputs.mlow, inputs.mhigh, inputs.mtheta)

    if !inputs.farwall_flag
        # ----------------------------------------------------------
        # Plasma–Wall block
        # ----------------------------------------------------------
        error("Haven't set up walls yet")
        j1, j2 = 1, 2
        ksgn = 2*j2 - 3
        grpw_block = similar(grdgre) # This should be grpp or a new matrix
        kernel!(grdgre, grpw_block, globals.xpla, globals.zpla, globals.xwal, globals.zwal, j1, j2, ksgn, 1, 1, true, wall)
        
        fourier_transform!(grpw_block, grdgre, cslth, 0, 0, lmin, lmax, mth)
        fourier_transform!(grpw_block, grdgre, snlth, 0, jmax1, lmin, lmax, mth)

        # ----------------------------------------------------------
        # Wall–Plasma block
        # ----------------------------------------------------------
        j1, j2 = 2, 1
        ksgn = 2*j2 - 3
        grwp_block = similar(grdgre) # This should be grpp or a new matrix
        kernel!(grdgre, grwp_block, globals.xwal, globals.zwal, globals.xpla, globals.zpla, j1, j2, ksgn, 1, 1, true, globals, wall)

        fourier_transform!(grwp_block, grdgre, cslth, 0, 0, lmin, lmax, mth)
        fourier_transform!(grwp_block, grdgre, snlth, 0, jmax1, lmin, lmax, mth)

        # ----------------------------------------------------------
        # Wall–Wall block
        # ----------------------------------------------------------
        j1, j2 = 2, 2
        ksgn = 2*j2 - 3
        grww_block = similar(grdgre) # This should be grpp or a new matrix
        kernel!(grdgre, grww_block, globals.xwal, globals.zwal, globals.xwal, globals.zwal, j1, j2, ksgn, 1, 1, true, globals, wall)

        fourier_transform!(grww_block, grdgre, cslth, 0, 0, lmin, lmax, mth)
        fourier_transform!(grww_block, grdgre, snlth, 0, jmax1, lmin, lmax, mth)

        # ----------------------------------------------------------
        # Assemble matrices for solving
        # ----------------------------------------------------------
        assemble_vacuum_matrix!(
            vacmat, vacmti, vacmatu, vacmtiu,
            grwp, grpw_block, grwp_block, grww_block,
            mth, jmax1, ln, lx, 2π
        )
    end

    # TODO: is this getting kept?
    # Add cn0 to make grdgre nonsingular for n=0 modes
    if (abs(inputs.n) <= 1e-5) && (!inputs.farwall_flag) && (wall.is_closed_toroidal)
        mth12 = inputs.farwall_flag ? inputs.mtheta : 2 * inputs.mtheta
        for i in 1:mth12, j in 1:mth12
            grdgre[i, j] += wall_settings.cn0
        end
    end

    # Only needed for mutual inductance with the wall calculations
    if inputs.kernelsign < 0
        grdgre .*= inputs.kernelsign
        # Account for factor of 2 in diagonal terms in eq. 90 of Chance
        for i in 1:2 * (inputs.mtheta + 5)
            grdgre[i, i] += 2.0
        end
    end

    # Invert the plasma response system of equations, eqs. 92-94ish of Chance 1997 (gelimb in Fortran)
    grri .= grdgre \ grri

    # TODO: I am not sure why we recall wwall here again? This is the third time...
    # globals.xwal, globals.zwal = wwall(wall_settings, globals)

    # There's some logic that computes xpass/zpass and chiwc/chiws here, might eventually be needed?

    # Perform inverse Fourier transforms to get response matrix components (eq. 115-118 of Chance 2007)
    arr = zeros(jmax1, jmax1)
    aii = zeros(jmax1, jmax1)
    ari = zeros(jmax1, jmax1)
    air = zeros(jmax1, jmax1)
    fourier_inverse_transform!(arr, grri, plasma_surf.cslth, 0, 0, inputs.mtheta, inputs.mlow, inputs.mhigh)
    fourier_inverse_transform!(aii, grri, plasma_surf.snlth, 0, inputs.mpert, inputs.mtheta, inputs.mlow, inputs.mhigh)
    fourier_inverse_transform!(ari, grri, plasma_surf.snlth, 0, 0, inputs.mtheta, inputs.mlow, inputs.mhigh)
    fourier_inverse_transform!(air, grri, plasma_surf.cslth, 0, inputs.mpert, inputs.mtheta, inputs.mlow, inputs.mhigh)

    # Final form of vacuum response matrix (eq. 114 of Chance 2007)
    vacmat = arr .+ aii
    vacmti = air .- ari
    # Force symmetry of response matrix if desired
    if inputs.force_wv_symmetry
        for l1 in 1:jmax1
            for l2 in l1:jmax1
                vacmat[l1, l2] = 0.5 * (vacmat[l1, l2] + vacmat[l2, l1])
                vacmti[l1, l2] = 0.5 * (vacmti[l1, l2] - vacmti[l2, l1])
            end
        end
    end
    wv = complex.(vacmat, vacmti)

    println("WV from Julia")
    display(wv)

    # There was an extra arrays call here in the Fortran - do we need any functionality from it here?
    
    # Fortran check2 data dump
    # todo clean up our own wall/plasma geometry outputs instead
    if false
        open("julia_vaccal_arrays.out", "w") do io
            println(io, "I, xp, zp, xw, zw, xpp, zpp, xwp, zwp =")
            # Derivatives (xplap, etc.) are not readily available here yet.
            # Writing NaN as placeholders.
            for i in 1:8:globals.mth1
                @printf(io, "%3d %13.2f %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f\n",
                        i, globals.xpla[i], globals.zpla[i], globals.xwal[i], globals.zwal[i],
                        NaN, NaN, NaN, NaN)
            end
        end
    end
end

"""
    kernel(xobs, zobs, xsource, zsce, j1, j2, isgn, iopw, iops, ischk, params)

Compute kernels of integral equation for Laplace's equation for a torus.

# Arguments
- `xobs`: Observer x coordinates
- `zobs`: Observer z coordinates  
- `xsource`: Source x coordinates
- `zsource`: Source z coordinates
- `j1, j2`: Boundary condition indices
- `isgn`: Sign parameter
- `iopw`: Wall option (0=inactive, 1=active)
- `iops`: Log singularity correction option
- `wallflag`: Check option for conductor position (ischk)
- `params`: Dictionary containing simulation parameters

# Returns
- `gradgreensfunction`: Gradient Green's function matrix
- `greensfunction`: Green's function matrix
"""
function kernel!(gradgreensfunction, greensfunction, x_obspoints, z_obspoints, x_sourcepoints, z_sourcepoints, j1, j2, isgn, iopw, iops, wallflag, inputs::VacuumInputType, wall_geo::WallGeometry)

    mth = inputs.mtheta
    mth1 = inputs.mtheta + 1
    dth = 2π / mth
    ak0i = 0.0
    jres = 1
    N_obs = length(x_obspoints)
    thetas = LinRange(0, mth*dth, mth)
    
    if N_obs != length(z_obspoints) || N_obs != length(x_sourcepoints) || N_obs != length(z_sourcepoints)
        error("Length of input arrays (xobs, zobs, xsource, zsce) are different. All length should be the same")
    end

    # matrix output greensfunction is accumulated in grwp of vaccal.
    # While grwp is 𝒢 befor fourier transform, grri is fourier transformed 𝒢

    # 1. definition for solving parameters
    wsimpb1=1*dth/3
    wsimpb2=2*dth/3
    wsimpb4=4*dth/3

    wsimpa1=1*dth/3
    wsimpa2=2*dth/3
    wsimpa4=4*dth/3

    algdth = log(dth) # log of dth
    alg = log(2*dth)

    slog1m = 3*dth*(algdth-1/3)
    slog1p = slog1m

    alg0=16.0*dth*(alg-68.0/15.0)/15.0
    alg1=128.0*dth*(alg-8.0/15.0)/45.0
    alg2=4.0*dth*(7.0*alg-11.0/15.0)/45.0

    isph = 0 # isph needs to be initialized

    # 2. check singular points for conductor.
    if wallflag == true
        # 2.0 initialize jbot and jtop, and get wall geometry from globals
        jbot=mth/2+1
        jtop=mth/2+1
        ww1 = globals.xwal
        ww2 = globals.zwal
        
        # 2.2 find where sign of wall r point acrosses zero. 
        # isph means there is 0-corssing point 
        for i in 1:mth
            if ww1[i] * ww1[i+1] <= 0.0
                jbot = ww1[i] > 0.0 ? i : jbot
                jtop = ww1[i] < 0.0 ? i + 1 : jtop
                isph = 1 
            end
        end
    end

    # 3 do spline and calc derivative for Z'_θ and X'_θ in eq.(51)

    # This doesn't work for me.. I'll use Iterpolations.jl for here temporary. WIP - Have to fix spline1d_deriv funciton in vacuum_math
    # xpr = [spline1d_deriv(the, xsource, θ) for θ in thetas]
    # zpr = [spline1d_deriv(the, zsce, θ) for θ in thetas]

    # Using Interpolations.jl, we can create periodic cubic splines
    itp_x = cubic_spline_interpolation(thetas, x_sourcepoints)
    itp_z = cubic_spline_interpolation(thetas, z_sourcepoints)

    gradients_x = (t -> Interpolations.gradient(itp_x, t)).(thetas)
    gradients_z = (t -> Interpolations.gradient(itp_z, t)).(thetas)
    xpr = first.(gradients_x) # d x / d theta
    zpr = first.(gradients_z) # d z / d theta



    # 4, begin obs loop.
    for j in 1:mth 

        # 4.1 initialize variable
        x_obs=x_obspoints[j] #observation point  
        z_obs=z_obspoints[j]
        theta_obs=thetas[j] # theta value  
        work = zeros(mth1)
        
        # 4.2 if the point of observation point is in negative, We cannot use green func
        # This is same for source point 
        if x_obs < 0.0
            if j2 == 2
                work[j] = 1.0
            end
            continue
        end
            
        # 4.3 set istart and iend
        iend = 2 # end point of integration
        aval1=0.0 # ∇𝒢_0
        # if wall crossing zero and wall is source, iend set.
        if isph == 1 && j2 == 2
            if jbot - j == 1
                iend = 3
            elseif jbot - j == 0
                iend = 4
            elseif j - jtop == 0
                iend = 0
            elseif j - jtop == 1
                iend = 1
            end
        end
        istart = 4 - iend # starting point of integration

        # 4-3 on mth. then mths is equals to mtheta+3 ??? I'm not sure
        mth_source=mth-(istart+iend-1) 

        
        # 4.4 loop for source point 
        for i in 1:mth_source

            # 4.5 get source point index(ic) theta(theta), X(xt), and Z(zt)
            ic = i + j + istart - 1
            if ic ≥ mth1
                ic = ic - mth
            end
            theta_source=(ic-1)*dth 
            x_source=x_sourcepoints[ic]  
            z_source=z_sourcepoints[ic]  

            # 4.6 if source point is in negative, we cannot use green function
            # & if source point(ic) and obs point (j) is same, it's singular
            if x_source < 0 
                continue 
            end
            if ic == j 
                continue 
            end

            # 4.7 calc X'_θ (xtp) and Z'_θ (ztp) and call green function
            # aval is 𝒥 ∇'𝒢ⁿ∇'ℒ, bval is 2pi𝒢ⁿ. aval0 is 𝒥 ∇'𝒢ⁿ∇'ℒ for n=0
            xtp=xpr[ic]  
            ztp=zpr[ic]  
            G, aval, aval0, bval = green(x_obs,z_obs,x_source,z_source,xtp,ztp,inputs.n,usechancebugs=false)

            # 4.8 simpson integral. 4 for odd, 2 for even, and 1 for others.
            wsimpb=wsimpb2
            if (i÷2)*2 == i  # Fixed: / to ÷ for integer division
                wsimpb=wsimpb4
            end
            if (i == 1)||(i == mth_source)
                wsimpb=wsimpb1
            end
            wsimpa=wsimpa2
            if (i÷2)*2 == i  # Fixed: / to ÷ for integer division
                wsimpa=wsimpa4
            end
            if (i == 1)||(i == mth_source)
                wsimpa=wsimpa1
            end

            # 4.9 work and gren
            # work : simpson integral for aval(𝒥 ∇'𝒢ⁿ∇'ℒ)
            # gren : log singularity values accumulated. simpson integral for bval
            # aval1 : aval1
            work[ic]=work[ic]+isgn*aval*wsimpa  
            greensfunction[j,ic]=greensfunction[j,ic]+bval*wsimpb # integral of bval (log singularity)  
            aval1 = aval1 + aval0 * wsimpa
        
        end
        
        # 5.1 if it's plasma/plsama, wall/wall and in negative wall point, skip loop
        # obs : j1 = 1(plasma), wall(2)
        # src : j1 = 1(plasma), wall(2)
        if (j1+j2) != 2 && isph == 1 && j > jbot && j < jtop
            continue
        end

        # 5.2 Get js
        # js2 : j - iend + 1
        js1=mod(j-iend+mth-1,mth)+1
        js2=mod(j-iend+mth,mth)+1
        js3=mod(j-iend+mth+1,mth)+1
        js4=mod(j-iend+mth+2,mth)+1
        js5=mod(j-iend+mth+3,mth)+1

        # 6 Singular points when source point and obs point are the same
        # 6.1 integration each left and right
        for ilr in [1,2]
            xl = theta_obs + (2*ilr-iend-2)*dth
            xu = xl + 2 * dth
            agaus = (xu + xl)/2
            bgaus = (xu - xl)/2
            tgaus = agaus .+ XGAUS .* bgaus # tgaus is 8 point gauss points, since xgauss is for only [-1,1]  # Fixed: xgaus to XGAUS
            # 6.2 for each 8 gaussian points
            for ig in 1:8
                tgaus0 = tgaus[ig] #i-th value of for 8 points, in theta  
                tgaus0 = mod(tgaus0, 2π)

                # 6.3 get X, X', Z, Z' for gaussian point
                xt = itp_x(tgaus0)
                xtp = Interpolations.gradient(itp_x, tgaus0)[1]
                zt = itp_z(tgaus0)
                ztp = Interpolations.gradient(itp_z, tgaus0)[1]

                # 6.4 call green function
                G, aval, aval0, bval = green(x_obs,z_obs,xt,zt,xtp,ztp,inputs.n,usechancebugs=false)

                # 6.5 add logarithm on G (not 𝒢_n). Chance eq.(75)
                # iops = 1
                bval = G + iops * log((theta_obs-tgaus[ig])^2)/x_obs  

                # 6.6 calc wgaus. bgaus refers Δ in theta. wgaus is weight for each 8 points
                wgbg=WGAUS[ig]*bgaus  # Fixed: wgaus() to WGAUS[]

                # 6.7 calc pgaus. below Chance eq.(77)
                pgaus=(tgaus[ig]-theta_obs-(2-iend)*dth)/dth  
                pgaus2=pgaus*pgaus
                amm = (pgaus2-1)*pgaus*(pgaus-2)/24.0 *wgbg
                am = -(pgaus-1)*pgaus*(pgaus2-4)/6.0 *wgbg
                a0 = (pgaus2-1)*(pgaus2-4)/4.0 *wgbg
                ap = -(pgaus+1)*pgaus*(pgaus2-4)/6.0 *wgbg
                app = (pgaus2-1)*pgaus*(pgaus+2)/24.0 *wgbg

                # 6.8 add up in work
                work[js1] += isgn * aval * amm  
                work[js2] += isgn * aval * am  
                work[js3] += isgn * aval * a0  
                work[js4] += isgn * aval * ap  
                work[js5] += isgn * aval * app  

                # 6.9 minus diverging value
                work[j] -= isgn * aval0 * wgbg  
                if j == jres
                    ak0i -= isgn * aval0 * wgbg
                end
                
                # 6.10 skip when plasma, no skip when considering wall
                if iopw == 0  # Fixed: opw to iopw
                    continue
                end

                # 6.11 if Wall is considered(wall/wall, plasma/wall, wall/plasma), add up bval value
                greensfunction[j,js1] += bval * amm  
                greensfunction[j,js2] += bval * am  
                greensfunction[j,js3] += bval * a0  
                greensfunction[j,js4] += bval * ap  
                greensfunction[j,js5] += bval * app  

            end
        end


        # 7. Residue
        # 7.1 Set default residue
        if j1 == j2 
            residue = 2.0
        else
            residue = 0.0
        end

        # 7.2 Change resdg, resk0 according to ishape
        # resdg, resk0 ???
        if wall_geo.is_closed_toroidal
            resdg=(2-j1)*(2-j2)+(j1-1)*(j2-1)
            resk0=(2-j1)*(2-j2)+(j1-3)*(j2-1)
            residue=resdg+resk0
        end

        # 7.3 minus residue value
        work[j] = work[j] - isgn * aval1 + residue
        if j == jres
            ak0i -= isgn * aval1
        end


        # 8.1 Only when plasma/plasma, log singularity activate. (S1)
        if iops == 1 && iopw != 0
            greensfunction[j,js1] -= alg2 / x_obs
            greensfunction[j,js2] -= alg1 / x_obs
            greensfunction[j,js3] -= alg0 / x_obs
            greensfunction[j,js4] -= alg1/x_obs
            greensfunction[j,js5] -= alg2/x_obs
        end

        # 4.3 Store all the datas of work in grdgre, gren
        gradgreensfunction[(j1-1)*mth + j, (j2-1)*mth .+ (1:mth)] .= work[1:mth]
        greensfunction[j, 1:mth] ./= 2π

    end

end

"""
    fourier_inverse!(gll, gil, cs, m00, l00, mth, lmin, lmax, dth)

    Purpose:
      This routine performs the inverse Fourier transform of gil onto gll
      using Fourier coefficients stored in cs.
    
    Inputs:
      gil(i,l)   : input matrix of size (mth × jmax1), the Fourier-space data
      cs(j,l)    : Fourier coefficient matrix (mth × jmax1)
      m00, l00   : integer offsets in the gil matrix
      lmin, lmax : define the active range of Fourier mode indices
      mth        : number of θ-grid points (dimension of gil along i)
      dth        : grid spacing in θ
    
    Output:
      gll(l2,l1) : output matrix updated in-place
"""
function fourier_inverse_transform!(gll::Matrix{Float64}, gil::Matrix{Float64}, cs::Matrix{Float64},
                 m00::Int, l00::Int, mth::Int, lmin::Int, lmax::Int)

    jmax1 = lmax - lmin + 1

    # Zero out gll block
    for l1 in 1:jmax1
        for l2 in 1:jmax1
            gll[l2, l1] = 0.0
        end
    end

    # Main accumulation (note: gll[l2, l1], not gll[l1, l2])
    dth = 2π / mth
    for l1 in 1:jmax1
        for l2 in 1:jmax1
            for i in 1:mth
                gll[l2, l1] += dth * cs[i, l2] * gil[m00+i, l00+l1] * 2π
            end
        end
    end

    return gll
end

"""
    fourier_transform!(gil, gij, cs, m00, l00, lmin, lmax, mth)

    Purpose:
      This routine performs a truncated Fourier transform of gij onto gil
      using Fourier coefficients stored in cs.
    
    Inputs:
      gij(i,j)   : input matrix of size (mth × mth), the "physical-space" data
      cs(j,l)    : Fourier coefficient matrix (mth × jmax1)
      m00, l00   : integer offsets in the gil matrix
      lmin, lmax : define the active range of Fourier mode indices
      mth        : number of θ-grid points (dimension of gij along i, j)
    
    Output:
      gil(i', l') : matrix updated in-place, where i' = m00 + i and l' = l00 + l
"""
function fourier_transform!(
    gil::Matrix{Float64},
    gij::Matrix{Float64},
    cs::Matrix{Float64},
    m00::Int,
    l00::Int,
    lmin::Int,
    lmax::Int,
    mth::Int
)
    # Compute jmax1 like Fortran
    jmax1 = lmax - lmin + 1

    # Zero out relevant gil block
    for l1 in 1:jmax1
        for i in 1:mth
            gil[m00 + i, l00 + l1] = 0.0
        end
    end

    # Accumulate with ll offset (critical to match Fortran)
    for l1 in 1:jmax1
        ll = l1 - 1 + lmin
        for j in 1:mth
            for i in 1:mth
                gil[m00 + i, l00 + l1] += cs[j, l1] * gij[i, j]
            end
        end
    end
end

function assemble_vacuum_matrix!(    
    vacmat, vacmti, vacmatu, vacmtiu,
    grwp, grpw, grwpw, grww,
    mth, jmax1, ln, lx, factpi
)
    """
    assemble_vacuum_matrix!(
        vacmat, vacmti, vacmatu, vacmtiu,
        grwp, grpw, grwpw, grww,
        mth, jmax1, ln, lx, factpi
    )

    Assemble the full vacuum response matrix from Green’s function sub-blocks.

    This routine constructs the four quadrants of the vacuum matrix that couple
    plasma and wall Fourier modes. The block structure is:

        ┌                 ┐
        │  PP   PW        │
        │                 │
        │  WP   WW        │
        └                 ┘

    where:
    - PP = Plasma–Plasma block
    - PW = Plasma–Wall block
    - WP = Wall–Plasma block
    - WW = Wall–Wall block

    Each sub-block is filled from the corresponding Green’s function kernel arrays.

    Arguments
    ---------
    - `vacmat::Array{Float64,2}` : Main vacuum response matrix (modified in place).
    - `vacmti::Array{Float64,2}` : Work array for inverse/iterative solver (zeroed here).
    - `vacmatu::Array{Float64,2}` : Work array for upper triangular factor (zeroed here).
    - `vacmtiu::Array{Float64,2}` : Work array for inverse of upper factor (zeroed here).
    - `grwp::Array{Float64,2}` : Green’s function values for Plasma–Plasma coupling.
    - `grpw::Array{Float64,2}` : Green’s function values for Plasma–Wall coupling.
    - `grwpw::Array{Float64,2}` : Green’s function values for Wall–Plasma coupling.
    - `grww::Array{Float64,2}` : Green’s function values for Wall–Wall coupling.
    - `mth::Int` : Number of poloidal modes.
    - `jmax1::Int` : Number of radial Fourier indices (per mode).
    - `ln::Int` : Toroidal mode index (reserved, not used here).
    - `lx::Int` : Radial surface index (reserved, not used here).
    - `factpi::Float64` : Normalization factor (π-related).

    Notes
    -----
    - The matrices `vacmti`, `vacmatu`, and `vacmtiu` are reset to zero here
    but not assembled; they are prepared for later linear algebra operations.
    - Indexing follows the (m, l) block structure, flattened to (row, col).
    - The resulting `vacmat` has dimensions `(2*mth*jmax1, 2*mth*jmax1)`.
    """
    
    # Dimensions
    nrows = 2 * mth * jmax1
    ncols = 2 * mth * jmax1

    # Reset all vacuum matrices
    vacmat  .= 0.0
    vacmti  .= 0.0
    vacmatu .= 0.0
    vacmtiu .= 0.0

    # ----------------------------------------------------------
    # Assemble Plasma–Plasma block into vacmat (upper-left)
    # ----------------------------------------------------------
    for m in 1:mth
        for l in 1:jmax1
            row = (m - 1) * jmax1 + l
            for mp in 1:mth
                for lp in 1:jmax1
                    col = (mp - 1) * jmax1 + lp
                    vacmat[row, col] += grwp[row, col] * factpi
                end
            end
        end
    end

    # ----------------------------------------------------------
    # Assemble Plasma–Wall block into vacmat (upper-right)
    # ----------------------------------------------------------
    offset_col = mth * jmax1
    for m in 1:mth
        for l in 1:jmax1
            row = (m - 1) * jmax1 + l
            for mp in 1:mth
                for lp in 1:jmax1
                    col = offset_col + (mp - 1) * jmax1 + lp
                    vacmat[row, col] += grpw[row, col - offset_col] * factpi
                end
            end
        end
    end

    # ----------------------------------------------------------
    # Assemble Wall–Plasma block into vacmat (lower-left)
    # ----------------------------------------------------------
    offset_row = mth * jmax1
    for m in 1:mth
        for l in 1:jmax1
            row = offset_row + (m - 1) * jmax1 + l
            for mp in 1:mth
                for lp in 1:jmax1
                    col = (mp - 1) * jmax1 + lp
                    vacmat[row, col] += grwpw[row - offset_row, col] * factpi
                end
            end
        end
    end

    # ----------------------------------------------------------
    # Assemble Wall–Wall block into vacmat (lower-right)
    # ----------------------------------------------------------
    for m in 1:mth
        for l in 1:jmax1
            row = offset_row + (m - 1) * jmax1 + l
            for mp in 1:mth
                for lp in 1:jmax1
                    col = offset_col + (mp - 1) * jmax1 + lp
                    vacmat[row, col] += grww[row - offset_row, col - offset_col] * factpi
                end
            end
        end
    end

    return nothing
end

# I think this is doing what gelimb was doing in the Fortran?
function solve_vacuum!(
    vacmat, vacmti, vacmatu, vacmtiu,
    rhs, chi,
    use_updown::Bool=false, use_symmetry::Bool=false
)
    """
    Solve the vacuum matrix equation for the Fourier coefficients.

    Parameters
    ----------
    vacmat, vacmti, vacmatu, vacmtiu : Arrays
        Vacuum matrices (preassembled).
    rhs : Vector
        Right-hand side vector (plasma boundary conditions).
    chi : Vector (preallocated)
        Output solution vector (vacuum potential coefficients).
    use_updown : Bool
        If true, use the up–down symmetric version.
    use_symmetry : Bool
        If true, use the toroidal symmetry version.

    Returns
    -------
    chi : updated in place
    """

    # Select which matrix to use based on flags
    mat = vacmat
    if use_updown && use_symmetry
        mat = vacmtiu
    elseif use_updown
        mat = vacmatu
    elseif use_symmetry
        mat = vacmti
    end

    # Solve system: mat * chi = rhs
    try
        chi .= mat \ rhs
    catch err
        @warn "Vacuum solve failed, using fallback (zeros)" exception=(err, catch_backtrace())
        chi .= 0.0
    end
end