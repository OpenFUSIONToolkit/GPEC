# VERY MUCH A WORK IN PROGRESS, NOTHING HERE WILL WORK YET
# I just pulled this relevant stuff from vac_vaccal.ipynb and threw it into this file
# and commented out the wall/debug stuff for now

# Gaussian quadrature weights and points for 8-point integration
const WGAUS = [0.101228536290376, 0.222381034453374, 0.313706645877887, 0.362683783378362,
               0.362683783378362, 0.313706645877887, 0.222381034453374, 0.101228536290376]

const XGAUS = [-0.960289856497536, -0.796666477413627, -0.525532409916329, -0.183434642495650,
                0.183434642495650,  0.525532409916329,  0.796666477413627,  0.960289856497536]


function vaccal!(globals::VacuumGlobalsType, settings::VacuumSettingsType)

    # ----------------------------------------------------------
    # Allocate and zero arrays
    # ----------------------------------------------------------
    grdgre = zeros(Float64, nths2, nths2)
    grwp   = zeros(Float64, nths2, nths2)
    dummy  = zeros(Float64, nths2, nfm2)
    ajll  .= 0.0
    vacmat .= 0.0
    vacmti .= 0.0

    println(outmod, "\n          Matrix Storage: K(obs_ji,sou_ji):\n")
    println(outmod, "          j = observer points, i = source points.")
    println(outmod, "          ie. K operates on chi from the left.\n")
    println(outmod, "          Observer Source  Block")
    println(outmod, "          plasma  plasma   1  1")
    println(outmod, "          plasma  wall     1  2")
    println(outmod, "          wall    plasma   2  1")
    println(outmod, "          wall    wall     2  2\n")

    # Initialization
    xwal[1] = 0.0
    zwal[1] = 0.0
    factpi  = twopi
    jmax    = 2*lmax[1] + 1
    jmax1   = lmax[1] - lmin[1] + 1
    lmax1   = lmax[1] + 1
    ln      = lmin[1]
    lx      = lmax[1]
    jdel    = 8
    nq      = n*q
    tmth    = 2*mth
    mthsq   = tmth * tmth
    lmth    = tmth * 2*jmax1
    j1v     = nfm
    j2v     = nfm

    # ----------------------------------------------------------
    # Apply wall boundary conditions
    # ----------------------------------------------------------
    # wwall!(mth, xwal, zwal)

    # ----------------------------------------------------------
    # Plasma–Plasma block
    # ----------------------------------------------------------
    j1, j2 = 1, 1
    ksgn = 2*j2 - 3
    grdgre, grwp = kernel!(xpla, zpla, xpla, zpla, j1, j2, ksgn, 1, 1, 0)

    fouran!(grwp, grdgre, cslth, 0, 0, lmin, lmax, mth)
    fouran!(grwp, grdgre, snlth, 0, jmax1, lmin, lmax, mth)

    # NOTE: there needs to be an if not farwall statement here, this entire section is skipped if farwall is true
    # # ----------------------------------------------------------
    # # Plasma–Wall block
    # # ----------------------------------------------------------
    # j1, j2 = 1, 2
    # ksgn = 2*j2 - 3
    # grpw_block = similar(grdgre)
    # #grdgre, grpw_block = kernel!(xpla, zpla, xwal, zwal, j1, j2, ksgn, 1, 1, 0)

    # if lfele == 1
    #     felang!(grpw_block, grdgre, cnqd, 0,0)
    #     felang!(grpw_block, grdgre, snqd, 0,jmax1)
    # end
    # if lfour == 1
    #     fouran!(grpw_block, grdgre, cslth, 0, 0, lmin, lmax, mth)
    #     fouran!(grpw_block, grdgre, snlth, 0, jmax1, lmin, lmax, mth)
    # end

    # # After plasma-wall kernel
    # dbg("grpw_block (plasma-wall)", grpw_block)

    # # ----------------------------------------------------------
    # # Wall–Plasma block
    # # ----------------------------------------------------------
    # j1, j2 = 2, 1
    # ksgn = 2*j2 - 3
    # grwpw_block = similar(grdgre)
    # #grdgre, grwpw_block = kernel!(xwal, zwal, xpla, zpla, j1, j2, ksgn, 1, 1, 0)

    # if lfele == 1
    #     felang!(grwpw_block, grdgre, cnqd, 0,0)
    #     felang!(grwpw_block, grdgre, snqd, 0,jmax1)
    # end
    # if lfour == 1
    #     fouran!(grwpw_block, grdgre, cslth, 0, 0, lmin, lmax, mth)
    #     fouran!(grwpw_block, grdgre, snlth, 0, jmax1, lmin, lmax, mth)
    # end

    # # After wall-plasma kernel
    # dbg("grwpw_block (wall-plasma)", grwpw_block)

    # # ----------------------------------------------------------
    # # Wall–Wall block
    # # ----------------------------------------------------------
    # j1, j2 = 2, 2
    # ksgn = 2*j2 - 3
    # grww_block = similar(grdgre)
    # #grdgre, grww_block = kernel!(xwal, zwal, xwal, zwal, j1, j2, ksgn, 1, 1, 0)

    # if lfele == 1
    #     felang!(grww_block, grdgre, cnqd, 0,0)
    #     felang!(grww_block, grdgre, snqd, 0,jmax1)
    # end
    # if lfour == 1
    #     fouran!(grww_block, grdgre, cslth, 0, 0, lmin, lmax, mth)
    #     fouran!(grww_block, grdgre, snlth, 0, jmax1, lmin, lmax, mth)
    # end

    # # After wall-wall kernel
    # dbg("grww_block (wall-wall)", grww_block)

    # # ----------------------------------------------------------
    # # Assemble matrices for solving
    # # ----------------------------------------------------------
    # assemble_vacuum_matrix!(
    #     vacmat, vacmti, vacmatu, vacmtiu,
    #     grwp, grpw_block, grwpw_block, grww_block,
    #     mth, jmax1, ln, lx, factpi
    # )

    # Need an equivalent assemble_vacuum_matrix! call here for the plasma-plasma block only

    # After assembling vacmat
    dbg("vacmat assembled", vacmat)
    dbg_scalar("factpi", factpi)

    # ----------------------------------------------------------
    # Solve vacuum system
    # ----------------------------------------------------------
    solve_vacuum!(
        vacmat, vacmti, vacmatu, 
        vacmtiu, ajll, chiwc
    )

    # After solving vacuum
    dbg("ajll", ajll)
    dbg("chiwc", chiwc)
    dbg("chiws", chiws)

    println("vacmat size = ", size(vacmat))
    println("ajll size = ", length(ajll))
    println("chiwc size = ", length(chiwc))

    # ----------------------------------------------------------
    # Call arrays! here with proper parameters
    # ----------------------------------------------------------
    grri = zeros(Float64, nths2, nths2)    # allocate grri
    delfac = 0.5                          # example value; set as needed
    # Replace this with setuparrays! ?
    # arrays!(mth, dth, lmin, lmax, qa1, n, delfac, delta, xpla, zpla, grri, farwal != 0)

end

"""
    kernel(xobs, zobs, xsce, zsce, j1, j2, isgn, iopw, iops, ischk, params)

Compute kernels of integral equation for Laplace's equation for a torus.

# Arguments
- `xobs`: Observer x coordinates
- `zobs`: Observer z coordinates  
- `xsce`: Source x coordinates
- `zsce`: Source z coordinates
- `j1, j2`: Boundary condition indices
- `isgn`: Sign parameter
- `iopw`: Wall option (0=inactive, 1=active)
- `iops`: Log singularity correction option
- `wall_flag`: Check option for conductor position (ischk)
- `params`: Dictionary containing simulation parameters

# Returns
- `grdgre`: Gradient Green's function matrix
- `gren`: Green's function matrix
"""
function kernel!(grdgre, gren, xobs, zobs, xsce, zsce, j1, j2, isgn, iopw, iops, wall_flag)

    # matrix output gren is accumulated in grwp of vaccal.
    # While grwp is 𝒢 befor fourier transform, grri is fourier transformed 𝒢

    the = theta_values = LinRange(0, 2*pi, 100)
    thetas = the

    # 1. definition for solving parameters
    wsimpb1=1*dth/3
    wsimpb2=2*dth/3
    wsimpb4=4*dth/3

    algdth = log(dth) # log of dth
    alg = log(2*dth)

    slog1m = 3*dth*(algdth-1/3)
    slog1p = slog1m

    alg0=16.0*dth*(alg-68.0/15.0)/15.0
    alg1=128.0*dth*(alg-8.0/15.0)/45.0
    alg2=4.0*dth*(7.0*alg-11.0/15.0)/45.0

    # 2. check singular points for conductor.
    if wall_flag == true
        # 2.0 initialize jbot and jtop
        jbot=mth/2+1
        jtop=mth/2+1
        
        # 2.1 call wall and restore points of wall in ww1, ww2
        wwall!(mth1,ww1,ww2)

        # 2.2 find where sign of wall r point acrosses zero. 
        # isph means there is 0-corssing point 
        for i in 1:mth
            if ww1[i] * ww1[i+1] ≤ zero
                jbot = ww1[i] > zero ? i : jbot
                jtop = ww1[i] < zero ? i + 1 : jtop
                isph = 1 
            end
        end
    end

    # 3 do spline and calc derivative for Z'_θ and X'_θ in eq.(51)
    xpr = [spline1d_deriv(the, xsce, θ) for θ in thetas]
    zpr = [spline1d_deriv(the, zsce, θ) for θ in thetas]

    # 4, begin obs loop.
    for j in 1:mth 

        # 4,1 initialize variable
        xs=xobs[j] #observation point  # Fixed: () to []
        zs=zobs[j]  # Fixed: () to []
        thes=the[j] # theta value  # Fixed: () to []
        work = zeros(mth1)
        aval1=zero # ∇𝒢_0

        # 4.2 if the point of observation point is in negative, We cannot use green func
        # This is same for source point 
        if xs < zero 
            if j2 == 2 
                work[j] = 1.0  # Fixed: () to []
            end
        else

            # 4.3 set istart and iend
            iend = 2 # end point of integration
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
            mths=mth-(istart+iend-1) 

            
            # 4.4 loop for source point 
            for i in 1:mths

                # 4.5 get source point index(ic) theta(theta), X(xt), and Z(zt)
                ic = i + j + istart - 1
                if ic ≥ mth1
                    ic = ic - mth
                end
                theta=(ic-1)*dth 
                xt=xsce[ic]  # Fixed: () to []
                zt=zsce[ic]  # Fixed: () to []

                # 4.6 if source point is in negative, we cannot use green function
                # & if source point(ic) and obs point (j) is same, it's singular
                if xt < 0 
                    continue 
                end
                if ic == j 
                    continue 
                end

                # 4.7 calc X'_θ (xtp) and Z'_θ (ztp) and call green function
                # aval is 𝒥 ∇'𝒢ⁿ∇'ℒ, bval is 2pi𝒢ⁿ. aval0 is 𝒥 ∇'𝒢ⁿ∇'ℒ for n=0
                xtp=xpr[ic]  # Fixed: () to []
                ztp=zpr[ic]  # Fixed: () to []
                G, aval, aval0, bval = green(xs,zs,xt,zt,xtp,ztp,n,usechancebugs=false)

                # 4.8 simpson integral. 4 for odd, 2 for even, and 1 for others.
                wsimpb=wsimpb2
                if (i÷2)*2 == i  # Fixed: / to ÷ for integer division
                    wsimpb=wsimpb4
                end
                if (i == 1)||(i == mths)
                    wsimpb=wsimpb1
                end
                wsimpa=wsimpa2
                if (i÷2)*2 == i  # Fixed: / to ÷ for integer division
                    wsimpa=wsimpa4
                end
                if (i == 1)||(i == mths)
                    wsimpa=wsimpa1
                end

                # 4.9 work and gren
                # work : simpson integral for aval(𝒥 ∇'𝒢ⁿ∇'ℒ)
                # gren : log singularity values accumulated. simpson integral for bval
                # aval1 : aval1
                work[ic]=work[ic]+isgn*aval*wsimpa  # Fixed: () to []
                gren[j,ic]=gren[j,ic]+bval*wsimpb # integral of bval (log singularity)  # Fixed: () to []
                aval1 = aval1 + aval0 * wsimpa
            
            end
            
            # 5.1 if it's plasma/plsama, wall/wall and in negative wall point, skip loop
            # obs : j1 = 1(plasma), wall(2)
            # src : j1 = 1(plasma), wall(2)
            j1j2 = j1+j2
            if j1j2 != 2 && isph == 1 && j > jbot && j < jtop  # Fixed: a&& to &&
                continue
            end

            # 5.2 Get js
            # js2 : j - iend + 1
            thes = the[j]  # Fixed: () to []
            js1=mod(j-iend+mth-1,mth)+1
            js2=mod(j-iend+mth,mth)+1
            js3=mod(j-iend+mth+1,mth)+1
            js4=mod(j-iend+mth+2,mth)+1
            js5=mod(j-iend+mth+3,mth)+1

            # 6 Singular points when source point and obs point are the same
            # 6.1 integration each left and right
            for ilr in [1,2]
                xl = thes + (2*ilr-iend-2)*dth
                xu = xl + 2 * dth
                agaus = (xu + xl)/2
                bgaus = (xu - xl)/2
                tgaus = agaus .+ XGAUS .* bgaus # tgaus is 8 point gauss points, since xgauss is for only [-1,1]  # Fixed: xgaus to XGAUS
                # 6.2 for each 8 gaussian points
                for ig in 1:8
                    tgaus0 = tgaus[ig] #i-th value of for 8 points, in theta  # Fixed: () to []
                    tgaus0 = mod(tgaus0, twopi)

                    # 6.3 get X, X', Z, Z' for gaussian point
                    spl1d2!(mth1,the,xsce,xpp,1,tgaus0,tab)
                    xt = tab[1] # xt is X  # Fixed: () to []
                    xtp = tab[2] # xtp is X'_θ  # Fixed: () to []
                    spl1d2!(mth1,the,zsce,zpp,1,tgaus0,tab)
                    zt=tab[1] # zt is Z  # Fixed: () to []
                    ztp=tab[2] # ztp is Z'_θ  # Fixed: () to []

                    # 6.4 call green function
                    G, aval, aval0, bval = green(xs,zs,xt,zt,xtp,ztp,n,usechancebugs=false)

                    # 6.5 add logarithm on G (not 𝒢_n). Chance eq.(75)
                    # iops = 1
                    bval = G + iops * log((thes-tgaus[ig])^2)/xs  # Fixed: () to []

                    # 6.6 calc wgaus. bgaus refers Δ in theta. wgaus is weight for each 8 points
                    wgbg=WGAUS[ig]*bgaus  # Fixed: wgaus() to WGAUS[]

                    # 6.7 calc pgaus. below Chance eq.(77)
                    pgaus=(tgaus[ig]-thes-(2-iend)*dth)/dth  # Fixed: () to []
                    pgaus2=pgaus*pgaus
                    amm = (pgaus2-1)*pgaus*(pgaus-2)/24.0 *wgbg
                    am = -(pgaus-1)*pgaus*(pgaus2-4)/6.0 *wgbg
                    a0 = (pgaus2-1)*(pgaus2-4)/4.0 *wgbg
                    ap = -(pgaus+1)*pgaus*(pgaus2-4)/6.0 *wgbg
                    app = (pgaus2-1)*pgaus*(pgaus+2)/24.0 *wgbg

                    # 6.8 add up in work
                    work[js1] += isgn * aval * amm  # Fixed: () to []
                    work[js2] += isgn * aval * am  # Fixed: () to []
                    work[js3] += isgn * aval * a0  # Fixed: () to []
                    work[js4] += isgn * aval * ap  # Fixed: () to []
                    work[js5] += isgn * aval * app  # Fixed: () to []

                    # 6.9 minus diverging value
                    work[j] -= isgn * aval0 * wgbg  # Fixed: () to []
                    if j == jres
                        ak0i -= isgn * aval0 * wgbg
                    end
                    
                    # 6.10 skip when plasma, no skip when considering wall
                    if iopw == 0  # Fixed: opw to iopw
                        continue
                    end

                    # 6.11 if Wall is considered(wall/wall, plasma/wall, wall/plasma), add up bval value
                    gren[j,js1] += bval * amm  # Fixed: () to []
                    gren[j,js2] += bval * am  # Fixed: () to []
                    gren[j,js3] += bval * a0  # Fixed: () to []
                    gren[j,js4] += bval * ap  # Fixed: () to []
                    gren[j,js5] += bval * app  # Fixed: () to []

                end
            end


            # 7. Residue
            # 7.1 Set default residu
            if j1 == j2 
                residu = 2.0
            else
                residu = 0.0
            end

            # 7.2 Change resdg, resk0 according to ishape
            # resdg, resk0 ???
            if ishape < 10 || ishape == 41 || ishape == 42
                resdg=(2-j1)*(2-j2)+(j1-1)*(j2-1)
                resk0=(2-j1)*(2-j2)+(j1-3)*(j2-1)
                residu=resdg+resk0
            end

            # 7.3 minus residu value
            work[j] -= (isgn * aval1 + residu)  # Fixed: () to []
            if j == jres
                ak0i -= isgn * aval1
            end


            # 8.1 Only when plasma/plasma, log singularity activate. (S1)
            if iops == 1 && iopw != 0
                gren[j,js1] -= alg2 / xs  # Fixed: () to []
                gren[j,js2] -= alg1 / xs  # Fixed: () to []
                gren[j,js3] -= alg0 / xs  # Fixed: () to []
                gren[j,js4] -= alg1/xs  # Fixed: () to []
                gren[j,js5] -= alg2/xs  # Fixed: () to []
            end

        end

        # 4.3 Store all the datas of work in grdgre, gren
        grdgre[(j1-1)*mth + j, (j2-1)*mth + (1:mth)] .= work[1:mth]
        gren[j, 1:mth] ./= twopi

    end
end

function foranv!(gil::Matrix{Float64}, gll::Matrix{Float64}, cs::Matrix{Float64},
                 m00::Int, l00::Int, mth::Int, lmin::Int, lmax::Int,
                 dth::Float64)
    
    # -------------------------------------------------------------------------
    # Purpose:
    #   This routine performs the inverse Fourier transform of gil using the
    #   coefficient matrix cs, producing the real-space matrix gll.
    #
    # Notes:
    #   • The summation is over the θ-grid index i.
    #   • The factor (dth ⋅ twopi) accounts for the discretized integration
    #     over poloidal angle θ (with spacing dth and full 2π periodicity).
    #   • gll is symmetric in the sense that its block depends on l1, l2
    #     but the roles of l1 and l2 are not interchangeable here because
    #     cs is indexed by (i, l2) and gil by (i, l1).
    #
    # Inputs:
    #   gil(i', l') : input Fourier-space data, dimensions (nths2 × nfm2)
    #   cs(i, l2)   : Fourier coefficient matrix (mth × jmax1)
    #   m00, l00    : integer offsets into gil
    #   mth         : number of θ-grid points
    #   lmin, lmax  : Fourier mode index bounds
    #   dth         : angular spacing (2π / mth)
    #
    # Output:
    #   gll(l2, l1) : jmax1 × jmax1 real-space block
    #
    # -------------------------------------------------------------------------

    jmax1 = lmax - lmin + 1

    # Zero out gll block
    for l1 in 1:jmax1
        for l2 in 1:jmax1
            gll[l2, l1] = 0.0
        end
    end

    # Main accumulation (note: gll[l2, l1], not gll[l1, l2])
    for l1 in 1:jmax1
        for l2 in 1:jmax1
            for i in 1:mth
                gll[l2, l1] += dth * cs[i, l2] * gil[m00+i, l00+l1] * 2π
            end
        end
    end

    return gll
end

function fouran!(
    gij::Matrix{Float64},
    gil::Matrix{Float64},
    cs::Matrix{Float64},
    m00::Int,
    l00::Int,
    lmin::Vector{Int},
    lmax::Vector{Int},
    mth::Int
)

    # -------------------------------------------------------------------------
    # Purpose:
    #   This routine performs a truncated Fourier transform of gij onto gil
    #   using Fourier coefficients stored in cs.
    #
    # Inputs:
    #   gij(i,j)   : input matrix of size (mth × mth), the "physical-space" data
    #   cs(j,l)    : Fourier coefficient matrix (mth × jmax1)
    #   m00, l00   : integer offsets in the gil matrix
    #   lmin, lmax : define the active range of Fourier mode indices
    #   mth        : number of θ-grid points (dimension of gij along i, j)
    #
    # Output:
    #   gil(i', l') : matrix updated in-place, where i' = m00 + i and l' = l00 + l
    #
    # -------------------------------------------------------------------------

    # Compute jmax1 like Fortran
    jmax1 = lmax[1] - lmin[1] + 1

    # Zero out relevant gil block
    for l1 in 1:jmax1
        for i in 1:mth
            gil[m00 + i, l00 + l1] = 0.0
        end
    end

    # Accumulate with ll offset (critical to match Fortran)
    for l1 in 1:jmax1
        ll = l1 - 1 + lmin[1]
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