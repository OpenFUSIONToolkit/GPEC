# Gaussian quadrature weights and points for 8-point integration
const GAUSSIANWEIGHTS = [0.101228536290376, 0.222381034453374, 0.313706645877887, 0.362683783378362,
               0.362683783378362, 0.313706645877887, 0.222381034453374, 0.101228536290376]

const GAUSSIANPOINTS = [-0.960289856497536, -0.796666477413627, -0.525532409916329, -0.183434642495650,
                0.183434642495650,  0.525532409916329,  0.796666477413627,  0.960289856497536]


function vaccal!(inputs::VacuumInput, plasma_surf::PlasmaGeometry, wall::WallGeometry)

    # Initialization
    (; mtheta, mpert, n, kernelsign, force_wv_symmetry) = inputs
    grri = zeros(2 * mtheta, 2 * mpert)
    grdgre = zeros(2 * mtheta, 2 * mtheta)
    grpp = zeros(2 * mtheta, 2 * mtheta)

    # ----------------------------------------------------------
    # Plasma–Plasma block
    # ----------------------------------------------------------
    j1, j2 = 1, 1
    ksgn = 2*j2 - 3
    kernel!(grdgre, grpp, plasma_surf.x, plasma_surf.z, plasma_surf.x, plasma_surf.z, j1, j2, ksgn, 1, 1, false, inputs)

    # TODO: remove this I think? Can't test right now since the full runthrough is broken. This should have been removed when mtheta+5 was removed
    # Enforce periodic boundary conditions
    for i = 1:mtheta + 2
        grpp[i,mtheta + 1] = grpp[i,1]
        grpp[i,mtheta + 2] = grpp[i,2]
    end

    # Fourier transform plasma-plasma block
    fourier_transform!(grri, grpp, plasma_surf.cslth, 0, 0, mtheta, mpert)
    fourier_transform!(grri, grpp, plasma_surf.snlth, 0, mpert, mtheta, mpert)

    if !wall.nowall
        # ----------------------------------------------------------
        # Plasma–Wall block
        # ----------------------------------------------------------
        error("Haven't set up walls yet")
        j1, j2 = 1, 2
        ksgn = 2*j2 - 3
        grpw_block = zeros(2 * mtheta, 2 * mtheta)
        kernel!(grdgre, grpw_block, globals.xpla, globals.zpla, globals.xwal, globals.zwal, j1, j2, ksgn, 1, 1, true, inputs)

        fourier_transform!(grpw_block, grdgre, cslth, 0, 0, mtheta, mpert)
        fourier_transform!(grpw_block, grdgre, snlth, 0, mpert, mtheta, mpert)

        # ----------------------------------------------------------
        # Wall–Plasma block
        # ----------------------------------------------------------
        j1, j2 = 2, 1
        ksgn = 2*j2 - 3
        grwp_block = similar(grdgre) # This should be grpp or a new matrix
        kernel!(grdgre, grwp_block, globals.xwal, globals.zwal, globals.xpla, globals.zpla, j1, j2, ksgn, 1, 1, true, inputs)

        fourier_transform!(grwp_block, grdgre, cslth, 0, 0, mtheta, mpert)
        fourier_transform!(grwp_block, grdgre, snlth, 0, mpert, mtheta, mpert)

        # ----------------------------------------------------------
        # Wall–Wall block
        # ----------------------------------------------------------
        j1, j2 = 2, 2
        ksgn = 2*j2 - 3
        grww_block = similar(grdgre) # This should be grpp or a new matrix
        kernel!(grdgre, grww_block, globals.xwal, globals.zwal, globals.xwal, globals.zwal, j1, j2, ksgn, 1, 1, true, inputs)

        fourier_transform!(grww_block, grdgre, cslth, 0, 0, mtheta, mpert)
        fourier_transform!(grww_block, grdgre, snlth, 0, mpert, mtheta, mpert)

        # ----------------------------------------------------------
        # Assemble matrices for solving
        # ----------------------------------------------------------
        assemble_vacuum_matrix!(
            vacmat, vacmti, vacmatu, vacmtiu,
            grwp, grpw_block, grwp_block, grww_block,
            mth, mpert, inputs.mlow, inputs.mhigh, 2π
        )
    end

    # TODO: is this getting kept?
    # Add cn0 to make grdgre nonsingular for n=0 modes
    cn0 = 1.0 # expose to user if anyone ever actually tries to use this
    if (abs(n) <= 1e-5) && (!wall.nowall) && (wall.is_closed_toroidal)
        @warn "Adding $cn0 to diagonal of grdgre to regularize n=0 mode; this may affect accuracy of results."
        mth12 = farwall_flag ? mtheta : 2 * mtheta
        for i in 1:mth12, j in 1:mth12
            grdgre[i, j] += cn0
        end
    end

    # Only needed for mutual inductance with the wall calculations
    if kernelsign < 0
        grdgre .*= kernelsign
        # Account for factor of 2 in diagonal terms in eq. 90 of Chance
        for i in 1:2 * mtheta
            grdgre[i, i] += 2.0
        end
    end

    # Invert the plasma response system of equations, eqs. 92-94ish of Chance 1997 (gelimb in Fortran)
    # TODO: this is for plasma only! Need to invert the full matrix if walls
    grri[1:mtheta, :] .= grdgre[1:mtheta, 1:mtheta] \ grri[1:mtheta, :]

    # There's some logic that computes xpass/zpass and chiwc/chiws here, might eventually be needed?

    # Perform inverse Fourier transforms to get response matrix components (eq. 115-118 of Chance 2007)
    arr = zeros(mpert, mpert)
    aii = zeros(mpert, mpert)
    ari = zeros(mpert, mpert)
    air = zeros(mpert, mpert)
    fourier_inverse_transform!(arr, grri, plasma_surf.cslth, 0, 0, mtheta, mpert)
    fourier_inverse_transform!(aii, grri, plasma_surf.snlth, 0, mpert, mtheta, mpert)
    fourier_inverse_transform!(ari, grri, plasma_surf.snlth, 0, 0, mtheta, mpert)
    fourier_inverse_transform!(air, grri, plasma_surf.cslth, 0, mpert, mtheta, mpert)

    # Final form of vacuum response matrix (eq. 114 of Chance 2007)
    vacmat = arr .+ aii
    vacmti = air .- ari
    # Force symmetry of response matrix if desired
    if force_wv_symmetry
        for l1 in 1:mpert
            for l2 in l1:mpert
                vacmat[l1, l2] = 0.5 * (vacmat[l1, l2] + vacmat[l2, l1])
                vacmti[l1, l2] = 0.5 * (vacmti[l1, l2] - vacmti[l2, l1])
            end
        end
    end
    wv = complex.(vacmat, vacmti)

    println("WV from Julia")
    display(wv)

    # There was an extra arrays call here in the Fortran - do we need any functionality from it here?

    # TODO: add our own wall/plasma geometry outputs if desired
    return wv, grri
end

"""
    kernel(x_obspoints, z_obspoints, x_sourcepoints, z_sourcepoints, j1, j2, isgn, iopw, iops, ischk, params)

Compute kernels of integral equation for Laplace's equation for a torus.

    ** WARNING: This kernel only supports closed toroidal walls currently. 
        The residue calculation needs to be updated for open walls. **


# Arguments
- `x_obspoints`: Observer x coordinates
- `z_obspoints`: Observer z coordinates
- `x_sourcepoints`: Source x coordinates
- `z_sourcepoints`: Source z coordinates
- `j1, j2`: Boundary condition indices
- `isgn`: Sign parameter
- `iopw`: Wall option (0=inactive, 1=active)
- `iops`: Log singularity correction option
- `wallflag`: Check option for conductor position (ischk)
- `params`: Dictionary containing simulation parameters

# Returns
- `grad_greenfunction_mat`: Gradient Green's function matrix
- `greenfunction_mat`: Green's function matrix
"""
function kernel!(grad_greenfunction_mat::Matrix{Float64}, greenfunction_mat::Matrix{Float64}, x_obspoints::Vector{Float64}, z_obspoints::Vector{Float64}, x_sourcepoints::Vector{Float64}, z_sourcepoints::Vector{Float64}, j1::Int, j2::Int, isgn::Int, iopw::Int, iops::Int, wallflag::Bool, inputs::VacuumInput)

    mtheta = inputs.mtheta
    mth1 = inputs.mtheta + 1
    dtheta = 2π / mtheta
    ak0i = 0.0
    jres = 1
    N_obs = length(x_obspoints)
    theta_grid = LinRange(0, mtheta*dtheta, mtheta)
    
    if N_obs != length(z_obspoints) || N_obs != length(x_sourcepoints) || N_obs != length(z_sourcepoints)
        error("Length of input arrays (xobs, zobs, xsource, zsce) are different. All length should be the same")
    end

    # matrix output greensfunction is accumulated in grwp of vaccal.
    # While grwp is 𝒢 befor fourier transform, grri is fourier transformed 𝒢

    # definition for solving parameters
    simpson_b1=1*dtheta/3
    simpson_b2=2*dtheta/3
    simpson_b4=4*dtheta/3

    simpson_a1=1*dtheta/3
    simpson_a2=2*dtheta/3
    simpson_a4=4*dtheta/3

    
    slog1m = 3*dtheta*(log(dtheta)-1/3)
    slog1p = slog1m
    
    log2dtheta = log(2*dtheta)
    log_correction_0=16.0*dtheta*(log2dtheta-68.0/15.0)/15.0
    log_correction_1=128.0*dtheta*(log2dtheta-8.0/15.0)/45.0
    log_correction_2=4.0*dtheta*(7.0*log2dtheta-11.0/15.0)/45.0

    has_zero_crossing = false # has_zero_crossing needs to be initialized

    # check singular points for conductor.
    if wallflag
        # 2.0 initialize jbot and jtop, and get wall geometry from globals
        jbot=mtheta/2+1
        jtop=mtheta/2+1
        ww1 = globals.xwal
        ww2 = globals.zwal
        
        # 2.2 find where sign of wall r point acrosses zero.
        # has_zero_crossing means there is 0-crossing point
        for i in 1:mtheta
            if ww1[i] * ww1[i+1] <= 0.0
                jbot = ww1[i] > 0.0 ? i : jbot
                jtop = ww1[i] < 0.0 ? i + 1 : jtop
                has_zero_crossing = true
            end
        end
    end

    # do spline and calc derivative for Z'_θ and X'_θ in eq.(51)

    # Using interpolations.jl for now seemes ok
    spline_x = cubic_spline_interpolation(theta_grid, x_sourcepoints, extrapolation_bc=Interpolations.Periodic())
    spline_z = cubic_spline_interpolation(theta_grid, z_sourcepoints, extrapolation_bc=Interpolations.Periodic())

    gradients_x = (t -> Interpolations.gradient(spline_x, t)).(theta_grid)
    gradients_z = (t -> Interpolations.gradient(spline_z, t)).(theta_grid)
    dx_dtheta = first.(gradients_x) # d x / d theta
    dz_dtheta = first.(gradients_z) # d z / d theta



    # observer loop
    for j in 1:mtheta

        # initialize variable
        x_obs=x_obspoints[j] #observation point
        z_obs=z_obspoints[j]
        theta_obs=theta_grid[j] # theta value
        work = zeros(mth1)
        
        # if the point of observation point is in negative, We cannot use green func
        # This is same for source point
        if x_obs < 0.0
            if j2 == 2
                work[j] = 1.0
            end
            continue
        end
            
        # set istart and iend
        iend = 2 # end point of integration
        aval1=0.0 # ∇𝒢_0
        # if wall crossing zero and wall is source, iend set.
        if has_zero_crossing && j2 == 2
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
        mth_source=mtheta-(istart+iend-1)

        
        # loop for source point
        for i in 1:mth_source

            # 4.5 get source point index(ic) theta(theta), X(xt), and Z(zt)
            ic = i + j + istart - 1
            if ic ≥ mth1
                ic = ic - mtheta
            end
            # theta_source=(ic-1)*dtheta
            x_source=x_sourcepoints[ic]
            z_source=z_sourcepoints[ic]

            # if source point is in negative, we cannot use green function
            # & if source point(ic) and obs point (j) is same, it's singular
            if x_source < 0
                continue
            end
            if ic == j
                continue
            end

            # calc X'_θ (xtp) and Z'_θ (ztp) and call green function
            # G_n is 2pi𝒢ⁿ; coupling_n is 𝒥 ∇'𝒢ⁿ∇'ℒ; coupling_0 is 𝒥 ∇'𝒢ⁿ∇'ℒ for n=0
            xtp=dx_dtheta[ic]
            ztp=dz_dtheta[ic]
            G_n, coupling_n, coupling_0 = green(x_obs,z_obs,x_source,z_source,xtp,ztp,inputs.n,uselegacygreenfunction=true)

            # simpson integral. 4 for odd, 2 for even, and 1 for others.
            simpson_b=simpson_b2
            if (i÷2)*2 == i
                simpson_b=simpson_b4
            end
            if (i == 1)||(i == mth_source)
                simpson_b=simpson_b1
            end
            simpson_a=simpson_a2
            if (i÷2)*2 == i
                simpson_a=simpson_a4
            end
            if (i == 1)||(i == mth_source)
                simpson_a=simpson_a1
            end

            # work and gren
            # work : simpson integral for coupling_n (𝒥 ∇'𝒢ⁿ∇'ℒ)
            # gren : log singularity values accumulated. simpson integral for G_n
            # aval1 : aval1
            work[ic]=work[ic]+isgn*coupling_n*simpson_a
            greenfunction_mat[j,ic]=greenfunction_mat[j,ic]+G_n*simpson_b # integral of G_n (log singularity)
            aval1 = aval1 + coupling_0 * simpson_a
        
        end
        
        # if it's plasma/plsama, wall/wall and in negative wall point, skip loop
        # obs : j1 = 1(plasma), wall(2)
        # src : j1 = 1(plasma), wall(2)
        if (j1+j2) != 2 && has_zero_crossing && j > jbot && j < jtop
            continue
        end

        # Get js
        js1=mod(j-iend+mtheta-1,mtheta)+1
        js2=mod(j-iend+mtheta,mtheta)+1
        js3=mod(j-iend+mtheta+1,mtheta)+1
        js4=mod(j-iend+mtheta+2,mtheta)+1
        js5=mod(j-iend+mtheta+3,mtheta)+1

        # Singular points when source point and obs point are the same
        # integration each left and right by gaussian quadrature
        for ilr in [1,2]
            gauss_xleft = theta_obs + (2*ilr-iend-2)*dtheta
            gauss_xright = gauss_xleft + 2 * dtheta
            gauss_xavg = (gauss_xright + gauss_xleft)/2
            gauss_halfdifference = (gauss_xright - gauss_xleft)/2
            tgaus = gauss_xavg .+ GAUSSIANPOINTS .* gauss_halfdifference # tgaus is 8 point gauss points, since GAUSSIANPOINTS is for only [-1,1]
            # for each of 8 gaussian points
            for ig in 1:8
                tgaus0 = tgaus[ig] #i-th value of for 8 points, in theta
                tgaus0 = mod(tgaus0, 2π)

                # get X, X', Z, Z' for gaussian point
                xt = spline_x(tgaus0)
                xtp = Interpolations.gradient(spline_x, tgaus0)[1]
                zt = spline_z(tgaus0)
                ztp = Interpolations.gradient(spline_z, tgaus0)[1]

                # call green function
                G_n, coupling_n, coupling_0 = green(x_obs,z_obs,xt,zt,xtp,ztp,inputs.n,uselegacygreenfunction=true)

                # add logarithm to G_n to analytically isolate the singularity. Chance eq.(75)
                # iops = 1
                G_n_nonsingular = G_n + iops * log((theta_obs-tgaus[ig])^2)/x_obs

                # calc GAUSSIANWEIGHTS. GAUSSIANWEIGHTS contains gaussian quadrature weights for each 8 points
                wgbg=GAUSSIANWEIGHTS[ig]*gauss_halfdifference

                # calc pgaus. below Chance eq.(77)
                # All are the noted index, times (wgbg)/(2 pgaus)
                # A0 has no extra factor
                # A1 has an extra factor of (pgaus +/- 1)
                # A2 has an extra factor of (pgaus +/- 2)
                pgaus=(tgaus[ig]-theta_obs-(2-iend)*dtheta)/dtheta
                pgaus2=pgaus*pgaus
                A0 = (pgaus2-1)*(pgaus2-4)/4.0 * wgbg
                A1_plus  = -(pgaus+1)*pgaus*(pgaus2-4)/6.0 * wgbg
                A1_minus = -(pgaus-1)*pgaus*(pgaus2-4)/6.0 * wgbg
                A2_plus  = (pgaus2-1)*pgaus*(pgaus+2)/24.0 * wgbg
                A2_minus = (pgaus2-1)*pgaus*(pgaus-2)/24.0 * wgbg

                # add up in work
                work[js1] += isgn * coupling_n * A2_minus
                work[js2] += isgn * coupling_n * A1_minus
                work[js3] += isgn * coupling_n * A0
                work[js4] += isgn * coupling_n * A1_plus
                work[js5] += isgn * coupling_n * A2_plus

                # minus diverging value
                work[j] -= isgn * coupling_0 * wgbg
                if j == jres
                    ak0i -= isgn * coupling_0 * wgbg
                end
                
                # skip when plasma, no skip when considering wall
                if iopw == 0
                    continue
                end

                # if Wall is considered(wall/wall, plasma/wall, wall/plasma), add up G_n_nonsingular value
                greenfunction_mat[j,js1] += G_n_nonsingular * A2_minus
                greenfunction_mat[j,js2] += G_n_nonsingular * A1_minus
                greenfunction_mat[j,js3] += G_n_nonsingular * A0
                greenfunction_mat[j,js4] += G_n_nonsingular * A1_plus
                greenfunction_mat[j,js5] += G_n_nonsingular * A2_plus

            end
        end


        # Set residue based on logic similar to Table I of Chance 1997
        if j1 == j2
            residue = 2.0
        else
            residue = 0.0
        end
        # Would need to pass in wall geometry to generalize this to open walls
        is_closed_toroidal = true
        if is_closed_toroidal
            residue_diagonal=(2-j1)*(2-j2)+(j1-1)*(j2-1)
            residue_k0=(2-j1)*(2-j2)+(j1-3)*(j2-1)
            residue=residue_diagonal+residue_k0
        end

        # minus residue value
        work[j] = work[j] - isgn * aval1 + residue
        if j == jres
            ak0i -= isgn * aval1
        end


        # Only when plasma/plasma, log singularity activate. (S1)
        if iops == 1 && iopw != 0
            greenfunction_mat[j,js1] -= log_correction_2 / x_obs
            greenfunction_mat[j,js2] -= log_correction_1 / x_obs
            greenfunction_mat[j,js3] -= log_correction_0 / x_obs
            greenfunction_mat[j,js4] -= log_correction_1/x_obs
            greenfunction_mat[j,js5] -= log_correction_2/x_obs
        end

        # Store all the datas of work in grdgre, gren
        grad_greenfunction_mat[(j1-1)*mtheta + j, (j2-1)*mtheta .+ (1:mtheta)] .= work[1:mtheta]
        greenfunction_mat[j, 1:mtheta] ./= 2π

    end

end

"""
    fourier_inverse_transform!(gll, gil, cs, m00, l00, mth, lmin, lmax, dth)

    Purpose:
      This routine performs the inverse Fourier transform of gil onto gll
      using Fourier coefficients stored in cs.
    
    Inputs:
      gil(i,l)   : input matrix of size (mth × mpert), the Fourier-space data
      cs(j,l)    : Fourier coefficient matrix (mth × mpert)
      m00, l00   : integer offsets in the gil matrix
      mth        : number of θ-grid points (dimension of gil along i)
      mpert      : number of Fourier modes
    
    Output:
      gll(l2,l1) : output matrix updated in-place
"""
function fourier_inverse_transform!(
    gll::Matrix{Float64}, 
    gil::Matrix{Float64}, 
    cs::Matrix{Float64},
    m00::Int, 
    l00::Int, 
    mth::Int, 
    mpert::Int
    )

    # Zero out gll block
    fill!(view(gll, 1:mpert, 1:mpert), 0.0)

    # Inverse Fourier transform via matrix multiply: gll = cs^T * gil * (2π * dth)
    # This computes: gll[l2, l1] = (2π * dth) * Σ_i cs[i, l2] * gil[i, l1]
    dth = 2π / mth
    mul!(gll, cs', view(gil, m00+1:m00+mth, l00+1:l00+mpert), 2π * dth, 0.0)
end

"""
    fourier_transform!(gil, gij, cs, m00, l00, mth, mpert)

    Purpose:
      This routine performs a truncated Fourier transform of gij onto gil
      using Fourier coefficients stored in cs.
    
    Inputs:
      gij(i,j)   : input matrix of size (mth × mth), the "physical-space" data
      cs(j,l)    : Fourier coefficient matrix (mth × mpert)
      m00, l00   : integer offsets in the gil matrix
      mth        : number of θ-grid points (dimension of gij along i, j)
      mpert      : number of Fourier modes

    Output:
      gil(i', l') : matrix updated in-place, where i' = m00 + i and l' = l00 + l
"""
function fourier_transform!(
    gil::Matrix{Float64},
    gij::Matrix{Float64},
    cs::Matrix{Float64},
    m00::Int,
    l00::Int,
    mth::Int,
    mpert::Int
)

    # Zero out relevant gil block
    fill!(view(gil, m00+1:m00+mth, l00+1:l00+mpert), 0.0)

    # Accumulate Fourier transform via matrix multiply: gil = gij * cs
    # This computes: gil[i, l] = Σ_j gij[i, j] * cs[j, l]
    mul!(view(gil, m00+1:m00+mth, l00+1:l00+mpert), gij, cs)

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
    # nrows = 2 * mth * jmax1
    # ncols = 2 * mth * jmax1

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