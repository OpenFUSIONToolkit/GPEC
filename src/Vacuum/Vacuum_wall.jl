"""
    bounds(x, z, istart, ifinish)

Calculates the minimum and maximum X and Z coordinates within a specified range of two input vectors.

# Arguments
- `x::AbstractVector{<:Real}`: Input vector for X coordinates.
- `z::AbstractVector{<:Real}`: Input vector for Z coordinates.
- `istart::Integer`: Starting index for the range (1-based).
- `ifinish::Integer`: Ending index for the range (1-based).

# Returns
- A `Tuple{Real, Real, Real, Real}` containing (xmin, xmax, zmin, zmax).

# Throws
- `ArgumentError`: If indices are out of bounds or `istart > ifinish`.
"""
function bounds(x::AbstractVector{<:Real}, z::AbstractVector{<:Real}, istart::Integer, ifinish::Integer)

    if istart > ifinish
        throw(ArgumentError("Starting index ($istart) cannot be greater than ending index ($ifinish)."))
    end
    
    if !(1 <= istart <= length(x)) || !(1 <= ifinish <= length(x))
        throw(ArgumentError("Indices [$istart, $ifinish] are out of bounds for x vector with length $(length(x))."))
    end
    if !(1 <= istart <= length(z)) || !(1 <= ifinish <= length(z))
        throw(ArgumentError("Indices [$istart, $ifinish] are out of bounds for z vector with length $(length(z))."))
    end

    range = istart:ifinish
    xmin = minimum(view(x, range))
    xmax = maximum(view(x, range))
    zmin = minimum(view(z, range))
    zmax = maximum(view(z, range))

    return xmin, xmax, zmin, zmax
end

"""
    eqarcw(xin, zin, mw1)

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
function eqarcw(xin::Vector{Float64}, zin::Vector{Float64}, mw1::Int)
    
    thlag = zeros(Float64, mw1)
    ell = zeros(Float64, mw1)
    thgr = zeros(Float64, mw1)
    xout = zeros(Float64, mw1)
    zout = zeros(Float64, mw1)

    # (Assuming the lagrange1d function is defined previously)
    
    # Initialize thlag as normalized parameter from 0 to 1
    for iw in 1:mw1
        thlag[iw] = (1.0 / (mw1 - 1)) * (iw - 1)
    end

    # Calculate cumulative arc length
    ell[1] = 1.0e-8 # Small non-zero initial length
    for iw in 2:mw1
        thet = (thlag[iw] + thlag[iw - 1]) / 2.0
        _ , d_xin_d_thlag = lagrange1d(thlag, xin, mw1, 3, thet, 1)
        _ , d_zin_d_thlag = lagrange1d(thlag, zin, mw1, 3, thet, 1)
        
        xtzt = sqrt(d_xin_d_thlag^2 + d_zin_d_thlag^2)
        ell[iw] = ell[iw - 1] + xtzt / (mw1 - 1)
    end

    # Re-parameterize based on equal arc lengths
    for i in 1:mw1
        elgr = (ell[mw1] / (mw1 - 1)) * (i - 1)
        f, _ = lagrange1d(ell, thlag, mw1, 3, elgr, 0)
        thgr[i] = f
    end

    # Get xout and zout using the re-parameterized thgr
    for i in 1:mw1
        ttt = thgr[i]

        f_x, _ = lagrange1d(thlag, xin, mw1, 3, ttt, 0)
        xout[i] = f_x
        f_z, _ = lagrange1d(thlag, zin, mw1, 3, ttt, 0)
        zout[i] = f_z
    end
    
    return xout, zout, ell, thgr, thlag
end

"""
    adjustb(betin, betout_ref, a_, bw_, cw_, dw_, xmaj_, plrad_, ishape_)

Adjusts the `betin` angle based on the `ishape_` and other wall/plasma parameters.
This function takes `betout_ref` as a `Ref` so it can modify the output value in-place.

# Arguments
- `betin::Float64`: Input angle.
- `a_::Float64`: Wall parameter.
- `bw_::Float64`: Wall parameter (elongation or height).
- `cw_::Float64`: Wall parameter (center or offset).
- `dw_::Float64`: Wall parameter (triangularity).
- `xmaj_::Float64`: Magnetic axis X coordinate.
- `plrad_::Float64`: Plasma radius.
- `ishape_::Int`: Integer indicating the wall shape type.
"""
function adjustb(betin::Float64, a_::Float64, bw_::Float64, cw_::Float64, dw_::Float64,
                 xmaj_::Float64, plrad_::Float64, ishape_::Int)

    # These local variables r0 and r are correctly scoped and used for intermediate calculations.
    local r0::Float64 = 0.0
    local r::Float64 = 0.0

    if ishape_ == 31
        r0 = cw_
        r  = a_
    elseif ishape_ == 21
        r0 = xmaj_ + cw_ * plrad_
        r  = plrad_ * (1.0 + a_ - cw_)
    else
        @warn "adjustb: Unsupported ishape_ value: $ishape_. r0 and r will remain 0.0."
    end

    bet2 = betin # No change here, bet2 is a local copy of betin

    if bw_ == 0.0
        @warn "adjustb: Division by zero detected (bw_ is 0.0). Setting betout_ref to NaN."
        betout = NaN # Correctly assign NaN to the value inside the Ref
        return nothing
    end
    betout = abs(atan(tan(bet2) / bw_)) # Correctly assign to the value inside the Ref

    return betout
end


"""
    d3dvesl!(r0, z0, a0, e0, ar, az, nval, zst, r, z, npts, ier_ref)

Defines the shape of the DIII-D vacuum vessel based on Fourier series coefficients.

# Arguments
- `r0::Float64`: Reference major radius.
- `z0::Float64`: Reference vertical position.
- `a0::Float64`: Minor radius scaling factor.
- `e0::Float64`: Elongation scaling factor.
- `ar::Vector{Float64}`: Fourier coefficients for radial component.
- `az::Vector{Float64}`: Fourier coefficients for vertical component.
- `nval::Int`: Number of Fourier coefficients.
- `zst::Float64`: Starting vertical position for calculation (used in an unused branch).
- `r::Vector{Float64}`: Output vector for R coordinates of the wall (modified in-place).
- `z::Vector{Float64}`: Output vector for Z coordinates of the wall (modified in-place).
- `npts::Int`: Number of points to generate for the wall.
"""
function d3dvesl!(r0::Float64, z0::Float64, a0::Float64, e0::Float64,
                 ar::Vector{Float64}, az::Vector{Float64}, nval::Int, zst::Float64,
                 r::Vector{Float64}, z::Vector{Float64}, npts::Int)

    pii = acos(-1.0) # Using Julia's built-in pi constant from π
    ier_ref = 0 # Initialize error flag

    local arcst::Float64

    if abs(z0 - zst) <= 1.0e-6
        # This branch is usually taken, as zstart is set to 0.0 in d3dwall
        arcst = 0.0
    else
        local isgn::Int
        local zpcmp::Float64
        local arci::Float64
        local arcf::Float64
        local dfi::Float64
        local arca::Float64
        local zza::Float64
        local arcb::Float64
        local zzb::Float64
        local ackb::Float64
        local arcc::Float64
        local zzc::Float64
        local ackc::Float64
        local dzp::Float64
        local dzm::Float64
        local dzt::Float64
        local dcf1::Float64
        local dcf2::Float64
        local zdf::Float64

        if z0 < zst
            isgn = +1
        end
        if z0 > zst
            isgn = -1
        end
        zpcmp = (zst - z0) / (a0 * e0)
        arci = 0.0
        arcf = 0.5 * pii
        dfi = (arcf - arci) / npts
        arca = arci
        zza = 0.0
        for j = 2:npts
            arcb = arca + isgn * dfi
            zzb = 0.0
            for k = 1:nval
                ackb = k * arcb
                zzb = zzb + az[k] * sin(ackb)
            end # 5 continue
            if (zza - zpcmp) * (zzb - zpcmp) <= 0.0
                arcc = arcb + isgn * dfi
                zzc = 0.0
                for k = 1:nval
                    ackc = k * arcc
                    zzc = zzc + az[k] * sin(ackc)
                end # 10 continue
                @goto label_25
            else
                arca = arcb
                zza = zzb
            end
        end # 20 continue
        ier_ref[] = 1
        return
        @label label_25 # 25 continue
        dzp = zzc - zzb
        dzm = zzb - zza
        dzt = zzc - zza
        dcf1 = dfi * (dzm / dzp + dzp / dzm) / dzt
        dcf2 = dfi * (1.0 / dzp - 1.0 / dzm) / dzt
        zdf = zpcmp - zzb
        arcst = arcb + dcf1 * zdf + dcf2 * zdf * zdf
    end

    arc0 = arcst
    arc1 = arcst + 2.0 * pii
    darc = (arc1 - arc0) / npts

    for j = 1:npts
        arc = arc0 + (j - 1.0) * darc
        sumr = 0.0
        sumz = 0.0
        for k = 1:nval
            arck = k * arc
            sumr = sumr + ar[k] * cos(arck)
            sumz = sumz + az[k] * sin(arck)
        end # 50 continue
        rpval = r0 + a0 * sumr
        zpval = z0 - e0 * a0 * sumz
        r[j] = rpval
        z[j] = zpval
    end # 100 continue

    return nothing
end



"""
    d3dwall!(xwall, ywall, mthh, iomod, iotty1, rext_in, zlim_val_in)

Defines the shape of the DIII-D wall by calling `d3dvesl`.

# Arguments
- `xwall::Vector{Float64}`: Output vector for X coordinates of the DIII-D wall (modified in-place).
- `ywall::Vector{Float64}`: Output vector for Y (Z in Cartesian) coordinates of the DIII-D wall (modified in-place).
- `mthh::Int`: Number of points to generate for the wall.
- `rext_in::Float64`: External scaling factor for the wall.
- `zlim_val_in::Float64`: Global z-limit value (used in `d3dvesl` implicitly).
"""
function d3dwall!(xwall::Vector{Float64}, ywall::Vector{Float64}, mthh::Int,rext_in::Float64)

    ncdf = 26
    rwi = [
        1.0, 0.05526794, -0.1738114, 0.01850757, 0.03714965, -0.02882647,
        -0.002357329, 0.009548103, -0.01214923, -0.001853416, 0.006837493,
        -0.001711245, 0.002270762, 0.003689963, -0.003959393, -0.001098017,
        0.003745465, -0.0002157904, -0.0003977743, -0.0002725623,
        -0.001005857, -0.000004579016, 0.002396789, -0.0007057043,
        0.001158347, 0.0003552319
    ]
    zwi = [
        1.0, -0.03236632, -0.1629422, 0.06013983, 0.01167756, -0.02579542,
        0.01626464, -0.002085857, -0.009098639, 0.01022163, -0.004388253,
        -0.009367258, 0.008308497, 0.004765150, -0.004611675, -0.001121423,
        -0.0002501100, 0.4282634e-03, 0.2669702e-02, -0.1073800e-02, # Corrected: -0.1073800d-02 from Fortran
        -0.2191338e-02, 0.1328267e-02, 0.5050959e-03, -0.5758863e-03,
        0.9348883e-03, 0.7094351e-03
    ]

    rwall0 = 1.6400000
    zwall0 = 0.0000000
    awall0 = 0.8839410
    ewall0 = 1.4037020

    nwcoef = ncdf

    rwll = rwall0
    zwll = zwall0
    awll = awall0 * rext_in # Use rext_in here
    ewll = ewall0

    # In the Fortran code, `zstart = zlim` and then `zlim = 0.0`, followed by `zstart = 0.0`.
    # This effectively makes `zstart` 0.0 for the call to `d3dvesl`.
    zstart = 0.0 # This overwrites zlim_val_in for the purpose of d3dvesl.

    nwalp = mthh

    d3dvesl!(rwll, zwll, awll, ewll, rwi, zwi, nwcoef, zstart, xwall, ywall, nwalp)

    xwall[mthh+1] = xwall[1]
    ywall[mthh+1] = ywall[1]
end

"""
    wwall(wall_settings::WallShapeSettings, vac_glob::VacuumGlobalsType)

    return x array and z array for length. 
        
"""
function wwall(inputs::VacuumInputType, wall_settings::WallShapeSettings, plasma_surf::PlasmaGeometry)

    mth = inputs.mtheta
    mth1 = inputs.mtheta + 1
    mth2 = inputs.mtheta + 2

    farwal = inputs.farwal_flag
    xinf = plasma_surf.x
    zinf = plasma_surf.z

    aw = wall_settings.aw
    bw = wall_settings.bw
    cw = wall_settings.cw
    dw = wall_settings.dw
    tw = wall_settings.tw

    ishape = wall_settings.ishape
    a = wall_settings.a
    b = wall_settings.b
    abulg = wall_settings.abulg
    bbulg = wall_settings.bbulg
    tbulg = wall_settings.tbulg
    xma = wall_settings.xma
    zma = wall_settings.zma
    isph = wall_settings.isph

    leqarcw = wall_settings.leqarcw

    dth = 2.0 * π / (mth1) # (2.0*pi / (mth+1))
    inside = 0

    # --- Array Initialization --- # <--- Added section
    xwal1 = zeros(Float64, mth2)
    zwal1 = zeros(Float64, mth2)
    thet = zeros(Float64, mth2)  # Used in ishape==3

    awsave = aw
    bwsave = bw
    insect = false
    isph = 0 # Corresponds to Fortran's `data iplt/0/`. iplt is only used at the end, so isph=0 is initialized here.
    iplt = 0 # <--- Added (Fortran's data iplt/0/ initialization)


    # --- Shape Logic ---
    if a < -100.0
        isph = 1
        ishape = -10
        xmnp, xmxp, zmnp, zmxp = bounds(xinf, zinf, 1, mth)
        xmin = xmnp
        xmax = xmxp
        zmin = zmnp
        zmax = zmxp
        plrad = 0.5 * (xmax - xmin)
        xmaj = 0.5 * (xmax + xmin)
        zmid = 0.5 * (zmax + zmin)
        hrad = xmax + aw * (xmax - xmaj)
        vrad = zmax + bw * (zmax - zmid)
        for i in 1:mth1
            xi = xinf[i] - xmaj
            zeta = zinf[i] - zmid
            bbb = (xi * vrad)^2 + (zeta * hrad)^2
            ccc = -xmaj * vrad * xi + hrad * sqrt(bbb - (zeta * xmaj)^2)
            xwal1[i] = xmaj + xi * vrad * ccc / bbb
            zwal1[i] = zmid + zeta * vrad * ccc / bbb
        end
        @goto cleanup # Direct jump to cleanup section
    elseif a > -10.0
        lfix = true
    else
        lfix = false # Initialize lfix for all paths
    end

    if farwal
        @info "No wall"
        # Return zeros, no wall defined
        return xwal1, zwal1
    end

    xshift = a
    mthalf = floor(Int, mth2 / 2) 
    xmin, xmax, zmin, zmax = bounds(xinf, zinf, 1, mth) # Replaced Fortran loop with bounds function (same as original Julia code)
    plrad = 0.5 * (xmax - xmin)
    xmaj = 0.5 * (xmax + xmin)
    zmid = 0.5 * (zmax + zmin)
    zrad = 0.5 * (zmax - zmin)
    scale = (zmax - zmin)

    if ((xmax - xmin) / 2.0) > scale
        scale = (xmax - xmin) / 2.0
    end

    scale = 1.0
    aw = aw * scale
    bw = bw * scale
    delta1 = dw * (xinf[1] - xma)

    # ishape=2 Elliptical shell
    if ishape == 2
        zh = sqrt(abs(zrad^2 - plrad^2))
        zah = a / zh
        zph = plrad / zh
        zmup = 0.5 * log((zrad + plrad) / (zrad - plrad)) 
        zmuw = log(zah + sqrt(zah^2 + 1)) 
        zxmup = exp(zmup)
        zxmuw = exp(zmuw)
        zbwal = zh * cosh(zmuw) # Major radius of wall
        bw = zbwal / a          # Elongation of wall
        for i in 1:mth2
            the = (i - 1) * dth
            xwal1[i] = xmaj + a * cos(the)
            zwal1[i] = -bw * a * sin(the)
        end
    end

    # ishape=3
    if ishape == 3
        # `thet` array was already allocated at the start of the function
        for i in 1:mth2
            rr = (xinf[i] - xma)^2 + (zinf[i] - zma)^2
            ro = sqrt(rr)
            the = atan((zinf[i] - zma), (xinf[i] - xma)) # Use atan2 for correct quadrant
            thex = abs(the)
            lsgn = 1
            if xma > xinf[i]
                the = the + π
            end
            if i <= mthalf
                if xma > xinf[i]
                    thex = π - thex
                end
                thet[i] = abs(thex) # <--- Now safe
            end
            if !lfix
                ro = ro + delta1
                xwal1[i] = xma + lsgn * ro * cos(the)
                zwal1[i] = zma + lsgn * ro * sin(the)
            else
                xshift = (xmax + xmin) / 2.0
                xshift = a
                the = (i - 1) * dth
                xwal1[i] = xshift + aw * cos(the + dw * sin(the))
                zwal1[i] = zma - bw * sin(the)
            end
            if i > mthalf
                continue
            end
            if zwal1[i] < zmin
                continue
            end
            j = i
            insect = false
            jsmall = j
            jlarge = j
            ics = 1
            if xma >= xinf[i]
                ics = -1
            end
            while true
                if j < 1 || j > mth # <--- Modified Fortran's j.lt.1 and j.ge.mthalf(mth?) loop conditions for Julia
                    j = j + ics # Increment j first to avoid Fortran's cycle
                    if j < 1 || j > mth # Check again
                         break # Stop if j is out of bounds
                    end
                end

                if zinf[j] >= zwal1[i]
                    jsmall = j
                else
                    break
                end

                if j == (mthalf) # Fortran's j.ge.mthalf cycle (only allowed up to mthalf)
                     break
                end
                
                j = j + ics
            end
            jlarge = (j < 1) ? 1 : ((j > mth) ? mth : j) # Handle case if j went out of bounds

            if abs(xinf[jsmall] - xma) >= abs(xwal1[i] - xma)
                insect = true
            end
            if jlarge > 0 && jlarge <= mth # Check if jlarge is a valid index
                if abs(xinf[jlarge] - xma) >= abs(xwal1[i] - xma)
                    insect = true
                end
            end
            if !insect
                continue
            end
            inside += 1
        end
    end

    # ishape=4 Modified dee-shaped wall independent of plasma geometry
    if ishape == 4
        wcentr = cw
        for i in 1:mth2
            the0 = (i - 1) * dth
            the = the0
            sn2th = sin(2.0 * the)
            xwal1[i] = cw + a * cos(the + dw * sin(the))
            zwal1[i] = -bw * a * sin(the + tw * sn2th) - aw * sn2th
        end
    end

    # ishape=5 Dee-shaped wall scaled to plasma
    if ishape == 5
        wcentr = xmaj + cw * plrad
        for i in 1:mth2
            the0 = (i - 1) * dth
            the = the0
            sn2th = sin(2.0 * the)
            xwal1[i] = xmaj + cw * plrad + plrad * (1.0 + a - cw) * cos(the + dw * sin(the))
            zwal1[i] = -bw * plrad * (1.0 + a - cw) * sin(the + tw * sn2th) - aw * plrad * sn2th
        end
    end

    # ishape=6 Conforming shell
    if ishape == 6
        wcentr = xmaj
        csmin = min(0.1, 1e-1 * minimum(view(xinf, 2:mth1)))
        for i in 2:mth1
            alph = atan(xinf[i+1] - xinf[i-1], zinf[i-1] - zinf[i+1]) # Fortran's ATAN2
            xwal1[i] = xinf[i] + a * plrad * cos(alph)
            zwal1[i] = zinf[i] + a * plrad * sin(alph)
            if xwal1[i] <= csmin
                xwal1[i] = csmin
            end
        end
        xwal1[1] = xwal1[mth1]
        zwal1[1] = zwal1[mth1]
        xwal1[mth2] = xwal1[2]
        zwal1[mth2] = zwal1[2]
    end

    # ishape=7 Enclosing bean-shaped wall
    if ishape == 7
        cwr = cw * π / 180.0
        for i in 1:mth2
            the0 = (i - 1) * dth
            the = the0
            rho = aw * (1.0 + bw * cos(the))
            the2 = cwr * sin(the)
            xofsw = xmax + a * plrad - aw * (1.0 + bw)
            xwal1[i] = xofsw + rho * cos(the2)
            zwal1[i] = -b * rho * sin(the2)
        end
    end

    # ishape=8 Wall of DIII-D
    if ishape == 8
        d3dwall!(xwal1, zwal1, mth, 1.0)
    end

    # ishape=11 Dee-shaped conductor
    if ishape == 11
        for i in 1:mth2
            the = (i - 1) * dth
            plrad = 0.5 * (xmax - xmin)
            xwal1[i] = xmax + plrad * (a + aw - aw * cos(the + dw * sin(the)))
            zwal1[i] = -plrad * bw * sin(the)
        end
    end

    # ishape=12 Solid bean-shaped conductor on right
    if ishape == 12
        plrad = 0.5 * (xmax - xmin)
        xmaj = 0.5 * (xmax + xmin)
        a0 = plrad * (1.0 + aw - cw + a)
        brad = b * π / 180.0
        for i in 1:mth2
            the0 = (i - 1) * dth
            the = the0
            rho = a0 - aw * plrad * cos(the)
            the2 = brad * sin(the)
            xwal1[i] = xmaj + cw * plrad + rho * cos(the2)
            zwal1[i] = -bw * rho * sin(the2)
        end
    end

    # ishape=13 Solid bean-shaped conductor on left
    if ishape == 13
        plrad = 0.5 * (xmax - xmin)
        xmaj = 0.5 * (xmax + xmin)
        a0 = plrad * (1.0 + aw - cw + a)
        brad = b * π / 180.0
        for i in 1:mth2
            the0 = (i - 1) * dth
            the = the0
            rho = a0 + aw * plrad * cos(the)
            the2 = brad * sin(the)
            xwal1[i] = xmaj + cw * plrad - rho * cos(the2)
            zwal1[i] = -bw * rho * sin(the2)
        end
    end

    # ishape=21 Shell scaled to plasma. Gap on the inner side.
    if ishape == 21
        plrad = 0.5 * (xmax - xmin)
        xmaj = 0.5 * (xmax + xmin)
        a0 = plrad * (1.0 + aw - cw + a)
        a0b = (a0 + plrad * aw) * bw
        brad0 = b * π / 180.0
        brad = brad0
        blgrad0 = bbulg * π / 180.0
        wcentr = xmaj + cw * plrad

        # Call adjustb
        blgrado = adjustb(blgrad0, a, bw, cw, dw, xmaj, plrad, ishape) # <--- Modified

        dthb = (2.0 * aw * plrad / a0b) * (1.0 - sin(blgrado)) / cos(blgrado)
        blgrad0 = blgrad0 - dthb
        
        blgradi = adjustb(blgrad0, a, bw, cw, dw, xmaj, plrad, ishape) # <--- Modified
        
        for i in 1:mth2
            the0 = (i - 1) * dth
            thbulg = (the0 > 0.5 * π && the0 < 1.5 * π) ? blgrado : blgradi
            cost2b = cos(2 * thbulg)
            the = the0
            cost = cos(the)
            ferm = +1.0 - 2.0 / (exp(cost / tw) + 1.0)
            rho = a0 - aw * plrad * ferm
            the2 = brad * sin(the)
            cost2 = cos(2.0 * the2)
            fermb = 1.0 / (exp((cost2b - cost2) / tbulg) + 1.0)
            bulge = abulg * plrad * fermb
            xwal1[i] = xmaj + cw * plrad + rho * cos(the2 + dw * sin(the2)) + bulge
            zwal1[i] = -bw * rho * sin(the2)
        end
    end

    # ishape=24 Shell scaled to plasma. Gap on the outer side.
    if ishape == 24
        plrad = 0.5 * (xmax - xmin)
        xmaj = 0.5 * (xmax + xmin)
        a0 = plrad * (1.0 + aw - cw + a)
        brad = b * π / 180.0
        wcentr = xmaj + cw * plrad
        for i in 1:mth2
            the0 = (i - 1) * dth
            the = the0
            cost = cos(the)
            ferm = +1.0 - 2.0 / (exp(cost / tw) + 1.0)
            rho = a0 + aw * plrad * ferm
            the2 = brad * sin(the)
            xwal1[i] = xmaj + cw * plrad - rho * cos(the2 - dw * sin(the2))
            zwal1[i] = -bw * rho * sin(the2)
        end
    end

    # ishape=31 Shell independent of plasma. Gap on the inner side.
    if ishape == 31
        a0 = a + aw
        a0b = (a0 + aw) * bw
        brad0 = b * π / 180.0
        brad = brad0
        blgrad0 = bbulg * π / 180.0
        wcentr = cw

        blgrado = adjustb(blgrad0,  a, bw, cw, dw, xmaj, plrad, ishape) # <--- Modified

        dthb = (2.0 * aw / a0b) * (1.0 - sin(blgrado)) / cos(blgrado)
        blgrad0 = blgrad0 - dthb
        
        blgradi = adjustb(blgrad0, a, bw, cw, dw, xmaj, plrad, ishape) # <--- Modified

        for i in 1:mth2
            the0 = (i - 1) * dth
            thbulg = (the0 > 0.5 * π && the0 < 1.5 * π) ? blgrado : blgradi
            cost2b = cos(2.0 * thbulg)
            the = the0
            cost = cos(the)
            ferm = +1.0 - 2.0 / (exp(cost / tw) + 1.0)
            rho = a0 - aw * ferm
            the2 = brad * sin(the)
            cost2 = cos(2.0 * the2)
            fermb = 1.0 / (exp((cost2b - cost2) / tbulg) + 1.0)
            bulge = abulg * fermb
            xwal1[i] = cw + rho * cos(the2 + dw * sin(the2)) + bulge
            zwal1[i] = -bw * rho * sin(the2)
        end
    end

    # ishape=34 Shell independent of plasma. Gap on the outer side.
    if ishape == 34
        a0 = a + aw
        brad = b * π / 180.0
        wcentr = cw
        for i in 1:mth2
            the0 = (i - 1) * dth
            the = the0
            cost = cos(the)
            ferm = +1.0 - 2.0 / (exp(cost / tw) + 1.0)
            rho = a0 + aw * ferm
            the2 = brad * sin(the)
            xwal1[i] = cw - rho * cos(the2 - dw * sin(the2))
            zwal1[i] = -bw * rho * sin(the2)
        end
    end

    # ishape=41 Arbitrary wall generated by spline data from wall_geo.io
    if ishape == 41
        # Fortran's npots=npots0+5-1 seems unnecessary in Julia. Using npots0.
        npots0 = 0
        wcentr = 0.0
        thetatmp = Float64[]
        xwaltmp = Float64[]
        zwaltmp = Float64[]

        open("wall_geo.in", "r") do io
            npots0 = parse(Int, readline(io))
            wcentr = parse(Float64, readline(io))
            readline(io) # Skip the next line

            # Allocate arrays
            thetatmp = zeros(Float64, npots0)
            xwaltmp = zeros(Float64, npots0)
            zwaltmp = zeros(Float64, npots0)
            
            for i in 1:npots0
                line = split(readline(io))
                thetatmp[i] = parse(Float64, line[1])
                xwaltmp[i] = parse(Float64, line[2]) - wcentr
                zwaltmp[i] = parse(Float64, line[3])
            end
        end # file closes here

        # Allocate spline work arrays
        rioptmp = zeros(Int, 2)
        xpptmp = zeros(Float64, npots0)
        zpptmp = zeros(Float64, npots0)  # <--- Added zpptmp allocation
        ww1tmp = zeros(Float64, npots0)
        ww2tmp = zeros(Float64, npots0)
        ww3tmp = zeros(Float64, npots0)
        tabtmp = zeros(Float64, 3)

        rioptmp[1] = 4
        rioptmp[2] = 4
        spl1d1!(npots0, thetatmp, xwaltmp, xpptmp, rioptmp, 1, ww1tmp, ww2tmp, ww3tmp)

        for i in 1:mth1
            the0 = (i - 1) * dth
            spl1d2(npots0,thetatmp,xwaltmp,xpptmp,1,the0,tabtmp)
            xwal1[i] = tabtmp[1]*a + wcentr
        end

        rioptmp[1] = 4
        rioptmp[2] = 4
        # <--- Modified: xwaltmp -> zwaltmp, xpptmp -> zpptmp
        spl1d1!(npots0, thetatmp, zwaltmp, zpptmp, rioptmp, 1, ww1tmp, ww2tmp, ww3tmp) 

        for i in 1:mth1
            the0 = (i - 1) * dth
            # <--- Modified: xpptmp -> zpptmp
            spl1d2(npots0,thetatmp,zwaltmp,zpptmp,1,the0,tabtmp)
            zwal1[i] = tabtmp[1]*a
        end

        xwal1[1] = xwal1[mth1]
        zwal1[1] = zwal1[mth1]
        xwal1[mth2] = xwal1[2]
        zwal1[mmth2] = zwal1[2]
    end

    # ishape=42 Arbitrary wall generated by position data
    if ishape == 42
        wcentr = 0.0
        open("wall_geo.in", "r") do io
            npots0 = parse(Int, readline(io))
            wcentr = parse(Float64, readline(io)) # wcentr is read but not used in Fortran
            readline(io) # Skip the next line

            if npots0 != mth + 2
                @error "ERROR: Number of points in wall_geo.in must be equal to mth+2."
                error("Wall geometry error") # Stop execution
            end

            thetatmp_dummy = zeros(Float64, npots0)
            for i in 1:npots0
                line = split(readline(io))
                thetatmp_dummy[i] = parse(Float64, line[1]) # Read but not used
                xwal1[i] = parse(Float64, line[2])
                zwal1[i] = parse(Float64, line[3])
            end
        end

        if xwal1[mth1] != xwal1[1] || zwal1[mth1] != zwal1[1]
            @error "ERROR: First point in wall_geo.in must be equal to 2nd last point."
            error("Wall geometry error")
        end
        if xwal1[mth2] != xwal1[2] || zwal1[mth2] != zwal1[2]
            @error "ERROR: Last point in wall_geo.in must be equal to 2nd point."
            error("Wall geometry error")
        end
    end

    xmx = xma + xshift

    # --- Cleanup ---
    @label cleanup # Target for the `goto` from ishape=-10
    xwal1[mth1] = xwal1[1]
    zwal1[mth1] = zwal1[1]
    xwal1[mth2] = xwal1[2]
    zwal1[mth2] = zwal1[2]
    if leqarcw == 1
        xpp, zpp, ww1, ww2, ww3 = eqarcw( xwal1, zwal1, mth1 ) # <--- Assuming eqarcw returns values
        for i in 1:mth1
            xwal1[i] = xpp[i]
            zwal1[i] = zpp[i]
        end
    end
    xwal = xwal1
    zwal = zwal1

    # Call to savewall(wcentr,xwal1,zwal1) from Fortran can be added here if implemented in Julia
    # savewall(wcentr, xwal1, zwal1) 

    if iplt <= 0 # <--- Fortran's if ( iplt .le. 0 ) then
        xmx = xmaj
        zma = 0.0
        iplt = 1
    end

    if insect # <--- Added (Fortran's warning message)
        @warn "There are at least $inside wall points in the plasma"
        # Corresponds to Fortran's errmes call
        # errmes("vacdat") 
    end

    aw = awsave # restore value of aw
    bw = bwsave # restore value of bw

    xwal = xwal1
    zwal = zwal1
    return xwal, zwal
end