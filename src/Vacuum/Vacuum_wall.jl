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
    # Temporary arrays for interpolation and arc-length calculation
    thlag = zeros(Float64, mw1) # Normalized input parameter [0, 1]
    ell   = zeros(Float64, mw1) # Cumulative arc length
    thgr  = zeros(Float64, mw1) # New parameter distribution for equal spacing
    xout  = zeros(Float64, mw1) # Uniformly spaced R-coordinates
    zout  = zeros(Float64, mw1) # Uniformly spaced Z-coordinates

    # 1. Define initial normalized parameter thlag
    dt = 1.0 / (mw1 - 1)
    for iw in 1:mw1
        thlag[iw] = dt * (iw - 1)
    end

    # 2. Calculate cumulative arc length using numerical integration
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

    # 3. Re-parameterize based on equal arc-length segments
    total_length = ell[mw1]
    ds_uniform = total_length / (mw1 - 1)
    
    for i in 1:mw1
        target_s = ds_uniform * (i - 1)
        # Find the value of 'thlag' that corresponds to the target arc length 's'
        f_th, _ = lagrange1d(ell, thlag, mw1, 3, target_s, 0)
        thgr[i] = f_th
    end

    # 4. Compute final output coordinates (xout, zout)
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
    wwall(wall_settings::WallShapeSettings, vac_glob::VacuumGlobalsType)

    return x array and z array for length. 
        
"""
function wwall(inputs::VacuumInput, wall_settings::WallShapeSettings, plasma_surf::PlasmaGeometry)
    mth = inputs.mtheta
    xinf = plasma_surf.x
    zinf = plasma_surf.z

    # Output arrays
    xwal1 = zeros(Float64, mth)
    zwal1 = zeros(Float64, mth)
    
    is_closed_toroidal = true
    if inputs.farwall_flag
        return xwal1, zwal1, is_closed_toroidal
    end

    # Common geometric parameters
    xmin, xmax, zmin, zmax = bounds(xinf, zinf, 1, mth)
    plrad = 0.5 * (xmax - xmin)
    xmaj = 0.5 * (xmax + xmin)
    
    # Destructuring settings for readability
    (; aw, bw, cw, dw, tw, a) = wall_settings
    wcentr = 0.0 # Initialize

    if wall_settings.shape == :conformal
        wcentr = xmaj
        csmin = min(0.1, 0.1 * minimum(xinf))
        for i in 1:mth
            j = (i == 1) ? mth : i - 1
            k = (i == mth) ? 1 : i + 1
            # Normal vector calculation
            alph = atan(xinf[k] - xinf[j], zinf[j] - zinf[k])
            xwal1[i] = max(csmin, xinf[i] + a * plrad * cos(alph))
            zwal1[i] = zinf[i] + a * plrad * sin(alph)
        end

    elseif wall_settings.shape == :elliptical
        # Only calculate these if elliptical
        zrad = 0.5 * (zmax - zmin)
        zh = sqrt(abs(zrad^2 - plrad^2))
        zmuw = log((a/zh) + sqrt((a/zh)^2 + 1)) 
        bw_eff = (zh * cosh(zmuw)) / a 
        
        for i in 1:mth
            the = (i - 1) * (2π / mth)
            xwal1[i] = xmaj + a * cos(the)
            zwal1[i] = -bw_eff * a * sin(the)
        end

    elseif wall_settings.shape == :dee
        wcentr = xmaj + cw * plrad
        for i in 1:mth
            the = (i - 1) * (2π / mth)
            xwal1[i] = wcentr + plrad * (1.0 + a - cw) * cos(the + dw * sin(the))
            zwal1[i] = -bw * plrad * (1.0 + a - cw) * sin(the + tw * sin(2.0*the)) - aw * plrad * sin(2.0*the)
        end

    elseif wall_settings.shape == :mod_dee
        wcentr = cw
        for i in 1:mth
            the = (i - 1) * (2π / mth)
            xwal1[i] = cw + a * cos(the + dw * sin(the))
            zwal1[i] = -bw * a * sin(the + tw * sin(2.0*the)) - aw * sin(2.0*the)
        end

    elseif wall_settings.shape == :from_file
        # Load wall geometry from external file "wall_geo.in"
        wcentr = 0.0
        open("wall_geo.in", "r") do io
            npots0 = parse(Int, readline(io))  # Number of points in file
            wcentr = parse(Float64, readline(io)) 
            readline(io) # Skip header/comment line

            if npots0 < mth
                @error "ERROR: wall_geo.in contains fewer points ($npots0) than mth ($mth)."
                error("Wall geometry file size mismatch")
            end

            for i in 1:mth
                line = split(readline(io))
                # Assumes file format: [index  R_coord  Z_coord]
                xwal1[i] = parse(Float64, line[2])
                zwal1[i] = parse(Float64, line[3])
            end
        end
    else
        error("Wall shape '$(wall_settings.shape)' is not implemented.")
    end

    # Optional: Re-parameterization
    if wall_settings.leqarcw == 1
        xwal1, zwal1, _, _, _ = eqarcw(xwal1, zwal1, mth)
    end

    # Save results
    h5open("wall_shape.h5", "w") do file
        attributes(file)["x_center"] = wcentr
        write(file, "x", xwal1)
        write(file, "z", zwal1)
    end

    return xwal1, zwal1, is_closed_toroidal
end

# function wwall(inputs::VacuumInput, wall_settings::WallShapeSettings, plasma_surf::PlasmaGeometry)

#     mth = inputs.mtheta
#     mth1 = inputs.mtheta + 1
#     mth2 = inputs.mtheta + 2

#     xinf = plasma_surf.x
#     zinf = plasma_surf.z

#     aw = wall_settings.aw
#     bw = wall_settings.bw
#     cw = wall_settings.cw
#     dw = wall_settings.dw
#     tw = wall_settings.tw

#     a = wall_settings.a
#     # b = wall_settings.b
#     # abulg = wall_settings.abulg
#     # bbulg = wall_settings.bbulg
#     # tbulg = wall_settings.tbulg
#     xma = wall_settings.xma
#     # zma = wall_settings.zma
#     # isph = wall_settings.isph


#     dth = 2.0 * π / (mth)
#     inside = 0

#     # --- Array Initialization --- # <--- Added section
#     xwal1 = zeros(Float64, mth)
#     zwal1 = zeros(Float64, mth)
#     # thet = zeros(Float64, mth)  # Used in ishape==3

#     insect = false
#     # isph = 0 # Corresponds to Fortran's `data iplt/0/`. iplt is only used at the end, so isph=0 is initialized here.
#     # iplt = 0 # <--- Added (Fortran's data iplt/0/ initialization)

#     is_closed_toroidal = true

#     if inputs.farwall_flag
#         @info "Enforcing no-wall vacuum energy conditions"
#         # Return zeros, no wall defined
#         return xwal1, zwal1, is_closed_toroidal
#     end

#     xshift = a
#     mthalf = floor(Int, mth2 / 2) 
#     xmin, xmax, zmin, zmax = bounds(xinf, zinf, 1, mth) # Replaced Fortran loop with bounds function (same as original Julia code)
#     plrad = 0.5 * (xmax - xmin)
#     xmaj = 0.5 * (xmax + xmin)
#     zmid = 0.5 * (zmax + zmin)
#     zrad = 0.5 * (zmax - zmin)
#     scale = (zmax - zmin)

#     if ((xmax - xmin) / 2.0) > scale
#         scale = (xmax - xmin) / 2.0
#     end

#     scale = 1.0
#     aw = aw * scale
#     bw = bw * scale
#     delta1 = dw * (xinf[1] - xma)
    
#     if wall_settings.shape == "conformal"
#         wcentr = xmaj
#         csmin = min(0.1, 1e-1 * minimum(view(xinf, 1:mth)))
#         for i in 1:mth
#             j = i - 1
#             if j<1
#                 j = mth - j
#             end
#             k = i + 1
#             if k>mth
#                 k = k - mth
#             end
#             alph = atan(xinf[k] - xinf[j], zinf[j] - zinf[k]) # Fortran's ATAN2
#             xwal1[i] = xinf[i] + a * plrad * cos(alph)
#             zwal1[i] = zinf[i] + a * plrad * sin(alph)
#             if xwal1[i] <= csmin
#                 xwal1[i] = csmin
#             end
#         end
#     else
#         error("Wall shape $(wall_settings.shape) not implemented yet.")
#     end

#     if wall_settings.shape == "elliptical"
#         zh = sqrt(abs(zrad^2 - plrad^2))
#         zah = a / zh
#         zph = plrad / zh
#         zmup = 0.5 * log((zrad + plrad) / (zrad - plrad)) 
#         zmuw = log(zah + sqrt(zah^2 + 1)) 
#         zxmup = exp(zmup)
#         zxmuw = exp(zmuw)
#         zbwal = zh * cosh(zmuw) # Major radius of wall
#         bw = zbwal / a          # Elongation of wall
#         for i in 1:mth2
#             the = (i - 1) * dth
#             xwal1[i] = xmaj + a * cos(the)
#             zwal1[i] = -bw * a * sin(the)
#         end
#     end

#     if wall_settings.shape == "dee"
#         wcentr = xmaj + cw * plrad
#         for i in 1:mth2
#             the0 = (i - 1) * dth
#             the = the0
#             sn2th = sin(2.0 * the)
#             xwal1[i] = xmaj + cw * plrad + plrad * (1.0 + a - cw) * cos(the + dw * sin(the))
#             zwal1[i] = -bw * plrad * (1.0 + a - cw) * sin(the + tw * sn2th) - aw * plrad * sn2th
#         end
#     end

#     if wall_settings.shape == "mod-dee"
#         wcentr = cw
#         for i in 1:mth2
#             the0 = (i - 1) * dth
#             the = the0
#             sn2th = sin(2.0 * the)
#             xwal1[i] = cw + a * cos(the + dw * sin(the))
#             zwal1[i] = -bw * a * sin(the + tw * sn2th) - aw * sn2th
#         end
#     end

#     if wall_settings.shape == "from_file"
#         wcentr = 0.0
#         open("wall_geo.in", "r") do io
#             npots0 = parse(Int, readline(io))
#             wcentr = parse(Float64, readline(io)) # wcentr is read but not used in Fortran
#             readline(io) # Skip the next line

#             if npots0 != mth + 2
#                 @error "ERROR: Number of points in wall_geo.in must be equal to mth+2."
#                 error("Wall geometry error") # Stop execution
#             end

#             thetatmp_dummy = zeros(Float64, npots0)
#             for i in 1:npots0
#                 line = split(readline(io))
#                 thetatmp_dummy[i] = parse(Float64, line[1]) # Read but not used
#                 xwal1[i] = parse(Float64, line[2])
#                 zwal1[i] = parse(Float64, line[3])
#             end
#         end

#         if xwal1[mth1] != xwal1[1] || zwal1[mth1] != zwal1[1]
#             @error "ERROR: First point in wall_geo.in must be equal to 2nd last point."
#             error("Wall geometry error")
#         end
#         if xwal1[mth2] != xwal1[2] || zwal1[mth2] != zwal1[2]
#             @error "ERROR: Last point in wall_geo.in must be equal to 2nd point."
#             error("Wall geometry error")
#         end
#     end

#     xmx = xma + xshift


#     if wall_settings.leqarcw == 1
#         # error("eqarcw function not implemented yet.")
#         xpp, zpp, ww1, ww2, ww3 = eqarcw( xwal1, zwal1, mth1 ) # <--- Assuming eqarcw returns values
#         for i in 1:mth1
#             xwal1[i] = xpp[i]
#             zwal1[i] = zpp[i]
#         end
#     end

#     h5open("wall_shape.h5", "w") do file
#         attributes(file)["x_center"] = wcentr
#         write(file, "x", xwal1)
#         write(file, "z", zwal1)
#     end

#         # if iplt <= 0 # <--- Fortran's if ( iplt .le. 0 ) then
#     #     xmx = xmaj
#     #     zma = 0.0
#     #     iplt = 1
#     # end

#     # if insect # <--- Added (Fortran's warning message)
#     #     @warn "There are at least $inside wall points in the plasma"
#     #     # Corresponds to Fortran's errmes call
#     #     # errmes("vacdat") 
#     # end

#     return xwal1, zwal1, is_closed_toroidal
# end