"""
Converts inverse equilibrium to straight-fieldline coordinates. Based on inverse.f
Of the entries we need to return: PlasmaEquilibrium(equil_params, sq_out, rzphi_out, eqfun_out, ro, zo, psio),
we only need to generate: sq_out, rzphi_out, eqfun_out. This is because we pass in InverseRunInput(equil_in, 
sq_in, rz_in, ro, zo, psio).

"""

"""
    inverse_extrap(xx::Matrix{Float64}, ff::Matrix{Float64}, x::Float64) -> Vector{Float64}

Performs component-wise Lagrange extrapolation for a vector-valued function.

## Arguments:
- `xx`: A (m × n) matrix where each row contains the x-values for each component.
- `ff`: A (m × n) matrix where each row contains function values at the corresponding `xx`.
- `x`: A scalar Float64 value at which to extrapolate.

## Returns:
- A vector of length n representing the extrapolated function values at `x`.
"""
function inverse_extrap(xx::Matrix{Float64}, ff::Matrix{Float64}, x::Float64)::Vector{Float64}
    m, n = size(ff)             # m = number of data points, n = number of components
    f = zeros(Float64, n)       # Output vector
    
    for i in 1:m
        term = copy(ff[i, :])   # Start with f_i (vector)
        for j in 1:m
            if j == i
                continue
            end
            term .= term .* ((x .- xx[j, :]) ./ (xx[i, :] .- xx[j, :]))
        end
        f .+= term              # Accumulate to output
    end
    
    return f
end



function equilibrium_solver(input::InverseRunInput)
    println("--- Starting Inverse Equilibrium Processing ---")
    # Extract input parameters

    config = input.config
    rz_in = input.rz_in
    sq_in = input.sq_in
    ro = input.ro
    zo = input.zo
    psio = input.psio

    grid_type = config.control.grid_type
    mpsi = config.control.mpsi
    mtheta = config.control.mtheta
    psilow = config.control.psilow
    psihigh = config.control.psihigh
    newq0 = config.control.newq0

    me = 3

    # Not using anymore
    # interp = false
    # diagnose_rz_in = false
    # diagnose_rzphi = false

    # c-----------------------------------------------------------------------
    # c     allocate and define local arrays.
    # c-----------------------------------------------------------------------
    # sq_in._fs[:, 3] .= sqrt.(sq_in._xs)
    rz_in._xs = sq_in._xs
    rz_in._ys = collect(0:rz_in.my) ./ rz_in.my

    mx = rz_in.mx
    my = rz_in.my

    x = rz_in.fs[:, :, 1] .- ro
    y = rz_in.fs[:, :, 2] .- zo
    r2 = x.^2 .+ y.^2

    twopi = 2 * π

    deta = zeros(Float64, mx+1, my+1)
    for ipsi in 0:mx, itheta in 0:my
        if r2[ipsi+1, itheta+1] == 0.0
            deta[ipsi+1, itheta+1] = 0.0
        else
            deta[ipsi+1, itheta+1] = atan(y[ipsi+1, itheta+1], x[ipsi+1, itheta+1]) / twopi
        end
    end

    # c-----------------------------------------------------------------------
    # c     transform input coordinates from cartesian to polar.
    # c-----------------------------------------------------------------------
    for ipsi in 0:mx
        for itheta in 1:my
            Δ = deta[ipsi+1, itheta+1] - deta[ipsi+1, itheta]
            if Δ > 0.5
                deta[ipsi+1, itheta+1] -= 1
            elseif Δ < -0.5
                deta[ipsi+1, itheta+1] += 1
            end
        end
        for itheta in 0:my
            if r2[ipsi+1, itheta+1] > 0
                deta[ipsi+1, itheta+1] -= rz_in.ys[itheta+1]
            end
        end
    end

    deta[1, :] = inverse_extrap(r2[2:me+1, :], deta[2:me+1, :], 0.0)

    rz_in_fs = zeros(Float64, mx+1, my+1, 3)
    rz_in_fs[:, :, 1] = r2
    rz_in_fs[:, :, 2] = deta

    rz_in_xs = copy(rz_in._xs)
    rz_in_ys = copy(rz_in._ys)

    new_rz_in = Spl.BicubicSpline(rz_in_xs, rz_in_ys, rz_in_fs, bctypex="extrap", bctypey="periodic")

    # c-----------------------------------------------------------------------
    # c     set up radial grid (only "ldp" implemented)
    # c-----------------------------------------------------------------------
    if grid_type == "ldp"
        sq_xs = psilow .+ (psihigh - psilow) .* (sin.(range(0.0, 1.0; length=mpsi+1) .* (π/2))).^2
        sq_fs = zeros(Float64, mpsi+1, 4) 
    else
        error("Only 'ldp' grid_type is implemented for now.")
    end

    local rzphi::Spl.BicubicSpline
    local eqfun::Spl.BicubicSpline

    # c-----------------------------------------------------------------------
    # c     prepare new bicube type for coordinates.
    # c-----------------------------------------------------------------------
    if mtheta == 0
        mtheta = rz_in.my
    end

    # (/"  r2  "," deta "," dphi ","  jac "/)
    rzphi_fs = zeros(Float64, mpsi+1, mtheta+1, 4)
    rzphi_xs = copy(sq_xs)
    rzphi_ys = collect(0:mtheta) ./ mtheta

    # (/"  b0  ","      ","      " /)
    eqfun_fs = zeros(Float64, mpsi+1, mtheta+1, 3)
    eqfun_xs = copy(sq_xs)
    eqfun_ys = collect(0:mtheta) ./ mtheta

    spl_xs = zeros(Float64, mtheta+1)
    spl_fs = zeros(Float64, mtheta+1, 5)


    for ipsi in 0:mpsi
        psifac = rzphi_xs[ipsi+1]
        f_sq_in = Spl.spline_eval!(sq_in, psifac)
        spl_xs .= rzphi_ys
        for itheta in 0:mtheta
            theta = rzphi_ys[itheta+1]
            f_rz_in, fx_rz_in, fy_rz_in = Spl.bicube_deriv1!(new_rz_in, psifac, theta)
            f_sq_in = Spl.spline_eval!(sq_in, psifac)

            if f_rz_in[1] < 0
                error("Invalid extrapolation near axis, rerun with larger value of psilow")
            end

            rfac = sqrt(f_rz_in[1])
            r = ro + rfac * cos(twopi * (theta + f_rz_in[2]))
            jacfac = fx_rz_in[1] * (1 + fy_rz_in[2]) - fy_rz_in[1] * fx_rz_in[2]
            w11 = (1 + fy_rz_in[2]) * twopi^ 2 * rfac / jacfac
            w12 = -fy_rz_in[1] * pi / (rfac * jacfac)
            bp = psio * sqrt(w11*w11 + w12*w12) / r
            bt = f_sq_in[1] / r
            b = sqrt(bp*bp + bt*bt)

            spl_fs[itheta+1, 1] = f_rz_in[1]
            spl_fs[itheta+1, 2] = f_rz_in[2]
            spl_fs[itheta+1, 3] = r * jacfac
            spl_fs[itheta+1, 4] = spl_fs[itheta+1, 3] / (r * r)
            spl_fs[itheta+1, 5] = spl_fs[itheta+1, 3] * bp^config.control.power_bp * b^config.control.power_b / r^config.control.power_r

        end
        # c-----------------------------------------------------------------------
        # c     fit to cubic splines and integrate.
        # c-----------------------------------------------------------------------

        spl = Spl.CubicSpline(spl_xs, spl_fs; bctype="periodic")
        Spl.spline_integrate!(spl)

        spl_xs = spl.fsi[:, 5] ./ spl.fsi[mtheta+1, 5]
        spl_fs[:, 2] .+= rzphi_ys .- spl_xs
        spl_fs[:, 4] = (spl_fs[:, 3] ./ spl.fsi[mtheta+1, 3]) ./ (spl_fs[:, 5] ./ spl.fsi[mtheta+1, 5]) * spl.fsi[mtheta+1, 3] * twopi * pi
        spl_fs[:, 3] = f_sq_in[1] * pi / psio * (spl.fsi[:, 4] - spl.fsi[mtheta+1, 4] .* spl_xs)

        for itheta in 0:mtheta
            theta = rzphi_ys[itheta+1]
            fs = Spl.spline_eval!(spl, theta)
            rzphi_fs[ipsi+1, itheta+1, :] = fs[1:4]
        end

        sq_fs[ipsi+1, 1] = f_sq_in[1] * twopi
        sq_fs[ipsi+1, 2] = f_sq_in[2]
        sq_fs[ipsi+1, 3] = spl.fsi[mtheta+1, 3] * twopi * pi # dV/d(psi)
        sq_fs[ipsi+1, 4] = spl.fsi[mtheta+1, 4] * sq_fs[ipsi+1, 1] / (2 * twopi * psio) # q-profile
    end

    sq = Spl.CubicSpline(sq_xs, sq_fs; bctype="extrap")
    
    f_sq, f1_sq = Spl.spline_eval(sq, sq_xs, 1)
    q0 = f_sq[1, 4] - f1_sq[1,4] * sq.xs[1] 
    if newq0 == -1
        newq0 = -q0
    end

    if newq0 != 0
        f0 = f_sq[1, 2] - f1_sq[1,2] * sq_xs[1]
        f0fac = f0^2 * ((newq0 / q0)^2 - 1)
        q0 = newq0
        for ipsi in 0:mpsi
            ffac = sqrt(1 + f0fac / f_sq[ipsi+1, 1]^2) * sign(newq0)
            sq_fs[ipsi+1, 1] *= ffac
            sq_fs[ipsi+1, 4] *= ffac
            rzphi_fs[ipsi+1, :, 3] *= ffac
        end
        sq = Spl.CubicSpline(sq_xs, sq_fs; bctype="extrap")
    end
    qa = f_sq[mpsi+1, 4] + f1_sq[mpsi+1, 4] * (1 - sq_xs[mpsi+1])
    rzphi = Spl.BicubicSpline(rzphi_xs, rzphi_ys, rzphi_fs; bctypex="extrap", bctypey="periodic")

    for ipsi in 0:mpsi
        f_sq= Spl.spline_eval!(sq, sq_xs[ipsi+1])
        q = f_sq[4]
        for itheta in 0:mtheta
            f_rzphi,fx_rzphi, fy_rzphi = Spl.bicube_deriv1!(rzphi, sq_xs[ipsi+1], rzphi_ys[itheta+1])
            rfac = sqrt(f_rzphi[1])
            eta = twopi * (itheta / mtheta + f_rzphi[2])
            r = ro + rfac * cos(eta)
            jacfac = fx_rzphi[4]

            v = zeros(Float64, 3, 3)
            v[1, 1] = fx_rzphi[1] / (2 * rfac)
            v[1, 2] = fx_rzphi[2] * twopi * rfac
            v[1, 3] = fx_rzphi[3] * r
            v[2, 1] = fy_rzphi[1] / (2 * rfac)
            v[2, 2] = (1 + fy_rzphi[2]) * twopi * rfac
            v[2, 3] = fy_rzphi[3] * r   
            v[3, 3] = twopi * r

            w11 = (1 + fy_rzphi[2]) * twopi^2 * rfac * r / jacfac
            w12 = -fy_rzphi[1] * pi * r / (rfac * jacfac)

            delpsi = sqrt(w11^2 + w12^2)
            eqfun_fs[ipsi+1, itheta+1, 1] = sqrt(((twopi * psio * delpsi)^2 + f_sq[1]^2) / (twopi * r)^2)
            eqfun_fs[ipsi+1, itheta+1, 2] = (sum(v[1, :] .* v[2, :]) + f_sq[4] * v[3, 3] * v[1, 3]) / (jacfac * eqfun_fs[ipsi+1, itheta+1, 1]^2)
            eqfun_fs[ipsi+1, itheta+1, 3] = (v[2, 3] * v[3, 3] + f_sq[4] * v[3, 3]^2) / (jacfac * eqfun_fs[ipsi+1, itheta+1, 1]^2)  
        end
    end
    eqfun = Spl.BicubicSpline(eqfun_xs, eqfun_ys, eqfun_fs; bctypex="extrap", bctypey="periodic")

    return PlasmaEquilibrium(
        input.config,
        EquilibriumParameters(),
        sq,
        rzphi,
        eqfun,
        ro,
        zo,
        psio,
    )
end