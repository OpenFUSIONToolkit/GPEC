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
    interp = false
    diagnose_rz_in = false
    diagnose_rzphi = false

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
    r2 = x .^ 2 .+ y .^ 2

    twopi = 2 * pi

    deta = zeros(Float64, mx + 1, my + 1)
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

    rz_in_fs = zeros(Float64, mx + 1, my + 1, 3)
    rz_in_fs[:, :, 1] = r2
    rz_in_fs[:, :, 2] = deta

    rz_spline = Spl.BicubicSpline(rz_, rz_in.xs, rz_in.ys; bctypex="extrap", bctypey="periodic")

    # c-----------------------------------------------------------------------
    # c     prepare new spline type for surface quantities.
    # c-----------------------------------------------------------------------

    if grid_type in ["original", "orig"]
        mpsi = size(sq_in._fs, 1) - 1
    end

    # c-----------------------------------------------------------------------
    # c     set up radial grid (only "ldp" implemented)
    # c-----------------------------------------------------------------------
    if grid_type == "ldp"
        xs = psilow .+ (psihigh - psilow) .* (sin.(range(0.0, 1.0; length=mpsi + 1) .* (π / 2))) .^ 2
        fs = zeros(Float64, mpsi + 1, 4)
        sq = Spl.spline_setup(xs, fs; bctype="extrap")
    else
        error("Only 'ldp' grid_type is implemented for now.")
    end

    local rzphi::Spl.BicubicSplineType
    local eqfun::Spl.BicubicSplineType

    # c-----------------------------------------------------------------------
    # c     prepare new bicube type for coordinates.
    # c-----------------------------------------------------------------------
    if mtheta == 0
        mtheta = rz_in.my
    end

    # (/"  r2  "," deta "," dphi ","  jac "/)
    rzphi_fs = zeros(Float64, mpsi + 1, mtheta + 1, 4)
    rzphi_xs = copy(sq_xs)
    rzphi_ys = collect(0:mtheta) ./ mtheta

    # (/"  b0  ","      ","      " /)
    eqfun_fs = zeros(Float64, mpsi + 1, mtheta + 1, 3)
    eqfun_xs = copy(sq_xs)
    eqfun_ys = collect(0:mtheta) ./ mtheta

    rzphi = Spl.bicube_setup(copy(sq.xs), collect(0:mtheta) ./ mtheta, rzphi_fs)
    eqfun = Spl.bicube_setup(copy(sq.xs), collect(0:mtheta) ./ mtheta, eqfun_fs)


    # spl_xs = zeros(Float64,mtheta+1)
    # spl_fs = zeros(Float64,mtheta+1, 5)
    # spl = Spl.spline_setup(spl_xs, spl_fs; bctype="extrap")

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
            w11 = (1 + fy_rz_in[2]) * twopi^2 * rfac / jacfac
            w12 = -fy_rz_in[1] * pi / (rfac * jacfac)
            bp = psio * sqrt(w11 * w11 + w12 * w12) / r
            bt = f_sq_in[1] / r
            b = sqrt(bp * bp + bt * bt)

            #         spl.fs[itheta+1, 1] = f_rz_[1]
            #         spl.fs[itheta+1, 2] = f_rz_[2]
            #         spl.fs[itheta+1, 3] = r*jacfac
            #         spl.fs[itheta+1, 4] = spl.fs[itheta+1, 3]/(r*r)
            #         spl.fs[itheta+1, 5] = spl.fs[itheta+1, 3]*bp^(config.power_bp)*b^(config.power_b)/r^(config.power_r)
        end

        #     spl.xs = spl.integral!()
        #     spl.xs = spl.fsi[:, 5] ./ spl.fs[end, 5]
        #     spl.fs[:, 2] .= spl.fs[:, 2] .+ rzphi.ys .- spl.xs
        #     spl.fs[:, 4] .= (spl.fs[:, 3] ./ spl.fsi[end, 3]) ./ (spl.fs[:, 5] ./ spl.fsi[end, 5]) .* spl.fsi[end, 3] * twopi * pi
        #     spl.fs[:, 3] .= f_sq[1] * pi / psio .* (spl.fsi[:, 4] .- spl.fsi[end, 4] .* spl.xs)
        #     # spl.xs = Spl.spline_setup(spl.xs, spl.fs; bctype="periodic").integral!()

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

    # # sq = Spl.spline_setup(sq.xs, sq.fs; bctype="extrap")
    # f = Spl.spline_eval!(sq, sq.xs[1])
    # _, f1 = Spl.spline_deriv1!(sq, sq.xs[1])
    # q0 = f[4] - f1[4] * sq.xs[1]

    # if newq0 == -1
    #     newq0 = -q0
    # end

end