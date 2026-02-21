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
    rz_in_xs = input.rz_in_xs
    rz_in_ys = input.rz_in_ys
    rz_in_R = input.rz_in_R
    rz_in_Z = input.rz_in_Z
    sq_in = input.sq_in
    ro = input.ro
    zo = input.zo
    psio = input.psio

    grid_type = config.grid_type
    mpsi = config.mpsi
    mtheta = config.mtheta
    psilow = config.psilow
    psihigh = config.psihigh
    newq0 = config.newq0

    me = 3

    # c-----------------------------------------------------------------------
    # c     allocate and define local arrays.
    # c-----------------------------------------------------------------------
    # Access grid dimensions from the interpolants
    mx = length(rz_in_xs) - 1
    my = length(rz_in_ys) - 1

    # Extract R and Z nodal values from interpolants
    R_data = rz_in_R.nodal_derivs.partials[1, :, :]
    Z_data = rz_in_Z.nodal_derivs.partials[1, :, :]

    x = R_data .- ro
    y = Z_data .- zo
    r2 = x .^ 2 .+ y .^ 2

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
                deta[ipsi+1, itheta+1] -= rz_in_ys[itheta+1]
            end
        end
    end

    deta[1, :] = inverse_extrap(r2[2:(me+1), :], deta[2:(me+1), :], 0.0)
    deta[1, :] = inverse_extrap(r2[2:(me+1), :], deta[2:(me+1), :], 0.0)

    # Ensure periodicity: copy first theta column to last
    # (The computation above may have broken periodicity due to subtracting rz_in_ys values)
    r2[:, end] .= r2[:, 1]
    deta[:, end] .= deta[:, 1]

    itp_opts2d = (search=LinearBinary(), bc=(CubicFit(), PeriodicBC()), extrap=(ExtendExtrap(), WrapExtrap()))

    # Create 2D interpolants for r² and dη
    rz_rsq = cubic_interp((rz_in_xs, rz_in_ys), r2; itp_opts2d...)
    rz_deta = cubic_interp((rz_in_xs, rz_in_ys), deta; itp_opts2d...)
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
        sq_xs = psilow .+ (psihigh - psilow) .* (sin.(range(0.0, 1.0; length=mpsi+1) .* (π/2))) .^ 2
        sq_fs = zeros(Float64, mpsi+1, 4)
        sq = cubic_interp(sq_xs, sq_fs; bc=CubicFit(), extrap=ExtendExtrap())
    else
        error("Only 'ldp' grid_type is implemented for now.")
    end

    # c-----------------------------------------------------------------------
    # c     prepare new bicube type for coordinates.
    # c-----------------------------------------------------------------------
    if mtheta == 0
        mtheta = my
    end

    theta_range = range(0.0, 1.0; length=mtheta+1)

    # (/"  r2  "," deta "," dphi ","  jac "/)
    rzphi_fs = zeros(Float64, mpsi+1, mtheta+1, 4)
    rzphi_xs = copy(sq_xs)
    # rzphi_ys is the materialized Vector stored in PlasmaEquilibrium for indexing/diagnostics.
    # The Range form (theta_range) is used for the interpolant: it skips index search during
    # evaluation (O(1) vs binary search) and may differ at machine-precision level from Vector.
    rzphi_ys = collect(theta_range)

    # (/"  b0  ","      ","      " /)
    eqfun_fs = zeros(Float64, mpsi+1, mtheta+1, 3)
    eqfun_xs = copy(sq_xs)

    # Preallocate arrays for periodic spline fitting in the loop
    spl_xs = zeros(Float64, mtheta+1)
    spl_fs = zeros(Float64, mtheta+1, 5)

    hint2d = (Ref(1), Ref(1))
    for ipsi in 0:mpsi
        psifac = rzphi_xs[ipsi+1]
        f_sq_in = sq_in(psifac)
        spl_xs .= rzphi_ys
        for itheta in 0:mtheta
            theta = rzphi_ys[itheta+1]
            # Evaluate r² and dη interpolants separately
            query_point = (psifac, theta)
            f_rsq = rz_rsq(query_point; hint=hint2d)
            f_deta = rz_deta(query_point; hint=hint2d)
            fx_rsq = rz_rsq(query_point; deriv=Val((1, 0)), hint=hint2d)
            fx_deta = rz_deta(query_point; deriv=Val((1, 0)), hint=hint2d)
            fy_rsq = rz_rsq(query_point; deriv=Val((0, 1)), hint=hint2d)
            fy_deta = rz_deta(query_point; deriv=Val((0, 1)), hint=hint2d)

            f_sq_in = sq_in(psifac)

            if f_rsq < 0
                error("Invalid extrapolation near axis, rerun with larger value of psilow")
            end

            rfac = sqrt(f_rsq)
            r = ro + rfac * cos(twopi * (theta + f_deta))
            jacfac = fx_rsq * (1 + fy_deta) - fy_rsq * fx_deta
            w11 = (1 + fy_deta) * twopi ^ 2 * rfac / jacfac
            w12 = -fy_rsq * pi / (rfac * jacfac)
            bp = psio * sqrt(w11*w11 + w12*w12) / r
            bt = f_sq_in[1] / r
            b = sqrt(bp*bp + bt*bt)

            spl_fs[itheta+1, 1] = f_rsq
            spl_fs[itheta+1, 2] = f_deta
            spl_fs[itheta+1, 3] = r * jacfac
            spl_fs[itheta+1, 4] = spl_fs[itheta+1, 3] / (r * r)
            spl_fs[itheta+1, 5] = spl_fs[itheta+1, 3] * bp^config.power_bp * b^config.power_b / r^config.power_r

        end
        # c-----------------------------------------------------------------------
        # c     fit to cubic splines and integrate.
        # c-----------------------------------------------------------------------

        # Ensure periodicity: copy first theta row to last
        # (Numerical operations may have broken exact periodicity)
        spl_fs[end, :] .= spl_fs[1, :]

        spl = cubic_interp(spl_xs, spl_fs; bc=PeriodicBC())
        spl_fsi = FastInterpolations.cumulative_integrate(spl)

        spl_xs = spl_fsi[:, 5] ./ spl_fsi[mtheta+1, 5]
        spl_fs[:, 2] .+= rzphi_ys .- spl_xs
        spl_fs[:, 4] = (spl_fs[:, 3] ./ spl_fsi[mtheta+1, 3]) ./ (spl_fs[:, 5] ./ spl_fsi[mtheta+1, 5]) * spl_fsi[mtheta+1, 3] * twopi * pi
        spl_fs[:, 3] = f_sq_in[1] * pi / psio * (spl_fsi[:, 4] - spl_fsi[mtheta+1, 4] .* spl_xs)

        for itheta in 0:mtheta
            theta = rzphi_ys[itheta+1]
            fs = spl(theta)
            rzphi_fs[ipsi+1, itheta+1, :] = fs[1:4]
        end

        sq_fs[ipsi+1, 1] = f_sq_in[1] * twopi
        sq_fs[ipsi+1, 2] = f_sq_in[2]
        sq_fs[ipsi+1, 3] = spl_fsi[mtheta+1, 3] * twopi * pi # dV/d(psi)
        sq_fs[ipsi+1, 4] = spl_fsi[mtheta+1, 4] * sq_fs[ipsi+1, 1] / (2 * twopi * psio) # q-profile
    end

    sq = cubic_interp(sq_xs, sq_fs; bc=CubicFit(), extrap=ExtendExtrap())

    # Evaluate sq and its derivative at all grid points
    f_sq = zeros(Float64, mpsi+1, 4)
    f1_sq = zeros(Float64, mpsi+1, 4)
    sq_deriv = deriv1(sq)
    for i in 1:(mpsi+1)
        f_sq[i, :] = sq(sq_xs[i])
        f1_sq[i, :] = sq_deriv(sq_xs[i])
    end
    q0 = f_sq[1, 4] - f1_sq[1, 4] * sq_xs[1]
    if newq0 == -1
        newq0 = -q0
    end

    if newq0 != 0
        f0 = f_sq[1, 2] - f1_sq[1, 2] * sq_xs[1]
        f0fac = f0^2 * ((newq0 / q0)^2 - 1)
        q0 = newq0
        for ipsi in 0:mpsi
            ffac = sqrt(1 + f0fac / f_sq[ipsi+1, 1]^2) * sign(newq0)
            sq_fs[ipsi+1, 1] *= ffac
            sq_fs[ipsi+1, 4] *= ffac
            rzphi_fs[ipsi+1, :, 3] *= ffac
        end
        sq = cubic_interp(sq_xs, sq_fs; bc=CubicFit(), extrap=ExtendExtrap())
    end
    qa = f_sq[mpsi+1, 4] + f1_sq[mpsi+1, 4] * (1 - sq_xs[mpsi+1])
    # Create 2D interpolants for geometric quantities (rzphi)
    rzphi_grid2d = (rzphi_xs, theta_range)

    rzphi_rsquared = cubic_interp(rzphi_grid2d, rzphi_fs[:, :, 1]; itp_opts2d...)
    rzphi_offset = cubic_interp(rzphi_grid2d, rzphi_fs[:, :, 2]; itp_opts2d...)
    rzphi_nu = cubic_interp(rzphi_grid2d, rzphi_fs[:, :, 3]; itp_opts2d...)
    rzphi_jac = cubic_interp(rzphi_grid2d, rzphi_fs[:, :, 4]; itp_opts2d...)

    for ipsi in 0:mpsi
        f_sq = sq(sq_xs[ipsi+1])
        q = f_sq[4]
        for itheta in 0:mtheta
            # Evaluate rzphi interpolants at grid points using nodal_derivs
            f_rzphi = SVector{4}(
                rzphi_rsquared.nodal_derivs.partials[1, ipsi+1, itheta+1],
                rzphi_offset.nodal_derivs.partials[1, ipsi+1, itheta+1],
                rzphi_nu.nodal_derivs.partials[1, ipsi+1, itheta+1],
                rzphi_jac.nodal_derivs.partials[1, ipsi+1, itheta+1]
            )
            fx_rzphi = SVector{4}(
                rzphi_rsquared.nodal_derivs.partials[2, ipsi+1, itheta+1],
                rzphi_offset.nodal_derivs.partials[2, ipsi+1, itheta+1],
                rzphi_nu.nodal_derivs.partials[2, ipsi+1, itheta+1],
                rzphi_jac.nodal_derivs.partials[2, ipsi+1, itheta+1]
            )
            fy_rzphi = SVector{4}(
                rzphi_rsquared.nodal_derivs.partials[3, ipsi+1, itheta+1],
                rzphi_offset.nodal_derivs.partials[3, ipsi+1, itheta+1],
                rzphi_nu.nodal_derivs.partials[3, ipsi+1, itheta+1],
                rzphi_jac.nodal_derivs.partials[3, ipsi+1, itheta+1]
            )
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
    # Create 2D interpolants for physics quantities (eqfun)
    eqfun_grid2d = (eqfun_xs, theta_range)
    eqfun_B = cubic_interp(eqfun_grid2d, eqfun_fs[:, :, 1]; itp_opts2d...)
    eqfun_metric1 = cubic_interp(eqfun_grid2d, eqfun_fs[:, :, 2]; itp_opts2d...)
    eqfun_metric2 = cubic_interp(eqfun_grid2d, eqfun_fs[:, :, 3]; itp_opts2d...)

    # Create ProfileSplines from sq interpolant
    # sq_fs columns: [F*2π, P, dV/dψ, q]
    profiles = ProfileSplines(
        sq_xs,
        sq_fs[:, 1],  # F values (already includes 2π factor)
        sq_fs[:, 2],  # P values
        sq_fs[:, 3],  # dV/dψ values
        sq_fs[:, 4]   # q values
    )

    return PlasmaEquilibrium(
        input.config,
        EquilibriumParameters(),
        profiles,
        rzphi_xs, rzphi_ys,
        rzphi_rsquared, rzphi_offset, rzphi_nu, rzphi_jac,
        eqfun_B, eqfun_metric1, eqfun_metric2,
        ro,
        zo,
        psio
    )
end
