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
    @info "Starting Inverse Equilibrium Processing"
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

    # Replicate Fortran inverse.f: overwrite deta at the axis (r²=0) by extrapolating from
    # the innermost non-zero surfaces. Only applies when the grid includes the axis (rz_in_xs[1]=0).
    me = 3
    if rz_in_xs[1] == 0.0
        deta[1, :] .= inverse_extrap(r2[2:(me+1), :], deta[2:(me+1), :], 0.0)
    end

    # Ensure periodicity: copy first theta column to last
    # (The computation above may have broken periodicity due to subtracting rz_in_ys values)
    @views r2[:, end] .= r2[:, 1]
    @views deta[:, end] .= deta[:, 1]

    itp_opts2d = (bc=(CubicFit(), PeriodicBC()), extrap=(ExtendExtrap(), WrapExtrap()))

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
    # c     set up radial grid
    # c-----------------------------------------------------------------------
    if grid_type == "log_asymptotic"
        n_core_mid_edge = nothing
        if mpsi == 0 && config.psi_accuracy > 0
            # Estimate A from q profile near psihigh (q is column 3 in sq_in)
            ε = max(1.0 - psihigh, 0.001)
            ψ₁ = clamp(1.0 - 3ε, psilow + 0.01, 0.999)
            ψ₂ = clamp(1.0 - 1.5ε, psilow + 0.01, 0.999)
            A = try
                buf = zeros(size(sq_in.y, 2))
                sq_in(buf, ψ₁)
                q1 = buf[3]
                sq_in(buf, ψ₂)
                q2 = buf[3]
                max(abs(q2 - q1) / log(2), 0.1)
            catch
                @warn "Could not estimate log slope from q profile, using default A=2.0"
                2.0
            end
            n_core_mid_edge = make_optimal_mpsi(psilow, psihigh, A, sq_in; tau=config.psi_accuracy)
            mpsi = sum(n_core_mid_edge)
        elseif mpsi == 0
            mpsi = 128
        end
        sq_xs = if n_core_mid_edge !== nothing
            make_optimal_psi_grid(psilow, psihigh, n_core_mid_edge...)
        else
            log_core = log(0.03 / psilow)
            log_mid = log(0.98 / 0.03)
            log_edge = log((1.0 - 0.98) / (1.0 - psihigh))
            log_total = log_core + log_mid + log_edge
            N_edge = clamp(round(Int, mpsi * log_edge / log_total), 2, mpsi ÷ 2)
            N_core = round(Int, mpsi * log_core / log_total)
            N_mid = mpsi - N_edge - N_core
            make_optimal_psi_grid(psilow, psihigh, N_core, N_mid, N_edge)
        end
    elseif grid_type == "ldp"
        if mpsi == 0
            mpsi = 128
        end
        sq_xs = psilow .+ (psihigh - psilow) .* (sin.(range(0.0, 1.0; length=mpsi+1) .* (π/2))) .^ 2
    else
        error("Unsupported grid_type: $grid_type")
    end
    sq_fs = zeros(Float64, mpsi+1, 4)
    sq = cubic_interp(sq_xs, Series(sq_fs); extrap=ExtendExtrap())

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
    spl_post_buf = Vector{Float64}(undef, 5)   # scratch for post-SFL spline evaluation

    f_sq_in_buf = Vector{Float64}(undef, size(sq_in.y, 2))
    sq_in_hint = Ref(1)
    hint2d = (Ref(1), Ref(1))
    for ipsi in 0:mpsi
        psifac = rzphi_xs[ipsi+1]
        sq_in(f_sq_in_buf, psifac; hint=sq_in_hint)
        spl_xs .= rzphi_ys
        for itheta in 0:mtheta
            theta = rzphi_ys[itheta+1]
            # Evaluate r² and dη interpolants separately
            query_point = (psifac, theta)
            f_rsq = rz_rsq(query_point; hint=hint2d)
            f_deta = rz_deta(query_point; hint=hint2d)
            fx_rsq = rz_rsq(query_point; deriv=DerivOp(1, 0), hint=hint2d)
            fx_deta = rz_deta(query_point; deriv=DerivOp(1, 0), hint=hint2d)
            fy_rsq = rz_rsq(query_point; deriv=DerivOp(0, 1), hint=hint2d)
            fy_deta = rz_deta(query_point; deriv=DerivOp(0, 1), hint=hint2d)

            if f_rsq < 0
                error("Invalid extrapolation near axis, rerun with larger value of psilow")
            end

            rfac = sqrt(f_rsq)
            r = ro + rfac * cos(twopi * (theta + f_deta))
            jacfac = fx_rsq * (1 + fy_deta) - fy_rsq * fx_deta
            w11 = (1 + fy_deta) * twopi ^ 2 * rfac / jacfac
            w12 = -fy_rsq * pi / (rfac * jacfac)
            bp = psio * sqrt(w11*w11 + w12*w12) / r
            bt = f_sq_in_buf[1] / r
            b = sqrt(bp*bp + bt*bt)

            spl_fs[itheta+1, 1] = f_rsq
            spl_fs[itheta+1, 2] = f_deta
            spl_fs[itheta+1, 3] = r * jacfac
            spl_fs[itheta+1, 4] = spl_fs[itheta+1, 3] / (r * r)
            spl_fs[itheta+1, 5] = spl_fs[itheta+1, 3] * bp^config.power_bp * b^config.power_b / (r^config.power_r * max(rfac, eps(Float64))^config.power_rc)

        end
        # c-----------------------------------------------------------------------
        # c     fit to cubic splines and integrate.
        # c-----------------------------------------------------------------------

        # Ensure periodicity: copy first theta row to last
        # (Numerical operations may have broken exact periodicity)
        @views spl_fs[end, :] .= spl_fs[1, :]

        spl = cubic_interp(spl_xs, Series(spl_fs); bc=PeriodicBC())
        spl_fsi = FastInterpolations.cumulative_integrate(spl)

        spl_xs .= spl_fsi[:, 5] ./ spl_fsi[mtheta+1, 5]
        if any(diff(spl_xs) .<= 0)
            @warn "InverseEquilibrium: SFL theta grid non-monotone at ipsi=$ipsi (psifac=$(round(psifac, sigdigits=4))). The SFL weight function (Jacobian × poloidal element) has a near-zero or negative segment; this surface's geometry may be corrupted."
        end
        @views spl_fs[:, 2] .+= rzphi_ys .- spl_xs
        @views spl_fs[:, 4] .= (spl_fs[:, 3] ./ spl_fsi[mtheta+1, 3]) ./ (spl_fs[:, 5] ./ spl_fsi[mtheta+1, 5]) .* (spl_fsi[mtheta+1, 3] * twopi * pi)
        @views spl_fs[:, 3] .= (f_sq_in_buf[1] * pi / psio) .* (spl_fsi[:, 4] .- spl_fsi[mtheta+1, 4] .* spl_xs)

        # Build spline on the post-SFL theta grid (spl_xs) from the modified spl_fs,
        # then evaluate at the uniform SFL theta grid (rzphi_ys). This correctly
        # propagates the SFL coordinate transformation into the rzphi splines.
        # (Using spl.y directly would give pre-transformation values — wrong for eqfun.)
        spl_post = cubic_interp(spl_xs, Series(spl_fs); bc=PeriodicBC())
        hint_post = Ref(1)
        for itheta in 0:mtheta
            spl_post(spl_post_buf, rzphi_ys[itheta+1]; hint=hint_post)
            @views rzphi_fs[ipsi+1, itheta+1, :] .= spl_post_buf[1:4]
        end

        sq_fs[ipsi+1, 1] = f_sq_in_buf[1] * twopi
        sq_fs[ipsi+1, 2] = f_sq_in_buf[2]
        sq_fs[ipsi+1, 3] = spl_fsi[mtheta+1, 3] * twopi * pi # dV/d(psi)
        sq_fs[ipsi+1, 4] = spl_fsi[mtheta+1, 4] * sq_fs[ipsi+1, 1] / (2 * twopi * psio) # q-profile
    end

    sq = cubic_interp(sq_xs, Series(sq_fs); extrap=ExtendExtrap())

    # Access sq nodal values directly (evaluating at own knots returns stored data)
    f_sq = sq.y
    sq_deriv = deriv1(sq)
    f1_sq_lo = sq_deriv(sq_xs[1])
    f1_sq_hi = sq_deriv(sq_xs[end])
    q0 = f_sq[1, 4] - f1_sq_lo[4] * sq_xs[1]
    if newq0 == -1
        newq0 = -q0
    end

    if newq0 != 0
        f0 = f_sq[1, 2] - f1_sq_lo[2] * sq_xs[1]
        f0fac = f0^2 * ((newq0 / q0)^2 - 1)
        q0 = newq0
        for ipsi in 0:mpsi
            ffac = sqrt(1 + f0fac / f_sq[ipsi+1, 1]^2) * sign(newq0)
            sq_fs[ipsi+1, 1] *= ffac
            sq_fs[ipsi+1, 4] *= ffac
            rzphi_fs[ipsi+1, :, 3] *= ffac
        end
        sq = cubic_interp(sq_xs, Series(sq_fs); extrap=ExtendExtrap())
        f_sq = sq.y
        sq_deriv = deriv1(sq)
        f1_sq_hi = sq_deriv(sq_xs[end])
    end
    qa = f_sq[mpsi+1, 4] + f1_sq_hi[4] * (1 - sq_xs[mpsi+1])
    # Create 2D interpolants for geometric quantities (rzphi)
    rzphi_grid2d = (rzphi_xs, theta_range)

    rzphi_rsquared = cubic_interp(rzphi_grid2d, rzphi_fs[:, :, 1]; itp_opts2d...)
    rzphi_offset = cubic_interp(rzphi_grid2d, rzphi_fs[:, :, 2]; itp_opts2d...)
    rzphi_nu = cubic_interp(rzphi_grid2d, rzphi_fs[:, :, 3]; itp_opts2d...)
    rzphi_jac = cubic_interp(rzphi_grid2d, rzphi_fs[:, :, 4]; itp_opts2d...)

    v = zeros(Float64, 3, 3)
    for ipsi in 0:mpsi
        f_sq_vec = @view sq.y[ipsi+1, :]
        for itheta in 0:mtheta
            # Evaluate rzphi interpolants at grid points using nodal_derivs
            f_rzphi = (
                rzphi_rsquared.nodal_derivs.partials[1, ipsi+1, itheta+1],
                rzphi_offset.nodal_derivs.partials[1, ipsi+1, itheta+1],
                rzphi_nu.nodal_derivs.partials[1, ipsi+1, itheta+1],
                rzphi_jac.nodal_derivs.partials[1, ipsi+1, itheta+1]
            )
            fx_rzphi = (
                rzphi_rsquared.nodal_derivs.partials[2, ipsi+1, itheta+1],
                rzphi_offset.nodal_derivs.partials[2, ipsi+1, itheta+1],
                rzphi_nu.nodal_derivs.partials[2, ipsi+1, itheta+1],
                rzphi_jac.nodal_derivs.partials[2, ipsi+1, itheta+1]
            )
            fy_rzphi = (
                rzphi_rsquared.nodal_derivs.partials[3, ipsi+1, itheta+1],
                rzphi_offset.nodal_derivs.partials[3, ipsi+1, itheta+1],
                rzphi_nu.nodal_derivs.partials[3, ipsi+1, itheta+1],
                rzphi_jac.nodal_derivs.partials[3, ipsi+1, itheta+1]
            )
            rfac = sqrt(f_rzphi[1])
            eta = twopi * (itheta / mtheta + f_rzphi[2])
            r = ro + rfac * cos(eta)
            jacfac = f_rzphi[4]

            fill!(v, 0.0)
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
            eqfun_fs[ipsi+1, itheta+1, 1] = sqrt(((twopi * psio * delpsi)^2 + f_sq_vec[1]^2) / (twopi * r)^2)
            eqfun_fs[ipsi+1, itheta+1, 2] = (sum(v[1, :] .* v[2, :]) + f_sq_vec[4] * v[3, 3] * v[1, 3]) / (jacfac * eqfun_fs[ipsi+1, itheta+1, 1]^2)
            eqfun_fs[ipsi+1, itheta+1, 3] = (v[2, 3] * v[3, 3] + f_sq_vec[4] * v[3, 3]^2) / (jacfac * eqfun_fs[ipsi+1, itheta+1, 1]^2)
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

    geometry = compute_geometry_profiles(rzphi_xs, rzphi_ys,
        rzphi_rsquared, rzphi_offset, rzphi_jac, ro)

    return PlasmaEquilibrium(
        input.config,
        EquilibriumParameters(),
        profiles,
        geometry,
        rzphi_xs, rzphi_ys,
        rzphi_rsquared, rzphi_offset, rzphi_nu, rzphi_jac,
        eqfun_B, eqfun_metric1, eqfun_metric2,
        ro,
        zo,
        psio
    )
end
