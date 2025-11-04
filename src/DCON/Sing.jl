"""
    sing_scan!(intr::DconInternal, ctrl::DconControl, equil::Equilibrium.PlasmaEquilibrium, ffit::FourFitVars, outp::DconOutput)

Scan all singular surfaces and calculate asymptotic vmat and mmat matrices
and Mericer criterion. Performs the same function as `sing_scan` in the Fortran code.
"""
function sing_scan!(intr::DconInternal, ctrl::DconControl, equil::Equilibrium.PlasmaEquilibrium, ffit::FourFitVars, outp::DconOutput)
    if outp.write_dcon_out
        write_output(outp, :dcon_out, "\n Singular Surfaces:")
        write_output(outp, :dcon_out, @sprintf("%3s %11s %11s %11s %11s %11s %11s %11s", "i", "psi", "rho", "q", "q1", "di0", "di", "err"))
    end
    for ising in 1:intr.msing
        sing_vmat!(intr, ctrl, equil, ffit, outp, ising)
    end
    if outp.write_dcon_out
        write_output(outp, :dcon_out, @sprintf("%3s %11s %11s %11s %11s %11s %11s %11s", "i", "psi", "rho", "q", "q1", "di0", "di", "err"))
    end
end

"""
    sing_find!(intr::DconInternal, ctrl::DconControl, equil::Equilibrium.PlasmaEquilibrium)

Locate singular rational q-surfaces (q = m/nn) using a bisection method
between extrema of the q-profile, and store their properties in `intr.sing`.
Performs the same function as `sing_find` in the Fortran code.
"""
function sing_find!(intr::DconInternal, ctrl::DconControl, equil::Equilibrium.PlasmaEquilibrium)

    # Loop over extrema of q, find all rational values in between
    for iex in 2:equil.params.mextrema
        dq = equil.params.qextrema_q[iex] - equil.params.qextrema_q[iex-1]
        m = trunc(Int, ctrl.nn * equil.params.qextrema_q[iex-1])
        if dq > 0
            m += 1
        end
        dm = Int(sign(dq * ctrl.nn))

        # Loop over possible m's in interval
        while (m - ctrl.nn * equil.params.qextrema_q[iex-1]) * (m - ctrl.nn * equil.params.qextrema_q[iex]) <= 0
            it = 0
            psi0 = equil.params.qextrema_psi[iex-1]
            psi1 = equil.params.qextrema_psi[iex]
            psifac = (psi0 + psi1) / 2 # initial guess for bisection

            # Bisection method to find singular surface
            while it < itmax
                it += 1
                psifac = (psi0 + psi1) / 2
                singfac = (m - ctrl.nn * Spl.spline_eval!(equil.sq, psifac)[4]) * dm
                if abs(singfac) <= 1e-8
                    break
                elseif singfac > 0
                    psi0 = psifac
                else
                    psi1 = psifac
                end
            end

            if it == itmax
                error("Bisection did not converge for m = $m")
            else
                push!(intr.sing, SingType(;
                    m=m,
                    psifac=psifac,
                    rho=sqrt(psifac),
                    q=m / ctrl.nn,
                    q1=Spl.spline_deriv1!(equil.sq, psifac)[2][4]
                ))
                intr.msing += 1
            end
            m += dm
        end
    end
end

"""
    sing_lim!(ctrl::DconControl, equil::Equilibrium.PlasmaEquilibrium, intr::DconInternal)

Compute and set integration ψ, q, and q' limits by handling cases where user truncates
before the last singular surface. Performs a similar function to `sing_lim`
in the Fortran code. Main differences include renaming of sas_flag -> set_psilim_via_dmlim,
removing dW edge storage variables since we now store all integration terms in memory, and
simplification of the logic.

The target value `qlim` is first determined from user-specified control parameters
(`ctrl.qhigh` or `ctrl.dmlim`), subject to the constraint that it does not exceed
`equil.params.qmax`. If set_psilim_via_dmlim is true, `qlim` is adjusted to the largest
rational surface such that `nq + dmlim < qmax`, where qmax is the maximum q value in the equilibrium.
If `qlim < qmax`, a Newton iteration is performed to find the corresponding
`psilim` to integrate to.

Note that the Newton iteration will be triggered if either `set_psilim_via_dmlim` is true
or `ctrl.qhigh < equil.params.qmax`. Otherwise, the equilibrium edge values are used.
"""
function sing_lim!(intr::DconInternal, ctrl::DconControl, equil::Equilibrium.PlasmaEquilibrium)

    # Initial guesses based on equilibrium
    intr.qlim = min(equil.params.qmax, ctrl.qhigh) # equilibrium solve only goes up to qmax, so we're capped there
    intr.q1lim = equil.sq.fs1[end, 4]
    intr.psilim = equil.config.control.psihigh

    # Optionally override qlim based on dmlim
    if ctrl.set_psilim_via_dmlim
        # Normalize dmlim ∈ [0,1)
        ctrl.dmlim = mod(ctrl.dmlim, 1.0)
        intr.qlim = (trunc(Int, ctrl.nn * intr.qlim) + ctrl.dmlim) / ctrl.nn

        # Reduce qlim if above qmax
        while intr.qlim > equil.params.qmax
            intr.qlim -= 1.0 / ctrl.nn
        end
    end

    # If set_psilim_via_dmlim decreased qlim or qhigh < qmax, we need to find the precise psilim via newton iteration
    if intr.qlim < equil.params.qmax
        # Find nearest ψ index where q ≈ qlim
        _, jpsi = findmin(abs.(equil.sq.fs[:, 4] .- intr.qlim))
        jpsi = min(jpsi, equil.config.control.mpsi - 1)

        # Shorthand to evaluate q/q1 inside newton iteration
        qval(ψ) = Spl.spline_eval!(equil.sq, ψ)[4]
        q1val(ψ) = Spl.spline_deriv1!(equil.sq, ψ)[2][4]

        intr.psilim = equil.sq.xs[jpsi]
        for _ in 1:itmax
            dpsi = (intr.qlim - qval(intr.psilim)) / q1val(intr.psilim)
            intr.psilim += dpsi
            abs(dpsi) < eps * abs(intr.psilim) && return intr.q1lim = q1val(intr.psilim)
        end
        error("Can't find psilim after $itmax iterations.")
    else
        # Just use the equilibrium values for psilim, qlim, and q1lim
        return
    end
end

"""
    sing_vmat!(intr::DconInternal, ctrl::DconControl, equil::Equilibrium.PlasmaEquilibrium, ffit::FourFitVars, outp::DconOutput, ising::Int)

Calculate asymptotic vmat and mmat matrices and Mercier criterion for
singular surface `ising`. Performs the same function as `sing_vmat` in the Fortran code.
Main differences are 1-indexing for the expansion orders. See equations 41-48 in
the 2016 Glasser DCON paper for the mathematical details.

### Arguments

  - `ising::Int`: Index of the singular surface to process (1 to `intr.msing`)

### TODOs

Check logic on typing of di
"""
function sing_vmat!(intr::DconInternal, ctrl::DconControl, equil::Equilibrium.PlasmaEquilibrium, ffit::FourFitVars, outp::DconOutput, ising::Int)

    # Allocations
    singp = intr.sing[ising]
    singp.vmat = zeros(ComplexF64, intr.mpert, 2 * intr.mpert, 2, 2 * ctrl.sing_order + 1)
    singp.mmat = zeros(ComplexF64, intr.mpert, 2 * intr.mpert, 2, 2 * ctrl.sing_order + 3)
    singp.power = zeros(ComplexF64, 2 * intr.mpert)

    if ising < 1 || ising > intr.msing
        return
    end

    ipert0 = round(Int, ctrl.nn * singp.q, RoundFromZero) - intr.mlow + 1 # resonant perturbation
    if ipert0 <= 0 || intr.mlow > ctrl.nn * singp.q || intr.mhigh < ctrl.nn * singp.q
        singp.di = 0
        return
    end

    # Compute ranges
    singp.r1 = [ipert0]
    singp.r2 = [ipert0, ipert0 + intr.mpert]
    singp.n1 = [i for i in 1:intr.mpert if i != ipert0]
    singp.n2 = vcat(singp.n1, [i + intr.mpert for i in singp.n1])

    psifac = singp.psifac
    q = singp.q
    di0 = Spl.spline_eval!(intr.locstab, singp.psifac)[1] / singp.psifac
    q1 = singp.q1
    rho = singp.rho

    # Compute Mercier criterion and singular power
    sing_mmat!(intr, ctrl, equil, ffit, ising)
    singp.m0mat = transpose(singp.mmat[singp.r1[1], singp.r2, :, 1])
    # TODO: this is a little odd. di is complex since m0 is complex (but the imaginaries are negligible),
    # its then stored in singp where the type is real, and then converted to complex again for alpha and power.
    # There's gotta be a more clear way to do this? But this is the most faithful to the Fortran
    di = singp.m0mat[1, 1] * singp.m0mat[2, 2] - singp.m0mat[2, 1] * singp.m0mat[1, 2]
    singp.di = real(di)
    singp.alpha = sqrt(-complex(singp.di))
    singp.power[ipert0] = -singp.alpha
    singp.power[ipert0+intr.mpert] = singp.alpha

    if outp.write_dcon_out
        write_output(outp, :dcon_out, @sprintf("%3d %11.3e %11.3e %11.3e %11.3e %11.3e %11.3e %11.3e", ising, psifac, rho, q, q1, di0, singp.di, singp.di / di0 - 1))
    end

    # Zeroth-order non-resonant solutions
    singp.vmat .= 0
    for ipert in 1:intr.mpert
        singp.vmat[ipert, ipert, 1, 1] = 1
        singp.vmat[ipert, ipert+intr.mpert, 2, 1] = 1
    end

    # Zeroth-order resonant solutions
    singp.vmat[ipert0, ipert0, 1, 1] = 1
    singp.vmat[ipert0, ipert0+intr.mpert, 1, 1] = 1
    singp.vmat[ipert0, ipert0, 2, 1] = -(singp.m0mat[1, 1] + singp.alpha) / singp.m0mat[1, 2]
    singp.vmat[ipert0, ipert0+intr.mpert, 2, 1] = -(singp.m0mat[1, 1] - singp.alpha) / singp.m0mat[1, 2]
    det =
        conj(singp.vmat[ipert0, ipert0, 1, 1]) * singp.vmat[ipert0, ipert0+intr.mpert, 2, 1] -
        conj(singp.vmat[ipert0, ipert0+intr.mpert, 1, 1]) * singp.vmat[ipert0, ipert0, 2, 1]
    singp.vmat[ipert0, :, :, 1] ./= sqrt(det)

    # Higher order solutions
    for k in 1:2*ctrl.sing_order
        sing_solve!(singp, intr, k)
    end
end

"""
    sing_mmat!(intr::DconInternal, ctrl::DconControl, equil::Equilibrium.PlasmaEquilibrium, ffit::FourFitVars, ising::Int)

Calculate asymptotic mmat matrix for singular surface `ising`.
Performs the same function as `sing_mmat` in the Fortran code.
Main differences are 1-indexing for the expansion orders and
using dense matrices instead of banded. We keep the Fortran
convention of only filling in the lower half of the Hermitian
matrices, and wrap the subsequent multiplications in `Hermitian()`
calls to take advantage of the symmetry. We have tried to be explicit
in which matrices have only their lower triangle stored to avoid confusion.

### Arguments

  - `ising::Int`: Index of the singular surface to process (1 to `intr.msing`)

### TODOs

Check third derivative accuracy in cubic splines or determine if it matters
Better way to unpack the cubic splines
"""
function sing_mmat!(intr::DconInternal, ctrl::DconControl, equil::Equilibrium.PlasmaEquilibrium, ffit::FourFitVars, ising::Int)

    # Initial allocations
    q = @MVector zeros(Float64, 4)
    singfac = zeros(Float64, intr.mpert, 4)
    f_lower_interp = zeros(ComplexF64, intr.mpert, intr.mpert, 4)
    g_interp = zeros(ComplexF64, intr.mpert, intr.mpert, 4)
    k_interp = zeros(ComplexF64, intr.mpert, intr.mpert, 4)
    f_lower = zeros(ComplexF64, intr.mpert, intr.mpert, ctrl.sing_order + 1)
    f0_lower = zeros(ComplexF64, intr.mpert, intr.mpert)
    ff_lower = zeros(ComplexF64, intr.mpert, intr.mpert, ctrl.sing_order + 1)
    g_lower = zeros(ComplexF64, intr.mpert, intr.mpert, ctrl.sing_order + 1)
    k = zeros(ComplexF64, intr.mpert, intr.mpert, ctrl.sing_order + 1)
    v = zeros(ComplexF64, intr.mpert, 2 * intr.mpert, 2)
    x = zeros(ComplexF64, intr.mpert, 2 * intr.mpert, 2, ctrl.sing_order + 1)

    singp = intr.sing[ising]

    # Evaluate cubic splines
    # TODO: third derivative has some error, but only included via sing_fac[ipert0] for sing_order < 3. Tests with solovev ideal indicate little sensitivity
    # TODO: this is an annoying way to have to take apart this tuple of vectors, I think
    # this is a planned fix already (i.e. separating cubic splines)
    q .= getindex.(Spl.spline_deriv3!(equil.sq, singp.psifac), 4)
    f_lower_interp[:, :, 1], f_lower_interp[:, :, 2], f_lower_interp[:, :, 3], f_lower_interp[:, :, 4] = Spl.spline_deriv3!(ffit.fmats_lower, singp.psifac)
    g_interp[:, :, 1], g_interp[:, :, 2], g_interp[:, :, 3], g_interp[:, :, 4] = Spl.spline_deriv3!(ffit.gmats, singp.psifac)
    k_interp[:, :, 1], k_interp[:, :, 2], k_interp[:, :, 3], k_interp[:, :, 4] = Spl.spline_deriv3!(ffit.kmats, singp.psifac)

    # Evaluate singfac and its derivatives
    ipert0 = singp.m - intr.mlow + 1
    mvec = intr.mlow:intr.mhigh
    singfac[:, 1] .= mvec .- ctrl.nn * q[1]
    singfac[:, 2] .= -ctrl.nn * q[2]
    singfac[:, 3] .= -ctrl.nn * q[3]
    singfac[:, 4] .= -ctrl.nn * q[4]
    singfac[ipert0, 1] = -ctrl.nn * q[2]
    singfac[ipert0, 2] = -ctrl.nn * q[3] / 2
    singfac[ipert0, 3] = -ctrl.nn * q[4] / 3
    singfac[ipert0, 4] = 0

    # Compute factored Hermitian F and its derivatives (lower half only)
    for jpert in 1:intr.mpert
        for ipert in jpert:min(intr.mpert, jpert + intr.mband)
            f_lower[ipert, jpert, 1] = singfac[ipert, 1] * f_lower_interp[ipert, jpert, 1]
            if ctrl.sing_order ≥ 1
                f_lower[ipert, jpert, 2] = singfac[ipert, 1] * f_lower_interp[ipert, jpert, 2] +
                                           singfac[ipert, 2] * f_lower_interp[ipert, jpert, 1]
            end
            if ctrl.sing_order ≥ 2
                f_lower[ipert, jpert, 3] =
                    singfac[ipert, 1] * f_lower_interp[ipert, jpert, 3] +
                    2 * singfac[ipert, 2] * f_lower_interp[ipert, jpert, 2] +
                    singfac[ipert, 3] * f_lower_interp[ipert, jpert, 1]
            end
            if ctrl.sing_order ≥ 3
                f_lower[ipert, jpert, 4] =
                    singfac[ipert, 1] * f_lower_interp[ipert, jpert, 4] +
                    3 * singfac[ipert, 2] * f_lower_interp[ipert, jpert, 3] +
                    3 * singfac[ipert, 3] * f_lower_interp[ipert, jpert, 2] +
                    singfac[ipert, 4] * f_lower_interp[ipert, jpert, 1]
            end
            if ctrl.sing_order ≥ 4
                f_lower[ipert, jpert, 5] =
                    4 * singfac[ipert, 2] * f_lower_interp[ipert, jpert, 4] +
                    6 * singfac[ipert, 3] * f_lower_interp[ipert, jpert, 3] +
                    4 * singfac[ipert, 4] * f_lower_interp[ipert, jpert, 2]
            end
            if ctrl.sing_order ≥ 5
                f_lower[ipert, jpert, 6] = 10 * singfac[ipert, 3] * f_lower_interp[ipert, jpert, 4] +
                                           10 * singfac[ipert, 4] * f_lower_interp[ipert, jpert, 3]
            end
            if ctrl.sing_order ≥ 6
                f_lower[ipert, jpert, 7] = 20 * singfac[ipert, 4] * f_lower_interp[ipert, jpert, 4]
            end
        end
    end
    @views f0_lower .= f_lower[:, :, 1]

    # Compute product (lower half only) of Hermitian matrix F
    # When we wrap the matrix multiplications later with Hermitian(ff),
    # Julia will handle filling the upper half via the Hermitian property
    # internally, just like LAPACK does in Fortran
    fac0 = 1
    for n in 0:ctrl.sing_order
        fac1 = 1
        for j in 0:n
            for jpert in 1:intr.mpert
                for ipert in jpert:min(intr.mpert, jpert + intr.mband)
                    for kpert in max(1, ipert - intr.mband):jpert
                        ff_lower[ipert, jpert, n+1] += fac1 * f_lower[ipert, kpert, j+1] * conj(f_lower[jpert, kpert, n-j+1])
                    end
                end
            end
            fac1 *= (n - j) / (j + 1)
        end
        @views ff_lower[:, :, n+1] ./= fac0
        fac0 *= (n + 1)
    end

    # Compute non-Hermitian matrix K
    for jpert in 1:intr.mpert
        for ipert in max(1, jpert - intr.mband):min(intr.mpert, jpert + intr.mband)
            k[ipert, jpert, 1] = singfac[ipert, 1] * k_interp[ipert, jpert, 1]
            if ctrl.sing_order ≥ 1
                k[ipert, jpert, 2] = singfac[ipert, 1] * k_interp[ipert, jpert, 2] +
                                     singfac[ipert, 2] * k_interp[ipert, jpert, 1]
            end
            if ctrl.sing_order ≥ 2
                k[ipert, jpert, 3] =
                    singfac[ipert, 1] * k_interp[ipert, jpert, 3] / 2 +
                    singfac[ipert, 2] * k_interp[ipert, jpert, 2] +
                    singfac[ipert, 3] * k_interp[ipert, jpert, 1] / 2
            end
            if ctrl.sing_order ≥ 3
                k[ipert, jpert, 4] =
                    singfac[ipert, 1] * k_interp[ipert, jpert, 4] / 6 +
                    singfac[ipert, 2] * k_interp[ipert, jpert, 3] / 2 +
                    singfac[ipert, 3] * k_interp[ipert, jpert, 2] / 2 +
                    singfac[ipert, 4] * k_interp[ipert, jpert, 1] / 6
            end
            if ctrl.sing_order ≥ 4
                k[ipert, jpert, 5] =
                    singfac[ipert, 2] * k_interp[ipert, jpert, 4] / 6 +
                    singfac[ipert, 3] * k_interp[ipert, jpert, 3] / 4 +
                    singfac[ipert, 4] * k_interp[ipert, jpert, 2] / 6
            end
            if ctrl.sing_order ≥ 5
                k[ipert, jpert, 6] = singfac[ipert, 3] * k_interp[ipert, jpert, 4] / 12 +
                                     singfac[ipert, 4] * k_interp[ipert, jpert, 3] / 12
            end
            if ctrl.sing_order ≥ 6
                k[ipert, jpert, 7] = singfac[ipert, 4] * k_interp[ipert, jpert, 4] / 36
            end
        end
    end

    # Compute Hermitian matrix G (lower half only)
    for jpert in 1:intr.mpert
        for ipert in jpert:min(intr.mpert, jpert + intr.mband)
            g_lower[ipert, jpert, 1] = g_interp[ipert, jpert, 1]
            if ctrl.sing_order < 1
                continue
            end
            g_lower[ipert, jpert, 2] = g_interp[ipert, jpert, 2]
            if ctrl.sing_order < 2
                continue
            end
            g_lower[ipert, jpert, 3] = g_interp[ipert, jpert, 3] / 2
            if ctrl.sing_order < 3
                continue
            end
            g_lower[ipert, jpert, 4] = g_interp[ipert, jpert, 4] / 6
        end
    end

    # Compute identity
    for ipert in 1:intr.mpert
        v[ipert, ipert, 1] = 1
        v[ipert, ipert+intr.mpert, 2] = 1
    end

    # Compute zeroth-order x1
    for isol in 1:2*intr.mpert
        @views x[:, isol, 1, 1] .= v[:, isol, 2] .- k[:, :, 1] * v[:, isol, 1]
    end
    # f is prefactorized so can just use this calculation to get F⁻¹x
    @views x[:, :, 1, 1] = UpperTriangular(f0_lower') \ (LowerTriangular(f0_lower) \ x[:, :, 1, 1])

    # Compute higher-order x1
    for i in 1:ctrl.sing_order
        for isol in 1:2*intr.mpert
            for j in 1:i
                @views x[:, isol, 1, i+1] .-= Hermitian(ff_lower[:, :, j+1], :L) * x[:, isol, 1, i-j+1]
            end
            @views x[:, isol, 1, i+1] .-= k[:, :, i+1] * v[:, isol, 1]
        end
        @views x[:, :, 1, i+1] = UpperTriangular(f0_lower') \ (LowerTriangular(f0_lower) \ x[:, :, 1, i+1])
    end

    # Compute x2
    for i in 0:ctrl.sing_order
        for isol in 1:2*intr.mpert
            for j in 0:i
                x[:, isol, 2, i+1] .+= k[:, :, j+1]' * x[:, isol, 1, i-j+1]
            end
            x[:, isol, 2, i+1] .+= Hermitian(g_lower[:, :, i+1], :L) * v[:, isol, 1]
        end
    end

    # Principal terms of mmat
    singp.mmat .= 0
    r1 = singp.r1
    r2 = singp.r2
    n1 = singp.n1
    n2 = singp.n2
    j = 0
    for i in 0:ctrl.sing_order
        singp.mmat[r1, r2, :, j+1] .= x[r1, r2, :, i+1]
        singp.mmat[r1, n2, :, j+2] .= x[r1, n2, :, i+1]
        singp.mmat[n1, r2, :, j+2] .= x[n1, r2, :, i+1]
        singp.mmat[n1, n2, :, j+3] .= x[n1, n2, :, i+1]
        j += 2
    end

    # Shearing terms
    singp.mmat[r1, r2[1], 1, 1] .+= 0.5
    singp.mmat[r1, r2[2], 2, 1] .-= 0.5
end

"""
    sing_solve!(singp::SingType, k::Int)

Solves iteratively for the next order in the power series `singp.vmat`.
See equation 47 in the Glass 2016 DCON paper. Identical to the Fortran
`sing_solve` subroutine.

## Arguments

  - `singp::SingType`: The singularity data structure containing all relevant matrices and parameters.
  - `k::Int`: The current order in the power series expansion.
"""
function sing_solve!(singp::SingType, intr::DconInternal, k::Int)
    for l in 1:k
        singp.vmat[:, :, :, k+1] .+= sing_matmul(singp.mmat[:, :, :, l+1], singp.vmat[:, :, :, k-l+1])
    end
    for isol in 1:2*intr.mpert
        a = copy(singp.m0mat)
        a[1, 1] -= k / 2.0 + singp.power[isol]
        a[2, 2] -= k / 2.0 + singp.power[isol]
        det = a[1, 1] * a[2, 2] - a[1, 2] * a[2, 1]
        x = -singp.vmat[singp.r1[1], isol, :, k+1]
        singp.vmat[singp.r1[1], isol, 1, k+1] = (a[2, 2] * x[1] - a[1, 2] * x[2]) / det
        singp.vmat[singp.r1[1], isol, 2, k+1] = (a[1, 1] * x[2] - a[2, 1] * x[1]) / det
        singp.vmat[singp.n1, isol, :, k+1] ./= (singp.power[isol] + k / 2.0)
    end
end

"""
    sing_matmul(a, b) -> c

Matrix multiplication specific to singular matrices.
Identical to the Fortran `sing_matmul` subroutine.

## Arguments

  - `a::Array{ComplexF64,3}`: shape (mpert, 2 * mpert, 2)
  - `b::Array{ComplexF64,3}`: shape (mmpert, 2 * mpert, 2)

## Returns

  - `c::Array{ComplexF64,3}`: shape (mpert, 2 * mpert, 2)
"""
function sing_matmul(a::Array{ComplexF64,3}, b::Array{ComplexF64,3})
    m = size(b, 1)
    n = size(b, 2)

    # consistency check
    if size(a, 2) != 2 * m
        error("Sing_matmul: size(a,2) = $(size(a,2)) != 2*size(b,1) = $(2*m)")
    end

    c = zeros(ComplexF64, size(a, 1), n, 2)

    # main computation
    tmp = zeros(ComplexF64, size(a, 1))
    for i in 1:n
        for j in 1:2
            @views mul!(tmp, a[:, 1:m, j], b[:, i, 1])
            @views c[:, i, j] .+= tmp
            @views mul!(tmp, a[:, m+1:2*m, j], b[:, i, 2])
            @views c[:, i, j] .+= tmp
        end
    end

    return c
end

"""
    sing_get_ua(ctrl::DconControl, intr::DconInternal, odet::OdeState)

Compute the asymptotic series solution for a given singularity.
Fills and returns `ua` with the asymptotic solution vmat
for the specified singular surface and psi value. Performs
the same function as `sing_get_ua` in the Fortran code.
"""
function sing_get_ua(ctrl::DconControl, intr::DconInternal, odet::OdeState)

    singp = intr.sing[odet.ising]
    r1 = singp.r1
    r2 = singp.r2

    # Compute distance from singular surface
    dpsi = odet.psifac - singp.psifac
    sqrtfac = sqrt(complex(dpsi))
    pfac = abs(dpsi)^singp.alpha

    # Compute power series via Horner's method
    ua = copy(singp.vmat[:, :, :, 2*ctrl.sing_order+1])
    for iorder in (2*ctrl.sing_order-1):-1:0
        ua .= ua .* sqrtfac .+ singp.vmat[:, :, :, iorder+1]
    end

    # Restore powers
    ua[r1, :, 1] ./= sqrtfac
    ua[r1, :, 2] .*= sqrtfac
    ua[:, r2[1], :] ./= pfac
    ua[:, r2[2], :] .*= pfac

    # Renormalize
    if odet.psifac < singp.psifac
        ua[:, r2[1], :] .*= abs(ua[r1[1], r2[1], 1]) / ua[r1[1], r2[1], 1]
        ua[:, r2[2], :] .*= abs(ua[r1[1], r2[2], 1]) / ua[r1[1], r2[2], 1]
    end

    return ua
end

"""
    sing_get_ca(ctrl::DconControl, intr::DconInternal, odet::OdeState)

Compute the asymptotic expansion coefficients according to equation
50 in Glasser 2016 DCON paper. Performs the same function as
`sing_get_ca` in the Fortran code.
"""
function sing_get_ca(ctrl::DconControl, intr::DconInternal, odet::OdeState)

    ua = sing_get_ua(ctrl, intr, odet)

    # Build temp1
    temp1 = zeros(ComplexF64, 2 * intr.mpert, 2 * intr.mpert)
    temp1[1:intr.mpert, :] .= ua[:, :, 1]
    temp1[intr.mpert+1:2*intr.mpert, :] .= ua[:, :, 2]

    # Built temp2
    temp2 = zeros(ComplexF64, 2 * intr.mpert, intr.mpert)
    temp2[1:intr.mpert, :] .= odet.u[:, :, 1]
    temp2[intr.mpert+1:2*intr.mpert, :] .= odet.u[:, :, 2]

    # LU factorization and solve
    temp2 .= lu(temp1) \ temp2

    # Build ca
    ca = zeros(ComplexF64, intr.mpert, intr.mpert, 2)
    ca[:, :, 1] .= temp2[1:intr.mpert, :]
    ca[:, :, 2] .= temp2[intr.mpert+1:2*intr.mpert, :]

    return ca
end

"""
    sing_der!(
        du::Array{ComplexF64,3},
        u::Array{ComplexF64,3},
        params::Tuple{DconControl, Equilibrium.PlasmaEquilibrium, FourFitVars, DconInternal, OdeState, DconOutput},
        psieval::Float64
    )

Evaluate the derivative of the Euler-Lagrange equations, i.e. u' in equation 24 of Glasser 2016.
This function performs the same role as `sing_der` in the Fortran code, with main differences
coming from hiding LAPACK operations under the hood via Julia's LinearAlgebra package,
so the code is much more straightforward.

This follows the Julia DifferentialEquations package format for in place updating.

    ode_function!(du, u, p, t)

From DifferentialEquations.jl docs: Defining your ODE function to be in-place updating
can have performance benefits. What this means is that, instead of writing a function
which outputs its solution, you write a function which updates a vector that is designated
to hold the solution. By doing this, DifferentialEquations.jl's solver packages are able
to reduce the amount of array allocations and achieve better performance.

Wherever possible, in-place operations on pre-allocated arrays are used to minimize memory allocations.
All LAPACK operations are handled under the hood by Julia's LinearAlgebra package, so we can obtain a much
more simplistic code with similar performance.

### Arguments

  - `du::Array{ComplexF64,3}`: Pre-allocated array to hold the derivative result, shape (mpert, mpert, 2), updated in-place
  - `u::Array{ComplexF64,3}`: Current state array, shape (mpert, mpert, 2)
  - `params::Tuple{DconControl, Equilibrium.PlasmaEquilibrium, FourFitVars, DconInternal, OdeState, DconOutput}`: Tuple of relevant structs
  - `psieval::Float64`: Current psi value at which to evaluate the derivative

### TODOs

Implement kin_flag functionality
"""
function sing_der!(du::Array{ComplexF64,3}, u::Array{ComplexF64,3},
    params::Tuple{DconControl,Equilibrium.PlasmaEquilibrium,
        FourFitVars,DconInternal,OdeState,DconOutput},
    psieval::Float64)

    # Unpack structs
    ctrl, equil, ffit, intr, odet, _ = params

    # Spline evaluation
    odet.q = Spl.spline_eval!(equil.sq, psieval)[4]
    odet.singfac_vec .= 1.0 ./ (intr.mlow .- ctrl.nn * odet.q .+ (0:intr.mpert-1))
    chi1 = 2π * equil.psio

    # kinetic stuff - skip for now
    if false #(TODO: kin_flag)
        error("kin_flag not implemented yet")
    else
        # Evaluate splines at psieval and reshape avoiding new allocations
        # TODO: is this actually more efficient?
        Spl.spline_eval!(odet.amat, ffit.amats, psieval)
        amat = reshape(odet.amat, intr.mpert, intr.mpert)
        Spl.spline_eval!(odet.bmat, ffit.bmats, psieval)
        bmat = reshape(odet.bmat, intr.mpert, intr.mpert)
        Spl.spline_eval!(odet.cmat, ffit.cmats, psieval)
        cmat = reshape(odet.cmat, intr.mpert, intr.mpert)
        Spl.spline_eval!(odet.fmat_lower, ffit.fmats_lower, psieval)
        fmat_lower = reshape(odet.fmat_lower, intr.mpert, intr.mpert)
        Spl.spline_eval!(odet.kmat, ffit.kmats, psieval)
        kmat = reshape(odet.kmat, intr.mpert, intr.mpert)
        Spl.spline_eval!(odet.gmat, ffit.gmats, psieval)
        gmat = reshape(odet.gmat, intr.mpert, intr.mpert)

        # ldiv!(A,B): Compute A \ B in-place and overwriting B to store the result,
        # where A is a factorization object.
        odet.Afact = cholesky(Hermitian(amat))
        # bmat = A⁻¹ * bmat
        ldiv!(odet.Afact, bmat)
        # cmat = A⁻¹ * cmat
        ldiv!(odet.Afact, cmat)
    end

    odet.tmp .= 0
    u1 = @view(u[:, :, 1])
    u2 = @view(u[:, :, 2])
    # Compute du
    if false #(TODO: kin_flag)
        error("kin_flag not implemented yet")
    else
        # See equation 23 in Glasser 2016 DCON paper for derivation
        # du[1] = - K * u[1] + u[2]

        du[:, :, 1] .= u2 .* odet.singfac_vec
        mul!(odet.tmp, kmat, u1)
        @views du[:, :, 1] .-= odet.tmp

        # du[1] = - F⁻¹ * K * u[1] + F⁻¹ * u[2] (remember F is already stored in factored form)
        @views ldiv!(LowerTriangular(fmat_lower), du[:, :, 1])
        @views ldiv!(UpperTriangular(fmat_lower'), du[:, :, 1])

        # du[2] = G * u[1] + K' * du[1] = G * u[1] - K^† * F⁻¹ * K * u[1] + K^† * F⁻¹ * u[2]
        mul!(odet.tmp, gmat, u1)
        @views du[:, :, 2] .= odet.tmp
        @views mul!(odet.tmp, adjoint(kmat), du[:, :, 1])
        @views du[:, :, 2] .+= odet.tmp
        @views du[:, :, 1] .*= odet.singfac_vec
    end

    # u-derivative used in GPEC
    @views odet.ud[:, :, 1] .= du[:, :, 1]
    @views mul!(odet.tmp, bmat, du[:, :, 1])
    odet.ud[:, :, 2] .= .-odet.tmp
    @views mul!(odet.tmp, cmat, u[:, :, 1])
    @views odet.ud[:, :, 2] .-= odet.tmp
end
