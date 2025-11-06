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
    sing_find!(intr::DconInternal, equil::Equilibrium.PlasmaEquilibrium)

Locate singular rational q-surfaces (q = m/nn) using a bisection method
between extrema of the q-profile, and store their properties in `intr.sing`.
Performs the same function as `sing_find` in the Fortran code.
"""
function sing_find!(intr::DconInternal, equil::Equilibrium.PlasmaEquilibrium)

    # loop over all toroidal mode numbers
    for n in intr.nlow:intr.nhigh
        # Loop over extrema of q, find all rational values in between
        for iex in 2:equil.params.mextrema
            dq = equil.params.qextrema_q[iex] - equil.params.qextrema_q[iex-1]
            m = trunc(Int, n * equil.params.qextrema_q[iex-1])
            if dq > 0
                m += 1
            end
            dm = Int(sign(dq * n))

            # Loop over possible m's in interval
            while (m - n * equil.params.qextrema_q[iex-1]) * (m - n * equil.params.qextrema_q[iex]) <= 0
                it = 0
                psi0 = equil.params.qextrema_psi[iex-1]
                psi1 = equil.params.qextrema_psi[iex]
                psifac = (psi0 + psi1) / 2 # initial guess for bisection

                # Bisection method to find singular surface
                converged = false
                for _ in 1:itmax
                    psifac = (psi0 + psi1) / 2
                    singfac = (m - n * Spl.spline_eval!(equil.sq, psifac)[4]) * dm
                    abs(singfac) < 1e-8 && (converged = true; break)
                    singfac > 0 ? (psi0 = psifac) : (psi1 = psifac)
                end

                if !converged
                    error("Bisection did not converge for m = $m after $itmax iterations.")
                elseif any(s -> isapprox(s.q, m / n; atol=1e-8), intr.sing)
                    # Rational surface with multiplicity > 1, add this m,n to the resonant mode numbers
                    # Technically only need one, but simplifies some later code and cheap to store both
                    push!(intr.sing[findfirst(s -> isapprox(s.q, m / n; atol=1e-8), intr.sing)].m, m)
                    push!(intr.sing[findfirst(s -> isapprox(s.q, m / n; atol=1e-8), intr.sing)].n, n)
                else
                    push!(intr.sing, SingType(;
                        m=[m],
                        n=[n],
                        psifac=psifac,
                        rho=sqrt(psifac),
                        q=m / n,
                        q1=Spl.spline_deriv1!(equil.sq, psifac)[2][4]
                    ))
                    intr.msing += 1
                end
                m += dm
            end
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
        if ctrl.nn_low != ctrl.nn_high
            @warn "When setting psilim via dmlim, only single n is currently supported. Setting nn_low = nn_high = $(ctrl.nn)."
            ctrl.nn_low = ctrl.nn_high = ctrl.nn
        end
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
        converged = false
        for _ in 1:itmax
            dpsi = (intr.qlim - qval(intr.psilim)) / q1val(intr.psilim)
            intr.psilim += dpsi
            abs(dpsi) < eps * abs(intr.psilim) && (converged = true; break)
        end

        if converged
            intr.q1lim = q1val(intr.psilim)
        else
            error("Can't find psilim after $itmax iterations.")
        end
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
    singp.vmat = zeros(ComplexF64, intr.numpert_total, 2 * intr.numpert_total, 2, 2 * ctrl.sing_order + 1)
    singp.mmat = zeros(ComplexF64, intr.numpert_total, 2 * intr.numpert_total, 2, 2 * ctrl.sing_order + 3)
    singp.power = zeros(ComplexF64, 2 * intr.numpert_total)

    # Compute the resonant (r) and nonresonant (n) indices of the shearing transformation matrix R
    # 1 indexes along the N*M dimension, and 2 along the 2*N*M dimension
    # In 2D, see eq. 41 of 2016 Glasser DCON paper
    ipert_res = 1 .+ singp.m .- intr.mlow .+ (singp.n .- intr.nlow) .* intr.mpert
    singp.r1 = ipert_res
    singp.r2 = vec([ipert_res[i] + j * intr.numpert_total for j in 0:1, i in eachindex(ipert_res)])
    singp.n1 = [i for i in 1:intr.numpert_total if !(i in ipert_res)]
    singp.n2 = vcat(singp.n1, [i + intr.numpert_total for i in singp.n1])

    psifac = singp.psifac
    q = singp.q
    di0 = Spl.spline_eval!(intr.locstab, singp.psifac)[1] / singp.psifac
    q1 = singp.q1
    rho = singp.rho

    # Compute Mercier criterion and singular power
    sing_mmat!(intr, ctrl, equil, ffit, ising)
    # TODO: I am not sure if this will generalize to 3D, but I think it works for 2D
    if length(singp.r1) == 1
        singp.m0mat = transpose(singp.mmat[singp.r1[1], singp.r2, :, 1])
    else
        singp.m0mat = vcat([transpose(singp.mmat[singp.r1[i], singp.r2, :, 1]) for i in eachindex(singp.r1)]...)
    end

    singp.alpha = eigen(singp.m0mat).values[length(singp.r1)+1:end] # take the M largest eigenvalues
    # Maybe? di isn't super well defined for multiplicity > 1
    singp.di = real(singp.alpha[1]^2)

    singp.power[ipert_res] .= -singp.alpha
    singp.power[ipert_res .+ intr.numpert_total] .= singp.alpha

    if outp.write_dcon_out
        write_output(outp, :dcon_out, @sprintf("%3d %11.3e %11.3e %11.3e %11.3e %11.3e %11.3e %11.3e", ising, psifac, rho, q, q1, di0, singp.di, singp.di / di0 - 1))
    end

    # Zeroth-order non-resonant solutions
    singp.vmat .= 0
    for ipert in 1:intr.numpert_total
        singp.vmat[ipert, ipert, 1, 1] = 1
        singp.vmat[ipert, ipert+intr.numpert_total, 2, 1] = 1
    end
    
    # Zeroth-order resonant solutions - solve (M₀ - αI)v₀ = 0
    # TODO: no idea if this is the correct way of doing this
    M = length(singp.r1) # multiplicity
    for i in 1:M # go block by block and do the same eigenvector solve/normalization?
        m0mat = singp.m0mat[(2*(i-1)+1):(2*i), (2*(i-1)+1):(2*i)]
        rpert_1 = singp.r1[i]
        rpert_2 = rpert_1 + intr.numpert_total
        alpha = singp.alpha[i]
        singp.vmat[rpert_1, rpert_1, 1, 1] = 1
        singp.vmat[rpert_1, rpert_2, 1, 1] = 1
        singp.vmat[rpert_1, rpert_1, 2, 1] = -(m0mat[1, 1] + alpha) / m0mat[1, 2]
        singp.vmat[rpert_1, rpert_2, 2, 1] = -(m0mat[1, 1] - alpha) / m0mat[1, 2]
        det =
            conj(singp.vmat[rpert_1, rpert_1, 1, 1]) * singp.vmat[rpert_1, rpert_2, 2, 1] -
            conj(singp.vmat[rpert_1, rpert_2, 1, 1]) * singp.vmat[rpert_1, rpert_1, 2, 1]
        singp.vmat[rpert_1, :, :, 1] ./= sqrt(det)
    end

    # Higher order solutions - need to solve iteratively
    for k in 1:2*ctrl.sing_order
        sing_solve!(singp, intr, k)
    end
end

"""
    sing_mmat!(intr::DconInternal, ctrl::DconControl, equil::Equilibrium.PlasmaEquilibrium, ffit::FourFitVars, ising::Int)

Calculate asymptotic mmat matrix for singular surface `ising`. Performs the same
function as `sing_mmat` in the Fortran code. Main differences are 1-indexing for
the expansion orders and using dense matrices instead of banded. We keep the Fortran
convention of only filling in the lower half of the Hermitian matrices, and wrap the
subsequent multiplications in `Hermitian()` calls to take advantage of the symmetry.
We have tried to be explicit in which matrices have only their lower triangle stored
to avoid confusion.

More specifically, in this function we construct the Taylor series M_k of the matrix M
(eq. 43-44 in Glasser 2016). This involves: evaluating the cubic splines and their
derivatives at the singular surface, constructing the Taylor series coefficients for each
composite matrix (simple for G, more complicated for F and K since they are stored as
their nonsingular forms and need to be multiplied by Q, which is itself a Taylor series
so the coefficients get complicated), solving x = L v for each order in the Taylor series,
and then reconstructing mmat from x.

### Arguments

  - `ising::Int`: Index of the singular surface to process (1 to `intr.msing`)

### TODOs

Check third derivative accuracy in cubic splines or determine if it matters
Better way to unpack the cubic splines
Rename variables to be more intuitive? I don't like ff - maybe f and f_fact instead of f_lower
Add a spline for F directly instead of the lower triangular factorization to avoid complexity?
"""
function sing_mmat!(intr::DconInternal, ctrl::DconControl, equil::Equilibrium.PlasmaEquilibrium, ffit::FourFitVars, ising::Int)

    # Initial allocations
    singp = intr.sing[ising]
    q = @MVector zeros(Float64, 4)
    singfac = zeros(Float64, intr.numpert_total, 4)
    f_lower_interp = zeros(ComplexF64, intr.numpert_total, intr.numpert_total, 4)
    g_interp = zeros(ComplexF64, intr.numpert_total, intr.numpert_total, 4)
    k_interp = zeros(ComplexF64, intr.numpert_total, intr.numpert_total, 4)
    f_lower = zeros(ComplexF64, intr.numpert_total, intr.numpert_total, ctrl.sing_order + 1)
    f0_lower = zeros(ComplexF64, intr.numpert_total, intr.numpert_total)
    ff_lower = zeros(ComplexF64, intr.numpert_total, intr.numpert_total, ctrl.sing_order + 1)
    g_lower = zeros(ComplexF64, intr.numpert_total, intr.numpert_total, ctrl.sing_order + 1)
    k = zeros(ComplexF64, intr.numpert_total, intr.numpert_total, ctrl.sing_order + 1)
    v = zeros(ComplexF64, intr.numpert_total, 2 * intr.numpert_total, 2)
    x = zeros(ComplexF64, intr.numpert_total, 2 * intr.numpert_total, 2, ctrl.sing_order + 1)

    # Evaluate cubic splines
    # TODO: third derivative has some error, but only included via sing_fac[ipert_res] for sing_order < 3. Tests with solovev ideal indicate little sensitivity
    # TODO: this is an annoying way to have to take apart this tuple of vectors, I think
    # this is a planned fix already (i.e. separating cubic splines)
    q .= getindex.(Spl.spline_deriv3!(equil.sq, singp.psifac), 4)
    f_lower_interp[:, :, 1], f_lower_interp[:, :, 2], f_lower_interp[:, :, 3], f_lower_interp[:, :, 4] = Spl.spline_deriv3!(ffit.fmats_lower, singp.psifac)
    g_interp[:, :, 1], g_interp[:, :, 2], g_interp[:, :, 3], g_interp[:, :, 4] = Spl.spline_deriv3!(ffit.gmats, singp.psifac)
    k_interp[:, :, 1], k_interp[:, :, 2], k_interp[:, :, 3], k_interp[:, :, 4] = Spl.spline_deriv3!(ffit.kmats, singp.psifac)

    # Evaluate Taylor series coefficients for Q = mᵢ - nᵢq(ψ) = [mᵢ - nᵢq, -nᵢq', -nᵢq'', -nᵢq''']
    singfac[:, 1] .= vec((intr.mlow:intr.mhigh) .- q[1] .* (intr.nlow:intr.nhigh)')
    for i in 2:4
        singfac[:, i] .= repeat(-(intr.nlow:intr.nhigh) .* q[i], inner=intr.mpert)
    end
    # For resonant modes mᵢ - nᵢq(ψ) = [-nᵢq', -nᵢq'', -nᵢq'''] - shift up terms by 1 index
    # Add scaling to account for hardcoding coefficients in computations below
    for (mres, nres) in zip(singp.m, singp.n)
        ipert_res = 1 + mres - intr.mlow + (nres - intr.nlow) * intr.mpert
        singfac[ipert_res, 1:4] .= (-nres * q[2], -nres * q[3] / 2, -nres * q[4] / 3, 0)
    end

    # This section becomes tricky because we need to reform F = QL̄L̄ᴴQᴴ
    # TODO: this section can absolutely be simplified using some intuition on the coefficients.
    # Possibly could just store an unfactorized F spline? Then logic would just look like K but
    # with an extra singfac/altered coefficients since it would just be F = QF̄Q
    # For now, leaving overly detailed comments to remind
    # First, compute Taylor series coefficients of QL̄ (but without scaling by 1/n!), so we get binomial coefficients leftover
    # f_lower = QL̄ = [QL̄, QL̄' + Q' L̄, 1/2 (QL̄'' + 2Q' L̄' + QQ'' L̄), 1/6 (QL̄''' + 3Q' L̄'' + 3Q'' L̄' + Q'''L̄), ...] (but without 1/2, 1/6, etc)
    for ipert_n in 1:intr.npert
        for jpert_m in 1:intr.mpert
            for ipert_m in jpert_m:min(intr.mpert, jpert_m + intr.mband)
                ipert = ipert_m + (ipert_n - 1) * intr.mpert
                jpert = jpert_m + (ipert_n - 1) * intr.mpert
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
    end
    @views f0_lower .= f_lower[:, :, 1]

    # Compute Taylor series coefficients of F = QL̄L̄ᴴQᴴ (lower half only due to indexing) from QL̄ computed above
    # Here, we build in the Taylor series coefficients iteratively (fac1 = (n choose j), fac0 = 1/n!)
    # so the final coefficient is 1/(n-j)!j! as desired (and hardcoded in the K computation below)
    # When we wrap the matrix multiplications later with Hermitian(ff),
    # Julia will handle filling the upper half via the Hermitian property
    # internally, just like LAPACK does in Fortran
    fac0 = 1
    for n in 0:ctrl.sing_order
        fac1 = 1
        for j in 0:n
            for ipert_n in 1:intr.npert
                for jpert_m in 1:intr.mpert
                    for ipert_m in jpert_m:min(intr.mpert, jpert_m + intr.mband)
                        for kpert_m in max(1, ipert_m - intr.mband):jpert_m
                            ipert = ipert_m + (ipert_n - 1) * intr.mpert
                            jpert = jpert_m + (ipert_n - 1) * intr.mpert
                            kpert = kpert_m + (ipert_n - 1) * intr.mpert
                            ff_lower[ipert, jpert, n+1] += fac1 * f_lower[ipert, kpert, j+1] * conj(f_lower[jpert, kpert, n-j+1])
                        end
                    end
                end
            end
            fac1 *= (n - j) / (j + 1)
        end
        @views ff_lower[:, :, n+1] ./= fac0
        fac0 *= (n + 1)
    end

    # Compute non-Hermitian matrix K = QK̄ Taylor series coefficients
    # K = [QK̄, QK̄' + Q'K̄, QK̄''/2 + Q'K̄' + Q̄''K̄/2, ...]
    for ipert_n in 1:intr.npert
        for jpert_m in 1:intr.mpert
            for ipert_m in max(1, jpert_m - intr.mband):min(intr.mpert, jpert_m + intr.mband)
                ipert = ipert_m + (ipert_n - 1) * intr.mpert
                jpert = jpert_m + (ipert_n - 1) * intr.mpert
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
    end

    # Compute Hermitian matrix G (lower half only) Taylor series coefficients
    # G = [G, G', G''/2, G'''/6]
    for ipert_n in 1:intr.npert
        for jpert_m in 1:intr.mpert
            for ipert_m in jpert_m:min(intr.mpert, jpert_m + intr.mband)
                ipert = ipert_m + (ipert_n - 1) * intr.mpert
                jpert = jpert_m + (ipert_n - 1) * intr.mpert
                g_lower[ipert, jpert, 1] = g_interp[ipert, jpert, 1]
                if ctrl.sing_order ≥ 1
                    g_lower[ipert, jpert, 2] = g_interp[ipert, jpert, 2]
                end
                if ctrl.sing_order ≥ 2
                    g_lower[ipert, jpert, 3] = g_interp[ipert, jpert, 3] / 2
                end
                if ctrl.sing_order ≥ 3
                    g_lower[ipert, jpert, 4] = g_interp[ipert, jpert, 4] / 6
                end
            end
        end
    end

    # Start with the identity matrix, (which can be used to index the projection onto resonant/nonresonant modes?)
    for ipert in 1:intr.numpert_total
        v[ipert, ipert, 1] = 1
        v[ipert, ipert+intr.numpert_total, 2] = 1
    end

    # Solve the Taylor expansion according to F * x¹ = v² - K v¹ at each order
    # 0ᵗʰ order: x¹₀ = F⁻¹(v² - K v¹)
    for isol in 1:2*intr.numpert_total
        @views x[:, isol, 1, 1] .= v[:, isol, 2] .- k[:, :, 1] * v[:, isol, 1]
    end
    @views x[:, :, 1, 1] = UpperTriangular(f0_lower') \ (LowerTriangular(f0_lower) \ x[:, :, 1, 1])
    # Higher-order: ∑Fⱼx¹ₙ₋ⱼ = -Kₙv¹ → x¹ₙ = F₀⁻¹(-∑Fⱼxₙ₋ⱼ - Kₙv¹)
    for i in 1:ctrl.sing_order
        for isol in 1:2*intr.numpert_total
            for j in 1:i
                @views x[:, isol, 1, i+1] .-= Hermitian(ff_lower[:, :, j+1], :L) * x[:, isol, 1, i-j+1]
            end
            @views x[:, isol, 1, i+1] .-= k[:, :, i+1] * v[:, isol, 1]
        end
        @views x[:, :, 1, i+1] = UpperTriangular(f0_lower') \ (LowerTriangular(f0_lower) \ x[:, :, 1, i+1])
    end

    # Solve x²ₙ = (G - K^†F⁻¹K)v¹ + K^†F⁻¹v² = Gₙv¹ - ∑Kⱼ^† x¹ₙ₋ⱼ at each order
    for i in 0:ctrl.sing_order
        for isol in 1:2*intr.numpert_total
            for j in 0:i
                x[:, isol, 2, i+1] .+= adjoint(k[:, :, j+1]) * x[:, isol, 1, i-j+1]
            end
            x[:, isol, 2, i+1] .+= Hermitian(g_lower[:, :, i+1], :L) * v[:, isol, 1]
        end
    end

    # Principal terms of mmat (what is going on here?)
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

    # Apply the shearing transformation matrix R for each set of resonant indices
    for i in eachindex(r1)
        singp.mmat[r1[i], r2[2 * i - 1], 1, 1] += 0.5
        singp.mmat[r1[i], r2[2 * i], 2, 1] -= 0.5
    end
end

"""
    sing_solve!(singp::SingType, k::Int)

Solves iteratively for the next order in the power series `singp.vmat`.
See equation 47 in the Glass 2016 DCON paper. Identical to the Fortran
`sing_solve` subroutine.

## Arguments

  - `singp::SingType`: The singular surface data structure containing all relevant matrices and parameters.
  - `k::Int`: The current order in the power series expansion.
"""
function sing_solve!(singp::SingType, intr::DconInternal, k::Int)
    for l in 1:k
        singp.vmat[:, :, :, k+1] .+= sing_matmul(singp.mmat[:, :, :, l+1], singp.vmat[:, :, :, k-l+1])
    end
    for isol in 1:2*intr.numpert_total
        # CONTINUE FROM HERE - maybe try the same block representation as above?
        # a = M₀ - (α + k/2)I
        a = copy(singp.m0mat)
        a[1, 1] -= k / 2.0 + singp.power[isol]
        a[2, 2] -= k / 2.0 + singp.power[isol]
        det = a[1, 1] * a[2, 2] - a[1, 2] * a[2, 1]
        # Solve the resonant indices
        x = -singp.vmat[singp.r1[1], isol, :, k+1]
        singp.vmat[singp.r1[1], isol, 1, k+1] = (a[2, 2] * x[1] - a[1, 2] * x[2]) / det
        singp.vmat[singp.r1[1], isol, 2, k+1] = (a[1, 1] * x[2] - a[2, 1] * x[1]) / det
        # Solve the non-resonant indices (where M₀v is in the null space?)
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
    sqrtfac = sqrt(complex(dpsi)) # √zᵏ
    pfac = abs(dpsi)^singp.alpha # zᵅ

    # Compute power series via Horner's method (eq. 45 in Glasser 2016)
    ua = copy(singp.vmat[:, :, :, 2*ctrl.sing_order+1])
    for iorder in (2*ctrl.sing_order-1):-1:0
        ua .= ua .* sqrtfac .+ singp.vmat[:, :, :, iorder+1]
    end

    # Restore powers (undo shearing transformation using z^(±0.5) and zᵅ)
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
        FourFitVars,DconInternal,OdeState},
    psieval::Float64)

    # Unpack structs and initialize
    ctrl, equil, ffit, intr, odet = params
    fill!(odet.tmp, 0)
    u1 = @view(u[:, :, 1])
    u2 = @view(u[:, :, 2])
    du1 = @view(du[:, :, 1])
    du2 = @view(du[:, :, 2])

    # Compute singfac = 1 / (m - nq)
    odet.q = Spl.spline_eval!(equil.sq, psieval)[4]
    odet.singfac_vec .= vec(1.0 ./ ((intr.mlow:intr.mhigh) .- odet.q .* (intr.nlow:intr.nhigh)'))

    # kinetic stuff - skip for now
    if false #(TODO: kin_flag)
        error("kin_flag not implemented yet")
    else
        # Evaluate matrix splines at the current psi value
        Spl.spline_eval!(odet.amat, ffit.amats, psieval)
        Spl.spline_eval!(odet.bmat, ffit.bmats, psieval)
        Spl.spline_eval!(odet.cmat, ffit.cmats, psieval)
        Spl.spline_eval!(odet.fmat_lower, ffit.fmats_lower, psieval)
        Spl.spline_eval!(odet.kmat, ffit.kmats, psieval)
        Spl.spline_eval!(odet.gmat, ffit.gmats, psieval)
        
        # Form full matrices from flat representations, either block-diagonal or full
        amat = reshape(odet.amat, intr.numpert_total, intr.numpert_total)
        bmat = reshape(odet.bmat, intr.numpert_total, intr.numpert_total)
        cmat = reshape(odet.cmat, intr.numpert_total, intr.numpert_total)
        fmat_lower = reshape(odet.fmat_lower, intr.numpert_total, intr.numpert_total)
        kmat = reshape(odet.kmat, intr.numpert_total, intr.numpert_total)
        gmat = reshape(odet.gmat, intr.numpert_total, intr.numpert_total)

        # ldiv!(A,B): Compute A \ B in-place and overwriting B to store the result,
        # where A is a factorization object.
        odet.Afact = cholesky(Hermitian(amat))
        # bmat = A⁻¹ * bmat
        ldiv!(odet.Afact, bmat)
        # cmat = A⁻¹ * cmat
        ldiv!(odet.Afact, cmat)
    end

    # Compute du
    if false #(TODO: kin_flag)
        error("kin_flag not implemented yet")
    else
        # See equations 22-24 in Glasser 2016 DCON paper for derivation
        # du[1] = - K̄ * u[1] + Q⁻¹ * u[2]
        du1 .= u2 .* odet.singfac_vec
        mul!(odet.tmp, kmat, u1)
        du1 .-= odet.tmp
        # du[1] = - F̄⁻¹ * K̄ * u[1] + F̄⁻¹ * Q⁻¹ * u[2]
        ldiv!(LowerTriangular(fmat_lower), du1)
        ldiv!(UpperTriangular(fmat_lower'), du1)
        # du[2] = G * u[1] + K̄^† * du[1] = G * u[1] - K̄^† * F̄⁻¹ * K̄ * u[1] + K̄^† * F̄⁻¹ * Q⁻¹ * u[2]
        mul!(odet.tmp, gmat, u1)
        du2 .= odet.tmp
        mul!(odet.tmp, adjoint(kmat), du1)
        du2 .+= odet.tmp
        # du[1] = - Q⁻¹ * F̄⁻¹ * K̄ * u[1] + Q⁻¹ * F̄⁻¹ * Q⁻¹ * u[2]
        du1 .*= odet.singfac_vec
    end

    # ud[1] = Ξ'_Ψ
    @views odet.ud[:, :, 1] .= du1
    # ud[2] = Ξ_s = - A⁻¹(B * Ξ'_Ψ - C * Ξ_Ψ), equation 18 of Glasser 2016
    mul!(odet.tmp, bmat, du1)
    odet.ud[:, :, 2] .= .-odet.tmp
    mul!(odet.tmp, cmat, u1)
    @views odet.ud[:, :, 2] .-= odet.tmp
end