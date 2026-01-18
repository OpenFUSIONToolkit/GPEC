"""
    sing_scan!(intr::DconInternal, ctrl::DconControl, equil::Equilibrium.PlasmaEquilibrium, ffit::FourFitVars)

Scan all singular surfaces and calculate asymptotic vmat and mmat matrices
and Mericer criterion. Performs the same function as `sing_scan` in the Fortran code.
"""
function sing_scan!(intr::DconInternal, ctrl::DconControl, equil::Equilibrium.PlasmaEquilibrium, ffit::FourFitVars)
    for ising in 1:intr.msing
        sing_vmat!(intr, ctrl, equil, ffit, ising)
    end
end

"""
    sing_find!(intr::DconInternal, equil::Equilibrium.PlasmaEquilibrium)

Locate singular rational q-surfaces (q = m/nn) using a bisection method
between extrema of the q-profile, and store their properties in `intr.sing`.
Performs the same function as `sing_find` in the Fortran code.
"""
function sing_find!(intr::DconInternal, equil::Equilibrium.PlasmaEquilibrium)

    # Loop over all toroidal mode numbers
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
                    abs(singfac) < 1e-8 && (converged=true; break)
                    singfac > 0 ? (psi0 = psifac) : (psi1 = psifac)
                end

                if !converged
                    error("Bisection did not converge for m = $m after $itmax iterations.")
                elseif any(s -> isapprox(s.q, m / n; atol=1e-8), intr.sing)
                    # Rational surface with multiplicity > 1, add this m,n to the resonant mode numbers
                    # Technically only need m or n, but simplifies some later code and cheap to store both
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
    # Sort singular surfaces by increasing ψ
    intr.sing = sort(intr.sing; by=s -> s.psifac)
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
            error("Setting psilim via dmlim is only valid for single n runs (nn_low == nn_high).")
        end
        @info "Setting psilim via dmlim: initial qlim = $(intr.qlim), dmlim = $(ctrl.dmlim)"
        # Normalize dmlim ∈ [0,1)
        ctrl.dmlim = mod(ctrl.dmlim, 1.0)
        intr.qlim = (trunc(Int, ctrl.nn_low * intr.qlim) + ctrl.dmlim) / ctrl.nn_low

        # Reduce qlim if above qmax
        while intr.qlim > equil.params.qmax
            intr.qlim -= 1.0 / ctrl.nn_low
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
            if abs(dpsi) < eps * abs(intr.psilim)
                converged = true
                intr.q1lim = q1val(intr.psilim)
                break
            end
        end
        if !converged
            error("Can't find psilim after $itmax iterations.")
        end
    end
end

"""
    sing_vmat!(intr::DconInternal, ctrl::DconControl, equil::Equilibrium.PlasmaEquilibrium, ffit::FourFitVars, ising::Int)

Calculate asymptotic vmat and mmat matrices and Mercier criterion for
singular surface `ising`. Performs the same function as `sing_vmat` in the Fortran code.
Main differences are 1-indexing for the expansion orders. See equations 41-48 in
the 2016 Glasser DCON paper for the mathematical details.

### Arguments

  - `ising::Int`: Index of the singular surface to process (1 to `intr.msing`)

### TODOs

Check logic on typing of di
"""
function sing_vmat!(intr::DconInternal, ctrl::DconControl, equil::Equilibrium.PlasmaEquilibrium, ffit::FourFitVars, ising::Int)

    # Allocations
    singp = intr.sing[ising]
    singp.vmat = zeros(ComplexF64, intr.numpert_total, 2 * intr.numpert_total, 2, 2 * ctrl.sing_order + 1)
    singp.mmat = zeros(ComplexF64, intr.numpert_total, 2 * intr.numpert_total, 2, 2 * ctrl.sing_order + 3)
    singp.power = zeros(ComplexF64, 2 * intr.numpert_total)

    # Compute the resonant (r) and nonresonant (n) indices of the shearing transformation matrix R
    # 1 indexes along the N*M dimension, and 2 along the 2*N*M dimension
    # In 2D, see eq. 41 of 2016 Glasser DCON paper
    # TODO: if we remove the 3rd dimension, no need for both r1 and r2
    ipert_res = 1 .+ singp.m .- intr.mlow .+ (singp.n .- intr.nlow) .* intr.mpert
    singp.r1 = ipert_res
    singp.r2 = vec([ipert_res[i] + j * intr.numpert_total for j in 0:1, i in eachindex(ipert_res)])
    singp.n1 = [i for i in 1:intr.numpert_total if !(i in ipert_res)]
    singp.n2 = vec([i + j * intr.numpert_total for j in 0:1, i in singp.n1])

    psifac = singp.psifac
    q = singp.q
    di0 = Spl.spline_eval!(intr.locstab, singp.psifac)[1] / singp.psifac
    q1 = singp.q1
    rho = singp.rho

    # Compute Mercier criterion and singular power
    sing_mmat!(intr, ctrl, equil, ffit, ising)
    # TODO: My approach for the following logic is to mimic the existing code but go block by block
    # in m0mat (i.e. looping through each resonance). I think it works for 2D, probably not 3D
    # Note: We only need the transpose here because the third dimension corresponds to the bottom half of the 2N X 2N matrix
    # If we get rid of the 3rd dimension, this becomes simpler
    if length(singp.r1) == 1
        singp.m0mat = transpose(singp.mmat[singp.r1[1], singp.r2, :, 1])
    else
        singp.m0mat = vcat([transpose(singp.mmat[singp.r1[i], singp.r2, :, 1]) for i in eachindex(singp.r1)]...)
    end

    singp.alpha = eigen(singp.m0mat).values[(length(singp.r1)+1):end] # take the M largest eigenvalues
    # In 3D, need to do a surface average to obtain the di computed in Mercier.jl
    # In 2D, I think alphas are the same for all resonances so can just take the first index
    singp.di = -real(singp.alpha[1]^2)

    # This is the parameter α but for all modes - α = 0 for non-resonant modes
    singp.power[ipert_res] .= -singp.alpha
    singp.power[ipert_res .+ intr.numpert_total] .= singp.alpha

    # Zeroth-order non-resonant solutions
    # TODO: without the third dimension, this is just setting to the identity
    singp.vmat .= 0
    for ipert in 1:intr.numpert_total
        singp.vmat[ipert, ipert, 1, 1] = 1
        singp.vmat[ipert, ipert+intr.numpert_total, 2, 1] = 1
    end

    # Zeroth-order resonant solutions - solve (M₀ - αI)v₀ = 0
    # TODO: this will probably need a better generalization in 3D
    for i in eachindex(singp.r1) # go block by block in M₀
        m0mat = singp.m0mat[(2*(i-1)+1):(2*i), (2*(i-1)+1):(2*i)]
        r1 = singp.r1[i]
        r2 = r1 + intr.numpert_total
        alpha = singp.alpha[i]
        singp.vmat[r1, r1, 1, 1] = 1
        singp.vmat[r1, r2, 1, 1] = 1
        singp.vmat[r1, r1, 2, 1] = -(m0mat[1, 1] + alpha) / m0mat[1, 2]
        singp.vmat[r1, r2, 2, 1] = -(m0mat[1, 1] - alpha) / m0mat[1, 2]
        det = conj(singp.vmat[r1, r1, 1, 1]) * singp.vmat[r1, r2, 2, 1] -
              conj(singp.vmat[r1, r2, 1, 1]) * singp.vmat[r1, r1, 2, 1]
        singp.vmat[r1, :, :, 1] ./= sqrt(det)
    end

    # Higher order solutions - need to solve iteratively
    for k in 1:(2*ctrl.sing_order)
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
    q .= getindex.(Spl.spline_deriv3!(equil.sq, singp.psifac), 4)
    f_lower_interp[:, :, 1], f_lower_interp[:, :, 2], f_lower_interp[:, :, 3], f_lower_interp[:, :, 4] = Spl.spline_deriv3!(ffit.fmats_lower, singp.psifac)
    g_interp[:, :, 1], g_interp[:, :, 2], g_interp[:, :, 3], g_interp[:, :, 4] = Spl.spline_deriv3!(ffit.gmats, singp.psifac)
    k_interp[:, :, 1], k_interp[:, :, 2], k_interp[:, :, 3], k_interp[:, :, 4] = Spl.spline_deriv3!(ffit.kmats, singp.psifac)

    # Evaluate Taylor series coefficients for diagonal matrix Qᵢ = mᵢ - nᵢq(ψ) = [mᵢ - nᵢq, -nᵢq', -nᵢq'', -nᵢq''']
    singfac[:, 1] .= vec((intr.mlow:intr.mhigh) .- q[1] .* (intr.nlow:intr.nhigh)')
    for i in 2:4
        singfac[:, i] .= repeat(-(intr.nlow:intr.nhigh) .* q[i]; inner=intr.mpert)
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
    # For now, leaving overly detailed comments to remind so I don't have to work through this again
    # First, compute Taylor series coefficients of QL̄ (but without scaling by 1/n!), so we get binomial coefficients leftover
    # f_lower = QL̄ = [QL̄, QL̄' + Q' L̄, 1/2 (QL̄'' + 2Q' L̄' + QQ'' L̄), 1/6 (QL̄''' + 3Q' L̄'' + 3Q'' L̄' + Q'''L̄), ...] (but without 1/2, 1/6, etc)
    for ipert_n in 1:intr.npert
        for jpert_m in 1:intr.mpert
            for ipert_m in jpert_m:min(intr.mpert, jpert_m+intr.mband)
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
                    for ipert_m in jpert_m:min(intr.mpert, jpert_m+intr.mband)
                        for kpert_m in max(1, ipert_m-intr.mband):jpert_m
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
            for ipert_m in max(1, jpert_m-intr.mband):min(intr.mpert, jpert_m+intr.mband)
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
            for ipert_m in jpert_m:min(intr.mpert, jpert_m+intr.mband)
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

    # We will now compute the Taylor series expansion of x = Lv, with L specified in eq. 23 of Glasser 2016
    # Start with the identity matrix (which can be indexed to project onto resonant/nonresonant modes)
    for ipert in 1:intr.numpert_total
        v[ipert, ipert, 1] = 1
        v[ipert, ipert+intr.numpert_total, 2] = 1
    end

    # Solve the Taylor expansion according to F * x¹ = v² - K v¹ at each order
    # 0ᵗʰ order: x¹₀ = F⁻¹(v² - K v¹)
    for isol in 1:(2*intr.numpert_total)
        @views x[:, isol, 1, 1] .= v[:, isol, 2] .- k[:, :, 1] * v[:, isol, 1]
    end
    @views x[:, :, 1, 1] = UpperTriangular(f0_lower') \ (LowerTriangular(f0_lower) \ x[:, :, 1, 1])

    # Higher-order: ∑Fⱼx¹ₙ₋ⱼ = -Kₙv¹ → x¹ₙ = F₀⁻¹(-∑Fⱼxₙ₋ⱼ - Kₙv¹)
    for i in 1:ctrl.sing_order
        for isol in 1:(2*intr.numpert_total)
            for j in 1:i
                @views x[:, isol, 1, i+1] .-= Hermitian(ff_lower[:, :, j+1], :L) * x[:, isol, 1, i-j+1]
            end
            @views x[:, isol, 1, i+1] .-= k[:, :, i+1] * v[:, isol, 1]
        end
        @views x[:, :, 1, i+1] = UpperTriangular(f0_lower') \ (LowerTriangular(f0_lower) \ x[:, :, 1, i+1])
    end

    # Solve x²ₙ = (G - K^†F⁻¹K)v¹ + K^†F⁻¹v² = Gₙv¹ + ∑Kⱼ^† x¹ₙ₋ⱼ at each order
    for i in 0:ctrl.sing_order
        for isol in 1:(2*intr.numpert_total)
            for j in 0:i
                x[:, isol, 2, i+1] .+= adjoint(k[:, :, j+1]) * x[:, isol, 1, i-j+1]
            end
            x[:, isol, 2, i+1] .+= Hermitian(g_lower[:, :, i+1], :L) * v[:, isol, 1]
        end
    end

    # Assemble power series coefficients of M = zS⁻¹(LS - S') at each order in √z
    # eq. 28 in Glasser 2023 PoP paper
    singp.mmat .= 0
    r1 = singp.r1
    r2 = singp.r2
    n1 = singp.n1
    n2 = singp.n2
    j = 0
    # Start with the S⁻¹LS components
    # Glasser PoP 2023 eq. 39: at each other of L, we get contributions to z^k from RLR,
    # z^k+0.5 from RLA and ALR, and z^k+1 from ALA (where A is the nonresonant part)
    for i in 0:ctrl.sing_order
        singp.mmat[r1, r2, :, j+1] .= x[r1, r2, :, i+1]
        singp.mmat[r1, n2, :, j+2] .= x[r1, n2, :, i+1]
        singp.mmat[n1, r2, :, j+2] .= x[n1, r2, :, i+1]
        singp.mmat[n1, n2, :, j+3] .= x[n1, n2, :, i+1]
        # Expansion of M is in half powers of z due to shearing transformation, so we jump by 2
        j += 2
    end
    # Apply the effect of the shearing transformation to the resonant indices R
    # Glasser PoP 2023 eq. 25 + 28: M = zS⁻¹LS - zS⁻¹S' = zS⁻¹LS + 0.5 [R, 0; 0, -R], 0ᵗʰ order only
    for i in eachindex(r1)
        singp.mmat[r1[i], r2[2*i-1], 1, 1] += 0.5
        singp.mmat[r1[i], r2[2*i], 2, 1] -= 0.5
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
    # TODO: rename this solver_higher_order_vmat?
    # Compute ∑Mₗvₖ₋ₗ
    for l in 1:k
        singp.vmat[:, :, :, k+1] .+= sing_matmul(singp.mmat[:, :, :, l+1], singp.vmat[:, :, :, k-l+1])
    end
    for isol in 1:(2*intr.numpert_total)
        for i in eachindex(singp.r1) # go block by block?
            # a = M₀ - (α + k/2)I = ∑Mₗvₖ₋ₗ (for multi-n 2D, we make a the ith block fo M₀)
            m0mat = singp.m0mat[(2*(i-1)+1):(2*i), (2*(i-1)+1):(2*i)]
            a = copy(m0mat)
            a[1, 1] -= k / 2.0 + singp.power[isol]
            a[2, 2] -= k / 2.0 + singp.power[isol]
            det = a[1, 1] * a[2, 2] - a[1, 2] * a[2, 1]
            # Solve the resonant indices
            x = -singp.vmat[singp.r1[i], isol, :, k+1]
            singp.vmat[singp.r1[i], isol, 1, k+1] = (a[2, 2] * x[1] - a[1, 2] * x[2]) / det
            singp.vmat[singp.r1[i], isol, 2, k+1] = (a[1, 1] * x[2] - a[2, 1] * x[1]) / det
        end
        # Solve the non-resonant indices (the eigenvalue α = 0, so M₀v = 0 (null space))
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
            @views mul!(tmp, a[:, (m+1):(2*m), j], b[:, i, 2])
            @views c[:, i, j] .+= tmp
        end
    end

    return c
end

"""
    sing_get_ua(ctrl::DconControl, intr::DconInternal, odet::OdeState)

Compute the asymptotic series solution for a given singular surface.
Fills and returns `ua` with the asymptotic solution vmat computed in
`sing_vmat`. We obtain the solution using equations 45 and 41 in the
2016 DCON paper. Performs the same function as `sing_get_ua` in the
Fortran code.
"""
function sing_get_ua(ctrl::DconControl, intr::DconInternal, odet::OdeState)

    singp = intr.sing[odet.ising]
    r1 = singp.r1
    r2 = singp.r2

    # Compute distance from singular surface (z)
    dpsi = odet.psifac - singp.psifac
    sqrtfac = sqrt(complex(dpsi))

    # Compute power series via Horner's method (eq. 45 in Glasser 2016)
    ua = copy(singp.vmat[:, :, :, 2*ctrl.sing_order+1])
    for iorder in (2*ctrl.sing_order-1):-1:0
        ua .= ua .* sqrtfac .+ singp.vmat[:, :, :, iorder+1] # sqrtfac becomes √zᵏ here
    end

    # Loop through resonances - this might change in 3D
    for i in eachindex(r1)
        # Form full power series solution for v by multiplying by zᵅ (eq. 45 in Glasser 2016)
        pfac = abs(dpsi) .^ singp.alpha[i] # zᵅ
        ua[:, r2[2*i-1], :] ./= pfac # /zᵅ = z⁻ᵅ
        ua[:, r2[2*i], :] .*= pfac

        # Apply shearing transformation u = Rv (eq. 41 in Glasser 2016)
        ua[r1[i], :, 1] ./= sqrtfac # z^-0.5
        ua[r1[i], :, 2] .*= sqrtfac # z^0.5

        # Renormalize
        if odet.psifac < singp.psifac
            ua[:, r2[2*i-1], :] .*= abs(ua[r1[i], r2[2*i-1], 1]) / ua[r1[i], r2[2*i-1], 1]
            ua[:, r2[2*i], :] .*= abs(ua[r1[i], r2[2*i], 1]) / ua[r1[i], r2[2*i], 1]
        end
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
    temp1 = zeros(ComplexF64, 2 * intr.numpert_total, 2 * intr.numpert_total)
    temp1[1:intr.numpert_total, :] .= ua[:, :, 1]
    temp1[(intr.numpert_total+1):(2*intr.numpert_total), :] .= ua[:, :, 2]

    # Built temp2
    temp2 = zeros(ComplexF64, 2 * intr.numpert_total, intr.numpert_total)
    temp2[1:intr.numpert_total, :] .= odet.u[:, :, 1]
    temp2[(intr.numpert_total+1):(2*intr.numpert_total), :] .= odet.u[:, :, 2]

    # LU factorization and solve
    temp2 .= lu(temp1) \ temp2

    # Build ca
    ca = zeros(ComplexF64, intr.numpert_total, intr.numpert_total, 2)
    ca[:, :, 1] .= temp2[1:intr.numpert_total, :]
    ca[:, :, 2] .= temp2[(intr.numpert_total+1):(2*intr.numpert_total), :]

    return ca
end

#
"""
    sing_der!(
        du::Array{ComplexF64,3},
        u::Array{ComplexF64,3},
        params::Tuple{DconControl, Equilibrium.PlasmaEquilibrium, FourFitVars, DconInternal, OdeState},
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
  - `params::Tuple{DconControl, Equilibrium.PlasmaEquilibrium, FourFitVars, DconInternal, OdeState}`: Tuple of relevant structs
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

    # Evaluate matrix splines at the current psi value
    if ctrl.kin_flag
        # Evaluate kinetic matrices (infrastructure ready, ODE formulation not yet implemented)
        Spl.spline_eval!(odet.amat, ffit.akmats, psieval)
        Spl.spline_eval!(odet.bmat, ffit.bkmats, psieval)
        Spl.spline_eval!(odet.cmat, ffit.ckmats, psieval)
        Spl.spline_eval!(odet.fmat_lower, ffit.f0mats, psieval)
        Spl.spline_eval!(odet.kmat, ffit.kkmats, psieval)
        Spl.spline_eval!(odet.gmat, ffit.gaats, psieval)

        # TODO: Kinetic ODE formulation differs from ideal MHD
        # Will need pmats, paats, r1mats, r2mats, r3mats, kkaats for complete implementation
        error("Kinetic ODE integration not yet implemented - matrix infrastructure ready")
    else
        # Evaluate ideal MHD matrix splines at the current psi value
        Spl.spline_eval!(odet.amat, ffit.amats, psieval)
        Spl.spline_eval!(odet.bmat, ffit.bmats, psieval)
        Spl.spline_eval!(odet.cmat, ffit.cmats, psieval)
        Spl.spline_eval!(odet.fmat_lower, ffit.fmats_lower, psieval)
        Spl.spline_eval!(odet.kmat, ffit.kmats, psieval)
        Spl.spline_eval!(odet.gmat, ffit.gmats, psieval)

        # Form full matrices from flat representations
        # TODO: make these block diagonal for multi-n?
        amat = reshape(odet.amat, intr.numpert_total, intr.numpert_total)
        bmat = reshape(odet.bmat, intr.numpert_total, intr.numpert_total)
        cmat = reshape(odet.cmat, intr.numpert_total, intr.numpert_total)
        fmat_lower = reshape(odet.fmat_lower, intr.numpert_total, intr.numpert_total)
        kmat = reshape(odet.kmat, intr.numpert_total, intr.numpert_total)
        gmat = reshape(odet.gmat, intr.numpert_total, intr.numpert_total)

        odet.Afact = cholesky(Hermitian(amat))
        # bmat = A⁻¹ * bmat
        ldiv!(odet.Afact, bmat)
        # cmat = A⁻¹ * cmat
        ldiv!(odet.Afact, cmat)
    end

    # Compute du (ideal MHD formulation)
    #TODO: implement kinetic formulation
    if !ctrl.kin_flag
        # See equations 22-24 in Glasser 2016 DCON paper for derivation
        # du[1] = - F̄⁻¹ * K̄ * u[1] + F̄⁻¹ * Q⁻¹ * u[2]
        du1 .= u2 .* odet.singfac_vec
        mul!(odet.tmp, kmat, u1)
        du1 .-= odet.tmp
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
    # Note: kinetic case handled above with error message

    # ud[1] = Ξ'_Ψ
    @views odet.ud[:, :, 1] .= du1
    # ud[2] = Ξ_s = - A⁻¹(B * Ξ'_Ψ - C * Ξ_Ψ), eq. 18 of Glasser 2016
    mul!(odet.tmp, bmat, du1)
    odet.ud[:, :, 2] .= .-odet.tmp
    mul!(odet.tmp, cmat, u1)
    @views odet.ud[:, :, 2] .-= odet.tmp
end


#= Claude's slightly less incorrect sing_der implementation
"""
    sing_der!(
        du::Array{ComplexF64,3},
        u::Array{ComplexF64,3},
        params::Tuple{DconControl, Equilibrium.PlasmaEquilibrium, FourFitVars, DconInternal, OdeState},
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

  - `du::Array{ComplexF64,3}`: Pre-allocated array to hold the derivative result, shape (mpert, msol, 2), updated in-place
  - `u::Array{ComplexF64,3}`: Current state array, shape (mpert, msol, 2)
  - `params::Tuple{DconControl, Equilibrium.PlasmaEquilibrium, FourFitVars, DconInternal, OdeState}`: Tuple of relevant structs
  - `psieval::Float64`: Current psi value at which to evaluate the derivative
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

    mpert = intr.numpert_total
    msol = intr.numpert_total  # Number of solutions

    # Get safety factor and compute singularity factors
    q_val = Spl.spline_eval!(equil.sq, psieval)[4]
    odet.q = q_val  # Store in odet for access elsewhere
    chi1 = 2π * equil.psio  # twopi * psio

    # Compute singfac = 1 / (m - n*q) for each mode
    odet.singfac_vec .= vec(1.0 ./ ((intr.mlow:intr.mhigh) .- q_val .* (intr.nlow:intr.nhigh)'))
    #=singfac_vec = zeros(Float64, mpert)
    for ipert in 1:mpert
        m_val = intr.mlow + ipert - 1
        singfac_vec[ipert] = 1.0 / (m_val - ctrl.nn * q_val)
    end
    =#

    if ctrl.kin_flag
        #===========================================
        KINETIC CASE
        ===========================================#

        # Evaluate base (ideal MHD) matrix splines
        amat_ideal_flat = zeros(ComplexF64, mpert * mpert)
        bmat_ideal_flat = zeros(ComplexF64, mpert * mpert)
        cmat_ideal_flat = zeros(ComplexF64, mpert * mpert)
        dmat_ideal_flat = zeros(ComplexF64, mpert * mpert)
        emat_ideal_flat = zeros(ComplexF64, mpert * mpert)
        hmat_ideal_flat = zeros(ComplexF64, mpert * mpert)
        dbat_flat = zeros(ComplexF64, mpert * mpert)
        ebat_flat = zeros(ComplexF64, mpert * mpert)
        fbat_flat = zeros(ComplexF64, mpert * mpert)

        Spl.spline_eval!(amat_ideal_flat, ffit.amats, psieval)
        Spl.spline_eval!(bmat_ideal_flat, ffit.bmats, psieval)
        Spl.spline_eval!(cmat_ideal_flat, ffit.cmats, psieval)
        Spl.spline_eval!(dmat_ideal_flat, ffit.dmats, psieval)
        Spl.spline_eval!(emat_ideal_flat, ffit.emats, psieval)
        Spl.spline_eval!(hmat_ideal_flat, ffit.hmats, psieval)
        Spl.spline_eval!(dbat_flat, ffit.dbats, psieval)
        Spl.spline_eval!(ebat_flat, ffit.ebats, psieval)
        Spl.spline_eval!(fbat_flat, ffit.fbats, psieval)

        # Reshape to matrices
        amat_ideal = reshape(amat_ideal_flat, mpert, mpert)
        bmat_ideal = reshape(bmat_ideal_flat, mpert, mpert)
        cmat_ideal = reshape(cmat_ideal_flat, mpert, mpert)
        dmat_ideal = reshape(dmat_ideal_flat, mpert, mpert)
        emat_ideal = reshape(emat_ideal_flat, mpert, mpert)
        hmat_ideal = reshape(hmat_ideal_flat, mpert, mpert)
        dbat = reshape(dbat_flat, mpert, mpert)
        ebat = reshape(ebat_flat, mpert, mpert)
        fbat = reshape(fbat_flat, mpert, mpert)

        # Evaluate kinetic contribution matrices (kwmat and ktmat)
        kwmat = zeros(ComplexF64, mpert, mpert, 6)
        ktmat = zeros(ComplexF64, mpert, mpert, 6)
        for i in 1:6
            kwmat_temp = zeros(ComplexF64, mpert * mpert)
            ktmat_temp = zeros(ComplexF64, mpert * mpert)
            Spl.spline_eval!(kwmat_temp, ffit.kwmats[i], psieval)
            Spl.spline_eval!(ktmat_temp, ffit.ktmats[i], psieval)
            kwmat[:, :, i] = reshape(kwmat_temp, mpert, mpert)
            ktmat[:, :, i] = reshape(ktmat_temp, mpert, mpert)
        end

        # Declare variables for kinetic matrices that will be computed
        local amat_work, bmat_work, cmat_work  # Matrices that will be factored/modified
        local fmat_kin, kmat_kin, kaat_kin, gaat_kin  # Final kinetic matrices for ODE

        # Check if using pre-computed kinetic matrices or computing on-the-fly
        if ctrl.fkg_kmats_flag
            #===========================================
            Use PRE-COMPUTED kinetic matrices
            ===========================================#

            amat_k_flat = zeros(ComplexF64, mpert * mpert)
            bmat_k_flat = zeros(ComplexF64, mpert * mpert)
            cmat_k_flat = zeros(ComplexF64, mpert * mpert)
            f0mat_flat = zeros(ComplexF64, mpert * mpert)
            pmat_flat = zeros(ComplexF64, mpert * mpert)
            paat_flat = zeros(ComplexF64, mpert * mpert)
            kkmat_flat = zeros(ComplexF64, mpert * mpert)
            kkaat_flat = zeros(ComplexF64, mpert * mpert)
            r1mat_flat = zeros(ComplexF64, mpert * mpert)
            r2mat_flat = zeros(ComplexF64, mpert * mpert)
            r3mat_flat = zeros(ComplexF64, mpert * mpert)
            gaat_flat = zeros(ComplexF64, mpert * mpert)

            Spl.spline_eval!(amat_k_flat, ffit.akmats, psieval)
            Spl.spline_eval!(bmat_k_flat, ffit.bkmats, psieval)
            Spl.spline_eval!(cmat_k_flat, ffit.ckmats, psieval)
            Spl.spline_eval!(f0mat_flat, ffit.f0mats, psieval)
            Spl.spline_eval!(pmat_flat, ffit.pmats, psieval)
            Spl.spline_eval!(paat_flat, ffit.paats, psieval)
            Spl.spline_eval!(kkmat_flat, ffit.kkmats, psieval)
            Spl.spline_eval!(kkaat_flat, ffit.kkaats, psieval)
            Spl.spline_eval!(r1mat_flat, ffit.r1mats, psieval)
            Spl.spline_eval!(r2mat_flat, ffit.r2mats, psieval)
            Spl.spline_eval!(r3mat_flat, ffit.r3mats, psieval)
            Spl.spline_eval!(gaat_flat, ffit.gaats, psieval)

            amat_precomp = reshape(amat_k_flat, mpert, mpert)
            bmat_precomp = reshape(bmat_k_flat, mpert, mpert)
            cmat_precomp = reshape(cmat_k_flat, mpert, mpert)
            f0mat = reshape(f0mat_flat, mpert, mpert)
            pmat = reshape(pmat_flat, mpert, mpert)
            paat = reshape(paat_flat, mpert, mpert)
            kkmat = reshape(kkmat_flat, mpert, mpert)
            kkaat = reshape(kkaat_flat, mpert, mpert)
            r1mat = reshape(r1mat_flat, mpert, mpert)
            r2mat = reshape(r2mat_flat, mpert, mpert)
            r3mat = reshape(r3mat_flat, mpert, mpert)
            gaat_kin = reshape(gaat_flat, mpert, mpert)

            # Factor matrix A (LU decomposition)
            amat_fact = lu(amat_precomp)
            if !issuccess(amat_fact)
                error("LU factorization failed: amat singular at psieval = $psieval, reduce delta_mband")
            end

            # These will be used in the ODE computation
            amat_work = amat_precomp
            bmat_work = bmat_precomp
            cmat_work = cmat_precomp

        else
            #===========================================
            COMPUTE kinetic matrices on-the-fly
            ===========================================#

            # Create kinetic versions by adding contributions to ideal MHD matrices
            # Note: Using .+ creates NEW arrays, not modifying the originals
            amat_kin_full = amat_ideal .+ kwmat[:, :, 1] .+ ktmat[:, :, 1]
            bmat_kin_full = bmat_ideal .+ kwmat[:, :, 2] .+ ktmat[:, :, 2]
            cmat_kin_full = cmat_ideal .+ kwmat[:, :, 3] .+ ktmat[:, :, 3]
            dmat_kin_full = dmat_ideal .+ kwmat[:, :, 4] .+ ktmat[:, :, 4]
            emat_kin_full = emat_ideal .+ kwmat[:, :, 5] .+ ktmat[:, :, 5]
            hmat_kin_full = hmat_ideal .+ kwmat[:, :, 6] .+ ktmat[:, :, 6]

            # Compute modified matrices (these are NEW arrays)
            baat = bmat_kin_full .- 2.0 .* ktmat[:, :, 2]
            caat = cmat_kin_full .- 2.0 .* ktmat[:, :, 3]
            eaat = emat_kin_full .- 2.0 .* ktmat[:, :, 5]
            b1mat = im .* dbat

            # Factor kinetic non-Hermitian matrix A
            amat_fact = lu(amat_kin_full)
            if !issuccess(amat_fact)
                error("LU factorization failed: amat singular at psieval = $psieval, reduce delta_mband")
            end

            # Compute f0mat = fbat - dbat^† * A⁻¹ * dbat
            temp1 = copy(dbat)
            ldiv!(amat_fact, temp1)
            f0mat = fbat .- adjoint(dbat) * temp1

            # Prepare matrices to separate Q factors
            # aamat = (A⁻¹)^†
            temp2 = copy(amat_kin_full)
            ldiv!(amat_fact, temp2)
            aamat = adjoint(temp2)
            umat = Matrix{ComplexF64}(I, mpert, mpert) .- aamat

            # Compute bkmat and bkaat
            bkmat = kwmat[:, :, 2] .+ ktmat[:, :, 2] .+
                   im .* chi1 / (2π * ctrl.nn) .* (kwmat[:, :, 1] .+ ktmat[:, :, 1])
            bkaat = kwmat[:, :, 2] .- ktmat[:, :, 2] .+
                   im .* chi1 / (2π * ctrl.nn) .* (kwmat[:, :, 1] .+ ktmat[:, :, 1])

            # Compute pmat = b1mat^† * A⁻¹ * bkmat
            temp2 = copy(bkmat)
            ldiv!(amat_fact, temp2)
            pmat = adjoint(b1mat) * temp2

            # Compute paat
            temp2 = copy(b1mat)
            ldiv!(amat_fact, temp2)
            paat = adjoint(bkaat) * temp2 .-
                   im .* chi1 / (2π * ctrl.nn) .* umat * b1mat
            paat = adjoint(paat)

            # Compute r1mat
            temp1 = kwmat[:, :, 1] .+ ktmat[:, :, 1]
            temp2 = copy(bkmat)
            ldiv!(amat_fact, temp2)
            r1mat = kwmat[:, :, 4] .+ ktmat[:, :, 4] .-
                   (chi1 / (2π * ctrl.nn))^2 .* adjoint(temp1) .+
                   im .* chi1 / (2π * ctrl.nn) .* adjoint(bkaat) .-
                   im .* chi1 / (2π * ctrl.nn) .* aamat * bkmat .-
                   adjoint(bkaat) * temp2

            # Compute r2mat
            temp1 = kwmat[:, :, 5] .+ ktmat[:, :, 5] .-
                   im .* chi1 / (2π * ctrl.nn) .* (kwmat[:, :, 3] .+ ktmat[:, :, 3])
            temp2 = copy(cmat_kin_full)
            ldiv!(amat_fact, temp2)
            r2mat = temp1 .+ im .* chi1 / (2π * ctrl.nn) .* umat * cmat_kin_full .-
                   adjoint(bkaat) * temp2

            # Compute r3mat
            temp1 = kwmat[:, :, 5] .- ktmat[:, :, 5] .-
                   im .* chi1 / (2π * ctrl.nn) .* (kwmat[:, :, 3] .- ktmat[:, :, 3])
            temp2 = copy(bkmat)
            ldiv!(amat_fact, temp2)
            r3mat = adjoint(temp1) .- adjoint(caat) * temp2

            # Compute kkmat = ebat - b1mat^† * A⁻¹ * cmat
            temp1 = copy(cmat_kin_full)
            ldiv!(amat_fact, temp1)
            kkmat = ebat .- adjoint(b1mat) * temp1

            # Compute kkaat = ebat^† - caat^† * A⁻¹ * b1mat
            temp1 = copy(b1mat)
            ldiv!(amat_fact, temp1)
            kkaat = adjoint(ebat) .- adjoint(caat) * temp1

            # Compute gaat = hmat - caat^† * A⁻¹ * cmat
            temp2 = copy(cmat_kin_full)
            ldiv!(amat_fact, temp2)
            gaat_kin = hmat_kin_full .- adjoint(caat) * temp2

            # Set work matrices for subsequent operations
            amat_work = amat_kin_full
            bmat_work = bmat_kin_full
            cmat_work = cmat_kin_full
        end

        # Solve A⁻¹ * bmat and A⁻¹ * cmat
        # Create copies since we'll modify them via ldiv!
        bmat_inv = copy(bmat_work)
        cmat_inv = copy(cmat_work)
        ldiv!(amat_fact, bmat_inv)
        ldiv!(amat_fact, cmat_inv)

        # Calculate kinetic non-Hermitian F, K, K† matrices
        fmat_kin = zeros(ComplexF64, mpert, mpert)
        kmat_kin = zeros(ComplexF64, mpert, mpert)
        kaat_kin = zeros(ComplexF64, mpert, mpert)

        for ipert in 1:mpert
            m1 = intr.mlow + ipert - 1
            singfac1 = m1 - ctrl.nn * q_val
            for jpert in 1:mpert
                m2 = intr.mlow + jpert - 1
                singfac2 = m2 - ctrl.nn * q_val

                fmat_kin[ipert, jpert] = singfac1 * f0mat[ipert, jpert] * singfac2 -
                                        singfac1 * pmat[ipert, jpert] -
                                        conj(paat[jpert, ipert]) * singfac2 +
                                        r1mat[ipert, jpert]

                kmat_kin[ipert, jpert] = singfac1 * kkmat[ipert, jpert] +
                                        r2mat[ipert, jpert]

                kaat_kin[ipert, jpert] = kkaat[ipert, jpert] * singfac2 +
                                        r3mat[ipert, jpert]
            end
        end

        #===========================================
        Compute du1 and du2 for KINETIC case
        ===========================================#

        # Compute du(:,:,1) = u(:,:,2) - K * u(:,:,1)
        for isol in 1:msol
            du[:, isol, 1] .= u[:, isol, 2]
            # du = du - K * u  (with factors -1.0 and 1.0 for mul!)
            mul!(du[:, isol, 1], kmat_kin, u[:, isol, 1], -1.0, 1.0)
        end

        # Factor fmat (LU decomposition)
        fmat_fact = lu(fmat_kin)
        if !issuccess(fmat_fact)
            error("LU factorization failed: fmat singular at psieval = $psieval, reduce delta_mband")
        end

        # Solve F * du(:,:,1) = rhs (modifies du[:,:,1] in place)
        for isol in 1:msol
            ldiv!(fmat_fact, view(du, :, isol, 1))
        end

        # Compute du(:,:,2) = gaat * u(:,:,1) + kaat * du(:,:,1)
        for isol in 1:msol
            # du(:,isol,2) = 0 + gaat * u(:,isol,1)
            mul!(du[:, isol, 2], gaat_kin, u[:, isol, 1], 1.0, 0.0)
            # du(:,isol,2) = du(:,isol,2) + kaat * du(:,isol,1)
            mul!(du[:, isol, 2], kaat_kin, du[:, isol, 1], 1.0, 1.0)
        end

        # Calculate and store u-derivative and xss
        # ud[1] = Ξ'_Ψ (derivative with respect to psi)
        odet.ud[:, :, 1] .= du[:, :, 1]

        # ud[2] = Ξ_s = -B * Ξ'_Ψ - C * Ξ_Ψ (eq. 18 of Glasser 2016)
        for isol in 1:msol
            temp1 = bmat_inv * du[:, isol, 1]
            temp2 = cmat_inv * u[:, isol, 1]
            odet.ud[:, isol, 2] .= .-temp1 .- temp2
        end

    else
        #===========================================
        IDEAL MHD CASE
        ===========================================#

        # Evaluate ideal MHD matrix splines at the current psi value
        amat_flat = zeros(ComplexF64, mpert * mpert)
        bmat_flat = zeros(ComplexF64, mpert * mpert)
        cmat_flat = zeros(ComplexF64, mpert * mpert)
        fmat_flat = zeros(ComplexF64, mpert * mpert)
        kmat_flat = zeros(ComplexF64, mpert * mpert)
        gmat_flat = zeros(ComplexF64, mpert * mpert)

        Spl.spline_eval!(amat_flat, ffit.amats, psieval)
        Spl.spline_eval!(bmat_flat, ffit.bmats, psieval)
        Spl.spline_eval!(cmat_flat, ffit.cmats, psieval)
        Spl.spline_eval!(fmat_flat, ffit.fmats_lower, psieval)
        Spl.spline_eval!(kmat_flat, ffit.kmats, psieval)
        Spl.spline_eval!(gmat_flat, ffit.gmats, psieval)

        # Form full matrices from flat representations
        amat_ideal = reshape(amat_flat, mpert, mpert)
        bmat_ideal = reshape(bmat_flat, mpert, mpert)
        cmat_ideal = reshape(cmat_flat, mpert, mpert)
        fmat_lower = reshape(fmat_flat, mpert, mpert)
        kmat_ideal = reshape(kmat_flat, mpert, mpert)
        gmat_ideal = reshape(gmat_flat, mpert, mpert)

        # Factor A (Cholesky for Hermitian positive definite)
        amat_fact = cholesky(Hermitian(amat_ideal))

        # Solve A⁻¹ * bmat and A⁻¹ * cmat
        # Create copies since ldiv! modifies in place
        bmat_inv = copy(bmat_ideal)
        cmat_inv = copy(cmat_ideal)
        ldiv!(amat_fact, bmat_inv)
        ldiv!(amat_fact, cmat_inv)

        #===========================================
        Compute du1 and du2 for IDEAL MHD case
        ===========================================#

        # Compute du(:,:,1) = u(:,:,2) * singfac - K * u(:,:,1)
        for isol in 1:msol
            du[:, isol, 1] .= u[:, isol, 2] .* odet.singfac_vec
            # du = du - K * u
            mul!(du[:, isol, 1], kmat_ideal, u[:, isol, 1], -1.0, 1.0)
        end

        # Solve F * du(:,:,1) = rhs using Cholesky factorization
        # F is Hermitian positive definite, stored in lower triangular form
        for isol in 1:msol
            ldiv!(LowerTriangular(fmat_lower), view(du, :, isol, 1))
            ldiv!(UpperTriangular(fmat_lower'), view(du, :, isol, 1))
        end

        # Compute du(:,:,2) = G * u(:,:,1) + K^† * du(:,:,1)
        # Then multiply du(:,:,1) by singfac
        for isol in 1:msol
            # du(:,isol,2) = 0 + G * u(:,isol,1)
            mul!(du[:, isol, 2], gmat_ideal, u[:, isol, 1], 1.0, 0.0)
            # du(:,isol,2) = du(:,isol,2) + K^† * du(:,isol,1)
            mul!(du[:, isol, 2], adjoint(kmat_ideal), du[:, isol, 1], 1.0, 1.0)
            # Finally multiply du(:,isol,1) by singfac
            du[:, isol, 1] .*= odet.singfac_vec
        end

        # Calculate and store u-derivative and xss
        # ud[1] = Ξ'_Ψ (derivative with respect to psi)
        odet.ud[:, :, 1] .= du[:, :, 1]

        # ud[2] = Ξ_s = -B * Ξ'_Ψ - C * Ξ_Ψ (eq. 18 of Glasser 2016)
        for isol in 1:msol
            temp1 = bmat_inv * du[:, isol, 1]
            temp2 = cmat_inv * u[:, isol, 1]
            odet.ud[:, isol, 2] .= .-temp1 .- temp2
        end
    end

    return nothing
end
=#
#= GitHub Copilot's incorrect sing_der! implementation
"""
    sing_der!(
        du::Array{ComplexF64,3},
        u::Array{ComplexF64,3},
        params::Tuple{DconControl, Equilibrium.PlasmaEquilibrium, FourFitVars, DconInternal, OdeState},
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

  - `du::Array{ComplexF64,3}`: Pre-allocated array to hold the derivative result, shape (mpert, msol, 2), updated in-place
  - `u::Array{ComplexF64,3}`: Current state array, shape (mpert, msol, 2)
  - `params::Tuple{DconControl, Equilibrium.PlasmaEquilibrium, FourFitVars, DconInternal, OdeState}`: Tuple of relevant structs
  - `psieval::Float64`: Current psi value at which to evaluate the derivative
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

    mpert = intr.numpert_total
    msol = intr.numpert_total  # Number of solutions
    mband = intr.mband > 0 ? intr.mband : (intr.mhigh - intr.mlow)  # Compute if not set

    # Get safety factor and compute singularity factors
    q_val = Spl.spline_eval!(equil.sq, psieval)[4]
    odet.q = q_val  # Store in odet for access elsewhere
    chi1 = 2π * equil.psio  # twopi * psio

    # Compute singfac = 1 / (m - n*q) for each mode
    singfac = ones(ComplexF64, mpert)
    for ipert in 1:mpert
        m_val = intr.mlow + ipert - 1
        singfac[ipert] = 1.0 / (m_val - intr.nlow * q_val)
    end

    # Pre-allocate ipiv for factorizations
    ipiv = zeros(Int64, mpert)
    work = zeros(ComplexF64, mpert * mpert)

    # Pre-allocate banded storage arrays
    amatlu = zeros(ComplexF64, 2*mband+1, mpert)
    fmatlu = zeros(ComplexF64, 2*mband+1, mpert)
    kmatb = zeros(ComplexF64, 2*mband+1, mpert)
    kaatb = zeros(ComplexF64, 2*mband+1, mpert)
    gaatb = zeros(ComplexF64, 2*mband+1, mpert)
    fmatb = zeros(ComplexF64, mband+1, mpert)
    gmatb = zeros(ComplexF64, mband+1, mpert)

    if ctrl.kin_flag
        #===========================================
        KINETIC CASE
        ===========================================#

        # Evaluate base (ideal MHD) matrix splines and reshape them
        amat = reshape(Spl.spline_eval!(ffit.amats, psieval), mpert, mpert)
        bmat = reshape(Spl.spline_eval!(ffit.bmats, psieval), mpert, mpert)
        cmat = reshape(Spl.spline_eval!(ffit.cmats, psieval), mpert, mpert)
        dmat = reshape(Spl.spline_eval!(ffit.dmats, psieval), mpert, mpert)
        emat = reshape(Spl.spline_eval!(ffit.emats, psieval), mpert, mpert)
        hmat = reshape(Spl.spline_eval!(ffit.hmats, psieval), mpert, mpert)
        dbat = reshape(Spl.spline_eval!(ffit.dbats, psieval), mpert, mpert)
        ebat = reshape(Spl.spline_eval!(ffit.ebats, psieval), mpert, mpert)
        fbat = reshape(Spl.spline_eval!(ffit.fbats, psieval), mpert, mpert)

        # Evaluate kinetic contribution matrices (6 components)
        kwmat = zeros(ComplexF64, mpert, mpert, 6)
        ktmat = zeros(ComplexF64, mpert, mpert, 6)
        for i in 1:6
            kwmat[:,:,i] = reshape(Spl.spline_eval!(ffit.kwmats[i], psieval), mpert, mpert)
            ktmat[:,:,i] = reshape(Spl.spline_eval!(ffit.ktmats[i], psieval), mpert, mpert)
        end

        # Compute kinetic matrices - either pre-computed or on-the-fly
        if ctrl.fkg_kmats_flag
            #===========================================
            Use PRE-COMPUTED kinetic matrices
            ===========================================#

            amat = reshape(Spl.spline_eval!(zeros(ComplexF64, mpert*mpert), ffit.akmats, psieval), mpert, mpert)
            bmat = reshape(Spl.spline_eval!(zeros(ComplexF64, mpert*mpert), ffit.bkmats, psieval), mpert, mpert)
            cmat = reshape(Spl.spline_eval!(zeros(ComplexF64, mpert*mpert), ffit.ckmats, psieval), mpert, mpert)
            f0mat = reshape(Spl.spline_eval!(zeros(ComplexF64, mpert*mpert), ffit.f0mats, psieval), mpert, mpert)
            pmat = reshape(Spl.spline_eval!(zeros(ComplexF64, mpert*mpert), ffit.pmats, psieval), mpert, mpert)
            paat = reshape(Spl.spline_eval!(zeros(ComplexF64, mpert*mpert), ffit.paats, psieval), mpert, mpert)
            kkmat = reshape(Spl.spline_eval!(zeros(ComplexF64, mpert*mpert), ffit.kkmats, psieval), mpert, mpert)
            kkaat = reshape(Spl.spline_eval!(zeros(ComplexF64, mpert*mpert), ffit.kkaats, psieval), mpert, mpert)
            r1mat = reshape(Spl.spline_eval!(zeros(ComplexF64, mpert*mpert), ffit.r1mats, psieval), mpert, mpert)
            r2mat = reshape(Spl.spline_eval!(zeros(ComplexF64, mpert*mpert), ffit.r2mats, psieval), mpert, mpert)
            r3mat = reshape(Spl.spline_eval!(zeros(ComplexF64, mpert*mpert), ffit.r3mats, psieval), mpert, mpert)
            gaat = reshape(Spl.spline_eval!(zeros(ComplexF64, mpert*mpert), ffit.gaats, psieval), mpert, mpert)

            # Factor A using banded storage
            # Convert to banded format: amatlu(2*mband+1+ipert-jpert, jpert) = amat(ipert, jpert)
            fill!(amatlu, 0)
            for jpert in 1:mpert, ipert in 1:mpert
                amatlu[2*mband+1+ipert-jpert, jpert] = amat[ipert, jpert]
            end

            LinearAlgebra.LAPACK.gbtrf!(mpert, mpert, mband, mband, amatlu, ipiv)

        else
            #===========================================
            COMPUTE kinetic matrices on-the-fly
            ===========================================#

            # Add kinetic contributions to ideal matrices
            amat .+= kwmat[:,:,1] .+ ktmat[:,:,1]
            bmat .+= kwmat[:,:,2] .+ ktmat[:,:,2]
            cmat .+= kwmat[:,:,3] .+ ktmat[:,:,3]
            dmat .+= kwmat[:,:,4] .+ ktmat[:,:,4]
            emat .+= kwmat[:,:,5] .+ ktmat[:,:,5]
            hmat .+= kwmat[:,:,6] .+ ktmat[:,:,6]

            # Compute auxiliary matrices
            baat = bmat .- 2.0 .* ktmat[:,:,2]
            caat = cmat .- 2.0 .* ktmat[:,:,3]
            eaat = emat .- 2.0 .* ktmat[:,:,5]
            b1mat = 1im .* dbat

            # Initialize umat as identity
            umat = zeros(ComplexF64, mpert, mpert)
            for ipert in 1:mpert
                umat[ipert, ipert] = 1.0
            end

            # Factor kinetic non-Hermitian matrix A
            # Convert to banded format
            fill!(amatlu, 0)
            for jpert in 1:mpert, ipert in 1:mpert
                amatlu[2*mband+1+ipert-jpert, jpert] = amat[ipert, jpert]
            end

            LinearAlgebra.LAPACK.gbtrf!(mpert, mpert, mband, mband, amatlu, ipiv)

            # Compute f0mat = fbat - dbat^† * A⁻¹ * dbat
            temp1 = copy(dbat)
            LinearAlgebra.LAPACK.gbtrs!('N', mpert, mband, mband, amatlu, ipiv, temp1)
            f0mat = fbat .- adjoint(dbat) * temp1

            # Prepare matrices to separate Q factors
            temp2 = copy(amat)
            LinearAlgebra.LAPACK.gbtrs!('C', mpert, mband, mband, amatlu, ipiv, temp2)
            aamat = adjoint(temp2)
            umat .= -aamat
            for ipert in 1:mpert
                umat[ipert, ipert] += 1.0
            end

            # Compute bkmat and bkaat
            bkmat = kwmat[:,:,2] .+ ktmat[:,:,2] .+
                   1im .* chi1 / (2π * intr.nlow) .* (kwmat[:,:,1] .+ ktmat[:,:,1])
            bkaat = kwmat[:,:,2] .- ktmat[:,:,2] .+
                   1im .* chi1 / (2π * intr.nlow) .* (kwmat[:,:,1] .+ ktmat[:,:,1])

            # Compute pmat = b1mat^† * A⁻¹ * bkmat
            temp2 = copy(bkmat)
            LinearAlgebra.LAPACK.gbtrs!('N', mpert, mband, mband, amatlu, ipiv, temp2)
            pmat = adjoint(b1mat) * temp2

            # Compute paat
            temp2 = copy(b1mat)
            LinearAlgebra.LAPACK.gbtrs!('N', mpert, mband, mband, amatlu, ipiv, temp2)
            paat = adjoint(bkaat) * temp2 .-
                   1im .* chi1 / (2π * intr.nlow) .* umat * b1mat
            paat = adjoint(paat)

            # Compute r1mat
            temp1 = kwmat[:,:,1] .+ ktmat[:,:,1]
            temp2 = copy(bkmat)
            LinearAlgebra.LAPACK.gbtrs!('N', mpert, mband, mband, amatlu, ipiv, temp2)
            r1mat = kwmat[:,:,4] .+ ktmat[:,:,4] .-
                   (chi1 / (2π * intr.nlow))^2 .* adjoint(temp1) .+
                   1im .* chi1 / (2π * intr.nlow) .* adjoint(bkaat) .-
                   1im .* chi1 / (2π * intr.nlow) .* aamat * bkmat .-
                   adjoint(bkaat) * temp2

            # Compute r2mat
            temp1 = kwmat[:,:,5] .+ ktmat[:,:,5] .-
                   1im .* chi1 / (2π * intr.nlow) .* (kwmat[:,:,3] .+ ktmat[:,:,3])
            temp2 = copy(cmat)
            LinearAlgebra.LAPACK.gbtrs!('N', mpert, mband, mband, amatlu, ipiv, temp2)
            r2mat = temp1 .+ 1im .* chi1 / (2π * intr.nlow) .* umat * cmat .-
                   adjoint(bkaat) * temp2

            # Compute r3mat
            temp1 = kwmat[:,:,5] .- ktmat[:,:,5] .-
                   1im .* chi1 / (2π * intr.nlow) .* (kwmat[:,:,3] .- ktmat[:,:,3])
            temp2 = copy(bkmat)
            LinearAlgebra.LAPACK.gbtrs!('N', mpert, mband, mband, amatlu, ipiv, temp2)
            r3mat = adjoint(temp1) .- adjoint(caat) * temp2

            # Compute kkmat = ebat - b1mat^† * A⁻¹ * cmat
            temp1 = copy(cmat)
            LinearAlgebra.LAPACK.gbtrs!('N', mpert, mband, mband, amatlu, ipiv, temp1)
            kkmat = ebat .- adjoint(b1mat) * temp1

            # Compute kkaat = ebat^† - caat^† * A⁻¹ * b1mat
            temp1 = copy(b1mat)
            LinearAlgebra.LAPACK.gbtrs!('N', mpert, mband, mband, amatlu, ipiv, temp1)
            kkaat = adjoint(ebat) .- adjoint(caat) * temp1

            # Compute gaat = hmat - caat^† * A⁻¹ * cmat
            temp2 = copy(cmat)
            LinearAlgebra.LAPACK.gbtrs!('N', mpert, mband, mband, amatlu, ipiv, temp2)
            gaat = hmat .- adjoint(caat) * temp2
        end

        # Solve A⁻¹ * bmat and A⁻¹ * cmat to compute derivatives
        LinearAlgebra.LAPACK.gbtrs!('N', mpert, mband, mband, amatlu, ipiv, bmat)
        LinearAlgebra.LAPACK.gbtrs!('N', mpert, mband, mband, amatlu, ipiv, cmat)

        # Compute F, K, KAAT matrices and store in banded format
        fmat = zeros(ComplexF64, mpert, mpert)
        kmat = zeros(ComplexF64, mpert, mpert)
        kaat = zeros(ComplexF64, mpert, mpert)

        for ipert in 1:mpert, jpert in 1:mpert
            m1 = intr.mlow + ipert - 1
            m2 = intr.mlow + jpert - 1
            singfac1 = m1 - intr.nlow * q_val
            singfac2 = m2 - intr.nlow * q_val

            fmat[ipert, jpert] = singfac1 * f0mat[ipert, jpert] * singfac2 -
                                singfac1 * pmat[ipert, jpert] -
                                conj(paat[jpert, ipert]) * singfac2 +
                                r1mat[ipert, jpert]

            kmat[ipert, jpert] = singfac1 * kkmat[ipert, jpert] +
                                r2mat[ipert, jpert]

            kaat[ipert, jpert] = kkaat[ipert, jpert] * singfac2 +
                                r3mat[ipert, jpert]
        end

        # Convert fmat, kmat, kaat, and gaat to banded storage for LAPACK
        fill!(fmatlu, 0)
        for jpert in 1:mpert, ipert in 1:mpert
            fmatlu[2*mband+1+ipert-jpert, jpert] = fmat[ipert, jpert]
        end

        fill!(kmatb, 0)
        fill!(kaatb, 0)
        for jpert in 1:mpert
            for ipert in max(1, jpert-mband):min(mpert, jpert+mband)
                kmatb[1+mband+ipert-jpert, jpert] = kmat[ipert, jpert]
                kaatb[1+mband+ipert-jpert, jpert] = kaat[ipert, jpert]
            end
        end

        fill!(gaatb, 0)
        for jpert in 1:mpert
            for ipert in max(1, jpert-mband):min(mpert, jpert+mband)
                gaatb[1+mband+ipert-jpert, jpert] = gaat[ipert, jpert]
            end
        end

        # Now compute du/dpsi according to Fortran: du = 0
        fill!(du, 0)

        # du(:,isol,1) = u(:,isol,2) - K * u(:,isol,1)
        for isol in 1:msol
            du[:, isol, 1] .= u[:, isol, 2]
            # Banded matrix-vector product: du -= K * u using zgbmv
            LinearAlgebra.BLAS.gemv!('N', -one, kmat, u[:, isol, 1], one, du[:, isol, 1])
        end

        # Factor fmat in banded format
        ipiv_f = zeros(Int32, mpert)
        LinearAlgebra.LAPACK.gbtrf!(mpert, mpert, mband, mband, fmatlu, ipiv_f)

        # Solve fmat * du(:,:,1) = rhs
        LinearAlgebra.LAPACK.gbtrs!('N', mpert, mband, mband, msol, fmatlu, ipiv_f, du)

        # du(:,isol,2) = gaat * u(:,isol,1) + kaat * du(:,isol,1)
        for isol in 1:msol
            # du(:,isol,2) = gaat * u(:,isol,1)
            LinearAlgebra.BLAS.gemv!('N', one, gaat, u[:, isol, 1], zero, du[:, isol, 2])
            # du(:,isol,2) += kaat * du(:,isol,1)
            LinearAlgebra.BLAS.gemv!('N', one, kaat, du[:, isol, 1], one, du[:, isol, 2])
        end

        # Calculate and store u-derivative (ud)
        # ud(:,:,1) = du(:,:,1)
        odet.ud[:, :, 1] .= du[:, :, 1]

        # ud(:,:,2) = -B * du(:,:,1) - C * u(:,:,1)
        for isol in 1:msol
            odet.ud[:, isol, 2] .= -bmat * du[:, isol, 1] .- cmat * u[:, isol, 1]
        end

    else
        #===========================================
        IDEAL MHD CASE
        ===========================================#

        # Initialize du
        fill!(du, 0)

        # Evaluate ideal MHD matrix splines
        amat = reshape(Spl.spline_eval!(ffit.amats, psieval), mpert, mpert)
        bmat = reshape(Spl.spline_eval!(ffit.bmats, psieval), mpert, mpert)
        cmat = reshape(Spl.spline_eval!(ffit.cmats, psieval), mpert, mpert)
        fmat = reshape(Spl.spline_eval!(ffit.fmats_lower, psieval), mpert, mpert)
        kmat = reshape(Spl.spline_eval!(ffit.kmats, psieval), mpert, mpert)
        gmat = reshape(Spl.spline_eval!(ffit.gmats, psieval), mpert, mpert)

        # Factor A using LDL factorization (like Fortran zhetrf)
        # Then solve A⁻¹ * bmat and A⁻¹ * cmat
        amat_fact = copy(amat)
        LinearAlgebra.LAPACK.hetrf!('L', amat_fact, ipiv)

        bmat_inv = copy(bmat)
        cmat_inv = copy(cmat)
        LinearAlgebra.LAPACK.hetrs!('L', amat_fact, ipiv, bmat_inv)
        LinearAlgebra.LAPACK.hetrs!('L', amat_fact, ipiv, cmat_inv)

        # Convert K to banded storage (2*mband+1 format for non-Hermitian, centered at 1+mband)
        fill!(kmatb, 0)
        for jpert in 1:mpert
            for ipert in max(1, jpert-mband):min(mpert, jpert+mband)
                kmatb[1+mband+ipert-jpert, jpert] = kmat[ipert, jpert]
            end
        end

        # Step 1: du(:,isol,1) = u(:,isol,2) * singfac - K * u(:,isol,1) via banded matrix-vector product
        for isol in 1:msol
            du[:, isol, 1] .= u[:, isol, 2] .* singfac
            # du -= K * u using zgbmv (general banded matrix-vector multiply)
            LinearAlgebra.BLAS.gbmv!('N', mpert, mband, mband, -one, kmatb, u[:, isol, 1], one, du[:, isol, 1])
        end

        # Step 2: Convert F to banded storage (2*mband+1 format for general banded factorization)
        # In LAPACK banded format: AB[ku+1+i-j, j] = A[i,j] where ku is upper bandwidth
        fill!(fmatlu, 0)
        for jpert in 1:mpert
            for ipert in max(1, jpert-mband):min(mpert, jpert+mband)
                fmatlu[mband+1+ipert-jpert, jpert] = fmat[ipert, jpert]
            end
        end

        # Factor F using zgbtrf (general banded factorization)
        # Signature: gbtrf!(kl::Integer, ku::Integer, m::Integer, AB::AbstractMatrix)
        ipiv_fmat = LinearAlgebra.LAPACK.gbtrf!(mband, mband, mpert, fmatlu)

        # Solve F * du = du using zgbtrs (general banded matrix solve)
        # gbtrs! modifies the RHS in place with the solution
        for isol in 1:msol
            LinearAlgebra.LAPACK.gbtrs!('N', mband, mband, mpert, fmatlu, ipiv_fmat, du[:, isol, 1])
        end

        # Convert G to banded Hermitian storage (mband+1 format)
        fill!(gmatb, 0)
        for jpert in 1:mpert
            for ipert in jpert:min(mpert, jpert+mband)
                gmatb[1+ipert-jpert, jpert] = gmat[ipert, jpert]
            end
        end

        # Step 3: Compute du(:,isol,2) = G * u(:,isol,1) + K† * du(:,isol,1)
        for isol in 1:msol
            # du(:,isol,2) = G * u(:,isol,1) using zhbmv (Hermitian banded matrix-vector multiply)
            LinearAlgebra.BLAS.hbmv!('L', mband, one, gmatb, u[:, isol, 1], zero, du[:, isol, 2])
            # du(:,isol,2) += K† * du(:,isol,1) using zgbmv with conjugate transpose
            LinearAlgebra.BLAS.gbmv!('C', mpert, mband, mband, one, kmatb, du[:, isol, 1], one, du[:, isol, 2])
            # du(:,isol,1) *= singfac
            du[:, isol, 1] .*= singfac
        end

        # Step 4: Calculate and store u-derivative (ud)
        # ud(:,:,1) = du(:,:,1)
        odet.ud[:, :, 1] .= du[:, :, 1]

        # ud(:,:,2) = -B * du(:,:,1) - C * u(:,:,1)
        for isol in 1:msol
            odet.ud[:, isol, 2] .= -bmat_inv * du[:, isol, 1] .- cmat_inv * u[:, isol, 1]
        end
    end

    return nothing
end
=#

"""
    sing_get_f_det(psifac::Float64) -> ComplexF64

Find determinant of non-Hermitian F matrix.
Subprogram 13 from GPEC code base - converted from Fortran to Julia.

# Arguments

  - `psifac`: Psi factor value at which to evaluate the determinant

# Returns

  - `det`: Complex determinant of the F matrix
"""
function sing_get_f_det!(ffit::FourFitVars, psifac::Float64, intr::DconInternal, equil::Equilibrium.PlasmaEquilibrium, ctrl::DconControl)

    #-----------------------------------------------------------------------
    # Compute q and singfac
    #-----------------------------------------------------------------------
    spline_eval!(equil.sq, psifac, 0)
    q = equil.sq.f[4]
    nn = intr.nlow #Choosing one for now but eventually going to need multi-n support here
    nq = nn * q
    singfac = [intr.mlow - nn*q + ipert for ipert in 0:(intr.mpert-1)]
    chi1 = twopi * equil.psio

    #-----------------------------------------------------------------------
    # Compute F matrix
    #-----------------------------------------------------------------------
    f = zeros(ComplexF64, intr.mpert, intr.mpert)

    if ctrl.kin_flag
        if intr.fkg_kmats_flag
            # Evaluate splines
            cspline_eval!(ffit.f0mats, psifac, 0)
            cspline_eval!(ffit.pmats, psifac, 0)
            cspline_eval!(ffit.paats, psifac, 0)
            cspline_eval!(ffit.r1mats, psifac, 0)

            f0mat = reshape(ffit.f0mats.f, intr.mpert, intr.mpert)
            pmat = reshape(ffit.pmats.f, intr.mpert, intr.mpert)
            paat = reshape(ffit.paats.f, intr.mpert, intr.mpert)
            r1mat = reshape(ffit.r1mats.f, intr.mpert, intr.mpert)
        else
            # Evaluate splines
            cspline_eval!(ffit.amats, psifac, 0)
            cspline_eval!(ffit.dbats, psifac, 0)
            cspline_eval!(ffit.fbats, psifac, 0)

            amat = reshape(ffit.amats.f, intr.mpert, intr.mpert)
            dbat = reshape(ffit.dbats.f, intr.mpert, intr.mpert)
            fmat = reshape(ffit.fbats.f, intr.mpert, intr.mpert)

            kwmat = zeros(ComplexF64, intr.mpert, intr.mpert, 4)
            ktmat = zeros(ComplexF64, intr.mpert, intr.mpert, 4)

            for i in 1:4
                cspline_eval!(ffit.kwmats[i], psifac, 0)
                cspline_eval!(ffit.ktmats[i], psifac, 0)
                kwmat[:, :, i] = reshape(ffit.kwmats[i].f, intr.mpert, intr.mpert)
                ktmat[:, :, i] = reshape(ffit.ktmats[i].f, intr.mpert, intr.mpert)
            end

            amat = amat + kwmat[:, :, 1] + ktmat[:, :, 1]
            b1mat = ifac * dbat

            #-----------------------------------------------------------------------
            # Factor kinetic non-Hermitian matrix A using banded storage
            #-----------------------------------------------------------------------
            # Convert to banded storage format (LAPACK style)
            amatlu = zeros(ComplexF64, 3*intr.mband+1, intr.mpert)
            umat = Matrix{ComplexF64}(I, intr.mpert, intr.mpert)

            for jpert in 1:intr.mpert
                for ipert in 1:intr.mpert
                    # Band storage: row index is (2*mband+1+ipert-jpert)
                    amatlu[2*intr.mband+1+ipert-jpert, jpert] = amat[ipert, jpert]
                end
            end

            # LU factorization of banded matrix
            ipiv, info = LAPACK.gbtrf!(intr.mband, intr.mband, amatlu)

            if info != 0
                error("gbtrf: amat singular at psifac = $psifac, ipert = $info, reduce delta_mband")
            end

            # Solve systems using banded LU
            temp1 = copy(dbat)
            LAPACK.gbtrs!('N', intr.mband, intr.mband, amatlu, ipiv, temp1)
            f0mat = fmat - adjoint(dbat) * temp1

            temp2 = copy(amat)
            LAPACK.gbtrs!('C', intr.mband, intr.mband, amatlu, ipiv, temp2)
            aamat = adjoint(temp2)
            umat = umat - aamat

            bkmat = kwmat[:, :, 2] + ktmat[:, :, 2] +
                    ifac * chi1 / (twopi*nn) * (kwmat[:, :, 1] + ktmat[:, :, 1])
            bkaat = kwmat[:, :, 2] - ktmat[:, :, 2] +
                    ifac * chi1 / (twopi*nn) * (kwmat[:, :, 1] + ktmat[:, :, 1])

            temp2 = copy(bkmat)
            LAPACK.gbtrs!('N', intr.mband, intr.mband, amatlu, ipiv, temp2)
            pmat = adjoint(b1mat) * temp2

            temp2 = copy(b1mat)
            LAPACK.gbtrs!('N', intr.mband, intr.mband, amatlu, ipiv, temp2)
            paat = adjoint(bkaat) * temp2 - ifac * chi1 / (twopi*nn) * umat * b1mat
            paat = adjoint(paat)

            temp1 = kwmat[:, :, 1] + ktmat[:, :, 1]
            temp2 = copy(bkmat)
            LAPACK.gbtrs!('N', intr.mband, intr.mband, amatlu, ipiv, temp2)

            r1mat =
                kwmat[:, :, 4] + ktmat[:, :, 4] -
                (chi1 / (twopi*nn))^2 * adjoint(temp1) +
                ifac * chi1 / (twopi*nn) * adjoint(bkaat) -
                ifac * chi1 / (twopi*nn) * aamat * bkmat -
                adjoint(bkaat) * temp2
        end

        # Construct F matrix
        for ipert in 1:intr.mpert
            m1 = intr.mlow + ipert - 1
            singfac1 = m1 - nn * q
            for jpert in 1:intr.mpert
                m2 = intr.mlow + jpert - 1
                singfac2 = m2 - nn * q
                f[ipert, jpert] = singfac1 * f0mat[ipert, jpert] * singfac2 -
                                  singfac1 * pmat[ipert, jpert] -
                                  conj(paat[jpert, ipert]) * singfac2 +
                                  r1mat[ipert, jpert]
            end
        end
    else
        # Non-kinetic case (Hermitian)
        cspline_eval!(ffit.amats, psifac, 0)
        cspline_eval!(ffit.dbats, psifac, 0)
        cspline_eval!(ffit.fbats, psifac, 0)

        amat = reshape(ffit.amats.f, intr.mpert, intr.mpert)
        dbat = reshape(ffit.dbats.f, intr.mpert, intr.mpert)
        fmat = reshape(ffit.fbats.f, intr.mpert, intr.mpert)

        # Hermitian factorization (Bunch-Kaufman)
        amat_copy = copy(amat)
        ipiv, info = LAPACK.hetrf!('L', amat_copy)

        if info != 0
            error("hetrf: amat singular at psifac = $psifac, ipert = $info, increase delta_mband")
        end

        temp1 = copy(dbat)
        LAPACK.hetrs!('L', amat_copy, ipiv, temp1)
        fmat = fmat - adjoint(dbat) * temp1

        # Construct F matrix
        for ipert in 1:intr.mpert
            m1 = intr.mlow + ipert - 1
            singfac1 = m1 - nn * q
            for jpert in 1:intr.mpert
                m2 = intr.mlow + jpert - 1
                singfac2 = m2 - nn * q
                f[ipert, jpert] = singfac1 * fmat[ipert, jpert] * singfac2
            end
        end
    end

    #-----------------------------------------------------------------------
    # Convert F to banded storage and compute LU factorization
    #-----------------------------------------------------------------------
    kl = intr.mpert - 1
    ku = intr.mpert - 1
    ldab = 2*kl + ku + 1
    m = intr.mpert
    n = intr.mpert

    lumat = zeros(ComplexF64, ldab, n)

    for jpert in 1:intr.mpert
        for ipert in 1:intr.mpert
            lumat[kl+ku+1+ipert-jpert, jpert] = f[ipert, jpert]
        end
    end

    fpiv, info = LAPACK.gbtrf!(kl, ku, lumat)

    if info != 0
        println("gbtrf info = ", info)
        error("Termination by galerkin_solve_equation")
    end

    #-----------------------------------------------------------------------
    # Calculate the determinant
    #-----------------------------------------------------------------------
    d = 1.0
    for i in 1:m
        if fpiv[i] != i
            d = -d
        end
    end

    det = prod(lumat[kl+ku+1, :]) * d

    return det
end



"""
    ksing_find(ctrl::DconControl, intr::DconInternal, odet::OdeState,
               ffit::FourFitVars, equil::Equilibrium.PlasmaEquilibrium)

Find new singular surfaces in plasma physics simulations.
Subprogram 14 from GPEC code base - converted from Fortran to Julia.
"""
function ksing_find(ctrl::DconControl, intr::DconInternal, odet::OdeState, ffit::FourFitVars,
    equil::Equilibrium.PlasmaEquilibrium; debug::Bool=false)
    # Parameters
    #TODO: these are probably things we want to pass in --> looks like they are often defined in the vac.in files
    tol = 1e-3
    keps1 = 1e-10
    keps2 = 1e-4
    nsing = 1000
    maxstep = 100000

    # Arrays
    psising = fill(-1.0, nsing)
    psising_check = fill(-1.0, nsing)
    tmp_record = zeros(ComplexF64, 2, maxstep)

    # Initialization
    println("Finding kinetically displaced singular surfaces")

    singnum = 0
    i_recur = 0
    i_depth = 0
    i_record = 0
    x0 = equil.config.control.psilow  #We should probably put this somewhere else
    x1 = ctrl.psilim

    #-----------------------------------------------------------------------
    # Adaptively search the singular point
    #-----------------------------------------------------------------------
    odet.sing_flag = false
    det0 = sing_get_f_det!(ffit, x0, intr, equil, ctrl)
    det1 = sing_get_f_det!(ffit, x1, intr, equil, ctrl)

    det_max = abs(det0) > abs(det1) ? det0 : det1

    singnum += 1
    psising[singnum] = x0
    sing_det = det0
    odet.sing_flag = true

    #= TODO: We are getting rid of all this writng stuff, right?
    # Open output files
    open("dcon_detf.out", "w") do io
        println(io, rpad("psi", 16), rpad("absdetF", 16),
                rpad("real(detF)", 16), rpad("imag(detF)", 16))
    end

    # Binary file handling would use Julia's serialization or HDF5
    bin_unit = bin_open("dcon_detf.bin", "w")

    =#

    #TODO: convert
    adp_find_sing!(x0, x1, det_max, det0, det1, psising,
        Ref(singnum), Ref(i_recur), Ref(i_depth), Ref(i_record),
        tol, Ref(sing_det), Ref(odet.sing_flag), tmp_record,
        ffit, equil, intr, ctrl)

    # bin_close(bin_unit)

    # Allocate and copy data
    sing_detf = tmp_record[:, 1:i_record]

    # Adjust boundaries
    if psising[1] > psilow
        psising[2:(singnum+1)] = psising[1:singnum]
        psising[1] = psilow
        singnum += 1
    end

    if psising[singnum] < psilim
        singnum += 1
        psising[singnum] = psilim
    end

    #-----------------------------------------------------------------------
    # Newton method to find accurate local minimum points
    #-----------------------------------------------------------------------
    # Create a closure that captures the necessary parameters for sing_get_f_det
    det_func(psi) = sing_get_f_det!(ffit, psi, intr, equil, ctrl)

    for i in 2:(singnum-1)
        x1 = psising[i]
        x1 = sing_newton(det_func, x1, psising[i-1], psising[i+1]) #TODO: convert this function and what is going on with the nested function call
        det0 = sing_get_f_det!(ffit, psising[i], intr, equil, ctrl)
        det1 = sing_get_f_det!(ffit, x1, intr, equil, ctrl)

        if abs(det0) > abs(det1)
            psising[i] = x1
        end
    end

    #-----------------------------------------------------------------------
    # Check the singular points
    #-----------------------------------------------------------------------
    singnum_check = singnum
    psising_check[:] = psising
    psising = fill(-1.0, nsing)
    singnum = 1

    if ctrl.verbose
        println("Looking for singularities below ", scientific(keps1),
            "x the maximum determinant of ", scientific(abs(det_max)))
    end

    psising[1] = psising_check[1]

    for i in 2:(singnum_check-1)
        det0 = sing_get_f_det!(ffit, psising_check[i], intr, equil, ctrl)
        reps = keps1 / keps2
        eps = keps2 * reps * 10^(psising_check[i] / log10(reps))

        if abs(det0) <= abs(det_max) * eps
            singnum += 1
            psising[singnum] = psising_check[i]
            if debug
                println("  > psi ", scientific(psising_check[i]),
                    " is singular")
            end
        else
            if debug
                println("  - psi ", scientific(psising_check[i]),
                    " is not singular. Determinant is ",
                    scientific(abs(det0) / (abs(det_max) * eps)),
                    "x the threshold")
            end
        end
    end

    singnum += 1
    psising[singnum] = psising_check[singnum_check]

    # Write output file
    open("sing_find.out", "w") do io
        println(io, rpad("psi", 16), rpad("real(det)", 16),
            rpad("imag(det)", 16))
        for i in 1:singnum
            det0 = sing_get_f_det!(ffit, psising[i], intr, equil, ctrl)
            println(io, rpad(string(psising[i]), 16),
                rpad(string(real(det0)), 16),
                rpad(string(imag(det0)), 16))
        end
    end

    # Create kinetic singular surface structures
    kmsing = singnum - 2
    kinsing = Vector{KineticSingular}(undef, kmsing)

    for ising in 1:kmsing
        kinsing[ising] = KineticSingular(;
            m=ising,
            psifac=psising[ising+1],
            rho=sqrt(psising[ising+1]),
            q=0.0,      # Will be set below
            q1=0.0      # Will be set below
        )

        # Evaluate spline
        spline_eval!(equil.sq, psising[ising+1], 1)
        #spline_result = spline_eval(equil.sq, psising[ising+1], 1)
        kinsing[ising].q = equil.sq.f[4] #spline_result.f[4]
        kinsing[ising].q1 = equil.sq.f1[4] #spline_result.f1[4]
    end

    # Print results
    if ctrl.verbose
        if kmsing > 0
            println("  > Found kinetic singular surfaces:")
            println("   ", rpad("psi", 16), rpad("q", 16))
            for ising in 1:kmsing
                println("   ", rpad(scientific(kinsing[ising].psifac), 16),
                    rpad(scientific(kinsing[ising].q), 16))
            end
        else
            println("  > Found no kinetic singular surfaces")
        end
    end

    return kinsing, singnum
end


"""
    adp_find_sing!(x0, x1, det0, det1, det_max, singpos, singnum, i_recur, i_depth,
                        i_record, tol, sing_det, sing_flag, record)
                        , det_max, sing_get_f_det, bin_unit)

Recursive adaptive finder for singular surfaces in plasma equilibrium.

Uses adaptive grid refinement to locate singularities (resonant surfaces) by
monitoring the gradient of the determinant function.

# Arguments

  - `x0::Float64`: Left boundary of search interval
  - `x1::Float64`: Right boundary of search interval
  - `det_max::Ref{ComplexF64}`: Maximum determinant encountered (modified)
  - `det0::ComplexF64`: Determinant value at x0
  - `det1::ComplexF64`: Determinant value at x1
  - `singpos::Vector{Float64}`: Array to store singular positions (modified)
  - `singnum::Ref{Int}`: Current number of singularities found (modified)
  - `i_recur::Ref{Int}`: Recursion counter (modified)
  - `i_depth::Ref{Int}`: Current recursion depth (modified)
  - `i_record::Ref{Int}`: Record counter (modified)
  - `tol::Float64`: Tolerance for grid partition criterion
  - `sing_det::Ref{ComplexF64}`: Best singular determinant value (modified)
  - `sing_flag::Ref{Bool}`: Flag indicating if inside singular region (modified)
  - `record::Matrix{ComplexF64}`: Record of (x, det) pairs (modified)
  - `sing_get_f_det::Function`: Function to compute determinant at a point
  - `bin_unit::IO`: Binary output file handle (optional)

# Algorithm

 1. Subdivides interval adaptively based on linearity test
 2. Identifies singularities by detecting local minima in |det|
 3. Tracks the sharpest singular point in each singular region
"""
function adp_find_sing!(x0::Float64, x1::Float64,
    det_max::ComplexF64, det0::ComplexF64, det1::ComplexF64,
    singpos::Vector{Float64},
    singnum::Ref{Int},
    i_recur::Ref{Int},
    i_depth::Ref{Int},
    i_record::Ref{Int},
    tol::Float64,
    sing_det::Ref{ComplexF64},
    sing_flag::Ref{Bool},
    record::Matrix{ComplexF64},
    ffit::FourFitVars,
    equil::Equilibrium.PlasmaEquilibrium,
    intr::DconInternal,
    ctrl::DconControl) # sing_get_f_det::Function, bin_unit::Union{IO,Nothing}=nothing)

    grid_tol = 1e-6

    # Increment depth and recursion counters
    i_depth[] += 1
    i_recur[] += 1

    # Set up 3-point stencil
    x = [x0, 0.5 * (x0 + x1), x1]
    det = ComplexF64[det0, sing_get_f_det!(ffit, x[2], intr, equil, ctrl), det1]

    # Track maximum determinant
    if abs(det[2]) > abs(det_max)
        det_max = det[2]
    end

    # Criteria for grid partition (linearity test)
    tmp1 = abs(det[1] + det[3])
    tmpm = abs(det[2]) * 2

    if abs(tmpm - tmp1) > tol * tmp1 && (x[3] - x[1]) > grid_tol
        # Grid is not linear enough - subdivide further
        adp_find_sing!(x[1], x[2], det_max, det[1], det[2],
            singpos, singnum, i_recur, i_depth, i_record,
            tol, sing_det, sing_flag, record,
            ffit, equil, intr, ctrl)

        adp_find_sing!(x[2], x[3], det_max, det[2], det[3],
            singpos, singnum, i_recur, i_depth, i_record,
            tol, sing_det, sing_flag, record,
            ffit, equil, intr, ctrl)
    else
        # Grid is linear enough - judge singularity with gradient of |det|
        tmp1 = abs(det[2]) - abs(det[1])
        tmp2 = abs(det[3]) - abs(det[2])

        # Case 1: tmp1 < 0 AND tmp2 < 0 (descending on both sides - peak before x[3])
        if tmp1 < 0 && tmp2 < 0
            if sing_flag[]
                # Already in singular region - update if this is sharper
                if abs(sing_det[]) > abs(det[3])
                    sing_det[] = det[3]
                    singpos[singnum[]] = x[3]
                end
            else
                # Entering new singular region
                singnum[] += 1
                if singnum[] + 3 > length(singpos)
                    error("Increase singpos array size")
                end
                singpos[singnum[]] = x[3]
                sing_det[] = det[3]
                sing_flag[] = true
            end
        end

        # Case 2: tmp1 < 0 AND tmp2 > 0 (local minimum at x[2])
        if tmp1 < 0 && tmp2 > 0
            if sing_flag[]
                # Already in singular region - update if this is sharper
                if abs(sing_det[]) > abs(det[2])
                    sing_det[] = det[2]
                    singpos[singnum[]] = x[2]
                end
            else
                # Entering new singular region
                singnum[] += 1
                if singnum[] + 3 > length(singpos)
                    error("Increase singpos array size")
                end
                singpos[singnum[]] = x[2]
                sing_det[] = det[2]
                sing_flag[] = true
            end
        end

        # Case 3: tmp1 > 0 AND tmp2 > 0 (ascending on both sides)
        if tmp1 > 0 && tmp2 > 0
            if sing_flag[]
                # Exiting singular region
                sing_flag[] = false
                # Update to sharpest point seen
                if abs(sing_det[]) > abs(det[1]) && singnum[] > 0
                    sing_det[] = det[1]
                    singpos[singnum[]] = x[1]
                end
            end
        end

        # Case 4: tmp1 > 0 AND tmp2 < 0 (ascending then descending - peak at x[2])
        if tmp1 > 0 && tmp2 < 0
            if sing_flag[]
                # Exiting singular region
                sing_flag[] = false
                # Update to sharpest point seen
                if abs(sing_det[]) > abs(det[1]) && singnum[] > 0
                    sing_det[] = det[1]
                    singpos[singnum[]] = x[1]
                end
            end
        end

        # Error check for exact zeros
        if tmp1 == 0 || tmp2 == 0
            error("det(2)-det(1)=0 or det(3)-det(2)=0")
        end

        #=
        # Write records to file (ASCII)
        open("singularity_search.out", "a") do f
            @printf(f, " %16.8e %16.8e %16.8e %16.8e\n",
                   x[2], abs(det[2]), real(det[2]), imag(det[2]))
            @printf(f, " %16.8e %16.8e %16.8e %16.8e\n",
                   x[3], abs(det[3]), real(det[3]), imag(det[3]))
        end
        =#
        #= TODO: Getting rid of binary output?
        # Write binary records if unit provided
        if bin_unit !== nothing
            write(bin_unit, Float32(x[2]), Float32(log10(abs(det[2]))),
                  Float32(real(det[2])), Float32(imag(det[2])))
            write(bin_unit, Float32(x[3]), Float32(log10(abs(det[3]))),
                  Float32(real(det[3])), Float32(imag(det[3])))
        end
        =#

        # Store record in memory (with bounds checking)
        if i_record[] < size(record, 2)
            i_record[] += 1
            record[:, i_record[]] = [ComplexF64(x[2]), det[2]]
        end
        if i_record[] < size(record, 2)
            i_record[] += 1
            record[:, i_record[]] = [ComplexF64(x[3]), det[3]]
        end
    end

    i_depth[] -= 1

    return nothing
end


"""
    sing_newton!(ff, z, bo0, bo1)

Newton iteration for singular surface finder.

Uses a modified Newton's method with adaptive step sizing to find local minima
of |ff(z)| within bounds. Includes safeguards against overshooting and climbing
out of sharp local wells.

# Arguments

  - `ff::Function`: Function returning complex determinant at position z
  - `z::Ref{Float64}`: Initial guess (modified to final position)
  - `bo0::Float64`: Lower bound estimate of neighboring minimum
  - `bo1::Float64`: Upper bound estimate of neighboring minimum

# Algorithm

 1. Sets conservative bounds inside estimated neighboring minima
 2. Uses modified Newton iteration with adaptive step control
 3. Tracks optimal position throughout iteration
 4. Includes safeguards for sharp local wells and peaks

# Returns

Modifies `z[]` in place to contain the position of the local minimum.
"""
function sing_newton!(ff::Function, z::Ref{Float64}, bo0::Float64, bo1::Float64)

    # Parameters
    dzfac = 1e-6
    dbfac = 1e-1
    tol = 1e-15
    itmax = 1000

    # Find initial guess - bounds well inside of estimated neighboring minima
    b0 = z[] - (z[] - bo0) * dbfac
    b1 = z[] + (bo1 - z[]) * dbfac

    f = abs(ff(z[]))
    zopt = z[]
    fopt = f

    # First step is a fraction of a half step towards the nearer boundary
    dz1 = (b0 + z[]) * 0.5 - z[]
    dz2 = (b1 + z[]) * 0.5 - z[]
    dz = dz1 * dzfac
    if abs(dz2) < abs(dz1)
        dz = dz2 * dzfac
    end

    it = 0

    # Iterate
    while true #TODO--> this is dangerous- do we want to do it this way really?
        it += 1
        err = abs(dz / z[])

        # Check convergence
        if err < tol
            z[] = zopt
            break
        end

        # Check iteration limit
        if it > itmax
            it = -1
            @warn @sprintf("  - search terminated at %.3e with large %.3e error",
                zopt, err)
            z[] = zopt
            break
        end

        # Check if step would go outside bounds
        if z[] + dz <= b0 || z[] + dz >= b1
            # We've climbed out of the sharp local well
            # Case 1: we are on the right side, but near a peak so the
            #         ~0 gradient way overshoots to the other side
            # Case 2: we are already on the other side of a peak and
            #         falling down towards the neighboring minimum
            dz *= 0.5
        else
            # Take Newton step
            z_old = z[]
            z[] = z[] + dz
            f_old = f
            f = abs(ff(z[]))

            # Track optimal position
            if f < fopt
                fopt = f
                zopt = z[]
            end

            # Compute new Newton step
            dz = -f * (z[] - z_old) / (f - f_old)
        end
    end

    # Final check for better position
    if f < fopt
        fopt = f
        zopt = z[]
    end

    return nothing
end
