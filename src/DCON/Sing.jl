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

Some of the kinetic part is a little sketchy
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

    ifac = 1im # Imaginary unit factor
    chi1 = 2π * equil.psio
    nn = intr.nlow # Later for multiple surfaces we'll have to loop over nn

    # Compute singfac = 1 / (m - nq)
    odet.q = Spl.spline_eval!(equil.sq, psieval)[4]
    odet.singfac_vec .= vec(1.0 ./ ((intr.mlow:intr.mhigh) .- odet.q .* (intr.nlow:intr.nhigh)'))

    # Evaluate matrix splines at the current psi value
    if ctrl.kin_flag
        # Evaluate ideal MHD matrix splines at the current psi value
        Spl.spline_eval!(odet.amat, ffit.amats, psieval)
        Spl.spline_eval!(odet.bmat, ffit.bmats, psieval)
        Spl.spline_eval!(odet.cmat, ffit.cmats, psieval)
        Spl.spline_eval!(odet.dmat, ffit.dmats, psieval)
        Spl.spline_eval!(odet.fmat_lower, ffit.fmats_lower, psieval)

        # Evaluate kinetic matrices into OdeState workspace
        Spl.spline_eval!(odet.emat, ffit.emats, psieval)
        Spl.spline_eval!(odet.hmat, ffit.hmats, psieval)
        Spl.spline_eval!(odet.dbat, ffit.dmats, psieval) #TODO: Verify this is correct
        Spl.spline_eval!(odet.ebat, ffit.emats, psieval) #TODO: Verify this is correct

        # if kwmat and ktmat not allocated or wrong size, allocate them
        if size(odet.kwmat, 1) != intr.numpert_total || size(odet.kwmat, 2) != intr.numpert_total
            odet.kwmat = zeros(ComplexF64, intr.numpert_total, intr.numpert_total, 6)
            odet.ktmat = zeros(ComplexF64, intr.numpert_total, intr.numpert_total, 6)
        end
        for i in 1:6
            kwmat_temp = zeros(ComplexF64, intr.numpert_total^2)
            ktmat_temp = zeros(ComplexF64, intr.numpert_total^2)
            Spl.spline_eval!(kwmat_temp, ffit.kwmats[i], psieval)
            Spl.spline_eval!(ktmat_temp, ffit.ktmats[i], psieval)
            odet.kwmat[:, :, i] .= reshape(kwmat_temp, intr.numpert_total, intr.numpert_total)
            odet.ktmat[:, :, i] .= reshape(ktmat_temp, intr.numpert_total, intr.numpert_total)
        end

        # Evaluate kinetic matrices
        if intr.fkg_kmats_flag
            # Make local storage for kinetic matrices- TODO: should these be included in a struct? I don't think they have to be passed around
            f0mat = Vector{ComplexF64}(undef, intr.numpert_total^2)
            pmat = Vector{ComplexF64}(undef, intr.numpert_total^2)
            paat = Vector{ComplexF64}(undef, intr.numpert_total^2)
            kkmat = Vector{ComplexF64}(undef, intr.numpert_total^2)
            kkaat = Vector{ComplexF64}(undef, intr.numpert_total^2)
            r1mat = Vector{ComplexF64}(undef, intr.numpert_total^2)
            r2mat = Vector{ComplexF64}(undef, intr.numpert_total^2)
            r3mat = Vector{ComplexF64}(undef, intr.numpert_total^2)
            gaat = Vector{ComplexF64}(undef, intr.numpert_total^2)

            Spl.spline_eval!(odet.amat, ffit.akmats, psieval)
            Spl.spline_eval!(odet.bmat, ffit.bkmats, psieval)
            Spl.spline_eval!(odet.cmat, ffit.ckmats, psieval)
            Spl.spline_eval!(f0mat, ffit.f0mats, psieval)
            Spl.spline_eval!(pmat, ffit.pmats, psieval)
            Spl.spline_eval!(paat, ffit.paats, psieval)
            Spl.spline_eval!(kkmat, ffit.kkmats, psieval)
            Spl.spline_eval!(kkaat, ffit.kkaats, psieval)
            Spl.spline_eval!(r1mat, ffit.r1mats, psieval)
            Spl.spline_eval!(r2mat, ffit.r2mats, psieval)
            Spl.spline_eval!(r3mat, ffit.r3mats, psieval)
            Spl.spline_eval!(gaat, ffit.gaats, psieval)

            # Initialize the banded matrix storage
            amatlu = zeros(ComplexF64, 3*intr.mband+1, intr.numpert_total)
            amat_reshaped = reshape(odet.amat, intr.numpert_total, intr.numpert_total)

            # Fill the banded matrix storage
            for jpert in 1:intr.numpert_total
                for ipert in 1:intr.numpert_total
                    amatlu[2*intr.mband+1+ipert-jpert, jpert] = amat_reshaped[ipert, jpert]
                end
            end

            # Perform LU factorization
            amatlu_fact, ipiv = LAPACK.gbtrf!(intr.mband, intr.mband, intr.numpert_total, amatlu)
            amatlu .= amatlu_fact

            # gbtrf! modifies amatlu in-place and returns (ab_modified, ipiv)

            # Precompute cmat_solved for later use
            cmat_mat = reshape(odet.cmat, intr.numpert_total, intr.numpert_total)
            cmat_solved = zeros(ComplexF64, intr.numpert_total, intr.numpert_total)
            for j in 1:intr.numpert_total
                temp_col = copy(cmat_mat[:, j])
                LAPACK.gbtrs!('N', intr.mband, intr.mband, intr.numpert_total, amatlu, ipiv, temp_col)
                cmat_solved[:, j] .= temp_col
            end
        else #TODO: This may be very inefficient- we need to figure out if this is necessary
            # Kinetic matrix computation using OdeState workspace
            # Note: baat, caat, eaat are computed as matrices for operations
            odet.amat .=
                odet.amat .+ vec(reshape(odet.kwmat[:, :, 1], intr.numpert_total, intr.numpert_total) .+ reshape(odet.ktmat[:, :, 1], intr.numpert_total, intr.numpert_total))
            odet.bmat .=
                odet.bmat .+ vec(reshape(odet.kwmat[:, :, 2], intr.numpert_total, intr.numpert_total) .+ reshape(odet.ktmat[:, :, 2], intr.numpert_total, intr.numpert_total))
            odet.cmat .=
                odet.cmat .+ vec(reshape(odet.kwmat[:, :, 3], intr.numpert_total, intr.numpert_total) .+ reshape(odet.ktmat[:, :, 3], intr.numpert_total, intr.numpert_total))
            odet.dmat .=
                odet.dmat .+ vec(reshape(odet.kwmat[:, :, 4], intr.numpert_total, intr.numpert_total) .+ reshape(odet.ktmat[:, :, 4], intr.numpert_total, intr.numpert_total))
            odet.emat .=
                odet.emat .+ vec(reshape(odet.kwmat[:, :, 5], intr.numpert_total, intr.numpert_total) .+ reshape(odet.ktmat[:, :, 5], intr.numpert_total, intr.numpert_total))
            odet.hmat .=
                odet.hmat .+ vec(reshape(odet.kwmat[:, :, 6], intr.numpert_total, intr.numpert_total) .+ reshape(odet.ktmat[:, :, 6], intr.numpert_total, intr.numpert_total))
            baat = reshape(odet.bmat, intr.numpert_total, intr.numpert_total) .- 2 .* odet.ktmat[:, :, 2]
            caat = reshape(odet.cmat, intr.numpert_total, intr.numpert_total) .- 2 .* odet.ktmat[:, :, 3]
            eaat = reshape(odet.emat, intr.numpert_total, intr.numpert_total) - 2*odet.ktmat[:, :, 5]
            odet.b1mat = ifac * odet.dbat

            # Initialize the banded matrix storage
            amatlu = zeros(ComplexF64, 3*intr.mband+1, intr.numpert_total)
            umat = zeros(ComplexF64, intr.numpert_total, intr.numpert_total)

            # Convert amat vector to banded matrix format
            amat_reshaped = reshape(odet.amat, intr.numpert_total, intr.numpert_total)
            for jpert in 1:intr.numpert_total
                for ipert in 1:intr.numpert_total
                    amatlu[2*intr.mband+1+ipert-jpert, jpert] = amat_reshaped[ipert, jpert]
                    if ipert == jpert
                        umat[ipert, jpert] = 1.0
                    end
                end
            end

            # Perform LU factorization
            amatlu_fact, ipiv = LAPACK.gbtrf!(intr.mband, intr.mband, intr.numpert_total, amatlu)
            amatlu .= amatlu_fact  # Copy factorized matrix back

            # gbtrf! returns (ab_modified, ipiv)

            # Compute f0mat = fmat - conj(transpose(dbat)) * A^{-1} * dbat
            # Need to solve A * X = dbat where dbat is a matrix (stored as flat vector)
            dbat_mat = reshape(odet.dbat, intr.numpert_total, intr.numpert_total)
            temp1_mat = zeros(ComplexF64, intr.numpert_total, intr.numpert_total)
            for j in 1:intr.numpert_total
                temp1_col = copy(dbat_mat[:, j])
                LAPACK.gbtrs!('N', intr.mband, intr.mband, intr.numpert_total, amatlu, ipiv, temp1_col)
                temp1_mat[:, j] .= temp1_col
            end
            f0mat_result = reshape(odet.fmat_lower, intr.numpert_total, intr.numpert_total) .- (adjoint(dbat_mat) * temp1_mat)
            odet.f0mat .= reshape(f0mat_result, intr.numpert_total^2)

            # Compute aamat = conj(transpose(A^{-1} * amat))
            # Need to solve A^H * X = amat where amat is a matrix (stored as flat vector)
            amat_mat = reshape(odet.amat, intr.numpert_total, intr.numpert_total)
            temp2_mat = zeros(ComplexF64, intr.numpert_total, intr.numpert_total)
            for j in 1:intr.numpert_total #TODO: Is this loop needed? See line 1005ish of Fortran sing_der
                temp2_col = copy(amat_mat[:, j])
                LAPACK.gbtrs!('C', intr.mband, intr.mband, intr.numpert_total, amatlu, ipiv, temp2_col)
                temp2_mat[:, j] .= temp2_col
            end
            odet.aamat .= reshape(adjoint(temp2_mat), intr.numpert_total^2)
            umat .-= reshape(odet.aamat, intr.numpert_total, intr.numpert_total)

            #TODO: This implementation seems really inefficient compared to Fortran- can we optimize?
            # Compute pmat and paat using bkmat and bkaat
            bkmat_mat = odet.kwmat[:, :, 2] .+ odet.ktmat[:, :, 2] .+ (ifac*chi1/(2π*nn)) .* (odet.kwmat[:, :, 1] .+ odet.ktmat[:, :, 1])
            bkaat_mat = odet.kwmat[:, :, 2] .- odet.ktmat[:, :, 2] .+ (ifac*chi1/(2π*nn)) .* (odet.kwmat[:, :, 1] .+ odet.ktmat[:, :, 1])
            odet.bkmat .= reshape(bkmat_mat, intr.numpert_total^2)
            odet.bkaat .= reshape(bkaat_mat, intr.numpert_total^2)

            # Solve A * X = bkmat column by column
            bkmat_solved = zeros(ComplexF64, intr.numpert_total, intr.numpert_total)
            for j in 1:intr.numpert_total
                temp_col = copy(bkmat_mat[:, j])
                LAPACK.gbtrs!('N', intr.mband, intr.mband, intr.numpert_total, amatlu, ipiv, temp_col)
                bkmat_solved[:, j] .= temp_col
            end
            odet.pmat .= reshape(adjoint(reshape(odet.b1mat, intr.numpert_total, intr.numpert_total)) * bkmat_solved, intr.numpert_total^2)

            # Solve A * X = b1mat column by column
            b1mat_mat = reshape(odet.b1mat, intr.numpert_total, intr.numpert_total)
            b1mat_solved = zeros(ComplexF64, intr.numpert_total, intr.numpert_total)
            for j in 1:intr.numpert_total
                temp_col = copy(b1mat_mat[:, j])
                LAPACK.gbtrs!('N', intr.mband, intr.mband, intr.numpert_total, amatlu, ipiv, temp_col)
                b1mat_solved[:, j] .= temp_col
            end
            paat_mat = adjoint(bkaat_mat) * b1mat_solved .- (ifac*chi1/(2π*nn)) .* (umat * b1mat_mat)
            odet.paat .= reshape(adjoint(paat_mat), intr.numpert_total^2)

            # Compute r1mat
            temp1_mat = odet.kwmat[:, :, 1] .+ odet.ktmat[:, :, 1]
            # Reuse bkmat_solved from previous computation
            r1mat_result =
                odet.kwmat[:, :, 4] .+ odet.ktmat[:, :, 4] .- ((chi1/(2π*nn))^2) .* adjoint(temp1_mat) .+
                (ifac*chi1/(2π*nn)) .* adjoint(bkaat_mat) .-
                (ifac*chi1/(2π*nn)) .* (reshape(odet.aamat, intr.numpert_total, intr.numpert_total) * bkmat_mat) .-
                (adjoint(bkaat_mat) * bkmat_solved)
            odet.r1mat .= reshape(r1mat_result, intr.numpert_total^2)

            # Compute r2mat
            temp1_mat = odet.kwmat[:, :, 5] .+ odet.ktmat[:, :, 5] .- (ifac*chi1/(2π*nn)) .* (odet.kwmat[:, :, 3] .+ odet.ktmat[:, :, 3])
            # Solve A * X = cmat column by column
            cmat_mat = reshape(odet.cmat, intr.numpert_total, intr.numpert_total)
            cmat_solved = zeros(ComplexF64, intr.numpert_total, intr.numpert_total)
            for j in 1:intr.numpert_total
                temp_col = copy(cmat_mat[:, j])
                LAPACK.gbtrs!('N', intr.mband, intr.mband, intr.numpert_total, amatlu, ipiv, temp_col)
                cmat_solved[:, j] .= temp_col
            end
            r2mat_result = temp1_mat .+ (ifac*chi1/(2π*nn)) .* (umat * cmat_mat) .-
                           (adjoint(bkaat_mat) * cmat_solved)
            odet.r2mat .= reshape(r2mat_result, intr.numpert_total^2)

            # Compute r3mat
            temp1_mat = odet.kwmat[:, :, 5] .- odet.ktmat[:, :, 5] .- (ifac*chi1/(2π*nn)) .* (odet.kwmat[:, :, 3] .- odet.ktmat[:, :, 3])
            # Reuse bkmat_solved from previous computation
            r3mat_result = adjoint(temp1_mat) .- (adjoint(caat) * bkmat_solved)
            odet.r3mat .= reshape(r3mat_result, intr.numpert_total^2)

            # Compute kkmat and kkaat
            # Reuse cmat_solved from previous computation
            kkmat_result = reshape(odet.ebat, intr.numpert_total, intr.numpert_total) .- (adjoint(b1mat_mat) * cmat_solved)
            odet.kkmat .= reshape(kkmat_result, intr.numpert_total^2)

            # Reuse b1mat_solved from previous computation
            kkaat_result = adjoint(reshape(odet.ebat, intr.numpert_total, intr.numpert_total)) .- (adjoint(caat) * b1mat_solved)
            odet.kkaat .= reshape(kkaat_result, intr.numpert_total^2)

            # Reuse cmat_solved from previous computation
            gaat_result = reshape(odet.hmat, intr.numpert_total, intr.numpert_total) .- (adjoint(caat) * cmat_solved)
            odet.gaat .= reshape(gaat_result, intr.numpert_total^2)
        end
        # Solve A * X = bmat column by column (overwriting bmat with solution)
        bmat_mat_final = reshape(odet.bmat, intr.numpert_total, intr.numpert_total)
        bmat_solved_final = zeros(ComplexF64, intr.numpert_total, intr.numpert_total)
        for j in 1:intr.numpert_total
            temp_col = copy(bmat_mat_final[:, j])
            LAPACK.gbtrs!('N', intr.mband, intr.mband, intr.numpert_total, amatlu, ipiv, temp_col)
            bmat_solved_final[:, j] .= temp_col
        end
        #TODO: are we missing lines 1076-1079ish after the end of the if statement from the Fortran code?

        odet.bmat .= reshape(bmat_solved_final, intr.numpert_total^2)

        # Solve A * X = cmat column by column (overwriting cmat with solution)
        # Note: cmat_solved from earlier still contains the solution, but we need to update odet.cmat
        odet.cmat .= reshape(cmat_solved, intr.numpert_total^2)

        kaat = zeros(ComplexF64, intr.numpert_total, intr.numpert_total)

        # Reshape flat vectors to matrices for easier indexing
        fmat_lower_mat = reshape(odet.fmat_lower, intr.numpert_total, intr.numpert_total)
        f0mat_mat = reshape(odet.f0mat, intr.numpert_total, intr.numpert_total)
        pmat_mat = reshape(odet.pmat, intr.numpert_total, intr.numpert_total)
        paat_mat = reshape(odet.paat, intr.numpert_total, intr.numpert_total)
        r1mat_mat = reshape(odet.r1mat, intr.numpert_total, intr.numpert_total)
        kkmat_mat = reshape(odet.kkmat, intr.numpert_total, intr.numpert_total)
        r2mat_mat = reshape(odet.r2mat, intr.numpert_total, intr.numpert_total)
        kkaat_mat = reshape(odet.kkaat, intr.numpert_total, intr.numpert_total)
        r3mat_mat = reshape(odet.r3mat, intr.numpert_total, intr.numpert_total)
        kmat_mat = reshape(odet.kmat, intr.numpert_total, intr.numpert_total)

        for ipert in 1:intr.numpert_total
            m1 = intr.mlow + ipert - 1
            singfac1 = m1 - odet.q * nn
            for jpert in 1:intr.numpert_total
                m2 = intr.mlow + jpert - 1
                singfac2 = m2 - odet.q * nn
                fmat_lower_mat[ipert, jpert] =
                    singfac1 * f0mat_mat[ipert, jpert] * singfac2 -
                    singfac1 * pmat_mat[ipert, jpert] -
                    conj(paat_mat[jpert, ipert]) * singfac2 +
                    r1mat_mat[ipert, jpert]
                kmat_mat[ipert, jpert] = singfac1 * kkmat_mat[ipert, jpert] +
                                         r2mat_mat[ipert, jpert]
                kaat[ipert, jpert] = kkaat_mat[ipert, jpert] * singfac2 +
                                     r3mat_mat[ipert, jpert]
            end
        end

        # Write results back to flat vectors
        odet.fmat_lower .= reshape(fmat_lower_mat, intr.numpert_total^2)
        odet.kmat .= reshape(kmat_mat, intr.numpert_total^2)

        # Store kinetic FKG matrices in banded format
        # LAPACK gbtrf! needs (2*kl+ku+1, n) storage for factorization
        fmatlu = zeros(ComplexF64, 3*intr.mband+1, intr.numpert_total)
        for jpert in 1:intr.numpert_total
            for ipert in max(1, jpert-intr.mband):min(intr.numpert_total, jpert+intr.mband)
                fmatlu[2*intr.mband+1+ipert-jpert, jpert] = fmat_lower_mat[ipert, jpert]
            end
        end

        #TODO: Do we need to add something that handles kmat to kmats and kaat to kaats here? (see lines 1131-1134ish in Fortran)

        # Convert kmat and kaat to banded matrix format for LAPACK gbmv
        kmatb = zeros(ComplexF64, 2*intr.mband+1, intr.numpert_total)
        kaatb = zeros(ComplexF64, 2*intr.mband+1, intr.numpert_total)
        for jpert in 1:intr.numpert_total
            for ipert in max(1, jpert-intr.mband):min(intr.numpert_total, jpert+intr.mband)
                kmatb[1+intr.mband+ipert-jpert, jpert] = kmat_mat[ipert, jpert]
                kaatb[1+intr.mband+ipert-jpert, jpert] = kaat[ipert, jpert]
            end
        end

        # Convert gaat to banded storage format for LAPACK gbmv
        gaatb = zeros(ComplexF64, 2*intr.mband+1, intr.numpert_total)
        gaat_mat = reshape(odet.gaat, intr.numpert_total, intr.numpert_total)
        for jpert in 1:intr.numpert_total
            for ipert in max(1, jpert-intr.mband):min(intr.numpert_total, jpert+intr.mband)
                gaatb[1+intr.mband+ipert-jpert, jpert] = gaat_mat[ipert, jpert]
            end
        end
    end

    # Always evaluate and reshape bmat and cmat (needed for ud[:,:,2] computation below)
    Spl.spline_eval!(odet.bmat, ffit.bmats, psieval)
    Spl.spline_eval!(odet.cmat, ffit.cmats, psieval)
    bmat = reshape(odet.bmat, intr.numpert_total, intr.numpert_total)
    cmat = reshape(odet.cmat, intr.numpert_total, intr.numpert_total)

    if ! ctrl.kin_flag
        # Evaluate ideal MHD matrix splines at the current psi value
        Spl.spline_eval!(odet.amat, ffit.amats, psieval)
        Spl.spline_eval!(odet.fmat_lower, ffit.fmats_lower, psieval)
        Spl.spline_eval!(odet.kmat, ffit.kmats, psieval)
        Spl.spline_eval!(odet.gmat, ffit.gmats, psieval)

        # Form full matrices from flat representations
        # TODO: make these block diagonal for multi-n?
        amat = reshape(odet.amat, intr.numpert_total, intr.numpert_total)
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
    if ctrl.kin_flag
        # Kinetic case: du1 = u2 - kmatb * u1
        # Flatten matrix views to vectors for BLAS
        u1_vec = vec(u1)
        u2_vec = vec(u2)
        du1_vec = vec(du1)
        du2_vec = vec(du2)

        du1_vec .= u2_vec
        LinearAlgebra.BLAS.gbmv!('N', intr.numpert_total, intr.mband, intr.mband, ComplexF64(-1.0), kmatb, u1_vec, ComplexF64(1.0), du1_vec)

        # Factor fmatlu for kinetic case
        fmatlu_fact, ipiv_kin = LAPACK.gbtrf!(intr.mband, intr.mband, intr.numpert_total, fmatlu)
        fmatlu .= fmatlu_fact

        # gbtrf! modifies fmatlu in-place and returns (ab_modified, ipiv)

        # Solve for du1: fmatlu * du1 = du1_rhs (column by column)
        du1_mat = reshape(du1_vec, intr.numpert_total, intr.numpert_total)
        for j in 1:intr.numpert_total
            du1_col = copy(du1_mat[:, j])
            LAPACK.gbtrs!('N', intr.mband, intr.mband, intr.numpert_total, fmatlu, ipiv_kin, du1_col)
            du1_mat[:, j] .= du1_col
        end

        # Compute du2 = gaatb * u1 + kaatb * du1
        du2_vec .= 0.0
        LinearAlgebra.BLAS.gbmv!('N', intr.numpert_total, intr.mband, intr.mband, ComplexF64(1.0), gaatb, u1_vec, ComplexF64(0.0), du2_vec)
        LinearAlgebra.BLAS.gbmv!('N', intr.numpert_total, intr.mband, intr.mband, ComplexF64(1.0), kaatb, du1_vec, ComplexF64(1.0), du2_vec)
    else
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
    q = Spl.spline_eval!(equil.sq, psifac)[4]
    nn = intr.nlow #Choosing one for now but eventually going to need multi-n support here
    nq = nn * q
    singfac = [intr.mlow - nn*q + ipert for ipert in 0:(intr.mpert-1)]
    chi1 = 2π * equil.psio
    ifac = 1im

    #-----------------------------------------------------------------------
    # Compute F matrix
    #-----------------------------------------------------------------------
    f = zeros(ComplexF64, intr.mpert, intr.mpert)

    if ctrl.kin_flag
        if intr.fkg_kmats_flag
            # Evaluate splines
            f0mat = reshape(Spl.spline_eval!(ffit.f0mats, psifac), intr.mpert, intr.mpert)
            pmat = reshape(Spl.spline_eval!(ffit.pmats, psifac), intr.mpert, intr.mpert)
            paat = reshape(Spl.spline_eval!(ffit.paats, psifac), intr.mpert, intr.mpert)
            r1mat = reshape(Spl.spline_eval!(ffit.r1mats, psifac), intr.mpert, intr.mpert)
        else
            # Evaluate splines
            amat = reshape(Spl.spline_eval!(ffit.amats, psifac), intr.mpert, intr.mpert)
            dbat = reshape(Spl.spline_eval!(ffit.dmats, psifac), intr.mpert, intr.mpert)
            fmat = reshape(Spl.spline_eval!(ffit.fmats_lower, psifac), intr.mpert, intr.mpert)

            kwmat = zeros(ComplexF64, intr.mpert, intr.mpert, 4)
            ktmat = zeros(ComplexF64, intr.mpert, intr.mpert, 4)

            for i in 1:4
                kwmat[:, :, i] = reshape(Spl.spline_eval!(ffit.kwmats[i], psifac), intr.mpert, intr.mpert)
                ktmat[:, :, i] = reshape(Spl.spline_eval!(ffit.ktmats[i], psifac), intr.mpert, intr.mpert)
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
            amatlu_fact, ipiv = LAPACK.gbtrf!(intr.mband, intr.mband, intr.mpert, amatlu)
            amatlu .= amatlu_fact

            if false
                error("gbtrf: amat singular at psifac = $psifac, ipert = $info, reduce delta_mband")
            end

            # Solve systems using banded LU
            temp1 = copy(dbat)
            LAPACK.gbtrs!('N', intr.mband, intr.mband, intr.mpert, amatlu, ipiv, temp1)
            f0mat = fmat - adjoint(dbat) * temp1

            temp2 = copy(amat)
            LAPACK.gbtrs!('C', intr.mband, intr.mband, intr.mpert, amatlu, ipiv, temp2)
            aamat = adjoint(temp2)
            umat = umat - aamat

            bkmat = kwmat[:, :, 2] + ktmat[:, :, 2] +
                    ifac * chi1 / (2π*nn) * (kwmat[:, :, 1] + ktmat[:, :, 1])
            bkaat = kwmat[:, :, 2] - ktmat[:, :, 2] +
                    ifac * chi1 / (2π*nn) * (kwmat[:, :, 1] + ktmat[:, :, 1])

            temp2 = copy(bkmat)
            LAPACK.gbtrs!('N', intr.mband, intr.mband, intr.mpert, amatlu, ipiv, temp2)
            pmat = adjoint(b1mat) * temp2

            temp2 = copy(b1mat)
            LAPACK.gbtrs!('N', intr.mband, intr.mband, intr.mpert, amatlu, ipiv, temp2)
            paat = adjoint(bkaat) * temp2 - ifac * chi1 / (2π*nn) * umat * b1mat
            paat = adjoint(paat)

            temp1 = kwmat[:, :, 1] + ktmat[:, :, 1]
            temp2 = copy(bkmat)
            LAPACK.gbtrs!('N', intr.mband, intr.mband, intr.mpert, amatlu, ipiv, temp2)

            r1mat =
                kwmat[:, :, 4] + ktmat[:, :, 4] -
                (chi1 / (2π*nn))^2 * adjoint(temp1) +
                ifac * chi1 / (2π*nn) * adjoint(bkaat) -
                ifac * chi1 / (2π*nn) * aamat * bkmat -
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
        amat = reshape(Spl.spline_eval!(ffit.amats, psifac), intr.mpert, intr.mpert)
        dbat = reshape(Spl.spline_eval!(ffit.dmats, psifac), intr.mpert, intr.mpert)
        fmat = reshape(Spl.spline_eval!(ffit.fmats_lower, psifac), intr.mpert, intr.mpert)

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

    lumat_fact, fpiv = LAPACK.gbtrf!(kl, ku, intr.mpert, lumat)
    lumat .= lumat_fact

    if false
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
    x1 = intr.psilim

    #-----------------------------------------------------------------------
    # Adaptively search the singular point
    #-----------------------------------------------------------------------
    sing_flag = Ref(false)
    det0 = sing_get_f_det!(ffit, x0, intr, equil, ctrl)
    det1 = sing_get_f_det!(ffit, x1, intr, equil, ctrl)

    det_max = abs(det0) > abs(det1) ? det0 : det1

    singnum += 1
    psising[singnum] = x0
    sing_det = det0
    sing_flag[] = true

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
        tol, Ref(sing_det), sing_flag, tmp_record,
        ffit, equil, intr, ctrl)

    # bin_close(bin_unit)

    # Allocate and copy data
    sing_detf = tmp_record[:, 1:i_record]

    psilow = equil.config.control.psilow

    # Adjust boundaries
    if psising[1] > psilow
        psising[2:(singnum+1)] = psising[1:singnum]
        psising[1] = psilow
        singnum += 1
    end

    if psising[singnum] < intr.psilim
        singnum += 1
        psising[singnum] = intr.psilim
    end

    #-----------------------------------------------------------------------
    # Newton method to find accurate local minimum points
    #-----------------------------------------------------------------------
    # Create a closure that captures the necessary parameters for sing_get_f_det
    det_func(psi) = sing_get_f_det!(ffit, psi, intr, equil, ctrl)

    for i in 2:(singnum-1)
        x1 = psising[i]
        x1_ref = Ref(x1)
        sing_newton!(det_func, x1_ref, psising[i-1], psising[i+1])
        x1 = x1_ref[]
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
        println("Looking for singularities below ", @sprintf("%.3e", keps1),
            "x the maximum determinant of ", @sprintf("%.3e", abs(det_max)))
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
                println("  > psi ", @sprintf("%.3e", abs(psising_check[i])),
                    " is singular")
            end
        else
            if debug
                println("  - psi ", @sprintf("%.3e", abs(psising_check[i])),
                    " is not singular. Determinant is ",
                    @sprintf("%.3e", abs(abs(det0) / (abs(det_max) * eps))),
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
    intr.kmsing = singnum - 2
    intr.kinsing = Vector{SingType}(undef, intr.kmsing)

    for ising in 1:intr.kmsing
        intr.kinsing[ising] = SingType(;
            m=[ising],
            psifac=psising[ising+1],
            rho=sqrt(psising[ising+1]),
            q=0.0,      # Will be set below
            q1=0.0      # Will be set below
        )

        # Evaluate spline and first derivative
        f, f1 = Spl.spline_deriv1!(equil.sq, psising[ising+1])
        intr.kinsing[ising].q = f[4]
        intr.kinsing[ising].q1 = f1[4]
    end

    # Print results
    if ctrl.verbose
        if intr.kmsing > 0
            println("  > Found kinetic singular surfaces:")
            println("   ", rpad("psi", 16), rpad("q", 16))
            for ising in 1:intr.kmsing
                println("   ", rpad(@sprintf("%.3e", intr.kinsing[ising].psifac), 16),
                    rpad(@sprintf("%.3e", intr.kinsing[ising].q), 16))
            end
        else
            println("  > Found no kinetic singular surfaces")
        end
    end
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
