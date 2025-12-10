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
            converged = false
            for _ in 1:itmax
                psifac = (psi0 + psi1) / 2
                singfac = (m - ctrl.nn * Spl.spline_eval!(equil.sq, psifac)[4]) * dm
                abs(singfac) < 1e-8 && (converged = true; break)
                singfac > 0 ? (psi0 = psifac) : (psi1 = psifac)
            end

            if !converged
                error("Bisection did not converge for m = $m after $itmax iterations.")
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

    # Compute the resonant (r) and nonresonant (n) indices of the shearing transformation matrix R
    # 1 indexes along the N*M dimension, and 2 along the 2*N*M dimension
    # In 2D, see eq. 41 of 2016 Glasser DCON paper
    singp.r1 = [ipert0]
    singp.r2 = [ipert0, ipert0 + intr.mpert]
    singp.n1 = [i for i in 1:intr.mpert if i != ipert0]
    singp.n2 = vcat(singp.n1, [i + intr.mpert for i in singp.n1])

    # Compute Mercier criterion and singular power
    sing_mmat!(intr, ctrl, equil, ffit, ising)
    singp.m0mat = transpose(singp.mmat[singp.r1[1], singp.r2, :, 1])
    # TODO: this is a little odd. di is complex since m0 is complex (but the imaginaries are negligible),
    # its then stored in singp where the type is real, and then converted to complex again for alpha and power.
    # There's gotta be a more clear way to do this? But this is the most faithful to the Fortran
    di = singp.m0mat[1, 1] * singp.m0mat[2, 2] - singp.m0mat[2, 1] * singp.m0mat[1, 2]
    singp.di = real(di)
    singp.alpha = sqrt(-complex(singp.di))

    # This is the parameter α but for all modes - α = 0 for non-resonant modes
    singp.power[ipert0] = -singp.alpha
    singp.power[ipert0+intr.mpert] = singp.alpha

    # Zeroth-order non-resonant solutions
    singp.vmat .= 0
    for ipert in 1:intr.mpert
        singp.vmat[ipert, ipert, 1, 1] = 1
        singp.vmat[ipert, ipert+intr.mpert, 2, 1] = 1
    end

    # Zeroth-order resonant solutions - solve (M₀ - αI)v₀ = 0
    singp.vmat[ipert0, ipert0, 1, 1] = 1
    singp.vmat[ipert0, ipert0+intr.mpert, 1, 1] = 1
    singp.vmat[ipert0, ipert0, 2, 1] = -(singp.m0mat[1, 1] + singp.alpha) / singp.m0mat[1, 2]
    singp.vmat[ipert0, ipert0+intr.mpert, 2, 1] = -(singp.m0mat[1, 1] - singp.alpha) / singp.m0mat[1, 2]
    det =
        conj(singp.vmat[ipert0, ipert0, 1, 1]) * singp.vmat[ipert0, ipert0+intr.mpert, 2, 1] -
        conj(singp.vmat[ipert0, ipert0+intr.mpert, 1, 1]) * singp.vmat[ipert0, ipert0, 2, 1]
    singp.vmat[ipert0, :, :, 1] ./= sqrt(det)

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

    # Evaluate Taylor series coefficients for diagonal matrix Qᵢ = mᵢ - nᵢq(ψ) = [mᵢ - nᵢq, -nᵢq', -nᵢq'', -nᵢq''']
    singfac[:, 1] .= collect(intr.mlow:intr.mhigh) .- ctrl.nn .* q[1]
    singfac[:, 2] .= -ctrl.nn * q[2]
    singfac[:, 3] .= -ctrl.nn * q[3]
    singfac[:, 4] .= -ctrl.nn * q[4]
    # For resonant modes mᵢ - nᵢq(ψ) = [-nᵢq', -nᵢq'', -nᵢq'''] - shift up terms by 1 index
    # Add scaling to account for hardcoding coefficients in computations below
    ipert0 = singp.m - intr.mlow + 1
    singfac[ipert0, 1] = -ctrl.nn * q[2]
    singfac[ipert0, 2] = -ctrl.nn * q[3] / 2
    singfac[ipert0, 3] = -ctrl.nn * q[4] / 3
    singfac[ipert0, 4] = 0

    # This section becomes tricky because we need to reform F = QL̄L̄ᴴQᴴ
    # TODO: this section can absolutely be simplified using some intuition on the coefficients.
    # Possibly could just store an unfactorized F spline? Then logic would just look like K but
    # with an extra singfac/altered coefficients since it would just be F = QF̄Q
    # For now, leaving overly detailed comments to remind so I don't have to work through this again
    # First, compute Taylor series coefficients of QL̄ (but without scaling by 1/n!), so we get binomial coefficients leftover
    # f_lower = QL̄ = [QL̄, QL̄' + Q' L̄, 1/2 (QL̄'' + 2Q' L̄' + QQ'' L̄), 1/6 (QL̄''' + 3Q' L̄'' + 3Q'' L̄' + Q'''L̄), ...] (but without 1/2, 1/6, etc)
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

    # Compute Taylor series coefficients of F = QL̄L̄ᴴQᴴ (lower half only due to indexing) from QL̄ computed above
    # Here, we build in the Taylor series coefficients iteratively (fac1 = (n choose j), fac0 = 1/n!)
    # so the final coefficient is 1/(n-j)!j! as desired (and hardcoded in the K computation below)
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

    # Compute non-Hermitian matrix K = QK̄ Taylor series coefficients
    # K = [QK̄, QK̄' + Q'K̄, QK̄''/2 + Q'K̄' + Q̄''K̄/2, ...]
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

    # Compute Hermitian matrix G (lower half only) Taylor series coefficients
    # G = [G, G', G''/2, G'''/6]
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

    # We will now compute the Taylor series expansion of x = Lv, with L specified in eq. 23 of Glasser 2016
    # Start with the identity matrix (which can be indexed to project onto resonant/nonresonant modes)
    for ipert in 1:intr.mpert
        v[ipert, ipert, 1] = 1
        v[ipert, ipert+intr.mpert, 2] = 1
    end

    # Solve the Taylor expansion according to F * x¹ = v² - K v¹ at each order
    # 0ᵗʰ order: x¹₀ = F⁻¹(v² - K v¹)
    for isol in 1:2*intr.mpert
        @views x[:, isol, 1, 1] .= v[:, isol, 2] .- k[:, :, 1] * v[:, isol, 1]
    end
    @views x[:, :, 1, 1] = UpperTriangular(f0_lower') \ (LowerTriangular(f0_lower) \ x[:, :, 1, 1])

    # Higher-order: ∑Fⱼx¹ₙ₋ⱼ = -Kₙv¹ → x¹ₙ = F₀⁻¹(-∑Fⱼxₙ₋ⱼ - Kₙv¹)
    for i in 1:ctrl.sing_order
        for isol in 1:2*intr.mpert
            for j in 1:i
                @views x[:, isol, 1, i+1] .-= Hermitian(ff_lower[:, :, j+1], :L) * x[:, isol, 1, i-j+1]
            end
            @views x[:, isol, 1, i+1] .-= k[:, :, i+1] * v[:, isol, 1]
        end
        @views x[:, :, 1, i+1] = UpperTriangular(f0_lower') \ (LowerTriangular(f0_lower) \ x[:, :, 1, i+1])
    end

    # Solve x²ₙ = (G - K^†F⁻¹K)v¹ + K^†F⁻¹v² = Gₙv¹ + ∑Kⱼ^† x¹ₙ₋ⱼ at each order
    for i in 0:ctrl.sing_order
        for isol in 1:2*intr.mpert
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
    singp.mmat[r1, r2[1], 1, 1] .+= 0.5
    singp.mmat[r1, r2[2], 2, 1] .-= 0.5
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
    # Solve a = M₀ - (α + k/2)I = ∑Mₗvₖ₋ₗ
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

    # Form full power series solution for v by multiplying by zᵅ (eq. 45 in Glasser 2016)
    pfac = abs(dpsi)^singp.alpha
    ua[:, r2[1], :] ./= pfac
    ua[:, r2[2], :] .*= pfac

    # Apply shearing transformation u = Rv (eq. 41 in Glasser 2016)
    ua[r1, :, 1] ./= sqrtfac
    ua[r1, :, 2] .*= sqrtfac

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
    odet.singfac_vec .= 1.0 ./ (collect(intr.mlow:intr.mhigh) .- ctrl.nn * odet.q )
    chi1 = 2π * equil.psio

    # kinetic stuff - skip for now
    if false #(TODO: kin_flag)
        error("kin_flag not implemented yet")
    else
        # Evaluate splines at psieval and reshape avoiding new allocations
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

    # ud[1] = Ξ'_Ψ
    @views odet.ud[:, :, 1] .= du1
    # ud[2] = Ξ_s = - A⁻¹(B * Ξ'_Ψ - C * Ξ_Ψ), eq. 18 of Glasser 2016
    mul!(odet.tmp, bmat, du1)
    odet.ud[:, :, 2] .= .-odet.tmp
    mul!(odet.tmp, cmat, u1)
    @views odet.ud[:, :, 2] .-= odet.tmp
end