"""
    sing_find!(intr::ForceFreeStatesInternal, equil::Equilibrium.PlasmaEquilibrium)

Locate singular rational q-surfaces (q = m/nn) using a bisection method
between extrema of the q-profile, and store their properties in `intr.sing`.
Performs the same function as `sing_find` in the Fortran code.
"""
function sing_find!(intr::ForceFreeStatesInternal, equil::Equilibrium.PlasmaEquilibrium)

    profiles = equil.profiles

    # Loop over all toroidal mode numbers
    for n in intr.nlow:intr.nhigh
        hint = Ref(1)
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
                psi0 = equil.params.qextrema_psi[iex-1]
                psi1 = equil.params.qextrema_psi[iex]
                psifac = (psi0 + psi1) / 2 # initial guess for bisection

                psifac = find_zero(psi -> m - n * profiles.q_spline(psi; hint=hint), (psi0, psi1), Roots.Brent())

                if any(s -> isapprox(s.q, m / n; atol=1e-8), intr.sing)
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
                        q1=profiles.q_deriv(psifac; hint=hint)
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
    sing_lim!(ctrl::ForceFreeStatesControl, equil::Equilibrium.PlasmaEquilibrium, intr::ForceFreeStatesInternal)

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
function sing_lim!(intr::ForceFreeStatesInternal, ctrl::ForceFreeStatesControl, equil::Equilibrium.PlasmaEquilibrium)

    profiles = equil.profiles

    # Initial guesses based on equilibrium
    intr.qlim = min(equil.params.qmax, ctrl.qhigh) # equilibrium solve only goes up to qmax, so we're capped there
    intr.q1lim = profiles.q_deriv(profiles.xs[end]; hint=Ref(profiles.npts_minus_1))
    intr.psilim = equil.config.psihigh

    # Optionally override qlim based on dmlim
    if ctrl.set_psilim_via_dmlim
        if ctrl.nn_low != ctrl.nn_high
            error("Setting psilim via dmlim is only valid for single n runs (nn_low == nn_high).")
        end
        @info "Setting psilim via dmlim: initial qlim = $(@sprintf("%.3f", intr.qlim)), dmlim = $(@sprintf("%.3f", ctrl.dmlim))"
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
        _, jpsi = findmin(abs.(profiles.q_spline.y .- intr.qlim))
        jpsi = min(jpsi, equil.config.mpsi - 1)

        hint = Ref(jpsi)
        intr.psilim = find_zero(
            (psi -> profiles.q_spline(psi; hint=hint) - intr.qlim,
                psi -> profiles.q_deriv(psi; hint=hint)),
            profiles.xs[jpsi], Roots.Newton()
        )
        intr.q1lim = profiles.q_deriv(intr.psilim)
    end
end

"""
    compute_sing_asymptotics(singp::SingType, ctrl::ForceFreeStatesControl, equil::Equilibrium.PlasmaEquilibrium, ffit::FourFitVars, intr::ForceFreeStatesInternal)

Calculate asymptotic vmat and mmat matrices for a singular surface.
Formerly `sing_vmat!`. Returns a `SingAsymptotics` struct with the computed data instead of
mutating the `SingType` struct. This makes it clear that asymptotics are computed on-demand
for ideal ForceFreeStates and are not inherent properties of the singular surface.

See equations 41-48 in the Glasser Phys. Plasmas 2016 112506 for the mathematical details.

### Arguments

  - `singp::SingType`: Singular surface parameters

### Returns

  - `SingAsymptotics`: Struct containing all asymptotic expansion data
"""
function compute_sing_asymptotics(singp::SingType, ctrl::ForceFreeStatesControl, equil::Equilibrium.PlasmaEquilibrium, ffit::FourFitVars, intr::ForceFreeStatesInternal)

    # Allocations
    vmat = zeros(ComplexF64, intr.numpert_total, 2 * intr.numpert_total, 2, 2 * ctrl.sing_order + 1)
    mmat = zeros(ComplexF64, intr.numpert_total, 2 * intr.numpert_total, 2, 2 * ctrl.sing_order + 3)
    power = zeros(ComplexF64, 2 * intr.numpert_total)

    # Compute the resonant (r) and nonresonant (n) indices of the shearing transformation matrix R
    # 1 indexes along the N*M dimension, and 2 along the 2*N*M dimension
    # In 2D, see eq. 41 of Glasser Phys. Plasmas 2016 112506
    # TODO: if we remove the 3rd dimension, no need for both r1 and r2
    ipert_res = 1 .+ singp.m .- intr.mlow .+ (singp.n .- intr.nlow) .* intr.mpert
    r1 = ipert_res
    r2 = vec([ipert_res[i] + j * intr.numpert_total for j in 0:1, i in eachindex(ipert_res)])
    n1 = [i for i in 1:intr.numpert_total if !(i in ipert_res)]
    n2 = vec([i + j * intr.numpert_total for j in 0:1, i in n1])

    # Compute Mercier criterion and singular power
    compute_sing_mmat!(mmat, singp, ctrl, equil.profiles, ffit, intr)

    # TODO: My approach for the following logic is to mimic the existing code but go block by block
    # in m0mat (i.e. looping through each resonance). I think it works for 2D, probably not 3D
    # Note: We only need the transpose here because the third dimension corresponds to the bottom half of the 2N X 2N matrix
    # If we get rid of the 3rd dimension, this becomes simpler
    m0mat = if length(r1) == 1
        Matrix(transpose(mmat[r1[1], r2, :, 1]))
    else
        Matrix(vcat([transpose(mmat[r1[i], r2, :, 1]) for i in eachindex(r1)]...))
    end

    alpha = eigen(m0mat).values[(length(r1)+1):end] # take the M largest eigenvalues

    # This is the parameter α but for all modes - α = 0 for non-resonant modes
    power[ipert_res] .= -alpha
    power[ipert_res .+ intr.numpert_total] .= alpha

    # Zeroth-order non-resonant solutions
    # TODO: without the third dimension, this is just setting to the identity
    for ipert in 1:intr.numpert_total
        vmat[ipert, ipert, 1, 1] = 1
        vmat[ipert, ipert+intr.numpert_total, 2, 1] = 1
    end

    # Zeroth-order resonant solutions - solve (M₀ - αI)v₀ = 0
    # TODO: this will probably need a better generalization in 3D
    for i in eachindex(r1) # go block by block in M₀
        m0mat_block = m0mat[(2*(i-1)+1):(2*i), (2*(i-1)+1):(2*i)]
        r1_i = r1[i]
        r2_i = r1_i + intr.numpert_total
        alpha_i = alpha[i]
        vmat[r1_i, r1_i, 1, 1] = 1
        vmat[r1_i, r2_i, 1, 1] = 1
        vmat[r1_i, r1_i, 2, 1] = -(m0mat_block[1, 1] + alpha_i) / m0mat_block[1, 2]
        vmat[r1_i, r2_i, 2, 1] = -(m0mat_block[1, 1] - alpha_i) / m0mat_block[1, 2]
        det = conj(vmat[r1_i, r1_i, 1, 1]) * vmat[r1_i, r2_i, 2, 1] -
              conj(vmat[r1_i, r2_i, 1, 1]) * vmat[r1_i, r1_i, 2, 1]
        vmat[r1_i, :, :, 1] ./= sqrt(det)
    end

    # Higher order solutions - need to solve iteratively
    for k in 1:(2*ctrl.sing_order)
        solve_higher_order_vmat!(vmat, mmat, m0mat, alpha, r1, r2, n1, n2, power, intr, k)
    end

    return SingAsymptotics(ctrl.sing_order, alpha, r1, r2, n1, n2, power, vmat, mmat, m0mat)
end

"""
    compute_sing_mmat!(mmat::Array{ComplexF64,4}, singp::SingType, ctrl::ForceFreeStatesControl, profiles::Equilibrium.ProfileSplines, ffit::FourFitVars, intr::ForceFreeStatesInternal)

Calculate asymptotic mmat matrix for a singular surface. Formerly `sing_mmat!`.
Performs the same function as `sing_mmat` in the Fortran code. Main differences are 1-indexing for
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

  - `mmat::Array{ComplexF64,4}`: Output array to store the computed mmat values
  - `singp::SingType`: Singular surface parameters

### TODOs

Check third derivative accuracy in cubic splines or determine if it matters
Better way to unpack the cubic splines
Rename variables to be more intuitive? I don't like ff - maybe f and f_fact instead of f_lower
Add a spline for F directly instead of the lower triangular factorization to avoid complexity?
"""
@with_pool pool function compute_sing_mmat!(
    mmat::Array{ComplexF64,4},
    singp::SingType,
    ctrl::ForceFreeStatesControl,
    profiles::Equilibrium.ProfileSplines,
    ffit::FourFitVars,
    intr::ForceFreeStatesInternal
)

    q_spline = profiles.q_spline
    q_d1 = profiles.q_deriv
    q_d2 = deriv2(q_spline)
    q_d3 = deriv3(q_spline)

    # Initial allocations
    Npert = intr.numpert_total

    singfac = zeros!(pool, Float64, Npert, 4)
    f_lower_interp = zeros!(pool, ComplexF64, Npert, Npert, 4)
    g_interp = zeros!(pool, ComplexF64, Npert, Npert, 4)
    k_interp = zeros!(pool, ComplexF64, Npert, Npert, 4)
    f_lower = zeros!(pool, ComplexF64, Npert, Npert, ctrl.sing_order + 1)
    f0_lower = zeros!(pool, ComplexF64, Npert, Npert)
    ff_lower = zeros!(pool, ComplexF64, Npert, Npert, ctrl.sing_order + 1)
    g_lower = zeros!(pool, ComplexF64, Npert, Npert, ctrl.sing_order + 1)
    k = zeros!(pool, ComplexF64, Npert, Npert, ctrl.sing_order + 1)
    v = zeros!(pool, ComplexF64, Npert, 2 * Npert, 2)
    x = zeros!(pool, ComplexF64, Npert, 2 * Npert, 2, ctrl.sing_order + 1)
    tmp_vec = acquire!(pool, ComplexF64, Npert)

    # Evaluate q spline and its derivatives
    q = (q_spline(singp.psifac),
        q_d1(singp.psifac),
        q_d2(singp.psifac),
        q_d3(singp.psifac))

    # Evaluate fmats_lower and derivatives using series interpolants
    ffit.fmats_lower(vec(@view(f_lower_interp[:, :, 1])), singp.psifac; hint=ffit._hint)
    ffit.fmats_lower(vec(@view(f_lower_interp[:, :, 2])), singp.psifac; deriv=DerivOp(1))
    ffit.fmats_lower(vec(@view(f_lower_interp[:, :, 3])), singp.psifac; deriv=DerivOp(2))
    ffit.fmats_lower(vec(@view(f_lower_interp[:, :, 4])), singp.psifac; deriv=DerivOp(3))

    # Evaluate gmats and derivatives
    ffit.gmats(vec(@view(g_interp[:, :, 1])), singp.psifac; hint=ffit._hint)
    ffit.gmats(vec(@view(g_interp[:, :, 2])), singp.psifac; deriv=DerivOp(1))
    ffit.gmats(vec(@view(g_interp[:, :, 3])), singp.psifac; deriv=DerivOp(2))
    ffit.gmats(vec(@view(g_interp[:, :, 4])), singp.psifac; deriv=DerivOp(3))

    # Evaluate kmats and derivatives
    ffit.kmats(vec(@view(k_interp[:, :, 1])), singp.psifac; hint=ffit._hint)
    ffit.kmats(vec(@view(k_interp[:, :, 2])), singp.psifac; deriv=DerivOp(1))
    ffit.kmats(vec(@view(k_interp[:, :, 3])), singp.psifac; deriv=DerivOp(2))
    ffit.kmats(vec(@view(k_interp[:, :, 4])), singp.psifac; deriv=DerivOp(3))

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
        @views mul!(tmp_vec, k[:, :, 1], v[:, isol, 1])
        @views x[:, isol, 1, 1] .= v[:, isol, 2] .- tmp_vec
    end
    @views x[:, :, 1, 1] = UpperTriangular(f0_lower') \ (LowerTriangular(f0_lower) \ x[:, :, 1, 1])

    # Higher-order: ∑Fⱼx¹ₙ₋ⱼ = -Kₙv¹ → x¹ₙ = F₀⁻¹(-∑Fⱼxₙ₋ⱼ - Kₙv¹)
    for i in 1:ctrl.sing_order
        for isol in 1:(2*intr.numpert_total)
            for j in 1:i
                @views mul!(tmp_vec, Hermitian(ff_lower[:, :, j+1], :L), x[:, isol, 1, i-j+1])
                @views x[:, isol, 1, i+1] .-= tmp_vec
            end
            @views mul!(tmp_vec, k[:, :, i+1], v[:, isol, 1])
            @views x[:, isol, 1, i+1] .-= tmp_vec
        end
        @views x[:, :, 1, i+1] = UpperTriangular(f0_lower') \ (LowerTriangular(f0_lower) \ x[:, :, 1, i+1])
    end

    # Solve x²ₙ = (G - K^†F⁻¹K)v¹ + K^†F⁻¹v² = Gₙv¹ + ∑Kⱼ^† x¹ₙ₋ⱼ at each order
    for i in 0:ctrl.sing_order
        for isol in 1:(2*intr.numpert_total)
            for j in 0:i
                @views mul!(tmp_vec, adjoint(k[:, :, j+1]), x[:, isol, 1, i-j+1])
                @views x[:, isol, 2, i+1] .+= tmp_vec
            end
            @views mul!(tmp_vec, Hermitian(g_lower[:, :, i+1], :L), v[:, isol, 1])
            @views x[:, isol, 2, i+1] .+= tmp_vec
        end
    end

    # Assemble power series coefficients of M = zS⁻¹(LS - S') at each order in √z
    # eq. 28 in Glasser 2023 PoP paper
    # Compute resonant and nonresonant indices
    ipert_res = 1 .+ singp.m .- intr.mlow .+ (singp.n .- intr.nlow) .* intr.mpert
    r1 = ipert_res
    r2 = vec([ipert_res[i] + j * intr.numpert_total for j in 0:1, i in eachindex(ipert_res)])
    n1 = [i for i in 1:intr.numpert_total if !(i in ipert_res)]
    n2 = vec([i + j * intr.numpert_total for j in 0:1, i in n1])

    j = 0
    # Start with the S⁻¹LS components
    # Glasser PoP 2023 eq. 39: at each other of L, we get contributions to z^k from RLR,
    # z^k+0.5 from RLA and ALR, and z^k+1 from ALA (where A is the nonresonant part)
    for i in 0:ctrl.sing_order
        mmat[r1, r2, :, j+1] .= x[r1, r2, :, i+1]
        mmat[r1, n2, :, j+2] .= x[r1, n2, :, i+1]
        mmat[n1, r2, :, j+2] .= x[n1, r2, :, i+1]
        mmat[n1, n2, :, j+3] .= x[n1, n2, :, i+1]
        # Expansion of M is in half powers of z due to shearing transformation, so we jump by 2
        j += 2
    end
    # Apply the effect of the shearing transformation to the resonant indices R
    # Glasser PoP 2023 eq. 25 + 28: M = zS⁻¹LS - zS⁻¹S' = zS⁻¹LS + 0.5 [R, 0; 0, -R], 0ᵗʰ order only
    for i in eachindex(r1)
        mmat[r1[i], r2[2*i-1], 1, 1] += 0.5
        mmat[r1[i], r2[2*i], 2, 1] -= 0.5
    end
end

"""
    solve_higher_order_vmat!(vmat::Array{ComplexF64,4}, mmat::Array{ComplexF64,4}, m0mat::Matrix{ComplexF64}, alpha::Vector{ComplexF64}, r1::Vector{Int}, r2::Vector{Int}, n1::Vector{Int}, n2::Vector{Int}, power::Vector{ComplexF64}, intr::ForceFreeStatesInternal, k::Int)

Solves iteratively for the next order in the power series `vmat`.
See equation 47 in the Glasser 2016 DCON paper. Identical to the Fortran
`sing_solve` subroutine. Now takes individual arrays instead of singp struct.

### Arguments

  - `vmat::Array{ComplexF64,4}`: V matrix power series (modified in-place)
  - `mmat::Array{ComplexF64,4}`: M matrix power series
  - `m0mat::Matrix{ComplexF64}`: Zeroth order M matrix
  - `alpha::Vector{ComplexF64}`: Eigenvalues of M₀ for resonant modes
  - `r1, r2, n1, n2::Vector{Int}`: Resonant and nonresonant indices
  - `power::Vector{ComplexF64}`: α values for all modes (0 for nonresonant)
  - `k::Int`: The current order in the power series expansion
"""
@with_pool pool function solve_higher_order_vmat!(
    vmat::Array{ComplexF64,4},
    mmat::Array{ComplexF64,4},
    m0mat::Matrix{ComplexF64},
    alpha::Vector{ComplexF64},
    r1::Vector{Int},
    r2::Vector{Int},
    n1::Vector{Int},
    n2::Vector{Int},
    power::Vector{ComplexF64},
    intr::ForceFreeStatesInternal,
    k::Int
)

    tmp_arr = zeros!(pool, ComplexF64, size(vmat)[1:3])
    # Compute ∑Mₗvₖ₋ₗ
    for l in 1:k
        @views sing_matmul!(tmp_arr, mmat[:, :, :, l+1], vmat[:, :, :, k-l+1])
        vmat[:, :, :, k+1] .+= tmp_arr
    end

    a = zeros!(pool, ComplexF64, 2, 2)
    for isol in 1:(2*intr.numpert_total)
        for i in eachindex(r1) # go block by block?
            # a = M₀ - (α + k/2)I = ∑Mₗvₖ₋ₗ (for multi-n 2D, we make a the ith block fo M₀)
            @views m0mat_block = m0mat[(2*(i-1)+1):(2*i), (2*(i-1)+1):(2*i)]
            a .= m0mat_block
            a[1, 1] -= k / 2.0 + power[isol]
            a[2, 2] -= k / 2.0 + power[isol]
            det = a[1, 1] * a[2, 2] - a[1, 2] * a[2, 1]
            # Solve the resonant indices
            x1 = -vmat[r1[i], isol, 1, k+1]
            x2 = -vmat[r1[i], isol, 2, k+1]
            vmat[r1[i], isol, 1, k+1] = (a[2, 2] * x1 - a[1, 2] * x2) / det
            vmat[r1[i], isol, 2, k+1] = (a[1, 1] * x2 - a[2, 1] * x1) / det
        end
        # Solve the non-resonant indices (the eigenvalue α = 0, so M₀v = 0 (null space))
        vmat[n1, isol, :, k+1] ./= (power[isol] + k / 2.0)
    end
end

"""
    sing_matmul(a, b) -> c

Matrix multiplication specific to singular matrices.
Identical to the Fortran `sing_matmul` subroutine.

### Arguments

  - `a::AbstractArray{ComplexF64,3}`: shape (mpert, 2 * mpert, 2)
  - `b::AbstractArray{ComplexF64,3}`: shape (mmpert, 2 * mpert, 2)

## Returns

  - `c::Array{ComplexF64,3}`: shape (mpert, 2 * mpert, 2)
"""
function sing_matmul(a::AbstractArray{ComplexF64,3}, b::AbstractArray{ComplexF64,3})
    c = zeros(ComplexF64, size(a, 1), size(b, 2), 2)
    return sing_matmul!(c, a, b)
end

@with_pool pool function sing_matmul!(out::AbstractArray{ComplexF64,3}, a::AbstractArray{ComplexF64,3}, b::AbstractArray{ComplexF64,3})
    m = size(b, 1)
    n = size(b, 2)

    # consistency check
    if size(a, 2) != 2 * m
        error("Sing_matmul: size(a,2) = $(size(a,2)) != 2*size(b,1) = $(2*m)")
    end

    fill!(out, zero(ComplexF64))
    # main computation
    tmp = acquire!(pool, ComplexF64, size(a, 1))
    for i in 1:n
        for j in 1:2
            @views mul!(tmp, a[:, 1:m, j], b[:, i, 1])
            @views out[:, i, j] .+= tmp
            @views mul!(tmp, a[:, (m+1):(2*m), j], b[:, i, 2])
            @views out[:, i, j] .+= tmp
        end
    end

    return out
end

"""
    sing_get_ua(sing_asymp::SingAsymptotics, z::Float64) -> ua

Compute the asymptotic series solution for a given singular surface.
Fills and returns `ua` with the asymptotic solution vmat from the provided asymptotics.
We obtain the solution using equations 45 and 41 in the 2016 DCON paper.
Performs the same function as `sing_get_ua` in the Fortran code.

### Arguments

  - `sing_asymp::SingAsymptotics`: Pre-computed asymptotic data
  - `z::Float64`: Distance from singular surface = ψ - ψ_res (Note this is -dpsi from cross_ideal_singular_surf)
"""
function sing_get_ua(sing_asymp::SingAsymptotics, z::Float64)

    r1 = sing_asymp.r1
    r2 = sing_asymp.r2
    sqrt_z = sqrt(complex(z)) # √z

    # Compute power series via Horner's method (eq. 45 in Glasser 2016)
    ua = copy(sing_asymp.vmat[:, :, :, 2*sing_asymp.sing_order+1])
    for iorder in (2*sing_asymp.sing_order-1):-1:0
        ua .= ua .* sqrt_z .+ sing_asymp.vmat[:, :, :, iorder+1] # sqrt_z becomes √zᵏ here
    end

    # Loop through resonances - this might change in 3D
    for i in eachindex(r1)
        # Form full power series solution for v by multiplying by zᵅ (eq. 45 in Glasser 2016)
        pfac = abs(z) .^ sing_asymp.alpha[i] # zᵅ
        ua[:, r2[2*i-1], :] ./= pfac # /zᵅ = z⁻ᵅ
        ua[:, r2[2*i], :] .*= pfac

        # Apply shearing transformation u = Rv (eq. 41 in Glasser 2016)
        ua[r1[i], :, 1] ./= sqrt_z # z^-0.5
        ua[r1[i], :, 2] .*= sqrt_z # z^0.5

        # Renormalize
        if z < 0
            ua[:, r2[2*i-1], :] .*= abs(ua[r1[i], r2[2*i-1], 1]) / ua[r1[i], r2[2*i-1], 1]
            ua[:, r2[2*i], :] .*= abs(ua[r1[i], r2[2*i], 1]) / ua[r1[i], r2[2*i], 1]
        end
    end

    return ua
end

"""
    sing_get_ca(u::Array{ComplexF64,3}, ua::Array{ComplexF64,3}, intr::ForceFreeStatesInternal)

Compute the asymptotic expansion coefficients according to equation
50 in Glasser 2016 DCON paper. Performs the same function as
`sing_get_ca` in the Fortran code.

### Arguments

  - `u::Array{ComplexF64,3}`: Current solution matrix, shape (numpert_total, numpert_total, 2)
  - `ua::Array{ComplexF64,3}`: Asymptotic solution matrix, shape (numpert_total, numpert_total, 2)
  - `intr::ForceFreeStatesInternal`: Internal ForceFreeStates data containing perturbation dimensions
"""
function sing_get_ca(u::Array{ComplexF64,3}, ua::Array{ComplexF64,3}, intr::ForceFreeStatesInternal)

    # Build temp1
    temp1 = zeros(ComplexF64, 2 * intr.numpert_total, 2 * intr.numpert_total)
    temp1[1:intr.numpert_total, :] .= ua[:, :, 1]
    temp1[(intr.numpert_total+1):(2*intr.numpert_total), :] .= ua[:, :, 2]

    # Built temp2
    temp2 = zeros(ComplexF64, 2 * intr.numpert_total, intr.numpert_total)
    temp2[1:intr.numpert_total, :] .= u[:, :, 1]
    temp2[(intr.numpert_total+1):(2*intr.numpert_total), :] .= u[:, :, 2]

    # LU factorization and solve
    temp2 .= lu!(temp1) \ temp2

    # Build ca
    ca = zeros(ComplexF64, intr.numpert_total, intr.numpert_total, 2)
    @views ca[:, :, 1] .= temp2[1:intr.numpert_total, :]
    @views ca[:, :, 2] .= temp2[(intr.numpert_total+1):(2*intr.numpert_total), :]

    return ca
end

"""
    sing_der!(
        du::Array{ComplexF64,3},
        u::Array{ComplexF64,3},
        params::Tuple{ForceFreeStatesControl, Equilibrium.PlasmaEquilibrium, FourFitVars, ForceFreeStatesInternal, OdeState},
        psieval::Float64
    )

Evaluate the derivative of the Euler-Lagrange equations [Glasser Phys. Plasmas 2016 112506 eq. 24].
This implements du/dψ for the ideal MHD eigenvalue problem.

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
  - `params::Tuple{ForceFreeStatesControl, Equilibrium.PlasmaEquilibrium, FourFitVars, ForceFreeStatesInternal, OdeState}`: Tuple of relevant structs
  - `psieval::Float64`: Current psi value at which to evaluate the derivative

### TODOs

Implement kin_flag functionality
"""
@with_pool pool function sing_der!(du::Array{ComplexF64,3}, u::Array{ComplexF64,3},
    params::Tuple{ForceFreeStatesControl,Equilibrium.PlasmaEquilibrium,
        FourFitVars,ForceFreeStatesInternal,OdeState,IntegrationChunk},
    psieval::Float64)

    # Unpack structs and initialize
    # note the two items not used here are needed in the integrator params for use in the integrator_callbackcallback
    _, equil, ffit, intr, odet, _ = params

    # Allocate temporary arrays from the pool
    Npert = intr.numpert_total

    singfac_vec = acquire!(pool, Float64, Npert)
    singfac_mat = reshape(singfac_vec, intr.mpert, intr.npert)

    amat = acquire!(pool, ComplexF64, Npert, Npert)
    bmat = similar!(pool, amat)
    cmat = similar!(pool, amat)
    fmat_lower = similar!(pool, amat)
    kmat = similar!(pool, amat)
    gmat = similar!(pool, amat)
    tmp_mat = similar!(pool, amat)

    fill!(tmp_mat, zero(ComplexF64))
    u1 = @view(u[:, :, 1])
    u2 = @view(u[:, :, 2])
    du1 = @view(du[:, :, 1])
    du2 = @view(du[:, :, 2])

    # Compute singfac = 1 / (m - nq)
    # Use shared hint with LinearBinary() search for O(1) interval lookup during sequential ODE integration
    odet.q = equil.profiles.q_spline(psieval; hint=odet.spline_hint)
    singfac_mat .= 1.0 ./ ((intr.mlow:intr.mhigh) .- odet.q .* (intr.nlow:intr.nhigh)')

    # kinetic stuff - skip for now
    if false #(TODO: kin_flag)
        error("kin_flag not implemented yet")
    else
        # Evaluate matrix splines at the current psi value using shared hint
        ffit.amats(vec(amat), psieval; hint=ffit._hint)
        ffit.bmats(vec(bmat), psieval; hint=ffit._hint)
        ffit.cmats(vec(cmat), psieval; hint=ffit._hint)
        ffit.fmats_lower(vec(fmat_lower), psieval; hint=ffit._hint)
        ffit.kmats(vec(kmat), psieval; hint=ffit._hint)
        ffit.gmats(vec(gmat), psieval; hint=ffit._hint)

        # Solve bmat = A⁻¹ * bmat, cmat = A⁻¹ * cmat in-place via Cholesky
        # Equivalent to: Afact = cholesky!(Hermitian(amat)); ldiv!(Afact, bmat); ldiv!(Afact, cmat)
        # but calls LAPACK directly to avoid Hermitian/Cholesky wrapper allocations in this hot loop
        LAPACK.potrf!('U', amat)
        LAPACK.potrs!('U', amat, bmat)
        LAPACK.potrs!('U', amat, cmat)

    end

    # Compute du
    if false #(TODO: kin_flag)
        error("kin_flag not implemented yet")
    else
        # See equations 22-24 in Glasser 2016 DCON paper for derivation
        # du[1] = - F̄⁻¹ * K̄ * u[1] + F̄⁻¹ * Q⁻¹ * u[2]
        du1 .= u2 .* singfac_vec
        mul!(tmp_mat, kmat, u1)
        du1 .-= tmp_mat
        ldiv!(LowerTriangular(fmat_lower), du1)
        ldiv!(UpperTriangular(fmat_lower'), du1)
        # du[2] = G * u[1] + K̄^† * du[1] = G * u[1] - K̄^† * F̄⁻¹ * K̄ * u[1] + K̄^† * F̄⁻¹ * Q⁻¹ * u[2]
        mul!(tmp_mat, gmat, u1)
        du2 .= tmp_mat
        mul!(tmp_mat, adjoint(kmat), du1)
        du2 .+= tmp_mat
        # du[1] = - Q⁻¹ * F̄⁻¹ * K̄ * u[1] + Q⁻¹ * F̄⁻¹ * Q⁻¹ * u[2]
        du1 .*= singfac_vec
    end

    # ud[1] = Ξ'_Ψ
    @views odet.ud[:, :, 1] .= du1
    # ud[2] = Ξ_s = - A⁻¹(B * Ξ'_Ψ - C * Ξ_Ψ), eq. 18 of Glasser 2016
    mul!(tmp_mat, bmat, du1)
    odet.ud[:, :, 2] .= .-tmp_mat
    mul!(tmp_mat, cmat, u1)
    @views odet.ud[:, :, 2] .-= tmp_mat
end
