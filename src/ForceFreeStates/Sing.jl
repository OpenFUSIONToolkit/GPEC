"""
    _find_rational_surfaces(equil::Equilibrium.PlasmaEquilibrium, nlow::Int, nhigh::Int)

Locate all rational q-surfaces q = m/n for n in `nlow:nhigh` by Brent bisection between
consecutive extrema of the q-profile (reverse shear gives one root per monotone segment).
Returns a vector of `(m, n, psifac)` named tuples in discovery order (n outer, ψ-interval
inner). Requires `equilibrium_qfind!` to have populated `equil.params.qextrema_*`.
"""
function _find_rational_surfaces(equil::Equilibrium.PlasmaEquilibrium, nlow::Int, nhigh::Int)
    profiles = equil.profiles
    surfaces = @NamedTuple{m::Int, n::Int, psifac::Float64}[]

    # Loop over all toroidal mode numbers
    for n in nlow:nhigh
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

                psifac = find_zero(psi -> m - n * profiles.q_spline(psi; hint=hint), (psi0, psi1), Roots.Brent())
                push!(surfaces, (m=m, n=n, psifac=psifac))
                m += dm
            end
        end
    end
    return surfaces
end

"""
    rational_psi_nodes(equil::Equilibrium.PlasmaEquilibrium; nlow::Int, nhigh::Int=nlow)

Unique ψ_N locations of all rational surfaces q = m/n for n in `nlow:nhigh`, sorted
increasing. Used as mandatory knots for the two-pass equilibrium grid refinement (the
same physical surface reached through several (m, n) pairs is deduplicated by q value).
"""
function rational_psi_nodes(equil::Equilibrium.PlasmaEquilibrium; nlow::Int, nhigh::Int=nlow)
    surfaces = _find_rational_surfaces(equil, nlow, nhigh)
    nodes = Float64[]
    qs = Float64[]
    for s in surfaces
        any(q -> isapprox(q, s.m / s.n; atol=1e-8), qs) && continue
        push!(qs, s.m / s.n)
        push!(nodes, s.psifac)
    end
    return sort!(nodes)
end

"""
    sing_find!(intr::ForceFreeStatesInternal, equil::Equilibrium.PlasmaEquilibrium)

Locate singular rational q-surfaces (q = m/nn) using a bisection method
between extrema of the q-profile, and store their properties in `intr.sing`.
Performs the same function as `sing_find` in the Fortran code.
"""
function sing_find!(intr::ForceFreeStatesInternal, equil::Equilibrium.PlasmaEquilibrium)
    profiles = equil.profiles
    hint = Ref(1)

    for s in _find_rational_surfaces(equil, intr.nlow, intr.nhigh)
        m, n, psifac = s.m, s.n, s.psifac
        if any(sg -> isapprox(sg.q, m / n; atol=1e-8), intr.sing)
            # Rational surface with multiplicity > 1, add this m,n to the resonant mode numbers
            # Technically only need m or n, but simplifies some later code and cheap to store both
            idx = findfirst(sg -> isapprox(sg.q, m / n; atol=1e-8), intr.sing)
            push!(intr.sing[idx].m, m)
            push!(intr.sing[idx].n, n)
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
`equil.params.qmax`. If `set_psilim_via_dmlim` is true, `qlim` is adjusted to the largest
rational surface such that `nq + dmlim < qmax`. If `qlim < qmax`, a Newton iteration is
performed to find the corresponding `psilim` to integrate to.

Note that the Newton iteration will be triggered if either `set_psilim_via_dmlim` is true
or `ctrl.qhigh < equil.params.qmax`. Otherwise, the equilibrium edge values are used.
"""
function sing_lim!(intr::ForceFreeStatesInternal, ctrl::ForceFreeStatesControl, equil::Equilibrium.PlasmaEquilibrium)

    profiles = equil.profiles

    # Initial guesses based on equilibrium
    intr.qlim = min(equil.params.qmax, ctrl.qhigh) # equilibrium solve only goes up to qmax, so we're capped there
    intr.q1lim = profiles.q_deriv(profiles.xs[end]; hint=Ref(profiles.npts_minus_1))
    intr.psilim = equil.config.psihigh

    # Optionally override qlim based on dmlim (Fortran sas_flag=t equivalent).
    # Multi-n runs (nn_low != nn_high) are not supported — the "outermost rational + dmlim/n"
    # cutoff depends on which n is used, so it isn't well-defined. Single-n with nn_low <= 0
    # (e.g. uninitialized default) is also skipped because the formula divides by nn_low.
    # Both cases fall back to qhigh / psihigh truncation with a warning.
    if ctrl.set_psilim_via_dmlim && ctrl.nn_low != ctrl.nn_high
        @warn "set_psilim_via_dmlim = true is ignored for multi-n runs (nn_low=$(ctrl.nn_low), nn_high=$(ctrl.nn_high)); falling back to qhigh / psihigh truncation."
    elseif ctrl.set_psilim_via_dmlim && ctrl.nn_low <= 0
        @warn "set_psilim_via_dmlim = true requires nn_low > 0; got nn_low=$(ctrl.nn_low). Falling back to qhigh / psihigh truncation."
    elseif ctrl.set_psilim_via_dmlim
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
        jpsi = min(jpsi, length(profiles.xs) - 1)

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
    sing_min!(intr::ForceFreeStatesInternal, ctrl::ForceFreeStatesControl, equil::Equilibrium.PlasmaEquilibrium)

Set the lower integration bound `intr.psilow`. Port of Fortran RDCON `sing_min` (sing.f):
when `qlow > qmin`, the q < qlow core (including any q ≤ 1 sawtooth/internal-kink surfaces) must be
excluded from the outer-region Galerkin domain — otherwise the Hermite FEM integrates through those
ideal singularities without imposing the ideal constraint, contaminating Δ′ at the innermost kept
surface. A Newton iteration locates ψ where q = qlow; scanning starts from the edge inward for
robustness in reverse-shear cores. When `qlow ≤ qmin` the axis value is kept.
"""
function sing_min!(intr::ForceFreeStatesInternal, ctrl::ForceFreeStatesControl, equil::Equilibrium.PlasmaEquilibrium)
    profiles = equil.profiles
    intr.psilow = profiles.xs[1]   # default: equilibrium axis-side bound
    ctrl.qlow > equil.params.qmin || return intr.psilow

    # Scan from the edge inward for the first node with q < qlow (robust for reverse-shear q).
    qy = profiles.q_spline.y
    jpsi = 1
    for j in (length(profiles.xs)-1):-1:1
        if qy[j] < ctrl.qlow
            jpsi = j
            break
        end
    end

    hint = Ref(jpsi)
    intr.psilow = find_zero(
        (psi -> profiles.q_spline(psi; hint=hint) - ctrl.qlow,
            psi -> profiles.q_deriv(psi; hint=hint)),
        profiles.xs[jpsi], Roots.Newton()
    )
    @info "sing_min: qlow=$(@sprintf("%.3f", ctrl.qlow)) > qmin=$(@sprintf("%.3f", equil.params.qmin)); " *
          "raising psilow from $(@sprintf("%.5f", profiles.xs[1])) to $(@sprintf("%.5f", intr.psilow)) (excludes q<qlow core)"
    return intr.psilow
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
function compute_sing_asymptotics(
    singp::SingType,
    ctrl::ForceFreeStatesControl,
    equil::Equilibrium.PlasmaEquilibrium,
    ffit::FourFitVars,
    intr::ForceFreeStatesInternal;
    sig::Float64=1.0,
    alpha_override::Union{Nothing,Vector{ComplexF64}}=nothing
)

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

    # Compute mmat Taylor coefficients with direction parameter sig.
    # Fortran computes separate mmatl (sig=-1) and mmatr (sig=+1) — the sig flips
    # odd derivatives of all input quantities (q, F, G, K splines).
    compute_sing_mmat!(mmat, singp, ctrl, equil.profiles, ffit, intr; sig=sig)

    # Extract direction-specific m0mat from zeroth-order mmat
    m0mat = if length(r1) == 1
        Matrix(transpose(mmat[r1[1], r2, :, 1]))
    else
        Matrix(vcat([transpose(mmat[r1[i], r2, :, 1]) for i in eachindex(r1)]...))
    end

    # Alpha (Mercier index) — Fortran computes this ONCE from the RIGHT-SIDE m0mat
    # and reuses it for both left and right vmat (matching Fortran STRIDE).
    # When alpha_override is provided (for the left-side call), use that instead.
    # Fortran: di = m0(1,1)*m0(2,2) - m0(2,1)*m0(1,2); alpha = sqrt(-di)
    # This matches eigenvalues only when tr(m0mat_block) = 0.
    alpha = if alpha_override !== nothing
        alpha_override
    else
        # Match Fortran exactly: alpha = sqrt(-det(m0mat_block)) for each resonant mode
        [sqrt(-ComplexF64(m0mat[(2*(i-1)+1), (2*(i-1)+1)] * m0mat[(2*i), (2*i)] -
                          m0mat[(2*i), (2*(i-1)+1)] * m0mat[(2*(i-1)+1), (2*i)]))
         for i in eachindex(r1)]
    end

    # This is the parameter α but for all modes - α = 0 for non-resonant modes
    power[ipert_res] .= -alpha
    power[ipert_res .+ intr.numpert_total] .= alpha

    # Zeroth-order non-resonant solutions
    for ipert in 1:intr.numpert_total
        vmat[ipert, ipert, 1, 1] = 1
        vmat[ipert, ipert+intr.numpert_total, 2, 1] = 1
    end

    # Zeroth-order resonant solutions: v_big_ξ' = -(m0(1,1) ± sig·α)/m0(1,2).
    # Matches Fortran STRIDE sing_vmat (sig·α sign convention separates left vs right side).
    for i in eachindex(r1)
        m0mat_block = m0mat[(2*(i-1)+1):(2*i), (2*(i-1)+1):(2*i)]
        r1_i = r1[i]
        r2_i = r1_i + intr.numpert_total
        alpha_i = alpha[i]
        vmat[r1_i, r1_i, 1, 1] = 1
        vmat[r1_i, r2_i, 1, 1] = 1
        vmat[r1_i, r1_i, 2, 1] = -(m0mat_block[1, 1] + sig * alpha_i) / m0mat_block[1, 2]
        vmat[r1_i, r2_i, 2, 1] = -(m0mat_block[1, 1] - sig * alpha_i) / m0mat_block[1, 2]
    end

    # Higher order solutions — sig propagates through the recursion (Fortran STRIDE sing_solve).
    for k in 1:(2*ctrl.sing_order)
        solve_higher_order_vmat!(vmat, mmat, m0mat, alpha, r1, r2, n1, n2, power, intr, k; sig=sig)
    end

    # Per-crossing m0mat / vmat diagnostics matching Fortran sing_vmat output.
    # @debug-only: enable via JULIA_DEBUG=GeneralizedPerturbedEquilibrium.
    @debug begin
        side_str = sig > 0 ? "right" : "left"
        ipert0 = r1[1]
        N = intr.numpert_total
        msg = "  === sing_asymptotics debug: m=$(singp.m[1]) sig=$sig ($side_str)\n"
        msg *= @sprintf("  m0mat(1,1)= %+.12e %+.12ei\n", real(m0mat[1, 1]), imag(m0mat[1, 1]))
        msg *= @sprintf("  m0mat(1,2)= %+.12e %+.12ei\n", real(m0mat[1, 2]), imag(m0mat[1, 2]))
        msg *= @sprintf("  m0mat(2,1)= %+.12e %+.12ei\n", real(m0mat[2, 1]), imag(m0mat[2, 1]))
        msg *= @sprintf("  m0mat(2,2)= %+.12e %+.12ei\n", real(m0mat[2, 2]), imag(m0mat[2, 2]))
        di = m0mat[1, 1]*m0mat[2, 2] - m0mat[2, 1]*m0mat[1, 2]
        msg *= @sprintf("  di= %+.12e, alpha= %+.12e %+.12ei\n", real(di), real(alpha[1]), imag(alpha[1]))
        msg *= @sprintf("  psifac= %+.12e, r1=%d, ipert0=%d\n", singp.psifac, r1[1], ipert0)
        msg *= @sprintf("  vmat(ip,ip,2,0)= %+.8e %+.8ei\n", real(vmat[ipert0, ipert0, 2, 1]), imag(vmat[ipert0, ipert0, 2, 1]))
        msg *= @sprintf("  vmat(ip,ip+N,2,0)= %+.8e %+.8ei\n", real(vmat[ipert0, ipert0+N, 2, 1]), imag(vmat[ipert0, ipert0+N, 2, 1]))
        for k in 0:(2*ctrl.sing_order)
            msg *= @sprintf("  k=%2d vmat(ip,ip,1)=%+.8e %+.8ei vmat(ip,ip,2)=%+.8e %+.8ei\n",
                k, real(vmat[ipert0, ipert0, 1, k+1]), imag(vmat[ipert0, ipert0, 1, k+1]),
                real(vmat[ipert0, ipert0, 2, k+1]), imag(vmat[ipert0, ipert0, 2, k+1]))
            msg *= @sprintf("  k=%2d vmat(ip,ip+N,1)=%+.8e %+.8ei vmat(ip,ip+N,2)=%+.8e %+.8ei\n",
                k, real(vmat[ipert0, ipert0+N, 1, k+1]), imag(vmat[ipert0, ipert0+N, 1, k+1]),
                real(vmat[ipert0, ipert0+N, 2, k+1]), imag(vmat[ipert0, ipert0+N, 2, k+1]))
        end
        msg
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
    intr::ForceFreeStatesInternal;
    sig::Float64=1.0
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

    # Evaluate q spline and its derivatives, applying sig to odd derivatives.
    # Fortran STRIDE sing_mmat: q(1)=sig*q', q(2)=q'', q(3)=sig*q'''
    q = (q_spline(singp.psifac),
        sig * q_d1(singp.psifac),
        q_d2(singp.psifac),
        sig * q_d3(singp.psifac))

    # Evaluate fmats_lower and derivatives, applying sig to odd derivatives.
    # Fortran sing_mmat multiplies fmats_f1 and fmats_f3 by sig in the Taylor products.
    ffit.fmats_lower(vec(@view(f_lower_interp[:, :, 1])), singp.psifac; hint=ffit._hint)
    ffit.fmats_lower(vec(@view(f_lower_interp[:, :, 2])), singp.psifac; deriv=DerivOp(1))
    ffit.fmats_lower(vec(@view(f_lower_interp[:, :, 3])), singp.psifac; deriv=DerivOp(2))
    ffit.fmats_lower(vec(@view(f_lower_interp[:, :, 4])), singp.psifac; deriv=DerivOp(3))
    @views f_lower_interp[:, :, 2] .*= sig  # 1st derivative
    @views f_lower_interp[:, :, 4] .*= sig  # 3rd derivative

    # Evaluate gmats and derivatives, applying sig to odd derivatives
    ffit.gmats(vec(@view(g_interp[:, :, 1])), singp.psifac; hint=ffit._hint)
    ffit.gmats(vec(@view(g_interp[:, :, 2])), singp.psifac; deriv=DerivOp(1))
    ffit.gmats(vec(@view(g_interp[:, :, 3])), singp.psifac; deriv=DerivOp(2))
    ffit.gmats(vec(@view(g_interp[:, :, 4])), singp.psifac; deriv=DerivOp(3))
    @views g_interp[:, :, 2] .*= sig
    @views g_interp[:, :, 4] .*= sig

    # Evaluate kmats and derivatives, applying sig to odd derivatives
    ffit.kmats(vec(@view(k_interp[:, :, 1])), singp.psifac; hint=ffit._hint)
    ffit.kmats(vec(@view(k_interp[:, :, 2])), singp.psifac; deriv=DerivOp(1))
    ffit.kmats(vec(@view(k_interp[:, :, 3])), singp.psifac; deriv=DerivOp(2))
    ffit.kmats(vec(@view(k_interp[:, :, 4])), singp.psifac; deriv=DerivOp(3))
    @views k_interp[:, :, 2] .*= sig
    @views k_interp[:, :, 4] .*= sig

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
            for ipert_m in jpert_m:intr.mpert
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
                    for ipert_m in jpert_m:intr.mpert
                        for kpert_m in 1:jpert_m
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
            for ipert_m in 1:intr.mpert
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
            for ipert_m in jpert_m:intr.mpert
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
        mmat[r1[i], r2[2*i-1], 1, 1] += 0.5 * sig
        mmat[r1[i], r2[2*i], 2, 1] -= 0.5 * sig
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
    k::Int;
    sig::Float64=1.0
)

    tmp_arr = zeros!(pool, ComplexF64, size(vmat)[1:3])
    # Compute ∑Mₗvₖ₋ₗ
    for l in 1:k
        @views sing_matmul!(tmp_arr, mmat[:, :, :, l+1], vmat[:, :, :, k-l+1])
        vmat[:, :, :, k+1] .+= tmp_arr
    end

    a = zeros!(pool, ComplexF64, 2, 2)
    for isol in 1:(2*intr.numpert_total)
        for i in eachindex(r1)
            # Fortran sing_solve: a(i,i) = m0mat(i,i) - sig*(k/2 + power(isol))
            @views m0mat_block = m0mat[(2*(i-1)+1):(2*i), (2*(i-1)+1):(2*i)]
            a .= m0mat_block
            a[1, 1] -= sig * (k / 2.0 + power[isol])
            a[2, 2] -= sig * (k / 2.0 + power[isol])
            det = a[1, 1] * a[2, 2] - a[1, 2] * a[2, 1]
            # Solve the resonant indices
            x1 = -vmat[r1[i], isol, 1, k+1]
            x2 = -vmat[r1[i], isol, 2, k+1]
            vmat[r1[i], isol, 1, k+1] = (a[2, 2] * x1 - a[1, 2] * x2) / det
            vmat[r1[i], isol, 2, k+1] = (a[1, 1] * x2 - a[2, 1] * x1) / det
        end
        # Fortran sing_solve: vmat(n1,isol,:,k) *= sig/(power(isol)+k/2)
        vmat[n1, isol, :, k+1] .*= sig / (power[isol] + k / 2.0)
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
    sing_get_ua(sing_asymp::SingAsymptotics, dpsi::Float64) -> ua

Compute the asymptotic series solution for a given singular surface.
Uses direction-specific asymptotics (left: sig=-1, right: sig=+1) with positive dpsi.
Matches Fortran STRIDE's `sing_get_ua`.

### Arguments

  - `sing_asymp::SingAsymptotics`: Pre-computed asymptotic data (must be left or right specific)
  - `dpsi::Float64`: Positive distance from singular surface = |ψ - ψ_res|
"""
function sing_get_ua(sing_asymp::SingAsymptotics, dpsi::Float64)

    r1 = sing_asymp.r1
    r2 = sing_asymp.r2

    # dpsi = |ψ - ψ_res| is always positive. Direction is handled by the
    # SingAsymptotics (left vs right vmat built with sig=-1 or sig=+1).
    # Matches Fortran STRIDE sing_get_ua: sqrtfac=SQRT(dpsi), always positive.
    sqrtfac = sqrt(dpsi)
    pfac_base = dpsi  # used for dpsi^alpha below

    # Compute power series via Horner's method (eq. 45 in Glasser 2016)
    ua = copy(sing_asymp.vmat[:, :, :, 2*sing_asymp.sing_order+1])
    for iorder in (2*sing_asymp.sing_order-1):-1:0
        ua .= ua .* sqrtfac .+ sing_asymp.vmat[:, :, :, iorder+1]
    end

    # Restore powers (unshear v→u) — matches Fortran STRIDE sing_get_ua
    for i in eachindex(r1)
        pfac = pfac_base ^ sing_asymp.alpha[i]  # dpsi^α
        ua[:, r2[2*i-1], :] ./= pfac  # big solution column: /dpsi^α
        ua[:, r2[2*i], :] .*= pfac    # small solution column: *dpsi^α
        ua[r1[i], :, 1] ./= sqrtfac   # resonant row ξ: /√dpsi
        ua[r1[i], :, 2] .*= sqrtfac   # resonant row ξ': *√dpsi
    end

    return ua
end

"""
    sing_get_dua(sing_asymp::SingAsymptotics, dpsi::Float64) -> dua

Compute the derivative of the asymptotic series solution with respect to the positive distance
`dpsi = |ψ − ψ_res|`, consistent with `sing_get_ua`. Port of Fortran `sing_get_dua`
(sing.f / sing1.f). Shape `(numpert_total, 2*numpert_total, 2)`. Used by the
outer-region Galerkin solver (`gal_extension`, `sing_matvec`).

Direction is carried by the `SingAsymptotics` (left vs right vmat built with sig=∓1), exactly as in
`sing_get_ua`, so `dpsi` is always positive and the arithmetic is real (no analytic continuation).
The term-by-term derivative is built via a `power` array tracking each term's total exponent (in
half-powers of `dpsi`: the 2*sing_order offset, the ∓1 shearing on resonant rows, and the ∓2α on
resonant columns), then the same shearing/`pfac` restoration and the chain-rule factor `1/(2·dpsi)`
are applied. The physical d/dψ sign for the left side (ψ = ψ_res − dpsi) is applied by the caller.

### Arguments

  - `sing_asymp::SingAsymptotics`: Pre-computed asymptotic data (must be left- or right-specific)
  - `dpsi::Float64`: Positive distance from singular surface = |ψ − ψ_res|
"""
function sing_get_dua(sing_asymp::SingAsymptotics, dpsi::Float64)

    r1 = sing_asymp.r1
    r2 = sing_asymp.r2
    order = sing_asymp.sing_order
    sqrtfac = sqrt(dpsi)
    N = size(sing_asymp.vmat, 1)

    # Per-term total exponent (×2, in half-powers of dpsi). See Fortran sing_get_dua.
    power = fill(ComplexF64(2 * order), N, 2 * N, 2)
    power[r1, :, 1] .-= 1
    power[r1, :, 2] .+= 1
    for i in eachindex(r1)
        power[:, r2[2*i-1], :] .-= 2 * sing_asymp.alpha[i]
        power[:, r2[2*i], :] .+= 2 * sing_asymp.alpha[i]
    end

    # Power series derivative by Horner's method (Fortran sing_get_dua)
    dua = sing_asymp.vmat[:, :, :, 2*order+1] .* power
    for iorder in (2*order-1):-1:0
        power .-= 1
        dua .= dua .* sqrtfac .+ sing_asymp.vmat[:, :, :, iorder+1] .* power
    end

    # Restore shearing and resonant powers, then the chain-rule factor 1/(2·dpsi) (Fortran sing_get_dua)
    for i in eachindex(r1)
        pfac = dpsi^sing_asymp.alpha[i]
        dua[:, r2[2*i-1], :] ./= pfac
        dua[:, r2[2*i], :] .*= pfac
        dua[r1[i], :, 1] ./= sqrtfac
        dua[r1[i], :, 2] .*= sqrtfac
    end
    dua ./= (2 * dpsi)

    return dua
end

# --- Column-restricted asymptotic evaluation for the outer-region Galerkin solver ---------------
#
# Every gal caller (gal_resonant!, gal_extension!, gal_get_solution) needs only the two resonant
# columns of a single-surface asymptotic: the big solution (column r2[1] = ipert_res) and the small
# solution (column r2[2] = ipert_res + numpert_total). The general sing_get_ua/sing_get_dua compute
# all 2·numpert_total columns and allocate full (N, 2N, 2) arrays at every evaluation point — ~34×
# more work than used. These kernels evaluate ONLY those two columns into a caller-provided
# (N, 2, 2) buffer (col 1 = big, col 2 = small), allocation-free, with scalar exponent bookkeeping
# instead of the full `power` array. They require a single-resonance asymptotic (length(r1) == 1),
# which always holds for the per-surface gal series. Numerically identical to slicing the full
# result to [r2[1], r2[2]].

"""
    sing_get_ua_res!(out, sing_asymp, dpsi) -> out

Evaluate only the big (col 1) and small (col 2) resonant columns of `sing_get_ua` into the
caller-provided `out` of shape `(numpert_total, 2, 2)`. Allocation-free. See the section comment.
"""
function sing_get_ua_res!(out::AbstractArray{ComplexF64,3}, sing_asymp::SingAsymptotics, dpsi::Float64)
    vmat = sing_asymp.vmat
    order = sing_asymp.sing_order
    N = size(vmat, 1)
    sqrtfac = sqrt(dpsi)
    ρ = sing_asymp.r1[1]
    cbig = sing_asymp.r2[1]
    csml = sing_asymp.r2[2]
    nterm = 2 * order

    @inbounds for k in 1:2, ii in 1:N
        ab = vmat[ii, cbig, k, nterm+1]
        as = vmat[ii, csml, k, nterm+1]
        for t in (nterm-1):-1:0
            ab = ab * sqrtfac + vmat[ii, cbig, k, t+1]
            as = as * sqrtfac + vmat[ii, csml, k, t+1]
        end
        out[ii, 1, k] = ab
        out[ii, 2, k] = as
    end

    # Unshear v→u: big column /dpsi^α, small column *dpsi^α; resonant row /√dpsi (qty1), *√dpsi (qty2).
    pfac = dpsi^sing_asymp.alpha[1]
    @inbounds for k in 1:2, ii in 1:N
        out[ii, 1, k] /= pfac
        out[ii, 2, k] *= pfac
    end
    @inbounds out[ρ, 1, 1] /= sqrtfac
    @inbounds out[ρ, 2, 1] /= sqrtfac
    @inbounds out[ρ, 1, 2] *= sqrtfac
    @inbounds out[ρ, 2, 2] *= sqrtfac
    return out
end

"""
    sing_get_dua_res!(out, sing_asymp, dpsi) -> out

Evaluate only the big (col 1) and small (col 2) resonant columns of `sing_get_dua` into the
caller-provided `out` of shape `(numpert_total, 2, 2)`. Allocation-free; scalar per-term exponents
replace the full `power` array. See the section comment.
"""
function sing_get_dua_res!(out::AbstractArray{ComplexF64,3}, sing_asymp::SingAsymptotics, dpsi::Float64)
    vmat = sing_asymp.vmat
    order = sing_asymp.sing_order
    N = size(vmat, 1)
    sqrtfac = sqrt(dpsi)
    ρ = sing_asymp.r1[1]
    cbig = sing_asymp.r2[1]
    csml = sing_asymp.r2[2]
    α = sing_asymp.alpha[1]
    nterm = 2 * order

    # Per-term exponent base (×1, in half-powers of dpsi): 2·order + row offset (∓1 on the resonant row,
    # by qty) + column offset (−2α big, +2α small). Tracked as scalars per column instead of a full array.
    @inbounds for k in 1:2, ii in 1:N
        roff = ii == ρ ? (k == 1 ? ComplexF64(-1) : ComplexF64(1)) : ComplexF64(0)
        pb = nterm + roff - 2 * α
        ps = nterm + roff + 2 * α
        ab = vmat[ii, cbig, k, nterm+1] * pb
        as = vmat[ii, csml, k, nterm+1] * ps
        for t in (nterm-1):-1:0
            pb -= 1
            ps -= 1
            ab = ab * sqrtfac + vmat[ii, cbig, k, t+1] * pb
            as = as * sqrtfac + vmat[ii, csml, k, t+1] * ps
        end
        out[ii, 1, k] = ab
        out[ii, 2, k] = as
    end

    pfac = dpsi^α
    twodpsi = 2 * dpsi
    @inbounds for k in 1:2, ii in 1:N
        out[ii, 1, k] /= pfac
        out[ii, 2, k] *= pfac
    end
    @inbounds out[ρ, 1, 1] /= sqrtfac
    @inbounds out[ρ, 2, 1] /= sqrtfac
    @inbounds out[ρ, 1, 2] *= sqrtfac
    @inbounds out[ρ, 2, 2] *= sqrtfac
    # Divide (not multiply by reciprocal) to match sing_get_dua bit-for-bit; the resonant QuadGK and the
    # ill-conditioned near-singular solve amplify a 1e-16 input change into ~1e-3 in Δ′.
    @inbounds for idx in eachindex(out)
        out[idx] /= twodpsi
    end
    return out
end

"""
Allocating wrapper for [`sing_get_ua_res!`](@ref); returns the big/small columns as `(N, 2, 2)`.
"""
sing_get_ua_res(sing_asymp::SingAsymptotics, dpsi::Float64) =
    sing_get_ua_res!(Array{ComplexF64,3}(undef, size(sing_asymp.vmat, 1), 2, 2), sing_asymp, dpsi)

"""
Allocating wrapper for [`sing_get_dua_res!`](@ref); returns the big/small columns as `(N, 2, 2)`.
"""
sing_get_dua_res(sing_asymp::SingAsymptotics, dpsi::Float64) =
    sing_get_dua_res!(Array{ComplexF64,3}(undef, size(sing_asymp.vmat, 1), 2, 2), sing_asymp, dpsi)

"""
    sing_matvec(ffit::FourFitVars, intr::ForceFreeStatesInternal, psi::Float64, q::Float64, ua, dua) -> matvec

Apply the Euler-Lagrange residual operator `L u = -(F u' + K u)' + (K† u' + G u)` to the asymptotic
solutions. Port of Fortran `sing_matvec` (sing.f). Returns `matvec`, shape
`(numpert_total, size(ua,2))`.

Uses the reduced (Schur-complemented) `ffit.kmats` (= K̄) and `ffit.gmats` (= Ḡ) directly, with the
**direct** singular factor `singfac = m - n q` applied to `u' = dua[:,:,1]`. The second component
`ua[:,:,2]` is the canonical momentum `F u' + K u`, so `-dua[:,:,2] = -(F u' + K u)'`.

### Arguments

  - `psi`: flux coordinate; `q`: safety factor at `psi` (passed in to avoid re-evaluating the spline)
  - `ua`, `dua`: asymptotic solution and its ψ-derivative from `sing_get_ua`/`sing_get_dua`
"""
function sing_matvec(ffit::FourFitVars, intr::ForceFreeStatesInternal, psi::Float64, q::Float64,
    ua::Array{ComplexF64,3}, dua::Array{ComplexF64,3})

    N = intr.numpert_total
    msol = size(ua, 2)

    # Direct singular factor (m - n q), NOT 1/(m - n q)
    sfvec = vec((intr.mlow:intr.mhigh) .- q .* (intr.nlow:intr.nhigh)')

    kmat = Matrix{ComplexF64}(undef, N, N)
    gmat = Matrix{ComplexF64}(undef, N, N)
    ffit.kmats(vec(kmat), psi; hint=ffit._hint)
    ffit.gmats(vec(gmat), psi; hint=ffit._hint)
    kdag = adjoint(kmat)

    matvec = zeros(ComplexF64, N, msol)
    d1 = Vector{ComplexF64}(undef, N)
    for isol in 1:msol
        @views d1 .= dua[:, isol, 1] .* sfvec          # u' · singfac
        @views matvec[:, isol] .= kdag * d1            # K̄† (u' · singfac)
        @views matvec[:, isol] .+= gmat * ua[:, isol, 1]  # + Ḡ u
        @views matvec[:, isol] .-= dua[:, isol, 2]     # - (F u' + K u)'
    end
    return matvec
end

"""
    sing_matvec!(matvec, kmat, gmat, d1, tmp, sfvec, ffit, intr, psi, q, ua, dua) -> matvec

Allocation-free, in-place form of [`sing_matvec`](@ref) for the hot resonant-quadrature integrand.
All scratch is caller-owned: `kmat`/`gmat` are `N×N`, `d1`/`tmp`/`sfvec` are length-`N`, `matvec` is
`N×size(ua,2)`. The reduced K̄/Ḡ splines write directly into `kmat`/`gmat`. The accumulation order
(separate `gmat*ua` into `tmp`, then add) reproduces `sing_matvec` **bit-for-bit** — a fused 5-arg
`mul!` would round differently, and the ill-conditioned near-singular resonant solve amplifies a
1e-16 change into ~1e-3 in Δ′.
"""
function sing_matvec!(matvec::AbstractMatrix{ComplexF64}, kmat::Matrix{ComplexF64}, gmat::Matrix{ComplexF64},
    d1::Vector{ComplexF64}, tmp::Vector{ComplexF64}, sfvec::Vector{Float64}, ffit::FourFitVars,
    intr::ForceFreeStatesInternal, psi::Float64, q::Float64, ua::AbstractArray{ComplexF64,3}, dua::AbstractArray{ComplexF64,3})

    N = intr.numpert_total
    msol = size(ua, 2)

    # Direct singular factor (m - n q), NOT 1/(m - n q); flat order matches vec((m) .- q.*(n)').
    idx = 0
    @inbounds for nn in intr.nlow:intr.nhigh, mm in intr.mlow:intr.mhigh
        idx += 1
        sfvec[idx] = mm - q * nn
    end

    ffit.kmats(vec(kmat), psi; hint=ffit._hint)
    ffit.gmats(vec(gmat), psi; hint=ffit._hint)
    kdag = adjoint(kmat)

    for isol in 1:msol
        @views d1 .= dua[:, isol, 1] .* sfvec             # u' · singfac
        @views mul!(matvec[:, isol], kdag, d1)            # K̄† (u' · singfac)
        @views mul!(tmp, gmat, ua[:, isol, 1])            # Ḡ u
        @views matvec[:, isol] .+= tmp                    # + Ḡ u
        @views matvec[:, isol] .-= dua[:, isol, 2]        # - (F u' + K u)'
    end
    return matvec
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
        params::Tuple{ForceFreeStatesControl, Equilibrium.PlasmaEquilibrium, FourFitVars, ForceFreeStatesInternal, OdeState, IntegrationChunk},
        psieval::Float64
    )

Evaluate the derivative of the Euler-Lagrange equations [Glasser Phys. Plasmas 2016 112506 eq. 24].
This implements du/dψ for both the ideal and kinetic MHD eigenvalue problems.

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
  - `params::Tuple{ForceFreeStatesControl, PlasmaEquilibrium, FourFitVars, ForceFreeStatesInternal, OdeState, IntegrationChunk}`: Tuple of relevant structs
  - `psieval::Float64`: Current psi value at which to evaluate the derivative
"""
@with_pool pool function sing_der!(du::Array{ComplexF64,3}, u::Array{ComplexF64,3},
    params::Tuple{ForceFreeStatesControl,Equilibrium.PlasmaEquilibrium,
        FourFitVars,ForceFreeStatesInternal,OdeState,IntegrationChunk},
    psieval::Float64)

    # Unpack structs
    ctrl, equil, ffit, intr, odet, _ = params

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
    # Use shared hint for O(1) interval lookup during sequential ODE integration
    odet.q = equil.profiles.q_spline(psieval; hint=odet.spline_hint)
    singfac_mat .= 1.0 ./ ((intr.mlow:intr.mhigh) .- odet.q .* (intr.nlow:intr.nhigh)')

    if ctrl.kinetic_factor > 0
        # ---- Kinetic path with pre-computed FKG matrices ----
        # Load pre-computed kinetic matrices from splines
        # amat/bmat/cmat here are the kinetic-modified A_kin/B_kin/C_kin
        # Use odet.ffit_hint (per-thread) instead of ffit._hint (shared, racy in parallel BVP)
        ffit.amats(vec(amat), psieval; hint=odet.ffit_hint)
        ffit.bmats(vec(bmat), psieval; hint=odet.ffit_hint)
        ffit.cmats(vec(cmat), psieval; hint=odet.ffit_hint)

        # Load FKG sub-matrices (note: reusing fmat_lower/kmat/gmat as workspace)
        f0mat = similar!(pool, amat)
        pmat_kin = similar!(pool, amat)
        paat_kin = similar!(pool, amat)
        kkmat_kin = similar!(pool, amat)
        kkaat_kin = similar!(pool, amat)
        r1mat_kin = similar!(pool, amat)
        r2mat_kin = similar!(pool, amat)
        r3mat_kin = similar!(pool, amat)
        gaat_kin = similar!(pool, amat)

        ffit.f0mats(vec(f0mat), psieval; hint=odet.ffit_hint)
        ffit.pmats(vec(pmat_kin), psieval; hint=odet.ffit_hint)
        ffit.paats(vec(paat_kin), psieval; hint=odet.ffit_hint)
        ffit.kkmats(vec(kkmat_kin), psieval; hint=odet.ffit_hint)
        ffit.kkaats(vec(kkaat_kin), psieval; hint=odet.ffit_hint)
        ffit.r1mats(vec(r1mat_kin), psieval; hint=odet.ffit_hint)
        ffit.r2mats(vec(r2mat_kin), psieval; hint=odet.ffit_hint)
        ffit.r3mats(vec(r3mat_kin), psieval; hint=odet.ffit_hint)
        ffit.gaats(vec(gaat_kin), psieval; hint=odet.ffit_hint)

        # A⁻¹B, A⁻¹C via LU (A is non-Hermitian with kinetic contributions)
        # Direct LAPACK to avoid the ipiv allocation that lu!/ldiv! would do in this hot loop
        _, ipiv, _ = LAPACK.getrf!(amat)
        LAPACK.getrs!('N', amat, ipiv, bmat)
        LAPACK.getrs!('N', amat, ipiv, cmat)

        # Build singfac-dependent F̄, K̄, K̄†, Ḡ† matrices (Logan 2015 Appendix C, Eqs C.5-C.11):
        # F̄(i,j) = q1*f0*q2 - q1*P - P†'*q2 + R1
        # K̄(i,j) = q1*KK + R2
        # K̄†(i,j) = KK†*q2 + R3
        # where q1 = (m₁ - n*q), q2 = (m₂ - n*q) — direct singfac, NOT 1/(m-nq) as in ideal path
        singfac_direct = acquire!(pool, Float64, Npert)
        singfac_direct_mat = reshape(singfac_direct, intr.mpert, intr.npert)
        singfac_direct_mat .= (intr.mlow:intr.mhigh) .- odet.q .* (intr.nlow:intr.nhigh)'

        # Build F, K, K† with singfac (using fmat_lower, kmat, gmat as workspace for F, K, K†)
        kaat_kin = similar!(pool, amat)  # K† matrix
        for j in 1:Npert
            q2 = singfac_direct[j]
            for i in 1:Npert
                q1 = singfac_direct[i]
                fmat_lower[i, j] = q1 * f0mat[i, j] * q2 - q1 * pmat_kin[i, j] -
                                   conj(paat_kin[j, i]) * q2 + r1mat_kin[i, j]
                kmat[i, j] = q1 * kkmat_kin[i, j] + r2mat_kin[i, j]
                kaat_kin[i, j] = kkaat_kin[i, j] * q2 + r3mat_kin[i, j]
            end
        end
        # gmat = gaat (already loaded)
        gmat .= gaat_kin

        # Kinetic ODE (Logan 2015 Eq 7.46): singfac absorbed into F̄/K̄/K̄†, no explicit Q⁻¹
        # du₁ = F̄⁻¹(u₂ - K̄·u₁)
        du1 .= u2
        mul!(tmp_mat, kmat, u1)
        du1 .-= tmp_mat
        # LU factorize F (non-Hermitian, non-symmetric); direct LAPACK for the same hot-loop reason
        _, ipiv2, _ = LAPACK.getrf!(fmat_lower)
        LAPACK.getrs!('N', fmat_lower, ipiv2, du1)

        # du₂ = Ḡ†·u₁ + K̄†·du₁  (Logan 2015 Eq C.10-C.11)
        mul!(tmp_mat, gmat, u1)
        du2 .= tmp_mat
        mul!(tmp_mat, kaat_kin, du1)
        du2 .+= tmp_mat

    else
        # ---- Ideal path ----
        # Evaluate matrix splines at the current psi (odet.ffit_hint is per-thread)
        ffit.amats(vec(amat), psieval; hint=odet.ffit_hint)
        ffit.bmats(vec(bmat), psieval; hint=odet.ffit_hint)
        ffit.cmats(vec(cmat), psieval; hint=odet.ffit_hint)
        ffit.fmats_lower(vec(fmat_lower), psieval; hint=odet.ffit_hint)
        ffit.kmats(vec(kmat), psieval; hint=odet.ffit_hint)
        ffit.gmats(vec(gmat), psieval; hint=odet.ffit_hint)

        # Solve bmat = A⁻¹ * bmat, cmat = A⁻¹ * cmat in-place via Cholesky
        LAPACK.potrf!('U', amat)
        LAPACK.potrs!('U', amat, bmat)
        LAPACK.potrs!('U', amat, cmat)

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

"""
    evaluate_fbar_condition(psi, ffit, equil, intr; hint=Ref(1))

Evaluate the condition number of the kinetic F̄ matrix at a given ψ. Uses cond(F̄)
as a scale-invariant measure of near-singularity. Mirrors the intent of Fortran
`sing_get_f_det` (`sing.f:1298-1481`) which computes det(F̄).

F̄(i,j) = q₁·f0(i,j)·q₂ - q₁·P(i,j) - conj(P†(j,i))·q₂ + R1(i,j)

where q₁ = m₁ - n·q(ψ), q₂ = m₂ - n·q(ψ) are the direct singularity factors.
"""
function evaluate_fbar_condition(psi::Float64, ffit::FourFitVars, equil::Equilibrium.PlasmaEquilibrium, intr::ForceFreeStatesInternal; hint=Ref(1))
    np = intr.numpert_total

    # Evaluate q(ψ) and compute singfac = m - n*q
    q = equil.profiles.q_spline(psi; hint=hint)
    singfac = Float64[(m - q * n) for m in intr.mlow:intr.mhigh for n in intr.nlow:intr.nhigh]

    # Evaluate FKG sub-matrices from splines
    f0_vec = zeros(ComplexF64, np * np)
    p_vec = zeros(ComplexF64, np * np)
    pa_vec = zeros(ComplexF64, np * np)
    r1_vec = zeros(ComplexF64, np * np)
    ffit.f0mats(f0_vec, psi; hint=hint)
    ffit.pmats(p_vec, psi; hint=hint)
    ffit.paats(pa_vec, psi; hint=hint)
    ffit.r1mats(r1_vec, psi; hint=hint)
    f0mat = reshape(f0_vec, np, np)
    pmat = reshape(p_vec, np, np)
    paat = reshape(pa_vec, np, np)
    r1mat = reshape(r1_vec, np, np)

    # Assemble F̄ [Fortran sing.f lines 1412-1423, sing_get_f_det with fkg_kmats_flag=true]
    fbar = zeros(ComplexF64, np, np)
    for j in 1:np
        q2 = singfac[j]
        for i in 1:np
            q1 = singfac[i]
            fbar[i, j] = q1 * f0mat[i, j] * q2 - q1 * pmat[i, j] - conj(paat[j, i]) * q2 + r1mat[i, j]
        end
    end

    return cond(fbar)
end

"""
    find_kinetic_singular_surfaces!(ffit, equil, intr; ngrid=2000, cond_threshold=1e8)

Find kinetically-displaced singular surfaces — locations where cond(F̄) peaks,
indicating near-singularity of the kinetic F̄ matrix in the ODE RHS. Populates
`intr.kinsing` and `intr.kmsing`.

Mirrors the intent of Fortran `ksing_find` (`sing.f:1486-1616`) which finds zeros of
det(F̄) via adaptive bisection. Here we use condition number peaks instead of
determinant zeros for better numerical robustness and scale invariance.

Algorithm:

 1. Evaluate cond(F̄) on a dense ψ grid
 2. Find local maxima (peaks where gradient changes from + to -)
 3. Refine each peak with golden-section minimization of -cond
 4. Filter by threshold and resonance condition
"""
function find_kinetic_singular_surfaces!(ffit::FourFitVars, equil::Equilibrium.PlasmaEquilibrium, intr::ForceFreeStatesInternal; ngrid::Int=2000, cond_threshold::Float64=1e8)
    psilow = equil.profiles.xs[1]
    psihigh = intr.psilim

    # Evaluate cond(F̄) on a dense grid
    psi_grid = collect(range(psilow, psihigh; length=ngrid))
    cond_vals = zeros(ngrid)
    hint = Ref(1)
    for i in 1:ngrid
        try
            cond_vals[i] = evaluate_fbar_condition(psi_grid[i], ffit, equil, intr; hint=hint)
        catch
            cond_vals[i] = Inf  # singular matrix — definitely a kinsing surface
        end
    end

    # Persist the scan so callers/HDF5 output can plot cond(F̄) vs ψ and show why peaks
    # were (or weren't) accepted as kinetic singular surfaces.
    intr.kinsing_scan_psi = psi_grid
    intr.kinsing_scan_cond = cond_vals
    intr.kinsing_scan_threshold = cond_threshold

    # Find local maxima of cond(F̄): points where cond increases then decreases
    peak_indices = Int[]
    for i in 2:(ngrid-1)
        if cond_vals[i] > cond_vals[i-1] && cond_vals[i] > cond_vals[i+1] && cond_vals[i] > cond_threshold
            push!(peak_indices, i)
        end
    end

    # Refine each peak to find the precise ψ location
    kinsing_surfaces = SingType[]
    for idx in peak_indices
        psi_lo = psi_grid[max(idx - 1, 1)]
        psi_hi = psi_grid[min(idx + 1, ngrid)]

        # Golden-section search to maximize cond (minimize -cond)
        psi_refined = _golden_section_max(psi_lo, psi_hi, psi -> evaluate_fbar_condition(psi, ffit, equil, intr))

        # Evaluate q and q' at refined location
        hint_ref = Ref(1)
        q_val = equil.profiles.q_spline(psi_refined; hint=hint_ref)
        q1_val = equil.profiles.q_deriv(psi_refined; hint=hint_ref)

        # Check resonance: at least one mode m satisfies mlow ≤ n*q ≤ mhigh
        has_resonant = false
        for n in intr.nlow:intr.nhigh
            nq = n * q_val
            if intr.mlow <= nq && nq <= intr.mhigh
                has_resonant = true
                break
            end
        end
        if !has_resonant
            continue
        end

        push!(
            kinsing_surfaces,
            SingType(;
                psifac=psi_refined,
                rho=sqrt(psi_refined),
                m=[round(Int, n * q_val) for n in intr.nlow:intr.nhigh],
                n=collect(intr.nlow:intr.nhigh),
                q=q_val,
                q1=q1_val
            )
        )
    end

    # Sort by ψ location
    sort!(kinsing_surfaces; by=s -> s.psifac)

    intr.kinsing = kinsing_surfaces
    intr.kmsing = length(kinsing_surfaces)

    if intr.kmsing > 0
        @info "Found $(intr.kmsing) kinetic singular surface(s):"
        for (i, ks) in enumerate(intr.kinsing)
            @info @sprintf("   kinsing[%d]: ψ = %.6f, q = %.4f", i, ks.psifac, ks.q)
        end
    else
        @info "No kinetic singular surfaces found (cond threshold = $(cond_threshold))"
    end

    return nothing
end

"""
Golden-section search to find the ψ that maximizes f(ψ) on [a, b].
"""
function _golden_section_max(a::Float64, b::Float64, f::Function; tol::Float64=1e-10)
    gr = (sqrt(5) + 1) / 2
    c = b - (b - a) / gr
    d = a + (b - a) / gr
    for _ in 1:100
        if abs(b - a) < tol
            break
        end
        if f(c) > f(d)
            b = d
        else
            a = c
        end
        c = b - (b - a) / gr
        d = a + (b - a) / gr
    end
    return (a + b) / 2
end
