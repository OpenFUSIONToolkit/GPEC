# Rational-surface finding: ideal (q = m/n) and kinetic (cond(F) peak) singular surfaces.

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
    sing_lim!(intr::ForceFreeStatesInternal, ctrl::ForceFreeStatesControl, equil::Equilibrium.PlasmaEquilibrium)

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
    intr.psilim = equil.params.psihigh_resolved

    # Optionally override qlim based on dmlim (Fortran sas_flag=t equivalent). The cutoff reads
    # the *resolved* toroidal range on `intr`, so callers must assign intr.nlow / intr.nhigh
    # before calling; an unresolved range is an error rather than a silent change of truncation
    # strategy. Multi-n runs are not supported — the "outermost rational + dmlim/n" cutoff depends
    # on which n is used — and fall back to qhigh / psihigh truncation with a warning.
    if ctrl.set_psilim_via_dmlim && intr.nlow <= 0
        error("sing_lim!: set_psilim_via_dmlim = true requires a resolved toroidal range, but got intr.nlow=$(intr.nlow). " *
              "Assign intr.nlow / intr.nhigh (from ctrl.nn_low / ctrl.nn_high) before calling sing_lim!, " *
              "or set set_psilim_via_dmlim = false to truncate via qhigh / psihigh instead.")
    elseif ctrl.set_psilim_via_dmlim && intr.nlow != intr.nhigh
        @warn "set_psilim_via_dmlim = true is ignored for multi-n runs (nn_low=$(intr.nlow), nn_high=$(intr.nhigh)); falling back to qhigh / psihigh truncation."
    elseif ctrl.set_psilim_via_dmlim
        @info "Setting psilim via dmlim: initial qlim = $(@sprintf("%.3f", intr.qlim)), dmlim = $(@sprintf("%.3f", ctrl.dmlim))"
        # Normalize dmlim ∈ [0,1)
        dmlim = mod(ctrl.dmlim, 1.0)
        intr.qlim = (trunc(Int, intr.nlow * intr.qlim) + dmlim) / intr.nlow

        # Reduce qlim if above qmax
        while intr.qlim > equil.params.qmax
            intr.qlim -= 1.0 / intr.nlow
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
    evaluate_fbar_condition(psi, kin, equil, intr; hint=Ref(1))

Evaluate the condition number of the kinetic F̄ matrix at a given ψ. Uses cond(F̄)
as a scale-invariant measure of near-singularity. Mirrors the intent of Fortran
`sing_get_f_det` (`sing.f:1298-1481`) which computes det(F̄).

F̄(i,j) = q₁·f0(i,j)·q₂ - q₁·P(i,j) - conj(P†(j,i))·q₂ + R1(i,j)

where q₁ = m₁ - n·q(ψ), q₂ = m₂ - n·q(ψ) are the direct singularity factors.
"""
function evaluate_fbar_condition(psi::Float64, kin::KineticMatrices, equil::Equilibrium.PlasmaEquilibrium, intr::ForceFreeStatesInternal; hint=Ref(1))
    np = intr.numpert_total

    # Evaluate q(ψ) and compute singfac = m - n*q
    q = equil.profiles.q_spline(psi; hint=hint)
    singfac = Float64[(m - q * n) for m in intr.mlow:intr.mhigh for n in intr.nlow:intr.nhigh]

    # Evaluate FKG sub-matrices from splines
    f0_vec = zeros(ComplexF64, np * np)
    p_vec = zeros(ComplexF64, np * np)
    pa_vec = zeros(ComplexF64, np * np)
    r1_vec = zeros(ComplexF64, np * np)
    kin.F0_spline(f0_vec, psi; hint=hint)
    kin.P_spline(p_vec, psi; hint=hint)
    kin.P_spline_adj(pa_vec, psi; hint=hint)
    kin.R1_spline(r1_vec, psi; hint=hint)
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
    find_kinetic_singular_surfaces!(mats, equil, intr; ngrid=2000, cond_threshold=1e8)

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
function find_kinetic_singular_surfaces!(mats::MatrixSplines, equil::Equilibrium.PlasmaEquilibrium, intr::ForceFreeStatesInternal; ngrid::Int=2000, cond_threshold::Float64=1e8)
    kin = mats.kinetic
    kin === nothing && error("find_kinetic_singular_surfaces! requires a kinetic fit; call make_kinetic_matrix first")
    psilow = equil.profiles.xs[1]
    psihigh = intr.psilim

    # Evaluate cond(F̄) on a dense grid
    psi_grid = collect(range(psilow, psihigh; length=ngrid))
    cond_vals = zeros(ngrid)
    hint = Ref(1)
    for i in 1:ngrid
        try
            cond_vals[i] = evaluate_fbar_condition(psi_grid[i], kin, equil, intr; hint=hint)
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
        psi_refined = _golden_section_max(psi_lo, psi_hi, psi -> evaluate_fbar_condition(psi, kin, equil, intr))

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
