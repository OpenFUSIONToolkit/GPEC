"""
    evaluate_stability_criterion!(odet, profiles) -> nzero

Fill `odet.crit_store` over all saved steps and return the number of zero crossings of the
critical eigenvalue (each signals a fixed-boundary instability). Warns with a count if W⁻¹ was
non-Hermitian beyond tolerance at any step.
"""
function evaluate_stability_criterion!(odet::OdeState, profiles::Equilibrium.ProfileSplines)

    resize!(odet.crit_store, odet.step)
    nzero = 0
    nonherm_count = 0

    for istep in 1:odet.step
        zero_cross, nonherm = check_for_zero_crossings!(odet, profiles, istep)
        if zero_cross
            nzero += 1
        end
        if nonherm
            nonherm_count += 1
        end
    end

    if nonherm_count > 0
        @warn "W inverse matrix was non-Hermitian beyond tolerance at $nonherm_count integration step(s)"
    end

    return nzero
end

"""
    check_for_zero_crossings!(odet, profiles, istep) -> zero_cross, nonherm

Compute `crit` (smallest-magnitude eigenvalue of W⁻¹, scaled by (dV/dψ)²) at step `istep` into
`odet.crit_store`, and report whether it changed sign since the previous step. A sign change counts
only if the interpolated midpoint confirms it is a genuine crossing rather than noise.

### Returns

  - `zero_cross::Bool`: True if a physical zero crossing was detected
  - `nonherm::Bool`: True if W⁻¹ was non-Hermitian beyond tolerance
"""
@with_pool pool function check_for_zero_crossings!(odet::OdeState, profiles::Equilibrium.ProfileSplines, istep::Int)

    u = acquire!(pool, eltype(odet.u_store), size(odet.u_store)[1:3])
    psi = odet.psi_store[istep]
    u .= odet.u_store[:, :, :, istep]
    dVdpsi = profiles.dVdpsi_spline(psi; hint=odet.spline_hint)
    crit_val, nonherm = compute_smallest_eigenvalue(u)
    odet.crit_store[istep] = crit_val * dVdpsi^2

    zero_cross = false
    if istep > 1 && odet.crit_store[istep] * odet.crit_store[istep-1] < 0
        crit = odet.crit_store[istep]
        crit_prev = odet.crit_store[istep-1]
        # Reject noise: crit at the interpolated crossing must lie between and well below both neighbors
        fac = crit / (crit - crit_prev)
        psi_mid = psi - fac * (psi - odet.psi_store[istep-1])
        u_mid = u .- fac .* (u .- @view(odet.u_store[:, :, :, istep-1]))
        dVdpsi = profiles.dVdpsi_spline(psi_mid; hint=odet.spline_hint)
        crit_mid_val, _ = compute_smallest_eigenvalue(u_mid)
        crit_mid = crit_mid_val * dVdpsi^2
        if (crit_mid - crit) * (crit_mid - crit_prev) < 0 && abs(crit_mid) < 0.5 * min(abs(crit), abs(crit_prev))
            zero_cross = true
            @info "Zero crossing detected at ψ = $(@sprintf("%.3f", psi_mid))"
        end
    end
    return zero_cross, nonherm
end

"""
    compute_smallest_eigenvalue(u) -> crit, nonherm

Form W⁻¹ = U₁U₂⁻¹ from the solution `u`, symmetrize it, and return its signed eigenvalue of
smallest magnitude. Also flags whether W⁻¹ was non-Hermitian beyond tolerance before symmetrizing.

### Returns

  - `crit::Float64`: the computed scaled critical eigenvalue
  - `nonherm::Bool`: true if W⁻¹ was non-Hermitian beyond tolerance (> 1e-3)
"""
@with_pool pool function compute_smallest_eigenvalue(u::AbstractArray{ComplexF64,3})

    # In-place wp_inverse = u[:, :, 1] / u[:, :, 2]
    wp_inverse = acquire!(pool, ComplexF64, size(u, 1), size(u, 2))
    U2_tmp = similar!(pool, wp_inverse)
    wp_inverse .= @view u[:, :, 1]
    U2_tmp .= @view u[:, :, 2]
    rdiv!(wp_inverse, lu!(U2_tmp))

    # Measure non-Hermiticity; taking the Hermitian part differs from Hermitian(), which reads one triangle only
    adjoint_wp_inverse = similar!(pool, wp_inverse)
    tmp_mat1 = similar!(pool, wp_inverse)
    tmp_mat2 = similar!(pool, wp_inverse)
    adjoint!(adjoint_wp_inverse, wp_inverse)
    @. tmp_mat1 = 0.5 * (wp_inverse + adjoint_wp_inverse)
    @. tmp_mat2 = 0.5 * (wp_inverse - adjoint_wp_inverse)

    nonherm_error = norm(tmp_mat2) / norm(tmp_mat1)
    nonherm = nonherm_error > 1e-3

    hermitianpart!(wp_inverse)

    # Return the eigenvalue of smallest magnitude, keeping its sign so the caller can detect crossings
    evals = eigvals!(Hermitian(wp_inverse))
    crit = evals[findmin(abs, evals)[2]]
    return crit, nonherm
end

"""
    cayley_eigen!(θ, V, u1, u2, s) -> defect

Eigen-decompose the Cayley transform Ψ = (U₁ − i s U₂)(U₁ + i s U₂)⁻¹ of the solution frame `[U₁; U₂]`,
writing the eigenphases in (−π, π] to `θ` and the eigenvectors to the columns of `V`, and return the
unitarity defect max| |λ| − 1 |. For a Lagrangian frame Ψ is unitary; an eigenvalue −1 occurs exactly where
det U₁ = 0, a conjugate point, while an eigenvalue +1 marks a singular U₂, a harmless pole of W⁻¹ (Arnold
1967, Funct. Anal. Appl. 1, 1; Robbin & Salamon 1993, Topology 32, 827). Ψ is invariant under U → U·T for
any invertible T, so Gaussian reduction leaves it unchanged and the frame can be passed with normalized
columns (see [`normalized_frame`](@ref)). When U₁†U₂ drifts from Hermitian, Ψ is not exactly unitary; the
eigenpairs are used regardless and the defect reports the drift.
"""
function cayley_eigen!(θ::AbstractVector{Float64}, V::AbstractMatrix{ComplexF64}, u1::AbstractMatrix{ComplexF64}, u2::AbstractMatrix{ComplexF64}, s::Real)
    psi = (u1 .- (im * s) .* u2) / (u1 .+ (im * s) .* u2)
    E = eigen!(psi)
    defect = 0.0
    for (j, l) in enumerate(E.values)
        defect = max(defect, abs(abs(l) - 1))
        a = angle(l)
        θ[j] = a <= -π ? Float64(π) : a
    end
    V .= E.vectors
    return defect
end

"""
    normalized_frame(u) -> (u1, u2)

Copies of the two halves of the solution frame `u = [U₁; U₂]` with each column of the stacked frame scaled
to unit norm, so that the Cayley inverse is conditioned by the frame's directions and not by the column
norms, which span many decades between Gaussian reductions.
"""
function normalized_frame(u::AbstractArray{ComplexF64,3})
    u1 = copy(@view u[:, :, 1])
    u2 = copy(@view u[:, :, 2])
    for j in axes(u1, 2)
        c = hypot(norm(@view u1[:, j]), norm(@view u2[:, j]))
        c > 0 && (u1[:, j] ./= c; u2[:, j] ./= c)
    end
    return u1, u2
end

"""
    maslov_match(θold, Vold, θnew, Vnew) -> (w, maxrot, overlap)

Follow each eigenphase from one step to the next through the eigenvectors, assigning old to new by greatest
overlap |⟨v_old, v_new⟩| first until every pair is matched (the eigenvectors of a unitary Ψ are orthonormal,
so a clear step gives overlaps near 1; within a cluster of nearly equal phases the assignment is arbitrary
but harmless, since swapped partners rotate by the same small arcs). Each phase moves along the shortest arc
δ = angle(exp(i(θnew − θold))); return the signed number of arcs that cross π (a conjugate point, +1 upward
and −1 downward), the largest |δ|, and the smallest overlap used.
"""
function maslov_match(θold::AbstractVector{Float64}, Vold::AbstractMatrix{ComplexF64}, θnew::AbstractVector{Float64}, Vnew::AbstractMatrix{ComplexF64})
    O = abs.(Vold' * Vnew)
    N = length(θnew)
    partner = zeros(Int, N)
    overlap = 1.0
    for _ in 1:N
        o, idx = findmax(O)
        j, k = Tuple(idx)
        partner[k] = j
        overlap = min(overlap, o)
        O[j, :] .= -1.0
        O[:, k] .= -1.0
    end
    w, maxrot = 0, 0.0
    for k in 1:N
        d = angle(cis(θnew[k] - θold[partner[k]]))
        maxrot = max(maxrot, abs(d))
        target = θold[partner[k]] + d
        w += target > π ? 1 : target <= -π ? -1 : 0
    end
    return w, maxrot, overlap
end

"""
    maslov_seed!(odet)

Mark the conjugate-point count for reseeding at the next accepted step. At a chunk start U₁ has vanishing
columns (every column at the axis, the zeroed resonant column after an ideal crossing), so eigenphases sit
exactly at π there; seeding one step later counts conjugate points in the open interval, as Newcomb's
theorem states it, and sets the Cayley scale s = ‖U₁‖/‖U₂‖ from a nonsingular frame.
"""
function maslov_seed!(odet::OdeState)
    odet.maslov_pending = true
    return odet
end

"""
    maslov_advance!(odet, u; refine=nothing, depth=0)

Advance the conjugate-point count from the stored eigenpairs to those of frame `u`: add the signed number
of eigenphases that crossed π to `odet.nconj`. A phase that rotated by more than π/4 in the step leaves its
direction of travel in doubt (an eigenvalue of W⁻¹ passing zero or infinity quickly), so when `refine(τ)`
can supply the frame at fraction τ of the step (the integrator's dense output), the step is bisected, up to
six times, until every arc is short. The step is counted in `odet.nconj_flags` when a rotation above π/2
remains or the unitarity defect exceeds 1e-3 (the frame is not Lagrangian to working precision, so its
eigenphases are unreliable). A pending seed stores the eigenpairs and the scale instead.
"""
function maslov_advance!(odet::OdeState, u::AbstractArray{ComplexF64,3}; refine=nothing, depth::Int=0)
    u1, u2 = normalized_frame(u)
    if odet.maslov_pending
        n1, n2 = norm(u1), norm(u2)
        odet.maslov_s = n1 > 0 && n2 > 0 ? n1 / n2 : 1.0
        odet.maslov_defect = max(odet.maslov_defect, cayley_eigen!(odet.maslov_theta, odet.maslov_vecs, u1, u2, odet.maslov_s))
        odet.maslov_pending = false
        return odet
    end
    θnew, Vnew = similar(odet.maslov_theta), similar(odet.maslov_vecs)
    defect = cayley_eigen!(θnew, Vnew, u1, u2, odet.maslov_s)
    w, maxrot, _ = maslov_match(odet.maslov_theta, odet.maslov_vecs, θnew, Vnew)
    if maxrot > π / 4 && refine !== nothing && depth < 6
        maslov_advance!(odet, refine(0.5); refine=τ -> refine(0.5 * τ), depth=depth + 1)
        return maslov_advance!(odet, u; refine=τ -> refine(0.5 + 0.5 * τ), depth=depth + 1)
    end
    odet.maslov_defect = max(odet.maslov_defect, defect)
    (maxrot > π / 2 || defect > 1e-3) && (odet.nconj_flags += 1)
    odet.nconj += w
    odet.maslov_theta .= θnew
    odet.maslov_vecs .= Vnew
    return odet
end
