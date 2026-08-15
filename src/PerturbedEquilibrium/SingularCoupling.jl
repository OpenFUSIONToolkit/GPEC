"""
Singular surface coupling and field calculations.

Implements GPEC's singular surface analysis (Fortran `gpout_resp` / `gpout_respinfo`):
- Delta' (tearing stability parameter)
- Resonant currents
- Island half-widths
- Chirikov parameters (island overlap)
- Coupling matrices

References:
- [Park Phys. Plasmas 2009 056115] - Plasma response calculation
- [Park Phys. Rev. Lett. 2007 195003] - RMP control
- [Glasser Phys. Plasmas 2016 072505] - Resistive Δ' calculation
"""

# Find bracketing indices and linear interpolation weight for psi in psi_store[1:nstep].
function _psi_bracket(psi_store::AbstractVector, psi::Float64, nstep::Int)
    idx_r = findfirst(p -> p >= psi, @view(psi_store[1:nstep]))
    if isnothing(idx_r) || idx_r >= nstep
        il, ir = nstep - 1, nstep
    elseif idx_r == 1
        il, ir = 1, 2
    else
        il, ir = idx_r - 1, idx_r
    end
    wt = (psi - psi_store[il]) / (psi_store[ir] - psi_store[il])
    return il, ir, wt
end


# Cubic Hermite interpolant value at psi given endpoint values and derivatives.
function _hermite_cubic_val(u_a, u_b, du_a, du_b, psi_a, psi_b, psi)
    h = psi_b - psi_a
    t = (psi - psi_a) / h
    h00 = 2t^3 - 3t^2 + 1
    h10 = t^3 - 2t^2 + t
    h01 = -2t^3 + 3t^2
    h11 = t^3 - t^2
    return @. h00 * u_a + h * h10 * du_a + h01 * u_b + h * h11 * du_b
end

# Cubic Hermite interpolant DERIVATIVE at psi, consistent with `_hermite_cubic_val`.
function _hermite_cubic_deriv(u_a, u_b, du_a, du_b, psi_a, psi_b, psi)
    h = psi_b - psi_a
    t = (psi - psi_a) / h
    d00 = (6t^2 - 6t) / h
    d10 = 3t^2 - 4t + 1
    d01 = (-6t^2 + 6t) / h
    d11 = 3t^2 - 2t
    return @. d00 * u_a + d10 * du_a + d01 * u_b + d11 * du_b
end

"""
    _chord_solution_at(psi, resnum, odet, nstep) -> (u, du)

Evaluate the `resnum` row of Ξ_ψ and Ξ′_ψ at `psi` from the stored solution:
cubic Hermite for the value, chord slope across the bracketing nodes for the derivative.
The least accurate of the Ξ′ evaluators, kept as the fallback for a solution whose stored
Ξ′ cannot be trusted. Currently unused: every basis that reaches the singular-coupling
loop carries a populated `du_store`, so the gal-native, ideal-EL and kinetic evaluators
cover all of them.
"""
function _chord_solution_at(psi::Float64, resnum::Int, odet::SolutionProfiles, nstep::Int)
    isempty(odet.du_store) && error(
        "_chord_solution_at: no derivative store — the solution carries no usable Ξ′."
    )
    il, ir, _ = _psi_bracket(odet.psi_store, psi, nstep)
    psi_a, psi_b = odet.psi_store[il], odet.psi_store[ir]

    u_a = odet.u_store[resnum, :, 1, il]
    u_b = odet.u_store[resnum, :, 1, ir]
    du_a = odet.du_store[resnum, :, il]
    du_b = odet.du_store[resnum, :, ir]

    u_e = _hermite_cubic_val(u_a, u_b, du_a, du_b, psi_a, psi_b, psi)
    du_e = (u_b .- u_a) ./ (psi_b - psi_a)
    return u_e, du_e
end

"""
    _gal_solution_at(psi, resnum, odet, nstep) -> (u, du)

Evaluate the `resnum` row of Ξ_ψ and Ξ′_ψ at `psi` for a gal-matched solution:
cubic Hermite for the value, and the analytic Ξ′ carried in `du_store` for the
derivative. Mirrors the `galsol%gal_flag` branch of Fortran `gpeq_sol`, which takes
Ξ′ from the analytic galerkin derivative rather than differentiating the value
spline. Differencing `u_store` here would discard that analytic content, and
near-cancellation in `bwp1` (singfac·Ξ′ against n q′·Ξ, with singfac → 0 at the
surface) amplifies the resulting error into Δ′ at the outer surfaces.
"""
function _gal_solution_at(psi::Float64, resnum::Int, odet::SolutionProfiles, nstep::Int)
    il, ir, _ = _psi_bracket(odet.psi_store, psi, nstep)
    psi_a, psi_b = odet.psi_store[il], odet.psi_store[ir]

    u_a = odet.u_store[resnum, :, 1, il]
    u_b = odet.u_store[resnum, :, 1, ir]
    du_a = odet.du_store[resnum, :, il]
    du_b = odet.du_store[resnum, :, ir]

    u_e = _hermite_cubic_val(u_a, u_b, du_a, du_b, psi_a, psi_b, psi)
    du_e = _hermite_cubic_deriv(u_a, u_b, du_a, du_b, psi_a, psi_b, psi)
    return u_e, du_e
end

"""
    _solution_at(psi, psi_surf, resnum, m_res, nn, odet, equil, nstep) -> (u, du)

Evaluate the `resnum` row of Ξ_ψ and Ξ′_ψ at `psi` from the stored ODE solution:
cubic Hermite for the value, Lagrange cubic through the singfac-weighted `du_store`
nodes for the derivative. Works for any formulation; ideal runs use the more accurate
`_el_solution_at`. The Ξ′ stencil stays on `psi`'s side of the singular surface
`psi_surf`, where the stored solution is discontinuous.
"""
function _solution_at(
    psi::Float64,
    psi_surf::Float64,
    resnum::Int,
    m_res::Int,
    nn::Int,
    odet::SolutionProfiles,
    equil::Equilibrium.PlasmaEquilibrium,
    nstep::Int
)
    il, ir, _ = _psi_bracket(odet.psi_store, psi, nstep)
    psi_a, psi_b = odet.psi_store[il], odet.psi_store[ir]

    u_a = odet.u_store[resnum, :, 1, il]
    u_b = odet.u_store[resnum, :, 1, ir]
    du_a = odet.du_store[resnum, :, il]
    du_b = odet.du_store[resnum, :, ir]

    u_e = _hermite_cubic_val(u_a, u_b, du_a, du_b, psi_a, psi_b, psi)

    # Same-side candidate nodes around the bracket, trimmed to the 4 nearest psi.
    side = sign(psi - psi_surf)
    idxs = [j for j in max(1, il-3):min(nstep, ir+3) if sign(odet.psi_store[j] - psi_surf) == side]
    while length(idxs) > 4
        abs(odet.psi_store[idxs[1]] - psi) > abs(odet.psi_store[idxs[end]] - psi) ? popfirst!(idxs) : pop!(idxs)
    end

    # Interpolate singfac·Ξ′ and divide the ~1/singfac pole back out at psi.
    singfac(p) = m_res - nn * equil.profiles.q_spline(p)
    du_e = zeros(ComplexF64, size(odet.du_store, 2))
    for k in eachindex(idxs)
        w = 1.0
        for j in eachindex(idxs)
            j == k && continue
            w *= (psi - odet.psi_store[idxs[j]]) / (odet.psi_store[idxs[k]] - odet.psi_store[idxs[j]])
        end
        du_e .+= (w * singfac(odet.psi_store[idxs[k]])) .* @view(odet.du_store[resnum, :, idxs[k]])
    end
    du_e ./= singfac(psi)

    return u_e, du_e
end

"""
    _el_solution_at(psi, resnum, odet, ffit, equil, ffs, nstep) -> (u, du)

Evaluate the `resnum` row of Ξ_ψ and Ξ′_ψ at `psi` from the stored ODE solution via the
ideal Euler-Lagrange relation Ξ′ = Q⁻¹·F̄⁻¹·(Q⁻¹·u₂ − K̄·u₁) [Glasser 2016 eqs. 22-24],
with u₁, u₂ Hermite-interpolated to `psi`. Only valid for ideal runs where
`ffit.fmats_lower` and `kmats` generated the solution.

The Hermite slopes need du₂ as well as du₁, and du₂ is not stored: both are evaluated here
from the derivative kernel at the two bracketing nodes, which is where the handful of
resonant evaluation points actually need them.
"""
function _el_solution_at(
    psi::Float64,
    resnum::Int,
    odet::SolutionProfiles,
    ffit::FourFitVars,
    equil::Equilibrium.PlasmaEquilibrium,
    ffs::ForceFreeStatesResult,
    nstep::Int
)
    npert = ffs.numpert_total
    il, ir, _ = _psi_bracket(odet.psi_store, psi, nstep)
    psi_a, psi_b = odet.psi_store[il], odet.psi_store[ir]

    u1_a = @view odet.u_store[:, :, 1, il]
    u1_b = @view odet.u_store[:, :, 1, ir]
    u2_a = @view odet.u_store[:, :, 2, il]
    u2_b = @view odet.u_store[:, :, 2, ir]

    # Own hints: this runs inside a threaded loop over rational surfaces.
    hint = Ref(1)
    q_hint = Ref(1)
    du_a = zeros(ComplexF64, npert, npert, 2)
    du_b = zeros(ComplexF64, npert, npert, 2)
    ForceFreeStates.el_derivatives!(du_a, odet.u_store[:, :, :, il], false, equil, ffit, ffs, psi_a, q_hint, hint)
    ForceFreeStates.el_derivatives!(du_b, odet.u_store[:, :, :, ir], false, equil, ffit, ffs, psi_b, q_hint, hint)
    du1_a = @view du_a[:, :, 1]
    du1_b = @view du_b[:, :, 1]
    du2_a = @view du_a[:, :, 2]
    du2_b = @view du_b[:, :, 2]

    kmat = Matrix{ComplexF64}(undef, npert, npert)

    u1_e = _hermite_cubic_val(u1_a, u1_b, du1_a, du1_b, psi_a, psi_b, psi)
    u2_e = _hermite_cubic_val(u2_a, u2_b, du2_a, du2_b, psi_a, psi_b, psi)

    # Ξ′ = Q⁻¹·F̄⁻¹·(Q⁻¹·u₂ − K̄·u₁) with Q⁻¹ = diag(1/(m − n·q))
    q_e = equil.profiles.q_spline(psi)
    singfac_inv = vec([1.0 / (m - q_e * n) for m in ffs.mlow:ffs.mhigh, n in ffs.nlow:ffs.nhigh])
    fmat_lower = Matrix{ComplexF64}(undef, npert, npert)
    ffit.fmats_lower(vec(fmat_lower), psi; hint=hint)
    ffit.kmats(vec(kmat), psi; hint=hint)
    du1_e = u2_e .* singfac_inv
    du1_e .-= kmat * u1_e
    ldiv!(LowerTriangular(fmat_lower), du1_e)
    ldiv!(UpperTriangular(fmat_lower'), du1_e)
    du1_e .*= singfac_inv

    return u1_e[resnum, :], du1_e[resnum, :]
end

"""
    compute_singular_coupling_metrics!(
        state::PerturbedEquilibriumState,
        equil::Equilibrium.PlasmaEquilibrium,
        ForceFreeStates_results::SolutionProfiles,
        mthvac::Int,
        ffs::ForceFreeStatesResult,
        intr::PerturbedEquilibriumInternal,
        ctrl::PerturbedEquilibriumControl,
        ffit::FourFitVars
    )

Compute singular layer coupling matrices and applied resonant vectors.

Matching Fortran GPEC output: `C_f_x_out` (coupling matrix) and `Phi_res`, `w_isl`, etc. (applied vectors).
[Park Phys. Plasmas 2009 056115]

## Modifies

Populates `state` with:

Coupling matrices `[n_rational × numpert_total]` — C[row, j] = coupling when forcing mode j has unit amplitude:

  - `C_resonant_area_weighted_field`, `C_resonant_current`, `C_island_width_sq`, `C_penetrated_area_weighted_field`, `C_delta_prime`

Applied resonant vectors `[n_rational]` = C · amp_vec:

  - `resonant_area_weighted_field`, `resonant_current`, `island_width_sq`, `penetrated_area_weighted_field`, `delta_prime`

Diagnostics `[n_rational]`: `island_half_width`, `chirikov_parameter`

Metadata `[n_rational]`: `rational_psi`, `rational_q`, `rational_m_res`, `rational_n`, `rational_surface_idx`
"""
function compute_singular_coupling_metrics!(
    state::PerturbedEquilibriumState,
    equil::Equilibrium.PlasmaEquilibrium,
    ForceFreeStates_results::SolutionProfiles,
    mthvac::Int,
    ffs::ForceFreeStatesResult,
    intr::PerturbedEquilibriumInternal,
    ctrl::PerturbedEquilibriumControl,
    ffit::FourFitVars
)
    ctrl.verbose && @info "Computing singular coupling metrics (GPEC method)"

    (; numpert_total, mlow, mhigh, nlow, nhigh) = ffs
    msing = length(ffs.surfaces)

    if msing == 0
        ctrl.verbose && @info "No singular surfaces found. Skipping singular coupling calculation."
        return
    end

    if size(intr.plasma_response, 1) == 0
        @warn "Permeability matrix not computed. Skipping singular coupling."
        return
    end

    chi1 = 2π * equil.psio
    twopi = 2π
    mtheta = mthvac

    # Phase 1: Collect all resonant (surface, n) pairs in psi order
    resonant_pairs = Tuple{Int,Int}[]
    for nn in nlow:nhigh
        for s in 1:msing
            m_res_float = ffs.surfaces[s].q * nn
            m_res = round(Int, m_res_float)
            abs(m_res_float - m_res) > 1e-6 && continue
            (m_res < mlow || m_res > mhigh) && continue
            push!(resonant_pairs, (s, nn))
        end
    end

    n_rational = length(resonant_pairs)
    if n_rational == 0
        ctrl.verbose && @info "No resonant surfaces found. Skipping singular coupling."
        return
    end

    ctrl.verbose && @info "Found $n_rational resonant (surface, n) pairs"

    # Phase 2: Allocate output arrays
    state.C_resonant_area_weighted_field = zeros(ComplexF64, n_rational, numpert_total)
    state.C_resonant_current = zeros(ComplexF64, n_rational, numpert_total)
    state.C_island_width_sq = zeros(ComplexF64, n_rational, numpert_total)
    have_inner_bpen = !isempty(intr.inner_bpen)
    if have_inner_bpen
        state.C_penetrated_area_weighted_field = zeros(ComplexF64, n_rational, numpert_total)
    else
        state.C_penetrated_area_weighted_field = zeros(ComplexF64, 0, 0)
        @warn "No inner-layer B_pen supplied; penetrated field not computed." maxlog=1
    end
    state.C_delta_prime = zeros(ComplexF64, n_rational, numpert_total)
    state.rational_psi = zeros(Float64, n_rational)
    state.rational_q = zeros(Float64, n_rational)
    state.rational_m_res = zeros(Int, n_rational)
    state.rational_n = zeros(Int, n_rational)
    state.rational_surface_idx = zeros(Int, n_rational)
    state.rational_area = zeros(Float64, n_rational)

    # Precompute ODE coefficient matrix C_coeffs for all PE forcing modes.
    # For each forcing mode k: c_k = u_bnd⁻¹ × edge_mn_k
    # where edge_mn_k[j] = plasma_response[j,k] / (chi1·singfac_lim[j]·2πi)
    # Matches Fortran gpout_resp: edge_mn = foutmn/(chi1·singfac·twopi·ifac)
    psi_lim = ForceFreeStates_results.psi_store[ForceFreeStates_results.step]
    q_lim = equil.profiles.q_spline(psi_lim)
    singfac_lim = [intr.m_modes[j] - intr.n_modes[j] * q_lim for j in 1:numpert_total]
    u_bnd = ForceFreeStates_results.u_store[:, :, 1, ForceFreeStates_results.step]
    # Divide each row j by singfac_lim[j] — reshape to column vector so Julia broadcasts row-wise, not column-wise.
    edge_mn = intr.plasma_response ./ (chi1 * 2π * im .* reshape(singfac_lim, :, 1))
    C_coeffs = u_bnd \ edge_mn  # mpert × numpert_total

    # Phase 3: Compute full coupling matrix rows. Each rational surface is independent -- it
    # builds its own Green's functions (a fresh, allocation-local vacuum solve) and writes its
    # own disjoint coupling-matrix row (state.C_*[row,:] / state.rational_*[row]) -- so thread
    # the loop over surfaces. Note a single surface `s` can appear on multiple rows in a
    # multi-n run (integer q resonates at several n), so per-row work must not write shared
    # sing[s] state. BLAS is pinned to one thread across the loop (each surface's solve is
    # small; multi-threaded BLAS would oversubscribe against the Julia threads) and restored in
    # a `finally` so an exception in the loop cannot leak the pinned count into the session. The
    # loop is top-level: its threadid()-indexed state must never be nested inside another
    # @threads region.
    nstep = ForceFreeStates_results.step
    # ξ′ evaluation preference: the ideal EL relation, or the interpolated stored RHS for kinetic runs.
    use_el = !ffit.kinetic_populated
    _blas_nthreads = BLAS.get_num_threads()
    BLAS.set_num_threads(1)
    try
        Threads.@threads :static for row in 1:length(resonant_pairs)
            (s, nn) = resonant_pairs[row]
            sing_surf = ffs.surfaces[s]
            m_res = round(Int, sing_surf.q * nn)

            resnum = findfirst(j -> intr.m_modes[j] == m_res && intr.n_modes[j] == nn, 1:numpert_total)
            if resnum === nothing
                @warn "Could not find index for resonant mode (m=$m_res, n=$nn)" maxlog = 1
                continue
            end

            # Surface inductance at this surface for this n (once per pair)
            L_surf = calc_surface_inductance(equil, sing_surf.psifac, mtheta, mlow:mhigh, nn)

            # Only the (m_res, m_res) diagonal element is needed for singflx
            m_idx = m_res - mlow + 1
            L_mm = L_surf[m_idx, m_idx]

            j_c = compute_current_density(equil, sing_surf.psifac)
            area = compute_surface_area(equil, sing_surf.psifac)
            # Matches Fortran gpout_resp: shear = m*dq/dψ / q² = n*dq/dψ / q (since m=n*q).
            # Uses abs(nn) because island_half_width = sqrt(abs(island_width_sq)), so the sign
            # of shear only affects the sign of C_island_width_sq, not the physical island width.
            shear = abs(nn) * sing_surf.q1 / sing_surf.q

            # Evaluate bwp1_mn = ∂b^ψ/∂ψ at lpsi and rpsi using permeability-weighted eigenstates.
            # Matches Fortran gpout_resp: evaluate bwp1_mn at lpsi/rpsi via gpeq_sol
            # spot = 5e-4 matches Fortran default singfac_min
            spot_psi = 5e-4 / (abs(nn) * abs(sing_surf.q1))
            lpsi = sing_surf.psifac - spot_psi
            rpsi = sing_surf.psifac + spot_psi

            # Evaluate u and dξ/dψ at lpsi and rpsi from the stored ODE solution.
            # Branch order mirrors Fortran gpeq_sol: gal-matched solutions take the analytic
            # galerkin Ξ′, ideal runs the EL relation, kinetic runs the stored Ξ′.
            if intr.odet_from_gal
                # interpolate u and the analytic galerkin dξ/dψ carried in du_store
                u_l, ud_l = _gal_solution_at(lpsi, resnum, ForceFreeStates_results, nstep)
                u_r, ud_r = _gal_solution_at(rpsi, resnum, ForceFreeStates_results, nstep)
            elseif use_el
                # interpolate u and evaluate dξ/dψ from the ideal EL relation
                u_l, ud_l = _el_solution_at(lpsi, resnum, ForceFreeStates_results, ffit, equil, ffs, nstep)
                u_r, ud_r = _el_solution_at(rpsi, resnum, ForceFreeStates_results, ffit, equil, ffs, nstep)
            else
                # interpolate u and the stored dξ/dψ, weighted to remove the resonant pole
                u_l, ud_l = _solution_at(lpsi, sing_surf.psifac, resnum, m_res, nn, ForceFreeStates_results, equil, nstep)
                u_r, ud_r = _solution_at(rpsi, sing_surf.psifac, resnum, m_res, nn, ForceFreeStates_results, equil, nstep)
            end

            q_l = equil.profiles.q_spline(lpsi)
            q1_l = equil.profiles.q_deriv(lpsi)
            q_r = equil.profiles.q_spline(rpsi)
            q1_r = equil.profiles.q_deriv(rpsi)
            singfac_l = m_res - nn * q_l
            singfac_r = m_res - nn * q_r

            jump_vec = Vector{ComplexF64}(undef, numpert_total)
            for k in 1:numpert_total
                ck = @view C_coeffs[:, k]
                xsp_l = transpose(u_l) * ck
                xsp1_l = transpose(ud_l) * ck
                xsp_r = transpose(u_r) * ck
                xsp1_r = transpose(ud_r) * ck
                bwp1_l = 2π * im * chi1 * (singfac_l * xsp1_l - nn * q1_l * xsp_l)
                bwp1_r = 2π * im * chi1 * (singfac_r * xsp1_r - nn * q1_r * xsp_r)
                jump_vec[k] = bwp1_r - bwp1_l
            end

            # Inner-layer (cusp-free) penetrated field: bpen[s, j] is linear in the same identity-at-edge
            # coil-drive columns as the outer solution, so it contracts with C_coeffs exactly like
            # the outer solution values above (xsp = transpose(u) * ck); /area matches the area-weighted
            # convention of the pointwise row.
            if have_inner_bpen && s <= size(intr.inner_bpen, 1)
                pen_row = (transpose(C_coeffs) * @view(intr.inner_bpen[s, :])) ./ area
                state.C_penetrated_area_weighted_field[row, :] = pen_row
            end

            # LHS normalization audit (#233) — output scalar coordinate-invariance per row:
            #  - Δ' (1/length): the resonant-surface jump in ∂b^ψ/∂ψ over 2π·χ₁; the tearing index is
            #    coordinate-invariant (its sign/zero-crossing set the stability boundary) [Glasser 2016].
            #  - resonant (shielding) current: j_c already integrates jac·|∇ψ| over the surface, so the
            #    Jacobian weighting is carried inside j_c — no separate area factor needed.
            #  - resonant flux → field: Φ^r/A^r [T], invariant [Park 2008; Pharr 2026].
            state.C_delta_prime[row, :] = jump_vec ./ (twopi * chi1)
            state.C_resonant_current[row, :] = jump_vec .* (-j_c / (twopi * m_res))
            # Matches Fortran gpout_resp: singflx = L·fkaxmn, resonant area-weighted field = singflx/area,
            # islandhwids = 4·singflx/(2π·shear·q·chi1)
            singflx_pre = (L_mm / (twopi * nn)) .* state.C_resonant_current[row, :]
            state.C_resonant_area_weighted_field[row, :] = singflx_pre ./ area
            if abs(shear) > 1e-10
                state.C_island_width_sq[row, :] = (4.0 / (twopi * shear * sing_surf.q * chi1)) .* singflx_pre
            end

            state.rational_psi[row] = sing_surf.psifac
            state.rational_area[row] = area
            state.rational_q[row] = sing_surf.q
            state.rational_m_res[row] = m_res
            state.rational_n[row] = nn
            state.rational_surface_idx[row] = s

            if ctrl.verbose
                dp_diag = real(state.C_delta_prime[row, resnum])
                @info "Row $row: q=$(@sprintf("%.3f", sing_surf.q)), ψ=$(@sprintf("%.3f", sing_surf.psifac)), m=$m_res, n=$nn, Δ'(diag)=$(@sprintf("%.3e", dp_diag))"
            end
        end
    finally
        BLAS.set_num_threads(_blas_nthreads)
    end

    # Phase 4: Apply forcing amplitudes → R = C · Φ_x. The applied resonant scalars are
    # physical, coordinate-invariant quantities, so evaluate them from the flux-space C and
    # the physical forcing Φ_x (bit-identical to pre-conform values).
    forcing_flux = zeros(ComplexF64, numpert_total)
    for mode in intr.forcing_modes
        j = findfirst(k -> intr.m_modes[k] == mode.m && intr.n_modes[k] == mode.n, 1:numpert_total)
        isnothing(j) || (forcing_flux[j] = mode.amplitude)
    end

    state.resonant_area_weighted_field = state.C_resonant_area_weighted_field * forcing_flux
    state.resonant_current = state.C_resonant_current * forcing_flux
    state.island_width_sq = state.C_island_width_sq * forcing_flux
    state.delta_prime = state.C_delta_prime * forcing_flux
    have_inner_bpen && (state.penetrated_area_weighted_field = state.C_penetrated_area_weighted_field * forcing_flux)
    state.forcing_solution_weights = C_coeffs * forcing_flux

    # Conform the stored coupling-matrix input basis to the coordinate-invariant root-area-weighted
    # field (b̃) space (#233 / Pharr 2026): C̃ = C·R, so each stored row acts on the applied field
    # b̃_x (Φ_x = R·b̃_x) and its singular values become coordinate-invariant. The conform operator
    # R = S·A (Σ·√A) is the only place flux briefly appears; it is built from the b̃→b̄ operator S and
    # the scalar surface area A. Done after the applied-vector evaluation above so those physical
    # scalars carry no round-trip noise.
    rootarea_to_area_weight, surface_area = build_control_surface_rootarea_to_area_weight(equil, ffs)
    flux_conform = rootarea_to_area_weight .* surface_area
    state.C_resonant_area_weighted_field = state.C_resonant_area_weighted_field * flux_conform
    state.C_resonant_current = state.C_resonant_current * flux_conform
    state.C_island_width_sq = state.C_island_width_sq * flux_conform
    state.C_delta_prime = state.C_delta_prime * flux_conform
    have_inner_bpen && (state.C_penetrated_area_weighted_field = state.C_penetrated_area_weighted_field * flux_conform)

    # Phase 5: Island diagnostics from applied resonant vectors
    compute_island_diagnostics!(state, n_rational)

    if ctrl.verbose
        max_island = isempty(state.island_half_width) ? 0.0 : maximum(state.island_half_width)
        max_dp = isempty(state.delta_prime) ? 0.0 : maximum(abs.(real.(state.delta_prime)))
        @info "Singular coupling complete. Max island half-width = $(@sprintf("%.3e", max_island)), max |Δ'| = $(@sprintf("%.3e", max_dp))"
    end
end

"""
    compute_current_density(
        equil::Equilibrium.PlasmaEquilibrium,
        psi::Float64
    )::Float64

Compute effective current density coefficient at given flux surface.

Implements GPEC's j_c calculation (Fortran `gpout_respinfo`):
j_c = χ₁² * q / (μ₀ * integral)

where the integral is computed via flux surface integration:
integral = ∫ (jac * |∇ψ| * sqreqb / |∇ψ|³) dθ

## GPEC Formula

```fortran
DO itheta=0,mthsurf
   CALL bicube_eval(rzphi,respsi,theta(itheta),1)
   rfac=SQRT(rzphi%f(1))
   jac=rzphi%f(4)
   w(1,1)=(1+rzphi%fy(2))*twopi**2*rfac*r(itheta)/jac
   w(1,2)=-rzphi%fy(1)*pi*r(itheta)/(rfac*jac)
   delpsi(itheta)=SQRT(w(1,1)**2+w(1,2)**2)
   sqreqb(itheta)=(sq%f(1)**2+chi1**2*delpsi(itheta)**2)/(twopi*r(itheta))**2
   jcfun(itheta)=sqreqb(itheta)/(delpsi(itheta)**3)
   j_c(ising)=j_c(ising)+jac*delpsi(itheta)*jcfun(itheta)/mthsurf
ENDDO
j_c(ising)=j_c(ising)-jac*delpsi(mthsurf)*jcfun(mthsurf)/mthsurf  ! trapezoidal rule
j_c(ising)=1.0/j_c(ising)*chi1**2*sq%f(4)/mu0
```

## Implementation

Uses trapezoidal rule integration around the flux surface with metric quantities
from the equilibrium bicubic spline (rzphi). The integrand includes:

  - jac: Jacobian of flux coordinates
  - |∇ψ|: Flux gradient magnitude (delpsi)
  - sqreqb: Magnetic field quantity (F² + χ₁²|∇ψ|²)/(2πR)² where F = R·B_tor
"""
function compute_current_density(
    equil::Equilibrium.PlasmaEquilibrium,
    psi::Float64
)::Float64
    # Physical constants
    μ₀ = 4π * 1e-7
    chi1 = 2π * equil.psio
    twopi = 2π

    # Get equilibrium quantities at this surface
    F_tor = equil.profiles.F_spline(psi)  # Toroidal field function (2π·R·B_tor in GPEC convention)
    q = equil.profiles.q_spline(psi)      # Safety factor

    ro = equil.ro

    # Number of theta points for integration
    # Match GPEC's mthsurf (typically 101 points from theta=0 to theta=1)
    mthsurf = length(equil.rzphi_ys) - 1

    # Integrate around flux surface using trapezoidal rule
    integral = 0.0

    # Storage for last point (needed for trapezoidal rule correction)
    last_jac = 0.0
    last_delpsi = 0.0
    last_jcfun = 0.0

    hint2d = (Ref(1), Ref(1))  # Shared 2D hint for hot loop optimization
    for itheta in 0:mthsurf
        # Theta coordinate normalized to [0, 1]
        theta = itheta / mthsurf

        m = Equilibrium.flux_surface_metric(equil, psi, theta; hint=hint2d)
        jac = m.jac
        delpsi = m.delpsi  # flux gradient magnitude |∇ψ|

        # sqreqb = (F² + χ₁²|∇ψ|²) / (2πR)²  where F = R·B_tor (Fortran sq%f(1))
        sqreqb = (F_tor^2 + chi1^2 * delpsi^2) / (twopi * m.r)^2

        # Integrand function
        jcfun = sqreqb / (delpsi^3)

        # Accumulate integral (trapezoidal rule)
        integral += jac * delpsi * jcfun / mthsurf

        # Store last point for correction
        if itheta == mthsurf
            last_jac = jac
            last_delpsi = delpsi
            last_jcfun = jcfun
        end
    end

    # Trapezoidal rule end correction (subtract half of last point contribution)
    integral -= last_jac * last_delpsi * last_jcfun / mthsurf

    # Final normalization: j_c = (1/integral) * χ₁² * q / μ₀
    j_c = (1.0 / integral) * chi1^2 * q / μ₀

    return j_c
end

"""
    compute_surface_area(
        equil::Equilibrium.PlasmaEquilibrium,
        psi::Float64
    )::Float64

Compute flux surface area at given ψ.

Implements GPEC's area calculation (Fortran `gpout_respinfo`):
area = ∫ jac * |∇ψ| dθ

where the integral is computed around the flux surface.

## GPEC Formula

```fortran
DO itheta=0,mthsurf
   CALL bicube_eval(rzphi,respsi,theta(itheta),1)
   rfac=SQRT(rzphi%f(1))
   jac=rzphi%f(4)
   w(1,1)=(1+rzphi%fy(2))*twopi**2*rfac*r(itheta)/jac
   w(1,2)=-rzphi%fy(1)*pi*r(itheta)/(rfac*jac)
   delpsi(itheta)=SQRT(w(1,1)**2+w(1,2)**2)
   area(ising)=area(ising)+jac*delpsi(itheta)/mthsurf
ENDDO
area(ising)=area(ising)-jac*delpsi(mthsurf)/mthsurf  ! trapezoidal rule
```

## Implementation

Uses trapezoidal rule integration around the flux surface with:

  - jac: Jacobian of flux coordinates from rzphi
  - |∇ψ|: Flux gradient magnitude (delpsi) from metric tensor
"""
function compute_surface_area(
    equil::Equilibrium.PlasmaEquilibrium,
    psi::Float64
)::Float64
    # mthsurf matches GPEC's flux-surface theta resolution
    mthsurf = length(equil.rzphi_ys) - 1
    return Equilibrium.flux_surface_area(equil, psi, mthsurf)
end

"""
    compute_island_diagnostics!(state::PerturbedEquilibriumState, n_rational::Int)

Compute island half-width and Chirikov parameter from applied resonant vectors.

  - `island_half_width[row]` = √|island_width_sq[row]|
  - `chirikov_parameter[row]` = half-width / (half-distance to nearest neighbor in rational_psi)
"""
function compute_island_diagnostics!(state::PerturbedEquilibriumState, n_rational::Int)
    state.island_half_width = sqrt.(abs.(state.island_width_sq))
    state.chirikov_parameter = zeros(Float64, n_rational)

    n_rational <= 1 && return

    for row in 1:n_rational
        psi_row = state.rational_psi[row]
        min_dist = Inf
        for row2 in 1:n_rational
            row2 == row && continue
            dist = abs(state.rational_psi[row2] - psi_row)
            min_dist = min(min_dist, dist)
        end
        if min_dist > 1e-10
            state.chirikov_parameter[row] = state.island_half_width[row] / (min_dist / 2.0)
        else
            state.chirikov_parameter[row] = Inf
        end
    end
end
