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
The least accurate Ξ′ evaluator; currently unused, kept as a fallback for a solution whose
stored Ξ′ cannot be trusted.
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
cubic Hermite for the value and the analytic Ξ′ carried in `du_store` for the derivative
(the `gal_flag` branch of Fortran `gpeq_sol`). Do not difference `u_store` instead:
near-cancellation in `bwp1` (singfac·Ξ′ against n·q′·Ξ) amplifies that error into Δ′.
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
    _el_solution_at(psi, resnum, odet, mats, equil, ffs, nstep) -> (u, du)

Evaluate the `resnum` row of Ξ_ψ and Ξ′_ψ at `psi` from the stored ODE solution via the
ideal Euler-Lagrange relation Ξ′ = Q⁻¹·F̄⁻¹·(Q⁻¹·u₂ − K̄·u₁) [Glasser 2016 eqs. 22-24],
with u₁, u₂ Hermite-interpolated to `psi`. Only valid for ideal runs where
`mats.ideal.F_spline_lower` and `K_spline` generated the solution.

du₂ is not stored, so the Hermite slopes du₁ and du₂ are both evaluated from the derivative
kernel at the two bracketing nodes.
"""
function _el_solution_at(
    psi::Float64,
    resnum::Int,
    odet::SolutionProfiles,
    mats::MatrixSplines,
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
    ForceFreeStates.el_derivatives!(du_a, odet.u_store[:, :, :, il], false, equil, mats, ffs, psi_a, q_hint, hint)
    ForceFreeStates.el_derivatives!(du_b, odet.u_store[:, :, :, ir], false, equil, mats, ffs, psi_b, q_hint, hint)
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
    mats.ideal.F_spline_lower(vec(fmat_lower), psi; hint=hint)
    mats.ideal.K_spline(vec(kmat), psi; hint=hint)
    du1_e = u2_e .* singfac_inv
    du1_e .-= kmat * u1_e
    ldiv!(LowerTriangular(fmat_lower), du1_e)
    ldiv!(UpperTriangular(fmat_lower'), du1_e)
    du1_e .*= singfac_inv

    return u1_e[resnum, :], du1_e[resnum, :]
end

"""
    build_coupling_operators!(resp; verbose=false)

Build the singular-layer coupling matrices of `resp.ffs` onto `resp`, matching Fortran GPEC's
`C_f_x_out` [Park Phys. Plasmas 2009 056115]: one row per resonant (surface, n) pair, giving the
resonant area-weighted field, current, (w/2)², penetrated field and Δ' per unit forcing mode, plus
the rational-surface metadata and the edge-to-solution coefficients `C_coeffs`. The matrices are
stored acting on b̃_x, with flux-space copies in `resp.coupling_flux` for applying a forcing.

## Normalization

  - Rows: the resonant and penetrated rows are a resonant flux divided by that surface's own area
    `rational_area[row]`, so both are fields in T. `C_resonant_current`, `C_island_width_sq` and
    `C_delta_prime` carry their own normalizations.
  - Columns: each field-space coupling matrix is right-multiplied by the control-surface `flux_conform`, so it
    acts on the root-area-weighted field b̃ in T and its singular values do not depend on `jac_type`.
    Applied vectors use the flux-space copies in `resp.coupling_flux` on the run's own forcing flux.
"""
function build_coupling_operators!(resp::PlasmaResponse; verbose::Bool=false)
    ffs = resp.ffs
    equil = ffs.equil
    solution = ffs.solution
    mats = ffs.mats
    mtheta_vacuum = ffs.control.mtheta_vacuum
    verbose && @info "Computing singular coupling metrics (GPEC method)"

    (; numpert_total, mlow, mhigh, nlow, nhigh) = ffs
    msing = length(ffs.surfaces)

    if msing == 0
        verbose && @info "No singular surfaces found. Skipping singular coupling calculation."
        return
    end

    if isempty(resp.permeability_flux)
        @warn "Permeability matrix not computed. Skipping singular coupling."
        return
    end

    chi1 = 2π * equil.psio
    twopi = 2π
    mtheta = mtheta_vacuum

    # Collect all resonant (surface, n) pairs
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
    # Rows run outward in ψ across every n; an integer-q surface keeps its n order.
    sort!(resonant_pairs; by=((s, nn),) -> (ffs.surfaces[s].psifac, nn))

    n_rational = length(resonant_pairs)
    if n_rational == 0
        verbose && @info "No resonant surfaces found. Skipping singular coupling."
        return
    end

    verbose && @info "Found $n_rational resonant (surface, n) pairs"

    resp.C_resonant_area_weighted_field = zeros(ComplexF64, n_rational, numpert_total)
    resp.C_resonant_current = zeros(ComplexF64, n_rational, numpert_total)
    resp.C_island_width_sq = zeros(ComplexF64, n_rational, numpert_total)
    have_inner_bpen = !isempty(ffs.bpen)
    if have_inner_bpen
        resp.C_penetrated_area_weighted_field = zeros(ComplexF64, n_rational, numpert_total)
    else
        resp.C_penetrated_area_weighted_field = zeros(ComplexF64, 0, 0)
        @warn "No inner-layer B_pen supplied; penetrated field not computed." maxlog=1
    end
    resp.C_delta_prime = zeros(ComplexF64, n_rational, numpert_total)
    resp.rational_psi = zeros(Float64, n_rational)
    resp.rational_q = zeros(Float64, n_rational)
    resp.rational_m_res = zeros(Int, n_rational)
    resp.rational_n = zeros(Int, n_rational)
    resp.rational_surface_idx = zeros(Int, n_rational)
    resp.rational_area = zeros(Float64, n_rational)

    # Edge-to-solution coefficients C_coeffs = u_bnd⁻¹·edge_mn, edge_mn = P/(chi1·singfac_lim·2πi) (Fortran gpout_resp)
    psi_lim = solution.psi_store[solution.step]
    q_lim = equil.profiles.q_spline(psi_lim)
    singfac_lim = [resp.m_modes[j] - resp.n_modes[j] * q_lim for j in 1:numpert_total]
    u_bnd = solution.u_store[:, :, 1, solution.step]
    # Divide each row j by singfac_lim[j] — reshape to column vector so Julia broadcasts row-wise, not column-wise.
    edge_mn = resp.permeability_flux ./ (chi1 * 2π * im .* reshape(singfac_lim, :, 1))
    C_coeffs = u_bnd \ edge_mn  # mpert × numpert_total
    resp.C_coeffs = C_coeffs

    # Rows are independent (own vacuum solve, disjoint output row), so thread them; a surface can own several rows
    # at integer q, so never write per-surface state. BLAS is pinned to one thread and restored in `finally`.
    nstep = solution.step
    # ξ′ evaluation preference: the ideal EL relation, or the interpolated stored RHS for kinetic runs.
    use_el = mats.kinetic === nothing
    _blas_nthreads = BLAS.get_num_threads()
    BLAS.set_num_threads(1)
    try
        Threads.@threads :static for row in 1:length(resonant_pairs)
            (s, nn) = resonant_pairs[row]
            sing_surf = ffs.surfaces[s]
            m_res = round(Int, sing_surf.q * nn)

            resnum = findfirst(j -> resp.m_modes[j] == m_res && resp.n_modes[j] == nn, 1:numpert_total)
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
            # shear = n·q′/q (Fortran gpout_resp); abs(nn) only sets the sign of C_island_width_sq, not the width.
            shear = abs(nn) * sing_surf.q1 / sing_surf.q

            # Evaluate ∂b^ψ/∂ψ either side of the surface; spot 5e-4 matches the Fortran singfac_min default.
            spot_psi = 5e-4 / (abs(nn) * abs(sing_surf.q1))
            lpsi = sing_surf.psifac - spot_psi
            rpsi = sing_surf.psifac + spot_psi

            # Ξ, Ξ′ at lpsi/rpsi as in Fortran gpeq_sol: gal-matched → analytic Ξ′, ideal → EL relation, kinetic → stored Ξ′.
            if resp.odet_from_gal
                u_l, ud_l = _gal_solution_at(lpsi, resnum, solution, nstep)
                u_r, ud_r = _gal_solution_at(rpsi, resnum, solution, nstep)
            elseif use_el
                u_l, ud_l = _el_solution_at(lpsi, resnum, solution, mats, equil, ffs, nstep)
                u_r, ud_r = _el_solution_at(rpsi, resnum, solution, mats, equil, ffs, nstep)
            else
                u_l, ud_l = _solution_at(lpsi, sing_surf.psifac, resnum, m_res, nn, solution, equil, nstep)
                u_r, ud_r = _solution_at(rpsi, sing_surf.psifac, resnum, m_res, nn, solution, equil, nstep)
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

            # Inner-layer penetrated field bpen[s, :] contracts with C_coeffs like the outer solution; /area makes it area-weighted.
            if have_inner_bpen && s <= size(ffs.bpen, 1)
                pen_row = (transpose(C_coeffs) * @view(ffs.bpen[s, :])) ./ area
                resp.C_penetrated_area_weighted_field[row, :] = pen_row
            end

            # Δ' is the ∂b^ψ/∂ψ jump over 2π·χ₁ [Glasser 2016]; j_c carries the surface Jacobian weighting of the current;
            # resonant flux/area is a coordinate-invariant field in T [Park 2008; Pharr 2026].
            resp.C_delta_prime[row, :] = jump_vec ./ (twopi * chi1)
            resp.C_resonant_current[row, :] = jump_vec .* (-j_c / (twopi * m_res))
            # Matches Fortran gpout_resp: singflx = L·fkaxmn, resonant area-weighted field = singflx/area,
            # islandhwids = 4·singflx/(2π·shear·q·chi1)
            singflx_pre = (L_mm / (twopi * nn)) .* resp.C_resonant_current[row, :]
            resp.C_resonant_area_weighted_field[row, :] = singflx_pre ./ area
            if abs(shear) > 1e-10
                resp.C_island_width_sq[row, :] = (4.0 / (twopi * shear * sing_surf.q * chi1)) .* singflx_pre
            end

            resp.rational_psi[row] = sing_surf.psifac
            resp.rational_area[row] = area
            resp.rational_q[row] = sing_surf.q
            resp.rational_m_res[row] = m_res
            resp.rational_n[row] = nn
            resp.rational_surface_idx[row] = s

            if verbose
                dp_diag = real(resp.C_delta_prime[row, resnum])
                @info "Row $row: q=$(@sprintf("%.3f", sing_surf.q)), ψ=$(@sprintf("%.3f", sing_surf.psifac)), m=$m_res, n=$nn, Δ'(diag)=$(@sprintf("%.3e", dp_diag))"
            end
        end
    finally
        BLAS.set_num_threads(_blas_nthreads)
    end

    # Conform the stored matrices to b̃ space, C̃ = C·R with R = S·A (Pharr 2026); `coupling_flux` keeps the flux-space
    # copies so a forcing is applied without round-trip noise.
    resp.coupling_flux = (; resonant_area_weighted_field=resp.C_resonant_area_weighted_field, resonant_current=resp.C_resonant_current,
        island_width_sq=resp.C_island_width_sq, penetrated_area_weighted_field=resp.C_penetrated_area_weighted_field, delta_prime=resp.C_delta_prime)
    flux_conform = resp.rootarea_to_area_weight .* resp.surface_area
    resp.C_resonant_area_weighted_field = resp.C_resonant_area_weighted_field * flux_conform
    resp.C_resonant_current = resp.C_resonant_current * flux_conform
    resp.C_island_width_sq = resp.C_island_width_sq * flux_conform
    resp.C_delta_prime = resp.C_delta_prime * flux_conform
    have_inner_bpen && (resp.C_penetrated_area_weighted_field = resp.C_penetrated_area_weighted_field * flux_conform)

    return resp
end

"""
    apply_coupling!(state; verbose=false)

Apply the coupling matrices of `state.response` to the forcing on `state`: the applied resonant vectors
(Fortran `Phi_res`, `w_isl`, `K_isl`, `Delta`), the forcing's solution weights, and the island
half-widths and Chirikov overlap.
"""
function apply_coupling!(state::PerturbedEquilibriumState; verbose::Bool=false)
    resp = state.response
    C = resp.coupling_flux
    numpert_total = resp.ffs.numpert_total

    # The applied resonant scalars are coordinate-invariant, so evaluate them from the flux-space C and Φ_x.
    forcing_flux = zeros(ComplexF64, numpert_total)
    for mode in state.forcing_modes
        j = findfirst(k -> resp.m_modes[k] == mode.m && resp.n_modes[k] == mode.n, 1:numpert_total)
        isnothing(j) || (forcing_flux[j] = mode.amplitude)
    end

    state.resonant_area_weighted_field = C.resonant_area_weighted_field * forcing_flux
    state.resonant_current = C.resonant_current * forcing_flux
    state.island_width_sq = C.island_width_sq * forcing_flux
    state.delta_prime = C.delta_prime * forcing_flux
    isempty(C.penetrated_area_weighted_field) || (state.penetrated_area_weighted_field = C.penetrated_area_weighted_field * forcing_flux)
    state.forcing_solution_weights = resp.C_coeffs * forcing_flux

    # Island diagnostics from the applied resonant vectors
    state.island_half_width, state.chirikov_parameter = island_diagnostics(state.island_width_sq, resp.rational_psi, resp.rational_surface_idx)

    if verbose
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

Effective current density coefficient `j_c = χ₁²·q / (μ₀·∮ J·|∇ψ|·sqreqb/|∇ψ|³ dθ)` at `psi`, with
`sqreqb = (F² + χ₁²|∇ψ|²)/(2πR)²` and `F` the equilibrium `F_spline`, by the trapezoidal rule on the
equilibrium θ grid. Port of the `j_c` calculation in Fortran `gpout_respinfo`.
"""
function compute_current_density(
    equil::Equilibrium.PlasmaEquilibrium,
    psi::Float64
)::Float64
    μ₀ = 4π * 1e-7
    chi1 = 2π * equil.psio
    twopi = 2π

    F_tor = equil.profiles.F_spline(psi)  # Toroidal field function (2π·R·B_tor in GPEC convention)
    q = equil.profiles.q_spline(psi)      # Safety factor

    ro = equil.ro

    # mthsurf intervals over θ ∈ [0, 1], matching Fortran
    mthsurf = length(equil.rzphi_ys) - 1

    # Trapezoidal rule around the flux surface
    integral = 0.0

    # Storage for last point (needed for trapezoidal rule correction)
    last_jac = 0.0
    last_delpsi = 0.0
    last_jcfun = 0.0

    hint2d = (Ref(1), Ref(1))  # Shared 2D hint for hot loop optimization
    for itheta in 0:mthsurf
        theta = itheta / mthsurf

        m = Equilibrium.flux_surface_metric(equil, psi, theta; hint=hint2d)
        jac = m.jac
        delpsi = m.delpsi  # flux gradient magnitude |∇ψ|

        # F_tor is Fortran sq%f(1)
        sqreqb = (F_tor^2 + chi1^2 * delpsi^2) / (twopi * m.r)^2

        jcfun = sqreqb / (delpsi^3)

        integral += jac * delpsi * jcfun / mthsurf

        if itheta == mthsurf
            last_jac = jac
            last_delpsi = delpsi
            last_jcfun = jcfun
        end
    end

    # Drop the duplicated periodic endpoint θ = 1
    integral -= last_jac * last_delpsi * last_jcfun / mthsurf

    j_c = (1.0 / integral) * chi1^2 * q / μ₀

    return j_c
end

"""
    compute_surface_area(
        equil::Equilibrium.PlasmaEquilibrium,
        psi::Float64
    )::Float64

Flux-surface area `∮ J·|∇ψ| dθ` at `psi` on the equilibrium θ grid (the area of Fortran `gpout_respinfo`).
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
    island_diagnostics(island_width_sq, rational_psi, rational_surface_idx) -> (island_half_width, chirikov_parameter)

Island half-width and Chirikov parameter from the applied (w/2)² on each rational row.

  - `island_half_width[row]` = √|island_width_sq[row]|
  - `chirikov_parameter[row]` = half-width / (half-distance to the nearest other rational surface);
    rows on the same surface at another n (integer q) are not neighbors
"""
function island_diagnostics(island_width_sq::AbstractVector, rational_psi::AbstractVector{Float64}, rational_surface_idx::AbstractVector{Int})
    island_half_width = sqrt.(abs.(island_width_sq))
    n_rational = length(rational_psi)
    chirikov_parameter = zeros(Float64, n_rational)
    n_rational <= 1 && return island_half_width, chirikov_parameter

    for row in 1:n_rational
        psi_row = rational_psi[row]
        min_dist = Inf
        for row2 in 1:n_rational
            rational_surface_idx[row2] == rational_surface_idx[row] && continue
            dist = abs(rational_psi[row2] - psi_row)
            min_dist = min(min_dist, dist)
        end
        chirikov_parameter[row] = min_dist > 1e-10 ? island_half_width[row] / (min_dist / 2.0) : Inf
    end
    return island_half_width, chirikov_parameter
end
