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

"""
    compute_singular_coupling_metrics!(
        state::PerturbedEquilibriumState,
        equil::Equilibrium.PlasmaEquilibrium,
        ForceFreeStates_results::OdeState,
        vac_data::VacuumData,
        ffs_intr::ForceFreeStatesInternal,
        intr::PerturbedEquilibriumInternal,
        ctrl::PerturbedEquilibriumControl
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
    ForceFreeStates_results::OdeState,
    vac_data::VacuumData,
    ffs_intr::ForceFreeStatesInternal,
    intr::PerturbedEquilibriumInternal,
    ctrl::PerturbedEquilibriumControl
)
    ctrl.verbose && @info "Computing singular coupling metrics (GPEC method)"

    (; msing, numpert_total, mlow, mhigh, nlow, nhigh) = ffs_intr

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
    mtheta = vac_data.mthvac
    wall_settings = Vacuum.WallShapeSettings(; shape="nowall")

    # Phase 1: Collect all resonant (surface, n) pairs in psi order
    resonant_pairs = Tuple{Int,Int}[]
    for nn in nlow:nhigh
        for s in 1:msing
            m_res_float = ffs_intr.sing[s].q * nn
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
    state.C_penetrated_area_weighted_field = zeros(ComplexF64, n_rational, numpert_total)
    state.C_delta_prime = zeros(ComplexF64, n_rational, numpert_total)
    state.rational_psi = zeros(Float64, n_rational)
    state.rational_q = zeros(Float64, n_rational)
    state.rational_m_res = zeros(Int, n_rational)
    state.rational_n = zeros(Int, n_rational)
    state.rational_surface_idx = zeros(Int, n_rational)

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
    psi_store_all = ForceFreeStates_results.psi_store
    nstep = ForceFreeStates_results.step
    _blas_nthreads = BLAS.get_num_threads()
    BLAS.set_num_threads(1)
    try
        Threads.@threads :static for row in 1:length(resonant_pairs)
            (s, nn) = resonant_pairs[row]
            sing_surf = ffs_intr.sing[s]
            m_res = round(Int, sing_surf.q * nn)

            resnum = findfirst(j -> intr.m_modes[j] == m_res && intr.n_modes[j] == nn, 1:numpert_total)
            if resnum === nothing
                @warn "Could not find index for resonant mode (m=$m_res, n=$nn)" maxlog=1
                continue
            end

            # Surface-current matrix at this surface for this n (once per pair)
            vac_input = Vacuum.VacuumInput(equil, sing_surf.psifac, mtheta, 1, mlow:mhigh, [nn])
            _, I_v, _, _ = Vacuum.compute_vacuum_response(vac_input, wall_settings; compute_L=true)

            # Precompute L_surf; only the (m_res, m_res) diagonal element is needed for singflx
            L_surf = calc_surface_inductance(I_v)
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

            # Interpolate u and du/dψ at lpsi and rpsi using stored ODE solution.
            # Value (u): Hermite cubic using ud_store for smoother interpolation than linear.
            # Derivative (ud): chord slope from u_store only — see comment below.
            il_l, ir_l, _ = _psi_bracket(psi_store_all, lpsi, nstep)
            il_r, ir_r, _ = _psi_bracket(psi_store_all, rpsi, nstep)

            u_node = ForceFreeStates_results.u_store
            ud_node = ForceFreeStates_results.ud_store
            ua_l = u_node[resnum, :, 1, il_l]
            ub_l = u_node[resnum, :, 1, ir_l]
            ua_r = u_node[resnum, :, 1, il_r]
            ub_r = u_node[resnum, :, 1, ir_r]
            dua_l = ud_node[resnum, :, 1, il_l]
            dub_l = ud_node[resnum, :, 1, ir_l]
            dua_r = ud_node[resnum, :, 1, il_r]
            dub_r = ud_node[resnum, :, 1, ir_r]

            psi_il_l = psi_store_all[il_l]
            psi_ir_l = psi_store_all[ir_l]
            psi_il_r = psi_store_all[il_r]
            psi_ir_r = psi_store_all[ir_r]

            u_l = _hermite_cubic_val(ua_l, ub_l, dua_l, dub_l, psi_il_l, psi_ir_l, lpsi)
            u_r = _hermite_cubic_val(ua_r, ub_r, dua_r, dub_r, psi_il_r, psi_ir_r, rpsi)
            # Derivative (ud): chord slope from u_store only — ud_store can be systematically off near
            # outer surfaces (q=4/5) where the ODE solution varies rapidly; near-cancellation in bwp1
            # then amplifies even a ~10% ud_store error into a large Delta' error. Chord slope avoids
            # this by using only u values, which are accurately stored by the ODE integrator.
            ud_l = (ub_l .- ua_l) ./ (psi_ir_l - psi_il_l)
            ud_r = (ub_r .- ua_r) ./ (psi_ir_r - psi_il_r)

            q_l = equil.profiles.q_spline(lpsi)
            q1_l = equil.profiles.q_deriv(lpsi)
            q_r = equil.profiles.q_spline(rpsi)
            q1_r = equil.profiles.q_deriv(rpsi)
            singfac_l = m_res - nn * q_l
            singfac_r = m_res - nn * q_r

            jump_vec = Vector{ComplexF64}(undef, numpert_total)
            for k in 1:numpert_total
                ck = @view C_coeffs[:, k]
                xsp_l = dot(u_l, ck)
                xsp1_l = dot(ud_l, ck)
                xsp_r = dot(u_r, ck)
                xsp1_r = dot(ud_r, ck)
                bwp1_l = 2π * im * chi1 * (singfac_l * xsp1_l - nn * q1_l * xsp_l)
                bwp1_r = 2π * im * chi1 * (singfac_r * xsp1_r - nn * q1_r * xsp_r)
                jump_vec[k] = bwp1_r - bwp1_l
                # C_penetrated_area_weighted_field: midpoint of b^ψ at lpsi/rpsi divided by the scalar surface area.
                # Matches Fortran gpout_resp: gpeq_interp_singsurf evaluates bwp_mn at respsi.
                # LHS normalization audit (#233): the resonant flux Φ^r divided by the scalar area A^r
                # is a genuine field amplitude in tesla and is coordinate-invariant [Pharr 2026; cf.
                # the resonant-field definition in the Conventions Reference].
                b_l = chi1 * singfac_l * 2π * im * xsp_l
                b_r = chi1 * singfac_r * 2π * im * xsp_r
                state.C_penetrated_area_weighted_field[row, k] = (b_l + b_r) / 2 / area
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
    state.penetrated_area_weighted_field = state.C_penetrated_area_weighted_field * forcing_flux
    state.delta_prime = state.C_delta_prime * forcing_flux

    # Conform the stored coupling-matrix input basis to the coordinate-invariant root-area-weighted
    # field (b̃) space (#233 / Pharr 2026): C̃ = C·R, so each stored row acts on the applied field
    # b̃_x (Φ_x = R·b̃_x) and its singular values become coordinate-invariant. The conform operator
    # R = S·A (Σ·√A) is the only place flux briefly appears; it is built from the b̃→b̄ operator S and
    # the scalar surface area A. Done after the applied-vector evaluation above so those physical
    # scalars carry no round-trip noise.
    rootarea_to_area_weight, surface_area = build_control_surface_rootarea_to_area_weight(equil, ffs_intr)
    flux_conform = rootarea_to_area_weight .* surface_area
    state.C_resonant_area_weighted_field = state.C_resonant_area_weighted_field * flux_conform
    state.C_resonant_current = state.C_resonant_current * flux_conform
    state.C_island_width_sq = state.C_island_width_sq * flux_conform
    state.C_penetrated_area_weighted_field = state.C_penetrated_area_weighted_field * flux_conform
    state.C_delta_prime = state.C_delta_prime * flux_conform

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
    calc_surface_inductance(I_v::AbstractMatrix{ComplexF64})::Matrix{ComplexF64}

Invert the vacuum surface-current matrix into surface inductance `L`.

Vacuum returns `I_v` from the Green’s functions, this helper applies GPEC normalization,
regularization, inversion, and Hermitianization.
"""
function calc_surface_inductance(I_v::Matrix{ComplexF64})::Matrix{ComplexF64}

    mpert = size(I_v, 1)
    μ₀ = 4π * 1e-7
    L_surf = zeros(ComplexF64, mpert, mpert)

    current_mag = maximum(abs.(I_v))

    if current_mag < 1e-15
        @warn "Current matrix is all zeros! Cannot compute surface inductance." maxlog=1
        for i in 1:mpert
            L_surf[i, i] = μ₀ * 1e-6
        end
    else
        try
            I_v ./= μ₀ * (2π)^2
            I_v += 1e-12 * current_mag * I
            # L = flux * inv(Iᵛ) with flux = I
            L_surf = inv(I_v)
            # Hermitianize (matches Fortran: temp1 = 0.5*(temp1 + CONJG(TRANSPOSE(temp1))))
            L_surf = 0.5 * (L_surf + L_surf')
        catch e
            @warn "Surface inductance inversion failed: $e" maxlog=1
            for i in 1:mpert
                L_surf[i, i] = μ₀ * 1e-6
            end
        end
    end

    return L_surf
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
