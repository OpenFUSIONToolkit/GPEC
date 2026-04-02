"""
Singular surface coupling and field calculations.

Implements GPEC's singular surface analysis (gpout.f lines 585-665, 1700-1810):
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
    h00 = 2t^3 - 3t^2 + 1;  h10 = t^3 - 2t^2 + t
    h01 = -2t^3 + 3t^2;     h11 = t^3 - t^2
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
  - `C_resonant_flux`, `C_resonant_current`, `C_island_width_sq`, `C_penetrated_field`, `C_delta_prime`

  Applied resonant vectors `[n_rational]` = C · amp_vec:
  - `resonant_flux`, `resonant_current`, `island_width_sq`, `penetrated_field`, `delta_prime`

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

    msing = ffs_intr.msing
    mpert = ffs_intr.mpert
    numpert_total = ffs_intr.numpert_total

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
    mlow = ffs_intr.mlow
    nlow = ffs_intr.nlow
    nhigh = ffs_intr.nhigh
    wall_settings = Vacuum.WallShapeSettings(; shape="nowall")

    # Phase 1: Collect all resonant (surface, n) pairs in psi order
    resonant_pairs = Tuple{Int,Int}[]
    for nn in nlow:nhigh
        for s in 1:msing
            m_res_float = ffs_intr.sing[s].q * nn
            m_res = round(Int, m_res_float)
            abs(m_res_float - m_res) > 1e-6 && continue
            (m_res < ffs_intr.mlow || m_res > ffs_intr.mhigh) && continue
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
    state.C_resonant_flux      = zeros(ComplexF64, n_rational, numpert_total)
    state.C_resonant_current   = zeros(ComplexF64, n_rational, numpert_total)
    state.C_island_width_sq    = zeros(ComplexF64, n_rational, numpert_total)
    state.C_penetrated_field   = zeros(ComplexF64, n_rational, numpert_total)
    state.C_delta_prime        = zeros(ComplexF64, n_rational, numpert_total)
    state.rational_psi         = zeros(Float64, n_rational)
    state.rational_q           = zeros(Float64, n_rational)
    state.rational_m_res       = zeros(Int, n_rational)
    state.rational_n           = zeros(Int, n_rational)
    state.rational_surface_idx = zeros(Int, n_rational)

    # Precompute ODE coefficient matrix C_coeffs for all PE forcing modes.
    # For each forcing mode k: c_k = u_bnd⁻¹ × edge_mn_k
    # where edge_mn_k[j] = plasma_response[j,k] / (chi1·singfac_lim[j]·2πi)
    # Matches Fortran gpout.f: edge_mn = foutmn/(chi1·singfac·twopi·ifac)  (line 604)
    psi_lim = ForceFreeStates_results.psi_store[ForceFreeStates_results.step]
    q_lim   = equil.profiles.q_spline(psi_lim)
    singfac_lim = [intr.m_modes[j] - intr.n_modes[j] * q_lim for j in 1:numpert_total]
    u_bnd   = ForceFreeStates_results.u_store[:, :, 1, ForceFreeStates_results.step]
    # Divide each row j by singfac_lim[j] — reshape to column vector so Julia broadcasts row-wise, not column-wise.
    edge_mn = intr.plasma_response ./ (chi1 * 2π * im .* reshape(singfac_lim, :, 1))
    C_coeffs = u_bnd \ edge_mn  # mpert × numpert_total

    # Phase 3: Compute full coupling matrix rows
    psi_store_all = ForceFreeStates_results.psi_store
    nstep = ForceFreeStates_results.step
    for (row, (s, nn)) in enumerate(resonant_pairs)
        sing_surf = ffs_intr.sing[s]
        m_res = round(Int, sing_surf.q * nn)

        resnum = findfirst(j -> intr.m_modes[j] == m_res && intr.n_modes[j] == nn, 1:numpert_total)
        if resnum === nothing
            @warn "Could not find index for resonant mode (m=$m_res, n=$nn)" maxlog=1
            continue
        end

        # Compute Green's functions at this surface for this n (once per pair)
        vac_input = Vacuum.VacuumInput(equil, sing_surf.psifac, mtheta, 1, mpert, mlow, 1, nn)
        _, grri_raw, grre_raw, _, _ = Vacuum.compute_vacuum_response(vac_input, wall_settings; green_only=true)
        grri = Matrix{Float64}(grri_raw)
        grre = Matrix{Float64}(grre_raw)
        ffs_intr.sing[s].grri = grri
        ffs_intr.sing[s].grre = grre

        # Get ν on the vacuum theta grid (same ν used in the vacuum Fourier basis computation)
        ν_vac = Vacuum.PlasmaGeometry(vac_input).ν

        # Precompute L_surf; only the (m_res, m_res) diagonal element is needed for singflx
        L_surf = compute_surface_inductance_from_greens(grri, grre, ffs_intr, nn, ν_vac)
        m_idx = m_res - mlow + 1
        L_mm = L_surf[m_idx, m_idx]

        j_c   = compute_current_density(equil, sing_surf.psifac)
        area  = compute_surface_area(equil, sing_surf.psifac)
        # Fortran: shear = mfac(resnum)*dq/dψ / q² = n*dq/dψ / q  (gpout.f line 579)
        shear = abs(nn) * sing_surf.q1 / sing_surf.q

        # Evaluate bwp1_mn = ∂b^ψ/∂ψ at lpsi and rpsi using permeability-weighted eigenstates.
        # Fortran: gpeq_sol(lpsi/rpsi) → bwp1_mn(resnum)  (gpout.f lines 618–624)
        # spot = 5e-4 matches Fortran default (gpec.f line 120)
        spot_psi = 5e-4 / (abs(nn) * abs(sing_surf.q1))
        lpsi = sing_surf.psifac - spot_psi
        rpsi = sing_surf.psifac + spot_psi

        # Interpolate u and du/dψ at lpsi and rpsi using stored ODE solution.
        # Value (u): Hermite cubic using ud_store for smoother interpolation than linear.
        # Derivative (ud): 2-point central diff of u_store only — avoids ud_store, which
        #   can be in the wrong basis at Gaussian reduction steps and from rejected ODE steps
        #   near outer surfaces (where the solution varies rapidly).
        il_l, ir_l, _ = _psi_bracket(psi_store_all, lpsi, nstep)
        il_r, ir_r, _ = _psi_bracket(psi_store_all, rpsi, nstep)

        u_node  = ForceFreeStates_results.u_store
        ud_node = ForceFreeStates_results.ud_store
        ua_l = u_node[ resnum, :, 1, il_l];  ub_l = u_node[ resnum, :, 1, ir_l]
        ua_r = u_node[ resnum, :, 1, il_r];  ub_r = u_node[ resnum, :, 1, ir_r]
        dua_l = ud_node[resnum, :, 1, il_l]; dub_l = ud_node[resnum, :, 1, ir_l]
        dua_r = ud_node[resnum, :, 1, il_r]; dub_r = ud_node[resnum, :, 1, ir_r]

        psi_il_l = psi_store_all[il_l];  psi_ir_l = psi_store_all[ir_l]
        psi_il_r = psi_store_all[il_r];  psi_ir_r = psi_store_all[ir_r]

        u_l  = _hermite_cubic_val(ua_l, ub_l, dua_l, dub_l, psi_il_l, psi_ir_l, lpsi)
        u_r  = _hermite_cubic_val(ua_r, ub_r, dua_r, dub_r, psi_il_r, psi_ir_r, rpsi)
        ud_l = (ub_l .- ua_l) ./ (psi_ir_l - psi_il_l)
        ud_r = (ub_r .- ua_r) ./ (psi_ir_r - psi_il_r)

        q_l = equil.profiles.q_spline(lpsi);  q1_l = equil.profiles.q_deriv(lpsi)
        q_r = equil.profiles.q_spline(rpsi);  q1_r = equil.profiles.q_deriv(rpsi)
        singfac_l = m_res - nn * q_l
        singfac_r = m_res - nn * q_r

        jump_vec = Vector{ComplexF64}(undef, numpert_total)
        for k in 1:numpert_total
            ck    = @view C_coeffs[:, k]
            xsp_l  = dot(u_l,  ck);  xsp1_l = dot(ud_l, ck)
            xsp_r  = dot(u_r,  ck);  xsp1_r = dot(ud_r, ck)
            bwp1_l = 2π * im * chi1 * (singfac_l * xsp1_l - nn * q1_l * xsp_l)
            bwp1_r = 2π * im * chi1 * (singfac_r * xsp1_r - nn * q1_r * xsp_r)
            jump_vec[k] = bwp1_r - bwp1_l
            # C_penetrated_field: midpoint of b^ψ at lpsi/rpsi divided by area.
            # Fortran: gpeq_interp_singsurf/sol evaluates bwp_mn spline at respsi  (gpout.f line 637)
            b_l = chi1 * singfac_l * 2π * im * xsp_l
            b_r = chi1 * singfac_r * 2π * im * xsp_r
            state.C_penetrated_field[row, k] = (b_l + b_r) / 2 / area
        end

        state.C_delta_prime[row, :]      = jump_vec ./ (twopi * chi1)
        state.C_resonant_current[row, :] = jump_vec .* (-j_c / (twopi * m_res))
        # singflx_pre = L_mm * singcurs / (twopi*nn)  (gpout.f line 632: singflx_mn = L * fkaxmn)
        # Phi_res = singflx_pre / area  (gpout.f line 639: singbnoflxs = singflx_mn / area)
        # island_width_sq uses singflx_pre WITHOUT /area  (gpout.f line 640-641: islandhwids = 4*singflx_mn/...)
        singflx_pre = (L_mm / (twopi * nn)) .* state.C_resonant_current[row, :]
        state.C_resonant_flux[row, :] = singflx_pre ./ area
        if abs(shear) > 1e-10
            state.C_island_width_sq[row, :] = (4.0 / (twopi * shear * sing_surf.q * chi1)) .* singflx_pre
        end

        state.rational_psi[row]          = sing_surf.psifac
        state.rational_q[row]            = sing_surf.q
        state.rational_m_res[row]        = m_res
        state.rational_n[row]            = nn
        state.rational_surface_idx[row]  = s

        if ctrl.verbose
            dp_diag = real(state.C_delta_prime[row, resnum])
            @info "Row $row: q=$(@sprintf("%.3f", sing_surf.q)), ψ=$(@sprintf("%.3f", sing_surf.psifac)), m=$m_res, n=$nn, Δ'(diag)=$(@sprintf("%.3e", dp_diag))"
        end
    end

    # Phase 4: Apply forcing amplitudes → R = C · amp_vec
    amp_vec = zeros(ComplexF64, numpert_total)
    for mode in intr.forcing_modes
        j = findfirst(k -> intr.m_modes[k] == mode.m && intr.n_modes[k] == mode.n, 1:numpert_total)
        isnothing(j) || (amp_vec[j] = mode.amplitude)
    end

    state.resonant_flux    = state.C_resonant_flux    * amp_vec
    state.resonant_current = state.C_resonant_current * amp_vec
    state.island_width_sq  = state.C_island_width_sq  * amp_vec
    state.penetrated_field = state.C_penetrated_field * amp_vec
    state.delta_prime      = state.C_delta_prime      * amp_vec

    # Phase 5: Island diagnostics from applied resonant vectors
    compute_island_diagnostics!(state, n_rational)

    if ctrl.verbose
        max_island = isempty(state.island_half_width) ? 0.0 : maximum(state.island_half_width)
        max_dp     = isempty(state.delta_prime)       ? 0.0 : maximum(abs.(real.(state.delta_prime)))
        @info "Singular coupling complete. Max island half-width = $(@sprintf("%.3e", max_island)), max |Δ'| = $(@sprintf("%.3e", max_dp))"
    end
end

"""
    interpolate_field_derivative(
        ForceFreeStates_results::OdeState,
        psi::Float64,
        mode_idx::Int,
        forcing_idx::Int
    )::ComplexF64

Interpolate field derivative ∂b/∂ψ at arbitrary flux coordinate.

Uses linear interpolation of stored ForceFreeStates solution.
"""
function interpolate_field_derivative(
    ForceFreeStates_results::OdeState,
    psi::Float64,
    mode_idx::Int,
    forcing_idx::Int
)::ComplexF64
    # Find bracketing points in psi_store
    psi_store = ForceFreeStates_results.psi_store[1:ForceFreeStates_results.step]

    # Find indices that bracket psi
    idx_right = findfirst(p -> p >= psi, psi_store)

    if idx_right === nothing
        # psi is beyond stored range, use last value
        return ForceFreeStates_results.ud_store[mode_idx, forcing_idx, 1, ForceFreeStates_results.step]
    elseif idx_right == 1
        # psi is before stored range, use first value
        return ForceFreeStates_results.ud_store[mode_idx, forcing_idx, 1, 1]
    else
        # Interpolate between idx_left and idx_right
        idx_left = idx_right - 1

        psi_left = psi_store[idx_left]
        psi_right = psi_store[idx_right]
        weight = (psi - psi_left) / (psi_right - psi_left)

        val_left = ForceFreeStates_results.ud_store[mode_idx, forcing_idx, 1, idx_left]
        val_right = ForceFreeStates_results.ud_store[mode_idx, forcing_idx, 1, idx_right]

        return val_left * (1.0 - weight) + val_right * weight
    end
end

"""
    interpolate_field_at_surface(
        ForceFreeStates_results::OdeState,
        psi::Float64,
        mode_idx::Int,
        forcing_idx::Int,
        equil::Equilibrium.PlasmaEquilibrium,
        m_mode::Int,
        n_mode::Int
    )::ComplexF64

Interpolate magnetic field at arbitrary flux surface.

Interpolates displacement from ForceFreeStates solution and converts to field using
ideal MHD relation from flux coordinates:
b^ψ = i * χ₁ * (m - n*q) * ξ_ψ

This matches the field reconstruction in FieldReconstruction.jl.
"""
function interpolate_field_at_surface(
    ForceFreeStates_results::OdeState,
    psi::Float64,
    mode_idx::Int,
    forcing_idx::Int,
    equil::Equilibrium.PlasmaEquilibrium,
    m_mode::Int,
    n_mode::Int
)::ComplexF64
    # Get displacement at this surface via interpolation
    psi_store = ForceFreeStates_results.psi_store[1:ForceFreeStates_results.step]

    idx_right = findfirst(p -> p >= psi, psi_store)

    if idx_right === nothing
        xi_psi = ForceFreeStates_results.u_store[mode_idx, forcing_idx, 1, ForceFreeStates_results.step]
    elseif idx_right == 1
        xi_psi = ForceFreeStates_results.u_store[mode_idx, forcing_idx, 1, 1]
    else
        idx_left = idx_right - 1
        psi_left = psi_store[idx_left]
        psi_right = psi_store[idx_right]
        weight = (psi - psi_left) / (psi_right - psi_left)

        val_left = ForceFreeStates_results.u_store[mode_idx, forcing_idx, 1, idx_left]
        val_right = ForceFreeStates_results.u_store[mode_idx, forcing_idx, 1, idx_right]

        xi_psi = val_left * (1.0 - weight) + val_right * weight
    end

    # Get safety factor at this surface
    q = equil.profiles.q_spline(psi)

    # Convert displacement to field using ideal MHD relation
    # b^ψ = i * χ₁ * (m - n*q) * ξ_ψ (from FieldReconstruction.jl line 304)
    chi1 = 2π * equil.psio
    twopi = 2π
    singfac = m_mode - n_mode * q
    b_psi = chi1 * singfac * twopi * im * xi_psi

    return b_psi
end

"""
    compute_bwp1_mn(
        ForceFreeStates_results::OdeState,
        psi::Float64,
        mode_idx::Int,
        forcing_idx::Int,
        equil::Equilibrium.PlasmaEquilibrium,
        n_mode::Int,
        m_mode::Int
    )::ComplexF64

Compute `∂b^ψ/∂ψ` at arbitrary flux surface for a given mode.

Implements Fortran GPEC's `bwp1_mn` formula from `gpeq.f` lines 106–107:

    bwp1_mn = 2πi·χ₁·(singfac·∂ξ_ψ/∂ψ − n·q'·ξ_ψ)

where:
- `ξ_ψ` = displacement (u_store[:, :, 1, :])
- `∂ξ_ψ/∂ψ` = displacement derivative (ud_store[:, :, 1, :])
- `singfac = m − n·q(ψ)`
- `q'` = dq/dψ

This is used in the delta-prime calculation: the jump
`bwp1_mn(rpsi) - bwp1_mn(lpsi)` normalized by `2π·χ₁` gives Δ'.
"""
function compute_bwp1_mn(
    ForceFreeStates_results::OdeState,
    psi::Float64,
    mode_idx::Int,
    forcing_idx::Int,
    equil::Equilibrium.PlasmaEquilibrium,
    n_mode::Int,
    m_mode::Int
)::ComplexF64
    psi_store = ForceFreeStates_results.psi_store[1:ForceFreeStates_results.step]
    idx_right = findfirst(p -> p >= psi, psi_store)

    if idx_right === nothing
        idx_left  = ForceFreeStates_results.step - 1
        idx_right = ForceFreeStates_results.step
    elseif idx_right == 1
        idx_left  = 1
        idx_right = 2
    else
        idx_left = idx_right - 1
    end

    psi_left  = psi_store[idx_left]
    psi_right = psi_store[idx_right]
    weight    = (psi - psi_left) / (psi_right - psi_left)

    xsp  = (1.0 - weight) * ForceFreeStates_results.u_store[ mode_idx, forcing_idx, 1, idx_left] +
                   weight  * ForceFreeStates_results.u_store[ mode_idx, forcing_idx, 1, idx_right]
    xsp1 = (1.0 - weight) * ForceFreeStates_results.ud_store[mode_idx, forcing_idx, 1, idx_left] +
                   weight  * ForceFreeStates_results.ud_store[mode_idx, forcing_idx, 1, idx_right]

    chi1    = 2π * equil.psio
    q       = equil.profiles.q_spline(psi)
    q1      = equil.profiles.q_deriv(psi)
    singfac = m_mode - n_mode * q

    return 2π * im * chi1 * (singfac * xsp1 - n_mode * q1 * xsp)
end

"""
    compute_current_density(
        equil::Equilibrium.PlasmaEquilibrium,
        psi::Float64
    )::Float64

Compute effective current density coefficient at given flux surface.

Implements GPEC's j_c calculation (gpout.f line 560-578):
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

        # Evaluate bicubic splines with derivatives at (psi, theta)
        # New API uses separate interpolants for each component
        r2 = equil.rzphi_rsquared((psi, theta); hint=hint2d)           # rfac²
        deta = equil.rzphi_offset((psi, theta); hint=hint2d)           # angle offset
        jac = equil.rzphi_jac((psi, theta); hint=hint2d)               # Jacobian
        r2_y = equil.rzphi_rsquared((psi, theta); deriv=Val((0, 1)), hint=hint2d)  # ∂(rfac²)/∂theta
        deta_y = equil.rzphi_offset((psi, theta); deriv=Val((0, 1)), hint=hint2d)  # ∂(deta)/∂theta

        rfac = sqrt(abs(r2))
        fy_rfac2 = r2_y
        fy_deta = deta_y

        # Compute R coordinate (Z not needed for this calculation)
        eta = twopi * (theta + deta)
        r = ro + rfac * cos(eta)

        # Compute metric components w(1,1) and w(1,2) for |∇ψ|
        # These come from the metric tensor in flux coordinates
        w11 = (1.0 + fy_deta) * twopi^2 * rfac * r / jac
        w12 = -fy_rfac2 * π * r / (rfac * jac)

        # Flux gradient magnitude |∇ψ|
        delpsi = sqrt(w11^2 + w12^2)

        # sqreqb = (F² + χ₁²|∇ψ|²) / (2πR)²  where F = R·B_tor (Fortran sq%f(1))
        sqreqb = (F_tor^2 + chi1^2 * delpsi^2) / (twopi * r)^2

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
    apply_green_function_simple(
        green::Matrix{Float64},
        mode_coeffs::Vector{ComplexF64}
    )::Vector{Float64}

Apply Green's function to Fourier mode coefficients.

Simplified version that extracts only plasma surface rows (first mtheta rows).

## Arguments

  - `green`: Green's function matrix [2*mtheta, 2*mpert]
  - `mode_coeffs`: Complex Fourier coefficients [mpert]

## Returns

  - Potential at plasma surface theta points [mtheta]
"""
function apply_green_function_simple(
    green::Matrix{Float64},
    mode_coeffs::Vector{ComplexF64}
)::Vector{Float64}
    mtheta = size(green, 1) ÷ 2
    mpert = length(mode_coeffs)

    # Pack complex to real/imag format
    packed = zeros(Float64, 2 * mpert)
    for i in 1:mpert
        packed[2*i-1] = real(mode_coeffs[i])
        packed[2*i] = imag(mode_coeffs[i])
    end

    # Apply Green's function (only plasma surface rows)
    chi_theta = green[1:mtheta, :] * packed

    return chi_theta
end

"""
    compute_surface_inductance_from_greens(
        grri::Matrix{Float64},
        grre::Matrix{Float64},
        ffs_intr::ForceFreeStatesInternal,
        nn::Int,
        ν::Vector{Float64}
    )::Matrix{ComplexF64}

Compute surface inductance matrix from Green's functions at flux surface.

Implements the GPEC gpvacuum_flxsurf algorithm (gpvacuum.f lines 299–331).

The Julia vacuum code uses SFL Fourier basis `cos(m*θ - n*ν)` in the column transform,
so the row DFT must apply the matching toroidal phase correction `exp(-i*n*ν)` before
the DFT (matching Fortran's `EXP(-ifac*nn*dphi)` in gpvacuum_flxsurf lines 308–309).

## Arguments

  - `grri`: Interior Green's function [2*mtheta, 2*mpert] (GROUPED: cos cols 1:mpert, sin cols mpert+1:2*mpert)
  - `grre`: Exterior Green's function [2*mtheta, 2*mpert]
  - `ffs_intr`: ForceFreeStates internal state
  - `nn`: Toroidal mode number
  - `ν`: Toroidal angle offset on the vacuum theta grid [mtheta]

## Returns

Surface inductance matrix [mpert × mpert]
"""
@with_pool pool function compute_surface_inductance_from_greens(
    grri::Matrix{Float64},
    grre::Matrix{Float64},
    ffs_intr::ForceFreeStatesInternal,
    nn::Int,
    ν::Vector{Float64},
)::Matrix{ComplexF64}
    mpert = ffs_intr.mpert
    mtheta = size(grri, 1) ÷ 2
    μ₀ = 4π * 1e-7

    ft = FourierTransforms.FourierTransform(mtheta, mpert, ffs_intr.mlow)

    flux_matrix    = zeros!(pool, ComplexF64, mpert, mpert)
    current_matrix = zeros!(pool, ComplexF64, mpert, mpert)

    grri_surf = @view grri[1:mtheta, :]
    grre_surf = @view grre[1:mtheta, :]

    kax_re = zeros!(pool, Float64, mtheta)
    kax_im = zeros!(pool, Float64, mtheta)

    # Toroidal phase correction: exp(-i*n*ν) matching Fortran gpvacuum_flxsurf line 308-309
    # EXP(-ifac*nn*dphi). Precompute since it's the same for all modes.
    cos_nν = cos.(nn .* ν)
    sin_nν = sin.(nn .* ν)

    for i in 1:mpert
        flux_matrix[i, i] = 1.0

        for k in 1:mtheta
            kax_re[k] = (grri_surf[k, i]        + grre_surf[k, i])        / (μ₀ * (2π)^2)
            kax_im[k] = (grri_surf[k, mpert+i]  + grre_surf[k, mpert+i])  / (μ₀ * (2π)^2)
        end

        # Apply exp(-i*n*ν): (kax_re - i*kax_im)*(cos(nν) - i*sin(nν))
        kax_re_corr = kax_re .* cos_nν .- kax_im .* sin_nν
        kax_im_corr = kax_re .* sin_nν .+ kax_im .* cos_nν

        current_matrix[:, i] = (ft(kax_re_corr) .- im .* ft(kax_im_corr)) ./ mtheta
    end

    # Compute surface inductance: L_surf = flux * inv(current) = inv(current)
    L_surf = zeros(ComplexF64, mpert, mpert)

    current_mag = maximum(abs.(current_matrix))

    if current_mag < 1e-15
        @warn "Current matrix is all zeros! Cannot compute surface inductance." maxlog=1
        for i in 1:mpert
            L_surf[i, i] = μ₀ * 1e-6
        end
    else
        try
            regularization = 1e-12 * current_mag
            current_reg = current_matrix + regularization * I

            L_surf = flux_matrix * inv(current_reg)

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

Implements GPEC's area calculation (gpout.f line 568):
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
    # Physical constants
    twopi = 2π

    # Magnetic axis location
    ro = equil.ro

    # Number of theta points for integration
    mthsurf = length(equil.rzphi_ys) - 1

    # Integrate around flux surface using trapezoidal rule
    area = 0.0

    # Storage for last point (needed for trapezoidal rule correction)
    last_jac = 0.0
    last_delpsi = 0.0

    hint2d = (Ref(1), Ref(1))  # Shared 2D hint for hot loop optimization
    for itheta in 0:mthsurf
        # Theta coordinate normalized to [0, 1]
        theta = itheta / mthsurf

        # Evaluate bicubic splines with derivatives at (psi, theta)
        r2 = equil.rzphi_rsquared((psi, theta); hint=hint2d)
        jac = equil.rzphi_jac((psi, theta); hint=hint2d)
        deta = equil.rzphi_offset((psi, theta); hint=hint2d)
        r2_y = equil.rzphi_rsquared((psi, theta); deriv=Val((0, 1)), hint=hint2d)
        deta_y = equil.rzphi_offset((psi, theta); deriv=Val((0, 1)), hint=hint2d)

        # Compute rfac
        rfac = sqrt(abs(r2))
        fy_rfac2 = r2_y
        fy_deta = deta_y

        # Compute R coordinate
        eta = twopi * (theta + deta)
        r = ro + rfac * cos(eta)

        # Compute metric components for |∇ψ|
        w11 = (1.0 + fy_deta) * twopi^2 * rfac * r / jac
        w12 = -fy_rfac2 * π * r / (rfac * jac)

        # Flux gradient magnitude
        delpsi = sqrt(w11^2 + w12^2)

        # Accumulate area integral (trapezoidal rule)
        area += jac * delpsi / mthsurf

        # Store last point for correction
        if itheta == mthsurf
            last_jac = jac
            last_delpsi = delpsi
        end
    end

    # Trapezoidal rule end correction
    area -= last_jac * last_delpsi / mthsurf

    return area
end

"""
    compute_island_diagnostics!(state::PerturbedEquilibriumState, n_rational::Int)

Compute island half-width and Chirikov parameter from applied resonant vectors.

  - `island_half_width[row]` = √|island_width_sq[row]|
  - `chirikov_parameter[row]` = half-width / (half-distance to nearest neighbor in rational_psi)
"""
function compute_island_diagnostics!(state::PerturbedEquilibriumState, n_rational::Int)
    state.island_half_width  = sqrt.(abs.(state.island_width_sq))
    state.chirikov_parameter = zeros(Float64, n_rational)

    n_rational <= 1 && return

    for row in 1:n_rational
        psi_row  = state.rational_psi[row]
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
