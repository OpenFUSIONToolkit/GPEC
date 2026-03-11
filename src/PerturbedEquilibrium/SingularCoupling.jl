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

Compute singular layer coupling metrics and island diagnostics.

Calculates the coupling between forcing modes and resonant surfaces, which determines
the effectiveness of external perturbations at rational surfaces where q = m/n.
[Park Phys. Plasmas 2009 056115]

Implements GPEC algorithm from gpout.f:

 1. For each forcing mode, apply unit field and compute plasma response
 2. For each singular surface, evaluate field jump and compute coupling quantities
 3. Calculate diagnostic island widths and overlap parameters

## Arguments

  - `state`: Output state structure to store results
  - `equil`: Equilibrium solution with q-profile and flux surfaces
  - `ForceFreeStates_results`: ForceFreeStates stability calculation results
  - `vac_data`: Vacuum response data including Green's functions
  - `ffs_intr`: ForceFreeStates internal state with singular surface data
  - `intr`: PerturbedEquilibrium internal state with permeability matrix
  - `ctrl`: Control parameters

## Modifies

Populates the following fields in `state`:

  - `resonant_flux[npert, msing]`: Normalized resonant flux Φ_r/A, indexed by (n-index, surface)
  - `resonant_current[npert, msing]`: Resonant current density
  - `island_width_sq[npert, msing]`: Square of island half-width (w/2)²
  - `penetrated_field[npert, msing]`: Normal field at resonant surface
  - `delta_prime[npert, msing]`: Tearing stability parameter Δ'
  - `island_half_width[msing]`: Dimensional island half-width (maximum over all n)
  - `chirikov_parameter[msing]`: Island overlap metric
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
    if ctrl.verbose
        @info "Computing singular coupling metrics (GPEC method)"
    end

    # Extract dimensions
    msing = ffs_intr.msing
    mpert = ffs_intr.mpert
    npert = ffs_intr.npert
    numpert_total = ffs_intr.numpert_total

    if msing == 0
        if ctrl.verbose
            @info "No singular surfaces found. Skipping singular coupling calculation."
        end
        return
    end

    # Initialize output arrays [npert, msing]
    # For single n: these collapse to [msing] vectors
    # For multiple n: [npert, msing] matrices with zeros for non-resonant combinations
    state.resonant_flux = zeros(ComplexF64, npert, msing)
    state.resonant_current = zeros(ComplexF64, npert, msing)
    state.island_width_sq = zeros(ComplexF64, npert, msing)
    state.penetrated_field = zeros(ComplexF64, npert, msing)
    state.delta_prime = zeros(ComplexF64, npert, msing)
    state.island_half_width = zeros(Float64, msing)
    state.chirikov_parameter = zeros(Float64, msing)

    # Physical constants
    chi1 = 2π * equil.psio  # Flux normalization
    twopi = 2π
    μ₀ = 4π * 1e-7

    # Get permeability matrix from internal state
    permeability = intr.plasma_response
    if size(permeability, 1) == 0
        @warn "Permeability matrix not computed. Skipping singular coupling."
        return
    end

    # Get vacuum calculation parameters
    mtheta = vac_data.mthvac  # Vacuum poloidal grid size
    mlow = ffs_intr.mlow
    nlow = ffs_intr.nlow
    nhigh = ffs_intr.nhigh

    # Perturbed equilibrium calculations always use nowall
    wall_settings = Vacuum.WallShapeSettings(; shape="nowall")

    if ctrl.verbose
        nstr = nlow == nhigh ? "$nlow" : "$nlow:$nhigh"
        @info "Computing surface Green's functions at $msing resonant surfaces (n = $nstr, no wall)"
    end

    # Main loop: Process each toroidal mode number separately
    for nn in nlow:nhigh
        n_idx = nn - nlow + 1  # Index for storing in [npert, msing] arrays

        if ctrl.verbose && npert > 1
            @info "Computing metrics for n = $nn"
        end

        # For each singular surface, compute Green's functions for this n
        # and calculate metrics if surface is resonant
        for s in 1:msing
            sing_surf = ffs_intr.sing[s]

            # Check if this surface is resonant for this n
            # m_res = round(q * n) must be an integer and within our m range
            m_res_float = sing_surf.q * nn
            m_res = round(Int, m_res_float)

            # Check if m/n is truly integer (within tolerance)
            if abs(m_res_float - m_res) > 1e-6
                # Not resonant for this n, leave metrics as zero
                continue
            end

            # Check if resonant mode is within our m range
            if m_res < ffs_intr.mlow || m_res > ffs_intr.mhigh
                continue
            end

            # Compute Green's functions at this surface for this n
            # TODO: This assumes an initial 2D equilibrum, getting 2D Green's functions for independent n
            vac_input = Vacuum.VacuumInput(equil, sing_surf.psifac, mtheta, 1, mpert, mlow, 1, nn)
            _, grri, grre, _, _ = Vacuum.compute_vacuum_response(vac_input, wall_settings; green_only=true)

            # Store in singular surface struct (overwrites for each n)
            ffs_intr.sing[s].grri = grri
            ffs_intr.sing[s].grre = grre

            # Find the linear index for the resonant mode (m_res, nn)
            resnum = findfirst(j -> intr.m_modes[j] == m_res && intr.n_modes[j] == nn, 1:numpert_total)
            if resnum === nothing
                @warn "Could not find index for resonant mode (m=$m_res, n=$nn)" maxlog=1
                continue
            end

            # Evaluate field derivative jump across surface
            respsi = sing_surf.psifac
            if size(ForceFreeStates_results.ca_l, 4) >= s && size(ForceFreeStates_results.ca_r, 4) >= s
                # Use asymptotic coefficients from ideal MHD solution
                # resnum is the resonant mode index, and we evaluate at forcing mode resnum
                lbwp1 = ForceFreeStates_results.ca_l[resnum, resnum, 2, s]
                rbwp1 = ForceFreeStates_results.ca_r[resnum, resnum, 2, s]
            else
                # Fallback: finite difference
                spot = 1e-6
                lpsi = respsi - spot / (abs(nn) * abs(sing_surf.q1))
                rpsi = respsi + spot / (abs(nn) * abs(sing_surf.q1))
                lbwp1 = interpolate_field_derivative(ForceFreeStates_results, lpsi, resnum, resnum)
                rbwp1 = interpolate_field_derivative(ForceFreeStates_results, rpsi, resnum, resnum)
            end

            # Compute Delta' (tearing stability parameter)
            delta_prime_val = (rbwp1 - lbwp1) / (twopi * chi1)
            state.delta_prime[n_idx, s] = delta_prime_val

            # Compute resonant current
            j_c = compute_current_density(equil, sing_surf.psifac)
            delcurs = (rbwp1 - lbwp1) * j_c * im / (twopi * m_res)
            resonant_current_val = -delcurs / im
            state.resonant_current[n_idx, s] = resonant_current_val

            # Compute singular flux from current using surface inductance
            singflx_mn = compute_singular_flux(
                resonant_current_val,
                vac_data,
                ffs_intr,
                intr,
                sing_surf,
                resnum,
                nn
            )

            # Compute island half-width squared
            shear = sing_surf.q1 * sing_surf.psifac / sing_surf.q
            if abs(shear) > 1e-10
                state.island_width_sq[n_idx, s] = 4.0 * singflx_mn / (twopi * shear * sing_surf.q * chi1)
            else
                state.island_width_sq[n_idx, s] = 0.0 + 0.0im
            end

            # Get interpolated field at resonant surface
            interpbwn = interpolate_field_at_surface(ForceFreeStates_results, respsi, resnum, resnum, equil, m_res, nn)

            # Normalize by surface area
            area = compute_surface_area(equil, sing_surf.psifac)
            state.penetrated_field[n_idx, s] = interpbwn / area
            state.resonant_flux[n_idx, s] = singflx_mn / area

            if ctrl.verbose && n_idx == 1 && s == 1
                @info "Surface 1: q = $(@sprintf("%.3f", sing_surf.q)), ψ = $(@sprintf("%.3f", sing_surf.psifac)), m = $m_res, n = $nn, Δ' = $(@sprintf("%.3e", real(delta_prime_val)))"
            end
        end
    end

    # Post-processing: Compute diagnostic quantities
    compute_island_diagnostics!(state, ffs_intr, equil, chi1, twopi)

    if ctrl.verbose
        max_island_width = maximum(abs.(state.island_half_width))
        max_delta_prime = maximum(abs.(real.(state.delta_prime)))
        @info "Singular coupling complete. Max island half-width = $(@sprintf("%.3e", max_island_width)), max Δ' = $(@sprintf("%.3e", max_delta_prime))"
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
  - sqreqb: Magnetic field quantity (F² + χ₁²|∇ψ|²)/(2πR)²
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
    F_tor = equil.profiles.F_spline(psi)  # Toroidal field function F = R*B_tor times 2π
    q = equil.profiles.q_spline(psi)      # Safety factor

    # Magnetic axis location
    ro = equil.ro
    zo = equil.zo

    # Number of theta points for integration
    # Match GPEC's mthsurf (typically 101 points from theta=0 to theta=1)
    mthsurf = length(equil.rzphi_xs) - 1

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

        # Magnetic field quantity: sqreqb = (F² + χ₁²|∇ψ|²) / (2πR)²
        # F is toroidal field function (already includes factor of 2π from sq)
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
        intr::PerturbedEquilibriumInternal
    )::Matrix{ComplexF64}

Compute surface inductance matrix from Green's functions at flux surface.

This implements the full GPEC gpvacuum_flxsurf algorithm (gpvacuum.f line 299-331):

 1. For each mode, apply Green's functions to unit flux
 2. Compute surface current from potential jump: kax = (chi - che) / μ₀
 3. Fourier transform to mode space
 4. Return inductance: L_surf = flux * inv(current)

## Arguments

  - `grri`: Interior Green's function [2*mtheta, 2*mpert] (stored in sing_surf)
  - `grre`: Exterior Green's function [2*mtheta, 2*mpert] (stored in sing_surf)
  - `ffs_intr`: ForceFreeStates internal state
  - `intr`: PerturbedEquilibrium internal state with mode arrays

## Returns

Surface inductance matrix [numpert_total, numpert_total]

## GPEC Reference

```fortran
DO i=1,mpert
   vbwp_mn=0; vbwp_mn(i)=1.0
   chi_fun = grri * vbwp_mn
   che_fun = grre * vbwp_mn
   kax_fun = (chi_fun - che_fun) / mu0
   CALL iscdftf(mfac, mpert, kax_fun, mthsurf, fkaxmats(:,i))
ENDDO
fsurf_indmats = fflxmats * inv(fkaxmats)
```
"""
@with_pool pool function compute_surface_inductance_from_greens(
    grri::Matrix{Float64},
    grre::Matrix{Float64},
    ffs_intr::ForceFreeStatesInternal,
    intr::PerturbedEquilibriumInternal
)::Matrix{ComplexF64}
    mpert = ffs_intr.mpert
    mtheta = size(grri, 1) ÷ 2

    # Physical constant
    μ₀ = 4π * 1e-7

    # Create Fourier transform object
    ft = FourierTransforms.FourierTransform(mtheta, mpert, ffs_intr.mlow)

    # Initialize matrices (mpert x mpert for single toroidal mode number)
    # Green's functions are computed for a specific n, so inductance is only over poloidal modes
    flux_matrix = zeros!(pool, ComplexF64, mpert, mpert)
    current_matrix = zeros!(pool, ComplexF64, mpert, mpert)

    # Pre-allocate loop buffers from pool
    vbwp_mn = zeros!(pool, ComplexF64, mpert)
    chi_theta = zeros!(pool, Float64, mtheta)
    che_theta = zeros!(pool, Float64, mtheta)
    kax_theta = zeros!(pool, Float64, mtheta)

    # For each poloidal mode, compute surface current from Green's functions
    for i in 1:mpert
        # Unit flux for mode i
        vbwp_mn .= 0.0
        vbwp_mn[i] = 1.0

        # Apply Green's functions using same approach as ResponseMatrices.jl
        apply_green_function!(chi_theta, grri, vbwp_mn)
        apply_green_function!(che_theta, grre, vbwp_mn)

        # Surface current from potential jump
        @. kax_theta = (chi_theta - che_theta) / μ₀

        # Transform back to Fourier mode space
        kax_modes = ft(kax_theta)

        # Store in matrices
        flux_matrix[:, i] = vbwp_mn
        current_matrix[:, i] = kax_modes
    end

    # Compute surface inductance: L_surf = flux * inv(current)
    # Initialize L_surf outside try-catch for scoping
    L_surf = zeros(ComplexF64, mpert, mpert)

    # Check matrix condition before inversion
    current_mag = maximum(abs.(current_matrix))
    current_min = minimum(abs.(current_matrix[abs.(current_matrix) .> 1e-15]))

    if current_mag < 1e-15
        @warn "Current matrix is all zeros! Cannot compute surface inductance." maxlog=1
        @warn "  This indicates grri and grre are identical or Green's functions failed." maxlog=1
        # Fallback: diagonal approximation
        for i in 1:mpert
            L_surf[i, i] = μ₀ * 1e-6
        end
    else
        try
            # Regularization for numerical stability
            regularization = 1e-12 * current_mag
            current_reg = current_matrix + regularization * I

            L_surf = flux_matrix * inv(current_reg)

            # Hermitianize
            L_surf = 0.5 * (L_surf + L_surf')
        catch e
            @warn "Surface inductance inversion failed: $e" maxlog=1
            @warn "  Current matrix magnitude: $current_mag, min non-zero: $current_min" maxlog=1
            @warn "  Check if Green's functions are computed correctly." maxlog=1
            # Fallback: diagonal approximation
            for i in 1:mpert
                L_surf[i, i] = μ₀ * 1e-6
            end
        end
    end

    return L_surf
end

"""
    compute_singular_flux(
        current::ComplexF64,
        vac_data::VacuumData,
        ffs_intr::ForceFreeStatesInternal,
        intr::PerturbedEquilibriumInternal,
        sing_surf::ForceFreeStates.SingType,
        mode_idx::Int,
        nn::Int
    )::ComplexF64

Compute singular flux from resonant current using surface inductance.

Uses surface inductance matrix at the singular surface:
Φ = L_surf * I

where L_surf is computed from Green's functions stored in sing_surf.

## GPEC Formula

```fortran
fkaxmn(resnum) = singcurs(ising,i) / (twopi*nn)
singflx_mn = MATMUL(fsurfindmats(ising,:,:), fkaxmn)
```

## Implementation

Uses surface inductance matrix computed from pre-stored Green's functions.
The Green's functions (grri, grre) are calculated at each singular surface
during initialization and stored in the sing_surf struct for efficiency.
"""
function compute_singular_flux(
    current::ComplexF64,
    vac_data::VacuumData,
    ffs_intr::ForceFreeStatesInternal,
    intr::PerturbedEquilibriumInternal,
    sing_surf::ForceFreeStates.SingType,
    mode_idx::Int,
    nn::Int
)::ComplexF64
    if abs(nn) == 0
        return 0.0 + 0.0im
    end

    # Get mode numbers for this index
    mpert = ffs_intr.mpert
    mlow = ffs_intr.mlow
    m_mode = intr.m_modes[mode_idx]

    # Poloidal mode index within mpert range
    m_idx = m_mode - mlow + 1

    # Build current vector (size mpert for single toroidal mode)
    # L_surf is mpert x mpert since Green's functions are for single n
    current_vector = zeros(ComplexF64, mpert)
    current_vector[m_idx] = current / (2π * nn)

    # Compute surface inductance from stored Green's functions
    # These were pre-computed at the singular surface location
    L_surf = compute_surface_inductance_from_greens(
        sing_surf.grri,
        sing_surf.grre,
        ffs_intr,
        intr
    )

    # Compute flux: Φ = L * I
    flux_vector = L_surf * current_vector

    # Return flux at resonant mode
    return flux_vector[m_idx]
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
    zo = equil.zo

    # Number of theta points for integration
    mthsurf = length(equil.rzphi_xs) - 1

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
    compute_island_diagnostics!(
        state::PerturbedEquilibriumState,
        ffs_intr::ForceFreeStatesInternal,
        equil::Equilibrium.PlasmaEquilibrium,
        chi1::Float64,
        twopi::Float64
    )

Compute island half-width and Chirikov parameter for each singular surface.

Uses the resonant flux and island_width_sq from singular coupling calculation.
"""
function compute_island_diagnostics!(
    state::PerturbedEquilibriumState,
    ffs_intr::ForceFreeStatesInternal,
    equil::Equilibrium.PlasmaEquilibrium,
    chi1::Float64,
    twopi::Float64
)
    msing = ffs_intr.msing

    # Compute island half-width for each singular surface
    # Take maximum over all toroidal modes n
    for s in 1:msing
        sing_surf = ffs_intr.sing[s]

        # Island half-width: w/2 = √|island_width_sq|
        # Array is [npert, msing], so index with [:, s] to get all n for this surface
        max_island_sq = maximum(abs.(state.island_width_sq[:, s]))
        state.island_half_width[s] = sqrt(abs(max_island_sq))

        # Compute Chirikov parameter: overlap = w/2 / distance_to_nearest_surface
        # Find distance to nearest singular surface
        if msing > 1
            distances = Float64[]
            for s2 in 1:msing
                if s2 != s
                    dist = abs(ffs_intr.sing[s].psifac - ffs_intr.sing[s2].psifac)
                    push!(distances, dist)
                end
            end

            if !isempty(distances)
                hdist = minimum(distances) / 2.0  # Half-distance
                if hdist > 1e-10
                    state.chirikov_parameter[s] = state.island_half_width[s] / hdist
                else
                    state.chirikov_parameter[s] = Inf
                end
            else
                state.chirikov_parameter[s] = 0.0
            end
        else
            state.chirikov_parameter[s] = 0.0
        end
    end
end
