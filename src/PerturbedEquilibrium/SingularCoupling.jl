"""
Singular surface coupling and field calculations.

Implements GPEC's singular surface analysis (gpout.f lines 585-665, 1700-1810):
- Delta' (tearing stability parameter)
- Resonant currents
- Island half-widths
- Chirikov parameters (island overlap)
- Coupling matrices

Reference: ~/Code/gpec/gpec/gpout.f
"""

"""
    compute_singular_coupling_metrics!(
        state::PerturbedEquilibriumState,
        equil::Equilibrium.PlasmaEquilibrium,
        dcon_results::OdeState,
        vac_data::VacuumData,
        ffs_intr::ForceFreeStatesInternal,
        intr::PerturbedEquilibriumInternal,
        ctrl::PerturbedEquilibriumControl
    )

Compute singular layer coupling metrics and island diagnostics.

Calculates the coupling between forcing modes and resonant surfaces, which determines
the effectiveness of external perturbations at rational surfaces where q = m/n.

Implements GPEC algorithm from gpout.f:
1. For each forcing mode, apply unit field and compute plasma response
2. For each singular surface, evaluate field jump and compute coupling quantities
3. Calculate diagnostic island widths and overlap parameters

## Arguments
- `state`: Output state structure to store results
- `equil`: Equilibrium solution with q-profile and flux surfaces
- `dcon_results`: DCON stability calculation results
- `vac_data`: Vacuum response data including Green's functions
- `ffs_intr`: ForceFreeStates internal state with singular surface data
- `intr`: PerturbedEquilibrium internal state with permeability matrix
- `ctrl`: Control parameters

## Modifies
Populates the following fields in `state`:
- `resonant_flux[msing, numpert_total]`: Normalized resonant flux Φ_r/A
- `resonant_current[msing, numpert_total]`: Resonant current density
- `island_width_sq[msing, numpert_total]`: Square of island half-width (w/2)²
- `penetrated_field[msing, numpert_total]`: Normal field at resonant surface
- `delta_prime[msing, numpert_total]`: Tearing stability parameter Δ'
- `island_half_width[msing]`: Dimensional island half-width
- `chirikov_parameter[msing]`: Island overlap metric

Note: numpert_total = mpert × npert handles all (m,n) mode combinations
"""
function compute_singular_coupling_metrics!(
    state::PerturbedEquilibriumState,
    equil::Equilibrium.PlasmaEquilibrium,
    dcon_results::OdeState,
    vac_data::VacuumData,
    ffs_intr::ForceFreeStatesInternal,
    intr::PerturbedEquilibriumInternal,
    ctrl::PerturbedEquilibriumControl
)
    if ctrl.verbose
        println("Computing singular coupling metrics (GPEC method)")
    end

    # Extract dimensions
    msing = ffs_intr.msing
    mpert = ffs_intr.mpert
    npert = ffs_intr.npert
    numpert_total = ffs_intr.numpert_total

    if msing == 0
        if ctrl.verbose
            println("  No singular surfaces found. Skipping singular coupling calculation.")
        end
        return
    end

    # Initialize output arrays
    state.resonant_flux = zeros(ComplexF64, msing, numpert_total)
    state.resonant_current = zeros(ComplexF64, msing, numpert_total)
    state.island_width_sq = zeros(ComplexF64, msing, numpert_total)
    state.penetrated_field = zeros(ComplexF64, msing, numpert_total)
    state.delta_prime = zeros(ComplexF64, msing, numpert_total)
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

    if ctrl.verbose
        println("  Number of singular surfaces: $msing")
        println("  Number of perturbing modes (m): $mpert")
        println("  Number of toroidal modes (n): $npert")
        println("  Total mode combinations: $numpert_total")
    end

    # Main loop: For each forcing mode (m,n) combination
    for i in 1:numpert_total
        # Extract m and n for this mode from pre-computed arrays
        m_mode = intr.m_modes[i]
        n_mode = intr.n_modes[i]

        # Step 1: Apply unit external field to mode i
        finmn = zeros(ComplexF64, numpert_total)
        finmn[i] = 1.0

        # Step 2: Get plasma response
        # For fixed boundary: foutmn = finmn (no plasma response)
        # For free boundary: foutmn = permeability * finmn
        if ctrl.fixed_boundary
            foutmn = finmn
        else
            foutmn = permeability * finmn
        end

        # Step 3: Convert to displacement at edge
        singfac = m_mode - n_mode * ffs_intr.qlim
        edge_displacement = foutmn ./ (chi1 * singfac * twopi * im)

        # Step 4: Loop over each singular surface
        for s in 1:msing
            sing_surf = ffs_intr.sing[s]

            # Get resonant mode number for this surface with this toroidal mode
            # m_res = round(q_sing * n)
            m_res = round(Int, sing_surf.q * n_mode)

            # Check if resonant mode is within our m range
            if m_res < ffs_intr.mlow || m_res > ffs_intr.mhigh
                continue
            end

            # Find the linear index for this (m_res, n_mode) combination
            # Search through pre-computed mode arrays for matching (m,n) pair
            resnum = findfirst(j -> intr.m_modes[j] == m_res && intr.n_modes[j] == n_mode, 1:numpert_total)

            if resnum === nothing
                # This shouldn't happen if mode arrays are set up correctly
                @warn "Could not find linear index for (m=$m_res, n=$n_mode)" maxlog=1
                continue
            end

            respsi = sing_surf.psifac  # Flux coordinate of singular surface

            # Step 4b: Evaluate field derivative jump across surface
            # Use asymptotic coefficients from DCON if available
            if size(dcon_results.ca_l, 4) >= s && size(dcon_results.ca_r, 4) >= s
                # Get field derivatives from asymptotic coefficients
                # ca_l and ca_r contain the solution just left and right of singular surface
                # Component 2 is the derivative (∂ξ_ψ/∂ψ)
                lbwp1 = dcon_results.ca_l[resnum, i, 2, s]
                rbwp1 = dcon_results.ca_r[resnum, i, 2, s]
            else
                # Fallback: use finite difference across singular surface
                spot = 1e-6  # Small step
                lpsi = respsi - spot / (abs(n_mode) * abs(sing_surf.q1))
                rpsi = respsi + spot / (abs(n_mode) * abs(sing_surf.q1))

                # Interpolate field derivative at left and right
                lbwp1 = interpolate_field_derivative(dcon_results, lpsi, resnum, i)
                rbwp1 = interpolate_field_derivative(dcon_results, rpsi, resnum, i)
            end

            # Step 4c: Compute Delta' (tearing stability parameter)
            # Δ' = (∂_ψ b_ψ|_right - ∂_ψ b_ψ|_left) / (2π χ₁)
            state.delta_prime[s, i] = (rbwp1 - lbwp1) / (twopi * chi1)

            # Step 4d: Compute resonant current
            # Get current density at singular surface from equilibrium
            j_c = compute_current_density(equil, sing_surf.psifac)

            # Current jump from field derivative jump
            delcurs = (rbwp1 - lbwp1) * j_c * im / (twopi * m_mode)
            state.resonant_current[s, i] = -delcurs / im

            # Step 4e: Compute singular flux from current
            # Uses surface inductance matrix at the singular surface
            singflx_mn = compute_singular_flux(
                state.resonant_current[s, i],
                vac_data,
                ffs_intr,
                equil,
                sing_surf.psifac,
                resnum,
                n_mode
            )

            # Step 4f: Compute island half-width squared
            # (w/2)² = 4*Φ_r / (2π * shear * q * χ₁)
            shear = sing_surf.q1 * sing_surf.psifac / sing_surf.q  # Magnetic shear
            if abs(shear) > 1e-10
                state.island_width_sq[s, i] = 4.0 * singflx_mn / (twopi * shear * sing_surf.q * chi1)
            else
                state.island_width_sq[s, i] = 0.0 + 0.0im
            end

            # Step 4g: Get interpolated field at resonant surface
            interpbwn = interpolate_field_at_surface(dcon_results, respsi, resnum, i, equil, m_res, n_mode)

            # Normalize by surface area
            area = compute_surface_area(equil, sing_surf.psifac)
            state.penetrated_field[s, i] = interpbwn / area

            # Step 4h: Compute normalized flux
            state.resonant_flux[s, i] = singflx_mn / area

            if ctrl.verbose && i == 1
                println("    Singular surface $s: q = $(sing_surf.q), ψ = $(sing_surf.psifac)")
                println("      Resonant mode: m = $m_res, n = $n_mode")
                println("      Delta': $(real(state.delta_prime[s, i]))")
            end
        end
    end

    # Post-processing: Compute diagnostic quantities
    compute_island_diagnostics!(state, ffs_intr, equil, chi1, twopi)

    if ctrl.verbose
        println("  Singular coupling calculation complete")
        max_island_width = maximum(abs.(state.island_half_width))
        max_delta_prime = maximum(abs.(real.(state.delta_prime)))
        println("    Max island half-width: $(@sprintf("%.3e", max_island_width))")
        println("    Max Delta': $(@sprintf("%.3e", max_delta_prime))")
    end
end

"""
    interpolate_field_derivative(
        dcon_results::OdeState,
        psi::Float64,
        mode_idx::Int,
        forcing_idx::Int
    )::ComplexF64

Interpolate field derivative ∂b/∂ψ at arbitrary flux coordinate.

Uses linear interpolation of stored DCON solution.
"""
function interpolate_field_derivative(
    dcon_results::OdeState,
    psi::Float64,
    mode_idx::Int,
    forcing_idx::Int
)::ComplexF64
    # Find bracketing points in psi_store
    psi_store = dcon_results.psi_store[1:dcon_results.step]

    # Find indices that bracket psi
    idx_right = findfirst(p -> p >= psi, psi_store)

    if idx_right === nothing
        # psi is beyond stored range, use last value
        return dcon_results.ud_store[mode_idx, forcing_idx, 1, dcon_results.step]
    elseif idx_right == 1
        # psi is before stored range, use first value
        return dcon_results.ud_store[mode_idx, forcing_idx, 1, 1]
    else
        # Interpolate between idx_left and idx_right
        idx_left = idx_right - 1

        psi_left = psi_store[idx_left]
        psi_right = psi_store[idx_right]
        weight = (psi - psi_left) / (psi_right - psi_left)

        val_left = dcon_results.ud_store[mode_idx, forcing_idx, 1, idx_left]
        val_right = dcon_results.ud_store[mode_idx, forcing_idx, 1, idx_right]

        return val_left * (1.0 - weight) + val_right * weight
    end
end

"""
    interpolate_field_at_surface(
        dcon_results::OdeState,
        psi::Float64,
        mode_idx::Int,
        forcing_idx::Int,
        equil::Equilibrium.PlasmaEquilibrium,
        m_mode::Int,
        n_mode::Int
    )::ComplexF64

Interpolate magnetic field at arbitrary flux surface.

Interpolates displacement from DCON solution and converts to field using
ideal MHD relation from flux coordinates:
    b^ψ = i * χ₁ * (m - n*q) * ξ_ψ

This matches the field reconstruction in FieldReconstruction.jl.
"""
function interpolate_field_at_surface(
    dcon_results::OdeState,
    psi::Float64,
    mode_idx::Int,
    forcing_idx::Int,
    equil::Equilibrium.PlasmaEquilibrium,
    m_mode::Int,
    n_mode::Int
)::ComplexF64
    # Get displacement at this surface via interpolation
    psi_store = dcon_results.psi_store[1:dcon_results.step]

    idx_right = findfirst(p -> p >= psi, psi_store)

    if idx_right === nothing
        xi_psi = dcon_results.u_store[mode_idx, forcing_idx, 1, dcon_results.step]
    elseif idx_right == 1
        xi_psi = dcon_results.u_store[mode_idx, forcing_idx, 1, 1]
    else
        idx_left = idx_right - 1
        psi_left = psi_store[idx_left]
        psi_right = psi_store[idx_right]
        weight = (psi - psi_left) / (psi_right - psi_left)

        val_left = dcon_results.u_store[mode_idx, forcing_idx, 1, idx_left]
        val_right = dcon_results.u_store[mode_idx, forcing_idx, 1, idx_right]

        xi_psi = val_left * (1.0 - weight) + val_right * weight
    end

    # Get safety factor at this surface
    sq_vals = Equilibrium.Splines.spline_eval!(equil.sq, psi)
    q = sq_vals[4]

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

This mimics GPEC's j_c calculation (gpout.f line 578):
    j_c = χ₁² * q / (μ₀ * surface_integral)

For now, uses simplified approximation. Full implementation requires
flux surface integrals of metric quantities.

## GPEC Formula

```fortran
j_c = chi1^2 * sq%f(4) / (mu0 * integral)
```

where integral involves flux surface geometric factors.

## Current Approximation

Uses `j_c ≈ χ₁² * q / μ₀` as simplified estimate.
This captures the dominant scaling but omits geometric correction factors.

## TODO

Implement full flux surface integral:
```julia
integral = ∫ (jac * |∇ψ| * sqreqb / |∇ψ|³) dθ / (2π)
```
where jac is Jacobian, sqreqb involves B², |∇ψ| is flux gradient magnitude.
"""
function compute_current_density(
    equil::Equilibrium.PlasmaEquilibrium,
    psi::Float64
)::Float64
    # Physical constants
    μ₀ = 4π * 1e-7
    chi1 = 2π * equil.psio

    # Get safety factor at this surface
    sq_vals = Equilibrium.Splines.spline_eval!(equil.sq, psi)
    q = sq_vals[4]

    # Simplified approximation: j_c ≈ χ₁² * q / μ₀
    # Full GPEC calculation involves surface integrals
    j_c = chi1^2 * q / μ₀

    return j_c
end

"""
    compute_surface_inductance_at_psi(
        equil::Equilibrium.PlasmaEquilibrium,
        vac_data::VacuumData,
        ffs_intr::ForceFreeStatesInternal,
        psi::Float64
    )::Matrix{ComplexF64}

Compute surface inductance matrix at arbitrary flux surface.

This mimics GPEC's gpvacuum_flxsurf (gpvacuum.f line 236) which:
1. Generates Green's functions at the specified flux surface
2. For each mode, computes surface current from potential jump
3. Returns inductance matrix: L_surf = flux * inv(current)

## GPEC Algorithm

```fortran
CALL mscvac(..., grri, ...)  ! Interior Green's function
CALL mscvac(..., grre, ...)  ! Exterior Green's function
kax = (chi - che) / mu0      ! Surface current from potential jump
fsurf_indmats = fflxmats * inv(fkaxmats)
```

## Current Implementation

Uses scaled approximation based on boundary surface inductance.
The scaling accounts for:
- Different flux surface area at interior vs. boundary
- Different field strength (∝ 1/√ψ roughly)

## Full Implementation (TODO)

Requires computing Green's functions at arbitrary flux surface:
1. Set up plasma boundary at psi (not psilim)
2. Call vacuum code to get grri, grre at that surface
3. Apply same algorithm as calc_surface_inductance()
4. Cache results for each singular surface
"""
function compute_surface_inductance_at_psi(
    equil::Equilibrium.PlasmaEquilibrium,
    vac_data::VacuumData,
    ffs_intr::ForceFreeStatesInternal,
    psi::Float64
)::Matrix{ComplexF64}
    # For now, use scaled approximation from boundary inductance
    # This captures the main physics but omits surface-specific Green's functions

    # Get boundary inductance (already computed)
    # This is stored in ResponseMatrices calculation
    # For this approximation, we'll construct a simplified version

    mpert = ffs_intr.mpert
    numpert_total = ffs_intr.numpert_total

    # Scaling factor: ratio of flux surfaces
    # L_surf(ψ) ≈ L_surf(ψ_edge) * (ψ_edge / ψ)^α
    # where α ≈ 1 for geometric scaling
    psi_edge = ffs_intr.psilim

    if psi > 1e-10 && psi_edge > 1e-10
        scale_factor = psi_edge / psi
    else
        scale_factor = 1.0
    end

    # Create simplified diagonal inductance matrix
    # This assumes weak mode coupling at singular surfaces
    μ₀ = 4π * 1e-7
    chi1 = 2π * equil.psio

    # Get safety factor at this surface
    sq_vals = Equilibrium.Splines.spline_eval!(equil.sq, psi)
    q = sq_vals[4]

    # Construct approximate surface inductance
    # L_ij ≈ δ_ij * μ₀ * χ₁ * scaling
    L_surf = zeros(ComplexF64, numpert_total, numpert_total)

    for i in 1:numpert_total
        # Diagonal approximation with geometric scaling
        L_surf[i, i] = μ₀ * chi1 * scale_factor
    end

    return L_surf
end

"""
    compute_singular_flux(
        current::ComplexF64,
        vac_data::VacuumData,
        ffs_intr::ForceFreeStatesInternal,
        equil::Equilibrium.PlasmaEquilibrium,
        psi::Float64,
        mode_idx::Int,
        nn::Int
    )::ComplexF64

Compute singular flux from resonant current using surface inductance.

Uses surface inductance matrix at the singular surface:
    Φ = L_surf * I

where L_surf is computed at the flux surface psi.

## GPEC Formula

```fortran
fkaxmn(resnum) = singcurs(ising,i) / (twopi*nn)
singflx_mn = MATMUL(fsurfindmats(ising,:,:), fkaxmn)
```

## Current Implementation

Uses simplified surface inductance matrix computed at the flux surface.
Full implementation requires Green's functions at that surface.
"""
function compute_singular_flux(
    current::ComplexF64,
    vac_data::VacuumData,
    ffs_intr::ForceFreeStatesInternal,
    equil::Equilibrium.PlasmaEquilibrium,
    psi::Float64,
    mode_idx::Int,
    nn::Int
)::ComplexF64
    # Build current vector with only resonant mode
    numpert_total = ffs_intr.numpert_total
    current_vector = zeros(ComplexF64, numpert_total)

    # Current at resonant mode
    if abs(nn) > 0
        current_vector[mode_idx] = current / (2π * nn)
    else
        return 0.0 + 0.0im
    end

    # Get surface inductance at this flux surface
    L_surf = compute_surface_inductance_at_psi(equil, vac_data, ffs_intr, psi)

    # Compute flux: Φ = L * I
    flux_vector = L_surf * current_vector

    # Return flux at resonant mode
    return flux_vector[mode_idx]
end

"""
    compute_surface_area(
        equil::Equilibrium.PlasmaEquilibrium,
        psi::Float64
    )::Float64

Compute flux surface area at given ψ.

This mimics GPEC's area calculation (gpout.f line 568):
    area = ∫ jac * |∇ψ| dθ / (2π)

## GPEC Formula

```fortran
area(ising) = ∫ jac * delpsi dθ / mthsurf
```

where:
- jac is the Jacobian from equilibrium
- delpsi = |∇ψ| is the flux gradient magnitude

## Current Approximation

Uses geometric estimate based on flux surface volume derivative:
    A ≈ dV/dψ / (2π √ψ)

This captures the major-radius-averaged flux surface area.
More accurate calculation requires full flux surface integration.

## TODO

Implement full integration using equilibrium bicubic spline:
```julia
area = ∫₀^(2π) jac(ψ, θ) * |∇ψ|(ψ, θ) dθ / (2π)
```
"""
function compute_surface_area(
    equil::Equilibrium.PlasmaEquilibrium,
    psi::Float64
)::Float64
    # Get equilibrium quantities at this surface
    sq_vals = Equilibrium.Splines.spline_eval!(equil.sq, psi)

    # sq(3) contains dV/dψ (volume derivative)
    dV_dpsi = sq_vals[3]

    # Approximate area from volume derivative
    # For a torus: dV/dψ ≈ A * 2π * √ψ (roughly)
    # So: A ≈ dV/dψ / (2π * √ψ)
    if psi > 1e-10
        area = abs(dV_dpsi) / (2π * sqrt(psi))
    else
        # Near axis, use limiting value
        area = abs(dV_dpsi) / (2π * 1e-5)
    end

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
    # Take maximum over all forcing modes
    for s in 1:msing
        sing_surf = ffs_intr.sing[s]

        # Island half-width: w/2 = √|island_width_sq|
        max_island_sq = maximum(abs.(state.island_width_sq[s, :]))
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
