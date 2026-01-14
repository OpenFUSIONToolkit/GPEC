"""
    compute_asymptotic_solutions(poloidal_angle, growth_parameter, asymptotic_spline, eigenfunctions, reference_angle) -> Matrix{Float64}

Computes the asymptotic pseudo-solutions (alpha series) for the ballooning equation
at a given extended poloidal angle. These solutions are used as boundary conditions for
the numerical ODE integration.

## Arguments

  - `poloidal_angle::Float64`: Extended poloidal angle θ.
  - `growth_parameter::Float64`: Growth parameter α (from Mercier criterion).
  - `asymptotic_spline::Splines.CubicSpline`: Spline containing asymptotic expansion coefficients.
  - `eigenfunctions::Matrix{Float64}`: Zeroth-order eigenfunctions (2×2 matrix).
  - `reference_angle::Float64`: Reference poloidal angle (usually 0).

## Returns

  - `asymptotic_matrix::Matrix{Float64}`: 2×2 matrix where each column is an asymptotic solution.
"""
function compute_asymptotic_solutions(poloidal_angle::Float64, growth_parameter::Float64,
    asymptotic_spline::Spl.CubicSpline1D,
    eigenfunctions::Matrix{Float64}, reference_angle::Float64)
    asymptotic_matrix = zeros(Float64, 2, 2)
    first_order_correction = zeros(Float64, 2, 2)

    # Evaluate alpha series expansion at poloidal_angle
    asymptotic_eval = Spl.spline_eval(asymptotic_spline, [poloidal_angle])
    asymptotic_coeffs = asymptotic_eval[1, :]
    angle_offset = poloidal_angle - reference_angle

    # First-order terms from the asymptotic expansion
    first_order_correction[1, 1] = eigenfunctions[1, 1] + asymptotic_coeffs[1] / angle_offset
    first_order_correction[2, 1] = eigenfunctions[2, 1] + asymptotic_coeffs[2] / angle_offset
    first_order_correction[1, 2] = eigenfunctions[1, 2] + asymptotic_coeffs[3] / angle_offset
    first_order_correction[2, 2] = eigenfunctions[2, 2] + asymptotic_coeffs[4] / angle_offset

    # Transformed solutions
    asymptotic_matrix[1, 1] = first_order_correction[1, 1]
    asymptotic_matrix[1, 2] = first_order_correction[1, 2]
    asymptotic_matrix[2, 1] = asymptotic_coeffs[5] * first_order_correction[1, 1] + first_order_correction[2, 1]
    asymptotic_matrix[2, 2] = asymptotic_coeffs[5] * first_order_correction[1, 2] + first_order_correction[2, 2]

    # CORRECTED: Match Fortran's thfac = ABS(dtheta)^(alpha+0.5) / dtheta
    # This equals: sign(dtheta) * |dtheta|^(alpha - 0.5)
    alpha_scaling = sign(angle_offset) * abs(angle_offset)^(growth_parameter - 0.5)

    asymptotic_matrix[1, 1] *= alpha_scaling
    asymptotic_matrix[2, 1] *= alpha_scaling * angle_offset
    asymptotic_matrix[1, 2] /= (alpha_scaling * angle_offset)
    asymptotic_matrix[2, 2] /= alpha_scaling

    return asymptotic_matrix
end

# ======================================================================
#   ODE System Definition
#   Differential equations for marginal ballooning stability
# ======================================================================
"""
    compute_ballooning_ode!(derivatives, solution, parameters, poloidal_angle)

Defines the RHS of the first-order ODE system for ideal marginal ballooning modes.
Used by the DifferentialEquations.jl ODE solver.

The second-order equation `d/dθ(f·dy/dθ) + g·y = 0` is rewritten as a 2x1 system:

  - `dy₁/dθ = y₂ / f`
  - `dy₂/dθ = -g · y₁`

where y₁ is the solution and y₂ = f·dy/dθ.

## Arguments

  - `derivatives::Vector{Float64}`: Output vector [dy₁/dθ, dy₂/dθ].
  - `solution::Vector{Float64}`: Current state vector [y₁, y₂].
  - `parameters::Tuple`: (ode_coefficient_spline, reference_angle).
  - `poloidal_angle::Float64`: Current poloidal angle θ (independent variable).
"""
function compute_ballooning_ode!(derivatives, solution, parameters, poloidal_angle)
    ode_coefficient_spline, reference_angle = parameters

    # Evaluate spline coefficients at current poloidal angle
    coeff_matrix = Spl.spline_eval(ode_coefficient_spline, [poloidal_angle])
    coefficients = coeff_matrix[1, :]  # Extract as vector
    angle_offset = poloidal_angle - reference_angle

    # ODE coefficient f: magnetic shear-related curvature term
    f_coefficient = coefficients[1] * (angle_offset + coefficients[2])^2 + coefficients[3]

    # ODE coefficient g: pressure-driven instability growth term
    g_coefficient = coefficients[4] + angle_offset * coefficients[5]

    # RHS system
    derivatives[1] = solution[2] / f_coefficient
    derivatives[2] = -solution[1] * g_coefficient
end

# ======================================================================
#   ODE Integration
#   Integrates the ideal marginal ballooning equations
# ======================================================================
"""
    integrate_ballooning_ode(flux_surface_index, growth_parameter, ode_coefficient_spline,
                             asymptotic_spline, eigenfunctions, reference_angle, control) -> (Float64, Float64)

Integrates the ballooning ODE from `-θ_max` to `+θ_max` using adaptive RK integration.

## Returns

  - `(coefficient_1, coefficient_2)`: Asymptotic coefficients determining stability.
"""
function integrate_ballooning_ode(flux_surface_index::Int, growth_parameter::Float64,
    ode_coefficient_spline::Spl.CubicSpline1D,
    asymptotic_spline::Spl.CubicSpline1D,
    eigenfunctions::Matrix{Float64}, reference_angle::Float64,
    control::DconControl)


    TOLERANCE = 1e-5
    MINIMUM_THETA = 1.0
    MINIMUM_STEP = 1e-10

    # Determine integration limits
    curvature_ratio = ode_coefficient_spline.fs[:, 3] ./ ode_coefficient_spline.fs[:, 1]
    theta_max_absolute = sqrt(maximum(abs.(curvature_ratio))) * 10.0 * control.thmax0
    theta_max = min(theta_max_absolute, 100.0)
    # Isn't it too low theta_max for high ipsi compared to fortran? we have to check this.

    # Initial conditions from asymptotic solution
    theta_start = -theta_max
    asymptotic_start = compute_asymptotic_solutions(theta_start, growth_parameter, asymptotic_spline,
        eigenfunctions, reference_angle)
    initial_condition = Vector{Float64}(asymptotic_start[:, 2]) * sinh(1.0)

    # Set up and solve ODE problem
    ode_problem = ODEProblem(compute_ballooning_ode!, initial_condition,
        (theta_start, theta_max),
        (ode_coefficient_spline, reference_angle))

    try
        ode_solution = solve(ode_problem, DP5(); reltol=TOLERANCE, abstol=TOLERANCE^2,
            dtmin=MINIMUM_STEP, adaptive=true)

        if ode_solution.retcode == ReturnCode.Success
            solution_final = ode_solution.u[end]
            theta_final = ode_solution.t[end]
            if theta_final < theta_max
                return NaN, NaN
            end

            # Asymptotic matching
            asymptotic_final = compute_asymptotic_solutions(theta_final, growth_parameter, asymptotic_spline,
                eigenfunctions, reference_angle)

            det_asymptotic = asymptotic_final[1, 1] * asymptotic_final[2, 2] -
                             asymptotic_final[1, 2] * asymptotic_final[2, 1]

            coefficient_1 = (solution_final[1] * asymptotic_final[2, 2] -
                             solution_final[2] * asymptotic_final[1, 2]) / det_asymptotic
            coefficient_2 = (solution_final[2] * asymptotic_final[1, 1] -
                             solution_final[1] * asymptotic_final[2, 1]) / det_asymptotic

            return coefficient_1, coefficient_2
        else
            # retcode=Success with early termination is normal (ODE reached stopping condition)
            if ode_solution.retcode != ReturnCode.Success
                @warn "ODE integration failed for surface ipsi = $flux_surface_index (retcode: $(ode_solution.retcode))"
            end
            return NaN, NaN
        end
    catch e
        @warn "ODE integration failed for surface ipsi = $flux_surface_index: $(e)"
        return NaN, NaN
    end
end


# ----------------------------------------------------------------------
#   subprogram 2. bal_prep.
#   computes coefficients for the ideal marginal ballooning equations.
# ----------------------------------------------------------------------
"""
    bal_prep(...) -> (di, alpha, bf, bg, v0, theta0)

Prepares all coefficients and splines required for the ballooning stability analysis
on a single magnetic flux surface.

## Returns

  - `mercier_criterion::Float64`: Mercier criterion di. If > 0, surface is stable.
  - `growth_parameter::Float64`: Growth rate exponent α.
  - `ode_coefficient_spline::CubicSpline`: Spline for ODE system.
  - `asymptotic_spline::CubicSpline`: Spline for asymptotic matching.
  - `zeroth_order_eigenfunctions::Matrix{Float64}`: Zeroth-order eigenfunctions (2×2).
  - `reference_angle::Float64`: Reference poloidal angle.
"""
function prepare_ballooning_coefficients(flux_surface_index::Int, plasma_eq::Equilibrium.PlasmaEquilibrium)
    # shorter aliases for equilibrium structs
    sq = plasma_eq.sq
    rzphi = plasma_eq.rzphi
    mtheta = length(rzphi.ys) - 1
    theta_grid = Vector(rzphi.ys)

    # surface quantities
    two_pi_f = sq.fs[flux_surface_index, 1]
    pressure_gradient = sq.fs1[flux_surface_index, 2] # p'
    q = sq.fs[flux_surface_index, 4]
    q_derivative = sq.fs1[flux_surface_index, 4] # q'
    chi_prime = 2pi * plasma_eq.psio

    # arrays to be filled
    jac = zeros(mtheta + 1)
    b1 = zeros(mtheta + 1)
    bsq = zeros(mtheta + 1)
    dbdb0 = zeros(mtheta + 1)
    dbdb1 = zeros(mtheta + 1)
    dbdb2 = zeros(mtheta + 1)
    kappan = zeros(mtheta + 1)
    kappas = zeros(mtheta + 1)
    fx_psi = zeros(4, mtheta + 1)  # Store fx values (4 components) for each theta point

    # loop over poloidal angle using direct array access at grid points
    ipsi = flux_surface_index
    for itheta in 1:(mtheta+1)
        # Direct array access for values and derivatives
        f1 = rzphi.fs[ipsi, itheta, 1]
        f2 = rzphi.fs[ipsi, itheta, 2]
        f4 = rzphi.fs[ipsi, itheta, 4]
        fx1 = rzphi.fsx[ipsi, itheta, 1]
        fx2 = rzphi.fsx[ipsi, itheta, 2]
        fx3 = rzphi.fsx[ipsi, itheta, 3]
        fx4 = rzphi.fsx[ipsi, itheta, 4]
        fy1 = rzphi.fsy[ipsi, itheta, 1]
        fy2 = rzphi.fsy[ipsi, itheta, 2]
        fy3 = rzphi.fsy[ipsi, itheta, 3]

        fx_psi[:, itheta] .= (fx1, fx2, fx3, fx4)  # Store fx for later use

        theta = rzphi.ys[itheta]
        rfac = sqrt(f1)
        eta = 2pi * (theta + f2)
        r = plasma_eq.ro + rfac * cos(eta)
        jac[itheta] = f4

        # contravariant basis vectors v^i
        v = zeros(3, 3)
        v[1, 1] = fx1 / (2 * rfac * jac[itheta])
        v[1, 2] = fx2 * 2pi * rfac / jac[itheta]
        v[1, 3] = fx3 * r / jac[itheta]
        v[2, 1] = fy1 / (2 * rfac * jac[itheta])
        v[2, 2] = (1 + fy2) * 2pi * rfac / jac[itheta]
        v[2, 3] = fy3 * r / jac[itheta]
        v[3, 3] = 2pi * r / jac[itheta]

        # covariant basis vectors w_i (direct method)
        w = zeros(3, 3)
        w[1, 1] = (1 + fy2) * (2pi)^2 * rfac * r / jac[itheta]
        w[1, 2] = -fy1 * pi * r / (rfac * jac[itheta])
        w[2, 1] = -fx2 * (2pi)^2 * r * rfac / jac[itheta]
        w[2, 2] = fx1 * pi * r / (rfac * jac[itheta])
        w[3, 1] = (fx2 * fy3 - fx3 * (1 + fy2)) * 2pi * r * rfac / jac[itheta]
        w[3, 2] = (fx3 * fy1 - fx1 * fy3) * r / (2 * rfac * jac[itheta])
        w[3, 3] = 1 / (2pi * r)

        # store physical quantities
        grad_psi = @view v[1, :]
        e_theta = @view v[2, :]
        e_phi = @view v[3, :]

        B_vec = (e_theta + q * e_phi) * chi_prime
        b1[itheta] = dot(B_vec, grad_psi)
        bsq[itheta] = dot(B_vec, B_vec)

        grad_alpha = @view w[1, :]
        grad_theta = @view w[2, :]
        grad_phi = @view w[3, :]
        gtheta = grad_phi - q * grad_theta

        dbdb0[itheta] = dot(grad_alpha, grad_alpha) * q_derivative^2
        dbdb1[itheta] = -2 * dot(gtheta, grad_alpha) * q_derivative
        dbdb2[itheta] = dot(gtheta, gtheta)
    end

    # compute curvature terms
    spl0 = Spl.CubicSpline1D(theta_grid, hcat(1 ./ bsq, jac .* b1 ./ bsq); bctype="periodic")

    kappas .= -spl0.fs1[:, 1] .* two_pi_f ./ (2 .* jac)
    kappan .= ((pressure_gradient ./ bsq .- fx_psi[4, :] ./ jac) ./ chi_prime .+
               two_pi_f .* q_derivative ./ (bsq .* jac) .+ spl0.fs1[:, 2] ./ jac) ./ 2.0

    # compute coefficients for ballooning equation and store in spline 'bf'
    bf_fs = zeros(mtheta + 1, 5)
    jacfac = jac ./ chi_prime
    bf_fs[:, 1] = dbdb0 ./ (bsq .* jacfac)
    bf_fs[:, 2] = dbdb1 ./ (2 .* dbdb0)
    bf_fs[:, 3] = (dbdb2 - dbdb1 .^ 2 ./ (4 .* dbdb0)) ./ (bsq .* jacfac)
    bf_fs[:, 4] = 2 .* kappan .* pressure_gradient ./ chi_prime .* jacfac
    bf_fs[:, 5] = -2 .* kappas .* pressure_gradient ./ chi_prime .* q_derivative .* jacfac
    ode_coefficient_spline = Spl.CubicSpline1D(theta_grid, bf_fs; bctype="periodic")

    # initialize spline 'bg'
    spl0_bg = Spl.CubicSpline1D(theta_grid, -pressure_gradient .* q_derivative .* two_pi_f ./ (bsq .* chi_prime^2); bctype="periodic")
    Spl.integrate!(spl0_bg)
    bg_fs = zeros(mtheta + 1, 5)
    bg_fs[:, 5] = spl0_bg.fs[:, 1] .- spl0_bg.fsi[end, 1]

    # Mercier criterion calculation
    spl1_fs = zeros(mtheta + 1, 4)
    for itheta in 1:(mtheta+1)
        c0 = zeros(2, 2)
        c0[1, 1] = 0.5
        c0[1, 2] = 1 / ode_coefficient_spline.fs[itheta, 1]
        c0[2, 1] = -ode_coefficient_spline.fs[itheta, 4]
        c0[2, 2] = -0.5
        spl1_fs[itheta, 1] = c0[1, 1] + c0[1, 2]*bg_fs[itheta, 5]
        spl1_fs[itheta, 2] = c0[1, 2]
        spl1_fs[itheta, 3] = c0[2, 1] + (c0[2, 2]-c0[1, 1]-c0[1, 2]*bg_fs[itheta, 5])*bg_fs[itheta, 5]
        spl1_fs[itheta, 4] = -spl1_fs[itheta, 1]
    end

    spl1 = Spl.CubicSpline1D(theta_grid, spl1_fs; bctype="periodic")
    Spl.integrate!(spl1)

    d0bar = [spl1.fsi[end, 1] spl1.fsi[end, 2]; spl1.fsi[end, 3] spl1.fsi[end, 4]]

    di = det(d0bar)

    # if stable by Mercier, no more work to do
    if di > 0
        return di, NaN, ode_coefficient_spline, Spl.CubicSpline1D(theta_grid, bg_fs; bctype="periodic"), zeros(2, 2), 0.0
    end

    alpha = sqrt(-di)

    # compute zeroth-order eigenfunctions
    v0 = zeros(2, 2)
    v0[1, 1] = 1.0
    v0[1, 2] = 1.0
    v0[2, 1] = -(d0bar[1, 1] - alpha) / d0bar[1, 2]
    v0[2, 2] = -(d0bar[1, 1] + alpha) / d0bar[1, 2]

    # ===== Higher-order corrections for the asymptotic solution =====
    # Compute derivatives for first-order terms (v1)
    # This follows the structure from bal_prep.f lines 300-340
    spl2_fs = zeros(mtheta + 1, 4)

    for itheta in 1:(mtheta+1)
        # Matrix product: (spl1 - alpha*I) * v0
        spl2_fs[itheta, 1] = (spl1.fs[itheta, 1] - alpha) * v0[1, 1] + spl1.fs[itheta, 2] * v0[2, 1]
        spl2_fs[itheta, 2] = spl1.fs[itheta, 3] * v0[1, 1] + (spl1.fs[itheta, 4] - alpha) * v0[2, 1]
        spl2_fs[itheta, 3] = (spl1.fs[itheta, 1] + alpha) * v0[1, 2] + spl1.fs[itheta, 2] * v0[2, 2]
        spl2_fs[itheta, 4] = spl1.fs[itheta, 3] * v0[1, 2] + (spl1.fs[itheta, 4] + alpha) * v0[2, 2]
    end

    spl2 = Spl.CubicSpline1D(theta_grid, spl2_fs; bctype="periodic")
    Spl.integrate!(spl2)
    # CRITICAL: Replace fs with integrated values (matching Fortran's spl2%fs=spl2%fsi)
    spl2_fs_integrated = copy(spl2.fsi)  # Save integrated values
    spl2 = Spl.CubicSpline1D(theta_grid, spl2_fs_integrated; bctype="periodic")


    # Compute derivatives for second-order terms
    spl3_fs = zeros(mtheta + 1, 4)

    for itheta in 1:(mtheta+1)
        # d1 matrix elements (first-order corrections to d matrix)
        fexp = -spl1.fs[itheta, 2] * ode_coefficient_spline.fs[itheta, 2] * 2
        d1_11 = fexp * bg_fs[itheta, 5]
        d1_12 = fexp
        d1_21 = -fexp * bg_fs[itheta, 5]^2
        d1_22 = -d1_11

        # Now use in spl3_fs calculation
        spl3_fs[itheta, 1] = (spl1.fs1[itheta, 1] + 1 - alpha) * spl2.fs[itheta, 1] +
                             spl1.fs[itheta, 2] * spl2.fs[itheta, 2] +
                             d1_11 * v0[1, 1] + d1_12 * v0[2, 1]

        spl3_fs[itheta, 2] = spl1.fs[itheta, 3] * spl2.fs[itheta, 1] +
                             (spl1.fs[itheta, 4] + 1 - alpha) * spl2.fs[itheta, 2] +
                             d1_21 * v0[1, 1] + d1_22 * v0[2, 1]

        spl3_fs[itheta, 3] = (spl1.fs[itheta, 1] + 1 + alpha) * spl2.fs[itheta, 3] +
                             spl1.fs[itheta, 2] * spl2.fs[itheta, 4] +
                             d1_11 * v0[1, 2] + d1_12 * v0[2, 2]

        spl3_fs[itheta, 4] = spl1.fs[itheta, 3] * spl2.fs[itheta, 3] +
                             (spl1.fs[itheta, 4] + 1 + alpha) * spl2.fs[itheta, 4] +
                             d1_21 * v0[1, 2] + d1_22 * v0[2, 2]
    end

    spl3 = Spl.CubicSpline1D(theta_grid, spl3_fs; bctype="periodic")
    Spl.integrate!(spl3)

    # Compute first-order constants for both eigenfunctions
    # First eigenfunction (with -alpha correction)
    d = zeros(2, 2)
    d[2, 1] = d0bar[2, 1]
    d[1, 2] = d0bar[1, 2]
    d[1, 1] = d0bar[1, 1] + 1 - alpha
    d[2, 2] = d0bar[2, 2] + 1 - alpha

    det_d = d[1, 1] * d[2, 2] - d[1, 2] * d[2, 1]
    v10 = zeros(2, 2)
    v10[1, 1] = (d[1, 2] * spl3.fsi[end, 2] - d[2, 2] * spl3.fsi[end, 1]) / det_d
    v10[2, 1] = (d[2, 1] * spl3.fsi[end, 1] - d[1, 1] * spl3.fsi[end, 2]) / det_d

    # Second eigenfunction (with +alpha correction)
    d[1, 1] = d0bar[1, 1] + 1 + alpha
    d[2, 2] = d0bar[2, 2] + 1 + alpha
    det_d = d[1, 1] * d[2, 2] - d[1, 2] * d[2, 1]
    v10[1, 2] = (d[1, 2] * spl3.fsi[end, 4] - d[2, 2] * spl3.fsi[end, 3]) / det_d
    v10[2, 2] = (d[2, 1] * spl3.fsi[end, 3] - d[1, 1] * spl3.fsi[end, 4]) / det_d

    # Assemble final bg spline with higher-order corrections
    bg_fs[:, 1] = spl2.fs[:, 1] .+ v10[1, 1]
    bg_fs[:, 2] = spl2.fs[:, 2] .+ v10[2, 1]
    bg_fs[:, 3] = spl2.fs[:, 3] .+ v10[1, 2]
    bg_fs[:, 4] = spl2.fs[:, 4] .+ v10[2, 2]

    bg = Spl.CubicSpline1D(theta_grid, bg_fs; bctype="periodic")

    reference_angle = 0.0 # Central poloidal angle (typically 0)
    return di, alpha, ode_coefficient_spline, bg, v0, reference_angle
end


# ======================================================================
#   Main Driver for Ballooning Stability Analysis
#   Computes ballooning stability criterion over all flux surfaces
# ======================================================================
"""
    compute_ballooning_stability!(ctrl, locstab_fs, plasma_eq)

Main driver routine for ballooning stability analysis. Iterates over all
magnetic flux surfaces, prepares coefficients, and integrates the ballooning
equation if the surface is Mercier unstable.

## Arguments

  - `ctrl::DconControl`: Control parameters for the analysis.
  - `locstab_fs::Matrix{Float64}`: Local stability matrix to store results (modified in place).
  - `plasma_eq::Equilibrium.PlasmaEquilibrium`: Plasma equilibrium data.

This function modifies `locstab_fs` in place with:

  - Column 1: Mercier criterion × ψ
  - Columns 4-5: Asymptotic coefficients ca₁ and ca₂
"""
function compute_ballooning_stability!(ctrl::DconControl, locstab_fs::Matrix{Float64}, plasma_eq::Equilibrium.PlasmaEquilibrium)

    if ctrl.verbose
        println("Evaluating high-n ballooning criterion...")
    end

    num_psi = length(plasma_eq.sq.xs)

    # Loop over flux surfaces
    for flux_surface_index in 1:num_psi

        # Prepare coefficients and check Mercier criterion
        mercier_criterion, growth_param, ode_coeff_spline, asymp_spline, zeroth_eigs, ref_angle = prepare_ballooning_coefficients(flux_surface_index, plasma_eq)

        # Store Mercier criterion in locstab matrix (matches Fortran output)
        # Assuming locstab columns are [di*psi, (di+(h-0.5)^2)*psi, h, ca1, ca2]
        locstab_fs[flux_surface_index, 1] = mercier_criterion * plasma_eq.sq.xs[flux_surface_index]
        # Note: h and the second term are not calculated here as in mercier_scan, focusing on ballooning.

        # If Mercier unstable, proceed with ballooning integration
        if mercier_criterion <= 0 && mercier_criterion >= -1e4 && plasma_eq.sq.xs[flux_surface_index] <= 1.0
            ca1, ca2 = integrate_ballooning_ode(flux_surface_index, growth_param, ode_coeff_spline, asymp_spline, zeroth_eigs, ref_angle, ctrl)

            # Store final asymptotic coefficients if integration reached theta_max
            if isfinite(ca1) && isfinite(ca2)
                locstab_fs[flux_surface_index, 4] = ca1
                locstab_fs[flux_surface_index, 5] = ca2
            end
        end
    end

    if ctrl.verbose
        println("Ballooning analysis complete.")
    end

end
