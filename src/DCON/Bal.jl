"""
    compute_asymptotic_solutions(poloidal_angle, growth_parameter, asymptotic_interp, asymp_buffer, eigenfunctions, reference_angle) -> Matrix{Float64}

Computes the asymptotic pseudo-solutions (alpha series) for the ballooning equation
at a given extended poloidal angle. These solutions are used as boundary conditions for
the numerical ODE integration.

## Arguments

  - `poloidal_angle::Float64`: Extended poloidal angle θ.
  - `growth_parameter::Float64`: Growth parameter α (from Mercier criterion).
  - `asymptotic_interp::CubicSeriesInterpolant`: SeriesInterpolant for 5 asymptotic coefficients.
  - `asymp_buffer::Vector{Float64}`: Pre-allocated buffer for in-place interpolation.
  - `eigenfunctions::Matrix{Float64}`: Zeroth-order eigenfunctions (2×2 matrix).
  - `reference_angle::Float64`: Reference poloidal angle (usually 0).

## Returns

  - `asymptotic_matrix::Matrix{Float64}`: 2×2 matrix where each column is an asymptotic solution.
"""
function compute_asymptotic_solutions(poloidal_angle::Float64, growth_parameter::Float64,
    asymptotic_interp::CubicSeriesInterpolant, asymp_buffer::Vector{Float64},
    eigenfunctions::Matrix{Float64}, reference_angle::Float64)
    asymptotic_matrix = zeros(Float64, 2, 2)
    first_order_correction = zeros(Float64, 2, 2)

    # Evaluate all asymptotic coefficients in-place (zero allocation)
    asymptotic_interp(asymp_buffer, poloidal_angle)
    a1, a2, a3, a4, a5 = asymp_buffer[1], asymp_buffer[2], asymp_buffer[3], asymp_buffer[4], asymp_buffer[5]
    angle_offset = poloidal_angle - reference_angle

    # First-order terms from the asymptotic expansion
    first_order_correction[1, 1] = eigenfunctions[1, 1] + a1 / angle_offset
    first_order_correction[2, 1] = eigenfunctions[2, 1] + a2 / angle_offset
    first_order_correction[1, 2] = eigenfunctions[1, 2] + a3 / angle_offset
    first_order_correction[2, 2] = eigenfunctions[2, 2] + a4 / angle_offset

    # Transformed solutions
    asymptotic_matrix[1, 1] = first_order_correction[1, 1]
    asymptotic_matrix[1, 2] = first_order_correction[1, 2]
    asymptotic_matrix[2, 1] = a5 * first_order_correction[1, 1] + first_order_correction[2, 1]
    asymptotic_matrix[2, 2] = a5 * first_order_correction[1, 2] + first_order_correction[2, 2]

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
  - `parameters::Tuple`: (ode_coeff_interp, reference_angle, coeff_buffer, hint).
  - `poloidal_angle::Float64`: Current poloidal angle θ (independent variable).
"""
function compute_ballooning_ode!(derivatives, solution, parameters, poloidal_angle)
    ode_coeff_interp, reference_angle, coeff_buffer, hint = parameters

    # Evaluate all ODE coefficients in-place (zero allocation)
    ode_coeff_interp(coeff_buffer, poloidal_angle; hint=hint)
    c1, c2, c3, c4, c5 = coeff_buffer[1], coeff_buffer[2], coeff_buffer[3], coeff_buffer[4], coeff_buffer[5]
    angle_offset = poloidal_angle - reference_angle

    # ODE coefficient f: magnetic shear-related curvature term
    f_coefficient = c1 * (angle_offset + c2)^2 + c3

    # ODE coefficient g: pressure-driven instability growth term
    g_coefficient = c4 + angle_offset * c5

    # RHS system
    derivatives[1] = solution[2] / f_coefficient
    derivatives[2] = -solution[1] * g_coefficient
end

# ======================================================================
#   ODE Integration
#   Integrates the ideal marginal ballooning equations
# ======================================================================
"""
    integrate_ballooning_ode(ipsi, growth_parameter, ode_coeff_interp,
                             asymptotic_interp, eigenfunctions, reference_angle, control) -> (Float64, Float64)

Integrates the ballooning ODE from `-θ_max` to `+θ_max` using adaptive RK integration.

## Returns

  - `(coefficient_1, coefficient_2)`: Asymptotic coefficients determining stability.
"""
function integrate_ballooning_ode(ipsi::Int, growth_parameter::Float64,
    ode_coeff_interp::CubicSeriesInterpolant,
    asymptotic_interp::CubicSeriesInterpolant,
    eigenfunctions::Matrix{Float64}, reference_angle::Float64,
    control::DconControl)

    TOLERANCE = 1e-5
    MINIMUM_STEP = 1e-10

    # Determine integration limits (access grid values from SeriesInterpolant)
    curvature_ratio = ode_coeff_interp.y[:, 3] ./ ode_coeff_interp.y[:, 1]
    theta_max_absolute = sqrt(maximum(abs.(curvature_ratio))) * 10.0 * control.thmax0
    theta_max = min(theta_max_absolute, 100.0)

    # Pre-allocate buffers for in-place interpolation (zero allocation)
    coeff_buffer = Vector{Float64}(undef, 5)
    asymp_buffer = Vector{Float64}(undef, 5)
    hint = Ref(1)

    # Initial conditions from asymptotic solution
    theta_start = -theta_max
    asymptotic_start = compute_asymptotic_solutions(theta_start, growth_parameter, asymptotic_interp,
        asymp_buffer, eigenfunctions, reference_angle)
    initial_condition = Vector{Float64}(asymptotic_start[:, 2]) * sinh(1.0)

    # Set up and solve ODE problem
    ode_problem = ODEProblem(compute_ballooning_ode!, initial_condition,
        (theta_start, theta_max),
        (ode_coeff_interp, reference_angle, coeff_buffer, hint))

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
            asymptotic_final = compute_asymptotic_solutions(theta_final, growth_parameter, asymptotic_interp,
                asymp_buffer, eigenfunctions, reference_angle)

            det_asymptotic = asymptotic_final[1, 1] * asymptotic_final[2, 2] -
                             asymptotic_final[1, 2] * asymptotic_final[2, 1]

            coefficient_1 = (solution_final[1] * asymptotic_final[2, 2] -
                             solution_final[2] * asymptotic_final[1, 2]) / det_asymptotic
            coefficient_2 = (solution_final[2] * asymptotic_final[1, 1] -
                             solution_final[1] * asymptotic_final[2, 1]) / det_asymptotic

            return coefficient_1, coefficient_2
        else
            @warn "ODE integration failed for surface ipsi = $ipsi (retcode: $(ode_solution.retcode))"
            return NaN, NaN
        end
    catch e
        @warn "ODE integration failed for surface ipsi = $ipsi: $(e)"
        return NaN, NaN
    end
end


# ----------------------------------------------------------------------
#   subprogram 2. bal_prep.
#   computes coefficients for the ideal marginal ballooning equations.
# ----------------------------------------------------------------------
"""
    prepare_ballooning_coefficients(ipsi, plasma_eq, hint) -> (di, alpha, ode_coeff_interp, asymptotic_interp, v0, theta0)

Prepares all coefficients and splines required for the ballooning stability analysis
on a single magnetic flux surface.

## Arguments

  - `ipsi::Int`: Flux surface index.
  - `plasma_eq::PlasmaEquilibrium`: Plasma equilibrium data.
  - `hint::Ref{Int}`: Spline search hint for sequential ipsi access.

## Returns

  - `mercier_criterion::Float64`: Mercier criterion di. If > 0, surface is stable.
  - `growth_parameter::Float64`: Growth rate exponent α.
  - `ode_coeff_interp::CubicSeriesInterpolant`: SeriesInterpolant for 5 ODE coefficients.
  - `asymptotic_interp::CubicSeriesInterpolant`: SeriesInterpolant for 5 asymptotic coefficients.
  - `zeroth_order_eigenfunctions::Matrix{Float64}`: Zeroth-order eigenfunctions (2×2).
  - `reference_angle::Float64`: Reference poloidal angle.
"""
function prepare_ballooning_coefficients(ipsi::Int, plasma_eq::Equilibrium.PlasmaEquilibrium, hint::Ref{Int})
    # shorter aliases for equilibrium structs
    profiles = plasma_eq.profiles
    rzphi = plasma_eq.rzphi
    mtheta = length(rzphi.ys) - 1
    theta_grid = Vector(rzphi.ys)

    # surface quantities
    psi = profiles.xs[ipsi]
    two_pi_f = profiles.F_spline.y[ipsi]
    pressure_gradient = profiles.P_deriv(psi; hint=hint)
    q = profiles.q_spline.y[ipsi]
    q_derivative = profiles.q_deriv(psi; hint=hint)
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

    # Pre-allocate basis vector matrices (reused each iteration)
    v = zeros(3, 3)  # contravariant basis vectors
    w = zeros(3, 3)  # covariant basis vectors

    # loop over poloidal angle using direct array access at grid points
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
        v[1, 1] = fx1 / (2 * rfac * jac[itheta])
        v[1, 2] = fx2 * 2pi * rfac / jac[itheta]
        v[1, 3] = fx3 * r / jac[itheta]
        v[2, 1] = fy1 / (2 * rfac * jac[itheta])
        v[2, 2] = (1 + fy2) * 2pi * rfac / jac[itheta]
        v[2, 3] = fy3 * r / jac[itheta]
        v[3, 3] = 2pi * r / jac[itheta]

        # covariant basis vectors w_i (direct method)
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

    # compute curvature terms using native cubic_interp with PeriodicBC
    spl0_interp = cubic_interp(theta_grid, hcat(1 ./ bsq, jac .* b1 ./ bsq); search=LinearBinary(), bc=PeriodicBC())
    spl0_d1 = deriv1(spl0_interp)
    # Evaluate derivatives at all theta points (returns Vector of Vectors, stack to matrix)
    spl0_fs1 = stack(spl0_d1.(theta_grid))

    kappas .= -spl0_fs1[:, 1] .* two_pi_f ./ (2 .* jac)
    kappan .= ((pressure_gradient ./ bsq .- fx_psi[4, :] ./ jac) ./ chi_prime .+
               two_pi_f .* q_derivative ./ (bsq .* jac) .+ spl0_fs1[:, 2] ./ jac) ./ 2.0

    # compute coefficients for ballooning equation and store
    bf_fs = zeros(mtheta + 1, 5)
    jacfac = jac ./ chi_prime
    bf_fs[:, 1] = dbdb0 ./ (bsq .* jacfac)
    bf_fs[:, 2] = dbdb1 ./ (2 .* dbdb0)
    bf_fs[:, 3] = (dbdb2 - dbdb1 .^ 2 ./ (4 .* dbdb0)) ./ (bsq .* jacfac)
    bf_fs[:, 4] = 2 .* kappan .* pressure_gradient ./ chi_prime .* jacfac
    bf_fs[:, 5] = -2 .* kappas .* pressure_gradient ./ chi_prime .* q_derivative .* jacfac
    ode_coeff_interp = cubic_interp(theta_grid, bf_fs; bc=PeriodicBC())

    # initialize bg values
    spl0_bg_fs = -pressure_gradient .* q_derivative .* two_pi_f ./ (bsq .* chi_prime^2)
    spl0_bg_fsi = Spl.cumulative_integral(theta_grid, spl0_bg_fs; bc=PeriodicBC())
    bg_fs = zeros(mtheta + 1, 5)
    bg_fs[:, 5] = spl0_bg_fsi .- spl0_bg_fsi[end]

    # Mercier criterion calculation
    spl1_fs = zeros(mtheta + 1, 4)
    for itheta in 1:(mtheta+1)
        c0 = zeros(2, 2)
        c0[1, 1] = 0.5
        c0[1, 2] = 1 / bf_fs[itheta, 1]
        c0[2, 1] = -bf_fs[itheta, 4]
        c0[2, 2] = -0.5
        spl1_fs[itheta, 1] = c0[1, 1] + c0[1, 2]*bg_fs[itheta, 5]
        spl1_fs[itheta, 2] = c0[1, 2]
        spl1_fs[itheta, 3] = c0[2, 1] + (c0[2, 2]-c0[1, 1]-c0[1, 2]*bg_fs[itheta, 5])*bg_fs[itheta, 5]
        spl1_fs[itheta, 4] = -spl1_fs[itheta, 1]
    end

    spl1_totals = Spl.total_integral(theta_grid, spl1_fs; bc=PeriodicBC())

    d0bar = [spl1_totals[1] spl1_totals[2]; spl1_totals[3] spl1_totals[4]]

    di = det(d0bar)

    # if stable by Mercier, no more work to do - return empty asymptotic data
    if di > 0
        asymptotic_interp = cubic_interp(theta_grid, bg_fs; bc=PeriodicBC())
        return di, NaN, ode_coeff_interp, asymptotic_interp, zeros(2, 2), 0.0
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
    spl2_fs = zeros(mtheta + 1, 4)

    for itheta in 1:(mtheta+1)
        # Matrix product: (spl1 - alpha*I) * v0
        spl2_fs[itheta, 1] = (spl1_fs[itheta, 1] - alpha) * v0[1, 1] + spl1_fs[itheta, 2] * v0[2, 1]
        spl2_fs[itheta, 2] = spl1_fs[itheta, 3] * v0[1, 1] + (spl1_fs[itheta, 4] - alpha) * v0[2, 1]
        spl2_fs[itheta, 3] = (spl1_fs[itheta, 1] + alpha) * v0[1, 2] + spl1_fs[itheta, 2] * v0[2, 2]
        spl2_fs[itheta, 4] = spl1_fs[itheta, 3] * v0[1, 2] + (spl1_fs[itheta, 4] + alpha) * v0[2, 2]
    end

    spl2_fsi = Spl.cumulative_integral(theta_grid, spl2_fs; bc=PeriodicBC())
    # CRITICAL: Use integrated values as the new fs (matching Fortran's spl2%fs=spl2%fsi)
    spl2_fs_new = copy(spl2_fsi)

    # Compute derivatives of spl1 for second-order terms
    spl1_interp = cubic_interp(theta_grid, spl1_fs; search=LinearBinary(), bc=PeriodicBC())
    spl1_d1 = deriv1(spl1_interp)
    # Evaluate derivatives at all theta points (returns Vector of Vectors, stack to matrix)
    spl1_fs1 = stack(spl1_d1.(theta_grid))

    # Compute derivatives for second-order terms
    spl3_fs = zeros(mtheta + 1, 4)

    for itheta in 1:(mtheta+1)
        # d1 matrix elements (first-order corrections to d matrix)
        fexp = -spl1_fs[itheta, 2] * bf_fs[itheta, 2] * 2
        d1_11 = fexp * bg_fs[itheta, 5]
        d1_12 = fexp
        d1_21 = -fexp * bg_fs[itheta, 5]^2
        d1_22 = -d1_11

        # Now use in spl3_fs calculation
        spl3_fs[itheta, 1] = (spl1_fs1[itheta, 1] + 1 - alpha) * spl2_fs_new[itheta, 1] +
                             spl1_fs[itheta, 2] * spl2_fs_new[itheta, 2] +
                             d1_11 * v0[1, 1] + d1_12 * v0[2, 1]

        spl3_fs[itheta, 2] = spl1_fs[itheta, 3] * spl2_fs_new[itheta, 1] +
                             (spl1_fs[itheta, 4] + 1 - alpha) * spl2_fs_new[itheta, 2] +
                             d1_21 * v0[1, 1] + d1_22 * v0[2, 1]

        spl3_fs[itheta, 3] = (spl1_fs[itheta, 1] + 1 + alpha) * spl2_fs_new[itheta, 3] +
                             spl1_fs[itheta, 2] * spl2_fs_new[itheta, 4] +
                             d1_11 * v0[1, 2] + d1_12 * v0[2, 2]

        spl3_fs[itheta, 4] = spl1_fs[itheta, 3] * spl2_fs_new[itheta, 3] +
                             (spl1_fs[itheta, 4] + 1 + alpha) * spl2_fs_new[itheta, 4] +
                             d1_21 * v0[1, 2] + d1_22 * v0[2, 2]
    end

    spl3_totals = Spl.total_integral(theta_grid, spl3_fs; bc=PeriodicBC())

    # Compute first-order constants for both eigenfunctions
    # First eigenfunction (with -alpha correction)
    d = zeros(2, 2)
    d[2, 1] = d0bar[2, 1]
    d[1, 2] = d0bar[1, 2]
    d[1, 1] = d0bar[1, 1] + 1 - alpha
    d[2, 2] = d0bar[2, 2] + 1 - alpha

    det_d = d[1, 1] * d[2, 2] - d[1, 2] * d[2, 1]
    v10 = zeros(2, 2)
    v10[1, 1] = (d[1, 2] * spl3_totals[2] - d[2, 2] * spl3_totals[1]) / det_d
    v10[2, 1] = (d[2, 1] * spl3_totals[1] - d[1, 1] * spl3_totals[2]) / det_d

    # Second eigenfunction (with +alpha correction)
    d[1, 1] = d0bar[1, 1] + 1 + alpha
    d[2, 2] = d0bar[2, 2] + 1 + alpha
    det_d = d[1, 1] * d[2, 2] - d[1, 2] * d[2, 1]
    v10[1, 2] = (d[1, 2] * spl3_totals[4] - d[2, 2] * spl3_totals[3]) / det_d
    v10[2, 2] = (d[2, 1] * spl3_totals[3] - d[1, 1] * spl3_totals[4]) / det_d

    # Assemble final bg with higher-order corrections
    bg_fs[:, 1] = spl2_fs_new[:, 1] .+ v10[1, 1]
    bg_fs[:, 2] = spl2_fs_new[:, 2] .+ v10[2, 1]
    bg_fs[:, 3] = spl2_fs_new[:, 3] .+ v10[1, 2]
    bg_fs[:, 4] = spl2_fs_new[:, 4] .+ v10[2, 2]

    asymptotic_interp = cubic_interp(theta_grid, bg_fs; bc=PeriodicBC())

    reference_angle = 0.0 # Central poloidal angle (typically 0)
    return di, alpha, ode_coeff_interp, asymptotic_interp, v0, reference_angle
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

    profiles = plasma_eq.profiles
    num_psi = length(profiles.xs)

    # Loop over flux surfaces
    hint = Ref(1)  # Shared hint for sequential psi access
    for ipsi in 1:num_psi

        # Prepare coefficients and check Mercier criterion
        mercier_criterion, growth_param, ode_coeff_interp, asymptotic_interp, zeroth_eigs, ref_angle = prepare_ballooning_coefficients(ipsi, plasma_eq, hint)

        # Store Mercier criterion in locstab matrix (matches Fortran output)
        locstab_fs[ipsi, 1] = mercier_criterion * profiles.xs[ipsi]

        # If Mercier unstable, proceed with ballooning integration
        if mercier_criterion <= 0 && mercier_criterion >= -1e4 && profiles.xs[ipsi] <= 1.0
            ca1, ca2 = integrate_ballooning_ode(ipsi, growth_param, ode_coeff_interp, asymptotic_interp, zeroth_eigs, ref_angle, ctrl)

            # Store final asymptotic coefficients if integration reached theta_max
            if isfinite(ca1) && isfinite(ca2)
                locstab_fs[ipsi, 4] = ca1
                locstab_fs[ipsi, 5] = ca2
            end
        end
    end

    if ctrl.verbose
        println("Ballooning analysis complete.")
    end

end
