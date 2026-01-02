"""
    mercier_scan!(locstab_fs::Array{Float64,5}, plasma_eq::PlasmaEquilibrium)

Evaluates Mercier criterion for local stability and modifies results in place
within the local stability array. Performs the same function as `mercier_scan`
in the Fortran code.
"""
function mercier_scan!(locstab_fs::Matrix{Float64}, plasma_eq::Equilibrium.PlasmaEquilibrium)

    # Allocate splines
    ff_fs = zeros(length(plasma_eq.theta_grid), 5)

    # Compute surface quantities
    for ipsi in 1:length(plasma_eq.psi_grid)
        psi = plasma_eq.psi_grid[ipsi]
        twopif = plasma_eq.F_values[ipsi]
        p1 = ForwardDiff.derivative(plasma_eq.P_spline, psi)
        v1 = plasma_eq.dVdpsi_values[ipsi]
        v2 = ForwardDiff.derivative(plasma_eq.dVdpsi_spline, psi)
        q = plasma_eq.q_values[ipsi]
        q1 = ForwardDiff.derivative(plasma_eq.q_spline, psi)
        chi1 = 2π * plasma_eq.psio

        # Evaluate coordinates and jacobian
        for itheta in 1:length(plasma_eq.theta_grid)
            theta = plasma_eq.theta_grid[itheta]

            # Evaluate spline values
            r2_val = plasma_eq.r2_spline(psi, theta)
            eta_val = plasma_eq.eta_spline(psi, theta)
            nu_val = plasma_eq.nu_spline(psi, theta)
            jac = plasma_eq.jac_spline(psi, theta)

            # Compute derivatives using ForwardDiff
            r2_dtheta = ForwardDiff.derivative(t -> plasma_eq.r2_spline(psi, t), theta)
            eta_dtheta = ForwardDiff.derivative(t -> plasma_eq.eta_spline(psi, t), theta)
            nu_dtheta = ForwardDiff.derivative(t -> plasma_eq.nu_spline(psi, t), theta)

            rfac = sqrt(r2_val)
            eta = 2π * (theta + eta_val)
            r = plasma_eq.ro + rfac * cos(eta)

            # Evaluate other local quantities
            v21 = r2_dtheta / (2.0 * rfac * jac)
            v22 = (1.0 + eta_dtheta) * 2π * rfac / jac
            v23 = nu_dtheta * r / jac
            v33 = 2π * r / jac
            bsq = chi1^2 * (v21^2 + v22^2 + (v23 + q * v33)^2)
            dpsisq = (2π * r)^2 * (v21^2 + v22^2)

            # Evaluate integrands
            ff_fs[itheta, 1] = bsq / dpsisq
            ff_fs[itheta, 2] = 1.0 / dpsisq
            ff_fs[itheta, 3] = 1.0 / bsq
            ff_fs[itheta, 4] = 1.0 / (bsq * dpsisq)
            ff_fs[itheta, 5] = bsq

            # Weight by jacobian and volume element
            @views ff_fs[itheta, :] .*= jac / v1
        end

        # Integrate quantities with respect to theta using trapezoidal rule
        # For periodic data, we use: integral ≈ (theta_max - theta_min) * mean(data)
        # Since theta_grid spans [0, 1], this simplifies to: mean(data)
        avg = vec(sum(ff_fs, dims=1) / size(ff_fs, 1))

        # Evaluate Mercier criterion and related quantities
        term = twopif * p1 * v1 / (q1 * chi1^3) * avg[2]
        di = -0.25 + term * (1 - term) +
             p1 * (v1 / (q1 * chi1^2))^2 * avg[1] *
             (p1 * (avg[3] + (twopif / chi1)^2 * avg[4]) - v2 / v1)
        h = twopif * p1 * v1 / (q1 * chi1^3) * (avg[2] - avg[1] / avg[5])

        # Store results in output spline structure
        locstab_fs[ipsi, 1] = di * plasma_eq.psi_grid[ipsi]
        locstab_fs[ipsi, 2] = (di + (h - 0.5)^2) * plasma_eq.psi_grid[ipsi]
        locstab_fs[ipsi, 3] = h
    end
end
