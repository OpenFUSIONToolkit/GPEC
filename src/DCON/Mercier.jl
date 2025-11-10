"""
    mercier_scan!(locstab_fs::Array{Float64,5}, plasma_eq::PlasmaEquilibrium)

Evaluates Mercier criterion for local stability and modifies results in place
within the local stability array. Performs the same function as `mercier_scan`
in the Fortran code.
"""
function mercier_scan!(locstab_fs::Matrix{Float64}, plasma_eq::Equilibrium.PlasmaEquilibrium)

    # Shorthand
    sq = plasma_eq.sq
    rzphi = plasma_eq.rzphi

    # Allocate splines
    ff_fs = zeros(length(rzphi.ys), 5)

    # Compute surface quantities
    for ipsi in 1:length(sq.xs)
        twopif = sq.fs[ipsi, 1]
        p1 = sq.fs1[ipsi, 2]
        v1 = sq.fs[ipsi, 3]
        v2 = sq.fs1[ipsi, 3]
        q = sq.fs[ipsi, 4]
        q1 = sq.fs1[ipsi, 4]
        chi1 = 2π * plasma_eq.psio

        # Evaluate coordinates and jacobian
        for itheta in 1:length(rzphi.ys)
            theta = rzphi.ys[itheta]

            # Evaluate bicubic spline at grid point
            f, _, fy = Spl.bicube_deriv1!(rzphi, rzphi.xs[ipsi], theta)

            rfac = sqrt(f[1])
            eta = 2π * (theta + f[2])
            r = plasma_eq.ro + rfac * cos(eta)
            jac = f[4]  # Jacobian

            # Evaluate other local quantities
            v21 = fy[1] / (2.0 * rfac * jac)
            v22 = (1.0 + fy[2]) * 2π * rfac / jac
            v23 = fy[3] * r / jac
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

        ff = Spl.CubicSpline(Vector(rzphi.ys), ff_fs; bctype=2) # bctype=2 is periodic

        # Integrate quantities with respect to theta
        Spl.spline_integrate!(ff)
        avg = ff.fsi[end, :]

        # Evaluate Mercier criterion and related quantities
        term = twopif * p1 * v1 / (q1 * chi1^3) * avg[2]
        di = -0.25 + term * (1 - term) +
             p1 * (v1 / (q1 * chi1^2))^2 * avg[1] *
             (p1 * (avg[3] + (twopif / chi1)^2 * avg[4]) - v2 / v1)
        h = twopif * p1 * v1 / (q1 * chi1^3) * (avg[2] - avg[1] / avg[5])

        # Store results in output spline structure
        locstab_fs[ipsi, 1] = di * sq.xs[ipsi]
        locstab_fs[ipsi, 2] = (di + (h - 0.5)^2) * sq.xs[ipsi]
        locstab_fs[ipsi, 3] = h
    end
end
