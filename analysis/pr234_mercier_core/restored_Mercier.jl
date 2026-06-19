"""
    restored_mercier_scan!(locstab_fs, plasma_eq)

Restored copy of the old `src/ForceFreeStates/Mercier.jl` calculation from
`origin/develop`, kept here only for PR #234 diagnostics. It evaluates the
Mercier ideal interchange `D_I`, resistive interchange `D_R`, and `H` on every
profile surface and writes:

- column 1: `D_I * psi`
- column 2: `D_R * psi`
- column 3: `H`
"""
function restored_mercier_scan!(locstab_fs::Matrix{Float64}, plasma_eq)
    profiles = plasma_eq.profiles
    ff_fs = zeros(length(plasma_eq.rzphi_ys), 5)

    hint = Ref(1)
    for ipsi in 1:profiles.npts
        psi = profiles.xs[ipsi]
        twopif = profiles.F_spline.y[ipsi]
        p1 = profiles.P_deriv(psi; hint=hint)
        v1 = profiles.dVdpsi_spline.y[ipsi]
        v2 = profiles.dVdpsi_deriv(psi; hint=hint)
        q = profiles.q_spline.y[ipsi]
        q1 = profiles.q_deriv(psi; hint=hint)
        chi1 = 2π * plasma_eq.psio

        for itheta in 1:length(plasma_eq.rzphi_ys)
            theta = plasma_eq.rzphi_ys[itheta]

            f1 = plasma_eq.rzphi_rsquared.nodal_derivs.partials[1, ipsi, itheta]
            f2 = plasma_eq.rzphi_offset.nodal_derivs.partials[1, ipsi, itheta]
            jac = plasma_eq.rzphi_jac.nodal_derivs.partials[1, ipsi, itheta]
            fy1 = plasma_eq.rzphi_rsquared.nodal_derivs.partials[3, ipsi, itheta]
            fy2 = plasma_eq.rzphi_offset.nodal_derivs.partials[3, ipsi, itheta]
            fy3 = plasma_eq.rzphi_nu.nodal_derivs.partials[3, ipsi, itheta]

            rfac = sqrt(f1)
            eta = 2π * (theta + f2)
            r = plasma_eq.ro + rfac * cos(eta)

            v21 = fy1 / (2.0 * rfac * jac)
            v22 = (1.0 + fy2) * 2π * rfac / jac
            v23 = fy3 * r / jac
            v33 = 2π * r / jac
            bsq = chi1^2 * (v21^2 + v22^2 + (v23 + q * v33)^2)
            dpsisq = (2π * r)^2 * (v21^2 + v22^2)

            ff_fs[itheta, 1] = bsq / dpsisq
            ff_fs[itheta, 2] = 1.0 / dpsisq
            ff_fs[itheta, 3] = 1.0 / bsq
            ff_fs[itheta, 4] = 1.0 / (bsq * dpsisq)
            ff_fs[itheta, 5] = bsq
            @views ff_fs[itheta, :] .*= jac / v1
        end

        itp = cubic_interp(plasma_eq.rzphi_ys, Series(ff_fs); bc=PeriodicBC())
        avg = FastInterpolations.integrate(itp)

        term = twopif * p1 * v1 / (q1 * chi1^3) * avg[2]
        di = -0.25 + term * (1 - term) +
             p1 * (v1 / (q1 * chi1^2))^2 * avg[1] *
             (p1 * (avg[3] + (twopif / chi1)^2 * avg[4]) - v2 / v1)
        h = twopif * p1 * v1 / (q1 * chi1^3) * (avg[2] - avg[1] / avg[5])

        locstab_fs[ipsi, 1] = di * profiles.xs[ipsi]
        locstab_fs[ipsi, 2] = (di + (h - 0.5)^2) * profiles.xs[ipsi]
        locstab_fs[ipsi, 3] = h
    end
end
