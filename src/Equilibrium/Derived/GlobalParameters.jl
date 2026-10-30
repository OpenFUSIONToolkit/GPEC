"""
    equilibrium_global_parameters!(pe::PlasmaEquilibrium)

Computes and populates global equilibrium parameters in the `PlasmaEquilibrium`
struct, such as rmean, amean, kappa, bt0, crnt, betat, betan, li1, etc. Performs
the same function as equil_out_global in the Fortran code.
"""
function equilibrium_global_parameters!(pe::PlasmaEquilibrium)
    profiles = pe.profiles
    mpsi = length(pe.rzphi_xs) - 1
    mtheta = length(pe.rzphi_ys) - 1

    # Use separatrix geometry
    rsep, zsep, rext, _ = equilibrium_separatrix_find!(pe)

    rmean = (rsep[1] + rsep[2]) / 2
    amean = (rsep[1] - rsep[2]) / 2
    aratio = rmean / amean
    kappa = (zsep[1] - zsep[2]) / (rsep[1] - rsep[2])
    delta1 = (rmean - rext[1]) / amean
    delta2 = (rmean - rext[2]) / amean
    dpsi = 1.0 - pe.rzphi_xs[mpsi+1]
    psi_edge = profiles.xs[end]
    bt0 = (profiles.F_spline.y[end] + profiles.F_deriv(psi_edge; hint=Ref(profiles.npts_minus_1)) * dpsi) / (2π * rmean)

    pe.params.rmean = rmean
    pe.params.amean = amean
    pe.params.aratio = aratio
    pe.params.kappa = kappa
    pe.params.delta1 = delta1
    pe.params.delta2 = delta2
    pe.params.bt0 = bt0

    psio = pe.psio
    gs1 = zeros(Float64, mtheta + 1)
    gs2 = zeros(Float64, mtheta + 1)

    # Direct array access at edge flux surface grid points using nodal_derivs
    for itheta in 0:mtheta
        # Function values (partials[1,:,:])
        r2 = pe.rzphi_rsquared.nodal_derivs.partials[1, end, itheta+1]
        offset = pe.rzphi_offset.nodal_derivs.partials[1, end, itheta+1]
        jac = pe.rzphi_jac.nodal_derivs.partials[1, end, itheta+1]
        # Theta derivatives (partials[3,:,:] = ∂f/∂y)
        r2_y = pe.rzphi_rsquared.nodal_derivs.partials[3, end, itheta+1]
        offset_y = pe.rzphi_offset.nodal_derivs.partials[3, end, itheta+1]

        chi1 = 2π * psio / jac
        jacfac = π / jac
        rfac = sqrt(r2)
        eta = 2π * (pe.rzphi_ys[itheta+1] + offset)
        r = pe.ro + rfac * cos(eta)
        v21 = jacfac * r2_y / (2π * rfac)
        v22 = jacfac * (1 + offset_y) * (2 * rfac)
        v33 = jacfac * 2π * (r / π)
        dvsq = (v21^2 + v22^2) * (v33 * jac^2)^2
        gs1[itheta+1] = sqrt(dvsq) / (2π * r)
        gs2[itheta+1] = chi1 * dvsq / (2π * r)^2
    end

    int1 = sum(gs1) / (mtheta + 1)
    int2 = sum(gs2) / (mtheta + 1)
    crnt = int2 / (1e6 * mu0)
    bp0 = int2 / int1
    bwall = 1e6 * mu0 * crnt / (2π * amean)

    pe.params.crnt = crnt
    pe.params.bwall = bwall

    # Flux-surface integrals over the full normalized flux [0, 1]; ExtendExtrap carries the integral past the first/last grid points (integrands are smooth/vanishing at the axis).
    xs = profiles.xs
    fsi(y) = FastInterpolations.integrate(cubic_interp(xs, y; extrap=ExtendExtrap()), 0.0, 1.0)
    P_vals = profiles.P_spline.y
    dVdpsi_vals = profiles.dVdpsi_spline.y

    fsi_pdv = fsi(P_vals .* dVdpsi_vals)         # ∫ p  dV/dψ
    fsi_dv = fsi(dVdpsi_vals)                   # ∫ dV/dψ
    fsi_p2dv = fsi(P_vals .^ 2 .* dVdpsi_vals)    # ∫ p² dV/dψ
    volume = fsi_dv                             # same integrand as hs col 2 in Fortran

    # Poloidal-field surface integral hs_bp2(ψ) = ψ₀² ∮dθ |∇ψ|² / (R² J).
    # This is Fortran equil_out.f's hs%fs(:,3) and is the correct integrand for
    # the internal inductance li1/li2/li3 (distinct from the p²·dV integrand
    # that drives betaj).
    hs_bp2 = zeros(Float64, mpsi + 1)
    for ipsi in 0:mpsi
        acc = 0.0
        for itheta in 0:mtheta
            r2 = pe.rzphi_rsquared.nodal_derivs.partials[1, ipsi+1, itheta+1]
            offset = pe.rzphi_offset.nodal_derivs.partials[1, ipsi+1, itheta+1]
            jac = pe.rzphi_jac.nodal_derivs.partials[1, ipsi+1, itheta+1]
            r2_y = pe.rzphi_rsquared.nodal_derivs.partials[3, ipsi+1, itheta+1]
            offset_y = pe.rzphi_offset.nodal_derivs.partials[3, ipsi+1, itheta+1]

            jacfac = π / jac
            rfac = sqrt(r2)
            eta = 2π * (pe.rzphi_ys[itheta+1] + offset)
            r = pe.ro + rfac * cos(eta)
            v21 = jacfac * r2_y / (2π * rfac)
            v22 = jacfac * (1 + offset_y) * (2 * rfac)
            v33 = jacfac * 2π * (r / π)
            dvsq = (v21^2 + v22^2) * (v33 * jac^2)^2
            acc += dvsq / (r^2) / jac
        end
        # Periodic trapezoidal rule on uniform θ grid reduces to a plain mean
        # because the first and last grid points coincide — matches the int1/int2
        # pattern used for the edge-surface integrals above.
        hs_bp2[ipsi+1] = (acc / (mtheta + 1)) * psio^2
    end
    fsi_bp2 = fsi(hs_bp2)

    p0 = P_vals[1] - profiles.P_deriv(profiles.xs[1]; hint=Ref(1)) * profiles.xs[1]  # linear extrapolation
    betat = 2 * (fsi_pdv / fsi_dv) / bt0^2
    betaj = 2 * sqrt(fsi_p2dv / fsi_dv) / bwall^2
    betan = 100 * amean * bt0 * betat / crnt
    betap1 = 2 * (fsi_pdv / fsi_dv) / bp0^2
    betap2 = 4 * fsi_pdv / ((1e6 * mu0 * crnt)^2 * pe.ro)
    betap3 = 4 * fsi_pdv / ((1e6 * mu0 * crnt)^2 * rmean)
    li1 = fsi_bp2 / fsi_dv / bp0^2
    li2 = 2 * fsi_bp2 / ((1e6 * mu0 * crnt)^2 * pe.ro)
    li3 = 2 * fsi_bp2 / ((1e6 * mu0 * crnt)^2 * rmean)

    pe.params.psi0 = psio
    pe.params.psi_axis = pe.psio
    pe.params.psi_boundary = 1.0
    pe.params.psi_boundary_norm = 1.0
    pe.params.psi_norm = 0.0
    pe.params.psi_axis_offset = 0.0
    pe.params.psi_boundary_offset = 0.0
    pe.params.psi_axis_sign = 1
    pe.params.psi_boundary_sign = -1
    pe.params.psi_boundary_zero = false

    pe.params.q0 = profiles.q_spline.y[1]
    pe.params.b0 = bt0

    pe.params.volume = volume
    pe.params.betat = betat
    pe.params.betan = betan
    pe.params.betaj = betaj
    pe.params.betap1 = betap1
    pe.params.betap2 = betap2
    pe.params.betap3 = betap3
    pe.params.li1 = li1
    pe.params.li2 = li2
    pe.params.li3 = li3
end

"""
    equilibrium_qfind!(equil::PlasmaEquilibrium)

Finds the extrema of the safety factor profile q(ψ) in the plasma equilibrium
and computes derived q-values such as q0, qmin, qmax, qa, and q95. Performs
the same function as equil_out_qfind in the Fortran code.
"""
function equilibrium_qfind!(equil::PlasmaEquilibrium)

    profiles = equil.profiles
    xs = profiles.xs
    mpsi = length(xs) - 1
    psiexl = Float64[]
    qexl = Float64[]

    # Create derivative views for q-spline
    q_spline = profiles.q_spline
    q_d1 = deriv1(q_spline)
    q_d2 = deriv2(q_spline)
    q_d3 = deriv3(q_spline)

    # Left endpoint
    push!(psiexl, xs[1])
    push!(qexl, q_spline.y[1])

    # Search for extrema in q(ψ)
    for ipsi in 1:mpsi
        x0 = xs[ipsi]
        x1 = xs[ipsi+1]
        xmax = x1 - x0

        a = q_spline(x0)
        b = q_d1(x0)
        c = q_d2(x0)
        d = q_d3(x0)

        if d != 0.0
            xcrit = -c / d
            dx2 = xcrit^2 - 2b / d
            if dx2 ≥ 0
                dx = sqrt(dx2)
                for delta in (dx, -dx)
                    x = xcrit - delta
                    if 0 ≤ x < xmax
                        ψ = x0 + x
                        push!(psiexl, ψ)
                        push!(qexl, q_spline(ψ))
                    end
                end
            end
        end
    end

    # Right endpoint
    push!(psiexl, xs[end])
    push!(qexl, q_spline.y[end])

    equil.params.qextrema_psi = psiexl
    equil.params.qextrema_q = qexl
    equil.params.mextrema = length(psiexl)
    # Compute derived q-values
    q0 = q_spline.y[1] - profiles.q_deriv(xs[1]; hint=Ref(1)) * xs[1]
    qmax_edge = q_spline.y[end]
    qmin = min(minimum(qexl), q0)
    qmax = max(maximum(qexl), qmax_edge)
    qa = q_spline.y[end] + profiles.q_deriv(xs[end]; hint=Ref(profiles.npts_minus_1)) * (1.0 - xs[end])

    q95 = q_spline(0.95)

    # Store derived values
    equil.params.q0 = q0
    equil.params.qmin = qmin
    equil.params.qmax = qmax
    equil.params.qa = qa
    equil.params.q95 = q95
end
