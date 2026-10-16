"""
    flux_surface_metric(equil, psi, theta; hint=(Ref(1), Ref(1))) -> (; r, jac, delpsi)

Evaluate the flux-surface geometry at `(psi, theta)`: major radius `r`, Jacobian `jac`, and the
flux-gradient magnitude `delpsi = |∇ψ|`. The metric components `w11`/`w12` follow GPEC's gpout.f
convention for the rzphi straight-field-line coordinates. This is the single source of truth for the
`|∇ψ|` computation reused across the FFS and PE modules.
"""
function flux_surface_metric(equil::PlasmaEquilibrium, psi::Float64, theta::Float64; hint=(Ref(1), Ref(1)))
    r2 = equil.rzphi_rsquared((psi, theta); hint=hint)
    jac = equil.rzphi_jac((psi, theta); hint=hint)
    deta = equil.rzphi_offset((psi, theta); hint=hint)
    r2_y = equil.rzphi_rsquared((psi, theta); deriv=DerivOp(0, 1), hint=hint)
    deta_y = equil.rzphi_offset((psi, theta); deriv=DerivOp(0, 1), hint=hint)

    rfac = sqrt(abs(r2))
    eta = 2π * (theta + deta)
    r = equil.ro + rfac * cos(eta)

    w11, w12 = _grad_psi_poloidal(rfac, r, jac, r2_y, deta_y)
    delpsi = sqrt(w11^2 + w12^2)

    return (; r, jac, delpsi)
end

# Poloidal-plane components (w11, w12) of ∇ψ in the rzphi straight-field-line coordinates (GPEC gpout.f).
@inline _grad_psi_poloidal(rfac, r, jac, r2_y, deta_y) = ((1.0 + deta_y) * (2π)^2 * rfac * r / jac, -r2_y * π * r / (rfac * jac))

"""
    flux_surface_basis(equil, psi, theta; hint=(Ref(1), Ref(1))) -> NamedTuple
    flux_surface_basis(equil, ipsi, itheta) -> NamedTuple

Full straight-field-line geometry at one point: `r`, `z`, Jacobian `jac` and its ψ-derivative
`jac_psi`, `delpsi2` = |∇ψ|², `dpdt` = ∇ψ·∇θ, `dpdz` = ∇ψ·∇ζ, and the covariant metric `g11`,
`g12`, `g13`, `g22`, `g23`, `g33` of the tangent basis e_i = ∂x/∂q^i. The gradient and tangent bases
are the `w` and `v` matrices of the Fortran `gpeq_c`; |∇ψ| and `jac` agree with
`flux_surface_metric`. Derivatives are with respect to ψ_N. The `(ipsi, itheta)` method reads the
values and first derivatives stored at equilibrium grid point `(rzphi_xs[ipsi], rzphi_ys[itheta])`,
as `make_metric` does.
"""
function flux_surface_basis(equil::PlasmaEquilibrium, psi::Float64, theta::Float64; hint=(Ref(1), Ref(1)))
    pt = (psi, theta)
    dx = DerivOp(1, 0)
    dy = DerivOp(0, 1)
    return _flux_surface_basis(equil, theta,
        equil.rzphi_rsquared(pt; hint=hint), equil.rzphi_rsquared(pt; deriv=dx, hint=hint), equil.rzphi_rsquared(pt; deriv=dy, hint=hint),
        equil.rzphi_offset(pt; hint=hint), equil.rzphi_offset(pt; deriv=dx, hint=hint), equil.rzphi_offset(pt; deriv=dy, hint=hint),
        equil.rzphi_nu(pt; deriv=dx, hint=hint), equil.rzphi_nu(pt; deriv=dy, hint=hint),
        equil.rzphi_jac(pt; hint=hint), equil.rzphi_jac(pt; deriv=dx, hint=hint))
end

function flux_surface_basis(equil::PlasmaEquilibrium, ipsi::Int, itheta::Int)
    r2 = equil.rzphi_rsquared.nodal_derivs.partials
    deta = equil.rzphi_offset.nodal_derivs.partials
    nu = equil.rzphi_nu.nodal_derivs.partials
    jac = equil.rzphi_jac.nodal_derivs.partials
    return _flux_surface_basis(equil, equil.rzphi_ys[itheta],
        r2[1, ipsi, itheta], r2[2, ipsi, itheta], r2[3, ipsi, itheta],
        deta[1, ipsi, itheta], deta[2, ipsi, itheta], deta[3, ipsi, itheta],
        nu[2, ipsi, itheta], nu[3, ipsi, itheta],
        jac[1, ipsi, itheta], jac[2, ipsi, itheta])
end

function _flux_surface_basis(equil::PlasmaEquilibrium, theta, r2, r2_x, r2_y, deta, deta_x, deta_y, nu_x, nu_y, jac, jac_psi)
    rfac = sqrt(abs(r2))
    eta = 2π * (theta + deta)
    r = equil.ro + rfac * cos(eta)
    z = equil.zo + rfac * sin(eta)

    # Gradient basis in the poloidal plane: rows ∇ψ, ∇θ, ∇ζ (Fortran gpeq_c w-matrix).
    w11, w12 = _grad_psi_poloidal(rfac, r, jac, r2_y, deta_y)
    w21 = -(2π)^2 * rfac * r * deta_x / jac
    w22 = π * r * r2_x / (rfac * jac)
    w31 = (2π * r * rfac / jac) * (deta_x * nu_y - nu_x * (1.0 + deta_y))
    w32 = (r / (2.0 * rfac * jac)) * (nu_x * r2_y - r2_x * nu_y)

    # Tangent basis e_ψ, e_θ, e_ζ without the Jacobian factor (Fortran gpeq_c v-matrix).
    v11 = r2_x / (2.0 * rfac)
    v12 = deta_x * 2π * rfac
    v13 = nu_x * r
    v21 = r2_y / (2.0 * rfac)
    v22 = (1.0 + deta_y) * 2π * rfac
    v23 = nu_y * r
    v33 = 2π * r

    return (; r, z, jac, jac_psi, delpsi2=w11^2 + w12^2, dpdt=w11 * w21 + w12 * w22, dpdz=w11 * w31 + w12 * w32,
        g11=v11^2 + v12^2 + v13^2, g12=v11 * v21 + v12 * v22 + v13 * v23, g13=v13 * v33,
        g22=v21^2 + v22^2 + v23^2, g23=v23 * v33, g33=v33^2)
end

"""
    metric_bsq_interpolant(equil) -> CubicInterpolantND

Bicubic interpolant of B² = (χ'/J)² (g_θθ + 2q g_θζ + q² g_ζζ) on the equilibrium (ψ,θ) grid, the
field strength of the straight-field-line metric that the Euler-Lagrange matrices and the perturbed
field are built from. The equilibrium's `eqfun_B` (from ψ₀|∇ψ| and F) agrees with it only to the
accuracy of the coordinate transformation.
"""
function metric_bsq_interpolant(equil::PlasmaEquilibrium)
    xs = equil.rzphi_xs
    ys = equil.rzphi_ys
    chi1 = 2π * equil.psio
    vals = zeros(length(xs), length(ys))
    for i in eachindex(xs)
        q = equil.profiles.q_spline(xs[i])
        for k in 1:(length(ys)-1)
            b = flux_surface_basis(equil, i, k)
            vals[i, k] = (chi1 / b.jac)^2 * (b.g22 + 2.0 * q * b.g23 + q^2 * b.g33)
        end
        vals[i, end] = vals[i, 1]
    end
    return cubic_interp((xs, ys), vals; bc=(CubicFit(), PeriodicBC()), extrap=(ExtendExtrap(), WrapExtrap()))
end

"""
    flux_surface_area(equil, psi, mtheta) -> Float64

Compute ∫ J·|∇ψ| dθ over the flux surface at `psi` using `mtheta` equally-spaced θ points and the
trapezoidal rule with periodic end correction.
"""
function flux_surface_area(equil::PlasmaEquilibrium, psi::Float64, mtheta::Int)
    area = 0.0
    last_contrib = 0.0

    hint = (Ref(1), Ref(1))  # Shared 2D hint for hot loop optimization
    for itheta in 0:mtheta
        theta = itheta / mtheta
        m = flux_surface_metric(equil, psi, theta; hint=hint)
        contrib = m.jac * m.delpsi / mtheta
        area += contrib

        if itheta == mtheta
            last_contrib = contrib
        end
    end

    # Trapezoidal rule end correction
    return area - last_contrib
end
