# ResistEval.jl
#
# Per-singular-surface Glasser-Greene-Johnson geometric coefficients (E, F,
# G, H, K, M) and the two flux-surface averages (⟨B²/|∇ψ|²⟩, ⟨B²⟩) that
# downstream callers need to turn geometry into τ_A / τ_R with kinetic
# profiles.
#
# Port of Fortran `rdcon/resist.f::resist_eval` (geometric part only).
# Unlike the Fortran, this routine produces *only* the pure-equilibrium
# quantities; kinetic timescales (τ_A, τ_R) are built on top in the
# downstream `build_ggj_inputs` helper using the same KineticProfiles that
# feed SLAYER, rather than Fortran's hardcoded `ne=1e14, te=3e3`
# parameter defaults.
#
# The 6 theta-integrands match the Fortran layout:
#   1: B² / |∇ψ|²
#   2: 1 / |∇ψ|²
#   3: 1 / B²
#   4: 1 / (B² · |∇ψ|²)
#   5: B²
#   6: |∇ψ|² / B²
# All weighted by `jac / v1` (jacobian / dV/dψ) before integration.

"""
    ResistGeometry

Per-singular-surface Glasser-Greene-Johnson geometric coefficients and
supporting flux-surface averages.

| field       | meaning                                              |
|-------------|------------------------------------------------------|
| `E`, `F`    | Glasser interchange parameters (enter `D_I = E+F+H-¼`) |
| `G`         | Coupling coefficient (curvature × pressure gradient) |
| `H`         | Pfirsch-Schlüter coefficient                         |
| `K`         | Glasser parameter                                    |
| `M`         | Mass factor                                          |
| `avg_bsq_over_dpsisq` | ⟨B²/|∇ψ|²⟩ — needed for τ_R         |
| `avg_bsq`   | ⟨B²⟩ — needed for τ_R                                |
| `p_local`   | Plasma pressure at this surface [Pa]                 |
| `p1_local`  | dp/dψ at this surface                                |
| `v1_local`  | dV/dψ at this surface                                |

`H` here is identical to the `H` reported by `mercier_scan!` and stored
in `locstab/h` — the GGJ routine recomputes it for convenience.
"""
struct ResistGeometry
    E::Float64
    F::Float64
    G::Float64
    H::Float64
    K::Float64
    M::Float64
    avg_bsq_over_dpsisq::Float64
    avg_bsq::Float64
    p_local::Float64
    p1_local::Float64
    v1_local::Float64
end

"""
    resist_geometry(equil, psifac, q1; gamma=5/3) -> ResistGeometry

Port of Fortran `rdcon/resist.f::resist_eval` restricted to the
pure-equilibrium geometric coefficients. Integrates the 6 theta integrands
at the given flux surface and combines them into E, F, G, H, K, M via the
standard GGJ formulas.

# Arguments

  - `equil::PlasmaEquilibrium` — the fully-solved equilibrium
  - `psifac` — normalized flux coordinate of the singular surface
  - `q1`     — dq/dψ at this surface (from `SingType.q1`)

# Keyword arguments

  - `gamma`  — adiabatic index (default 5/3)
"""
function resist_geometry(equil::Equilibrium.PlasmaEquilibrium,
                          psifac::Real, q1::Real; gamma::Real=5/3)
    profiles = equil.profiles
    twopi    = 2π
    chi1     = twopi * equil.psio
    psi_f    = Float64(psifac)

    # Surface-profile quantities (evaluate via the existing splines)
    twopif = profiles.F_spline(psi_f)
    p      = profiles.P_spline(psi_f)
    p1     = profiles.P_deriv(psi_f)
    v1     = profiles.dVdpsi_spline(psi_f)
    v2     = profiles.dVdpsi_deriv(psi_f)
    q      = profiles.q_spline(psi_f)

    # Build the 6 theta-integrands by evaluating rzphi-derived metric
    # terms at every poloidal grid point, then integrate around θ.
    ntheta = length(equil.rzphi_ys)
    ff     = zeros(Float64, ntheta, 6)
    for itheta in 1:ntheta
        theta = equil.rzphi_ys[itheta]
        f1  = equil.rzphi_rsquared((psi_f, theta))
        f2  = equil.rzphi_offset((psi_f, theta))
        jac = equil.rzphi_jac((psi_f, theta))
        fy1 = FastInterpolations.deriv_view(equil.rzphi_rsquared, (0, 1))((psi_f, theta))
        fy2 = FastInterpolations.deriv_view(equil.rzphi_offset,   (0, 1))((psi_f, theta))
        fy3 = FastInterpolations.deriv_view(equil.rzphi_nu,       (0, 1))((psi_f, theta))

        rfac = sqrt(f1)
        eta  = twopi * (theta + f2)
        r    = equil.ro + rfac * cos(eta)

        v21 = fy1 / (2 * rfac * jac)
        v22 = (1 + fy2) * twopi * rfac / jac
        v23 = fy3 * r / jac
        v33 = twopi * r / jac
        bsq    = chi1^2 * (v21^2 + v22^2 + (v23 + q*v33)^2)
        dpsisq = (twopi * r)^2 * (v21^2 + v22^2)

        ff[itheta, 1] = bsq / dpsisq
        ff[itheta, 2] = 1.0 / dpsisq
        ff[itheta, 3] = 1.0 / bsq
        ff[itheta, 4] = 1.0 / (bsq * dpsisq)
        ff[itheta, 5] = bsq
        ff[itheta, 6] = dpsisq / bsq
        @views ff[itheta, :] .*= jac / v1
    end

    # Integrate each column around θ using the same periodic cubic-spline
    # integrator Mercier.jl uses
    itp = cubic_interp(equil.rzphi_ys, Series(ff); bc=PeriodicBC())
    avg = FastInterpolations.integrate(itp)

    # GGJ coefficients (resist.f:107-125)
    E_coef = p1 * v1 / (q1 * chi1^2)^2 * avg[1] *
             (twopif * q1 * chi1 / avg[5] - v2)
    F_coef = (p1 * v1 / (q1 * chi1^2))^2 *
             (avg[1] * avg[3] + (twopif / chi1)^2 *
              (avg[1] * avg[4] - avg[2]^2))
    H_coef = twopif * p1 * v1 / (q1 * chi1^3) * (avg[2] - avg[1] / avg[5])
    M_coef = avg[1] *
             (avg[6] + (twopif / chi1)^2 * (avg[3] - 1.0 / avg[5]))
    G_coef = avg[5] / (M_coef * gamma * p)
    K_coef = (q1 * chi1^2 / (p1 * v1))^2 *
             avg[5] / (M_coef * avg[1])

    return ResistGeometry(
        E_coef, F_coef, G_coef, H_coef, K_coef, M_coef,
        avg[1], avg[5], p, p1, v1,
    )
end

"""
    resist_eval_all!(intr::ForceFreeStatesInternal, equil; gamma=5/3)

Populate `sing.restype` for every `SingType` in `intr.sing` using
`resist_geometry`. No-op for surfaces whose `restype` has already been
filled.
"""
function resist_eval_all!(intr::ForceFreeStatesInternal,
                           equil::Equilibrium.PlasmaEquilibrium;
                           gamma::Real=5/3)
    for sing in intr.sing
        sing.restype === nothing || continue
        sing.restype = resist_geometry(equil, sing.psifac, sing.q1; gamma=gamma)
    end
    return intr
end
