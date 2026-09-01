# ResistEval.jl
#
# Per-singular-surface Glasser-Greene-Johnson geometric coefficients (E, F,
# G, H, K, M) and the two flux-surface averages (⟨B²/|∇ψ|²⟩, ⟨B²⟩) that
# downstream callers need to turn geometry into τ_A / τ_R with kinetic
# profiles.
#
# Port of Fortran RDCON `resist_eval` (geometric part only).
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
#
# A seventh integrand, B, is added (beyond the Fortran set) so that ⟨B⟩ is
# available for the Lin-Liu & Miller 1995 trapped-fraction formula used by
# the shared NeoclassicalResistivity closure. B_max, B_min, and the flux-
# surface-averaged major radius R_major are accumulated alongside by
# running extrema over the θ-loop.

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
| `avg_B`     | ⟨B⟩ — needed for Lin-Liu-Miller f_t                  |
| `B_max`, `B_min` | θ-extrema of B on the surface [T]               |
| `f_trap`    | Lin-Liu & Miller 1995 trapped-particle fraction      |
| `R_major`   | flux-surface-averaged major radius ⟨R⟩ [m]           |
| `eps_local` | (R_max − R_min)/2 / R_major — local inverse aspect ratio |
| `p_local`   | Plasma pressure at this surface [Pa]                 |
| `p1_local`  | dp/dψ at this surface                                |
| `v1_local`  | dV/dψ at this surface                                |

`H` here is identical to the `H` reported by `mercier_scan!` and stored
in `LocalStability/h` — the GGJ routine recomputes it for convenience.

`avg_B`, `B_max`, `B_min`, `f_trap`, `R_major`, and `eps_local` are used
by `NeoclassicalResistivity.eta_neoclassical` to form the Sauter/Redl
F_33 correction to Spitzer resistivity. See Sauter, Angioni & Lin-Liu
1999, Phys. Plasmas 6, 2834 and Lin-Liu & Miller 1995, Phys. Plasmas 2,
1666.
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
    avg_B::Float64
    B_max::Float64
    B_min::Float64
    f_trap::Float64
    R_major::Float64
    eps_local::Float64
    p_local::Float64
    p1_local::Float64
    v1_local::Float64
end

"""
    resist_geometry(equil, psifac, q1; gamma=5/3) -> ResistGeometry

Port of Fortran RDCON `resist_eval` restricted to the
pure-equilibrium geometric coefficients. Integrates the 6 theta integrands
at the given flux surface and combines them into E, F, G, H, K, M via the
standard GGJ formulas.

# Arguments

  - `equil::PlasmaEquilibrium` — the fully-solved equilibrium
  - `psifac` — normalized flux coordinate of the singular surface
  - `q1`     — dq/dψ at this surface (from `SingType.q1`)

# Keyword arguments

  - `gamma`  — adiabatic index (default 5/3)

!!! note "Contract"
    `psifac` must be a genuine interior rational surface (`0 < ψ < 1`) with
    nonzero `q1`, `p1 = dp/dψ`, and `p`. The GGJ combination divides by these
    and by `|∇ψ|²` (which → 0 at the axis), so calling on the magnetic axis,
    a flat-pressure surface, or a zero-shear surface yields `Inf`/`NaN`. This
    matches the Fortran `resist_eval`, which is only ever invoked on interior
    rationals.
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

    # Build the 6 GGJ θ-integrands plus a 7th (B) for the neoclassical
    # resistivity f_t calculation, and accumulate running extrema of
    # (B, R) for Lin-Liu-Miller f_t and the local ε.
    ntheta = length(equil.rzphi_ys)
    ff     = zeros(Float64, ntheta, 7)
    B_max  = -Inf
    B_min  =  Inf
    R_max  = -Inf
    R_min  =  Inf
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

        B_here = sqrt(bsq)
        B_max = max(B_max, B_here)
        B_min = min(B_min, B_here)
        R_max = max(R_max, r)
        R_min = min(R_min, r)

        ff[itheta, 1] = bsq / dpsisq
        ff[itheta, 2] = 1.0 / dpsisq
        ff[itheta, 3] = 1.0 / bsq
        ff[itheta, 4] = 1.0 / (bsq * dpsisq)
        ff[itheta, 5] = bsq
        ff[itheta, 6] = dpsisq / bsq
        ff[itheta, 7] = B_here
        @views ff[itheta, :] .*= jac / v1
    end
    # Snap the repeated endpoint exactly equal to the start (see Equilibrium.jl's
    # GS-residual integrator and issue #240 -- independent spline evaluations at
    # theta=0/1 can differ by machine epsilon and trip PeriodicBC's check)
    @views ff[end, :] .= ff[1, :]

    # Integrate each column around θ using the same periodic cubic-spline
    # integrator Mercier.jl uses
    itp = cubic_interp(equil.rzphi_ys, Series(ff); bc=PeriodicBC())
    avg = FastInterpolations.integrate(itp)
    avg_B = avg[7]
    R_major = 0.5 * (R_max + R_min)
    eps_local = R_major > 0 ? 0.5 * (R_max - R_min) / R_major : 0.0
    f_trap = Utilities.NeoclassicalResistivity.trapped_fraction(avg_B, avg[5], B_min, B_max)

    # GGJ coefficients (Fortran RDCON resist_eval)
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
        avg[1], avg[5],
        avg_B, B_max, B_min, f_trap, R_major, eps_local,
        p, p1, v1,
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
