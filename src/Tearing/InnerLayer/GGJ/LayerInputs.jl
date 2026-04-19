# LayerInputs.jl (GGJ)
#
# Build per-surface `GGJParameters` from a solved `PlasmaEquilibrium`, the
# `SingType` rational-surface list (each carrying a populated
# `restype::ResistGeometry` from `ForceFreeStates.resist_eval_all!`), and a
# `KineticProfiles` object — the same three ingredients `build_slayer_inputs`
# consumes. Produces the (E, F, G, H, K, τ_A, τ_R) tuple that GGJ's
# `solve_inner` needs, with τ_A / τ_R built from kinetic profiles using the
# same Spitzer resistivity and mass-density formulas SLAYER uses.
#
# Deliberately does *not* mirror the Fortran `rdcon/resist.f` hardcoded
# `ne = 1e14 cm⁻³, te = 3 keV` PARAMETER defaults. The kinetic content
# enters through `profiles` alone; this keeps GGJ and SLAYER using
# bit-identical plasma inputs when both are driven by the same
# `KineticProfiles`.

using ...Utilities: KineticProfiles
using ....Utilities.PhysicalConstants: MU_0, M_E, M_P, E_CHG, EPS_0
using ....ForceFreeStates: ResistGeometry

"""
    build_ggj_inputs(equil, sings, profiles; mu_i=2.0, zeff=1.0,
                      v1_scale=1.0) -> Vector{GGJParameters}

Construct a `GGJParameters` for each rational surface in `sings`. Each
surface's geometric coefficients (E, F, G, H, K, M) come from the
`sing.restype::ResistGeometry` populated by `resist_eval_all!`. Kinetic
timescales are derived from the `KineticProfiles` at `sing.psifac`:

```
ρ(ψ)   = μ_i · m_p · n_e(ψ)
ln Λ   = 24 + 3 ln 10 − ½ ln n_e + ln T_e
η(ψ)   = 1.65e-9 · ln Λ / (T_e / 1 keV)^(3/2)         [Ω·m, Spitzer]
τ_A    = √(ρ · M · μ_0) / |2π · n · q' · χ₁ / V'|     [Alfvén time]
τ_R    = (⟨B²/|∇ψ|²⟩ / ⟨B²⟩) · μ_0 / η                 [resistive diffusion]
```

The mode number `n` is taken from `sings[k].n[1]` (first resonant mode at
the surface). `χ₁ = 2π · psio`. The `v1_scale` kwarg is an optional
multiplicative factor on `V'` in the τ_A denominator — matches the
Fortran `sing%restype%v1 = v1 / volume` normalization option from
`rdcon/resist.f:144`; default `1.0` means use the raw `V'`.

Throws if any surface's `restype` is still `nothing` — call
`ForceFreeStates.resist_eval_all!(intr, equil)` first.
"""
function build_ggj_inputs(equil, sings, profiles::KineticProfiles;
                           mu_i::Real=2.0, zeff::Real=1.0,
                           v1_scale::Real=1.0)
    psio  = equil.psio
    chi1  = 2π * psio

    out = Vector{GGJParameters}(undef, length(sings))
    for (k, sing) in enumerate(sings)
        rg = sing.restype
        rg === nothing &&
            throw(ArgumentError("build_ggj_inputs: surface $k has " *
                                "restype = nothing. Call " *
                                "ForceFreeStates.resist_eval_all!(intr, equil) " *
                                "after sing_find! to populate it."))
        rg isa ResistGeometry ||
            throw(ArgumentError("build_ggj_inputs: surface $k has " *
                                "restype of unexpected type $(typeof(rg))."))

        # Kinetic profiles at this surface
        prof = profiles(sing.psifac)
        n_e  = prof.n_e          # [m⁻³]
        t_e  = prof.T_e          # [eV]

        # Mass density and Spitzer resistivity — same formulas as
        # slayer_parameters so SLAYER and GGJ see identical plasma inputs
        lnLamb = 24.0 + 3.0 * log(10.0) - 0.5 * log(n_e) + log(t_e)
        eta_sp = 1.65e-9 * lnLamb / (t_e / 1e3)^1.5
        rho    = mu_i * M_P * n_e

        # Alfvén time at the rational surface (resist.f:136-137)
        n_tor = Int(sing.n[1])
        v1    = rg.v1_local * v1_scale
        taua  = sqrt(rho * rg.M * MU_0) /
                abs(2π * n_tor * sing.q1 * chi1 / v1)

        # Resistive diffusion time (resist.f:138)
        taur  = (rg.avg_bsq_over_dpsisq / rg.avg_bsq) * MU_0 / eta_sp

        out[k] = GGJParameters(
            E=rg.E, F=rg.F, G=rg.G, H=rg.H, K=rg.K, M=rg.M,
            taua=taua, taur=taur, v1=1.0, ising=k,
        )
    end
    return out
end
