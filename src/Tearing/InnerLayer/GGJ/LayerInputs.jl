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
# Deliberately does *not* use any hardcoded `ne`/`te` defaults. The kinetic
# content enters through `profiles` alone; this keeps GGJ and SLAYER using
# bit-identical plasma inputs when both are driven by the same
# `KineticProfiles`.

using ...Utilities: KineticProfiles
using ....Utilities.PhysicalConstants: MU_0, M_E, M_P, E_CHG, EPS_0
using ....Utilities.NeoclassicalResistivity
using ....Utilities.NeoclassicalResistivity: NeoResistivityModel, SpitzerModel,
    SauterNeoModel, RedlNeoModel,
    coulomb_log_e, eta_spitzer, nu_star_e, eta_neoclassical
using ....ForceFreeStates: ResistGeometry

"""
    build_ggj_inputs(equil, sings, profiles; mu_i=2.0, zeff=1.0,
                      v1_scale=1.0,
                      resistivity_model::NeoResistivityModel=SpitzerModel(),
                      lnLambda_form::Symbol=:nrl) -> Vector{GGJParameters}

Construct a `GGJParameters` for each rational surface in `sings`. Each
surface's geometric coefficients (E, F, G, H, K, M) come from the
`sing.restype::ResistGeometry` populated by `resist_eval_all!`. Kinetic
timescales are derived from the `KineticProfiles` at `sing.psifac`:

```
ρ(ψ)   = μ_i · m_p · n_e(ψ)
η(ψ)   = eta_neoclassical(model, n_e, T_e, Z_eff, f_t, ν*_e)     [Ω·m]
τ_A    = √(ρ · M · μ_0) / |2π · n · q' · χ₁ / V'|                 [Alfvén time]
τ_R    = (⟨B²/|∇ψ|²⟩ / ⟨B²⟩) · μ_0 / η                             [resistive diffusion]
```

The mode number `n` is taken from `sings[k].n[1]` (first resonant mode at
the surface). `χ₁ = 2π · psio`. The `v1_scale` kwarg is an optional
multiplicative factor on `V'` in the τ_A denominator (a `v1 / volume`
normalization option); default `1.0` means use the raw `V'`.

# Resistivity model

`resistivity_model` selects the η closure:

  - `SpitzerModel()` (default) — Sauter 1999 Eq. 18a (Zeff-aware Spitzer),
    with the NRL Coulomb log.
  - `SauterNeoModel()` — multiplies by Sauter 1999 F_33 using f_t and ν*_e
    from the surface's `ResistGeometry`. Produces the physically-correct
    trapped-particle-corrected η for H-mode tearing stability.
  - `RedlNeoModel()` — Redl 2021 F_33 (improved high-ν* fit).

`lnLambda_form` selects `:nrl` (default), `:sauter`, or `:wesson`.

Throws if any surface's `restype` is still `nothing` — call
`ForceFreeStates.resist_eval_all!(intr, equil)` first.
"""
function build_ggj_inputs(equil, sings, profiles::KineticProfiles;
    mu_i::Real=2.0, zeff::Real=1.0,
    v1_scale::Real=1.0,
    resistivity_model::NeoResistivityModel=SpitzerModel(),
    lnLambda_form::Symbol=:nrl)
    psio = equil.psio
    chi1 = 2π * psio

    out = Vector{GGJParameters}(undef, length(sings))
    for (k, sing) in enumerate(sings)
        rg = sing.restype
        rg === nothing &&
            throw(
                ArgumentError(
                    "build_ggj_inputs: surface $k has " *
                    "restype = nothing. Call " *
                    "ForceFreeStates.resist_eval_all!(intr, equil) " *
                    "after sing_find! to populate it."
                )
            )
        rg isa ResistGeometry ||
            throw(ArgumentError("build_ggj_inputs: surface $k has " *
                                "restype of unexpected type $(typeof(rg))."))

        # Kinetic profiles at this surface
        prof = profiles(sing.psifac)
        n_e = prof.n_e          # [m⁻³]
        t_e = prof.T_e          # [eV]

        # Shared Coulomb log and resistivity closure (identical to SLAYER
        # when the same resistivity_model is selected).
        lnLamb = coulomb_log_e(n_e, t_e; form=lnLambda_form)
        if resistivity_model isa SpitzerModel
            eta_use = eta_spitzer(n_e, t_e, zeff; lnLamb=lnLamb)
        else
            nuestar = nu_star_e(n_e, t_e, rg.R_major, rg.eps_local,
                sing.q, zeff; lnLamb=lnLamb)
            eta_use = eta_neoclassical(resistivity_model, n_e, t_e, zeff,
                rg.f_trap, nuestar; lnLamb=lnLamb)
        end
        rho = mu_i * M_P * n_e

        # Alfvén time at the rational surface
        n_tor = Int(sing.n[1])
        v1 = rg.v1_local * v1_scale
        taua = sqrt(rho * rg.M * MU_0) /
               abs(2π * n_tor * sing.q1 * chi1 / v1)

        # Resistive diffusion time
        taur = (rg.avg_bsq_over_dpsisq / rg.avg_bsq) * MU_0 / eta_use

        # dV/dψ normalized by total plasma volume (`v1 = v1/volume`). This is
        # the `v1` consumed by `rescale_delta` as v1^(2p1); NOT the raw V'
        # used in τ_A above.
        equil.params.volume === nothing &&
            throw(
                ArgumentError(
                    "build_ggj_inputs: equil.params.volume " *
                    "is nothing. Ensure the equilibrium " *
                    "solver populated the total plasma " *
                    "volume before building GGJ inputs."
                )
            )
        v1_norm = rg.v1_local / equil.params.volume

        out[k] = GGJParameters(;
            E=rg.E, F=rg.F, G=rg.G, H=rg.H, K=rg.K, M=rg.M,
            taua=taua, taur=taur, v1=v1_norm, ising=k
        )
    end
    return out
end
