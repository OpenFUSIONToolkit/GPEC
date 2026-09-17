# Scripting API

GPEC's pipeline is normally driven from a `gpec.toml` deck through
`GeneralizedPerturbedEquilibrium.main`. The scripting API exposes the same stages as
ordinary Julia functions, so a run can be built, parameterized and looped over in a script
without writing a deck.

```julia
using GeneralizedPerturbedEquilibrium

eq  = PlasmaEquilibrium("input.geqdsk"; jac_type="hamada")
ffs = solve(eq, Riccati(); nn=1, delta_mlow=8, delta_mhigh=8, vac_flag=true)
rmp = RMPField("coils.dat")
pe  = perturbed_equilibrium(ffs, rmp)
```

`solve(eq, alg; kwargs...)` is sugar for the canonical problem form: a
[`EulerLagrangeProblem`](@ref) names WHAT is being solved (the perturbed-plasma
Euler-Lagrange system posed on this equilibrium, with this mode range, wall and closure)
and the integrator names HOW. A `PlasmaEquilibrium` hosts many possible problems; the
problem type keeps `solve` unambiguous as other problem classes appear.

```julia
prob = EulerLagrangeProblem(eq; nn=1, delta_mlow=8, delta_mhigh=8, vac_flag=true)
ffs  = solve(prob, Riccati())
```

The four objects map one-to-one onto the pipeline stages:

  - [`PlasmaEquilibrium`](@ref) reads and processes the equilibrium — the `[Equilibrium]`
    section. Analytic equilibria (`sol`, `lar`, `tj_analytic`) take their parameters from a
    separate config object and are built with
    `setup_equilibrium(eq_config, analytic_config)` instead.
  - `solve` runs the force-free-states stage — the `[ForceFreeStates]` section — and returns
    a `ForceFreeStatesResult`, exactly the object the TOML driver publishes.
  - [`RMPField`](@ref) describes the external field — the `[ForcingTerms]` section — without
    reading anything from disk until it is applied.
  - `perturbed_equilibrium` runs the plasma-response stage — the `[PerturbedEquilibrium]`
    section.

## Combining forcing sources

`RMPField`s form a vector space: `+`, `-` and scalar `*` build lazy linear combinations that
record their terms and compute nothing until the perturbed-equilibrium stage materializes
them. Because the plasma response is linear in the forcing, driving with a combination is
exactly equivalent to combining the individually materialized fields — each term is
evaluated on the control surface and the mode amplitudes are summed. A complex scalar
phase-rotates a source.

```julia
nominal    = RMPField("nominal_efc.dat")
weld_field = RMPField("weld_fields.h5")
pe = perturbed_equilibrium(ffs, 2.0 * nominal + 0.5 * weld_field)
```

This is the substrate for error-field workflows: build per-unit sources independently
(a coil set per ampere, a displacement field per millimeter), then assemble physical cases
by weighting and summing — without recomputing anything per combination until the final
`perturbed_equilibrium` call.

The weights are linear-combination coefficients, not physical amplitudes: sources that are
not scalar multiples of each other get their own description and the algebra. A coil set
with a failed conductor is `nominal - failed_coil`, not `0.9 * nominal`.

## Choosing an integrator

The second argument of `solve` picks the formalism, and its fields are that formalism's
tunables. Everything else is a `ForceFreeStatesControl` keyword, so the TOML keys and the
`solve` keywords are the same knobs:

```julia
ffs = solve(eq, Forward(); nn=1, vac_flag=true)                 # dense ξ profiles for PerturbedEquilibrium
ffs = solve(eq, Riccati(; nchunks=40); nn=1, vac_flag=true)     # chunked propagators, Δ′ matrix
ffs = solve(eq, Galerkin(; nx=512); nn=1)                       # RDCON outer-region Galerkin Δ′
```

Which products each formalism can supply differs; a result carries `nothing` in the fields
its integrator does not produce and consumers warn and skip rather than erroring. See the
[Stability Analysis](stability.md) page for the result struct and its capability gates.

Inner-layer matching is requested with the integrator-agnostic `match` keyword, which closes
the basis with a resistive layer solution instead of the ideal jump condition:

```julia
ffs = solve(eq, Galerkin(); nn=1,
    match=ResistiveMatch(; eta=[1e-6, 2e-6], rho=[1e-7, 1e-7], rotation=[0.0, 0.0]))
@assert ffs.closure === :matched
```

Only the Galerkin formalism implements the match today; requesting one from `Forward` or
`Riccati` errors. Kinetic runs (`kinetic_factor > 0`) need the `[KineticForces]` profiles and
remain TOML-driven.

## Entry points

```@docs
GeneralizedPerturbedEquilibrium.EulerLagrangeProblem
GeneralizedPerturbedEquilibrium.solve
GeneralizedPerturbedEquilibrium.perturbed_equilibrium
```

## Integrator selectors

```@autodocs
Modules = [GeneralizedPerturbedEquilibrium.ForceFreeStates]
Pages = ["Integrators.jl"]
```
