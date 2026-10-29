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
resp = plasma_response(ffs)
pe   = perturbed_equilibrium(resp, rmp)
```

`solve(eq, alg; kwargs...)` is sugar for the canonical problem form: a
[`ForceBalanceProblem`](@ref) names WHAT is being solved (the perturbed-plasma
Euler-Lagrange system posed on this equilibrium, with this mode range, wall and closure)
and the integrator names HOW. A `PlasmaEquilibrium` hosts many possible problems; the
problem type keeps `solve` unambiguous as other problem classes appear.

```julia
prob = ForceBalanceProblem(eq; nn=1, delta_mlow=8, delta_mhigh=8, vac_flag=true)
ffs  = solve(prob, Riccati())
save_to_hdf5("gpec.h5", ffs)
```

`solve` returns its result without writing a file. Call `save_to_hdf5(path, ffs)`
or pass an open HDF5 file or group to save it explicitly. The deck driver saves its
FFS result after the solve.

The same `save_to_hdf5(target, result)` function saves an equilibrium, local-stability
result, perturbed-equilibrium state, kinetic-force state, tearing result, or error-field
result. Each module owns the datasets for its result. Paths open or create an HDF5
file; an open file or group remains owned by the caller. Stage computations return
results, and the deck driver applies the configured output switches and filenames.

The objects map one-to-one onto the pipeline stages:

  - [`PlasmaEquilibrium`](@ref) reads and processes the equilibrium — the `[Equilibrium]`
    section. Analytic equilibria take their parameters from their config object, with the
    `eq_type` implied, and IMAS equilibria from a data dictionary:

    ```julia
    eq = PlasmaEquilibrium(Equilibrium.SolovevConfig(; elongation=1.5); mpsi=0)
    eq = PlasmaEquilibrium(dd; imas_cocos=11)
    ```

    Every equilibrium remembers its source, so with `mpsi=0` `solve` re-forms it on the same
    refined two-pass grid the deck uses; `Equilibrium.reform(eq, psi_nodes)` re-forms it on any grid.
  - `solve` runs the force-free-states stage — the `[ForceFreeStates]` section — and returns
    a `ForceFreeStatesResult`, exactly the object the TOML driver publishes.
  - [`RMPField`](@ref) describes the external field — the `[ForcingTerms]` section — without
    reading anything from disk until it is applied.
  - [`plasma_response`](@ref) builds the plasma-response and singular-coupling operators of a
    `ForceFreeStatesResult` once, and `perturbed_equilibrium(resp, rmp)` drives them with one field —
    together the `[PerturbedEquilibrium]` section.

## Kinetic profiles

The equilibrium carries the kinetic profiles. Name the file with the `kinetic_file` (and,
for HDF5, `kinetic_group`) keyword, the same keys as the `[Equilibrium]` deck section, and it
is read once into `equil.kinetic`; every kinetic stage derives its own view from it.
[`attach_kinetic_profiles!`](@ref GeneralizedPerturbedEquilibrium.Equilibrium.attach_kinetic_profiles!) attaches a file or in-memory `KineticProfileData` to an
equilibrium built without one, such as an analytic equilibrium.

```julia
eq = PlasmaEquilibrium("input.geqdsk"; jac_type="hamada", kinetic_file="kinetic.h5")
```

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
pe = perturbed_equilibrium(resp, 2.0 * nominal + 0.5 * weld_field)
```

This is the substrate for error-field workflows: build per-unit sources independently
(a coil set per ampere, a displacement field per millimeter), then assemble physical cases
by weighting and summing — without recomputing anything per combination until the final
`perturbed_equilibrium` call, and with one `plasma_response` serving every combination.
`RMPField(modes)` wraps a control-surface spectrum computed elsewhere, already in the unit-norm
`Φ_x` convention, as a leaf that combines like any other.

The weights are linear-combination coefficients, not physical amplitudes: sources that are
not scalar multiples of each other get their own description and the algebra. A coil set
with a failed conductor is `nominal - failed_coil`, not `0.9 * nominal`.

## Choosing an integrator

The second argument of `solve` picks the formalism, and its fields are that formalism's
numerical settings. Physical inputs belong on `ForceBalanceProblem`; ODE tolerances and
storage settings belong on `Forward` or `Riccati`, and `extended_precision_bvp` belongs only
on `Riccati`. The current deck reader translates its existing section layout into those objects:

```julia
ffs = solve(eq, Forward(; eulerlagrange_tolerance=1e-8); nn=1, vac_flag=true) # dense ξ profiles for PerturbedEquilibrium
ffs = solve(eq, Riccati(; nchunks=40); nn=1, vac_flag=true)     # chunked propagators, Δ′ matrix
ffs = solve(eq, Galerkin(; nx=512); nn=1)                       # RDCON outer-region Galerkin Δ′
```

Which products each formalism can supply differs; a result carries `nothing` in the fields
its integrator does not produce and consumers warn and skip rather than erroring. See the
[Stability Analysis](stability.md) page for the result struct and its capability gates.

`singfac_min`, `frobenius_order`, `frobenius_psi_max`, `truncate_at_dW_peak`, and
`dW_edge_scan_start` remain problem settings. `Galerkin` rejects non-default values of the
last three. Its per-surface order adjustment is controlled by `frobenius_order_ceiling`,
and its factorization is selected automatically.

## Inner-layer matching

Inner-layer matching is its own problem, posed on a finished solve: a `MatchProblem` holds
the outer Δ′ the solve published plus the per-surface layer parameters, and the inner-layer
model passed to `solve` computes the layer response at the prescribed rotation. Solving it
returns a new result with the closure changed from `:ideal` to `:matched` and the
eigenfunctions replaced — the expensive outer solve is reused, so layer-parameter scans
cost one cheap match solve per point:

```julia
ffs = solve(eq, Galerkin(; rpec_flag=true, cut_solution=true); nn=1)

matched = solve(MatchProblem(ffs; eta=1e-6, rho=1e-7, rotation=0.0), GGJModel())
@assert matched.closure === :matched

# A rotation scan reuses the one outer solve:
bpens = [solve(MatchProblem(ffs; eta=1e-6, rho=1e-7, rotation=f), GGJModel()).bpen for f in 0.0:50.0:500.0]
```

A scalar applies to every surface; a vector gives one value per entry of `ffs.surfaces`,
core to edge. The problem needs a Δ′ payload with coil-response columns, so it accepts Galerkin (`rpec_flag=true`) and
Riccati (`vac_flag=true`) results; a Riccati-fed match fills `bpen` and the resonant data but keeps
`solution === nothing` (no outer basis is retained). Only a closure-capable model is
accepted — `GGJModel()` today; `SLAYERModel()` is slab-only and drives the free-eigenvalue tearing
solve instead.

## Tearing stability

A `TearingProblem` holds the outer solution and the surface coupling. The plasma is the
equilibrium's kinetic data (profiles, species, transport defaults, resistivity closure), the
inner-layer model holds its own closures, and `TearingConfig` holds the scan and root-search
settings.

```julia
tear = solve(TearingProblem(ffs), SLAYERModel(; P_perp_model=:chi_perp_e);
    config=TearingConfig(; Q_re_range=(-2.0, 2.0), Q_im_range=(-0.5, 3.0)))
ggj = solve(TearingProblem(ffs; coupling_mode=:coupled), GGJModel(; solver=:ray);
    config=TearingConfig(; gamma_range=(1.0, 1e4), omega_range=(-1e4, 1e4)))
save_to_hdf5("gpec.h5", tear)
```

GGJ needs the raw Δ′ matrix and validates the surface ordering. Its η and ρ default from the
kinetic data: η by the resistivity closure SLAYER uses, ρ = m_i·n_i of the main ion; pass `eta`
and `rho` to `TearingProblem` to override them. GGJ takes no rotation and reports plasma-frame
frequencies; SLAYER reports lab-frame frequencies with its per-surface E×B shifts. Both expose
`omega` in rad/s and `gamma` in 1/s, positive for instability, and the HDF5 roots group records
`frequency_frame`. SLAYER searches a normalized Q box, GGJ a physical (γ, ω) box.

Kinetic runs (`kinetic_factor > 0`) take `Forward()`. The `"calculated"` source reads the
equilibrium's kinetic profiles and species, with the default `KineticForcesControl` collision
settings.

## Local stability

The Mercier, resistive interchange and ballooning profiles are their own stage, posed on the
equilibrium. A [`LocalStabilityProblem`](@ref) holds the equilibrium, ballooning angle
`theta_k`, and `alpha_boundary_scan` choice. [`Ballooning`](@ref) holds numerical tolerances
and scan settings. `solve` returns a [`LocalStabilityResult`](@ref); its `verbose` keyword
defaults to `false`. The current deck reader obtains all of these from `[LocalStability]`.

```julia
problem = LocalStabilityProblem(eq; theta_k=0.0, alpha_boundary_scan=true)
ls = solve(problem, Ballooning(; n_scan=64); verbose=false)
ls.psi, ls.D_I, ls.D_R, ls.ballooning_Delta_prime     # profiles on the equilibrium ψ grid
ls.ballooning_psi, ls.alpha, ls.alpha_critical        # first ballooning boundary, NaN outside its scan window
```

## Entry points

```@docs
GeneralizedPerturbedEquilibrium.ForceBalanceProblem
GeneralizedPerturbedEquilibrium.solve
GeneralizedPerturbedEquilibrium.perturbed_equilibrium
```

[`plasma_response`](@ref GeneralizedPerturbedEquilibrium.PerturbedEquilibrium.plasma_response), which builds the
[`PlasmaResponse`](@ref GeneralizedPerturbedEquilibrium.PerturbedEquilibrium.PlasmaResponse) that `perturbed_equilibrium`
applies, is documented with the [perturbed equilibrium](perturbed_equilibrium.md) module.

## Integrator selectors

```@autodocs
Modules = [GeneralizedPerturbedEquilibrium.ForceFreeStates]
Pages = ["Integrators.jl"]
```
