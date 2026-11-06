# Scripting API

A deck runs through `GeneralizedPerturbedEquilibrium.main`; the scripting API exposes the same stages as Julia functions, so a run can be built, parameterized and looped over without a deck, and a deck can never take a path a script cannot.
Every stage is a `solve` of a problem type with an algorithm, or a function on the previous stage's result; nothing is written until `save_to_hdf5` is called.

```julia
using GeneralizedPerturbedEquilibrium

eq   = PlasmaEquilibrium("input.geqdsk"; jac_type="hamada", kinetic_file="kinetic.h5")
ffs  = solve(eq, Forward(); nn=1, delta_mlow=8, delta_mhigh=8, vac_flag=true)           # [ForceFreeStates]: Forward, no wall, for the response below
ls   = solve(LocalStabilityProblem(eq; alpha_boundary_scan=true), Ballooning())          # [LocalStability]
resp = plasma_response(ffs; compute_singular_coupling=true)                               # [PerturbedEquilibrium]
pe   = perturbed_equilibrium(resp, RMPField("coils.dat"))                                 #   with the [ForcingTerms] drive
kf   = kinetic_torque(pe, KineticForcesControl())                                         # [KineticForces]
ef   = error_fields(resp, pe.coil_sets, ForcingTerms.CoilConfig(ft_ctrl);                  # [ErrorFields]; ft_ctrl: the [ForcingTerms] control
                    tolerances=ErrorFields.read_tolerance_toml("tolerances.toml"))
tear = solve(TearingProblem(ffs), SLAYERModel(); config=TearingConfig())                  # [Tearing]

save_to_hdf5("gpec.h5", ffs; local_stability=ls)
foreach(r -> save_to_hdf5("gpec.h5", r), (pe, kf, ef, tear))
```

`solve(eq, alg; kwargs...)` is sugar for the problem form `solve(ForceBalanceProblem(eq; kwargs...), alg)`: the problem names what is solved (the force-balance equation on this equilibrium, with this mode range, wall and closure) and the algorithm names how.
`main` translates each deck section into the problem, algorithm or control it names, calls these functions for every section present, saves each result into `[Run] output_file`, and returns `(; equil, local_stability, ffs, match, resp, pe, kinetic, error_fields, tearing)`, each `nothing` when its section is absent.

| Stage | Deck section | Call | Result |
|---|---|---|---|
| Equilibrium | `[Equilibrium]` (or `[Solovev]`, `[LargeAspectRatio]`, `[TJAnalytic]`) | `PlasmaEquilibrium(file; kwargs...)`, `PlasmaEquilibrium(config)`, `PlasmaEquilibrium(dd)` | `PlasmaEquilibrium` |
| ForceFreeStates | `[ForceFreeStates]` + `[Forward]`, `[Riccati]` or `[Galerkin]` | `solve(ForceBalanceProblem(eq; ...), Forward() \| Riccati() \| Galerkin())` | `ForceFreeStatesResult` |
| LocalStability | `[LocalStability]` + `[Ballooning]` | `solve(LocalStabilityProblem(eq; ...), Ballooning(; ...))` | `LocalStabilityResult` |
| Match | `[Match]` + `[GGJ]` | `solve(MatchProblem(ffs; ...), GGJModel())` | `ForceFreeStatesResult` with `closure == :matched` |
| Drive | `[ForcingTerms]` | `RMPField(ft_ctrl)` (a `ForcingTermsControl`), `RMPField(file)`, `RMPField(modes)` | `RMPField` |
| PerturbedEquilibrium | `[PerturbedEquilibrium]` | `plasma_response(ffs; ...)`, then `perturbed_equilibrium(resp, rmp; ...)` | `PlasmaResponse`, `PerturbedEquilibriumState` |
| KineticForces | `[KineticForces]` | `kinetic_torque(pe, KineticForcesControl(; ...))` | `KineticForcesResult` |
| ErrorFields | `[ErrorFields]` | `error_fields(resp, pe.coil_sets, CoilConfig(...); ...)` | `ErrorFieldsResult` |
| Tearing | `[Tearing]` + `[SLAYER]` or `[GGJ]` | `solve(TearingProblem(ffs; ...), SLAYERModel() \| GGJModel(); config=TearingConfig(...))` | `TearingResult` |

The [Workflow](workflow.md) page describes what each stage computes; the module pages document the keywords, which are the deck keys of the matching section.

## Inspecting a result

Each result prints a short summary at the REPL: its headline numbers, which optional products it carries and, when one is missing, why. Arrays are summarized by size or range, never printed. Inside a container, such as the named tuple `main` returns, a result prints on one line.

```julia-repl
julia> ffs
ForceFreeStatesResult  Riccati, ideal closure
  Physics            ideal MHD
  Modes              n = 1, m = -4:3 (8 total)
  Domain             ψ ∈ [0.0001, 0.9995], q_lim = 3.262
  Surfaces           2: 2/1 @ ψ = 0.1221, 3/1 @ ψ = 0.8804
  Fixed boundary     stable (no zero crossings of the critical determinant)
  Free boundary      stable: et₁ = 16.4, ep₁ = 14.04, ev₁ = 2.365, vacuum λ = 2.315
  Δ′_ii (ψ_N)        2/1: 0.1781, 3/1: -215.7
  Solution           not computed (Riccati stores no ξ profiles)
  Wall               ConformalWall, standoff = 0.2415 a
  Plots              13 of 15 available (see available_plots)
```

The fields themselves are documented in each type's docstring.

## Plotting a result

Every plot in [`Analysis`](analysis.md) takes the result in memory or the path of the `gpec.h5` it was saved to, and draws with the Makie backend the session has loaded (`using CairoMakie` or `using GLMakie`). `available_plots` lists the plots for a result, a path, or the named tuple `main` returns, and for each one the run cannot draw, what to change:

```julia-repl
julia> available_plots(ffs)
Plots for ForceFreeStatesResult (riccati): 13 of 15 available

ForceFreeStates
  ✓ plot_ffs_summary           the stability panels the run supplies
  ✗ plot_mode_displacement     needs stored ξ profiles: use the Forward integrator or a matched Galerkin solve (Riccati and unmatched Galerkin store none)
  ✓ plot_critical_determinant  critical determinant against ψ_N
  ⋮
```

A plot the source cannot draw throws an `ArgumentError` with the same reason, and a summary leaves out the panels it cannot draw:

```julia
plot_ffs_summary(ffs)         # the result in memory
plot_ffs_summary("gpec.h5")   # or the file to which it was saved
```

## Saving and rerunning

`save_to_hdf5(target, result)` saves any stage result; `target` is a path, which opens or creates the file, or an open HDF5 file or group, which stays owned by the caller.
Each save records in `Input/gpec_toml_raw` the deck sections that state the problem, algorithm and controls the result was computed with, so a script run's file replays like a deck run's:

```bash
./gpec gpec.h5                 # main(["gpec.h5"]) in Julia
./gpec gpec.h5 --coil-source coils
```

A coil run is replayed either from the stored deck coil configuration (`--coil-source forcing-modes`, the default, which needs the original geometry files) or from the coil geometry stored under `Input/RawInputs/Coils/` (`--coil-source coils`, which needs no original files and allows the equilibrium to be overridden).
A forcing file is replayed from its raw contents under `Input/RawInputs/ForcingTerms/`, and a forcing no deck can state (a weighted field, a spectrum or a sum) from the control-surface modes the save records.

## Equilibrium and kinetic profiles

Analytic equilibria take their parameters from a config object, with the `eq_type` implied, and IMAS equilibria from a data dictionary:

```julia
eq = PlasmaEquilibrium(Equilibrium.SolovevConfig(; elongation=1.5); mpsi=0)
eq = PlasmaEquilibrium(dd; imas_cocos=11)
```

Every equilibrium remembers its source, so with `mpsi=0` `solve` re-forms it on the refined two-pass grid the deck uses; `Equilibrium.reform(eq, psi_nodes)` re-forms it on any grid.
The `kinetic_file` (and, for HDF5, `kinetic_group`) keyword reads the kinetic profiles once into `equil.kinetic`, from which every kinetic stage derives its view; [`Equilibrium.attach_kinetic_profiles!`](@ref) attaches a file or an in-memory `KineticProfileData` to an equilibrium built without one.

## Combining forcing sources

`RMPField`s form a vector space: `+`, `-` and scalar `*` build lazy linear combinations that compute nothing until `perturbed_equilibrium` materializes them, and because the response is linear in the forcing, driving with a combination equals combining the materialized fields; a complex scalar phase-rotates a source.

```julia
nominal    = RMPField("nominal_efc.dat")
weld_field = RMPField("weld_fields.h5")
pe = perturbed_equilibrium(resp, 2.0 * nominal + 0.5 * weld_field)
```

This is the substrate for error-field workflows: build per-unit sources (a coil set per ampere, a displacement field per millimetre), then assemble physical cases by weighting and summing, with one `plasma_response` serving every combination.
`RMPField(modes)` wraps a control-surface spectrum computed elsewhere, in the unit-norm ``\Phi_x`` convention, as a leaf.
The weights are linear-combination coefficients, not physical amplitudes: a coil set with a failed conductor is `nominal - failed_coil`, not `0.9 * nominal`.

## Choosing an integrator

The second argument of `solve` picks the formalism and its fields are that formalism's numerical settings; physical inputs belong on `ForceBalanceProblem`:

```julia
ffs = solve(eq, Forward(; eulerlagrange_tolerance=1e-8); nn=1, vac_flag=true)   # dense ξ profiles: the only input plasma_response accepts; kinetic solves
ffs = solve(eq, Riccati(; nchunks=40); nn=1, vac_flag=true)                      # chunked propagators, Δ′ matrix (default)
ffs = solve(eq, Galerkin(; nx=512); nn=1)                                        # finite-element outer-region Δ′
```

A result carries `nothing` in the fields its integrator does not produce, and consumers warn and skip rather than erroring; `plasma_response` in particular needs a `Forward()` solve with `wall=nothing` (it errors on a wall or a Galerkin result and skips with a warning on a Riccati one), see [Stability](stability.md).
A kinetic solve takes `Forward()` with the problem's `kinetic` keyword: `kinetic=KineticForcesControl(; ...)` evaluates the drift-kinetic matrices on the equilibrium's kinetic profiles, `kinetic=TestKineticMatrices(; scale)` uses synthetic matrices, `kinetic_factor` scales either, and `km = kinetic_matrices(prob)` evaluates the source once for reuse in later solves ([`KineticMatrices`](@ref GeneralizedPerturbedEquilibrium.ForceFreeStates.KineticMatrices)).

## Inner-layer matching

A `MatchProblem` holds the outer Δ′ a solve published plus per-surface layer parameters, and the inner-layer model passed to `solve` computes the layer response at the prescribed rotation; the result is a new `ForceFreeStatesResult` with the closure changed from `:ideal` to `:matched` and the eigenfunctions replaced, so layer-parameter scans cost one cheap match per point:

```julia
ffs = solve(eq, Galerkin(; rpec_flag=true, cut_solution=true); nn=1)
matched = solve(MatchProblem(ffs; eta=1e-6, rho=1e-7, rotation=0.0), GGJModel())
bpens = [solve(MatchProblem(ffs; eta=1e-6, rho=1e-7, rotation=f), GGJModel()).bpen for f in 0.0:50.0:500.0]
```

A scalar applies to every surface and a vector gives one value per entry of `ffs.surfaces`, core to edge.
The problem needs a Δ′ payload with coil-response columns, so it accepts Galerkin (`rpec_flag=true`) and Riccati (`vac_flag=true`) results; a Riccati-fed match fills `bpen` and the resonant data but keeps `solution === nothing`.
`GGJModel()` is the closure-capable model; `SLAYERModel()` is slab-only and drives the tearing solve instead.

## Tearing stability

A `TearingProblem` holds the outer solution and the surface coupling, the inner-layer model its own closures, and `TearingConfig` the scan and root-search settings:

```julia
tear = solve(TearingProblem(ffs), SLAYERModel(; P_perp_model=:chi_perp_e);
    config=TearingConfig(; Q_re_range=(-2.0, 2.0), Q_im_range=(-0.5, 3.0)))
ggj = solve(TearingProblem(ffs; coupling_mode=:coupled), GGJModel(; solver=:ray);
    config=TearingConfig(; gamma_range=(1.0, 1e4), omega_range=(-1e4, 1e4)))
```

Both models report lab-frame frequencies (`omega` in rad/s, `gamma` in 1/s, positive for instability, with `frequency_frame` recorded on the roots group); SLAYER searches a normalized Q box and GGJ a physical (γ, ω) box.

## Local stability

```julia
problem = LocalStabilityProblem(eq; theta_k=0.0, alpha_boundary_scan=true)
ls = solve(problem, Ballooning(; n_scan=64); verbose=false)
ls.psi, ls.D_I, ls.D_R, ls.ballooning_Delta_prime     # profiles on the equilibrium ψ grid
ls.ballooning_psi, ls.alpha, ls.alpha_critical        # the ballooning boundary, NaN outside its scan window
```

## Entry points

```@docs
GeneralizedPerturbedEquilibrium.ForceBalanceProblem
GeneralizedPerturbedEquilibrium.solve
GeneralizedPerturbedEquilibrium.kinetic_matrices
GeneralizedPerturbedEquilibrium.perturbed_equilibrium
```

[`plasma_response`](@ref GeneralizedPerturbedEquilibrium.PerturbedEquilibrium.plasma_response), which builds the
[`PlasmaResponse`](@ref GeneralizedPerturbedEquilibrium.PerturbedEquilibrium.PlasmaResponse) that `perturbed_equilibrium`
applies, is documented with the [Perturbed Equilibrium](perturbed_equilibrium.md) module.

## Integrator selectors

```@autodocs
Modules = [GeneralizedPerturbedEquilibrium.ForceFreeStates]
Pages = ["Integrators.jl"]
```
