```@meta
CurrentModule = GeneralizedPerturbedEquilibrium
```

```@eval
using Markdown
using GeneralizedPerturbedEquilibrium
readme = read(joinpath(pkgdir(GeneralizedPerturbedEquilibrium), "README.md"), String)
# The "Full documentation" self-link belongs in the GitHub README, not on the doc site itself.
# If that line is ever removed from README.md this `replace` silently no-ops, which is fine —
# the strip is defensive, not load-bearing.
readme = replace(readme, r"^Full documentation:.*$"m => "")
Markdown.parse(readme)
```

## What GPEC computes

One deck, `gpec.toml`, runs the stages below in order, each when its section is present, into one output file, `gpec.h5`.
The [Workflow](workflow.md) page maps every stage to its physics, deck sections and datasets.

| Stage | Deck section | Main output |
|---|---|---|
| `Equilibrium` — the axisymmetric equilibrium, its flux-surface coordinates and rational surfaces | `[Equilibrium]` | `Equilibrium/` |
| `ForceFreeStates` — ideal (or drift-kinetic) MHD stability by the Newcomb criterion, with the `Vacuum` response of the `[Wall]` | `[ForceFreeStates]` | `ForceFreeStates/FreeBoundaryStability/`, `SingularSurfaces/Delta_prime_matrix` |
| `LocalStability` — Mercier, resistive-interchange and ballooning criteria | `[LocalStability]` | `LocalStability/` |
| `Match` — the resistive inner layer matched to the outer-region Δ′ | `[Match]` | `SingularSurfaces/Match/` |
| `PerturbedEquilibrium` — the plasma response to the field of `[ForcingTerms]`: resonant fields, islands, overlap | `[PerturbedEquilibrium]` | `PerturbedEquilibrium/SingularCoupling/` |
| `KineticForces` — the neoclassical toroidal viscosity torque of that response | `[KineticForces]` | `KineticForces/` |
| `ErrorFields` — coil tolerances, locking risk and correction limits | `[ErrorFields]` | `ErrorFields/Risk/` |
| `Tearing` — tearing growth rates and critical resonant fields from an inner-layer model | `[Tearing]` | `Tearing/Roots/` |

## Documentation

- [**Setup**](set_up.md) — Installation on Linux, macOS and Windows (WSL), and a first run
- [**Workflow**](workflow.md) — The pipeline: physics of each stage, deck sections, examples and the output file
- [**Scripting API**](api.md) — The same pipeline from Julia scripts
- [**Conventions**](conventions.md) — Coordinates, signs, normalizations and field amplitudes
- [**Modules**](equilibrium.md) — Governing equations, deck keys, outputs and function reference per module
- [**Citations**](citations.md) — Papers underpinning GPEC's algorithms
- [**Developer Notes**](developer_notes.md) — Git workflow, commit conventions and the regression-testing harness
- [**Introduction slides**](https://docs.google.com/presentation/d/1Iey7bsvpDG1WDRow24OD0NOWyKYOXMs0U7QLc7Y-wj4/edit?slide=id.g2798d4bd85b_0_15#slide=id.g2798d4bd85b_0_15) — An overview of GPEC and its physics

## Contact

GPEC is developed and maintained by **Nikolas Logan** ([ncl2128@columbia.edu](mailto:ncl2128@columbia.edu)) and **Jong-Kyu Park** ([jkpark@snu.ac.kr](mailto:jkpark@snu.ac.kr)) as lead developers.

New users and developers are warmly encouraged to reach out — whether you have questions about running GPEC, want to discuss the physics, or are interested in contributing or collaborating, please get in touch.

For bug reports and feature requests, please open an issue on the [GitHub repository](https://github.com/OpenFUSIONToolkit/GPEC/issues).
