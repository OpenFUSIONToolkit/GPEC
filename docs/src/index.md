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

## Documentation

- [**Setup**](set_up.md) — Installation instructions for macOS and Windows (WSL)
- [**Workflow**](workflow.md) — Full pipeline description: inputs, outputs, and key physics per module
- [**API Reference**](vacuum.md) — Detailed function and type documentation for each module
- [**Citations**](citations.md) — Papers underpinning GPEC's algorithms
- [**Developer Notes**](developer_notes.md) — Git workflow, commit conventions, and the regression-testing harness

## Contact

GPEC is developed and maintained by **Nikolas Logan** ([ncl2128@columbia.edu](mailto:ncl2128@columbia.edu)) and **Jong-Kyu Park** ([jkpark@snu.ac.kr](mailto:jkpark@snu.ac.kr)) as lead developers.

New users and developers are warmly encouraged to reach out — whether you have questions about running GPEC, want to discuss the physics, or are interested in contributing or collaborating, please get in touch.

For bug reports and feature requests, please open an issue on the [GitHub repository](https://github.com/OpenFUSIONToolkit/GPEC/issues).
