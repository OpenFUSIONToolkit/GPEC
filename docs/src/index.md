# GPEC.jl

*A Julia implementation of the Generalized Perturbed Equilibrium Code suite*

GPEC.jl is a work-in-progress Julia port of the [Generalized Perturbed Equilibrium Code (GPEC)](https://github.com/PrincetonUniversity/GPEC) suite, providing tools for magnetohydrodynamic (MHD) equilibrium and stability analysis in fusion plasmas.

## Overview

GPEC provides functionality for:

- **some stuff** Fill this in later

## Installation

```julia
using Pkg
Pkg.add("GeneralizedPerturbedEquilibrium")
```

## Quick Start

### Running GPEC as a Script

GPEC includes an executable script in the project root for easy command-line usage:

```bash
./gpec path/to/directory
```

This will:
1. Read the `gpec.toml` configuration file from the specified directory
2. Set up the equilibrium
3. Compute force-free states (stability analysis)
4. Optionally compute perturbed equilibrium response (if configured)

If no directory is provided, GPEC will use the current directory (`./`):

```bash
./gpec
```

### Using GPEC as a Library

You can also use GPEC programmatically in your own Julia code:

```julia
using GeneralizedPerturbedEquilibrium

# Run full analysis from a directory containing gpec.toml
GeneralizedPerturbedEquilibrium.main(["path/to/directory"])

# Or set up equilibrium only
using GeneralizedPerturbedEquilibrium.Equilibrium
equil = setup_equilibrium("path/to/gpec.toml")
```

See the [Examples](@ref) section for detailed usage examples.

## Modules

```@contents
Pages = ["splines.md", "vacuum.md", "equilibrium.md"]
Depth = 1
```

## Examples

The package includes several Jupyter notebook examples:

- `example.ipynb`: Fill this in later

## Developer Notes

### Commit Messages

To assist with release note compilation, please follow the commit message format:
```
CODE - TAG - Detailed message
```
where CODE is Equil, ForceFreeStates, Vacuum, etc. and TAGs are descriptors like WIP, MINOR, IMPROVEMENT, BUG FIX, NEW FEATURE, etc.

Additionally, please see [this](https://docs.google.com/document/d/1XAOTz1IV8ErZAAk-iSuEuddNOLB5XcoVZsAbPKRUUuA) google doc for more details on using the GitHub.

## Links

- [Source Repository](https://github.com/OpenFUSIONToolkit/GPEC)
- [Original GPEC](https://github.com/PrincetonUniversity/GPEC)
