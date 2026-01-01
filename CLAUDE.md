# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

JPEC (Julia Perturbed Equilibrium Code) is a work-in-progress Julia port of the Generalized Perturbed Equilibrium Code (GPEC) suite for magnetohydrodynamic (MHD) equilibrium and stability analysis in fusion plasmas. The codebase is a hybrid Julia/Fortran implementation, with active Julia development alongside legacy Fortran code that is called via ccall.

## Common Commands

### Building and Testing

```bash
# Run all tests (includes building Fortran)
julia --project=. -e 'using Pkg; Pkg.activate("."); Pkg.build(); Pkg.instantiate(); include("test/runtests.jl")'

# Build Fortran libraries only
julia --project=. -e 'using Pkg; Pkg.activate("."); Pkg.build()'

# Run specific test file
julia --project=. test/runtests.jl test/runtests_spline.jl

# Available test files:
# - test/runtests_build.jl
# - test/runtests_spline.jl
# - test/runtests_vacuum_fortran.jl
# - test/runtests_vacuum_julia.jl
# - test/runtests_solovev.jl
# - test/runtests_ode.jl
# - test/runtests_sing.jl
# - test/runtests_fullruns.jl
```

### Building Documentation

```bash
# Build documentation locally
julia --project=. build_docs_local.jl
```

### Development with Revise

For faster recompilation during development, use Revise.jl (installed in global environment, not in Project.toml):

```julia
using Revise
using JPEC
```

## Architecture

### Module Structure

JPEC consists of four main modules, each organized in `src/`:

1. **Splines** (`src/Splines/`) - Numerical interpolation library
   - `CubicSpline.jl` - 1D cubic spline interpolation
   - `BicubicSpline.jl` - 2D bicubic spline interpolation
   - `FourierSpline.jl` - Fourier-based spline interpolation
   - Supports both pure Julia and Fortran implementations (via `fortran/` subdirectory)

2. **Equilibrium** (`src/Equilibrium/`) - MHD equilibrium solvers
   - Main entry point: `setup_equilibrium(path)` or `setup_equilibrium(config)`
   - Supports multiple equilibrium types: efit, chease, chease2, lar (Large Aspect Ratio), sol (Solovev)
   - `EquilibriumTypes.jl` - Core data structures (`PlasmaEquilibrium`, `EquilibriumConfig`, etc.)
   - `ReadEquilibrium.jl` - Parsing equilibrium files
   - `DirectEquilibrium.jl` - Direct equilibrium solver
   - `InverseEquilibrium.jl` - Inverse equilibrium solver
   - `AnalyticEquilibrium.jl` - Analytical equilibrium solutions

3. **Vacuum** (`src/Vacuum/`) - Vacuum field calculations
   - Hybrid Julia/Fortran implementation actively being converted to pure Julia
   - Main function: `mscvac()` (Fortran) and `compute_vacuum_response()` (Julia)
   - Fortran interface via ccall to `libvac` shared library
   - `Vacuum_data.jl` - Data structures
   - `Vacuum_init.jl` - Initialization routines
   - `Vacuum_vac.jl` - Core vacuum calculation (Julia conversion in progress)
   - `Vacuum_math.jl` - Mathematical utilities
   - `fortran/` - Legacy Fortran code compiled to shared library

4. **DCON** (`src/DCON/`) - Stability analysis
   - `DconStructs.jl` - Core data structures
   - `Main.jl` - Main entry points
   - `Ode.jl` - ODE solver for stability equations
   - `Sing.jl` - Singular point handling
   - `Fourfit.jl` - Fourier fitting routines
   - `FixedBoundaryStability.jl` - Fixed boundary stability analysis
   - `Free.jl` - Free boundary stability
   - `Mercier.jl` - Mercier stability criterion

### Fortran Integration

The build system compiles Fortran code into shared libraries:

- Build configuration in `deps/build.jl` and `deps/build_helpers.jl`
- OS-specific compiler flags (macOS uses Accelerate framework, Linux uses OpenBLAS)
- Current Fortran modules: Splines, Vacuum
- Platform support: macOS and Linux only (Windows unsupported, use WSL)
- Add new Fortran builds by creating `build_*_fortran()` functions in `deps/build_helpers.jl`

### Data Flow

1. **Equilibrium Setup**: `setup_equilibrium()` reads equilibrium files (TOML config) → parses equilibrium data → runs solver (direct/inverse) → computes global parameters (q-profile, beta, etc.) → diagnoses Grad-Shafranov solution
2. **Vacuum Calculations**: Initialize plasma/wall surfaces → compute vacuum response matrix → return wv, grri, xzpts arrays
3. **Stability Analysis**: Uses equilibrium data and vacuum response to analyze MHD stability

### Key Data Structures

- `PlasmaEquilibrium` - Main equilibrium container with bicubic splines (rzphi), 1D profiles (sq), and parameters
- `EquilibriumConfig` - Configuration loaded from TOML files
- `VacuumInput` - Input parameters for vacuum calculations
- `WallShapeSettings` - Wall geometry configuration

## Git Workflow

This project uses GitFlow:
- Two permanent branches: `main` and `develop`
- `main` branch updated only at release-ready stages
- Feature branches off `develop`, merged back with `--no-ff`
- Current work is on `vacuum_julia` branch

### Commit Message Format

```
CODE - TAG - Detailed message
```

Where:
- CODE: Module name (EQUIL, DCON, VAC, VACUUM, etc.)
- TAG: Type descriptor (WIP, MINOR, IMPROVEMENT, BUG FIX, NEW FEATURE, etc.)

Examples:
- `VAC - WIP - Converting vaccal wall code to Julia`
- `EQUIL - BUG FIX - Fixed separatrix finding for high kappa`
- `DCON - NEW FEATURE - Added Mercier criterion calculation`

This format is used for compiling release notes, so tags should be human-readable and descriptive.

## Important Notes

- Julia 1.11 is the target version
- Tests include both Fortran and Julia implementations to ensure parity during conversion
- The Vacuum module is actively being converted from Fortran to Julia (see `vacuum_julia` branch)
- When modifying equilibrium code, remember to update diagnostic outputs (gsec.h5, gse.h5, gsei.h5)
- The codebase uses 0-based indexing in many places to match Fortran conventions, then converts to 1-based Julia indexing
- Pre-commit hooks are configured for notebook cleaning and Julia formatting (see `docs/src/set_up.md` for developer setup)
