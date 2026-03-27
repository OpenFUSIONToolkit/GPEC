# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

GPEC (Generalized Perturbed Equilibrium Code, Julia implementation) is a comprehensive Julia reimplementation of the GPEC suite for MHD analysis of fusion plasmas. The code performs equilibrium reconstruction, ideal MHD stability analysis, and perturbed equilibrium calculations including plasma response and singular surface coupling diagnostics.

**Relationship to Fortran GPEC**: This Julia GPEC is an evolution of the Fortran GPEC code suite, available at https://github.com/PrincetonUniversity/GPEC. When users reference "the Fortran code", "the original GPEC", or "Fortran GPEC", they are referring to that Fortran codebase. This Julia implementation reimplements and extends GPEC's functionality with improved performance and maintainability.

**Local GPEC Repository**: For code conversion or comparison with the original Fortran implementation, check for a local GPEC repository at `~/Code/gpec`. If not found at this location, ask the user for the correct path.

This codebase is implemented in Julia. References to “Fortran GPEC” or “legacy VACUUM” refer to the original upstream Fortran codebase, not a runtime dependency of this Julia implementation.

**Current Development Focus**: The `perturbed_equilibrium` branch is implementing full GPEC-style perturbed equilibrium functionality, including singular coupling analysis, island formation diagnostics, and mode-space field reconstruction.

## Key References

**IMPORTANT**: The papers in `docs/resources/` provide the theoretical foundation for GPEC's algorithms and should be referenced to understand what the code is doing. **Citing equations from these papers in code comments and annotations is strongly encouraged** to maintain traceability between theory and implementation.

### Vacuum Module

The Vacuum module implements the methods described in:

- **Chance et al. (1997)**: "Vacuum calculations in azimuthally symmetric geometry"
  - Location: `docs/resources/1997-Chance-Vacuum_calculations_in_azimuthally_symmetric_geometry.pdf`
  - Published: Physics of Plasmas **4**, 2161 (1997)
  - Link: https://pubs.aip.org/aip/pop/article-abstract/4/6/2161/263192
  - Describes: Fundamental vacuum response calculation method for tokamak geometry

- **Chance et al. (2007)**: "Calculation of the vacuum Green's function valid even for high toroidal mode numbers in tokamaks"
  - Location: `docs/resources/2007-Chance-Calculation of the vacuum Greens function valid even for high toroidal mode numbers in tokamaks.pdf`
  - Published: Physics of Plasmas **14**, 052506 (2007)
  - Describes: Improved Green's function calculation for high-n modes

### ForceFreeStates Module

The ForceFreeStates module (ideal MHD stability analysis) implements methods from:

- **Glasser (2016)**: "The direct criterion of Newcomb for the ideal MHD stability of an axisymmetric toroidal plasma"
  - Location: `docs/resources/2016-Glasser-The_direct_criterion_of_Newcomb_for_the_ideal_MHD_stability_of_an_axisymmetric_toroidal_plasma.pdf`
  - Published: Physics of Plasmas **23**, 112506 (2016)
  - **Describes: FUNDAMENTAL PAPER - Newcomb's criterion for ideal MHD stability (current implementation)**

- **Glasser (2018)**: "A Riccati solution for the ideal MHD plasma response with applications to real-time stability control"
  - Location: `docs/resources/2018-Glasser-A Riccati solution for the ideal MHD plasma response with applications to real-time stability control.pdf`
  - Published: Physics of Plasmas **25**, 032507 (2018)
  - Describes: Riccati method for ideal MHD eigenvalue problem

### PerturbedEquilibrium Module

The PerturbedEquilibrium module implements GPEC-style perturbed equilibrium calculations from:

- **Park et al. (2007a)**: "Computation of three-dimensional tokamak and spherical torus equilibria"
  - Location: `docs/resources/2007-Park-Computation_of_three-dimensional_tokamak_and_spherical_torus_equilibria-compressed.pdf`
  - Published: Physics of Plasmas **14**, 052110 (2007)
  - Describes: 3D equilibrium perturbations in toroidal geometry

- **Park et al. (2007b)**: "Control of Asymmetric Magnetic Perturbations in Tokamaks"
  - Location: `docs/resources/2007-Park-Control_of_Asymmetric_Magnetic_Perturbations_in_Tokamaks.pdf`
  - Published: Physical Review Letters **99**, 195003 (2007)
  - Describes: Plasma response to resonant magnetic perturbations (RMP)

- **Park et al. (2009)**: "Importance of plasma response to nonaxisymmetric perturbations in tokamaks"
  - Location: `docs/resources/2009-Park-Importance_of_plasma_response_to_nonaxisymmetric_perturbations_in_tokamaks-compressed.pdf`
  - Published: Physics of Plasmas **16**, 056115 (2009)
  - Describes: Self-consistent plasma response calculation

- **Park et al. (2011)**: "Kinetic energy principle and neoclassical toroidal torque in tokamaks"
  - Location: `docs/resources/2011-Park-Physics_of_Plasmas_Kinetic_energy_principle_and_neoclassical_toroidal_torque_in_tokamaks.pdf`
  - Published: Physics of Plasmas **18**, 110702 (2011)
  - Describes: Energy principle for perturbed equilibria

- **Park et al. (2017)**: "Self-consistent perturbed equilibrium with neoclassical toroidal torque in tokamaks"
  - Location: `docs/resources/2017-Park-Self_consistent_perturbed_equilibrium_with_neoclassical_toroidal_torque_in_toka.pdf`
  - Published: Physics of Plasmas **24**, 032505 (2017)
  - Describes: Self-consistent coupling with neoclassical effects

### Resistive MHD Stability Analysis (Future Work)

GPEC will eventually implement resistive MHD stability analysis based on:

- **Glasser (2016)**: "Computation of resistive instabilities by matched asymptotic expansions"
  - Location: `docs/resources/2016-Glasser-Computation_of_resistive_instabilities_by_matched_asymptotic_expansions-compressed.pdf`
  - Published: Physics of Plasmas **23**, 072505 (2016)
  - Describes: Resistive stability analysis and Δ' calculation via matched asymptotic expansions

- **Glasser (2018)**: "A robust solution for the resistive MHD toroidal Δ′ matrix in near real-time"
  - Location: `docs/resources/2018-Glasser-A robust solution for the resistive MHD toroidal Delta-prime matrix in near real-time.pdf`
  - Published: Physics of Plasmas **25**, 032501 (2018)
  - Describes: Fast computation of Δ' matrix for resistive stability

- **Wang et al. (2020)**: "Modeling of resistive plasma response in toroidal geometry using an asymptotic matching approach"
  - Location: `docs/resources/2020-Wang-Modeling of resistive plasma response in toroidal geometry using an asymptotic matching approach.pdf`
  - Published: Physics of Plasmas **27**, 122509 (2020)
  - Describes: Asymptotic matching for resistive plasma response

### PENTRC Module (Future Work)

GPEC will eventually port the PENTRC (Perturbed Equilibrium Neoclassical Toroidal viscosity in Realistic geometry Code) functionality from the Fortran GPEC suite. This is described in:

- **Logan & Park (2013)**: "Neoclassical toroidal viscosity in perturbed equilibria with general tokamak geometry"
  - Location: `docs/resources/2013-Logan-Neoclassical_toroidal_viscosity_in_perturbed_equilibria_with_general_tokamak_geometry.pdf`
  - Published: Physics of Plasmas **20**, 122507 (2013)
  - Describes: Neoclassical toroidal viscosity (NTV) in perturbed equilibria

- **Logan (2015)**: "Electromagnetic Torque in Tokamaks with Toroidal Asymmetries"
  - Location: `docs/resources/2015-Logan-Electromagnetic_Torque_in_Tokamaks_with_Toroidal_Asymmetries-compressed.pdf`
  - Published: PhD Thesis, Princeton University (2015)
  - Describes: Complete PENTRC theory and implementation

### Additional References

- **Various (2009)**: "Nonambipolar Transport by Trapped Particles in Tokamaks"
  - Location: `docs/resources/2009-Nonambipolar_Transport_by_Trapped_Particles_in_Tokamaks.pdf`
  - Describes: Neoclassical transport theory

## Common Commands

### Building and Testing

```bash
# Run all tests
julia --project=. -e 'using Pkg; Pkg.activate("."); Pkg.instantiate(); include("test/runtests.jl")'

# Run specific test file
julia --project=. test/runtests.jl test/runtests_solovev.jl

# Available test files:

# - test/runtests_vacuum_julia.jl   # Julia vacuum module
# - test/runtests_solovev.jl        # Analytical equilibrium
# - test/runtests_ode.jl            # ODE integration
# - test/runtests_sing.jl           # Singular surface handling
# - test/runtests_fullruns.jl       # End-to-end tests
```

### Building Documentation

```bash
# Build documentation locally
julia --project=. build_docs_local.jl

# Documentation hosted at: https://openfusiontoolkit.github.io/GPEC/dev/
```

### Development with Revise

For faster recompilation during development, use Revise.jl (installed in global environment, not in Project.toml):

```julia
using Revise
using GeneralizedPerturbedEquilibrium
```

### Benchmarking

Use the generic benchmarking tool at `benchmarks/benchmark_git_branches.jl` to compare performance between branches or commits.

**Tool usage:**

```bash
# Compare feature branch against develop
julia benchmarks/benchmark_git_branches.jl \
  --example examples/DIIID-like_ideal_example \
  --branch1 develop \
  --branch2 feature-branch

# Compare specific commits
julia benchmarks/benchmark_git_branches.jl \
  --example examples/DIIID-like_ideal_example \
  --commit1 abc123 \
  --commit2 def456

# Compare current develop vs develop from 1 month ago
julia benchmarks/benchmark_git_branches.jl \
  --example examples/DIIID-like_ideal_example \
  --branch1 develop \
  --commit1 HEAD~10 \
  --branch2 develop
```

**Default benchmark case:** `examples/DIIID-like_ideal_example`

**Reported metrics:**
1. **Eigenmode energy (`et[1]`)** - First eigenvalue; verifies calculation correctness
2. **Integration steps** - Total ODE solver steps
3. **Runtime (warmed)** - Wall-clock time averaged over multiple warm runs (JIT warmup handled automatically)
4. **Commit hash** - Git commit of code tested

**The tool automatically:**
- Handles JIT warmup (runs example 3 times, averages last 2)
- Switches between branches/commits
- Stashes uncommitted changes if necessary
- Restores original branch when done
- Reports comparison with percentage differences

**Important notes:**
- Working directory should be clean or changes will be stashed during branch switching
- Tool requires HDF5.jl for reading euler.h5 output
- Each benchmark run takes several minutes per branch (includes compilation + warm runs)

**Benchmark script conventions:**
- Benchmark scripts must reference input data from `examples/` (e.g., `joinpath(@__DIR__, "..", "examples", "DIIID-like_ideal_example")`). Never duplicate example inputs into `benchmarks/`.
- If a benchmark needs modified TOML settings or a parameter scan, copy inputs to a temporary local directory at runtime — do not commit these copies.
- All outputs (figures, CSVs, HDF5 files) must be saved into `benchmarks/` itself (or a self-described subdirectory within it, e.g., `benchmarks/coil_scan_results/`). Output files are not committed.


## Architecture

### Computational Workflow

GPEC follows a three-stage analysis pipeline:

1. **Equilibrium** → Solve Grad-Shafranov equation, compute flux surfaces, safety factor q-profile
2. **Stability Analysis** → Solve ideal MHD eigenvalue problem (DCON-style), identify singular surfaces
3. **Perturbed Equilibrium** → Compute plasma response to external fields, analyze singular coupling and island formation

This workflow is reflected in the modular structure and data flow.

### Module Structure

GPEC consists of **seven main modules** organized in `src/`:

#### Foundation Modules

1. **Splines** (`src/Splines/`) - Numerical interpolation library
   - `CubicSpline.jl` - 1D cubic spline interpolation
   - `BicubicSpline.jl` - 2D bicubic spline interpolation
   - `FourierSpline.jl` - Fourier-based spline interpolation
   - Status: Mature, pure Julia implementation

2. **Utilities** (`src/Utilities/`) - Shared computational tools
   - `FourierTransforms.jl` - Efficient Fourier transform utilities with pre-computed basis functions
   - Provides type-stable functor pattern for repeated transforms
   - Used by Vacuum and PerturbedEquilibrium modules

#### Core Physics Modules

3. **Equilibrium** (`src/Equilibrium/`) - MHD equilibrium solvers
   - Main entry point: `setup_equilibrium(path)` or `setup_equilibrium(config)`
   - Supports multiple equilibrium types:
     - `efit` - EFIT g-file format
     - `chease`, `chease2` - CHEASE equilibrium code formats
     - `lar` - Large Aspect Ratio analytical model
     - `sol` - Solovev analytical equilibrium
   - Key files:
     - `EquilibriumTypes.jl` - Core data structures
     - `ReadEquilibrium.jl` - Parsing equilibrium files
     - `DirectEquilibrium.jl` - Direct Grad-Shafranov solver
     - `InverseEquilibrium.jl` - Inverse equilibrium solver
     - `AnalyticEquilibrium.jl` - Analytical solutions
   - Status: Stable and feature-complete

4. **Vacuum** (`src/Vacuum/`) - Vacuum field calculations and Green's functions
   - Computes vacuum response matrices for ideal MHD analysis
   - Calculates both **interior** (grri) and **exterior** (grre) Green's functions
   - Main functions:
     - `compute_vacuum_response()` - Pure Julia implementation
   - Key files:
     - `VacuumStructs.jl` - Data structures
     - `VacuumInternals.jl` - Core algorithms
     - `VacuumFromEquilibrium.jl` - Integration with equilibrium data
   - Status: **Pure Julia implementation complete and available**

5. **ForceFreeStates** (`src/ForceFreeStates/`) - Ideal MHD stability analysis (DCON-style)
   - Solves ideal MHD eigenvalue problem with force-free boundary conditions
   - Identifies singular surfaces where ξ·∇ψ = 0
   - Key files:
     - `ForceFreeStatesStructs.jl` - Core data structures
     - `Ode.jl` - ODE solver for Euler-Lagrange equations
     - `Sing.jl` - Singular point handling and layer analysis
     - `Fourfit.jl` - Fourier fitting routines
     - `FixedBoundaryStability.jl` - Fixed boundary analysis
     - `Free.jl` - Free boundary stability
     - `Mercier.jl` - Mercier stability criterion
   - Status: Stable, core DCON functionality implemented

#### Perturbed Equilibrium Modules

6. **ForcingTerms** (`src/ForcingTerms/`) - External field specification
   - Handles external magnetic field perturbations (coils, RMP, etc.)
   - Supports ASCII and HDF5 forcing data formats
   - `ForcingMode` data structure specifies amplitude and phase for each (m,n) component
   - Status: Complete and functional

7. **PerturbedEquilibrium** (`src/PerturbedEquilibrium/`) - **GPEC-style plasma response**
   - Computes plasma response to external forcing
   - Calculates singular coupling metrics at rational surfaces
   - Key files:
     - `PerturbedEquilibrium.jl` - Main entry point
     - `PerturbedEquilibriumStructs.jl` - Data structures
     - `ResponseMatrices.jl` - Permeability matrix calculation
     - `FieldReconstruction.jl` - Mode-space field reconstruction
     - `Response.jl` - Plasma response computation
     - `SingularCoupling.jl` - **Singular surface analysis** including:
       - Delta prime (Δ') tearing stability parameter
       - Resonant flux and currents at rational surfaces
       - Island half-widths and Chirikov parameters
       - Green's functions at interior flux surfaces
       - Surface inductance for singular surfaces
     - `Utils.jl` - Helper functions
   - Status: **Active development on `perturbed_equilibrium` branch**

### Configuration

**Unified Configuration File**: `gpec.toml`

All GPEC modules are configured via a single TOML file with the following sections:

- `[Equilibrium]` - Equilibrium solver settings
- `[Wall]` - Wall geometry and vacuum region
- `[ForceFreeStates]` - Stability analysis parameters
- `[PerturbedEquilibrium]` - Perturbed equilibrium settings
- `[ForcingTerms]` - External field specification

Key parameters:
- `force_termination` - Set to `true` to exit after equilibrium/stability (skip perturbed equilibrium)
- `output_file` - Output filename (default: `gpec.h5`)

Example configuration files are provided in:
- `examples/Solovev_ideal_example/gpec.toml`
- `examples/DIIID-like_ideal_example/gpec.toml`

**Note**: Legacy configuration files (`equil.toml`, `vac.in`) are deprecated.

### Data Flow

The complete GPEC analysis pipeline:

1. **Equilibrium Setup**:
   - `setup_equilibrium(config)` reads configuration from `gpec.toml`
   - Parses equilibrium data (EFIT, CHEASE, or analytical)
   - Runs Grad-Shafranov solver (direct or inverse)
   - Computes global parameters: q-profile, pressure, current density, β
   - Creates bicubic splines for (ψ, θ, φ) → (R, Z, Φ) mapping
   - Outputs: `PlasmaEquilibrium` object

2. **Vacuum Response**:
   - Initialize plasma and wall surfaces from equilibrium
   - Compute vacuum response matrices (wv, grri, grre)
   - Calculate both interior and exterior Green's functions
   - Pure Julia implementation

3. **Stability Analysis** (ForceFreeStates):
   - Solve ideal MHD Euler-Lagrange equations via ODE integration
   - Identify singular surfaces where q = m/n
   - Compute Δ' at each singular surface
   - Calculate potential and kinetic energies
   - Check Mercier and ballooning stability criteria
   - Outputs: Eigenmode structure ξ(ψ,θ)

4. **Perturbed Equilibrium** (GPEC-style):
   - Load external forcing data (coil fields, RMP configuration)
   - Compute plasma response using permeability matrices
   - Reconstruct mode-space fields (ξ_modes, b_modes)
   - Calculate singular coupling metrics at rational surfaces:
     - Δ' (tearing stability parameter)
     - Island half-widths
     - Chirikov overlap parameter
     - Resonant flux and currents
   - Outputs: `PerturbedEquilibriumState` with response fields and diagnostics

5. **Output**:
   - All results saved to single HDF5 file (default: `gpec.h5`)
   - HDF5 groups: `input/`, `info/`, `equil/`, `splines/`, `locstab/`, `integration/`, `singular/`, `vacuum/`, and perturbed equilibrium data

### Key Data Structures

#### Equilibrium
- `PlasmaEquilibrium` - Main equilibrium container with bicubic splines (rzphi), 1D profiles (sq), and global parameters
- `EquilibriumConfig` - Configuration loaded from TOML files

#### Vacuum
- `VacuumInput` - Input parameters for vacuum calculations
- `WallShapeSettings` - Wall geometry configuration

#### Stability
- `SingType` - Singular surface data including:
  - Rational surface location (ψ, q = m/n)
  - Δ' (tearing stability parameter)
  - Eigenmode structure at singular surface
  - **NEW**: Green's functions (grri, grre) at interior singular surfaces
  - Surface inductance

#### Perturbed Equilibrium
- `PerturbedEquilibriumControl` - User-facing TOML configuration parameters
- `PerturbedEquilibriumInternal` - Internal state with mode arrays
- `PerturbedEquilibriumState` - Results including:
  - Response fields (ξ_modes, b_modes) in mode space
  - Singular coupling matrices [msing × numpert_total]
  - Island diagnostics (half-widths, Chirikov parameters)
- `ForcingMode` - External forcing specification (m, n, amplitude, phase)

### Module Dependencies

```
GeneralizedPerturbedEquilibrium
├── Splines (foundation)
├── Utilities (shared tools)
│   └── FourierTransforms
├── Equilibrium (uses Splines)
├── Vacuum (uses Splines, Equilibrium, Utilities)
├── ForcingTerms (data I/O)
├── ForceFreeStates (uses Equilibrium, Vacuum, Splines)
└── PerturbedEquilibrium (uses ForceFreeStates, Vacuum, ForcingTerms, Utilities)
```

## Git Workflow

This project uses GitFlow (http://nvie.com/posts/a-successful-git-branching-model):

- Two permanent branches: `main` and `develop`
- `main` is updated only at release-ready stages via pull request from `develop`
- `develop` is the integration branch — all feature branches merge here

**IMPORTANT**: All development must be done on feature branches. No commits should be made directly to `develop` or `main`. Always create a branch from `develop`, do all work there, and open a pull request back into `develop`.

### Branch Naming

Branches use a typed prefix and a lowercase hyphen-separated description:

| Prefix | Purpose | Branches from | Merges into |
|---|---|---|---|
| `feature/` | New functionality | `develop` | `develop` |
| `bugfix/` | Non-critical bug fixes | `develop` | `develop` |
| `hotfix/` | Critical production fix | `main` | `main` + `develop` |
| `performance/` | Performance improvements | `develop` | `develop` |
| `refactor/` | Refactoring without behavior change | `develop` | `develop` |
| `docs/` | Documentation only | `develop` | `develop` |
| `test/` | Test additions/improvements | `develop` | `develop` |
| `experiment/` | Exploratory work, may not merge | `develop` | — |

Examples: `bugfix/sing-lim-bounds-error`, `feature/kinetic-damping`, `performance/green-function-prefactor`

Author-named branches (e.g. `jmh/`, `nlogan/`) are not used — git history already records authorship on every commit.

### Hotfix Workflow

Hotfixes address critical bugs in production (`main`) that cannot wait for the next release cycle:

1. Branch `hotfix/description` from the current tagged `main` commit
2. Fix the bug with one or more commits
3. Merge into `main` via pull request; tag the merge commit with a new patch version (e.g. `v0.1.1`)
4. Merge the same branch into `develop` so the fix is not lost in the next release

### Versioning

This project uses semantic versioning: `v{major}.{minor}.{patch}`

- **major**: breaking API or file-format changes
- **minor**: new features, backward-compatible
- **patch**: bug fixes (typically via hotfix branches)

Tags are applied to merge commits on `main`.

**Current Development**:
- Active branch: `perturbed_equilibrium` - Major feature implementing GPEC-style perturbed equilibrium calculations
- Previous work: `vacuum_julia` branch (merged) - Pure Julia vacuum implementation

### Commit Message Format

```
CODE - TAG - Detailed message
```

Where:
- **CODE**: Module name (EQUIL, VAC, VACUUM, ForceFreeStates, PERTURBED EQUILIBRIUM, etc.)
- **TAG**: Type descriptor (WIP, MINOR, IMPROVEMENT, BUG FIX, NEW FEATURE, REFACTOR, CLEANUP, etc.)

Examples:
- `PERTURBED EQUILIBRIUM - NEW FEATURE - Implement singular coupling diagnostics`
- `VAC - IMPROVEMENT - Add dual Green's function computation`
- `EQUIL - BUG FIX - Fixed separatrix finding for high kappa`
- `ForceFreeStates - REFACTOR - Unified singular surface data structure`

This format is used for compiling release notes, so tags should be human-readable and descriptive.

## Important Notes

### General
- **Julia version**: 1.11 is the target version
- **Indexing**: The codebase uses 0-based indexing in many places to match Fortran conventions, then converts to 1-based Julia indexing
- **No step numbering in code comments** - Avoid annotations like "Step 1: do this" followed by "Step 2: do that". These get out of sync as code changes. Just describe the action without numbering.
- **Keep code comments concise** - A comment should be one line where possible. Do not write multi-line block comments explaining the current session's investigation, what was tried, what was wrong before, or why a specific file/path behaves differently. State what the code does and why at a general level. Example of too much detail: a 6-line block explaining that efit_by_inversion uses psilow>0 while CHEASE starts at 0, that the old code was removed, and that spline spikes result. Preferred: `# Replicate Fortran inverse.f: overwrite deta at axis (r²=0) by extrapolating from innermost surfaces.`

### Output Files
- **Default output**: `gpec.h5` (previously `euler.h5` in older versions)
- **Legacy diagnostic files**: When modifying equilibrium code, remember that older versions output `gsec.h5`, `gse.h5`, `gsei.h5` - these are now consolidated into `gpec.h5`

### Current Development Priorities
- **Perturbed equilibrium module**: Active development of GPEC-style singular coupling analysis
- **Configuration**: All settings now in unified `gpec.toml` file

### Plotting
- **Spectrum plots** (any plot with discrete mode numbers m or n on the x-axis) must use `seriestype=:steppre` with a `step_series` helper that pads zeros on both ends. Pattern from `benchmarks/benchmark_coil_ForcingTerms_against_fortran.jl`:
  ```julia
  function step_series(m_vals, amps)
      m_ext   = [m_vals[1] - 1; m_vals; m_vals[end] + 1]
      amp_ext = [0.0; amps; 0.0]
      return m_ext, amp_ext
  end
  # Usage:
  m_ext, a_ext = step_series(m_vals, amplitudes)
  plot!(p, m_ext, a_ext; seriestype=:steppre, lw=2, label="...")
  ```

### Performance
- Pure Julia implementations are available for all major components and offer comparable or better performance than Fortran
- Benchmarks available in `benchmark/` directory for Fourier transforms and vacuum calculations
- Pre-commit hooks are configured for notebook cleaning and Julia formatting (see `docs/src/set_up.md` for developer setup)

## Git Merge conflict resolution policy

- When resolving git conflicts, do not simply accept one side.
- Analyze what each side changed and WHY before producing a resolution.
- Produce a merged version incorporating both sets of changes.
- If both sides renamed the same symbol differently, prefer the current (ours) branch convention.
- When a rename on one side conflicts with a logic change on the other, apply the logic change using the renamed symbol.
- If a conflict involves changes to numerical parameters (tolerances, boundary conditions, grid sizes), flag for human review rather than guessing.
- Flag any conflicts where the combination is ambiguous for human review.
