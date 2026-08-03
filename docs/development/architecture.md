# Architecture

## Computational Workflow

GPEC follows a three-stage analysis pipeline:

1. **Equilibrium** → Solve Grad-Shafranov equation, compute flux surfaces, safety factor q-profile
2. **Stability Analysis** → Solve ideal MHD eigenvalue problem (DCON-style), identify singular surfaces
3. **Perturbed Equilibrium** → Compute plasma response to external fields, analyze singular coupling and island formation

This workflow is reflected in the modular structure and data flow.

## Module Structure

GPEC consists of **seven main modules** organized in `src/`:

### Foundation Modules

1. **Splines** (`src/Splines/`) - Numerical interpolation library
   - `CubicSpline.jl` - 1D cubic spline interpolation
   - `BicubicSpline.jl` - 2D bicubic spline interpolation
   - `FourierSpline.jl` - Fourier-based spline interpolation
   - Status: Mature, pure Julia implementation

2. **Utilities** (`src/Utilities/`) - Shared computational tools
   - `FourierTransforms.jl` - Efficient Fourier transform utilities with pre-computed basis functions
   - Provides type-stable functor pattern for repeated transforms
   - Used by Vacuum and PerturbedEquilibrium modules

### Core Physics Modules

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
   - Solves the exterior boundary-integral system for the vacuum energy matrix `wv`, and optionally
     (`compute_Iv=true`) the interior system as well to build the surface-current matrix `I_v`
     (Park 2007 eq. 21b). The interior/exterior Green's functions themselves are internal scratch.
   - Main functions:
     - `compute_vacuum_response()` / `compute_vacuum_response!()` - allocating and in-place entry points
   - Key files:
     - `DataTypes.jl` - Data structures (`VacuumInput`, `PlasmaGeometry`, `WallGeometry`)
     - `Kernel2D.jl` / `Kernel3D.jl` - Single-/double-layer kernel assembly
     - `Field.jl` - Vacuum field and potential evaluation off the surface
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
     - `Ballooning.jl` - Local stability scan: Mercier D_I, resistive interchange D_R, and high-n ballooning Δ' (s–α). Replaces the former standalone `Mercier.jl`.
   - Status: Stable, core DCON functionality implemented

### Perturbed Equilibrium Modules

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
   - Status: Core plasma response and singular coupling calculations implemented; active area of development

## Configuration

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

## Data Flow

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
   - Compute the vacuum energy matrix `wv` (and `I_v` when `compute_Iv=true`)
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

## Key Data Structures

### Equilibrium
- `PlasmaEquilibrium` - Main equilibrium container with bicubic splines (rzphi), 1D profiles (sq), and global parameters
- `EquilibriumConfig` - Configuration loaded from TOML files

### Vacuum
- `VacuumInput` - Input parameters for vacuum calculations
- `WallShapeSettings` - Wall geometry configuration

### Stability
- `SingType` - Singular surface data including:
  - Rational surface location (ψ, ρ, q = m/n, dq/dψ)
  - Δ' (tearing stability parameter) — **stub**; the valid Δ' is `ForceFreeStatesInternal.delta_prime_matrix`
  - Asymptotic solution bases at the inner-layer boundaries

### Perturbed Equilibrium
- `PerturbedEquilibriumControl` - User-facing TOML configuration parameters
- `PerturbedEquilibriumInternal` - Internal state with mode arrays
- `PerturbedEquilibriumState` - Results including:
  - Response fields (ξ_modes, b_modes) in mode space
  - Singular coupling matrices [msing × numpert_total]
  - Island diagnostics (half-widths, Chirikov parameters)
- `ForcingMode` - External forcing specification (m, n, amplitude, phase)

## Module Dependencies

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
