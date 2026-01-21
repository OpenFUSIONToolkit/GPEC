# JPEC Copilot Instructions

JPEC (Julia Perturbed Equilibrium Code) is a hybrid Julia/Fortran MHD equilibrium and stability analysis suite for fusion plasmas. This is an active port of GPEC with legacy Fortran called via `ccall`.

## Architecture Overview

Four main modules in `src/`:

1. **Splines** - Numerical interpolation (1D/2D cubic, Fourier). Hybrid Julia/Fortran implementations.
2. **Equilibrium** - MHD equilibrium solvers. Entry: `setup_equilibrium(path)` or `setup_equilibrium(config)`. Supports efit, chease, chease2, lar (Large Aspect Ratio), sol (Solovev).
3. **Vacuum** - Vacuum field calculations. **Actively being converted from Fortran to Julia**. Main functions: `mscvac()` (Fortran) and `compute_vacuum_response()` (Julia).
4. **DCON** - Stability analysis. Solves ODE for MHD stability, handles singular surfaces. Entry: functions in `src/DCON/Main.jl`.

**BIEST** (`src/BIEST/`) - C++ boundary integral operator for 3D vacuum geometries, called from Julia via wrapper.

## Data Flow

1. **Equilibrium**: `setup_equilibrium()` → parse TOML config → read equilibrium file → run solver (direct/inverse) → compute q-profile, beta → diagnose GS solution → output gsec.h5, gse.h5, gsei.h5
2. **Vacuum**: Initialize plasma/wall surfaces → compute vacuum response matrix → return wv, grri, xzpts arrays
3. **Stability**: Use equilibrium + vacuum response → integrate stability ODEs → compute energies → determine stability

### Key Data Structures

- `PlasmaEquilibrium` - Main equilibrium container with bicubic splines (rzphi), 1D profiles (sq), and parameters
- `EquilibriumConfig` - Configuration from TOML files (sections: `[EQUIL_CONTROL]`, `[EQUIL_OUTPUT]`)
- `VacuumInput` - Vacuum calculation parameters
- `DconControl` - DCON configuration from TOML (sections: `[DCON_CONTROL]`, `[WALL]`)

## Build & Test Commands

```bash
# Build Fortran libraries (required before first use)
julia --project=. -e 'using Pkg; Pkg.build()'

# Run all tests
julia --project=. test/runtests.jl

# Run specific test (add others as arguments)
julia --project=. test/runtests.jl test/runtests_spline.jl

# Build documentation locally
julia --project=. build_docs_local.jl
```

**Available test files**: `runtests_build.jl`, `runtests_spline.jl`, `runtests_vacuum_fortran.jl`, `runtests_vacuum_julia.jl`, `runtests_solovev.jl`, `runtests_ode.jl`, `runtests_sing.jl`, `runtests_fullruns.jl`

## Fortran Integration

Build system in `deps/build.jl` and `deps/build_helpers.jl`:
- Compiles Fortran → shared libraries (`.dylib` on macOS, `.so` on Linux)
- OS-specific flags: macOS uses Accelerate, Linux uses OpenBLAS
- **No Windows support** - use WSL
- Current Fortran modules: Splines (`libspline`), Vacuum (`libvac`), BIEST (`libbiest`)
- Add new builds by creating `build_*_fortran()` in `deps/build_helpers.jl`

### Calling Fortran from Julia

Use `ccall` with mangled names:
```julia
ccall((:function_name_, libname), ReturnType, (ArgTypes...), args...)
# Module functions: (:__module_MOD_function, libname)
```

See examples in `src/Vacuum/Vacuum.jl` (lines 59-93) and `src/Splines/CubicSpline.jl`.

## BIEST C++ Integration

Located in `src/BIEST/` with C++ sources and Julia wrapper:
- Build: `cd src/BIEST && make`
- C++ wrapper: `src/BIEST/wrapper.cpp` exposes functions via `extern "C"`
- Julia interface: `src/BIEST.jl` calls via `ccall((:function, libbiest), ...)`
- To add functions: update `wrapper.cpp`, add to `BIEST.jl`, export, recompile

## Project Conventions

### Commit Message Format

```
CODE - TAG - Detailed message
```

- **CODE**: Module (EQUIL, DCON, VAC, VACUUM, BIEST, etc.)
- **TAG**: WIP, MINOR, IMPROVEMENT, BUG FIX, NEW FEATURE, etc.

Examples:
- `VAC - WIP - Converting vaccal wall code to Julia`
- `EQUIL - BUG FIX - Fixed separatrix finding for high kappa`

### Code Style

- **NO step numbering in comments** - Avoid "Step 1:", "Step 2:" annotations (they get out of sync)
- Many places use **0-based indexing** (Fortran convention) then convert to 1-based Julia indexing
- Document index conventions in comments when converting between Fortran/Julia

### Git Workflow (GitFlow)

- Two permanent branches: `main` and `develop`
- `main` updated only at releases
- Feature branches off `develop`, merge with `--no-ff`
- Current active work: `3D_vacuum` branch (PR #131)

## Configuration Files

TOML-based configuration for equilibrium and stability runs:

**equil.toml** - `[EQUIL_CONTROL]` (solver params, grid) + `[EQUIL_OUTPUT]` (diagnostics)
**dcon.toml** - `[DCON_CONTROL]` (stability params, mode numbers) + `[WALL]` (wall geometry)

Examples in `examples/DIIID-like_ideal_example/` and `examples/Solovev_ideal_example/`

## Development Tips

- Use **Revise.jl** for faster recompilation (install in global env, not Project.toml):
  ```julia
  using Revise
  using JPEC
  ```
- Target Julia version: **1.11**
- When modifying equilibrium code, update diagnostic outputs (gsec.h5, gse.h5, gsei.h5)
- Tests include Fortran and Julia implementations to ensure parity during conversion
- Pre-commit hooks configured for notebook cleaning and Julia formatting (see `docs/src/set_up.md`)

## Benchmarking

**Default benchmark case**: `examples/DIIID-like_ideal_example`
**Reference**: `origin/develop` branch (fetch latest before running)

**Required metrics**:
1. **Least stable eigenmode energy** - First value of `et` array (verifies consistency)
2. **Number of steps** - ODE solver integration steps
3. **Runtime** - Wall-clock execution time

Compare current branch vs `origin/develop` for performance regression testing.

## Important Files

- `src/JPEC.jl` - Main module, includes all submodules
- `deps/build_helpers.jl` - Fortran build configuration
- `src/Equilibrium/EquilibriumTypes.jl` - Core equilibrium data structures
- `src/DCON/DconStructs.jl` - Stability analysis data structures
- `src/Vacuum/VacuumInternals.jl` - Julia vacuum implementation (active development)
- `examples/*/equil.toml`, `examples/*/dcon.toml` - Configuration examples

## Current Development Focus

Active conversion of Vacuum module from Fortran to Julia on `3D_vacuum` branch. When working on vacuum code:
- Maintain parity between `mscvac()` (Fortran) and `compute_vacuum_response()` (Julia)
- Test both implementations with `runtests_vacuum_fortran.jl` and `runtests_vacuum_julia.jl`
- Document indexing conversions between 0-based (Fortran) and 1-based (Julia)
