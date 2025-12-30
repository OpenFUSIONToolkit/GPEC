# PerturbedEquilibrium Implementation Plan

## Overview

This document outlines the skeleton implementation for perturbed equilibrium (GPEC) functionality in JPEC, following the patterns established by DCON and Vacuum modules.

## Design Principles

1. **Follow DCON/Vacuum patterns**: Struct-based design with clear separation of control parameters, internal state, and results
2. **Explicit arguments**: Prefer passing specific parameters rather than large structs (except where it makes sense)
3. **TOML configuration**: Consolidated jpec.toml with `[PerturbedEquilibrium]` section
4. **Module organization**: Code in `src/PerturbedEquilibrium/`

## Directory Structure

```
src/PerturbedEquilibrium/
├── PerturbedEquilibrium.jl              # Main module file
├── PerturbedEquilibriumStructs.jl       # Data structure definitions
├── Response.jl                          # Plasma response calculations
├── SingularCoupling.jl                  # Singular coupling metrics (key result)
├── ModeOutput.jl                        # Mode field output functionality
└── Utils.jl                             # Helper functions
```

## Data Structures

### PerturbedEquilibriumControl

User-facing parameters from TOML `[PerturbedEquilibrium]` section:

```julia
@kwdef mutable struct PerturbedEquilibriumControl
    # High Priority (MWE)
    forcing_data_file::String = "forcing.h5"      # Path to forcing data (m, b_x)
    forcing_data_format::String = "hdf5"          # "hdf5" or "ascii"
    fixed_boundary::Bool = false                  # Fixed boundary flag
    output_eigenmodes::Bool = true                # Output mode fields
    compute_response::Bool = true                 # Always true for MWE
    compute_singular_coupling::Bool = true        # Key metric calculation
    verbose::Bool = true                          # Logging

    # Medium Priority (include but can be simple)
    filter_modes::Bool = false                    # Mode filtering (defer for MWE)
    singular_point_method::String = "standard"    # Hardcode for MWE

    # Output settings
    output_filename::String = "perturbed_eq.h5"   # Output file
    write_outputs_to_HDF5::Bool = true            # HDF5 output flag
end
```

### PerturbedEquilibriumInternal

Runtime/derived quantities:

```julia
@kwdef mutable struct PerturbedEquilibriumInternal
    dir_path::String = "./"                       # Working directory

    # Forcing data (loaded from file)
    forcing_modes::Vector{Int} = Int[]            # Poloidal mode numbers
    forcing_amplitudes::Vector{ComplexF64} = ComplexF64[]  # Complex amplitudes

    # Response quantities
    plasma_response::Matrix{ComplexF64} = zeros(ComplexF64, 0, 0)

    # Coupling metrics
    singular_coupling_metrics::Dict{String,Float64} = Dict{String,Float64}()
end
```

### PerturbedEquilibriumState

Calculation results:

```julia
@kwdef mutable struct PerturbedEquilibriumState
    # Response fields
    xi_perturbed::Array{ComplexF64,3} = zeros(ComplexF64, 0, 0, 0)  # Displacement field
    b_perturbed::Array{ComplexF64,3} = zeros(ComplexF64, 0, 0, 0)   # Magnetic field

    # Singular coupling results
    coupling_coefficient::ComplexF64 = 0.0 + 0.0im
    resonant_amplitude::Float64 = 0.0

    # Energies (if needed)
    plasma_energy::Float64 = 0.0
    vacuum_energy::Float64 = 0.0
    total_energy::Float64 = 0.0
end
```

## Configuration: jpec.toml

Rename `dcon.toml` → `jpec.toml` and add new section:

```toml
[DCON_CONTROL]
# ... existing DCON parameters ...

[WALL]
# ... existing wall parameters ...

[PerturbedEquilibrium]
forcing_data_file = "forcing.h5"
forcing_data_format = "hdf5"
fixed_boundary = false
output_eigenmodes = true
compute_response = true
compute_singular_coupling = true
verbose = true
output_filename = "perturbed_eq.h5"
write_outputs_to_HDF5 = true
```

## Skeleton Functions (MWE)

### Main Entry Point

```julia
function compute_perturbed_equilibrium(
    equil::Equilibrium.PlasmaEquilibrium,
    dcon_results::OdeState,
    ctrl::PerturbedEquilibriumControl,
    intr::PerturbedEquilibriumInternal
)::PerturbedEquilibriumState

    if ctrl.verbose
        println("PERTURBED EQUILIBRIUM START")
        println("----------------------------------")
    end

    state = PerturbedEquilibriumState()

    # Load forcing data
    load_forcing_data!(intr, ctrl)

    # Compute plasma response
    if ctrl.compute_response
        compute_plasma_response!(state, equil, dcon_results, intr, ctrl)
    end

    # Compute singular coupling metrics
    if ctrl.compute_singular_coupling
        compute_singular_coupling_metrics!(state, equil, dcon_results, intr, ctrl)
    end

    # Output eigenmodes
    if ctrl.output_eigenmodes
        output_eigenmode_fields(state, equil, ctrl)
    end

    if ctrl.verbose
        println("----------------------------------")
        println("PERTURBED EQUILIBRIUM COMPLETE")
    end

    return state
end
```

### Response.jl - Core Calculations

```julia
"""Load forcing data from HDF5 or ASCII file"""
function load_forcing_data!(
    intr::PerturbedEquilibriumInternal,
    ctrl::PerturbedEquilibriumControl
)
    # TODO: Implement HDF5/ASCII reader for (m, b_x) data
    # For MWE: simple HDF5 read of mode numbers and amplitudes
    if ctrl.verbose
        println("Loading forcing data from $(ctrl.forcing_data_file)")
    end
end

"""Compute plasma response to forcing"""
function compute_plasma_response!(
    state::PerturbedEquilibriumState,
    equil::Equilibrium.PlasmaEquilibrium,
    dcon_results::OdeState,
    intr::PerturbedEquilibriumInternal,
    ctrl::PerturbedEquilibriumControl
)
    # TODO: Convert from Fortran perturbed equilibrium calculation
    # Use DCON eigenmode solutions and forcing to compute response
    if ctrl.verbose
        println("Computing plasma response")
    end
end
```

### SingularCoupling.jl - Key Metrics

```julia
"""Compute singular layer coupling metrics"""
function compute_singular_coupling_metrics!(
    state::PerturbedEquilibriumState,
    equil::Equilibrium.PlasmaEquilibrium,
    dcon_results::OdeState,
    intr::PerturbedEquilibriumInternal,
    ctrl::PerturbedEquilibriumControl
)
    # TODO: Implement singular layer coupling calculation
    # This is the key scientific result
    if ctrl.verbose
        println("Computing singular coupling metrics")
    end
end
```

### ModeOutput.jl - Visualization

```julia
"""Output eigenmode fields as magnetic field components"""
function output_eigenmode_fields(
    state::PerturbedEquilibriumState,
    equil::Equilibrium.PlasmaEquilibrium,
    ctrl::PerturbedEquilibriumControl
)
    # TODO: Write mode fields to HDF5
    # Convert displacement to b-fields for visualization
    if ctrl.verbose
        println("Writing eigenmode fields to $(ctrl.output_filename)")
    end
end
```

## Integration with JPEC.main()

Modify `src/JPEC.jl`:

```julia
function main(args::Vector{String}=String[])
    # ... existing DCON code ...

    # Read jpec.toml instead of dcon.toml
    inputs = TOML.parsefile(joinpath(intr.dir_path, "jpec.toml"))

    # ... DCON calculations ...

    # Check for PerturbedEquilibrium section
    if "PerturbedEquilibrium" in keys(inputs)
        if ctrl.verbose
            println("\nRunning perturbed equilibrium calculations")
        end

        pe_ctrl = PerturbedEquilibriumControl(;
            (Symbol(k) => v for (k, v) in inputs["PerturbedEquilibrium"])...
        )
        pe_intr = PerturbedEquilibriumInternal(; dir_path=intr.dir_path)

        pe_state = PerturbedEquilibrium.compute_perturbed_equilibrium(
            equil, odet, pe_ctrl, pe_intr
        )

        if pe_ctrl.write_outputs_to_HDF5
            PerturbedEquilibrium.write_outputs_to_HDF5(pe_state, pe_ctrl, equil)
        end
    end

    # ... rest of main ...
end
```

## Migration Path

1. **Phase 1 (Skeleton)**: Create module structure, structs, and stub functions
2. **Phase 2 (I/O)**: Implement forcing data reader and output writer
3. **Phase 3 (Core)**: Convert Fortran subroutines one-by-one:
   - Response calculation
   - Singular coupling
   - Mode field computation
4. **Phase 4 (Integration)**: Full integration with DCON workflow

## File Renaming Strategy

```bash
# Rename all dcon.toml → jpec.toml
find . -name "dcon.toml" -type f -execdir mv {} jpec.toml \;

# Update code references
# In JPEC.jl and tests: "dcon.toml" → "jpec.toml"
```

## Testing Strategy

Create minimal test case:
```
test/test_data/perturbed_equilibrium_simple/
├── jpec.toml              # With [PerturbedEquilibrium] section
├── equil.toml             # Equilibrium config
└── forcing.h5             # Simple forcing data
```

## Deferred for Later

- Eigenmode file handoff (keep internal)
- Coil flag functionality
- Displacement flag
- Chebyshev methods
- NetCDF/ASCII/binary output variants
- Advanced filtering and mode selection

## Questions for Clarification

1. Should forcing data format support both HDF5 and ASCII initially, or just HDF5?
2. Do you want the perturbed equilibrium to run automatically if DCON runs, or only if explicitly requested?
3. Should we create a combined output file or separate DCON and PerturbedEquilibrium outputs?

---

This plan provides a clear roadmap for implementing the minimum working example while maintaining consistency with the existing JPEC architecture.
