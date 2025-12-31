"""
    PerturbedEquilibriumControl

User-facing control parameters from TOML [PerturbedEquilibrium] section.

## Fields

Note: Forcing data file settings are now in [ForcingTerms] section.

High Priority (MWE):
  - `fixed_boundary::Bool` - Fixed boundary flag (default: false)
  - `output_eigenmodes::Bool` - Output mode fields as b-fields (default: true)
  - `compute_response::Bool` - Compute plasma response (default: true)
  - `compute_singular_coupling::Bool` - Compute singular coupling metrics (default: true)
  - `verbose::Bool` - Enable verbose logging (default: true)

Output Settings:
  - `output_filename::String` - Combined output file with DCON results (default: uses DCON HDF5_filename)
  - `write_outputs_to_HDF5::Bool` - Write perturbed equilibrium outputs to HDF5 (default: true)

Medium Priority (defer for MWE):
  - `filter_modes::Bool` - Enable mode filtering (default: false)
  - `singular_point_method::String` - Method for singular point treatment (default: "standard")
"""
@kwdef mutable struct PerturbedEquilibriumControl
    # High Priority (MWE)
    fixed_boundary::Bool = false
    output_eigenmodes::Bool = true
    compute_response::Bool = true
    compute_singular_coupling::Bool = true
    verbose::Bool = true

    # Output settings
    output_filename::String = ""  # Empty means use DCON filename
    write_outputs_to_HDF5::Bool = true

    # Medium Priority (include but simple for MWE)
    filter_modes::Bool = false
    singular_point_method::String = "standard"
end

"""
    PerturbedEquilibriumInternal

Internal state variables for perturbed equilibrium calculations.

## Fields

  - `dir_path::String` - Working directory path
  - `forcing_modes::Vector{ForcingMode}` - Loaded forcing mode data
  - `plasma_response::Matrix{ComplexF64}` - Plasma response matrix
  - `singular_coupling_metrics::Dict{String,Float64}` - Coupling metrics at singular surfaces
"""
@kwdef mutable struct PerturbedEquilibriumInternal
    dir_path::String = "./"
    forcing_modes::Vector{ForcingMode} = ForcingMode[]
    plasma_response::Matrix{ComplexF64} = zeros(ComplexF64, 0, 0)
    singular_coupling_metrics::Dict{String,Float64} = Dict{String,Float64}()
end

"""
    PerturbedEquilibriumState

Results from perturbed equilibrium calculations.

## Fields

  - `xi_perturbed::Array{ComplexF64,3}` - Perturbed displacement field [psi, theta, mode]
  - `b_perturbed::Array{ComplexF64,3}` - Perturbed magnetic field [psi, theta, mode]
  - `coupling_coefficient::ComplexF64` - Singular layer coupling coefficient
  - `resonant_amplitude::Float64` - Resonant mode amplitude
  - `plasma_energy::Float64` - Plasma perturbation energy
  - `vacuum_energy::Float64` - Vacuum perturbation energy
  - `total_energy::Float64` - Total perturbation energy
"""
@kwdef mutable struct PerturbedEquilibriumState
    # Response fields
    xi_perturbed::Array{ComplexF64,3} = zeros(ComplexF64, 0, 0, 0)
    b_perturbed::Array{ComplexF64,3} = zeros(ComplexF64, 0, 0, 0)

    # Singular coupling results
    coupling_coefficient::ComplexF64 = 0.0 + 0.0im
    resonant_amplitude::Float64 = 0.0

    # Energies
    plasma_energy::Float64 = 0.0
    vacuum_energy::Float64 = 0.0
    total_energy::Float64 = 0.0
end
