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

  - `output_filename::String` - Combined output file with ForceFreeStates results (default: uses ForceFreeStates HDF5_filename)
  - `write_outputs_to_HDF5::Bool` - Write perturbed equilibrium outputs to HDF5 (default: true)

Medium Priority (defer for MWE):

  - `filter_modes::Bool` - Enable mode filtering (default: false)
  - `singular_point_method::String` - Method for singular point treatment (default: "standard")    # High Priority (MWE)
"""
@kwdef mutable struct PerturbedEquilibriumControl
    # High Priority (MWE)
    fixed_boundary::Bool = false
    output_eigenmodes::Bool = true
    compute_response::Bool = true
    compute_singular_coupling::Bool = true
    verbose::Bool = true

    # Output settings
    output_filename::String = ""  # Empty means use ForceFreeStates filename
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
  - `m_modes::Vector{Int}` - Poloidal mode numbers for each index i in 1:numpert_total
  - `n_modes::Vector{Int}` - Toroidal mode numbers for each index i in 1:numpert_total
"""
@kwdef mutable struct PerturbedEquilibriumInternal
    dir_path::String = "./"
    forcing_modes::Vector{ForcingMode} = ForcingMode[]
    plasma_response::Matrix{ComplexF64} = zeros(ComplexF64, 0, 0)
    singular_coupling_metrics::Dict{String,Float64} = Dict{String,Float64}()
    m_modes::Vector{Int} = Int[]
    n_modes::Vector{Int} = Int[]
end

"""
    PerturbedEquilibriumState

Results from perturbed equilibrium calculations.

## Fields

Response fields (mode space):

  - `xi_modes::Union{Nothing, NamedTuple}` - Displacement (psi, theta, zeta) [npsi, mpert]
  - `b_modes::Union{Nothing, NamedTuple}` - Magnetic field (psi, theta, zeta) [npsi, mpert]

Singular coupling matrices [msing, numpert_total]:

  - `resonant_flux::Matrix{ComplexF64}` - Resonant flux Φ_r/A (singcoup(1,:,:))
  - `resonant_current::Matrix{ComplexF64}` - Resonant current (singcoup(2,:,:))
  - `island_width_sq::Matrix{ComplexF64}` - (w/2)² in ψ_n (singcoup(3,:,:))
  - `penetrated_field::Matrix{ComplexF64}` - Resonant field (singcoup(4,:,:))
  - `delta_prime::Matrix{ComplexF64}` - Tearing stability Δ' (singcoup(5,:,:))

Diagnostic quantities [msing]:

  - `island_half_width::Vector{Float64}` - Actual w/2 (meters or ψ_n)
  - `chirikov_parameter::Vector{Float64}` - Island overlap metric

Note: numpert_total = mpert × npert handles all (m,n) mode combinations

Legacy fields (deprecated):

  - `coupling_coefficient::ComplexF64` - Use singular coupling matrices instead
  - `resonant_amplitude::Float64` - Use island diagnostics instead

Energies:

  - `plasma_energy::Float64` - Plasma perturbation energy    # Response fields in mode space [npsi, mpert] following GPEC notation
  - `vacuum_energy::Float64` - Vacuum perturbation energy    # NamedTuples contain (psi, theta, zeta) components in flux coordinates
  - `total_energy::Float64` - Total perturbation energy
"""
@kwdef mutable struct PerturbedEquilibriumState
    # Response fields in mode space [npsi, mpert] following GPEC notation
    # NamedTuples contain (psi, theta, zeta) components in flux coordinates
    xi_modes::Union{Nothing,NamedTuple} = nothing
    b_modes::Union{Nothing,NamedTuple} = nothing

    # Singular coupling matrices [msing × mpert] - GPEC singcoup(1:5,:,:)
    resonant_flux::Matrix{ComplexF64} = zeros(ComplexF64, 0, 0)        # singcoup(1,:,:)
    resonant_current::Matrix{ComplexF64} = zeros(ComplexF64, 0, 0)     # singcoup(2,:,:)
    island_width_sq::Matrix{ComplexF64} = zeros(ComplexF64, 0, 0)      # singcoup(3,:,:)
    penetrated_field::Matrix{ComplexF64} = zeros(ComplexF64, 0, 0)     # singcoup(4,:,:)
    delta_prime::Matrix{ComplexF64} = zeros(ComplexF64, 0, 0)          # singcoup(5,:,:)

    # Diagnostic quantities [msing]
    island_half_width::Vector{Float64} = Float64[]     # Actual w/2 (meters or ψ_n)
    chirikov_parameter::Vector{Float64} = Float64[]    # Island overlap

    # Legacy singular coupling results (deprecated - use matrices above)
    coupling_coefficient::ComplexF64 = 0.0 + 0.0im
    resonant_amplitude::Float64 = 0.0

    # Energies
    plasma_energy::Float64 = 0.0
    vacuum_energy::Float64 = 0.0
    total_energy::Float64 = 0.0
end
