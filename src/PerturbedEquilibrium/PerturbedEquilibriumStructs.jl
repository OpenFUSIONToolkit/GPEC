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
  - `singular_point_method::String` - Method for singular point treatment (default: "standard")

Regularization:
  - `reg_spot::Float64` - Regularization width for singular surface smoothing (default: 0.05). Set to 0 to disable. Must be ≥ 0.
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

    # Regularization width for singular surface smoothing (matches Fortran gpec.f reg_spot).
    # Set to 0 to disable regularization. Must be non-negative.
    reg_spot::Float64 = 5e-2
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
  - `b_modes::Union{Nothing, NamedTuple}` - Magnetic field; psi=b^ψ, psi_area=b^ψ/⟨J·|∇ψ|⟩_θ, theta/zeta=unregularized, theta_reg/zeta_reg=regularized [npsi, mpert]
  - `b_n_modes::Union{Nothing, Matrix{ComplexF64}}` - Physical normal field b_n [npsi, mpert]
  - `xi_n_modes::Union{Nothing, Matrix{ComplexF64}}` - Physical normal displacement xi_n [npsi, mpert]

Coupling matrices [n_rational × numpert_total] — one row per resonant (surface, n) pair.
Each row maps the full applied field to the resonant response at that surface.
Matches Fortran `C_f_x_out`, `C_i_x_out`, etc. (shape [mode_C, m_out]).
  - `C_resonant_flux`    - Φ_r/A coupling (singcoup row 1)
  - `C_resonant_current` - Resonant current coupling (singcoup row 2)
  - `C_island_width_sq`  - (w/2)² coupling (singcoup row 3)
  - `C_penetrated_field` - Penetrated field coupling (singcoup row 4)
  - `C_delta_prime`      - Δ' coupling (singcoup row 5)

Applied resonant vectors [n_rational] = C · forcing_amplitudes.
Matches Fortran `Phi_res`, `w_isl`, `K_isl`, `Delta`.
  - `resonant_flux`, `resonant_current`, `island_width_sq`, `penetrated_field`, `delta_prime`

Diagnostics [n_rational]:
  - `island_half_width::Vector{Float64}` - w/2 = sqrt(|island_width_sq|) from applied forcing
  - `chirikov_parameter::Vector{Float64}` - Island overlap metric

Metadata [n_rational] — identifies each (surface, n) row:
  - `rational_psi`, `rational_q`, `rational_m_res`, `rational_n`, `rational_surface_idx`

Control surface matrices [numpert_total × numpert_total]:
  - `plasma_inductance` - Lambda (wt0-based plasma inductance)
  - `surface_inductance` - L (vacuum surface inductance from Green's functions)
  - `permeability` - P = Lambda * L^{-1} (plasma response matrix, Phi_tot = P * Phi_x)
  - `reluctance` - Rho = L^{-1} * (Lambda - L) * L^{-1}

Energies:
  - `plasma_energy`, `vacuum_energy`, `total_energy`
"""
@kwdef mutable struct PerturbedEquilibriumState
    # Radial grid (FFS ODE integration ψ_n values) [npsi]
    psi_grid::Vector{Float64} = Float64[]

    # Response fields in mode space [npsi, mpert]
    xi_modes::Union{Nothing, NamedTuple} = nothing
    b_modes::Union{Nothing, NamedTuple} = nothing
    b_n_modes::Union{Nothing, Matrix{ComplexF64}}  = nothing  # physical normal field b_n [npsi, mpert]
    xi_n_modes::Union{Nothing, Matrix{ComplexF64}} = nothing  # physical normal displacement xi_n [npsi, mpert]

    # Coupling matrices [n_rational × numpert_total]
    C_resonant_flux::Matrix{ComplexF64}     = zeros(ComplexF64, 0, 0)
    C_resonant_current::Matrix{ComplexF64}  = zeros(ComplexF64, 0, 0)
    C_island_width_sq::Matrix{ComplexF64}   = zeros(ComplexF64, 0, 0)
    C_penetrated_field::Matrix{ComplexF64}  = zeros(ComplexF64, 0, 0)
    C_delta_prime::Matrix{ComplexF64}       = zeros(ComplexF64, 0, 0)

    # Applied resonant vectors [n_rational] = C · amp_vec
    resonant_flux::Vector{ComplexF64}       = ComplexF64[]
    resonant_current::Vector{ComplexF64}    = ComplexF64[]
    island_width_sq::Vector{ComplexF64}     = ComplexF64[]
    penetrated_field::Vector{ComplexF64}    = ComplexF64[]
    delta_prime::Vector{ComplexF64}         = ComplexF64[]

    # Diagnostics [n_rational]
    island_half_width::Vector{Float64}      = Float64[]
    chirikov_parameter::Vector{Float64}     = Float64[]

    # Metadata [n_rational]
    rational_psi::Vector{Float64}           = Float64[]
    rational_q::Vector{Float64}             = Float64[]
    rational_m_res::Vector{Int}             = Int[]
    rational_n::Vector{Int}                 = Int[]
    rational_surface_idx::Vector{Int}       = Int[]

    # Control surface perturbation vectors [numpert_total]
    forcing_vec::Vector{ComplexF64}   = ComplexF64[]  # Phi_x: external forcing in eigenmode basis
    response_vec::Vector{ComplexF64}  = ComplexF64[]  # Phi_tot = P * Phi_x: total plasma response

    # Control surface matrices [numpert_total × numpert_total]
    plasma_inductance::Matrix{ComplexF64}  = zeros(ComplexF64, 0, 0)  # Lambda
    surface_inductance::Matrix{ComplexF64} = zeros(ComplexF64, 0, 0)  # L
    permeability::Matrix{ComplexF64}       = zeros(ComplexF64, 0, 0)  # P = Lambda * L^{-1}
    reluctance::Matrix{ComplexF64}         = zeros(ComplexF64, 0, 0)  # Rho = L^{-1}*(Lambda-L)*L^{-1}

    # Energies
    plasma_energy::Float64  = 0.0
    vacuum_energy::Float64  = 0.0
    total_energy::Float64   = 0.0
end
