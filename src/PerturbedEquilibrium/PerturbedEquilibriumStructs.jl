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
  - `b_modes::Union{Nothing, NamedTuple}` - Magnetic field; psi=b^ψ, b_psi_area_weighted=b^ψ/⟨J·|∇ψ|⟩_θ, theta/zeta=unregularized, theta_reg/zeta_reg=regularized [npsi, mpert]
  - `b_n_modes::Union{Nothing, Matrix{ComplexF64}}` - Physical normal field b_n [npsi, mpert]
  - `xi_n_modes::Union{Nothing, Matrix{ComplexF64}}` - Physical normal displacement xi_n [npsi, mpert]

Coupling matrices [n_rational × numpert_total] — one row per resonant (surface, n) pair.
Each row maps the full applied field to the resonant response at that surface.
Matches Fortran `C_f_x_out`, `C_i_x_out`, etc. (shape [mode_C, m_out]).
  - `C_resonant_area_weighted_field`   - Φ_r/A^r coupling (resonant area-weighted field b^r in tesla; singcoup row 1) [Pharr 2026]
  - `C_resonant_current` - Resonant current coupling (singcoup row 2)
  - `C_island_width_sq`  - (w/2)² coupling (singcoup row 3)
  - `C_penetrated_area_weighted_field` - Penetrated area-weighted field coupling (singcoup row 4)
  - `C_delta_prime`      - Δ' coupling (singcoup row 5)

Applied resonant vectors [n_rational] = C · forcing_amplitudes.
Matches Fortran `Phi_res`, `w_isl`, `K_isl`, `Delta`.
  - `resonant_area_weighted_field`, `resonant_current`, `island_width_sq`, `penetrated_area_weighted_field`, `delta_prime`

Diagnostics [n_rational]:
  - `island_half_width::Vector{Float64}` - w/2 = sqrt(|island_width_sq|) from applied forcing
  - `chirikov_parameter::Vector{Float64}` - Island overlap metric

Metadata [n_rational] — identifies each (surface, n) row:
  - `rational_psi`, `rational_q`, `rational_m_res`, `rational_n`, `rational_surface_idx`

Control surface matrices [numpert_total × numpert_total], stored in the coordinate-invariant
root-area-weighted field (b̃) space (issue #233 / Pharr 2026). Recover the flux-space forms with
the `control_surface_field_operator` (`rootareafield_to_flux`, Φ = rootareafield_to_flux·b̃): e.g. `P_flux = rootareafield_to_flux·P̃·rootareafield_to_flux⁻¹`,
`L_flux = rootareafield_to_flux·L̃·rootareafield_to_flux†`.
  - `plasma_inductance` - Λ̃ = rootareafield_to_flux⁻¹·Λ·rootareafield_to_flux⁻† (wt0-based plasma inductance, congruence)
  - `surface_inductance` - L̃ = rootareafield_to_flux⁻¹·L·rootareafield_to_flux⁻† (vacuum surface inductance, congruence)
  - `permeability` - P̃ = rootareafield_to_flux⁻¹·P·rootareafield_to_flux (plasma response operator P=Λ·L⁻¹, similarity)
  - `reluctance` - ϱ̃ = rootareafield_to_flux†·ϱ·rootareafield_to_flux (ϱ = L⁻¹·(Λ−L)·L⁻¹, congruence)
  - `control_surface_field_operator` - rootareafield_to_flux = sqrtamat·√jarea at psilim (field → flux recovery operator)

Energies (Fortran gpout convention; Φ_x external flux, Φ_tot total flux, L/Λ inductances):
  - `vacuum_energy`  - Re( ⟨Φ_x,  L⁻¹·Φ_x⟩ ) / 4   (energy to perturb the vacuum)
  - `surface_energy` - Re( ⟨Φ_tot, L⁻¹·Φ_tot⟩ ) / 4 (energy at the control surface)
  - `plasma_energy`  - Re( ⟨Φ_tot, Λ⁻¹·Φ_tot⟩ ) / 4 (energy to perturb the plasma; Fortran's "total energy")
  - `toroidal_torque` - -2·n·Im( ⟨Φ_tot, Λ⁻¹·Φ_tot⟩ / 4 )
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
    C_resonant_area_weighted_field::Matrix{ComplexF64}    = zeros(ComplexF64, 0, 0)
    C_resonant_current::Matrix{ComplexF64}  = zeros(ComplexF64, 0, 0)
    C_island_width_sq::Matrix{ComplexF64}   = zeros(ComplexF64, 0, 0)
    C_penetrated_area_weighted_field::Matrix{ComplexF64}  = zeros(ComplexF64, 0, 0)
    C_delta_prime::Matrix{ComplexF64}       = zeros(ComplexF64, 0, 0)

    # Applied resonant vectors [n_rational] = C · amp_vec
    resonant_area_weighted_field::Vector{ComplexF64}      = ComplexF64[]
    resonant_current::Vector{ComplexF64}    = ComplexF64[]
    island_width_sq::Vector{ComplexF64}     = ComplexF64[]
    penetrated_area_weighted_field::Vector{ComplexF64}    = ComplexF64[]
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
    plasma_inductance::Matrix{ComplexF64}  = zeros(ComplexF64, 0, 0)  # Λ̃ (field space)
    surface_inductance::Matrix{ComplexF64} = zeros(ComplexF64, 0, 0)  # L̃ (field space)
    permeability::Matrix{ComplexF64}       = zeros(ComplexF64, 0, 0)  # P̃ = rootareafield_to_flux⁻¹·Λ·L⁻¹·rootareafield_to_flux
    reluctance::Matrix{ComplexF64}         = zeros(ComplexF64, 0, 0)  # ϱ̃ = rootareafield_to_flux†·L⁻¹·(Λ−L)·L⁻¹·rootareafield_to_flux
    control_surface_field_operator::Matrix{ComplexF64} = zeros(ComplexF64, 0, 0)  # rootareafield_to_flux: field → flux recovery

    # Energies — see the struct docstring for formulas
    vacuum_energy::Float64   = 0.0
    surface_energy::Float64  = 0.0
    plasma_energy::Float64   = 0.0
    toroidal_torque::Float64 = 0.0
end
