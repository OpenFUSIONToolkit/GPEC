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
# High Priority (MWE)

  - `reg_spot::Float64` - Regularization width for singular surface smoothing (default: 0.05). Set to 0 to disable. Must be ≥ 0.
"""
@kwdef struct PerturbedEquilibriumControl
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
  - `coil_sets::Vector{CoilSet}` - Coil geometry used (when `forcing_data_format = "coil"`);
    captured for the gpec.h5 rerun snapshot
  - `plasma_response::Matrix{ComplexF64}` - Plasma response matrix
  - `singular_coupling_metrics::Dict{String,Float64}` - Coupling metrics at singular surfaces
  - `m_modes::Vector{Int}` - Poloidal mode numbers for each index i in 1:numpert_total
  - `n_modes::Vector{Int}` - Toroidal mode numbers for each index i in 1:numpert_total
"""
@kwdef mutable struct PerturbedEquilibriumInternal
    dir_path::String = "./"
    forcing_modes::Vector{ForcingMode} = ForcingMode[]
    coil_sets::Vector{CoilSet} = CoilSet[]
    plasma_response::Matrix{ComplexF64} = zeros(ComplexF64, 0, 0)
    singular_coupling_metrics::Dict{String,Float64} = Dict{String,Float64}()
    m_modes::Vector{Int} = Int[]
    n_modes::Vector{Int} = Int[]
    # ForceFreeStates-provided B_pen per (match surface × coil-drive column) from inner layer.
    inner_bpen::Matrix{ComplexF64} = zeros(ComplexF64, 0, 0)
    # True when the consumed solution came from gal matching, whose du_store carries the
    # analytic galerkin Ξ′; selects the gal branch of the singular-coupling Ξ′ evaluation.
    odet_from_gal::Bool = false
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

Control-surface forcing/response spectra [numpert_total], in the three Pharr (2026) field
representations (all tesla; no flux/weber is stored):

  - `forcing_b`/`response_b` - bare normal field b (Σ⁻¹·b̃)
  - `forcing_b_rootarea`/`response_b_rootarea` - root-area-weighted field b̃ (coordinate-invariant)
  - `forcing_b_area`/`response_b_area` - area-weighted field b̄ (= S·b̃; flux is Φ = A·b̄)

Control surface matrices [numpert_total × numpert_total], stored in the coordinate-invariant
root-area-weighted field (b̃) space (issue #233 / Pharr 2026). Writing `S ≡ rootarea_to_area_weight`
(b̃→b̄) and `A ≡ surface_area`, the brief internal flux-conform operator is `R = S·A` (Φ = R·b̃).
Recover the area-weighted (b̄) forms by conforming with `S` (e.g. `L_b̄ = S·L̃·S†`); recover flux
with `Φ = A·b̄`.

  - `plasma_inductance` - Λ̃ = R⁻¹·Λ·R⁻† (wt0-based plasma inductance, congruence)
  - `surface_inductance` - L̃ = R⁻¹·L·R⁻† (vacuum surface inductance, congruence)
  - `permeability` - P̃ = R⁻¹·P·R (plasma response operator P=Λ·L⁻¹, similarity)
  - `reluctance` - ϱ̃ = R†·ϱ·R (ϱ = L⁻¹·(Λ−L)·L⁻¹, congruence)
  - `rootarea_to_area_weight` - S = Σ/√A at psilim (b̃→b̄ recovery operator)
  - `surface_area` - scalar A = ∫J|∇ψ|dθ at psilim (flux recovery Φ = A·b̄)

Energies (Joules; Fortran gpout convention). Congruence-invariant scalars
(energy = Φ†·G⁻¹·Φ = b̃†·G̃⁻¹·b̃), evaluated from the brief internal flux vectors Φ_x, Φ_tot with the
well-conditioned flux-space inductances L, Λ:

  - `vacuum_energy`  - Re( ⟨Φ_x,  L⁻¹·Φ_x⟩ ) / 4   (energy to perturb the vacuum)
  - `surface_energy` - Re( ⟨Φ_tot, L⁻¹·Φ_tot⟩ ) / 4 (energy at the control surface)
  - `plasma_energy`  - Re( ⟨Φ_tot, Λ⁻¹·Φ_tot⟩ ) / 4 (energy to perturb the plasma; Fortran's "total energy")    # Response fields in mode space [npsi, mpert]
  - `toroidal_torque` - -2·n·Im( ⟨Φ_tot, Λ⁻¹·Φ_tot⟩ / 4 ) — the **boundary-response torque**:
    the net toroidal torque implied by the anti-Hermitian part of the plasma inductance at the
    control surface. For exact self-consistent solutions this equals the volume-integrated
    kinetic torque of the same Euler-Lagrange model (the δW surface-term identity), so any
    numerical difference from a ψ-resolved torque profile is a solution/grid-quality diagnostic,
    not physics. It is a *distinct construction* from the NTV torque under `KineticForces/`
    (independent energy-space bounce-average evaluation); agreement between the two tests whether
    the EL matrices faithfully discretize the drift-kinetic operator. Zero for ideal (Hermitian) runs.
"""
@kwdef mutable struct PerturbedEquilibriumState
    # Radial grid (FFS ODE integration ψ_n values) [npsi]
    psi_grid::Vector{Float64} = Float64[]

    # Response fields in mode space [npsi, mpert]
    xi_modes::Union{Nothing,NamedTuple} = nothing
    b_modes::Union{Nothing,NamedTuple} = nothing
    b_n_modes::Union{Nothing,Matrix{ComplexF64}} = nothing  # physical normal field b_n [npsi, mpert]
    xi_n_modes::Union{Nothing,Matrix{ComplexF64}} = nothing  # physical normal displacement xi_n [npsi, mpert]

    # Coupling matrices [n_rational × numpert_total]
    C_resonant_area_weighted_field::Matrix{ComplexF64} = zeros(ComplexF64, 0, 0)
    C_resonant_current::Matrix{ComplexF64} = zeros(ComplexF64, 0, 0)
    C_island_width_sq::Matrix{ComplexF64} = zeros(ComplexF64, 0, 0)
    C_penetrated_area_weighted_field::Matrix{ComplexF64} = zeros(ComplexF64, 0, 0)
    C_delta_prime::Matrix{ComplexF64} = zeros(ComplexF64, 0, 0)

    # Applied resonant vectors [n_rational] = C · amp_vec
    resonant_area_weighted_field::Vector{ComplexF64} = ComplexF64[]
    resonant_current::Vector{ComplexF64} = ComplexF64[]
    island_width_sq::Vector{ComplexF64} = ComplexF64[]
    penetrated_area_weighted_field::Vector{ComplexF64} = ComplexF64[]
    delta_prime::Vector{ComplexF64} = ComplexF64[]
    forcing_solution_weights::Vector{ComplexF64} = ComplexF64[]
    rational_area::Vector{Float64} = Float64[]

    # Diagnostics [n_rational]
    island_half_width::Vector{Float64} = Float64[]
    chirikov_parameter::Vector{Float64} = Float64[]

    # Metadata [n_rational]
    rational_psi::Vector{Float64} = Float64[]
    rational_q::Vector{Float64} = Float64[]
    rational_m_res::Vector{Int} = Int[]
    rational_n::Vector{Int} = Int[]
    rational_surface_idx::Vector{Int} = Int[]

    # Control-surface forcing/response spectra in the three weightings of field representations [numpert_total], tesla
    forcing_b::Vector{ComplexF64} = ComplexF64[]  # bare normal field b (forcing Φ_x)
    forcing_b_rootarea::Vector{ComplexF64} = ComplexF64[]  # root-area-weighted field b̃ (coordinate-invariant)
    forcing_b_area::Vector{ComplexF64} = ComplexF64[]  # area-weighted field b̄
    response_b::Vector{ComplexF64} = ComplexF64[]  # bare normal field b (response Φ_tot = P·Φ_x)
    response_b_rootarea::Vector{ComplexF64} = ComplexF64[]  # root-area-weighted field b̃
    response_b_area::Vector{ComplexF64} = ComplexF64[]  # area-weighted field b̄

    # Control surface matrices [numpert_total × numpert_total], root-area-weighted field (b̃) space
    plasma_inductance::Matrix{ComplexF64} = zeros(ComplexF64, 0, 0)  # Λ̃ (field space)
    surface_inductance::Matrix{ComplexF64} = zeros(ComplexF64, 0, 0)  # L̃ (field space)
    permeability::Matrix{ComplexF64} = zeros(ComplexF64, 0, 0)  # P̃ = R⁻¹·Λ·L⁻¹·R
    reluctance::Matrix{ComplexF64} = zeros(ComplexF64, 0, 0)  # ϱ̃ = R†·L⁻¹·(Λ−L)·L⁻¹·R
    rootarea_to_area_weight::Matrix{ComplexF64} = zeros(ComplexF64, 0, 0)  # S = Σ/√A at psilim: b̃→b̄ recovery operator
    surface_area::Float64 = 0.0  # scalar control-surface area A = ∫J|∇ψ|dθ (flux: Φ = A·b̄; conform R = S·A)

    # Energies — see the struct docstring for formulas
    vacuum_energy::Float64 = 0.0
    surface_energy::Float64 = 0.0
    plasma_energy::Float64 = 0.0
    toroidal_torque::Float64 = 0.0
end
