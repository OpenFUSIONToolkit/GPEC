"""
    TorqueResponse

ψ-resolved torque response of a forward kinetic Euler-Lagrange solve, expressed on the
root-area-weighted control-surface field b̃ (the basis of `forcing_b_rootarea` and of the
stored response matrices). The matrices are cumulative in ψ: `T_xe[:, :, k]` is the response
of the plasma inside `psi[k]`, and the last slice is the total.

The quadratic form `b̃†·T_xe(ψ)·b̃` is the complex "2×" torque, matching the Fortran GPEC
`T_xe` array: the physical torque and energy of the applied spectrum `b̃` are recovered with a
factor 1/2 (`torque_profile`), `T(ψ) = b̃†·T_xe(ψ)·b̃/2`, whose real part is the toroidal torque
on the plasma inside ψ and whose imaginary part is `2n·δW` of that plasma (ideal plus kinetic).

## Fields

  - `psi::Vector{Float64}` - ψ_N of each stored node of the solve `[npsi]`
  - `m_modes`, `n_modes::Vector{Int}` - (m, n) of each row and column of `T_xe` `[numpert_total]`
  - `T_xe::Array{ComplexF64,3}` - torque response on b̃, N·m/T² `[numpert_total × numpert_total × npsi]`
  - `source_names::Vector{String}` - name of each forcing source of `T_source`: the coil sets, or the forcing file `[nsource]`
  - `T_source::Array{ComplexF64,3}` - torque response in forcing-source space `[nsource × nsource × npsi]`,
    N·m per squared scale factor of each source as built: driving the sources at `s` times their deck
    currents or file amplitudes gives the complex 2× torque `s†·T_source·s`. Empty without forcing.
  - `T_applied::Vector{ComplexF64}` - the run's own forcing contracted, `b̃ₓ†·T_xe(ψ)·b̃ₓ/2` `[npsi]`;
    empty when no forcing spectrum was supplied
"""
struct TorqueResponse
    psi::Vector{Float64}
    m_modes::Vector{Int}
    n_modes::Vector{Int}
    T_xe::Array{ComplexF64,3}
    source_names::Vector{String}
    T_source::Array{ComplexF64,3}
    T_applied::Vector{ComplexF64}
end

"""
    PlasmaResponse

The source-independent half of a perturbed equilibrium: the control-surface response and
singular-coupling operators of one force-free-states result, built once by
[`plasma_response`](@ref) and applied to any number of external fields by
`perturbed_equilibrium(resp, rmp)` (Park et al., Phys. Plasmas 14, 052110 (2007); Park,
Phys. Plasmas 16, 056115 (2009)). Operators the result could not supply (no free-boundary
energies or no ξ solution) are left empty, and forcing such a response yields only the forcing
spectra.

## Fields

  - `ffs::ForceFreeStatesResult` - The result the operators were built from.
  - `m_modes`, `n_modes` - Poloidal and toroidal mode number of each entry of a spectrum [numpert_total].
  - `odet_from_gal::Bool` - The solution is the gal-native matched basis, whose `du_store` carries the
    analytic Galerkin Ξ′; selects that branch of the singular-coupling Ξ′ evaluation.
  - `compute_singular_coupling::Bool`, `compute_torque_response::Bool` - The [`plasma_response`](@ref) options it was built with.
  - `flux_matrix::Matrix{ComplexF64}` - Control-surface flux of the free-boundary eigenmodes [mode × eigenmode].
  - `plasma_inductance_flux`, `surface_inductance_flux`, `permeability_flux` - Λ, L and P = Λ·L⁻¹ acting
    on the physical control-surface flux Φ_x; the applied quantities and energies are evaluated with these.

Control surface matrices [numpert_total × numpert_total], stored in the coordinate-invariant
root-area-weighted field (b̃) space (Pharr 2026). Writing `S ≡ rootarea_to_area_weight`
(b̃→b̄) and `A ≡ surface_area`, the brief internal flux-conform operator is `R = S·A` (Φ = R·b̃).
Recover the area-weighted (b̄) forms by conforming with `S` (e.g. `L_b̄ = S·L̃·S†`); recover flux
with `Φ = A·b̄`.

  - `plasma_inductance` - Λ̃ = R⁻¹·Λ·R⁻† (wt0-based plasma inductance, congruence)
  - `surface_inductance` - L̃ = R⁻¹·L·R⁻† (vacuum surface inductance, congruence)
  - `permeability` - P̃ = R⁻¹·P·R (plasma response operator P=Λ·L⁻¹, similarity)
  - `reluctance` - ϱ̃ = R†·ϱ·R (ϱ = L⁻¹·(Λ−L)·L⁻¹, congruence)
  - `rootarea_to_area_weight` - S = Σ/√A at psilim (b̃→b̄ recovery operator)
  - `surface_area` - scalar A = ∫J|∇ψ|dθ at psilim (flux recovery Φ = A·b̄)

Coupling matrices [n_rational × numpert_total] — one row per resonant (surface, n) pair, mapping the
applied root-area-weighted field b̃_x to the resonant response at that surface. Matches Fortran
`C_f_x_out`, `C_i_x_out`, etc. (shape [mode_C, m_out]).

  - `C_resonant_area_weighted_field`   - Φ_r/A^r coupling (resonant area-weighted field b^r in tesla; singcoup row 1) [Pharr 2026]
  - `C_resonant_current` - Resonant current coupling (singcoup row 2)
  - `C_island_width_sq`  - (w/2)² coupling (singcoup row 3)
  - `C_penetrated_area_weighted_field` - Penetrated area-weighted field coupling (singcoup row 4)
  - `C_delta_prime`      - Δ' coupling (singcoup row 5)
  - `coupling_flux::NamedTuple` - The same five matrices acting on the physical flux Φ_x (keys without the
    `C_` prefix), used for the applied resonant vectors; empty when the coupling was not built.
  - `C_coeffs::Matrix{ComplexF64}` - Edge-to-solution coefficients `u_bnd⁻¹·edge_mn` of each unit forcing mode.

Metadata [n_rational] — identifies each (surface, n) row:

  - `rational_psi`, `rational_q`, `rational_m_res`, `rational_n`, `rational_surface_idx`, `rational_area`

Dominant resonant-coupling modes — the run summary SVD `U·diag(σ)·Vᴴ` of
`C_resonant_area_weighted_field` over the core window (window post hoc with `dominant_coupling`
on a `ResonantCoupling`):

  - `dominant_singular_values` - σ, descending [rank]
  - `dominant_right_singular_vectors` - V, applied-b̃ spectra ranked by resonant drive [numpert_total × rank]
  - `dominant_left_singular_vectors` - U, resonant-field patterns over the retained surfaces [n_retained × rank]
  - `dominant_rational_index` - rows of the `rational_*` arrays retained in the window [n_retained]
  - `T_xe` - ψ-resolved torque response on b̃ of a single-`n` forward kinetic solve, cumulative in ψ
    [numpert_total × numpert_total × npsi]; built only when `plasma_response` is asked for it, else empty.
    Each `perturbed_equilibrium` call contracts it with its forcing into a [`TorqueResponse`](@ref).
"""
@kwdef mutable struct PlasmaResponse
    ffs::ForceFreeStatesResult
    m_modes::Vector{Int} = Int[]
    n_modes::Vector{Int} = Int[]
    odet_from_gal::Bool = false
    compute_singular_coupling::Bool = true
    compute_torque_response::Bool = false

    # Flux-space operators [numpert_total × numpert_total]
    flux_matrix::Matrix{ComplexF64} = zeros(ComplexF64, 0, 0)
    plasma_inductance_flux::Matrix{ComplexF64} = zeros(ComplexF64, 0, 0)
    surface_inductance_flux::Matrix{ComplexF64} = zeros(ComplexF64, 0, 0)
    permeability_flux::Matrix{ComplexF64} = zeros(ComplexF64, 0, 0)

    # Control surface matrices, root-area-weighted field (b̃) space
    plasma_inductance::Matrix{ComplexF64} = zeros(ComplexF64, 0, 0)
    surface_inductance::Matrix{ComplexF64} = zeros(ComplexF64, 0, 0)
    permeability::Matrix{ComplexF64} = zeros(ComplexF64, 0, 0)
    reluctance::Matrix{ComplexF64} = zeros(ComplexF64, 0, 0)
    rootarea_to_area_weight::Matrix{ComplexF64} = zeros(ComplexF64, 0, 0)
    surface_area::Float64 = 0.0

    # Coupling matrices [n_rational × numpert_total], b̃ input basis and flux input basis
    C_resonant_area_weighted_field::Matrix{ComplexF64} = zeros(ComplexF64, 0, 0)
    C_resonant_current::Matrix{ComplexF64} = zeros(ComplexF64, 0, 0)
    C_island_width_sq::Matrix{ComplexF64} = zeros(ComplexF64, 0, 0)
    C_penetrated_area_weighted_field::Matrix{ComplexF64} = zeros(ComplexF64, 0, 0)
    C_delta_prime::Matrix{ComplexF64} = zeros(ComplexF64, 0, 0)
    coupling_flux::NamedTuple = (;)
    C_coeffs::Matrix{ComplexF64} = zeros(ComplexF64, 0, 0)

    # Metadata [n_rational]
    rational_psi::Vector{Float64} = Float64[]
    rational_q::Vector{Float64} = Float64[]
    rational_m_res::Vector{Int} = Int[]
    rational_n::Vector{Int} = Int[]
    rational_surface_idx::Vector{Int} = Int[]
    rational_area::Vector{Float64} = Float64[]

    # Dominant resonant-coupling modes
    dominant_singular_values::Vector{Float64} = Float64[]
    dominant_right_singular_vectors::Matrix{ComplexF64} = zeros(ComplexF64, 0, 0)
    dominant_left_singular_vectors::Matrix{ComplexF64} = zeros(ComplexF64, 0, 0)
    dominant_rational_index::Vector{Int} = Int[]

    # Torque response of a kinetic forward solve, opt-in [numpert_total × numpert_total × npsi]
    T_xe::Array{ComplexF64,3} = zeros(ComplexF64, 0, 0, 0)
end

"""
    PerturbedEquilibriumState

The perturbed equilibrium driven by one external field: the forcing materialized against a
[`PlasmaResponse`](@ref) and the response, profiles and resonant quantities it produces.

## Fields

  - `response::PlasmaResponse` - The operators this field was applied to.
  - `forcing` - The [`ForcingTerms.RMPField`](@ref) applied.
  - `forcing_modes::Vector{ForcingMode}` - The materialized control-surface forcing Φ_x.
  - `coil_sets::Vector{CoilSet}` - Coil geometry used (coil-format forcing); captured for the gpec.h5 rerun snapshot.
  - `regularization_width::Float64` - Width the profiles were reconstructed with (Fortran `reg_spot`): 0, no
    regularization (Fortran `reg_flag = f`). Set it, typically to 0.05, only to drive KineticForces with an
    ideal solve, whose 1/(m−nq) divergence leaves the NTV torque integral unconverged; a kinetic solve has
    none to smooth (Park & Logan, Phys. Plasmas 24, 032505 (2017) §III D).

Response fields (mode space):

  - `psi_grid` - FFS ODE integration ψ_n values [npsi]
  - `xi_modes::Union{Nothing, NamedTuple}` - Displacement (psi, theta, zeta) [npsi, numpert_total]
  - `b_modes::Union{Nothing, NamedTuple}` - Magnetic field; psi=J·b^ψ, b_psi_area_weighted=J·b^ψ/⟨J·|∇ψ|⟩_θ, theta/zeta=unregularized, theta_reg/zeta_reg=regularized [npsi, numpert_total]
  - `b_n_modes::Union{Nothing, Matrix{ComplexF64}}` - Physical normal field b_n [npsi, numpert_total]
  - `xi_n_modes::Union{Nothing, Matrix{ComplexF64}}` - Physical normal displacement xi_n [npsi, numpert_total]

Applied resonant vectors [n_rational] = C · forcing, on the rows of `response`.
Matches Fortran `Phi_res`, `w_isl`, `K_isl`, `Delta`.

  - `resonant_area_weighted_field`, `resonant_current`, `island_width_sq`, `penetrated_area_weighted_field`, `delta_prime`
  - `forcing_solution_weights` - `C_coeffs`·Φ_x, the forcing's weights on the edge-driven solutions

Diagnostics [n_rational]:

  - `island_half_width::Vector{Float64}` - w/2 = sqrt(|island_width_sq|) from applied forcing
  - `chirikov_parameter::Vector{Float64}` - Island overlap metric
  - `dominant_forcing_overlap` - Vᴴ·b̃_x, the applied forcing's coefficient on each dominant mode [rank]

Control-surface forcing/response spectra [numpert_total], in the three Pharr (2026) field
representations (all tesla; no flux/weber is stored):

  - `forcing_b`/`response_b` - bare normal field b (Σ⁻¹·b̃)
  - `forcing_b_rootarea`/`response_b_rootarea` - root-area-weighted field b̃ (coordinate-invariant)
  - `forcing_b_area`/`response_b_area` - area-weighted field b̄ (= S·b̃; flux is Φ = A·b̄)

Energies (Joules; Fortran gpout convention). Congruence-invariant scalars
(energy = Φ†·G⁻¹·Φ = b̃†·G̃⁻¹·b̃), evaluated from the brief internal flux vectors Φ_x, Φ_tot with the
well-conditioned flux-space inductances L, Λ:

  - `vacuum_energy`  - Re( ⟨Φ_x,  L⁻¹·Φ_x⟩ ) / 4   (energy to perturb the vacuum)
  - `surface_energy` - Re( ⟨Φ_tot, L⁻¹·Φ_tot⟩ ) / 4 (energy at the control surface)
  - `plasma_energy`  - Σₙ Re( ⟨Φ_tot,n, Λₙₙ⁻¹·Φ_tot,n⟩ ) / 4 over the diagonal n blocks of Λ (energy to
    perturb the plasma; Fortran's "total energy")
  - `toroidal_torque` - Σₙ −2n·Im( ⟨Φ_tot,n, Λₙₙ⁻¹·Φ_tot,n⟩ / 4 ) [Park 2011 PoP 18 110702, eq. 1] —
    the boundary-response torque, zero for ideal
    (Hermitian) runs. Equals the volume-integrated Euler-Lagrange kinetic torque only for converged
    self-consistent solutions, and is a distinct construction from the KineticForces NTV torque.

Torque response (kinetic forward solves with `compute_torque_response`):

  - `torque_response::Union{Nothing,TorqueResponse}` - the ψ-resolved torque response matrices on the
    applied b̃ and on the coil sets, with the run's own forcing contracted; see [`TorqueResponse`](@ref)
"""
@kwdef mutable struct PerturbedEquilibriumState
    response::PlasmaResponse
    forcing::Union{Nothing,ForcingTerms.RMPField} = nothing
    forcing_modes::Vector{ForcingMode} = ForcingMode[]
    coil_sets::Vector{CoilSet} = CoilSet[]
    regularization_width::Float64 = 0.0

    # Radial grid (FFS ODE integration ψ_n values) [npsi]
    psi_grid::Vector{Float64} = Float64[]

    # Response fields in mode space [npsi, numpert_total]
    xi_modes::Union{Nothing,NamedTuple} = nothing
    b_modes::Union{Nothing,NamedTuple} = nothing
    b_n_modes::Union{Nothing,Matrix{ComplexF64}} = nothing  # physical normal field b_n [npsi, numpert_total]
    xi_n_modes::Union{Nothing,Matrix{ComplexF64}} = nothing  # physical normal displacement xi_n [npsi, numpert_total]

    # Applied resonant vectors [n_rational] = C · amp_vec
    resonant_area_weighted_field::Vector{ComplexF64} = ComplexF64[]
    resonant_current::Vector{ComplexF64} = ComplexF64[]
    island_width_sq::Vector{ComplexF64} = ComplexF64[]
    penetrated_area_weighted_field::Vector{ComplexF64} = ComplexF64[]
    delta_prime::Vector{ComplexF64} = ComplexF64[]
    forcing_solution_weights::Vector{ComplexF64} = ComplexF64[]

    # Diagnostics [n_rational]
    island_half_width::Vector{Float64} = Float64[]
    chirikov_parameter::Vector{Float64} = Float64[]
    dominant_forcing_overlap::Vector{ComplexF64} = ComplexF64[]  # Vᴴ·b̃_x [rank]

    # Control-surface forcing/response spectra in the three weightings of field representations [numpert_total], tesla
    forcing_b::Vector{ComplexF64} = ComplexF64[]  # bare normal field b (forcing Φ_x)
    forcing_b_rootarea::Vector{ComplexF64} = ComplexF64[]  # root-area-weighted field b̃ (coordinate-invariant)
    forcing_b_area::Vector{ComplexF64} = ComplexF64[]  # area-weighted field b̄
    response_b::Vector{ComplexF64} = ComplexF64[]  # bare normal field b (response Φ_tot = P·Φ_x)
    response_b_rootarea::Vector{ComplexF64} = ComplexF64[]  # root-area-weighted field b̃
    response_b_area::Vector{ComplexF64} = ComplexF64[]  # area-weighted field b̄

    # Energies — see the struct docstring for formulas
    vacuum_energy::Float64 = 0.0
    surface_energy::Float64 = 0.0
    plasma_energy::Float64 = 0.0
    toroidal_torque::Float64 = 0.0

    # ψ-resolved torque response matrices of a kinetic forward solve
    torque_response::Union{Nothing,TorqueResponse} = nothing
end
