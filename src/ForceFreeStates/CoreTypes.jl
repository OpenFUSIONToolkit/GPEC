# Module-wide types: ModeSpace, ForceFreeStatesInternal, ForceFreeStatesControl.

"""
    ModeSpace

Supertype for objects that carry the resolved (m, n) mode space — `mlow`, `mhigh`, `mpert`,
`nlow`, `nhigh`, `npert`, `numpert_total`. Both the solve-time scratch
[`ForceFreeStatesInternal`](@ref) and the published [`ForceFreeStatesResult`](@ref) are
`ModeSpace`s, so kernels that need nothing but the mode indexing (`el_derivatives!`,
`materialize_derivative_stores!`, `build_kinetic_metric_matrices`) accept either.
"""
abstract type ModeSpace end

"""
    ForceFreeStatesInternal

Solve-time scratch state for a force-balance solve: the resolved mode space, singular surfaces and integration limits.

## Fields

  - `mlow::Int` - Lowest poloidal mode number
  - `mhigh::Int` - Highest poloidal mode number
  - `mpert::Int` - Number of poloidal modes (mhigh - mlow + 1)
  - `nlow::Int` - Lowest toroidal mode number, resolved from `ctrl.nn_low`/`nn_high`
  - `nhigh::Int` - Highest toroidal mode number, resolved from `ctrl.nn_low`/`nn_high`
  - `npert::Int` - Number of toroidal modes (nhigh - nlow + 1)
  - `numpert_total::Int` - Total number of modes (mpert × npert)
  - `msing::Int` - Number of ideal singular surfaces
  - `kmsing::Int` - Number of kinetic singular surfaces (det(F̄) near-zeros)
  - `sing::Vector{SingType}` - Vector of ideal singular surface data
  - `kinsing::Vector{SingType}` - Vector of kinetic singular surface data
  - `kinsing_scan_psi::Vector{Float64}` - ψ grid used by `find_kinetic_singular_surfaces!` for the cond(F̄) scan (empty unless the finder has run)
  - `kinsing_scan_cond::Vector{Float64}` - cond(F̄) values on that grid; the finder locates peaks that exceed `kinsing_scan_threshold`
  - `kinsing_scan_threshold::Float64` - Threshold on cond(F̄) used to accept a peak as a kinetic singular surface
  - `psilim::Float64` - Flux limit for integration
  - `qlim::Float64` - Safety factor at psilim
  - `q1lim::Float64` - Safety factor derivative at psilim
  - `wall_settings::Union{Nothing,Vacuum.WallSettings}` - Wall shape settings for vacuum calculations
  - `verbose::Bool` - Log solve progress; set from the `verbose` keyword of `solve`
"""
@kwdef mutable struct ForceFreeStatesInternal <: ModeSpace
    mlow::Int = 0
    mhigh::Int = 0
    mpert::Int = 0
    nlow::Int = 0
    nhigh::Int = 0
    npert::Int = 0
    numpert_total::Int = 0
    msing::Int = 0
    kmsing::Int = 0
    sing::Vector{SingType} = SingType[]
    kinsing::Vector{SingType} = SingType[]
    kinsing_scan_psi::Vector{Float64} = Float64[]
    kinsing_scan_cond::Vector{Float64} = Float64[]
    kinsing_scan_threshold::Float64 = 0.0
    psilim::Float64 = 0.0
    psilow::Float64 = 0.0   # lower integration bound; the outermost crossing of q = qlow (sing_min!), else the first grid node
    qlim::Float64 = 0.0
    q1lim::Float64 = 0.0
    wall_settings::Union{Nothing,Vacuum.WallSettings} = nothing
    verbose::Bool = false
    """
    Inter-surface Δ' matrix, `msing × msing`, PEST3 convention: the odd-parity projection of
    `delta_prime_raw`, set by `compute_delta_prime_matrix!` (Riccati path only).
    """
    delta_prime_matrix::Matrix{ComplexF64} = Matrix{ComplexF64}(undef, 0, 0)

    """
    Edge coil-response matrix, `2msing × numpert_total`: column k is the resonant small-solution
    response at each surface side to a unit source on edge mode k (Eq. 37 rpec edge condition,
    `_solve_bvp_edge_coil`). Empty unless the S-axis vacuum-edge BVP was assembled.
    """
    delta_coil::Matrix{ComplexF64} = Matrix{ComplexF64}(undef, 0, 0)

    """
    Raw `2msing × 2msing` outer-region matching matrix D' (Pletzer–Dewar 1991, before parity
    rotation), side-major ordering `[L_s1, R_s1, …, L_sm, R_sm]`. `pest3_decompose` recovers the
    A', B', Γ', Δ' blocks. No ½ prefactor (matches Fortran rdcon; the paper applies ½).
    Empty unless the Riccati integrator was used.
    """
    delta_prime_raw::Matrix{ComplexF64} = Matrix{ComplexF64}(undef, 0, 0)
end

"""
    ForceFreeStatesControl

Physical and workflow controls for a force-balance solve, together with its selected algorithm.

## Fields

  - `algorithm::AbstractIntegrator` - Selected integrator and its numerical settings.
  - `vac_flag::Bool` - Enable vacuum region calculation
  - `mtheta_vacuum::Int` - Number of vacuum poloidal grid points (corresponds to `mtheta` in VacuumInput)
  - `nzeta_vacuum::Int` - Number of vacuum toroidal grid points (corresponds to `nzeta` in VacuumInput3D)
  - `frobenius_psi_max::Float64` - Largest starting ψ_N at which the opt-in diagonal Frobenius start is used; default `0` always uses the fixed start U₁ = 0, U₂ = I. The
    Frobenius start is currently unreliable (warns when used); leave at 0 unless testing it.
  - `nn_low::Int` - Lower bound for toroidal modes
  - `nn_high::Int` - Upper bound for toroidal modes
  - `delta_mlow::Int` - Expands lower bound of Fourier harmonics by delta_mlow
  - `delta_mhigh::Int` - Expands upper bound of Fourier harmonics by delta_mhigh
  - `singfac_min::Float64` - Fractional distance from rational q at which the ideal jump condition is enforced. Default 1e-4 suits the Riccati Δ′ (smaller values add a
    spurious imaginary part); Forward runs feeding PerturbedEquilibrium need about 1e-6 for the outer-surface singular coupling to converge.
  - `truncate_at_rational_offset::Bool` - Truncate the domain at `(last_rational_q + rational_offset_fraction) / n` instead of at `qhigh` / `psihigh`, avoiding a spurious
    δW kink when the integration ends just below a rational. Right for diverted equilibria (default `true`); set `false` for limited/analytic equilibria with finite edge
    q (Solovev, LAR), where it cuts too much edge. Multi-n runs warn and fall back to `qhigh` / `psihigh`.
  - `psilim_from_layer_overlap::Union{Bool,InnerLayer.SLAYERModel}` - Cap `qlim` where adjacent rational surfaces' resistive layers overlap (Fitzpatrick, Nucl. Fusion
    2025 Sect. 5.9); `rational_offset_fraction` / `qhigh` still pick the final surface inside the cap. `true` uses the default `SLAYERModel()` closures, a
    `SLAYERModel(; ...)` its own (a deck's `[SLAYER]` section). Requires the equilibrium's kinetic data; the result carries the scan as `layer_overlap`. Default `false`.
  - `rational_offset_fraction::Float64` - Distance beyond the last rational surface, in units of 1/n, in [0, 1). Used only when `truncate_at_rational_offset` is true. Default 0.2.
  - `frobenius_order::Int` - Order of the Frobenius expansion at rational surfaces. Default 6.
  - `qhigh::Float64` - Integration terminated at q limit determined by minimum of qhigh and qa from equil
  - `kinetic` - Kinetic source of the force operator: `nothing` (ideal), a `KineticForces.KineticForcesControl` (the drift-kinetic matrices of the KineticForces model
    evaluated on the equilibrium's kinetic profiles; needs `Forward()`), `TestKineticMatrices` (synthetic test matrices), or the `KineticMatrices` of an earlier
    evaluation, reused. A deck selects the KineticForces model with `kinetic = "KineticForces"`, using its `[KineticForces]` section.
  - `kinetic_factor::Float64` - Uniform scale on the kinetic matrices of a kinetic run. Default 1.
  - `qlow::Float64` - Integration starts at the outermost crossing of q = qlow, excluding the q < qlow region inside it (a core, or a reversed-shear ring); no effect when
    q ≥ qlow everywhere
  - `dW_edge_scan_start::Float64` - If below psilim, records a dW(ψ) diagnostic scan over [dW_edge_scan_start, psilim] on `odet.edge_scan`. Does not change the domain
    unless `truncate_at_dW_peak` is set.
  - `truncate_at_dW_peak::Bool` - When `true` and `dW_edge_scan_start < psilim`, adopt the edge-dW scan's peak as the plasma edge, equivalent to setting `psihigh` to the
    peak by hand. Δ' then depends on where the peak falls in the edge band. Default `false`.
  - `write_el_matrices::Bool` - Write the ideal (and on a kinetic run, kinetic) Euler-Lagrange matrices on the ψ grid to `ForceFreeStates/EulerLagrangeMatrices`; scales
    as numpert_total²·npsi and dominates the file size. Default `false`.
"""
@kwdef struct ForceFreeStatesControl{A<:AbstractIntegrator}
    algorithm::A = Riccati()
    vac_flag::Bool = false
    mtheta_vacuum::Int = 480
    nzeta_vacuum::Int = 1
    nn_low::Int = 0
    nn_high::Int = 0
    delta_mlow::Int = 0
    delta_mhigh::Int = 0
    singfac_min::Float64 = 1e-4   # Required nonzero for the Riccati path.
    truncate_at_rational_offset::Bool = true
    psilim_from_layer_overlap::Union{Bool,InnerLayer.SLAYERModel} = false
    rational_offset_fraction::Float64 = 0.2
    frobenius_order::Int = 6
    qhigh::Float64 = 1e3
    kinetic::Any = nothing
    kinetic_factor::Float64 = 1.0
    qlow::Float64 = 0.0
    dW_edge_scan_start::Float64 = 0.99
    truncate_at_dW_peak::Bool = false
    write_el_matrices::Bool = false

    frobenius_psi_max::Float64 = 0.0
end

ForceFreeStatesControl(algorithm::A, fields...) where {A<:AbstractIntegrator} = ForceFreeStatesControl{A}(algorithm, fields...)

# Galerkin solves from the fixed start on the full domain, so the start and edge-truncation controls must keep their defaults.
function validate_algorithm(ctrl::ForceFreeStatesControl)
    if ctrl.algorithm isa Galerkin
        for (name, default) in ((:frobenius_psi_max, 0.0), (:truncate_at_dW_peak, false), (:dW_edge_scan_start, 0.99))
            getfield(ctrl, name) == default || throw(ArgumentError("Galerkin does not support non-default `$name`."))
        end
    end
    return ctrl
end
