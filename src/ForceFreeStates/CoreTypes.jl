# Module-wide types: ModeSpace, DebugSettings, ForceFreeStatesInternal, ForceFreeStatesControl.

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
DebugSettings

A mutable struct containing settings for debugging and benchmarking output.

## Fields

  - `output_benchmark_data::Bool` - Flag to output benchmark data for comparison between codes
"""
@kwdef mutable struct DebugSettings
    output_benchmark_data::Bool = false
end

"""
    ForceFreeStatesInternal

A mutable struct holding internal state variables for stability calculations.

## Fields

  - `dir_path::String` - Directory path for input/output files
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
    dir_path::String = ""
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
    psilow::Float64 = 0.0   # lower integration bound; raised above the axis by sing_min! when qlow > qmin (RDCON gal)
    qlim::Float64 = 0.0
    q1lim::Float64 = 0.0
    debug_settings::DebugSettings = DebugSettings()
    wall_settings::Union{Nothing,Vacuum.WallSettings} = nothing
    verbose::Bool = false
    """
    Inter-surface Δ' matrix of shape (msing × msing) in PEST3 convention.
    Computed by `compute_delta_prime_matrix!` (parallel FM path only) using the STRIDE
    global BVP with vacuum coupling. The deltap linear combination is applied to the
    raw 2msing×2msing BVP solution to produce the PEST3-compatible tearing parameter.
    """
    delta_prime_matrix::Matrix{ComplexF64} = Matrix{ComplexF64}(undef, 0, 0)

    """
    Edge coil-response matrix of shape (2msing × numpert_total). Column k is the resonant
    small-solution response at each surface side to a unit source on edge poloidal mode k,
    built by imposing the Eq. (37) rpec edge boundary condition on the Riccati BVP
    (`_solve_bvp_edge_coil`). Empty unless the S-axis vacuum-edge BVP was assembled.
    """
    delta_coil::Matrix{ComplexF64} = Matrix{ComplexF64}(undef, 0, 0)

    """
    Raw 2msing × 2msing outer-region matching matrix `D'` from the STRIDE global
    BVP, in the side-major ordering `[L_s1, R_s1, L_s2, R_s2, …, L_sm, R_sm]`
    (left vs right of each singular surface, interleaved surface-by-surface).
    This is the Pletzer–Dewar 1991 outer-region matrix before parity rotation,
    and is stored byte-compatibly with the Fortran `rdcon/gal.f::gal_write_delta`
    convention (top 2msing×2msing block of `delta_gw.dat`). The PEST3 Δ' matrix
    stored in `delta_prime_matrix` is the odd-parity tearing projection of this
    raw matrix; the even-parity A' and off-parity B', Γ' blocks are recovered
    via `pest3_decompose(dp_raw)` — needed for the full det(D' − D(γ)) = 0
    eigenvalue problem with Glasser stabilization.

    Empty unless the Riccati integrator was used. No ½ prefactor is applied (matches
    Fortran rdcon; Pletzer–Dewar paper multiplies by ½).
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
  - `sing_start::Int` - Start integration at the `sing_start`-th singular surface
  - `frobenius_psi_max::Float64` - Largest starting ψ_N at which the opt-in Frobenius start is used. Default `0` selects the fixed start everywhere and silently: U₁ = 0, U₂ = I, the initialization Fortran DCON uses. A positive value uses the diagonal Frobenius 2×2 when the integration starts at or below it, and warns that this start is currently unreliable. A start above the threshold uses the fixed start with no warning. That 2×2 does not recover a displacement that vanishes as a positive power of the starting ψ, and its m = 0 branch is an imaginary pair whose sign follows roundoff. Leave this at 0 unless the Frobenius start itself is under test.
  - `nn_low::Int` - Lower bound for toroidal modes
  - `nn_high::Int` - Upper bound for toroidal modes
  - `delta_mlow::Int` - Expands lower bound of Fourier harmonics by delta_mlow
  - `delta_mhigh::Int` - Expands upper bound of Fourier harmonics by delta_mhigh
  - `singfac_min::Float64` - Fractional distance from rational q at which ideal jump condition is enforced. The default 1e-4 suits the Riccati Δ′, whose asymptotic matching gains a spurious imaginary part as it shrinks; `"forward"` runs feeding PerturbedEquilibrium need about 1e-6 for the singular coupling to converge at the outer rational surfaces.
  - `truncate_at_rational_offset::Bool` - Truncate the integration domain at `(last_rational_q + rational_offset_fraction) / n` rather than at `qhigh` / `psihigh`. Fortran STRIDE found that truncating ~20 % above the outermost rational (`rational_offset_fraction = 0.2`) avoids a numerical kink instability in δW that appears when the integration ends too close to or just below a rational surface. **For diverted equilibria where q → ∞ at the separatrix** (e.g. DIII-D geqdsks, the bulk of production use) this costs negligible physical domain because rationals get arbitrarily dense near the LCFS — `truncate_at_rational_offset = true` is the safe and recommended default. **For limited circular / analytical equilibria with finite q at the edge** (Solovev, LAR scans), rationals are sparse and 20 % above the last rational chops off too much edge, so set `truncate_at_rational_offset = false` and let `qhigh` / `psihigh` control the truncation. Multi-`n` runs are not supported by this truncation (the "outermost rational + rational_offset_fraction / n" depends on which `n`); when `truncate_at_rational_offset = true` with `nn_low != nn_high`, `sing_lim!` warns and falls back to `qhigh` / `psihigh`. Default `true`.
  - `psilim_from_layer_overlap::Union{Bool,InnerLayer.SLAYERModel}` - Cap the integration domain where adjacent rational surfaces' resistive layers overlap, after Fitzpatrick, Nucl. Fusion 2025 Sect. 5.9: past that point no surface retains a well-separated inner region. Applied as an upper bound on `qlim` at the top of `sing_lim!`, so `rational_offset_fraction` / `qhigh` still select the final surface from inside it. `true` scans with the default `SLAYERModel()` layer closures; a `SLAYERModel(; ...)` scans with its closures (a deck takes them from its `[SLAYER]` section). The plasma is the equilibrium's kinetic data, which the scan requires; the scan runs only when the cap is on, and the result carries it as `layer_overlap`. With the shipped DIII-D-like profiles the overlap point (ψ ≈ 0.9985) lies outside the 0.995 decks' domain but inside the SLAYER deck's 0.9995. See the integration-domain truncation section of the stability documentation. Default `false`.
  - `rational_offset_fraction::Float64` - Distance beyond last rational surface (normalised ∈ [0,1) in units of 1/n). Only used when `truncate_at_rational_offset` is true. Fortran STRIDE convention is 0.2 (truncate 20 % of one rational-surface spacing above the last surface), retained here.
  - `frobenius_order::Int` - Order of singular layer (Frobenius) expansion at rational surfaces. Default 6 (Fortran STRIDE convention for Δ' calculations; lower values trade accuracy for speed).
  - `qhigh::Float64` - Integration terminated at q limit determined by minimum of qhigh and qa from equil
  - `kinetic_source::String` - Kinetic matrix source: "fixed" (X-shaped test matrices scaled by kinetic_factor relative to ideal matrix Frobenius norms; Ak, Dk, Hk Hermitian, Bk, Ck, Ek non-Hermitian), "calculated" (the drift-kinetic matrices of the KineticForces model, evaluated on the `[KineticForces]` profiles; needs `integrator = "forward"` and is deck-driven through `main`)
  - `kinetic_factor::Float64` - Dimensionless scaling factor for kinetic matrices. Zero (the default) disables the kinetic path; any positive value enables it and scales the kinetic matrices: when kinetic_source="fixed", scales X-shaped test matrices relative to ideal matrix norms; when kinetic_source="calculated", applied as uniform post-hoc multiplier to W and T components.
  - `qlow::Float64` - Integration terminated at q limit determined by minimum of qlow and q0 from equil
  - `dW_edge_scan_start::Float64` - If less than psilim, records a dW(ψ) diagnostic scan over [dW_edge_scan_start, psilim] on odet.edge_scan. The integration domain (psilim) is always controlled by qhigh / psihigh and is not modified by this scan (unless `truncate_at_dW_peak=true`, see caveats below).
  - `truncate_at_dW_peak::Bool` - When `true` and `dW_edge_scan_start < psilim`, the edge-dW scan's peak is adopted as the physical plasma edge, reproducing the ode_record_edge heuristic of Fortran STRIDE. `intr.psilim`/`intr.qlim`/`odet.u` are pulled back to the peak, fixups recorded beyond it are discarded, `q1lim` and the rational-surface list are re-derived at the new edge, and on the Riccati path the Δ' chunks are cut back to it. The result is the same physics as setting `psihigh` to the peak by hand. Δ' still depends on where in the edge band the peak falls, so set this deliberately when you mean to measure against the peak-defined edge (e.g. edge-mode regimes); leave at `false` (default) for the full domain at `qhigh` / `psihigh` / `rational_offset_fraction`.
  - `write_el_matrices::Bool` - Write the Euler-Lagrange matrices (`ForceFreeStates/EulerLagrangeMatrices`: the ideal A, B, C, D, E, H, F, K and G and, on a kinetic run, the kinetic set) sampled on the ψ grid. Each is `numpert_total × numpert_total` complex at every ψ, so the group scales as numpert_total²·npsi and dominates the file size. Default `false`.
"""
@kwdef struct ForceFreeStatesControl{A<:AbstractIntegrator}
    algorithm::A = Riccati()
    vac_flag::Bool = false
    mtheta_vacuum::Int = 480
    nzeta_vacuum::Int = 1
    sing_start::Int = 0
    nn_low::Int = 0
    nn_high::Int = 0
    delta_mlow::Int = 0
    delta_mhigh::Int = 0
    singfac_min::Float64 = 1e-4   # Matches Fortran STRIDE; required nonzero for the Riccati path.
    truncate_at_rational_offset::Bool = true   # Safe default for diverted equilibria (most production use); set false for limited/analytical (LAR, Solovev). Auto-skipped for multi-n. See docstring.
    psilim_from_layer_overlap::Union{Bool,InnerLayer.SLAYERModel} = false  # Cap psilim where adjacent resistive layers overlap. See docstring.
    rational_offset_fraction::Float64 = 0.2
    frobenius_order::Int = 6
    qhigh::Float64 = 1e3
    kinetic_source::String = "fixed"
    kinetic_factor::Float64 = 0.0
    qlow::Float64 = 0.0
    dW_edge_scan_start::Float64 = 0.99
    truncate_at_dW_peak::Bool = false   # Edge-dW peak becomes new physical edge; Δ' BVP made self-consistent. See docstring.
    write_el_matrices::Bool = false

    # --- DRIVEN (RPEC) outer↔inner asymptotic matching (rmatch match_rpec port) ---
    gal_match_flag::Bool = false    # match the inner layer on the finished solve (coil-driven ξ from Δ′ + inner Δ(Q)); needs Galerkin(rpec_flag=true), or riccati with vac_flag=true
    gal_ideal_flag::Bool = false    # within the match, build the IDEAL solution: skip the inner-layer Δ, use bare coil columns (cout=0). Mirrors Fortran rmatch coil%ideal_flag (the EL reference). eta/rho/rotation ignored.
    gal_inner_solver::String = "ray" # inner-layer Δ backend for the match: "ray" (rotated contour + inward march; recommended for every Q) or "galerkin" (real-axis Hermite FEM; accuracy depends on the surface and arg Q)
    # --- Inner-layer "galerkin" backend overrides (gal_inner_solver = "galerkin"; unset keeps the backend default) ---
    gal_inner_xfac::Union{Nothing,Float64} = nothing  # asymptotic-matching radius multiplier
    gal_inner_nx::Union{Nothing,Int} = nothing        # inner-layer grid cells
    gal_inner_nq::Union{Nothing,Int} = nothing        # quadrature order per cell
    gal_inner_cutoff::Union{Nothing,Int} = nothing    # cells carrying the large solution as driving term
    gal_inner_kmax::Union{Nothing,Int} = nothing      # large-x asymptotic series order
    gal_eta::Vector{Float64} = Float64[]      # per-surface resistivity η (length msing, core→edge); Fortran rmatch `eta`
    gal_rho::Vector{Float64} = Float64[]      # per-surface mass density ρ [kg/m³] (length msing, core→edge); Fortran rmatch `massden`
    gal_rotation::Vector{Float64} = Float64[] # per-surface rotation frequency f [Hz] (length msing, core→edge); forced eigenvalue γ_s = 2πi·n·f. Fortran rmatch `rotation`
    gal_gamma::Float64 = 5 / 3       # ratio of specific heats Γ for the resistive-layer coefficients (ggj_parameters G term)
    frobenius_psi_max::Float64 = 0.0
end

ForceFreeStatesControl(algorithm::A, fields...) where {A<:AbstractIntegrator} = ForceFreeStatesControl{A}(algorithm, fields...)
