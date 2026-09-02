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
  - `gal_basis_output::Bool` - Write the raw Galerkin outer-region basis functions (per-interval, unconstrained at the rationals) under `GalerkinIntegration/Basis/`. Solver internals for development verification, not physics output.
"""
@kwdef mutable struct DebugSettings
    output_benchmark_data::Bool = false
    gal_basis_output::Bool = false
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
  - `wall_settings::Vacuum.WallShapeSettings` - Wall shape settings for vacuum calculations
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
    wall_settings::Vacuum.WallShapeSettings = Vacuum.WallShapeSettings()
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

An immutable struct containing 'ForceFreeStates' parameters set by the user in
gpec.toml.

## Fields

  - `verbose::Bool` - Enable verbose output
  - `local_stability_flag::Bool` - Enable local stability analysis (`D_I` and ballooning)
  - `vac_flag::Bool` - Enable vacuum region calculation
  - `mthvac::Int` - Number of vacuum poloidal grid points (corresponds to `mtheta` in VacuumInput)
  - `nzvac::Int` - Number of vacuum toroidal grid points (corresponds to `nzeta` in VacuumInput3D)
  - `sing_start::Int` - Start integration at the `sing_start`-th singular surface
  - `nn_low::Int` - Lower bound for toroidal modes
  - `nn_high::Int` - Upper bound for toroidal modes
  - `delta_mlow::Int` - Expands lower bound of Fourier harmonics by delta_mlow
  - `delta_mhigh::Int` - Expands upper bound of Fourier harmonics by delta_mhigh
  - `nstep::Int` - Maximum number of integration steps (not yet implemented)
  - `ksing::Int` - Singular surface handling parameter
  - `eulerlagrange_tolerance::Float64` - Relative tolerance for ODE integration of Euler-Lagrange equations
  - `ucrit::Float64` - Critical value of unorm ratio to trigger solution normalization. In the standard path it triggers Gaussian reduction; in the Riccati path it triggers `renormalize_riccati_inplace!`. Default `1e4` empirically keeps max(|U₁|, |U₂|) in O(1)–O(10⁴) over the integration domain on DIII-D / Solovev sweeps; lower triggers excess renorms without accuracy gain, higher risks overflow before the next renorm.
  - `numsteps_init::Int` - Initial array size for ODE data storage
  - `numunorms_init::Int` - Initial array size for solution normalization data
  - `singfac_min::Float64` - Fractional distance from rational q at which ideal jump condition is enforced
  - `set_psilim_via_dmlim::Bool` - Truncate the integration domain at `(last_rational_q + dmlim) / n` rather than at `qhigh` / `psihigh`. Fortran STRIDE found that truncating ~20 % above the outermost rational (`dmlim = 0.2`) avoids a numerical kink instability in δW that appears when the integration ends too close to or just below a rational surface. **For diverted equilibria where q → ∞ at the separatrix** (e.g. DIII-D geqdsks, the bulk of production use) this costs negligible physical domain because rationals get arbitrarily dense near the LCFS — `set_psilim_via_dmlim = true` is the safe and recommended default. **For limited circular / analytical equilibria with finite q at the edge** (Solovev, LAR scans), rationals are sparse and 20 % above the last rational chops off too much edge, so set `set_psilim_via_dmlim = false` and let `qhigh` / `psihigh` control the truncation. Multi-`n` runs are not supported by this truncation (the "outermost rational + dmlim / n" depends on which `n`); when `set_psilim_via_dmlim = true` with `nn_low != nn_high`, `sing_lim!` warns and falls back to `qhigh` / `psihigh`. Default `true`.
  - `dmlim::Float64` - Distance beyond last rational surface (normalised ∈ [0,1) in units of 1/n). Only used when `set_psilim_via_dmlim` is true. Fortran STRIDE convention is 0.2 (truncate 20 % of one rational-surface spacing above the last surface), retained here.
  - `sing_order::Int` - Order of singular layer (Frobenius) expansion at rational surfaces. Default 6 (Fortran STRIDE convention for Δ' calculations; lower values trade accuracy for speed).
  - `qhigh::Float64` - Integration terminated at q limit determined by minimum of qhigh and qa from equil
  - `kinetic_source::String` - Kinetic matrix source: "fixed" (X-shaped test matrices scaled by kinetic_factor relative to ideal matrix Frobenius norms; Ak, Dk, Hk Hermitian, Bk, Ck, Ek non-Hermitian), "calculated" (PENTRC — not yet implemented)
  - `kinetic_factor::Float64` - Dimensionless scaling factor for kinetic matrices. Zero (the default) disables the kinetic path; any positive value enables it and scales the kinetic matrices: when kinetic_source="fixed", scales X-shaped test matrices relative to ideal matrix norms; when kinetic_source="calculated", applied as uniform post-hoc multiplier to W and T components.
  - `qlow::Float64` - Integration terminated at q limit determined by minimum of qlow and q0 from equil
  - `psiedge::Float64` - If less than psilim, records a dW(ψ) diagnostic scan over [psiedge, psilim] on odet.edge_scan. The integration domain (psilim) is always controlled by qhigh / psihigh and is not modified by this scan (unless `truncate_at_dW_peak=true`, see caveats below).
  - `truncate_at_dW_peak::Bool` - When `true` and `psiedge < psilim`, the edge-dW scan's peak location is adopted as the new physical plasma edge — `intr.psilim`/`intr.qlim`/`odet.u` are pulled back to the peak, AND the FM Δ' chunks/propagators are made self-consistent with the new boundary (the chunk that straddles the peak is rebuilt + re-integrated; any chunks past the peak are dropped). This reproduces the spirit of the original ode_record_edge heuristic from Fortran STRIDE while keeping Δ' and δW well-defined at the new boundary. The Δ' metric is still physically dependent on where the peak falls in the edge band, so use this flag deliberately when you mean to scan against the peak-defined edge (e.g. for studying edge-mode regimes); leave at `false` (default) for the full-domain Δ' at `qhigh` / `psihigh` / `dmlim`.
  - `diagnose::Bool` - Enable diagnostic output (not yet implemented)
  - `diagnose_ca::Bool` - Enable asymptotic coefficient diagnostics (not yet implemented)
  - `write_outputs_to_HDF5::Bool` - Write results to HDF5 format
  - `HDF5_filename::String` - Name of HDF5 output file
  - `save_interval::Int` - Save every Nth ODE step (1=all, 10=every 10th). Always saves near rational surfaces. (Same as `euler_step` in the Fortran)
  - `force_termination::Bool` - Terminate after force-free states (skip perturbed equilibrium calculations)
  - `integrator::String` - Which formalism integrates the Euler-Lagrange system. `"forward"` sweeps the plasma serially with Gaussian reduction and returns `u_store` / `du_store` / `xi_s_store` dense in the axis (EL) basis — the only convention PerturbedEquilibrium and FieldReconstruction consume correctly, and the only path that supports `kinetic_factor > 0`. `"riccati"` (default) runs the chunked fundamental-matrix propagator driver (Glasser 2018 Phys. Plasmas 25, 032507): chunks are integrated independently from identity initial conditions and assembled serially with Riccati-style crossings, which is the only way to obtain the singular-surface Δ' matrix for the tearing-mode solvers downstream, but leaves `u_store` as sparse chunk-endpoint Riccati states, so dense ξ profiles are unavailable. `"galerkin"` solves the same Euler-Lagrange system variationally instead of by radial ODE integration — the RDCON outer-region singular Galerkin method (Glasser, Wang & Park 2016 Phys. Plasmas 23, 112506), which discretizes the displacement on packed Hermite-cubic elements and solves one global banded system — producing the resistive Δ′ matrix and, when `gal_match_flag` is set, the RPEC inner-layer-matched ξ; it computes its own vacuum response and returns no free-boundary energies, and does not support `kinetic_factor > 0`. Requires `singfac_min != 0` for `"riccati"`.
  - `nchunks::Int` - Target number of Riccati integration chunks. `0` (the default) derives the count from problem structure alone: `max(2·msing + 3, 8·(msing + 1) + msing)`, enough sub-chunks per segment to keep the accumulated propagator products well-conditioned. An explicit value below `2·msing + 3` is clamped up with a warning. Chunk sizing never consults `Threads.nthreads()`, so Riccati outputs are identical whatever thread count `julia -t` provides; threads only change wall-clock.
  - `extended_precision_bvp::Bool` - When `true` (default), promote the Δ' BVP linear system to `Complex{Double64}` (~31 digits) for the LU solve and PEST3 combination. Guards against catastrophic cancellation in the PEST3 four-term combination (dp_raw entries can be 10⁴–10⁵× larger than the result; the imaginary part of off-diagonal Δ' is particularly sensitive). Disabling (`false`) saves ~1.5–2× the BVP solve time but on DIIID-class equilibria the imaginary Δ' components can drift by factors of 2–5×; only disable for performance experiments on cases where Float64 has been validated against Double64.
  - `fixed_axis::Bool` - Axis initial condition of the Euler-Lagrange integration. `true` (default) starts from U₁ = 0, U₂ = I at `psilow`, i.e. ξ^ψ = 0 at the magnetic axis — the condition used by Fortran DCON (`dcon/ode.f`, `ode_axis_init`) and by Glasser 2016 §VI. `false` starts from the free-axis Frobenius state returned by [`compute_axis_init`](@ref) (Glasser 2016 Eq. 51). The Frobenius state is meant to reduce to the [0, I] limit as `psilow → 0`, but at the `psilow` values used in practice (1e-2 … 1e-4) it returns U₁ entries of order 0.1–10 with mixed signs. Started from that state, the signed critical eigenvalue of W_p⁻¹ (the DCON `crit`) is displaced from the axis onward and never recovers: one eigenvalue of the plasma response matrix W_p stays ~10× off its DCON value in every equilibrium tested, and where it crosses zero the free-boundary energy `et[1]` reports a spurious −10²…−10⁵ "instability" (diverted DIII-D-like n = 1 scans, verified against Fortran DCON on identical geqdsks). With `true` the `crit(ψ)` trace, W_p spectrum and `et` agree with DCON. The Δ′ BVP is unaffected either way (≤ 1e-4 relative). Set `false` only to reproduce pre-existing free-axis results.
"""
@kwdef struct ForceFreeStatesControl
    verbose::Bool = true
    local_stability_flag::Bool = false
    vac_flag::Bool = false
    mthvac::Int = 480
    nzvac::Int = 1
    sing_start::Int = 0
    nn_low::Int = 0
    nn_high::Int = 0
    delta_mlow::Int = 0
    delta_mhigh::Int = 0
    nstep::Int = typemax(Int)
    ksing::Int = -1
    eulerlagrange_tolerance::Float64 = 1e-8
    ucrit::Float64 = 1e4
    numsteps_init::Int = 4000
    numunorms_init::Int = 100
    singfac_min::Float64 = 1e-4   # Matches Fortran STRIDE; required nonzero for the Riccati path.
    set_psilim_via_dmlim::Bool = true   # Safe default for diverted equilibria (most production use); set false for limited/analytical (LAR, Solovev). Auto-skipped for multi-n. See docstring.
    dmlim::Float64 = 0.2
    sing_order::Int = 6
    qhigh::Float64 = 1e3
    kinetic_source::String = "fixed"
    kinetic_factor::Float64 = 0.0
    qlow::Float64 = 0.0
    psiedge::Float64 = 0.99
    truncate_at_dW_peak::Bool = false   # Edge-dW peak becomes new physical edge; Δ' BVP made self-consistent. See docstring.
    diagnose::Bool = false
    diagnose_ca::Bool = false
    write_outputs_to_HDF5::Bool = true
    HDF5_filename::String = "gpec.h5"
    save_interval::Int = 3
    force_termination::Bool = false
    integrator::String = "riccati"   # Default: unlocks SingularSurfaces/Delta_prime_matrix (STRIDE BVP Δ′ matrix) used by SLAYER/GGJ downstream. Use "forward" for dense ξ (PerturbedEquilibrium) or kinetic runs.
    nchunks::Int = 0                 # Riccati chunk-count target; 0 = auto (derived from msing alone, never from Threads.nthreads()).
    extended_precision_bvp::Bool = true   # Promote Δ' BVP to Complex{Double64}; default on (Float64 drifts the imaginary Δ' by 2–5× on DIIID-class cases).

    # --- RDCON outer-region Galerkin Δ′ solver (gal_solve port); selected by integrator = "galerkin" ---
    gal_solver::String = "LU"       # "LU" (zgbtrf/zgbtrs) or "cholesky" (zpbtrf/zpbtrs)
    gal_nx::Int = 256               # elements per interval between singular surfaces
    gal_nq::Int = 6                 # Gauss-Lobatto quadrature order per element
    gal_pfac::Float64 = 0.001       # grid packing ratio near singular surfaces
    gal_dx0::Float64 = 5e-4         # resonant-element integration truncation distance (×1/|n q'|)
    gal_dx1::Float64 = 1e-3         # resonant-element size (×1/|n q'|)
    gal_dx2::Float64 = 1e-3         # extension-element size (×1/|n q'|)
    gal_cutoff::Int = 10            # # of elements carrying the large solution as driving term
    gal_tol::Float64 = 1e-10        # resonant-quadrature (QuadGK) tolerance
    gal_gnstep::Int = 20000         # max resonant-quadrature evaluations (QuadGK maxevals in gal_resonant!)
    gal_dx1dx2_flag::Bool = true    # enable special dx1/dx2 treatment for resonant/extension elements
    gal_sing_order::Int = 6         # base power-series order for the Galerkin singular asymptotics
    gal_sing_order_ceiling::Bool = true  # auto-raise order by ceil(2·Re(α)) per surface (high Mercier index)
    gal_rpec_flag::Bool = false     # append mpert coil-response columns to the Δ′ solve (RDCON rpec_flag): unit boundary sources whose plasma response is recorded; needed for the driven (resistive perturbed-equilibrium) Δ_gw
    gal_edge_onesided::Bool = false # pack the two end intervals one-sided toward their single rational end (vs the Fortran symmetric "both" pack); avoids the fine edge cell that inflates cond(A). Default false = faithful to gal.f.
    # --- DRIVEN (RPEC) outer↔inner asymptotic matching (rmatch match_rpec port) ---
    gal_match_flag::Bool = false    # enable the RPEC inner-layer matching: solve the coil-driven matched ξ(ψ) from the gal Δ′ + the inner-layer Δ(Q). Requires gal_rpec_flag=true.
    gal_ideal_flag::Bool = false    # within the match, build the IDEAL solution: skip the inner-layer Δ, use bare coil columns (cout=0). Mirrors Fortran rmatch coil%ideal_flag (the EL reference). eta/rho/rotation ignored.
    gal_inner_solver::String = "ray" # inner-layer Δ backend for the match: "ray" (rotated-contour collocation, certified Δ at the optimal θ = arg(Q)/4; robust for |Q| ≳ 1) or "galerkin" (Hermite-cubic inps; drifts for |Q| ≳ 1)
    # --- Inner-layer "galerkin" backend knobs (used only when gal_inner_solver = "galerkin") ---
    gal_inner_xfac::Float64 = 10.0  # asymptotic-matching radius multiplier (inps_xfac: xmax × 10)
    gal_inner_nx::Int = 1280        # inner-layer grid cells (128 · xfac in the reference)
    gal_inner_nq::Int = 5           # quadrature order per cell
    gal_inner_cutoff::Int = 5       # cells carrying the large solution as driving term
    gal_inner_kmax::Int = 8         # large-x asymptotic series order (↔ order_pow)
    gal_eta::Vector{Float64} = Float64[]      # per-surface resistivity η (length msing, core→edge); Fortran rmatch `eta`
    gal_rho::Vector{Float64} = Float64[]      # per-surface mass density ρ [kg/m³] (length msing, core→edge); Fortran rmatch `massden`
    gal_rotation::Vector{Float64} = Float64[] # per-surface rotation frequency f [Hz] (length msing, core→edge); forced eigenvalue γ_s = 2πi·n·f. Fortran rmatch `rotation`
    gal_gamma::Float64 = 5 / 3       # ratio of specific heats Γ for the resistive-layer coefficients (resist_eval G term)
    fixed_axis::Bool = true          # DCON axis condition U₁=0, U₂=I (ξ^ψ=0 at the axis). `false` = Frobenius free-axis init (compute_axis_init), opt-in only; see docstring.
end
