"""
    SingType

A mutable struct holding data related to the singular surfaces in the equilibrium.

## Fields

  - `psifac::Float64` - Normalized flux coordinate at the singular surface
  - `rho::Float64` - Radial coordinate (√ψ)
  - `m::Vector{Int}` - Poloidal mode number(s)
  - `n::Vector{Int}` - Toroidal mode number(s)
  - `q::Float64` - Safety factor (= m/n)
  - `q1::Float64` - Derivative of safety factor with respect to ψ
  - `grri::Array{Float64,2}` - Interior Green's function at this surface [2*mthvac, 2*mpert]
  - `grre::Array{Float64,2}` - Exterior Green's function at this surface [2*mthvac, 2*mpert]
  - `delta_prime::Vector{ComplexF64}` - **STUB (not physically valid)**. Per-surface ca-based Δ' estimate retained for future work / debugging only. The physically valid Δ' is `ForceFreeStatesInternal.delta_prime_matrix`, computed via the STRIDE global BVP (Glasser 2018 PoP 25, 032501). Do not use this field for tearing-stability analysis; do not expect agreement with `delta_prime_matrix`.
  - `delta_prime_col::Matrix{ComplexF64}` - **STUB (not physically valid)**. Per-surface ca-based Δ' column retained for future work / debugging only. Shape (numpert_total × n_res_modes); `delta_prime_col[j, i] = (ca_r[j,ipert_res_i,2] - ca_l[j,ipert_res_i,2]) / (4π²·psio)`. The diagonal element matches the (also stubbed) `delta_prime[i]`. Only populated for the Riccati/parallel FM paths. The physically valid Δ' is `ForceFreeStatesInternal.delta_prime_matrix`; this field exists for future development on intra-surface coupling diagnostics, not for production use.
"""
@kwdef mutable struct SingType
    psifac::Float64 = 0.0
    rho::Float64 = 0.0
    m::Vector{Int} = Int[]
    n::Vector{Int} = Int[]
    q::Float64 = 0.0
    q1::Float64 = 0.0
    grri::Array{Float64,2} = Array{Float64}(undef, 0, 0)
    grre::Array{Float64,2} = Array{Float64}(undef, 0, 0)
    delta_prime::Vector{ComplexF64} = ComplexF64[]
    delta_prime_col::Matrix{ComplexF64} = Matrix{ComplexF64}(undef, 0, 0)
    ua_left::Array{ComplexF64,3} = Array{ComplexF64}(undef, 0, 0, 0)   # asymptotic basis at left inner-layer boundary
    ua_right::Array{ComplexF64,3} = Array{ComplexF64}(undef, 0, 0, 0)  # asymptotic basis at right inner-layer boundary
    psi_ua_left::Float64 = 0.0   # ψ where ua_left was evaluated (left inner-layer boundary)
    psi_ua_right::Float64 = 0.0  # ψ where ua_right was evaluated (right inner-layer boundary)
end

"""
    SingAsymptotics

A struct containing asymptotic expansion data for ideal ForceFreeStates calculations at a singular surface.
This data is computed on-demand during singular surface crossings in `cross_ideal_singular_surf!`.

## Fields

  - `alpha::Vector{ComplexF64}` - Resonant matrix eigenvalues
  - `r1::Vector{Int}` - Resonant indices along first index
  - `r2::Vector{Int}` - Resonant indices along second index
  - `n1::Vector{Int}` - Nonresonant indices along first index
  - `n2::Vector{Int}` - Nonresonant indices along second index
  - `power::Vector{ComplexF64}` - Power series coefficients
  - `vmat::Array{ComplexF64,4}` - Power series of V matrix for asymptotic analysis
  - `mmat::Array{ComplexF64,4}` - Power series of M matrix for asymptotic analysis
  - `m0mat::Matrix{ComplexF64}` - Zeroth order M matrix projected onto resonant subspace
"""
struct SingAsymptotics
    sing_order::Int
    alpha::Vector{ComplexF64}
    r1::Vector{Int}
    r2::Vector{Int}
    n1::Vector{Int}
    n2::Vector{Int}
    power::Vector{ComplexF64}
    vmat::Array{ComplexF64,4}
    mmat::Array{ComplexF64,4}
    m0mat::Matrix{ComplexF64}
end

"""
    IntegrationChunk

A struct representing a region of integration in the Euler-Lagrange solver.

## Fields

  - `psi_start::Float64` - Starting ψ coordinate for this integration region
  - `psi_end::Float64` - Ending ψ coordinate for this integration region
  - `needs_crossing::Bool` - Whether a rational surface crossing is needed after this chunk
  - `ising::Int` - Index of the singular surface associated with this chunk (0 if none)
  - `direction::Int` - Integration direction: +1 forward (axis→edge), -1 backward (edge→axis).
    For `direction=-1` chunks, `psi_start` < `psi_end` but integration proceeds from `psi_end`
    toward `psi_start`. The resulting propagator maps state at `psi_end` → state at `psi_start`.
    Used in bidirectional parallel FM to produce well-conditioned crossing-chunk propagators:
    solutions that grow exponentially forward (toward a singularity) decay when integrated
    backward, so the backward propagator is well-conditioned.
"""
@kwdef struct IntegrationChunk
    psi_start::Float64
    psi_end::Float64
    needs_crossing::Bool
    ising::Int = 0
    direction::Int = 1   # +1 forward, -1 backward
end

"""
    ChunkPropagator

Fundamental matrix for one integration chunk, stored as two N×N×2 solution blocks.
Represents the propagator Φ(ψ₂,ψ₁) computed by integrating the EL ODE from two
identity-block initial conditions:

  - `block_upper_ic`: result of integrating with IC = (I_N, 0_N)  (U₁ = I, U₂ = 0)
  - `block_lower_ic`: result of integrating with IC = (0_N, I_N)  (U₁ = 0, U₂ = I)

Applying the propagator to the current state `u_prev`:

  u₁_new = block_upper_ic[:,:,1] · u₁_prev + block_lower_ic[:,:,1] · u₂_prev
  u₂_new = block_upper_ic[:,:,2] · u₁_prev + block_lower_ic[:,:,2] · u₂_prev

Since each chunk starts from a bounded identity IC (rather than the accumulated state),
exponential growth within a chunk does not affect the conditioning of the overall
assembly. This enables `Threads.@threads` parallel integration across all chunks.
"""
struct ChunkPropagator
    block_upper_ic::Array{ComplexF64,3}   # shape (N, N, 2) — result from IC = (I, 0)
    block_lower_ic::Array{ComplexF64,3}   # shape (N, N, 2) — result from IC = (0, I)
end
ChunkPropagator(N::Int) = ChunkPropagator(zeros(ComplexF64, N, N, 2), zeros(ComplexF64, N, N, 2))

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
  - `mband::Int` - Bandwidth for matrix operations (mpert - 1 - delta_mband)
  - `nlow::Int` - Lowest toroidal mode number
  - `nhigh::Int` - Highest toroidal mode number
  - `npert::Int` - Number of toroidal modes (nhigh - nlow + 1)
  - `numpert_total::Int` - Total number of modes (mpert × npert)
  - `keq_out::Bool` - Flag to output equilibrium quantities (not yet implemented)
  - `theta_out::Bool` - Flag to output theta coordinate data (not yet implemented)
  - `xlmda_out::Bool` - Flag to output eigenvalue data (not yet implemented)
  - `sol_base::Int` - Base index for solution vectors (not yet implemented)
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
  - `locstab::CubicSeriesInterpolant` - Spline for local stability analysis
  - `wall_settings::Vacuum.WallShapeSettings` - Wall shape settings for vacuum calculations
"""
@kwdef mutable struct ForceFreeStatesInternal
    dir_path::String = ""
    mlow::Int = 0
    mhigh::Int = 0
    mpert::Int = 0
    mband::Int = 0
    nlow::Int = 0
    nhigh::Int = 0
    npert::Int = 0
    numpert_total::Int = 0
    keq_out::Bool = false
    theta_out::Bool = false
    xlmda_out::Bool = false
    sol_base::Int = 50
    msing::Int = 0
    kmsing::Int = 0
    sing::Vector{SingType} = SingType[]
    kinsing::Vector{SingType} = SingType[]
    kinsing_scan_psi::Vector{Float64} = Float64[]
    kinsing_scan_cond::Vector{Float64} = Float64[]
    kinsing_scan_threshold::Float64 = 0.0
    psilim::Float64 = 0.0
    qlim::Float64 = 0.0
    q1lim::Float64 = 0.0
    locstab::FastInterpolations.CubicSeriesInterpolant = cubic_interp(collect(0.0:0.25:1.0), Series(zeros(5, 5)); bc=ZeroCurvBC())
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
    small-solution response at each surface side to driving edge poloidal mode k, built by
    looping the Eq. (37) edge boundary condition through the Riccati BVP (`loop_edge_boundary_conditions`).
    """
    delta_coil_matrix::Matrix{ComplexF64} = Matrix{ComplexF64}(undef, 0, 0)
end

"""
    ForceFreeStatesControl

A mutable struct containing control parameters for stability analysis, set by the user in gpec.toml.

## Fields

  - `verbose::Bool` - Enable verbose output
  - `bal_flag::Bool` - Enable ballooning mode analysis
  - `mat_flag::Bool` - Enable matrix output
  - `ode_flag::Bool` - Enable ODE integration diagnostics
  - `vac_flag::Bool` - Enable vacuum region calculation
  - `mer_flag::Bool` - Enable Mercier stability criterion
  - `mthvac::Int` - Number of vacuum poloidal grid points (corresponds to `mtheta` in VacuumInput)
  - `nzvac::Int` - Number of vacuum toroidal grid points (corresponds to `nzeta` in VacuumInput3D)
  - `sing_start::Int` - Start integration at the `sing_start`-th singular surface
  - `nn_low::Int` - Lower bound for toroidal modes
  - `nn_high::Int` - Upper bound for toroidal modes
  - `delta_mlow::Int` - Expands lower bound of Fourier harmonics by delta_mlow
  - `delta_mhigh::Int` - Expands upper bound of Fourier harmonics by delta_mhigh
  - `delta_mband::Int` - Integration keeps only this wide a band of solutions along the diagonal in m,m'
  - `thmax0::Float64` - Maximum integration step size (not yet implemented)
  - `nstep::Int` - Maximum number of integration steps (not yet implemented)
  - `ksing::Int` - Singular surface handling parameter
  - `eulerlagrange_tolerance::Float64` - Relative tolerance for ODE integration of Euler-Lagrange equations
  - `ucrit::Float64` - Critical value of unorm ratio to trigger solution normalization. In the standard path it triggers Gaussian reduction; in the Riccati path it triggers `renormalize_riccati_inplace!`. Default `1e4` empirically keeps max(|U₁|, |U₂|) in O(1)–O(10⁴) over the integration domain on DIII-D / Solovev sweeps; lower triggers excess renorms without accuracy gain, higher risks overflow before the next renorm.
  - `numsteps_init::Int` - Initial array size for ODE data storage
  - `numunorms_init::Int` - Initial array size for solution normalization data
  - `singfac_min::Float64` - Fractional distance from rational q at which ideal jump condition is enforced
  - `cyl_flag::Bool` - Make delta_mlow and delta_mhigh set the actual m truncation bounds. Default is to expand (n*qmin-4, n*qmax).
  - `set_psilim_via_dmlim::Bool` - Truncate the integration domain at `(last_rational_q + dmlim) / n` rather than at `qhigh` / `psihigh`. Fortran STRIDE found that truncating ~20 % above the outermost rational (`dmlim = 0.2`) avoids a numerical kink instability in δW that appears when the integration ends too close to or just below a rational surface. **For diverted equilibria where q → ∞ at the separatrix** (e.g. DIII-D geqdsks, the bulk of production use) this costs negligible physical domain because rationals get arbitrarily dense near the LCFS — `set_psilim_via_dmlim = true` is the safe and recommended default. **For limited circular / analytical equilibria with finite q at the edge** (Solovev, LAR scans), rationals are sparse and 20 % above the last rational chops off too much edge, so set `set_psilim_via_dmlim = false` and let `qhigh` / `psihigh` control the truncation. Multi-`n` runs are not supported by this truncation (the "outermost rational + dmlim / n" depends on which `n`); when `set_psilim_via_dmlim = true` with `nn_low != nn_high`, `sing_lim!` warns and falls back to `qhigh` / `psihigh`. Default `true`.
  - `dmlim::Float64` - Distance beyond last rational surface (normalised ∈ [0,1) in units of 1/n). Only used when `set_psilim_via_dmlim` is true. Fortran STRIDE convention is 0.2 (truncate 20 % of one rational-surface spacing above the last surface), retained here.
  - `sing_order::Int` - Order of singular layer (Frobenius) expansion at rational surfaces. Default 6 (Fortran STRIDE convention for Δ' calculations; lower values trade accuracy for speed).
  - `qhigh::Float64` - Integration terminated at q limit determined by minimum of qhigh and qa from equil
  - `kinetic_source::String` - Kinetic matrix source: "fixed" (X-shaped test matrices scaled by kinetic_factor relative to ideal matrix Frobenius norms; Ak, Dk, Hk Hermitian, Bk, Ck, Ek non-Hermitian), "calculated" (PENTRC — not yet implemented)
  - `kinetic_factor::Float64` - Dimensionless scaling factor for kinetic matrices. Zero (the default) disables the kinetic path; any positive value enables it and scales the kinetic matrices: when kinetic_source="fixed", scales X-shaped test matrices relative to ideal matrix norms; when kinetic_source="calculated", applied as uniform post-hoc multiplier to W and T components.
  - `qlow::Float64` - Integration terminated at q limit determined by minimum of qlow and q0 from equil
  - `reform_eq_with_psilim::Bool` - Reform equilibrium with computed psilim (not yet implemented)
  - `psiedge::Float64` - If less than psilim, records a dW(ψ) diagnostic scan over [psiedge, psilim] on odet.edge_scan. The integration domain (psilim) is always controlled by qhigh / psihigh and is not modified by this scan (unless `truncate_at_dW_peak=true`, see caveats below).
  - `truncate_at_dW_peak::Bool` - When `true` and `psiedge < psilim`, the edge-dW scan's peak location is adopted as the new physical plasma edge — `intr.psilim`/`intr.qlim`/`odet.u` are pulled back to the peak, AND the FM Δ' chunks/propagators are made self-consistent with the new boundary (the chunk that straddles the peak is rebuilt + re-integrated; any chunks past the peak are dropped). This reproduces the spirit of the original ode_record_edge heuristic from Fortran STRIDE while keeping Δ' and δW well-defined at the new boundary. The Δ' metric is still physically dependent on where the peak falls in the edge band, so use this flag deliberately when you mean to scan against the peak-defined edge (e.g. for studying edge-mode regimes); leave at `false` (default) for the full-domain Δ' at `qhigh` / `psihigh` / `dmlim`.
  - `diagnose::Bool` - Enable diagnostic output (not yet implemented)
  - `diagnose_ca::Bool` - Enable asymptotic coefficient diagnostics (not yet implemented)
  - `write_outputs_to_HDF5::Bool` - Write results to HDF5 format
  - `HDF5_filename::String` - Name of HDF5 output file
  - `force_wv_symmetry::Bool` - Boolean flag to enforce symmetry in the vacuum response matrix
  - `save_interval::Int` - Save every Nth ODE step (1=all, 10=every 10th). Always saves near rational surfaces. (Same as `euler_step` in the Fortran)
  - `force_termination::Bool` - Terminate after force-free states (skip perturbed equilibrium calculations)
  - `use_riccati::Bool` - Use the dual Riccati reformulation S = U₁·U₂⁻¹ instead of the standard U₁/U₂ ODE. Reduces stiffness for faster integration. See Glasser (2018) Phys. Plasmas 25, 032507.
  - `use_parallel::Bool` - Parallel fundamental matrix (propagator) integration using `Threads.@threads`. Each chunk is integrated independently from identity IC and assembled serially. Requires `singfac_min != 0`. Uses the same chunk bounds as the standard path but sub-divides chunks for load balancing. Crossings use the Riccati-style algorithm (no Gaussian reduction).
  - `parallel_threads::Int` - Cap on the number of threads the parallel BVP uses. **Default `2`** parallelises the FM chunks across two threads (the BVP has ~10 chunks; 2 threads is enough to amortize them — speedup saturates here, raising to 4 adds scheduling overhead). Set `parallel_threads = 1` to run the FM chunks SERIALLY (no `Threads.@threads`), which is bit-deterministic and immune to the thread-schedule sensitivity that historically caused intermittent BVP divergences on numerically delicate equilibria like DIII-D 147131 (see CONVENTIONS.md §7). Empirical reliability sweep (5 trials × {1,2,4} on DIII-D 147131 βₚ≈0.07): 15/15 bit-identical Δ′ at every setting; pt=2 ≈ pt=4 ≈ 20 % faster than serial. If a parallel run diverges, drop to `parallel_threads = 1` rather than switching `use_parallel = false` — the latter is silently wrong. Capped at `Threads.nthreads()`.
  - `populate_dense_xi::Bool` - When `use_parallel = true`, append a serial Euler-Lagrange pass at the end of the propagator BVP and let it replace the `odet` returned to the main pipeline.  This populates `u_store` / `ud_store` densely in the axis (EL) basis — the only convention the PerturbedEquilibrium / FieldReconstruction downstream code consumes correctly.  Without it the parallel path stores only chunk-endpoint Riccati S matrices and zeros for `ud_store` (see Riccati.jl docstring caveats), and HDF5 `integration/xi_psi`/`dxi_psi`/`xi_s` are unusable.  Δ' (`singular/delta_prime_matrix`) is computed from the parallel BVP and is bit-identical between `populate_dense_xi=true` and `false`.  Energies (`vacuum/ep`/`ev`/`et`) are computed by `free_run!` from `odet`, so with `populate_dense_xi=true` they match what a pure serial run (`use_parallel=false`) would produce; with `populate_dense_xi=false` they use the parallel-pass Riccati `odet.u` instead (differs by the ~0.12 % Riccati-vs-axis algorithmic gap on DIIID-class cases).  **Default `false`** to avoid paying the dense-pass cost on Δ'/vacuum/ideal-stability-only runs; **PerturbedEquilibrium-using configs must set `populate_dense_xi = true` explicitly** when `use_parallel = true` (otherwise PE silently reads Riccati-basis garbage).  Auto-disabled when `force_termination = true` regardless of the user setting, since the dense pass has no downstream consumer in that case.  Approximate cost when enabled: one extra serial EL integration (~1× the parallel BVP wall-clock for typical N).
  - `extended_precision_bvp::Bool` - When `true` (default), promote the Δ' BVP linear system to `Complex{Double64}` (~31 digits) for the LU solve and PEST3 combination. Guards against catastrophic cancellation in the PEST3 four-term combination (dp_raw entries can be 10⁴–10⁵× larger than the result; the imaginary part of off-diagonal Δ' is particularly sensitive). Disabling (`false`) saves ~1.5–2× the BVP solve time but on DIIID-class equilibria the imaginary Δ' components can drift by factors of 2–5×; only disable for performance experiments on cases where Float64 has been validated against Double64.
"""
@kwdef mutable struct ForceFreeStatesControl
    verbose::Bool = true
    bal_flag::Bool = false
    mat_flag::Bool = false
    ode_flag::Bool = false
    vac_flag::Bool = false
    mer_flag::Bool = false
    mthvac::Int = 480
    nzvac::Int = 1
    sing_start::Int = 0
    nn_low::Int = 0
    nn_high::Int = 0
    delta_mlow::Int = 0
    delta_mhigh::Int = 0
    delta_mband::Int = 0
    thmax0::Float64 = 1.0
    nstep::Int = typemax(Int)
    ksing::Int = -1
    eulerlagrange_tolerance::Float64 = 1e-8
    ucrit::Float64 = 1e4
    numsteps_init::Int = 4000
    numunorms_init::Int = 100
    singfac_min::Float64 = 1e-4   # Matches Fortran STRIDE; required nonzero for use_parallel path.
    cyl_flag::Bool = false
    set_psilim_via_dmlim::Bool = true   # Safe default for diverted equilibria (most production use); set false for limited/analytical (LAR, Solovev). Auto-skipped for multi-n. See docstring.
    dmlim::Float64 = 0.2
    sing_order::Int = 6
    qhigh::Float64 = 1e3
    kinetic_source::String = "fixed"
    kinetic_factor::Float64 = 0.0
    qlow::Float64 = 0.0
    reform_eq_with_psilim::Bool = false
    psiedge::Float64 = 0.99
    truncate_at_dW_peak::Bool = false   # Edge-dW peak becomes new physical edge; Δ' BVP made self-consistent. See docstring.
    parallel_threads::Int = 2
    diagnose::Bool = false
    diagnose_ca::Bool = false
    write_outputs_to_HDF5::Bool = true
    HDF5_filename::String = "gpec.h5"
    force_wv_symmetry::Bool = true
    save_interval::Int = 3
    force_termination::Bool = false
    use_riccati::Bool = false
    use_parallel::Bool = true    # Default on: unlocks singular/delta_prime_matrix (STRIDE BVP Δ' matrix) used by SLAYER/GGJ downstream.
    populate_dense_xi::Bool = false  # When use_parallel=true, set to true ONLY if a PerturbedEquilibrium pipeline will consume dense ξ. Default false avoids the ~1× parallel-BVP serial-EL re-run for non-PE runs (Δ'/vacuum/ideal-stability only). See ForceFreeStatesControl docstring for the full trade-off (et[1] convention differs by ~0.12% on DIIID between populate=true vs false).
    extended_precision_bvp::Bool = true   # Promote Δ' BVP to Complex{Double64}; default on (Float64 drifts the imaginary Δ' by 2–5× on DIIID-class cases).
    fixed_axis::Bool = false
end

@kwdef mutable struct FourFitVars{S<:CubicSeriesInterpolant,Opts<:NamedTuple}
    mpert::Int
    mband::Int
    numpert_total::Int  # = mpert * npert (total series count per matrix = numpert_total^2)

    # Complex-valued CubicSeriesInterpolant for stability matrices
    # Each matrix is flattened to (npsi × numpert_total^2) series
    # FastInterpolations natively supports complex values: CubicSeriesInterpolant{Tgrid, Tvalue}
    # NOTE: itp_opts must precede interpolant fields — @kwdef evaluates defaults in declaration order
    itp_opts::Opts = (; extrap=ExtendExtrap())

    amats::S = _empty_series_interp_complex(numpert_total^2, itp_opts)
    bmats::S = _empty_series_interp_complex(numpert_total^2, itp_opts)
    cmats::S = _empty_series_interp_complex(numpert_total^2, itp_opts)
    # `dmats_prim`, `emats_prim` are the pre-Schur-reduction geometric forms
    # (D = χ₁·(g23 + q·g33·m/n); E = (-χ₁/n)·(q'·χ₁·g33 - 2π·i·χ₁·g31·singfac + jθ·I)).
    # The `_prim` suffix follows `fmats_prim`. Downstream kinetic FKG Schur complements
    # consume these primitive forms; the alternate singular-layer path that would need
    # kinetic-added overwrites of D and E is not implemented here (see Kinetic.jl).
    dmats_prim::S = _empty_series_interp_complex(numpert_total^2, itp_opts)
    emats_prim::S = _empty_series_interp_complex(numpert_total^2, itp_opts)
    hmats::S = _empty_series_interp_complex(numpert_total^2, itp_opts)
    fmats_lower::S = _empty_series_interp_complex(numpert_total^2, itp_opts)
    fmats_prim::S = _empty_series_interp_complex(numpert_total^2, itp_opts)  # primitive F before Schur complement (for kinetic)
    kmats::S = _empty_series_interp_complex(numpert_total^2, itp_opts)
    gmats::S = _empty_series_interp_complex(numpert_total^2, itp_opts)

    # Ideal A,B,C splines preserved before kinetic overwrite (for mat_flag output)
    amats_ideal::S = _empty_series_interp_complex(numpert_total^2, itp_opts)
    bmats_ideal::S = _empty_series_interp_complex(numpert_total^2, itp_opts)
    cmats_ideal::S = _empty_series_interp_complex(numpert_total^2, itp_opts)

    # Kinetic energy matrix splines: 6 components (A,B,C,D,E,H perturbations)
    kwmats::Vector{S} = [_empty_series_interp_complex(numpert_total^2, itp_opts) for _ in 1:6]
    # Kinetic torque matrix splines: 6 components
    ktmats::Vector{S} = [_empty_series_interp_complex(numpert_total^2, itp_opts) for _ in 1:6]

    # Pre-computed FKG kinetic matrices (populated by make_kinetic_matrix)
    f0mats::S = _empty_series_interp_complex(numpert_total^2, itp_opts)
    pmats::S = _empty_series_interp_complex(numpert_total^2, itp_opts)
    paats::S = _empty_series_interp_complex(numpert_total^2, itp_opts)
    kkmats::S = _empty_series_interp_complex(numpert_total^2, itp_opts)
    kkaats::S = _empty_series_interp_complex(numpert_total^2, itp_opts)
    r1mats::S = _empty_series_interp_complex(numpert_total^2, itp_opts)
    r2mats::S = _empty_series_interp_complex(numpert_total^2, itp_opts)
    r3mats::S = _empty_series_interp_complex(numpert_total^2, itp_opts)
    gaats::S = _empty_series_interp_complex(numpert_total^2, itp_opts)

    # Pre-allocated evaluation buffer for matrix output
    _mat_out::Matrix{ComplexF64} = Matrix{ComplexF64}(undef, numpert_total, numpert_total)

    # Shared hint for sequential evaluation (all splines evaluated at same psi)
    _hint::Base.RefValue{Int} = Ref(1)

    # Used in Free.jl
    jmat::Vector{ComplexF64} = Vector{ComplexF64}(undef, 2 * mband + 1)
end

# Helper to create empty complex series interpolant for default initialization
function _empty_series_interp_complex(n_series::Int)
    xs = collect(range(0.0, 1.0; length=5))
    Y = zeros(ComplexF64, 5, n_series)
    return cubic_interp(xs, Series(Y))
end

function _empty_series_interp_complex(n_series::Int, itp_opts::NamedTuple)
    xs = collect(range(0.0, 1.0; length=5))
    Y = zeros(ComplexF64, 5, n_series)
    return cubic_interp(xs, Series(Y); itp_opts...)
end

# Convenience constructor
FourFitVars(mpert::Int, mband::Int, numpert_total::Int) = FourFitVars(; mpert, mband, numpert_total)

"""
    VacuumData

A struct containing relevant data from the vacuum calculation.
Populated in `Free.jl`.

## Fields

  - `numpoints::Int` - Total number of points in the vacuum calculation (mthvac * nzvac)
  - `numpert_total::Int` - Total number of modes (mpert × npert)
  - `mthvac::Int` - Number of vacuum poloidal grid points (corresponds to `mtheta` in VacuumInput) - only needed for GPEC functionality currently
  - `wt::Array{ComplexF64, 2}` - Free-boundary eigenvector matrix after diagonalising W (numpert_total × numpert_total). Columns are the eigenmodes sorted most-unstable first. **ξ-space.**
  - `wt0::Array{ComplexF64, 2}` - Free-boundary total-energy matrix W = wp + wv before diagonalisation (numpert_total × numpert_total). **ξ-space.**
  - `wp::Array{ComplexF64, 2}` - Plasma energy matrix (numpert_total × numpert_total). **ξ-space.**
  - `wv::Array{ComplexF64, 2}` - Vacuum energy matrix (numpert_total × numpert_total). **ξ-space.**
  - `pn_wt0, pn_wp, pn_wv::Array{ComplexF64,2}` - Power-normalized flux (Φ-space) counterparts of `wt0`, `wp`, `wv` at the plasma edge (see PowerNorm.jl). Jacobian-invariant up to the M†·W·M transform.
  - `pn_wt::Array{ComplexF64,2}` - Φ-space eigenvector matrix of `pn_wt0` (columns sorted most-unstable first, phase-normalized so each column's largest-magnitude entry is real-positive).
  - `ep::Vector{ComplexF64}` - Plasma eigenvalues
  - `ev::Vector{ComplexF64}` - Vacuum eigenvalues
  - `et::Vector{ComplexF64}` - Total eigenvalues of plasma + vacuum
  - `n_tor_idx::Vector{Int}` -  0-based toroidal mode number index of each sorted eigenvalue (numpert_total). Needed in `write_imas`
  - `vacuum_eigenvalue::Float64` - Least stable (minimum) eigenvalue of the vacuum matrix wv, clamped to zero
  - `grri::Array{Float64, 2}` - Interior Green's function matrices (2 * mthvac * nzvac × 2 * numpert_total)
  - `grre::Array{Float64, 2}` - Exterior Green's function matrices (2 * mthvac * nzvac × 2 * numpert_total)
  - `plasma_pts::Array{Float64, 3}` - Cartesian coordinates of plasma points, shape (mthvac * nzvac) × 3 for (x, y, z)
  - `wall_pts::Array{Float64, 3}` - Cartesian coordinates of wall points, shape (mthvac * nzvac) × 3 for (x, y, z)
"""
@kwdef mutable struct VacuumData
    numpoints::Int
    numpert_total::Int
    mthvac::Int # this is only needed to not break GPEC functionality currently

    wt::Array{ComplexF64,2} = Array{ComplexF64}(undef, numpert_total, numpert_total)
    wt0::Array{ComplexF64,2} = Array{ComplexF64}(undef, numpert_total, numpert_total)
    wp::Array{ComplexF64,2} = Array{ComplexF64}(undef, numpert_total, numpert_total)
    wv::Array{ComplexF64,2} = Array{ComplexF64}(undef, numpert_total, numpert_total)
    ep::Vector{ComplexF64} = Vector{ComplexF64}(undef, numpert_total)
    ev::Vector{ComplexF64} = Vector{ComplexF64}(undef, numpert_total)
    et::Vector{ComplexF64} = Vector{ComplexF64}(undef, numpert_total)
    n_tor_idx::Vector{Int} = zeros(Int, numpert_total)
    vacuum_eigenvalue::Float64 = NaN

    # Power-normalized flux eigenvalues (Jacobian-invariant)
    pn_et::Vector{ComplexF64} = fill(complex(NaN), numpert_total)
    pn_ep::Vector{ComplexF64} = fill(complex(NaN), numpert_total)
    pn_ev::Vector{ComplexF64} = fill(complex(NaN), numpert_total)

    # Power-normalized flux matrices at the plasma edge (Jacobian-invariant).
    pn_wt0::Array{ComplexF64,2} = fill(complex(NaN), numpert_total, numpert_total)
    pn_wp::Array{ComplexF64,2} = fill(complex(NaN), numpert_total, numpert_total)
    pn_wv::Array{ComplexF64,2} = fill(complex(NaN), numpert_total, numpert_total)
    pn_wt::Array{ComplexF64,2} = fill(complex(NaN), numpert_total, numpert_total)

    grri::Array{Float64,2} = Array{Float64}(undef, 2 * numpoints, 2 * numpert_total)
    grre::Array{Float64,2} = Array{Float64}(undef, 2 * numpoints, 2 * numpert_total)
    plasma_pts::Array{Float64,2} = Array{Float64}(undef, numpoints, 3)
    wall_pts::Array{Float64,2} = Array{Float64}(undef, numpoints, 3)
end

VacuumData(numpoints::Int, numpert_total::Int, mthvac::Int) = VacuumData(; numpoints, numpert_total, mthvac)

"""
EdgeScanState

Holds the state and results for the edge dW stability scan over ψ ∈ [psiedge, psilim].
Initialized and populated by `findmax_dW_edge!`; results written to HDF5 under `EdgeScan/`
(power-norm values) and `EdgeScan/XiNorm/` (ξ-space values, for benchmarking).

## Fields

  - `wvmat` - Precomputed wv matrix spline (raw, no singfac); singfac applied analytically in `free_compute_total`.
  - `wv_hint::Base.RefValue{Int}` - Search hint for wvmat spline (different grid from equilibrium profiles).
  - `sqrtamat_spline` - Precomputed √A convolution matrix + surface area spline for power-norm eigenvalues.
  - `sqrtamat_hint::Base.RefValue{Int}` - Search hint for sqrtamat spline.
  - `psi, q` - ψ and q values at each edge scan step.
  - `total_eigenvalue, plasma_energy, vacuum_energy, vacuum_eigenvalue` - ξ-space energy components at each step (NaN for failed steps). Kept for Fortran benchmarking.
  - `pn_total_eigenvalue, pn_plasma_energy, pn_vacuum_energy, pn_vacuum_eigenvalue` - Power-normalized flux (Φ-space) energies (Jacobian-invariant; NaN for failed/singular steps). These are the values written to `EdgeScan/` by default.
"""
@kwdef mutable struct EdgeScanState
    numpert_total::Int
    N_edge::Int

    # Vacuum matrix spline and evaluation infrastructure
    wvmat::CubicSeriesInterpolant{Float64,ComplexF64} = _empty_series_interp_complex(numpert_total^2)
    wv_hint::Base.RefValue{Int} = Ref(1)

    # Power-norm sqrtamat + jarea spline (mpert^2 + 1 series: flattened sqrtamat then jarea)
    sqrtamat_spline::CubicSeriesInterpolant{Float64,ComplexF64} = _empty_series_interp_complex(numpert_total^2 + 1)
    sqrtamat_hint::Base.RefValue{Int} = Ref(1)

    # Scan results (written to HDF5 under EdgeScan/; NaN where free_compute_total raised SingularException)
    psi::Vector{Float64} = Vector{Float64}(undef, N_edge)
    q::Vector{Float64} = Vector{Float64}(undef, N_edge)
    total_eigenvalue::Vector{ComplexF64} = fill(complex(NaN), N_edge)
    plasma_energy::Vector{ComplexF64} = fill(complex(NaN), N_edge)
    vacuum_energy::Vector{ComplexF64} = fill(complex(NaN), N_edge)
    vacuum_eigenvalue::Vector{Float64} = fill(NaN, N_edge)

    # Power-normalized flux eigenvalues (Jacobian-invariant)
    pn_total_eigenvalue::Vector{ComplexF64} = fill(complex(NaN), N_edge)
    pn_plasma_energy::Vector{ComplexF64} = fill(complex(NaN), N_edge)
    pn_vacuum_energy::Vector{ComplexF64} = fill(complex(NaN), N_edge)
    pn_vacuum_eigenvalue::Vector{Float64} = fill(NaN, N_edge)
end

EdgeScanState(numpert_total::Int, N_edge::Int) = EdgeScanState(; numpert_total, N_edge)

"""
OdeState

A mutable struct to hold the state of the ODE solver used by the ForceFreeStates integration routines.
This struct stores configuration parameters used to allocate arrays, the evolving stored
solution during integration, diagnostic arrays used for normalization / Gaussian reduction,
and a small set of temporary matrices and factors used to compute singular-layer corrections.

## Fields

  - `numpert_total::Int` - Total number of Fourier mode combinations (m × n) used in the calculation.

  - `numunorms_init::Int` - Initial allocation size for the number of normalization operations recorded.

  - `msing::Int` - Number of singular surfaces in the equilibrium (used to size asymptotic coefficient arrays).

  - `numsteps_init::Int` - Initial allocation size for the number of integration steps to store.

  - `step::Int` - Current integration step index (1-based, like `istep` in the original Fortran).

  - `psi_store::Vector{Float64}` - Stored psi values at each saved integration step (length `numsteps_init`).

  - `q_store::Vector{Float64}` - Stored q values at each saved integration step (length `numsteps_init`).

  - `u_store::Array{ComplexF64,4}` - Stored solution arrays at each saved step with shape
    `(numpert_total, numpert_total, 2, numsteps_init)` (complex solution state used by the solver).

  - `ud_store::Array{ComplexF64,4}` - Stored derivatives of the solution at each saved step with same shape as `u_store`.

  - `crit_store::Vector{Float64}` - Stored crit parameter values (smallest eigenvalue of W⁻ꜝ) (length `numsteps_init`).

  - `ca_r::Array{ComplexF64,4}` - Asymptotic coefficients just to the right of each singular surface
    with shape `(numpert_total, numpert_total, 2, msing)`.

  - `ca_l::Array{ComplexF64,4}` - Asymptotic coefficients just to the left of each singular surface
    with shape `(numpert_total, numpert_total, 2, msing)`.

  - `edge_scan::EdgeScanState` - Edge dW scan state and results. Initialized as a disabled sentinel (N_edge=0) and replaced by `findmax_dW_edge!` when a scan runs.

  - `psifac::Float64` - Current normalized flux coordinate for the integrator.

  - `q::Float64` - Safety factor value at `psifac` (current q during integration).

  - `u::Array{ComplexF64,3}` - Current working solution arrays with shape `(numpert_total, numpert_total, 2)`.

  - `ud::Array{ComplexF64,3}` - Current working solution derivative (different than du) arrays with shape `(numpert_total, numpert_total, 2)`.

  - `ising_start::Int` - Index of the starting singular surface to be crossed during integration.

    # Initialization parameters

  - `psimax::Float64` - Maximum psi value for which the integrator is allowed to run in next integration region.

  - `needs_crossing::Bool` - Flag indicating whether a rational surface needs to be crossed after the current integration region.

  - `nzero::Int` - Count of detected zero crossings (used for diagnostics).

    # Saved data throughout integration

  - `new::Bool` - Flag indicating whether a new `unorm0` should be computed after a fixup.

# Total ODE solver steps taken (all steps, not just saved ones)

  - `unorm::Vector{Float64}` - Current norms of the solution vectors (length `numpert_total`).

  - `unorm0::Vector{Float64}` - Reference/initial norms of the solution vectors (length `numpert_total`).

  - `ifix::Int` - Number of normalization operations performed (index into normalization arrays).

  - `index::Array{Int,2}` - Index matrix used for sorting solution norms with shape `(numpert_total, numunorms_init)`.

  - `sing_flag::Vector{Bool}` - Boolean flags indicating which stored normalizations correspond to singular solutions    # Edge dW scan state and results (disabled sentinel when psiedge >= psilim, i.e. no edge scan)
    (length `numunorms_init`).

  - `zeroed_idx::Vector{Vector{Int}}` - For each ideal rational surface jump, a vector of indices of solutions that were zeroed.    # Data for integrator

  - `fixfac::Array{ComplexF64,3}` - Fix-up factors for Gaussian reduction with shape    # Initialization parameters
    `(numpert_total, numpert_total, numunorms_init)`.

  - `fixstep::Vector{Int64}` - Step indices (psi step positions) at which normalization/fixups were performed (length `numunorms_init`).
"""
@kwdef mutable struct OdeState
    # Initialization parameters
    numpert_total::Int
    numunorms_init::Int
    msing::Int
    numsteps_init::Int

    # Saved data throughout integration
    step::Int = 1
    total_steps::Int = 0  # Total ODE solver steps taken (all steps, not just saved ones)
    psi_store::Vector{Float64} = Vector{Float64}(undef, numsteps_init)
    q_store::Vector{Float64} = Vector{Float64}(undef, numsteps_init)
    u_store::Array{ComplexF64,4} = Array{ComplexF64}(undef, numpert_total, numpert_total, 2, numsteps_init)
    ud_store::Array{ComplexF64,4} = Array{ComplexF64}(undef, numpert_total, numpert_total, 2, numsteps_init)
    crit_store::Vector{Float64} = Vector{Float64}(undef, numsteps_init)
    ca_r::Array{ComplexF64,4} = Array{ComplexF64}(undef, numpert_total, numpert_total, 2, msing)
    ca_l::Array{ComplexF64,4} = Array{ComplexF64}(undef, numpert_total, numpert_total, 2, msing)

    # Edge dW scan state and results (disabled sentinel when psiedge >= psilim, i.e. no edge scan)
    edge_scan::EdgeScanState = EdgeScanState(numpert_total, 0)

    # Data for integrator
    psifac::Float64 = 0.0
    q::Float64 = 0.0
    u::Array{ComplexF64,3} = zeros(ComplexF64, numpert_total, numpert_total, 2)
    ud::Array{ComplexF64,3} = zeros(ComplexF64, numpert_total, numpert_total, 2)
    ising_start::Int = 0
    psimax::Float64 = 0.0
    needs_crossing::Bool = false
    nzero::Int = 0

    # Used for Gaussian reduction
    new::Bool = true
    unorm::Vector{Float64} = zeros(Float64, numpert_total)
    unorm0::Vector{Float64} = zeros(Float64, numpert_total)
    ifix::Int = 0
    index::Array{Int,2} = zeros(Int, numpert_total, numunorms_init)
    sing_flag::Vector{Bool} = falses(numunorms_init)
    zeroed_idx::Vector{Vector{Int}} = [Int[] for _ in 1:numunorms_init]
    fixfac::Array{ComplexF64,3} = zeros(ComplexF64, numpert_total, numpert_total, numunorms_init)
    fixstep::Vector{Int64} = zeros(Int64, numunorms_init)

    # Kinetic workspace arrays: evaluated from kwmats/ktmats splines at current psi
    kwmat::Array{ComplexF64,3} = zeros(ComplexF64, numpert_total, numpert_total, 6)
    ktmat::Array{ComplexF64,3} = zeros(ComplexF64, numpert_total, numpert_total, 6)

    # Shared hint for CubicInterpolant interval search optimization during ODE integration
    # All splines evaluated at the same psi can share this hint for O(1) interval lookups
    spline_hint::Base.RefValue{Int} = Ref(1)
    # Shared 2D hint for CubicInterpolantND (rzphi splines) during ODE integration
    # Tuple of (psi_hint, theta_hint) for O(1) interval lookups in 2D bicubic splines
    rzphi_hint::Tuple{Base.RefValue{Int},Base.RefValue{Int}} = (Ref(1), Ref(1))
    # Per-thread hint for FourFitVars matrix splines (amats/bmats/cmats/fmats_lower/kmats/gmats
    # and kinetic equivalents). Lives on OdeState — which is already cloned per thread in the
    # parallel BVP path — so concurrent sing_der! invocations don't race on a shared Ref.
    ffit_hint::Base.RefValue{Int} = Ref(1)
end

OdeState(numpert_total::Int, numsteps_init::Int, numunorms_init::Int, msing::Int) =
    OdeState(; numpert_total, numsteps_init, numunorms_init, msing)
