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
"""
@kwdef struct IntegrationChunk
    psi_start::Float64
    psi_end::Float64
    needs_crossing::Bool
    ising::Int = 0
end

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
  - `fft_flag::Bool` - Enable Fourier transform analysis
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
  - `ucrit::Float64` - Critical value of unorm ratio to trigger solution normalization
  - `numsteps_init::Int` - Initial array size for ODE data storage
  - `numunorms_init::Int` - Initial array size for solution normalization data
  - `singfac_min::Float64` - Fractional distance from rational q at which ideal jump condition is enforced
  - `cyl_flag::Bool` - Make delta_mlow and delta_mhigh set the actual m truncation bounds. Default is to expand (n*qmin-4, n*qmax).
  - `sing_order::Int` - Order of singular layer expansion
  - `qhigh::Float64` - Integration terminated at q limit determined by minimum of qhigh and qa from equil
  - `kinetic_source::String` - Kinetic matrix source: "fixed" (X-shaped test matrices scaled by kinetic_factor relative to ideal matrix Frobenius norms; Ak, Dk, Hk Hermitian, Bk, Ck, Ek non-Hermitian), "calculated" (PENTRC — not yet implemented)
  - `kinetic_factor::Float64` - Dimensionless scaling factor for kinetic matrices. Zero (the default) disables the kinetic path; any positive value enables it and scales the kinetic matrices: when kinetic_source="fixed", scales X-shaped test matrices relative to ideal matrix norms; when kinetic_source="calculated", applied as uniform post-hoc multiplier to W and T components.
  - `qlow::Float64` - Integration terminated at q limit determined by minimum of qlow and q0 from equil
  - `reform_eq_with_psilim::Bool` - Reform equilibrium with computed psilim (not yet implemented)
  - `psiedge::Float64` - If less then psilim, calculates dW(psi) between psiedge and psilim, then runs with truncation at max(dW)
  - `parallel_threads::Int` - Number of parallel threads (not yet implemented)
  - `diagnose::Bool` - Enable diagnostic output (not yet implemented)
  - `diagnose_ca::Bool` - Enable asymptotic coefficient diagnostics (not yet implemented)
  - `write_outputs_to_HDF5::Bool` - Write results to HDF5 format
  - `HDF5_filename::String` - Name of HDF5 output file
  - `force_wv_symmetry::Bool` - Boolean flag to enforce symmetry in the vacuum response matrix
  - `save_interval::Int` - Save every Nth ODE step (1=all, 10=every 10th). Always saves near rational surfaces. (Same as `euler_step` in the Fortran)
  - `force_termination::Bool` - Terminate after force-free states (skip perturbed equilibrium calculations)
"""
@kwdef mutable struct ForceFreeStatesControl
    verbose::Bool = true
    bal_flag::Bool = false
    mat_flag::Bool = false
    ode_flag::Bool = false
    vac_flag::Bool = false
    mer_flag::Bool = false
    fft_flag::Bool = false
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
    eulerlagrange_tolerance::Float64 = 1e-7
    ucrit::Float64 = 1e4
    numsteps_init::Int = 4000
    numunorms_init::Int = 100
    singfac_min::Float64 = 0.0
    cyl_flag::Bool = false
    sing_order::Int = 2
    qhigh::Float64 = 1e3
    kinetic_source::String = "fixed"
    kinetic_factor::Float64 = 0.0
    qlow::Float64 = 0.0
    reform_eq_with_psilim::Bool = false
    psiedge::Float64 = 0.99
    parallel_threads::Int = 1
    diagnose::Bool = false
    diagnose_ca::Bool = false
    write_outputs_to_HDF5::Bool = true
    HDF5_filename::String = "gpec.h5"
    force_wv_symmetry::Bool = true
    save_interval::Int = 3
    force_termination::Bool = false
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
  - `wt::Array{ComplexF64, 2}` - Toroidal vacuum response matrix (numpert_total × numpert_total)
  - `wt0::Array{ComplexF64, 2}` - Reference toroidal vacuum matrix (numpert_total × numpert_total)
  - `wv::Array{ComplexF64, 2}` - Vacuum energy matrix (numpert_total × numpert_total)
  - `ep::Vector{ComplexF64}` - Plasma eigenvalues
  - `ev::Vector{ComplexF64}` - Vacuum eigenvalues
  - `et::Vector{ComplexF64}` - Total eigenvalues of plasma + vacuum
  - `n_tor_idx::Vector{Int}` -  0-based toroidal mode number index of each sorted eigenvalue (numpert_total). Needed in `write_imas`
  - `vacuum_eigenvalue::Float64` - Least stable (minimum) eigenvalue of the vacuum matrix wv, clamped to zero
  - `grri::Array{Float64, 2}` - Interior Green's function matrices (2 * mthvac * nzvac × 2 * numpert_total)
  - `grre::Array{Float64, 2}` - Exterior Green's function matrices (2 * mthvac * nzvac × 2 * numpert_total)
  - `plasma_pts::Array{Float64, 3}` - Cartesian coordinates of plasma points [x, y, z] (mthvac * nzvac × 3)
  - `wall_pts::Array{Float64, 3}` - Cartesian coordinates of wall points [x, y, z] (mthvac * nzvac × 3)
"""
@kwdef mutable struct VacuumData
    numpoints::Int
    numpert_total::Int
    mthvac::Int # this is only needed to not break GPEC functionality currently

    wt::Array{ComplexF64,2} = Array{ComplexF64}(undef, numpert_total, numpert_total)
    wt0::Array{ComplexF64,2} = Array{ComplexF64}(undef, numpert_total, numpert_total)
    wv::Array{ComplexF64,2} = Array{ComplexF64}(undef, numpert_total, numpert_total)
    ep::Vector{ComplexF64} = Vector{ComplexF64}(undef, numpert_total)
    ev::Vector{ComplexF64} = Vector{ComplexF64}(undef, numpert_total)
    et::Vector{ComplexF64} = Vector{ComplexF64}(undef, numpert_total)
    n_tor_idx::Vector{Int} = zeros(Int, numpert_total)
    vacuum_eigenvalue::Float64 = NaN
    grri::Array{Float64,2} = Array{Float64}(undef, 2 * numpoints, 2 * numpert_total)
    grre::Array{Float64,2} = Array{Float64}(undef, 2 * numpoints, 2 * numpert_total)
    plasma_pts::Array{Float64,2} = Array{Float64}(undef, numpoints, 3)
    wall_pts::Array{Float64,2} = Array{Float64}(undef, numpoints, 3)
end

VacuumData(numpoints::Int, numpert_total::Int, mthvac::Int) = VacuumData(; numpoints, numpert_total, mthvac)

"""
EdgeScanState

Holds the state and results for the edge dW stability scan over ψ ∈ [psiedge, psilim].
Initialized and populated by `findmax_dW_edge!`; results written to HDF5 under `edge_scan/`.

## Fields

  - `wvmat` - Precomputed wv matrix spline (raw, no singfac); singfac applied analytically in `free_compute_total`.
  - `wv_hint::Base.RefValue{Int}` - Search hint for wvmat spline (different grid from equilibrium profiles).
  - `psi, q` - ψ and q values at each edge scan step.
  - `total_eigenvalue, plasma_energy, vacuum_energy, vacuum_eigenvalue` - Energy components at each step (NaN for failed steps).
"""
@kwdef mutable struct EdgeScanState
    numpert_total::Int
    N_edge::Int

    # Vacuum matrix spline and evaluation infrastructure
    wvmat::CubicSeriesInterpolant{Float64,ComplexF64} = _empty_series_interp_complex(numpert_total^2)
    wv_hint::Base.RefValue{Int} = Ref(1)

    # Scan results (written to HDF5 under edge_scan/; NaN where free_compute_total raised SingularException)
    psi::Vector{Float64} = Vector{Float64}(undef, N_edge)
    q::Vector{Float64} = Vector{Float64}(undef, N_edge)
    total_eigenvalue::Vector{ComplexF64} = fill(complex(NaN), N_edge)
    plasma_energy::Vector{ComplexF64} = fill(complex(NaN), N_edge)
    vacuum_energy::Vector{ComplexF64} = fill(complex(NaN), N_edge)
    vacuum_eigenvalue::Vector{Float64} = fill(NaN, N_edge)
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
end

OdeState(numpert_total::Int, numsteps_init::Int, numunorms_init::Int, msing::Int) =
    OdeState(; numpert_total, numsteps_init, numunorms_init, msing)
