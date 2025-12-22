"""
    SingType

A mutable struct containing data for singular surfaces in the plasma stability analysis.

## Fields

  - `psifac::Float64` - Normalized flux coordinate at the singular surface
  - `rho::Float64` - Radial coordinate (√ψ)
  - `m::Vector{Int}` - Poloidal mode number(s)
  - `n::Vector{Int}` - Toroidal mode number(s)
  - `q::Float64` - Safety factor (= m/n)
  - `q1::Float64` - Derivative of safety factor with respect to ψ
  - `di::Float64` - Mercier criterion
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
@kwdef mutable struct SingType
    psifac::Float64 = 0.0
    rho::Float64 = 0.0
    m::Vector{Int} = Int[]
    n::Vector{Int} = Int[]
    q::Float64 = 0.0
    q1::Float64 = 0.0
    di::Float64 = 0.0
    alpha::Vector{ComplexF64} = ComplexF64[]
    r1::Vector{Int} = Int[]
    r2::Vector{Int} = Int[]
    n1::Vector{Int} = Int[]
    n2::Vector{Int} = Int[]
    power::Vector{ComplexF64} = ComplexF64[]
    vmat::Array{ComplexF64,4} = Array{ComplexF64}(undef, 0, 0, 0, 0)
    mmat::Array{ComplexF64,4} = Array{ComplexF64}(undef, 0, 0, 0, 0)
    m0mat::Matrix{ComplexF64} = zeros(ComplexF64, 2, 2)
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
    KineticParams

Minimal parameters for kinetic damping calculations.

## Fields

  - `nl::Int` - Number of parallel wave numbers (loop runs -nl:nl)

Future fields will be added as real kinetic physics is implemented
(e.g., ion/electron mass, charge, distribution parameters, etc.)
"""
@kwdef mutable struct KineticParams
    nl::Int = 10  # Default: -10 to +10 ell values
end

"""
    DconInternal

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
  - `fkg_kmats_flag::Bool` - Flag for kinetic matrix computation (not yet implemented)
  - `sol_base::Int` - Base index for solution vectors (not yet implemented)
  - `msing::Int` - Number of ideal singular surfaces
  - `kmsing::Int` - Number of kinetic singular surfaces (not yet implemented)
  - `sing::Vector{SingType}` - Vector of ideal singular surface data
  - `kinsing::Vector{SingType}` - Vector of kinetic singular surface data (not yet implemented)
  - `psilim::Float64` - Flux limit for integration
  - `qlim::Float64` - Safety factor at psilim
  - `q1lim::Float64` - Safety factor derivative at psilim
  - `locstab::Spl.CubicSpline{Float64}` - Spline for local stability analysis
"""
@kwdef mutable struct DconInternal
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
    fkg_kmats_flag::Bool = false
    sol_base::Int = 50
    msing::Int = 0
    kmsing::Int = 0
    sing::Vector{SingType} = SingType[]
    kinsing::Vector{SingType} = SingType[]
    psilim::Float64 = 0.0
    qlim::Float64 = 0.0
    q1lim::Float64 = 0.0
    locstab::Spl.CubicSpline{Float64} = Spl.empty_CubicSpline(Float64)
    debug_settings::DebugSettings = DebugSettings()
end

"""
    DconControl

A mutable struct containing control parameters for stability analysis, set by the user in dcon.toml.

## Fields

  - `verbose::Bool` - Enable verbose output
  - `bal_flag::Bool` - Enable ballooning mode analysis
  - `mat_flag::Bool` - Enable matrix output
  - `ode_flag::Bool` - Enable ODE integration diagnostics
  - `vac_flag::Bool` - Enable vacuum region calculation
  - `mer_flag::Bool` - Enable Mercier stability criterion
  - `fft_flag::Bool` - Enable Fourier transform analysis
  - `mthvac::Int` - Number of vacuum poloidal grid points (corresponds to `mtheta` in VacuumInput)
  - `sing_start::Int` - Start integration at the `sing_start`-th singular surface
  - `nn_low::Int` - Lower bound for toroidal modes
  - `nn_high::Int` - Upper bound for toroidal modes
  - `delta_mlow::Int` - Expands lower bound of Fourier harmonics by delta_mlow
  - `delta_mhigh::Int` - Expands upper bound of Fourier harmonics by delta_mhigh
  - `delta_mband::Int` - Integration keeps only this wide a band of solutions along the diagonal in m,m'
  - `thmax0::Float64` - Maximum integration step size (not yet implemented)
  - `nstep::Int` - Maximum number of integration steps (not yet implemented)
  - `ksing::Int` - Singular surface handling parameter
  - `tol_nr::Float64` - Relative tolerance of dynamic integration steps away from rationals
  - `tol_r::Float64` - Relative tolerance of dynamic integration steps near rationals
  - `crossover::Float64` - Fractional distance from rational q at which tolerance is switched to tol_r
  - `ucrit::Float64` - Critical value of unorm ratio to trigger solution normalization
  - `numsteps_init::Int` - Initial array size for ODE data storage
  - `numunorms_init::Int` - Initial array size for solution normalization data
  - `singfac_min::Float64` - Fractional distance from rational q at which ideal jump condition is enforced
  - `cyl_flag::Bool` - Make delta_mlow and delta_mhigh set the actual m truncation bounds. Default is to expand (n*qmin-4, n*qmax).
  - `set_psilim_via_dmlim::Bool` - Determine psilim truncation from outermost rational + dmlim
  - `dmlim::Float64` - Distance beyond last rational surface (as percentage)
  - `sing_order::Int` - Order of singular layer expansion
  - `qhigh::Float64` - Integration terminated at q limit determined by minimum of qhigh and qa from equil
  - `kin_flag::Bool` - Enable kinetic effects
  - `con_flag::Bool` - Continue integration through rationals without zeroing singular solutions
  - `kinfac1::Float64` - First kinetic scaling factor (not yet implemented)
  - `kinfac2::Float64` - Second kinetic scaling factor (not yet implemented)
  - `kingridtype::Int` - Type of kinetic grid (0=standard) (not yet implemented)
  - `ktanh_flag::Bool` - Enable hyperbolic tangent profile (not yet implemented)
  - `passing_flag::Bool` - Include passing particles (not yet implemented)
  - `trapped_flag::Bool` - Include trapped particles (not yet implemented)
  - `ion_flag::Bool` - Include ion kinetic effects (not yet implemented)
  - `electron_flag::Bool` - Include electron kinetic effects (not yet implemented)
  - `ktc::Float64` - Kinetic collision parameter (not yet implemented)
  - `ktw::Float64` - Kinetic width parameter (not yet implemented)
  - `qlow::Float64` - Integration terminated at q limit determined by minimum of qlow and q0 from equil
  - `reform_eq_with_psilim::Bool` - Reform equilibrium with computed psilim (not yet implemented)
  - `psiedge::Float64` - If less then psilim, calculates dW(psi) between psiedge and psilim, then runs with truncation at max(dW)
  - `nperq_edge::Int` - Number of points per q value at edge (not yet implemented)
  - `wv_farwall_flag::Bool` - Force nowall gpec calculations while calculating mutual inductance with the wall, when set true.
  - `dcon_kin_threads::Int` - Number of threads for kinetic calculations (not yet implemented)
  - `parallel_threads::Int` - Number of parallel threads (not yet implemented)
  - `diagnose::Bool` - Enable diagnostic output (not yet implemented)
  - `diagnose_ca::Bool` - Enable asymptotic coefficient diagnostics (not yet implemented)
  - `write_outputs_to_HDF5::Bool` - Write results to HDF5 format
  - `HDF5_filename::String` - Name of HDF5 output file
  - `force_wv_symmetry::Bool` - Boolean flag to enforce symmetry in the vacuum response matrix
  - `kinetic::KineticParams` - Kinetic damping parameters (separate struct for organization)
"""
@kwdef mutable struct DconControl
    verbose::Bool = true
    bal_flag::Bool = false
    mat_flag::Bool = false
    ode_flag::Bool = false
    vac_flag::Bool = false
    mer_flag::Bool = false
    fft_flag::Bool = false
    mthvac::Int = 480
    sing_start::Int = 0
    nn_low::Int = 0
    nn_high::Int = 0
    delta_mlow::Int = 0
    delta_mhigh::Int = 0
    delta_mband::Int = 0
    thmax0::Float64 = 1.0
    nstep::Int = typemax(Int)
    ksing::Int = -1
    tol_nr::Float64 = 1e-5
    tol_r::Float64 = 1e-5
    crossover::Float64 = 1e-2
    ucrit::Float64 = 1e4
    numsteps_init::Int = 4000
    numunorms_init::Int = 100
    singfac_min::Float64 = 0.0
    cyl_flag::Bool = false
    set_psilim_via_dmlim::Bool = false
    dmlim::Float64 = 0.2
    sing_order::Int = 2
    qhigh::Float64 = 1e3
    kin_flag::Bool = false
    con_flag::Bool = false
    kinfac1::Float64 = 1.0
    kinfac2::Float64 = 1.0
    kingridtype::Int = 0
    ktanh_flag::Bool = false
    passing_flag::Bool = false
    trapped_flag::Bool = true
    ion_flag::Bool = true
    electron_flag::Bool = false
    ktc::Float64 = 0.1
    ktw::Float64 = 50.0
    qlow::Float64 = 0.0
    reform_eq_with_psilim::Bool = false
    psiedge::Float64 = 1.0
    nperq_edge::Int = 20
    wv_farwall_flag::Bool = true
    dcon_kin_threads::Int = 1
    parallel_threads::Int = 1
    diagnose::Bool = false
    diagnose_ca::Bool = false
    write_outputs_to_HDF5::Bool = true
    HDF5_filename::String = "euler.h5"
    force_wv_symmetry::Bool = true

    # Kinetic parameters (separate struct for organization)
    kinetic::KineticParams = KineticParams()
end

@kwdef mutable struct FourFitVars
    mpert::Int
    mband::Int

    # Spline matrices
    amats::Spl.CubicSpline{ComplexF64} = Spl.empty_CubicSpline(ComplexF64)
    bmats::Spl.CubicSpline{ComplexF64} = Spl.empty_CubicSpline(ComplexF64)
    cmats::Spl.CubicSpline{ComplexF64} = Spl.empty_CubicSpline(ComplexF64)
    dmats::Spl.CubicSpline{ComplexF64} = Spl.empty_CubicSpline(ComplexF64)
    emats::Spl.CubicSpline{ComplexF64} = Spl.empty_CubicSpline(ComplexF64)
    hmats::Spl.CubicSpline{ComplexF64} = Spl.empty_CubicSpline(ComplexF64)
    fmats_lower::Spl.CubicSpline{ComplexF64} = Spl.empty_CubicSpline(ComplexF64)
    kmats::Spl.CubicSpline{ComplexF64} = Spl.empty_CubicSpline(ComplexF64)
    gmats::Spl.CubicSpline{ComplexF64} = Spl.empty_CubicSpline(ComplexF64)

    # Kinetic matrices (only allocated if kin_flag=true)
    kwmats::Vector{Spl.CubicSpline{ComplexF64}} = [Spl.empty_CubicSpline(ComplexF64) for _ in 1:6]
    ktmats::Vector{Spl.CubicSpline{ComplexF64}} = [Spl.empty_CubicSpline(ComplexF64) for _ in 1:6]

    # Composite kinetic matrices (used by ODE solver)
    akmats::Spl.CubicSpline{ComplexF64} = Spl.empty_CubicSpline(ComplexF64)
    bkmats::Spl.CubicSpline{ComplexF64} = Spl.empty_CubicSpline(ComplexF64)
    ckmats::Spl.CubicSpline{ComplexF64} = Spl.empty_CubicSpline(ComplexF64)
    f0mats::Spl.CubicSpline{ComplexF64} = Spl.empty_CubicSpline(ComplexF64)
    pmats::Spl.CubicSpline{ComplexF64} = Spl.empty_CubicSpline(ComplexF64)
    paats::Spl.CubicSpline{ComplexF64} = Spl.empty_CubicSpline(ComplexF64)
    kkmats::Spl.CubicSpline{ComplexF64} = Spl.empty_CubicSpline(ComplexF64)
    kkaats::Spl.CubicSpline{ComplexF64} = Spl.empty_CubicSpline(ComplexF64)
    r1mats::Spl.CubicSpline{ComplexF64} = Spl.empty_CubicSpline(ComplexF64)
    r2mats::Spl.CubicSpline{ComplexF64} = Spl.empty_CubicSpline(ComplexF64)
    r3mats::Spl.CubicSpline{ComplexF64} = Spl.empty_CubicSpline(ComplexF64)
    gaats::Spl.CubicSpline{ComplexF64} = Spl.empty_CubicSpline(ComplexF64)

    # Used in Free.jl
    jmat::Vector{ComplexF64} = Vector{ComplexF64}(undef, 2 * mband + 1)
end

# TODO: I think this initialization is funky - just need mband, not mpert. Fix later
FourFitVars(mpert::Int) = FourFitVars(; mpert)

"""
    VacuumData

A struct containing relevant data from the vacuum calculation.
Populated in `Free.jl`.

## Fields

- `mthvac::Int` - Number of vacuum poloidal grid points (corresponds to `mtheta` in VacuumInput)
- `mpert::Int` - Number of poloidal modes
- `numpert_total::Int` - Total number of modes (mpert × npert)
- `wt::Array{ComplexF64, 2}` - Toroidal vacuum response matrix (numpert_total × numpert_total)
- `wt0::Array{ComplexF64, 2}` - Reference toroidal vacuum matrix (numpert_total × numpert_total)
- `wv::Array{ComplexF64, 2}` - Vacuum energy matrix (numpert_total × numpert_total)
- `ep::Vector{ComplexF64}` - Plasma eigenvalues
- `ev::Vector{ComplexF64}` - Vacuum eigenvalues
- `et::Vector{ComplexF64}` - Total eigenvalues of plasma + vacuum
- `grri::Array{Float64, 2}` - Green's function radial integrals (2×mthvac × 2×mpert)
- `xzpts::Array{Float64, 2}` - Coordinate points [R_plasma, Z_plasma, R_wall, Z_wall] (mthvac × 4)
"""
@kwdef mutable struct VacuumData
    mthvac::Int
    mpert::Int
    numpert_total::Int

    wt::Array{ComplexF64,2} = Array{ComplexF64}(undef, numpert_total, numpert_total)
    wt0::Array{ComplexF64,2} = Array{ComplexF64}(undef, numpert_total, numpert_total)
    wv::Array{ComplexF64,2} = Array{ComplexF64}(undef, numpert_total, numpert_total)
    ep::Vector{ComplexF64} = Vector{ComplexF64}(undef, numpert_total)
    ev::Vector{ComplexF64} = Vector{ComplexF64}(undef, numpert_total)
    et::Vector{ComplexF64} = Vector{ComplexF64}(undef, numpert_total)

    # VACUUM can't handle 3D yet, so these are temporary mpert arrays
    # TODO: Matt separated grri into a few arrays for IPEC, will need to do that later
    grri::Array{Float64,2} = Array{Float64}(undef, 2 * (mthvac + 5), 2 * mpert)
    xzpts::Array{Float64,2} = Array{Float64}(undef, mthvac + 5, 4)
end

VacuumData(mthvac::Int, mpert::Int, numpert_total::Int) = VacuumData(; mthvac, mpert, numpert_total)

"""
OdeState

A mutable struct to hold the state of the ODE solver used by the DCON integration routines.
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
  - `dW_edge::Vector{ComplexF64}` - dW values computed in the psiedge < psilim region for each stored step (length `numsteps_init`).
  - `wvmat_spline::Spl.CubicSpline{ComplexF64}` - Spline representation of precomputed wv matrices used by `free_test`/vacuum routines.
  - `psifac::Float64` - Current normalized flux coordinate for the integrator.
  - `q::Float64` - Safety factor value at `psifac` (current q during integration).
  - `u::Array{ComplexF64,3}` - Current working solution arrays with shape `(numpert_total, numpert_total, 2)`.
  - `ud::Array{ComplexF64,3}` - Current working solution derivative (different than du) arrays with shape `(numpert_total, numpert_total, 2)`.
  - `ising::Int` - Index of the next singular surface to be crossed during integration.
  - `psimax::Float64` - Maximum psi value for which the integrator is allowed to run in next integration region.
  - `next::String` - Next integration action to take (e.g. `"cross"` to cross a rational surface or `"finish"`).
  - `nzero::Int` - Count of detected zero crossings (used for diagnostics).
  - `new::Bool` - Flag indicating whether a new `unorm0` should be computed after a fixup.
  - `unorm::Vector{Float64}` - Current norms of the solution vectors (length `numpert_total`).
  - `unorm0::Vector{Float64}` - Reference/initial norms of the solution vectors (length `numpert_total`).
  - `ifix::Int` - Number of normalization operations performed (index into normalization arrays).
  - `index::Array{Int,2}` - Index matrix used for sorting solution norms with shape `(numpert_total, numunorms_init)`.
  - `sing_flag::Vector{Bool}` - Boolean flags indicating which stored normalizations correspond to singular solutions
    (length `numunorms_init`).
  - `zeroed_idx::Vector{Vector{Int}}` - For each ideal rational surface jump, a vector of indices of solutions that were zeroed.
  - `fixfac::Array{ComplexF64,3}` - Fix-up factors for Gaussian reduction with shape
    `(numpert_total, numpert_total, numunorms_init)`.
  - `fixstep::Vector{Int64}` - Step indices (psi step positions) at which normalization/fixups were performed (length `numunorms_init`).
  - Temporary workspaces used during integration calculations:

      + `amat::Vector{ComplexF64}` - Flattened A matrix (length `numpert_total^2`)
      + `bmat::Vector{ComplexF64}` - Flattened B matrix (length `numpert_total^2`)
      + `cmat::Vector{ComplexF64}` - Flattened C matrix (length `numpert_total^2`)
      + `fmat_lower::Vector{ComplexF64}` - Lower-triangle factor of F (length `numpert_total^2`)
      + `kmat::Vector{ComplexF64}` - Flattened K matrix (length `numpert_total^2`)
      + `gmat::Vector{ComplexF64}` - Flattened G matrix (length `numpert_total^2`)
      + `tmp::Matrix{ComplexF64}` - Workspace matrix for EL derivative calculations with shape `(numpert_total, numpert_total)`.
      + `Afact::Union{Cholesky{ComplexF64, Matrix{ComplexF64}}, Nothing}` - Cholesky factor
      + `singfac_vec::Vector{Float64}` - Vector of m-nq factors
"""
@kwdef mutable struct OdeState
    # Initialization parameters
    numpert_total::Int
    numunorms_init::Int
    msing::Int
    numsteps_init::Int

    # Saved data throughout integration
    step::Int = 1
    psi_store::Vector{Float64} = Vector{Float64}(undef, numsteps_init)
    q_store::Vector{Float64} = Vector{Float64}(undef, numsteps_init)
    u_store::Array{ComplexF64,4} = Array{ComplexF64}(undef, numpert_total, numpert_total, 2, numsteps_init)
    ud_store::Array{ComplexF64,4} = Array{ComplexF64}(undef, numpert_total, numpert_total, 2, numsteps_init)
    crit_store::Vector{Float64} = Vector{Float64}(undef, numsteps_init)
    ca_r::Array{ComplexF64,4} = Array{ComplexF64}(undef, numpert_total, numpert_total, 2, msing)
    ca_l::Array{ComplexF64,4} = Array{ComplexF64}(undef, numpert_total, numpert_total, 2, msing)

    # Used for to find peak dW in the edge
    dW_edge::Vector{ComplexF64} = Array{ComplexF64}(undef, numsteps_init)
    wvmat_spline::Spl.CubicSpline{ComplexF64} = Spl.empty_CubicSpline(ComplexF64)

    # Data for integrator
    psifac::Float64 = 0.0
    q::Float64 = 0.0
    u::Array{ComplexF64,3} = zeros(ComplexF64, numpert_total, numpert_total, 2)
    ud::Array{ComplexF64,3} = zeros(ComplexF64, numpert_total, numpert_total, 2)
    ising::Int = 0
    psimax::Float64 = 0.0
    next::String = ""
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

    # Temporary matrices for sing_der calculations
    amat::Vector{ComplexF64} = Vector{ComplexF64}(undef, numpert_total^2)
    bmat::Vector{ComplexF64} = Vector{ComplexF64}(undef, numpert_total^2)
    cmat::Vector{ComplexF64} = Vector{ComplexF64}(undef, numpert_total^2)
    fmat_lower::Vector{ComplexF64} = Vector{ComplexF64}(undef, numpert_total^2)
    kmat::Vector{ComplexF64} = Vector{ComplexF64}(undef, numpert_total^2)
    gmat::Vector{ComplexF64} = Vector{ComplexF64}(undef, numpert_total^2)
    tmp::Matrix{ComplexF64} = Matrix{ComplexF64}(undef, numpert_total, numpert_total)
    Afact::Union{Cholesky{ComplexF64,Matrix{ComplexF64}},Nothing} = nothing
    singfac_vec::Vector{Float64} = Vector{Float64}(undef, numpert_total)
end

# Initialize function for OdeState with relevant parameters for array initialization
OdeState(numpert_total::Int, numsteps_init::Int, numunorms_init::Int, msing::Int) = OdeState(; numpert_total, numsteps_init, numunorms_init, msing)


# Below here are debug output structs used for benchmarking and unit testing




"""
    VacuumBenchmarkInputs

A struct to hold all inputs required for vacuum benchmarking between Fortran and Julia implementations.

## Fields

- `wv_block::Matrix{ComplexF64}` - Vacuum response matrix block
- `mpert::Int` - Number of poloidal modes
- `mtheta_eq::Int` - Number of poloidal grid points in input equilibrium (corresponds to `mtheta_eq` in VacuumInput)
- `mthvac::Int` - Number of poloidal grid points in vacuum calculations (corresponds to `mtheta` in VacuumInput)
- `complex_flag::Bool` - Flag indicating if complex arithmetic is used
- `kernelsign::Float64` - Sign of the kernel for vacuum calculation
- `wall_flag::Bool` - Flag indicating presence of wall
- `farwall_flag::Bool` - Flag indicating presence of far wall
- `grri::Matrix{Float64}` - Green's function response matrix
- `xzpts::Matrix{Float64}` - Coordinate points on plasma boundary [R, Z]
- `ahg_file::String` - Filename for AHG data
- `dir_path::String` - Directory path for input/output files
- `vac_inputs::Vacuum.VacuumInput` - VacuumInput struct for Julia vacuum code
- `wall_settings::Vacuum.WallShapeSettings` - Wall shape settings
- `n::Int` - Toroidal mode number
- `ipert_n::Int` - Index of perturbed toroidal mode
- `psifac::Float64` - Normalized flux coordinate
"""
@kwdef struct VacuumBenchmarkInputs
    # Vacuum computation parameters
    wv_block::Matrix{ComplexF64}
    mpert::Int
    mtheta_eq::Int
    mthvac::Int
    complex_flag::Bool
    kernelsign::Float64
    wall_flag::Bool
    farwall_flag::Bool
    grri::Matrix{Float64}
    xzpts::Matrix{Float64}
    ahg_file::String
    dir_path::String
    
    # VacuumInput struct for Julia code
    vac_inputs::Vacuum.VacuumInput
    
    # Wall settings
    wall_settings::Vacuum.WallShapeSettings
    
    # Additional context
    n::Int
    ipert_n::Int
    psifac::Float64
end