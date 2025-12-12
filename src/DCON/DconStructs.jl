"""
    SingType

A mutable struct containing data for singular surfaces in the plasma stability analysis.

## Fields

- `psifac::Float64` - Normalized flux coordinate at the singular surface
- `rho::Float64` - Radial coordinate (√ψ)
- `m::Vector{Int}` - Poloidal mode numbers
- `n::Vector{Int}` - Toroidal mode numbers
- `q::Float64` - Safety factor at the singular surface
- `q1::Float64` - Derivative of safety factor with respect to ψ
- `di::Float64` - Inertial layer width parameter
- `alpha::Vector{ComplexF64}` - Complex phase angles for mode coupling
- `r1::Vector{Int}` - Primary resonance indices
- `r2::Vector{Int}` - Secondary resonance indices
- `n1::Vector{Int}` - Primary toroidal mode indices
- `n2::Vector{Int}` - Secondary toroidal mode indices
- `power::Vector{ComplexF64}` - Power series coefficients
- `vmat::Array{ComplexF64,4}` - Velocity matrix for singular layer analysis
- `mmat::Array{ComplexF64,4}` - Mode coupling matrix for singular layer analysis
- `m0mat::Matrix{ComplexF64}` - Base mode matrix (2×2)
"""
# TODO: ideally, everything is allocated at construction, but mpert is determined after
# since these are allocated in sing_find. What's the best way to handle this?
# For now, leaving as missing, per Brendan's github comment
# Something simple could be not creating sing_types within sing_find, but instead saving the data,
# then creating them all at once after mpert is determined in dcon.jl
# This wouldn't be as clean, but would allow preallocation. Does this greatly impact performance?
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
# @kwdef mutable struct SingType
#     mpert::Int
#     order::Int
#     m::Int = 0
#     psifac::Float64 = 0.0
#     rho::Float64 = 0.0
#     q::Float64 = 0.0
#     q1::Float64 = 0.0
#     di::Float64 = 0.0
#     alpha::ComplexF64 = 0.0 + 0.0im
#     r1::Vector{Int} = [0]
#     r2::Vector{Int} = [0, 0]
#     n1::Vector{Int} = Vector{Int}(undef, mpert - 1)
#     n2::Vector{Int} = Vector{Int}(undef, 2 * mpert - 2)
#     power::Vector{ComplexF64} = Vector{ComplexF64}(undef, 2 * mpert)
#     vmat::Array{ComplexF64,4} = Array{ComplexF64}(undef, mpert, 2 * mpert, 2, order + 1)
#     mmat::Array{ComplexF64,4} = Array{ComplexF64}(undef, mpert, 2 * mpert, 2, order + 3)
#     m0mat::Union{Nothing, Matrix{ComplexF64}} = zeros(ComplexF64, 2, 2)
# end
# # Constructor to allocate matrices
# SingType(mpert::Int, order::Int; kwargs...) = SingType(; mpert, order, kwargs...)

"""
    DconInternal

A mutable struct holding internal state variables for DCON stability calculations.

## Fields

- `dir_path::String` - Directory path for output files
- `mlow::Int` - Lowest poloidal mode number
- `mhigh::Int` - Highest poloidal mode number
- `mpert::Int` - Number of poloidal modes (mhigh - mlow + 1)
- `mband::Int` - Bandwidth for matrix operations (mpert - 1 - delta_mband)
- `nlow::Int` - Lowest toroidal mode number
- `nhigh::Int` - Highest toroidal mode number
- `npert::Int` - Number of toroidal modes (nhigh - nlow + 1)
- `numpert_total::Int` - Total number of perturbation modes (mpert × npert)
- `vac_memory::Bool` - Memory allocation flag for vacuum calculations (not yet implemented)
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
    mpert::Int = 0 # mpert = mhigh-mlow+1
    mband::Int = 0 # mband = mpert-1-delta_mband
    nlow::Int = 0
    nhigh::Int = 0
    npert::Int = 0 # npert = nhigh-nlow+1
    numpert_total::Int = 0 # numpert_total = mpert*npert
    vac_memory::Bool = true # TODO: most likely just remove, always true in ahg_flag is deprecated
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
end

"""
    DconControl

A mutable struct containing control parameters for DCON stability analysis.

## Fields

- `verbose::Bool` - Enable verbose output
- `bal_flag::Bool` - Enable ballooning mode analysis
- `mat_flag::Bool` - Enable matrix output
- `ode_flag::Bool` - Enable ODE integration diagnostics
- `vac_flag::Bool` - Enable vacuum region calculation
- `mer_flag::Bool` - Enable Mercier stability criterion
- `fft_flag::Bool` - Enable Fourier transform analysis
- `mthvac::Int` - Number of vacuum poloidal mesh points
- `sing_start::Int` - Starting index for singular surface treatment
- `nn_low::Int` - Lower bound for toroidal mode scan
- `nn_high::Int` - Upper bound for toroidal mode scan
- `delta_mlow::Int` - Offset for lowest poloidal mode
- `delta_mhigh::Int` - Offset for highest poloidal mode
- `delta_mband::Int` - Bandwidth reduction parameter
- `thmax0::Float64` - Maximum integration step size
- `nstep::Int` - Maximum number of integration steps
- `ksing::Int` - Singular surface handling parameter
- `tol_nr::Float64` - Newton-Raphson tolerance
- `tol_r::Float64` - Residual tolerance
- `crossover::Float64` - Crossover threshold for singular layer
- `ucrit::Float64` - Critical value for solution normalization
- `numsteps_init::Int` - Initial array size for ODE data storage
- `numunorms_init::Int` - Initial array size for normalization data
- `singfac_min::Float64` - Minimum singular factor threshold
- `cyl_flag::Bool` - Enable cylindrical approximation
- `set_psilim_via_dmlim::Bool` - Determine psilim from outermost rational + dmlim
- `dmlim::Float64` - Distance beyond last rational surface (as percentage)
- `sing_order::Int` - Order of singular layer expansion
- `qhigh::Float64` - Upper limit for safety factor
- `kin_flag::Bool` - Enable kinetic effects
- `con_flag::Bool` - Enable continuum damping
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
- `qlow::Float64` - Lower limit for safety factor
- `use_classic_splines::Bool` - Use classic spline interpolation
- `reform_eq_with_psilim::Bool` - Reform equilibrium with computed psilim
- `psiedge::Float64` - Normalized flux at edge
- `nperq_edge::Int` - Number of points per q value at edge (not yet implemented)
- `wv_farwall_flag::Bool` - Enable far wall vacuum calculation
- `dcon_kin_threads::Int` - Number of threads for kinetic calculations
- `parallel_threads::Int` - Number of parallel threads
- `diagnose::Bool` - Enable diagnostic output
- `diagnose_ca::Bool` - Enable asymptotic coefficient diagnostics
- `write_outputs_to_HDF5::Bool` - Write results to HDF5 format
- `HDF5_filename::String` - Name of HDF5 output file
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
    numsteps_init::Int = 4000 # used to set initial size of data store in OdeState
    numunorms_init::Int = 100 # used to set initial size of saved unorm data in OdeState
    singfac_min::Float64 = 0.0
    cyl_flag::Bool = false
    set_psilim_via_dmlim::Bool = false # previously sas_flag, if true, determines psilim using outermost rational + dmlim
    dmlim::Float64 = 0.2 # % outside the last rational surface to go out to determine dW if set_psilim_via_dmlim is true
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
    use_classic_splines::Bool = false
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
end

"""
    FourFitVars

A mutable struct containing variables for Fourier fitting in DCON calculations.

## Fields

- `mpert::Int` - Number of poloidal modes
- `mband::Int` - Bandwidth for matrix operations
- `amats::Spl.CubicSpline{ComplexF64}` - Spline for A matrix coefficients
- `bmats::Spl.CubicSpline{ComplexF64}` - Spline for B matrix coefficients
- `cmats::Spl.CubicSpline{ComplexF64}` - Spline for C matrix coefficients
- `dmats::Spl.CubicSpline{ComplexF64}` - Spline for D matrix coefficients
- `emats::Spl.CubicSpline{ComplexF64}` - Spline for E matrix coefficients
- `hmats::Spl.CubicSpline{ComplexF64}` - Spline for H matrix coefficients
- `fmats_lower::Spl.CubicSpline{ComplexF64}` - Spline for lower F matrix coefficients
- `kmats::Spl.CubicSpline{ComplexF64}` - Spline for K matrix coefficients
- `gmats::Spl.CubicSpline{ComplexF64}` - Spline for G matrix coefficients
- `jmat::Vector{ComplexF64}` - J matrix vector (size 2×mband + 1)
- `parallel_threads::Int` - Number of parallel threads for computation
- `dcon_kin_threads::Int` - Number of threads for kinetic calculations
"""
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

    # Used in Free.jl
    jmat::Vector{ComplexF64} = Vector{ComplexF64}(undef, 2 * mband + 1)

    parallel_threads::Int = 0
    dcon_kin_threads::Int = 0
end

FourFitVars(mpert::Int) = FourFitVars(; mpert)

"""
    VacuumData

A struct containing vacuum region calculation data for DCON stability analysis.

## Fields

- `mthvac::Int` - Number of vacuum poloidal mesh points
- `mpert::Int` - Number of poloidal modes
- `numpert_total::Int` - Total number of perturbation modes
- `wt::Array{ComplexF64, 2}` - Toroidal vacuum response matrix (numpert_total × numpert_total)
- `wt0::Array{ComplexF64, 2}` - Reference toroidal vacuum matrix (numpert_total × numpert_total)
- `wv::Array{ComplexF64, 2}` - Vacuum energy matrix (numpert_total × numpert_total)
- `ep::Vector{ComplexF64}` - Plasma edge displacement eigenvector
- `ev::Vector{ComplexF64}` - Vacuum edge displacement eigenvector
- `et::Vector{ComplexF64}` - Total edge displacement eigenvector
- `grri::Array{Float64, 2}` - Green's function radial integrals (2×(mthvac+5) × 2×mpert)
- `xzpts::Array{Float64, 2}` - X-Z coordinate points on plasma boundary (mthvac+5 × 4)
"""
# TODO: Matt separated grri into a few arrays for IPEC, will need to do that later
@kwdef struct VacuumData
    mthvac::Int
    mpert::Int
    numpert_total::Int

    wt::Array{ComplexF64, 2} = Array{ComplexF64}(undef, numpert_total, numpert_total)
    wt0::Array{ComplexF64, 2} = Array{ComplexF64}(undef, numpert_total, numpert_total)
    wv::Array{ComplexF64, 2} = Array{ComplexF64}(undef, numpert_total, numpert_total)
    ep::Vector{ComplexF64} = Vector{ComplexF64}(undef, numpert_total)
    ev::Vector{ComplexF64} = Vector{ComplexF64}(undef, numpert_total)
    et::Vector{ComplexF64} = Vector{ComplexF64}(undef, numpert_total)

    # VACUUM can't handle 3D yet, so these are temporary mpert arrays
    grri::Array{Float64, 2} = Array{Float64}(undef, 2 * (mthvac + 5), 2 * mpert)
    xzpts::Array{Float64, 2} = Array{Float64}(undef, mthvac + 5, 4)
end

VacuumData(mthvac::Int, mpert::Int, numpert_total::Int) = VacuumData(; mthvac, mpert, numpert_total)

"""
OdeState

A mutable struct to hold the state of the ODE solver for DCON.
This struct contains all necessary fields to manage the ODE integration process,
including solution vectors, tolerances, and flags for the integration process.
"""
@kwdef mutable struct OdeState
    # Initialization parameters
    numpert_total::Int                  # total number of modes
    numunorms_init::Int             # initial storage size for unorm data
    msing::Int                   # number of singular surfaces
    numsteps_init::Int             # initial size of data store

    # Saved data throughout integration
    step::Int = 1                    # current step of integration (this is like istep in Fortran)
    psi_store::Vector{Float64} = Vector{Float64}(undef, numsteps_init)  # psi at each step of integration
    q_store::Vector{Float64} = Vector{Float64}(undef, numsteps_init)    # q at each step of integration
    u_store::Array{ComplexF64,4} = Array{ComplexF64}(undef, numpert_total, numpert_total, 2, numsteps_init) # store of u at each step of integration
    ud_store::Array{ComplexF64,4} = Array{ComplexF64}(undef, numpert_total, numpert_total, 2, numsteps_init) # store of ud at each step of integration
    crit_store::Vector{Float64} = Vector{Float64}(undef, numsteps_init)  # store of crit at each step of integration
    ca_r::Array{ComplexF64,4} = Array{ComplexF64}(undef, numpert_total, numpert_total, 2, msing) # asymptotic coefficients just right of singular surface
    ca_l::Array{ComplexF64,4} = Array{ComplexF64}(undef, numpert_total, numpert_total, 2, msing) # asymptotic coefficients just left of singular surface

    # Used for to find peak dW in the edge
    dW_edge::Vector{ComplexF64} = Array{ComplexF64}(undef, numsteps_init)  # dW at each step in the edge
    wvmat_spline::Spl.CubicSpline{ComplexF64} = Spl.empty_CubicSpline(ComplexF64)  # spline of wv matrices for free_test

    # Data for integrator
    psifac::Float64 = 0.0       # normalized flux coordinate
    q::Float64 = 0.0            # q value at psifac
    u::Array{ComplexF64,3} = zeros(ComplexF64, numpert_total, numpert_total, 2)            # solution vectors
    ud::Array{ComplexF64,3} = zeros(ComplexF64, numpert_total, numpert_total, 2)           # derivative of solution vectors used in GPEC
    ising::Int = 0               # index of next singular surface
    psimax::Float64 = 0.0         # maximum psi value for the integrator
    next::String = ""           # next integration action ("cross" or "finish")
    nzero::Int = 0              # count of zero crossings detected

    # Used for Gaussian reduction
    new::Bool = true            # flag for computing new unorm0 after a fixup
    unorm::Vector{Float64} = zeros(Float64, numpert_total)                        # norms of solution vectors
    unorm0::Vector{Float64} = zeros(Float64, numpert_total)                       # initial norms of solution vectors
    ifix::Int = 0                # index for number of unorms performed
    index::Array{Int,2} = zeros(Int, numpert_total, numunorms_init)                                   # indices for sorting solutions
    sing_flag::Vector{Bool} = falses(numunorms_init)                     # flags for singular solutions
    zeroed_idx::Vector{Vector{Int}} = Vector{Vector{Int}}(undef, numunorms_init)  # indices of zeroed solutions at each unorm
    fixfac::Array{ComplexF64,3} = zeros(ComplexF64, numpert_total, numpert_total, numunorms_init)             # fixup factors for Gaussian reduction
    fixstep::Vector{Int64} = zeros(Int64, numunorms_init)               # psi values at which unorms were performed

    # Temporary matrices for sing_der calculations
    amat::Vector{ComplexF64} = Vector{ComplexF64}(undef, numpert_total^2)
    bmat::Vector{ComplexF64} = Vector{ComplexF64}(undef, numpert_total^2)
    cmat::Vector{ComplexF64} = Vector{ComplexF64}(undef, numpert_total^2)
    fmat_lower::Vector{ComplexF64} = Vector{ComplexF64}(undef, numpert_total^2)
    kmat::Vector{ComplexF64} = Vector{ComplexF64}(undef, numpert_total^2)
    gmat::Vector{ComplexF64} = Vector{ComplexF64}(undef, numpert_total^2)
    tmp::Matrix{ComplexF64} = Matrix{ComplexF64}(undef, numpert_total, numpert_total)
    Afact::Union{Cholesky{ComplexF64, Matrix{ComplexF64}}, Nothing} = nothing
    singfac_vec::Vector{Float64} = Vector{Float64}(undef, numpert_total)
end

# Initialize function for OdeState with relevant parameters for array initialization
OdeState(numpert_total::Int, numsteps_init::Int, numunorms_init::Int, msing::Int) = OdeState(; numpert_total, numsteps_init, numunorms_init, msing)