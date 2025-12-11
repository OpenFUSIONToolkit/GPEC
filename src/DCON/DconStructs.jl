# TODO: add descriptions of what all variables are and/or explanation of defaults

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
    power::Union{Missing,Vector{ComplexF64}} = missing # we know the size of these, but it depends on mpert
    vmat::Union{Missing,Array{ComplexF64,4}} = missing
    mmat::Union{Missing,Array{ComplexF64,4}} = missing
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

@kwdef mutable struct DconInternal
    dir_path::String = ""
    mlow::Int = 0
    mhigh::Int = 0
    mpert::Int = 0 # mpert = mhigh-mlow+1
    mband::Int = 0 # mband = mpert-1-delta_mband
    nlow::Int = 0
    nhigh::Int = 0
    npert::Int = 0 # npert = nhigh-nlow+1
    numpert_total = 0 # numpert_total = mpert*npert
    vac_memory::Bool = true # TODO: most likely just remove, always true in ahg_flag is deprecated
    keq_out::Bool = false
    theta_out::Bool = false
    xlmda_out::Bool = false
    fkg_kmats_flag::Bool = false
    sol_base::Int = 50
    msing::Int = 0
    kmsing::Int = 0
    sing::Union{Nothing,Vector{SingType}} = SingType[]
    kinsing::Union{Nothing,Vector{SingType}} = SingType[]
    psilim::Float64 = 0.0
    qlim::Float64 = 0.0
    q1lim::Float64 = 0.0
    # TODO: how to initialize a spline? This will be a spline of size mpsi x 5
    locstab::Union{Missing,Spl.CubicSpline{Float64}} = missing
end

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

# TODO: how can we initialize the splines to not be nothings?
@kwdef mutable struct FourFitVars
    mpert::Int
    mband::Int

    # Spline matrices
    # TODO: how to initialize these splines?
    amats::Union{Missing,Spl.CubicSpline{ComplexF64}} = missing
    bmats::Union{Missing,Spl.CubicSpline{ComplexF64}} = missing
    cmats::Union{Missing,Spl.CubicSpline{ComplexF64}} = missing
    dmats::Union{Missing,Spl.CubicSpline{ComplexF64}} = missing
    emats::Union{Missing,Spl.CubicSpline{ComplexF64}} = missing
    hmats::Union{Missing,Spl.CubicSpline{ComplexF64}} = missing
    fmats_lower::Union{Missing,Spl.CubicSpline{ComplexF64}} = missing
    kmats::Union{Missing,Spl.CubicSpline{ComplexF64}} = missing
    gmats::Union{Missing,Spl.CubicSpline{ComplexF64}} = missing

    # Used in Free.jl
    jmat::Vector{ComplexF64} = Vector{ComplexF64}(undef, 2 * mband + 1)

    parallel_threads::Int = 0
    dcon_kin_threads::Int = 0
end

FourFitVars(mpert::Int) = FourFitVars(; mpert)

# TODO: Matt separated grri into a few arrays for IPEC, will need to do that later
@kwdef mutable struct VacuumData
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
    wvmat_spline::Union{Nothing,Spl.CubicSpline{ComplexF64}} = nothing  # spline of wv matrices for free_test # TODO: how to initialize a spline?

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