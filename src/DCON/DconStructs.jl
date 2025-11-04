# TODO: add descriptions of what all variables are and/or explanation of defaults

# TODO: ideally, everything is allocated at construction, but mpert is determined after
# since these are allocated in sing_find. What's the best way to handle this?
# For now, leaving as missing, per Brendan's github comment
# Something simple could be not creating sing_types within sing_find, but instead saving the data,
# then creating them all at once after mpert is determined in dcon.jl
# This wouldn't be as clean, but would allow preallocation. Does this greatly impact performance?
@kwdef mutable struct SingType
    m::Int = 0
    r1::Vector{Int} = [0]
    r2::Vector{Int} = [0, 0]
    psifac::Float64 = 0.0
    rho::Float64 = 0.0
    q::Float64 = 0.0
    q1::Float64 = 0.0
    di::Float64 = 0.0
    alpha::ComplexF64 = 0.0 + 0.0im
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
    mlow::Int = 0 #Copy of delta_mlow, but with limits enforced
    mhigh::Int = 0 #Copy of delta_mhigh, but with limits enforced
    mpert::Int = 0 #mpert = mhigh-mlow+1
    mband::Int = 0 #mband = mpert-1-delta_mband
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
    node_flag::Bool = false
    mthvac::Int = 480
    sing_start::Int = 0
    nn::Int = 0
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
    termbycross_flag::Bool = false
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
end

# TODO: a lot of these will be deprecated, this will be a general data structure that handles output
# Since file I/O in Julia will be very different than Fortran, this will likely be reworked significantly
@kwdef mutable struct DconOutput
    # output switches
    write_crit_out::Bool = false
    write_dcon_out::Bool = false
    write_euler_h5::Bool = true
    write_eqdata_h5::Bool = true

    # filenames
    fname_crit_out::String = "crit.out"
    fname_dcon_out::String = "dcon.out"
    fname_euler_h5::String = "euler.h5"
    fname_eqdata_h5::String = "eqdata.h5"

    handles::Dict{Symbol,Any} = Dict()

    # Old or unused yet
    interp::Bool = false
    crit_break::Bool = true
    out_bal1::Bool = false
    bin_bal1::Bool = false
    out_bal2::Bool = false
    bin_bal2::Bool = false
    out_metric::Bool = false
    bin_metric::Bool = false
    feval_flag::Bool = false
    out_fmat::Bool = false
    bin_fmat::Bool = false
    out_gmat::Bool = false
    bin_gmat::Bool = false
    out_kmat::Bool = false
    bin_kmat::Bool = false
    out_sol::Bool = false
    out_sol_min::Int = 0
    out_sol_max::Int = 0
    bin_sol::Bool = false
    bin_sol_min::Int = 0
    bin_sol_max::Int = 0
    out_fl::Bool = false
    bin_fl::Bool = false
    out_evals::Bool = false
    bin_evals::Bool = false
    bin_euler::Bool = false
    euler_stride::Int = 1
    bin_vac::Bool = false # TODO: deprecated
    mthsurf0::Float64 = 1.0 # TODO: deprecated
    msol_ahb::Int = 0 # TODO: deprecated
    netcdf_out::Bool = true # TODO: might be deprecated
    out_fund::Bool = false
    out_ahg2msc::Bool = false # TODO: deprecated
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
