"""
    PentrcStructs

Data structures for the PENTRC module.
Consolidates all global variables into logical structures.
"""

# ============================================================================
# Control/Input Parameters
# ============================================================================
"""
    PentrcControl

Main control structure for PENTRC simulations.
Corresponds to input namelist variables.
"""
struct PentrcControl
    # Moment type
    moment::String              # "heat" or "pressure"
    qt::Bool                    # true if moment == "heat"
    
    # Method flags (which methods to calculate)
    fgar_flag::Bool
    tgar_flag::Bool
    pgar_flag::Bool
    rlar_flag::Bool
    clar_flag::Bool
    fcgl_flag::Bool
    wxyz_flag::Bool
    fkmm_flag::Bool
    tkmm_flag::Bool
    pkmm_flag::Bool
    frmm_flag::Bool
    trmm_flag::Bool
    prmm_flag::Bool
    fwmm_flag::Bool
    twmm_flag::Bool
    pwmm_flag::Bool
    ftmm_flag::Bool
    ttmm_flag::Bool
    ptmm_flag::Bool
    
    # Output flags
    eq_out::Bool
    theta_out::Bool
    xlmda_out::Bool
    eqpsi_out::Bool
    output_ascii::Bool
    output_netcdf::Bool
    output_orbit::Bool
    output_ffuns::Bool
    output_torque::Bool
    
    # Grid flags
    equil_grid::Bool
    input_grid::Bool
    dynamic_grid::Bool
    
    # Diagnostic flags
    fnml_flag::Bool
    ellip_flag::Bool
    diag_flag::Bool
    term_flag::Bool
    force_xialpha::Bool
    clean::Bool
    
    # Plasma species parameters
    zi::Int                     # Ion charge
    mi::Int                     # Ion mass (proton masses)
    zimp::Int                   # Impurity charge
    mimp::Int                   # Impurity mass
    electron::Bool              # Calculate for electrons
    
    # Mode numbers
    nl::Int                     # Bounce harmonic number
    nn::Int                     # Toroidal mode number (from equilibrium)
    
    # Grid parameters
    tmag_in::Int
    jsurf_in::Int
    power_bin::Int
    power_bpin::Int
    power_rin::Int
    power_rcin::Int
    
    # Thread control
    pentrc_threads::Int
    openmp_threads::Int
    
    # Tolerances
    atol_xlmda::Float64
    rtol_xlmda::Float64
    
    # Scaling factors
    nfac::Float64
    tfac::Float64
    wefac::Float64
    wdfac::Float64
    wpfac::Float64
    nufac::Float64
    divxfac::Float64
    
    # Diagnostic parameters
    diag_psi::Float64
    psi_out::Vector{Float64}
    psilims::Vector{Float64}
    
    # Output parameters
    data_dir::String
    nutype::String
    f0type::String
    jac_in::String
    
    # File references
    idconfile::String
    kinetic_file::String
    gpec_file::String
    peq_file::String
    pmodb_file::String
    
    # Debugging
    verbose::Bool
    indebug::Bool
end


# ============================================================================
# Internal State/Working Variables
# ============================================================================
"""
    PentrcInternal

Internal working state for PENTRC calculations.
Holds intermediate results and temporary data.
"""
struct PentrcInternal
    dir_path::String                    # Working directory
    ro::Float64                         # Major radius
    bo::Float64                         # Field on axis
    
    # Integration results
    tphi::ComplexF64                    # Total torque/energy
    tsurf::ComplexF64                   # Surface torque/energy  
    teq::ComplexF64                     # Equilibrium grid result
    
    # Grid arrays
    grid_type::String                   # "equil", "input", or "lsode"
    psi_grid::Vector{Float64}
    
    # Output state
    nstring::String
    method::String
    wtw::Array{ComplexF64, 3}           # Kinetic matrix storage
    
    # Status tracking
    psi_out_valid::Vector{Float64}
    nvalid::Int
    
    # Method/flags array
    flags::Vector{Bool}                 # [fgar, tgar, ..., prmm]
    methods::Vector{String}
    docs::Vector{String}
end

function PentrcInternal(; dir_path::String=".", ro::Float64=1.0, bo::Float64=1.0)
    return PentrcInternal(
        dir_path,
        ro,
        bo,
        ComplexF64(0, 0),
        ComplexF64(0, 0),
        ComplexF64(0, 0),
        "equil",
        Float64[],
        "",
        "",
        Array{ComplexF64,3}(undef, 0, 0, 0),
        Float64[],
        0,
        fill(false, 18),
        ["fgar", "tgar", "pgar", "rlar", "clar", "fcgl",
         "fwmm", "twmm", "pwmm", "ftmm", "ttmm", "ptmm",
         "fkmm", "tkmm", "pkmm", "frmm", "trmm", "prmm"],
        ["Full general-aspect-ratio calculation",
         "Trapped particle general-aspect-ratio calculation",
         "Passing particle general-aspect-ratio calculation",
         "Trapped particle large-aspect-ratio calculation",
         "Trapped particle cylindrical large-aspect-ratio calculation",
         "Fluid Chew–Goldberger–Low calculation",
         "Full energy calculation using MXM Euler–Lagrange matrix",
         "Trapped energy calculation using MXM Euler–Lagrange matrix",
         "Passing energy calculation using MXM Euler–Lagrange matrix",
         "Full torque calculation using MXM Euler–Lagrange matrix",
         "Trapped torque calculation using MXM Euler–Lagrange matrix",
         "Passing torque calculation using MXM Euler–Lagrange matrix",
         "Full MXM Euler–Lagrange energy matrix norm calculation",
         "Trapped MXM Euler–Lagrange energy matrix norm calculation",
         "Passing MXM Euler–Lagrange energy matrix norm calculation",
         "Full MXM Euler–Lagrange torque matrix norm calculation",
         "Trapped MXM Euler–Lagrange torque matrix norm calculation",
         "Passing MXM Euler–Lagrange torque matrix norm calculation"]
    )
end


# ============================================================================
# Output Configuration
# ============================================================================
"""
    PentrcOutput

Output file and formatting configuration.
"""
struct PentrcOutput
    output_dir::String
    write_orbit_ascii::Bool
    write_orbit_netcdf::Bool
    write_pitch_ascii::Bool
    write_pitch_netcdf::Bool
    write_energy_ascii::Bool
    write_energy_netcdf::Bool
    write_torque_ascii::Bool
    write_torque_netcdf::Bool
    write_elmat_ascii::Bool
    fname_format::String               # "ascii", "netcdf", "both"
end

function PentrcOutput(; output_dir::String=".", 
                      write_orbit_ascii::Bool=false,
                      write_orbit_netcdf::Bool=false,
                      write_pitch_ascii::Bool=false,
                      write_pitch_netcdf::Bool=false,
                      write_energy_ascii::Bool=false,
                      write_energy_netcdf::Bool=false,
                      write_torque_ascii::Bool=true,
                      write_torque_netcdf::Bool=false,
                      write_elmat_ascii::Bool=false,
                      fname_format::String="ascii")
    return PentrcOutput(
        output_dir,
        write_orbit_ascii,
        write_orbit_netcdf,
        write_pitch_ascii,
        write_pitch_netcdf,
        write_energy_ascii,
        write_energy_netcdf,
        write_torque_ascii,
        write_torque_netcdf,
        write_elmat_ascii,
        fname_format
    )
end
