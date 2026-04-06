"""
    Input

Input reading and parsing for PENTRC module.
Handles namelists, TOML files, and equilibrium data.
"""

using TOML
using DelimitedFiles

"""
    read_pentrc_config(config_path::String)::PentrcControl

Read PENTRC configuration from TOML file and return PentrcControl structure.
Provides sensible defaults for missing parameters.

# Arguments
- `config_path::String`: Path to pentrc.toml configuration file

# Returns
- `PentrcControl`: Control structure with all parameters
"""
function read_pentrc_config(config_path::String)::PentrcControl
    
    # Load TOML file
    if !isfile(config_path)
        error("Configuration file not found: $config_path")
    end
    cfg = TOML.parsefile(config_path)
    
    # Get PENTRC section or use whole file
    pentrc_cfg = get(cfg, "PENTRC", cfg)
    
    # Helper function for safe extraction with defaults
    function get_cfg(key::String, default)
        return get(pentrc_cfg, key, default)
    end
    
    # Moment type
    moment = get_cfg("moment", "pressure")
    qt = moment == "heat"
    
    # Method flags
    fgar_flag = get_cfg("fgar_flag", true)
    tgar_flag = get_cfg("tgar_flag", false)
    pgar_flag = get_cfg("pgar_flag", false)
    rlar_flag = get_cfg("rlar_flag", false)
    clar_flag = get_cfg("clar_flag", false)
    fcgl_flag = get_cfg("fcgl_flag", false)
    wxyz_flag = get_cfg("wxyz_flag", false)
    fkmm_flag = get_cfg("fkmm_flag", false)
    tkmm_flag = get_cfg("tkmm_flag", false)
    pkmm_flag = get_cfg("pkmm_flag", false)
    frmm_flag = get_cfg("frmm_flag", false)
    trmm_flag = get_cfg("trmm_flag", false)
    prmm_flag = get_cfg("prmm_flag", false)
    fwmm_flag = get_cfg("fwmm_flag", false)
    twmm_flag = get_cfg("twmm_flag", false)
    pwmm_flag = get_cfg("pwmm_flag", false)
    ftmm_flag = get_cfg("ftmm_flag", false)
    ttmm_flag = get_cfg("ttmm_flag", false)
    ptmm_flag = get_cfg("ptmm_flag", false)
    
    # Output flags
    eq_out = get_cfg("eq_out", false)
    theta_out = get_cfg("theta_out", false)
    xlmda_out = get_cfg("xlmda_out", false)
    eqpsi_out = get_cfg("eqpsi_out", false)
    output_ascii = get_cfg("output_ascii", true)
    output_netcdf = get_cfg("output_netcdf", false)
    output_orbit = get_cfg("output_orbit", false)
    output_ffuns = get_cfg("output_ffuns", false)
    output_torque = get_cfg("output_torque", true)
    
    # Grid flags
    equil_grid = get_cfg("equil_grid", false)
    input_grid = get_cfg("input_grid", false)
    dynamic_grid = get_cfg("dynamic_grid", true)
    
    # Diagnostic flags
    fnml_flag = get_cfg("fnml_flag", false)
    ellip_flag = get_cfg("ellip_flag", false)
    diag_flag = get_cfg("diag_flag", false)
    term_flag = get_cfg("term_flag", false)
    force_xialpha = get_cfg("force_xialpha", false)
    clean = get_cfg("clean", true)
    
    # Plasma species
    zi = Int(get_cfg("zi", 1))
    mi = Int(get_cfg("mi", 2))
    zimp = Int(get_cfg("zimp", 6))
    mimp = Int(get_cfg("mimp", 12))
    electron = get_cfg("electron", false)
    
    # Mode numbers
    nl = Int(get_cfg("nl", 0))
    nn = Int(get_cfg("nn", 1))
    
    # Grid parameters
    tmag_in = Int(get_cfg("tmag_in", 1))
    jsurf_in = Int(get_cfg("jsurf_in", 0))
    power_bin = Int(get_cfg("power_bin", -1))
    power_bpin = Int(get_cfg("power_bpin", -1))
    power_rin = Int(get_cfg("power_rin", -1))
    power_rcin = Int(get_cfg("power_rcin", -1))
    
    # Thread control
    pentrc_threads = Int(get_cfg("pentrc_threads", 0))
    openmp_threads = Int(get_cfg("openmp_threads", 0))
    
    # Tolerances
    atol_xlmda = Float64(get_cfg("atol_xlmda", 1e-6))
    rtol_xlmda = Float64(get_cfg("rtol_xlmda", 1e-3))
    
    # Scaling factors
    nfac = Float64(get_cfg("nfac", 1.0))
    tfac = Float64(get_cfg("tfac", 1.0))
    wefac = Float64(get_cfg("wefac", 1.0))
    wdfac = Float64(get_cfg("wdfac", 1.0))
    wpfac = Float64(get_cfg("wpfac", 1.0))
    nufac = Float64(get_cfg("nufac", 1.0))
    divxfac = Float64(get_cfg("divxfac", 1.0))
    
    # Diagnostic parameters
    diag_psi = Float64(get_cfg("diag_psi", 0.7))
    psi_out_raw = get_cfg("psi_out", fill(-1.0, 30))
    psi_out = isa(psi_out_raw, Vector) ? Float64.(psi_out_raw) : [Float64(psi_out_raw)]
    psilims = Float64.(get_cfg("psilims", [0.0, 1.0]))
    
    # Set default psi_out if user didn't specify
    if all(psi_out .== -1)
        psi_out = collect(1:30) ./ 30.6
    end
    
    # File and path parameters
    data_dir = get_cfg("data_dir", ".")
    nutype = get_cfg("nutype", "harmonic")
    f0type = get_cfg("f0type", "maxwellian")
    jac_in = get_cfg("jac_in", "default")
    
    idconfile = get_cfg("idconfile", "euler.bin")
    kinetic_file = get_cfg("kinetic_file", "kin.dat")
    gpec_file = get_cfg("gpec_file", "gpec_order1_n1.bin")
    peq_file = get_cfg("peq_file", "")
    pmodb_file = get_cfg("pmodb_file", "none")
    
    # Debugging
    verbose = get_cfg("verbose", false)
    indebug = get_cfg("indebug", false)
    
    # Create and return control structure
    return PentrcControl(
        moment, qt,
        fgar_flag, tgar_flag, pgar_flag, rlar_flag, clar_flag, fcgl_flag, wxyz_flag,
        fkmm_flag, tkmm_flag, pkmm_flag, frmm_flag, trmm_flag, prmm_flag,
        fwmm_flag, twmm_flag, pwmm_flag, ftmm_flag, ttmm_flag, ptmm_flag,
        eq_out, theta_out, xlmda_out, eqpsi_out,
        output_ascii, output_netcdf, output_orbit, output_ffuns, output_torque,
        equil_grid, input_grid, dynamic_grid,
        fnml_flag, ellip_flag, diag_flag, term_flag, force_xialpha, clean,
        zi, mi, zimp, mimp, electron,
        nl, nn,
        tmag_in, jsurf_in, power_bin, power_bpin, power_rin, power_rcin,
        pentrc_threads, openmp_threads,
        atol_xlmda, rtol_xlmda,
        nfac, tfac, wefac, wdfac, wpfac, nufac, divxfac,
        diag_psi, psi_out, psilims,
        data_dir, nutype, f0type, jac_in,
        idconfile, kinetic_file, gpec_file, peq_file, pmodb_file,
        verbose, indebug
    )
end

"""
    get_method_flags(ctrl::PentrcControl)::Vector{Bool}

Extract method flags from control structure in correct order.

# Returns
- `Vector{Bool}`: Flags in order [fgar, tgar, ..., prmm]
"""
function get_method_flags(ctrl::PentrcControl)::Vector{Bool}
    return [
        ctrl.fgar_flag, ctrl.tgar_flag, ctrl.pgar_flag, 
        ctrl.rlar_flag, ctrl.clar_flag, ctrl.fcgl_flag,
        ctrl.fwmm_flag, ctrl.twmm_flag, ctrl.pwmm_flag,
        ctrl.ftmm_flag, ctrl.ttmm_flag, ctrl.ptmm_flag,
        ctrl.fkmm_flag, ctrl.tkmm_flag, ctrl.pkmm_flag,
        ctrl.frmm_flag, ctrl.trmm_flag, ctrl.prmm_flag
    ]
end


# ============================================================================
# Initialization functions (PENTRC setup)
# ============================================================================

"""
    initialize_pentrc(ctrl::PentrcControl, equil)

Initialize PENTRC with equilibrium and kinetic data.
Called both when PENTRC runs standalone and when ForceFreeStates kinetic_flag=true.

# Arguments
- `ctrl::PentrcControl`: Configuration from config file
- `equil`: Equilibrium structure from ForceFreeStates

# Side effects
- Loads kinetic profiles
- Sets up equilibrium splines
- Initializes perturbation data
"""
function initialize_pentrc(ctrl::PentrcControl, equil)
    
    # TODO: Implement full initialization
    # 1. Read kinetic profiles (read_kin)
    # 2. Set up equilibrium (set_eq)
    # 3. Read perturbations (read_peq, set_peq)
    # 4. Validate all data
    
    if ctrl.verbose
        println("Initializing PENTRC...")
    end
    
    # Stub implementation - will be expanded
    return nothing
end


"""
    read_equil(equil; verbose=false)

Initialize PENTRC equilibrium data from a PlasmaEquilibrium.
In the Julia implementation, equilibrium data is passed directly in memory
rather than read from binary files as in the Fortran PENTRC.

# Arguments
- `equil`: PlasmaEquilibrium from Equilibrium module
- `verbose::Bool`: Print progress messages
"""
function read_equil(equil; verbose=false)
    if verbose
        println("Initializing PENTRC equilibrium from PlasmaEquilibrium")
    end
    # TODO: Extract PENTRC-specific profiles (sq, geom, kin) from equil
    return nothing
end


"""
    read_kin(kin_file::String; zi=1, zimp=6, mi=2, mimp=12, 
             nfac=1.0, tfac=1.0, wefac=1.0, wpfac=1.0, 
             write_log=false, verbose=false)

Read kinetic profiles (density, temperature, etc.) from ASCII file.
Creates spline interpolants for use in torque calculations.

# Arguments
- `kin_file::String`: Path to kinetic profile data file
- `zi::Int`: Ion charge in fundamental units
- `zimp::Int`: Impurity charge
- `mi::Int`: Ion mass in units of proton mass
- `mimp::Int`: Impurity mass in units of proton mass
- `nfac::Float64`: Density scaling factor
- `tfac::Float64`: Temperature scaling factor
- `wefac::Float64`: ExB rotation direct scaling
- `wpfac::Float64`: Indirect rotation scaling (through omegae)
- `write_log::Bool`: Write diagnostic output
- `verbose::Bool`: Print progress messages

# Returns
- Spline structure with kinetic profiles (kin)

Expected file format: 6 columns
  psi_n  n_i(m^-3)  n_e(m^-3)  T_i(eV)  T_e(eV)  omega_E(rad/s)
"""
function read_kin(kin_file::String; zi=1, zimp=6, mi=2, mimp=12,
                  nfac=1.0, tfac=1.0, wefac=1.0, wpfac=1.0, 
                  write_log=false, verbose=false)
    
    if !isfile(kin_file)
        error("Kinetic profile file not found: $kin_file")
    end

    # Read ASCII table (skipping header lines)
    table = DelimitedFiles.readdlm(kin_file)
    
    # Filter out non-numeric rows
    table_clean = Float64[]
    for row in eachrow(table)
        if all(x -> isa(x, Number), row)
            push!(table_clean, vec(row)...)
        end
    end
    table = reshape(table_clean, :, 6)
    
    if write_log
        println("Table shape: ", size(table))
    end
    
    # Physical constants (use module-level PENTRC constants where possible)
    eV_to_J = 1.602e-19
    
    ndata = size(table, 1)
    nkin = 100
    
    # Create splines for input data first
    psi_input = @view table[:, 1]
    n_i_input = @view table[:, 2]
    n_e_input = @view table[:, 3]
    T_i_input = @view table[:, 4]  # in eV
    T_e_input = @view table[:, 5]  # in eV
    omega_e_input = @view table[:, 6]
    
    # Initialize output arrays for regular grid (0:nkin steps)
    psi_reg = collect(0:nkin) ./ nkin
    kin_data = zeros(nkin+1, 10)  # 10 quantities: psi, n_i, n_e, T_i, T_e, omega_e, loglam, nu_i, nu_e, zeff
    
    # Interpolate input data to regular psi grid
    kin_data[:, 1] = psi_reg
    kin_data[:, 2] = linear_interp_extrap(psi_input, n_i_input, psi_reg)
    kin_data[:, 3] = linear_interp_extrap(psi_input, n_e_input, psi_reg)
    kin_data[:, 4] = linear_interp_extrap(psi_input, T_i_input, psi_reg)
    kin_data[:, 5] = linear_interp_extrap(psi_input, T_e_input, psi_reg)
    kin_data[:, 6] = linear_interp_extrap(psi_input, omega_e_input, psi_reg)
    
    # Convert temperatures from eV to J
    kin_data[:, 4] = kin_data[:, 4] .* eV_to_J
    kin_data[:, 5] = kin_data[:, 5] .* eV_to_J
    
    # Calculate collisionalities (assumes SI units for n, T)
    zeff = similar(kin_data[:, 1])
    zpitch = similar(kin_data[:, 1])
    
    for i in 1:(nkin+1)
        n_i = kin_data[i, 2]
        n_e = kin_data[i, 3]
        T_i = kin_data[i, 4]  # in Joules
        T_e = kin_data[i, 5]  # in Joules
        
        if n_e > 0
            zeff[i] = zimp - (n_i / n_e) * zi * (zimp - zi)
        else
            zeff[i] = zimp
        end
        
        zpitch[i] = 1.0 + (1.0 + mimp) / (2.0 * mimp) * zimp * (zeff[i] - 1.0) / (zimp - zeff[i])
        
        # Coulomb logarithm
        loglam = 17.3 - 0.5 * log10(n_e / 1.0e20) + 1.5 * log10(T_e / 1.602e-16)
        kin_data[i, 7] = loglam
        
        # Ion collision frequency
        if T_i > 0
            kin_data[i, 8] = (zpitch[i] / 3.5e17) * n_i * loglam / 
                            (sqrt(1.0 * mi) * (T_i / 1.602e-16)^1.5)
        else
            kin_data[i, 8] = 0.0
        end
        
        # Electron collision frequency
        if T_e > 0
            kin_data[i, 9] = (zpitch[i] / 3.5e17) * n_e * loglam /
                            (sqrt(me / mp) * (T_e / 1.602e-16)^1.5)
        else
            kin_data[i, 9] = 0.0
        end
        
        kin_data[i, 10] = zeff[i]
    end
    
    if write_log
        println("Formed kinetic profile on $(nkin+1) point grid")
    end
    
    # Apply profile manipulations
    kin_data[:, 2] = kin_data[:, 2] .* nfac  # n_i scaling
    kin_data[:, 3] = kin_data[:, 3] .* nfac  # n_e scaling
    kin_data[:, 4] = kin_data[:, 4] .* tfac  # T_i scaling
    kin_data[:, 5] = kin_data[:, 5] .* tfac  # T_e scaling
    
    # Direct manipulation of omega_ExB
    welec = kin_data[:, 6] .* wefac
    
    # Avoid division by zero
    warning_printed = false
    for i in 1:(nkin+1)
        if welec[i] == 0
            welec[i] = 1e-9
            if !warning_printed
                println("  ! WARNING ! Modifying omega_ExB = 0 to 1e-9 to avoid nans")
                warning_printed = true
            end
        end
    end
    
    if wefac != 1.0 && verbose
        println("  -> applying direct manipulation of omegaE by factor ", wefac)
    end
    
    # Indirect rotation manipulation (through omegae)
    # This requires derivatives of density, which we approximate numerically
    wdian = zeros(nkin+1)
    wdiat = zeros(nkin+1)
    
    for i in 2:(nkin)
        dn_i_dpsi = (kin_data[i+1, 2] - kin_data[i-1, 2]) / (2.0 / nkin)
        dT_i_dpsi = (kin_data[i+1, 4] - kin_data[i-1, 4]) / (2.0 / nkin)
        
        chi1 = 1.0  # TODO: Get from ForceFreeStates equilibrium
        
        wdian[i] = -twopi * kin_data[i, 4] * dn_i_dpsi / (e * zi * chi1 * kin_data[i, 2])
        wdiat[i] = -twopi * dT_i_dpsi / (e * zi * chi1)
    end
    
    # Endpoints
    wdian[1] = wdian[2]
    wdian[nkin+1] = wdian[nkin]
    wdiat[1] = wdiat[2]
    wdiat[nkin+1] = wdiat[nkin]
    
    # Apply indirect rotation scaling
    for i in 1:(nkin+1)
        if welec[i] != 0
            wpefac_i = (wpfac * (welec[i] + wdian[i] + wdiat[i]) - (wdian[i] + wdiat[i])) / welec[i]
            welec[i] = wpefac_i * welec[i]
        end
    end
    
    kin_data[:, 6] = welec
    
    if wpfac != 1.0 && verbose
        println("  -> manipulating rotation by factor of ", wpfac)
        println("     by indirect manipulation of omegae profile")
    end
    
    # Write log file for debugging
    if write_log
        open("pentrc_kinetics.out", "w") do f
            write(f, "PENTRC KINETIC PROFILE OUTPUT\n")
            write(f, "psi_n\t\tn_i\t\tn_e\t\tT_i\t\tT_e\t\tomegaE\t\tloglam\t\tnu_i\t\tnu_e\t\tzeff\n")
            for i in 1:(nkin+1)
                write(f, @sprintf("%.6e\t%.6e\t%.6e\t%.6e\t%.6e\t%.6e\t%.6e\t%.6e\t%.6e\t%.6e\n",
                    kin_data[i, 1], kin_data[i, 2], kin_data[i, 3], kin_data[i, 4], kin_data[i, 5],
                    kin_data[i, 6], kin_data[i, 7], kin_data[i, 8], kin_data[i, 9], kin_data[i, 10]))
            end
        end
        println("Wrote kinetic spline to pentrc_kinetics.out")
    end
    
    return kin_data
end


# Helper function for linear interpolation with extrapolation
function linear_interp_extrap(x::AbstractVector, y::AbstractVector, x_new::AbstractVector)
    y_new = similar(x_new, Float64)
    for (i, xval) in enumerate(x_new)
        if xval < x[1]
            # Linear extrapolation below
            slope = (y[2] - y[1]) / (x[2] - x[1])
            y_new[i] = y[1] + slope * (xval - x[1])
        elseif xval > x[end]
            # Linear extrapolation above
            slope = (y[end] - y[end-1]) / (x[end] - x[end-1])
            y_new[i] = y[end] + slope * (xval - x[end])
        else
            # Linear interpolation
            idx = searchsortin(x, xval) - 1
            if idx < 1
                idx = 1
            end
            if idx >= length(x)
                idx = length(x) - 1
            end
            t = (xval - x[idx]) / (x[idx+1] - x[idx])
            y_new[i] = y[idx] * (1 - t) + y[idx+1] * t
        end
    end
    return y_new
end


"""
    read_peq(peq_file::String; jac_in="default", jsurf_in=0, tmag_in=1,
             force_xialpha=false, debug=false, op_powin=nothing, verbose=false)

Read perturbation equilibrium displacement profiles from ASCII file.
Expects file with columns: psi, m, xi^psi_real, xi^psi_imag, 
                                    xi'^psi_real, xi'^psi_imag,
                                    xi^alpha_real, xi^alpha_imag

# Arguments
- `peq_file::String`: Path to perturbation equilibrium data file
- `jac_in::String`: Input jacobian type ("hamada", "pest", "boozer", "park", "polar", etc.)
- `jsurf_in::Int`: Surface weighted input (1=yes, 0=no)
- `tmag_in::Int`: Toroidal angle spec (1=magnetic, 0=cylindrical)
- `force_xialpha::Bool`: Recalculate xi^alpha from xi^psi using force balance
- `debug::Bool`: Print intermediate debug messages
- `op_powin::Vector`: Powers of [B, Bp, R, Rc] for jac_in="other"
- `verbose::Bool`: Print progress messages

# Returns
- Perturbation mode data structure with xi components as splines
"""
function read_peq(peq_file::String; jac_in="default", jsurf_in=0, tmag_in=1,
                  force_xialpha=false, debug=false, op_powin=nothing, verbose=false)
    
    if !isfile(peq_file)
        error("Perturbation equilibrium file not found: $peq_file")
    end
    
    # TODO: Full implementation of ASCII peq reading
    # This is complex - requires:
    # 1. Reading ASCII table with complex mode data
    # 2. Detecting psi-outer vs m-outer loop ordering
    # 3. Coordinate transformation to ForceFreeStates jacobian
    # 4. Mode spectrum interpolation (newm function)
    # 5. Clebsch to flux surface decomposition
    
    if debug
        println("Reading perturbation equilibrium from: $peq_file")
    end
    
    # For now, delegate to read_gpec_peq for binary format
    if endswith(peq_file, ".bin") || endswith(peq_file, ".dat")
        return read_gpec_peq(peq_file; write_log=debug, verbose=verbose)
    end
    
    error("ASCII peq reading not yet implemented - use GPEC binary format (.bin)")
end


"""
    read_gpec_peq(gpec_file::String; write_log=false, verbose=false)

Read GPEC binary format perturbation equilibrium file.
Contains pre-computed dB/B and div(xi_perp) mode matrices.

File format:
  - ms, mp: mode numbers
  - lagbpar(mstep, mpert): dB/B mode matrix
  - divxprp(mstep, mpert): div(xi_perp) mode matrix

# Arguments
- `gpec_file::String`: Path to GPEC binary file
- `write_log::Bool`: Write diagnostic output
- `verbose::Bool`: Print progress messages

# Returns
- Data structure with dbob_m and divx_m complex splines
"""
function read_gpec_peq(gpec_file::String; write_log=false, verbose=false)
    
    if !isfile(gpec_file)
        error("GPEC perturbation file not found: $gpec_file")
    end
    
    if verbose
        println("Reading GPEC perturbation equilibrium from: $gpec_file")
    end
    
    # For Julia, we'll support both binary and HDF5 formats
    if endswith(gpec_file, ".h5") || endswith(gpec_file, ".hdf5")
        # HDF5 format
        h5open(gpec_file, "r") do fid
            ms = read(fid, "ms")
            mp = read(fid, "mp")
            lagbpar = read(fid, "lagbpar")
            divxprp = read(fid, "divxprp")
            
            return (; ms=ms, mp=mp, lagbpar=lagbpar, divxprp=divxprp)
        end
    else
        # Binary format support (Julia Fortran-compatible binary I/O)
        open(gpec_file, "r") do io
            # Read binary formatted data
            # Format: Fortran unformatted sequential access
            # Each record starts with record length (4 bytes), then data, then record length again
            
            # Skip record markers (they depend on Fortran compiler)
            # Try to read as if it's a simple binary dump
            ms = read(io, Int32)
            mp = read(io, Int32)
            
            if verbose
                println("  -> GPEC file has $ms steps in psi, $mp modes")
            end
            
            # Read complex matrices
            lagbpar = zeros(ComplexF64, ms, mp)
            divxprp = zeros(ComplexF64, ms, mp)
            
            for i in 1:ms, j in 1:mp
                re = read(io, Float64)
                im = read(io, Float64)
                lagbpar[i, j] = re + 1im * im
            end
            
            for i in 1:ms, j in 1:mp
                re = read(io, Float64)
                im = read(io, Float64)
                divxprp[i, j] = re + 1im * im
            end
            
            return (; ms=ms, mp=mp, lagbpar=lagbpar, divxprp=divxprp)
        end
    end
end


"""
    set_peq(psi::Vector, ms::Vector, xmp1mns::Matrix, xspmns::Matrix, xmsmns::Matrix;
            set_dbdx=true, debug=false)

Set perturbation equilibrium data after reading from file.
Decomposes Clebsch displacements into Fourier modes and computes dB/B, div(ξ_perp).

# Arguments
- `psi::Vector`: Normalized poloidal flux grid
- `ms::Vector`: Poloidal mode numbers
- `xmp1mns::Matrix`: xi^(psi-1) component (psi, m)
- `xspmns::Matrix`: xi^psi component (psi, m)  
- `xmsmns::Matrix`: xi^alpha component (psi, m) [in ForceFreeStates units/chi1]
- `set_dbdx::Bool`: Calculate and set dB/B and div splines (default=true)
- `debug::Bool`: Print intermediate messages

# Side effects
Updates global xs_m, dbob_m, divx_m splines for all perturbation components.

# Notes
Requires ForceFreeStates matrix structure access for metric calculations.
"""
function set_peq(psi::Vector, ms::Vector, xmp1mns::Matrix, xspmns::Matrix, xmsmns::Matrix;
                 set_dbdx=true, debug=false)
    
    # TODO: Full implementation requires ForceFreeStates metric matrices
    # 1. Allocate xs_m splines for 3 components
    # 2. Fill from input (psi, m) -> (psi, mfac) mapping
    # 3. Calculate dB/B = -xi_psi' + (S-matrix operations)
    # 4. Calculate div(xi_perp) = -(dB/B + kappa·xi_perp)
    # 5. Fit resulting splines with extrapolation
    
    if debug
        println("Setting perturbation equilibrium")
        println("  psi range: $(psi[1]) to $(psi[end])")
        println("  input modes: $ms")
    end
    
    npsi = length(psi)
    nm = length(ms)
    
    if debug
        println("  $(npsi) flux surfaces, $(nm) modes")
    end
    
    # For now, stub implementation
    # Full implementation needs ForceFreeStates accessor functions
    ## xs_m splines would be set here
    ## dbob_m and divx_m would be calculated
    
    return nothing
end


"""
    set_eq(equil)

Set up equilibrium from equilibrium structure.
Initializes all equilibrium splines and parameters needed for PENTRC.

# Arguments
- `equil`: Equilibrium structure (typically from ForceFreeStates)
"""
function set_eq(equil)
    # TODO: Implement equilibrium setup
    # 1. Extract flux functions from equil structure
    # 2. Copy/reference sq, geom, rzphi, eqfun splines
    # 3. Set global variables: ro, bo, chi1, shotnum, shottime, machine
    
    return nothing
end


"""
    read_fnml(fnml_file::String; verbose=false)

Read Fourier mode coupling data (pre-integrated special functions).
Provides F^-1/2_mnl matrix used for bounce-averaged kappa integrals.

From [Park, Phys. Rev. Let. 2009], Eq. (13).

File format (binary):
  - nfk, nft: dimensions (integers)
  - xs(0:nfk): first dimension grid
  - ys(0:nft): second dimension grid
  - fs(0:nfk, 0:nft): function values (complex)

# Arguments
- `fnml_file::String`: Path to Fourier mode coupling file
- `verbose::Bool`: Print progress messages

# Returns
- fnml data structure with xs, ys, fs arrays
"""
function read_fnml(fnml_file::String; verbose=false)
    
    if !isfile(fnml_file)
        error("Fourier mode coupling file not found: $fnml_file")
    end
    
    if verbose
        println("Reading Fourier mode coupling data from: $fnml_file")
    end
    
    # Support both binary and HDF5 formats
    if endswith(fnml_file, ".h5") || endswith(fnml_file, ".hdf5")
        # HDF5 format
        h5open(fnml_file, "r") do fid
            nfk = read(fid, "nfk")
            nft = read(fid, "nft")
            xs = read(fid, "xs")
            ys = read(fid, "ys")
            fs = read(fid, "fs")
            
            return (; nfk=nfk, nft=nft, xs=xs, ys=ys, fs=fs)
        end
    else
        # Binary format (Fortran unformatted sequential)
        open(fnml_file, "r") do io
            # Read dimensions
            nfk = read(io, Int32)
            nft = read(io, Int32)
            
            if verbose
                println("  -> fnml grid: $nfk x $nft")
            end
            
            # Read xs grid (nfk+1 values)
            xs = zeros(Float64, nfk+1)
            for i in 1:(nfk+1)
                xs[i] = read(io, Float64)
            end
            
            # Read ys grid (nft+1 values)
            ys = zeros(Float64, nft+1)
            for i in 1:(nft+1)
                ys[i] = read(io, Float64)
            end
            
            # Read function values (complex, (nfk+1) x (nft+1))
            fs = zeros(ComplexF64, nfk+1, nft+1)
            for j in 1:(nft+1), i in 1:(nfk+1)
                re = read(io, Float64)
                im = read(io, Float64)
                fs[i, j] = re + 1im * im
            end
            
            return (; nfk=nfk, nft=nft, xs=xs, ys=ys, fs=fs)
        end
    end
end


# ============================================================================
# Perturbation equilibrium reading and setup
# ============================================================================

"""
    read_pmodb(pmodb_file::String; jac_in="default", jsurf_in=0, tmag_in=1,
               debug=false, op_powin=nothing, verbose=false)

Read psi-m matrix of Lagrangian dB/B and div(xi_perp) from file.

Converts between different jacobian coordinate systems (PEST, Boozer, etc.)
and interpolates to ForceFreeStates mode spectrum (mfac).

# Arguments
- `pmodb_file::String`: Path to perturbation equilibrium matrix file
- `jac_in::String`: Input jacobian type ("hamada", "pest", "boozer", "park", "polar")
- `jsurf_in::Int`: Surface-weighted input (0 or 1)
- `tmag_in::Int`: Toroidal angle (1=magnetic, 0=cylindrical)
- `debug::Bool`: Print detailed debug output
- `op_powin::Vector{Int}`: Powers [B, Bp, R, Rc] for jacobian="other"
- `verbose::Bool`: Print progress

# Side effects
Sets global dbob_m and divx_m complex splines after 1/B weighting.

Expected file columns:
  psi, m, real(dB/B), imag(dB/B), real(div), imag(div), real(kappa), imag(kappa)
"""
function read_pmodb(pmodb_file::String; jac_in="default", jsurf_in=0, tmag_in=1,
                    debug=false, op_powin=nothing, verbose=false)
    
    if !isfile(pmodb_file)
        error("Perturbation matrix file not found: $pmodb_file")
    end
    
    if verbose
        println("Reading perturbation equilibrium matrix from: $pmodb_file")
    end
    
    # Read ASCII table
    table = DelimitedFiles.readdlm(pmodb_file)
    
    # Filter numeric rows
    table_clean = Float64[]
    for row in eachrow(table)
        if all(x -> isa(x, Number), row)
            push!(table_clean, vec(row)...)
        end
    end
    
    if isempty(table_clean)
        error("No numeric data found in $pmodb_file")
    end
    
    table = reshape(table_clean, :, 8)  # Expected: 8 columns
    
    if debug
        println("  -> Read $(size(table,1)) rows x $(size(table,2)) columns")
    end
    
    # TODO: Complete implementation requires:
    # 1. Unique extraction of psi and m grids
    # 2. Jacobian coordinate transformation
    # 3. Mode spectrum interpolation (newm function)
    # 4. 1/B weighting with Lagrangian mod-B spline
    # 5. Complex spline fitting with extrapolation
    
    # For now, return parsed data
    return table
end
