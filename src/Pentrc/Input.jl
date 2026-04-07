"""
    Input

Input reading and parsing for PENTRC module.
Handles namelists, TOML files, and equilibrium data.
Currently this file just reads the euler.bin but after integrating with kinetic part in DCON these will be replaced with julia file input. 
"""

using TOML
using DelimitedFiles
import ..Spl

function _read_fortran_record(io::IO)
    n = read(io, Int32)
    n < 0 && error("Unsupported negative Fortran record marker: $n")
    data = read(io, n)
    n2 = read(io, Int32)
    n == n2 || error("Fortran record marker mismatch: $n != $n2")
    return data
end

function _read_idcon_header(idconfile::String)
    isfile(idconfile) || error("idconfile not found: $idconfile")
    open(idconfile, "r") do io
        rec1 = IOBuffer(_read_fortran_record(io))
        mlow = read(rec1, Int32)
        mhigh = read(rec1, Int32)
        nn = read(rec1, Int32)
        mpsi = read(rec1, Int32)
        mtheta = read(rec1, Int32)
        ro = read(rec1, Float64)
        zo = read(rec1, Float64)

        rec2 = IOBuffer(_read_fortran_record(io))
        bband = read(rec2, Int32)
        mthsurf0 = read(rec2, Float64)
        mthvac = read(rec2, Int32)
        psio = read(rec2, Float64)
        psilow = read(rec2, Float64)
        psilim = read(rec2, Float64)
        qlim = read(rec2, Float64)
        singfac_min = read(rec2, Float64)

        return (
            mlow=Int(mlow),
            mhigh=Int(mhigh),
            nn=Int(nn),
            mpsi=Int(mpsi),
            mtheta=Int(mtheta),
            ro=ro,
            zo=zo,
            bband=Int(bband),
            mthsurf0=mthsurf0,
            mthvac=Int(mthvac),
            psio=psio,
            psilow=psilow,
            psilim=psilim,
            qlim=qlim,
            singfac_min=singfac_min,
        )
    end
end

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
Called both when PENTRC runs standalone and when DCON kinetic_flag=true.

# Arguments
- `ctrl::PentrcControl`: Configuration from config file
- `equil`: Equilibrium structure from DCON

# Side effects
- Loads kinetic profiles
- Sets up equilibrium splines
- Initializes perturbation data
"""
function initialize_pentrc(ctrl::PentrcControl, equil)
    global verbose = ctrl.verbose
    global tdebug = ctrl.indebug
    global qt = ctrl.qt

    if ctrl.verbose
        println("Initializing PENTRC...")
    end

    idcon_path = ""
    if !isempty(ctrl.idconfile)
        idcon_path = isabspath(ctrl.idconfile) ? ctrl.idconfile : joinpath(pwd(), ctrl.idconfile)
        if !isfile(idcon_path)
            idcon_path = ctrl.idconfile
        end
    end

    load_equilibrium!(equil; idconfile=isfile(idcon_path) ? idcon_path : nothing)
    global nn = ctrl.nn

    if !isempty(ctrl.kinetic_file) && isfile(ctrl.kinetic_file)
        load_kinetic_profiles!(
            ctrl.kinetic_file;
            zi=ctrl.zi,
            zimp=ctrl.zimp,
            mi=ctrl.mi,
            mimp=ctrl.mimp,
            nfac=ctrl.nfac,
            tfac=ctrl.tfac,
            wefac=ctrl.wefac,
            wpfac=ctrl.wpfac,
            write_log=ctrl.indebug,
            verbose=ctrl.verbose,
        )
    end

    if (ctrl.rlar_flag || ctrl.clar_flag) && !isempty(ctrl.data_dir)
        fnml_file = joinpath(ctrl.data_dir, "fkmnl.dat")
        if isfile(fnml_file)
            load_fnml!(fnml_file; verbose=ctrl.verbose)
        end
    end

    load_perturbation_input!(ctrl)

    return nothing
end

function load_perturbation_input!(ctrl::PentrcControl)
    if !isempty(ctrl.peq_file) && lowercase(ctrl.peq_file) != "none" && isfile(ctrl.peq_file)
        return load_peq!(
            ctrl.peq_file;
            jac_in=ctrl.jac_in,
            jsurf_in=ctrl.jsurf_in,
            tmag_in=ctrl.tmag_in,
            force_xialpha=ctrl.force_xialpha,
            debug=ctrl.indebug,
            verbose=ctrl.verbose,
        )
    elseif !isempty(ctrl.gpec_file) && lowercase(ctrl.gpec_file) != "none" && isfile(ctrl.gpec_file)
        return load_gpec_peq!(ctrl.gpec_file; verbose=ctrl.verbose)
    elseif !isempty(ctrl.pmodb_file) && lowercase(ctrl.pmodb_file) != "none" && isfile(ctrl.pmodb_file)
        return load_pmodb!(
            ctrl.pmodb_file;
            jac_in=ctrl.jac_in,
            jsurf_in=ctrl.jsurf_in,
            tmag_in=ctrl.tmag_in,
            debug=ctrl.indebug,
            verbose=ctrl.verbose,
        )
    end

    return nothing
end

"""
    load_kinetic_profiles!(kin_file::String; zi=1, zimp=6, mi=2, mimp=12,
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
function load_kinetic_profiles!(kin_file::String; zi=1, zimp=6, mi=2, mimp=12,
                                nfac=1.0, tfac=1.0, wefac=1.0, wpfac=1.0,
                                write_log=false, verbose=false)
    
    if !isfile(kin_file)
        error("Kinetic profile file not found: $kin_file")
    end
    
    table, _ = readtable(kin_file; verbose=verbose, debug=write_log)

    if size(table, 2) < 6
        error("Kinetic profile file must contain at least 6 columns: psi, n_i, n_e, T_i, T_e, omega_E")
    end

    if write_log
        println("Table shape: ", size(table))
    end

    eV_to_J = 1.602e-19
    nkin = 100

    psi_reg = collect(0:nkin) ./ nkin

    tmp = Spl.CubicSpline(Float64.(table[:, 1]), Float64.(table[:, 2:6]); bctype="extrap")
    kin_fs = zeros(nkin + 1, 9)
    for (i, psi) in enumerate(psi_reg)
        kin_fs[i, 1:5] .= Spl.spline_eval!(tmp, psi)
    end

    kin_fs[:, 3:4] .*= eV_to_J

    zeff = zimp .- (kin_fs[:, 1] ./ kin_fs[:, 2]) .* zi .* (zimp - zi)
    zpitch = 1.0 .+ ((1.0 + mimp) / (2.0 * mimp)) .* zimp .* (zeff .- 1.0) ./ (zimp .- zeff)
    kin_fs[:, 6] .= 17.3 .- 0.5 .* log.(kin_fs[:, 2] ./ 1.0e20) .+ 1.5 .* log.(kin_fs[:, 4] ./ 1.602e-16)
    kin_fs[:, 7] .= (zpitch ./ 3.5e17) .* kin_fs[:, 1] .* kin_fs[:, 6] ./
                    (sqrt(1.0 * mi) .* (kin_fs[:, 3] ./ 1.602e-16) .^ 1.5)
    kin_fs[:, 8] .= (zpitch ./ 3.5e17) .* kin_fs[:, 2] .* kin_fs[:, 6] ./
                    (sqrt(me / mp) .* (kin_fs[:, 4] ./ 1.602e-16) .^ 1.5)
    kin_fs[:, 9] .= zeff

    if write_log
        println("Formed kinetic profile on $(nkin+1) point grid")
    end

    kin_spline = Spl.CubicSpline(psi_reg, kin_fs; bctype="extrap")

    kin_spline._fs[:, 1:2] .*= nfac
    kin_spline._fs[:, 3:4] .*= tfac
    Spl.spline_fit!(kin_spline, "extrap")

    welec = wefac .* kin_spline._fs[:, 5]
    warning_printed = false
    for i in eachindex(welec)
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

    wdian = -twopi .* kin_spline._fs[:, 3] .* kin_spline.fs1[:, 1] ./ (e * zi * chi1 .* kin_spline._fs[:, 1])
    wdiat = -twopi .* kin_spline.fs1[:, 3] ./ (e * zi * chi1)
    wpefac = (wpfac .* (welec .+ wdian .+ wdiat) .- (wdian .+ wdiat)) ./ welec
    welec .= wpefac .* welec
    kin_spline._fs[:, 5] .= welec

    if wpfac != 1.0 && verbose
        println("  -> manipulating rotation by factor of ", wpfac)
        println("     by indirect manipulation of omegae profile")
    end

    Spl.spline_fit!(kin_spline, "extrap")

    if write_log
        open("pentrc_kinetics.out", "w") do f
            write(f, "PENTRC KINETIC PROFILE OUTPUT\n")
            write(f, "psi_n\t\tn_i\t\tn_e\t\tT_i\t\tT_e\t\tomegaE\t\tloglam\t\tnu_i\t\tnu_e\t\tzeff\n")
            for i in 1:(nkin + 1)
                write(f, @sprintf("%.6e\t%.6e\t%.6e\t%.6e\t%.6e\t%.6e\t%.6e\t%.6e\t%.6e\t%.6e\n",
                    psi_reg[i], kin_spline._fs[i, 1], kin_spline._fs[i, 2], kin_spline._fs[i, 3], kin_spline._fs[i, 4],
                    kin_spline._fs[i, 5], kin_spline._fs[i, 6], kin_spline._fs[i, 7], kin_spline._fs[i, 8], kin_spline._fs[i, 9]))
            end
        end
        println("Wrote kinetic spline to pentrc_kinetics.out")
    end

    global kin = kin_spline

    return kin_spline
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
            idx = searchsortedlast(x, xval)
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

function _read_fortran_record(io::IO)
    n1 = Int(read(io, Int32))
    payload = read(io, n1)
    n2 = Int(read(io, Int32))
    n1 == n2 || error("Fortran unformatted record length mismatch: $n1 != $n2")
    return payload
end

function _read_fortran_scalars(io::IO, ::Type{T}, count::Integer=1) where {T}
    payload = _read_fortran_record(io)
    expected = count * sizeof(T)
    length(payload) == expected || error("Fortran record size mismatch for $(T): expected $expected bytes, got $(length(payload))")
    vals = reinterpret(T, payload)
    return count == 1 ? vals[1] : collect(vals)
end

function _read_fortran_array(io::IO, ::Type{T}, dims::Vararg{Int,N}) where {T,N}
    payload = _read_fortran_record(io)
    expected = prod(dims) * sizeof(T)
    length(payload) == expected || error("Fortran record size mismatch for $(T) array: expected $expected bytes, got $(length(payload))")
    vals = reinterpret(T, payload)
    return reshape(collect(vals), dims...)
end


"""
    load_peq!(peq_file::String; jac_in="default", jsurf_in=0, tmag_in=1,
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
function load_peq!(peq_file::String; jac_in="default", jsurf_in=0, tmag_in=1,
                   force_xialpha=false, debug=false, op_powin=nothing, verbose=false)
    
    if !isfile(peq_file)
        error("Perturbation equilibrium file not found: $peq_file")
    end
    
    # TODO: Full implementation of ASCII peq reading
    # This is complex - requires:
    # 1. Reading ASCII table with complex mode data
    # 2. Detecting psi-outer vs m-outer loop ordering
    # 3. Coordinate transformation to DCON jacobian
    # 4. Mode spectrum interpolation (newm function)
    # 5. Clebsch to flux surface decomposition
    
    if debug
        println("Reading perturbation equilibrium from: $peq_file")
    end
    
    # For now, delegate to load_gpec_peq! for binary format
    if endswith(peq_file, ".bin") || endswith(peq_file, ".dat")
        return load_gpec_peq!(peq_file; write_log=debug, verbose=verbose)
    end
    
    error("ASCII peq reading not yet implemented - use GPEC binary format (.bin)")
end

"""
    load_gpec_peq!(gpec_file::String; write_log=false, verbose=false)

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
function load_gpec_peq!(gpec_file::String; write_log=false, verbose=false)
    
    if !isfile(gpec_file)
        error("GPEC perturbation file not found: $gpec_file")
    end
    
    if verbose
        println("Reading GPEC perturbation equilibrium from: $gpec_file")
    end
    
    # For Julia, we'll support both binary and HDF5 formats
    if endswith(gpec_file, ".h5") || endswith(gpec_file, ".hdf5")
        h5open(gpec_file, "r") do fid
            ms = read(fid, "ms")
            mp = read(fid, "mp")
            lagbpar = read(fid, "lagbpar")
            divxprp = read(fid, "divxprp")
            data = (; ms=Int(ms), mp=Int(mp), lagbpar=ComplexF64.(lagbpar), divxprp=ComplexF64.(divxprp))
            _set_dbdx_from_modes(data.lagbpar, data.divxprp; verbose=verbose)
            return data
        end
    else
        # Fortran unformatted sequential records.
        open(gpec_file, "r") do io
            dims = _read_fortran_scalars(io, Int32, 2)
            ms, mp = Int.(dims)

            if verbose
                println("  -> GPEC file has $ms steps in psi, $mp modes")
            end

            lagbpar = _read_fortran_array(io, ComplexF64, ms, mp)
            divxprp = _read_fortran_array(io, ComplexF64, ms, mp)
            data = (; ms=Int(ms), mp=Int(mp), lagbpar=lagbpar, divxprp=divxprp)
            _set_dbdx_from_modes(data.lagbpar, data.divxprp; verbose=verbose)
            return data
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
- `xmsmns::Matrix`: xi^alpha component (psi, m) [in DCON units/chi1]
- `set_dbdx::Bool`: Calculate and set dB/B and div splines (default=true)
- `debug::Bool`: Print intermediate messages

# Side effects
Updates global xs_m, dbob_m, divx_m splines for all perturbation components.

# Notes
Requires DCON matrix structure access for metric calculations.
"""
function set_peq(psi::Vector, ms::Vector, xmp1mns::Matrix, xspmns::Matrix, xmsmns::Matrix;
                 set_dbdx=true, debug=false)
    if debug
        println("Setting perturbation equilibrium")
        println("  psi range: $(psi[1]) to $(psi[end])")
        println("  input modes: $ms")
    end
    
    npsi = length(psi)
    target_modes = active_mode_numbers(length(ms); mode_numbers=ms)
    nm = length(target_modes)
    
    if debug
        println("  $(npsi) flux surfaces, $(nm) modes")
    end
    
    aligned_x1 = _align_modes(ComplexF64.(xmp1mns), Int.(ms), target_modes)
    aligned_x2 = _align_modes(ComplexF64.(xspmns), Int.(ms), target_modes)
    aligned_x3 = _align_modes(ComplexF64.(xmsmns), Int.(ms), target_modes)

    global xs_m = Any[
        Spl.CubicSpline(collect(psi), aligned_x1; bctype="extrap"),
        Spl.CubicSpline(collect(psi), aligned_x2; bctype="extrap"),
        Spl.CubicSpline(collect(psi), aligned_x3; bctype="extrap"),
    ]

    if set_dbdx && smats !== nothing && tmats !== nothing && xmats !== nothing && ymats !== nothing && zmats !== nothing
        @warn "set_peq stores xi splines but does not yet reconstruct dB/B and div(xi⊥) from DCON action matrices. Use load_pmodb!/load_gpec_peq! for tpsi-ready perturbations." maxlog=1
    end

    if dbob_m === nothing || divx_m === nothing
        zero_modes = zeros(ComplexF64, npsi, nm)
        global dbob_m = Spl.CubicSpline(collect(psi), zero_modes; bctype="extrap")
        global divx_m = Spl.CubicSpline(collect(psi), zero_modes; bctype="extrap")
    end

    return nothing
end


"""
    load_equilibrium!(equil)

Set up equilibrium from equilibrium structure.
Initializes all equilibrium splines and parameters needed for PENTRC.

# Arguments
- `equil`: Equilibrium structure (typically from DCON)
"""
function load_equilibrium!(equil; idconfile::Union{Nothing,String}=nothing)
    global sq = equil.sq
    global rzphi = equil.rzphi
    global eqfun = equil.eqfun
    global ro = equil.ro
    global zo = equil.zo
    global psio = equil.psio
    global chi1 = twopi * equil.psio
    global bo = something(equil.params.b0, abs(Spl.spline_eval!(sq, 0.0)[1]) / max(twopi * ro, eps()))
    global jac_type = equil.config.control.jac_type
    global power_bp = equil.config.control.power_bp
    global power_b = equil.config.control.power_b
    global power_r = equil.config.control.power_r
    global shotnum = 0.0
    global shottime = 0.0
    global machine = "julia"
    mthsurf_idcon = nothing
    if idconfile !== nothing
        try
            hdr = _read_idcon_header(idconfile)
            mthsurf_idcon = hdr.mthvac
        catch err
            @warn "Failed to read mthsurf from idconfile, falling back to equilibrium theta grid" idconfile exception=(err, catch_backtrace())
        end
    end
    global mthsurf = something(mthsurf_idcon, length(rzphi.ys) - 1)
    global theta = collect(range(0.0, 1.0; length=mthsurf + 1))
    global psifac = collect(rzphi.xs)
    global geom = _build_geom_spline(equil)

    return nothing
end

function _surface_rz(psi::Float64, θgrid::AbstractVector{<:Real})
    rvals = zeros(Float64, length(θgrid))
    zvals = zeros(Float64, length(θgrid))
    for (i, θ) in enumerate(θgrid)
        rzphi_f = Spl.bicube_eval!(rzphi, psi, Float64(θ))
        rsq = max(rzphi_f[1], 0.0)
        rfac = sqrt(rsq)
        eta = twopi * (Float64(θ) + rzphi_f[2])
        rvals[i] = ro + rfac * cos(eta)
        zvals[i] = zo + rfac * sin(eta)
    end
    return rvals, zvals
end

function _polygon_area(rvals::AbstractVector{<:Real}, zvals::AbstractVector{<:Real})
    n = min(length(rvals), length(zvals))
    n < 3 && return 0.0
    area2 = 0.0
    for i in 1:(n - 1)
        area2 += rvals[i] * zvals[i + 1] - rvals[i + 1] * zvals[i]
    end
    return 0.5 * abs(area2)
end

function _build_geom_spline(equil)
    ψgrid = collect(sq.xs)
    θgrid = collect(rzphi.ys)
    geom_data = zeros(length(ψgrid), 3)

    for (i, ψ) in enumerate(ψgrid)
        rvals, zvals = _surface_rz(ψ, θgrid)
        rmin = minimum(rvals)
        rmax = maximum(rvals)
        rmajor = max(0.5 * (rmax + rmin), eps())
        rminor = max(0.5 * (rmax - rmin), eps())
        area = _polygon_area(rvals, zvals)

        geom_data[i, 1] = area
        geom_data[i, 2] = rminor
        geom_data[i, 3] = rmajor
    end

    geom_data[1, 1] = max(geom_data[1, 1], π * geom_data[1, 2]^2)
    return Spl.CubicSpline(ψgrid, geom_data; bctype="extrap")
end

"""
    load_fnml!(fnml_file::String; verbose=false)

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
function load_fnml!(fnml_file::String; verbose=false)
    
    if !isfile(fnml_file)
        error("Fourier mode coupling file not found: $fnml_file")
    end
    
    if verbose
        println("Reading Fourier mode coupling data from: $fnml_file")
    end
    
    # Support both binary and HDF5 formats
    if endswith(fnml_file, ".h5") || endswith(fnml_file, ".hdf5")
        h5open(fnml_file, "r") do fid
            nfk = read(fid, "nfk")
            nft = read(fid, "nft")
            xs = read(fid, "xs")
            ys = read(fid, "ys")
            fs = read(fid, "fs")
            fs3 = ndims(fs) == 2 ? reshape(Float64.(fs), length(xs), length(ys), 1) : Float64.(fs)
            fnml_spline = Spl.BicubicSpline(Float64.(xs), Float64.(ys), fs3; bctypex="extrap", bctypey="extrap")
            global fnml = fnml_spline
            return fnml_spline
        end
    else
        # Fortran unformatted sequential records.
        open(fnml_file, "r") do io
            dims = _read_fortran_scalars(io, Int32, 2)
            nfk, nft = Int.(dims)

            if verbose
                println("  -> fnml grid: $nfk x $nft")
            end

            xs = _read_fortran_scalars(io, Float64, nfk + 1)
            ys = _read_fortran_scalars(io, Float64, nft + 1)
            fs2 = _read_fortran_array(io, Float64, nfk + 1, nft + 1)
            fs = reshape(fs2, nfk + 1, nft + 1, 1)

            fnml_spline = Spl.BicubicSpline(xs, ys, fs; bctypex="extrap", bctypey="extrap")
            global fnml = fnml_spline
            return fnml_spline
        end
    end
end

# ============================================================================
# Perturbation equilibrium reading and setup
# ============================================================================

"""
    load_pmodb!(pmodb_file::String; jac_in="default", jsurf_in=0, tmag_in=1,
                debug=false, op_powin=nothing, verbose=false)

Read psi-m matrix of Lagrangian dB/B and div(xi_perp) from file.

Converts between different jacobian coordinate systems (PEST, Boozer, etc.)
and interpolates to DCON mode spectrum (mfac).

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
function load_pmodb!(pmodb_file::String; jac_in="default", jsurf_in=0, tmag_in=1,
                     debug=false, op_powin=nothing, verbose=false)
    
    if !isfile(pmodb_file)
        error("Perturbation matrix file not found: $pmodb_file")
    end
    
    if verbose
        println("Reading perturbation equilibrium matrix from: $pmodb_file")
    end
    
    table, _ = readtable(pmodb_file; verbose=verbose, debug=debug)
    ncol = size(table, 2)
    ncol in (8, 10) || error("Unsupported pmodb table width $(ncol); expected 8 or 10 columns")
    
    if debug
        println("  -> Read $(size(table,1)) rows x $(size(table,2)) columns")
    end
    
    nm = nunique(table[:, 2])
    npsi = size(table, 1) ÷ nm
    npsi * nm == size(table, 1) || error("pmodb table is not a rectangular (psi,m) grid")

    psi_outer = nunique(table[1:nm, 2]) == nm
    ms = psi_outer ? round.(Int, table[1:nm, 2]) : round.(Int, table[1:npsi:end, 2])
    psi = psi_outer ? Float64.(table[1:nm:end, 1]) : Float64.(table[1:npsi, 1])

    lagbvec = ComplexF64.(table[:, 5], table[:, 6])
    divvec =
        if ncol >= 10
            kapxvec = ComplexF64.(table[:, 9], table[:, 10])
            -(lagbvec .+ kapxvec)
        else
            ComplexF64.(table[:, 7], table[:, 8])
        end

    lagbmat = psi_outer ? reshape(lagbvec, nm, npsi)' : reshape(lagbvec, npsi, nm)
    divmat = psi_outer ? reshape(divvec, nm, npsi)' : reshape(divvec, npsi, nm)

    target_modes = active_mode_numbers(length(ms); mode_numbers=ms)
    lagb_aligned = _align_modes(lagbmat, ms, target_modes)
    div_aligned = _align_modes(divmat, ms, target_modes)

    jac_spec = _resolve_input_jacobian(jac_in, op_powin)
    if jac_spec.name != jac_type || tmag_in != 1
        for i in axes(lagb_aligned, 1)
            _idcon_coords!(
                view(lagb_aligned, i, :),
                target_modes,
                psi[i],
                jac_spec.power_r,
                jac_spec.power_bp,
                jac_spec.power_b,
                jac_spec.power_rc,
                tmag_in,
                jsurf_in,
            )
            _idcon_coords!(
                view(div_aligned, i, :),
                target_modes,
                psi[i],
                jac_spec.power_r,
                jac_spec.power_bp,
                jac_spec.power_b,
                jac_spec.power_rc,
                tmag_in,
                jsurf_in,
            )
        end
    end

    db_weighted = similar(lagb_aligned)
    div_weighted = similar(div_aligned)
    for i in axes(lagb_aligned, 1)
        lagbfun = iscdftb(target_modes, vec(lagb_aligned[i, :]), mthsurf)
        divxfun = iscdftb(target_modes, vec(div_aligned[i, :]), mthsurf)

        for j in eachindex(lagbfun)
            bmag = Spl.bicube_eval!(eqfun, psi[i], theta[j])[1]
            lagbfun[j] /= bmag
            divxfun[j] /= bmag
        end

        db_weighted[i, :] .= iscdftf(target_modes, lagbfun)
        div_weighted[i, :] .= iscdftf(target_modes, divxfun)
    end

    global dbob_m = Spl.CubicSpline(collect(psi), db_weighted; bctype="extrap")
    global divx_m = Spl.CubicSpline(collect(psi), div_weighted; bctype="extrap")

    return (; psi=collect(psi), ms=target_modes, dbob=db_weighted, divx=div_weighted)
end

function _resolve_input_jacobian(jac_in::AbstractString, op_powin)
    if isempty(jac_in) || jac_in == "default"
        return (; name=jac_type, power_b=power_b, power_bp=power_bp, power_r=power_r, power_rc=0)
    elseif jac_in == "hamada"
        return (; name="hamada", power_b=0, power_bp=0, power_r=0, power_rc=0)
    elseif jac_in == "pest"
        return (; name="pest", power_b=0, power_bp=0, power_r=2, power_rc=0)
    elseif jac_in == "equal_arc"
        return (; name="equal_arc", power_b=0, power_bp=1, power_r=0, power_rc=0)
    elseif jac_in == "boozer"
        return (; name="boozer", power_b=2, power_bp=0, power_r=0, power_rc=0)
    elseif jac_in == "park"
        return (; name="park", power_b=1, power_bp=0, power_r=0, power_rc=0)
    elseif jac_in == "polar"
        return (; name="polar", power_b=0, power_bp=1, power_r=0, power_rc=1)
    elseif jac_in == "other"
        op_powin === nothing && error("`jac_in=\"other\"` requires `op_powin = [B, Bp, R, Rc]`.")
        length(op_powin) == 4 || error("`op_powin` must have four entries: [B, Bp, R, Rc].")
        return (; name="other", power_b=Int(op_powin[1]), power_bp=Int(op_powin[2]), power_r=Int(op_powin[3]), power_rc=Int(op_powin[4]))
    end
    error("Unsupported jac_in=$(repr(jac_in)).")
end

function _inverse_monotone(xgrid::AbstractVector{<:Real}, ygrid::AbstractVector{<:Real}, y::Float64)
    y <= ygrid[1] && return Float64(xgrid[1])
    y >= ygrid[end] && return Float64(xgrid[end])
    idx = searchsortedfirst(ygrid, y)
    idx = clamp(idx, 2, length(ygrid))
    y0 = Float64(ygrid[idx - 1])
    y1 = Float64(ygrid[idx])
    x0 = Float64(xgrid[idx - 1])
    x1 = Float64(xgrid[idx])
    frac = abs(y1 - y0) <= eps() ? 0.0 : (y - y0) / (y1 - y0)
    return x0 + frac * (x1 - x0)
end

function _idcon_coords!(ftnmn::AbstractVector{ComplexF64}, amf::AbstractVector{<:Integer}, psi::Float64,
                        ri::Int, bpi::Int, bi::Int, rci::Int, ti::Int, ji::Int)
    θgrid = theta
    nθ = length(θgrid)
    dphi = zeros(Float64, nθ)
    thetas = zeros(Float64, nθ)
    jacfac = ones(Float64, nθ)
    spl = Spl.CubicSpline(collect(θgrid), zeros(Float64, nθ, 2); bctype="periodic")
    sq_f = Spl.spline_eval!(sq, psi)
    q = sq_f[4]

    for i in eachindex(θgrid)
        θ = Float64(θgrid[i])
        rzphi_f, _, rzphi_fy = Spl.bicube_deriv1!(rzphi, psi, θ)
        rsq = max(rzphi_f[1], eps())
        rfac = sqrt(rsq)
        eta = twopi * (θ + rzphi_f[2])
        r = ro + rfac * cos(eta)
        jac = rzphi_f[4]
        w11 = (1 + rzphi_fy[2]) * twopi^2 * rfac * r / jac
        w12 = -rzphi_fy[1] * π * r / (rfac * jac)
        delpsi = sqrt(w11^2 + w12^2)
        bp = psio * delpsi / r
        bt = sq_f[1] / (twopi * r)
        b = sqrt(bp^2 + bt^2)
        fac = r^power_r / (bp^power_bp * b^power_b)
        spl.fs[i, 1] = fac / (r^ri * rfac^rci) * bp^bpi * b^bi
        spl.fs[i, 2] = delpsi * r^ri * rfac^rci / (bp^bpi * b^bi)
        if ti == 0
            dphi[i] = rzphi_f[3]
        end
    end

    Spl.spline_fit!(spl, "periodic")
    Spl.spline_integrate!(spl)
    total = spl.fsi[end, 1]
    thetas .= spl.fsi[:, 1] ./ total

    if ji == 1
        for i in eachindex(θgrid)
            θi = _inverse_monotone(θgrid, thetas, Float64(θgrid[i]))
            jacfac[i] = Spl.spline_eval!(spl, θi)[2]
        end
        jarea = sum(@view jacfac[1:(end - 1)]) / max(length(jacfac) - 1, 1)
        ftnfun = iscdftb(Int.(amf), collect(ftnmn), mthsurf)
        ftnfun ./= ComplexF64.(jacfac)
        ftnfun .*= jarea
        ftnmn .= iscdftf(Int.(amf), ftnfun)
    end

    ftnfun = zeros(ComplexF64, nθ)
    for i in eachindex(θgrid)
        θmap = thetas[i]
        acc = 0.0 + 0.0im
        for j in eachindex(amf)
            acc += ftnmn[j] * exp(iunit * twopi * amf[j] * θmap)
        end
        ftnfun[i] = acc
    end

    if ti == 0
        ftnfun .*= exp.(-iunit * nn .* dphi)
    else
        ftnfun .*= exp.(-twopi * iunit * nn * q .* (thetas .- θgrid))
    end

    ftnmn .= iscdftf(Int.(amf), ftnfun)
    return nothing
end


function _newm(source_modes::AbstractVector{<:Integer}, source_vals::AbstractVector{ComplexF64},
               target_modes::AbstractVector{<:Integer})
    len_i = length(source_modes)
    len_o = length(target_modes)
    out = zeros(ComplexF64, len_o)
    len_i == 0 && return out
    len_o == 0 && return out

    i = max(abs(Int(source_modes[1])) - abs(Int(target_modes[1])) + 1, 1)
    ii = len_i - max(abs(Int(source_modes[end])) - abs(Int(target_modes[end])), 0)
    j = max(abs(Int(target_modes[1])) - abs(Int(source_modes[1])) + 1, 1)
    jj = len_o - max(abs(Int(target_modes[end])) - abs(Int(source_modes[end])), 0)

    if !(Int(target_modes[j]) == Int(source_modes[i]) && Int(target_modes[jj]) == Int(source_modes[ii]))
        error("Misalignment in m-spaces: source=$(source_modes[1]):$(source_modes[end]), target=$(target_modes[1]):$(target_modes[end])")
    end

    out[j:jj] .= source_vals[i:ii]
    return out
end

function _align_modes(data::AbstractMatrix{ComplexF64}, source_modes::AbstractVector{<:Integer},
                      target_modes::AbstractVector{<:Integer})
    aligned = zeros(ComplexF64, size(data, 1), length(target_modes))
    for i in axes(data, 1)
        aligned[i, :] .= _newm(source_modes, vec(data[i, :]), target_modes)
    end
    return aligned
end

function _set_dbdx_from_modes(lagbpar::AbstractMatrix{ComplexF64}, divxprp::AbstractMatrix{ComplexF64};
                              verbose::Bool=false, mode_numbers=nothing)
    nm = size(lagbpar, 2)
    target_modes = active_mode_numbers(nm; mode_numbers=mode_numbers)

    psi_grid =
        if !isempty(psifac) && length(psifac) == size(lagbpar, 1)
            collect(psifac)
        elseif sq !== nothing
            collect(range(first(sq.xs), last(sq.xs), length=size(lagbpar, 1)))
        else
            collect(range(0.0, 1.0, length=size(lagbpar, 1)))
        end

    global dbob_m = Spl.CubicSpline(psi_grid, ComplexF64.(lagbpar); bctype="extrap")
    global divx_m = Spl.CubicSpline(psi_grid, ComplexF64.(divxprp); bctype="extrap")

    if verbose
        println("  -> Loaded perturbation mode splines on $(length(psi_grid)) psi points and $(length(target_modes)) modes")
    end

    return nothing
end
