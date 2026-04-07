"""
    Input

Input reading and parsing for the KineticForces module.
"""

using TOML
using DelimitedFiles

"""
    get_method_flags(ctrl::KineticForcesControl)::Vector{Bool}

Extract method flags from control structure in correct order.
"""
function get_method_flags(ctrl::KineticForcesControl)::Vector{Bool}
    return [
        ctrl.fgar_flag, ctrl.tgar_flag, ctrl.pgar_flag, 
        ctrl.rlar_flag, ctrl.clar_flag, ctrl.fcgl_flag,
        ctrl.fwmm_flag, ctrl.twmm_flag, ctrl.pwmm_flag,
        ctrl.ftmm_flag, ctrl.ttmm_flag, ctrl.ptmm_flag,
        ctrl.fkmm_flag, ctrl.tkmm_flag, ctrl.pkmm_flag,
        ctrl.frmm_flag, ctrl.trmm_flag, ctrl.prmm_flag
    ]
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
    initialize_from_equilibrium!(intr::KineticForcesInternal, equil)

Populate KineticForcesInternal fields from a PlasmaEquilibrium.
Extracts geometry, profiles, and grid info needed for NTV calculations.

# Arguments
- `intr::KineticForcesInternal`: Internal state to populate
- `equil`: PlasmaEquilibrium from Equilibrium module
"""
function initialize_from_equilibrium!(intr::KineticForcesInternal, equil)
    intr.ro = equil.ro
    intr.bo = equil.params.bo
    intr.chi1 = 2π * equil.psio
    intr.mthsurf = length(equil.rzphi_ys) - 1

    # TODO: Construct profile interpolants (sq, kin, geom) from equil
    # TODO: Construct perturbation interpolants (dbob_m, divx_m) from ForceFreeStates output
    # TODO: Extract poloidal mode numbers (mfac) from perturbation data
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
