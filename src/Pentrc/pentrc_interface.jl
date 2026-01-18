# PERTURBED EQUILIBRIUM NONAMBIPOLAR TRANSPORT CODE

"""
pentrc_interface module

DESCRIPTION:
    Top level interface for PENTRC.
    This was essentially the PENTRC program without any of the calculations,
    and was moved to an independent module in order to allow kinetic DCON
    to interface with and use the top-tier PENTRC functions / routines.

PUBLIC MEMBER SUBPROGRAMS:
    initialize_pentrc    - Read input namelists and distribute module variables
    get_pentrc          - Get pentrc configuration variables

PUBLIC DATA MEMBERS:
    ro, real                - Major radius (idcon)
    bo, real                - Field on axis (determined from sq)
    kin, spline_type        - (psi) Kinetic profiles
    xs_m, fspline_type      - (psi,theta) First order perturbations
    eqfun, bicube_type      - (psi,theta) Equilibrium mod b
    sq, spline_type         - (psi,theta) Equilibrium flux functions
    rzphi, bicube_type      - (psi,theta) Cylindrical coordinate & Jacobian

REVISION HISTORY:
    2014.03.06 -Logan- initial writing.
    2025.11.23 -Fairborn- Converted to Julia
    Converted using Claude AI for first pass

AUTHOR: Logan
EMAIL: nikolas.logan@columbia.edu
"""

include("params.jl")  # For r8, xj, npsi_out, nmethods, methods, docs, version
## Expose parameter constants for older code expecting lowercase globals
if isdefined(Main, :Params)
    const npsi_out = Params.Nψ_out
    const nmethods = Params.Nmethods
    const methods = Params.METHODS
    const docs = Params.DOCS
    const ngrids = Params.Ngrids
else
    const npsi_out = 30
    const nmethods = 18
    const methods = ["fgar"]
    const docs = ["Fallback docs"]
    const ngrids = 3
end
const version = "v3.00"
include("utilities.jl")  # For timer, to_upper, get_free_file_unit
include("special.jl")  # For set_fymnl, set_ellip
include("dcon_interface.jl")  # For set_eq, idcon_harvest
include("inputs.jl")  # For read_kin, read_equil, nn, read_peq, read_pmodb, set_peq, read_fnml, verbose
include("diagnostics.jl")  # For diagnose_all
include("spline_mod.jl")  # For use_classic_splines
include("energy_integration.jl")
include("pitch_integration.jl")
include("torque.jl")

# Import parameter constants from the Params module created by params.jl
if isdefined(Main, :Params)
    const npsi_out = Params.Nψ_out
    const nmethods = Params.Nmethods
    const methods = Params.METHODS
    const docs = Params.DOCS
else
    # Fallback defaults used for tests
    const npsi_out = 30
    const nmethods = 18
    const methods = ["fgar"]
    const docs = ["Fallback docs"]
end

# Version string (used in harvest payloads)
const version = "v3.00"

using Printf

# Module-level variables with defaults
# Logical flags
fgar_flag = true
tgar_flag = false
pgar_flag = false
rlar_flag = false
clar_flag = false
fcgl_flag = false
wxyz_flag = false
fkmm_flag = false
tkmm_flag = false
pkmm_flag = false
frmm_flag = false
trmm_flag = false
prmm_flag = false
fwmm_flag = false
twmm_flag = false
pwmm_flag = false
ftmm_flag = false
ttmm_flag = false
ptmm_flag = false
electron = false
eq_out = false
theta_out = false
xlmda_out = false
eqpsi_out = false
equil_grid = false
input_grid = false
dynamic_grid = true
fnml_flag = false
ellip_flag = false
diag_flag = false
term_flag = false
force_xialpha = false
clean = true
flags = fill(false, nmethods)
indebug = false

# Integer variables
mi = 2
zi = 1
zimp = 6
mimp = 12
nl = 0
tmag_in = 1
jsurf_in = 0
power_bin = -1
power_bpin = -1
power_rin = -1
power_rcin = -1
pentrc_threads = 0
openmp_threads = 0

# Real variables
atol_xlmda = 1e-6
rtol_xlmda = 1e-3
nfac = 1.0
tfac = 1.0
wefac = 1.0
wdfac = 1.0
wpfac = 1.0
nufac = 1.0
divxfac = 1.0
diag_psi = 0.7
psi_out = fill(-1.0, npsi_out)
psilims = [0.0, 1.0]

# Complex variables
tphi = ComplexF64(0, 0)
tsurf = ComplexF64(0, 0)
teq = ComplexF64(0, 0)
wtw = Array{ComplexF64, 3}(undef, 0, 0, 0)  # Will be allocated later

# String variables
nstring = ""
method = ""
idconfile = "euler.bin"
kinetic_file = "kin.dat"
gpec_file = "gpec_order1_n1.bin"
peq_file = ""
pmodb_file = "none"
data_dir = "."
nutype = "harmonic"
f0type = "maxwellian"
jac_in = "default"
moment = "pressure"

"""
    initialize_pentrc(; op_kin=true, op_deq=true, op_peq=true)

Read the pentrc namelists and distribute the necessary variables
to the appropriate module circles.
Optionally, read the other standard input files (otherwise they must be set)

# Arguments
- `op_kin::Bool`: Read ascii table of kinetic profiles (default true)
- `op_deq::Bool`: Read dcon euler.bin description of the equilibrium (default true)
- `op_peq::Bool`: Read ascii table of clebsch displacement profiles (default true)
"""
function initialize_pentrc(; op_kin=true, op_deq=true, op_peq=true)
    in_kin = op_kin
    in_deq = op_deq
    in_peq = op_peq
    
    # Read namelists from pentrc.in
    read_pentrc_input()
    
    # Defaults
    if all(psi_out .== -1)
        global psi_out = collect(1:30) ./ 30.6  # even spread if user doesn't pick any
    end
    
    # Warnings if using deprecated inputs
    if openmp_threads > 0
        @warn "openmp_threads is deprecated. Use pentrc_threads or JULIA_NUM_THREADS env variable."
    end
    if eq_out
        @warn "eq_out has been deprecated. Behavior is always true."
    end
    if eqpsi_out
        @warn "eqpsi_out has been deprecated. Use equil_grid."
    end
    if term_flag
        @warn "term_flag has been deprecated. Use verbose."
    end
    
    # Distribute some simplified inputs to module circles
    global xatol = atol_xlmda
    global xrtol = rtol_xlmda
    global xnufac = nufac
    global xnutype = nutype
    global xf0type = f0type
    global lambdaatol = atol_xlmda
    global lambdartol = rtol_xlmda
    
    # Read (perturbed) equilibrium inputs
    if indebug && in_kin
        println("  read_kin args: ", kinetic_file, " ", zi, " ", zimp, " ", mi, " ", mimp, " ", 
                nfac, " ", tfac, " ", wefac, " ", wpfac, " ", indebug)
    end
    
    if in_deq
        read_equil(idconfile)
    end
    
    if in_kin
        read_kin(kinetic_file, zi, zimp, mi, mimp, nfac, tfac, wefac, wpfac, indebug)
    end
    
    if in_peq
        if isempty(peq_file)
            nstr = @sprintf("%3d", nn)
            global peq_file = "gpec_xclebsch_n$(strip(nstr)).out"
        end
        
        read_peq(peq_file, jac_in, jsurf_in, tmag_in, force_xialpha, indebug;
                 op_powin=[power_bin, power_bpin, power_rin, power_rcin])
        
        if !isempty(pmodb_file) && pmodb_file != "none"
            read_pmodb(pmodb_file, jac_in, jsurf_in, tmag_in, indebug;
                      op_powin=[power_bin, power_bpin, power_rin, power_rcin])
        end
    end
end

"""
    get_pentrc()

Read the pentrc namelists and distribute the necessary variables 
to the appropriate module circles. Returns configuration parameters.

# Returns
- `get_nl::Int`: Toroidal mode number
- `get_zi::Int`: Ion charge number
- `get_mi::Int`: Ion mass number
- `get_wdfac::Float64`: Drift factor
- `get_divxfac::Float64`: Divergence factor
- `get_electron::Bool`: Electron flag
- `get_eq_out::Bool`: Equilibrium output flag
- `get_theta_out::Bool`: Theta output flag
- `get_xlmda_out::Bool`: Lambda output flag
"""
function get_pentrc()
    # Read interface and set modules
    println("Interfacing with PENTRC => v3.00")
    
    read_pentrc_input()
    
    # Distribute inputs to PENTRC module circles
    global xatol = atol_xlmda
    global xrtol = rtol_xlmda
    global xnufac = nufac
    global xnutype = nutype
    global xf0type = f0type
    global lambdaatol = atol_xlmda
    global lambdartol = rtol_xlmda
    
    # Set the kinetic spline
    if indebug
        println("read_kin args:")
        println("  ", kinetic_file, " ", zi, " ", zimp, " ", mi, " ", mimp, " ",
                nfac, " ", tfac, " ", wefac, " ", wpfac, " ", tdebug)
    end
    
    read_kin(kinetic_file, zi, zimp, mi, mimp, nfac, tfac, wefac, wpfac, indebug)
    
    # Return torque function variables to external program
    return nl, zi, mi, wdfac, divxfac, electron, eq_out, theta_out, xlmda_out
end

"""
    read_pentrc_input()

Internal function to read namelists from pentrc.in file.
This is a simplified version - in Julia, you might want to use a 
configuration file format like TOML, JSON, or YAML instead of Fortran namelists.
"""
function read_pentrc_input()
    # Julia doesn't have native namelist support
    # You would typically use a different configuration format
    # This is a placeholder that shows the structure
    
    # For now, this would need to be implemented based on your specific needs
    # Options include:
    # 1. Use TOML.jl and convert pentrc.in to TOML format
    # 2. Use ConfParser.jl for INI-style files
    # 3. Write a custom parser for Fortran namelists
    # 4. Use JSON or YAML for configuration
    
    if isfile("pentrc.in")
        @warn "Namelist reading not yet fully implemented. Using defaults."
        # TODO: Implement namelist parsing or use alternative config format
    else
        @warn "pentrc.in not found. Using default values."
    end
end