# PERTURBED EQUILIBRIUM NONAMBIPOLAR TRANSPORT CODE

"""
energy_integration module

DESCRIPTION:
    Functional form of the energy integrand as found in
    [Logan, Park, et al., Phys. Plasma, 2013] and various integration
    methods.

PUBLIC MEMBER FUNCTIONS:
    xintgnd             - Resonance operator Eq. (20)
    xintgrl_lsode       - Dynamic energy integration
    xintrgl_spline      - Spline integration

PUBLIC DATA MEMBERS:
    xatol               - absolute tolerance of lsode
    xrtol               - relative tolerance of lsode

REVISION HISTORY:
    2014.03.06 -Logan- initial writing.
    2025.11.30 -Fairborn- Converted to Julia 
    Converted using Claude AI for first pass

AUTHOR: Logan
EMAIL: nikolas.logan@columbia.edu
"""

include("params.jl")  # For r8, mp, me, e, npsi_out, nell_out, nlambda_out, nmethods, methods, version
include("utilities.jl")  # For get_free_file_unit, append_2d, check
include("dcon_interface.jl")  # For shotnum, shottime, machine

try
    using NCDatasets  # NetCDF library
catch
    @warn "NCDatasets not available; NetCDF functionality disabled."
end
using DifferentialEquations  # Julia's ODE solver (replaces LSODE)
using Printf

# Global variables with defaults
xatol = 1e-12
xrtol = 1e-9
xmax = 72.0
ximag = 0.00
xnufac = 1.00
xnutype = "harmonic"
xf0type = "maxwellian"
qt = false
xdebug = false

# Module variables for internal use
const maxstep = 10000
energy_imaxis = false

# Thread-local variables (using task-local storage in Julia)
const energy_state = TaskLocalValue{Dict{Symbol,Any}}(() -> Dict(
    :energy_n => 0,
    :energy_wn => 0.0,
    :energy_wt => 0.0,
    :energy_we => 0.0,
    :energy_wd => 0.0,
    :energy_wb => 0.0,
    :energy_nuk => 0.0,
    :energy_leff => 0.0
))

# Helper functions to access thread-local state
get_energy_var(key) = energy_state[][key]
set_energy_var!(key, val) = (energy_state[][key] = val)

# Record structure
mutable struct Record
    is_recorded::Bool
    psi_index::Int
    lambda_index::Int
    ell_index::Int
    ell::Vector{Int}
    psi::Vector{Float64}
    lambda::Array{Float64,3}
    leff::Array{Float64,3}
    fs::Array{Float64,5}
end

# Constructor for empty record
function Record()
    Record(false, 0, 0, 0, Int[], Float64[], 
           Array{Float64}(undef, 0, 0, 0),
           Array{Float64}(undef, 0, 0, 0),
           Array{Float64}(undef, 0, 0, 0, 0, 0))
end

# Global array of records for each method
energy_record = [Record() for _ in 1:nmethods]

"""
    xintgrl_lsode(wn, wt, we, wd, wb, nuk, ell, leff, n, psi, lambda, method; op_record=false)

Dynamic energy integration using ODE solver. Module global variables
xatol, xrtol are used for tolerances and xmax, ximag are used
to define integration path (0->i*ximag, i*ximag->xmax+i*ximag).

Module global variable xnutype is used to determine collision
operator. Default is harmonic, and valid choices are:
    "zero" : collisionless
    "small" : 1e-6*(we+wd) (krook value ignored)
    "krook" : krook operator (unmodified argument)
    "harmonic" : (1+(l/2)^2)*krook*x^-3/2

Module's global variable xf0type is used to determine
denominator. Default is "maxwellian" and valid options are:
    "maxwellian" : x^(5/2)*exp(-x)

Pro-tip: To calculate offset rotation use we=-wn-wt.

# Arguments
- `wn`: density gradient diamagnetic drift frequency
- `wt`: temperature gradient diamagnetic drift frequency
- `we`: electric precession frequency
- `wd`: magnetic precession frequency
- `wb`: bounce frequency divided by x (x=E/T)
- `nuk`: effective Krook collision frequency (i.e. /2eps for trapped)
- `ell`: Bounce harmonic
- `leff`: effective bounce harmonic ell-sigma*n*q where sigma=0(1) for trapped(passing)
- `n`: toroidal mode number
- `psi`: Flux surface (for record keeping only)
- `lambda`: Normalized pitch angle muB0/E (for record keeping only)
- `method`: Integration method name
- `op_record`: Store integration energy-space profile in memory

# Returns
- `ComplexF64`: energy integral
"""
function xintgrl_lsode(wn::Float64, wt::Float64, we::Float64, wd::Float64, 
                       wb::Float64, nuk::Float64, ell::Int, leff::Float64, 
                       n::Int, psi::Float64, lambda::Float64, method::String; 
                       op_record::Bool=false)
    
    # Initialize variables
    xprofile = zeros(6, maxstep)
    record_this = op_record
    
    # Set module-level variables for access in integrand
    set_energy_var!(:energy_wn, wn)
    set_energy_var!(:energy_wt, wt)
    set_energy_var!(:energy_we, we)
    set_energy_var!(:energy_wd, wd)
    set_energy_var!(:energy_wb, wb)
    set_energy_var!(:energy_nuk, nuk)
    set_energy_var!(:energy_leff, leff)
    set_energy_var!(:energy_n, n)
    
    y = [0.0, 0.0]   # [real, imag] parts of integral
    yi = [0.0, 0.0]  # Integration along imaginary axis
    i = 0  # Step counter
    
    # Optionally step off real axis to avoid poles
    if ximag != 0.0
        global energy_imaxis = true
        x_start = 1e-15
        x_end = ximag
        
        if record_this
            # Single step integration with recording
            prob = ODEProblem(xintgrnd!, yi, (x_start, x_end))
            sol = solve(prob, Tsit5(); 
                       abstol=xatol, reltol=xrtol, 
                       saveat=range(x_start, x_end, length=min(1000, maxstep)),
                       maxiters=maxstep)
            
            for (idx, t) in enumerate(sol.t)
                if i >= maxstep
                    break
                end
                i += 1
                integrand = zeros(2)
                xintgrnd!(integrand, sol.u[idx], nothing, t)
                xprofile[:, i] = [0.0, t, integrand[1], integrand[2], 
                                  sol.u[idx][1], sol.u[idx][2]]
            end
            yi = sol.u[end]
        else
            # Full integration
            prob = ODEProblem(xintgrnd!, yi, (x_start, x_end))
            sol = solve(prob, Tsit5(); 
                       abstol=xatol, reltol=xrtol, maxiters=maxstep)
            yi = sol.u[end]
            
            if !sol.retcode == :Success
                @warn "Integration along imaginary axis may have issues"
            end
        end
    end
    
    # Integration in real space to xmax
    global energy_imaxis = false
    x_start = 1e-15
    x_end = xmax
    
    if record_this
        # Single step integration with recording
        prob = ODEProblem(xintgrnd!, y, (x_start, x_end))
        sol = solve(prob, Tsit5(); 
                   abstol=xatol, reltol=xrtol,
                   saveat=range(x_start, x_end, length=min(1000, maxstep-i)),
                   maxiters=maxstep-i)
        
        for (idx, t) in enumerate(sol.t)
            if i >= maxstep
                break
            end
            i += 1
            integrand = zeros(2)
            xintgrnd!(integrand, sol.u[idx], nothing, t)
            xprofile[:, i] = [t, 0.0, integrand[1], integrand[2],
                              sol.u[idx][1], sol.u[idx][2]]
        end
        y = sol.u[end]
    else
        # Full integration
        prob = ODEProblem(xintgrnd!, y, (x_start, x_end))
        sol = solve(prob, Tsit5(); 
                   abstol=xatol, reltol=xrtol, maxiters=maxstep)
        y = sol.u[end]
        
        if sol.retcode != :Success
            println("psi = $psi, lambda = $lambda, leff = $leff")
            
            # Write error file
            open("pentrc_xintrgl_lsode.err", "w") do f
                println(f, "psi = $psi lambda = $lambda leff = $leff")
                println(f, "x                T_phi            2ndeltaW         int(T_phi)       int(2ndeltaW)")
                
                # Try integration with output
                prob_err = ODEProblem(xintgrnd!, [0.0, 0.0], (1e-15, xmax))
                sol_err = solve(prob_err, Tsit5(); 
                               abstol=xatol, reltol=xrtol,
                               saveat=0.1, maxiters=maxstep)
                
                for (idx, t) in enumerate(sol_err.t)
                    integrand = zeros(2)
                    xintgrnd!(integrand, sol_err.u[idx], nothing, t)
                    @printf(f, " %16.8e %16.8e %16.8e %16.8e %16.8e\n",
                           t, integrand[1], integrand[2], 
                           sol_err.u[idx][1], sol_err.u[idx][2])
                end
            end
            
            error("ERROR: xintgrl_lsode - Integration failed. Consider complex contour (ximag>0).")
        end
    end
    
    # Save integration profile to memory
    if record_this
        # Fill in unused x's with endpoints
        for j in i+1:maxstep
            xprofile[1, j] = xprofile[1, i] + 1e-9
            xprofile[2, j] = 0.0
            xprofile[3:4, j] .= 0.0
            xprofile[5:6, j] .= xprofile[5:6, i]
        end
        record_method(method, psi, ell, leff, lambda, xprofile)
    end
    
    # Return complex integral
    return ComplexF64(y[1] + yi[1], y[2] + yi[2])
end

"""
    xintgrnd!(ydot, y, p, x)

Energy integrand (y) as a function of normalized energy (x) as
defined in Eq. (8) of [Logan, Park, et al., Phys. Plasma, 2013].

This is the ODE form for DifferentialEquations.jl

# Arguments
- `ydot`: Output derivative [real, imag]
- `y`: Current state [real integral, imag integral]
- `p`: Parameters (unused, for DifferentialEquations.jl compatibility)
- `x`: Normalized energy (E/T)
"""
function xintgrnd!(ydot, y, p, x)
    # Complex contour determined by global variable
    cx = energy_imaxis ? im * x : x + im * ximag
    
    # Get thread-local variables
    energy_wn = get_energy_var(:energy_wn)
    energy_wt = get_energy_var(:energy_wt)
    energy_we = get_energy_var(:energy_we)
    energy_wd = get_energy_var(:energy_wd)
    energy_wb = get_energy_var(:energy_wb)
    energy_nuk = get_energy_var(:energy_nuk)
    energy_leff = get_energy_var(:energy_leff)
    energy_n = get_energy_var(:energy_n)
    
    # Collisionality determined by global variable
    if xnutype == "zero"
        nux = 0.0
    elseif xnutype == "small"
        nux = 1e-5 * energy_we
    elseif xnutype == "krook"
        nux = energy_nuk
    elseif xnutype == "harmonic"
        if x == 0
            nux = floatmax(Float64)  # Avoid 0^-3 = NaN
        else
            nux = energy_nuk * (1 + 0.25 * energy_leff^2) * cx^(-1.5)
        end
    else
        error("ERROR: xintgrnd - nutype must be zero, small, krook, or harmonic")
    end
    nux = xnufac * nux
    
    # Zeroth order distribution behavior
    denom = im * (energy_leff * energy_wb * sqrt(cx) + 
                  energy_n * (energy_we + energy_wd * cx)) - nux
    
    if xf0type == "maxwellian"
        # Standard solution from [Logan, Park, et al., Phys. Plasma, 2013]
        fx = (energy_we + energy_wn + energy_wt * (cx - 1.5)) * 
             cx^2.5 * exp(-cx) / denom
    elseif xf0type == "jkp"
        # Jong-Kyu Park [Park,Boozer,Menard, PRL 2009] approx neoclassical offset
        fx = (energy_we + energy_wn + energy_wt * 2) * 
             cx^2.5 * exp(-cx) / denom
    elseif xf0type == "cgl"
        # Chew-Goldberger-Low limit (we+wd -> inf)
        fx = cx^2.5 * exp(-cx) / (im * energy_n) / 1.0
    else
        error("ERROR: xintgrnd - f0 type must be maxwellian, jkp, or cgl")
    end
    
    # Heat flux calculation
    if qt
        fx = (cx - 2.5) * fx
    end
    
    if false  # Debug output
        println("nutype = ", xnutype)
        println("f0type = ", xf0type)
        println("ximag  = ", ximag)
        println("imaxis = ", energy_imaxis)
        println("omegas = ", energy_wn, " ", energy_wt, " ", energy_we, " ", 
                energy_wd, " ", energy_wb, " ", energy_nuk)
        println("n,l    = ", energy_n, " ", energy_leff)
        println("x      = ", x)
        println("fx     = ", fx)
    end
    
    # Decouple two real space solutions
    ydot[1] = real(fx)
    ydot[2] = imag(fx)
    
    return nothing
end

"""
    record_method(method, psi, ell, leff, lambda, fs)

Save the energy integrand and integral profiles for this method,
at this surface and pitch.
"""
function record_method(method::String, psi::Float64, ell::Int, leff::Float64, 
                       lambda::Float64, fs::Matrix{Float64})
    debug = false
    
    if debug
        println("Recording method")
    end
    
    # Find the right record
    for m in 1:nmethods
        if method == methods[m]
            if debug
                println("  method ", method)
            end
            
            # Initialize the record if needed
            if !energy_record[m].is_recorded
                if debug
                    println("   - is not recorded")
                end
                energy_record[m].is_recorded = true
                energy_record[m].psi_index = 0
                energy_record[m].lambda_index = 0
                energy_record[m].ell_index = 0
                energy_record[m].psi = zeros(npsi_out)
                energy_record[m].ell = zeros(Int, nell_out)
                energy_record[m].leff = zeros(nlambda_out, nell_out, npsi_out)
                energy_record[m].lambda = zeros(nlambda_out, nell_out, npsi_out)
                energy_record[m].fs = zeros(6, maxstep, nlambda_out, nell_out, npsi_out)
            end
            
            # Bump indexes
            if debug
                println("   - psi,ell,leff,lambda = ", psi, " ", ell, " ", leff, " ", lambda)
            end
            
            k = energy_record[m].psi_index
            j = energy_record[m].ell_index
            i = energy_record[m].lambda_index
            
            if k == 0 || psi != energy_record[m].psi[k]
                k += 1
            end
            if j == 0 || ell != energy_record[m].ell[j]
                j = mod(j, nell_out) + 1
            end
            if i == 0 || lambda != energy_record[m].lambda[i, j, k]
                i = mod(i, nlambda_out) + 1
            end
            
            if debug
                println("   - new i,j,k = ", i, " ", j, " ", k)
            end
            
            # Force fail if buggy
            if k > npsi_out
                error("ERROR: Too many psi energy records for method $method")
            end
            
            # Fill in new profiles
            energy_record[m].fs[:, :, i, j, k] = fs
            energy_record[m].lambda[i, j, k] = lambda
            energy_record[m].leff[i, j, k] = leff
            energy_record[m].ell[j] = ell
            energy_record[m].psi[k] = psi
            energy_record[m].psi_index = k
            energy_record[m].ell_index = j
            energy_record[m].lambda_index = i
        end
    end
    
    if debug
        println("Done recording")
    end
end

"""
    record_reset()

Deallocate all energy records.
"""
function record_reset()
    debug = false
    
    if debug
        println("Energy record resetting...")
    end
    
    for m in 1:nmethods
        if energy_record[m].is_recorded
            global energy_record[m] = Record()
        end
    end
    
    if debug
        println("Energy record reset done")
    end
end

"""
    output_energy_netcdf(n; op_label="")

Write recorded energy space integrations to a netcdf file.

# Arguments
- `n::Int`: Toroidal mode number for filename
- `op_label::String`: Extra label inserted into filename (optional)
"""
function output_energy_netcdf(n::Int; op_label::String="")
    debug = false
    
    println("Writing energy record output to netcdf")
    
    # Optional labeling
    label = isempty(op_label) ? "" : "_" * strip(op_label)
    
    # Assume all methods on the same psi's and ell's
    npsi = 0
    nell = 0
    nlambda = 0
    psi_out = Float64[]
    ell_out = Int[]
    
    for m in 1:nmethods
        if energy_record[m].is_recorded
            if debug
                println("  Getting dims from ", methods[m])
            end
            npsi = energy_record[m].psi_index
            nell = energy_record[m].ell_index
            nlambda = energy_record[m].lambda_index
            psi_out = energy_record[m].psi[1:npsi]
            ell_out = energy_record[m].ell[1:nell]
            break
        end
    end
    
    # Do nothing if no records
    if npsi == 0
        println("  WARNING: No energy records to output to netcdf")
        return
    end
    
    # Create and open netcdf file
    ncfile = "pentrc_energy_output$(label)_n$(n).nc"
    if debug
        println("  opening ", ncfile)
    end
    
    NCDataset(ncfile, "c") do ds
        # Store attributes
        if debug
            println("  storing attributes")
        end
        ds.attrib["title"] = "PENTRC energy space outputs"
        ds.attrib["version"] = version
        ds.attrib["shot"] = Int(shotnum)
        ds.attrib["time"] = Int(shottime)
        ds.attrib["machine"] = machine
        ds.attrib["n"] = n
        
        # Define dimensions
        if debug
            println("  defining dimensions")
            println("  npsi,nell,nlambda,nx = ", npsi, " ", nell, " ", nlambda, " ", maxstep)
        end
        
        defDim(ds, "i", 2)
        defDim(ds, "ell", nell)
        defDim(ds, "psi_n", npsi)
        defDim(ds, "Lambda_index", nlambda)
        defDim(ds, "x_index", maxstep)
        
        # Define and store dimension variables
        if debug
            println("  storing dimensions")
        end
        defVar(ds, "i", Int, ("i",))
        ds["i"][:] = [0, 1]
        
        defVar(ds, "ell", Int, ("ell",))
        ds["ell"][:] = ell_out
        
        defVar(ds, "psi_n", Float64, ("psi_n",))
        ds["psi_n"][:] = psi_out
        
        defVar(ds, "Lambda_index", Float64, ("Lambda_index",))
        ds["Lambda_index"][:] = collect(1:nlambda)
        
        defVar(ds, "x_index", Float64, ("x_index",))
        ds["x_index"][:] = collect(1:maxstep)
        
        # Store each method
        for m in 1:nmethods
            if energy_record[m].is_recorded
                if debug
                    println("  defining ", methods[m])
                end
                
                suffix = "_" * methods[m]
                
                # Check sizes
                if npsi != energy_record[m].psi_index
                    println(npsi, " ", energy_record[m].psi_index)
                    error("Error: Record sizes are inconsistent")
                end
                
                # Define variables
                defVar(ds, "Lambda" * suffix, Float64, 
                       ("Lambda_index", "ell", "psi_n"))
                defVar(ds, "ell_eff" * suffix, Float64, 
                       ("Lambda_index", "ell", "psi_n"))
                defVar(ds, "x" * suffix, Float64, 
                       ("i", "x_index", "Lambda_index", "ell", "psi_n"))
                defVar(ds, "T_psi_Lambda_x" * suffix, Float64, 
                       ("i", "x_index", "Lambda_index", "ell", "psi_n"))
                defVar(ds, "T_psi_Lambda" * suffix, Float64, 
                       ("i", "x_index", "Lambda_index", "ell", "psi_n"))
                
                # Put in variables
                if debug
                    println("  storing ", methods[m])
                end
                ds["Lambda" * suffix][:, :, :] = 
                    energy_record[m].lambda[1:nlambda, 1:nell, 1:npsi]
                ds["ell_eff" * suffix][:, :, :] = 
                    energy_record[m].leff[1:nlambda, 1:nell, 1:npsi]
                ds["x" * suffix][:, :, :, :, :] = 
                    energy_record[m].fs[1:2, :, 1:nlambda, 1:nell, 1:npsi]
                ds["T_psi_Lambda_x" * suffix][:, :, :, :, :] = 
                    energy_record[m].fs[3:4, :, 1:nlambda, 1:nell, 1:npsi]
                ds["T_psi_Lambda" * suffix][:, :, :, :, :] = 
                    energy_record[m].fs[5:6, :, 1:nlambda, 1:nell, 1:npsi]
            end
        end
    end
    
    # Clear the memory
    record_reset()
    
    if debug
        println("Finished energy netcdf output")
    end
end

"""
    output_energy_ascii(n; op_label="")

Write ascii energy output files.

# Arguments
- `n::Int`: Mode number
- `op_label::String`: Optional addition to the file name
"""
function output_energy_ascii(n::Int; op_label::String="")
    # Optional labeling
    label = isempty(op_label) ? "" : "_" * strip(op_label)
    
    # Open and prepare file
    file = "pentrc_energy_output$(label)_n$(n).out"
    open(file, "w") do f
        # Write header material
        println(f, "PERTURBED EQUILIBRIUM NONAMBIPOLAR TRANSPORT CODE:")
        println(f, " Energy integrand")
        println(f, " - variables are:   lambda =  B0*m*v_perp^2/(2B),  x = E/T")
        println(f, " - normalization is: 1/(-2n^2tn*chi'/sqrt(pi))")
        println(f)
        @printf(f, "%-10s%4d\n", "n =", n)
        
        # Write each method in a new table
        for m in 1:nmethods
            if energy_record[m].is_recorded
                println(f)
                @printf(f, "%-17s%s\n", "method =", methods[m])
                @printf(f, "%-17s%-17s%-17s%-17s%-17s%-17s%-17s%-17s%-17s%-17s%-17s%-17s\n",
                       "psi_n", "ell", "Lambda_index", "x_index",
                       "Lambda", "ell_eff", "real(x)", "imag(x)",
                       "T_phi", "2ndeltaW", "int(T_phi)", "int(2ndeltaW)")
                
                for k in 1:energy_record[m].psi_index
                    for j in 1:energy_record[m].ell_index
                        for i in 1:energy_record[m].lambda_index
                            for istep in 1:maxstep
                                @printf(f, "%17.8e%17.8e%17.8e%17.8e%17.8e%17.8e%17.8e%17.8e%17.8e%17.8e%17.8e%17.8e\n",
                                       energy_record[m].psi[k],
                                       Float64(energy_record[m].ell[j]),
                                       Float64(i), Float64(istep),
                                       energy_record[m].lambda[i, j, k],
                                       energy_record[m].leff[i, j, k],
                                       energy_record[m].fs[:, istep, i, j, k]...)
                            end
                        end
                    end
                end
            end
        end
    end
end