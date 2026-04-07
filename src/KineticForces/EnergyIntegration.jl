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

using OrdinaryDiffEq
using Printf

# Integration defaults
xatol = 1e-12
xrtol = 1e-9
xmax = 72.0
ximag = 0.00
xnufac = 1.00
xnutype = "harmonic"
xf0type = "maxwellian"
qt = false

# Module variables for internal use
const maxstep = 10000
energy_imaxis = false

# Module-level state for energy integrand (passed via closure in commit 8)
const energy_state = Dict{Symbol,Any}(
    :energy_n => 0,
    :energy_wn => 0.0,
    :energy_wt => 0.0,
    :energy_we => 0.0,
    :energy_wd => 0.0,
    :energy_wb => 0.0,
    :energy_nuk => 0.0,
    :energy_leff => 0.0
)

get_energy_var(key) = energy_state[key]
set_energy_var!(key, val) = (energy_state[key] = val)

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
                       n::Int, psi::Float64, lambda::Float64, method::String)

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
    
    # Optionally step off real axis to avoid poles
    if ximag != 0.0
        global energy_imaxis = true
        x_start = 1e-15
        x_end = ximag

        prob = ODEProblem(xintgrnd!, yi, (x_start, x_end))
        sol = solve(prob, Tsit5();
                    abstol=xatol, reltol=xrtol, maxiters=maxstep)
        yi = sol.u[end]

        if sol.retcode != :Success
            @warn "Integration along imaginary axis may have issues"
        end
    end

    # Integration in real space to xmax
    global energy_imaxis = false
    x_start = 1e-15
    x_end = xmax

    prob = ODEProblem(xintgrnd!, y, (x_start, x_end))
    sol = solve(prob, Tsit5();
                abstol=xatol, reltol=xrtol, maxiters=maxstep)
    y = sol.u[end]

    if sol.retcode != :Success
        @warn "xintgrl_lsode failed at psi=$psi, lambda=$lambda, leff=$leff. Consider complex contour (ximag>0)."
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

