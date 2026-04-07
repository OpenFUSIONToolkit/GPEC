"""
    EnergyIntegration

Energy-space ODE integration for the kinetic resonance operator.
Implements the energy integrand from [Logan, Park, et al., Phys. Plasmas, 2013] Eq. (8).
"""

using OrdinaryDiffEq

# Integration defaults
const ENERGY_XMAX = 72.0
const ENERGY_MAXITERS = 10000

"""
    EnergyParams

Parameters passed to the energy ODE integrand via the ODE problem's `p` field.
"""
struct EnergyParams
    wn::Float64       # density gradient diamagnetic drift frequency
    wt::Float64       # temperature gradient diamagnetic drift frequency
    we::Float64       # electric precession frequency
    wd::Float64       # magnetic precession frequency
    wb::Float64       # bounce frequency / sqrt(x)
    nuk::Float64      # effective Krook collision frequency
    leff::Float64     # effective bounce harmonic
    n::Int            # toroidal mode number
    nutype::String    # collision operator type
    f0type::String    # distribution function type
    nufac::Float64    # collisionality scaling factor
    ximag::Float64    # imaginary contour offset
    qt::Bool          # heat flux calculation flag
    imag_axis::Bool   # true when integrating along imaginary axis
end

"""
    integrate_energy_ode(wn, wt, we, wd, wb, nuk, ell, leff, n, psi, lambda, method;
                         nutype="harmonic", f0type="maxwellian", nufac=1.0,
                         ximag=0.0, qt=false,
                         atol=1e-12, rtol=1e-9)::ComplexF64

Integrate the kinetic resonance operator over normalized energy x = E/T.
Uses OrdinaryDiffEq/Tsit5 adaptive ODE solver.

The integration path optionally steps off the real axis (0 → i*ximag)
then integrates along (0 → xmax + i*ximag) to handle poles.

Collision operator types (`nutype`):
- `"zero"`: collisionless
- `"small"`: 1e-5 * we
- `"krook"`: unmodified Krook operator
- `"harmonic"`: (1 + (l/2)^2) * krook * x^(-3/2)

Distribution function types (`f0type`):
- `"maxwellian"`: standard Maxwellian, Eq. (8) of [Logan, Park, et al., 2013]
- `"jkp"`: Jong-Kyu Park approximation [Park, Boozer, Menard, PRL 2009]
- `"cgl"`: Chew-Goldberger-Low limit

# Returns
- `ComplexF64`: energy integral value
"""
function integrate_energy_ode(wn::Float64, wt::Float64, we::Float64, wd::Float64,
                              wb::Float64, nuk::Float64, ell::Int, leff::Float64,
                              n::Int, psi::Float64, lambda::Float64, method::String;
                              nutype::String="harmonic", f0type::String="maxwellian",
                              nufac::Float64=1.0, ximag::Float64=0.0, qt::Bool=false,
                              atol::Float64=1e-12, rtol::Float64=1e-9)::ComplexF64

    y = [0.0, 0.0]    # [real, imag] parts of integral
    yi = [0.0, 0.0]   # Integration along imaginary axis

    # Optionally step off real axis to avoid poles
    if ximag != 0.0
        p_imag = EnergyParams(wn, wt, we, wd, wb, nuk, leff, n,
                              nutype, f0type, nufac, ximag, qt, true)
        prob = ODEProblem(energy_integrand!, yi, (1e-15, ximag), p_imag)
        sol = solve(prob, Tsit5(); abstol=atol, reltol=rtol, maxiters=ENERGY_MAXITERS)
        yi = sol.u[end]

        if sol.retcode != :Success
            @warn "Integration along imaginary axis may have issues"
        end
    end

    # Integration along real axis to xmax
    p_real = EnergyParams(wn, wt, we, wd, wb, nuk, leff, n,
                          nutype, f0type, nufac, ximag, qt, false)
    prob = ODEProblem(energy_integrand!, y, (1e-15, ENERGY_XMAX), p_real)
    sol = solve(prob, Tsit5(); abstol=atol, reltol=rtol, maxiters=ENERGY_MAXITERS)
    y = sol.u[end]

    if sol.retcode != :Success
        @warn "integrate_energy_ode failed at psi=$psi, lambda=$lambda, leff=$leff. Consider complex contour (ximag>0)."
    end

    return ComplexF64(y[1] + yi[1], y[2] + yi[2])
end

"""
    energy_integrand!(ydot, y, p::EnergyParams, x)

Energy integrand as a function of normalized energy x = E/T.
Implements the resonance operator from [Logan, Park, et al., Phys. Plasmas, 2013] Eq. (8).

# Arguments
- `ydot`: output derivative [real(f), imag(f)]
- `y`: current state (unused by the integrand, integral accumulates via ODE solver)
- `p::EnergyParams`: physical parameters and integration mode
- `x`: normalized energy E/T
"""
function energy_integrand!(ydot, y, p::EnergyParams, x)
    cx = p.imag_axis ? im * x : x + im * p.ximag

    # Collision frequency
    nux = if p.nutype == "zero"
        0.0
    elseif p.nutype == "small"
        1e-5 * p.we
    elseif p.nutype == "krook"
        p.nuk
    elseif p.nutype == "harmonic"
        x == 0 ? floatmax(Float64) : p.nuk * (1 + 0.25 * p.leff^2) * cx^(-1.5)
    else
        error("nutype must be zero, small, krook, or harmonic")
    end
    nux = p.nufac * nux

    # Resonance denominator: i*(l_eff*wb*sqrt(x) + n*(we + wd*x)) - nu
    denom = im * (p.leff * p.wb * sqrt(cx) + p.n * (p.we + p.wd * cx)) - nux

    # Distribution function contribution
    fx = if p.f0type == "maxwellian"
        (p.we + p.wn + p.wt * (cx - 1.5)) * cx^2.5 * exp(-cx) / denom
    elseif p.f0type == "jkp"
        (p.we + p.wn + p.wt * 2) * cx^2.5 * exp(-cx) / denom
    elseif p.f0type == "cgl"
        cx^2.5 * exp(-cx) / (im * p.n)
    else
        error("f0type must be maxwellian, jkp, or cgl")
    end

    # Heat flux moment
    if p.qt
        fx = (cx - 2.5) * fx
    end

    ydot[1] = real(fx)
    ydot[2] = imag(fx)
    return nothing
end
