"""
    EnergyIntegration

Energy-space integration for the kinetic resonance operator.
Implements the energy integrand from [Logan, Park, et al., Phys. Plasmas, 2013] Eq. (8).

Uses QuadGK adaptive Gauss-Kronrod quadrature (the integrand is independent of the
state variable, so this is pure quadrature — not an ODE).
"""

# Integration defaults
const ENERGY_XMAX = 128.0

"""
    EnergyParams

Parameters for the energy integrand evaluation.
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
end

"""
    energy_integrand_scalar(x, p::EnergyParams; imag_axis=false) → ComplexF64

Evaluate the energy integrand at normalized energy x = E/T.
Implements [Logan, Park, et al., Phys. Plasmas, 2013] Eq. (8).
"""
function energy_integrand_scalar(x::Float64, p::EnergyParams; imag_axis::Bool=false)::ComplexF64
    cx = imag_axis ? im * x : x + im * p.ximag

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

    return fx
end

"""
    integrate_energy_ode(wn, wt, we, wd, wb, nuk, ell, leff, n, psi, lambda, method;
                         nutype="harmonic", f0type="maxwellian", nufac=1.0,
                         ximag=0.0, qt=false,
                         atol=1e-12, rtol=1e-9)::ComplexF64

Integrate the kinetic resonance operator over normalized energy x = E/T.
Uses QuadGK adaptive Gauss-Kronrod quadrature.

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
                              atol::Float64=1e-7, rtol::Float64=1e-5)::ComplexF64

    p = EnergyParams(wn, wt, we, wd, wb, nuk, leff, n,
                     nutype, f0type, nufac, ximag, qt)

    result = ComplexF64(0.0)

    # Optionally step off real axis to avoid poles
    if ximag != 0.0
        val, _ = quadgk(x -> energy_integrand_scalar(x, p; imag_axis=true),
                         1e-15, ximag; atol=atol, rtol=rtol)
        result += val
    end

    # Main integration along real axis to xmax
    val, _ = quadgk(x -> energy_integrand_scalar(x, p; imag_axis=false),
                     1e-15, ENERGY_XMAX; atol=atol, rtol=rtol)
    result += val

    return result
end
