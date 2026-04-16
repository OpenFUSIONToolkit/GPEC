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
    _energy_integrand_real(x::Float64, p::EnergyParams) → ComplexF64

Fast Float64 kernel for the energy integrand on the real axis.
Algebraically identical to `energy_integrand_scalar(x, p; imag_axis=false)`
when `p.ximag == 0`, but avoids complex `_cpow` / `power_by_squaring` by
rewriting the half-integer powers as multiplicative `sqrt(x)` forms.

For real `x > 0` the identities are exact:
- `x^(-1.5) = 1 / (x * sqrt(x))`
- `x^2.5   = x * x * sqrt(x)`
"""
@inline function _energy_integrand_real(x::Float64, p::EnergyParams)::ComplexF64
    sqx = sqrt(x)
    inv_xsqx = 1.0 / (x * sqx)     # x^(-1.5)
    x25 = x * x * sqx              # x^2.5

    nux = if p.nutype == "zero"
        0.0
    elseif p.nutype == "small"
        1e-5 * p.we
    elseif p.nutype == "krook"
        p.nuk
    elseif p.nutype == "harmonic"
        x == 0 ? floatmax(Float64) : p.nuk * (1 + 0.25 * p.leff^2) * inv_xsqx
    else
        error("nutype must be zero, small, krook, or harmonic")
    end
    nux = p.nufac * nux

    # Resonance denominator: i*(l_eff*wb*sqrt(x) + n*(we + wd*x)) - nu
    denom = complex(-nux, p.leff * p.wb * sqx + p.n * (p.we + p.wd * x))

    emx = exp(-x)
    fx = if p.f0type == "maxwellian"
        (p.we + p.wn + p.wt * (x - 1.5)) * x25 * emx / denom
    elseif p.f0type == "jkp"
        (p.we + p.wn + p.wt * 2) * x25 * emx / denom
    elseif p.f0type == "cgl"
        complex(0.0, -x25 * emx / p.n)   # (x^2.5 * exp(-x)) / (i*n) = -i * (...)/n
    else
        error("f0type must be maxwellian, jkp, or cgl")
    end

    if p.qt
        fx = (x - 2.5) * fx
    end

    return fx
end

"""
    _energy_integrand_complex(cx::ComplexF64, p::EnergyParams) → ComplexF64

ComplexF64 kernel for the energy integrand at a complex contour point
`cx`. Handles the `ximag > 0` pole-avoidance contour (upper half plane)
and the imaginary-axis pre-step `cx = i*x`.

Uses the same `sqrt`-based half-integer rewrites as `_energy_integrand_real`
but with complex `sqrt` on the principal branch.
"""
@inline function _energy_integrand_complex(cx::ComplexF64, p::EnergyParams)::ComplexF64
    sqcx = sqrt(cx)
    inv_cxsqcx = 1.0 / (cx * sqcx)    # cx^(-1.5)
    cx25 = cx * cx * sqcx             # cx^2.5

    nux = if p.nutype == "zero"
        ComplexF64(0.0)
    elseif p.nutype == "small"
        ComplexF64(1e-5 * p.we)
    elseif p.nutype == "krook"
        ComplexF64(p.nuk)
    elseif p.nutype == "harmonic"
        cx == 0 ? ComplexF64(floatmax(Float64)) : p.nuk * (1 + 0.25 * p.leff^2) * inv_cxsqcx
    else
        error("nutype must be zero, small, krook, or harmonic")
    end
    nux = p.nufac * nux

    denom = im * (p.leff * p.wb * sqcx + p.n * (p.we + p.wd * cx)) - nux

    fx = if p.f0type == "maxwellian"
        (p.we + p.wn + p.wt * (cx - 1.5)) * cx25 * exp(-cx) / denom
    elseif p.f0type == "jkp"
        (p.we + p.wn + p.wt * 2) * cx25 * exp(-cx) / denom
    elseif p.f0type == "cgl"
        cx25 * exp(-cx) / (im * p.n)
    else
        error("f0type must be maxwellian, jkp, or cgl")
    end

    if p.qt
        fx = (cx - 2.5) * fx
    end

    return fx
end

"""
    energy_integrand_scalar(x, p::EnergyParams; imag_axis=false) → ComplexF64

Evaluate the energy integrand at normalized energy x = E/T.
Implements [Logan, Park, et al., Phys. Plasmas, 2013] Eq. (8).

Thin wrapper dispatching to `_energy_integrand_real` on the fast
Float64 path when `p.ximag == 0 && !imag_axis`, otherwise calling
`_energy_integrand_complex` with `cx = imag_axis ? im*x : x + im*p.ximag`.
"""
function energy_integrand_scalar(x::Float64, p::EnergyParams; imag_axis::Bool=false)::ComplexF64
    if !imag_axis && p.ximag == 0.0
        return _energy_integrand_real(x, p)
    else
        cx = imag_axis ? im * x : complex(x, p.ximag)
        return _energy_integrand_complex(cx, p)
    end
end

"""
    integrate_energy(wn, wt, we, wd, wb, nuk, ell, leff, n, psi, lambda, method;
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
function integrate_energy(wn::Float64, wt::Float64, we::Float64, wd::Float64,
                              wb::Float64, nuk::Float64, ell::Int, leff::Float64,
                              n::Int, psi::Float64, lambda::Float64, method::String;
                              nutype::String="harmonic", f0type::String="maxwellian",
                              nufac::Float64=1.0, ximag::Float64=0.0, qt::Bool=false,
                              atol::Float64=1e-7, rtol::Float64=1e-5)::ComplexF64

    p = EnergyParams(wn, wt, we, wd, wb, nuk, leff, n,
                     nutype, f0type, nufac, ximag, qt)

    # Fast path: ximag==0 (every production TOML), all-real-axis Float64 kernel.
    if ximag == 0.0
        val, _ = quadgk(x -> _energy_integrand_real(x, p),
                         1e-15, ENERGY_XMAX; atol=atol, rtol=rtol)
        return val
    end

    # Contour path: ximag>0 for nu=0 pole avoidance.
    # Step from (1e-15, 0) up to (1e-15, ximag) on the imaginary axis, then
    # integrate along (x + i*ximag) for x in [1e-15, xmax].
    result = ComplexF64(0.0)
    val, _ = quadgk(x -> _energy_integrand_complex(im * x, p),
                     1e-15, ximag; atol=atol, rtol=rtol)
    result += val
    val, _ = quadgk(x -> _energy_integrand_complex(complex(x, ximag), p),
                     1e-15, ENERGY_XMAX; atol=atol, rtol=rtol)
    result += val

    return result
end


"""
    evaluate_energy_integrand(x_grid; wn, wt, we, wd, wb, nuk, leff, n,
                               nutype="harmonic", f0type="maxwellian",
                               nufac=1.0, ximag=0.0, qt=false) → Vector{ComplexF64}

Diagnostic convenience: evaluate the energy integrand at specified x = E/T values.
Returns the integrand value (not the integral) at each point in `x_grid`.
Useful for plotting the energy integrand shape and verifying kinetic resonance
resolution.

# Example
```julia
x = 10 .^ range(-2, stop=2, length=500)
f = KineticForces.evaluate_energy_integrand(x; wn=1e3, wt=2e3, we=5e4,
        wd=1e2, wb=3e4, nuk=1e3, leff=1.0, n=1)
plot(x, real.(f); xscale=:log10, xlabel="x = E/T", ylabel="Re(integrand)")
```
"""
function evaluate_energy_integrand(x_grid::AbstractVector{Float64};
                                    wn::Float64, wt::Float64, we::Float64,
                                    wd::Float64, wb::Float64, nuk::Float64,
                                    leff::Float64, n::Int,
                                    nutype::String="harmonic", f0type::String="maxwellian",
                                    nufac::Float64=1.0, ximag::Float64=0.0, qt::Bool=false)
    p = EnergyParams(wn, wt, we, wd, wb, nuk, leff, n, nutype, f0type, nufac, ximag, qt)
    return [energy_integrand_scalar(x, p) for x in x_grid]
end
