"""
    EnergyIntegration

Energy-space integration for the kinetic resonance operator.
Implements the energy integrand from [Logan, Park, et al., Phys. Plasmas, 2013] Eq. (8).

The integral over normalized energy x = E/T is evaluated in real x-space over
`[0, X_ENERGY_MAX]`. The integrand and every resonance pole decay as `x^p·exp(-x)`,
so the upper limit is set where that tail falls far below any quadrature tolerance
(see `X_ENERGY_MAX`). Resonance poles (where the denominator i·Ω(x) - ν vanishes,
shifted off the real axis to x_pole = x_res - i·ν/Ω′ by collisions) are removed
analytically by a Sokhotski-Plemelj decomposition, so the remaining integrand
is smooth and integrated by QuadGK adaptive Gauss-Kronrod quadrature. One path
serves all collisionalities; the collisionless case is the exact ν→0 limit.
"""

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
    ximag::Float64    # deprecated: imaginary contour offset, no longer used
    qt::Bool          # heat flux calculation flag
end

"""
    _energy_collision_frequency(x::Float64, p::EnergyParams) → Float64

Collision frequency ν(x) at normalized energy x = E/T.

- `"zero"`: collisionless (ν = 0)
- `"small"`: 1e-5 * we
- `"krook"`: unmodified Krook operator
- `"harmonic"`: (1 + (l/2)²) * krook * x^(-3/2)
"""
@inline function _energy_collision_frequency(x::Float64, p::EnergyParams)::Float64
    nux = if p.nutype == "zero"
        0.0
    elseif p.nutype == "small"
        1e-5 * p.we
    elseif p.nutype == "krook"
        p.nuk
    elseif p.nutype == "harmonic"
        x <= 0.0 ? floatmax(Float64) : p.nuk * (1 + 0.25 * p.leff^2) / (x * sqrt(x))
    else
        error("nutype must be zero, small, krook, or harmonic")
    end
    return p.nufac * nux
end

"""
    _energy_numerator(x::Float64, p::EnergyParams) → ComplexF64

Numerator N(x) of the energy integrand, **without** the resonance denominator
and **without** the Maxwellian weight exp(-x). The physical x-space integrand is
N(x)·exp(-x)/denom(x); the residue at a pole uses N(x_res)·exp(-x_res).

For CGL there is no resonance denominator: N_cgl = x^2.5 / (i·n).
"""
@inline function _energy_numerator(x::Float64, p::EnergyParams)::ComplexF64
    x25 = x * x * sqrt(x)   # x^2.5
    fx = if p.f0type == "maxwellian"
        ComplexF64((p.we + p.wn + p.wt * (x - 1.5)) * x25)
    elseif p.f0type == "jkp"
        ComplexF64((p.we + p.wn + p.wt * 2) * x25)
    elseif p.f0type == "cgl"
        complex(0.0, -x25 / p.n)   # x^2.5 / (i*n)
    else
        error("f0type must be maxwellian, jkp, or cgl")
    end
    if p.qt
        fx *= (x - 2.5)
    end
    return fx
end

"""
    _energy_numerator_deriv(x::Float64, p::EnergyParams) → ComplexF64

x-derivative dN/dx of `_energy_numerator`, used for the analytic regular-part
limit at a real-axis (ν=0) resonance pole. CGL is excluded — the real-pole path
never carries a CGL numerator (CGL has no pole).
"""
@inline function _energy_numerator_deriv(x::Float64, p::EnergyParams)::ComplexF64
    sx = sqrt(x)
    x15 = x * sx            # x^1.5
    x25 = x * x15           # x^2.5
    if p.f0type == "maxwellian"
        a = p.we + p.wn + p.wt * (x - 1.5)
        nn = ComplexF64(a * x25)
        dn = ComplexF64(p.wt * x25 + a * 2.5 * x15)   # d/dx[(…)·x^2.5]
    elseif p.f0type == "jkp"
        a = p.we + p.wn + p.wt * 2
        nn = ComplexF64(a * x25)
        dn = ComplexF64(a * 2.5 * x15)
    else
        error("_energy_numerator_deriv supports maxwellian and jkp")
    end
    return p.qt ? dn * (x - 2.5) + nn : dn   # d/dx[N·(x-2.5)] = N′·(x-2.5) + N
end

"""
    _energy_integrand_real(x::Float64, p::EnergyParams) → ComplexF64

Physical energy integrand in x-space, N(x)·exp(-x)/denom(x), evaluated on the
real axis. Used both by the production integral (`integrate_energy` via
`_integrate_energy_resonant`) and for diagnostics (`evaluate_energy_integrand`).
"""
@inline function _energy_integrand_real(x::Float64, p::EnergyParams)::ComplexF64
    sqx = sqrt(x)
    nux = _energy_collision_frequency(x, p)
    # Resonance denominator: i*(l_eff*wb*sqrt(x) + n*(we + wd*x)) - nu
    denom = complex(-nux, p.leff * p.wb * sqx + p.n * (p.we + p.wd * x))
    emx = exp(-x)
    if p.f0type == "cgl"
        # CGL has no resonance denominator.
        fx = complex(0.0, -x * x * sqx * emx / p.n)
        return p.qt ? (x - 2.5) * fx : fx
    end
    return _energy_numerator(x, p) * emx / denom
end

"""
    energy_integrand_scalar(x::Float64, p::EnergyParams) → ComplexF64

Evaluate the physical energy integrand N(x)·exp(-x)/denom(x) at normalized
energy x = E/T. Implements [Logan, Park, et al., Phys. Plasmas, 2013] Eq. (8).
"""
energy_integrand_scalar(x::Float64, p::EnergyParams)::ComplexF64 = _energy_integrand_real(x, p)

"""
    find_resonance_energies(leff, wb, n, we, wd) → Vector{Float64}

Real positive energies x_res where the resonance condition vanishes:

    Ω(x) = leff·wb·√x + n·(we + wd·x) = 0

With s = √x this is the quadratic n·wd·s² + leff·wb·s + n·we = 0. Returns the
x = s² values for the positive real roots (the locations of the resonance
poles of the energy integrand).
"""
function find_resonance_energies(leff::Float64, wb::Float64, n::Int, we::Float64, wd::Float64)::Vector{Float64}
    a = n * wd
    b = leff * wb
    c = n * we
    roots = Float64[]
    if abs(a) < SINGULAR_EPS
        # Linear case: b·s + c = 0
        abs(b) < SINGULAR_EPS && return roots
        s = -c / b
        s > 0.0 && push!(roots, s^2)
    else
        disc = b^2 - 4 * a * c
        disc < 0.0 && return roots
        sd = sqrt(disc)
        for s in ((-b + sd) / (2 * a), (-b - sd) / (2 * a))
            s > 0.0 && push!(roots, s^2)
        end
    end
    return roots
end

# Upper limit of the real-x energy integral. Integrand and poles decay as x^p·exp(-x)
# (p ≤ 3.5); at x=100 the tail is ~1e-37, far below any quadrature tolerance.
const X_ENERGY_MAX = 100.0

"""
    _real_pole_regular_part(xr, p, leff, wb, n, wd) → ComplexF64

Laurent regular part (finite limit at `x → x_res`) of the pole-subtracted
real-axis (ν=0) integrand `N(x)·exp(-x)/(i·Ω(x)) − R/(x − x_res)`, with
`Ω(x) = leff·wb·√x + n·(we+wd·x)` and `R = N(x_res)·exp(-x_res)/(i·Ω′)`.

Writing `h(x) = N(x)·exp(-x)`, the limit is
`[h′(x_res) − h(x_res)·Ω″/(2Ω′)] / (i·Ω′)`, with `h′ = (N′ − N)·exp(-x)`,
`Ω′ = leff·wb/(2√x) + n·wd`, `Ω″ = −leff·wb/(4·x^{3/2})`.
"""
@inline function _real_pole_regular_part(xr::Float64, p::EnergyParams, leff::Float64,
                                                 wb::Float64, n::Int, wd::Float64)::ComplexF64
    sx = sqrt(xr)
    op = leff * wb / (2.0 * sx) + n * wd          # Ω′(x_res)
    opp = -leff * wb / (4.0 * xr * sx)            # Ω″(x_res)
    nn = _energy_numerator(xr, p)                 # N(x_res)
    dn = _energy_numerator_deriv(xr, p)           # N′(x_res)
    return exp(-xr) * ((dn - nn) - nn * opp / (2.0 * op)) / (im * op)
end

"""
    _integrate_energy_resonant(p, leff, wb, n, we, wd, atol, rtol) → ComplexF64

Energy integral in **real x-space** over `[0, X_ENERGY_MAX]`, matching Fortran
PENTRC's real-space integration (energy.f90). One path for any collisionality:
each resonance contributes a pole `x_pole = x_res − i·ν/Ω′`, removed analytically by
principal-value + residue (Sokhotski-Plemelj), leaving a smooth integrand for QuadGK.
The collisionless case (ν ≡ 0, real-axis pole) is the exact ν→0⁺ limit of this single
formula; its causal branch is carried by the signed zero of `−pole_offset` (see the
add-back below), and its on-axis `0/0` window by the analytic regular-part limit.
"""
# Real x-space resonant integrand with pole subtractions
# Explicit function keeps memory allocation out of the QuadGK inner loop. 
@inline function _resonant_integrand(x::Float64, p::EnergyParams,
        residues::Vector{ComplexF64}, x_poles::Vector{ComplexF64}, npole::Int,
        leff::Float64, wb::Float64, n::Int, wd::Float64)::ComplexF64
    val = _energy_integrand_real(x, p)
    @inbounds for k in 1:npole
        val -= residues[k] / (x - x_poles[k])
    end
    if !isfinite(val)
        k = 1
        @inbounds for j in 2:npole
            abs(x - real(x_poles[j])) < abs(x - real(x_poles[k])) && (k = j)
        end
        val = _real_pole_regular_part(real(x_poles[k]), p, leff, wb, n, wd)
        @inbounds for j in 1:npole
            j == k && continue
            val -= residues[j] / (x - x_poles[j])
        end
    end
    return val
end

function _integrate_energy_resonant(p::EnergyParams, leff::Float64, wb::Float64,
                                        n::Int, we::Float64, wd::Float64,
                                        atol::Float64, rtol::Float64, segbuf=nothing)::ComplexF64
    x_res_list = find_resonance_energies(leff, wb, n, we, wd)   # ≤ 2 roots of a quadratic in √x
    x_poles = ComplexF64[]
    residues = ComplexF64[]
    sizehint!(x_poles, length(x_res_list))
    sizehint!(residues, length(x_res_list))
    pole_contribution = ComplexF64(0.0)

    for xr in x_res_list
        # Poles beyond X_ENERGY_MAX (exp(-xr) negligible) lie outside the domain — drop them.
        (xr <= 0.0 || xr >= X_ENERGY_MAX) && continue
        # Ω′(x_res) = d/dx[leff·wb·√x + n·(we + wd·x)]
        omega_prime = leff * wb / (2.0 * sqrt(xr)) + n * wd
        abs(omega_prime) < SINGULAR_EPS && continue
        # Collisions shift the pole off-axis: x_pole = x_res - i·ν/Ω′. Skip if ν overflows
        # (harmonic ν at tiny x_res): pole infinitely broadened, integrand stays smooth.
        pole_offset = _energy_collision_frequency(xr, p) / omega_prime
        isfinite(pole_offset) || continue
        x_pole = complex(xr, -pole_offset)   # signed-zero imag part for ν=0 carries the causal branch
        # Residue of N(x)·exp(-x)/(i·Ω(x)) at x_res, where Ω ≈ Ω′·(x - x_res) near the pole.
        R = _energy_numerator(xr, p) * exp(-xr) / (im * omega_prime)
        # Add-back ∫₀^xmax R/(x - x_pole) dx = R·[log(xmax - x_pole) - log(-x_pole)].
        pole_contribution += R * (log(X_ENERGY_MAX - x_pole) - log(-x_pole))
        push!(x_poles, x_pole)
        push!(residues, R)
    end

    npole = length(residues)

    # Physical x-space integrand N(x)·exp(-x)/(i·Ω) minus the subtracted pole parts.
    integrand = x -> _resonant_integrand(x, p, residues, x_poles, npole, leff, wb, n, wd)

    if npole == 0
        val, _ = quadgk(integrand, 0.0, X_ENERGY_MAX; atol=atol, rtol=rtol, segbuf=segbuf)
        return val
    end

    # Pole real parts as QuadGK breakpoints so the smooth-but-sharp integrand is resolved
    # near each pole. With ≤ 2 poles, branch on the breakpoint count to pass them as
    # positional args — a `breaks...` splat of a runtime-length Vector is type-unstable.
    breaks = unique(sort(real.(x_poles)))
    smooth_val, _ = if length(breaks) == 1
        quadgk(integrand, 0.0, breaks[1], X_ENERGY_MAX; atol=atol, rtol=rtol, segbuf=segbuf)
    else
        quadgk(integrand, 0.0, breaks[1], breaks[2], X_ENERGY_MAX; atol=atol, rtol=rtol, segbuf=segbuf)
    end
    return smooth_val + pole_contribution
end

"""
    integrate_energy(wn, wt, we, wd, wb, nuk, ell, leff, n, psi, lambda, method;
                     nutype="harmonic", f0type="maxwellian", nufac=1.0,
                     ximag=0.0, qt=false, atol=1e-7, rtol=1e-5) → ComplexF64

Integrate the kinetic resonance operator over normalized energy x = E/T.

The integral ∫₀^∞ N(x)·exp(-x)/denom(x) dx is evaluated in real x-space over
`[0, X_ENERGY_MAX]` (the integrand and its poles decay as `x^p·exp(-x)`, so the
tail there is far below any tolerance) via `_integrate_energy_resonant`. Each
resonance pole (root of
Ω(x) = leff·wb·√x + n·(we + wd·x), shifted off the real axis by collisions to
x_pole = x_res - i·ν/Ω′) is removed by subtracting its singular part R/(x - x_pole)
and adding back the analytic principal-value + residue. A single formula handles
all collisionalities: the collisionless case (ν ≡ 0) is the exact ν→0 limit, with
its real-axis pole resolved analytically (see `_integrate_energy_resonant`).

Collision operator types (`nutype`): `"zero"`, `"small"`, `"krook"`, `"harmonic"`.
Distribution function types (`f0type`): `"maxwellian"`, `"jkp"`, `"cgl"`.

`ximag` is accepted for backward compatibility but no longer used — resonance
poles are now handled analytically rather than by contour deformation.

# Returns
- `ComplexF64`: energy integral value
"""
function integrate_energy(wn::Float64, wt::Float64, we::Float64, wd::Float64,
                              wb::Float64, nuk::Float64, ell::Int, leff::Float64,
                              n::Int, psi::Float64, lambda::Float64, method::String;
                              nutype::String="harmonic", f0type::String="maxwellian",
                              nufac::Float64=1.0, ximag::Float64=0.0, qt::Bool=false,
                              atol::Float64=1e-7, rtol::Float64=1e-5, segbuf=nothing)::ComplexF64

    p = EnergyParams(wn, wt, we, wd, wb, nuk, leff, n,
                     nutype, f0type, nufac, ximag, qt)

    # CGL has no resonance denominator and no pole — integrate the physical
    # x-space integrand directly over the half line (QuadGK maps [0,∞) itself).
    if f0type == "cgl"
        val, _ = quadgk(x -> _energy_integrand_real(x, p), 0.0, Inf; atol=atol, rtol=rtol, segbuf=segbuf)
        return val
    end

    # Single resonant path for any collisionality: real x-space PV+residue with a
    # pole x_pole = x_res - i·ν/Ω′ (off-axis for ν>0, real for ν≡0).
    return _integrate_energy_resonant(p, leff, wb, n, we, wd, atol, rtol, segbuf)
end

"""
    evaluate_energy_integrand(x_grid; wn, wt, we, wd, wb, nuk, leff, n,
                               nutype="harmonic", f0type="maxwellian",
                               nufac=1.0, ximag=0.0, qt=false) → Vector{ComplexF64}

Diagnostic convenience: evaluate the physical x-space energy integrand
N(x)·exp(-x)/denom(x) at specified x = E/T values. Returns the integrand value
(not the integral) at each point in `x_grid`. Useful for plotting the energy
integrand shape and verifying kinetic resonance resolution.

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
