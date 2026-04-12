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


# ============================================================================
# QuadGK-based energy integration
# ============================================================================

"""
    find_resonance_energies(leff, wb, n, we, wd) → Vector{Float64}

Find real positive energy roots x_res where the resonance denominator vanishes:
    leff·wb·√x + n·(we + wd·x) = 0

Substituting u = √x gives a quadratic in u:
    n·wd·u² + leff·wb·u + n·we = 0

Returns only real positive x = u² values.
"""
function find_resonance_energies(leff::Float64, wb::Float64, n::Int, we::Float64, wd::Float64)::Vector{Float64}
    a = n * wd
    b = leff * wb
    c = n * we

    roots = Float64[]

    if abs(a) < 1e-30
        # Linear case: b*u + c = 0  →  u = -c/b
        abs(b) < 1e-30 && return roots
        u = -c / b
        u > 0.0 && push!(roots, u^2)
    else
        disc = b^2 - 4 * a * c
        disc < 0.0 && return roots
        sd = sqrt(disc)
        # Two roots via the standard quadratic formula
        for u in ((-b + sd) / (2 * a), (-b - sd) / (2 * a))
            u > 0.0 && push!(roots, u^2)
        end
    end

    return roots
end


"""
    collision_frequency(x, nuk, leff, nutype, nufac) → Float64

Evaluate the collision frequency ν(x) at energy x, matching `energy_integrand!`.
"""
function collision_frequency(x::Float64, nuk::Float64, leff::Float64, nutype::String, nufac::Float64)::Float64
    nux = if nutype == "zero"
        0.0
    elseif nutype == "small"
        1e-5
    elseif nutype == "krook"
        nuk
    elseif nutype == "harmonic"
        x <= 0.0 ? floatmax(Float64) : nuk * (1 + 0.25 * leff^2) * x^(-1.5)
    else
        error("nutype must be zero, small, krook, or harmonic")
    end
    return nufac * nux
end


"""
    energy_numerator(x, wn, wt, we, n, f0type, qt) → ComplexF64

Evaluate the numerator N(x) of the energy integrand (without the resonance denominator
or Maxwellian weight exp(-x)). Under the u-substitution u = 1-exp(-x), the factor
exp(-x)·dx is absorbed into du, so N(x) = (drift frequency terms) × x^2.5 × (optional qt).

For CGL, the integrand has no resonance denominator: N_cgl = x^2.5 / (i·n).
"""
function energy_numerator(x::Float64, wn::Float64, wt::Float64, we::Float64, n::Int, f0type::String, qt::Bool)::ComplexF64
    fx = if f0type == "maxwellian"
        (we + wn + wt * (x - 1.5)) * x^2.5
    elseif f0type == "jkp"
        (we + wn + wt * 2) * x^2.5
    elseif f0type == "cgl"
        x^2.5 / (im * n)
    else
        error("f0type must be maxwellian, jkp, or cgl")
    end
    qt && (fx *= (x - 2.5))
    return ComplexF64(fx)
end


"""
    integrate_energy_quadgk(wn, wt, we, wd, wb, nuk, ell, leff, n, psi, lambda, method;
                            nutype, f0type, nufac, qt, atol, rtol) → ComplexF64

Integrate the kinetic resonance operator over energy using QuadGK adaptive quadrature
with the u-substitution u = 1 - exp(-x) mapping [0,∞) → [0,1).

Resonance poles are handled analytically via Sokhotski-Plemelj decomposition:
the singular part is subtracted from the integrand and its contribution computed
via the complex logarithm formula, which handles both ν > 0 and ν → 0 correctly.

See plan section 2d for mathematical details.
"""
function integrate_energy_quadgk(wn::Float64, wt::Float64, we::Float64, wd::Float64,
                                 wb::Float64, nuk::Float64, ell::Int, leff::Float64,
                                 n::Int, psi::Float64, lambda::Float64, method::String;
                                 nutype::String="harmonic", f0type::String="maxwellian",
                                 nufac::Float64=1.0, qt::Bool=false,
                                 atol::Float64=1e-12, rtol::Float64=1e-9)::ComplexF64
    # CGL has no resonance denominator — integrate directly
    if f0type == "cgl"
        integrand_cgl = u -> begin
            u >= 1.0 && return ComplexF64(0.0)
            x = -log(1.0 - u)
            return energy_numerator(x, wn, wt, we, n, "cgl", qt)
        end
        val, _ = quadgk(integrand_cgl, 0.0, 1.0; atol, rtol)
        return val
    end

    # Find resonance locations
    x_res_list = find_resonance_energies(leff, wb, n, we, wd)

    # Build residue data for Sokhotski-Plemelj subtraction
    u_breaks = Float64[]
    residues_u = ComplexF64[]
    pole_contributions = ComplexF64[]

    for xr in x_res_list
        xr <= 0.0 && continue

        # Ω'(x_res) = d/dx[leff*wb*√x + n*(we + wd*x)] at x=xr
        omega_prime = leff * wb / (2.0 * sqrt(xr)) + n * wd
        abs(omega_prime) < 1e-30 && continue

        ur = 1.0 - exp(-xr)
        push!(u_breaks, ur)

        # Numerator at resonance (no exp(-x) weight — absorbed by substitution)
        N_res = energy_numerator(xr, wn, wt, we, n, f0type, qt)

        # Residue in u-space: R_u = N(x_res) * (1-u_res) / (i*Ω')
        # The (1-u_res) = exp(-x_res) factor comes from dx/du = 1/(1-u)
        R_u = N_res * (1.0 - ur) / (im * omega_prime)
        push!(residues_u, R_u)

        # Pole location in u-space (complex for ν > 0, real for ν = 0)
        nux_res = collision_frequency(xr, nuk, leff, nutype, nufac)
        # x_pole = x_res + i*ν/(Ω') — the pole of 1/(i*Ω - ν) moves off real axis
        x_pole = xr + im * nux_res / omega_prime
        u_pole = 1.0 - exp(-x_pole)

        # Analytical integral of R_u/(u - u_pole) over [0, 1):
        # ∫₀¹ R_u/(u - u_pole) du = R_u * [log(1 - u_pole) - log(-u_pole)]
        push!(pole_contributions, R_u * (log(1.0 - u_pole) - log(-u_pole)))
    end

    n_poles = length(u_breaks)

    # Smooth integrand with poles subtracted
    integrand_u = u -> begin
        u >= 1.0 && return ComplexF64(0.0)
        x = -log(1.0 - u)

        # Collision frequency at this x
        nux = collision_frequency(x, nuk, leff, nutype, nufac)

        # Full resonance denominator
        denom = im * (leff * wb * sqrt(x) + n * (we + wd * x)) - nux

        # Full integrand value (numerator / denominator, no exp(-x) weight)
        val = energy_numerator(x, wn, wt, we, n, f0type, qt) / denom

        # Subtract singular parts at each resonance
        for k in 1:n_poles
            val -= residues_u[k] / (u - u_breaks[k])
        end

        return val
    end

    # QuadGK with resonance locations as breakpoints
    breaks = sort(u_breaks)
    smooth_val, _ = quadgk(integrand_u, 0.0, breaks..., 1.0; atol, rtol)

    return smooth_val + sum(pole_contributions; init=ComplexF64(0.0))
end
