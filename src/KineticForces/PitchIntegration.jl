"""
    PitchIntegration

Pitch-angle (λ) ODE integration for the kinetic resonance operator.
Equivalent to Fortran `pitch.f90:lambdaintgrl_lsode`.

The pitch integral is the outer loop over pitch angle λ = μB₀/E.
At each λ, the inner loop calls `integrate_energy_ode()` for the
energy-space integration over x = E/T.

The full resonance operator integral is:
    T(ψ) = normalization × ∫₀^λmax f(λ) × [∫₀^xmax R(x,λ) dx] dλ

For circulating particles (λ ≤ B₀/Bmax): both co- and counter-passing contribute
For trapped particles (λ > B₀/Bmax): single bounce contribution
"""

using OrdinaryDiffEq

"""
    PitchParams

Parameters for the pitch-angle ODE integrand.
"""
struct PitchParams
    wn::Float64        # density gradient diamagnetic drift frequency
    wt::Float64        # temperature gradient diamagnetic drift frequency
    we::Float64        # electric precession frequency
    nuk::Float64       # effective collision frequency
    bobmax::Float64    # trapped-passing boundary (B₀/Bmax)
    epsr::Float64      # inverse aspect ratio
    q::Float64         # safety factor
    ell::Int           # bounce harmonic
    n::Int             # toroidal mode number
    psi::Float64       # normalized flux
    method::String     # method label
    nutype::String     # collision operator type
    f0type::String     # distribution function type
    nufac::Float64     # collisionality scaling factor
    ximag::Float64     # imaginary contour offset
    qt::Bool           # heat flux flag
    energy_atol::Float64  # energy integration absolute tolerance
    energy_rtol::Float64  # energy integration relative tolerance
    # Pitch-angle equilibrium functions (callable interpolants)
    wb_interp::Any     # bounce frequency ωb(λ)
    wd_interp::Any     # magnetic precession ωd(λ)
    flux_interps::Vector{Any}  # pitch-angle flux function interpolants f_i(λ)
end

"""
    integrate_pitch_ode(wn, wt, we, nuk, bobmax, epsr, q,
                        wb_interp, wd_interp, flux_interps,
                        ell, n, psi, method;
                        lambda_min=0.0, lambda_max=1.0,
                        nutype="harmonic", f0type="maxwellian",
                        nufac=1.0, ximag=0.0, qt=false,
                        energy_atol=1e-12, energy_rtol=1e-9,
                        pitch_atol=1e-12, pitch_rtol=1e-9)

Integrate the kinetic resonance operator over pitch angle λ using ODE solver.
At each λ, calls `integrate_energy_ode()` for the inner energy-space integration.

# Arguments
- `wn`: density gradient diamagnetic drift frequency
- `wt`: temperature gradient diamagnetic drift frequency
- `we`: electric precession frequency
- `nuk`: effective collision frequency
- `bobmax`: trapped-passing boundary B₀/Bmax at this ψ
- `epsr`: inverse aspect ratio
- `q`: safety factor
- `wb_interp`: bounce frequency interpolant ωb(λ)
- `wd_interp`: magnetic precession interpolant ωd(λ)
- `flux_interps`: vector of pitch-angle flux function interpolants
- `ell`: bounce harmonic
- `n`: toroidal mode number
- `psi`: normalized flux
- `method`: torque calculation method label

# Returns
- `Vector{ComplexF64}`: integrated results, one per flux function component
"""
function integrate_pitch_ode(wn::Float64, wt::Float64, we::Float64, nuk::Float64,
                             bobmax::Float64, epsr::Float64, q::Float64,
                             wb_interp, wd_interp, flux_interps::Vector,
                             ell::Int, n::Int, psi::Float64, method::String;
                             lambda_min::Float64=0.0, lambda_max::Float64=1.0,
                             nutype::String="harmonic", f0type::String="maxwellian",
                             nufac::Float64=1.0, ximag::Float64=0.0, qt::Bool=false,
                             energy_atol::Float64=1e-12, energy_rtol::Float64=1e-9,
                             pitch_atol::Float64=1e-12, pitch_rtol::Float64=1e-9)

    nqty = length(flux_interps)
    neq = 2 * nqty  # real + imag for each component

    params = PitchParams(wn, wt, we, nuk, bobmax, epsr, q, ell, n, psi, method,
                         nutype, f0type, nufac, ximag, qt,
                         energy_atol, energy_rtol,
                         wb_interp, wd_interp, flux_interps)

    y0 = zeros(neq)
    prob = ODEProblem(pitch_integrand!, y0, (lambda_min, lambda_max), params)
    sol = solve(prob, Tsit5(); reltol=pitch_rtol, abstol=pitch_atol)

    if sol.retcode != :Success
        @warn "integrate_pitch_ode may have issues at psi=$psi, method=$method"
    end

    # Extract final complex results
    yf = sol.u[end]
    return [complex(yf[2*(i-1)+1], yf[2*(i-1)+2]) for i in 1:nqty]
end


"""
    pitch_integrand!(dy, y, p::PitchParams, lambda)

Pitch-angle integrand for the ODE solver.
At each λ, evaluates the energy integral and multiplies by pitch-angle flux functions.

For circulating particles (λ ≤ bobmax): sum co-passing (ωb > 0) and counter-passing (ωb < 0)
For trapped particles (λ > bobmax): single bounce contribution
"""
function pitch_integrand!(dy, y, p::PitchParams, lambda)
    wb = p.wb_interp(lambda)
    wd = p.wd_interp(lambda)

    # Effective bounce harmonic: ℓ + nq for circulating, ℓ for trapped
    is_circulating = lambda <= p.bobmax
    leff = is_circulating ? Float64(p.ell) + p.n * p.q : Float64(p.ell)

    # Collision frequency: unmodified for circulating, scaled by 1/(2ε) for trapped
    nueff = is_circulating ? p.nuk : p.nuk / (2 * p.epsr)

    # Inner energy integration
    if is_circulating
        # Co-passing (ωb > 0) + counter-passing (ωb < 0)
        xint_co = integrate_energy_ode(p.wn, p.wt, p.we, wd, wb, nueff,
                                        p.ell, leff, p.n, p.psi, lambda, p.method;
                                        nutype=p.nutype, f0type=p.f0type,
                                        nufac=p.nufac, ximag=p.ximag, qt=p.qt,
                                        atol=p.energy_atol, rtol=p.energy_rtol)
        xint_counter = integrate_energy_ode(p.wn, p.wt, p.we, wd, -wb, nueff,
                                             p.ell, leff, p.n, p.psi, lambda, p.method;
                                             nutype=p.nutype, f0type=p.f0type,
                                             nufac=p.nufac, ximag=p.ximag, qt=p.qt,
                                             atol=p.energy_atol, rtol=p.energy_rtol)
        xint = xint_co + xint_counter
    else
        # Trapped: single bounce
        xint = integrate_energy_ode(p.wn, p.wt, p.we, wd, wb, nueff,
                                     p.ell, leff, p.n, p.psi, lambda, p.method;
                                     nutype=p.nutype, f0type=p.f0type,
                                     nufac=p.nufac, ximag=p.ximag, qt=p.qt,
                                     atol=p.energy_atol, rtol=p.energy_rtol)
    end

    # Multiply by pitch-angle flux functions
    nqty = length(p.flux_interps)
    for i in 1:nqty
        fres = p.flux_interps[i](lambda) * xint
        dy[2*(i-1)+1] = real(fres)
        dy[2*(i-1)+2] = imag(fres)
    end

    return nothing
end
