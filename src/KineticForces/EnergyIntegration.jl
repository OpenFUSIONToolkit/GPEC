"""
    EnergyIntegration

Energy-space integration for the kinetic resonance operator.
Implements the energy integrand from [Logan, Park, et al., Phys. Plasmas, 2013] Eq. (8).

The integral over normalized energy x = E/T is mapped to the finite interval
u ∈ [0,1) by the substitution u = 1 - exp(-x), which absorbs the Maxwellian
weight exp(-x) into du and covers the full [0,∞) domain without truncation.
Resonance poles (where the denominator i·Ω(x) - ν vanishes) are removed
analytically by a Sokhotski-Plemelj decomposition, so the remaining integrand
is smooth and integrated by QuadGK adaptive Gauss-Kronrod quadrature.
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
and **without** the Maxwellian weight exp(-x). Under the u = 1-exp(-x)
substitution the factor exp(-x)·dx is absorbed into du, so the u-space
integrand is N(x)/denom(x).

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
    _energy_integrand_real(x::Float64, p::EnergyParams) → ComplexF64

Physical energy integrand in x-space, N(x)·exp(-x)/denom(x), evaluated on the
real axis. Used for diagnostics (`evaluate_energy_integrand`) and tests; the
production integral works in u-space via `integrate_energy`.
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

# Upper energy limit for the collisionless real-x-space integral, matching the
# Fortran PENTRC convention (energy.f90, `xmax`). exp(-72) ≈ 5e-32, so the energy
# tail beyond this is negligible to any quadrature tolerance.
const X_ENERGY_MAX = 72.0

"""
    _integrate_energy_collisionless(p, leff, wb, n, we, wd, atol, rtol) → ComplexF64

Collisionless (ν ≡ 0) energy integral in **real x-space** over `[0, X_ENERGY_MAX]`,
matching Fortran PENTRC's real-space integration (energy.f90). Each real resonance
pole `x_res ∈ (0, X_ENERGY_MAX)` is removed analytically by principal-value +
residue (Sokhotski-Plemelj, causal ν→0⁺ branch ∓iπ following sign Ω′), leaving a
smooth integrand for QuadGK.

The collisionless pole sits on the real axis. In x-space it stays a
well-conditioned `O(1)` number, so the pole subtraction and its breakpoint behave
cleanly. The `u = 1-exp(-x)` substitution used for the collisional path
(`integrate_energy`) compresses tail poles toward `u=1`, where the subtraction is
numerically unresolvable — hence the dedicated real-space branch here.
"""
function _integrate_energy_collisionless(p::EnergyParams, leff::Float64, wb::Float64,
                                             n::Int, we::Float64, wd::Float64,
                                             atol::Float64, rtol::Float64)::ComplexF64
    x_res_list = find_resonance_energies(leff, wb, n, we, wd)
    x_poles = Float64[]
    residues = ComplexF64[]
    pole_contribution = ComplexF64(0.0)

    for xr in x_res_list
        # Poles beyond X_ENERGY_MAX (exp(-xr) negligible) lie outside the domain — drop them.
        (xr <= 0.0 || xr >= X_ENERGY_MAX) && continue
        # Ω′(x_res) = d/dx[leff·wb·√x + n·(we + wd·x)]
        omega_prime = leff * wb / (2.0 * sqrt(xr)) + n * wd
        abs(omega_prime) < SINGULAR_EPS && continue
        # Residue of N(x)·exp(-x)/(i·Ω(x)) at x_res, where Ω ≈ Ω′·(x - x_res) near the pole.
        R = _energy_numerator(xr, p) * exp(-xr) / (im * omega_prime)
        # Causal (ν→0⁺) add-back: ∫₀^xmax R/(x - x_res) dx = R·[log(xmax - x_res) - log(x_res) ∓ iπ].
        pole_contribution += R * (log(X_ENERGY_MAX - xr) - log(xr) - im * π * sign(omega_prime))
        push!(x_poles, xr)
        push!(residues, R)
    end

    npole = length(residues)

    # Physical x-space integrand N(x)·exp(-x)/(i·Ω) minus the subtracted pole parts.
    integrand = x -> begin
        val = _energy_integrand_real(x, p)
        @inbounds for k in 1:npole
            val -= residues[k] / (x - x_poles[k])
        end
        return val
    end

    if npole == 0
        val, _ = quadgk(integrand, 0.0, X_ENERGY_MAX; atol=atol, rtol=rtol)
        return val
    end

    # Resonance locations as QuadGK breakpoints so the smooth-but-sharp integrand
    # is resolved near each pole.
    breaks = unique!(sort!(x_poles))
    smooth_val, _ = quadgk(integrand, 0.0, breaks..., X_ENERGY_MAX; atol=atol, rtol=rtol)
    return smooth_val + pole_contribution
end

"""
    integrate_energy(wn, wt, we, wd, wb, nuk, ell, leff, n, psi, lambda, method;
                     nutype="harmonic", f0type="maxwellian", nufac=1.0,
                     ximag=0.0, qt=false, atol=1e-7, rtol=1e-5) → ComplexF64

Integrate the kinetic resonance operator over normalized energy x = E/T.

The integral ∫₀^∞ N(x)·exp(-x)/denom(x) dx is mapped to u ∈ [0,1) by
u = 1 - exp(-x) (Jacobian dx/du = 1/(1-u)), giving ∫₀¹ N(x(u))/denom(x(u)) du
with no energy truncation.

Each resonance pole (root of Ω(x) = leff·wb·√x + n·(we + wd·x), shifted off
the real axis by collisions to x_pole = x_res - i·ν/Ω′) is removed by
subtracting its singular part R/(u - u_pole) from the integrand and adding
back the analytic contribution ∫₀¹ R/(u - u_pole) du = R·[log(1-u_pole) -
log(-u_pole)]. The same pole u_pole is used in both the subtraction and the
add-back, so the decomposition is exact and the remaining integrand is smooth.

This u-space path is used only when collisions are present (ν > 0), where the
pole is off the real axis. The collisionless case (ν ≡ 0) has a real-axis pole
that the u-substitution would compress toward u=1 (numerically unresolvable in
the Maxwellian tail), so it is dispatched to `_integrate_energy_collisionless`,
which integrates in real x-space over `[0, X_ENERGY_MAX]` à la Fortran PENTRC.

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
                              atol::Float64=1e-7, rtol::Float64=1e-5)::ComplexF64

    p = EnergyParams(wn, wt, we, wd, wb, nuk, leff, n,
                     nutype, f0type, nufac, ximag, qt)

    # CGL has no resonance denominator and no pole — integrate the physical
    # x-space integrand directly over the half line (QuadGK maps [0,∞) itself).
    if f0type == "cgl"
        val, _ = quadgk(x -> _energy_integrand_real(x, p), 0.0, Inf; atol=atol, rtol=rtol)
        return val
    end

    # Collisionless (ν ≡ 0): real-axis pole. Dispatch to the real-x-space integral
    # (Fortran PENTRC convention); the u-substitution below is only well-behaved
    # for the off-axis pole of the collisional case. ν vanishes identically iff it
    # vanishes at any x, so probe at x = 1.
    if _energy_collision_frequency(1.0, p) == 0.0
        return _integrate_energy_collisionless(p, leff, wb, n, we, wd, atol, rtol)
    end

    # Locate resonance poles and build their Sokhotski-Plemelj decomposition.
    x_res_list = find_resonance_energies(leff, wb, n, we, wd)
    u_poles = ComplexF64[]
    residues = ComplexF64[]
    u_breaks = Float64[]
    pole_contribution = ComplexF64(0.0)

    for xr in x_res_list
        xr <= 0.0 && continue
        # A resonance beyond x ≈ 700 sits where the Maxwellian weight exp(-x_res) underflows
        # to zero in Float64. Its pole contribution is genuinely zero, and the formula
        # R·log(-u_pole) with R=0 and u_pole≈1 trips the 0·∞ = NaN trap.
        xr > 700.0 && continue
        # Ω′(x_res) = d/dx[leff·wb·√x + n·(we + wd·x)]
        omega_prime = leff * wb / (2.0 * sqrt(xr)) + n * wd
        abs(omega_prime) < SINGULAR_EPS && continue

        u_res = -expm1(-xr)   # 1 - exp(-x_res)
        # Reached only for the collisional case (the ν ≡ 0 case is dispatched above),
        # so ν(x_res) > 0 and the pole is off the real axis. Guard defensively.
        nu_res = _energy_collision_frequency(xr, p)
        nu_res > 0.0 || continue

        # i·Ω - ν = 0 ⟹ x_pole = x_res - i·ν/Ω′. If ν overflows at a (tiny) x_res, the
        # pole is infinitely collisionally broadened — no localized pole; skip and let
        # QuadGK integrate the smooth integrand directly.
        pole_offset = nu_res / omega_prime
        isfinite(pole_offset) || continue
        x_pole = complex(xr, -pole_offset)
        u_pole = 1.0 - exp(-x_pole)
        R = _energy_numerator(xr, p) * (1.0 - u_pole) / (im * omega_prime)
        pole_contribution += R * (log(1.0 - u_pole) - log(-u_pole))

        push!(u_poles, u_pole)
        push!(residues, R)
        push!(u_breaks, u_res)
    end

    npole = length(residues)

    # Smooth integrand: full u-space integrand minus the subtracted pole parts.
    integrand = u -> begin
        u >= 1.0 && return ComplexF64(0.0)
        x = -log1p(-u)
        nux = _energy_collision_frequency(x, p)
        denom = complex(-nux, leff * wb * sqrt(x) + n * (we + wd * x))
        val = _energy_numerator(x, p) / denom
        @inbounds for k in 1:npole
            val -= residues[k] / (u - u_poles[k])
        end
        return val
    end

    if npole == 0
        val, _ = quadgk(integrand, 0.0, 1.0; atol=atol, rtol=rtol)
        return val
    end

    # Resonance locations as QuadGK breakpoints so the (now smooth but possibly
    # sharp) integrand is resolved near each pole.
    breaks = unique!(sort!(filter(b -> 0.0 < b < 1.0, u_breaks)))
    smooth_val, _ = quadgk(integrand, 0.0, breaks..., 1.0; atol=atol, rtol=rtol)
    return smooth_val + pole_contribution
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
