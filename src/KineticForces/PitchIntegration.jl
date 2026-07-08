"""
    PitchIntegration

Pitch-angle (λ) integration for the kinetic resonance operator using
adaptive Gauss-Kronrod quadrature (QuadGK).

The pitch integral is the outer loop over pitch angle λ = μB₀/E.
At each λ, the inner loop calls `integrate_energy()` for the
energy-space integration over x = E/T.

The full resonance operator integral is:
    T(ψ) = normalization × ∫₀^λmax f(λ) × [∫₀^xmax R(x,λ) dx] dλ

For circulating particles (λ ≤ B₀/Bmax): both co- and counter-passing contribute
For trapped particles (λ > B₀/Bmax): single bounce contribution
"""

"""
    PitchGARParams

Parameters for the GAR pitch-angle integrand.
Uses a unified fbnce interpolant that returns [ωb, ωd, f₁, f₂, ...] at each λ,
matching Fortran `lambdaintgrl_lsode`.
"""
struct PitchGARParams{F}
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
    energy_atol::Float64
    energy_rtol::Float64
    rex::Float64       # real part multiplier for resonance operator
    imx::Float64       # imaginary part multiplier for resonance operator
    nqty::Int          # number of flux quantities to integrate
    fbnce::F           # CubicSeriesInterpolant: fbnce(λ) → [wb, wd, f₁, ...] (typed for stability)
    fbnce_norm::Vector{Float64}  # normalization factors (1/median)
    fbnce_hint::Base.RefValue{Int}  # sticky bracket-search hint for fbnce(λ)
    fvals::Vector{ComplexF64}    # reusable buffer for the in-place fbnce(fvals, λ) evaluation
    esegbuf::Vector{QuadGK.Segment{Float64,ComplexF64,Float64}}  # reusable energy-integral QuadGK segment buffer
end


"""
    integrate_pitch_gar_quadgk(wn, wt, we, nuk, bobmax, epsr, q, fbnce, fbnce_norm,
                                nqty, ell, n, rex, imx, psi, method; ...) → Vector{ComplexF64}

Integrate the kinetic resonance operator over pitch angle λ using adaptive
Gauss-Kronrod quadrature. Uses `QuadGK.quadgk!` with an in-place ComplexF64
kernel buffer.

The fbnce interpolant returns [ωb, ωd, f₁, f₂, ...] at each λ, where:
- f₁ = ωb|δJ|²/ro² (scalar torque)
- f₂:end = ωb·W_outer_products/ro² (kinetic matrix elements, if present)

Splits the domain at the trapped/passing boundary so Gauss-Kronrod resolves
the kink in leff = ell + n*q (circulating) → ell (trapped). One `quadgk!`
call writes all `nqty` complex quantities per λ-evaluation.

# Returns
- `Vector{ComplexF64}` of length nqty: integrated pitch-angle results
"""
function integrate_pitch_gar_quadgk(
    wn::Float64, wt::Float64, we::Float64, nuk::Float64,
    bobmax::Float64, epsr::Float64, q::Float64,
    fbnce, fbnce_norm::Vector{Float64},
    nqty::Int, ell::Int, n::Int,
    rex::Float64, imx::Float64, psi::Float64, method::String;
    nutype::String="harmonic", f0type::String="maxwellian",
    nufac::Float64=1.0, ximag::Float64=0.0, qt::Bool=false,
    energy_atol::Float64=1e-9, energy_rtol::Float64=1e-6,
    pitch_atol::Float64=1e-9, pitch_rtol::Float64=1e-6
)
    params = PitchGARParams(
        wn, wt, we, nuk, bobmax, epsr, q, ell, n, psi, method,
        nutype, f0type, nufac, ximag, qt,
        energy_atol, energy_rtol,
        rex, imx, nqty, fbnce, fbnce_norm, Ref(1),
        Vector{ComplexF64}(undef, 2 + nqty),
        QuadGK.alloc_segbuf(Float64, ComplexF64, Float64))

    lambda_min = first(fbnce.cache.x)
    lambda_max = last(fbnce.cache.x)

    # Split domain at trapped/passing boundary so Gauss-Kronrod resolves
    # the kink in leff = ell + n*q (circulating) → ell (trapped).
    bobmax_clip = clamp(bobmax, lambda_min, lambda_max)
    segments = if lambda_min < bobmax_clip < lambda_max
        (lambda_min, bobmax_clip, lambda_max)
    else
        (lambda_min, lambda_max)
    end

    # In-place quadgk! buffer; copy the result out so the returned vector is
    # owned by the caller.
    buf = zeros(ComplexF64, nqty)
    kernel! = (out, λ) -> _pitch_gar_kernel_quadgk!(out, λ, params)
    I, _ = quadgk!(kernel!, buf, segments...; atol=pitch_atol, rtol=pitch_rtol)
    return copy(I)
end


"""
    _pitch_gar_kernel_quadgk!(out::Vector{ComplexF64}, lambda, p::PitchGARParams)

In-place complex-valued kernel for `quadgk!`. Writes `out[i] = fvals[i+2] * xint_decomposed`
for i in 1..nqty. QuadGK natively handles ComplexF64.
"""
function _pitch_gar_kernel_quadgk!(out::Vector{ComplexF64}, lambda, p::PitchGARParams)
    p.fbnce(p.fvals, lambda; hint=p.fbnce_hint)
    fvals = p.fvals
    wb = real(fvals[1])
    wd = real(fvals[2])

    is_circulating = lambda <= p.bobmax
    leff = is_circulating ? Float64(p.ell) + p.n * p.q : Float64(p.ell)
    nueff = is_circulating ? p.nuk : p.nuk / (2 * p.epsr)

    if is_circulating
        xint_co = integrate_energy(p.wn, p.wt, p.we, wd, wb, nueff,
                                        p.ell, leff, p.n, p.psi, lambda, p.method;
                                        nutype=p.nutype, f0type=p.f0type,
                                        nufac=p.nufac, ximag=p.ximag, qt=p.qt,
                                        atol=p.energy_atol, rtol=p.energy_rtol, segbuf=p.esegbuf)
        xint_counter = integrate_energy(p.wn, p.wt, p.we, wd, -wb, nueff,
                                             p.ell, leff, p.n, p.psi, lambda, p.method;
                                             nutype=p.nutype, f0type=p.f0type,
                                             nufac=p.nufac, ximag=p.ximag, qt=p.qt,
                                             atol=p.energy_atol, rtol=p.energy_rtol, segbuf=p.esegbuf)
        xint = xint_co + xint_counter
    else
        xint = integrate_energy(p.wn, p.wt, p.we, wd, wb, nueff,
                                     p.ell, leff, p.n, p.psi, lambda, p.method;
                                     nutype=p.nutype, f0type=p.f0type,
                                     nufac=p.nufac, ximag=p.ximag, qt=p.qt,
                                     atol=p.energy_atol, rtol=p.energy_rtol)
    end

    xint_decomposed = complex(p.rex * real(xint), p.imx * imag(xint))

    @inbounds for i in 1:p.nqty
        out[i] = fvals[i + 2] * xint_decomposed
    end
    return nothing
end


"""
    integrate_pitch_gar_quadgk_wt(wn, wt, we, nuk, bobmax, epsr, q, fbnce, fbnce_norm,
                                   nqty, ell, n, psi, method; ...) → Vector{ComplexF64}

Dual-output variant for the kinetic-matrix path. Emits both the wmm half
(rex=0, imx=1 → Fortran kwmat) and the tmm half (rex=1, imx=0 → Fortran ktmat)
in a single pitch integration, sharing one energy integration per (λ, E).

Returns a length-`2*nqty` packed buffer: `[wmm | tmm]`. The two halves each
reproduce Fortran's independent-pass result at Fortran's element-by-element
convention (verified via matrix-dump comparison vs `~/Code/gpec/dcon/fourfit.F`
`kwmat_l`/`ktmat_l`). Downstream `kwmat ± ktmat` combinations in
`ForceFreeStates/Kinetic.jl` then reproduce `sing.f:967-1075` exactly for the
non-Hermitian B_k, C_k, E_k diagonals.
"""
function integrate_pitch_gar_quadgk_wt(
    wn::Float64, wt::Float64, we::Float64, nuk::Float64,
    bobmax::Float64, epsr::Float64, q::Float64,
    fbnce, fbnce_norm::Vector{Float64},
    nqty::Int, ell::Int, n::Int,
    psi::Float64, method::String;
    nutype::String="harmonic", f0type::String="maxwellian",
    nufac::Float64=1.0, ximag::Float64=0.0, qt::Bool=false,
    energy_atol::Float64=1e-9, energy_rtol::Float64=1e-6,
    pitch_atol::Float64=1e-9, pitch_rtol::Float64=1e-6
)
    # rex/imx unused in the dual kernel; carry 1.0 placeholders for the struct.
    params = PitchGARParams(
        wn, wt, we, nuk, bobmax, epsr, q, ell, n, psi, method,
        nutype, f0type, nufac, ximag, qt,
        energy_atol, energy_rtol,
        1.0, 1.0, nqty, fbnce, fbnce_norm, Ref(1),
        Vector{ComplexF64}(undef, 2 + nqty),
        QuadGK.alloc_segbuf(Float64, ComplexF64, Float64))

    lambda_min = first(fbnce.cache.x)
    lambda_max = last(fbnce.cache.x)

    bobmax_clip = clamp(bobmax, lambda_min, lambda_max)
    segments = if lambda_min < bobmax_clip < lambda_max
        (lambda_min, bobmax_clip, lambda_max)
    else
        (lambda_min, lambda_max)
    end

    buf = zeros(ComplexF64, 2 * nqty)
    kernel! = (out, λ) -> _pitch_gar_kernel_quadgk_wt!(out, λ, params)
    I, _ = quadgk!(kernel!, buf, segments...; atol=pitch_atol, rtol=pitch_rtol)
    return copy(I)
end


"""
    _pitch_gar_kernel_quadgk_wt!(out::Vector{ComplexF64}, lambda, p::PitchGARParams)

Dual-output pitch kernel. Fills a length-`2*nqty` buffer:
  out[1:nqty]          — fwmm half: `fvals * complex(0, imag(xint))`
  out[nqty+1:2*nqty]   — ftmm half: `fvals * complex(real(xint), 0)`

One energy integration per λ; both halves share it.
"""
function _pitch_gar_kernel_quadgk_wt!(out::Vector{ComplexF64}, lambda, p::PitchGARParams)
    p.fbnce(p.fvals, lambda; hint=p.fbnce_hint)
    fvals = p.fvals
    wb = real(fvals[1])
    wd = real(fvals[2])

    is_circulating = lambda <= p.bobmax
    leff = is_circulating ? Float64(p.ell) + p.n * p.q : Float64(p.ell)
    nueff = is_circulating ? p.nuk : p.nuk / (2 * p.epsr)

    if is_circulating
        xint_co = integrate_energy(p.wn, p.wt, p.we, wd, wb, nueff,
                                        p.ell, leff, p.n, p.psi, lambda, p.method;
                                        nutype=p.nutype, f0type=p.f0type,
                                        nufac=p.nufac, ximag=p.ximag, qt=p.qt,
                                        atol=p.energy_atol, rtol=p.energy_rtol, segbuf=p.esegbuf)
        xint_counter = integrate_energy(p.wn, p.wt, p.we, wd, -wb, nueff,
                                             p.ell, leff, p.n, p.psi, lambda, p.method;
                                             nutype=p.nutype, f0type=p.f0type,
                                             nufac=p.nufac, ximag=p.ximag, qt=p.qt,
                                             atol=p.energy_atol, rtol=p.energy_rtol, segbuf=p.esegbuf)
        xint = xint_co + xint_counter
    else
        xint = integrate_energy(p.wn, p.wt, p.we, wd, wb, nueff,
                                     p.ell, leff, p.n, p.psi, lambda, p.method;
                                     nutype=p.nutype, f0type=p.f0type,
                                     nufac=p.nufac, ximag=p.ximag, qt=p.qt,
                                     atol=p.energy_atol, rtol=p.energy_rtol)
    end

    xint_w = complex(0.0, imag(xint))   # fwmm: rex=0, imx=1
    xint_t = complex(real(xint), 0.0)   # ftmm: rex=1, imx=0

    nq = p.nqty
    @inbounds for i in 1:nq
        f = fvals[i + 2]
        out[i]      = f * xint_w
        out[i + nq] = f * xint_t
    end
    return nothing
end
