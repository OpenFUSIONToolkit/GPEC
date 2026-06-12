"""
FourierCoefficients - Lightweight FFT-based Fourier decomposition

Computes and stores Fourier coefficients from 2D periodic data without
creating spline interpolants. Use this when you only need to access
coefficients at original grid points.

# Usage

```julia
xs = range(0, 1; length=100) |> collect
ys = range(0, 2π; length=64) |> collect
fs = zeros(100, 64, 2)
for i in 1:100, j in 1:64
    fs[i, j, 1] = exp(-xs[i]) * cos(3*ys[j])
end
fc = FourierCoefficients(xs, ys, fs, 10)  # Keep 10 modes
c = get_complex_coeff(fc, 50, 3, 1)  # Get mode 3 at ipsi=50
```
"""

using FFTW
import FastInterpolations: cubic_interp, PeriodicBC, deriv1, Series

"""
    FourierCoefficients

Lightweight container for Fourier coefficients without spline interpolation.
Use this when you only need to access coefficients at original grid points.

# Fields

  - `xs::Vector{Float64}`: Radial coordinates
  - `mband::Int`: Number of Fourier modes (0 to mband inclusive)
  - `nqty::Int`: Number of quantities
  - `cos_coeffs::Array{Float64,3}`: Cosine coefficients (npsi × nmodes × nqty)
  - `sin_coeffs::Array{Float64,3}`: Sine coefficients (npsi × nmodes × nqty)
"""
struct FourierCoefficients
    xs::Vector{Float64}
    mband::Int
    nqty::Int
    cos_coeffs::Array{Float64,3}
    sin_coeffs::Array{Float64,3}
end

"""
    empty_FourierCoefficients()

Create an empty FourierCoefficients for initialization purposes.
"""
function empty_FourierCoefficients()
    FourierCoefficients(Float64[], 0, 0, zeros(Float64, 0, 1, 0), zeros(Float64, 0, 1, 0))
end

"""
    FourierCoefficients(xs, ys, fs, mband; fft_flag=true)

Compute poloidal Fourier coefficients of `fs` without retaining a spline interpolant.

Two methods are available, selected by `fft_flag` (mirroring the Fortran `fspline_fit`
dispatch and the `fft_flag` namelist control):

  - `fft_flag=true` — bare FFT of the `ntheta` samples (Fortran `fspline_fit_2`). Fast, but the
    DFT aliases any spectral content above the Nyquist limit, so the retained low-`m` coefficients
    (and hence Δ′) converge only as `mtheta` grows when `fs` is not band-limited (e.g. the metric
    tensor on shaped geometry).
  - `fft_flag=false` — spline quadrature (Fortran `fspline_fit_1`): a periodic cubic spline is fit
    through the samples and each coefficient is the **analytic integral** of that spline against
    `exp(-imθ)`. Alias-free and high-order, so the coefficients are essentially `mtheta`-independent.
    This is the default in the Fortran code and the recommended path for the metric tensor.

Both paths use the identical convention `c_m = (1/2π) ∫ f(θ) e^{-imθ} dθ`, stored such that
[`get_complex_coeff`](@ref) returns `c_m`, so they are interchangeable downstream.

# Arguments

  - `xs::Vector{Float64}`: Radial coordinates
  - `ys::Vector{Float64}`: Poloidal coordinates (radians). For the spline path this is the closed
    grid `[0, 2π]` with `ntheta = mtheta+1` points (last point duplicates θ=0), matching the
    equilibrium `rzphi` θ-knots.
  - `fs::Array{Float64,3}`: Function values (npsi × ntheta × nqty)
  - `mband::Int`: Number of Fourier modes to retain (0:mband)
  - `fft_flag::Bool=true`: select FFT (`true`) or spline-quadrature (`false`) method
"""
function FourierCoefficients(xs::Vector{Float64}, ys::Vector{Float64},
    fs::Array{Float64,3}, mband::Int; fft_flag::Bool=true)
    npsi, ntheta, nqty = size(fs)
    @assert length(xs) == npsi "xs length must match first dimension of fs"
    @assert length(ys) == ntheta "ys length must match second dimension of fs"
    @assert mband >= 0 "mband must be non-negative"
    return fft_flag ? _ffit_fft(xs, ys, fs, mband) : _ffit_spline(xs, ys, fs, mband)
end

# --- FFT method (Fortran fspline_fit_2): bare DFT of the ntheta samples ---
function _ffit_fft(xs::Vector{Float64}, ys::Vector{Float64}, fs::Array{Float64,3}, mband::Int)
    npsi, ntheta, nqty = size(fs)

    # Clamp mband to Nyquist limit
    nyquist_limit = ntheta ÷ 2
    actual_mband = min(mband, nyquist_limit)
    nmodes = actual_mband + 1

    # Compute Fourier coefficients using batched FFT
    fs_reshaped = reshape(permutedims(fs, (2, 1, 3)), ntheta, npsi * nqty)
    fft_results = fft(fs_reshaped, 1)

    cos_coeffs = zeros(Float64, npsi, nmodes, nqty)
    sin_coeffs = zeros(Float64, npsi, nmodes, nqty)
    for iq in 1:nqty
        for ipsi in 1:npsi
            col_idx = (iq - 1) * npsi + ipsi
            fft_col = fft_results[:, col_idx]
            @inbounds cos_coeffs[ipsi, 1, iq] = real(fft_col[1]) / ntheta
            @inbounds for m in 1:actual_mband
                cos_coeffs[ipsi, m+1, iq] = 2 * real(fft_col[m+1]) / ntheta
                sin_coeffs[ipsi, m+1, iq] = -2 * imag(fft_col[m+1]) / ntheta
            end
        end
    end
    FourierCoefficients(xs, actual_mband, nqty, cos_coeffs, sin_coeffs)
end

"""
    _periodic_nodal_derivs(ys, fs) -> Array{Float64,3}

Nodal first derivatives ∂f/∂θ of the periodic cubic spline through each θ-column of `fs`
(npsi × ntheta × nqty), returned with the same shape. `ys` is the closed θ grid (ntheta points,
ys[1]=0, ys[end]=2π). Mirrors Fortran `spline_fit(...,"periodic")` → `fs1` in `fspline_fit_1`.
"""
function _periodic_nodal_derivs(ys::Vector{Float64}, fs::Array{Float64,3})
    npsi, ntheta, nqty = size(fs)
    my = ntheta - 1                                       # intervals; drop the duplicated endpoint
    fsy = similar(fs)
    yopen = ys[1:my]                                      # open grid [0, 2π), my points
    period = ys[ntheta] - ys[1]                           # closed grid spans exactly one period (=2π)
    # endpoint=:exclusive: data is not repeated, so we avoid the bit-identical-endpoint requirement.
    bc = PeriodicBC(; endpoint=:exclusive, period=period)
    for iq in 1:nqty
        for ipsi in 1:npsi
            fcol = collect(@view fs[ipsi, 1:my, iq])
            d1 = deriv1(cubic_interp(yopen, fcol; bc=bc))
            @views fsy[ipsi, 1:my, iq] .= d1.(yopen)
            fsy[ipsi, ntheta, iq] = fsy[ipsi, 1, iq]                 # periodic wrap to θ=2π node
        end
    end
    return fsy
end

# --- Spline-quadrature method (faithful port of Fortran equil/fspline.f fspline_fit_1) ---
# Integrates the periodic cubic spline of each θ-column analytically against exp(-imθ).
# Convention c_m = (1/2π)∫ f e^{-imθ}dθ, identical to the FFT path. Alias-free, mtheta-robust.
function _ffit_spline(xs::Vector{Float64}, ys::Vector{Float64}, fs::Array{Float64,3}, mband::Int)
    npsi, ntheta, nqty = size(fs)
    my = ntheta - 1                       # number of θ intervals (ys is the closed grid 0..2π)
    @assert my >= 3 "spline Fourier fit needs at least 4 θ points"
    nmodes = mband + 1
    twopi = 2π
    eps = 1e-3                             # small-(m·Δθ) Taylor cutoff (matches Fortran)

    cos_coeffs = zeros(Float64, npsi, nmodes, nqty)
    sin_coeffs = zeros(Float64, npsi, nmodes, nqty)

    delta = ys[2:ntheta] .- ys[1:ntheta-1]            # interval widths, length my
    d4fac = 1.0 ./ delta .^ 4
    fsy = _periodic_nodal_derivs(ys, fs)              # nodal ∂f/∂θ of the periodic spline

    expfac0 = cis.(-ys)                               # exp(-i ys), length ntheta
    dexpfac0 = expfac0[1:my] ./ expfac0[2:ntheta]     # exp(i·delta), length my
    E = ones(ComplexF64, ntheta)                      # = exp(-i m ys) during iteration m
    De = ones(ComplexF64, my)                         # = exp(i m delta) during iteration m
    a1 = Vector{ComplexF64}(undef, my)
    b1 = Vector{ComplexF64}(undef, my)

    for mm in 0:mband
        @inbounds for j in 1:my
            x = mm * delta[j]
            if abs(x) > eps
                de = De[j]
                a = (de * (12 - im * x * (6 + x^2)) - 6 * (2 + im * x)) * d4fac[j] / mm^4
                b = (de * (6 - x * (4im + x)) - 2 * (3 + im * x)) * d4fac[j] / mm^4
            else
                a = 0.5 + 7im * x / 20 - 2 * x^2 / 15
                b = 1 / 12 + im * x / 20 - x^2 / 60
            end
            a1[j] = a * delta[j] / twopi
            b1[j] = b * delta[j]^2 / twopi
        end
        @inbounds for iq in 1:nqty
            for ipsi in 1:npsi
                c = zero(ComplexF64)
                for j in 1:my
                    Er = E[j+1]; El = E[j]
                    c += a1[j] * Er * fs[ipsi, j, iq] + conj(a1[j]) * El * fs[ipsi, j+1, iq] +
                         b1[j] * Er * fsy[ipsi, j, iq] - conj(b1[j]) * El * fsy[ipsi, j+1, iq]
                end
                if mm == 0
                    cos_coeffs[ipsi, 1, iq] = real(c)
                else
                    cos_coeffs[ipsi, mm+1, iq] = 2 * real(c)
                    sin_coeffs[ipsi, mm+1, iq] = -2 * imag(c)
                end
            end
        end
        E .*= expfac0
        De .*= dexpfac0
    end
    FourierCoefficients(xs, mband, nqty, cos_coeffs, sin_coeffs)
end

"""
    get_complex_coeff(fc::FourierCoefficients, ipsi, mode, qty) -> ComplexF64

Get normalized complex FFT coefficient at grid point.

The FourierCoefficients internally stores:

  - `cos_coeffs = 2 * real(FFT) / ntheta` (for m > 0)
  - `sin_coeffs = -2 * imag(FFT) / ntheta` (for m > 0)

This function returns the normalized FFT coefficient (FFT/ntheta):

  - `c[0] = cos_coeffs[0]` (DC component)
  - `c[m] = cos_coeffs[m]/2 - i*sin_coeffs[m]/2` (for m > 0)
"""
function get_complex_coeff(fc::FourierCoefficients, ipsi::Int, mode::Int, qty::Int)
    @boundscheck begin
        @assert 1 <= ipsi <= length(fc.xs) "ipsi out of bounds"
        @assert 0 <= mode <= fc.mband "mode out of bounds"
        @assert 1 <= qty <= fc.nqty "qty out of bounds"
    end
    if mode == 0
        return complex(fc.cos_coeffs[ipsi, 1, qty], 0.0)
    else
        return complex(fc.cos_coeffs[ipsi, mode+1, qty] / 2,
            -fc.sin_coeffs[ipsi, mode+1, qty] / 2)
    end
end

"""
    get_complex_coeffs!(out, fc::FourierCoefficients, ipsi, qty)

Fill vector with normalized complex FFT coefficients for modes 0:mband.
"""
function get_complex_coeffs!(out::AbstractVector{ComplexF64}, fc::FourierCoefficients,
    ipsi::Int, qty::Int)
    nmodes = fc.mband + 1
    @assert length(out) >= nmodes "output vector too short"

    @inbounds out[1] = complex(fc.cos_coeffs[ipsi, 1, qty], 0.0)
    @inbounds for m in 1:fc.mband
        out[m+1] = complex(fc.cos_coeffs[ipsi, m+1, qty] / 2,
            -fc.sin_coeffs[ipsi, m+1, qty] / 2)
    end
    return out
end
