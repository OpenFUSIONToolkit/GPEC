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

"""
    FourierCoefficients

Lightweight container for Fourier coefficients without spline interpolation.
Use this when you only need to access coefficients at original grid points.

# Fields

  - `xs::Vector{Float64}`: Radial coordinates
  - `mmax::Int`: Highest retained poloidal Fourier mode (modes 0:mmax inclusive)
  - `nqty::Int`: Number of quantities
  - `cos_coeffs::Array{Float64,3}`: Cosine coefficients (npsi × nmodes × nqty)
  - `sin_coeffs::Array{Float64,3}`: Sine coefficients (npsi × nmodes × nqty)
"""
struct FourierCoefficients
    xs::Vector{Float64}
    mmax::Int
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
    FourierCoefficients(xs, ys, fs, mmax)

Compute Fourier coefficients via FFT without creating splines.

# Arguments

  - `xs::Vector{Float64}`: Radial coordinates
  - `ys::Vector{Float64}`: Poloidal coordinates (periodic domain)
  - `fs::Array{Float64,3}`: Function values (npsi × ntheta × nqty)
  - `mmax::Int`: Highest poloidal Fourier mode to retain
"""
function FourierCoefficients(xs::Vector{Float64}, ys::Vector{Float64},
    fs::Array{Float64,3}, mmax::Int)
    npsi, ny_full, nqty = size(fs)

    @assert length(xs) == npsi "xs length must match first dimension of fs"
    @assert length(ys) == ny_full "ys length must match second dimension of fs"
    @assert mmax >= 0 "mmax must be non-negative"

    # Drop periodic-duplicate endpoint before FFT to match Fortran fspline_fit_2
    # (equil/fspline.f:293 uses `f = fst%fs(:, 0:my-1, iq)`). The equilibrium θ-grids
    # in this codebase store θ=0 and θ=2π as both endpoints (length mtheta+1);
    # including the duplicate biases the DC coefficient by ~(f(0) − mean)/N.
    has_duplicate = ny_full > 1 && isapprox(ys[end] - ys[1], 2π; rtol=1e-10)
    ntheta = has_duplicate ? ny_full - 1 : ny_full
    fs_view = has_duplicate ? view(fs, :, (1:ntheta), :) : fs

    @assert mmax <= ntheta ÷ 2 "Requested mmax=$mmax exceeds the θ-grid Nyquist limit $(ntheta ÷ 2) (ntheta=$ntheta); increase the poloidal grid resolution or reduce mpert."

    # Compute Fourier coefficients using batched FFT
    fs_reshaped = reshape(permutedims(fs_view, (2, 1, 3)), ntheta, npsi * nqty)
    fft_results = fft(fs_reshaped, 1)

    # Extract and normalize coefficients
    cos_coeffs = zeros(Float64, npsi, mmax + 1, nqty)
    sin_coeffs = zeros(Float64, npsi, mmax + 1, nqty)

    for iq in 1:nqty
        for ipsi in 1:npsi
            col_idx = (iq - 1) * npsi + ipsi
            fft_col = fft_results[:, col_idx]

            # Mode 0 (DC component)
            @inbounds cos_coeffs[ipsi, 1, iq] = real(fft_col[1]) / ntheta

            # Higher modes
            @inbounds for m in 1:mmax
                cos_coeffs[ipsi, m+1, iq] = 2 * real(fft_col[m+1]) / ntheta
                sin_coeffs[ipsi, m+1, iq] = -2 * imag(fft_col[m+1]) / ntheta
            end
        end
    end

    FourierCoefficients(xs, mmax, nqty, cos_coeffs, sin_coeffs)
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
        @assert 0 <= mode <= fc.mmax "mode out of bounds"
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

Fill vector with normalized complex FFT coefficients for modes 0:mmax.
"""
function get_complex_coeffs!(out::AbstractVector{ComplexF64}, fc::FourierCoefficients,
    ipsi::Int, qty::Int)
    nmodes = fc.mmax + 1
    @assert length(out) >= nmodes "output vector too short"

    @inbounds out[1] = complex(fc.cos_coeffs[ipsi, 1, qty], 0.0)
    @inbounds for m in 1:fc.mmax
        out[m+1] = complex(fc.cos_coeffs[ipsi, m+1, qty] / 2,
            -fc.sin_coeffs[ipsi, m+1, qty] / 2)
    end
    return out
end
