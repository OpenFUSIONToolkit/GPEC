"""
FourierModeSplines - Fourier decomposition with 1D splines per mode

Replaces the Fortran-backed FourierSpline with explicit FFT decomposition
and individual cubic splines for each Fourier mode in the radial direction.

This approach:
1. Applies FFT to each radial slice to get Fourier coefficients
2. Creates 1D splines for each (mode, quantity) combination
3. Evaluates by summing Fourier modes at the query point

# Performance Notes

Trigonometric recurrence is used for Fourier mode evaluation to avoid
repeated calls to cos/sin. This uses:
    cos(m·θ) = 2·cos(θ)·cos((m-1)·θ) - cos((m-2)·θ)
    sin(m·θ) = 2·cos(θ)·sin((m-1)·θ) - sin((m-2)·θ)

This is numerically stable for reasonable mode numbers (mband < 100).

# Thread Safety Warning

The evaluation functions (`evaluate!`, `deriv1!`, `deriv2!`) mutate internal work
arrays and are NOT thread-safe. For multi-threaded usage:
- Use separate FourierModeSplines instances per thread

# References

- FFT normalization: FFTW convention with DFT of length N
- Trigonometric recurrence: Numerical Recipes, Press et al. (1992), Sec. 5.5
"""

using FFTW

# Import from SplineAdapter (will be in same module)
# CubicSpline1D is defined in SplineAdapter.jl

"""
    FourierModeSplines{S}

Fourier-based 2D interpolation: cubic in x (radial), Fourier series in y (poloidal).

# Type Parameters

  - `S`: Type of the radial splines (for type stability)

# Fields

  - `xs::Vector{Float64}`: Radial coordinates (psi)
  - `ys::Vector{Float64}`: Poloidal coordinates (theta)
  - `period::Float64`: Angular period (typically 2π or 1.0 for normalized coordinates)
  - `mband::Int`: Number of Fourier modes (0 to mband inclusive)
  - `nqty::Int`: Number of quantities
  - `cos_coeffs::Array{Float64,3}`: Cosine coefficients (npsi × nmodes × nqty)
  - `sin_coeffs::Array{Float64,3}`: Sine coefficients (npsi × nmodes × nqty)
  - `cos_splines::Matrix{S}`: Splines for cosine coefficients [mode, qty]
  - `sin_splines::Matrix{S}`: Splines for sine coefficients [mode, qty]
"""
struct FourierModeSplines{S}
    xs::Vector{Float64}
    ys::Vector{Float64}
    period::Float64
    mband::Int
    nqty::Int
    cos_coeffs::Array{Float64,3}
    sin_coeffs::Array{Float64,3}
    cos_splines::Matrix{S}
    sin_splines::Matrix{S}
    # Work arrays
    _f::Vector{Float64}
    _fx::Vector{Float64}
    _fy::Vector{Float64}
    _fxx::Vector{Float64}
    _fxy::Vector{Float64}
    _fyy::Vector{Float64}
end

"""
    FourierModeSplines(xs, ys, fs, mband; bctype="extrap", period=nothing)

Create a Fourier-based 2D interpolator.

# Arguments

  - `xs::Vector{Float64}`: Radial coordinates (sorted)
  - `ys::Vector{Float64}`: Poloidal coordinates (sorted, periodic domain)
  - `fs::Array{Float64,3}`: Function values (npsi × ntheta × nqty)
  - `mband::Int`: Number of Fourier modes to retain (0 to mband inclusive)
  - `bctype`: Boundary condition for radial splines ("natural", "periodic", "extrap")
  - `period`: Angular period. If nothing, auto-detected from ys (2π if max > 2, else 1.0)

# Bounds Checking

  - `mband` must be ≤ ntheta ÷ 2 (Nyquist limit)
  - Larger mband values will be silently clamped to the Nyquist limit

# Example

```julia
xs = range(0, 1; length=100) |> collect
ys = range(0, 2π; length=64) |> collect  # 64 poloidal points
fs = zeros(100, 64, 2)
for i in 1:100, j in 1:64
    fs[i, j, 1] = exp(-xs[i]) * cos(3*ys[j])
    fs[i, j, 2] = exp(-xs[i]) * sin(3*ys[j])
end
fms = FourierModeSplines(xs, ys, fs, 10)  # Keep 10 modes
f, fx, fy = deriv1!(fms, 0.5, π/4)
```
"""
function FourierModeSplines(xs::Vector{Float64}, ys::Vector{Float64},
    fs::Array{Float64,3}, mband::Int;
    bctype::String="extrap", period::Union{Nothing,Float64}=nothing)
    npsi, ntheta, nqty = size(fs)

    @assert length(xs) == npsi "xs length must match first dimension of fs"
    @assert length(ys) == ntheta "ys length must match second dimension of fs"
    @assert mband >= 0 "mband must be non-negative"

    # Clamp mband to Nyquist limit
    nyquist_limit = ntheta ÷ 2
    actual_mband = min(mband, nyquist_limit)
    if actual_mband < mband
        @warn "mband=$mband exceeds Nyquist limit for ntheta=$ntheta. Using mband=$actual_mband"
    end

    nmodes = actual_mband + 1  # modes 0, 1, ..., mband

    # Determine the angular period
    if period === nothing
        # Auto-detect: if ys[end] < 2.0, assume normalized [0, 1)
        period = ys[end] < 2.0 ? 1.0 : 2π
    end

    # Compute Fourier coefficients using batched FFT
    # Reshape for batched operation: (ntheta, npsi * nqty)
    fs_reshaped = reshape(permutedims(fs, (2, 1, 3)), ntheta, npsi * nqty)

    # Apply batched FFT along first dimension
    fft_results = fft(fs_reshaped, 1)

    # Extract and normalize Fourier coefficients
    cos_coeffs = zeros(Float64, npsi, nmodes, nqty)
    sin_coeffs = zeros(Float64, npsi, nmodes, nqty)

    for iq in 1:nqty
        for ipsi in 1:npsi
            col_idx = (iq - 1) * npsi + ipsi
            fft_col = fft_results[:, col_idx]

            # Mode 0 (DC component)
            @inbounds cos_coeffs[ipsi, 1, iq] = real(fft_col[1]) / ntheta

            # Higher modes
            @inbounds for m in 1:actual_mband
                if m < nyquist_limit + 1
                    cos_coeffs[ipsi, m+1, iq] = 2 * real(fft_col[m+1]) / ntheta
                    sin_coeffs[ipsi, m+1, iq] = -2 * imag(fft_col[m+1]) / ntheta
                end
            end
        end
    end

    # Create 1D splines for each Fourier coefficient in the radial direction
    # First, create a template spline to get the type
    template_spline = CubicSpline1D(xs, cos_coeffs[:, 1, 1]; bctype=bctype)
    SplineType = typeof(template_spline)

    cos_splines = Matrix{SplineType}(undef, nmodes, nqty)
    sin_splines = Matrix{SplineType}(undef, nmodes, nqty)

    for m in 1:nmodes
        for iq in 1:nqty
            cos_splines[m, iq] = CubicSpline1D(xs, cos_coeffs[:, m, iq]; bctype=bctype)
            sin_splines[m, iq] = CubicSpline1D(xs, sin_coeffs[:, m, iq]; bctype=bctype)
        end
    end

    # Work arrays
    _f = zeros(Float64, nqty)
    _fx = zeros(Float64, nqty)
    _fy = zeros(Float64, nqty)
    _fxx = zeros(Float64, nqty)
    _fxy = zeros(Float64, nqty)
    _fyy = zeros(Float64, nqty)

    FourierModeSplines{SplineType}(xs, ys, period, actual_mband, nqty,
        cos_coeffs, sin_coeffs,
        cos_splines, sin_splines,
        _f, _fx, _fy, _fxx, _fxy, _fyy)
end

"""
    _compute_trig_values(mband, omega, y)

Compute cos(m·ω·y) and sin(m·ω·y) for m = 0, 1, ..., mband using trigonometric recurrence.

Returns (cos_vals, sin_vals) where each is a Vector{Float64} of length mband+1.
"""
function _compute_trig_values(mband::Int, omega::Float64, y::Float64)
    nmodes = mband + 1
    cos_vals = Vector{Float64}(undef, nmodes)
    sin_vals = Vector{Float64}(undef, nmodes)

    # Base cases
    theta = omega * y
    c1 = cos(theta)
    s1 = sin(theta)
    two_c1 = 2.0 * c1

    @inbounds cos_vals[1] = 1.0  # cos(0) = 1
    @inbounds sin_vals[1] = 0.0  # sin(0) = 0

    if mband >= 1
        @inbounds cos_vals[2] = c1
        @inbounds sin_vals[2] = s1
    end

    # Recurrence: cos(m·θ) = 2·cos(θ)·cos((m-1)·θ) - cos((m-2)·θ)
    #             sin(m·θ) = 2·cos(θ)·sin((m-1)·θ) - sin((m-2)·θ)
    @inbounds for m in 2:mband
        cos_vals[m+1] = muladd(two_c1, cos_vals[m], -cos_vals[m-1])
        sin_vals[m+1] = muladd(two_c1, sin_vals[m], -sin_vals[m-1])
    end

    return cos_vals, sin_vals
end

"""
    evaluate!(fms, x, y) -> Vector{Float64}

Evaluate the Fourier-mode spline at point (x, y).

Uses trigonometric recurrence for efficient Fourier mode computation.

Thread-safety: NOT thread-safe. Use separate instances per thread.
"""
function evaluate!(fms::FourierModeSplines{S}, x::Float64, y::Float64) where {S}
    fill!(fms._f, 0.0)

    # Angular frequency
    omega = 2π / fms.period

    # Compute all trig values using recurrence
    cos_vals, sin_vals = _compute_trig_values(fms.mband, omega, y)

    # Sum over Fourier modes
    @inbounds for m in 0:fms.mband
        cos_my = cos_vals[m+1]
        sin_my = sin_vals[m+1]

        for iq in 1:fms.nqty
            cm = evaluate!(fms.cos_splines[m+1, iq], x)[1]
            sm = evaluate!(fms.sin_splines[m+1, iq], x)[1]
            fms._f[iq] = muladd(cm, cos_my, muladd(sm, sin_my, fms._f[iq]))
        end
    end

    return fms._f
end

"""
    deriv1!(fms, x, y) -> (f, fx, fy)

Evaluate Fourier-mode spline and first derivatives at point (x, y).

Uses trigonometric recurrence for efficient Fourier mode computation.

Thread-safety: NOT thread-safe. Use separate instances per thread.
"""
function deriv1!(fms::FourierModeSplines{S}, x::Float64, y::Float64) where {S}
    fill!(fms._f, 0.0)
    fill!(fms._fx, 0.0)
    fill!(fms._fy, 0.0)

    omega = 2π / fms.period
    cos_vals, sin_vals = _compute_trig_values(fms.mband, omega, y)

    @inbounds for m in 0:fms.mband
        cos_my = cos_vals[m+1]
        sin_my = sin_vals[m+1]
        # d/dy[cos(m·ω·y)] = -m·ω·sin(m·ω·y)
        # d/dy[sin(m·ω·y)] = m·ω·cos(m·ω·y)
        m_omega = m * omega
        dcos_dy = -m_omega * sin_my
        dsin_dy = m_omega * cos_my

        for iq in 1:fms.nqty
            f_cm, f1_cm = deriv1!(fms.cos_splines[m+1, iq], x)
            f_sm, f1_sm = deriv1!(fms.sin_splines[m+1, iq], x)

            cm = f_cm[1]
            sm = f_sm[1]
            dcm_dx = f1_cm[1]
            dsm_dx = f1_sm[1]

            fms._f[iq] = muladd(cm, cos_my, muladd(sm, sin_my, fms._f[iq]))
            fms._fx[iq] = muladd(dcm_dx, cos_my, muladd(dsm_dx, sin_my, fms._fx[iq]))
            fms._fy[iq] = muladd(cm, dcos_dy, muladd(sm, dsin_dy, fms._fy[iq]))
        end
    end

    return fms._f, fms._fx, fms._fy
end

"""
    deriv2!(fms, x, y) -> (f, fx, fy, fxx, fxy, fyy)

Evaluate Fourier-mode spline through second derivatives at point (x, y).
Includes cross-derivative fxy.

Uses trigonometric recurrence for efficient Fourier mode computation.

Thread-safety: NOT thread-safe. Use separate instances per thread.
"""
function deriv2!(fms::FourierModeSplines{S}, x::Float64, y::Float64) where {S}
    fill!(fms._f, 0.0)
    fill!(fms._fx, 0.0)
    fill!(fms._fy, 0.0)
    fill!(fms._fxx, 0.0)
    fill!(fms._fxy, 0.0)
    fill!(fms._fyy, 0.0)

    omega = 2π / fms.period
    cos_vals, sin_vals = _compute_trig_values(fms.mband, omega, y)

    @inbounds for m in 0:fms.mband
        cos_my = cos_vals[m+1]
        sin_my = sin_vals[m+1]
        m_omega = m * omega
        m_omega_sq = m_omega * m_omega

        # First derivatives
        dcos_dy = -m_omega * sin_my
        dsin_dy = m_omega * cos_my

        # Second derivatives
        d2cos_dy2 = -m_omega_sq * cos_my
        d2sin_dy2 = -m_omega_sq * sin_my

        for iq in 1:fms.nqty
            f_cm, f1_cm, f2_cm = deriv2!(fms.cos_splines[m+1, iq], x)
            f_sm, f1_sm, f2_sm = deriv2!(fms.sin_splines[m+1, iq], x)

            cm = f_cm[1]
            sm = f_sm[1]
            dcm_dx = f1_cm[1]
            dsm_dx = f1_sm[1]
            d2cm_dx2 = f2_cm[1]
            d2sm_dx2 = f2_sm[1]

            # f = Σ [cm·cos(m·ω·y) + sm·sin(m·ω·y)]
            fms._f[iq] = muladd(cm, cos_my, muladd(sm, sin_my, fms._f[iq]))

            # fx = Σ [dcm/dx·cos(m·ω·y) + dsm/dx·sin(m·ω·y)]
            fms._fx[iq] = muladd(dcm_dx, cos_my, muladd(dsm_dx, sin_my, fms._fx[iq]))

            # fy = Σ [cm·dcos/dy + sm·dsin/dy]
            fms._fy[iq] = muladd(cm, dcos_dy, muladd(sm, dsin_dy, fms._fy[iq]))

            # fxx = Σ [d²cm/dx²·cos(m·ω·y) + d²sm/dx²·sin(m·ω·y)]
            fms._fxx[iq] = muladd(d2cm_dx2, cos_my, muladd(d2sm_dx2, sin_my, fms._fxx[iq]))

            # fxy = Σ [dcm/dx·dcos/dy + dsm/dx·dsin/dy]
            fms._fxy[iq] = muladd(dcm_dx, dcos_dy, muladd(dsm_dx, dsin_dy, fms._fxy[iq]))

            # fyy = Σ [cm·d²cos/dy² + sm·d²sin/dy²]
            fms._fyy[iq] = muladd(cm, d2cos_dy2, muladd(sm, d2sin_dy2, fms._fyy[iq]))
        end
    end

    return fms._f, fms._fx, fms._fy, fms._fxx, fms._fxy, fms._fyy
end

"""
    evaluate(fms, xs_eval, ys_eval) -> Array{Float64,3}

Vectorized evaluation at multiple points.
Returns array of shape (length(xs_eval), length(ys_eval), nqty).

Note: This allocates and is thread-safe.
"""
function evaluate(fms::FourierModeSplines, xs_eval::Vector{Float64}, ys_eval::Vector{Float64})
    nx = length(xs_eval)
    ny = length(ys_eval)
    result = zeros(Float64, nx, ny, fms.nqty)

    for (ix, x) in enumerate(xs_eval)
        for (iy, y) in enumerate(ys_eval)
            result[ix, iy, :] .= evaluate!(fms, x, y)
        end
    end

    return result
end

"""
    empty_FourierModeSplines()

Create an empty/placeholder FourierModeSplines for type stability.
"""
function empty_FourierModeSplines()
    xs = Float64[0.0, 1.0]
    ys = Float64[0.0, 0.5]
    fs = zeros(Float64, 2, 2, 1)
    FourierModeSplines(xs, ys, fs, 1)
end

# Convenience evaluation
(fms::FourierModeSplines)(x::Float64, y::Float64) = evaluate!(fms, x, y)

"""
    get_complex_coeff(fms, ipsi, mode, qty) -> ComplexF64

Get the complex Fourier coefficient for a specific grid point, mode, and quantity.

For real-valued signals, the complex DFT coefficient is:

  - c_0 = a_0 (DC component, purely real)
  - c_m = (a_m - i*b_m) / 2 for m > 0

This matches the normalization: f(θ) = Σ_m c_m * exp(i*m*θ)

# Arguments

  - `fms`: FourierModeSplines object
  - `ipsi`: Radial grid index (1-based)
  - `mode`: Fourier mode number (0, 1, ..., mband)
  - `qty`: Quantity index (1-based)
"""
function get_complex_coeff(fms::FourierModeSplines, ipsi::Int, mode::Int, qty::Int)
    @boundscheck begin
        @assert 1 <= ipsi <= length(fms.xs) "ipsi out of bounds"
        @assert 0 <= mode <= fms.mband "mode out of bounds"
        @assert 1 <= qty <= fms.nqty "qty out of bounds"
    end
    if mode == 0
        # DC component is purely real
        return complex(fms.cos_coeffs[ipsi, 1, qty], 0.0)
    else
        # c_m = (a_m - i*b_m) / 2 = cos/2 + i*sin/2 (since b_m = -2*Im(c_m))
        return complex(fms.cos_coeffs[ipsi, mode+1, qty] / 2,
            fms.sin_coeffs[ipsi, mode+1, qty] / 2)
    end
end

"""
    get_complex_coeffs!(out, fms, ipsi, qty)

Fill vector `out` with complex Fourier coefficients for modes 0:mband.

The output has length mband+1, with out[k+1] = coefficient for mode k.
"""
function get_complex_coeffs!(out::AbstractVector{ComplexF64}, fms::FourierModeSplines,
    ipsi::Int, qty::Int)
    nmodes = fms.mband + 1
    @assert length(out) >= nmodes "output vector too short"

    # DC component
    @inbounds out[1] = complex(fms.cos_coeffs[ipsi, 1, qty], 0.0)

    # Higher modes
    @inbounds for m in 1:fms.mband
        out[m+1] = complex(fms.cos_coeffs[ipsi, m+1, qty] / 2,
            fms.sin_coeffs[ipsi, m+1, qty] / 2)
    end

    return out
end
