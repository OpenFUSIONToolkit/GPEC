"""
FourierModeSplines - Fourier decomposition with 1D splines per mode

Replaces the Fortran-backed FourierSpline with explicit FFT decomposition
and individual cubic splines for each Fourier mode in the radial direction.

This approach:
1. Applies FFT to each radial slice to get Fourier coefficients
2. Creates 1D splines for each (mode, quantity) combination
3. Evaluates by summing Fourier modes at the query point
"""

using FFTW

# Import from SplineAdapter (will be in same module)
# CubicSpline1D is defined in SplineAdapter.jl

"""
    FourierModeSplines{S}

Fourier-based 2D interpolation: cubic in x (radial), Fourier series in y (poloidal).

# Fields

  - `xs::Vector{Float64}`: Radial coordinates (psi)
  - `ys::Vector{Float64}`: Poloidal coordinates (theta), assumed [0, 2π) or normalized
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
end

"""
    FourierModeSplines(xs, ys, fs, mband; bctype="extrap")

Create a Fourier-based 2D interpolator.

# Arguments

  - `xs::Vector{Float64}`: Radial coordinates (sorted)
  - `ys::Vector{Float64}`: Poloidal coordinates (sorted, periodic domain)
  - `fs::Array{Float64,3}`: Function values (npsi × ntheta × nqty)
  - `mband::Int`: Number of Fourier modes to retain
  - `bctype`: Boundary condition for radial splines
"""
function FourierModeSplines(xs::Vector{Float64}, ys::Vector{Float64},
    fs::Array{Float64,3}, mband::Int;
    bctype::String="extrap")
    npsi, ntheta, nqty = size(fs)
    nmodes = mband + 1  # modes 0, 1, ..., mband

    @assert length(xs) == npsi "xs length must match first dimension of fs"
    @assert length(ys) == ntheta "ys length must match second dimension of fs"

    # Compute Fourier coefficients at each radial location
    cos_coeffs = zeros(Float64, npsi, nmodes, nqty)
    sin_coeffs = zeros(Float64, npsi, nmodes, nqty)

    # Determine the angular period (assume ys spans [0, period))
    period = 2π  # Default assumption
    if length(ys) > 1
        # Check if ys is normalized [0, 1] or [0, 2π]
        if ys[end] < 2.0
            period = 1.0
        end
    end

    for ipsi in 1:npsi
        for iq in 1:nqty
            # Get the poloidal slice
            f_theta = fs[ipsi, :, iq]

            # Use FFT to compute Fourier coefficients
            fft_result = fft(f_theta)

            # Extract coefficients (normalized)
            # Mode 0 (DC component)
            cos_coeffs[ipsi, 1, iq] = real(fft_result[1]) / ntheta

            # Higher modes
            for m in 1:mband
                if m < ntheta ÷ 2 + 1
                    # Positive frequency
                    cos_coeffs[ipsi, m+1, iq] = 2 * real(fft_result[m+1]) / ntheta
                    sin_coeffs[ipsi, m+1, iq] = -2 * imag(fft_result[m+1]) / ntheta
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

    FourierModeSplines{SplineType}(xs, ys, mband, nqty,
        cos_coeffs, sin_coeffs,
        cos_splines, sin_splines,
        _f, _fx, _fy)
end

"""
    evaluate!(fms, x, y) -> Vector{Float64}

Evaluate the Fourier-mode spline at point (x, y).
"""
function evaluate!(fms::FourierModeSplines, x::Float64, y::Float64)
    fill!(fms._f, 0.0)

    # Sum over Fourier modes
    for m in 0:fms.mband
        cos_my = cos(m * 2π * y)  # Assuming y is normalized to [0, 1]
        sin_my = sin(m * 2π * y)

        for iq in 1:fms.nqty
            cm = evaluate!(fms.cos_splines[m+1, iq], x)[1]
            sm = evaluate!(fms.sin_splines[m+1, iq], x)[1]
            fms._f[iq] += cm * cos_my + sm * sin_my
        end
    end

    return fms._f
end

"""
    deriv1!(fms, x, y) -> (f, fx, fy)

Evaluate Fourier-mode spline and first derivatives at point (x, y).
"""
function deriv1!(fms::FourierModeSplines, x::Float64, y::Float64)
    fill!(fms._f, 0.0)
    fill!(fms._fx, 0.0)
    fill!(fms._fy, 0.0)

    for m in 0:fms.mband
        cos_my = cos(m * 2π * y)
        sin_my = sin(m * 2π * y)
        dcos_dy = -m * 2π * sin_my
        dsin_dy = m * 2π * cos_my

        for iq in 1:fms.nqty
            f_cm, f1_cm = deriv1!(fms.cos_splines[m+1, iq], x)
            f_sm, f1_sm = deriv1!(fms.sin_splines[m+1, iq], x)

            cm = f_cm[1]
            sm = f_sm[1]
            dcm_dx = f1_cm[1]
            dsm_dx = f1_sm[1]

            fms._f[iq] += cm * cos_my + sm * sin_my
            fms._fx[iq] += dcm_dx * cos_my + dsm_dx * sin_my
            fms._fy[iq] += cm * dcos_dy + sm * dsin_dy
        end
    end

    return fms._f, fms._fx, fms._fy
end

"""
    evaluate(fms, xs_eval, ys_eval) -> Array{Float64,3}

Vectorized evaluation at multiple points.
Returns array of shape (length(xs_eval), length(ys_eval), nqty).
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
