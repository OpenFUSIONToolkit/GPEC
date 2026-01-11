"""
RZPhiSplines - Flux-coordinate mapping using 1D periodic splines in theta

This type replaces the 2D BicubicSpline/BicubicWrapper for rzphi data, leveraging
the fact that rzphi is only ever evaluated at psi grid points. Each flux surface
has its own set of periodic 1D splines in theta.

# Advantages over 2D splines
- Simpler implementation (no 2D interpolation complexity)
- Faster grid-point access (no psi binary search during evaluation)
- Pure Julia implementation
- Matches the physics (each flux surface is independent)

# Thread Safety Warning
The evaluation functions (`evaluate!`, `deriv1!`, `deriv2!`) mutate internal work
arrays and are NOT thread-safe. For multi-threaded usage, create separate instances.
"""

"""
    RZPhiSplines{S<:CubicSpline1D}

Flux-coordinate mapping using 1D periodic splines in theta for each flux surface.

# Type Parameters

  - `S`: Type of the 1D cubic spline (CubicSpline1D with periodic BC)

# Fields

  - `xs::Vector{Float64}`: Psi grid points (normalized flux, length npsi)
  - `ys::Vector{Float64}`: Theta grid points (normalized poloidal angle [0,1], endpoint-inclusive)
  - `nqty::Int`: Number of quantities (4 for rzphi)
  - `npsi::Int`: Number of psi surfaces
  - `ntheta::Int`: Number of theta points (including endpoint)
  - `theta_splines::Matrix{S}`: Shape (npsi, nqty) - 1D periodic splines in theta
  - `psi_deriv_splines::Matrix{S}`: Shape (npsi, nqty) - splines of pre-computed d/dpsi
  - `fs::Array{Float64,3}`: Original data (npsi, ntheta, nqty)
  - `fs_psi::Array{Float64,3}`: Pre-computed d/dpsi via finite differences

# Work Arrays (pre-allocated, not thread-safe)

  - `_f`, `_fx`, `_fy`, `_fxx`, `_fxy`, `_fyy`: Length-nqty output vectors
"""
struct RZPhiSplines{S<:CubicSpline1D}
    xs::Vector{Float64}
    ys::Vector{Float64}
    nqty::Int
    npsi::Int
    ntheta::Int

    theta_splines::Matrix{S}
    psi_deriv_splines::Matrix{S}

    fs::Array{Float64,3}
    fs_psi::Array{Float64,3}

    # Work arrays
    _f::Vector{Float64}
    _fx::Vector{Float64}
    _fy::Vector{Float64}
    _fxx::Vector{Float64}
    _fxy::Vector{Float64}
    _fyy::Vector{Float64}
end

"""
    RZPhiSplines(xs, ys, fs)

Create RZPhiSplines from grid data.

# Arguments

  - `xs::Vector{Float64}`: Psi grid points (length npsi)
  - `ys::Vector{Float64}`: Theta grid points (length ntheta), endpoint-inclusive [0,1]
  - `fs::Array{Float64,3}`: Data array (npsi, ntheta, nqty)

The theta data is assumed to be endpoint-inclusive (both 0 and 1 included),
matching the convention used in DirectEquilibrium.jl. The endpoint is stripped
internally for proper periodic spline construction.
"""
function RZPhiSplines(xs::Vector{Float64}, ys::Vector{Float64}, fs::Array{Float64,3})
    npsi, ntheta, nqty = size(fs)
    @assert length(xs) == npsi "xs length must match first dimension of fs"
    @assert length(ys) == ntheta "ys length must match second dimension of fs"
    @assert ntheta >= 3 "Need at least 3 theta points for periodic splines"

    # Strip endpoint for periodic splines (data is endpoint-inclusive)
    # The period is ys[end] - ys[1] = 1.0 for normalized theta
    ys_internal = ys[1:(end-1)]

    # Pre-compute psi derivatives using finite differences
    fs_psi = _compute_psi_derivatives(xs, fs)

    # Create template spline to get the concrete type
    template = CubicSpline1D(ys_internal, fs[1, 1:(end-1), 1]; bctype="periodic")
    SplineType = typeof(template)

    # Allocate spline matrices
    theta_splines = Matrix{SplineType}(undef, npsi, nqty)
    psi_deriv_splines = Matrix{SplineType}(undef, npsi, nqty)

    # Create 1D periodic splines for each (ipsi, iqty) combination
    for ipsi in 1:npsi
        for iq in 1:nqty
            theta_splines[ipsi, iq] = CubicSpline1D(
                ys_internal, fs[ipsi, 1:(end-1), iq]; bctype="periodic"
            )
            psi_deriv_splines[ipsi, iq] = CubicSpline1D(
                ys_internal, fs_psi[ipsi, 1:(end-1), iq]; bctype="periodic"
            )
        end
    end

    # Allocate work arrays
    _f = zeros(Float64, nqty)
    _fx = zeros(Float64, nqty)
    _fy = zeros(Float64, nqty)
    _fxx = zeros(Float64, nqty)
    _fxy = zeros(Float64, nqty)
    _fyy = zeros(Float64, nqty)

    RZPhiSplines{SplineType}(
        xs, ys, nqty, npsi, ntheta,
        theta_splines, psi_deriv_splines,
        copy(fs), fs_psi,
        _f, _fx, _fy, _fxx, _fxy, _fyy
    )
end

"""
    _compute_psi_derivatives(xs, fs) -> Array{Float64,3}

Pre-compute d/dpsi using 1D spline derivatives.
For each (theta, quantity) combination, fit a cubic spline in psi and
evaluate its derivative at all grid points. This matches the behavior
of Hermite bicubic splines better than finite differences.
"""
function _compute_psi_derivatives(xs::Vector{Float64}, fs::Array{Float64,3})
    npsi, ntheta, nqty = size(fs)
    fs_psi = zeros(Float64, npsi, ntheta, nqty)

    for iq in 1:nqty
        for itheta in 1:ntheta
            # Fit a 1D cubic spline in psi for this (theta, quantity) slice
            psi_data = fs[:, itheta, iq]
            psi_spline = CubicSpline1D(xs, reshape(psi_data, :, 1); bctype="not-a-knot")

            # Evaluate the derivative at each psi grid point
            for ipsi in 1:npsi
                _, f1 = SplinesMod.deriv1!(psi_spline, xs[ipsi])
                fs_psi[ipsi, itheta, iq] = f1[1]
            end
        end
    end

    return fs_psi
end

"""
    _find_psi_index(rzphi, psi) -> Int

Find the psi grid index for a given psi value.
Assumes psi is at or very close to a grid point.
Uses binary search for efficiency.
"""
@inline function _find_psi_index(rzphi::RZPhiSplines, psi::Float64)::Int
    xs = rzphi.xs
    npsi = rzphi.npsi

    # Binary search for closest grid point
    idx = searchsortedfirst(xs, psi)

    if idx > npsi
        return npsi
    elseif idx == 1
        return 1
    else
        # Check which neighbor is closer
        if abs(xs[idx-1] - psi) <= abs(xs[idx] - psi)
            return idx - 1
        else
            return idx
        end
    end
end

"""
    _wrap_theta(rzphi, theta) -> Float64

Wrap theta to the range [ys[1], ys[end]) for periodic evaluation.
"""
@inline function _wrap_theta(rzphi::RZPhiSplines, theta::Float64)::Float64
    y_min = rzphi.ys[1]
    y_period = rzphi.ys[end] - rzphi.ys[1]

    if y_period <= 0
        return theta
    end

    y_shifted = theta - y_min
    y_wrapped = mod(y_shifted, y_period)
    return y_wrapped + y_min
end

# =============================================================================
# Evaluation Methods
# =============================================================================

"""
    evaluate!(rzphi, psi, theta) -> Vector{Float64}

Evaluate at (psi, theta). Psi must be at a grid point.
Returns reference to internal work array (not thread-safe).
"""
function evaluate!(rzphi::RZPhiSplines{S}, psi::Float64, theta::Float64) where {S}
    ipsi = _find_psi_index(rzphi, psi)
    theta_wrapped = _wrap_theta(rzphi, theta)

    @inbounds for iq in 1:rzphi.nqty
        f_val = SplinesMod.evaluate!(rzphi.theta_splines[ipsi, iq], theta_wrapped)
        rzphi._f[iq] = f_val[1]
    end

    return rzphi._f
end

"""
    deriv1!(rzphi, psi, theta) -> (f, fx, fy)

Evaluate value and first derivatives at (psi, theta). Psi must be at a grid point.
Returns references to internal work arrays (not thread-safe).
"""
function deriv1!(rzphi::RZPhiSplines{S}, psi::Float64, theta::Float64) where {S}
    ipsi = _find_psi_index(rzphi, psi)
    theta_wrapped = _wrap_theta(rzphi, theta)

    @inbounds for iq in 1:rzphi.nqty
        # Get f and fy from theta spline
        f_val, f1_val = SplinesMod.deriv1!(rzphi.theta_splines[ipsi, iq], theta_wrapped)
        rzphi._f[iq] = f_val[1]
        rzphi._fy[iq] = f1_val[1]

        # Get fx from pre-computed psi derivative spline
        fx_val = SplinesMod.evaluate!(rzphi.psi_deriv_splines[ipsi, iq], theta_wrapped)
        rzphi._fx[iq] = fx_val[1]
    end

    return rzphi._f, rzphi._fx, rzphi._fy
end

"""
    deriv2!(rzphi, psi, theta) -> (f, fx, fy, fxx, fxy, fyy)

Evaluate through second derivatives at (psi, theta). Psi must be at a grid point.
Returns references to internal work arrays (not thread-safe).

Note: fxx (second psi derivative) is computed from pre-computed first psi derivatives
using finite differences between neighboring surfaces.
"""
function deriv2!(rzphi::RZPhiSplines{S}, psi::Float64, theta::Float64) where {S}
    ipsi = _find_psi_index(rzphi, psi)
    theta_wrapped = _wrap_theta(rzphi, theta)

    @inbounds for iq in 1:rzphi.nqty
        # Get f, fy, fyy from theta spline
        f_val, f1_val, f2_val = SplinesMod.deriv2!(rzphi.theta_splines[ipsi, iq], theta_wrapped)
        rzphi._f[iq] = f_val[1]
        rzphi._fy[iq] = f1_val[1]
        rzphi._fyy[iq] = f2_val[1]

        # Get fx from psi derivative spline
        fx_val = SplinesMod.evaluate!(rzphi.psi_deriv_splines[ipsi, iq], theta_wrapped)
        rzphi._fx[iq] = fx_val[1]

        # Get fxy (d/dtheta of d/dpsi) from psi derivative spline
        _, fxy_val = SplinesMod.deriv1!(rzphi.psi_deriv_splines[ipsi, iq], theta_wrapped)
        rzphi._fxy[iq] = fxy_val[1]

        # Compute fxx using finite differences on fs_psi
        rzphi._fxx[iq] = _compute_fxx(rzphi, ipsi, iq, theta_wrapped)
    end

    return rzphi._f, rzphi._fx, rzphi._fy, rzphi._fxx, rzphi._fxy, rzphi._fyy
end

"""
Compute second psi derivative (fxx) using finite differences on the pre-computed
first psi derivatives, interpolated at the given theta.
"""
function _compute_fxx(rzphi::RZPhiSplines{S}, ipsi::Int, iq::Int, theta::Float64) where {S}
    npsi = rzphi.npsi
    xs = rzphi.xs

    if npsi == 1
        return 0.0
    elseif npsi == 2
        return 0.0  # Can't compute second derivative with only 2 points
    end

    # Evaluate psi derivatives at neighboring surfaces
    if ipsi == 1
        # Forward difference
        fx1 = SplinesMod.evaluate!(rzphi.psi_deriv_splines[1, iq], theta)[1]
        fx2 = SplinesMod.evaluate!(rzphi.psi_deriv_splines[2, iq], theta)[1]
        fx3 = SplinesMod.evaluate!(rzphi.psi_deriv_splines[3, iq], theta)[1]
        h1 = xs[2] - xs[1]
        h2 = xs[3] - xs[2]
        # 3-point forward difference for second derivative
        return 2 * (fx3 * h1 - fx2 * (h1 + h2) + fx1 * h2) / (h1 * h2 * (h1 + h2))
    elseif ipsi == npsi
        # Backward difference
        fx1 = SplinesMod.evaluate!(rzphi.psi_deriv_splines[npsi-2, iq], theta)[1]
        fx2 = SplinesMod.evaluate!(rzphi.psi_deriv_splines[npsi-1, iq], theta)[1]
        fx3 = SplinesMod.evaluate!(rzphi.psi_deriv_splines[npsi, iq], theta)[1]
        h1 = xs[npsi-1] - xs[npsi-2]
        h2 = xs[npsi] - xs[npsi-1]
        return 2 * (fx3 * h1 - fx2 * (h1 + h2) + fx1 * h2) / (h1 * h2 * (h1 + h2))
    else
        # Centered difference
        fx_prev = SplinesMod.evaluate!(rzphi.psi_deriv_splines[ipsi-1, iq], theta)[1]
        fx_curr = SplinesMod.evaluate!(rzphi.psi_deriv_splines[ipsi, iq], theta)[1]
        fx_next = SplinesMod.evaluate!(rzphi.psi_deriv_splines[ipsi+1, iq], theta)[1]
        hp = xs[ipsi+1] - xs[ipsi]
        hm = xs[ipsi] - xs[ipsi-1]
        # Non-uniform second derivative formula
        return 2 * (fx_next * hm - fx_curr * (hp + hm) + fx_prev * hp) / (hp * hm * (hp + hm))
    end
end

# =============================================================================
# Utility Functions
# =============================================================================

"""
    empty_RZPhiSplines()

Create an empty/placeholder RZPhiSplines for type stability.
"""
function empty_RZPhiSplines()
    xs = Float64[0.0, 1.0]
    ys = Float64[0.0, 0.5, 1.0]
    fs = zeros(Float64, 2, 3, 4)
    RZPhiSplines(xs, ys, fs)
end

# Convenience evaluation (allocates, but thread-safe)
(rzphi::RZPhiSplines)(psi::Float64, theta::Float64) = copy(evaluate!(rzphi, psi, theta))

# =============================================================================
# API Compatibility (matching BicubicSpline/BicubicWrapper interface)
# =============================================================================

"""
    bicube_eval!(rzphi, x, y)

Legacy API alias for `evaluate!(rzphi, x, y)`.
"""
bicube_eval!(rzphi::RZPhiSplines, x::Float64, y::Float64) = evaluate!(rzphi, x, y)

"""
    bicube_deriv1!(rzphi, x, y)

Legacy API alias for `deriv1!(rzphi, x, y)`.
"""
bicube_deriv1!(rzphi::RZPhiSplines, x::Float64, y::Float64) = deriv1!(rzphi, x, y)

"""
    bicube_deriv2!(rzphi, x, y)

Legacy API alias for `deriv2!(rzphi, x, y)`.
"""
bicube_deriv2!(rzphi::RZPhiSplines, x::Float64, y::Float64) = deriv2!(rzphi, x, y)
