"""
    fourier_inverse_transform!(gll, gil, cs, m00, l00)

Perform the inverse Fourier transform of `gil` onto `gll` using Fourier coefficients stored in `cs`.

# Arguments

  - `gll`: Output matrix (mpert × mpert) updated in-place
  - `gil`: Input matrix (mtheta × mpert) containing Fourier-space data
  - `cs`: Fourier coefficient matrix (mtheta × mpert)
  - `m00`: Integer offset in the gil matrix (row offset)
  - `l00`: Integer offset in the gil matrix (column offset)

# Notes

  - Computes: `gll[l2, l1] = (2π * dth) * Σ_i cs[i, l2] * gil[i, l1]`
  - Performs the same function as fouranv in the Fortran code.

# Returns

  - gll(l2,l1) : output matrix updated in-place (mpert × mpert)
"""
function fourier_inverse_transform!(gll::Matrix{Float64}, gil::Matrix{Float64}, cs::Matrix{Float64}, m00::Int, l00::Int)

    # Zero out gll block
    mtheta, mpert = size(cs)
    fill!(view(gll, 1:mpert, 1:mpert), 0.0)

    # Inverse Fourier transform via matrix multiply: gll = cs^T * gil * (2π * dth)
    # This computes: gll[l2, l1] = (2π * dth) * Σ_i cs[i, l2] * gil[i, l1]
    dth = 2π / mtheta
    mul!(gll, cs', view(gil, (m00+1):(m00+mtheta), (l00+1):(l00+mpert)), 2π * dth, 0.0)
end

"""
    fourier_transform!(gil, gij, cs, m00, l00, mth, mpert)

    Purpose:
      This routine performs a truncated Fourier transform of gij onto gil
      using Fourier coefficients stored in cs.

    Inputs:
      gij(i,j)   : input matrix of size (mth × mth), the "physical-space" data
      cs(j,l)    : Fourier coefficient matrix (mth × mpert)
      m00, l00   : integer offsets in the gil matrix
      mth        : number of θ-grid points (dimension of gij along i, j)
      mpert      : number of Fourier modes

    Output:
      gil(i', l') : output matrix updated in-place (mth × mpert), where i' = m00 + i and l' = l00 + l
"""
function fourier_transform!(gil::Matrix{Float64}, gij::Matrix{Float64}, cs::Matrix{Float64}, m00::Int, l00::Int)

    # Zero out relevant gil block
    mtheta, mpert = size(cs)
    fill!(view(gil, (m00+1):(m00+mtheta), (l00+1):(l00+mpert)), 0.0)

    # Fourier transform via matrix multiply: gil[i, l] = Σ_j gij[i, j] * cs[j, l]
    mul!(view(gil, (m00+1):(m00+mtheta), (l00+1):(l00+mpert)), gij, cs)
end

# Returns the array of derivatives at all x points, I think this acts like difspl
# in the Fortran but need to check/consolidate spline routines later
function periodic_cubic_deriv(theta, vals)
    # Close the loop for periodic BC by appending first point at the end
    theta_closed = vcat(collect(theta), theta[end] + (theta[2] - theta[1]))
    vals_closed = vcat(vals, vals[1])
    spline = cubic_interp(theta_closed, vals_closed; bc=PeriodicBC())
    d1 = deriv1(spline)
    return d1.(collect(theta))
end

"""
    lagrange1d(ax, af, m, nl, x, iop)

Perform Lagrange interpolation and optionally compute its derivative.
Replaces `lagp`, `lagpe4`, and `lag` from Fortran code.

# Arguments

  - `ax::Vector{Float64}`: Array of x-coordinates for the interpolation points
  - `af::Vector{Float64}`: Array of y-coordinates (function values) for the interpolation points
  - `m::Int`: Number of interpolation points
  - `nl::Int`: Number of points to use for the local interpolation (polynomial degree + 1)
  - `x::Float64`: The x-value at which to evaluate the interpolated function and/or its derivative
  - `iop::Int`: Flag controlling output (0 = value only, 1 = value and derivative)

# Returns

  - `f::Float64`: The interpolated function value at `x`
  - `df::Float64`: The interpolated function derivative at `x` (0.0 if iop=0)

# Notes

  - Uses local Lagrange interpolation with `nl` points centered around `x`
  - Automatically adjusts interpolation window to stay within array bounds
"""
function lagrange1d(ax::Vector{Float64}, af::Vector{Float64}, m::Int, nl::Int, x::Float64, iop::Int)

    # --- Error fix: Initialize f and df internally to 0.0 ---
    f::Float64 = 0.0
    df::Float64 = 0.0

    jn = findfirst(i -> ax[i] >= x, 1:m)
    jn = (jn === nothing) ? m : jn
    jn = max(jn - 1, 1)
    if jn < m && abs(ax[jn+1] - x) < abs(x - ax[jn])
        jn += 1
    end

    # Determine the range of indices for interpolation
    jnmm = floor(Int, (nl - 0.1) / 2)
    jnpp = floor(Int, (nl + 0.1) / 2)

    nll = jn - jnmm
    nlr = jn + jnpp

    # Adjust for even nl when ax[jn] > x (Window shift)
    if (nl % 2 == 0) && (ax[jn] > x)
        nll -= 1
        nlr -= 1 # <--- Shifting the window (was nlr += 1)
    end

    # Clamp indices to valid array bounds
    if nlr > m
        nlr = m
        nll = nlr - nl + 1
    elseif nll < 1
        nll = 1
        nlr = nl
    end

    # Compute function value f
    for i in nll:nlr
        alag = 1.0
        for j in nll:nlr
            (i == j) && continue
            alag *= (x - ax[j]) / (ax[i] - ax[j])
        end
        f += alag * af[i]
    end

    # --- Error fix: Use the 'iop' argument ---
    (iop == 0) && return f, df # df is returned as 0.0

    # Compute derivative df
    for i in nll:nlr
        slag = 0.0
        for id in nll:nlr
            (id == i) && continue
            alag = 1.0
            for j in nll:nlr
                (j == i) && continue
                alag *= (j != id) ? ((x - ax[j]) / (ax[i] - ax[j])) : (1.0 / (ax[i] - ax[id]))
            end
            slag += alag
        end
        df += slag * af[i]
    end

    return f, df # Return value and derivative
end

"""
    interp_to_new_grid(vecin, mtheta; dx0=0.0, dx1=0.0)

Resample the input array `vecin` using a periodic cubic spline to an output array of length `mtheta`.

This function unifies the Fortran functions `trans`, `transdx`, and `transdxx` into a single
function with optional offset parameters.

# Arguments

  - `vecin::Vector{Float64}`: Input array to be resampled
  - `mtheta::Int`: Desired length of the output array
  - `dx0::Float64`: Global offset added to all x-coordinates (default 0, applied as `x += dx0 / mtheta_in`)
  - `dx1::Float64`: Fine offset added to each index (default 0, applied as `ai = (i-1) + dx1`)

# Returns

  - `vecout::Vector{Float64}`: The resampled output array (length `mtheta`)

# Notes

  - If `mtheta == length(vecin)`, returns the input vector unchanged
  - Uses periodic cubic spline interpolation for resampling
  - Input grid is normalized to [0, 1] for interpolation
"""
function interp_to_new_grid(vecin::Vector{Float64}, mtheta::Int; dx0=0.0, dx1=0.0)

    # Initialize
    mtheta_in = length(vecin)

    # If mtheta == mtheta_in, just return the input vector
    if mtheta == mtheta_in
        return vecin
    end

    # Input grids are from [0, 1] inclusive, since no interpolants will fall outside of this, we don't need periodic extrapolation
    θin = collect(range(0.0, 1.0; length=mtheta_in))
    spline = cubic_interp(θin, vecin)

    # Interpolate to new grid with optional offsets
    vecout = zeros(mtheta)
    for i in 1:mtheta
        x = (i - 1 + dx1) / mtheta + dx0 / mtheta_in
        x = x % 1.0  # This is for periodicity in the case of dx1/dx0 ≠ 0
        vecout[i] = spline(x)
    end
    return vecout
end

"""
    distribute_to_equal_arc_grid(xin, zin, mw1)

Perform arc length re-parameterization of a 2D curve.

Takes an input curve defined by `(xin, zin)` coordinates and re-samples it such that
the new points `(xout, zout)` are equally spaced in arc length along the curve.

# Arguments

  - `xin::Vector{Float64}`: Array of x-coordinates of the input curve
  - `zin::Vector{Float64}`: Array of z-coordinates of the input curve
  - `mw1::Int`: Number of points in the input and output curves

# Returns

  - `xout::Vector{Float64}`: Array of x-coordinates of the arc-length re-parameterized curve
  - `zout::Vector{Float64}`: Array of z-coordinates of the arc-length re-parameterized curve
  - `ell::Vector{Float64}`: Array of cumulative arc lengths for the input curve
  - `thgr::Vector{Float64}`: Array of re-parameterized 'theta' values corresponding to equal arc lengths
  - `thlag::Vector{Float64}`: Array of normalized 'theta' values for the input curve (0 to 1)

# Notes

  - Uses Lagrange interpolation for calculating arc length and resampling
  - Ensures uniform spacing in arc length for improved numerical stability
"""
function distribute_to_equal_arc_grid(xin::Vector{Float64}, zin::Vector{Float64}, mtheta::Int)

    # Temporary arrays for interpolation and arc-length calculation
    theta_in = zeros(Float64, mtheta) # Normalized input parameter [0, 1)
    theta_out = zeros(Float64, mtheta) # New parameter distribution for equal spacing
    xout = zeros(Float64, mtheta) # Uniformly spaced R-coordinates
    zout = zeros(Float64, mtheta) # Uniformly spaced Z-coordinates

    # Define initial normalized parameter theta_in
    dt = 1.0 / mtheta
    theta_in .= range(; start=0, length=mtheta, step=dt) # θ ∈ [0, 1)
    # we need a closed loop for arc length calculation
    mtheta1 = mtheta + 1
    xin1 = vcat(xin, xin[1])
    zin1 = vcat(zin, zin[1])
    theta_in1 = vcat(theta_in, [1.0])
    ell = zeros(Float64, mtheta1) # Cumulative arc length of closed loop

    # Calculate cumulative arc length using numerical integration
    # We use a mid-point derivative approximation to find the path length
    for iw in 2:mtheta1
        # Evaluate derivative at the midpoint of the interval
        theta = (theta_in1[iw] + theta_in1[iw-1]) / 2.0

        # Calculate dx/dt and dz/dt using Lagrange interpolation (order 3)
        _, d_xin = lagrange1d(theta_in1, xin1, mtheta1, 3, theta, 1)
        _, d_zin = lagrange1d(theta_in1, zin1, mtheta1, 3, theta, 1)

        # Instantaneous speed (ds/dt)
        ds_dt = sqrt(d_xin^2 + d_zin^2)

        # Accumulate length: ds = (ds/dt) * dt
        ell[iw] = ell[iw-1] + ds_dt * dt
    end

    # Re-parameterize based on equal arc-length segments
    ell_targets = collect(range(0; step=ell[end]/mtheta, length=mtheta)) # [0, Length) for open loop result
    for i in 2:mtheta
        # Find the value of 'theta_in' that corresponds to the target arc length 's'
        f_th, _ = lagrange1d(ell, theta_in1, mtheta1, 3, ell_targets[i], 0)
        theta_out[i] = f_th
    end

    # Interpolate the original (x,z) data at the new parameter points to get (xout, zout)
    # Chance interpolates in theta_out to get xin, zin but this introduces small errors in arc lengths
    for i in 1:mtheta
        f_x, _ = lagrange1d(ell, xin1, mtheta1, 3, ell_targets[i], 0)
        f_z, _ = lagrange1d(ell, zin1, mtheta1, 3, ell_targets[i], 0)
        xout[i] = f_x
        zout[i] = f_z
    end

    return xout, zout, ell, theta_out, theta_in
end

# Helper functions for compute_vacuum_field

"""
    _pickup_field(inputs, plasma_surf, grri, Bn_real, Bn_imag, R_grid, Z_grid)

Calculate the magnetic field on a specified grid using finite differencing
of the magnetic scalar potential `chi`.

This is the Julia version of the Fortran `pickup` routine. It computes the vacuum
magnetic field perturbation at a set of grid points given the plasma surface perturbation.

# Arguments

  - `inputs::VacuumInput`: Struct containing vacuum calculation parameters
  - `plasma_surf::PlasmaGeometry`: Struct with plasma surface geometry and basis functions
  - `grri::Matrix{Float64}`: Inverted Green's function response matrix from vaccal!
  - `Bn_real::Vector{Float64}`: Real part of normal field Fourier harmonics at plasma surface
  - `Bn_imag::Vector{Float64}`: Imaginary part of normal field Fourier harmonics at plasma surface
  - `R_grid::AbstractVector`: R-coordinates for output field evaluation
  - `Z_grid::AbstractVector`: Z-coordinates for output field evaluation

# Returns

  - `B_R::Matrix{ComplexF64}`: R-component of magnetic field on grid (nx × nz)
  - `B_Z::Matrix{ComplexF64}`: Z-component of magnetic field on grid (nx × nz)
  - `B_phi::Matrix{ComplexF64}`: Toroidal component of magnetic field on grid (nx × nz)
  - `grid_info::Matrix{Int}`: Grid point classification (1=inside plasma, 0=outside)

# Notes

  - Uses 5-point finite difference stencil for computing field from potential
  - Field components computed as: B_R = -∂χ/∂R, B_Z = -∂χ/∂Z, B_φ = inχ/R
"""
function _pickup_field(inputs::VacuumInput, plasma_surf::PlasmaGeometry, grri::Matrix{Float64},
    Bn_real::Vector{Float64}, Bn_imag::Vector{Float64},
    R_grid::AbstractVector, Z_grid::AbstractVector)

    nx = length(R_grid)
    nz = length(Z_grid)
    ifac = 1im

    # Create the grid of points where the potential will be calculated
    R_points, Z_points = _create_pickup_grid(R_grid, Z_grid)
    n_points = length(R_points)

    # Output arrays
    B_R_complex = zeros(ComplexF64, nx, nz)
    B_Z_complex = zeros(ComplexF64, nx, nz)
    B_phi_complex = zeros(ComplexF64, nx, nz)
    grid_info = zeros(Int, nx, nz)

    # Finite difference steps
    del_R = 1e-5 * (maximum(plasma_surf.x) - minimum(plasma_surf.x))
    del_Z = 1e-5 * (maximum(plasma_surf.x) - minimum(plasma_surf.x))

    # Calculate potential `chi` at 5 points for each grid location for finite differencing
    # 1: (R, Z + dZ), 2: (R, Z - dZ), 3: (R + dR, Z), 4: (R - dR, Z), 5: (R, Z)
    chi_r = zeros(5, n_points)
    chi_i = zeros(5, n_points)

    Threads.@threads for i in 1:n_points
        R, Z = R_points[i], Z_points[i]

        # Points for finite difference stencil
        observe_points = [
            (R, Z + del_Z),
            (R, Z - del_Z),
            (R + del_R, Z),
            (R - del_R, Z),
            (R, Z)
        ]

        for (j, (obs_R, obs_Z)) in enumerate(observe_points)
            chi_r[j, i], chi_i[j, i] = _calculate_potential_chi(
                obs_R, obs_Z, inputs, plasma_surf, grri, Bn_real, Bn_imag
            )
        end
    end

    # Calculate fields using finite differences and reshape into a grid
    for i in 1:nx
        for j in 1:nz
            idx = (i - 1) * nz + j

            # B_R = -d(chi)/dR, B_Z = -d(chi)/dZ
            br_r = -(chi_r[3, idx] - chi_r[4, idx]) / (2.0 * del_R)
            br_i = -(chi_i[3, idx] - chi_i[4, idx]) / (2.0 * del_R)

            bz_r = -(chi_r[1, idx] - chi_r[2, idx]) / (2.0 * del_Z)
            bz_i = -(chi_i[1, idx] - chi_i[2, idx]) / (2.0 * del_Z)

            # B_phi = i*n*chi / R
            # Bphi = i*n*(chi_r + i*chi_i)/R = (-n*chi_i + i*n*chi_r)/R
            bphi_r = -inputs.n * chi_i[5, idx] / R_points[idx]
            bphi_i = inputs.n * chi_r[5, idx] / R_points[idx]

            B_R_complex[i, j] = br_r + ifac * br_i
            B_Z_complex[i, j] = bz_r + ifac * bz_i
            B_phi_complex[i, j] = bphi_r + ifac * bphi_i

            # Check if point is inside the plasma
            fintjj = 0.0
            for k in 1:inputs.mtheta
                dx = R_points[idx] - plasma_surf.x[k]
                dz = Z_points[idx] - plasma_surf.z[k]
                rho2 = dx^2 + dz^2
                if rho2 > 1e-16
                    fintjj += (plasma_surf.dz_dtheta[k] * dx - plasma_surf.dx_dtheta[k] * dz) / rho2
                end
            end
            grid_info[i, j] = (fintjj > 0.1) ? 1 : 0 # 1 for interior, 0 for exterior
        end
    end

    return B_R_complex, B_Z_complex, B_phi_complex, grid_info
end

"""
    _create_pickup_grid(R_grid, Z_grid)

Create a flattened 1D list of (R, Z) coordinates from 2D grid vectors.

This is the Julia version of the Fortran `loops` subroutine.

# Arguments

  - `R_grid::AbstractVector`: Vector of R-coordinates defining the grid
  - `Z_grid::AbstractVector`: Vector of Z-coordinates defining the grid

# Returns

  - `R_points::Vector{Float64}`: Flattened array of R-coordinates (length nx*nz)
  - `Z_points::Vector{Float64}`: Flattened array of Z-coordinates (length nx*nz)

# Notes

  - Grid points are ordered as: [(R[1],Z[1]), (R[1],Z[2]), ..., (R[1],Z[nz]), (R[2],Z[1]), ...]
"""
function _create_pickup_grid(R_grid::AbstractVector, Z_grid::AbstractVector)
    nx = length(R_grid)
    nz = length(Z_grid)
    R_points = zeros(Float64, nx * nz)
    Z_points = zeros(Float64, nx * nz)

    for i in 1:nx
        for j in 1:nz
            idx = (i - 1) * nz + j
            R_points[idx] = R_grid[i]
            Z_points[idx] = Z_grid[j]
        end
    end
    return R_points, Z_points
end

"""
    _calculate_potential_chi(R_obs, Z_obs, inputs, plasma_surf, grri, Bn_real, Bn_imag)

Calculate the magnetic scalar potential chi at a single observation point (R_obs, Z_obs).

This is the Julia version of the Fortran `chi` subroutine. The potential is computed
by integrating the Green's function response with the source perturbation at the plasma surface.

# Arguments

  - `R_obs::Float64`: R-coordinate of observation point
  - `Z_obs::Float64`: Z-coordinate of observation point
  - `inputs::VacuumInput`: Struct containing vacuum calculation parameters
  - `plasma_surf::PlasmaGeometry`: Struct with plasma surface geometry
  - `grri::Matrix{Float64}`: Inverted Green's function response matrix
  - `Bn_real::Vector{Float64}`: Real part of normal field Fourier harmonics
  - `Bn_imag::Vector{Float64}`: Imaginary part of normal field Fourier harmonics

# Returns

  - `chi_real::Float64`: Real part of the magnetic scalar potential at (R_obs, Z_obs)
  - `chi_imag::Float64`: Imaginary part of the magnetic scalar potential at (R_obs, Z_obs)

# Notes

  - The potential is computed via Fourier series over poloidal modes
  - Includes coupling term from Green's function derivative
  - Factor of -0.5 * dtheta applied from Fortran convention
"""
function _calculate_potential_chi(R_obs::Float64, Z_obs::Float64,
    inputs::VacuumInput, plasma_surf::PlasmaGeometry,
    grri::Matrix{Float64},
    Bn_real::Vector{Float64}, Bn_imag::Vector{Float64})

    chi_real = 0.0
    chi_imag = 0.0

    mtheta = inputs.mtheta
    mpert = inputs.mpert
    n = inputs.n
    dtheta = 2pi / mtheta

    # Pre-calculate Green's function for the observation point
    g_real = zeros(mtheta, mpert)
    g_imag = zeros(mtheta, mpert)

    l_modes = (inputs.mlow:inputs.mhigh)

    for i_theta in 1:mtheta
        R_src = plasma_surf.x[i_theta]
        Z_src = plasma_surf.z[i_theta]

        # Call the low-level Green's function calculator.
        # The `green` function returns the Green's function value itself (G_n) and
        # the coupling terms for mode n and mode 0.
        G_n, coupling_n, coupling_0 = green(R_obs, Z_obs, R_src, Z_src, plasma_surf.dx_dtheta[i_theta], plasma_surf.dz_dtheta[i_theta], n)

        # The term `aval` in the original Fortran CHI routine corresponds to the coupling term 𝒥 ∇'𝒢ⁿ∇'ℒ,
        # which is directly returned as `coupling_n` by the Julia `green` function.
        aval = coupling_n

        # Accumulate Fourier series for g_real and g_imag at this source point
        for l_idx in 1:mpert
            g_real[i_theta, l_idx] = aval * plasma_surf.cos_ln_basis[i_theta, l_idx]
            g_imag[i_theta, l_idx] = aval * plasma_surf.sin_ln_basis[i_theta, l_idx]
        end
    end

    # Now, combine with the inverted response `grri` and the source `Bn`
    # This corresponds to the main loop in the Fortran `chi` subroutine
    for l1 in 1:mpert
        term_r = 0.0
        term_i = 0.0
        for i_theta in 1:mtheta
            # grri has structure [ (grri_cc, grri_cs), (grri_sc, grri_ss) ]
            # The indices for chiwc, chiws in Fortran map to columns of grri
            chi_wc = grri[i_theta, l1]          # Real part kernel
            chi_ws = grri[i_theta, mpert+l1]  # Imaginary part kernel

            term_r += g_real[i_theta, l1] * chi_wc - g_imag[i_theta, l1] * chi_ws
            term_i += g_imag[i_theta, l1] * chi_wc + g_real[i_theta, l1] * chi_ws
        end

        chi_real += term_r * Bn_real[l1] - term_i * Bn_imag[l1]
        chi_imag += term_i * Bn_real[l1] + term_r * Bn_imag[l1]
    end

    # The factor of 0.5 * isg * dth in Fortran
    # isg is -1 for plasma surface
    factor = -0.5 * dtheta
    return chi_real * factor, chi_imag * factor
end
