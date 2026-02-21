"""
    distribute_to_equal_arc_grid(xin, zin)

Given a set of points (xin, zin) that define a closed curve in 2D, redistribute these points to be equally spaced
in terms of arc length along the curve. This is useful for ensuring that points on a wall or plasma surface are
uniformly distributed in space rather than in the parameter θ. This performs the same function as eqarcw in the
Fortran code. We now use FastInterpolations instead of a manual Lagrange interpolation.

# Arguments

  - `xin::Vector{Float64}`: x-coordinates of the original points defining the curve (endpoint not included)
  - `zin::Vector{Float64}`: z-coordinates of the original points defining the curve (endpoint not included)

# Returns

  - `xout::Vector{Float64}`: x-coordinates of the redistributed points, equally spaced in arc length
  - `zout::Vector{Float64}`: z-coordinates of the redistributed points, equally spaced in arc length
"""
function distribute_to_equal_arc_grid(xin::Vector{Float64}, zin::Vector{Float64})

    mtheta = length(xin)
    dθ = 2π / mtheta
    θ_grid = range(; start=0, length=mtheta, step=dθ)

    # Build periodic splines for derivatives on the closed loop
    spline_x = cubic_interp(θ_grid, xin; bc=PeriodicBC(; endpoint=:exclusive))
    spline_z = cubic_interp(θ_grid, zin; bc=PeriodicBC(; endpoint=:exclusive))

    # Calculate cumulative arc length using numerical integration
    arc_length = zeros(mtheta + 1)
    for i in 1:mtheta
        # Use a mid-point derivative approximation
        theta_mid = θ_grid[i] + dθ / 2.0
        d_xin = spline_x(theta_mid; deriv=1)
        d_zin = spline_z(theta_mid; deriv=1)
        # Accumulate length: ds = (ds/dt) * dt
        ds_dθ = sqrt(d_xin^2 + d_zin^2)
        arc_length[i+1] = arc_length[i] + ds_dθ * dθ
    end

    # Re-parameterize based on equal arc-length segments by interpolate the original (x,z) data at the equal arc length points
    arc_length_targets = range(; start=0, length=mtheta, step=arc_length[end]/mtheta)
    # arc_length is an uneven grid so we have to specify the period explicitly
    periodic_bc = PeriodicBC(; endpoint=:exclusive, period=arc_length[end])

    # Use One-Shot API
    xout = cubic_interp(arc_length[1:(end-1)], xin, arc_length_targets; bc=periodic_bc)
    zout = cubic_interp(arc_length[1:(end-1)], zin, arc_length_targets; bc=periodic_bc)
    return xout, zout
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
