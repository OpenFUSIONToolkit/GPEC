# Helper functions for mscfld

"""
    _pickup_field(inputs, plasma_surf, grri, Bn_real, Bn_imag, R_grid, Z_grid)

Internal function to calculate the magnetic field on a specified grid.
This is a Julia version of the Fortran `pickup` routine. It uses finite differencing
of the magnetic scalar potential `chi` to find the field components.
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
            bphi_i =  inputs.n * chi_r[5, idx] / R_points[idx]
            
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

Creates a flattened 1D list of (R, Z) coordinates from 2D grid vectors.
This is a Julia version of the Fortran `loops` subroutine.
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

Calculates the magnetic scalar potential chi at a single observation point (R_obs, Z_obs).
This is a Julia version of the Fortran `chi` subroutine.
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
    qa = inputs.qa
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
            l = l_modes[l_idx]
            arg = l * (i_theta-1) * dtheta + n * qa * plasma_surf.delta[i_theta]
            cos_val = cos(arg)
            sin_val = sin(arg)

            g_real[i_theta, l_idx] = aval * cos_val
            g_imag[i_theta, l_idx] = aval * sin_val
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
            chi_ws = grri[i_theta, mpert + l1]  # Imaginary part kernel
            
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

"""
    mscfld(inputs::VacuumInput, plasma_surf::PlasmaGeometry, wall::WallGeometry, 
           Bn::Vector{<:Number}, R_grid::AbstractVector, Z_grid::AbstractVector)

Calculates the perturbed magnetic field in the vacuum region resulting from a normal
magnetic field perturbation (`Bn`) at the plasma surface.

This function orchestrates the vacuum field calculation by:
1. Calling `vaccal!` to compute the vacuum response kernel (`grri`).
2. Defining a grid of points (`R_grid`, `Z_grid`) where the field is to be calculated.
3. Calling `_pickup_field` to compute the magnetic field components on that grid using the kernel
   and the source perturbation `Bn`.

# Arguments
- `inputs::VacuumInput`: Struct containing vacuum calculation parameters (n, mpert, etc.).
- `plasma_surf::PlasmaGeometry`: Struct with plasma surface geometry and basis functions.
- `wall::WallGeometry`: Struct with wall geometry.
- `Bn::Vector{<:Number}`: Complex vector of Fourier harmonics of the normal magnetic field
  perturbation at the plasma surface, `B_n = B_n_real + i*B_n_imag`. Length must be `mpert`.
- `R_grid::AbstractVector`: Vector of R coordinates for the output field grid.
- `Z_grid::AbstractVector`: Vector of Z coordinates for the output field grid.

# Returns
- `B_R::Matrix{ComplexF64}`: The R-component of the magnetic field on the grid.
- `B_Z::Matrix{ComplexF64}`: The Z-component of the magnetic field on the grid.
- `B_phi::Matrix{ComplexF64}`: The toroidal component of the magnetic field on the grid.
- `grid_info::Matrix{Int}`: Information about the grid points (1=inside plasma, 0=outside).
"""
function mscfld(inputs::VacuumInput, plasma_surf::PlasmaGeometry, wall::WallGeometry, 
                Bn::Vector{<:Number}, R_grid::AbstractVector, Z_grid::AbstractVector)

    # 1. Call vaccal! to get the inverted Green's function matrix
    # The Fortran version calls the whole chain (ent33 -> vaccal), 
    # here we assume vaccal! provides what we need.
    wv, grri = vaccal!(inputs, plasma_surf, wall)

    # Separate real and imaginary parts of the source perturbation
    Bn_real = real.(Bn)
    Bn_imag = imag.(Bn)

    # 2. Define grid and parameters for pickup routine
    nx = length(R_grid)
    nz = length(Z_grid)
    
    # 3. Call the field pickup routine
    B_R, B_Z, B_phi, grid_info = _pickup_field(
        inputs, plasma_surf, grri, Bn_real, Bn_imag, R_grid, Z_grid
    )

    return B_R, B_Z, B_phi, grid_info
end


