"""
Functions to extract vacuum calculation inputs from equilibrium at arbitrary flux surfaces.

This allows computing Green's functions at interior flux surfaces (e.g., singular surfaces)
by extracting the plasma geometry from the equilibrium bicubic spline.
"""

# Note: Equilibrium module is imported in parent scope via ..Equilibrium
# Import FourierTransforms for coefficient calculation
using ..Utilities.FourierTransforms: compute_fourier_coefficients, fourier_transform!

"""
    extract_plasma_surface_at_psi(
        equil::Equilibrium.PlasmaEquilibrium,
        psi::Float64,
        mtheta_eq::Int
    ) -> (r, z, delta, qa)

Extract plasma surface geometry from equilibrium at specified flux coordinate.

Evaluates equilibrium bicubic spline around the flux surface to get R, Z coordinates
and computes the toroidal angle offset delta for vacuum calculations.

## Arguments
- `equil`: Equilibrium solution with rzphi bicubic spline
- `psi`: Normalized flux coordinate (0 at axis, 1 at edge)
- `mtheta_eq`: Number of poloidal grid points for surface extraction

## Returns
Tuple of:
- `r::Vector{Float64}`: R-coordinates around flux surface [mtheta_eq]
- `z::Vector{Float64}`: Z-coordinates around flux surface [mtheta_eq]
- `delta::Vector{Float64}`: Toroidal angle offset δ = -ν/qa [mtheta_eq]
- `qa::Float64`: Safety factor at this surface

## Implementation

The equilibrium bicubic spline `rzphi` stores:
- rzphi.f[1] = r² (or rfac²)
- rzphi.f[2] = deta (angle offset)
- rzphi.f[3] = dphi (toroidal angle)
- rzphi.f[4] = jac (Jacobian)

From these we compute:
- rfac = √(rzphi.f[1])
- eta = 2π*(θ + rzphi.f[2])
- R = R₀ + rfac*cos(eta)
- Z = Z₀ + rfac*sin(eta)
- delta = rzphi.f[3] / qa

## Reference
Matches GPEC's ahg_write and gpvacuum_flxsurf approach (gpvacuum.f line 291-296)
"""
function extract_plasma_surface_at_psi(
    equil::Equilibrium.PlasmaEquilibrium,
    psi::Float64,
    mtheta_eq::Int
)
    # Get magnetic axis location
    ro = equil.ro
    zo = equil.zo

    # Get safety factor at this surface
    qa = equil.profiles.q_spline(psi)

    # Allocate output arrays
    r = zeros(Float64, mtheta_eq)
    z = zeros(Float64, mtheta_eq)
    delta = zeros(Float64, mtheta_eq)

    # Evaluate equilibrium around the flux surface
    twopi = 2π
    for itheta in 0:mtheta_eq-1
        # Theta coordinate normalized to [0, 1)
        theta = itheta / mtheta_eq

        # Evaluate bicubic splines at (psi, theta)
        # New API uses separate interpolants for each component
        r2 = equil.rzphi_rsquared((psi, theta))      # r² or rfac²
        deta = equil.rzphi_offset((psi, theta))      # angle offset
        dphi = equil.rzphi_nu((psi, theta))          # toroidal angle offset (nu)

        rfac = sqrt(abs(r2))

        # Compute R, Z coordinates
        eta = twopi * (theta + deta)
        r[itheta+1] = ro + rfac * cos(eta)
        z[itheta+1] = zo + rfac * sin(eta)

        # Toroidal angle offset: delta = -ν/qa where ϕ = 2πζ + ν
        # In GPEC: delta = dphi (already normalized)
        delta[itheta+1] = dphi
    end

    return r, z, delta, qa
end

"""
    create_vacuum_input_at_psi(
        equil::Equilibrium.PlasmaEquilibrium,
        psi::Float64,
        mtheta_eq::Int,
        mtheta::Int,
        mpert::Int,
        mlow::Int,
        n::Int
    ) -> VacuumInput

Create VacuumInput structure for computing Green's functions at arbitrary flux surface.

Extracts plasma geometry from equilibrium and packages it into VacuumInput format.

## Arguments
- `equil`: Equilibrium solution
- `psi`: Normalized flux coordinate
- `mtheta_eq`: Number of equilibrium poloidal points
- `mtheta`: Number of vacuum calculation poloidal points
- `mpert`: Number of perturbing poloidal modes
- `mlow`: Lowest poloidal mode number
- `n`: Toroidal mode number

## Returns
VacuumInput structure ready for compute_vacuum_response()

## Usage
This is used to compute Green's functions at singular surfaces:
```julia
vac_input = create_vacuum_input_at_psi(equil, sing_surf.psifac, ...)
grri, grre = compute_greens_functions_only(vac_input, wall_settings)
```
"""
function create_vacuum_input_at_psi(
    equil::Equilibrium.PlasmaEquilibrium,
    psi::Float64,
    mtheta_eq::Int,
    mtheta::Int,
    mpert::Int,
    mlow::Int,
    n::Int
)
    # Extract plasma surface geometry at this psi
    r, z, delta, qa = extract_plasma_surface_at_psi(equil, psi, mtheta_eq)

    # Convert delta to ν for VacuumInput
    # Relationship: delta = -ν/qa, so ν = -delta*qa
    ν = -delta .* qa

    # Create VacuumInput structure
    mhigh = mlow + mpert - 1

    return VacuumInput(
        r = r,
        z = z,
        ν = ν,
        mlow = mlow,
        mhigh = mhigh,
        mpert = mpert,
        n = n,
        qa = qa,
        mtheta_eq = mtheta_eq,
        mtheta = mtheta,
        force_wv_symmetry = true
    )
end

"""
    compute_greens_functions_only(
        inputs::VacuumInput,
        wall_settings::WallShapeSettings
    ) -> (grri, grre)

Compute only Green's functions without full vacuum response matrix.

This is a lightweight version of compute_vacuum_response() that skips
the wv matrix calculation and only returns the Green's functions needed
for surface inductance calculations.

## Arguments
- `inputs`: Vacuum input with plasma geometry
- `wall_settings`: Wall shape settings

## Returns
Tuple of:
- `grri::Matrix{Float64}`: Interior Green's function [2*mtheta, 2*mpert]
- `grre::Matrix{Float64}`: Exterior Green's function [2*mtheta, 2*mpert]

## Performance
Much faster than full compute_vacuum_response() since it skips:
- wv matrix construction
- Eigenvalue calculations
- Mode coupling matrices

## Reference
Mimics GPEC's gpvacuum_flxsurf which only computes Green's functions
for surface inductance (gpvacuum.f line 279-283)
"""
function compute_greens_functions_only(
    inputs::VacuumInput,
    wall_settings::WallShapeSettings
)

    # Initialization and allocations
    (; mtheta, mpert, mlow, n, qa, force_wv_symmetry) = inputs
    plasma_surf = initialize_plasma_surface(inputs)
    wall = initialize_wall(inputs, plasma_surf, wall_settings)

    # Compute Fourier basis coefficients using FourierTransforms utility
    # We only need the coefficient arrays for the existing fourier_transform! functions
    cos_ln_basis, sin_ln_basis = compute_fourier_coefficients(mtheta, mpert, mlow; n=n, qa=qa, delta=plasma_surf.delta)

    # Allocate arrays for both Green's functions
    grri = zeros(2 * mtheta, 2 * mpert)  # Interior (kernelsign=-1)
    grre = zeros(2 * mtheta, 2 * mpert)  # Exterior (kernelsign=+1)
    grad_green = zeros(2 * mtheta, 2 * mtheta)
    green_temp = zeros(mtheta, mtheta)

    PLASMA_ROW_OFFSET = 0
    WALL_ROW_OFFSET = mtheta
    COS_COL_OFFSET = 0
    SIN_COL_OFFSET = mpert

    # Plasma–Plasma block
    kernel!(grad_green, green_temp, plasma_surf, plasma_surf, n)

    # Fourier transform obs=plasma, src=plasma block
    fourier_transform!(grri, green_temp, cos_ln_basis, PLASMA_ROW_OFFSET, COS_COL_OFFSET)
    fourier_transform!(grri, green_temp, sin_ln_basis, PLASMA_ROW_OFFSET, SIN_COL_OFFSET)
    fourier_transform!(grre, green_temp, cos_ln_basis, PLASMA_ROW_OFFSET, COS_COL_OFFSET)
    fourier_transform!(grre, green_temp, sin_ln_basis, PLASMA_ROW_OFFSET, SIN_COL_OFFSET)

    if !wall.nowall
        # Plasma–Wall block
        kernel!(grad_green, green_temp, plasma_surf, wall, n)
        # Wall–Wall block
        kernel!(grad_green, green_temp, wall, wall, n)
        # Wall–Plasma block
        kernel!(grad_green, green_temp, wall, plasma_surf, n)

        # Fourier transform obs=wall, src=plasma block
        fourier_transform!(grri, green_temp, cos_ln_basis, WALL_ROW_OFFSET, COS_COL_OFFSET)
        fourier_transform!(grri, green_temp, sin_ln_basis, WALL_ROW_OFFSET, SIN_COL_OFFSET)
        fourier_transform!(grre, green_temp, cos_ln_basis, WALL_ROW_OFFSET, COS_COL_OFFSET)
        fourier_transform!(grre, green_temp, sin_ln_basis, WALL_ROW_OFFSET, SIN_COL_OFFSET)
    end

    # Add cn0 to make grdgre nonsingular for n=0 modes
    cn0 = 1.0 # expose to user if anyone ever actually tries to use this
    (n == 0 && !wall.nowall && wall.is_closed_toroidal) && begin
        @warn "Adding $cn0 to diagonal of grdgre to regularize n=0 mode; this may affect accuracy of results."
        mth12 = wall.nowall ? mtheta : 2 * mtheta
        for i in 1:mth12, j in 1:mth12
            grad_green[i, j] += cn0
        end
    end

    # Compute both Green's functions with different kernel signs
    # grri: interior potential (kernelsign=-1)
    # grre: exterior potential (kernelsign=+1)

    # Make copies for each kernelsign
    grad_green_interior = copy(grad_green)
    grad_green_exterior = copy(grad_green)

    # Apply kernelsign transformations
    apply_kernelsign!(grad_green_interior, -1.0, mtheta)  # Interior
    apply_kernelsign!(grad_green_exterior, +1.0, mtheta)  # Exterior (no-op)

    # Invert the vacuum response system for both cases
    if wall.nowall
        @views grri[1:mtheta, :] .= grad_green_interior[1:mtheta, 1:mtheta] \ grri[1:mtheta, :]
        @views grre[1:mtheta, :] .= grad_green_exterior[1:mtheta, 1:mtheta] \ grre[1:mtheta, :]
    else
        grri .= grad_green_interior \ grri
        grre .= grad_green_exterior \ grre
    end

    return grri, grre
end
