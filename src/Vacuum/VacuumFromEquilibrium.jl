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
)

    mtheta = len(equil.rzphi_ys)
    # Get magnetic axis location
    ro = equil.ro
    zo = equil.zo

    # Get safety factor at this surface
    qa = equil.profiles.q_spline(psi)

    # Allocate output arrays
    r = zeros(Float64, mtheta)
    z = zeros(Float64, mtheta)
    delta = zeros(Float64, mtheta)

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
