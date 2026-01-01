"""
Functions to extract vacuum calculation inputs from equilibrium at arbitrary flux surfaces.

This allows computing Green's functions at interior flux surfaces (e.g., singular surfaces)
by extracting the plasma geometry from the equilibrium bicubic spline.
"""

# Note: Equilibrium module is imported in parent scope via ..Equilibrium

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
    sq_vals = Spl.spline_eval!(equil.sq, psi)
    qa = sq_vals[4]

    # Allocate output arrays
    r = zeros(Float64, mtheta_eq)
    z = zeros(Float64, mtheta_eq)
    delta = zeros(Float64, mtheta_eq)

    # Evaluate equilibrium around the flux surface
    twopi = 2π
    for itheta in 0:mtheta_eq-1
        # Theta coordinate normalized to [0, 1)
        theta = itheta / mtheta_eq

        # Evaluate bicubic spline at (psi, theta)
        rzphi_vals = Spl.bicube_eval!(equil.rzphi, psi, theta)

        # Extract values
        # rzphi_vals is the _f array which gets filled by bicube_eval!
        # Component 1: r² or rfac²
        # Component 2: deta (angle offset)
        # Component 3: dphi (toroidal angle offset)
        # Component 4: jac (Jacobian)

        rfac = sqrt(abs(rzphi_vals[1]))
        deta = rzphi_vals[2]
        dphi = rzphi_vals[3]

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

    # Create VacuumInput structure
    mhigh = mlow + mpert - 1

    return Vacuum.VacuumInput(
        r = r,
        z = z,
        delta = delta,
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
    # Reuse initialization from compute_vacuum_response
    (; mtheta, mpert, mlow, n, qa) = inputs
    plasma_surf = Vacuum.initialize_plasma_surface(inputs)
    wall = Vacuum.initialize_wall(inputs, plasma_surf, wall_settings)

    # Compute Fourier basis coefficients
    cslth, snlth = Vacuum.compute_fourier_coefficients(mtheta, mpert, mlow; n=n, qa=qa, delta=plasma_surf.delta)

    # Allocate arrays for both Green's functions
    grri = zeros(2 * mtheta, 2 * mpert)
    grre = zeros(2 * mtheta, 2 * mpert)
    grad_greenfunction_mat = zeros(2 * mtheta, 2 * mtheta)
    greenfunction_temp = zeros(mtheta, mtheta)

    # Plasma–Plasma block
    j1, j2 = 1, 1
    Vacuum.kernel!(grad_greenfunction_mat, greenfunction_temp, plasma_surf.x, plasma_surf.z, plasma_surf.x, plasma_surf.z, j1, j2, 1, n)

    # Fourier transform plasma-plasma block
    Vacuum.fourier_transform!(grri, greenfunction_temp, cslth, 0, 0)
    Vacuum.fourier_transform!(grri, greenfunction_temp, snlth, 0, mpert)
    Vacuum.fourier_transform!(grre, greenfunction_temp, cslth, 0, 0)
    Vacuum.fourier_transform!(grre, greenfunction_temp, snlth, 0, mpert)

    !wall.nowall && begin
        # Plasma–Wall block
        j1, j2 = 1, 2
        Vacuum.kernel!(grad_greenfunction_mat, greenfunction_temp, plasma_surf.x, plasma_surf.z, wall.x, wall.z, j1, j2, 0, n)

        # Wall–Wall block
        j1, j2 = 2, 2
        Vacuum.kernel!(grad_greenfunction_mat, greenfunction_temp, wall.x, wall.z, wall.x, wall.z, j1, j2, 0, n)

        # Wall–Plasma block
        j1, j2 = 2, 1
        Vacuum.kernel!(grad_greenfunction_mat, greenfunction_temp, wall.x, wall.z, plasma_surf.x, plasma_surf.z, j1, j2, 1, n)

        # Fourier transform wall blocks
        Vacuum.fourier_transform!(grri, greenfunction_temp, cslth, mtheta, 0)
        Vacuum.fourier_transform!(grri, greenfunction_temp, snlth, mtheta, mpert)
        Vacuum.fourier_transform!(grre, greenfunction_temp, cslth, mtheta, 0)
        Vacuum.fourier_transform!(grre, greenfunction_temp, snlth, mtheta, mpert)
    end

    # Add cn0 for n=0 modes if needed
    cn0 = 1.0
    (abs(n) <= 1e-5 && !wall.nowall && wall.is_closed_toroidal) && begin
        grri[1:mtheta, 1:mpert] .+= cn0
        grre[1:mtheta, 1:mpert] .+= cn0
    end

    return grri, grre
end
