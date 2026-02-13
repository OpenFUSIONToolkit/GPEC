"""
Functions to extract vacuum calculation inputs from equilibrium at arbitrary flux surfaces.

This allows computing Green's functions at interior flux surfaces (e.g., singular surfaces)
by extracting the plasma geometry from the equilibrium bicubic spline.
"""

# Note: Equilibrium module is imported in parent scope via ..Equilibrium
# Import FourierTransforms for coefficient calculation
using ..Utilities.FourierTransforms: compute_fourier_coefficients, fourier_transform!

"""
    extract_plasma_surface_at_psi(equil::Equilibrium.PlasmaEquilibrium, ψ::Float64) -> (r, z, ν, qa)

Extract plasma surface geometry from equilibrium at specified flux coordinate.

Evaluates equilibrium bicubic spline around the flux surface to get R, Z coordinates
and computes the toroidal angle offset ν for vacuum calculations.

## Arguments
- `equil`: Equilibrium solution with rzphi bicubic spline
- `ψ`: Normalized flux coordinate (0 at axis, 1 at edge)

## Returns
Tuple of:
- `r::Vector{Float64}`: R-coordinates around flux surface [mtheta]
- `z::Vector{Float64}`: Z-coordinates around flux surface [mtheta]
- `ν::Vector{Float64}`: Toroidal angle offset ν [mtheta]
- `q::Float64`: Safety factor at this surface

## Implementation

The equilibrium bicubic spline `rzphi` stores:
- rzphi.f[1] = r² (or rfac²)
- rzphi.f[2] = offset (straight-fieldline poloidal angle offset from geometric poloidal angle)
- rzphi.f[3] = ν (toroidal angle offset from geometric toroidal angle)
- rzphi.f[4] = jac (Jacobian)

From these we compute:
- r_minor = √(rzphi.f[1])
- θ_cyl = 2π*(θ_sfl + rzphi.f[2])
- R = R₀ + r_minor * cos(θ_cyl)
- Z = Z₀ + r_minor * sin(θ_cyl)

## Reference
Matches GPEC's ahg_write and gpvacuum_flxsurf approach (gpvacuum.f line 291-296)
"""
function extract_plasma_surface_at_psi(equil::Equilibrium.PlasmaEquilibrium, ψ::Float64)

    # Get safety factor at this surface
    q = equil.profiles.q_spline(ψ)

    # Allocate output arrays
    mtheta = length(equil.rzphi_ys)
    r_minor = zeros(mtheta)
    θ_cyl = zeros(mtheta)
    ν = zeros(mtheta)

    # Evaluate equilibrium around the flux surface
    for (i, θ_sfl) in enumerate(equil.rzphi_ys)
        # Get minor radius, geometric poloidal angle, and toroidal angle offset
        r_minor[i] = sqrt(equil.rzphi_rsquared((ψ, θ_sfl)))
        θ_cyl[i] = 2π * (θ_sfl + equil.rzphi_offset((ψ, θ_sfl)))
        ν[i] = equil.rzphi_nu((ψ, θ_sfl))
    end

    # Compute R and Z using the computed cylindrical coordinates
    r = equil.ro .+ r_minor .* cos.(θ_cyl)
    z = equil.zo .+ r_minor .* sin.(θ_cyl)
    return r, z, ν, q
end

"""
    create_vacuum_input_at_psi(
        equil::Equilibrium.PlasmaEquilibrium,
        psi::Float64,
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
    mtheta::Int,
    mpert::Int,
    mlow::Int,
    n::Int;
    force_wv_symmetry::Bool = true
)
    # Extract plasma surface geometry at this psi
    r, z, ν, q = extract_plasma_surface_at_psi(equil, psi)

    # TODO: this was in Free.jl - is this general for GPEC too?
    # Invert values for n < 0
    if n < 0
        ν .= -ν
        n = -n
    end

    # Create VacuumInput structure
    mhigh = mlow + mpert - 1

    return VacuumInput(
        r=reverse(r),
        z=reverse(z),
        ν=reverse(ν),
        mlow = mlow,
        mhigh = mhigh,
        mpert = mpert,
        n = n,
        qa = q,
        mtheta_eq = equil.config.mtheta,
        mtheta = mtheta,
        force_wv_symmetry = force_wv_symmetry
    )
end
