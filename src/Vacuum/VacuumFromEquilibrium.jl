"""
    extract_plasma_surface_at_psi(equil::Equilibrium.PlasmaEquilibrium, ψ::Float64) -> (r, z, ν)

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

## Reference    # Allocate output arrays

Matches GPEC's ahg_write and gpvacuum_flxsurf approach (gpvacuum.f line 291-296)
"""
function extract_plasma_surface_at_psi(equil::Equilibrium.PlasmaEquilibrium, ψ::Float64)

    # Allocate output arrays
    mtheta = length(equil.rzphi_ys)
    r_minor = zeros(mtheta)
    θ_cyl = zeros(mtheta)
    ν = zeros(mtheta)

    # Evaluate equilibrium around the flux surface
    hint2d = (Ref(1), Ref(1))
    for (i, θ_sfl) in enumerate(equil.rzphi_ys)
        # Get minor radius, geometric poloidal angle, and toroidal angle offset
        r_minor[i] = sqrt(equil.rzphi_rsquared((ψ, θ_sfl); hint=hint2d))
        θ_cyl[i] = 2π * (θ_sfl + equil.rzphi_offset((ψ, θ_sfl); hint=hint2d))
        ν[i] = equil.rzphi_nu((ψ, θ_sfl); hint=hint2d)
    end

    # Compute R and Z using the computed cylindrical coordinates
    r = equil.ro .+ r_minor .* cos.(θ_cyl)
    z = equil.zo .+ r_minor .* sin.(θ_cyl)
    return r, z, ν
end
