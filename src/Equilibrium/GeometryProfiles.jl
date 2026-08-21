"""
    GeometryProfiles

Flux-surface-averaged geometric quantities (area, ⟨r⟩, ⟨R⟩) computed once at
equilibrium construction time. Mirrors Fortran `pentrc/dcon_interface.f:set_geom`
via the same `issurfint` weighting (`wegt ∈ {0,1,3}`, `ave ∈ {0,1}`), and uses
the nodal_derivs access pattern from `ForceFreeStates/Mercier.jl`.
"""

"""
    compute_geometry_profiles(rzphi_xs, rzphi_ys,
                              rzphi_rsquared, rzphi_offset, rzphi_jac, ro)
        → GeometryProfileSplines

Build named cubic splines for flux-surface-averaged area, ⟨r⟩, and ⟨R⟩ from the
equilibrium's 2D rzphi interpolants. Uses trapezoidal sums in θ across the
non-periodic θ nodes (the trailing duplicate at θ = 1 is skipped, mthsurf =
length(rzphi_ys) - 1).

The integrand `delpsi = |∇ψ|` (in geometry units) is built from the same
nodal derivatives the GS-residual diagnostic uses; see `Equilibrium.jl:407-419`
and the Fortran reference at `pentrc/dcon_interface.f:1199-1254`.
"""
function compute_geometry_profiles(
    rzphi_xs::Vector{Float64},
    rzphi_ys::Vector{Float64},
    rzphi_rsquared::FastInterpolations.CubicInterpolantND,
    rzphi_offset::FastInterpolations.CubicInterpolantND,
    rzphi_jac::FastInterpolations.CubicInterpolantND,
    ro::Float64
)
    npsi = length(rzphi_xs)
    mthsurf = length(rzphi_ys) - 1     # skip the periodic duplicate at θ = 1

    area_vals = zeros(Float64, npsi)
    avg_r_vals = zeros(Float64, npsi)
    avg_R_vals = zeros(Float64, npsi)

    for ipsi in 1:npsi
        area_psi = 0.0
        r_int = 0.0
        R_int = 0.0
        for itheta in 1:mthsurf
            f1 = rzphi_rsquared.nodal_derivs.partials[1, ipsi, itheta]
            offset = rzphi_offset.nodal_derivs.partials[1, ipsi, itheta]
            jac = rzphi_jac.nodal_derivs.partials[1, ipsi, itheta]
            fy1 = rzphi_rsquared.nodal_derivs.partials[3, ipsi, itheta]
            fy2 = rzphi_offset.nodal_derivs.partials[3, ipsi, itheta]

            rfac = sqrt(f1)
            eta = 2π * (rzphi_ys[itheta] + offset)
            R = ro + rfac * cos(eta)

            w1 = (1 + fy2) * (2π)^2 * rfac * R / jac
            w2 = (rfac > 0) ? (-fy1 * π * R / (rfac * jac)) : 0.0
            delpsi = sqrt(w1^2 + w2^2)
            weight = jac * delpsi / mthsurf

            area_psi += weight                  # wegt=0, ave=0
            r_int += weight * rfac              # wegt=3, ave=1
            R_int += weight * R                 # wegt=1, ave=1
        end
        area_vals[ipsi] = area_psi
        avg_r_vals[ipsi] = (area_psi > 0) ? r_int / area_psi : 0.0
        avg_R_vals[ipsi] = (area_psi > 0) ? R_int / area_psi : ro
    end

    return GeometryProfileSplines(rzphi_xs, area_vals, avg_r_vals, avg_R_vals)
end

"""
    compute_geometry_profiles(pe::PlasmaEquilibrium) → GeometryProfileSplines

Convenience method that pulls the rzphi grids and interpolants directly off a
`PlasmaEquilibrium`. Useful for tests and after-the-fact recomputation.
"""
function compute_geometry_profiles(pe::PlasmaEquilibrium)
    return compute_geometry_profiles(
        pe.rzphi_xs, pe.rzphi_ys,
        pe.rzphi_rsquared, pe.rzphi_offset, pe.rzphi_jac,
        pe.ro
    )
end
