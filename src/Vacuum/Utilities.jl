"""
    extract_plasma_surface_at_psi(equil::Equilibrium.PlasmaEquilibrium, ψ::Float64) -> (r, z, ν)

Evaluate the `rzphi` splines on the straight-field-line θ grid at normalized flux `ψ` (0 at axis, 1 at edge),
returning the surface `R`, `Z` and toroidal angle offset `ν`, each of length `mtheta`.
`R = R₀ + r cos θ_cyl`, `Z = Z₀ + r sin θ_cyl` with `r = √rzphi_rsquared` and `θ_cyl = 2π(θ + rzphi_offset)`.
Reference: Chance, Phys. Plasmas 1997 2161. Follows Fortran `gpvacuum_flxsurf` (gpvacuum.f).
"""
function extract_plasma_surface_at_psi(equil::Equilibrium.PlasmaEquilibrium, ψ::Float64)

    mtheta = length(equil.rzphi_ys)
    r_minor = zeros(mtheta)
    θ_cyl = zeros(mtheta)
    ν = zeros(mtheta)

    hint2d = (Ref(1), Ref(1))
    for (i, θ_sfl) in enumerate(equil.rzphi_ys)
        r_minor[i] = sqrt(equil.rzphi_rsquared((ψ, θ_sfl); hint=hint2d))
        θ_cyl[i] = 2π * (θ_sfl + equil.rzphi_offset((ψ, θ_sfl); hint=hint2d))
        ν[i] = equil.rzphi_nu((ψ, θ_sfl); hint=hint2d)
    end

    r = equil.ro .+ r_minor .* cos.(θ_cyl)
    z = equil.zo .+ r_minor .* sin.(θ_cyl)
    return r, z, ν
end

"""
    signed_area(x, z)

Shoelace area of the closed (x, z) contour: positive when it runs counter-clockwise.
"""
function signed_area(x::AbstractVector{<:Real}, z::AbstractVector{<:Real})
    n = length(x)
    return sum(x[i] * z[mod1(i + 1, n)] - x[mod1(i + 1, n)] * z[i] for i in 1:n) / 2
end

"""
    assert_counterclockwise(x, z, name)

Throw unless the closed (x, z) contour runs counter-clockwise. The 2D kernel takes its normals from the
direction of travel, so a clockwise contour would give a wrong operator silently.
"""
function assert_counterclockwise(x::AbstractVector{<:Real}, z::AbstractVector{<:Real}, name::String)
    signed_area(x, z) > 0 || throw(ArgumentError("$name contour runs clockwise in the (R, Z) plane. The vacuum solve needs it " *
                        "counter-clockwise, GPEC's θ direction: reverse the point order."))
end

"""
    distribute_to_equal_arc_grid(xin, zin)

Resample the closed curve `(xin, zin)` (endpoint not repeated) to the same number of points equally spaced
in arc length, returning `(xout, zout)`. Same role as Fortran `eqarcw`, using periodic cubic splines.
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
        # Midpoint rule for ds/dθ
        theta_mid = θ_grid[i] + dθ / 2.0
        d_xin = spline_x(theta_mid; deriv=DerivOp(1))
        d_zin = spline_z(theta_mid; deriv=DerivOp(1))
        ds_dθ = sqrt(d_xin^2 + d_zin^2)
        arc_length[i+1] = arc_length[i] + ds_dθ * dθ
    end

    # Interpolate the original (x, z) at equal arc-length targets
    arc_length_targets = range(; start=0, length=mtheta, step=arc_length[end]/mtheta)
    # arc_length is an uneven grid so we have to specify the period explicitly
    periodic_bc = PeriodicBC(; endpoint=:exclusive, period=arc_length[end])
    xout = cubic_interp(arc_length[1:(end-1)], xin, arc_length_targets; bc=periodic_bc)
    zout = cubic_interp(arc_length[1:(end-1)], zin, arc_length_targets; bc=periodic_bc)
    return xout, zout
end

"""
    periodic_wrap(x, n) -> Int

`mod1(x, n)` without a division. Only valid for `1 - n ≤ x ≤ 2n`.
"""
@inline periodic_wrap(x::Int, n::Int) = x < 1 ? x + n : (x > n ? x - n : x)

"""
    cross3!(c, a, b, idx)

Write the cross product of rows `idx` of the `(N, 3)` matrices `a` and `b` into row `idx` of `c`.
"""
@inline function cross3!(
    c::AbstractMatrix,
    a::AbstractMatrix,
    b::AbstractMatrix,
    idx::Int
)
    @inbounds begin
        a1 = a[idx, 1]
        a2 = a[idx, 2]
        a3 = a[idx, 3]
        b1 = b[idx, 1]
        b2 = b[idx, 2]
        b3 = b[idx, 3]

        c[idx, 1] = a2*b3 - a3*b2
        c[idx, 2] = a3*b1 - a1*b3
        c[idx, 3] = a1*b2 - a2*b1
    end
end
