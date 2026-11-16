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
    arc_length_targets = range(; start=0, length=mtheta, step=arc_length[end] / mtheta)
    # arc_length is an uneven grid so we have to specify the period explicitly
    periodic_bc = PeriodicBC(; endpoint=:exclusive, period=arc_length[end])
    xout = cubic_interp(arc_length[1:(end-1)], xin, arc_length_targets; bc=periodic_bc)
    zout = cubic_interp(arc_length[1:(end-1)], zin, arc_length_targets; bc=periodic_bc)
    return xout, zout
end

"""
    trig_derivative_matrix(n) -> Matrix{Float64}

Derivative at the nodes of the trigonometric interpolant on `n` equispaced periodic nodes.
For even `n` the Nyquist mode is a cosine, whose derivative vanishes at the nodes.
"""
function trig_derivative_matrix(n::Int)
    D = zeros(n, n)
    for j in 1:n, i in 1:n
        r = mod(i - j, n)
        r == 0 && continue
        # Symmetric range keeps sin(πr/n) away from π, where it loses relative precision; the sign factor absorbs the shift
        r > n ÷ 2 && (r -= n)
        s, c = sincospi(r / n)
        D[i, j] = (isodd(r) ? -0.5 : 0.5) * (iseven(n) ? c / s : 1 / s)
    end
    return D
end

"""
    trig_tangents(r, mtheta, nzeta) -> (dr_dθ, dr_dζ)

Tangents of the trigonometric interpolant of the `(mtheta * nzeta, 3)` grid samples `r`.
"""
function trig_tangents(r::Matrix{Float64}, mtheta::Int, nzeta::Int)
    Dθ, Dζ = trig_derivative_matrix(mtheta), trig_derivative_matrix(nzeta)
    dr_dθ, dr_dζ = similar(r), similar(r)
    for k in 1:3
        R = reshape(r[:, k], mtheta, nzeta)
        dr_dθ[:, k] = vec(Dθ * R)
        dr_dζ[:, k] = vec(R * transpose(Dζ))
    end
    return dr_dθ, dr_dζ
end

"""
    trig_upsample_matrix(n) -> Matrix{Float64}

Values at the `2n` nodes of the twice finer grid of the trigonometric interpolant on `n` equispaced periodic nodes.
Odd rows are the identity.
"""
function trig_upsample_matrix(n::Int)
    U = zeros(2n, n)
    for j in 1:n, i in 1:2n
        # Offset of fine node i from coarse node j, in fine spacings
        r = mod(i - 2j + 1, 2n)
        if iseven(r)
            U[i, j] = r == 0
        else
            # Symmetric range, as in `trig_derivative_matrix`; mod(r, 4) flips with the shift exactly when csc does (odd n)
            r > n && (r -= 2n)
            s, c = sincospi(r / 2n)
            # Periodic sinc at a midpoint: sign alternates every other midpoint; cot for even n (Nyquist cosine), csc for odd n
            U[i, j] = (mod(r, 4) == 1 ? 1 : -1) * (iseven(n) ? c / s : 1 / s) / n
        end
    end
    return U
end

"""
    trig_upsample(a, Uθ, Uζ, mtheta, nzeta) -> Matrix{Float64}

Columns of the `(mtheta * nzeta, k)` grid samples `a` on the 2× finer grid; `Uθ`, `Uζ` from [`trig_upsample_matrix`](@ref).
"""
function trig_upsample(a::AbstractMatrix{Float64}, Uθ::Matrix{Float64}, Uζ::Matrix{Float64}, mtheta::Int, nzeta::Int)
    # 2× in θ and 2× in ζ → 4× the nodes, still θ-fastest
    a_up = Matrix{Float64}(undef, 4 * mtheta * nzeta, size(a, 2))
    for k in axes(a, 2)
        a_up[:, k] = vec(Uθ * reshape(a[:, k], mtheta, nzeta) * transpose(Uζ))
    end
    return a_up
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

        c[idx, 1] = a2 * b3 - a3 * b2
        c[idx, 2] = a3 * b1 - a1 * b3
        c[idx, 3] = a1 * b2 - a2 * b1
    end
end

# cᵢ = 1/∏_{j≠i}(i - j) on the nodes 0:N-1, for every order up to 21 (factorial(21) overflows Int)
const LAGRANGE_DENOMINATORS = ntuple(N -> ntuple(i -> (-1)^(N - i) / (factorial(i - 1) * factorial(N - i)), N), 21)

"""
    lagrange_weights(ξ, Val(N)) -> SVector{N,Float64}

Lagrange cardinal weights at `ξ` on the uniform nodes `0:(N - 1)`: wᵢ = cᵢ ∏_{j≠i} (ξ - j + 1).
"""
@inline function lagrange_weights(ξ, ::Val{N}) where {N}
    c = LAGRANGE_DENOMINATORS[N]
    return SVector(ntuple(i -> foldl(*, ntuple(j -> j == i ? 1.0 : ξ - (j - 1), Val(N)); init=c[i]), Val(N)))
end

# Plasma–wall rows whose observer is more than NEAR_GAP source node spacings h from the source curve take the periodic
# trapezoidal rule alone: its error ∝ exp(-2πd/h) is ~1e-11 there. Nearer rows are integrated about the closest source
# point
const NEAR_GAP = 4.0

"""
    sinh_rule(Q, nmax, step) -> (; rules, step)

Gauss–Legendre rules on [0, 1] with `Q, 2Q, …, nmax Q` nodes for [`sinh_nodes`](@ref), which takes `⌈Δs/step⌉ Q` nodes
on a span `Δs` of the sinh variable.
"""
function sinh_rule(Q::Int, nmax::Int, step::Float64)
    rules = map(1:nmax) do m
        t, w = gausslegendre(m * Q)
        return (t .+ 1) ./ 2, w ./ 2
    end
    return (; rules, step)
end

"""
    sinh_nodes(ε, a, b, rule) -> iterator of (ρ, w)

Quadrature nodes and weights on `[a, b]` for a kernel peaked at `ρ = 0` with width `ε`: `ρ = ε sinh(s)`, Gauss in `s`
between `asinh(a/ε)` and `asinh(b/ε)` [Johnston & Elliott, Int. J. Numer. Meth. Engng 62 (2005) 564], from a
[`sinh_rule`](@ref).
"""
@inline function sinh_nodes(ε::Float64, a::Float64, b::Float64, rule)
    sa, sb = asinh(a / ε), asinh(b / ε)
    Δs = sb - sa
    t_nodes, t_weights = rule.rules[clamp(ceil(Int, Δs / rule.step), 1, length(rule.rules))]
    return ((ε * sinh(sa + Δs * t_nodes[i]), t_weights[i] * Δs * ε * cosh(sa + Δs * t_nodes[i])) for i in eachindex(t_nodes))
end

"""
    periodic_splines(x, z) -> (dx_dθ, dz_dθ, spline_x, spline_z)

Periodic cubic splines of a closed curve sampled on the uniform grid `θ = 2π(k - 1)/M`, and their derivatives on it.
"""
function periodic_splines(x::Vector{Float64}, z::Vector{Float64})
    θ = range(; start=0, length=length(x), step=2π / length(x))
    bc = PeriodicBC(; endpoint=:exclusive, period=2π)
    spline_x, spline_z = cubic_interp(θ, x; bc), cubic_interp(θ, z; bc)
    return spline_x.(θ; deriv=DerivOp(1)), spline_z.(θ; deriv=DerivOp(1)), spline_x, spline_z
end

"""
    curve_point(curve, θ) -> (x, z, ∂x/∂θ, ∂z/∂θ)

Point and tangent of a 2D geometry's spline at any angle `θ`.
"""
@inline function curve_point(curve, θ)
    θ = mod(θ, 2π)
    return curve.spline_x(θ), curve.spline_z(θ), curve.spline_x(θ; deriv=DerivOp(1)), curve.spline_z(θ; deriv=DerivOp(1))
end

"""
    closest_point(curve, x, z) -> (; θ, d, h)

Point of a counter-clockwise 2D geometry closest to `(x, z)`, by Newton on its spline from the nearest node: its angle `θ`,
the signed distance `d` along the outward normal, and the local node spacing `h`.
"""
function closest_point(curve, x, z)
    M = length(curve.x)
    θ = 2π / M * (argmin(@. (curve.x - x)^2 + (curve.z - z)^2) - 1)
    for _ in 1:30
        X, Z, Xθ, Zθ = curve_point(curve, θ)
        Xθθ, Zθθ = curve.spline_x(mod(θ, 2π); deriv=DerivOp(2)), curve.spline_z(mod(θ, 2π); deriv=DerivOp(2))
        Δ = clamp(((x - X) * Xθ + (z - Z) * Zθ) / (Xθ^2 + Zθ^2 - (x - X) * Xθθ - (z - Z) * Zθθ), -2π / M, 2π / M)
        θ += Δ
        abs(Δ) < 1e-14 && break
    end
    X, Z, Xθ, Zθ = curve_point(curve, θ)
    J = hypot(Xθ, Zθ)
    return (; θ=mod(θ, 2π), d=((x - X) * Zθ - (z - Z) * Xθ) / J, h=J * 2π / M)
end
