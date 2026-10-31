"""
    HermiteFEM

The Hermite-cubic finite-element pieces shared by the two Galerkin solvers: the outer-region ForceFreeStates
solver and the GGJ inner-layer solver. Both are ports of the Fortran `gal.f` / GGJ `deltac` kernels.
"""
module HermiteFEM

using StaticArrays: SVector

export hermite_basis, pack_grid

"""
    hermite_basis(x, x0, x1) -> (pb, qb)

Hermite cubic basis on `[x0, x1]`: values `pb` and derivatives `qb`, slots 1:4 ↔ (u₀, u₀', u₁, u₁'). Port of
Fortran `gal_hermite` (gal.f).
"""
function hermite_basis(x::Real, x0::Real, x1::Real)
    dx = x1 - x0
    t0 = (x - x0) / dx
    t1 = 1 - t0
    t02 = t0 * t0
    t12 = t1 * t1
    pb = SVector{4,Float64}(t12 * (1 + 2t0), t12 * t0 * dx, t02 * (1 + 2t1), -t02 * t1 * dx)
    qb = SVector{4,Float64}(-6t0 * t1 / dx, t0 * (3t0 - 4) + 1, 6t1 * t0 / dx, t0 * (3t0 - 2))
    return pb, qb
end

"""
    pack_grid(nx, pfac, side) -> Vector{Float64}

Packed logical grid on `[-1, 1]`, port of Fortran `gal_pack` (gal.f). `side = "both"` gives `2nx+1` nodes
(Fortran `-nx:nx`); `"left"` and `"right"` give `2nx+1` nodes mapped onto `[-1, 1]` from one side. `pfac > 1`
packs toward the centre (GWP2016 Eq. 33), `pfac == 1` is uniform, `0 < pfac < 1` packs toward the ends (Eq. 5).
"""
function pack_grid(nx::Int, pfac::Float64, side::String)
    if side == "left"
        xi = collect(-2nx:0) ./ (2nx)
    elseif side == "right"
        xi = collect(0:2nx) ./ (2nx)
    elseif side == "both"
        xi = collect(-nx:nx) ./ nx
    else
        error("pack_grid: unrecognized side = $side")
    end

    if pfac > 1
        λ = sqrt(1 - 1 / pfac)
        x = @. log((1 + λ * xi) / (1 - λ * xi)) / log((1 + λ) / (1 - λ))
    elseif pfac == 1
        x = copy(xi)
    elseif pfac > 0
        λ = sqrt(1 - pfac)
        a = log((1 + λ) / (1 - λ))
        x = @. (exp(a * xi) - 1) / (exp(a * xi) + 1) / λ
    else
        x = @. xi^(-pfac)
    end

    if side == "left"
        @. x = 2x + 1
    elseif side == "right"
        @. x = 2x - 1
    end
    return x
end

end # module HermiteFEM
