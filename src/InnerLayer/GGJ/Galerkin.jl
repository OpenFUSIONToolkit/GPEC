# Galerkin.jl
#
# Hermite-cubic finite-element (Galerkin) solver for the GGJ inner layer, porting the GWP2016
# Sec. III singular-Galerkin method: the half-domain weak form (Eq. 32) on a packed grid
# (Eq. 33), parity conditions at x = 0, and resonant/extension cells at x_max supplying the
# matching data (Eqs. 34–35). Fortran deltac configuration: "resonant" + noexp + inps,
# Hermite basis, mpert = 3 (Ψ, Ξ, Υ). `_assemble_and_solve_point!` reuses the assembly on the
# rotated ray with a point far-field condition (Ray.jl).

using LinearAlgebra
using FastGaussQuadrature: gausslobatto
using QuadGK: quadgk

"""
    _physical_uv(params, Q, x) -> (I, U, V)

3×3 coefficients of `I·u'' − V·u' − U·u = 0` for u = (Ψ, Ξ, Υ) at real or complex `x`, with
(I, V, U) = (A, −B, −C) of GWP2016 Eqs. 12–15; each row is one equation of GWP2016 Eq. 11.
"""
function _physical_uv(params::GGJParameters, Q::ComplexF64, x::Number)
    e = ComplexF64(params.E)
    f = ComplexF64(params.F)
    h = ComplexF64(params.H)
    g = ComplexF64(params.G)
    k = ComplexF64(params.K)
    q = Q
    q2 = q * q
    x2 = x * x

    Imat = @SMatrix ComplexF64[1 0 0; 0 q2 0; 0 0 q]

    # U = −C (GWP2016 Eq. 15) and V = −B (Eq. 14), rows scaled by diag(1, Q², Q) like `Imat`.
    #! format: off
    U = @SMatrix ComplexF64[
        q                     (-q * x)                       0
        (-x / q) * q2         (x2 / q) * q2                  (-(e + f) / q2) * q2
        (-x / q) * q          (-(g - k * e) * q) * q         (x2 / q + (g + k * f) * q) * q
    ]
    V = @SMatrix ComplexF64[
        0                       0         h
        (-h / q2) * q2          0         0
        (h * k * q) * q         0         0
    ]
    #! format: on

    return Imat, U, V
end

# Hermite cubic basis on [x0, x1]: values pb and derivatives qb, slots 1:4 ↔ (u₀, u₀', u₁, u₁').
function _hermite(x::Real, x0::Real, x1::Real)
    dx = x1 - x0
    t0 = (x - x0) / dx
    t1 = 1 - t0
    t02 = t0 * t0
    t12 = t1 * t1
    pb = SVector{4,Float64}(
        t12 * (1 + 2t0),
        t12 * t0 * dx,
        t02 * (1 + 2t1),
        -t02 * t1 * dx
    )
    qb = SVector{4,Float64}(
        -6t0 * t1 / dx,
        t0 * (3t0 - 4) + 1,
        6t1 * t0 / dx,
        t0 * (3t0 - 2)
    )
    return pb, qb
end

# Grid packing: pfac > 1 concentrates nodes near X = 0 (GWP2016 Eq. 33), pfac = 1 is uniform,
# 0 < pfac < 1 packs toward the edge (Eq. 5).
function _pack(nx::Int, pfac::Float64, side::String)
    xi = zeros(Float64, 2nx + 1)
    if side == "right"
        for i in 0:(2nx)
            xi[i+1] = i / (2.0 * nx)
        end
    elseif side == "left"
        for i in (-2nx):0
            xi[i+2nx+1] = i / (2.0 * nx)
        end
    else # "both"
        for i in (-nx):nx
            xi[i+nx+1] = i / Float64(nx)
        end
    end

    x = similar(xi)
    if pfac > 1
        λ = sqrt(1 - 1 / pfac)
        @. x = log((1 + λ * xi) / (1 - λ * xi)) / log((1 + λ) / (1 - λ))
    elseif pfac == 1
        x .= xi
    elseif pfac > 0
        λ = sqrt(1 - pfac)
        a = log((1 + λ) / (1 - λ))
        @. x = (exp(a * xi) - 1) / (exp(a * xi) + 1) / λ
    else
        @. x = xi^(-pfac)
    end

    if side == "left"
        @. x = 2x + 1
    elseif side == "right"
        @. x = 2x - 1
    end
    return x
end

# x_max and the two thin matching cells inside it, from where the series residual (GW2020 Eq. 54)
# crosses three tolerances. Returns (xmax, dx1, dx2, cache).
function _xmax_3level(params::GGJParameters, Q::ComplexF64;
    kmax::Int=8, xfac::Float64=1.0,
    eps_vec::NTuple{3,Float64}=(1e-2, 5e-7, 1e-7))
    cache = build_asymptotics(params, Q; kmax=kmax)

    # Bracket each crossing on a coarse log sweep, then bisect.
    dxfac = 10.0^0.01
    xmax_vals = zeros(Float64, 3)
    brackets = Vector{Tuple{Float64,Float64}}(undef, 3)
    set = [true, true, true]
    x_prev = 0.1
    delta_prev = maximum(asymptotic_residual(cache, x_prev))
    x = x_prev * dxfac
    for _ in 1:1000
        dmax = maximum(asymptotic_residual(cache, x))
        for i in 1:3
            if delta_prev >= eps_vec[i] && dmax < eps_vec[i] && set[i]
                brackets[i] = (x_prev, x)
                set[i] = false
            end
        end
        if !any(set)
            break
        end
        x_prev = x
        delta_prev = dmax
        x *= dxfac
    end
    any(set) && error("_xmax_3level: failed to bracket all xmax levels")

    # Bisect each bracket to find smooth xmax
    for i in 1:3
        lo, hi = brackets[i]
        for _ in 1:60  # 60 bisections ≈ machine precision
            mid = (lo + hi) / 2
            if maximum(asymptotic_residual(cache, mid)) < eps_vec[i]
                hi = mid
            else
                lo = mid
            end
        end
        xmax_vals[i] = (lo + hi) / 2
    end

    xmax_vals[2] *= xfac
    xmax_vals[3] *= xfac
    xmax = xmax_vals[3]
    dx1 = xmax_vals[3] - xmax_vals[2]
    dx2 = dx1 / 2
    return (; xmax, dx1, dx2, cache)
end

# Cell types toward x_max (GWP2016 Sec. III text after Eq. 33; the outer-region analogue is Fig. 1):
# normal Hermite cells (CT_NONE), extension cells blending the asymptotic solutions into the Hermite
# basis (CT_EXT2, CT_EXT1, CT_EXT), and the resonant cell (CT_RES) whose large-solution coefficient is
# the matching data. x = 0 carries the parity conditions.
@enum CellType CT_NONE CT_EXT2 CT_EXT1 CT_EXT CT_RES

"""
    GalerkinCell

One finite element of the Galerkin grid.

## Fields

  - `etype` - Cell type (`CT_NONE`, `CT_EXT2`, `CT_EXT1`, `CT_EXT`, `CT_RES`).
  - `x_left`, `x_right` - Cell end points.
  - `x_lsode` - Upper limit of the resonant-cell integral and the x_max surface term (`CT_RES` only).
  - `np` - Highest local Hermite slot per component: 3 for Hermite cells, 1 for `CT_EXT` (left node only), −1 for `CT_RES`.
  - `map` - `(mpert, np+1)` local-to-global DOF map; `CT_EXT` appends a column of asymptotic DOFs, `CT_RES` holds only those.
  - `emap` - Asymptotic DOF indices shared by `CT_EXT` and `CT_RES`; only `emap[1]` is active (noexp).
"""
struct GalerkinCell
    etype::CellType
    x_left::Float64
    x_right::Float64
    x_lsode::Float64
    np::Int
    map::Matrix{Int}
    emap::Vector{Int}
end

"""
    GalerkinWorkspace

Grid, banded system and scratch buffers of one Galerkin solve.

## Fields

  - `cells` - Grid cells from x = 0 outward.
  - `ndim` - Number of global unknowns.
  - `nx` - Number of cells.
  - `kl` - Banded half-bandwidth (`ku = kl`).
  - `mat` - `(2kl+ku+1, ndim, 2)` LAPACK banded matrices, one per parity (1 = even/Δ₊, 2 = odd/Δ₋).
  - `rhs`, `sol` - `(ndim, 2)` right-hand sides and solutions per parity.
  - `cell_mat_buf`, `cell_mat_ext_buf` - `(3, 3, 4, 4)` per-cell scratch for the Hermite and extension terms.
  - `cell_rhs_ext_buf` - `(3, 4)` per-cell extension RHS scratch.
  - `ab_buf`, `rhs_buf` - Scratch copies for the banded LU solve.
"""
struct GalerkinWorkspace
    cells::Vector{GalerkinCell}
    ndim::Int
    nx::Int
    kl::Int
    mat::Array{ComplexF64,3}
    rhs::Matrix{ComplexF64}
    sol::Matrix{ComplexF64}
    cell_mat_buf::Array{ComplexF64,4}
    cell_mat_ext_buf::Array{ComplexF64,4}
    cell_rhs_ext_buf::Matrix{ComplexF64}
    ab_buf::Matrix{ComplexF64}
    rhs_buf::Vector{ComplexF64}
end

function _build_grid_and_workspace(nx::Int, xmax::Float64, dx1::Float64, dx2::Float64,
    pfac::Float64, cutoff::Int)
    side = pfac < 1 ? "left" : "right"

    # Cell types
    etypes = fill(CT_NONE, nx)
    etypes[nx] = CT_RES
    etypes[nx-1] = CT_EXT
    for ix in (nx-2):-1:(nx-cutoff)
        etypes[ix] = CT_EXT1
    end
    etypes[nx-cutoff-1] = CT_EXT2

    # Grid nodes: x[0..nx] in 1-based → x[1..nx+1]
    x_nodes = zeros(Float64, nx + 1)
    x_nodes[1] = 0.0
    x_nodes[nx+1] = xmax
    x_nodes[nx] = xmax - dx1
    x_nodes[nx-1] = xmax - (dx1 + dx2)
    ixmax = nx - 2  # packed region is 0..ixmax

    x0 = x_nodes[1]
    x1_packed = x_nodes[ixmax+1]
    xm = (x1_packed + x0) / 2
    dxp = (x1_packed - x0) / 2
    mx = ixmax ÷ 2
    packed = xm .+ dxp .* _pack(mx, pfac, side)
    for i in 1:(ixmax+1)
        x_nodes[i] = packed[i]
    end
    x_nodes[1] = 0.0
    x_nodes[ixmax+1] = x1_packed

    return _galerkin_workspace(x_nodes, etypes)
end

# Cells, local-to-global Hermite DOF maps, bandwidth and buffers for given nodes and cell types.
# The stock layout ends in an EXT/RES pair; an all-CT_NONE layout is the plain Hermite grid
# used with an external far-field condition (`_assemble_and_solve_point!`).
function _galerkin_workspace(x_nodes::Vector{Float64}, etypes::Vector{CellType})
    mpert = 3
    np = 3
    nx = length(etypes)
    has_res = etypes[end] == CT_RES

    # Build cells and local-to-global mapping
    cells = Vector{GalerkinCell}(undef, nx)
    imap = 1
    for ix in 1:nx
        et = etypes[ix]
        xl = x_nodes[ix]
        xr = x_nodes[ix+1]

        cell_np = if et == CT_NONE || et == CT_EXT1 || et == CT_EXT2
            np
        elseif et == CT_EXT
            1  # left-node Hermite pair only, plus one asymptotic DOF
        else # CT_RES
            -1  # no Hermite DOFs
        end

        x_lsode = (et == CT_RES) ? xr : 0.0

        # Build local-to-global Hermite DOF map.
        if et == CT_RES
            # Will be set after the loop
            map_local = zeros(Int, mpert, 1)  # map(:, 0:0)
            emap_local = zeros(Int, 1)
            cells[ix] = GalerkinCell(et, xl, xr, x_lsode, cell_np, map_local, emap_local)
            continue
        end

        # Standard Hermite mapping with overlap
        if imap > 1
            imap -= 2 * mpert
        end
        map_local = zeros(Int, mpert, cell_np + 1)  # map(:, 0:cell_np)
        for ip in 0:cell_np
            for ipert in 1:mpert
                map_local[ipert, ip+1] = imap
                imap += 1
            end
        end

        emap_local = Int[]
        if et == CT_EXT
            # Extra asymptotic DOFs: emap = (imap, imap+1, imap+2).
            # With noexp, ndim = imap so only emap[1] is active; emap[2,3] > ndim
            # and are skipped during assembly.
            emap_local = [imap, imap + 1, imap + 2]
            map_extra = [imap; imap + 1; imap + 2]  # 3-element column
            map_local = hcat(map_local, map_extra)
        end

        cells[ix] = GalerkinCell(et, xl, xr, x_lsode, cell_np, map_local, emap_local)
    end

    # Set resonant cell mapping: shares emap with ext cell
    if has_res
        ext_cell = cells[nx-1]
        emap_vals = ext_cell.emap
        res_map = reshape(copy(emap_vals), mpert, 1)
        cells[nx] = GalerkinCell(CT_RES, x_nodes[nx], x_nodes[nx+1], x_nodes[nx+1],
            -1, res_map, copy(emap_vals))
    end

    ndim = has_res ? imap : imap - 1  # noexp → ndim = imap (only emap[1] ≤ ndim); plain grid: Hermite DOFs only

    # Bandwidth
    kl = mpert * (np + 1)  # one cell's mpert·(np+1) DOFs
    ku = kl
    ldab = 2kl + ku + 1

    mat = zeros(ComplexF64, ldab, ndim, 2)
    rhs = zeros(ComplexF64, ndim, 2)
    sol = zeros(ComplexF64, ndim, 2)
    # Scratch buffers sized for np + 1 = 4; smaller cells use a zeroed sub-slice.
    cell_mat_buf = zeros(ComplexF64, mpert, mpert, np + 1, np + 1)
    cell_mat_ext_buf = zeros(ComplexF64, mpert, mpert, np + 1, np + 1)
    cell_rhs_ext_buf = zeros(ComplexF64, mpert, np + 1)
    ab_buf = zeros(ComplexF64, ldab, ndim)
    rhs_buf = zeros(ComplexF64, ndim)

    return GalerkinWorkspace(cells, ndim, nx, kl, mat, rhs, sol,
        cell_mat_buf, cell_mat_ext_buf, cell_rhs_ext_buf,
        ab_buf, rhs_buf)
end

# One cell's GWP2016 Eq. 32 bilinear form after integrating the A-term by parts,
# I·(αᵢ' αⱼ') + V·(αᵢ αⱼ') + U·(αᵢ αⱼ), by Gauss–Lobatto quadrature.
function _gauss_quad!(cell_mat::Array{ComplexF64,4}, cell::GalerkinCell,
    quad_nodes::Vector{Float64}, quad_weights::Vector{Float64},
    params::GGJParameters, Q::ComplexF64; θ::Float64=0.0)
    np_cell = cell.np
    nq = length(quad_nodes)
    x0 = (cell.x_right + cell.x_left) / 2
    dx = (cell.x_right - cell.x_left) / 2
    fill!(cell_mat, 0)

    @inbounds for iq in 1:nq
        x = x0 + dx * quad_nodes[iq]
        w = dx * quad_weights[iq]
        if iszero(θ)
            Imat, Umat, Vmat = _physical_uv(params, Q, x)
        else
            # Rotated contour x = e^{iθ}s: d/dx = e^{-iθ} d/ds; times e^{2iθ}, I is unchanged, V gains e^{iθ}, U gains e^{2iθ}.
            Imat, U0, V0 = _physical_uv(params, Q, cis(θ) * x)
            Umat, Vmat = cis(2θ) * U0, cis(θ) * V0
        end
        pb, qb = _hermite(x, cell.x_left, cell.x_right)

        for ip in 0:np_cell, jp in 0:np_cell
            for jpert in 1:3, ipert in 1:3
                cell_mat[ipert, jpert, ip+1, jp+1] +=
                    w * (
                        Imat[ipert, jpert] * qb[ip+1] * qb[jp+1] +
                        Vmat[ipert, jpert] * pb[ip+1] * qb[jp+1] +
                        Umat[ipert, jpert] * pb[ip+1] * pb[jp+1]
                    )
            end
        end
    end
end

# Extension cells (GWP2016 Sec. III): the large resonant solution (column 1) is blended into the
# Hermite cubics with C¹ continuity as an extra DOF; the small one (column 2) drives the RHS.
function _extension!(cell_mat::Array{ComplexF64,4}, cell_rhs::Matrix{ComplexF64},
    cell::GalerkinCell,
    quad_nodes::Vector{Float64}, quad_weights::Vector{Float64},
    params::GGJParameters, Q::ComplexF64,
    cache::InnerAsymptoticsCache)
    np_cell = cell.np
    x0c = (cell.x_right + cell.x_left) / 2
    dxc = (cell.x_right - cell.x_left) / 2
    nq = length(quad_nodes)

    if cell.etype == CT_EXT
        # Asymptotic basis at right boundary of ext cell
        ua_bdy, dua_bdy = physical_ua_dua(cache, cell.x_right)
        ep = np_cell + 1  # index of the extra DOF (1-based → ep+1 in array)

        @inbounds for iq in 1:nq
            x = x0c + dxc * quad_nodes[iq]
            w = dxc * quad_weights[iq]
            uax, duax = physical_ua_dua(cache, x)
            Imat, Umat, Vmat = _physical_uv(params, Q, x)
            pb, qb = _hermite(x, cell.x_left, cell.x_right)

            # Extra DOF φ = ua₁(x_r)·H₁₀ + ua₁'(x_r)·H₁₁, the large solution Hermite-blended at the right
            # node; assembles (hᵢ, Lφ), (φ, Lhᵢ) and (φ, Lφ).
            jp = 1  # column 1 = large solution (noexp)
            ua1 = ua_bdy[:, jp] * pb[3] + dua_bdy[:, jp] * pb[4]
            dua1 = ua_bdy[:, jp] * qb[3] + dua_bdy[:, jp] * qb[4]

            for ip in 0:np_cell
                term1 = qb[ip+1] * (Imat * dua1) + pb[ip+1] * (Vmat * dua1) + pb[ip+1] * (Umat * ua1)
                cell_mat[:, 1, ip+1, ep+1] .+= w .* term1

                ua2 = ua_bdy[:, jp] * pb[3] + dua_bdy[:, jp] * pb[4]
                dua2 = ua_bdy[:, jp] * qb[3] + dua_bdy[:, jp] * qb[4]
                term2 = qb[ip+1] * transpose(transpose(dua2) * Imat) + qb[ip+1] * transpose(transpose(ua2) * Vmat) + pb[ip+1] * transpose(transpose(ua2) * Umat)
                cell_mat[1, :, ep+1, ip+1] .+= w .* term2
            end

            # (phi_1^T, L*phi_1) self-interaction
            ua2_s = ua_bdy[:, 1] * pb[3] + dua_bdy[:, 1] * pb[4]
            dua2_s = ua_bdy[:, 1] * qb[3] + dua_bdy[:, 1] * qb[4]
            term = transpose(dua2_s) * Imat * dua1 + transpose(ua2_s) * Vmat * dua1 + transpose(ua2_s) * Umat * ua1
            cell_mat[1, 1, ep+1, ep+1] += w * term

            # RHS: driven by the small solution (column 2)
            ua_drv = uax[:, 2]
            dua_drv = duax[:, 2]
            for ip in 0:np_cell
                term_r = qb[ip+1] * (Imat * dua_drv) + pb[ip+1] * (Vmat * dua_drv) + pb[ip+1] * (Umat * ua_drv)
                cell_rhs[:, ip+1] .-= w .* term_r
            end
            # RHS for asymptotic test function
            term_re = transpose(dua2_s) * Imat * dua_drv + transpose(ua2_s) * Vmat * dua_drv + transpose(ua2_s) * Umat * ua_drv
            cell_rhs[1, ep+1] -= w * term_re
        end

    elseif cell.etype == CT_EXT1 || cell.etype == CT_EXT2
        # Only RHS contribution (driving term)
        @inbounds for iq in 1:nq
            x = x0c + dxc * quad_nodes[iq]
            w = dxc * quad_weights[iq]
            Imat, Umat, Vmat = _physical_uv(params, Q, x)
            pb, qb = _hermite(x, cell.x_left, cell.x_right)

            if cell.etype == CT_EXT1
                uax, duax = physical_ua_dua(cache, x)
                ua_drv = uax[:, 2]
                dua_drv = duax[:, 2]
            else # CT_EXT2
                ua_bdy, dua_bdy = physical_ua_dua(cache, cell.x_right)
                ua_drv = ua_bdy[:, 2] * pb[3] + dua_bdy[:, 2] * pb[4]
                dua_drv = ua_bdy[:, 2] * qb[3] + dua_bdy[:, 2] * qb[4]
            end

            for ip in 0:np_cell
                term_r = qb[ip+1] * (Imat * dua_drv) + pb[ip+1] * (Vmat * dua_drv) + pb[ip+1] * (Umat * ua_drv)
                cell_rhs[:, ip+1] .-= w .* term_r
            end
        end
    end
end

# Resonant-cell scalar products ∫ ua₁ᵀ L ua_j dx (j = 1 stiffness, j = 2 RHS) with the asymptotic
# solutions as test and trial functions; non-analytic, so adaptive QuadGK (GWP2016 Sec. III).
function _resonant_integral(cell::GalerkinCell, params::GGJParameters,
    Q::ComplexF64, cache::InnerAsymptoticsCache;
    tol::Float64=1e-5)
    x_left = cell.x_left
    x_right = cell.x_lsode

    function integrand_11(x)
        ua, dua = physical_ua_dua(cache, x)
        Imat, Umat, Vmat = _physical_uv(params, Q, x)
        ua1 = ua[:, 1]
        dua1 = dua[:, 1]
        return transpose(dua1) * Imat * dua1 + transpose(ua1) * Vmat * dua1 + transpose(ua1) * Umat * ua1
    end

    function integrand_12(x)
        ua, dua = physical_ua_dua(cache, x)
        Imat, Umat, Vmat = _physical_uv(params, Q, x)
        dua1 = dua[:, 1]
        ua1 = ua[:, 1]
        ua2 = ua[:, 2]
        dua2 = dua[:, 2]
        return transpose(dua1) * Imat * dua2 + transpose(ua1) * Vmat * dua2 + transpose(ua1) * Umat * ua2
    end

    val_11, _ = quadgk(integrand_11, x_left, x_right; rtol=tol)
    val_12, _ = quadgk(integrand_12, x_left, x_right; rtol=tol)

    return ComplexF64(val_11), ComplexF64(val_12)
end

# -----------------------------------------------------------------------
# Global assembly and solve.
# -----------------------------------------------------------------------

function _assemble_and_solve!(ws::GalerkinWorkspace,
    params::GGJParameters, Q::ComplexF64,
    cache::InnerAsymptoticsCache;
    nq::Int=4, tol_res::Float64=1e-5)
    mpert = 3
    np = 3
    quad_nodes, quad_weights = gausslobatto(nq + 1)
    offset = ws.kl + ws.kl + 1  # kl + ku + 1 since ku = kl

    fill!(ws.mat, 0)
    fill!(ws.rhs, 0)

    cell_mat = ws.cell_mat_buf
    cell_mat_ext = ws.cell_mat_ext_buf
    cell_rhs_ext = ws.cell_rhs_ext_buf
    for ix in 1:ws.nx
        cell = ws.cells[ix]

        # Gauss quadrature for the Hermite contribution (all cells but CT_RES)
        if cell.np >= 0
            np_eff = cell.np
            fill!(cell_mat, 0)
            _gauss_quad!(cell_mat, cell, quad_nodes, quad_weights, params, Q)
            _scatter_cell!(ws, cell, cell_mat, np_eff)
        end

        # Extension terms
        if cell.etype in (CT_EXT, CT_EXT1, CT_EXT2)
            # CT_EXT carries one extra (asymptotic) slot after its Hermite pair.
            np_eff = cell.etype == CT_EXT ? cell.np + 1 : cell.np
            fill!(cell_mat_ext, 0)
            fill!(cell_rhs_ext, 0)
            _extension!(cell_mat_ext, cell_rhs_ext, cell, quad_nodes, quad_weights, params, Q, cache)

            # Assemble ext contributions
            npp = np_eff
            for ip in 0:npp, ipert in 1:mpert
                i = ip < size(cell.map, 2) ? cell.map[ipert, ip+1] : cell.emap[1]
                # For the extra DOF, only ipert=1 is meaningful (noexp)
                if cell.etype == CT_EXT && ip == cell.np + 1 && ipert > 1
                    continue
                end
                if i > ws.ndim
                    continue
                end
                for jp in 0:npp, jpert in 1:mpert
                    j = jp < size(cell.map, 2) ? cell.map[jpert, jp+1] : cell.emap[1]
                    if cell.etype == CT_EXT && jp == cell.np + 1 && jpert > 1
                        continue
                    end
                    if j > ws.ndim
                        continue
                    end
                    ws.mat[offset+i-j, j, 1] += cell_mat_ext[ipert, jpert, ip+1, jp+1]
                end
                # RHS (both parities get the same base RHS)
                ws.rhs[i, 1] += cell_rhs_ext[ipert, ip+1]
            end
        end

        # Resonant integral
        if cell.etype == CT_RES
            val_11, val_12 = _resonant_integral(cell, params, Q, cache; tol=tol_res)
            emap1 = cell.emap[1]
            if emap1 <= ws.ndim
                ws.mat[offset, emap1, 1] += val_11  # diagonal: i=j=emap1
                ws.rhs[emap1, 1] -= val_12
            end
        end
    end

    # IBP surface term [αᵢ A uⱼ'] at x_max, dropped in GWP2016 Eq. 32 and kept here because the resonant cell carries Δ.
    res_cell = ws.cells[ws.nx]
    x_lsode = res_cell.x_lsode
    ua_xmax, dua_xmax = physical_ua_dua(cache, x_lsode)
    Imat_xmax, _, _ = _physical_uv(params, Q, x_lsode)
    emap1 = res_cell.emap[1]
    if emap1 <= ws.ndim
        # Surface term: -ua_1^T I dua_1 at x_lsode (stiffness)
        surf_11 = transpose(ua_xmax[:, 1]) * Imat_xmax * dua_xmax[:, 1]
        ws.mat[offset, emap1, 1] -= surf_11
        # Surface term: +ua_1^T I dua_2 at x_lsode (RHS)
        surf_12 = transpose(ua_xmax[:, 1]) * Imat_xmax * dua_xmax[:, 2]
        ws.rhs[emap1, 1] += surf_12
    end

    _parity_bcs_and_solve!(ws, params, Q, ws.ndim)
end

# Add one cell's Hermite block to the base (isol = 1) banded matrix; both parities share it.
function _scatter_cell!(ws::GalerkinWorkspace, cell::GalerkinCell, cell_mat::Array{ComplexF64,4}, np_eff::Int)
    mpert = 3
    offset = ws.kl + ws.kl + 1
    for ip in 0:np_eff, ipert in 1:mpert
        i = cell.map[ipert, ip+1]
        if i > ws.ndim
            continue
        end
        for jp in 0:np_eff, jpert in 1:mpert
            j = cell.map[jpert, jp+1]
            if j > ws.ndim
                continue
            end
            ws.mat[offset+i-j, j, 1] += cell_mat[ipert, jpert, ip+1, jp+1]
        end
    end
end

# Copy the base system to both parities, impose the x = 0 parity conditions, and solve the
# leading n×n banded system of each parity by LAPACK banded LU.
function _parity_bcs_and_solve!(ws::GalerkinWorkspace, params::GGJParameters, Q::ComplexF64, n::Int)
    mpert = 3
    offset = ws.kl + ws.kl + 1

    # Copy base matrix/rhs for both parities
    ws.mat[:, :, 2] .= ws.mat[:, :, 1]
    ws.rhs[:, 2] .= ws.rhs[:, 1]

    # Boundary condition at x = 0 (parity BCs)
    cell1 = ws.cells[1]
    Imat0, _, _ = _physical_uv(params, Q, cell1.x_left)

    # IBP surface term at x = 0
    for ipert in 1:mpert, jpert in 1:mpert
        i = cell1.map[ipert, 1]   # ip=0
        j = cell1.map[jpert, 2]   # ip=1
        if i <= ws.ndim && j <= ws.ndim
            ws.mat[offset+i-j, j, 1] += Imat0[ipert, jpert]
            ws.mat[offset+i-j, j, 2] += Imat0[ipert, jpert]
        end
    end

    # Parity conditions at X = 0 (GWP2016 Sec. III), parity of the displacement Ξ (Eq. 34):
    # isol = 1 even (Δ₊, interchange), isol = 2 odd (Δ₋, tearing). The ip = 0 rows are replaced.
    for isol in 1:2
        # Zero out ip=0 rows in the global matrix
        for ipert in 1:mpert
            i = cell1.map[ipert, 1]  # ip=0 DOFs
            if i > ws.ndim
                continue
            end
            for jj in max(1, i-ws.kl):min(ws.ndim, i+ws.kl)
                ws.mat[offset+i-jj, jj, isol] = 0
            end
        end
        # Even: Ψ(0) = Ξ'(0) = Υ'(0) = 0; odd: Ψ'(0) = Ξ(0) = Υ(0) = 0 (ip = 1 slots are derivatives).
        if isol == 1
            i = cell1.map[1, 1]
            j = cell1.map[1, 1]
            ws.mat[offset+i-j, j, isol] = 1
            for ipert in 2:3
                i = cell1.map[ipert, 1]
                j = cell1.map[ipert, 2]
                ws.mat[offset+i-j, j, isol] = 1
            end
        else
            i = cell1.map[1, 1]
            j = cell1.map[1, 2]
            ws.mat[offset+i-j, j, isol] = 1
            for ipert in 2:3
                i = cell1.map[ipert, 1]
                j = cell1.map[ipert, 1]
                ws.mat[offset+i-j, j, isol] = 1
            end
        end
        for ipert in 1:mpert
            i = cell1.map[ipert, 1]
            if i <= ws.ndim
                ws.rhs[i, isol] = 0
            end
        end
    end

    # Solve for each parity using LAPACK banded LU (gbtrf! + gbtrs!)
    kl = ws.kl
    ku = kl
    ab = @view ws.ab_buf[:, 1:n]
    rb = @view ws.rhs_buf[1:n]
    for isol in 1:2
        copyto!(ab, @view(ws.mat[:, 1:n, isol]))
        copyto!(rb, @view(ws.rhs[1:n, isol]))
        _, ipiv = LinearAlgebra.LAPACK.gbtrf!(kl, ku, n, ab)
        LinearAlgebra.LAPACK.gbtrs!('N', kl, ku, n, ab, ipiv, rb)
        ws.sol[1:n, isol] .= rb
    end
end

"""
    _assemble_and_solve_point!(ws, params, Q, θ, Us, Ub, E; nq=6) -> (Δraw, cexp)

Hermite Galerkin solve on the all-`CT_NONE` grid of `ws` along the ray x = e^{iθ}s, with the
far-field condition imposed strongly at the last node: its state is v = Us + Δ·Ub + c₁E₁ + c₂E₂,
with Us, Ub, E given as (Ψ, Ξ, Υ, Ψ', Ξ', Υ') in x-derivatives. The last node's six DOFs are
replaced by (Δ, c₁, c₂), tested in the same three directions. Returns the raw Δ and decaying-pair coefficients per parity
(isol = 1 even/Δ₊, 2 odd/Δ₋); `ws.sol` holds the full nodal solution afterwards.
"""
function _assemble_and_solve_point!(ws::GalerkinWorkspace, params::GGJParameters, Q::ComplexF64,
    θ::Float64, Us::Vector{ComplexF64}, Ub::Vector{ComplexF64}, E::Matrix{ComplexF64}; nq::Int=6)
    quad_nodes, quad_weights = gausslobatto(nq + 1)
    offset = ws.kl + ws.kl + 1
    N = ws.nx
    fill!(ws.mat, 0)
    fill!(ws.rhs, 0)
    cell_mat = ws.cell_mat_buf
    for ix in 1:(N-1)
        _gauss_quad!(cell_mat, ws.cells[ix], quad_nodes, quad_weights, params, Q; θ=θ)
        _scatter_cell!(ws, ws.cells[ix], cell_mat, ws.cells[ix].np)
    end

    # Last cell as a dense 12×12 block, local DOF index 3·ip + ipert (ip = 0..3 Hermite slots); the
    # far-field state is imposed at its right node.
    cell = ws.cells[N]
    _gauss_quad!(cell_mat, cell, quad_nodes, quad_weights, params, Q; θ=θ)
    Kc = zeros(ComplexF64, 12, 12)
    for ip in 0:3, jp in 0:3, ipert in 1:3, jpert in 1:3
        Kc[3ip+ipert, 3jp+jpert] = cell_mat[ipert, jpert, ip+1, jp+1]
    end
    # IBP boundary term at the far end, mirroring the x = 0 term of `_parity_bcs_and_solve!` (I is x-independent).
    Imat, _, _ = _physical_uv(params, Q, 0.0)
    for ipert in 1:3, jpert in 1:3
        Kc[6+ipert, 9+jpert] -= Imat[ipert, jpert]
    end

    # Far-field trace in DOF units (values, s-derivatives = e^{iθ}·x-derivatives); Ub normalized.
    dof(v) = ComplexF64[v[1], v[2], v[3], cis(θ)*v[4], cis(θ)*v[5], cis(θ)*v[6]]
    βb = norm(Ub)
    Bd = hcat(dof(Ub) ./ βb, dof(E[:, 1]), dof(E[:, 2]))
    usd = dof(Us)
    # Tl maps the 9 reduced unknowns (left node's 6 DOFs, Δ̃, c₁, c₂) to the cell's 12 Hermite DOFs; test space = Tlᵀ.
    Tl = zeros(ComplexF64, 12, 9)
    for k in 1:6
        Tl[k, k] = 1
    end
    Tl[7:12, 7:9] .= Bd
    K9 = transpose(Tl) * Kc * Tl
    r9 = -(transpose(Tl) * (Kc[:, 7:12] * usd))
    # (Δ̃, c₁, c₂) occupy the last node's value slots; its derivative slots drop out of the reduced system.
    yslots = cell.map[:, 3]
    glob = vcat(cell.map[:, 1], cell.map[:, 2], yslots)
    for a in 1:9
        i = glob[a]
        ws.rhs[i, 1] += r9[a]
        for b in 1:9
            j = glob[b]
            ws.mat[offset+i-j, j, 1] += K9[a, b]
        end
    end

    _parity_bcs_and_solve!(ws, params, Q, yslots[end])

    Δraw = MVector{2,ComplexF64}(0, 0)
    cexp = zeros(MMatrix{2,2,ComplexF64})
    for isol in 1:2
        y = ws.sol[yslots, isol]
        Δraw[isol] = y[1] / βb
        cexp[1, isol] = y[2]
        cexp[2, isol] = y[3]
        ws.sol[vcat(cell.map[:, 3], cell.map[:, 4]), isol] .= usd .+ Bd * y   # reconstruct the last node for profiles
    end
    return SVector(Δraw), SMatrix(cexp)
end

# Profiles (Ψ, Ξ, Υ) from the Hermite DOFs, columns 1 = even/Δ₊, 2 = odd/Δ₋; CT_EXT/CT_RES cells are
# skipped because their solution lives in the asymptotic emap DOF.
function _solution_profile(ws::GalerkinWorkspace; npc::Int=10)
    mpert = 3
    # Every Hermite cell contributes npc points.
    npts = 0
    for cell in ws.cells
        (cell.etype == CT_RES || cell.etype == CT_EXT) && continue
        cell.np >= 0 || continue
        npts += npc
    end
    xs = Vector{Float64}(undef, npts)
    Ψ = Matrix{ComplexF64}(undef, npts, 2)
    Ξ = Matrix{ComplexF64}(undef, npts, 2)
    Υ = Matrix{ComplexF64}(undef, npts, 2)
    vals = zeros(ComplexF64, mpert, 2)
    k = 0
    for cell in ws.cells
        (cell.etype == CT_RES || cell.etype == CT_EXT) && continue
        cell.np >= 0 || continue
        for t in range(0, 1; length=npc + 1)[1:(end-1)]
            x = cell.x_left + t * (cell.x_right - cell.x_left)
            pb, _ = _hermite(x, cell.x_left, cell.x_right)
            fill!(vals, 0)
            for isol in 1:2, ipert in 1:mpert, ip in 0:cell.np
                g = cell.map[ipert, ip+1]
                g <= ws.ndim && (vals[ipert, isol] += ws.sol[g, isol] * pb[ip+1])
            end
            k += 1
            xs[k] = x
            Ψ[k, 1] = vals[1, 1]
            Ψ[k, 2] = vals[1, 2]
            Ξ[k, 1] = vals[2, 1]
            Ξ[k, 2] = vals[2, 2]
            Υ[k, 1] = vals[3, 1]
            Υ[k, 2] = vals[3, 2]
        end
    end
    p = sortperm(xs)
    return (; x=xs[p], Ψ=Ψ[p, :], Ξ=Ξ[p, :], Υ=Υ[p, :])
end

# Stock Galerkin solve shared by `solve_inner` and `solve_inner_profile`. The matching data are the
# resonant cell's emap DOF, the coefficient of the large resonant solution (GWP2016 Eqs. 34–35), per
# parity isol = 1 (even, Δ₊), 2 (odd, Δ₋); returns them rescaled, with the solved workspace.
function _galerkin_solve(params::GGJParameters, γ::Number;
    kmax::Int=8, nx::Int=512, nq::Int=4, pfac::Float64=1.0,
    cutoff::Int=5, xfac::Float64=1.0, tol_res::Float64=1e-5)
    Q = inner_Q(params, γ)
    xmax_info = _xmax_3level(params, Q; kmax=kmax, xfac=xfac)
    ws = _build_grid_and_workspace(nx, xmax_info.xmax, xmax_info.dx1, xmax_info.dx2, pfac, cutoff)
    _assemble_and_solve!(ws, params, Q, xmax_info.cache; nq=nq, tol_res=tol_res)
    emap1 = ws.cells[ws.nx].emap[1]
    return rescale_delta(SVector{2,ComplexF64}(ws.sol[emap1, 1], ws.sol[emap1, 2]), params), ws
end

"""
    solve_inner_profile(::GGJModel{:galerkin}, params::GGJParameters, γ::Number; kwargs...)
        -> (; Δ, x, Ψ, Ξ, dψdx, rescale)

Hermite-FEM implementation of the [`solve_inner_profile`](@ref) interface:
real-axis solve, so `Δ` and the profiles come from the same solution. Same
numerics/kwargs as `solve_inner(GGJModel(; solver=:galerkin), ...)`.
"""
function solve_inner_profile(::_BareGGJ{:galerkin}, params::GGJParameters, γ::Number; kwargs...)
    Δ, ws = _galerkin_solve(params, γ; kwargs...)
    prof = _solution_profile(ws)
    return (; Δ=Δ, x=prof.x, Ψ=prof.Ψ, Ξ=prof.Ξ, _profile_conversions(params)...)
end

"""
    solve_inner(::GGJModel{:galerkin}, params::GGJParameters, γ::Number;
                kmax::Int=8, nx::Int=512, nq::Int=4, pfac::Float64=1.0,
                cutoff::Int=5, xfac::Float64=1.0, tol_res::Float64=1e-5)
                -> InnerLayerResponse

Solve the GGJ inner-layer matching problem using the Hermite-cubic finite
element (Galerkin) method (GWP2016 Sec. III), in the "resonant + noexp + inps"
configuration.

Returns the parity-projected matching data (GWP2016 Eqs. 34–35), rescaled by
[`rescale_delta`](@ref), as an `InnerLayerResponse`: `interchange` is the even-parity Δ₊
(isol = 1), `tearing` the odd-parity Δ₋ (isol = 2).

Knobs: `kmax` series order; `nx` cell count; `nq` Gauss–Lobatto order; `pfac` grid packing;
`cutoff` extension-cell span (cells nx−cutoff−1 … nx−1); `xfac` scale on the two inner matching radii; `tol_res`
resonant-cell quadrature tolerance.
"""
function solve_inner(::_BareGGJ{:galerkin}, params::GGJParameters, γ::Number; kwargs...)
    Δ, _ = _galerkin_solve(params, γ; kwargs...)
    return InnerLayerResponse(; interchange=Δ[1], tearing=Δ[2])
end
