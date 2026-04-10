# Galerkin.jl
#
# Hermite-cubic finite element (Galerkin) solver for the GGJ inner-layer
# model. Direct port of rmatch/deltac.f in the "resonant + noexp + inps"
# configuration. The half-domain problem (x ∈ [0, xmax]) is solved with
# parity boundary conditions at x = 0 and asymptotic matching at x = xmax.
#
# Configuration assumed (matches deltac.f defaults when called with inps):
#   gal_method = "resonant"    method = true     noexp = true
#   basis_type = 0 (Hermite)   fulldomain = 0    inps_type = "inps"
#   mpert = 3 (W, N, Θ)        np = 3 (Hermite cubic → 4 DOFs/node)
#
# Reference: A. H. Glasser, Z. R. Wang & J.-K. Park,
# Phys. Plasmas 23, 112506 (2016).

using LinearAlgebra
using FastGaussQuadrature: gausslobatto
using QuadGK: quadgk

# -----------------------------------------------------------------------
# Physical-variable helpers (replacements for inpso_get_ua/dua/uv).
# -----------------------------------------------------------------------

"""
Convert the inps 6×2 output U_inps at coordinate `x` to the physical
(W, N, Θ) and (W', N', Θ') representation used by deltac/inpso.
Returns `(ua, dua)` each 3×2 complex, where columns are the two algebraic
solutions and rows are (W, N, Θ).
"""
function _physical_ua_dua(cache::InnerAsymptoticsCache, x::Real)
    U, dU = evaluate_asymptotics(cache, x; derivative=true, apply_T=true)
    ua  = zeros(ComplexF64, 3, 2)
    dua = zeros(ComplexF64, 3, 2)
    @inbounds for j in 1:2
        ua[1, j]  = U[1, j] / x
        ua[2, j]  = U[2, j]
        ua[3, j]  = U[3, j]
        dua[1, j] = U[4, j] - U[1, j] / (x * x)
        dua[2, j] = U[5, j] * x
        dua[3, j] = U[6, j] * x
    end
    return ua, dua
end

"""
Build the (I, U, V) coefficient matrices of the second-order system
`I·u'' − V·u' − U·u = 0` at coordinate `x`. Port of inpso_get_uv.
All matrices are 3×3 complex.
"""
function _physical_uv(params::GGJParameters, Q::ComplexF64, x::Real)
    e = ComplexF64(params.E); f = ComplexF64(params.F)
    h = ComplexF64(params.H); g = ComplexF64(params.G)
    k = ComplexF64(params.K); q = Q
    q2 = q * q
    x2 = x * x

    Imat = @SMatrix ComplexF64[1 0 0; 0 q2 0; 0 0 q]

    # Build U and V matching inpso_get_uv: pre-scaled from column-major RESHAPE, then row 2 *=q², row 3 *=q.
    # Pre-scaled U (column-major Fortran RESHAPE → Julia row-major):
    #   U[1,:] = (q,      -q*x,         0)
    #   U[2,:] = (-x/q,    x²/q,        -(e+f)/q²)
    #   U[3,:] = (-x/q,   -(g-ke)*q,     x²/q+(g+kf)*q)
    U = @SMatrix ComplexF64[
        q                (-q * x)                     0
        (-x / q) * q2    (x2 / q) * q2                (-(e + f) / q2) * q2
        (-x / q) * q     (-(g - k * e) * q) * q       (x2 / q + (g + k * f) * q) * q
    ]

    # Pre-scaled V (column-major Fortran RESHAPE → Julia row-major):
    #   V[1,:] = (0,       0,  h)
    #   V[2,:] = (-h/q²,   0,  0)
    #   V[3,:] = (h*k*q,   0,  0)
    V = @SMatrix ComplexF64[
        0                0  h
        (-h / q2) * q2   0  0
        (h * k * q) * q  0  0
    ]

    return Imat, U, V
end

# -----------------------------------------------------------------------
# Hermite cubic basis functions — port of deltac_hermite.
# Returns (pb[1:4], qb[1:4]) where pb = values, qb = derivatives,
# indexed 1:4 → Fortran 0:3.
# -----------------------------------------------------------------------

function _hermite(x::Real, x0::Real, x1::Real)
    dx = x1 - x0
    t0 = (x - x0) / dx
    t1 = 1 - t0
    t02 = t0 * t0; t12 = t1 * t1
    pb = SVector{4,Float64}(
        t12 * (1 + 2t0),
        t12 * t0 * dx,
        t02 * (1 + 2t1),
        -t02 * t1 * dx,
    )
    qb = SVector{4,Float64}(
        -6t0 * t1 / dx,
        t0 * (3t0 - 4) + 1,
        6t1 * t0 / dx,
        t0 * (3t0 - 2),
    )
    return pb, qb
end

# -----------------------------------------------------------------------
# Grid packing — port of deltac_pack.
# -----------------------------------------------------------------------

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

# -----------------------------------------------------------------------
# Three-level xmax sweep — replaces inpso_xmax for inps mode.
# Returns (xmax, dx1, dx2, cache).
# -----------------------------------------------------------------------

function _xmax_3level(params::GGJParameters, Q::ComplexF64;
                     kmax::Int=8, xfac::Float64=1.0,
                     eps_vec::NTuple{3,Float64}=(1e-2, 5e-7, 1e-7))
    cache = build_asymptotics(params, Q; kmax=kmax)
    dxfac = 10.0^0.01
    xmax_vals = zeros(Float64, 3)
    set = [true, true, true]
    x = 0.1
    for _ in 1:1000
        delta = asymptotic_residual(cache, x)
        dmax = maximum(delta)
        for i in 1:3
            if dmax < eps_vec[i] && set[i]
                xmax_vals[i] = x
                set[i] = false
            end
        end
        if !any(set)
            break
        end
        x *= dxfac
    end
    any(set) && error("_xmax_3level: failed to find all xmax levels")
    # Fortran 0-indexed: inps_xmaxx(1) → Julia xmax_vals[2], inps_xmaxx(2) → xmax_vals[3]
    xmax_vals[2] *= xfac
    xmax_vals[3] *= xfac
    xmax = xmax_vals[3]
    dx1 = xmax_vals[3] - xmax_vals[2]
    dx2 = dx1 / 2
    return (; xmax, dx1, dx2, cache)
end

# -----------------------------------------------------------------------
# Cell types and grid construction.
# -----------------------------------------------------------------------

@enum CellType CT_NONE CT_EXT2 CT_EXT1 CT_EXT CT_RES

struct GalerkinCell
    etype::CellType
    x_left::Float64
    x_right::Float64
    x_lsode::Float64     # right integration limit for res cell
    np::Int               # Hermite DOF count per component (-1 for res)
    map::Matrix{Int}      # (mpert, 0:np_eff) local-to-global DOF map
    emap::Vector{Int}     # extra (asymptotic) DOF indices (length 1 for noexp)
end

struct GalerkinWorkspace
    cells::Vector{GalerkinCell}
    ndim::Int
    nx::Int
    kl::Int
    mat::Array{ComplexF64,3}   # (ldab, ndim, 2) banded storage
    rhs::Matrix{ComplexF64}    # (ndim, 2)
    sol::Matrix{ComplexF64}    # (ndim, 2)
end

function _build_grid_and_workspace(nx::Int, xmax::Float64, dx1::Float64, dx2::Float64,
                                    pfac::Float64, cutoff::Int, nq::Int)
    mpert = 3
    np = 3
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

    x0 = x_nodes[1]; x1_packed = x_nodes[ixmax+1]
    xm = (x1_packed + x0) / 2; dxp = (x1_packed - x0) / 2
    mx = ixmax ÷ 2
    packed = xm .+ dxp .* _pack(mx, pfac, side)
    for i in 1:(ixmax+1)
        x_nodes[i] = packed[i]
    end
    x_nodes[1] = 0.0
    x_nodes[ixmax+1] = x1_packed

    # Build cells and local-to-global mapping
    cells = Vector{GalerkinCell}(undef, nx)
    imap = 1
    for ix in 1:nx
        et = etypes[ix]
        xl = x_nodes[ix]; xr = x_nodes[ix+1]

        cell_np = if et == CT_NONE || et == CT_EXT1 || et == CT_EXT2
            np
        elseif et == CT_EXT
            1  # method=true: left Hermite pair only + 1 extra
        else # CT_RES
            -1  # no Hermite DOFs
        end

        x_lsode = (et == CT_RES) ? xr : 0.0

        # Build map (matching deltac_make_map_hermite logic)
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
            # and are skipped during assembly. This matches Fortran convention.
            emap_local = [imap, imap + 1, imap + 2]
            map_extra = [imap; imap + 1; imap + 2]  # 3-element column
            map_local = hcat(map_local, map_extra)
        end

        cells[ix] = GalerkinCell(et, xl, xr, x_lsode, cell_np, map_local, emap_local)
    end

    # Set resonant cell mapping: shares emap with ext cell
    ext_cell = cells[nx-1]
    emap_vals = ext_cell.emap
    res_map = reshape(copy(emap_vals), mpert, 1)
    cells[nx] = GalerkinCell(CT_RES, x_nodes[nx], x_nodes[nx+1], x_nodes[nx+1],
                             -1, res_map, copy(emap_vals))

    ndim = imap  # noexp → ndim = imap (only emap[1] ≤ ndim)

    # Bandwidth
    kl = mpert * (np + 1) + 1 - 1  # resonant method with noexp
    ku = kl
    ldab = 2kl + ku + 1

    mat = zeros(ComplexF64, ldab, ndim, 2)
    rhs = zeros(ComplexF64, ndim, 2)
    sol = zeros(ComplexF64, ndim, 2)

    return GalerkinWorkspace(cells, ndim, nx, kl, mat, rhs, sol)
end

# -----------------------------------------------------------------------
# Local assembly: Gauss quadrature for "none" cells.
# Port of deltac_gauss_quad.
# -----------------------------------------------------------------------

function _gauss_quad!(cell_mat::Array{ComplexF64,4}, cell::GalerkinCell,
                     quad_nodes::Vector{Float64}, quad_weights::Vector{Float64},
                     params::GGJParameters, Q::ComplexF64)
    np_cell = cell.np
    nq = length(quad_nodes)
    x0 = (cell.x_right + cell.x_left) / 2
    dx = (cell.x_right - cell.x_left) / 2
    fill!(cell_mat, 0)

    @inbounds for iq in 1:nq
        x = x0 + dx * quad_nodes[iq]
        w = dx * quad_weights[iq]
        Imat, Umat, Vmat = _physical_uv(params, Q, x)
        pb, qb = _hermite(x, cell.x_left, cell.x_right)

        for ip in 0:np_cell, jp in 0:np_cell
            # cell_mat[:,:,ip+1,jp+1] += w*(I*qb[ip]*qb[jp] + V*pb[ip]*qb[jp] + U*pb[ip]*pb[jp])
            for jpert in 1:3, ipert in 1:3
                cell_mat[ipert, jpert, ip+1, jp+1] += w * (
                    Imat[ipert, jpert] * qb[ip+1] * qb[jp+1] +
                    Vmat[ipert, jpert] * pb[ip+1] * qb[jp+1] +
                    Umat[ipert, jpert] * pb[ip+1] * pb[jp+1]
                )
            end
        end
    end
end

# -----------------------------------------------------------------------
# Extension assembly — port of deltac_extension.
# Handles ext, ext1, ext2 cell types.
# -----------------------------------------------------------------------

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
        ua_bdy, dua_bdy = _physical_ua_dua(cache, cell.x_right)
        ep = np_cell + 1  # index of the extra DOF (1-based → ep+1 in array)

        @inbounds for iq in 1:nq
            x = x0c + dxc * quad_nodes[iq]
            w = dxc * quad_weights[iq]
            uax, duax = _physical_ua_dua(cache, x)
            Imat, Umat, Vmat = _physical_uv(params, Q, x)
            pb, qb = _hermite(x, cell.x_left, cell.x_right)

            # The asymptotic basis function in the ext cell uses Hermite blending
            # at the right node: phi_j(x) = ua_j * pb[3] + dua_j * pb[4]
            # (pb[3] = H_10, pb[4] = H_11 at right node)

            # (h_i, L*phi_j) for i=0:np_cell, j=1 (only large algebraic, noexp)
            # and (phi_j^T, L*h_i) for the transpose
            jp = 1  # only tid(1) = column 1 = large algebraic (with noexp)
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

            # RHS: driving from small algebraic (column 2 = tid(4))
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
                uax, duax = _physical_ua_dua(cache, x)
                ua_drv = uax[:, 2]
                dua_drv = duax[:, 2]
            else # CT_EXT2
                ua_bdy, dua_bdy = _physical_ua_dua(cache, cell.x_right)
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

# -----------------------------------------------------------------------
# Resonant integral — replaces deltac_lsode_int with QuadGK.
# Computes ∫_{x_left}^{x_lsode} ua_1^T L ua_j dx for j=1 and j=2.
# With noexp, only (1,1) and (1,2) entries of the 1×2 result matter
# (ip=1 only for test function, jp=1 for stiffness, jp=2 for RHS).
# -----------------------------------------------------------------------

function _resonant_integral(cell::GalerkinCell, params::GGJParameters,
                           Q::ComplexF64, cache::InnerAsymptoticsCache;
                           tol::Float64=1e-5)
    x_left = cell.x_left
    x_right = cell.x_lsode

    # u_res[1,1] = ∫ ua_1^T L ua_1 dx  (stiffness)
    # u_res[1,2] = ∫ ua_1^T L ua_2 dx  (RHS/driving)
    # Weak form after IBP: (test') I (trial') + (test) V (trial') + (test) U (trial)
    function integrand_11(x)
        ua, dua = _physical_ua_dua(cache, x)
        Imat, Umat, Vmat = _physical_uv(params, Q, x)
        ua1 = ua[:, 1]; dua1 = dua[:, 1]
        return transpose(dua1) * Imat * dua1 + transpose(ua1) * Vmat * dua1 + transpose(ua1) * Umat * ua1
    end

    function integrand_12(x)
        ua, dua = _physical_ua_dua(cache, x)
        Imat, Umat, Vmat = _physical_uv(params, Q, x)
        dua1 = dua[:, 1]
        ua1 = ua[:, 1]; ua2 = ua[:, 2]; dua2 = dua[:, 2]
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
    mpert = 3; np = 3
    quad_nodes, quad_weights = gausslobatto(nq + 1)
    offset = ws.kl + ws.kl + 1  # kl + ku + 1 since ku = kl

    fill!(ws.mat, 0)
    fill!(ws.rhs, 0)

    # Per-cell assembly
    for ix in 1:ws.nx
        cell = ws.cells[ix]

        # Gauss quadrature for Hermite contribution (all cell types)
        if cell.np >= 0
            np_eff = cell.np
            cell_mat = zeros(ComplexF64, mpert, mpert, np_eff + 1, np_eff + 1)
            _gauss_quad!(cell_mat, cell, quad_nodes, quad_weights, params, Q)

            # Assemble into global banded matrix (both parities use same base matrix)
            for ip in 0:np_eff, ipert in 1:mpert
                i = cell.map[ipert, ip+1]
                if i > ws.ndim; continue; end
                for jp in 0:np_eff, jpert in 1:mpert
                    j = cell.map[jpert, jp+1]
                    if j > ws.ndim; continue; end
                    ws.mat[offset + i - j, j, 1] += cell_mat[ipert, jpert, ip+1, jp+1]
                end
            end
        end

        # Extension terms
        if cell.etype in (CT_EXT, CT_EXT1, CT_EXT2)
            np_eff = cell.etype == CT_EXT ? cell.np + 1 : cell.np
            cell_mat_ext = zeros(ComplexF64, mpert, mpert, np_eff + 1, np_eff + 1)
            cell_rhs_ext = zeros(ComplexF64, mpert, np_eff + 1)
            # For ext, we need to create a temporary cell_mat that includes the extra DOF
            if cell.etype == CT_EXT
                cell_mat_ext = zeros(ComplexF64, mpert, mpert, cell.np + 2, cell.np + 2)
                cell_rhs_ext = zeros(ComplexF64, mpert, cell.np + 2)
            else
                cell_mat_ext = zeros(ComplexF64, mpert, mpert, cell.np + 1, cell.np + 1)
                cell_rhs_ext = zeros(ComplexF64, mpert, cell.np + 1)
            end
            _extension!(cell_mat_ext, cell_rhs_ext, cell, quad_nodes, quad_weights, params, Q, cache)

            # Assemble ext contributions
            npp = size(cell_mat_ext, 3) - 1
            for ip in 0:npp, ipert in 1:mpert
                i = ip < size(cell.map, 2) ? cell.map[ipert, ip+1] : cell.emap[1]
                # For the extra DOF, only ipert=1 is meaningful (noexp)
                if cell.etype == CT_EXT && ip == cell.np + 1 && ipert > 1
                    continue
                end
                if i > ws.ndim; continue; end
                for jp in 0:npp, jpert in 1:mpert
                    j = jp < size(cell.map, 2) ? cell.map[jpert, jp+1] : cell.emap[1]
                    if cell.etype == CT_EXT && jp == cell.np + 1 && jpert > 1
                        continue
                    end
                    if j > ws.ndim; continue; end
                    ws.mat[offset + i - j, j, 1] += cell_mat_ext[ipert, jpert, ip+1, jp+1]
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

    # Surface term at xmax for "resonant" method
    res_cell = ws.cells[ws.nx]
    x_lsode = res_cell.x_lsode
    ua_xmax, dua_xmax = _physical_ua_dua(cache, x_lsode)
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

    # Copy base matrix/rhs for both parities
    ws.mat[:, :, 2] .= ws.mat[:, :, 1]
    ws.rhs[:, 2] .= ws.rhs[:, 1]

    # Boundary condition at x = 0 (parity BCs)
    cell1 = ws.cells[1]
    Imat0, _, _ = _physical_uv(params, Q, cell1.x_left)

    # Restore surface term at x=0 (IBP boundary)
    for ipert in 1:mpert, jpert in 1:mpert
        i = cell1.map[ipert, 1]   # ip=0
        j = cell1.map[jpert, 2]   # ip=1
        if i <= ws.ndim && j <= ws.ndim
            ws.mat[offset + i - j, j, 1] += Imat0[ipert, jpert]
            ws.mat[offset + i - j, j, 2] += Imat0[ipert, jpert]
        end
    end

    # Apply parity BCs for each solution (isol=1: odd, isol=2: even).
    # Mirrors deltac_set_boundary: for each isol, build a modified local
    # matrix for ip=0..1 of cell 1, then write it into the global matrix.
    for isol in 1:2
        # Zero out ip=0 rows in the global matrix
        for ipert in 1:mpert
            i = cell1.map[ipert, 1]  # ip=0 DOFs
            if i > ws.ndim; continue; end
            for jj in max(1, i - ws.kl):min(ws.ndim, i + ws.kl)
                ws.mat[offset + i - jj, jj, isol] = 0
            end
        end
        # Odd parity (isol=1): W'(0)=0, N(0)=0, Θ(0)=0
        # → row=W(ip=0), col=W(ip=1): A[map[1,1], map[1,2]] = 1
        # → row=N(ip=0), col=N(ip=0): A[map[2,1], map[2,1]] = 1
        # → row=Θ(ip=0), col=Θ(ip=0): A[map[3,1], map[3,1]] = 1
        # Even parity (isol=2): W(0)=0, N'(0)=0, Θ'(0)=0
        # → row=W(ip=0), col=W(ip=0): A[map[1,1], map[1,1]] = 1
        # → row=N(ip=0), col=N(ip=1): A[map[2,1], map[2,2]] = 1
        # → row=Θ(ip=0), col=Θ(ip=1): A[map[3,1], map[3,2]] = 1
        if isol == 1
            i = cell1.map[1, 1]; j = cell1.map[1, 2]
            ws.mat[offset + i - j, j, isol] = 1
            for ipert in 2:3
                i = cell1.map[ipert, 1]; j = cell1.map[ipert, 1]
                ws.mat[offset + i - j, j, isol] = 1
            end
        else
            i = cell1.map[1, 1]; j = cell1.map[1, 1]
            ws.mat[offset + i - j, j, isol] = 1
            for ipert in 2:3
                i = cell1.map[ipert, 1]; j = cell1.map[ipert, 2]
                ws.mat[offset + i - j, j, isol] = 1
            end
        end
        for ipert in 1:mpert
            i = cell1.map[ipert, 1]
            if i <= ws.ndim
                ws.rhs[i, isol] = 0
            end
        end
    end

    # Solve for each parity
    for isol in 1:2
        ws.sol[:, isol] .= ws.rhs[:, isol]
        band_mat = @view ws.mat[:, :, isol]
        # Convert banded to dense for LU solve
        A = zeros(ComplexF64, ws.ndim, ws.ndim)
        for j in 1:ws.ndim
            for ii in max(1, j - ws.kl):min(ws.ndim, j + ws.kl)
                A[ii, j] = band_mat[offset + ii - j, j]
            end
        end
        ws.sol[:, isol] .= A \ ws.rhs[:, isol]
    end
end

# -----------------------------------------------------------------------
# Top-level Galerkin solver.
# -----------------------------------------------------------------------

"""
    solve_inner(::GGJModel{:galerkin}, params::GGJParameters, γ::Number;
                kmax::Int=8, nx::Int=128, nq::Int=4, pfac::Float64=1.0,
                cutoff::Int=5, xfac::Float64=1.0, tol_res::Float64=1e-5)
                -> SVector{2,ComplexF64}

Solve the GGJ inner-layer matching problem using the Hermite-cubic finite
element (Galerkin) method. Direct port of rmatch/deltac.f in the
"resonant + noexp + inps" configuration.

Returns `(Δ₁, Δ₂)` with rescaling applied. The ordering matches deltac.f's
output convention (swapped relative to deltar.f).
"""
function solve_inner(::GGJModel{:galerkin}, params::GGJParameters, γ::Number;
                     kmax::Int=8, nx::Int=128, nq::Int=4, pfac::Float64=1.0,
                     cutoff::Int=5, xfac::Float64=1.0, tol_res::Float64=1e-5)
    Q = inner_Q(params, γ)

    # Build asymptotic cache and determine grid parameters
    xmax_info = _xmax_3level(params, Q; kmax=kmax, xfac=xfac)
    cache = xmax_info.cache

    # Build grid and workspace
    ws = _build_grid_and_workspace(nx, xmax_info.xmax, xmax_info.dx1, xmax_info.dx2,
                                   pfac, cutoff, nq)

    # Assemble and solve
    _assemble_and_solve!(ws, params, Q, cache; nq=nq, tol_res=tol_res)

    # Extract delta from the resonant cell's emap DOF
    res_cell = ws.cells[ws.nx]
    emap1 = res_cell.emap[1]
    Δ_raw = SVector{2,ComplexF64}(ws.sol[emap1, 1], ws.sol[emap1, 2])

    # Apply deltac.f's swap convention (line 194-196)
    Δ_swapped = SVector{2,ComplexF64}(Δ_raw[2], Δ_raw[1])

    return rescale_delta(Δ_swapped, params)
end
