# Galerkin.jl
#
# Generalized singular-Galerkin Hermite finite-element solver for the shared
# Wasow–Galerkin engine. Refactor of the GGJ-specific `GGJ/Galerkin.jl`: the
# physical-component count is `spec.ncomp` (was a literal 3), the FEM coefficient
# matrices come from `spec.coeffs`, the asymptotic large-x basis from
# `spec.physical` ∘ `evaluate_wasow`, and the half-domain boundary conditions
# from `spec.bc`. The half-domain problem (x ∈ [0, xmax]) is solved once per
# parity with parity boundary conditions at x = 0 and asymptotic matching at
# x = xmax, in the "resonant + noexp + inps" configuration of deltac.f.
#
# Hermite elements are cubic (`np = 3`, 4 DOFs/node); the engine asserts this.
# The algebraic matching uses one large + one small algebraic solution
# (`n_alg = 2`); a higher matching-mode count is a C_LAYER Phase-4 input.
#
# Reference: A. H. Glasser, Z. R. Wang & J.-K. Park, Phys. Plasmas 23, 112506 (2016).

using LinearAlgebra
using FastGaussQuadrature: gausslobatto
using QuadGK: quadgk
using StaticArrays

# Evaluate the physical asymptotic basis (ncomp × n_alg) and its derivative at x.
# For x ≥ 0 (the half-domain path, and the right half of the full domain) this is
# the plain asymptotic evaluation. For x < 0 (the full-domain left end) the basis
# is continued from the right by the system's parity: with per-component parity
# σ_c (spec.component_parity), a solution u_c(x) maps to the solution
# u_c(-x) = σ_c u_c(|x|), u'_c(-x) = -σ_c u'_c(|x|). For GGJ σ = (+1,-1,-1) (W
# even, N/Θ odd); a uniform flip would be wrong because GGJ is mixed-parity. The
# half-domain path never evaluates x < 0, so it is unaffected.
function _eval_physical(spec::SystemSpec, cache::WasowCache, x::Real)
    if x >= 0
        U, dU = evaluate_wasow(cache, x; derivative=true, apply_T=true)
        return spec.physical(U, dU, x)
    else
        U, dU = evaluate_wasow(cache, -x; derivative=true, apply_T=true)
        ua, dua = spec.physical(U, dU, -x)
        σ = spec.component_parity
        @inbounds for c in 1:spec.ncomp
            for j in axes(ua, 2)
                ua[c, j] *= σ[c]
                dua[c, j] *= -σ[c]
            end
        end
        return ua, dua
    end
end

# Hermite cubic basis functions (port of deltac_hermite). Returns (pb, qb) with
# pb = values, qb = derivatives, indexed 1:4 → Fortran 0:3.
function _hermite(x::Real, x0::Real, x1::Real)
    dx = x1 - x0
    t0 = (x - x0) / dx
    t1 = 1 - t0
    t02 = t0 * t0
    t12 = t1 * t1
    pb = SVector{4,Float64}(t12 * (1 + 2t0), t12 * t0 * dx, t02 * (1 + 2t1), -t02 * t1 * dx)
    qb = SVector{4,Float64}(-6t0 * t1 / dx, t0 * (3t0 - 4) + 1, 6t1 * t0 / dx, t0 * (3t0 - 2))
    return pb, qb
end

# Grid packing (port of deltac_pack).
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
    else
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
# Cell types and grid construction.
# -----------------------------------------------------------------------

@enum CellType CT_NONE CT_EXT2 CT_EXT1 CT_EXT CT_RES

struct GalerkinCell
    etype::CellType
    x_left::Float64
    x_right::Float64
    x_lsode::Float64
    np::Int
    map::Matrix{Int}
    emap::Vector{Int}
end

struct GalerkinWorkspace
    cells::Vector{GalerkinCell}
    ndim::Int
    nx::Int
    kl::Int
    nparity::Int
    mat::Array{ComplexF64,3}   # (ldab, ndim, nparity)
    rhs::Matrix{ComplexF64}    # (ndim, nparity)
    sol::Matrix{ComplexF64}    # (ndim, nparity)
end

function _build_grid_and_workspace(spec::SystemSpec, nx::Int, xmax::Float64,
    dx1::Float64, dx2::Float64, pfac::Float64,
    cutoff::Int, nparity::Int)
    nc = spec.ncomp
    np = spec.np
    side = pfac < 1 ? "left" : "right"

    etypes = fill(CT_NONE, nx)
    etypes[nx] = CT_RES
    etypes[nx-1] = CT_EXT
    for ix in (nx-2):-1:(nx-cutoff)
        etypes[ix] = CT_EXT1
    end
    etypes[nx-cutoff-1] = CT_EXT2

    x_nodes = zeros(Float64, nx + 1)
    x_nodes[1] = 0.0
    x_nodes[nx+1] = xmax
    x_nodes[nx] = xmax - dx1
    x_nodes[nx-1] = xmax - (dx1 + dx2)
    ixmax = nx - 2

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

    cells = Vector{GalerkinCell}(undef, nx)
    imap = 1
    for ix in 1:nx
        et = etypes[ix]
        xl = x_nodes[ix]
        xr = x_nodes[ix+1]

        cell_np = if et == CT_NONE || et == CT_EXT1 || et == CT_EXT2
            np
        elseif et == CT_EXT
            1
        else
            -1
        end

        x_lsode = (et == CT_RES) ? xr : 0.0

        if et == CT_RES
            map_local = zeros(Int, nc, 1)
            emap_local = zeros(Int, 1)
            cells[ix] = GalerkinCell(et, xl, xr, x_lsode, cell_np, map_local, emap_local)
            continue
        end

        if imap > 1
            imap -= 2 * nc
        end
        map_local = zeros(Int, nc, cell_np + 1)
        for ip in 0:cell_np
            for ipert in 1:nc
                map_local[ipert, ip+1] = imap
                imap += 1
            end
        end

        emap_local = Int[]
        if et == CT_EXT
            emap_local = collect(imap:(imap+nc-1))
            map_extra = collect(imap:(imap+nc-1))
            map_local = hcat(map_local, map_extra)
        end

        cells[ix] = GalerkinCell(et, xl, xr, x_lsode, cell_np, map_local, emap_local)
    end

    ext_cell = cells[nx-1]
    emap_vals = ext_cell.emap
    res_map = reshape(copy(emap_vals), nc, 1)
    cells[nx] = GalerkinCell(CT_RES, x_nodes[nx], x_nodes[nx+1], x_nodes[nx+1],
        -1, res_map, copy(emap_vals))

    ndim = imap

    kl = nc * (np + 1)
    ku = kl
    ldab = 2kl + ku + 1

    mat = zeros(ComplexF64, ldab, ndim, nparity)
    rhs = zeros(ComplexF64, ndim, nparity)
    sol = zeros(ComplexF64, ndim, nparity)

    return GalerkinWorkspace(cells, ndim, nx, kl, nparity, mat, rhs, sol)
end

# -----------------------------------------------------------------------
# Local assembly: Gauss quadrature for interior cells (port of deltac_gauss_quad).
# -----------------------------------------------------------------------

function _gauss_quad!(cell_mat::Array{ComplexF64,4}, cell::GalerkinCell,
    quad_nodes::Vector{Float64}, quad_weights::Vector{Float64},
    spec::SystemSpec, params, Q::ComplexF64)
    nc = spec.ncomp
    np_cell = cell.np
    nq = length(quad_nodes)
    x0 = (cell.x_right + cell.x_left) / 2
    dx = (cell.x_right - cell.x_left) / 2
    fill!(cell_mat, 0)

    @inbounds for iq in 1:nq
        x = x0 + dx * quad_nodes[iq]
        w = dx * quad_weights[iq]
        Imat, Umat, Vmat = spec.coeffs(params, Q, x)
        pb, qb = _hermite(x, cell.x_left, cell.x_right)
        for ip in 0:np_cell, jp in 0:np_cell
            for jpert in 1:nc, ipert in 1:nc
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

# -----------------------------------------------------------------------
# Extension assembly (port of deltac_extension): ext, ext1, ext2 cell types.
# -----------------------------------------------------------------------

function _extension!(cell_mat::Array{ComplexF64,4}, cell_rhs::Matrix{ComplexF64},
    cell::GalerkinCell, quad_nodes::Vector{Float64},
    quad_weights::Vector{Float64}, spec::SystemSpec, params,
    Q::ComplexF64, cache::WasowCache)
    np_cell = cell.np
    x0c = (cell.x_right + cell.x_left) / 2
    dxc = (cell.x_right - cell.x_left) / 2
    nq = length(quad_nodes)

    if cell.etype == CT_EXT
        ua_bdy, dua_bdy = _eval_physical(spec, cache, cell.x_right)
        ep = np_cell + 1

        @inbounds for iq in 1:nq
            x = x0c + dxc * quad_nodes[iq]
            w = dxc * quad_weights[iq]
            uax, duax = _eval_physical(spec, cache, x)
            Imat, Umat, Vmat = spec.coeffs(params, Q, x)
            pb, qb = _hermite(x, cell.x_left, cell.x_right)

            jp = 1  # large algebraic column (noexp)
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

            ua2_s = ua_bdy[:, 1] * pb[3] + dua_bdy[:, 1] * pb[4]
            dua2_s = ua_bdy[:, 1] * qb[3] + dua_bdy[:, 1] * qb[4]
            term = transpose(dua2_s) * Imat * dua1 + transpose(ua2_s) * Vmat * dua1 + transpose(ua2_s) * Umat * ua1
            cell_mat[1, 1, ep+1, ep+1] += w * term

            ua_drv = uax[:, 2]
            dua_drv = duax[:, 2]
            for ip in 0:np_cell
                term_r = qb[ip+1] * (Imat * dua_drv) + pb[ip+1] * (Vmat * dua_drv) + pb[ip+1] * (Umat * ua_drv)
                cell_rhs[:, ip+1] .-= w .* term_r
            end
            term_re = transpose(dua2_s) * Imat * dua_drv + transpose(ua2_s) * Vmat * dua_drv + transpose(ua2_s) * Umat * ua_drv
            cell_rhs[1, ep+1] -= w * term_re
        end

    elseif cell.etype == CT_EXT1 || cell.etype == CT_EXT2
        @inbounds for iq in 1:nq
            x = x0c + dxc * quad_nodes[iq]
            w = dxc * quad_weights[iq]
            Imat, Umat, Vmat = spec.coeffs(params, Q, x)
            pb, qb = _hermite(x, cell.x_left, cell.x_right)

            if cell.etype == CT_EXT1
                uax, duax = _eval_physical(spec, cache, x)
                ua_drv = uax[:, 2]
                dua_drv = duax[:, 2]
            else
                ua_bdy, dua_bdy = _eval_physical(spec, cache, cell.x_right)
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
# Resonant integral (replaces deltac_lsode_int with QuadGK).
# -----------------------------------------------------------------------

function _resonant_integral(cell::GalerkinCell, spec::SystemSpec, params,
    Q::ComplexF64, cache::WasowCache; tol::Float64=1e-5)
    x_left = cell.x_left
    x_right = cell.x_lsode

    function integrand_11(x)
        ua, dua = _eval_physical(spec, cache, x)
        Imat, Umat, Vmat = spec.coeffs(params, Q, x)
        ua1 = ua[:, 1]
        dua1 = dua[:, 1]
        return transpose(dua1) * Imat * dua1 + transpose(ua1) * Vmat * dua1 + transpose(ua1) * Umat * ua1
    end
    function integrand_12(x)
        ua, dua = _eval_physical(spec, cache, x)
        Imat, Umat, Vmat = spec.coeffs(params, Q, x)
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
# Global assembly and parity solve.
# -----------------------------------------------------------------------

function _assemble_and_solve!(ws::GalerkinWorkspace, spec::SystemSpec, params,
    Q::ComplexF64, cache::WasowCache; nq::Int=4,
    tol_res::Float64=1e-5)
    nc = spec.ncomp
    nparity = ws.nparity
    quad_nodes, quad_weights = gausslobatto(nq + 1)
    offset = ws.kl + ws.kl + 1

    fill!(ws.mat, 0)
    fill!(ws.rhs, 0)

    for ix in 1:ws.nx
        cell = ws.cells[ix]

        if cell.np >= 0
            np_eff = cell.np
            cell_mat = zeros(ComplexF64, nc, nc, np_eff + 1, np_eff + 1)
            _gauss_quad!(cell_mat, cell, quad_nodes, quad_weights, spec, params, Q)
            for ip in 0:np_eff, ipert in 1:nc
                i = cell.map[ipert, ip+1]
                if i > ws.ndim
                    continue
                end
                for jp in 0:np_eff, jpert in 1:nc
                    j = cell.map[jpert, jp+1]
                    if j > ws.ndim
                        continue
                    end
                    ws.mat[offset+i-j, j, 1] += cell_mat[ipert, jpert, ip+1, jp+1]
                end
            end
        end

        if cell.etype in (CT_EXT, CT_EXT1, CT_EXT2)
            if cell.etype == CT_EXT
                cell_mat_ext = zeros(ComplexF64, nc, nc, cell.np + 2, cell.np + 2)
                cell_rhs_ext = zeros(ComplexF64, nc, cell.np + 2)
            else
                cell_mat_ext = zeros(ComplexF64, nc, nc, cell.np + 1, cell.np + 1)
                cell_rhs_ext = zeros(ComplexF64, nc, cell.np + 1)
            end
            _extension!(cell_mat_ext, cell_rhs_ext, cell, quad_nodes, quad_weights, spec, params, Q, cache)

            npp = size(cell_mat_ext, 3) - 1
            for ip in 0:npp, ipert in 1:nc
                i = ip < size(cell.map, 2) ? cell.map[ipert, ip+1] : cell.emap[1]
                if cell.etype == CT_EXT && ip == cell.np + 1 && ipert > 1
                    continue
                end
                if i > ws.ndim
                    continue
                end
                for jp in 0:npp, jpert in 1:nc
                    j = jp < size(cell.map, 2) ? cell.map[jpert, jp+1] : cell.emap[1]
                    if cell.etype == CT_EXT && jp == cell.np + 1 && jpert > 1
                        continue
                    end
                    if j > ws.ndim
                        continue
                    end
                    ws.mat[offset+i-j, j, 1] += cell_mat_ext[ipert, jpert, ip+1, jp+1]
                end
                ws.rhs[i, 1] += cell_rhs_ext[ipert, ip+1]
            end
        end

        if cell.etype == CT_RES
            val_11, val_12 = _resonant_integral(cell, spec, params, Q, cache; tol=tol_res)
            emap1 = cell.emap[1]
            if emap1 <= ws.ndim
                ws.mat[offset, emap1, 1] += val_11
                ws.rhs[emap1, 1] -= val_12
            end
        end
    end

    # Surface term at xmax for the resonant method.
    res_cell = ws.cells[ws.nx]
    x_lsode = res_cell.x_lsode
    ua_xmax, dua_xmax = _eval_physical(spec, cache, x_lsode)
    Imat_xmax, _, _ = spec.coeffs(params, Q, x_lsode)
    emap1 = res_cell.emap[1]
    if emap1 <= ws.ndim
        surf_11 = transpose(ua_xmax[:, 1]) * Imat_xmax * dua_xmax[:, 1]
        ws.mat[offset, emap1, 1] -= surf_11
        surf_12 = transpose(ua_xmax[:, 1]) * Imat_xmax * dua_xmax[:, 2]
        ws.rhs[emap1, 1] += surf_12
    end

    # Replicate the base matrix/rhs across the parity solves.
    for isol in 2:nparity
        ws.mat[:, :, isol] .= ws.mat[:, :, 1]
        ws.rhs[:, isol] .= ws.rhs[:, 1]
    end

    # Boundary surface term at x = 0 (IBP) for all parities.
    cell1 = ws.cells[1]
    Imat0, _, _ = spec.coeffs(params, Q, cell1.x_left)
    for ipert in 1:nc, jpert in 1:nc
        i = cell1.map[ipert, 1]
        j = cell1.map[jpert, 2]
        if i <= ws.ndim && j <= ws.ndim
            for isol in 1:nparity
                ws.mat[offset+i-j, j, isol] += Imat0[ipert, jpert]
            end
        end
    end

    # Apply parity boundary conditions at x = 0 (from spec.bc).
    for isol in 1:nparity
        selectors = spec.bc.parities[isol]
        for ipert in 1:nc
            i = cell1.map[ipert, 1]
            if i > ws.ndim
                continue
            end
            for jj in max(1, i - ws.kl):min(ws.ndim, i + ws.kl)
                ws.mat[offset+i-jj, jj, isol] = 0
            end
        end
        for ipert in 1:nc
            kind = selectors[ipert]
            i = cell1.map[ipert, 1]
            # DIRICHLET pins the value DOF (ip=0); NEUMANN pins the derivative DOF (ip=1).
            j = kind == DIRICHLET ? cell1.map[ipert, 1] : cell1.map[ipert, 2]
            ws.mat[offset+i-j, j, isol] = 1
        end
        for ipert in 1:nc
            i = cell1.map[ipert, 1]
            if i <= ws.ndim
                ws.rhs[i, isol] = 0
            end
        end
    end

    # Banded LU solve per parity.
    n = ws.ndim
    kl = ws.kl
    ku = kl
    for isol in 1:nparity
        ab = copy(ws.mat[:, :, isol])
        rhs_col = copy(ws.rhs[:, isol])
        ab, ipiv = LinearAlgebra.LAPACK.gbtrf!(kl, ku, n, ab)
        LinearAlgebra.LAPACK.gbtrs!('N', kl, ku, n, ab, ipiv, rhs_col)
        ws.sol[:, isol] .= rhs_col
    end
end

# -----------------------------------------------------------------------
# Top-level half-domain Galerkin solve.
# -----------------------------------------------------------------------

"""
    solve_galerkin(spec::SystemSpec, params, Q::ComplexF64;
                   kmax=8, nx=512, nq=4, pfac=1.0, cutoff=5, xfac=1.0,
                   tol_res=1e-5) -> SVector{2,ComplexF64}

Solve the half-domain inner-layer matching problem for `spec` with the
singular-Galerkin Hermite FEM, returning the **raw** matching coefficient for
each parity (in `spec.bc` order), before any model-specific reordering or
rescaling. Requires `spec.domain == :half` and exactly two parities.
"""
function solve_galerkin(spec::SystemSpec, params, Q::ComplexF64;
    kmax::Int=8, nx::Int=512, nq::Int=4, pfac::Float64=1.0,
    cutoff::Int=5, xfac::Float64=1.0, tol_res::Float64=1e-5)
    spec.domain == :half || throw(ArgumentError("solve_galerkin: spec.domain must be :half (got $(spec.domain)); use solve_galerkin_full"))
    spec.np == 3 || throw(ArgumentError("solve_galerkin: Hermite order np must be 3 (cubic); got $(spec.np)"))
    nparity = nparities(spec.bc)
    nparity == 2 || throw(ArgumentError("solve_galerkin: half-domain contract expects 2 parities, got $nparity"))

    info = xmax_3level(spec, params, Q; kmax=kmax, xfac=xfac)
    cache = info.cache
    ws = _build_grid_and_workspace(spec, nx, info.xmax, info.dx1, info.dx2, pfac, cutoff, nparity)
    _assemble_and_solve!(ws, spec, params, Q, cache; nq=nq, tol_res=tol_res)

    res_cell = ws.cells[ws.nx]
    emap1 = res_cell.emap[1]
    return SVector{2,ComplexF64}(ws.sol[emap1, 1], ws.sol[emap1, 2])
end

solve_galerkin(spec::SystemSpec, params, Q::Number; kwargs...) =
    solve_galerkin(spec, params, ComplexF64(Q); kwargs...)
