# FullDomain.jl
#
# Full-domain (parity-coupled) inner-layer solve for the Wasow–Galerkin engine.
#
# Whereas the half-domain path folds the problem to X ∈ [0, Xmax] and imposes a
# parity boundary condition at the rational surface (X = 0), the full-domain path
# discretizes X ∈ [−Xmax, +Xmax] directly, with asymptotic matching at *both*
# ends and no parity reduction. The inner-layer coefficient matrices are regular
# at X = 0 (the rational-surface singularity is resolved in the stretched
# coordinate), so X = 0 is an ordinary interior point.
#
# This path is required when the inner-layer physics breaks the X → −X parity of
# the single-fluid system (e.g. a rotating two-fluid drift-MHD layer). The
# present implementation is the domain-structure **scaffolding** described in the
# project plan: it constructs the genuine two-sided grid (matching cells at both
# ends, regular interior across X = 0), assembles, and returns a
# [`FullDomainMatching`](@ref) object that subsumes the half-domain parity pair.
# The negative-side asymptotic-basis convention (see `_eval_physical`) and the
# physical validation of the matching values are finalized at C_LAYER Phase 4,
# where the parity-broken two-fluid asymptotics define the left-end basis.

"""
    FullDomainMatching

Matching data from a full-domain inner-layer solve. Generalizes the half-domain
`(Δ_odd, Δ_even)` pair: `amp` holds the raw matching coefficient extracted at
each end of the domain (`amp[1]` at −Xmax, `amp[2]` at +Xmax) for the
correspondingly-driven solve. For a parity-symmetric system these recombine to
the half-domain parity pair; for a parity-broken system they are the independent
two-sided matching data.
"""
struct FullDomainMatching
    amp::SVector{2,ComplexF64}
    xmax::Float64
    ndim::Int
end

# Build the symmetric two-sided grid and workspace spanning [−xmax, +xmax].
# Cell layout (contiguous left → right over 2·nx cells): a resonant matching cell
# at each extreme end, an extension cell just inside each end, ext1/ext2 driving
# cells further in, and regular Gauss-quadrature cells through the interior
# (including across X = 0). DOFs are numbered by the same overlapping Hermite
# walk as the half domain; the two resonant cells share the asymptotic DOFs of
# their adjacent extension cells.
function _build_full_grid_and_workspace(spec::SystemSpec, nx::Int, xmax::Float64,
    dx1::Float64, dx2::Float64, pfac::Float64,
    cutoff::Int)
    nc = spec.ncomp
    np = spec.np
    ncell = 2 * nx
    side = pfac < 1 ? "left" : "right"

    # Positive-half packed nodes (0 … xmax), identical to the half-domain grid.
    pos = zeros(Float64, nx + 1)
    pos[1] = 0.0
    pos[nx+1] = xmax
    pos[nx] = xmax - dx1
    pos[nx-1] = xmax - (dx1 + dx2)
    ixmax = nx - 2
    x1_packed = pos[ixmax+1]
    xm = x1_packed / 2
    dxp = x1_packed / 2
    mx = ixmax ÷ 2
    packed = xm .+ dxp .* _pack(mx, pfac, side)
    for i in 1:(ixmax+1)
        pos[i] = packed[i]
    end
    pos[1] = 0.0
    pos[ixmax+1] = x1_packed

    # Full node array: mirror the positive half to the negative side, 0 at center.
    nodes = zeros(Float64, ncell + 1)
    for i in 1:nx
        nodes[i] = -pos[nx+2-i]       # −xmax … (just below 0)
    end
    nodes[nx+1] = 0.0
    for i in 1:nx
        nodes[nx+1+i] = pos[i+1]     # (just above 0) … +xmax
    end

    # Cell types: resonant at both ends, extension just inside, ext1/ext2 driving
    # cells, regular interior in between.
    etypes = fill(CT_NONE, ncell)
    etypes[1] = CT_RES
    etypes[2] = CT_EXT
    for ic in 3:(2+cutoff-1)
        etypes[ic] = CT_EXT1
    end
    etypes[2+cutoff] = CT_EXT2
    etypes[ncell] = CT_RES
    etypes[ncell-1] = CT_EXT
    for ic in (ncell-2):-1:(ncell-cutoff)
        etypes[ic] = CT_EXT1
    end
    etypes[ncell-cutoff-1] = CT_EXT2

    cells = Vector{GalerkinCell}(undef, ncell)
    imap = 1
    for ic in 1:ncell
        et = etypes[ic]
        xl = nodes[ic]
        xr = nodes[ic+1]

        cell_np = if et == CT_NONE || et == CT_EXT1 || et == CT_EXT2
            np
        elseif et == CT_EXT
            1
        else
            -1
        end
        x_lsode = (et == CT_RES) ? (ic == 1 ? xl : xr) : 0.0

        if et == CT_RES
            cells[ic] = GalerkinCell(et, xl, xr, x_lsode, -1, zeros(Int, nc, 1), zeros(Int, 1))
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
            map_local = hcat(map_local, collect(imap:(imap+nc-1)))
        end
        cells[ic] = GalerkinCell(et, xl, xr, x_lsode, cell_np, map_local, emap_local)
    end

    # Resonant cells share the asymptotic DOFs of their adjacent extension cells.
    left_ext = cells[2]
    right_ext = cells[ncell-1]
    cells[1] = GalerkinCell(CT_RES, nodes[1], nodes[2], nodes[1], -1,
        reshape(copy(left_ext.emap), nc, 1), copy(left_ext.emap))
    cells[ncell] = GalerkinCell(CT_RES, nodes[ncell], nodes[ncell+1], nodes[ncell+1], -1,
        reshape(copy(right_ext.emap), nc, 1), copy(right_ext.emap))

    ndim = imap
    kl = nc * (np + 1)
    ku = kl
    ldab = 2kl + ku + 1
    mat = zeros(ComplexF64, ldab, ndim, 2)
    rhs = zeros(ComplexF64, ndim, 2)
    sol = zeros(ComplexF64, ndim, 2)
    return GalerkinWorkspace(cells, ndim, ncell, kl, 2, mat, rhs, sol)
end

# Assemble and solve the two-sided system. The two RHS columns separate the
# left-end (X < 0) and right-end (X > 0) driving (deltac.f fulldomain convention):
# left-half cells contribute to column 1, right-half cells to column 2.
function _assemble_and_solve_full!(ws::GalerkinWorkspace, spec::SystemSpec, params,
    Q::ComplexF64, cache::WasowCache; nq::Int=4,
    tol_res::Float64=1e-5)
    nc = spec.ncomp
    quad_nodes, quad_weights = gausslobatto(nq + 1)
    offset = ws.kl + ws.kl + 1
    fill!(ws.mat, 0)
    fill!(ws.rhs, 0)

    for ix in 1:ws.nx
        cell = ws.cells[ix]
        # Driving column: left half (cell midpoint < 0) → 1, right half → 2.
        col = (cell.x_left + cell.x_right) / 2 < 0 ? 1 : 2

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
                ws.rhs[i, col] += cell_rhs_ext[ipert, ip+1]
            end
        end

        if cell.etype == CT_RES
            val_11, val_12 = _resonant_integral(cell, spec, params, Q, cache; tol=tol_res)
            emap1 = cell.emap[1]
            if emap1 <= ws.ndim
                ws.mat[offset, emap1, 1] += val_11
                ws.rhs[emap1, col] -= val_12
            end
            # Surface term at this end.
            ua_e, dua_e = _eval_physical(spec, cache, cell.x_lsode)
            Imat_e, _, _ = spec.coeffs(params, Q, cell.x_lsode)
            if emap1 <= ws.ndim
                ws.mat[offset, emap1, 1] -= transpose(ua_e[:, 1]) * Imat_e * dua_e[:, 1]
                ws.rhs[emap1, col] += transpose(ua_e[:, 1]) * Imat_e * dua_e[:, 2]
            end
        end
    end

    # The stiffness matrix (column 1 of mat) is shared by both driving solves.
    ws.mat[:, :, 2] .= ws.mat[:, :, 1]

    n = ws.ndim
    kl = ws.kl
    ku = kl
    for isol in 1:2
        ab = copy(ws.mat[:, :, isol])
        rhs_col = copy(ws.rhs[:, isol])
        ab, ipiv = LinearAlgebra.LAPACK.gbtrf!(kl, ku, n, ab)
        LinearAlgebra.LAPACK.gbtrs!('N', kl, ku, n, ab, ipiv, rhs_col)
        ws.sol[:, isol] .= rhs_col
    end
end

"""
    solve_galerkin_full(spec::SystemSpec, params, Q::ComplexF64;
                        kmax=8, nx=512, nq=4, pfac=1.0, cutoff=5, xfac=1.0,
                        tol_res=1e-5) -> FullDomainMatching

Full-domain inner-layer solve for `spec` on X ∈ [−Xmax, +Xmax] (no parity
reduction). Builds the two-sided grid, assembles, solves the two-sided system,
and returns the [`FullDomainMatching`](@ref) data. See the file header for the
scaffolding status of this path.
"""
function solve_galerkin_full(spec::SystemSpec, params, Q::ComplexF64;
    kmax::Int=8, nx::Int=512, nq::Int=4, pfac::Float64=1.0,
    cutoff::Int=5, xfac::Float64=1.0, tol_res::Float64=1e-5)
    spec.np == 3 || throw(ArgumentError("solve_galerkin_full: Hermite order np must be 3 (cubic); got $(spec.np)"))
    info = xmax_3level(spec, params, Q; kmax=kmax, xfac=xfac)
    cache = info.cache
    ws = _build_full_grid_and_workspace(spec, nx, info.xmax, info.dx1, info.dx2, pfac, cutoff)
    _assemble_and_solve_full!(ws, spec, params, Q, cache; nq=nq, tol_res=tol_res)

    left_emap = ws.cells[1].emap[1]
    right_emap = ws.cells[ws.nx].emap[1]
    amp = SVector{2,ComplexF64}(ws.sol[left_emap, 1], ws.sol[right_emap, 2])
    return FullDomainMatching(amp, info.xmax, ws.ndim)
end

solve_galerkin_full(spec::SystemSpec, params, Q::Number; kwargs...) =
    solve_galerkin_full(spec, params, ComplexF64(Q); kwargs...)
