# GalerkinSolution.jl
#
# Reconstruct the outer-region radial displacement ξ(ψ) AND its analytic derivative ξ′(ψ) from the
# solved Galerkin coefficients. Ports:
#   gal_get_solution    (gal.f:1833-1997)  ξ (and optional ξ′) at one ψ for one solution column
#   gal_output_solution (gal.f:2005-2273)  evaluate (ξ, ξ′) on a gal-native packed grid, all columns
#
# The whole point of carrying ξ′ here is to evaluate it analytically from the Hermite derivative basis
# (gal_hermite qb) and the asymptotic-series derivative (sing_get_dua_gal) rather than ever
# spline-differentiating ξ at the packed edge — the spline-endpoint derivative was the root cause of
# GPEC's fake ~100% shielding (fixed in Fortran by RDCON_singcoup commit c3e8c5bc).
#
# Fortran module flags restore_uh/us/ul default .TRUE. (gal.f:81); the full reconstruction is always
# performed. The cut path (gal.f:1974-1992, used only for the matching diagnostic) is NOT ported here.

# Dense / coarse sampling per cell, matching Fortran interp_np_res / interp_np (gal.f:88).
const GAL_INTERP_NP = 3
const GAL_INTERP_NP_RES = 60

# Advance the monotone (iintvl, icell) cursor to the cell containing x. Port of gal.f:1863-1876.
@inline function gal_locate_cell(ws::GalWorkspace, x::Float64, iintvl::Int, icell::Int)
    msing = length(ws.intvl) - 1
    for _ in 1:((msing+1)*ws.nx+1)
        cell = ws.intvl[iintvl+1].cells[icell]
        if cell.x[1] <= x <= cell.x[2]
            return iintvl, icell, cell
        end
        icell += 1
        if icell > ws.nx
            icell = 1
            iintvl += 1
        end
        if iintvl > msing
            iintvl = 0
        end
    end
    error("gal_locate_cell: x=$x not found in any cell")
end

"""
    gal_get_solution(ws, asymps, sings, intr, x, iintvl, icell, isol; want_d=true)
        -> (sol, dsol, iintvl, icell)

Reconstruct the displacement `sol` (length `mpert`) — and, when `want_d`, its analytic radial
derivative `dsol = d(sol)/dψ` — at flux `x` for solution column `isol`, from the solved Galerkin
coefficients `ws.sol`. Port of `gal_get_solution` (gal.f:1833-1997, non-cut path). `iintvl`/`icell`
are a monotone cell cursor advanced and returned for the next call. `dsol` is `nothing` if `!want_d`.
"""
function gal_get_solution(ws::GalWorkspace, asymps::Vector{GalSingAsymp}, sings::Vector{SingType},
    intr::ForceFreeStatesInternal, x::Float64, iintvl::Int, icell::Int, isol::Int; want_d::Bool=true)

    msing = length(ws.intvl) - 1
    mpert = intr.numpert_total
    np = GAL_NP
    iintvl, icell, cell = gal_locate_cell(ws, x, iintvl, icell)

    sol = zeros(ComplexF64, mpert)
    dsol = want_d ? zeros(ComplexF64, mpert) : nothing

    # --- non-resonant (Hermite) part: same coefficients, value basis pb → ξ, derivative basis qb → ξ′ ---
    pbt, qbt = gal_hermite(x, cell.x[1], cell.x[2])
    pb = collect(pbt)
    qb = collect(qbt)
    if iintvl == msing && icell == ws.nx          # last edge cell: swap node-2/3 DOFs (gal.f:1888-1897)
        pb[3], pb[4] = pb[4], pb[3]
        qb[3], qb[4] = qb[4], qb[3]
    end
    for ip in 0:np
        u = @view ws.sol[cell.map[:, ip+1], isol]
        @views sol .+= u .* pb[ip+1]
        want_d && (@views dsol .+= u .* qb[ip+1])
    end

    cell.etype == GCT_NONE && return sol, dsol, iintvl, icell

    # --- small & large resonant/extension solutions (gal.f:1904-1970) ---
    jsing = _cell_jsing(cell, iintvl)             # left → iintvl+1, right → iintvl (gal.f:1911/1919)
    asymp = asymps[jsing]
    psi_s = sings[jsing].psifac
    ipert0 = _ipert_res(sings[jsing], intr)       # resonant (big) row; small row = ipert0+mpert

    # Edge Hermite weights (fresh, unswapped) for ext/ext2 cells (gal.f:1912-1925)
    pbe, qbe = gal_hermite(x, cell.x[1], cell.x[2])
    if cell.extra == GAL_SIDE_LEFT
        epb1, epb2 = pbe[3], pbe[4]
        eqb1, eqb2 = qbe[3], qbe[4]
        xext = cell.x[2]
    else
        epb1, epb2 = pbe[1], pbe[2]
        eqb1, eqb2 = qbe[1], qbe[2]
        xext = cell.x[1]
    end

    need_in = cell.etype == GCT_RES || cell.etype == GCT_EXT || cell.etype == GCT_EXT1
    need_ext = cell.etype == GCT_EXT || cell.etype == GCT_EXT2
    ua = need_in ? sing_get_ua_gal(asymp, x - psi_s) : nothing
    dua = (need_in && want_d) ? sing_get_dua_gal(asymp, x - psi_s) : nothing
    uaext = need_ext ? sing_get_ua_gal(asymp, xext - psi_s) : nothing
    duaext = need_ext ? sing_get_dua_gal(asymp, xext - psi_s) : nothing

    # small solution Δ-coefficient restore (ext/res only), coeff = ws.sol[emap,isol] (gal.f:1937-1951)
    if cell.etype == GCT_RES || cell.etype == GCT_EXT
        delta = ws.sol[cell.emap, isol]
        if cell.etype == GCT_RES
            @views sol .+= delta .* ua[:, ipert0+mpert, 1]
            want_d && (@views dsol .+= delta .* dua[:, ipert0+mpert, 1])
        else  # GCT_EXT: series held at the fixed edge xext, x-dependence only via Hermite weights
            @views sol .+= delta .* (epb1 .* uaext[:, ipert0+mpert, 1] .+ epb2 .* duaext[:, ipert0+mpert, 1])
            want_d && (@views dsol .+= delta .* (eqb1 .* uaext[:, ipert0+mpert, 1] .+ eqb2 .* duaext[:, ipert0+mpert, 1]))
        end
    end

    # large solution (coeff 1) only for the column driving this surface/side (gal.f:1952-1970)
    if (isol == 2jsing - 1 && cell.extra == GAL_SIDE_LEFT) ||
       (isol == 2jsing && cell.extra == GAL_SIDE_RIGHT)
        if cell.etype == GCT_RES || cell.etype == GCT_EXT || cell.etype == GCT_EXT1
            @views sol .+= ua[:, ipert0, 1]
            want_d && (@views dsol .+= dua[:, ipert0, 1])
        elseif cell.etype == GCT_EXT2
            @views sol .+= epb1 .* uaext[:, ipert0, 1] .+ epb2 .* duaext[:, ipert0, 1]
            want_d && (@views dsol .+= eqb1 .* uaext[:, ipert0, 1] .+ eqb2 .* duaext[:, ipert0, 1])
        end
    end

    return sol, dsol, iintvl, icell
end

"""
    gal_output_solution(ws, asymps, sings, intr, profiles, psihigh) -> GalerkinSolution

Build the gal-native packed radial grid (inner→edge; `GAL_INTERP_NP_RES` points per resonant/extension
cell, `GAL_INTERP_NP` elsewhere, plus the edge point ψ=psihigh) and evaluate ξ AND the analytic ξ′ over
it for every solution column. Port of `gal_output_solution` (gal.f:2027-2103); the binary/ASCII writes
and the `b_flag` ξ→b^ψ conversion (off by default) are not ported — we keep ξ itself.
"""
function gal_output_solution(ws::GalWorkspace, asymps::Vector{GalSingAsymp}, sings::Vector{SingType},
    intr::ForceFreeStatesInternal, profiles, psihigh::Float64)

    mpert = intr.numpert_total
    msing = length(ws.intvl) - 1

    # --- grid (gal.f:2038-2079): per-cell left-inclusive sampling, then the edge point ---
    psi = Float64[]
    issing = Bool[]
    for iintvl in 0:msing
        for icell in 1:ws.nx
            cell = ws.intvl[iintvl+1].cells[icell]
            npts = (cell.etype == GCT_RES || cell.etype == GCT_EXT) ? GAL_INTERP_NP_RES : GAL_INTERP_NP
            x1, x2 = cell.x
            for k in 0:(npts-1)
                push!(psi, x1 + (x2 - x1) * (k / npts))
                # res-right cell's first point sits exactly on the surface (gal.f:2070-2071)
                push!(issing, k == 0 && cell.extra == GAL_SIDE_RIGHT && cell.etype == GCT_RES)
            end
        end
    end
    push!(psi, psihigh)
    push!(issing, false)

    qhint = Ref(1)
    q = [profiles.q_spline(p; hint=qhint) for p in psi]

    # --- evaluate (gal.f:2090-2103) ---
    ngrid = length(psi)
    xi = zeros(ComplexF64, mpert, ngrid, ws.nsol)
    xi_deriv = zeros(ComplexF64, mpert, ngrid, ws.nsol)
    for isol in 1:ws.nsol
        iintvl = 0
        icell = 1
        for ip in 1:ngrid
            issing[ip] && continue
            sol, dsol, iintvl, icell = gal_get_solution(ws, asymps, sings, intr, psi[ip], iintvl, icell, isol)
            @views xi[:, ip, isol] .= sol
            @views xi_deriv[:, ip, isol] .= dsol
        end
    end

    return GalerkinSolution(psi, q, issing, xi, xi_deriv)
end
