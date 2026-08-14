# GalerkinSolution.jl
#
# Reconstruct the outer-region radial displacement ξ(ψ) AND its analytic derivative ξ′(ψ) from the
# solved Galerkin coefficients. Ports:
#   gal_get_solution    (gal.f)  ξ (and optional ξ′) at one ψ for one solution column
#   gal_output_solution (gal.f)  evaluate (ξ, ξ′) on a gal-native packed grid, all columns
#
# ξ′ is evaluated analytically from the Hermite derivative basis (gal_hermite qb) and the asymptotic-series
# derivative (sing_get_dua_gal), never by spline-differentiating ξ at the packed edge.
#
# Fortran module flags restore_uh/us/ul default .TRUE. (gal.f); the full reconstruction is always
# performed. The `cut=true` path swaps the resonant Frobenius series for its leading-order term
# (`sing_get_ua_gal_cut`); the resulting cut solution supplies the regular background that the
# resistive inner-layer solution is added to when forming the composite solution at a rational
# surface (see `gal_match_rpec`).

# Sampling points per cell, matching Fortran interp_np_res / interp_np (gal.f): coarse in regular cells,
# dense in resonant/extension cells to resolve the near-singular asymptotic series.
const GAL_INTERP_NP = 3
const GAL_INTERP_NP_RES = 60

# Advance the monotone (iintvl, icell) cursor to the cell containing x. Port of gal.f.
@inline function gal_locate_cell(ws::GalWorkspace, x::Float64, iintvl::Int, icell::Int)
    msing = length(ws.intvl) - 1
    for _ in 1:((msing+1)*ws.nx+1)   # bounded scan: at most one full wrap over all cells
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
    gal_get_solution(ws, asymps, sings, intr, x, iintvl, icell, isol; want_d=true, cut_delta=nothing)
        -> (sol, dsol, iintvl, icell)

Reconstruct the displacement `sol` (length `mpert`) — and, when `want_d`, its analytic radial
derivative `dsol = d(sol)/dψ` — at flux `x` for solution column `isol`, from the solved Galerkin
coefficients `ws.sol`. Port of `gal_get_solution` (gal.f). `iintvl`/`icell`
are a monotone cell cursor advanced and returned for the next call. `dsol` is `nothing` if `!want_d`.

Passing `cut_delta` (the `(nsol, 2·msing)` small-solution coefficient matrix, i.e. `GalerkinResult.delta`)
selects the *cut* solution: the leading-order resonant content is subtracted from the reconstruction,
leaving the smooth background that the resistive inner-layer solution is added to when forming the
composite solution at a rational surface. `want_d` is forced off in that case.
"""
function gal_get_solution(ws::GalWorkspace, asymps::Vector{GalSingAsymp}, sings::Vector{SingType},
    intr::ForceFreeStatesInternal, x::Float64, iintvl::Int, icell::Int, isol::Int; want_d::Bool=true,
    cut_delta::Union{Nothing,AbstractMatrix{ComplexF64}}=nothing)

    cut_delta === nothing || (want_d = false)

    msing = length(ws.intvl) - 1
    mpert = intr.numpert_total
    np = GAL_NP
    iintvl, icell, cell = gal_locate_cell(ws, x, iintvl, icell)

    sol = zeros(ComplexF64, mpert)
    dsol = want_d ? zeros(ComplexF64, mpert) : nothing

    # --- non-resonant (Hermite) part: same coefficients, value basis pb → ξ, derivative basis qb → ξ′ ---
    pbt, qbt = gal_hermite(x, cell.x[1], cell.x[2])
    swap = iintvl == msing && icell == ws.nx       # last edge cell: swap node-2/3 DOFs (gal.f)
    pb = swap ? (pbt[1], pbt[2], pbt[4], pbt[3]) : pbt
    qb = swap ? (qbt[1], qbt[2], qbt[4], qbt[3]) : qbt
    for ip in 0:np
        u = @view ws.sol[cell.map[:, ip+1], isol]
        @views sol .+= u .* pb[ip+1]
        want_d && (@views dsol .+= u .* qb[ip+1])
    end

    cell.etype == GCT_NONE && return sol, dsol, iintvl, icell

    # --- small & large resonant/extension solutions (gal.f) ---
    jsing = _cell_jsing(cell, iintvl)             # left → iintvl+1, right → iintvl (gal_get_solution)
    asymp = asymps[jsing]
    psi_s = sings[jsing].psifac
    # sing_get_*_gal returns only the two resonant columns: 1 = big solution, 2 = small solution.

    # Edge Hermite weights (fresh, unswapped) for ext/ext2 cells (gal.f)
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

    # small solution Δ-coefficient restore (ext/res only), coeff = ws.sol[emap,isol] (gal.f)
    if cell.etype == GCT_RES || cell.etype == GCT_EXT
        delta = ws.sol[cell.emap, isol]
        if cell.etype == GCT_RES
            @views sol .+= delta .* ua[:, 2, 1]
            want_d && (@views dsol .+= delta .* dua[:, 2, 1])
        else  # GCT_EXT: series held at the fixed edge xext, x-dependence only via Hermite weights
            @views sol .+= delta .* (epb1 .* uaext[:, 2, 1] .+ epb2 .* duaext[:, 2, 1])
            want_d && (@views dsol .+= delta .* (eqb1 .* uaext[:, 2, 1] .+ eqb2 .* duaext[:, 2, 1]))
        end
    end

    # large solution (coeff 1) only for the column driving this surface/side (gal.f)
    driving = (isol == 2jsing - 1 && cell.extra == GAL_SIDE_LEFT) ||
              (isol == 2jsing && cell.extra == GAL_SIDE_RIGHT)
    if driving
        if cell.etype == GCT_RES || cell.etype == GCT_EXT || cell.etype == GCT_EXT1
            @views sol .+= ua[:, 1, 1]
            want_d && (@views dsol .+= dua[:, 1, 1])
        elseif cell.etype == GCT_EXT2
            @views sol .+= epb1 .* uaext[:, 1, 1] .+ epb2 .* duaext[:, 1, 1]
            want_d && (@views dsol .+= eqb1 .* uaext[:, 1, 1] .+ eqb2 .* duaext[:, 1, 1])
        end
    end

    # Cut solution (gal.f cut_flag block): remove the leading-order resonant content, evaluated at x
    # for every resonant cell type. The small coefficient comes from the Δ′ matrix rather than
    # `cell.emap` because ext1/ext2 cells carry no emap.
    if cut_delta !== nothing
        jsol = cell.extra == GAL_SIDE_LEFT ? 2jsing - 1 : 2jsing
        ua_cut = sing_get_ua_gal_cut(asymp, x - psi_s)
        @views sol .-= cut_delta[isol, jsol] .* ua_cut[:, 2, 1]
        driving && (@views sol .-= ua_cut[:, 1, 1])
    end

    return sol, dsol, iintvl, icell
end

"""
    gal_output_solution(ws, asymps, sings, intr, profiles, psihigh; delta=nothing) -> GalerkinSolution

Build the gal-native packed radial grid (inner→edge; `GAL_INTERP_NP_RES` points per resonant/extension
cell, `GAL_INTERP_NP` elsewhere, plus the edge point ψ=psihigh) and evaluate ξ AND the analytic ξ′ over
it for every solution column. Port of `gal_output_solution` (gal.f); the binary/ASCII writes
and the `b_flag` ξ→b^ψ conversion (off by default) are not ported — we keep ξ itself.

When `delta` (the `(nsol, 2·msing)` small-solution coefficient matrix) is supplied, the cut solution
`xi_cut` is evaluated on the same grid; it is the background of the composite inner-region solution
built by `gal_match_rpec`.
"""
function gal_output_solution(ws::GalWorkspace, asymps::Vector{GalSingAsymp}, sings::Vector{SingType},
    intr::ForceFreeStatesInternal, profiles, psihigh::Float64;
    delta::Union{Nothing,AbstractMatrix{ComplexF64}}=nothing)

    mpert = intr.numpert_total
    msing = length(ws.intvl) - 1

    # --- grid (gal.f): per-cell left-inclusive sampling, then the edge point ---
    psi = Float64[]
    issing = Bool[]
    for iintvl in 0:msing
        for icell in 1:ws.nx
            cell = ws.intvl[iintvl+1].cells[icell]
            npts = (cell.etype == GCT_RES || cell.etype == GCT_EXT) ? GAL_INTERP_NP_RES : GAL_INTERP_NP
            x1, x2 = cell.x
            for k in 0:(npts-1)
                push!(psi, x1 + (x2 - x1) * (k / npts))
                # res-right cell's first point sits exactly on the surface (gal.f)
                push!(issing, k == 0 && cell.extra == GAL_SIDE_RIGHT && cell.etype == GCT_RES)
            end
        end
    end
    push!(psi, psihigh)
    push!(issing, false)

    qhint = Ref(1)
    q = [profiles.q_spline(p; hint=qhint) for p in psi]

    # --- evaluate (gal.f) ---
    ngrid = length(psi)
    xi = zeros(ComplexF64, mpert, ngrid, ws.nsol)
    xi_deriv = zeros(ComplexF64, mpert, ngrid, ws.nsol)
    xi_cut = delta === nothing ? zeros(ComplexF64, 0, 0, 0) : zeros(ComplexF64, mpert, ngrid, ws.nsol)
    for isol in 1:ws.nsol
        iintvl = 0
        icell = 1
        for ip in 1:ngrid
            issing[ip] && continue
            sol, dsol, iintvl, icell = gal_get_solution(ws, asymps, sings, intr, psi[ip], iintvl, icell, isol)
            @views xi[:, ip, isol] .= sol
            @views xi_deriv[:, ip, isol] .= dsol
        end
        delta === nothing && continue
        iintvl = 0
        icell = 1
        for ip in 1:ngrid
            issing[ip] && continue
            solc, _, iintvl, icell = gal_get_solution(ws, asymps, sings, intr, psi[ip], iintvl, icell, isol;
                cut_delta=delta)
            @views xi_cut[:, ip, isol] .= solc
        end
    end

    # ψ span of the resonant + extension cells flanking each surface: outside it the cut subtracts
    # nothing, so the composite inner-region solution is only defined within these bounds
    # (Fortran `outs%xext`).
    cut_range = zeros(Float64, 0, 0)
    if delta !== nothing
        cut_range = hcat(fill(Inf, msing), fill(-Inf, msing))
        for iintvl in 0:msing, icell in 1:ws.nx
            cell = ws.intvl[iintvl+1].cells[icell]
            cell.etype == GCT_NONE && continue
            js = _cell_jsing(cell, iintvl)
            cut_range[js, 1] = min(cut_range[js, 1], cell.x[1])
            cut_range[js, 2] = max(cut_range[js, 2], cell.x[2])
        end
    end

    return GalerkinSolution(psi, q, issing, xi, xi_deriv, xi_cut, cut_range)
end
