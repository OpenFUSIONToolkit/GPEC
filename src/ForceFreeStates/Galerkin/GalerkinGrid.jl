# GalerkinGrid.jl
#
# Grid construction and local→global DOF mapping for the outer-region Galerkin solver.
# Ports gal_pack (gal.f:262-318), gal_make_grid (gal.f:326-451), and gal_make_map (gal.f:509-575).
#
# Node arrays use Fortran 0-based `x(0:nx)` mapped to Julia `xn[i+1]` (so Fortran `x(i)` is `xn[i+1]`).
# Cells are 1-based in both. Intervals `0:msing` (Fortran) are `ws.intvl[ising+1]` in Julia.
# The optional `gal_xmin_flag`/`sing1_xmin` adaptive-width branch is intentionally not ported.

"""
    gal_pack(nx::Int, pfac::Float64, side::String) -> Vector{Float64}

Packed logical grid on `[-1, 1]`. Port of Fortran `gal_pack` (gal.f:262-318). For `side="both"`
returns `2*nx+1` nodes (Fortran `-nx:nx`) clustered toward both ends when `pfac < 1` (strong packing
near the bracketing singular surfaces). `pfac > 1` packs toward the centre; `pfac == 1` is uniform.
"""
function gal_pack(nx::Int, pfac::Float64, side::String)
    if side == "left"
        xi = collect(-2nx:0) ./ (2nx)
    elseif side == "right"
        xi = collect(0:2nx) ./ (2nx)
    elseif side == "both"
        xi = collect(-nx:nx) ./ nx
    else
        error("gal_pack: unrecognized side = $side")
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

"""
    gal_make_grid!(intvl::GalInterval, ising::Int, msing::Int, sings::Vector{SingType},
                   nn::Int, psilow::Float64, psihigh::Float64, ctrl::ForceFreeStatesControl)

Set the cell types (`res`/`ext`/`ext1`/`ext2`/`none`), sides, resonant integration limits, and packed
node positions for interval `ising` (0-based, `0:msing`). Port of Fortran `gal_make_grid`
(gal.f:326-451), `dx1dx2_flag=true` path. `intvl.x`/`intvl.dx` have length `nx+1`, `intvl.cells` length
`nx` (preallocated by the caller). `sings` is the filtered list of resonant surfaces (1-based,
`sings[ising]` = Fortran `sing(ising)`).

Note (faithful to Fortran): the lower-surface width scale uses `|nn·q1|` (gal.f:356) but the
upper-surface scale uses `|q1|` without `nn` (gal.f:398). Identical for `nn = 1`.
"""
function gal_make_grid!(intvl::GalInterval, ising::Int, msing::Int, sings::Vector{SingType},
    nn::Int, psilow::Float64, psihigh::Float64, ctrl::ForceFreeStatesControl)

    nx = ctrl.gal_nx
    dx0 = ctrl.gal_dx0
    dx1 = ctrl.gal_dx1
    dx2 = ctrl.gal_dx2
    pfac = ctrl.gal_pfac
    cutoff = ctrl.gal_cutoff
    xn = intvl.x
    cells = intvl.cells

    # Reset cell flags
    for c in cells
        c.etype = GCT_NONE
        c.extra = GAL_SIDE_NONE
    end

    # --- lower bound (gal.f:349-389) ---
    local ixmin::Int
    if ising == 0
        ixmin = 0
        xn[0+1] = psilow
    else
        x0 = sings[ising].psifac
        nq1 = abs(nn * sings[ising].q1)
        xn[0+1] = x0
        cells[1].x_lsode = x0 + dx0 / nq1
        if ctrl.gal_dx1dx2_flag
            xn[1+1] = x0 + dx1 / nq1
            xn[2+1] = x0 + (dx1 + dx2) / nq1
            ixmin = 2
        else
            ixmin = 0
        end
        cells[1].extra = GAL_SIDE_RIGHT
        cells[2].extra = GAL_SIDE_RIGHT
        cells[1].etype = GCT_RES
        cells[2].etype = GCT_EXT
        for ix in 3:(cutoff+1)
            cells[ix].etype = GCT_EXT1
            cells[ix].extra = GAL_SIDE_RIGHT
        end
        ix_ext2 = max(cutoff + 2, 3)   # Fortran loop-exit value of ix (gal.f:387)
        (ix_ext2 > nx || cells[ix_ext2].etype != GCT_NONE) &&
            error("gal_make_grid: too many elements include the big solution (right side)")
        cells[ix_ext2].etype = GCT_EXT2
        cells[ix_ext2].extra = GAL_SIDE_RIGHT
    end

    # --- upper bound (gal.f:391-429) ---
    local ixmax::Int
    if ising == msing
        ixmax = nx
        xn[nx+1] = psihigh
    else
        x1 = sings[ising+1].psifac
        nq1 = abs(sings[ising+1].q1)   # NOTE: no nn factor, faithful to gal.f:398
        xn[nx+1] = x1
        cells[nx].x_lsode = x1 - dx0 / nq1
        if ctrl.gal_dx1dx2_flag
            xn[(nx-1)+1] = x1 - dx1 / nq1
            xn[(nx-2)+1] = x1 - (dx1 + dx2) / nq1
            ixmax = nx - 2
        else
            ixmax = nx
        end
        cells[nx-1].extra = GAL_SIDE_LEFT
        cells[nx].extra = GAL_SIDE_LEFT
        cells[nx-1].etype = GCT_EXT
        cells[nx].etype = GCT_RES
        for ix in (nx-2):-1:(nx-cutoff)
            cells[ix].etype = GCT_EXT1
            cells[ix].extra = GAL_SIDE_LEFT
        end
        ix_ext2 = min(nx - cutoff - 1, nx - 2)   # Fortran loop-exit value of ix (gal.f:427)
        (ix_ext2 < 1 || cells[ix_ext2].etype != GCT_NONE) &&
            error("gal_make_grid: too many elements include the big solution (left side)")
        cells[ix_ext2].etype = GCT_EXT2
        cells[ix_ext2].extra = GAL_SIDE_LEFT
    end

    # --- interior packed grid (gal.f:431-446) ---
    x0 = xn[ixmin+1]
    x1 = xn[ixmax+1]
    xm = (xn[ixmax+1] + xn[ixmin+1]) / 2
    dxh = (xn[ixmax+1] - xn[ixmin+1]) / 2
    mx = (ixmax - ixmin) ÷ 2
    # Default: symmetric "both" pack (faithful to gal.f:438). With gal_edge_onesided, the two end
    # intervals pack only toward their single rational end, leaving the regular boundary (psilow at
    # interval 0, psihigh at interval msing) at coarse spacing — removes the gratuitous fine edge cell.
    side = "both"
    if ctrl.gal_edge_onesided && msing > 0
        if ising == 0
            side = "right"   # rational sits at the upper (ixmax) end
        elseif ising == msing
            side = "left"    # rational sits at the lower (ixmin) end
        end
    end
    packed = gal_pack(mx, pfac, side)          # length 2*mx+1 = (ixmax-ixmin)+1
    for k in 0:(ixmax-ixmin)
        xn[ixmin+1+k] = xm + dxh * packed[k+1]
    end
    xn[ixmin+1] = x0
    xn[ixmax+1] = x1

    intvl.dx[0+1] = 0.0
    for i in 1:nx
        intvl.dx[i+1] = xn[i+1] - xn[i]
    end
    for ix in 1:nx
        cells[ix].x = (xn[ix], xn[ix+1])       # Fortran (x(ix-1), x(ix))
    end
    return intvl
end

"""
    gal_make_map!(ws::GalWorkspace, mpert::Int)

Build the local→global DOF numbering with the `2*mpert` C¹ overlap between adjacent Hermite cells and
the interleaved extra (small resonant) DOFs (`emap`). Port of Fortran `gal_make_map` (gal.f:509-575).
Sets `ws.ndim`. The `skip`/`emap` bookkeeping reserves a global slot for each resonant coefficient and
shares it between the `res` cell and its neighbouring `ext` cell.
"""
function gal_make_map!(ws::GalWorkspace, mpert::Int)
    np = ws.np
    msing = length(ws.intvl) - 1
    imap = 1
    skip = false
    emap = 0
    for ising in 0:msing
        intvl = ws.intvl[ising+1]
        for ix in 1:ws.nx
            cell = intvl.cells[ix]
            # Nonresonant Hermite DOFs (with 2*mpert overlap onto the previous cell)
            if imap > 1
                imap -= 2 * mpert
            end
            for ip in 0:np
                for ipert in 1:mpert
                    cell.map[ipert, ip+1] = imap
                    imap += 1
                end
                if ip == 1 && skip
                    imap += 1
                    skip = false
                end
            end
            # Extra (small resonant) DOF, shared between res and adjacent ext cell
            if cell.extra == GAL_SIDE_LEFT
                if cell.etype == GCT_EXT
                    emap = imap
                    cell.emap = emap
                    skip = true
                elseif cell.etype == GCT_RES
                    cell.emap = emap
                end
            elseif cell.extra == GAL_SIDE_RIGHT
                if cell.etype == GCT_RES
                    emap = imap
                    cell.emap = emap
                    skip = true
                elseif cell.etype == GCT_EXT
                    cell.emap = emap
                end
            end
        end
    end
    ws.ndim = imap - 1
    return ws
end
