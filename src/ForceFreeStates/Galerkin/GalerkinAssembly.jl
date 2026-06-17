# GalerkinAssembly.jl
#
# Element-level assembly for the outer-region Galerkin solver. Ports:
#   gal_hermite       (gal.f)   Hermite cubic basis
#   gal_get_fkg       (gal.f)  F, K, G at a point from F̄/K̄/Ḡ + singfac
#   gal_gauss_quad    (gal.f) nonresonant stiffness via Gauss-Lobatto quadrature
#   gal_extension     (gal.f) extension-cell matrix/rhs (big-solution driving + surface terms)
#   gal_resonant      (gal.f)   resonant-cell integral (QuadGK in place of Fortran LSODE)
#   gal_assemble_mat  (gal.f)   scatter cell blocks into the banded global matrix (LU/Cholesky)
#   gal_assemble_rhs  (gal.f)   scatter cell rhs into the global RHS
#   gal_set_boundary  (gal.f) axis identity + edge (vacuum or fixed) BCs
#
# Hermite DOF index ip (Fortran 0:3) ↔ Julia array index ip+1 throughout.

"""Resonant flat mode index `1 + (m-mlow) + (n-nlow)*mpert` for a singular surface (single resonance)."""
@inline function _ipert_res(sing::SingType, intr::ForceFreeStatesInternal)
    return 1 + (sing.m[1] - intr.mlow) + (sing.n[1] - intr.nlow) * intr.mpert
end

# Two-sided asymptotic evaluation (z = ψ - ψ_s). Right (z≥0) uses the sig=+1 series at z; left (z<0)
# uses the sig=-1 series at the positive distance |z| (real √, no renorm), matching Fortran sing.f.
# The derivative on the left picks up d|z|/dψ = -1.
@inline sing_get_ua_gal(g::GalSingAsymp, z::Float64) =
    z >= 0 ? sing_get_ua(g.right, z) : sing_get_ua(g.left, -z)
@inline sing_get_dua_gal(g::GalSingAsymp, z::Float64) =
    z >= 0 ? sing_get_dua(g.right, z) : (-1) .* sing_get_dua(g.left, -z)

"""
    gal_hermite(x, x0, x1) -> (pb, qb)

Hermite cubic basis values `pb` and derivatives `qb` on `[x0, x1]` (length 4, Fortran `0:3`).
Port of `gal_hermite` (gal.f).
"""
function gal_hermite(x::Real, x0::Real, x1::Real)
    dx = x1 - x0
    t0 = (x - x0) / dx
    t1 = 1 - t0
    t02 = t0 * t0
    t12 = t1 * t1
    pb = (t12 * (1 + 2t0), t12 * t0 * dx, t02 * (1 + 2t1), -t02 * t1 * dx)
    qb = (-6t0 * t1 / dx, t0 * (3t0 - 4) + 1, 6t1 * t0 / dx, t0 * (3t0 - 2))
    return pb, qb
end

"""
    gal_get_fkg(ffit, intr, x, q) -> (F, K, G)

Evaluate the `mpert×mpert` matrices `F = Q F̄ Qᴴ`, `K = Q K̄`, `G = Ḡ` at flux `x` with safety factor
`q`, where `Q = diag(singfac)` and `singfac = m - n q` (direct). Port of `gal_get_fkg` (gal.f).
Uses the un-factored reduced `ffit.fmats_gal` (F̄), `ffit.kmats` (K̄), `ffit.gmats` (Ḡ). F/K/G are the
ideal-MHD Euler–Lagrange coefficient matrices of the outer-region weak form (Glasser 2016, PoP 23, 112506).
"""
function gal_get_fkg(ffit::FourFitVars, intr::ForceFreeStatesInternal, x::Float64, q::Float64)
    N = intr.numpert_total
    sf = vec((intr.mlow:intr.mhigh) .- q .* (intr.nlow:intr.nhigh)')

    Fbar = Matrix{ComplexF64}(undef, N, N)
    K = Matrix{ComplexF64}(undef, N, N)
    G = Matrix{ComplexF64}(undef, N, N)
    ffit.fmats_gal(vec(Fbar), x; hint=ffit._hint)
    ffit.kmats(vec(K), x; hint=ffit._hint)
    ffit.gmats(vec(G), x; hint=ffit._hint)

    F = Matrix{ComplexF64}(undef, N, N)
    @inbounds for j in 1:N, i in 1:N
        F[i, j] = Fbar[i, j] * sf[i] * sf[j]
        K[i, j] = K[i, j] * sf[i]
    end
    return F, K, G
end

"""
    gal_gauss_quad!(cell, ffit, profiles, intr, nodes, weights, swap_edge)

Accumulate the nonresonant Hermite stiffness block `cell.mat` by Gauss-Lobatto quadrature. Port of
`gal_gauss_quad` (gal.f). When `swap_edge` (the final cell of the last interval), the
right-node value/slope DOFs (Fortran pb(2)↔pb(3)) are swapped so the edge-value DOF lands at index 3,
matching the free-boundary BC (gal.f).
"""
function gal_gauss_quad!(cell::GalCell, ffit::FourFitVars, profiles, intr::ForceFreeStatesInternal,
    nodes::Vector{Float64}, weights::Vector{Float64}, swap_edge::Bool)

    N = intr.numpert_total
    np = GAL_NP
    fill!(cell.mat, 0)
    x1, x2 = cell.x
    x0c = (x2 + x1) / 2
    dxc = (x2 - x1) / 2
    qhint = Ref(1)

    @inbounds for iq in eachindex(nodes)
        x = x0c + dxc * nodes[iq]
        w = dxc * weights[iq]
        q = profiles.q_spline(x; hint=qhint)
        F, K, G = gal_get_fkg(ffit, intr, x, q)
        pbt, qbt = gal_hermite(x, x1, x2)
        pb = collect(pbt)
        qb = collect(qbt)
        if swap_edge
            pb[3], pb[4] = pb[4], pb[3]
            qb[3], qb[4] = qb[4], qb[3]
        end
        for ip in 0:np, ipert in 1:N, jp in 0:np, jpert in 1:N
            cell.mat[ipert, jpert, ip+1, jp+1] += w * (
                F[ipert, jpert] * qb[ip+1] * qb[jp+1] +
                K[ipert, jpert] * qb[ip+1] * pb[jp+1] +
                conj(K[jpert, ipert]) * pb[ip+1] * qb[jp+1] +
                G[ipert, jpert] * pb[ip+1] * pb[jp+1])
        end
    end
    return cell
end

# Resolve (jsing, ψ_s, sign) for an extension/resonant cell from its side.
@inline function _cell_jsing(cell::GalCell, ising::Int)
    return cell.extra == GAL_SIDE_LEFT ? ising + 1 : ising
end

"""
    gal_extension!(cell, ising, ffit, profiles, intr, asymps, sings, nn, nodes, weights)

Build `cell.emat`, `cell.ediag`, `cell.rhs`, `cell.erhs` for an extension cell (`ext`/`ext1`/`ext2`).
Port of `gal_extension` (gal.f). The big solution (`isol` column) drives the RHS; for `ext`
cells the small solution (`isol+N` column) also forms the resonant coupling `emat`/`ediag`.
"""
function gal_extension!(cell::GalCell, ising::Int, ffit::FourFitVars, profiles,
    intr::ForceFreeStatesInternal, asymps::Vector{GalSingAsymp}, sings::Vector{SingType},
    nn::Int, nodes::Vector{Float64}, weights::Vector{Float64})

    N = intr.numpert_total
    np = GAL_NP
    x1, x2 = cell.x
    qhint = Ref(1)

    jsing = _cell_jsing(cell, ising)
    asymp = asymps[jsing]
    psi_s = sings[jsing].psifac
    isol = _ipert_res(sings[jsing], intr)
    ua_at(x) = sing_get_ua_gal(asymp, x - psi_s)
    dua_at(x) = sing_get_dua_gal(asymp, x - psi_s)

    # Boundary node and weights (gal.f)
    if cell.extra == GAL_SIDE_LEFT
        jp = 2          # Fortran jp (0-based) -> Julia jp+1
        xb = x2
        ws = -1.0
    else
        jp = 0
        xb = x1
        ws = 1.0
    end
    if cell.etype == GCT_EXT || cell.etype == GCT_EXT1
        w1 = 1.0
        w2 = 1.0
    else # GCT_EXT2
        w1 = (cell.extra == GAL_SIDE_LEFT) ? 0.0 : 1.0
        w2 = (cell.extra == GAL_SIDE_LEFT) ? 1.0 : 0.0
    end

    fill!(cell.emat, 0)
    cell.ediag = zero(ComplexF64)

    # --- emat and ediag (ext cells only) (gal.f) ---
    if cell.etype == GCT_EXT
        uab = ua_at(xb)
        duab = dua_at(xb)
        u = uab[:, isol+N, 1]      # small solution
        du = duab[:, isol+N, 1]
        for ipert in 1:N
            @views cell.emat .+= cell.mat[:, ipert, :, jp+1] .* u[ipert] .+
                                 cell.mat[:, ipert, :, jp+2] .* du[ipert]
        end
        for ipert in 1:N
            cell.ediag += conj(u[ipert]) * cell.emat[ipert, jp+1] +
                          conj(du[ipert]) * cell.emat[ipert, jp+2]
        end
        # surface term at xb
        qb = profiles.q_spline(xb; hint=qhint)
        Fb, Kb, _ = gal_get_fkg(ffit, intr, xb, qb)
        pbt, _ = gal_hermite(xb, x1, x2)
        Fdu_Ku = Fb * du .+ Kb * u
        for ip in 0:np
            @views cell.emat[:, ip+1] .+= Fdu_Ku .* (pbt[ip+1] * ws)
        end
        cell.ediag += sum(conj.(u) .* Fdu_Ku) * ws
    end

    # --- rhs (gal.f) ---
    ua1 = ua_at(x1)[:, isol, 1]    # big solution at left node
    du1 = dua_at(x1)[:, isol, 1]
    ua2 = ua_at(x2)[:, isol, 1]    # big solution at right node
    du2 = dua_at(x2)[:, isol, 1]
    fill!(cell.rhs, 0)

    if cell.etype == GCT_EXT || cell.etype == GCT_EXT1
        cell.erhs = zero(ComplexF64)
        x0c = (x2 + x1) / 2
        dxc = (x2 - x1) / 2
        for iq in eachindex(nodes)
            x = x0c + dxc * nodes[iq]
            w = dxc * weights[iq]
            q = profiles.q_spline(x; hint=qhint)
            F, K, G = gal_get_fkg(ffit, intr, x, q)
            pbt, qbt = gal_hermite(x, x1, x2)
            uax = ua_at(x)
            ub = uax[:, isol, 1]
            dub = dua_at(x)[:, isol, 1]
            term1 = F * dub .+ K * ub
            term2 = adjoint(K) * dub .+ G * ub
            for ip in 0:np
                @views cell.rhs[:, ip+1] .-= (term1 .* qbt[ip+1] .+ term2 .* pbt[ip+1]) .* w
            end
        end
        if cell.etype == GCT_EXT
            for ipert in 1:N
                cell.erhs += conj(ua_at(xb)[ipert, isol+N, 1]) * cell.rhs[ipert, jp+1] +
                             conj(dua_at(xb)[ipert, isol+N, 1]) * cell.rhs[ipert, jp+2]
            end
        end
    elseif cell.etype == GCT_EXT2
        for ipert in 1:N
            @views cell.rhs .-= w1 .* (cell.mat[:, ipert, :, 0+1] .* ua1[ipert] .+
                                       cell.mat[:, ipert, :, 1+1] .* du1[ipert])
            @views cell.rhs .-= w2 .* (cell.mat[:, ipert, :, 2+1] .* ua2[ipert] .+
                                       cell.mat[:, ipert, :, 3+1] .* du2[ipert])
        end
    end

    # --- surface terms (always; gal.f) ---
    q_l = profiles.q_spline(x1; hint=qhint)
    Fl, Kl, _ = gal_get_fkg(ffit, intr, x1, q_l)
    pbt, _ = gal_hermite(x1, x1, x2)
    surf_l = Fl * du1 .+ Kl * ua1
    for ip in 0:np
        @views cell.rhs[:, ip+1] .-= surf_l .* (pbt[ip+1] * w1)
    end
    if cell.etype == GCT_EXT && cell.extra == GAL_SIDE_RIGHT
        cell.erhs -= sum(conj.(ua_at(x1)[:, isol+N, 1]) .* surf_l)
    end

    q_r = profiles.q_spline(x2; hint=qhint)
    Fr, Kr, _ = gal_get_fkg(ffit, intr, x2, q_r)
    pbt, _ = gal_hermite(x2, x1, x2)
    surf_r = Fr * du2 .+ Kr * ua2
    for ip in 0:np
        @views cell.rhs[:, ip+1] .+= surf_r .* (pbt[ip+1] * w2)
    end
    if cell.etype == GCT_EXT && cell.extra == GAL_SIDE_LEFT
        cell.erhs += sum(conj.(ua_at(x2)[:, isol+N, 1]) .* surf_r)
    end
    return cell
end

"""
    gal_resonant!(cell, ising, ffit, profiles, intr, asymps, sings, nn, gal_tol, gal_gnstep, verbose)

Compute the resonant-cell contributions `cell.erhs`, `cell.ediag`, `cell.rhs`, `cell.emat` by adaptive
quadrature (QuadGK) of the residual-operator integrand. Port of `gal_lsode_int`/`gal_lsode_der`
(gal.f): the Fortran LSODE accumulation is replaced by `quadgk` over `[x_bdy, x_lsode]`.
"""
function gal_resonant!(cell::GalCell, ising::Int, ffit::FourFitVars, profiles,
    intr::ForceFreeStatesInternal, asymps::Vector{GalSingAsymp}, sings::Vector{SingType},
    nn::Int, gal_tol::Float64, gal_gnstep::Int, verbose::Bool)

    N = intr.numpert_total
    np = GAL_NP
    x1, x2 = cell.x
    qhint = Ref(1)

    jsing = _cell_jsing(cell, ising)
    asymp = asymps[jsing]
    psi_s = sings[jsing].psifac
    isol_big = _ipert_res(sings[jsing], intr)
    isol_small = isol_big + N
    cols = [isol_big, isol_small]

    # Integration limits (gal.f): right → [x2, x_lsode]; left → [x1, x_lsode]
    x0 = (cell.extra == GAL_SIDE_LEFT) ? x1 : x2
    x1l = cell.x_lsode
    sgn_u = (cell.extra == GAL_SIDE_RIGHT) ? -1.0 : 1.0

    L = 2 + 2 * N * (np + 1)
    function integrand(x)
        z = x - psi_s
        ua = sing_get_ua_gal(asymp, z)
        dua = sing_get_dua_gal(asymp, z)
        q = profiles.q_spline(x; hint=qhint)
        ua2 = ua[:, cols, :]
        dua2 = dua[:, cols, :]
        mv = sing_matvec(ffit, intr, x, q, ua2, dua2)   # N×2 (col1=big, col2=small)
        pbt, _ = gal_hermite(x, x1, x2)
        w = conj.(@view ua2[:, 2, 1])                   # conj small solution, qty1
        out = Vector{ComplexF64}(undef, L)
        out[1] = sum(w .* @view mv[:, 1])               # res1
        out[2] = sum(w .* @view mv[:, 2])               # res2
        idx = 3
        for ip in 0:np, ipert in 1:N
            out[idx] = pbt[ip+1] * mv[ipert, 1]         # hbig (ipert fastest, then ip)
            idx += 1
        end
        for ip in 0:np, ipert in 1:N
            out[idx] = pbt[ip+1] * mv[ipert, 2]         # hsmall
            idx += 1
        end
        return out
    end

    # Numerical-method deviation from gal_lsode_int: adaptive QuadGK (Gauss-Kronrod) replaces Fortran's
    # LSODE accumulation over the same integrand/limits, so this path is NOT bit-exact with gal.f. Match
    # Fortran's LSODE tolerance (gal_tol) and cap the evaluations at gal_gnstep (Fortran's gnstep limit)
    # so a near-singular integrand cannot hang.
    raw, qerr = quadgk(integrand, x0, x1l; rtol=gal_tol, atol=1e-30, maxevals=gal_gnstep)
    if verbose
        @info "  resonant jsing=$jsing side=$(cell.extra) qerr=$(qerr) res1=$(raw[1]) res2=$(raw[2])"
    end
    hbig = reshape(@view(raw[3:2+N*(np+1)]), N, np + 1)
    hsmall = reshape(@view(raw[3+N*(np+1):end]), N, np + 1)

    # Apply the right-side negation and the gal_make_arrays signs (gal.f gal_make_arrays)
    cell.erhs = -sgn_u * raw[1]
    cell.ediag = sgn_u * raw[2]
    cell.rhs .= (-sgn_u) .* hbig
    cell.emat .= sgn_u .* hsmall
    return cell
end

"""
    gal_assemble_mat!(ws, mpert)

Scatter every cell's `mat`/`emat`/`ediag` into the banded global matrix `ws.mat`. Port of
`gal_assemble_mat` (gal.f). Supports both `"LU"` (full band, `offset = kl+ku+1`) and
`"cholesky"` (lower band only, `offset = 1`).
"""
function gal_assemble_mat!(ws::GalWorkspace, mpert::Int)
    np = ws.np
    msing = length(ws.intvl) - 1
    lu = ws.solver == "LU"
    offset = lu ? ws.kl + ws.ku + 1 : 1

    for ising in 0:msing
        for cell in ws.intvl[ising+1].cells
            res = cell.etype == GCT_RES || cell.etype == GCT_EXT
            for ip in 0:np, ipert in 1:mpert
                i = cell.map[ipert, ip+1]
                # nonresonant stiffness
                for jp in 0:np, jpert in 1:mpert
                    j = cell.map[jpert, jp+1]
                    if lu
                        ws.mat[offset+i-j, j] += cell.mat[ipert, jpert, ip+1, jp+1]
                    elseif j <= i
                        ws.mat[offset+i-j, j] += cell.mat[ipert, jpert, ip+1, jp+1]
                    end
                end
                # resonant coupling (res/ext only)
                if res
                    j = cell.emap
                    e = cell.emat[ipert, ip+1]
                    if lu
                        ws.mat[offset+i-j, j] += e
                        ws.mat[offset+j-i, i] += conj(e)
                    elseif j < i
                        ws.mat[offset+i-j, j] += e
                    elseif j > i
                        ws.mat[offset+j-i, i] += conj(e)
                    end
                end
            end
            if res
                ws.mat[offset, cell.emap] += cell.ediag
            end
        end
    end
    return ws
end

"""
    gal_assemble_rhs!(ws, mpert)

Scatter every cell's `rhs`/`erhs` into the global RHS `ws.rhs`. Port of `gal_assemble_rhs`
(gal.f). A new RHS column (`isol`) begins at each big-solution driver: an `ext2`-left cell or a
`res`-right cell.
"""
function gal_assemble_rhs!(ws::GalWorkspace, mpert::Int)
    np = ws.np
    msing = length(ws.intvl) - 1
    isol = 0
    for ising in 0:msing
        for cell in ws.intvl[ising+1].cells
            cell.etype == GCT_NONE && continue
            if cell.etype == GCT_EXT2 && cell.extra == GAL_SIDE_LEFT
                isol += 1
            elseif cell.etype == GCT_RES && cell.extra == GAL_SIDE_RIGHT
                isol += 1
            end
            for ip in 0:np, ipert in 1:mpert
                imap = cell.map[ipert, ip+1]
                ws.rhs[imap, isol] += cell.rhs[ipert, ip+1]
            end
            if cell.etype == GCT_EXT || cell.etype == GCT_RES
                ws.rhs[cell.emap, isol] += cell.erhs
            end
        end
    end
    return ws
end

"""
    gal_set_boundary!(ws, mpert, wv_edge)

Apply the axis (`u(0)=0` identity) and edge boundary conditions to the boundary cells' `mat`, plus the
rpec coil-source RHS injection, before global assembly. Port of `gal_set_boundary` (gal.f),
including its full three-way edge branch (rpec count `ncoil = ws.nsol - 2*msing`):
  - rpec (`ncoil > 0`): identity edge **and** a unit source at the edge value DOF of coil column
    `2*msing+ipert` for each poloidal mode `ipert` (gal.f). `ws.rhs` is already allocated and
    `gal_assemble_rhs!` only fills columns `1:2*msing`, so the injection is not clobbered.
  - free boundary (`wv_edge !== nothing`): add the vacuum block `wvac·psio²` at the edge value DOF
    (gal.f).
  - fixed boundary (else): identity edge (gal.f).
`wv_edge` is `nothing` for the rpec and fixed cases.
"""
function gal_set_boundary!(ws::GalWorkspace, mpert::Int, wv_edge::Union{Nothing,Matrix{ComplexF64}})
    chol = ws.solver == "cholesky"
    msing = length(ws.intvl) - 1
    ncoil = ws.nsol - 2 * msing

    # axis u(0): first cell of interval 0, Hermite value DOF ip=0 (Julia index 1)
    cell = ws.intvl[1].cells[1]
    cell.mat[:, :, 1, :] .= 0
    chol && (cell.mat[:, :, :, 1] .= 0)
    for idx in 1:mpert
        cell.mat[idx, idx, 1, 1] = 1
    end

    # edge u(1): last cell of last interval, edge-value DOF ip=3 (Julia index 4, see gauss_quad swap)
    cell = ws.intvl[msing+1].cells[ws.nx]
    if ncoil > 0
        # rpec: identity edge + unit coil sources (gal.f)
        cell.mat[:, :, 4, :] .= 0
        chol && (cell.mat[:, :, :, 4] .= 0)
        for idx in 1:mpert
            cell.mat[idx, idx, 4, 4] = 1
        end
        for ipert in 1:mpert
            ws.rhs[cell.map[ipert, 4], 2 * msing + ipert] = 1
        end
    elseif wv_edge !== nothing
        # free boundary: vacuum block wvac·psio² (gal.f)
        @views cell.mat[:, :, 4, 4] .+= wv_edge
    else
        # fixed boundary: identity edge (gal.f)
        cell.mat[:, :, 4, :] .= 0
        chol && (cell.mat[:, :, :, 4] .= 0)
        for idx in 1:mpert
            cell.mat[idx, idx, 4, 4] = 1
        end
    end
    return ws
end
