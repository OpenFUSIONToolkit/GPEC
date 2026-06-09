# GalerkinSolve.jl
#
# Top-level driver for the RDCON outer-region singular Galerkin Δ′ solve, plus the per-cell assembly
# orchestration, the banded solve, Δ′ extraction, PEST-3 blocks, and HDF5 output.
# Ports gal_make_arrays (gal.f:1300-1359), gal_solve (gal.f:1367-1431), and gal_write_pest3_data
# (gal.f:1712-1745). No inner-layer wiring (future thrust).

"""
    gal_make_arrays!(ws, ctrl, equil, ffit, intr, asymps, sings, nn, wv_edge)

Assemble the global banded matrix and RHS. For each cell: Gauss-Lobatto Hermite stiffness
(`gal_gauss_quad!`), then the resonant (`gal_resonant!`) or extension (`gal_extension!`) contributions;
then the boundary conditions and the scatter into `ws.mat`/`ws.rhs`. Port of `gal_make_arrays`.
"""
function gal_make_arrays!(ws::GalWorkspace, ctrl::ForceFreeStatesControl, equil, ffit::FourFitVars,
    intr::ForceFreeStatesInternal, asymps::Vector{GalSingAsymp}, sings::Vector{SingType},
    nn::Int, wv_edge::Union{Nothing,Matrix{ComplexF64}})

    mpert = intr.numpert_total
    msing = length(ws.intvl) - 1
    profiles = equil.profiles
    nodes, weights = gausslobatto(ws.nq + 1)

    for ising in 0:msing
        for (ix, cell) in enumerate(ws.intvl[ising+1].cells)
            swap_edge = (ising == msing && ix == ws.nx)
            gal_gauss_quad!(cell, ffit, profiles, intr, nodes, weights, swap_edge)
            if cell.etype == GCT_RES
                gal_resonant!(cell, ising, ffit, profiles, intr, asymps, sings, nn, ctrl.gal_tol)
            elseif cell.etype == GCT_EXT || cell.etype == GCT_EXT1 || cell.etype == GCT_EXT2
                gal_extension!(cell, ising, ffit, profiles, intr, asymps, sings, nn, nodes, weights)
            end
        end
    end

    gal_set_boundary!(ws, mpert, wv_edge)
    gal_assemble_mat!(ws, mpert)
    gal_assemble_rhs!(ws, mpert)
    return ws
end

"""
    galerkin_solve(ctrl::ForceFreeStatesControl, equil, ffit::FourFitVars,
                   intr::ForceFreeStatesInternal; vac_data=nothing) -> GalerkinResult

Compute the outer-region Δ′ matching matrix by the singular Galerkin method. Port of `gal_solve`
(gal.f:1367-1431). Single toroidal mode only (`intr.npert == 1`). Returns a `GalerkinResult`; if there
are no resonant surfaces in the domain it returns an empty result.

`vac_data` (a `VacuumData` from `free_run!`) supplies the free-boundary edge term
`wv_edge = vac_data.wv · psio²`; pass `nothing` for a fixed-boundary edge.
"""
function galerkin_solve(ctrl::ForceFreeStatesControl, equil, ffit::FourFitVars,
    intr::ForceFreeStatesInternal; vac_data=nothing)

    intr.npert == 1 || error("galerkin_solve: only single-n (npert == 1) is supported")
    ctrl.gal_solver in ("LU", "cholesky") || error("galerkin_solve: gal_solver must be \"LU\" or \"cholesky\"")
    nn = intr.nlow
    mpert = intr.numpert_total
    np = GAL_NP

    # Resonant surfaces inside the integration domain, in increasing ψ (Fortran `sing(1:msing)`).
    # psilow is raised above the axis by sing_min! when qlow > qmin (excludes the q<qlow core);
    # fall back to the equilibrium axis bound if sing_min! was not run (psilow still 0).
    psilow = intr.psilow > 0 ? intr.psilow : equil.profiles.xs[1]
    psihigh = intr.psilim
    sings = [s for s in intr.sing if psilow < s.psifac < psihigh && intr.mlow <= s.m[1] <= intr.mhigh]
    msing = length(sings)
    if msing == 0
        ctrl.verbose && @info "galerkin_solve: no resonant surfaces in domain; skipping Δ′ solve"
        empty2 = Matrix{ComplexF64}(undef, 0, 0)
        return GalerkinResult(Matrix{ComplexF64}(undef, 0, 0), empty2, empty2, empty2, empty2, 0,
            Float64[], Float64[], Int[], Int[], Float64[], ComplexF64[])
    end

    ctrl.verbose && @info "Starting outer-region Galerkin Δ′ solve (msing=$msing, solver=$(ctrl.gal_solver))"

    # Per-surface two-sided asymptotic series (pr178's validated convention, matching Fortran sing.f
    # vmatr/vmatl): right = sig=+1, left = sig=-1, no √det normalization. The Mercier exponent α is a
    # property of the surface, so the left series reuses the right's α (alpha_override). The order is
    # raised to gal_sing_order + ceil(2·Re(α)) for high-Mercier-index surfaces (Fortran sing1_vmat).
    ctrl_gal = deepcopy(ctrl)
    asymps = GalSingAsymp[]
    for s in sings
        ctrl_gal.sing_order = ctrl.gal_sing_order
        ar = compute_sing_asymptotics(s, ctrl_gal, equil, ffit, intr; sig=1.0)
        if ctrl.gal_sing_order_ceiling
            order = ctrl.gal_sing_order + ceil(Int, 2 * real(ar.alpha[1]))
            if order > ctrl.gal_sing_order
                ctrl_gal.sing_order = order
                ar = compute_sing_asymptotics(s, ctrl_gal, equil, ffit, intr; sig=1.0)
            end
        end
        al = compute_sing_asymptotics(s, ctrl_gal, equil, ffit, intr; sig=-1.0, alpha_override=ar.alpha)
        push!(asymps, GalSingAsymp(ar, al))
    end

    # --- allocate workspace ---
    nx = ctrl.gal_nx
    kl = mpert * (np + 1)
    ku = kl
    ldab = ctrl.gal_solver == "LU" ? 2kl + ku + 1 : kl + 1
    nsol = 2 * msing
    intvl = [GalInterval(zeros(Float64, nx + 1), zeros(Float64, nx + 1), [GalCell(mpert) for _ in 1:nx])
             for _ in 0:msing]
    ws = GalWorkspace(ctrl.gal_solver, nx, ctrl.gal_nq, np, 0, kl, ku, ldab, nsol,
        intvl, Matrix{ComplexF64}(undef, 0, 0), Matrix{ComplexF64}(undef, 0, 0), Matrix{ComplexF64}(undef, 0, 0))

    for ising in 0:msing
        gal_make_grid!(ws.intvl[ising+1], ising, msing, sings, nn, psilow, psihigh, ctrl)
    end
    gal_make_map!(ws, mpert)

    ws.mat = zeros(ComplexF64, ldab, ws.ndim)
    ws.rhs = zeros(ComplexF64, ws.ndim, nsol)
    ws.sol = zeros(ComplexF64, ws.ndim, nsol)

    # Free-boundary edge term wvac·psio² (vac_data.wv is already singfac-scaled at qlim in free_run!)
    wv_edge = nothing
    if ctrl.vac_flag && vac_data !== nothing
        wv_edge = Matrix{ComplexF64}(vac_data.wv .* equil.psio^2)
    end

    gal_make_arrays!(ws, ctrl, equil, ffit, intr, asymps, sings, nn, wv_edge)

    if get(ENV, "GAL_DEBUG", "") == "1"
        offdbg = ws.solver == "LU" ? ws.kl + ws.ku + 1 : 1
        @info "GAL_DEBUG ndim=$(ws.ndim) nsol=$nsol kl=$(ws.kl) ldab=$(ws.ldab) |rhs|=$(norm(ws.rhs))"
        for ising in 0:msing
            for cell in ws.intvl[ising+1].cells
                if cell.etype == GCT_RES || cell.etype == GCT_EXT
                    @info "  cell ising=$ising etype=$(cell.etype) side=$(cell.extra) emap=$(cell.emap) ediag=$(cell.ediag) erhs=$(cell.erhs) |rhs|=$(norm(cell.rhs)) |emat|=$(norm(cell.emat)) matdiag[emap]=$(ws.mat[offdbg, cell.emap])"
                end
            end
        end
    end

    # --- banded solve (gal.f:1383-1396) ---
    ws.sol .= ws.rhs
    if ws.solver == "LU"
        ctrl.verbose && @info "Galerkin LU banded factorization + solve"
        ab, ipiv = LinearAlgebra.LAPACK.gbtrf!(ws.kl, ws.ku, ws.ndim, ws.mat)
        LinearAlgebra.LAPACK.gbtrs!('N', ws.kl, ws.ku, ws.ndim, ab, ipiv, ws.sol)
    else
        ctrl.verbose && @info "Galerkin Cholesky banded factorization + solve"
        LinearAlgebra.LAPACK.pbtrf!('L', ws.kl, ws.mat)
        LinearAlgebra.LAPACK.pbtrs!('L', ws.kl, ws.mat, ws.sol)
    end

    # --- extract Δ′ from the small resonant coefficients (gal.f:1405-1417) ---
    delta = zeros(ComplexF64, nsol, 2 * msing)
    jsol = 0
    for ising in 0:msing
        for cell in ws.intvl[ising+1].cells
            if cell.etype == GCT_RES
                jsol += 1
                @views delta[:, jsol] .= ws.sol[cell.emap, :]
            end
        end
    end

    Ap, Bp, Gammap, Deltap = gal_pest3_blocks(delta, msing)

    di = [real(-asymps[i].right.alpha[1]^2) for i in 1:msing]
    alpha = [asymps[i].right.alpha[1] for i in 1:msing]
    return GalerkinResult(delta, Ap, Bp, Gammap, Deltap, msing,
        [s.psifac for s in sings], [s.q for s in sings],
        [s.m[1] for s in sings], [s.n[1] for s in sings], di, alpha)
end

"""
    gal_pest3_blocks(delta, msing) -> (Ap, Bp, Gammap, Deltap)

PEST-3 matching blocks (each `msing×msing`) as ± combinations of the left/right Δ′ entries. Port of
`gal_write_pest3_data` (gal.f:1726-1745). Row index `2i-1`/`2i` = left/right of surface `i`.
"""
function gal_pest3_blocks(delta::Matrix{ComplexF64}, msing::Int)
    Ap = zeros(ComplexF64, msing, msing)
    Bp = zeros(ComplexF64, msing, msing)
    Gammap = zeros(ComplexF64, msing, msing)
    Deltap = zeros(ComplexF64, msing, msing)
    for i in 1:msing, j in 1:msing
        d11 = delta[2i-1, 2j-1]
        d12 = delta[2i-1, 2j]
        d21 = delta[2i, 2j-1]
        d22 = delta[2i, 2j]
        Ap[i, j] = d22 + d21 + d12 + d11
        Bp[i, j] = d22 - d21 + d12 - d11
        Gammap[i, j] = d22 + d21 - d12 - d11
        Deltap[i, j] = d22 - d21 - d12 + d11
    end
    return Ap, Bp, Gammap, Deltap
end

"""
    write_galerkin!(out_h5, result::GalerkinResult)

Write the Galerkin Δ′ outputs into the open HDF5 file under the `galerkin/` group. Replaces the Fortran
`delta_gw`/`pest3_data` ASCII/binary outputs.
"""
function write_galerkin!(out_h5, result::GalerkinResult)
    out_h5["galerkin/msing"] = result.msing
    result.msing == 0 && return nothing
    out_h5["galerkin/delta"] = result.delta
    out_h5["galerkin/pest3_A"] = result.Ap
    out_h5["galerkin/pest3_B"] = result.Bp
    out_h5["galerkin/pest3_Gamma"] = result.Gammap
    out_h5["galerkin/pest3_Delta"] = result.Deltap
    out_h5["galerkin/sing_psi"] = result.sing_psi
    out_h5["galerkin/sing_q"] = result.sing_q
    out_h5["galerkin/sing_m"] = result.sing_m
    out_h5["galerkin/sing_n"] = result.sing_n
    out_h5["galerkin/di"] = result.di
    out_h5["galerkin/alpha"] = result.alpha
    return nothing
end
