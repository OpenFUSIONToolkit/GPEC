# GalerkinSolve.jl
#
# Top-level driver for the RDCON outer-region singular Galerkin Δ′ solve, plus the per-cell assembly
# orchestration, the banded solve, Δ′ extraction, PEST-3 blocks, and HDF5 output.
# Ports gal_make_arrays (gal.f), gal_solve (gal.f), and gal_write_pest3_data
# (gal.f). The DRIVEN/RPEC inner-layer matching is wired in via gal_match_rpec (GalerkinMatch.jl).

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
                gal_resonant!(cell, ising, ffit, profiles, intr, asymps, sings, nn, ctrl.gal_tol, ctrl.gal_gnstep, ctrl.verbose)
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

"""Empty `GalerkinResult` for a domain with no resonant surfaces."""
function empty_galerkin_result()
    empty2 = Matrix{ComplexF64}(undef, 0, 0)
    return GalerkinResult(empty2, empty2, empty2, empty2, empty2, 0,
        Float64[], Float64[], Int[], Int[], Float64[], ComplexF64[], empty2, nothing, nothing)
end

"""
    galerkin_solve(ctrl::ForceFreeStatesControl, equil, ffit::FourFitVars,
                   intr::ForceFreeStatesInternal; vac_data=nothing) -> GalerkinResult

Compute the outer-region Δ′ matching matrix by the singular Galerkin method. Port of `gal_solve`
(gal.f). Single toroidal mode only (`intr.npert == 1`). Returns a `GalerkinResult`; if there
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

    sings, psilow, psihigh = gal_resonant_surfaces(intr, equil)
    msing = length(sings)
    if msing == 0
        ctrl.verbose && @info "galerkin_solve: no resonant surfaces in domain; skipping Δ′ solve"
        return empty_galerkin_result()
    end

    ctrl.verbose && @info "Starting outer-region Galerkin Δ′ solve (msing=$msing, solver=$(ctrl.gal_solver))"

    # Per-surface two-sided asymptotic series matching Fortran sing.f vmatr/vmatl:
    # right = sig=+1, left = sig=-1, no √det normalization. The Mercier exponent α is a
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
    # rpec_flag (RDCON, gal.f): append mpert coil-response columns. Each is a unit source at
    # the edge value DOF in one poloidal mode; the recorded plasma response is the coil block of Δ_gw.
    # Cholesky's lower-only edge zeroing can't represent the rpec identity edge, so require LU.
    ncoil = ctrl.gal_rpec_flag ? mpert : 0
    if ncoil > 0 && ctrl.gal_solver != "LU"
        error("galerkin_solve: gal_rpec_flag=true requires gal_solver=\"LU\" (coil edge BC needs the full-band path)")
    end
    nsol = 2 * msing + ncoil
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

    # Free-boundary edge term wvac·psio² (vac_data.wv is already singfac-scaled at qlim in free_run!).
    # rpec passes wv_edge=nothing; gal_set_boundary! then applies the identity edge AND injects the coil
    # unit sources (its three-way branch). The vacuum block is only built for the non-rpec free case.
    wv_edge = nothing
    if ctrl.vac_flag && vac_data !== nothing && ncoil == 0
        wv_edge = Matrix{ComplexF64}(vac_data.wv .* equil.psio^2)
    end

    gal_make_arrays!(ws, ctrl, equil, ffit, intr, asymps, sings, nn, wv_edge)

    if ctrl.verbose
        offdbg = ws.solver == "LU" ? ws.kl + ws.ku + 1 : 1
        @info "Galerkin assembly: ndim=$(ws.ndim) nsol=$nsol kl=$(ws.kl) ldab=$(ws.ldab) |rhs|=$(norm(ws.rhs))"
        for ising in 0:msing
            for cell in ws.intvl[ising+1].cells
                if cell.etype == GCT_RES || cell.etype == GCT_EXT
                    @info "  cell ising=$ising etype=$(cell.etype) side=$(cell.extra) emap=$(cell.emap) ediag=$(cell.ediag) erhs=$(cell.erhs) |rhs|=$(norm(cell.rhs)) |emat|=$(norm(cell.emat)) matdiag[emap]=$(ws.mat[offdbg, cell.emap])"
                end
            end
        end
    end

    # --- banded solve (gal.f) ---
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

    # --- extract Δ′ from the small resonant coefficients (gal.f) ---
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

    # Mercier index D_I = -Re(α²); α is the small/large solution exponent (a surface property, taken
    # from the right series — the left series reuses the same α via alpha_override above).
    di = [real(-asymps[i].right.alpha[1]^2) for i in 1:msing]
    alpha = [asymps[i].right.alpha[1] for i in 1:msing]
    # Coil-response block (rpec_flag): rows 2*msing+1 : 2*msing+mpert of delta (empty otherwise).
    delta_coil = ncoil > 0 ? delta[(2*msing+1):(2*msing+ncoil), :] : Matrix{ComplexF64}(undef, 0, 0)

    # Reconstruct ξ(ψ) AND analytic ξ′(ψ) on the gal-native grid (gal_output_solution).
    ctrl.verbose && @info "Reconstructing outer-region ξ and analytic ξ′ on the gal grid"
    solution = gal_output_solution(ws, asymps, sings, intr, equil.profiles, psihigh;
        delta=(ctrl.gal_match_flag ? delta : nothing))

    sing_psi = [s.psifac for s in sings]
    sing_q = [s.q for s in sings]
    sing_m = [s.m[1] for s in sings]
    sing_n = [s.n[1] for s in sings]
    result = GalerkinResult(delta, Ap, Bp, Gammap, Deltap, msing,
        sing_psi, sing_q, sing_m, sing_n, di, alpha, delta_coil, solution, nothing)

    # DRIVEN (RPEC) outer↔inner matching: build the coil-driven matched ξ/ξ′ (gal_match_rpec).
    ctrl.gal_match_flag || return result
    ctrl.gal_rpec_flag || error("galerkin_solve: gal_match_flag=true requires gal_rpec_flag=true")
    ctrl.verbose && @info(
        ctrl.gal_ideal_flag ?
        "RPEC matching: IDEAL solution (inner layer skipped, bare coil columns)" :
        "RPEC matching: inner-layer Δ(Q) + outer↔inner solve for the coil-driven ξ"
    )
    match = gal_match_rpec(ctrl, equil, intr, result)
    ctrl.gal_ideal_flag || (ctrl.verbose && @info "RPEC matching: linear-solve residual = $(match.residual)")
    return GalerkinResult(delta, Ap, Bp, Gammap, Deltap, msing,
        sing_psi, sing_q, sing_m, sing_n, di, alpha, delta_coil, solution, match)
end

"""
    gal_pest3_blocks(delta, msing) -> (Ap, Bp, Gammap, Deltap)

PEST-3 matching blocks (each `msing×msing`) as ± combinations of the left/right Δ′ entries. Port of
`gal_write_pest3_data`. Row index `2i-1`/`2i` = left/right of surface `i`.
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
    if !isempty(result.delta_coil)
        out_h5["galerkin/delta_coil"] = result.delta_coil
    end
    if result.solution !== nothing
        sol = result.solution
        out_h5["galerkin/solution/psi"] = sol.psi
        out_h5["galerkin/solution/q"] = sol.q
        out_h5["galerkin/solution/issing"] = collect(sol.issing)
        out_h5["galerkin/solution/xi"] = sol.xi
        out_h5["galerkin/solution/xi_deriv"] = sol.xi_deriv
        isempty(sol.xi_cut) || (out_h5["galerkin/solution/xi_cut"] = sol.xi_cut)
        isempty(sol.cut_range) || (out_h5["galerkin/solution/cut_range"] = sol.cut_range)
    end
    if result.match !== nothing
        m = result.match
        out_h5["galerkin/match/cout"] = m.cout
        out_h5["galerkin/match/cin"] = m.cin
        out_h5["galerkin/match/xi"] = m.xi
        out_h5["galerkin/match/xi_deriv"] = m.xi_deriv
        out_h5["galerkin/match/deltar"] = m.deltar
        out_h5["galerkin/match/bpen"] = m.bpen
        out_h5["galerkin/match/rpec_eig"] = m.rpec_eig
        # Per-surface inner-layer ξ_ψ(ψ) (match.f intotsol); ragged grids → one dataset pair per surface.
        for i in eachindex(m.inner_psi)
            out_h5["galerkin/match/inner/psi_$i"] = m.inner_psi[i]
            out_h5["galerkin/match/inner/xi_$i"] = m.inner_xi[i]
            out_h5["galerkin/match/inner/b_$i"] = m.inner_b[i]
        end
        out_h5["galerkin/match/residual"] = m.residual
        # Per-surface GGJ inner-layer coefficients (resist_eval), replacing the old
        # GAL_DUMP_INNER temp-CSV hook: everything needed to reconstruct the layer
        # problem (E,F,G,H,K,M) and its scales (taua, taur ⇒ S = taur/taua, v1).
        if !isempty(m.inner_params)
            for f in (:E, :F, :G, :H, :K, :M, :taua, :taur, :v1)
                out_h5["galerkin/match/inner_params/$(f)"] = [getfield(pp, f) for pp in m.inner_params]
            end
        end
    end
    return nothing
end
