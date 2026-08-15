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
                   intr::ForceFreeStatesInternal; wv=nothing) -> GalerkinResult

Compute the outer-region Δ′ matching matrix by the singular Galerkin method. Port of `gal_solve`
(gal.f). Single toroidal mode only (`intr.npert == 1`). Returns a `GalerkinResult`; if there
are no resonant surfaces in the domain it returns an empty result.

`wv` is the vacuum energy matrix from `free_run`, which supplies the free-boundary edge term
`wv_edge = wv · psio²`; pass `nothing` for a fixed-boundary edge.
"""
function galerkin_solve(ctrl::ForceFreeStatesControl, equil, ffit::FourFitVars,
    intr::ForceFreeStatesInternal; wv=nothing)

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
    asymps = GalSingAsymp[]
    for s in sings
        sing_order = ctrl.gal_sing_order
        ar = compute_sing_asymptotics(s, ctrl, equil, ffit, intr; sig=1.0, sing_order=sing_order)
        if ctrl.gal_sing_order_ceiling
            order = ctrl.gal_sing_order + ceil(Int, 2 * real(ar.alpha[1]))
            if order > ctrl.gal_sing_order
                sing_order = order
                ar = compute_sing_asymptotics(s, ctrl, equil, ffit, intr; sig=1.0, sing_order=sing_order)
            end
        end
        al = compute_sing_asymptotics(s, ctrl, equil, ffit, intr; sig=-1.0, alpha_override=ar.alpha, sing_order=sing_order)
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

    # Free-boundary edge term wvac·psio² (wv is already singfac-scaled at qlim in free_run).
    # rpec passes wv_edge=nothing; gal_set_boundary! then applies the identity edge AND injects the coil
    # unit sources (its three-way branch). The vacuum block is only built for the non-rpec free case.
    wv_edge = nothing
    if ctrl.vac_flag && wv !== nothing && ncoil == 0
        wv_edge = Matrix{ComplexF64}(wv .* equil.psio^2)
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

Write the Galerkin outputs into the open HDF5 file. The integrator's solution functions and
RPEC matching data go under `ForceFreeStates/Solutions/GalerkinIntegration/`; the per-surface
Δ′/PEST-3 matching results consolidate with the other rational-surface stability results under
`SingularSurfaces/GalerkinDeltaPrime/`. Replaces the Fortran `delta_gw`/`pest3_data`
ASCII/binary outputs.
"""
function write_galerkin!(out_h5, result::GalerkinResult)
    gal = "ForceFreeStates/Solutions/GalerkinIntegration"
    gdp = "SingularSurfaces/GalerkinDeltaPrime"
    out_h5["$gal/rational_count"] = result.msing
    if result.msing == 0
        annotate_galerkin!(out_h5)
        return nothing
    end
    out_h5["$gdp/Delta_prime_raw"] = result.delta
    out_h5["$gdp/pest3_A"] = result.Ap
    out_h5["$gdp/pest3_B"] = result.Bp
    out_h5["$gdp/pest3_Gamma"] = result.Gammap
    out_h5["$gdp/pest3_Delta"] = result.Deltap
    out_h5["$gdp/rational_psi"] = result.sing_psi
    out_h5["$gdp/rational_q"] = result.sing_q
    out_h5["$gdp/rational_m"] = result.sing_m
    out_h5["$gdp/rational_n"] = result.sing_n
    out_h5["$gdp/D_I"] = result.di
    out_h5["$gdp/alpha"] = result.alpha
    if !isempty(result.delta_coil)
        out_h5["$gdp/Delta_coil"] = result.delta_coil
    end
    if result.solution !== nothing
        sol = result.solution
        out_h5["$gal/Solution/psi"] = sol.psi
        out_h5["$gal/Solution/is_rational"] = collect(sol.issing)
        out_h5["$gal/Solution/xi_psi"] = sol.xi
        out_h5["$gal/Solution/dxi_psidpsi"] = sol.xi_deriv
        isempty(sol.xi_cut) || (out_h5["$gal/Solution/xi_psi_cut"] = sol.xi_cut)
        isempty(sol.cut_range) || (out_h5["$gal/Solution/cut_range"] = sol.cut_range)
    end
    if result.match !== nothing
        m = result.match
        out_h5["$gal/Match/cout"] = m.cout
        out_h5["$gal/Match/cin"] = m.cin
        out_h5["$gal/Match/xi"] = m.xi
        out_h5["$gal/Match/dxidpsi"] = m.xi_deriv
        out_h5["$gal/Match/Delta_r"] = m.deltar
        out_h5["$gal/Match/bpen"] = m.bpen
        out_h5["$gal/Match/rpec_eig"] = m.rpec_eig
        # Per-surface inner-layer ξ_ψ(ψ) (match.f intotsol); ragged grids → one dataset pair per surface.
        for i in eachindex(m.inner_psi)
            out_h5["$gal/Match/Inner/psi_$i"] = m.inner_psi[i]
            out_h5["$gal/Match/Inner/xi_$i"] = m.inner_xi[i]
            out_h5["$gal/Match/Inner/b_$i"] = m.inner_b[i]
        end
        out_h5["$gal/Match/residual"] = m.residual
        if !isempty(m.inner_params)
            for f in (:E, :F, :G, :H, :K, :M)
                out_h5["$gal/Match/InnerParams/$(f)"] = [getfield(pp, f) for pp in m.inner_params]
            end
            # Literature names, matching the Tearing PerSurface mapping for the same fields.
            out_h5["$gal/Match/InnerParams/tau_A"] = [pp.taua for pp in m.inner_params]
            out_h5["$gal/Match/InnerParams/tau_R"] = [pp.taur for pp in m.inner_params]
            out_h5["$gal/Match/InnerParams/dVdpsi"] = [pp.v1 for pp in m.inner_params]
        end
    end
    annotate_galerkin!(out_h5)
    return nothing
end

# Metadata tables for the Galerkin outputs (Match/** is debug-only and exempt from the
# metadata contract; see docs/development/hdf5-conventions.md).
const GALERKIN_H5_ANNOTATIONS = [
    "ForceFreeStates/Solutions/GalerkinIntegration/rational_count" => (; long_name="number of rational (singular) surfaces in the Galerkin solve"),
    "ForceFreeStates/Solutions/GalerkinIntegration/Solution/psi" => (; long_name="normalized poloidal flux ψ_N grid of the Galerkin solution", scale="psi"),
    "ForceFreeStates/Solutions/GalerkinIntegration/Solution/is_rational" =>
        (; long_name="flag: grid node lies on a rational surface", dims=("psi",), attach=(1 => "ForceFreeStates/Solutions/GalerkinIntegration/Solution/psi",)),
    "ForceFreeStates/Solutions/GalerkinIntegration/Solution/xi_psi" =>
        (; long_name="Galerkin solution functions ξ^ψ (arbitrary amplitude; note the psi/solution axis order differs from ForwardIntegration)", dims=("mode", "psi", "solution")),
    "ForceFreeStates/Solutions/GalerkinIntegration/Solution/dxi_psidpsi" =>
        (; long_name="ψ_N derivative of the Galerkin solution functions ξ^ψ (arbitrary amplitude)", dims=("mode", "psi", "solution")),
    "ForceFreeStates/Solutions/GalerkinIntegration/Solution/xi_psi_cut" =>
        (; long_name="Galerkin solution functions ξ^ψ with the leading-order resonant response excised", dims=("mode", "psi", "solution")),
    "ForceFreeStates/Solutions/GalerkinIntegration/Solution/cut_range" =>
        (; long_name="ψ_N bounds of the excised resonant + extension cells per surface", dims=("surface", "bound")),
    "SingularSurfaces/GalerkinDeltaPrime/Delta_prime_raw" =>
        (; long_name="outer-region Δ' matrix (2msing×2msing, side-major [L_s1, R_s1, ...])", dims=("surface_side_row", "surface_side_col")),
    "SingularSurfaces/GalerkinDeltaPrime/pest3_A" => (; long_name="PEST-3 matching block A' (Galerkin outer region)", dims=("surface_row", "surface_col")),
    "SingularSurfaces/GalerkinDeltaPrime/pest3_B" => (; long_name="PEST-3 matching block B' (Galerkin outer region)", dims=("surface_row", "surface_col")),
    "SingularSurfaces/GalerkinDeltaPrime/pest3_Gamma" => (; long_name="PEST-3 matching block Γ' (Galerkin outer region)", dims=("surface_row", "surface_col")),
    "SingularSurfaces/GalerkinDeltaPrime/pest3_Delta" => (; long_name="PEST-3 matching block Δ' (Galerkin outer region)", dims=("surface_row", "surface_col")),
    "SingularSurfaces/GalerkinDeltaPrime/rational_psi" => (; long_name="normalized poloidal flux ψ_N of each rational surface", scale="psi_rational"),
    "SingularSurfaces/GalerkinDeltaPrime/rational_q" =>
        (; long_name="safety factor q = m/n at each rational surface", dims=("surface",), attach=(1 => "SingularSurfaces/GalerkinDeltaPrime/rational_psi",)),
    "SingularSurfaces/GalerkinDeltaPrime/rational_m" =>
        (; long_name="resonant poloidal mode number m at each rational surface", dims=("surface",), attach=(1 => "SingularSurfaces/GalerkinDeltaPrime/rational_psi",)),
    "SingularSurfaces/GalerkinDeltaPrime/rational_n" =>
        (; long_name="resonant toroidal mode number n at each rational surface", dims=("surface",), attach=(1 => "SingularSurfaces/GalerkinDeltaPrime/rational_psi",)),
    "SingularSurfaces/GalerkinDeltaPrime/D_I" =>
        (; long_name="Mercier D_I at each rational surface", dims=("surface",), attach=(1 => "SingularSurfaces/GalerkinDeltaPrime/rational_psi",)),
    "SingularSurfaces/GalerkinDeltaPrime/alpha" =>
        (; long_name="Frobenius small-solution exponent α at each rational surface", dims=("surface",), attach=(1 => "SingularSurfaces/GalerkinDeltaPrime/rational_psi",)),
    "SingularSurfaces/GalerkinDeltaPrime/Delta_coil" => (; long_name="edge coil-response matrix (edge mode × surface-side; RPEC columns)", dims=("mode", "surface_side"))
]

# Attach long_name/units/dims + dimension scales (declared in-table) to everything
# write_galerkin! wrote.
function annotate_galerkin!(out_h5)
    Utilities.HDF5Annotations.annotate!(out_h5, GALERKIN_H5_ANNOTATIONS)
    return nothing
end
