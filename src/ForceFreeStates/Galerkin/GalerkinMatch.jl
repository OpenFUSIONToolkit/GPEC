# GalerkinMatch.jl
#
# DRIVEN (RPEC) outer↔inner asymptotic matching. Port of rmatch `match_rpec` (match.f:1318-1510) and the
# outer total-solution construction of `match_output_solution` (match.f:1654-1671).
#
# Per rational surface: the forced eigenvalue γ_s = 2πi·n·f_s (rpec, match.f:1359), the inner-layer
# matching data Δ(Q) via the Galerkin GGJ solver (InnerLayer.solve_inner), then the 4·msing matching
# system mat·[cout;cin] = rmat is assembled (rmat = −transpose of the gal Δ′ coil block) and solved for
# the per-coil outer/inner coefficients. The matched outer solution for coil drive j is
#   ξ_j = Σ_{isol=1}^{2 msing} cout[isol,j]·sols[:,:,isol] + sols[:,:,2 msing+j]
# and identically for ξ′ with sols_deriv. Stacking over j gives the matched fundamental matrix in the
# identity-at-edge basis (coil column j has ξ_edge = e_j).

"""
    gal_match_rpec(ctrl, equil, intr, gal_result) -> GalMatchResult

Solve the coil-driven RPEC matching from the outer Δ′ (`gal_result`) and the per-surface inner-layer Δ.
Requires `gal_result.solution` (reconstructed outer ξ/ξ′) and the rpec coil block `gal_result.delta_coil`.

The resistive path uses per-surface inputs `ctrl.gal_eta`/`gal_rho`/`gal_rotation` (length `msing`) +
`ctrl.gal_gamma`. When `ctrl.gal_ideal_flag`, the inner layer is skipped and `cout=0`, so the matched
solution is the bare ideal coil column (Fortran rmatch `coil%ideal_flag`) — the DCON/EL reference.
"""
function gal_match_rpec(ctrl::ForceFreeStatesControl, equil, intr::ForceFreeStatesInternal,
    gal_result::GalerkinResult)

    msing = gal_result.msing
    mpert = intr.numpert_total
    nn = intr.nlow
    mcoil = mpert   # rpec: one coil column per poloidal harmonic

    gal_result.solution !== nothing ||
        error("gal_match_rpec: gal_result.solution is missing (need the gal-reconstructed ξ/ξ′)")
    isempty(gal_result.delta_coil) &&
        error("gal_match_rpec: delta_coil is empty — gal_rpec_flag must be true for RPEC matching")

    if ctrl.gal_ideal_flag
        # Ideal limit (Fortran rmatch coil%ideal_flag, match.f:1665-1671): skip the inner layer entirely
        # and set the resistive plasma combination to zero. The gal outer solve is already fully ideal, so
        # the shared construction below collapses to the bare ideal coil column sols(:,:,csol).
        cout = zeros(ComplexF64, 2msing, mcoil)
        cin = zeros(ComplexF64, 2msing, mcoil)
        deltar = zeros(ComplexF64, msing, 2)
        rpec_eig = zeros(ComplexF64, msing)
        residual = 0.0
    else
        for (name, v) in (("gal_eta", ctrl.gal_eta), ("gal_rho", ctrl.gal_rho), ("gal_rotation", ctrl.gal_rotation))
            length(v) == msing || error("gal_match_rpec: $name has length $(length(v)), expected msing=$msing (one value per surface, core→edge)")
        end

        # Re-derive the gal singular-surface set (must match galerkin_solve's filter) to get the SingType
        # objects resist_eval needs; verify alignment with the stored gal Δ′ surfaces.
        psilow = intr.psilow > 0 ? intr.psilow : equil.profiles.xs[1]
        psihigh = intr.psilim
        sings = [s for s in intr.sing if psilow < s.psifac < psihigh && intr.mlow <= s.m[1] <= intr.mhigh]
        length(sings) == msing && isapprox([s.psifac for s in sings], gal_result.sing_psi; rtol=1e-8) ||
            error("gal_match_rpec: re-derived surface set does not match gal Δ′ surfaces (filter drift?)")

        # --- inner-layer matching data Δ(Q) per surface (deltac_run; match.f:1359-1363) ---
        inner = InnerLayer.GGJModel(solver=:galerkin)
        deltar = zeros(ComplexF64, msing, 2)
        rpec_eig = zeros(ComplexF64, msing)
        for i in 1:msing
            params = resist_eval(sings[i], equil, intr; eta=ctrl.gal_eta[i], rho=ctrl.gal_rho[i],
                gamma=ctrl.gal_gamma, ising=i)
            γ = 2π * im * nn * ctrl.gal_rotation[i]    # rpec forced eigenvalue 2πi·n·f
            rpec_eig[i] = γ
            Δ = InnerLayer.solve_inner(inner, params, γ)   # (Δ₁, Δ₂) in deltac.f convention
            deltar[i, 1] = Δ[1]
            deltar[i, 2] = Δ[2]
        end

        # --- assemble the 4·msing matching system (match.f:1337-1376) ---
        delta_out = gal_result.delta[1:2msing, 1:2msing]   # outer Δ′ plasma block
        mat = zeros(ComplexF64, 4msing, 4msing)
        rmat = zeros(ComplexF64, 4msing, mcoil)
        @views mat[2msing+1:4msing, 1:2msing] .= transpose(delta_out)          # Δ_out
        @views rmat[2msing+1:4msing, :] .= .-transpose(gal_result.delta_coil)  # −Δ_coil source
        for ising in 1:msing
            idx1 = 2ising - 1
            idx2 = 2ising
            idx3 = idx1 + 2msing
            idx4 = idx2 + 2msing
            delta1 = deltar[ising, 1]
            delta2 = deltar[ising, 2]
            mat[idx1, idx1] = 1
            mat[idx2, idx2] = 1
            mat[idx1, idx3] = -1
            mat[idx1, idx4] = 1
            mat[idx2, idx3] = -1
            mat[idx2, idx4] = -1
            mat[idx3, idx3] = -delta1   # Δ_in
            mat[idx3, idx4] = delta2
            mat[idx4, idx3] = -delta1
            mat[idx4, idx4] = -delta2
        end

        # --- solve mat·cof = rmat for the outer/inner coefficients (match.f:1402-1408) ---
        cof = mat \ rmat
        residual = norm(mat * cof - rmat) / max(norm(rmat), 1e-300)
        cout = cof[1:2msing, :]
        cin = cof[2msing+1:4msing, :]
    end

    # --- matched outer ξ/ξ′ per coil drive (match.f:1654-1671); ideal: cout=0 ⇒ bare coil column ---
    sols = gal_result.solution.xi          # (mpert, ngrid, nsol)
    sols_deriv = gal_result.solution.xi_deriv
    ngrid = size(sols, 2)
    xi = zeros(ComplexF64, mpert, ngrid, mcoil)
    xi_deriv = zeros(ComplexF64, mpert, ngrid, mcoil)
    for j in 1:mcoil
        csol = 2msing + j                  # this coil's particular-solution column
        @views xi[:, :, j] .= sols[:, :, csol]
        @views xi_deriv[:, :, j] .= sols_deriv[:, :, csol]
        for isol in 1:2msing
            @views xi[:, :, j] .+= cout[isol, j] .* sols[:, :, isol]
            @views xi_deriv[:, :, j] .+= cout[isol, j] .* sols_deriv[:, :, isol]
        end
    end

    return GalMatchResult(cout, cin, xi, xi_deriv, deltar, rpec_eig, residual)
end

"""
    gal_matched_odestate(gal_result, ffit, intr) -> OdeState

Pack the RPEC-matched outer solution into an `OdeState` shaped exactly like the shooting integrator's,
so `PerturbedEquilibrium` consumes it unchanged. Mirrors Fortran `idcon_build`'s gal branch
(idcon.f:479-493) and `globalsol.bin` (the on-surface `issing` points are dropped, match.f:1468-1497):

  - `u_store[:,:,1,ip] = ξ_ψ`  (matched fundamental matrix, mode×coil-drive, identity-at-edge basis)
  - `ud_store[:,:,1,ip] = dξ_ψ/dψ`  (analytic)
  - `ud_store[:,:,2,ip] = ξ_s = −A⁻¹(B·ξ′ + C·ξ)`  via `ffit` (same outer ideal-MHD relation as `sing_der!`)
  - `u_store[:,:,2] = 0`  (PE never reads it; matches Fortran's unused u2 in the gal path)

The grid is the gal-native grid (inner→edge); `step` indexes the edge so `build_flux_matrix` derives the
edge BC from `u_store[:,:,1,step]`.
"""
function gal_matched_odestate(gal_result::GalerkinResult, ffit::FourFitVars, intr::ForceFreeStatesInternal)
    gal_result.match !== nothing || error("gal_matched_odestate: no match result (run with gal_match_flag=true)")
    sol = gal_result.solution
    m = gal_result.match
    mpert = intr.numpert_total

    # Drop the on-surface (issing) grid points — zero placeholders where the resonant series diverges.
    keep = .!sol.issing
    psi_f = sol.psi[keep]
    q_f = sol.q[keep]
    xi_f = m.xi[:, keep, :]          # (mpert, ngrid_f, mcoil)
    dxi_f = m.xi_deriv[:, keep, :]
    ngrid_f = length(psi_f)

    u_store = zeros(ComplexF64, mpert, mpert, 2, ngrid_f)
    ud_store = zeros(ComplexF64, mpert, mpert, 2, ngrid_f)

    amat = Matrix{ComplexF64}(undef, mpert, mpert)
    bmat = Matrix{ComplexF64}(undef, mpert, mpert)
    cmat = Matrix{ComplexF64}(undef, mpert, mpert)
    tmp = Matrix{ComplexF64}(undef, mpert, mpert)
    hint = Ref(1)
    for ip in 1:ngrid_f
        ξ = @view xi_f[:, ip, :]
        ξ′ = @view dxi_f[:, ip, :]
        @views u_store[:, :, 1, ip] .= ξ
        @views ud_store[:, :, 1, ip] .= ξ′
        # ξ_s = −A⁻¹(B·ξ′ + C·ξ), exactly as the ideal path of sing_der! (Sing.jl:1015-1049)
        ffit.amats(vec(amat), psi_f[ip]; hint=hint)
        ffit.bmats(vec(bmat), psi_f[ip]; hint=hint)
        ffit.cmats(vec(cmat), psi_f[ip]; hint=hint)
        LinearAlgebra.LAPACK.potrf!('U', amat)
        LinearAlgebra.LAPACK.potrs!('U', amat, bmat)   # bmat ← A⁻¹ B
        LinearAlgebra.LAPACK.potrs!('U', amat, cmat)   # cmat ← A⁻¹ C
        xs = @view ud_store[:, :, 2, ip]
        mul!(tmp, bmat, ξ′)
        xs .= .-tmp
        mul!(tmp, cmat, ξ)
        xs .-= tmp
    end

    return OdeState(; numpert_total=mpert, numunorms_init=1, msing=gal_result.msing, numsteps_init=ngrid_f,
        step=ngrid_f, total_steps=ngrid_f, psi_store=psi_f, q_store=q_f, u_store=u_store, ud_store=ud_store)
end
