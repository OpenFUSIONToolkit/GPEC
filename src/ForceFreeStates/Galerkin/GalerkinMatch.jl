# GalerkinMatch.jl
#
# DRIVEN (RPEC) outer↔inner asymptotic matching. Port of rmatch `match_rpec` (match.f) and the
# outer total-solution construction of `match_output_solution` (match.f).
#
# Per rational surface: the forced eigenvalue γ_s = 2πi·n·f_s (gal_rotation is the rotation f in Hz, so γ
# is the layer Doppler frequency), the inner-layer matching data Δ(Q) via the Galerkin GGJ solver
# (InnerLayer.solve_inner), then the 4·msing matching
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
        # Ideal limit (Fortran rmatch coil%ideal_flag, match.f): skip the inner layer entirely
        # and set the resistive plasma combination to zero. The gal outer solve is already fully ideal, so
        # the shared construction below collapses to the bare ideal coil column sols(:,:,csol).
        cout = zeros(ComplexF64, 2msing, mcoil)
        cin = zeros(ComplexF64, 2msing, mcoil)
        deltar = zeros(ComplexF64, msing, 2)
        bpen = zeros(ComplexF64, msing, mcoil)
        inner_psi = Vector{Float64}[]   # no inner layer in the ideal limit
        inner_xi = Matrix{ComplexF64}[]
        inner_b = Matrix{ComplexF64}[]
        inner_params = InnerLayer.GGJParameters[]   # inner layer skipped in the ideal limit
        rpec_eig = zeros(ComplexF64, msing)
        residual = 0.0
    else
        for (name, v) in (("gal_eta", ctrl.gal_eta), ("gal_rho", ctrl.gal_rho), ("gal_rotation", ctrl.gal_rotation))
            length(v) == msing || error("gal_match_rpec: $name has length $(length(v)), expected msing=$msing (one value per surface, core→edge)")
        end

        # Re-derive the gal singular-surface set (same helper as galerkin_solve) for the SingType objects
        # resist_eval needs.
        sings, _, _ = gal_resonant_surfaces(intr, equil)
        length(sings) == msing ||
            error("gal_match_rpec: re-derived $(length(sings)) surfaces, expected msing=$msing")
        ctrl.gal_inner_solver in ("ray", "galerkin") ||
            error("gal_match_rpec: gal_inner_solver = \"$(ctrl.gal_inner_solver)\" (expected \"ray\" or \"galerkin\")")

        # --- inner-layer matching data Δ(Q) per surface (deltac_run; match.f) ---
        # solve_inner_profile returns the same Δ as solve_inner plus the reconstructed inner-layer field,
        # so the layer-center value (penetrated field) comes for free from the single matching solve.
        deltar = zeros(ComplexF64, msing, 2)
        rpec_eig = zeros(ComplexF64, msing)
        # Per-surface layer-center field weights pen[i,k] = scale·Ψ_k(0), parity k=1,2 (match.f intotsol_b).
        chi1 = 2π * equil.psio
        pen = zeros(ComplexF64, msing, 2)
        # Per-surface inner-layer ξ_ψ building blocks for the matched solution (match.f intotsol, deltac comp 2):
        # the ψ grid ψ_s ± X·x0/v1 (left reversed then right) and the resc-scaled odd/even parity profiles.
        inner_psi = Vector{Vector{Float64}}(undef, msing)
        inner_odd = Vector{Vector{ComplexF64}}(undef, msing)   # Ξ₁, antisymmetric across ψ_s
        inner_even = Vector{Vector{ComplexF64}}(undef, msing)  # Ξ₂, symmetric across ψ_s
        inner_bodd = Vector{Vector{ComplexF64}}(undef, msing)  # b^ψ₁ = scale·resc·Ψ₁, symmetric (Ψ′₁(0)=0)
        inner_beven = Vector{Vector{ComplexF64}}(undef, msing) # b^ψ₂ = scale·resc·Ψ₂, antisymmetric (Ψ₂(0)=0)
        inner_params = Vector{InnerLayer.GGJParameters}(undef, msing)
        for i in 1:msing
            params = resist_eval(sings[i], equil, intr; eta=ctrl.gal_eta[i], rho=ctrl.gal_rho[i],
                gamma=ctrl.gal_gamma, ising=i)
            inner_params[i] = params
            γ = 2π * im * nn * ctrl.gal_rotation[i]    # forced eigenvalue; gal_rotation is f [Hz], γ = 2πi·n·f
            rpec_eig[i] = γ
            # gal_inner_solver = "ray" (default) uses the rotated-ray backend (certified optimal-θ Δ
            # + θ=0 real-axis profile re-solve with runtime certificate — see the InnerLayer
            # solve_inner_profile(::GGJModel{:ray}, ...) docstring); "galerkin" uses the Hermite-FEM
            # real-axis solve with the ctrl.gal_inner_* knobs (defaults match the Fortran rmatch
            # deltac/inps reference).
            inner = if ctrl.gal_inner_solver == "ray"
                InnerLayer.solve_inner_profile(InnerLayer.GGJModel(; solver=:ray), params, γ)
            else
                InnerLayer.solve_inner_profile(InnerLayer.GGJModel(; solver=:galerkin), params, γ;
                    xfac=ctrl.gal_inner_xfac, nx=ctrl.gal_inner_nx, nq=ctrl.gal_inner_nq, cutoff=ctrl.gal_inner_cutoff, kmax=ctrl.gal_inner_kmax)
            end
            deltar[i, 1] = inner.Δ[1]
            deltar[i, 2] = inner.Δ[2]
            profΨ, profΞ, xg = inner.Ψ, inner.Ξ, inner.x
            # Amplitude rescale: inner profiles are normalized in X = v1·δψ/x0 (inner_psi below);
            # inner.rescale converts a big-branch amplitude to the outer δψ-normalization.
            resc = inner.rescale
            # b-field scaling, derived from the code's own outer convention (SingularCoupling):
            #   b_m = 2πi·χ₁·(m−nq)·ξ_m,  m−nq = −n·q′·δψ,  δψ = dψdx·X,  resc·Ξ(X) ↔ ξ_m,
            # and the far-field identity Ψ = X·Ξ (GWP2016 Eq. 16; Ψ is the normal-field variable, Eq. A17):
            #   b_m = −2πi·χ₁·n·q′·dψdx·resc·Ψ.
            scale = -2π * chi1 * im * nn * sings[i].q1 * inner.dψdx
            pen[i, 1] = scale * profΨ[1, 1] * resc                      # layer center X=0, parity 1 (Ψ(0)≠0)
            pen[i, 2] = scale * profΨ[1, 2] * resc                      # parity 2 (Ψ(0)=0 ⇒ ~0)
            xvar = xg .* inner.dψdx                                     # inner X → ψ-distance (deltac.f:1822)
            inner_psi[i] = vcat(reverse(sings[i].psifac .- xvar), sings[i].psifac .+ xvar)
            inner_odd[i] = resc .* vcat(reverse(.-profΞ[:, 1]), profΞ[:, 1])   # comp 2, parity 1 (odd: −left,+right)
            inner_even[i] = resc .* vcat(reverse(profΞ[:, 2]), profΞ[:, 2])    # comp 2, parity 2 (even)
            # b^ψ profiles on the same two-sided grid: Ψ is the normal-field variable (GWP2016 A17),
            # b_m(δψ) = scale·resc·Ψ(X) throughout the layer (→ outer frozen-in relation via Ψ = XΞ).
            inner_bodd[i] = (scale * resc) .* vcat(reverse(profΨ[:, 1]), profΨ[:, 1])     # parity 1: Ψ even
            inner_beven[i] = (scale * resc) .* vcat(reverse(.-profΨ[:, 2]), profΨ[:, 2])  # parity 2: Ψ odd
        end

        # --- assemble the 4·msing matching system (match.f) ---
        delta_out = gal_result.delta[1:2msing, 1:2msing]   # outer Δ′ plasma block
        mat = zeros(ComplexF64, 4msing, 4msing)
        rmat = zeros(ComplexF64, 4msing, mcoil)
        @views mat[(2msing+1):4msing, 1:2msing] .= transpose(delta_out)          # Δ_out
        @views rmat[(2msing+1):4msing, :] .= .-transpose(gal_result.delta_coil)  # −Δ_coil source
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
            # inner-layer Δ block signs per match.f match_rpec
            mat[idx3, idx3] = -delta1
            mat[idx3, idx4] = delta2
            mat[idx4, idx3] = -delta1
            mat[idx4, idx4] = -delta2
        end

        # --- solve mat·cof = rmat for the outer/inner coefficients (match.f) ---
        cof = mat \ rmat
        residual = norm(mat * cof - rmat) / max(norm(rmat), 1e-300)
        cout = cof[1:2msing, :]
        cin = cof[(2msing+1):4msing, :]

        # Inner-layer penetrated (reconnected) resonant field at each rational surface, read off the GGJ
        # inner solution at the layer center exactly as Fortran match_output_solution builds intotsol_b
        # (match.f) — cusp-free, fit-free. bpen[i,j] = pen₁(i)·cin[2i,j] + pen₂(i)·cin[2i-1,j].
        bpen = zeros(ComplexF64, msing, mcoil)
        for i in 1:msing, j in 1:mcoil
            bpen[i, j] = pen[i, 1] * cin[2i, j] + pen[i, 2] * cin[2i-1, j]
        end

        # Matched inner-layer ξ_ψ(ψ) per surface, per coil drive (match.f intotsol): odd parity weighted by
        # cin[2i] (cofin(2·ising)), even parity by cin[2i-1] (cofin(2·ising-1)).
        inner_xi = [inner_odd[i] * transpose(cin[2i, :]) .+ inner_even[i] * transpose(cin[2i-1, :]) for i in 1:msing]
        # Matched inner-layer b^ψ(ψ) per surface, per coil drive.
        inner_b = [inner_bodd[i] * transpose(cin[2i, :]) .+ inner_beven[i] * transpose(cin[2i-1, :]) for i in 1:msing]
    end

    # --- matched outer ξ/ξ′ per coil drive (match.f); ideal: cout=0 ⇒ bare coil column ---
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

    # --- composite inner-region solution: cut outer background + layer (match.f intotsol/intotsol_b) ---
    # The layer solution alone carries only the resonant content it resolves; the smooth background
    # removed by the cut has to be added back for the inner and outer solutions to overlap in the
    # matching region. Without this the inner profile does not graft onto the outer eigenfunction.
    if !isempty(inner_psi)
        sols_cut = gal_result.solution.xi_cut
        isempty(sols_cut) && error("gal_match_rpec: solution.xi_cut is empty — the cut solution is " *
                                   "required to build the composite inner-region solution")
        cut_range = gal_result.solution.cut_range
        keep = .!gal_result.solution.issing
        psi_keep = gal_result.solution.psi[keep]
        chi1_c = 2π * equil.psio
        for i in 1:msing
            m_res = round(Int, nn * sings[i].q)
            ires = m_res - intr.mlow + 1
            # Clip the layer to the window where the cut is active; outside it the cut removes
            # nothing and the composite is undefined (Fortran writes no points there).
            lo, hi = cut_range[i, 1], cut_range[i, 2]
            inside = findall(p -> lo <= p <= hi, inner_psi[i])
            if isempty(inside)
                @warn "gal_match_rpec: inner layer at ψ=$(round(sings[i].psifac, digits=5)) lies outside the " *
                      "resonant/extension cells (ψ ∈ [$lo, $hi]); raise gal_dx1/gal_dx2 to overlap the layer" surface = i
                continue
            end
            length(inside) < length(inner_psi[i]) && @info "gal_match_rpec: surface $i inner region clipped to the " *
                  "cut window ($(length(inside)) of $(length(inner_psi[i])) points)"
            inner_psi[i] = inner_psi[i][inside]
            inner_xi[i] = inner_xi[i][inside, :]
            inner_b[i] = inner_b[i][inside, :]

            # cut outer background for this surface's resonant harmonic, per coil drive
            cutmn = Matrix{ComplexF64}(undef, length(psi_keep), mcoil)
            for j in 1:mcoil
                @views cutmn[:, j] .= sols_cut[ires, keep, 2msing+j]
                for isol in 1:2msing
                    @views cutmn[:, j] .+= cout[isol, j] .* sols_cut[ires, keep, isol]
                end
            end
            itp = cubic_interp(psi_keep, Series(cutmn); extrap=ExtendExtrap())
            buf = Vector{ComplexF64}(undef, mcoil)
            hint = Ref(1)
            for (ip, psi_p) in enumerate(inner_psi[i])
                itp(buf, psi_p; hint=hint)
                singfac = m_res - nn * equil.profiles.q_spline(psi_p)
                @views inner_xi[i][ip, :] .+= buf
                @views inner_b[i][ip, :] .+= (2π * im * chi1_c * singfac) .* buf
            end
        end
    end

    return GalMatchResult(cout, cin, xi, xi_deriv, deltar, bpen, inner_psi, inner_xi, inner_b, inner_params, rpec_eig, residual)
end

"""
    gal_matched_odestate(gal_result, ffit, intr) -> OdeState

Pack the RPEC-matched outer solution into an `OdeState` shaped exactly like the shooting integrator's,
so `PerturbedEquilibrium` consumes it unchanged. Mirrors Fortran `idcon_build`'s gal branch
(idcon.f) and `globalsol.bin` (the on-surface `issing` points are dropped, match.f):

  - `u_store[:,:,1,ip] = ξ_ψ`  (matched fundamental matrix, mode×coil-drive, identity-at-edge basis)
  - `du_store[:,:,1,ip] = dξ_ψ/dψ`  (analytic)
  - `xi_s_store[:,:,ip] = ξ_s = −A⁻¹(B·ξ′ + C·ξ)`  via `ffit` (same outer ideal-MHD relation as `sing_der!`)
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
    du_store = zeros(ComplexF64, mpert, mpert, 2, ngrid_f)
    xi_s_store = zeros(ComplexF64, mpert, mpert, ngrid_f)

    amat = Matrix{ComplexF64}(undef, mpert, mpert)
    bmat = Matrix{ComplexF64}(undef, mpert, mpert)
    cmat = Matrix{ComplexF64}(undef, mpert, mpert)
    tmp = Matrix{ComplexF64}(undef, mpert, mpert)
    hint = Ref(1)
    for ip in 1:ngrid_f
        ξ = @view xi_f[:, ip, :]
        ξ′ = @view dxi_f[:, ip, :]
        @views u_store[:, :, 1, ip] .= ξ
        @views du_store[:, :, 1, ip] .= ξ′
        # ξ_s = −A⁻¹(B·ξ′ + C·ξ), exactly as the ideal path of sing_der! (Sing.jl:1015-1049)
        ffit.amats(vec(amat), psi_f[ip]; hint=hint)
        ffit.bmats(vec(bmat), psi_f[ip]; hint=hint)
        ffit.cmats(vec(cmat), psi_f[ip]; hint=hint)
        LinearAlgebra.LAPACK.potrf!('U', amat)
        LinearAlgebra.LAPACK.potrs!('U', amat, bmat)   # bmat ← A⁻¹ B
        LinearAlgebra.LAPACK.potrs!('U', amat, cmat)   # cmat ← A⁻¹ C
        xs = @view xi_s_store[:, :, ip]
        mul!(tmp, bmat, ξ′)
        xs .= .-tmp
        mul!(tmp, cmat, ξ)
        xs .-= tmp
    end

    return OdeState(; numpert_total=mpert, numunorms_init=1, msing=gal_result.msing, numsteps_init=ngrid_f,
        step=ngrid_f, total_steps=ngrid_f, psi_store=psi_f, q_store=q_f, u_store=u_store, du_store=du_store, xi_s_store=xi_s_store)
end
