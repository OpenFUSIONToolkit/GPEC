# Outer<->inner resistive match, Wang et al. 2020 (PoP 27, 122509) Eq. 11:
#   C = -(Δ_out - Δ_in(i2πf))^{-1} Δ_coil
# Raw STRIDE outer Δ' + raw coil drive matched to the GGJ inner layer (resist_eval -> solve_inner).
struct ResonantMatchResult
    cout::Matrix{ComplexF64}               # outer coeffs (2msing × ncoil)
    cin::Matrix{ComplexF64}                # inner coeffs (2msing × ncoil)
    deltar::Matrix{ComplexF64}             # inner-layer Δ per surface (msing × 2)
    rpec_eig::Vector{ComplexF64}           # forced eigenvalue γ_s = 2πi·n·f
    reconnected_flux::Matrix{ComplexF64}   # reconnected resonant flux (2msing × ncoil)
    bpen::Matrix{ComplexF64}               # area-weighted penetrated field (msing × ncoil); empty until Stage 2
    residual::Float64
end

function resonant_match_rpec(delta_out_raw::AbstractMatrix, delta_coil_raw::AbstractMatrix,
    sings::Vector{SingType}, equil::Equilibrium.PlasmaEquilibrium,
    intr::ForceFreeStatesInternal, ctrl::ForceFreeStatesControl)

    msing = size(delta_out_raw, 1) ÷ 2
    ncoil = size(delta_coil_raw, 2)
    nn = intr.nlow
    empty_bpen = Matrix{ComplexF64}(undef, 0, 0)

    size(delta_out_raw) == (2msing, 2msing) || error("delta_out_raw $(size(delta_out_raw)) != (2msing,2msing)")
    length(sings) == msing || error("sings $(length(sings)) != msing $msing")
    size(delta_coil_raw, 1) == 2msing || error("delta_coil_raw rows $(size(delta_coil_raw,1)) != 2msing")

    if ctrl.gal_ideal_flag   # ideal limit: no inner layer, no reconnection
        return ResonantMatchResult(zeros(ComplexF64,2msing,ncoil), zeros(ComplexF64,2msing,ncoil),
            zeros(ComplexF64,msing,2), zeros(ComplexF64,msing), Matrix{ComplexF64}(delta_coil_raw), empty_bpen, 0.0)
    end
    for (nm,v) in (("gal_eta",ctrl.gal_eta),("gal_rho",ctrl.gal_rho),("gal_rotation",ctrl.gal_rotation))
        length(v) == msing || error("$nm length $(length(v)) != msing $msing")
    end

    chi1 = 2π * equil.psio
    deltar = zeros(ComplexF64, msing, 2)
    rpec_eig = zeros(ComplexF64, msing)
    # Layer-center (X=0) penetrated-field weights pen[i,k] = scale·Ψ_k(0)·rescale (match.f intotsol_b);
    # solve_inner_profile returns the same Δ as solve_inner plus the inner-layer field needed for pen.
    pen = zeros(ComplexF64, msing, 2)
    for i in 1:msing
        params = resist_eval(sings[i], equil, intr; eta=ctrl.gal_eta[i], rho=ctrl.gal_rho[i], gamma=ctrl.gal_gamma, ising=i)
        γ = 2π*im*nn*ctrl.gal_rotation[i]
        rpec_eig[i] = γ
        inner = InnerLayer.solve_inner_profile(InnerLayer.GGJModel(; solver=:galerkin), params, γ;
            xfac=ctrl.gal_inner_xfac, nx=ctrl.gal_inner_nx, nq=ctrl.gal_inner_nq, cutoff=ctrl.gal_inner_cutoff, kmax=ctrl.gal_inner_kmax)
        deltar[i,1] = inner.Δ[1]; deltar[i,2] = inner.Δ[2]
        scale = -2π * chi1 * im * nn * sings[i].q1 * inner.dψdx   # b_m = −2πi·χ₁·n·q′·dψdx·rescale·Ψ (GalerkinMatch.jl)
        pen[i,1] = scale * inner.Ψ[1,1] * inner.rescale          # parity 1 (Ψ(0)≠0)
        pen[i,2] = scale * inner.Ψ[1,2] * inner.rescale          # parity 2 (Ψ(0)=0 ⇒ ~0)
    end

    mat  = zeros(ComplexF64, 4msing, 4msing)
    rmat = zeros(ComplexF64, 4msing, ncoil)
    @views mat[2msing+1:4msing, 1:2msing] .= transpose(delta_out_raw)
    @views rmat[2msing+1:4msing, :] .= .-delta_coil_raw
    for i in 1:msing
        a=2i-1; b=2i; c=a+2msing; d=b+2msing
        d1=deltar[i,1]; d2=deltar[i,2]
        mat[a,a]=1;  mat[b,b]=1
        mat[a,c]=-1; mat[a,d]=1
        mat[b,c]=-1; mat[b,d]=-1
        mat[c,c]=-d1; mat[c,d]=d2
        mat[d,c]=-d1; mat[d,d]=-d2
    end

    cof = mat \ rmat
    residual = norm(mat*cof - rmat) / max(norm(rmat), 1e-300)
    cout = cof[1:2msing, :]
    cin  = cof[2msing+1:4msing, :]
    
    reconnected_flux = delta_coil_raw .+ transpose(delta_out_raw)*cout
    # Inner-layer penetrated (reconnected) resonant field per surface — ONE quantity per surface, read off
    # the inner solution at the layer center (match.f intotsol_b; GalerkinMatch.jl): bpen[i,j] = pen₁(i)·cin[2i,j] + pen₂(i)·cin[2i-1,j].
    bpen = zeros(ComplexF64, msing, ncoil)
    for i in 1:msing, j in 1:ncoil
        bpen[i,j] = pen[i,1]*cin[2i,j] + pen[i,2]*cin[2i-1,j]
    end
    return ResonantMatchResult(cout, cin, deltar, rpec_eig, reconnected_flux, bpen, residual)
end
