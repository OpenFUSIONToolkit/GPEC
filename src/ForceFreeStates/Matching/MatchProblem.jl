# MatchProblem.jl
#
# Inner-layer matching as a POST-SOLVE transformation: `solve(MatchProblem(ffs; ...), GGJModel())`
# consumes a published ForceFreeStatesResult and returns a new one with the closure changed
# from :ideal to :matched and the eigenfunctions replaced — the expensive outer solve is
# reused across arbitrarily many cheap match solves (η/ρ/rotation scans). Port of rmatch
# `match_rpec` + `match_output_solution` (match.f): per surface the forced eigenvalue
# γ_s = 2πi·n·f_s, the inner-layer Δ(Q), then the 4·msing system for the per-coil
# outer/inner coefficients; the matched outer solution for coil drive j is
#   ξ_j = Σ_{isol=1}^{2 msing} cout[isol,j]·sols[:,:,isol] + sols[:,:,2 msing+j].

"""
    MatchProblem(ffs; eta=nothing, rho=nothing, rotation=nothing, gamma=5/3, ideal=false,
                 mu_i=2.0, zeff=1.0, resistivity_model=SpitzerModel(), lnLambda_form=:nrl)

The driven (RPEC) inner-layer matching problem posed on a finished force-free-states solve:
match the outer Δ′ the solve published against an inner-layer response at PRESCRIBED
per-surface eigenvalues γ_s = 2πi·n·f_s. This is the WHAT; the inner-layer model passed to
[`solve`](@ref) — `GGJModel()` today — is the HOW. Solving it returns a NEW
`ForceFreeStatesResult` with `closure = :matched`, `bpen` filled, and (when the producing
formalism retained its outer basis) the ξ solution replaced by the matched profiles, so
layer-parameter scans reuse one outer solve across many cheap match solves.

Construction needs `ffs.delta_prime` with a populated coil block (`Galerkin(; rpec_flag=true)`,
or the Riccati BVP with the vacuum edge coupling) and errors otherwise. The per-surface
plasma parameters come from [`layer_parameters`](@ref): derived from `ffs.equil.kinetic`
with the `mu_i`/`zeff`/`resistivity_model`/`lnLambda_form` knobs, or taken from the
explicit `eta`/`rho`/`rotation` override vectors (one value per matched surface, core to
edge). With `ideal=true` the inner layer is skipped and the matched solution is the bare
ideal coil column (Fortran rmatch `coil%ideal_flag`) — η/ρ/rotation are then unread.

## Fields

  - `ffs::ForceFreeStatesResult` - The outer solve being matched.
  - `surfaces::Vector{SingType}` - The matched surface set (the solve's Δ′ ordering, core to edge).
  - `eta`, `rho`, `rotation::Vector{Float64}` - Resolved per-surface η in Ω·m, ρ in kg/m³ and f in Hz.
  - `gamma::Float64` - Ratio of specific heats Γ in the resistive-layer coefficients.
  - `ideal::Bool` - Skip the inner layer and build the perfectly-shielded reference solution.
"""
struct MatchProblem{R<:ForceFreeStatesResult}
    ffs::R
    surfaces::Vector{SingType}
    eta::Vector{Float64}
    rho::Vector{Float64}
    rotation::Vector{Float64}
    gamma::Float64
    ideal::Bool
end

function MatchProblem(
    ffs::ForceFreeStatesResult;
    eta::Union{Nothing,AbstractVector{<:Real}}=nothing,
    rho::Union{Nothing,AbstractVector{<:Real}}=nothing,
    rotation::Union{Nothing,AbstractVector{<:Real}}=nothing,
    gamma::Real=5 / 3,
    ideal::Bool=false,
    mu_i::Real=2.0,
    zeff::Real=1.0,
    resistivity_model::NeoResistivityModel=SpitzerModel(),
    lnLambda_form::Symbol=:nrl
)
    dp = ffs.delta_prime
    dp === nothing &&
        error("MatchProblem: the $(ffs.integrator) result carries no Δ′ payload — inner-layer matching needs a Riccati or Galerkin solve")
    isempty(dp.coil) &&
        error("MatchProblem: the Δ′ coil-response block is empty — solve with Galerkin(; rpec_flag=true) (or the Riccati vacuum edge coupling) first")

    sings = _matched_surfaces(ffs)
    size(dp.raw, 1) == 2 * length(sings) ||
        error("MatchProblem: Δ′ raw block is $(size(dp.raw, 1))×$(size(dp.raw, 2)) but the result carries $(length(sings)) matchable surfaces")

    if ideal
        msing = length(sings)
        return MatchProblem(ffs, sings, zeros(msing), zeros(msing), zeros(msing), Float64(gamma), true)
    end
    params = layer_parameters(sings, ffs.equil; eta=eta, rho=rho, rotation=rotation,
        mu_i=mu_i, zeff=zeff, resistivity_model=resistivity_model, lnLambda_form=lnLambda_form)
    return MatchProblem(ffs, sings, params.eta, params.rho, params.rotation, Float64(gamma), false)
end

# The matched surface set: the solve's rational surfaces restricted to the integration
# domain and the resolved m-band — the same filter `gal_resonant_surfaces` applies at
# solve time, re-derived off the published result so the Δ′ row ordering is reproduced.
function _matched_surfaces(ffs::ForceFreeStatesResult)
    psilow = ffs.psilow > 0 ? ffs.psilow : ffs.equil.profiles.xs[1]
    return [s for s in ffs.surfaces if psilow < s.psifac < ffs.psilim && ffs.mlow <= s.m[1] <= ffs.mhigh]
end

"""
    closure_capable(model) -> Bool

Whether an inner-layer model can CLOSE a matched outer solution: that takes both parity
channels of the matching data and the reconstructed layer field profiles. `GGJModel` can;
`SLAYERModel` cannot (slab: single parity, no interchange channel, no layer profiles) — it
is restricted to the free-eigenvalue tearing solve.
"""
closure_capable(::InnerLayer.GGJModel) = true
closure_capable(::InnerLayer.SLAYERModel) = false

"""
    solve(prob::MatchProblem, model) -> ForceFreeStatesResult

Solve the driven inner-layer matching with the given [`InnerLayer.InnerLayerModel`](@ref)
and return a NEW result: `closure = :matched` (`:ideal` for the perfectly-shielded
reference), `bpen` filled from the inner solutions at the layer centers, the matched
`MatchResult` attached to `result.galerkin.match`, and the ξ solution replaced by the
matched identity-at-edge profiles when the producing formalism retained its outer basis
(a Riccati-fed match keeps `solution === nothing`; the capability gates handle every
consumer). Only a [`closure_capable`](@ref) model is accepted here.
"""
function CommonSolve.solve(prob::MatchProblem, model::InnerLayer.InnerLayerModel)
    closure_capable(model) ||
        error("a $(nameof(typeof(model))) inner-layer model cannot close a matched solution " *
              "(slab: single parity, no interchange channel, no reconstructable layer profiles); " *
              "use GGJModel() here — SLAYERModel drives the free-eigenvalue tearing solve instead")
    match = _compute_match(prob, model)
    return _matched_result(prob.ffs, match, prob.ideal)
end

# The unified match computation (port of the former gal_match_rpec, parameterized by the
# problem and model instead of the control struct). The matching system and resonant
# products are basis-free; the outer-profile recombination and the composite inner-region
# graft run only when the producing solve retained its outer basis.
function _compute_match(prob::MatchProblem, model::InnerLayer.GGJModel)
    ffs = prob.ffs
    dp = ffs.delta_prime
    sings = prob.surfaces
    msing = length(sings)
    mcoil = size(dp.coil, 2)
    nn = ffs.nlow
    equil = ffs.equil

    gal_sol = ffs.galerkin === nothing ? nothing : ffs.galerkin.solution

    if prob.ideal
        # Ideal limit (Fortran rmatch coil%ideal_flag, match.f): skip the inner layer entirely
        # and set the resistive plasma combination to zero, so the shared construction below
        # collapses to the bare ideal coil column sols(:,:,csol).
        cout = zeros(ComplexF64, 2msing, mcoil)
        cin = zeros(ComplexF64, 2msing, mcoil)
        deltar = zeros(ComplexF64, msing, 2)
        bpen = zeros(ComplexF64, msing, mcoil)
        inner_psi = Vector{Float64}[]
        inner_xi = Matrix{ComplexF64}[]
        inner_b = Matrix{ComplexF64}[]
        inner_params = InnerLayer.GGJParameters[]
        rpec_eig = zeros(ComplexF64, msing)
        residual = 0.0
    else
        # --- inner-layer matching data Δ(Q) per surface (deltac_run; match.f) ---
        # solve_inner_profile returns the same Δ as solve_inner plus the reconstructed
        # inner-layer field, so the layer-center value (penetrated field) comes for free.
        deltar = zeros(ComplexF64, msing, 2)
        rpec_eig = zeros(ComplexF64, msing)
        # Per-surface layer-center field weights pen[i,k] = scale·Ψ_k(0), parity k=1,2 (match.f intotsol_b).
        chi1 = 2π * equil.psio
        pen = zeros(ComplexF64, msing, 2)
        # Per-surface inner-layer ξ_ψ building blocks for the matched solution (match.f intotsol, deltac
        # comp 2): the ψ grid ψ_s ± X·x0/v1 (left reversed then right) and the resc-scaled parity profiles.
        inner_psi = Vector{Vector{Float64}}(undef, msing)
        inner_odd = Vector{Vector{ComplexF64}}(undef, msing)   # Ξ₁, antisymmetric across ψ_s
        inner_even = Vector{Vector{ComplexF64}}(undef, msing)  # Ξ₂, symmetric across ψ_s
        inner_bodd = Vector{Vector{ComplexF64}}(undef, msing)  # b^ψ₁ = scale·resc·Ψ₁, symmetric (Ψ′₁(0)=0)
        inner_beven = Vector{Vector{ComplexF64}}(undef, msing) # b^ψ₂ = scale·resc·Ψ₂, antisymmetric (Ψ₂(0)=0)
        inner_params = Vector{InnerLayer.GGJParameters}(undef, msing)
        for i in 1:msing
            params = ggj_parameters(sings[i], equil; eta=prob.eta[i], rho=prob.rho[i],
                gamma=prob.gamma, ising=i)
            inner_params[i] = params
            γ = 2π * im * nn * prob.rotation[i]    # forced eigenvalue; rotation is f in Hz, γ = 2πi·n·f
            rpec_eig[i] = γ
            inner = InnerLayer.solve_inner_profile(model, params, γ)
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

        cout, cin, residual = _match_system(dp.raw, dp.coil, deltar)

        # Inner-layer penetrated (reconnected) resonant field at each rational surface, read off the
        # inner solution at the layer center exactly as Fortran match_output_solution builds intotsol_b
        # (match.f) — cusp-free, fit-free. bpen[i,j] = pen₁(i)·cin[2i,j] + pen₂(i)·cin[2i-1,j].
        bpen = zeros(ComplexF64, msing, mcoil)
        for i in 1:msing, j in 1:mcoil
            bpen[i, j] = pen[i, 1] * cin[2i, j] + pen[i, 2] * cin[2i-1, j]
        end

        # Matched inner-layer ξ_ψ(ψ) per surface, per coil drive (match.f intotsol): odd parity weighted
        # by cin[2i] (cofin(2·ising)), even parity by cin[2i-1] (cofin(2·ising-1)).
        inner_xi = [inner_odd[i] * transpose(cin[2i, :]) .+ inner_even[i] * transpose(cin[2i-1, :]) for i in 1:msing]
        # Matched inner-layer b^ψ(ψ) per surface, per coil drive.
        inner_b = [inner_bodd[i] * transpose(cin[2i, :]) .+ inner_beven[i] * transpose(cin[2i-1, :]) for i in 1:msing]
    end

    reconnected_flux = Matrix{ComplexF64}(dp.coil .+ transpose(dp.raw) * cout)

    # --- matched outer ξ/ξ′ per coil drive (match.f); ideal: cout=0 ⇒ bare coil column ---
    # Only when the producing solve retained its outer basis; the matching system above is basis-free.
    if gal_sol === nothing
        xi = Array{ComplexF64,3}(undef, 0, 0, 0)
        xi_deriv = Array{ComplexF64,3}(undef, 0, 0, 0)
        empty!(inner_psi); empty!(inner_xi); empty!(inner_b)
    else
        sols = gal_sol.xi          # (mpert, ngrid, nsol)
        sols_deriv = gal_sol.xi_deriv
        mpert = size(sols, 1)
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
            sols_cut = gal_sol.xi_cut
            if isempty(sols_cut)
                @warn "MatchProblem: the solve retained no cut solution (set cut_solution=true on Galerkin), " *
                      "so the composite inner-region profiles are skipped; bpen and the matched outer ξ are unaffected"
                empty!(inner_psi); empty!(inner_xi); empty!(inner_b)
            else
                cut_range = gal_sol.cut_range
                keep = .!gal_sol.issing
                psi_keep = gal_sol.psi[keep]
                chi1_c = 2π * equil.psio
                for i in 1:msing
                    m_res = round(Int, nn * sings[i].q)
                    ires = m_res - ffs.mlow + 1
                    # Clip the layer to the window where the cut is active; outside it the cut removes
                    # nothing and the composite is undefined (Fortran writes no points there).
                    lo, hi = cut_range[i, 1], cut_range[i, 2]
                    inside = findall(p -> lo <= p <= hi, inner_psi[i])
                    if isempty(inside)
                        @warn "MatchProblem: inner layer at ψ=$(round(sings[i].psifac, digits=5)) lies outside the " *
                              "resonant/extension cells (ψ ∈ [$lo, $hi]); raise gal_dx1/gal_dx2 to overlap the layer" surface = i
                        continue
                    end
                    length(inside) < length(inner_psi[i]) && @info "MatchProblem: surface $i inner region clipped to the " *
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
        end
    end

    return MatchResult(cout, cin, deltar, rpec_eig, residual, bpen, reconnected_flux,
        xi, xi_deriv, inner_psi, inner_xi, inner_b, inner_params)
end

# Rebuild the published result around the match: same solve products, new closure. The ideal
# reference keeps the :ideal closure and its all-zero bpen (preserving the solve-time row
# convention); a resistive match carries the matched-surface-set bpen. The matched profiles
# replace the solution whenever the outer basis allowed their construction.
function _matched_result(ffs::ForceFreeStatesResult, match::MatchResult, ideal::Bool)
    gal = ffs.galerkin
    new_gal = gal === nothing ? nothing :
              GalerkinResult(gal.msing, gal.sing_psi, gal.sing_q, gal.sing_m, gal.sing_n,
        gal.di, gal.alpha, gal.solution, match)
    solution = (new_gal !== nothing && new_gal.solution !== nothing && !isempty(match.xi)) ?
               _matched_gal_profiles(new_gal, ffs.mats, ffs) : ffs.solution
    closure = ideal ? :ideal : :matched
    bpen = ideal ? ffs.bpen : match.bpen

    return ForceFreeStatesResult(
        ffs.integrator, ffs.control, ffs.equil,
        ffs.mlow, ffs.mhigh, ffs.mpert, ffs.nlow, ffs.nhigh, ffs.npert, ffs.numpert_total,
        ffs.psilow, ffs.psilim, ffs.qlim, ffs.q1lim, ffs.dir_path, ffs.wall_settings, ffs.debug_settings,
        ffs.metric, ffs.mats, ffs.surfaces, ffs.kinetic,
        closure, bpen,
        solution, ffs.diagnostics, ffs.wp, ffs.free_boundary, ffs.delta_prime, new_gal
    )
end
