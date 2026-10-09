# MatchProblem.jl
#
# Driven inner-layer matching on a finished solve (rmatch match_rpec + match_output_solution, match.f).

"""
    MatchProblem(ffs; eta=nothing, rho=nothing, rotation=nothing, gamma=5/3, ideal=false,
                 mu_i=2.0, zeff=1.0, resistivity_model=SauterNeoModel(), lnLambda_form=:nrl)

Driven inner-layer matching on a finished force-free-states solve `ffs`: each rational surface
is matched at the forced eigenvalue γ = 2πi·n·f, with f the plasma rotation frequency there.
`solve(prob, GGJModel())` returns a copy of `ffs` with the matched solution and the penetrated
field `bpen`. `ffs` needs the Δ′ coil block, from `Galerkin(; rpec_flag=true)` or `Riccati()`
with `vac_flag=true`.

# Keywords
- `eta`, `rho`, `rotation`: per-surface resistivity in Ω·m, mass density in kg/m³ and rotation
  frequency in Hz, core to edge. Any left `nothing` is derived from `ffs.equil.kinetic` by
  [`layer_parameters`](@ref) with `mu_i`, `zeff`, `resistivity_model` and `lnLambda_form`.
- `gamma`: ratio of specific heats.
- `ideal`: skip the inner layer and return the perfectly shielding solution.

Fields: `ffs`, the matched `surfaces`, and the resolved `eta`, `rho`, `rotation`, `gamma`, `ideal`.

Ref: Fortran rmatch `match_rpec` (match.f).
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
    resistivity_model::NeoResistivityModel=SauterNeoModel(),
    lnLambda_form::Symbol=:nrl
)
    ffs.npert == 1 || error("MatchProblem: matching is single-n; the result spans n = $(ffs.nlow):$(ffs.nhigh)")
    ffs.closure === :matched && error("MatchProblem: the result is already matched; pose the match on the outer solve")
    dp = ffs.delta_prime
    dp === nothing &&
        error("MatchProblem: the $(ffs.integrator) result has no Δ′; solve with Galerkin(; rpec_flag=true), or Riccati() with vac_flag=true")
    isempty(dp.coil) &&
        error("MatchProblem: the Δ′ coil block is empty; solve with Galerkin(; rpec_flag=true), or Riccati() with vac_flag=true")

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

# The surfaces the Δ′ rows are indexed by (see DeltaPrimeData): all of them for Riccati, the
# Galerkin solve's own subset otherwise.
_matched_surfaces(ffs::ForceFreeStatesResult) =
    ffs.galerkin === nothing ? copy(ffs.surfaces) : [s for s in ffs.surfaces if s.psifac in ffs.galerkin.sing_psi]

"""
    closure_capable(model) -> Bool

Whether `model` can solve a [`MatchProblem`](@ref): `GGJModel` with the `:ray` or `:galerkin`
backend can (they reconstruct the layer profiles); `SLAYERModel` (slab, tearing parity only)
cannot.
"""
closure_capable(::InnerLayer.GGJModel{S}) where {S} = S in (:ray, :galerkin)
closure_capable(::InnerLayer.SLAYERModel) = false

"""
    solve(prob::MatchProblem, model) -> ForceFreeStatesResult

Match `prob.ffs` to the inner layer of `model`. The result has `closure = :matched` (`:ideal`
with `ideal=true`), `bpen`, the [`MatchResult`](@ref) in `result.match` and, for a Galerkin
solve, the matched ξ. A `:matched` result drops the ideal δW (`wp`, `free_boundary`), so
PerturbedEquilibrium skips its response rather than mixing ideal and matched physics.
"""
function CommonSolve.solve(prob::MatchProblem, model::InnerLayer.InnerLayerModel)
    closure_capable(model) ||
        error("$(nameof(typeof(model))) cannot solve a MatchProblem (use GGJModel(); SLAYERModel is for TearingProblem)")
    match = _compute_match(prob, model)
    return _matched_result(prob.ffs, match, prob.ideal)
end

# Inner-layer Δ per surface, the matching system, then the matched outer and inner profiles.
function _compute_match(prob::MatchProblem, model::InnerLayer.GGJModel)
    ffs = prob.ffs
    dp = ffs.delta_prime
    sings = prob.surfaces
    msing = length(sings)
    mcoil = size(dp.coil, 2)
    nn = ffs.nlow
    equil = ffs.equil
    chi1 = 2π * equil.psio

    gal_sol = ffs.galerkin === nothing ? nothing : ffs.galerkin.solution

    if prob.ideal
        # Ideal limit (rmatch coil%ideal_flag): no inner layer, cout = 0.
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
        # Inner-layer Δ and profiles per surface (match.f deltac_run, intotsol).
        deltar = zeros(ComplexF64, msing, 2)
        rpec_eig = zeros(ComplexF64, msing)
        # Layer-center field weight per parity (match.f intotsol_b).
        pen = zeros(ComplexF64, msing, 2)
        inner_psi = Vector{Vector{Float64}}(undef, msing)      # ψ_s ± X·dψdx, ascending
        inner_odd = Vector{Vector{ComplexF64}}(undef, msing)   # Ξ₁, odd about ψ_s
        inner_even = Vector{Vector{ComplexF64}}(undef, msing)  # Ξ₂, even
        inner_bodd = Vector{Vector{ComplexF64}}(undef, msing)  # b^ψ from Ψ₁, even
        inner_beven = Vector{Vector{ComplexF64}}(undef, msing) # b^ψ from Ψ₂, odd
        inner_params = Vector{InnerLayer.GGJParameters}(undef, msing)
        for i in 1:msing
            params = ggj_parameters(sings[i], equil; eta=prob.eta[i], rho=prob.rho[i],
                gamma=prob.gamma, ising=i)
            inner_params[i] = params
            γ = 2π * im * nn * prob.rotation[i]    # forced eigenvalue, f in Hz
            rpec_eig[i] = γ
            inner = InnerLayer.solve_inner_profile(model, params, γ)
            deltar[i, 1] = inner.Δ[1]
            deltar[i, 2] = inner.Δ[2]
            profΨ, profΞ, xg = inner.Ψ, inner.Ξ, inner.x
            resc = inner.rescale                    # inner big-branch amplitude → outer δψ normalization
            # b_m = 2πi·χ₁·(m−nq)·ξ_m with m−nq = −n·q′·dψdx·X and Ψ = XΞ (GWP2016 Eq. 16) ⇒ b_m = scale·resc·Ψ.
            scale = -2π * chi1 * im * nn * sings[i].q1 * inner.dψdx
            pen[i, 1] = scale * profΨ[1, 1] * resc   # X = 0, parity 1
            pen[i, 2] = scale * profΨ[1, 2] * resc   # parity 2 (Ψ(0) = 0)
            xvar = xg .* inner.dψdx                  # X → ψ distance
            inner_psi[i] = vcat(reverse(sings[i].psifac .- xvar), sings[i].psifac .+ xvar)
            inner_odd[i] = resc .* vcat(reverse(.-profΞ[:, 1]), profΞ[:, 1])
            inner_even[i] = resc .* vcat(reverse(profΞ[:, 2]), profΞ[:, 2])
            inner_bodd[i] = (scale * resc) .* vcat(reverse(profΨ[:, 1]), profΨ[:, 1])
            inner_beven[i] = (scale * resc) .* vcat(reverse(.-profΨ[:, 2]), profΨ[:, 2])
        end

        cout, cin, residual = _match_system(dp.raw, dp.coil, deltar)

        # Penetrated field at each layer center (match.f intotsol_b).
        bpen = zeros(ComplexF64, msing, mcoil)
        for i in 1:msing, j in 1:mcoil
            bpen[i, j] = pen[i, 1] * cin[2i, j] + pen[i, 2] * cin[2i-1, j]
        end

        # Matched inner ξ_ψ and b^ψ per coil: odd parity × cin[2i], even × cin[2i-1] (match.f intotsol).
        inner_xi = [inner_odd[i] * transpose(cin[2i, :]) .+ inner_even[i] * transpose(cin[2i-1, :]) for i in 1:msing]
        inner_b = [inner_bodd[i] * transpose(cin[2i, :]) .+ inner_beven[i] * transpose(cin[2i-1, :]) for i in 1:msing]
    end

    reconnected_flux = Matrix{ComplexF64}(dp.coil .+ transpose(dp.raw) * cout)

    # Matched outer ξ per coil: coil column + Σ cout·plasma columns (needs the Galerkin basis).
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
            csol = 2msing + j
            @views xi[:, :, j] .= sols[:, :, csol]
            @views xi_deriv[:, :, j] .= sols_deriv[:, :, csol]
            for isol in 1:2msing
                @views xi[:, :, j] .+= cout[isol, j] .* sols[:, :, isol]
                @views xi_deriv[:, :, j] .+= cout[isol, j] .* sols_deriv[:, :, isol]
            end
        end

        # Composite inner solution: add back the outer background the cut removed (match.f intotsol).
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
                for i in 1:msing
                    m_res = round(Int, nn * sings[i].q)
                    ires = m_res - ffs.mlow + 1
                    # Keep only points inside the cut window.
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

                    # Cut outer background of the resonant harmonic, per coil.
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
                        @views inner_b[i][ip, :] .+= (2π * im * chi1 * singfac) .* buf
                    end
                end
            end
        end
    end

    return MatchResult(cout, cin, deltar, rpec_eig, residual, bpen, reconnected_flux,
        xi, xi_deriv, inner_psi, inner_xi, inner_b, inner_params)
end

# Copy of ffs with the match applied; the ideal reference keeps closure :ideal, its zero bpen and δW.
function _matched_result(ffs::ForceFreeStatesResult, match::MatchResult, ideal::Bool)
    gal = ffs.galerkin
    solution = (gal !== nothing && gal.solution !== nothing && !isempty(match.xi)) ?
               _matched_gal_profiles(gal, match, ffs.mats, ffs) : ffs.solution
    ideal && return _with(ffs; match=match, solution=solution)
    # The ideal δW does not describe the matched plasma; PE gates on its absence.
    return _with(ffs; closure=:matched, bpen=match.bpen, match=match, solution=solution,
        wp=nothing, free_boundary=nothing)
end

# A copy of the immutable `x` with the named fields replaced.
_with(x; fields...) = typeof(x).name.wrapper((get(fields, f, getfield(x, f)) for f in fieldnames(typeof(x)))...)
