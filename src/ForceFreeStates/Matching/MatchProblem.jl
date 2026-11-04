# MatchProblem.jl
#
# Driven inner-layer matching on a finished solve (rmatch match_rpec + match_output_solution, match.f).

"""
    MatchProblem(ffs; eta=nothing, rho=nothing, rotation=nothing, gamma=5/3)

Driven inner-layer matching on a finished force-free-states solve `ffs`: each rational surface
is matched at the forced eigenvalue γ = 2πi·n·f, with f the plasma rotation frequency there.
`solve(prob, GGJModel())` returns a copy of `ffs` with the matched solution and the penetrated
field `bpen`. `ffs` needs `Delta_EF`, from `Galerkin()` or `Riccati()` with `error_field_response` (Riccati
also needs `vac_flag`).

# Keywords

  - `eta`, `rho`, `rotation`: resistivity in Ω·m, mass density in kg/m³ and rotation frequency
    in Hz, a scalar for every surface or one value per `ffs.surfaces` entry (core to edge).
    Each one not given comes from the equilibrium's kinetic data, as the tearing problem's do:
    η from [`InnerLayer.layer_eta`](@ref), ρ from [`InnerLayer.layer_rho`](@ref), and the
    rotation f = Ω_E/2π from the E×B rotation of [`InnerLayer.layer_omega_E`](@ref).
  - `gamma`: ratio of specific heats.

The perfectly shielding (ideal) solution needs no match: it is the Galerkin solve's own result.

## Fields

  - `ffs` - The outer solve being matched; held by reference, not copied.
  - `eta`, `rho`, `rotation` - Per-surface η in Ω·m, ρ in kg/m³ and rotation frequency in Hz, following `ffs.surfaces`.
  - `gamma` - Ratio of specific heats.

Ref: Fortran rmatch `match_rpec` (match.f).
"""
struct MatchProblem{R<:ForceFreeStatesResult}
    ffs::R
    eta::Vector{Float64}
    rho::Vector{Float64}
    rotation::Vector{Float64}
    gamma::Float64
end

function MatchProblem(
    ffs::ForceFreeStatesResult;
    eta::Union{Nothing,Real,AbstractVector{<:Real}}=nothing,
    rho::Union{Nothing,Real,AbstractVector{<:Real}}=nothing,
    rotation::Union{Nothing,Real,AbstractVector{<:Real}}=nothing,
    gamma::Real=5 / 3
)
    ffs.npert == 1 || error("MatchProblem: matching is single-n; the result spans n = $(ffs.nlow):$(ffs.nhigh)")
    ffs.closure === :matched && error("MatchProblem: the result is already matched; pose the match on the outer solve")
    dp = ffs.delta_prime
    dp === nothing &&
        error("MatchProblem: the $(ffs.integrator) result has no Δ′; solve with Galerkin() or Riccati() (Riccati needs vac_flag=true)")
    (isempty(dp.ef) || isempty(dp.raw_fixedboundary)) &&
        error("MatchProblem: needs Δ_EF and the fixed-boundary Δ′; solve with error_field_response=true (Riccati also needs vac_flag=true)")

    msing = length(ffs.surfaces)
    if ffs.galerkin !== nothing
        dropped = [s.psifac for s in ffs.surfaces if !(s.psifac in ffs.galerkin.sing_psi)]
        isempty(dropped) || error("MatchProblem: the Galerkin solve dropped the surfaces at ψ = $dropped (on the domain edge or " *
              "outside mlow:mhigh); matching needs every surface")
    end
    size(dp.raw_fixedboundary, 1) == 2msing ||
        error("MatchProblem: Δ′ raw block is $(size(dp.raw_fixedboundary, 1))×$(size(dp.raw_fixedboundary, 2)) but the result carries $msing surfaces")

    InnerLayer._require_layer_inputs("MatchProblem", ffs.equil; eta, rho, rotation)
    eta === nothing && (eta = InnerLayer.layer_eta(ffs.equil, ffs.surfaces))
    rho === nothing && (rho = InnerLayer.layer_rho(ffs.equil, ffs.surfaces))
    if rotation === nothing
        omega_E = InnerLayer.layer_omega_E(ffs.equil, ffs.surfaces)
        any(isnan, omega_E) && error("MatchProblem: the kinetic data carry no omega_E; give rotation")
        rotation = omega_E ./ 2π
    end
    per_surface(name, v) =
        v isa Real ? fill(Float64(v), msing) :
        length(v) == msing ? collect(Float64, v) :
        error("MatchProblem: $name has length $(length(v)), expected $msing (one per surface, core to edge)")
    return MatchProblem(ffs, per_surface("eta", eta), per_surface("rho", rho), per_surface("rotation", rotation), Float64(gamma))
end

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

Match `prob.ffs` to the inner layer of `model`. The result has `closure = :matched`, `bpen`, the
[`MatchResult`](@ref) in `result.match` and, for a Galerkin solve, the matched ξ, its edge response
`wp` (see [`ForceFreeStatesResult`](@ref)) and, when the solve had `vac_flag`, the free-boundary energies
of that response. A Riccati-fed match drops `wp` and `free_boundary`, so PerturbedEquilibrium refuses
it rather than mixing ideal and matched physics.
"""
function CommonSolve.solve(prob::MatchProblem, model::InnerLayer.InnerLayerModel)
    closure_capable(model) ||
        error("$(nameof(typeof(model))) cannot solve a MatchProblem (use GGJModel(); SLAYERModel is for TearingProblem)")
    match = _compute_match(prob, model)
    return _matched_result(prob.ffs, match)
end

# Inner-layer Δ per surface, the matching system, then the matched outer and inner profiles.
function _compute_match(prob::MatchProblem, model::InnerLayer.GGJModel)
    ffs = prob.ffs
    dp = ffs.delta_prime
    sings = ffs.surfaces
    msing = length(sings)
    nef = size(dp.ef, 2)
    nn = ffs.nlow
    equil = ffs.equil
    chi1 = 2π * equil.psio

    gal_sol = ffs.galerkin === nothing ? nothing : ffs.galerkin.solution

    # Inner-layer Δ and profiles per surface (match.f deltac_run, intotsol).
    deltar = zeros(ComplexF64, msing, 2)
    forced_eigenvalue = zeros(ComplexF64, msing)
    # Layer-center field weight per parity (match.f intotsol_b).
    pen = zeros(ComplexF64, msing, 2)
    inner_psi = Vector{Vector{Float64}}(undef, msing)      # ψ_s ± X·dψdx, ascending
    inner_odd = Vector{Vector{ComplexF64}}(undef, msing)   # Ξ₋, odd about ψ_s
    inner_even = Vector{Vector{ComplexF64}}(undef, msing)  # Ξ₊, even
    inner_bodd = Vector{Vector{ComplexF64}}(undef, msing)  # b^ψ from Ψ₋, even
    inner_beven = Vector{Vector{ComplexF64}}(undef, msing) # b^ψ from Ψ₊, odd
    inner_params = Vector{InnerLayer.GGJParameters}(undef, msing)
    profiles = true
    for i in 1:msing
        params = ggj_parameters(sings[i], equil; eta=prob.eta[i], rho=prob.rho[i],
            gamma=prob.gamma, ising=i)
        inner_params[i] = params
        γ = 2π * im * nn * prob.rotation[i]    # forced eigenvalue, f in Hz
        forced_eigenvalue[i] = γ
        inner = InnerLayer.solve_inner_profile(model, params, γ)   # parity columns (Δ₊, Δ₋)
        deltar[i, 1] = inner.Δ[1]
        deltar[i, 2] = inner.Δ[2]
        profΨ, profΞ, xg = inner.Ψ, inner.Ξ, inner.x
        resc = inner.rescale                    # inner big-branch amplitude → outer δψ normalization
        # b_m = 2πi·χ₁·(m−nq)·ξ_m with m−nq = −n·q′·dψdx·X and Ψ = XΞ at leading order (GWP2016 Eqs. 16, 27; Ψ normalization Eq. A17) ⇒ b_m = scale·resc·Ψ.
        scale = -2π * chi1 * im * nn * sings[i].q1 * inner.dψdx
        pen[i, 1] = scale * profΨ[1, 1] * resc   # X = 0, even parity (Ψ(0) = 0)
        pen[i, 2] = scale * profΨ[1, 2] * resc   # odd parity
        # A backend returning only layer-center values (real-ψ profiles not requested) skips the inner profiles.
        if length(xg) == 1
            profiles = false
            continue
        end
        xvar = xg .* inner.dψdx                  # X → ψ distance
        inner_psi[i] = vcat(reverse(sings[i].psifac .- xvar), sings[i].psifac .+ xvar)
        inner_odd[i] = resc .* vcat(reverse(.-profΞ[:, 2]), profΞ[:, 2])
        inner_even[i] = resc .* vcat(reverse(profΞ[:, 1]), profΞ[:, 1])
        inner_bodd[i] = (scale * resc) .* vcat(reverse(profΨ[:, 2]), profΨ[:, 2])
        inner_beven[i] = (scale * resc) .* vcat(reverse(.-profΨ[:, 1]), profΨ[:, 1])
    end

    cout, cin, residual = _match_system(dp.raw_fixedboundary, dp.ef, deltar)

    # Penetrated field at each layer center (match.f intotsol_b); cin rows per surface are (d₊, d₋).
    bpen = zeros(ComplexF64, msing, nef)
    for i in 1:msing, j in 1:nef
        bpen[i, j] = pen[i, 1] * cin[2i-1, j] + pen[i, 2] * cin[2i, j]
    end

    # Matched inner ξ_ψ and b^ψ per EF column: odd parity × cin[2i], even × cin[2i-1] (match.f intotsol).
    if profiles
        inner_xi = [inner_odd[i] * transpose(cin[2i, :]) .+ inner_even[i] * transpose(cin[2i-1, :]) for i in 1:msing]
        inner_b = [inner_bodd[i] * transpose(cin[2i, :]) .+ inner_beven[i] * transpose(cin[2i-1, :]) for i in 1:msing]
    else
        inner_psi = Vector{Float64}[]
        inner_xi = Matrix{ComplexF64}[]
        inner_b = Matrix{ComplexF64}[]
    end

    reconnected_flux = Matrix{ComplexF64}(dp.ef .+ transpose(dp.raw_fixedboundary) * cout)

    # Matched outer ξ per EF column: EF column + Σ cout·plasma columns (needs the Galerkin basis).
    if gal_sol === nothing
        xi = Array{ComplexF64,3}(undef, 0, 0, 0)
        xi_deriv = Array{ComplexF64,3}(undef, 0, 0, 0)
        empty!(inner_psi)
        empty!(inner_xi)
        empty!(inner_b)
    else
        sols = gal_sol.xi          # (mpert, ngrid, nsol)
        sols_deriv = gal_sol.xi_deriv
        mpert = size(sols, 1)
        ngrid = size(sols, 2)
        xi = zeros(ComplexF64, mpert, ngrid, nef)
        xi_deriv = zeros(ComplexF64, mpert, ngrid, nef)
        for j in 1:nef
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
                      "so the composite inner-region profiles are skipped; bpen and the matched outer ξ are unaffected" maxlog = 1
                empty!(inner_psi)
                empty!(inner_xi)
                empty!(inner_b)
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

                    # Cut outer background of the resonant harmonic, per EF column.
                    cutmn = Matrix{ComplexF64}(undef, length(psi_keep), nef)
                    for j in 1:nef
                        @views cutmn[:, j] .= sols_cut[ires, keep, 2msing+j]
                        for isol in 1:2msing
                            @views cutmn[:, j] .+= cout[isol, j] .* sols_cut[ires, keep, isol]
                        end
                    end
                    itp = cubic_interp(psi_keep, Series(cutmn); extrap=ExtendExtrap())
                    buf = Vector{ComplexF64}(undef, nef)
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

    return MatchResult(cout, cin, deltar, forced_eigenvalue, residual, bpen, reconnected_flux,
        xi, xi_deriv, inner_psi, inner_xi, inner_b, inner_params, model, prob)
end

# Copy of ffs with the match applied. A Galerkin match publishes the matched ξ, its edge response (the matched
# columns are EF + plasma·cout, so their edge response is too) and the free-boundary energies of that response with
# the solve's own vacuum. For Riccati match these remain TODO.
function _matched_result(ffs::ForceFreeStatesResult, match::MatchResult)
    gal = ffs.galerkin
    solution = nothing
    wp = nothing
    free_boundary = nothing
    if gal !== nothing && gal.solution !== nothing && !isempty(match.xi)
        solution = _gal_profiles(gal.solution, match.xi, match.xi_deriv, ffs.mats, ffs)
        wp = gal_wp(gal, ffs.equil.psio; cout=match.cout)
        ffs.free_boundary === nothing || (free_boundary = free_run(wp, ffs.control, ffs.equil, ffs.mats, ffs))
    end
    return _with(ffs; closure=:matched, bpen=match.bpen, match=match, solution=solution, wp=wp, free_boundary=free_boundary)
end

# A copy of the immutable `x` with the named fields replaced.
_with(x; fields...) = typeof(x).name.wrapper((get(fields, f, getfield(x, f)) for f in fieldnames(typeof(x)))...)
