# Ray.jl
#
# Rotated-contour backend (`GGJModel{:ray}`): the GGJ equations continued onto x = e^{iθ}s,
# θ = arg(Q)/4. Far-field series data at the radius S are marched inward to s_m modulo the
# decaying exponential pair, then the Galerkin.jl Hermite assembly solves [0, s_m] with
# v(s_m) = Us + Δ·Ub + c₁E₁ + c₂E₂. Method and rationale: docs/src/inner_layer.md.

"""
    ode_matrix([CT,] p::GGJParameters, Q, x) -> SMatrix{6,6,CT}

Coefficient matrix `M(x)` of GWP2016 Eq. 11 as `dv/dx = M v`, `v = (Ψ, Ξ, Υ, Ψ', Ξ', Υ')`, for
real or complex `x`. `M` is polynomial in `x`, so x = 0 is an ordinary point and the system
continues analytically to complex `x`. `CT` selects the element type (default `ComplexF64`;
`Complex{Double64}` for the extended-precision march).
"""
ode_matrix(p::GGJParameters, Q::ComplexF64, x::Number) =
    ode_matrix(ComplexF64, p, Q, x)

function ode_matrix(::Type{CT}, p::GGJParameters, Q::Number, x::Number) where {CT<:Complex}
    e = CT(p.E)
    f = CT(p.F)
    g = CT(p.G)
    h = CT(p.H)
    k = CT(p.K)
    q = CT(Q)
    q2 = q * q
    xc = CT(x)
    x2 = xc * xc

    m = zeros(MMatrix{6,6,CT})
    # rows 1–3: (Ψ, Ξ, Υ)' = (Ψ', Ξ', Υ')
    m[1, 4] = 1
    m[2, 5] = 1
    m[3, 6] = 1
    # row 4: Ψ'' = Q Ψ − Q x Ξ + H Υ'
    m[4, 1] = q
    m[4, 2] = -q * xc
    m[4, 6] = h
    # row 5: Ξ'' = −(x/Q) Ψ + (x²/Q) Ξ − ((E+F)/Q²) Υ − (H/Q²) Ψ'
    m[5, 1] = -xc / q
    m[5, 2] = x2 / q
    m[5, 3] = -(e + f) / q2
    m[5, 4] = -h / q2
    # row 6: Υ'' = −(x/Q) Ψ − Q(G−KE) Ξ + (x²/Q + Q(G+KF)) Υ + HKQ Ψ'
    m[6, 1] = -xc / q
    m[6, 2] = -q * (g - k * e)
    m[6, 3] = x2 / q + q * (g + k * f)
    m[6, 4] = h * k * q

    return SMatrix{6,6,CT}(m)
end

"""
    ray_breaks(params, Q, θ, S; cfl=0.2, min_cells=1024) -> Vector{Float64}

Cell boundaries `0 = s₀ < s₁ < … = S` for the Hermite BVP: each cell resolves the full
local WKB rate μ + ρs (fast Ohm and Υ-family pair μ = |√Q| + √(|Q|·|G+KF|), parabolic pair
ρ = Re(e^{2iθ}/√Q)) whether or not the mode is present, since unresolved dormant modes
alias into spurious discrete ones. `min_cells` floors the resolution where the local rates
are small (|Q| ≪ 1).
"""
function ray_breaks(params::GGJParameters, Q::ComplexF64, θ::Float64, S::Float64;
    cfl::Float64=0.2, min_cells::Int=1024)
    sq = sqrt(Q)
    ρ = real(cis(2θ) / sq)
    ρ > 0 || throw(ArgumentError("contour angle θ=$θ gives non-decaying parabolic pair (ρ=$ρ ≤ 0); use θ ≈ arg(Q)/4"))
    μ = abs(sq) + sqrt(abs(Q) * abs(params.G + params.K * params.F))
    hmax = S / min_cells
    breaks = [0.0]
    while breaks[end] < S
        h = min(cfl / (μ + ρ * breaks[end]), hmax)
        snew = breaks[end] + h
        push!(breaks, snew >= S - 0.25 * h ? S : snew)
    end
    return breaks
end

# Far-field boundary data. The admissible solutions at large s are the power pair (u_small, u_big)
# from the inps series and the forward-decaying exponential pair E. The series is trusted only at
# S, which can lie far beyond s_m, so the power pair is marched S → s_m by 2-stage Radau IIA,
# R(z) = 2(3+z)/(z² − 4z + 6). Unresolved steps (z > 8, clear of |R| > 1 on 0 < z < 6) damp the
# exponential directions, which is harmless because Δ is defined modulo E.

# Small and large power-like solutions at x as 6-component states (Ψ, Ξ, Υ, Ψ', Ξ', Υ').
function _power_states(cache::InnerAsymptoticsCache, x::Number)
    ua, dua = physical_ua_dua(cache, x)
    return vcat(ua[:, 2], dua[:, 2]), vcat(ua[:, 1], dua[:, 1])
end

# Condition number of the column-normalized boundary basis [Us, Ub, E].
_basis_cond(Us, Ub, E) = cond(hcat(Us ./ norm(Us), Ub ./ norm(Ub), E))

"""
    decaying_pair(params, Q, θ, S; growth=30.0, seed=20260707, dp_res=20.0) -> E (6×2)

Orthonormal basis of the forward-decaying pair at s = S on the ray: random
seeds integrated backward over [S, S+ΔS] with fully-resolved RK4, ΔS sized so
the pair separates from everything else by ≈ `growth` e-folds.
"""
function decaying_pair(params::GGJParameters, Q::ComplexF64, θ::Float64, S::Float64;
    growth::Float64=30.0, seed::Int=20260707, dp_res::Float64=20.0)
    sq = sqrt(Q)
    ρ = real(cis(2θ) / sq)
    ρ > 0 || throw(ArgumentError("θ=$θ gives ρ=$ρ ≤ 0; decaying pair undefined"))
    ph = cis(θ)
    # Window ΔS: the slowest backward grower separates from the power pair by `growth` e-folds.
    g2 = Inf
    gmax_loc = 0.0
    for λ in eigvals(Matrix(ph * ode_matrix(params, Q, ph * S)))
        g = -real(λ)
        g > 20.0 / S && (g2 = min(g2, g))
        gmax_loc = max(gmax_loc, abs(real(λ)))
    end
    gsep = isfinite(g2) ? max(g2 - 20.0 / S, 0.05 * g2) : ρ * S
    ΔS = min(growth / gsep, 3.0 * S)

    # Resolve the true local spectrum; ρs + |√Q| under-counts the Υ-family rate √|Q(G+KF)|.
    for λ in eigvals(Matrix(ph * ode_matrix(params, Q, ph * (S + ΔS))))
        gmax_loc = max(gmax_loc, abs(real(λ)))
    end
    rate = max(ρ * (S + ΔS) + abs(sq), gmax_loc)
    nsteps = clamp(ceil(Int, dp_res * rate * ΔS), 200, 2_000_000)
    h = -ΔS / nsteps

    rng = MersenneTwister(seed)
    v = randn(rng, ComplexF64, 6, 2)
    v = Matrix(qr(v).Q)

    f(s, y) = (ph * ode_matrix(params, Q, ph * s)) * y

    s = S + ΔS
    for istep in 1:nsteps
        k1 = f(s, v)
        k2 = f(s + h / 2, v + (h / 2) * k1)
        k3 = f(s + h / 2, v + (h / 2) * k2)
        k4 = f(s + h, v + h * k3)
        v = v + (h / 6) * (k1 + 2k2 + 2k3 + k4)
        s += h
        if istep % 50 == 0
            v = Matrix(qr(v).Q)
        end
    end
    return Matrix(qr(v).Q)
end

"""
    march_boundary(params, Q, θ, S, s_m, Us, Ub; rtol=1e-9, ztrk=0.3, purge_budget=2.5,
                   growth=30.0, seed=20260707, dp_res=20.0)
        -> (Us_m, Ub_m, F)

Transport the power-pair boundary data from the series radius `S` inward to
`s_m` along the ray, modulo the decaying exponential pair (the quotient in
which Δ is defined): 2-stage Radau IIA with purges against a carried decaying frame.
`rtol` sets the step factor c = (rtol/100)^{1/3}; `ztrk` caps resolved-band steps at ztrk/g_max;
`purge_budget` is the e-folds between purges; `growth`, `seed`, `dp_res` go to [`decaying_pair`](@ref).
"""
function march_boundary(params::GGJParameters, Q::ComplexF64, θ::Float64,
    S::Float64, s_m::Float64, Us::Vector{ComplexF64}, Ub::Vector{ComplexF64};
    rtol::Float64=1e-9, growth::Float64=30.0, seed::Int=20260707,
    ztrk::Float64=0.3, purge_budget::Float64=2.5, dp_res::Float64=20.0)
    ph = cis(θ)
    𝔐(s) = ph * ode_matrix(params, Q, ph * s)

    W = hcat(copy(Us), copy(Ub))
    function purge!(W, s)
        Floc = decaying_pair(params, Q, θ, s; growth=growth, seed=seed, dp_res=dp_res)
        W .-= Floc * (Floc' * W)
        return Floc
    end

    # One Radau IIA step in the precision of W (`Mfun` returns the matrix in that precision):
    # A = [5/12 -1/12; 3/4 1/4], c = (1/3, 1), b = A[2,:].
    function radau_step(Mfun, s, W, h)
        CT = eltype(W)
        hc = real(CT)(h)
        Id = Matrix{CT}(I, 6, 6)
        M1 = Mfun(s + h / 3)
        M2 = Mfun(s + h)
        Ablk = [Id-(5hc/12)*M1 (hc/12)*M1; -(3hc / 4)*M2 Id-(hc/4)*M2]
        K = Ablk \ [M1 * W; M2 * W]
        return W + hc * ((CT(3) / 4) * @view(K[1:6, :]) + (CT(1) / 4) * @view(K[7:12, :]))
    end

    # Damped-zone steps have cond ~ z ≫ 1; their structured LU backward error mixes u_small into u_big
    # by ~eps·ρS²/2·‖u_big‖/‖u_small‖, so that zone runs in Double64.
    CTd = Complex{Double64}
    phd = CTd(cos(Double64(θ)), sin(Double64(θ)))
    𝔐d(s) = phd * Matrix(ode_matrix(CTd, params, Q, phd * Double64(s)))

    # Step law, from the local growth rates g of 𝔐(s):
    #   damped zone (every grower has g·c·s > 8): h = −c·s with c = (rtol/100)^{1/3}, so the step
    #     count grows only like ln(S/s_m);
    #   resolved band: h = −min(c·s, ztrk/g_max), with the decaying frame F carried by the same
    #     steps and purged from W every `purge_budget` e-folds.
    # On the real axis with imaginary Q the band entry lands on the turning point x² = −Q²(G+KF),
    # where F cannot be identified; the default θ keeps it off the contour.
    c = clamp(cbrt(rtol / 100.0), 2e-5, 2e-3)

    # Damped zone, in Double64 from the first purge at S to the resolved-band entry.
    Wd = CTd.(W)
    let Floc = decaying_pair(params, Q, θ, S; growth=growth, seed=seed, dp_res=dp_res)
        Fd = CTd.(Floc)
        Wd .-= Fd * (Fd' * Wd)
    end
    s = S
    nstep = 0
    while s > s_m + 1e-12 * S
        λs = eigvals(Matrix(𝔐(s)))
        gmin = Inf
        for λ in λs
            g = -real(λ)
            g > 10.0 / s && (gmin = min(gmin, g))
        end
        (isfinite(gmin) && gmin * c * s < 8.0) && break   # resolved-band entry
        h = -min(c * s, s - s_m)
        Wd = radau_step(𝔐d, s, Wd, h)
        s += h
        nstep += 1
        nstep > 5_000_000 && error("march_boundary: step limit exceeded (s=$s)")
    end
    W = ComplexF64.(Wd)

    # Resolved band, in Float64 with the carried frame.
    Φ = 0.0
    F = Matrix{ComplexF64}(undef, 6, 2)
    have_F = false
    while s > s_m + 1e-12 * S
        # Backward-growing rates (Re λ < 0); rates ≲ 10/s belong to the algebraic power pair.
        λs = eigvals(Matrix(𝔐(s)))
        gmin, gmax = Inf, 0.0
        for λ in λs
            g = -real(λ)
            if g > 10.0 / s
                gmin = min(gmin, g)
                gmax = max(gmax, g)
            end
        end
        resolved = isfinite(gmin) && gmin * c * s < 8.0
        h = resolved ? -min(c * s, ztrk / gmax, s - s_m) : -min(c * s, s - s_m)
        if resolved && !have_F
            F = decaying_pair(params, Q, θ, s; growth=growth, seed=seed, dp_res=dp_res)
            W .-= F * (F' * W)
            have_F = true
            Φ = 0.0
        end
        W = radau_step(𝔐, s, W, h)
        if have_F
            F = radau_step(𝔐, s, F, h)
            F = Matrix(qr(F).Q)
            Φ += gmax * abs(h)
            if Φ > purge_budget
                W .-= F * (F' * W)
                Φ = 0.0
            end
        end
        s += h
        nstep += 1
        nstep > 5_000_000 && error("march_boundary: step limit exceeded (s=$s)")
    end
    # E must span the solutions that decay at infinity, which differ from the locally decaying pair at
    # s_m once the mode families morph (s ~ |Q|); the carried F is that subspace. Local extraction is
    # the fallback when the march never entered the resolved band.
    if have_F
        W .-= F * (F' * W)
    else
        F = purge!(W, s_m)
    end
    return W[:, 1], W[:, 2], F
end

"""
    boundary_basis(params, Q, cache, θ, S; growth=30.0, seed=20260707, dp_res=20.0)
        -> (Us, Ub, E, cond_est)

Boundary data at s = S taken directly from the inps series, used when the BVP edge is S itself
(s_m ≳ S): `Us`, `Ub` are the small/large
power-like solutions as 6-component states in the inps normalization, `E`
the numerically-generated decaying pair, `cond_est` the condition number of
the column-normalized basis.
"""
function boundary_basis(params::GGJParameters, Q::ComplexF64,
    cache::InnerAsymptoticsCache, θ::Float64, S::Float64;
    growth::Float64=30.0, seed::Int=20260707, dp_res::Float64=20.0)
    Us, Ub = _power_states(cache, cis(θ) * S)
    E = decaying_pair(params, Q, θ, S; growth=growth, seed=seed, dp_res=dp_res)
    return Us, Ub, E, _basis_cond(Us, Ub, E)
end

# Driver: series radius S, BVP edge s_m, march S → s_m, Hermite BVP on [0, s_m].

"""
    RaySolveResult

Output of [`solve_ray`](@ref).

## Fields

  - `Δ`: rescaled matching data (Δ₊, Δ₋) = (interchange, tearing), the even (isol = 1) and
    odd (isol = 2) parities of the displacement Ξ.
  - `Δraw`: the same pair in raw inps normalization, before `rescale_delta`.
  - `cexp`: decaying-pair coefficients (c₁, c₂) at s_m, one column per parity.
  - `Q`, `θ`: scaled eigenvalue and contour angle.
  - `S`: series radius where the far-field data were taken.
  - `series_ok`: whether `pick_smax` reached its residual tolerance.
  - `bc_cond`: condition number of the column-normalized boundary basis [Us, Ub, E].
  - `breaks`: Hermite cell boundaries on [0, s_m].
  - `cfl`, `min_cells`: mesh resolution parameters used by [`ray_breaks`](@ref).
  - `ws`: the full Galerkin workspace of the BVP (system buffers included); the nodal solution is `ws.sol`, both parities.
"""
struct RaySolveResult
    Δ::SVector{2,ComplexF64}
    Δraw::SVector{2,ComplexF64}
    cexp::SMatrix{2,2,ComplexF64,4}
    Q::ComplexF64
    θ::Float64
    S::Float64
    series_ok::Bool
    bc_cond::Float64
    breaks::Vector{Float64}
    cfl::Float64
    min_cells::Int
    ws::GalerkinWorkspace
end

"""
    solve_ray(params::GGJParameters, Q; θ=angle(Q)/4, kmax=12, S=nothing, smax_tol=1e-9,
              cfl=0.2, min_cells=1024, nq=6, verbose=false, …) -> RaySolveResult

Solve the GGJ inner-layer problem on the rotated ray x = e^{iθ}s. `Δ` is the rescaled pair
(Δ₊, Δ₋) = (interchange, tearing), directly comparable to the `:galerkin` backend; θ = 0 is a
real-axis solve.

Both boundary points are computed per solve. `S` defaults to the smallest radius where the series
residual is below `smax_tol` with order `kmax` ([`pick_smax`](@ref); it warns and uses the best
point if the tolerance is not reached). The BVP edge s_m is where every exponential family has
decayed 45 e-folds from the origin, from the local spectrum, scaled by `sm_fac`. `cfl`,
`min_cells` set the Hermite grid ([`ray_breaks`](@ref)) and `nq` its quadrature. `march_rtol`
is [`march_boundary`](@ref)'s `rtol`; `growth`, `seed`, `ztrk`, `purge_budget`, `dp_res` go to
`march_boundary`/[`decaying_pair`](@ref); `handoff` (diagnostic) supplies (Us, Ub) at S.
"""
function solve_ray(params::GGJParameters, Q::ComplexF64;
    θ::Float64=angle(Q) / 4, kmax::Int=12,
    S::Union{Nothing,Float64}=nothing, smax_tol::Float64=1e-9,
    growth::Float64=30.0, march_rtol::Float64=1e-9, sm_fac::Float64=1.0,
    cfl::Float64=0.2, min_cells::Int=1024, nq::Int=6, seed::Int=20260707,
    ztrk::Float64=0.3, purge_budget::Float64=2.5, dp_res::Float64=20.0,
    handoff::Union{Nothing,NTuple{2,Vector{ComplexF64}}}=nothing,
    verbose::Bool=false)

    cache = build_asymptotics(params, Q; kmax=kmax)
    series_ok = true
    if S === nothing
        Sv, cache, series_ok = pick_smax(params, Q; θ=θ, eps=smax_tol, kmax=kmax, cache=cache)
        series_ok || @warn "pick_smax: series residual target $smax_tol not reached; using best point S=$Sv"
    else
        Sv = S
    end

    # s_m: where every exponential family has decayed 45 e-folds from the origin, from the local spectrum.
    # When s_m ≳ S (small |Q|) the series data are applied at S directly.
    s_m = let acc = 0.0, sprev = 0.0, out = Sv
        for sk in exp10.(range(-2, log10(Sv); length=400))
            gdec = Inf
            for λ in eigvals(Matrix(cis(θ) * ode_matrix(params, Q, cis(θ) * sk)))
                g = -real(λ)                    # backward-growth = forward-decay rate
                g > 10.0 / sk && (gdec = min(gdec, g))
            end
            isfinite(gdec) && (acc += gdec * (sk - sprev))
            sprev = sk
            if acc >= 45.0
                out = sk
                break
            end
        end
        sm_fac * max(out, 2.0)
    end

    local Us, Ub, E, bc_cond, Sbvp
    if s_m < 0.98 * Sv
        # `handoff` is a diagnostic: externally supplied series data (Us, Ub) at x = e^{iθ}S.
        UsS, UbS = handoff === nothing ? _power_states(cache, cis(θ) * Sv) : handoff
        Us, Ub, E = march_boundary(params, Q, θ, Sv, s_m, UsS, UbS;
            rtol=march_rtol, growth=growth, seed=seed,
            ztrk=ztrk, purge_budget=purge_budget, dp_res=dp_res)
        Sbvp = s_m
        bc_cond = _basis_cond(Us, Ub, E)
        verbose && @printf("  march: S=%.4g → s_m=%.4g, bc_cond=%.1e\n", Sv, s_m, bc_cond)
    else
        Us, Ub, E, bc_cond = boundary_basis(params, Q, cache, θ, Sv;
            growth=growth, seed=seed, dp_res=dp_res)
        Sbvp = Sv
    end
    bc_cond < 1e12 || @warn "boundary basis poorly conditioned (cond ≈ $bc_cond)"

    breaks = ray_breaks(params, Q, θ, Sbvp; cfl=cfl, min_cells=min_cells)
    ws = _galerkin_workspace(breaks, fill(CT_NONE, length(breaks) - 1))
    Δraw, cexp = _assemble_and_solve_point!(ws, params, Q, θ, Us, Ub, E; nq=nq)
    verbose && @printf("  BVP: %d Hermite cells on [0, %.4g], Δraw[1] = %.6e %+.6eim\n",
        length(breaks) - 1, Sbvp, real(Δraw[1]), imag(Δraw[1]))

    Δ = rescale_delta(Δraw, params)

    return RaySolveResult(Δ, Δraw, cexp, Q, θ, Sv, series_ok, bc_cond, breaks, cfl, min_cells, ws)
end

solve_ray(params::GGJParameters, Q::Number; kwargs...) =
    solve_ray(params, ComplexF64(Q); kwargs...)

"""
    solve_inner(::GGJModel{:ray}, params::GGJParameters, γ::Number; kwargs...)
        -> InnerLayerResponse

Rotated-ray backend: converts γ via `inner_Q`, forwards keywords to
[`solve_ray`](@ref), and returns the parity-projected matching data (Δ₊, Δ₋) as the
`interchange` and `tearing` fields of [`InnerLayerResponse`](@ref).
Preferred for |Q| ≳ 1 and near the imaginary axis; use `solve_ray` directly
when the full [`RaySolveResult`](@ref) is wanted. `real_profiles` is a profile-only model
option (see `solve_inner_profile`) and is ignored here.
"""
function solve_inner(::_BareGGJ{:ray}, params::GGJParameters, γ::Number; real_profiles::Bool=false, kwargs...)
    res = solve_ray(params, inner_Q(params, γ); kwargs...)
    return InnerLayerResponse(; interchange=res.Δ[1], tearing=res.Δ[2])
end

"""
    solve_inner_profile(::GGJModel{:ray}, params::GGJParameters, γ::Number;
                        real_profiles=false, npc=8, certify_rtol=1e-3, kwargs...)
        -> (; Δ, x, Ψ, Ξ, dψdx, rescale, certΔ)

Rotated-ray implementation of the [`solve_inner_profile`](@ref) interface. `Δ` and the
layer-center fields come from the solve at θ = arg(Q)/4; x = 0 lies on every contour, so
Ψ(0), Ξ(0) are the real-axis values, and by default only they are returned (`x = [0.0]`,
`certΔ = NaN`). With `real_profiles = true` the fields across the layer come from an extra
θ = 0 solve (`npc` points per cell), and the relative difference of its Δ₊ is `certΔ`,
warned above `certify_rtol`. Other keywords forward to [`solve_ray`](@ref), except θ.
"""
function solve_inner_profile(::_BareGGJ{:ray}, params::GGJParameters, γ::Number;
    real_profiles::Bool=false, npc::Int=8, certify_rtol::Float64=1e-3, kwargs...)
    Q = inner_Q(params, γ)
    rr = solve_ray(params, Q; kwargs...)             # Δ and layer-center fields (optimal θ)
    if !real_profiles
        c = solution_profile(rr; npc=1)              # first point is s = 0
        return (; Δ=rr.Δ, x=[0.0], Ψ=c.Ψ[1:1, :], Ξ=c.Ξ[1:1, :], _profile_conversions(params)..., certΔ=NaN)
    end
    r0 = solve_ray(params, Q; θ=0.0, kwargs...)      # real-axis profile solve
    certΔ = abs(r0.Δ[1] - rr.Δ[1]) / max(abs(rr.Δ[1]), 1e-300)
    certΔ > certify_rtol && @warn "GGJ ray: θ=0 profile solve disagrees with certified Δ" ising = params.ising Q certΔ
    prof = solution_profile(r0; npc=npc)
    return (; Δ=rr.Δ, x=prof.s, Ψ=prof.Ψ, Ξ=prof.Ξ, _profile_conversions(params)..., certΔ=certΔ)
end

"""
    solution_profile(res::RaySolveResult; npc=8) -> (; s, x, Ψ, Ξ, Υ)

Reconstruct (Ψ, Ξ, Υ) on the solution contour from the Hermite nodal solution, `npc`
points per cell; each field is `npts × 2` with columns (even, odd) as in `Δ`, and `x = e^{iθ}s`.
For rotated solves the fields live on the ray, not the real axis.
"""
function solution_profile(res::RaySolveResult; npc::Int=8)
    prof = _solution_profile(res.ws; npc=npc)
    return (; s=prof.x, x=cis(res.θ) .* prof.x, Ψ=prof.Ψ, Ξ=prof.Ξ, Υ=prof.Υ)
end

"""
    asymptotic_profile(params, res::RaySolveResult, srange) -> (; s, x, Ψ, Ξ, Υ)

Evaluate the analytic representation `u_small + Δ·u_big` on the ray for
`s ≥ res.S`; concatenated with [`solution_profile`](@ref) it should join
the numeric solution seamlessly across the march zone.
"""
function asymptotic_profile(params::GGJParameters, res::RaySolveResult,
    srange::AbstractVector{<:Real})
    cache = build_asymptotics(params, res.Q; kmax=12)
    n = length(srange)
    Ψ = Matrix{ComplexF64}(undef, n, 2)
    Ξ = Matrix{ComplexF64}(undef, n, 2)
    Υ = Matrix{ComplexF64}(undef, n, 2)
    for (k, s) in enumerate(srange)
        ua, _ = physical_ua_dua(cache, cis(res.θ) * s)
        for isol in 1:2
            Δ = res.Δraw[isol]
            Ψ[k, isol] = ua[1, 2] + Δ * ua[1, 1]
            Ξ[k, isol] = ua[2, 2] + Δ * ua[2, 1]
            Υ[k, isol] = ua[3, 2] + Δ * ua[3, 1]
        end
    end
    return (; s=collect(float.(srange)), x=cis(res.θ) .* srange, Ψ=Ψ, Ξ=Ξ, Υ=Υ)
end

"""
    delta_convergence(params, Q; verbose=true, kwargs...) -> (; Δ, spread, base, table)

Error bar for the matching data: re-solve with each numerical knob perturbed in turn (contour
angle, mesh, series order and radius, march tolerance, BVP edge s_m, decaying-pair separation) and
report the relative change of (Δ₊, Δ₋); `spread` is the worst case per parity. θ moves outward
only, away from the pseudo-resonance x² = −Q²(G+KF).
"""
function delta_convergence(params::GGJParameters, Q::ComplexF64; verbose::Bool=true, kwargs...)
    base = solve_ray(params, Q; kwargs...)
    variations = [
        ("θ → 1.2θ", (; θ=(base.θ == 0 ? 0.1 : 1.2 * base.θ))),
        ("mesh ×2", (; cfl=(base.cfl / 2), min_cells=2 * base.min_cells)),
        ("kmax+4, S↑", (; kmax=16, smax_tol=1e-10)),
        ("march_rtol/100", (; march_rtol=1e-11)),
        ("sm_fac 1→1.4", (; sm_fac=1.4)),
        ("growth 30→45", (; growth=45.0))
    ]
    table = NamedTuple[]
    spread = MVector{2,Float64}(0.0, 0.0)
    verbose && @printf("Δ convergence check, Q = %s:\n", string(Q))
    verbose && @printf("  baseline: Δ₊ = %+.8e %+.8eim, Δ₋ = %+.8e %+.8eim\n",
        real(base.Δ[1]), imag(base.Δ[1]), real(base.Δ[2]), imag(base.Δ[2]))
    for (name, kw) in variations
        r = solve_ray(params, Q; kwargs..., kw...)
        d1 = abs(r.Δ[1] - base.Δ[1]) / abs(base.Δ[1])
        d2 = abs(r.Δ[2] - base.Δ[2]) / abs(base.Δ[2])
        spread[1] = max(spread[1], d1)
        spread[2] = max(spread[2], d2)
        push!(table, (; name, d1, d2))
        verbose && @printf("  %-16s δΔ₊ = %.2e   δΔ₋ = %.2e\n", name, d1, d2)
    end
    verbose && @printf("  %-16s δΔ₊ = %.2e   δΔ₋ = %.2e\n", "WORST CASE", spread[1], spread[2])
    return (; Δ=base.Δ, spread=SVector(spread), base=base, table=table)
end

delta_convergence(params::GGJParameters, Q::Number; kwargs...) =
    delta_convergence(params, ComplexF64(Q); kwargs...)
