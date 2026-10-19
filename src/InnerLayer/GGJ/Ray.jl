# Ray.jl
#
# Rotated-contour backend (`GGJModel{:ray}`) for the GGJ inner-layer matching data,
# robust at large |Q| where the stock `:galerkin` matching degrades. Method write-up:
# docs/src/inner_layer.md, "Numerical method".
#
#   The equations are continued analytically onto the ray x = e^{iθ}s, θ = arg(Q)/4
#   (parabolic-cylinder exponent exactly real; pseudo-resonance off the contour). The
#   far-field condition uses the same inps Wasow basis as the other backends, applied at
#   the series radius S and transported S → s_m by an L-stable 2-stage Radau IIA march in
#   the quotient modulo the decaying exponential pair (damped zone in Complex{Double64}).
#   On [0, s_m] the stock Hermite-cubic Galerkin assembly (Galerkin.jl) solves the BVP with
#   parity conditions at the origin and v(s_m) = Us + Δ·Ub + c₁E₁ + c₂E₂ at the far end.

# The GGJ inner-region equations (GWP2016 Eq. 11 ≡ GW2020 Eq. 1) as a plain
# first-order system in v = (Ψ, Ξ, Υ, Ψ', Ξ', Υ'), dv/dx = M(x)·v.
# All coefficients are POLYNOMIAL in x, so x = 0 is an ordinary point (the
# x⁻²/x⁻⁴ singularities of the GW2020 Eq. 2 form are artifacts of the
# (xΨ, Ψ'/x) scaling) and the system continues analytically to complex x.
#
#   Ψ'' = H Υ' + Q(Ψ − xΞ)
#   Ξ'' = [Q x² Ξ − Q x Ψ − (E+F) Υ − H Ψ'] / Q²
#   Υ'' = [x² Υ − x Ψ − Q²(G(Ξ−Υ) − K(EΞ + FΥ + HΨ'))] / Q
#
# Row-by-row this matches the deltac `inpso_get_uv` matrices (I, V, U) after
# dividing each equation by its diagonal second-derivative coefficient
# (1, Q², Q).

"""
    ode_matrix([CT,] p::GGJParameters, Q, x) -> SMatrix{6,6,CT}

Coefficient matrix `M(x)` of `dv/dx = M v`, `v = (Ψ, Ξ, Υ, Ψ', Ξ', Υ')`, for
real or complex `x`. `CT` selects the element type (default `ComplexF64`;
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

# Large-s boundary data at the matching point on the ray x = e^{iθ}s.
#
# The admissible (bounded-as-s→∞) solution space is spanned by
#   { u_small, u_big }  — the two power-like Mercier solutions, taken from
#                         the inps series in the standard normalization, and
#   { E₁, E₂ }          — the two forward-DECAYING exponential solutions.
#
# The inps series is only trusted at radius S (residual ≤ smax_tol), which
# for extreme-coefficient surfaces (K ~ 10⁶) can be S ~ 10⁴–10⁵, while the
# BVP ends at s_m ~ 30–150. The gap is bridged by MARCHING the
# power-pair data W = [u_small, u_big] backward S → s_m.
#
# Marching design (the hard-won part):
#   - Backward, the decaying pair GROWS at rate ρs (ρ = Re(e^{2iθ}/√Q)), up
#     to ~10³/unit at large s. Any explicit integrator — including
#     projected/continuous-QR variants, whose unprojected stage points
#     reintroduce the transverse stiffness — is stability-limited to
#     h ≲ 3/(ρs), i.e. O(ρS²) steps: hopeless at S ~ 10⁵.
#   - The system is LINEAR, so we use an implicit method (2-stage Radau IIA,
#     order 3; one 12×12 solve per step) with stability function
#     R(z) = 2(3+z)/(z² − 4z + 6) → 2/z as |z| → ∞ in any direction. Resolved
#     slow content is integrated to 3rd order; under-resolved exponential
#     directions are DAMPED whatever their sign. That is normally an accuracy
#     vice; here it is exactly the required behavior, because Δ is defined
#     modulo the decaying pair. For a backward-growing mode |R| > 1 on
#     0 < z < 6 (poles at z = 2 ± i√2), so damped steps need z > 8.
#   - In the partially-resolved band (|z| = ρs·h ≲ 10) genuine backward
#     growth of rounding-injected decaying-pair content still occurs, so W
#     is PURGED against the decaying frame every `purge_budget` e-folds of
#     ∫g_max ds. Purging only ever removes span{E} content —
#     legitimate by the definition of Δ — so the transported coefficients
#     of u_small/u_big are exact up to integrator tolerance.

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
    # Purification window: BOTH backward-growing directions must separate
    # from the power pair by ≥ `growth` e-folds over [S, S+ΔS], so the
    # budget is set by the SLOWEST grower's rate margin over the algebraic
    # power rates (from the frozen spectrum) — not by ρS. At moderate |Q|
    # the fast-Ohm grower can be ~0.6/unit while ρS ~ 10²: budgeting on ρS
    # leaves the second frame column O(1)-contaminated with power content,
    # which then poisons every march purge that uses the frame.
    g2 = Inf
    gmax_loc = 0.0
    for λ in eigvals(Matrix(ph * ode_matrix(params, Q, ph * S)))
        g = -real(λ)
        g > 20.0 / S && (g2 = min(g2, g))
        gmax_loc = max(gmax_loc, abs(real(λ)))
    end
    gsep = isfinite(g2) ? max(g2 - 20.0 / S, 0.05 * g2) : ρ * S
    ΔS = min(growth / gsep, 3.0 * S)

    # Step resolution must use the TRUE local spectrum, not the analytic
    # ρs + |√Q| estimate: on extreme-coefficient surfaces the Υ-family rate
    # √|Q(G+KF)| dominates at small s and the analytic estimate under-counts
    # it 10×, leaving h·λ ~ 0.2 — stable but phase-inaccurate, which corrupts
    # the extracted frame directions (observed as O(1) θ-drift downstream).
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
    march_boundary(params, Q, θ, S, s_m, Us, Ub; rtol=1e-9, growth=30.0, seed=20260707)
        -> (Us_m, Ub_m, F)

Transport the power-pair boundary data from the series radius `S` inward to
`s_m` along the ray, modulo the decaying exponential pair (the quotient in
which Δ is defined). 2-stage Radau IIA (L-stable) with Φ-budgeted purges
against a carried decaying frame; explicit marchers cannot do this at large S
(see file header).
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

    # Extended precision (Double64) for the damped-zone W march.
    # At z = ρ·c·s² ≫ 1 the step matrices have cond ~ z and the LU backward
    # error has a SYSTEMATIC direction set by the elimination structure — it
    # does not dither away, accumulating a relative error Σ eps·z ≈ eps·ρS²/2
    # on the power columns that is INDEPENDENT of the step law (c cancels).
    # The near-parallel power-pair geometry at large S turns that into
    # u_small/u_big coefficient mixing β ~ eps·ρS²/2 · (‖u_big‖/‖u_small‖)(S)
    # ~ 1e-4 at |Q| = 500 — saturating Δ of the near-pole parity at ~1/β and
    # flooring the other at |Δ|·β — the Δ-mixing defect extended precision removes.
    # Double64 arithmetic (eps ~ 5e-33) removes it outright; the resolved
    # band stays Float64 (z ≤ ztrk there, solves well-conditioned).
    CTd = Complex{Double64}
    phd = CTd(cos(Double64(θ)), sin(Double64(θ)))
    𝔐d(s) = phd * Matrix(ode_matrix(CTd, params, Q, phd * Double64(s)))

    # --- Schedule: geometric where damped, growth-capped where resolved. ---
    # No adaptive controller (three generations died on estimator artifacts);
    # the step law is derived from the local mode structure instead. At each
    # step the eigenvalues of 𝔐(s) give the two backward-growing exponential
    # rates g (Re λ < 0, excluding the algebraic power-pair rates ~ r/s):
    #
    #   damped zone (g_min·c·s > 8 for all growers): h = -c·s. All growing
    #     content — including the O(smax_tol) fast-direction contamination of
    #     the series data — sits at z > 8 where Radau DAMPS it
    #     (|R| ≈ 2/z). Slow-mode global error ≈ 10²c³ → c = (rtol/100)^{1/3};
    #     step count ~ ln(S/s_m)/c, growing only logarithmically with S.
    #
    #   resolved band (some grower at z ≲ 8): genuine backward growth must be
    #     tracked and removed. h = -min(c·s, 0.3/g_max) keeps z ≤ 0.3 so the
    #     DISCRETE growing eigendirection matches the continuous one to
    #     O(z²) ~ 1% (an orthogonal purge against a frame misaligned by δ
    #     leaks δ·content — with e-fold budget Φ ≤ 2.5 the cycle gain
    #     e^Φ·δ < 0.2 and contamination stays at the rounding floor). The
    #     frame F is carried CONTINUOUSLY with the same Radau steps + QR
    #     (backward-attracting ⇒ self-correcting), so no re-extraction cost.
    #
    #   The band is entered where the slowest grower's rate drops. On the real
    #   axis with imaginary Q that is the Υ-family turning point x² = −Q²(G+KF),
    #   where the decaying frame cannot be identified and the march fails; the
    #   default θ = arg(Q)/4 keeps the turning point off the contour.
    c = clamp(cbrt(rtol / 100.0), 2e-5, 2e-3)

    # --- Phase 1: damped zone, W in Double64 (see the Double64 comment above). ---
    # First purge (at S) and all damped-zone steps run in extended precision;
    # on resolved-band entry W drops back to Float64 for phase 2.
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

    # --- Phase 2: resolved band (Float64, carried frame + purges). ---
    Φ = 0.0
    F = Matrix{ComplexF64}(undef, 6, 2)
    have_F = false
    while s > s_m + 1e-12 * S
        # Backward-growing exponential rates from the frozen-coefficient
        # spectrum (Re λ < 0 ⇒ grows as s decreases); rates ≲ 10/s are the
        # algebraic power pair, not exponential growers.
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
    # The boundary basis must span the traces AT s_m of the solutions that
    # decay AT INFINITY — which, after the mode families morph around
    # s ~ |Q|, is NOT the same subspace as the locally-decaying pair at s_m.
    # The carried frame F is exactly the marched ∞-decaying pair (it enters
    # the resolved band before the morphing region and tracks the subspace
    # through it), so it IS the correct E-basis; re-extracting locally at
    # s_m substitutes the wrong subspace and corrupts Δ at O(1) on
    # extreme-coefficient surfaces. Local extraction remains only as the
    # fallback when the march never entered the resolved band (no morphing
    # crossed under resolution — the subspaces then coincide).
    if have_F
        W .-= F * (F' * W)
    else
        F = purge!(W, s_m)
    end
    return W[:, 1], W[:, 2], F
end

"""
    boundary_basis(params, Q, cache, θ, S; growth=30.0)
        -> (Us, Ub, E, cond_est)

Boundary data at s = S directly from the inps series (no march; used when
the series is already valid at the BVP edge): `Us`, `Ub` are the small/large
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

# Driver: series radius S, BVP edge s_m, march S → s_m, then the Hermite Galerkin BVP on
# [0, s_m] with the far-field state v(s_m) = Us + Δ·Ub + c₁E₁ + c₂E₂ (Galerkin.jl,
# `_assemble_and_solve_point!`).

"""
    RaySolveResult

Output of [`solve_ray`](@ref).

## Fields

  - `Δ`: rescaled matching data in the deltac output ordering (index 1 ↔ even-BC solution,
    2 ↔ odd-BC solution, the same swap deltac_solve applies).
  - `Δraw`: (isol = 1, isol = 2) in raw inps normalization, before `rescale_delta`.
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
              cfl=0.2, min_cells=1024, nq=6, verbose=false, kwargs...) -> RaySolveResult

Solve the GGJ inner-layer problem on the rotated ray x = e^{iθ}s.
`Δ` follows the deltac output convention (swap + `rescale_delta`), directly
comparable to the `:galerkin` backend. θ = 0 reproduces a real-axis solve;
`S` defaults to the inps series-residual radius (`pick_smax`); `cfl`, `min_cells` set the
Hermite grid ([`ray_breaks`](@ref)) and `nq` its quadrature; remaining keywords are march
knobs (see `march_boundary`, `decaying_pair`).
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

    # Beyond s_m the solution is pure power law, carried by the quotient march; for small |Q| (s_m ≳ S) no march is needed.
    # s_m: where every exponential solution family has decayed ≥ 45 e-folds from the origin, computed from the actual
    # frozen-coefficient spectrum (the asymptotic ρs rate is wrong near the
    # origin for extreme-coefficient surfaces, where the Υ-family rate
    # √|Q(G+KF)| dominates and kills exponential content within s ~ O(1)).
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

    Δswapped = SVector{2,ComplexF64}(Δraw[2], Δraw[1])   # deltac_solve convention
    Δ = rescale_delta(Δswapped, params)

    return RaySolveResult(Δ, Δraw, cexp, Q, θ, Sv, series_ok, bc_cond, breaks, cfl, min_cells, ws)
end

solve_ray(params::GGJParameters, Q::Number; kwargs...) =
    solve_ray(params, ComplexF64(Q); kwargs...)

"""
    solve_inner(::GGJModel{:ray}, params::GGJParameters, γ::Number; kwargs...)
        -> InnerLayerResponse

Rotated-ray backend: converts γ via `inner_Q`, forwards keywords to
[`solve_ray`](@ref), and returns the parity-projected matching data. The
`solve_ray` pair is ordered `(Δ₁, Δ₂) = (interchange, tearing)`, so the two
channels are swapped into the named fields of [`InnerLayerResponse`](@ref).
Preferred for |Q| ≳ 1 and near the imaginary axis; use `solve_ray` directly
when the full [`RaySolveResult`](@ref) is wanted. `real_profiles` is a profile-only model
option (see `solve_inner_profile`) and is ignored here.
"""
function solve_inner(::_BareGGJ{:ray}, params::GGJParameters, γ::Number; real_profiles::Bool=false, kwargs...)
    res = solve_ray(params, inner_Q(params, γ); kwargs...)
    return InnerLayerResponse(res.Δ[2], res.Δ[1])
end

"""
    solve_inner_profile(::GGJModel{:ray}, params::GGJParameters, γ::Number;
                        real_profiles=false, npc=8, certify_rtol=1e-3, kwargs...)
        -> (; Δ, x, Ψ, Ξ, dψdx, rescale, certΔ)

Rotated-ray implementation of the [`solve_inner_profile`](@ref) interface. `Δ` and the
layer-center fields come from the solve at θ = arg(Q)/4; x = 0 lies on every contour, so
Ψ(0), Ξ(0) there are the real-axis values. By default only those center values are
returned (`x = [0.0]`, one-row `Ψ`, `Ξ`; `certΔ = NaN`), which is all the penetrated field
needs. With `real_profiles = true` the fields across the layer come from an extra θ = 0
re-solve on the real axis, and the θ = 0 vs θ = arg(Q)/4 Δ agreement is returned as `certΔ`
(a runtime certificate; warns above `certify_rtol`). `npc` sets the profile points per mesh
cell; extra keywords forward to [`solve_ray`](@ref) (θ is fixed by the method — do not pass it).
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
points per cell; each field is `npts × 2` with columns (odd, even) and `x = e^{iθ}s`.
For rotated solves the fields live on the ray — continuation back to real x is a
separate (unstable) problem, by design.
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

Error bar for the matching data: solve at the baseline, then re-solve with each
numerical knob perturbed on an independent axis (contour angle θ, mesh resolution,
series order/radius, march tolerance, handoff radius, purification e-folds — see
`variations` below) and report the relative change of (Δ₁, Δ₂) per knob. θ is perturbed
outward only: inward moves toward the x² ≈ −Q²(G+KF) pseudo-resonance and measures the
perturbed solve, not the baseline. `spread` is the per-parity worst case over all knobs.
"""
function delta_convergence(params::GGJParameters, Q::ComplexF64; verbose::Bool=true, kwargs...)
    base = solve_ray(params, Q; kwargs...)
    variations = [
        ("θ → 1.2θ", (; θ=(base.θ == 0 ? 0.1 : 1.2 * base.θ))),
        ("mesh ×2", (; cfl=base.cfl / 2, min_cells=2 * base.min_cells)),
        ("kmax+4, S↑", (; kmax=16, smax_tol=1e-10)),
        ("march_rtol/100", (; march_rtol=1e-11)),
        ("sm_fac 1→1.4", (; sm_fac=1.4)),
        ("growth 30→45", (; growth=45.0))
    ]
    table = NamedTuple[]
    spread = MVector{2,Float64}(0.0, 0.0)
    verbose && @printf("Δ convergence check, Q = %s:\n", string(Q))
    verbose && @printf("  baseline: Δ₁ = %+.8e %+.8eim, Δ₂ = %+.8e %+.8eim\n",
        real(base.Δ[1]), imag(base.Δ[1]), real(base.Δ[2]), imag(base.Δ[2]))
    for (name, kw) in variations
        r = solve_ray(params, Q; kwargs..., kw...)
        d1 = abs(r.Δ[1] - base.Δ[1]) / abs(base.Δ[1])
        d2 = abs(r.Δ[2] - base.Δ[2]) / abs(base.Δ[2])
        spread[1] = max(spread[1], d1)
        spread[2] = max(spread[2], d2)
        push!(table, (; name, d1, d2))
        verbose && @printf("  %-16s δΔ₁ = %.2e   δΔ₂ = %.2e\n", name, d1, d2)
    end
    verbose && @printf("  %-16s δΔ₁ = %.2e   δΔ₂ = %.2e\n", "WORST CASE", spread[1], spread[2])
    return (; Δ=base.Δ, spread=SVector(spread), base=base, table=table)
end

delta_convergence(params::GGJParameters, Q::Number; kwargs...) =
    delta_convergence(params, ComplexF64(Q); kwargs...)
