# Ray.jl
#
# Rotated-contour spectral-element collocation backend (`GGJModel{:ray}`)
# for the GGJ inner-layer matching data — robust at large |Q| on and near
# the imaginary axis, where both `:shooting` (exponential dichotomy) and
# `:galerkin` (real-axis oscillation + on-axis pseudo-resonance) degrade.
#
# Validated against the Fortran rmatch pins, the `:galerkin` backend at
# moderate Q, and physical benchmarks to Q = 500i. Full method write-up:
# docs/src/inner_layer.md, "Numerical method". In one paragraph:
#
#   The equations are continued analytically onto the ray x = e^{iθ}s with
#   θ = arg(Q)/4 (parabolic-cylinder exponent exactly real; pseudo-resonance
#   cleared). On [0, s_m] a global Chebyshev spectral-element collocation BVP
#   in the plain variables v = (Ψ, Ξ, Υ, Ψ', Ξ', Υ') is solved with parity
#   conditions at the (ordinary-point) origin and Δ, c₁, c₂ as bordered
#   unknowns of one sparse LU per parity (rank-3 Woodbury for the second).
#   The far-field condition uses the SAME inps Wasow basis as the other
#   backends (RayAsymptotics.jl evaluates it at complex x; the rdcon
#   normalization of Δ is retained), applied at the series radius S and
#   transported S → s_m by an L-stable 2-stage Radau IIA march in the
#   quotient modulo the decaying exponential pair. The damped-zone march
#   arithmetic runs in Complex{Double64}: at large S the near-parallel
#   power-pair geometry amplifies the structured backward error of the
#   ill-conditioned implicit solves (Σ eps·z ≈ eps·ρS²/2) into Δ-mixing
#   ~1e-4 at |Q| = 500 in Float64 (docs/src/inner_layer.md, "Far-field
#   boundary and the inward march").
#
# File layout:
#   1. system   — plain-variable first-order ODE matrix, parity sets
#   2. mesh     — Chebyshev cells, barycentric interpolation, WKB grading
#   3. boundary — decaying pair, Radau IIA quotient march
#   4. solve    — assembly, bordered solve, refinement, driver, diagnostics

# system.jl
#
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

Coefficient matrix `M(x)` of `dv/dx = M v`, `v = (Ψ, Ξ, Υ, Ψ', Ξ', Υ')`,
valid for real or complex `x`. The optional leading type argument selects
the element type (e.g. `Complex{Double64}` for the extended-precision
damped-zone march); default `ComplexF64`.
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
    parity_rows(isol) -> SVector{3,Int}

Component indices of `v = (Ψ, Ξ, Υ, Ψ', Ξ', Υ')` that vanish at s = 0 for
each parity solution, mirroring deltac_set_boundary:

  - `isol = 1` ("odd"):  Ψ'(0) = 0, Ξ(0) = 0, Υ(0) = 0  → components (4, 2, 3)
  - `isol = 2` ("even"): Ψ(0) = 0, Ξ'(0) = 0, Υ'(0) = 0 → components (1, 5, 6)
"""
parity_rows(isol::Int) = isol == 1 ? SVector(4, 2, 3) : SVector(1, 5, 6)
# mesh.jl
#
# Graded spectral-element mesh on the ray parameter s ∈ [0, S], plus the
# Chebyshev–Lobatto reference-cell machinery (nodes, differentiation matrix,
# barycentric interpolation).
#
# The initial grading tracks the local stiffness scales of the rotated
# system:
#   - the fast Ohm's-law pair e^{±√Q x}: active near the origin, decaying
#     at rate Re(√Q e^{iθ}) — resolved on a fine zone [0, s₁];
#   - the parabolic-cylinder pair e^{±ρ s²/2}, ρ = Re(e^{2iθ}/√Q): decaying
#     content is above relative machine floor only for s ≲ s_d = √(2D/ρ) —
#     resolved on a rate-adapted zone [s₁, s_d];
#   - beyond s_d the physical solution is a smooth power law: geometric
#     (log-spaced) cells suffice out to S.
# Residual-driven bisection refinement (solve.jl) then repairs whatever the
# initial guess missed.

"""
    cheblobatto(p) -> (t, D)

Chebyshev–Lobatto nodes on [-1, 1] in ASCENDING order and the spectral
differentiation matrix `D` for those nodes (Trefethen's `cheb`, reflected).
"""
function cheblobatto(p::Int)
    p >= 1 || throw(ArgumentError("polynomial order p must be ≥ 1"))
    # Standard descending Trefethen nodes x_j = cos(jπ/p), j = 0..p.
    x = [cos(pi * j / p) for j in 0:p]
    c = [j == 0 || j == p ? 2.0 : 1.0 for j in 0:p] .* [(-1.0)^j for j in 0:p]
    D = zeros(p + 1, p + 1)
    for i in 1:(p+1), j in 1:(p+1)
        if i != j
            D[i, j] = (c[i] / c[j]) / (x[i] - x[j])
        end
    end
    for i in 1:(p+1)
        D[i, i] = -sum(D[i, j] for j in 1:(p+1) if j != i)
    end
    # Reflect to ascending order: y = -x is ascending with the SAME index
    # order (x is descending), and for g(y) := f(-y) sampled on y the chain
    # rule gives D_y = -D. No index permutation.
    t = -x
    Dt = -D
    return t, Dt
end

"""
    barycentric_weights(t) -> w

Barycentric interpolation weights for nodes `t` (any distribution).
"""
function barycentric_weights(t::AbstractVector{Float64})
    n = length(t)
    w = ones(Float64, n)
    for j in 1:n
        for k in 1:n
            k == j && continue
            w[j] /= (t[j] - t[k])
        end
    end
    return w
end

"""
    barycentric_eval(t, w, vals, τ) -> value

Evaluate the interpolant of `vals` (matrix: nodes × components) at `τ`.
Returns a vector over components.
"""
function barycentric_eval(t::AbstractVector{Float64}, w::AbstractVector{Float64},
    vals::AbstractMatrix{ComplexF64}, τ::Float64)
    n = length(t)
    for j in 1:n
        if τ == t[j]
            return vals[j, :]
        end
    end
    num = zeros(ComplexF64, size(vals, 2))
    den = 0.0
    for j in 1:n
        fac = w[j] / (τ - t[j])
        den += fac
        @views num .+= fac .* vals[j, :]
    end
    return num ./ den
end

"""
    initial_breaks(params, Q, θ, S, p; cfl=0.4, drop=40.0, ratio=1.3,
                   hmax_frac=0.08) -> Vector{Float64}

Build the initial cell-boundary vector `0 = s₀ < s₁ < … = S` using the
three-zone grading described in the header. `cfl` ≈ resolved-exponential
rate per node; `drop` = e-folds after which decaying exponential content is
considered machine-dead; `ratio` = geometric growth in the outer zone.
"""
function initial_breaks(params::GGJParameters, Q::ComplexF64, θ::Float64,
    S::Float64, p::Int; cfl::Float64=0.4, drop::Float64=40.0,
    ratio::Float64=1.3, hmax_frac::Float64=0.08)
    sq = sqrt(Q)
    ρ = real(cis(2θ) / sq)                 # parabolic pair: exponent ρ s²/2
    ρ > 0 || throw(ArgumentError("contour angle θ=$θ gives non-decaying parabolic pair (ρ=$ρ ≤ 0); use θ ≈ arg(Q)/4"))
    # Fast local WKB scales: the Ohm pair |√Q| and the Υ-family
    # √|Q(G+KF)| (dominant on extreme-coefficient surfaces, K ~ 10⁶ —
    # dormant almost everywhere but must be resolved; see solve.jl header).
    μ = abs(sq) + sqrt(abs(Q) * abs(params.G + params.K * params.F))
    μdec = max(real(sq * cis(θ)), 0.1 * μ) # its decay rate along the ray

    s1 = min(drop / μdec, 0.7 * S)         # fast pair dead beyond s1
    sd = min(sqrt(2 * drop / ρ), 0.8 * S)  # parabolic pair dead beyond sd
    sd = max(sd, min(1.05 * s1, S))
    hmax = hmax_frac * S

    # Single grading rule: every cell resolves the full local WKB rate
    # μ + ρs (fast pair + parabolic pair), whether the corresponding mode
    # amplitudes are alive or dormant — unresolved dormant modes alias into
    # spurious bounded discrete modes and poison the global solve. The
    # outer power-law region is not discretized at all (handled by the
    # boundary march), so this never gets expensive: the BVP domain ends at
    # s_m ~ max(s_d, s₁) where μ dominates the rate budget.
    _ = (s1, sd, ratio)  # zone markers retained in signature for tuning
    breaks = [0.0]
    while breaks[end] < S
        s = breaks[end]
        h = cfl * p / (μ + ρ * s)
        h = min(h, hmax, S / 8)
        snew = s + h
        if snew >= S - 0.25 * h
            snew = S
        end
        push!(breaks, min(snew, S))
    end
    # Guard against a degenerate final sliver.
    if length(breaks) >= 3 && (breaks[end] - breaks[end-1]) < 0.2 * (breaks[end-1] - breaks[end-2])
        deleteat!(breaks, length(breaks) - 1)
    end
    return breaks
end
# boundary.jl
#
# Large-s boundary data at the matching point on the ray x = e^{iθ}s.
#
# The admissible (bounded-as-s→∞) solution space is spanned by
#   { u_small, u_big }  — the two power-like Mercier solutions, taken from
#                         the inps series in the standard normalization, and
#   { E₁, E₂ }          — the two forward-DECAYING exponential solutions.
#
# The inps series is only trusted at radius S (residual ≤ smax_tol), which
# for extreme-coefficient surfaces (K ~ 10⁶) can be S ~ 10⁴–10⁵, while the
# collocation BVP ends at s_m ~ 30–150. The gap is bridged by MARCHING the
# power-pair data W = [u_small, u_big] backward S → s_m.
#
# Marching design (the hard-won part):
#   - Backward, the decaying pair GROWS at rate ρs (ρ = Re(e^{2iθ}/√Q)), up
#     to ~10³/unit at large s. Any explicit integrator — including
#     projected/continuous-QR variants, whose unprojected stage points
#     reintroduce the transverse stiffness — is stability-limited to
#     h ≲ 3/(ρs), i.e. O(ρS²) steps: hopeless at S ~ 10⁵.
#   - The system is LINEAR, so we use a stiffly-accurate L-STABLE implicit
#     method (2-stage Radau IIA, order 3; one 12×12 solve per step). For
#     resolved slow content it is 3rd-order accurate; for the unresolvable
#     backward-growing exponential directions |R(z)| < 1 at large |z| —
#     the method DAMPS what it cannot resolve instead of amplifying it.
#     That is normally an accuracy vice; here it is exactly the required
#     behavior, because Δ is defined modulo the decaying pair.
#   - In the partially-resolved band (|z| = ρs·h ≲ 10) genuine backward
#     growth of rounding-injected decaying-pair content still occurs, so W
#     is PURGED against a locally-extracted decaying frame every
#     Φ = ∫ρs ds ≈ 8 e-folds. Purging only ever removes span{E} content —
#     legitimate by the definition of Δ — so the transported coefficients
#     of u_small/u_big are exact up to integrator tolerance.

"""
    decaying_pair(params, Q, θ, S; growth=30.0, seed=20260707) -> E::Matrix{ComplexF64} (6×2)

Orthonormal basis of the forward-decaying pair at s = S on the ray, via a
SHORT backward RK4 integration over [S, S+ΔS], with ΔS chosen so the pair is
amplified by ≈ e^growth relative to everything else (backward-attracting
purification of random seeds). The integration is fully resolved
(h·rate ≈ 0.05), so plain RK4 is stable over this short stretch.
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
which the matching data Δ is defined). Adaptive 2-stage Radau IIA (L-stable,
stiffly accurate) on the raw linear system, with Φ-budgeted purges against
locally-extracted decaying frames; see the file header for why explicit
(including projected-QR) marchers cannot do this job at large S.
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

    I6 = Matrix{ComplexF64}(I, 6, 6)
    # One Radau IIA step: A = [5/12 -1/12; 3/4 1/4], c = (1/3, 1), b = A[2,:].
    function radau_step(s, W, h)
        M1 = 𝔐(s + h / 3)
        M2 = 𝔐(s + h)
        Ablk = [I6-(5h/12)*M1 (h/12)*M1; -(3h/4)*M2 I6-(h/4)*M2]
        rhs = [M1 * W; M2 * W]
        K = Ablk \ rhs
        return W + h * ((3 / 4) * @view(K[1:6, :]) + (1 / 4) * @view(K[7:12, :]))
    end

    # Extended-precision (Double64) Radau step for the damped-zone W march.
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
    I6d = Matrix{CTd}(I, 6, 6)
    𝔐d(s) = phd * Matrix(ode_matrix(CTd, params, Q, phd * Double64(s)))
    function radau_step_d(s, Wd, h)
        hd = Double64(h)
        M1 = 𝔐d(s + h / 3)
        M2 = 𝔐d(s + h)
        Ablk = [I6d-(5hd/12)*M1 (hd/12)*M1; -(3hd/4)*M2 I6d-(hd/4)*M2]
        rhs = [M1 * Wd; M2 * Wd]
        K = Ablk \ rhs
        return Wd + hd * ((CTd(3) / 4) * K[1:6, :] + (CTd(1) / 4) * K[7:12, :])
    end

    # --- Schedule: geometric where damped, growth-capped where resolved. ---
    # No adaptive controller (three generations died on estimator artifacts);
    # the step law is derived from the local mode structure instead. At each
    # step the eigenvalues of 𝔐(s) give the two backward-growing exponential
    # rates g (Re λ < 0, excluding the algebraic power-pair rates ~ r/s):
    #
    #   damped zone (g_min·c·s > 8 for all growers): h = -c·s. All growing
    #     content — including the O(smax_tol) fast-direction contamination of
    #     the series data — sits at z ≫ 1 where L-stable Radau DAMPS it
    #     (|R| ~ 6/z²). Slow-mode global error ≈ 10²c³ → c = (rtol/100)^{1/3};
    #     step count ~ ln(S/s_m)/c independent of S.
    #
    #   resolved band (some grower at z ≲ 8): genuine backward growth must be
    #     tracked and removed. h = -min(c·s, 0.3/g_max) keeps z ≤ 0.3 so the
    #     DISCRETE growing eigendirection matches the continuous one to
    #     O(z²) ~ 1% (an orthogonal purge against a frame misaligned by δ
    #     leaks δ·content — with e-fold budget Φ ≤ 2.5 the cycle gain
    #     e^Φ·δ < 0.2 and contamination stays at the rounding floor). The
    #     frame F is carried CONTINUOUSLY with the same Radau steps + QR
    #     (backward-attracting ⇒ self-correcting), so no re-extraction cost.
    c = clamp(cbrt(rtol / 100.0), 2e-5, 2e-3)

    # --- Phase 1: damped zone, W in Double64 (see radau_step_d comment). ---
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
        Wd = radau_step_d(s, Wd, h)
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
        W = radau_step(s, W, h)
        if have_F
            F = radau_step(s, F, h)
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
    xS = cis(θ) * S
    ua, dua = physical_ua_dua(cache, xS)
    Ub = ComplexF64[ua[1, 1], ua[2, 1], ua[3, 1], dua[1, 1], dua[2, 1], dua[3, 1]]
    Us = ComplexF64[ua[1, 2], ua[2, 2], ua[3, 2], dua[1, 2], dua[2, 2], dua[3, 2]]
    E = decaying_pair(params, Q, θ, S; growth=growth, seed=seed, dp_res=dp_res)

    Bmat = hcat(Us ./ norm(Us), Ub ./ norm(Ub), E)
    cond_est = cond(Bmat)
    return Us, Ub, E, cond_est
end
# solve.jl
#
# Global spectral-element collocation solve of the GGJ inner-layer BVP on
# the rotated ray, with the matching data Δ as a bordered unknown.
#
# For each parity (deltac isol = 1 "odd", 2 "even"):
#
#   unknowns : v at all global nodes (6 components each), plus (Δ, c₁, c₂)
#   equations: dv/ds = e^{iθ} M(e^{iθ}s) v   collocated at the p non-left
#              Lobatto nodes of every cell (right-biased collocation —
#              Radau-IIA-like, stable for the dormant stiff modes);
#              3 parity conditions at s = 0;
#              6 matching conditions at s = S:
#                  v(S) − Δ·Ub − c₁E₁ − c₂E₂ = Us .
#
# One sparse LU per parity per mesh; residual-driven bisection refinement.
# The exponential dichotomy never enters: no quantity is ever propagated
# across the layer, and the boundary condition splits the dichotomy exactly.

"""
    RaySolveResult

Output of [`solve_ray`](@ref). `Δ` is the rescaled matching data in the
deltac output ordering (index 1 ↔ even-BC solution, 2 ↔ odd-BC solution,
i.e. the same swap deltac_solve applies); `Δraw` is (isol=1, isol=2) in raw
inps normalization before `rescale_delta`.
"""
struct RaySolveResult
    Δ::SVector{2,ComplexF64}
    Δraw::SVector{2,ComplexF64}
    cexp::SMatrix{2,2,ComplexF64,4}   # decaying-pair coefficients (col = isol)
    Q::ComplexF64
    θ::Float64
    S::Float64
    series_ok::Bool                    # pick_smax reached its tolerance
    bc_cond::Float64                   # boundary-basis condition number
    breaks::Vector{Float64}
    p::Int
    sols::Matrix{ComplexF64}           # ndof × 2 nodal solution (isol = 1, 2)
    resid::Float64                     # final max relative collocation residual
    δΔ_refine::Float64                 # |Δ change| in last refinement round (rel)
    nrounds::Int
end

# -----------------------------------------------------------------------
# Assembly.
# -----------------------------------------------------------------------

function _assemble_base(params::GGJParameters, Q::ComplexF64, θ::Float64,
    breaks::Vector{Float64}, p::Int,
    t::Vector{Float64}, D::Matrix{Float64},
    Us::Vector{ComplexF64}, Ub::Vector{ComplexF64}, E::Matrix{ComplexF64})
    N = length(breaks) - 1
    Ng = N * p + 1
    ndof = 6 * Ng + 3
    ph = cis(θ)
    βb = norm(Ub)

    nnz_est = N * p * 6 * (p + 7) + 24 + 3
    Ic = Vector{Int}()
    Jc = Vector{Int}()
    Vc = Vector{ComplexF64}()
    sizehint!(Ic, nnz_est)
    sizehint!(Jc, nnz_est)
    sizehint!(Vc, nnz_est)

    @inbounds for c in 1:N
        a, b = breaks[c], breaks[c+1]
        Dscale = 2 / (b - a)
        for j in 1:p
            s_j = a + (b - a) * (t[j+1] + 1) / 2
            M = ph * ode_matrix(params, Q, ph * s_j)
            row0 = 6 * ((c - 1) * p + (j - 1))
            gj = (c - 1) * p + j + 1
            for i in 1:6
                row = row0 + i
                for k in 0:p
                    gk = (c - 1) * p + k + 1
                    push!(Ic, row)
                    push!(Jc, 6 * (gk - 1) + i)
                    push!(Vc, Dscale * D[j+1, k+1])
                end
                for i2 in 1:6
                    Mv = M[i, i2]
                    if Mv != 0
                        push!(Ic, row)
                        push!(Jc, 6 * (gj - 1) + i2)
                        push!(Vc, -Mv)
                    end
                end
            end
        end
    end

    # Matching rows at s = S.
    rowm0 = 6 * N * p
    @inbounds for i in 1:6
        row = rowm0 + i
        push!(Ic, row)
        push!(Jc, 6 * (Ng - 1) + i)
        push!(Vc, one(ComplexF64))
        push!(Ic, row)
        push!(Jc, 6 * Ng + 1)
        push!(Vc, -Ub[i] / βb)
        push!(Ic, row)
        push!(Jc, 6 * Ng + 2)
        push!(Vc, -E[i, 1])
        push!(Ic, row)
        push!(Jc, 6 * Ng + 3)
        push!(Vc, -E[i, 2])
    end

    rhs = zeros(ComplexF64, ndof)
    rhs[rowm0.+(1:6)] .= Us

    return Ic, Jc, Vc, rhs, βb, ndof, Ng
end

"""
    _solve_parities(Ic, Jc, Vc, rhs, ndof, N, p) -> (sol_odd, sol_even)

Solve BOTH parity systems from one factorization. The two matrices differ
only in the 3 parity rows at s = 0 (odd: unit entries at components (4,2,3);
even: at (1,5,6)), i.e. A_even = A_odd + Σₖ e_{rₖ}(e_{c2ₖ} − e_{c1ₖ})ᵀ — a
rank-3 update. One sparse LU (of the row-equilibrated odd system) plus a
Woodbury correction (3 extra triangular solves + a 3×3 solve) replaces the
second factorization, which dominates the linear-algebra cost.
"""
function _solve_parities(Ic::Vector{Int}, Jc::Vector{Int}, Vc::Vector{ComplexF64},
    rhs::Vector{ComplexF64}, ndof::Int, N::Int, p::Int)
    rowp0 = 6 * N * p + 6
    c1 = parity_rows(1)
    c2 = parity_rows(2)
    I2 = copy(Ic)
    J2 = copy(Jc)
    V2 = copy(Vc)
    for k in 1:3
        push!(I2, rowp0 + k)
        push!(J2, c1[k])
        push!(V2, one(ComplexF64))
    end
    A = sparse(I2, J2, V2, ndof, ndof)

    # Row equilibration.
    rmax = zeros(Float64, ndof)
    rows = rowvals(A)
    vals = nonzeros(A)
    for col in 1:ndof
        for idx in nzrange(A, col)
            r = rows[idx]
            a = abs(vals[idx])
            a > rmax[r] && (rmax[r] = a)
        end
    end
    @inbounds for i in 1:ndof
        rmax[i] = rmax[i] > 0 ? 1 / rmax[i] : 1.0
    end
    Dr = Diagonal(rmax)
    F = lu(Dr * A)
    x1 = F \ (Dr * rhs)

    # Woodbury: (DrA + U Vᵀ)⁻¹ b = x1 − A⁻¹U (I₃ + Vᵀ A⁻¹U)⁻¹ Vᵀ x1,
    # with U[:,k] = Dr[rₖ,rₖ]·e_{rₖ} and Vᵀ[k,:] = (e_{c2ₖ} − e_{c1ₖ})ᵀ.
    U = zeros(ComplexF64, ndof, 3)
    for k in 1:3
        U[rowp0+k, k] = rmax[rowp0+k]
    end
    AinvU = F \ U
    Vx1 = ComplexF64[x1[c2[k]] - x1[c1[k]] for k in 1:3]
    Smat = Matrix{ComplexF64}(I, 3, 3)
    for k in 1:3, j in 1:3
        Smat[k, j] += AinvU[c2[k], j] - AinvU[c1[k], j]
    end
    x2 = x1 - AinvU * (Smat \ Vx1)
    # UMFPACK numeric factorizations live in C malloc, invisible to the GC's
    # heap accounting — under the refinement loop, lazily-finalized multi-GB
    # factorizations pile up (observed: tens of GB on 24k-cell meshes). Free
    # eagerly now that both solutions are extracted.
    finalize(F)
    return x1, x2
end

# -----------------------------------------------------------------------
# Residual estimation for adaptive refinement.
# -----------------------------------------------------------------------

function _cell_errors(params::GGJParameters, Q::ComplexF64, θ::Float64,
    breaks::Vector{Float64}, p::Int,
    t::Vector{Float64}, D::Matrix{Float64}, w::Vector{Float64},
    sol::Vector{ComplexF64})
    N = length(breaks) - 1
    ph = cis(θ)
    errs = zeros(Float64, N)

    # Reference-cell test points: left endpoint + inter-node midpoints.
    τs = Float64[-1.0]
    for j in 1:p
        push!(τs, (t[j] + t[j+1]) / 2)
    end

    vals = Matrix{ComplexF64}(undef, p + 1, 6)
    for c in 1:N
        a, b = breaks[c], breaks[c+1]
        Dscale = 2 / (b - a)
        for j in 0:p
            g = (c - 1) * p + j + 1
            for i in 1:6
                vals[j+1, i] = sol[6*(g-1)+i]
            end
        end
        dvals = (Dscale .* D) * vals

        emax = 0.0
        for τ in τs
            vτ = barycentric_eval(t, w, vals, τ)
            dvτ = barycentric_eval(t, w, dvals, τ)
            s_τ = a + (b - a) * (τ + 1) / 2
            Mv = (ph * ode_matrix(params, Q, ph * s_τ)) * vτ
            rnorm = 0.0
            scale = 1e-300
            for i in 1:6
                rnorm = max(rnorm, abs(dvτ[i] - Mv[i]))
                scale = max(scale, abs(dvτ[i]), abs(Mv[i]))
            end
            emax = max(emax, rnorm / scale)
        end
        errs[c] = emax
    end
    return errs
end

# -----------------------------------------------------------------------
# Driver.
# -----------------------------------------------------------------------

"""
    solve_ray(params::GGJParameters, Q; θ=angle(Q)/4, kmax=12, p=12,
              S=nothing, smax_tol=1e-9, growth=30.0,
              refine_tol=1e-8, max_rounds=8, max_cells=6000,
              verbose=false) -> RaySolveResult

Solve the GGJ inner-layer matching problem on the rotated ray x = e^{iθ}s
and return the parity matching data. `Δ` follows the deltac output
convention (swap + `rescale_delta`) so it is directly comparable to the
GPEC_julia Galerkin solver.

Keyword highlights:
  - `θ`     : contour angle; default arg(Q)/4 makes the parabolic exponent
              exactly real. θ = 0 reproduces a real-axis solve.
  - `S`     : matching radius; default from the inps series residual
              criterion along the ray (`pick_smax`).
  - `p`     : polynomial order per spectral element.
  - `refine_tol`/`max_rounds`: residual-driven bisection refinement control.
"""
function solve_ray(params::GGJParameters, Q::ComplexF64;
    θ::Float64=angle(Q) / 4, kmax::Int=12, p::Int=12,
    S::Union{Nothing,Float64}=nothing, smax_tol::Float64=1e-9,
    growth::Float64=30.0, refine_tol::Float64=1e-8,
    march_rtol::Float64=1e-9, sm_fac::Float64=1.0, cfl::Float64=0.4,
    max_rounds::Int=8, max_cells::Int=6000, seed::Int=20260707,
    ztrk::Float64=0.3, purge_budget::Float64=2.5, dp_res::Float64=20.0,
    handoff::Union{Nothing,NTuple{2,Vector{ComplexF64}}}=nothing,
    boundary::Union{Nothing,Tuple{Vector{ComplexF64},Vector{ComplexF64},Matrix{ComplexF64}}}=nothing,
    verbose::Bool=false)

    cache = build_asymptotics(params, Q; kmax=kmax)
    series_ok = true
    if S === nothing
        Sv, cache, series_ok = pick_smax(params, Q; θ=θ, eps=smax_tol, kmax=kmax, cache=cache)
        series_ok || @warn "pick_smax: series residual target $smax_tol not reached; using best point S=$Sv"
    else
        Sv = S
    end

    # Decide the BVP outer radius: the layer physics (exponential content
    # above machine floor) ends at s_d; beyond that the solution is pure
    # power law and is handled by the projected quotient MARCH, not by
    # collocation. For small |Q| (s_d ≳ S) no march is needed.
    sq = sqrt(Q)
    ρ = real(cis(2θ) / sq)
    # BVP outer radius s_m: where every exponential solution family has
    # decayed ≥ 45 e-folds from the origin, computed from the actual
    # frozen-coefficient spectrum (the asymptotic ρs rate is wrong near the
    # origin for extreme-coefficient surfaces, where the Υ-family rate
    # √|Q(G+KF)| dominates and kills exponential content within s ~ O(1)).
    s_m = let acc = 0.0, sprev = 0.0, out = Sv
        for sk in exp10.(range(-2, log10(Sv), length=400))
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
    if boundary !== nothing
        # diagnostic: externally supplied boundary data (Us, Ub, E) at s_m,
        # e.g. from an extended-precision march
        Us, Ub, E = boundary
        Sbvp = s_m
        bc_cond = cond(hcat(Us ./ norm(Us), Ub ./ norm(Ub), E))
    elseif s_m < 0.98 * Sv
        local UsS, UbS
        if handoff === nothing
            ua, dua = physical_ua_dua(cache, cis(θ) * Sv)
            UbS = ComplexF64[ua[1, 1], ua[2, 1], ua[3, 1], dua[1, 1], dua[2, 1], dua[3, 1]]
            UsS = ComplexF64[ua[1, 2], ua[2, 2], ua[3, 2], dua[1, 2], dua[2, 2], dua[3, 2]]
        else
            # diagnostic: externally supplied series hand-off (Us, Ub) at
            # x = e^{iθ}S, e.g. evaluated in extended precision
            UsS, UbS = handoff
        end
        Us, Ub, E = march_boundary(params, Q, θ, Sv, s_m, UsS, UbS;
            rtol=march_rtol, growth=growth, seed=seed,
            ztrk=ztrk, purge_budget=purge_budget, dp_res=dp_res)
        Sbvp = s_m
        bc_cond = cond(hcat(Us ./ norm(Us), Ub ./ norm(Ub), E))
        verbose && @printf("  march: S=%.4g → s_m=%.4g, bc_cond=%.1e\n", Sv, s_m, bc_cond)
    else
        Us, Ub, E, bc_cond = boundary_basis(params, Q, cache, θ, Sv;
            growth=growth, seed=seed, dp_res=dp_res)
        Sbvp = Sv
    end
    bc_cond < 1e12 || @warn "boundary basis poorly conditioned (cond ≈ $bc_cond)"

    t, D = cheblobatto(p)
    w = barycentric_weights(t)
    breaks = initial_breaks(params, Q, θ, Sbvp, p; cfl=cfl)

    Δraw = MVector{2,ComplexF64}(0, 0)
    Δprev = MVector{2,ComplexF64}(NaN, NaN)
    cexp = zeros(MMatrix{2,2,ComplexF64})
    local sols
    resid = Inf
    resid_prev = Inf
    nstall = 0
    δΔ = Inf
    round_used = 0

    for round in 0:max_rounds
        round_used = round
        N = length(breaks) - 1
        Ic, Jc, Vc, rhs, βb, ndof, Ng = _assemble_base(params, Q, θ, breaks, p, t, D, Us, Ub, E)
        sols = Matrix{ComplexF64}(undef, ndof, 2)
        errs = zeros(Float64, N)
        sol_odd, sol_even = _solve_parities(Ic, Jc, Vc, rhs, ndof, N, p)
        for (isol, sol) in ((1, sol_odd), (2, sol_even))
            sols[:, isol] = sol
            Δraw[isol] = sol[6*Ng+1] / βb
            cexp[1, isol] = sol[6*Ng+2]
            cexp[2, isol] = sol[6*Ng+3]
            errs = max.(errs, _cell_errors(params, Q, θ, breaks, p, t, D, w, sol))
        end
        resid = maximum(errs)
        δΔ = maximum(abs.(Δraw .- Δprev) ./ (abs.(Δraw) .+ 1e-300))
        Δprev .= Δraw
        verbose && @printf("  round %d: %d cells, resid %.2e, Δraw[1] = %.6e %+.6eim, δΔ = %.1e\n",
            round, N, resid, real(Δraw[1]), imag(Δraw[1]), δΔ)

        (resid < refine_tol || round == max_rounds) && break
        # Stagnation guard: on extreme-coefficient surfaces the residual
        # bottoms out at the roundoff floor ε·‖M‖ (~10⁻⁸ ≫ refine_tol);
        # refining past that plateau burns cells without informing Δ. Two
        # consecutive rounds with < 2× improvement ⇒ converged-to-plateau.
        nstall = resid > 0.5 * resid_prev ? nstall + 1 : 0
        resid_prev = resid
        if nstall >= 2
            verbose && @printf("  refinement stagnated at resid %.2e (roundoff plateau) — stopping\n", resid)
            break
        end
        marked = findall(>(refine_tol), errs)
        isempty(marked) && break
        if N + length(marked) > max_cells
            @warn "solve_ray: cell budget ($max_cells) reached with resid=$resid"
            break
        end
        newbreaks = Float64[breaks[1]]
        for c in 1:N
            if c in marked
                push!(newbreaks, (breaks[c] + breaks[c+1]) / 2)
            end
            push!(newbreaks, breaks[c+1])
        end
        breaks = newbreaks
    end

    Δswapped = SVector{2,ComplexF64}(Δraw[2], Δraw[1])   # deltac_solve convention
    Δ = rescale_delta(Δswapped, params)

    return RaySolveResult(Δ, SVector(Δraw), SMatrix(cexp), Q, θ, Sv, series_ok,
        bc_cond, breaks, p, sols, resid, δΔ, round_used)
end

solve_ray(params::GGJParameters, Q::Number; kwargs...) =
    solve_ray(params, ComplexF64(Q); kwargs...)

"""
    solve_inner(::GGJModel{:ray}, params::GGJParameters, γ::Number; kwargs...)
        -> SVector{2,ComplexF64}

Solve the GGJ inner-layer matching problem with the rotated-ray collocation
backend and return the parity-projected matching data `(Δ₁, Δ₂)` in the same
convention as the `:shooting` and `:galerkin` backends (deltac output swap +
`X₀^{2√(−D_I)}` physical rescale applied), directly interchangeable with
them. Converts γ via `inner_Q` and forwards all keywords to
[`solve_ray`](@ref); use `solve_ray` directly when the full
[`RaySolveResult`](@ref) (raw Δ, contour, mesh, nodal solution, residuals)
is wanted.

Preferred backend for |Q| ≳ 1 and for Q near the imaginary axis (rotation /
resistivity scans): validated to |Q| = 500 on the imaginary axis, where the
other backends fail. At |Q| ≪ 1 all three backends agree; `:ray` remains
accurate but `:galerkin` may be faster for very small |Q| real Q.
"""
function solve_inner(::GGJModel{:ray}, params::GGJParameters, γ::Number; kwargs...)
    res = solve_ray(params, inner_Q(params, γ); kwargs...)
    return res.Δ
end

"""
    evaluate_solution(res::RaySolveResult, s::Real; isol=1) -> SVector{6,ComplexF64}

Evaluate the solved state v = (Ψ, Ξ, Υ, Ψ', Ξ', Υ') at ray parameter `s`
(diagnostic; barycentric interpolation on the containing cell).
"""
function evaluate_solution(res::RaySolveResult, s::Real; isol::Int=1)
    breaks = res.breaks
    p = res.p
    (breaks[1] <= s <= breaks[end]) || throw(ArgumentError("s=$s outside [0, $(breaks[end])]"))
    c = clamp(searchsortedlast(breaks, s), 1, length(breaks) - 1)
    a, b = breaks[c], breaks[c+1]
    t, _ = cheblobatto(p)
    w = barycentric_weights(t)
    vals = Matrix{ComplexF64}(undef, p + 1, 6)
    for j in 0:p
        g = (c - 1) * p + j + 1
        for i in 1:6
            vals[j+1, i] = res.sols[6*(g-1)+i, isol]
        end
    end
    τ = 2 * (s - a) / (b - a) - 1
    return SVector{6,ComplexF64}(barycentric_eval(t, w, vals, τ)...)
end

"""
    solution_profile(res::RaySolveResult; npc=8) -> (; s, x, Ψ, Ξ, Υ)

Reconstruct the physical inner-layer fields (Ψ, Ξ, Υ) on the solution
contour, `npc` points per cell over the collocation domain [0, s_m].
Output layout mirrors the legacy `solve_inner_profile`: each field is an
`npts × 2` matrix with columns (isol=1 "odd", isol=2 "even"); `x = e^{iθ}s`
is the (complex) layer coordinate. For real Q (θ = 0) this is the real-axis
profile directly; for rotated solves the fields live on the ray — analytic
continuation back to real x is a separate (unstable) problem, by design.
"""
function solution_profile(res::RaySolveResult; npc::Int=8)
    breaks = res.breaks
    p = res.p
    t, _ = cheblobatto(p)
    w = barycentric_weights(t)
    N = length(breaks) - 1
    npts = N * npc + 1
    s = Vector{Float64}(undef, npts)
    Ψ = Matrix{ComplexF64}(undef, npts, 2)
    Ξ = Matrix{ComplexF64}(undef, npts, 2)
    Υ = Matrix{ComplexF64}(undef, npts, 2)
    vals = Matrix{ComplexF64}(undef, p + 1, 6)
    k = 0
    for c in 1:N
        a, b = breaks[c], breaks[c+1]
        for m in 0:(npc-1)
            k += 1
            τ = -1 + 2 * m / npc
            s[k] = a + (b - a) * (τ + 1) / 2
            for isol in 1:2
                for j in 0:p, i in 1:6
                    vals[j+1, i] = res.sols[6*((c-1)*p+j)+i, isol]
                end
                v = barycentric_eval(t, w, vals, τ)
                Ψ[k, isol] = v[1]
                Ξ[k, isol] = v[2]
                Υ[k, isol] = v[3]
            end
        end
    end
    # right endpoint
    k += 1
    s[k] = breaks[end]
    for isol in 1:2
        v = evaluate_solution(res, breaks[end]; isol=isol)
        Ψ[k, isol] = v[1]
        Ξ[k, isol] = v[2]
        Υ[k, isol] = v[3]
    end
    return (; s=s, x=cis(res.θ) .* s, Ψ=Ψ, Ξ=Ξ, Υ=Υ)
end

"""
    asymptotic_profile(params, res::RaySolveResult, srange) -> (; s, x, Ψ, Ξ, Υ)

Evaluate the ANALYTIC representation `u_small + Δ·u_big` on the ray for
`s ≥ res.S` (where the inps series is trusted; decaying-exponential content
is machine-dead there). Concatenating with [`solution_profile`](@ref)
demonstrates the seamless numeric ↔ asymptotic match — the same overlap the
outer-region matching relies on.
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
    delta_convergence(params, Q; verbose=true, refine_tol=1e-9,
                      max_rounds=16, max_cells=12000, kwargs...)
        -> (; Δ, spread, base, table)

Convergence battery for the matching data: solve at a trusted baseline, then
re-solve with every numerical knob perturbed on an INDEPENDENT axis, and
report the relative change of (Δ₁, Δ₂) per knob. The knobs probe orthogonal
error sources, so the worst-case `spread` is an honest error bar on Δ:

  - `θ → 1.2θ`       : contour angle — Δ is an analytic invariant of the ray,
                       so any drift is pure numerical error (sharpest test).
                       OUTWARD: the natural angle maximizes clearance from the
                       x² ≈ −Q²(G+KF) pseudo-resonance, so inward checks
                       measure their own degraded solve, not the baseline;
  - `p → p+4`        : spectral-element order (interior discretization);
  - `kmax+4, smax_tol/10` : series order and matching radius S (far-field BC);
  - `refine_tol/10`  : mesh refinement depth;
  - `march_rtol/100` : outer-region quotient-march tolerance;
  - `sm_fac 1→1.4`   : BVP/march handoff radius;
  - `growth 30→45`   : decaying-pair purification e-folds.

Returns the baseline `Δ`, the per-parity worst-case relative `spread`, the
baseline result struct, and the per-knob table `(name, δΔ₁, δΔ₂)`.
"""
function delta_convergence(params::GGJParameters, Q::ComplexF64;
    verbose::Bool=true, refine_tol::Float64=1e-9,
    max_rounds::Int=16, max_cells::Int=12000, kwargs...)
    base = solve_ray(params, Q; refine_tol=refine_tol, max_rounds=max_rounds,
        max_cells=max_cells, kwargs...)
    variations = [
        ("θ → 1.2θ", (; θ=(base.θ == 0 ? 0.1 : 1.2 * base.θ))),
        ("p → p+4", (; p=base.p + 4)),
        ("kmax+4, S↑", (; kmax=16, smax_tol=1e-10)),
        ("refine_tol/10", (; refine_tol=refine_tol / 10)),
        ("march_rtol/100", (; march_rtol=1e-11)),
        ("sm_fac 1→1.4", (; sm_fac=1.4)),
        ("growth 30→45", (; growth=45.0)),
    ]
    table = NamedTuple[]
    spread = MVector{2,Float64}(0.0, 0.0)
    verbose && @printf("Δ convergence battery, Q = %s:\n", string(Q))
    verbose && @printf("  baseline: Δ₁ = %+.8e %+.8eim, Δ₂ = %+.8e %+.8eim\n",
        real(base.Δ[1]), imag(base.Δ[1]), real(base.Δ[2]), imag(base.Δ[2]))
    for (name, kw) in variations
        r = solve_ray(params, Q; refine_tol=refine_tol, max_rounds=max_rounds,
            max_cells=max_cells, kwargs..., kw...)
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

