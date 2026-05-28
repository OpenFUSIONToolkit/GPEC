# Asymptotics.jl
#
# Generalized Wasow asymptotic-basis construction for the shared Wasow–Galerkin
# engine. Refactor of the GGJ-specific `GGJ/InnerAsymptotics.jl`: all sizes are
# values (system order `n`, algebraic dimension `n_alg`, series length `nterms`)
# rather than `SMatrix{6,6}` type literals, the hand-derived closed-form Lyapunov
# solve is replaced by a general block-Sylvester solve, and the closed-form 2×2
# Vandermonde inverse is replaced by a general LU inverse.
#
# Reference: A. H. Glasser & Z. R. Wang, Phys. Plasmas 27, 012506 (2020),
# Sections II.A–II.G (Eqs. 4–53). Notation map:
#
#   A₀/A₁/…   -> Eqs. 4–5   (physical-system coefficient matrices, supplied by spec)
#   T, Tinv   -> Eqs. 7–8   (block-diagonalizing basis of A₀, supplied by spec)
#   Jᵢ        -> Eqs. 9–10  (Jᵢ = T⁻¹ Aᵢ T)
#   P_k, B_k  -> Eqs. 16,22 (splitting transformation; block-Sylvester solve)
#   Q_k, C_k  -> Eqs. 32–39 (algebraic-block companion diagonalization)
#   D_k       -> Eq. 43     (shearing transformation)
#   Y0, R     -> Eq. 49     (lowest-order solution and Frobenius exponents)
#   Z_k       -> Eq. 52     (Y-series in shifted-exponent form)
#   U(x)      -> Eq. 53     (final asymptotic solution at large x)

using LinearAlgebra

"""
    WasowCache{RT}

Frozen state of the generalized Wasow asymptotic-basis construction for a single
`(model, Q)` pair, with Frobenius-exponent element type `RT`. All matrices are
stored as dense `Matrix`/`Vector` so the order is a runtime value.

Index convention: `P[k+1]` holds the k-th-order matrix `P_k`, etc.

Fields:

  - `Q, kmax` — growth-rate parameter and series truncation order.
  - `n, n_alg, nterms` — system size, algebraic-block size, number of `A` matrices.
  - `alg_idx, exp_idx` — algebraic / exponential index partition of `1:n`.
  - `R` — Frobenius exponents at infinity (length `n_alg`).
  - `T, Tinv` — `n×n` block-diagonalizing basis of `A₀` and its inverse.
  - `J` — `(J₀, …, J_{nterms-1})`, the rotated coefficient matrices.
  - `P, B` — splitting matrices, k = 0..kmax+2.
  - `Qmat` — `n_alg×n_alg` companion matrices, k = 0..kmax+2.
  - `Y0, Y0inv` — lowest-order `n_alg×n_alg` Y matrix and its inverse.
  - `Y, Z` — `n_alg×n_alg` series matrices, k = 0..kmax.
  - `spow` — shear powers (in the `xfac = x⁻²` variable) of the `n_alg` algebraic
    columns, defining the diagonal scaling `S(x) = diag(xfac^spow)`.
  - `cc, dd, rvec` — `x`-independent Horner-coefficient packings of `Y` and of
    `(Qmat, P[exp,alg])`, and the fractional-power exponents `-R[j]/2`,
    precomputed once so [`evaluate_wasow`](@ref) does not rebuild them per call.
"""
struct WasowCache{RT<:Number}
    Q::ComplexF64
    kmax::Int
    n::Int
    n_alg::Int
    nterms::Int
    alg_idx::Vector{Int}
    exp_idx::Vector{Int}
    R::Vector{RT}
    T::Matrix{ComplexF64}
    Tinv::Matrix{ComplexF64}
    J::Vector{Matrix{ComplexF64}}
    P::Vector{Matrix{ComplexF64}}
    B::Vector{Matrix{ComplexF64}}
    Qmat::Vector{Matrix{ComplexF64}}
    Y0::Matrix{ComplexF64}
    Y0inv::Matrix{ComplexF64}
    Y::Vector{Matrix{ComplexF64}}
    Z::Vector{Matrix{ComplexF64}}
    spow::Vector{Int}
    cc::Matrix{ComplexF64}
    dd::Matrix{ComplexF64}
    rvec::Vector{RT}
end

# -----------------------------------------------------------------------
# Block-Sylvester split (generalizes GGJ `_lyap_solve`).
#
# Given the current k-th coefficient matrix `K` (n×n) and the leading rotated
# matrix `J₀`, return `(B, P)` where:
#   - `B` is block-diagonal in the (alg, exp) split: `B = diag(K_aa, K_ee)`.
#   - `P` has zero diagonal blocks; its off-diagonal blocks solve the Sylvester
#     equations that block-diagonalize `J₀ + …` to this order. Because the
#     algebraic block of `J₀` has eigenvalues ≈ 0 and the exponential block has
#     nonzero eigenvalues, the two blocks have disjoint spectra and the Sylvester
#     solves are well-posed.
#
# Sylvester convention (LinearAlgebra.sylvester solves `A X + X B + C = 0`):
#   P_ae solves  J₀_aa P_ae − P_ae J₀_ee = −K_ae   ⇒ sylvester(J₀_aa, −J₀_ee, K_ae)
#   P_ea solves  J₀_ee P_ea − P_ea J₀_aa = −K_ea   ⇒ sylvester(J₀_ee, −J₀_aa, K_ea)
# -----------------------------------------------------------------------

function _sylvester_split(K::AbstractMatrix{ComplexF64}, J0_aa::Matrix{ComplexF64},
    J0_ee::Matrix{ComplexF64}, alg::Vector{Int}, exp::Vector{Int}, n::Int)
    B = zeros(ComplexF64, n, n)
    P = zeros(ComplexF64, n, n)
    @views begin
        B[alg, alg] .= K[alg, alg]
        B[exp, exp] .= K[exp, exp]
        K_ae = Matrix(K[alg, exp])
        K_ea = Matrix(K[exp, alg])
        P[alg, exp] .= sylvester(J0_aa, -J0_ee, K_ae)
        P[exp, alg] .= sylvester(J0_ee, -J0_aa, K_ea)
    end
    return B, P
end

# -----------------------------------------------------------------------
# Split recurrence (generalizes GGJ `_split_recurrence`).
# Builds B_k, P_k for k = 0..kmax+2 with an `nterms`-length series.
# -----------------------------------------------------------------------

function _split_recurrence(J::Vector{Matrix{ComplexF64}}, nterms::Int,
    alg::Vector{Int}, exp::Vector{Int}, n::Int, kmax::Int)
    N = kmax + 3
    P = Vector{Matrix{ComplexF64}}(undef, N)
    B = Vector{Matrix{ComplexF64}}(undef, N)

    J0 = J[1]
    J0_aa = Matrix(@view J0[alg, alg])
    J0_ee = Matrix(@view J0[exp, exp])

    P[1] = Matrix{ComplexF64}(I, n, n)   # P_0 = I
    B[1] = copy(J0)                       # B_0 = J_0

    for k in 1:(kmax+2)
        # K_k starts as J_k (only the first `nterms-1` orders contribute).
        Kacc = (k <= nterms - 1) ? copy(J[k+1]) : zeros(ComplexF64, n, n)
        if k > 1
            Kacc .+= ComplexF64(2 * (k - 1)) .* P[k]
        end
        for l in 1:(k-1)
            kml = k - l
            if kml <= nterms - 1
                Kacc .+= J[kml+1] * P[l+1]
            end
            Kacc .-= P[l+1] * B[kml+1]
        end
        Bk, Pk = _sylvester_split(Kacc, J0_aa, J0_ee, alg, exp, n)
        B[k+1] = Bk
        P[k+1] = Pk
    end
    return P, B
end

# -----------------------------------------------------------------------
# Algebraic-block coefficients (generalizes GGJ `_coefs`).
#
# Builds the companion matrices Q_k, C_k, the shearing D_k, and the Y/Z series of
# the n_alg-dimensional algebraic block from its B_k[alg,alg] sub-blocks and the
# Frobenius exponents R. The leading Vandermonde Y0 and its inverse are general
# (LU-based) over n_alg.
#
# NOTE (Phase-4 extension point): the companion construction of Q_k, C_k, D_k and
# the shear powers below implement the single Jordan-degenerate algebraic pair of
# the GGJ system (n_alg = 2) — the floor that any generalization must reproduce.
# A higher-order layer with additional / confluent algebraic exponents needs the
# confluent-Vandermonde generalization of this block; it is deliberately
# localized here. The surrounding machinery (sizes, Sylvester split, Vandermonde
# inverse, Y/Z recurrence, evaluator) is already size-generic.
# -----------------------------------------------------------------------

function _algebraic_coefs(B::Vector{Matrix{ComplexF64}}, R::Vector{RT},
    n_alg::Int, alg::Vector{Int}, kmax::Int) where {RT<:Number}
    na = n_alg
    Z2 = zeros(ComplexF64, na, na)
    I2 = Matrix{ComplexF64}(I, na, na)
    N = kmax + 3

    K2 = Vector{Matrix{ComplexF64}}(undef, N)
    Qm = Vector{Matrix{ComplexF64}}(undef, N)
    Cm = Vector{Matrix{ComplexF64}}(undef, N)

    Balg(k) = Matrix(@view B[k][alg, alg])

    K2[1] = copy(Z2)
    Qm[1] = copy(I2)
    Cm[1] = Balg(1)   # C_0 = B_0[alg,alg]

    for k in 1:(kmax+2)
        K2acc = Balg(k + 1) + ComplexF64(2 * (k - 1)) * Qm[k]
        for l in 1:(k-1)
            K2acc += Balg(k - l + 1) * Qm[l+1] - Qm[l+1] * Cm[k-l+1]
        end
        K2[k+1] = K2acc
        na == 2 || throw(
            ArgumentError(
                "_algebraic_coefs: companion construction currently implements n_alg == 2 (GGJ floor); n_alg = $na needs confluent-Vandermonde generalization"
            )
        )
        # Companion form of the 2×2 algebraic block (Eqs. 32–39).
        Qm[k+1] = ComplexF64[
            0 0
            -K2acc[1, 1] -K2acc[1, 2]
        ]
        Cm[k+1] = ComplexF64[
            0 0
            K2acc[2, 1] K2acc[1, 1]+K2acc[2, 2]
        ]
    end

    # Shearing matrices D_k (Eq. 43), 2×2 (GGJ floor).
    D = Vector{Matrix{ComplexF64}}(undef, kmax + 1)
    D[1] = ComplexF64[
        0 1
        Cm[3][2, 1] 3
    ]
    for k in 1:kmax
        D[k+1] = ComplexF64[
            0 0
            Cm[k+3][2, 1] Cm[k+2][2, 2]
        ]
    end

    # Lowest-order Vandermonde Y0 from the Frobenius exponents (Eq. 49), general
    # over n_alg, inverted by LU.
    Y0 = Matrix{ComplexF64}(undef, na, na)
    @inbounds for j in 1:na, i in 1:na
        Y0[i, j] = ComplexF64(R[j])^(i - 1)
    end
    Y0inv = inv(Y0)

    # E_k, Z_k, Y_k recurrence (Eqs. 52). The (R[j] − R[i] − 2k) denominator
    # generalizes over n_alg.
    Y = Vector{Matrix{ComplexF64}}(undef, kmax + 1)
    Z = Vector{Matrix{ComplexF64}}(undef, kmax + 1)
    Z[1] = copy(I2)
    Y[1] = copy(Y0)
    for k in 1:kmax
        Zacc = copy(Z2)
        for l in 1:k
            El = Y0inv * D[l+1] * Y0
            Zacc += El * Z[k-l+1]
        end
        @inbounds for j in 1:na, i in 1:na
            Zacc[i, j] /= (ComplexF64(R[j]) - ComplexF64(R[i]) - 2 * k)
        end
        Z[k+1] = Zacc
        Y[k+1] = Y0 * Z[k+1]
    end

    # Shear powers (in the xfac = x⁻² variable) of the algebraic columns. For the
    # GGJ 2×2 block this is S(x) = diag(1, xfac); the natural generalization is
    # diag(xfac^0, …, xfac^(n_alg-1)).
    spow = collect(0:(na-1))

    return Y0, Y0inv, Qm, Y, Z, spow
end

# -----------------------------------------------------------------------
# Public builder.
# -----------------------------------------------------------------------

"""
    build_wasow(spec::SystemSpec, params, Q::ComplexF64; kmax::Int=8) -> WasowCache

Construct the generalized Wasow asymptotic basis for `spec` at growth-rate
parameter `Q`, truncating each power series at order `kmax`. The returned cache
is evaluated at `x > 0` via [`evaluate_wasow`](@ref) and queried for an adaptive
`X_max` via [`pick_xmax`](@ref).
"""
function build_wasow(spec::SystemSpec, params, Q::ComplexF64; kmax::Int=8)
    A = spec.asymptotic_matrices(params, Q)
    T, Tinv = spec.eigenbasis(params, Q)
    Tm = Matrix{ComplexF64}(T)
    Tim = Matrix{ComplexF64}(Tinv)
    J = Matrix{ComplexF64}[Tim * Matrix{ComplexF64}(A[i]) * Tm for i in 1:spec.nterms]
    R = collect(spec.exponents(params, Q))

    P, B = _split_recurrence(J, spec.nterms, spec.alg_idx, spec.exp_idx, spec.n, kmax)
    Y0, Y0inv, Qmat, Y, Z, spow = _algebraic_coefs(B, R, spec.n_alg, spec.alg_idx, kmax)

    # x-independent quantities used on every evaluation, packed once here.
    na = spec.n_alg
    cc = _pack_y_coefs(Y, kmax, na)
    dd = _pack_qp_coefs(Qmat, P, kmax, na, spec.n - na, spec.alg_idx, spec.exp_idx)
    rvec = [-R[j] / 2 for j in 1:na for _ in 1:na]   # column-major: rvec[(j-1)*na+i]

    return WasowCache(Q, kmax, spec.n, spec.n_alg, spec.nterms,
        copy(spec.alg_idx), copy(spec.exp_idx), R, Tm, Tim, J,
        P, B, Qmat, Y0, Y0inv, Y, Z, spow, cc, dd, rvec)
end

build_wasow(spec::SystemSpec, params, Q::Number; kmax::Int=8) =
    build_wasow(spec, params, ComplexF64(Q); kmax=kmax)

# -----------------------------------------------------------------------
# Horner evaluator with optional fractional-power prefactor (port of GGJ
# `_horner`; already size-generic in the number of components).
#
# Computes y[i] = (Σ_{k=0..n} c[i,k] x^k) x^rvec[i] and its x-derivative.
# `c` is `nrows × (n+1)`; the second axis is k = 0..n, 1-indexed.
# -----------------------------------------------------------------------

function _horner(x::Real, c::AbstractMatrix{ComplexF64};
    rvec::Union{Nothing,AbstractVector{<:Number}}=nothing,
    derivative::Bool=false)
    nrows, ncols = size(c)
    n = ncols - 1

    y = ComplexF64.(c[:, ncols])
    @inbounds for k in (n-1):-1:0
        for i in 1:nrows
            y[i] = y[i] * x + c[i, k+1]
        end
    end

    if rvec !== nothing
        xrvec = [x^rvec[i] for i in 1:nrows]
        @inbounds for i in 1:nrows
            y[i] *= xrvec[i]
        end
    end

    if !derivative
        return y, nothing
    end

    dy = similar(y)
    if rvec !== nothing
        @inbounds for i in 1:nrows
            dy[i] = c[i, ncols] * (rvec[i] + n)
        end
        @inbounds for k in (n-1):-1:0
            for i in 1:nrows
                dy[i] = dy[i] * x + c[i, k+1] * (rvec[i] + k)
            end
        end
        @inbounds for i in 1:nrows
            xrv_i = x^rvec[i]
            dy[i] = dy[i] * xrv_i / x
        end
    else
        @inbounds for i in 1:nrows
            dy[i] = c[i, ncols] * n
        end
        @inbounds for k in (n-1):-1:0
            for i in 1:nrows
                dy[i] = dy[i] * x + c[i, k+1] * k
            end
        end
        @inbounds for i in 1:nrows
            dy[i] = dy[i] / x
        end
    end
    return y, dy
end

# Pack the Y series into Horner-coefficient form (n_alg² × (kmax+1)), column-major
# in the algebraic indices. `x`-independent; built once in `build_wasow`.
function _pack_y_coefs(Y::Vector{Matrix{ComplexF64}}, kmax::Int, na::Int)
    cc = Matrix{ComplexF64}(undef, na * na, kmax + 1)
    @inbounds for k in 0:kmax
        Yk = Y[k+1]
        for j in 1:na, i in 1:na
            cc[(j-1)*na+i, k+1] = Yk[i, j]
        end
    end
    return cc
end

# Pack the companion Q series and the P[exp,alg] blocks into Horner-coefficient
# form ((n_alg² + n_exp·n_alg) × (kmax+1)), column-major. `x`-independent.
function _pack_qp_coefs(Qmat::Vector{Matrix{ComplexF64}}, P::Vector{Matrix{ComplexF64}},
    kmax::Int, na::Int, ne::Int, alg::Vector{Int}, exp::Vector{Int})
    dd = Matrix{ComplexF64}(undef, na * na + ne * na, kmax + 1)
    @inbounds for k in 0:kmax
        Qk = Qmat[k+1]
        Pk = P[k+1]
        for j in 1:na, i in 1:na
            dd[(j-1)*na+i, k+1] = Qk[i, j]
        end
        for j in 1:na, i in 1:ne
            dd[na*na+(j-1)*ne+i, k+1] = Pk[exp[i], alg[j]]
        end
    end
    return dd
end

# -----------------------------------------------------------------------
# Evaluator (generalizes GGJ `evaluate_asymptotics`).
# -----------------------------------------------------------------------

"""
    evaluate_wasow(cache::WasowCache, x::Real; derivative::Bool=true, apply_T::Bool=true)
        -> (U, dU)

Evaluate the Wasow asymptotic basis at `x > 0`. Returns the `n × n_alg` matrix
`U` whose columns are the algebraically-decaying ("small") asymptotic solutions,
and (if `derivative=true`) the `n × n_alg` matrix `dU = dU/dx`.

With `apply_T=false` the result is left in the rotated coordinate basis (used by
the residual check); the default `apply_T=true` returns it in the original
first-order-system basis used by the FEM and shooting solvers.
"""
function evaluate_wasow(cache::WasowCache, x::Real; derivative::Bool=true, apply_T::Bool=true)
    na = cache.n_alg
    ne = cache.n - na
    alg = cache.alg_idx
    exp = cache.exp_idx
    xfac = 1.0 / (x * x)

    # cc, dd (Horner packings) and rvec (= -R[j]/2) are x-independent: precomputed
    # in build_wasow. Each Y column j carries a prefactor x^{R[j]} = xfac^{-R[j]/2}.
    yy, dyy = _horner(xfac, cache.cc; rvec=cache.rvec, derivative=derivative)
    zz, dzz = _horner(xfac, cache.dd; derivative=derivative)

    y = reshape(yy, na, na)
    q = reshape(view(zz, 1:(na*na)), na, na) |> Matrix
    p_ea = reshape(view(zz, (na*na+1):(na*na+ne*na)), ne, na) |> Matrix

    pp = zeros(ComplexF64, cache.n, na)
    @inbounds for i in 1:na
        pp[alg[i], i] = 1
    end
    @inbounds for j in 1:na, i in 1:ne
        pp[exp[i], j] = p_ea[i, j]
    end

    smat = Diagonal(ComplexF64[xfac^cache.spow[j] for j in 1:na])
    qsy = q * smat * y
    U = pp * qsy

    if derivative
        dxfac = -2 * xfac / x
        dy = reshape(dxfac .* dyy, na, na) |> Matrix
        dz = dxfac .* dzz
        dq = reshape(view(dz, 1:(na*na)), na, na) |> Matrix
        dp_ea = reshape(view(dz, (na*na+1):(na*na+ne*na)), ne, na) |> Matrix

        dpp = zeros(ComplexF64, cache.n, na)
        @inbounds for j in 1:na, i in 1:ne
            dpp[exp[i], j] = dp_ea[i, j]
        end

        dsdiag = ComplexF64[cache.spow[j] == 0 ? 0.0 + 0.0im : cache.spow[j] * xfac^(cache.spow[j] - 1) * dxfac for j in 1:na]
        dsmat = Diagonal(dsdiag)
        dqsy = q * smat * dy + q * dsmat * y + dq * smat * y
        dU = pp * dqsy + dpp * qsy

        if apply_T
            U = cache.T * U
            dU = cache.T * dU
        end
        return U, dU
    else
        if apply_T
            U = cache.T * U
        end
        return U, nothing
    end
end

# -----------------------------------------------------------------------
# Residual and adaptive xmax (generalize GGJ `asymptotic_residual` / `pick_xmax`).
# -----------------------------------------------------------------------

"""
    wasow_residual(cache::WasowCache, x::Real) -> Vector{Float64}

Per-column residual `‖dU − x·M·U‖∞ / max(‖dU‖∞, ‖x·M·U‖∞)` of the asymptotic
basis at `x`, where `M = Σ_{i=0}^{nterms-1} xfac^i J_i` (xfac = x⁻²) is the
rotated coefficient matrix. One entry per algebraic column.
"""
function wasow_residual(cache::WasowCache, x::Real)
    U, dU = evaluate_wasow(cache, x; derivative=true, apply_T=false)
    xfac = 1.0 / (x * x)
    M = copy(cache.J[1])
    @inbounds for i in 1:(cache.nterms-1)
        M .+= (xfac^i) .* cache.J[i+1]
    end

    matvec1 = dU
    matvec2 = -x * (M * U)
    matvec0 = matvec1 + matvec2

    na = cache.n_alg
    delta = zeros(Float64, na)
    @inbounds for j in 1:na
        n0 = 0.0
        n1 = 0.0
        n2 = 0.0
        for i in 1:cache.n
            n0 = max(n0, abs(matvec0[i, j]))
            n1 = max(n1, abs(matvec1[i, j]))
            n2 = max(n2, abs(matvec2[i, j]))
        end
        denom = max(n1, n2)
        delta[j] = denom == 0 ? 0.0 : n0 / denom
    end
    return delta
end

"""
    pick_xmax(spec::SystemSpec, params, Q::ComplexF64;
              eps::Float64=1e-7, kmax::Int=8,
              xlogmin::Float64=-1.0, xlogmax::Float64=4.0,
              dxlog::Float64=0.01) -> (Float64, WasowCache)

Sweep `x` log-uniformly upward and return the smallest `x` at which
`max(wasow_residual) < eps`, together with the `WasowCache` it built.
"""
function pick_xmax(spec::SystemSpec, params, Q::ComplexF64;
    eps::Float64=1e-7, kmax::Int=8,
    xlogmin::Float64=-1.0, xlogmax::Float64=4.0, dxlog::Float64=0.01)
    cache = build_wasow(spec, params, Q; kmax=kmax)
    dxfac = 10.0^dxlog
    xlog = xlogmin
    x = 10.0^xlog
    while xlog <= xlogmax
        if maximum(wasow_residual(cache, x)) < eps
            return x, cache
        end
        xlog += dxlog
        x *= dxfac
    end
    error("pick_xmax: failed to reach residual < $eps for x in [10^$xlogmin, 10^$xlogmax]")
end

pick_xmax(spec::SystemSpec, params, Q::Number; kwargs...) =
    pick_xmax(spec, params, ComplexF64(Q); kwargs...)

# -----------------------------------------------------------------------
# Three-level xmax sweep used by the Galerkin grid (port of GGJ `_xmax_3level`).
# -----------------------------------------------------------------------

"""
    xmax_3level(spec::SystemSpec, params, Q::ComplexF64;
                kmax::Int=8, xfac::Float64=1.0,
                eps_vec::NTuple{3,Float64}=(1e-2, 5e-7, 1e-7))
        -> (; xmax, dx1, dx2, cache)

Locate the three nested matching radii used to build the Galerkin grid: the
outer matching point `xmax` (where the asymptotic residual falls below
`eps_vec[3]`) and two interval widths `dx1, dx2` separating the extension cells,
together with the [`WasowCache`](@ref) it built. Each radius is bracketed by a
coarse log-uniform sweep and refined by bisection.
"""
function xmax_3level(spec::SystemSpec, params, Q::ComplexF64;
    kmax::Int=8, xfac::Float64=1.0,
    eps_vec::NTuple{3,Float64}=(1e-2, 5e-7, 1e-7))
    cache = build_wasow(spec, params, Q; kmax=kmax)

    dxfac = 10.0^0.01
    xmax_vals = zeros(Float64, 3)
    brackets = Vector{Tuple{Float64,Float64}}(undef, 3)
    set = [true, true, true]
    x_prev = 0.1
    delta_prev = maximum(wasow_residual(cache, x_prev))
    x = x_prev * dxfac
    for _ in 1:1000
        dmax = maximum(wasow_residual(cache, x))
        for i in 1:3
            if delta_prev >= eps_vec[i] && dmax < eps_vec[i] && set[i]
                brackets[i] = (x_prev, x)
                set[i] = false
            end
        end
        if !any(set)
            break
        end
        x_prev = x
        delta_prev = dmax
        x *= dxfac
    end
    any(set) && error("xmax_3level: failed to bracket all xmax levels")

    for i in 1:3
        lo, hi = brackets[i]
        for _ in 1:60
            mid = (lo + hi) / 2
            if maximum(wasow_residual(cache, mid)) < eps_vec[i]
                hi = mid
            else
                lo = mid
            end
        end
        xmax_vals[i] = (lo + hi) / 2
    end

    xmax_vals[2] *= xfac
    xmax_vals[3] *= xfac
    xmax = xmax_vals[3]
    dx1 = xmax_vals[3] - xmax_vals[2]
    dx2 = dx1 / 2
    return (; xmax, dx1, dx2, cache)
end
