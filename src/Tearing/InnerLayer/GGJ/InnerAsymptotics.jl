# InnerAsymptotics.jl
#
# Wasow asymptotic basis ("inps" basis) for the GGJ inner-layer system.
#
# Reference: A. H. Glasser & Z. R. Wang, Phys. Plasmas 27, 012506 (2020),
# Sections II.A–II.G (Eqs. 4–53). Notation map:
#
#   A0/A1/A2  -> Eqs. 4–5      (physical-system coefficient matrices)
#   T, Tinv   -> Eqs. 7–8      (eigenvector basis of A_0)
#   J0/J1/J2  -> Eqs. 9–10     (J_i = T^{-1} A_i T)
#   P_k, B_k  -> Eqs. 16, 22   (splitting transformation; Lyapunov solve)
#   Q_k, C_k  -> Eqs. 32–39    (2×2 inner sub-block diagonalization)
#   D_k       -> Eq. 43        (shearing transformation)
#   Y0, R     -> Eq. 49        (lowest-order solution and Frobenius exponents)
#   Z_k       -> Eq. 52        (Y-series in shifted-exponent form)
#   U(x)      -> Eq. 53        (final asymptotic solution at large x)

using LinearAlgebra
using StaticArrays

# Index ranges that partition the 6-component first-order system into the
# 2-dimensional algebraic ("small") subspace and the 4-dimensional
# exponential ("large") subspace.
const _R1 = SVector(1, 2)
const _R2 = SVector(3, 4, 5, 6)

"""
    InnerAsymptoticsCache

Frozen state of the `inps` Wasow asymptotic-basis construction for a single
`(GGJParameters, Q)` pair. All matrices are stored as `SMatrix`/`SVector`
so the evaluator can run allocation-free on a hot path.

Index convention: `P[k+1]` holds the k-th-order matrix `P_k`, `B[k+1]`
holds `B_k`, etc., for k = 0, 1, …, the upper bound documented in each
field.

Fields:

  - `params, Q, kmax` — input parameters and series truncation order.
  - `λ = 1/√Q` — complex scale factor used by the Wasow split.
  - `R = (r₊, r₋)` — Mercier-shifted Frobenius exponents at infinity (Eq. 49).
  - `T, Tinv` — 6×6 eigenvector basis of A₀ (Eq. 7–8).
  - `J` — `(J₀, J₁, J₂)`, the J-rotated coefficient matrices (Eq. 9–10).
  - `P, B` — splitting matrices, k = 0..kmax+2 (Eqs. 16, 22).
  - `K2` — 2×2 inner working matrices, k = 0..kmax+2 (Eq. 32; entry k=0 unused).
  - `Qmat, Cmat` — 2×2 inner-block transformation matrices, k = 0..kmax+2 (Eqs. 32–39).
  - `Dmat` — 2×2 shearing matrices, k = 0..kmax (Eq. 43).
  - `Y0, Y0inv` — lowest-order Y matrix and its inverse (Eq. 49).
  - `Y, Z` — 2×2 series matrices, k = 0..kmax (Eq. 52).
"""
struct InnerAsymptoticsCache
    params::GGJParameters
    Q::ComplexF64
    kmax::Int
    λ::ComplexF64
    R::SVector{2,Float64}
    T::SMatrix{6,6,ComplexF64,36}
    Tinv::SMatrix{6,6,ComplexF64,36}
    J::NTuple{3,SMatrix{6,6,ComplexF64,36}}
    P::Vector{SMatrix{6,6,ComplexF64,36}}    # k = 0..kmax+2
    B::Vector{SMatrix{6,6,ComplexF64,36}}    # k = 0..kmax+2
    K2::Vector{SMatrix{2,2,ComplexF64,4}}    # k = 0..kmax+2
    Qmat::Vector{SMatrix{2,2,ComplexF64,4}}  # k = 0..kmax+2
    Cmat::Vector{SMatrix{2,2,ComplexF64,4}}  # k = 0..kmax+2
    Dmat::Vector{SMatrix{2,2,ComplexF64,4}}  # k = 0..kmax
    Y0::SMatrix{2,2,ComplexF64,4}
    Y0inv::SMatrix{2,2,ComplexF64,4}
    Y::Vector{SMatrix{2,2,ComplexF64,4}}     # k = 0..kmax
    Z::Vector{SMatrix{2,2,ComplexF64,4}}     # k = 0..kmax
end

# Helpers for cache indexing using physical (zero-based) order k.
@inline _at(v::Vector, k::Int) = v[k+1]

# -----------------------------------------------------------------------
# Step 1: build T, Tinv, A_0..A_2, J_0..J_2.
# -----------------------------------------------------------------------

function _build_tjmat(p::GGJParameters, Q::ComplexF64)
    e = ComplexF64(p.E)
    f = ComplexF64(p.F)
    h = ComplexF64(p.H)
    g = ComplexF64(p.G)
    k = ComplexF64(p.K)
    q = Q
    q2 = q * q
    λ = 1 / sqrt(q)

    # T (Eq. 7) — built by column-major reshape; the listing below is
    # row-major Julia order matching that layout.
    T = @SMatrix ComplexF64[
        1 0 h*q q2/λ h*q -q2/λ
        0 0 0 -1/λ 0 1/λ
        0 0 -1/λ 0 1/λ 0
        0 1 -h/λ -q2 h/λ -q2
        0 0 0 1 0 1
        0 0 1 0 1 0
    ]

    Tinv = @SMatrix ComplexF64[
        1 q2 0 0 0 -h*q
        0 0 -h 1 q2 0
        0 0 -λ/2 0 0 1/2
        0 -λ/2 0 0 1/2 0
        0 0 λ/2 0 0 1/2
        0 λ/2 0 0 1/2 0
    ]

    # A_0, A_1, A_2 — physical-system coefficient matrices. Build mutable then freeze.
    A0 = zeros(ComplexF64, 6, 6)
    A1 = zeros(ComplexF64, 6, 6)
    A2 = zeros(ComplexF64, 6, 6)

    # A0
    A0[4, 2] = -q
    A0[5, 2] = 1 / q
    A0[6, 3] = 1 / q
    A0[1, 4] = 1
    A0[2, 5] = 1
    A0[3, 6] = 1
    A0[4, 6] = h

    # A1
    A1[4, 1] = q
    A1[5, 1] = -1 / q
    A1[6, 1] = -1 / q
    A1[6, 2] = -(g - k * e) * q
    A1[5, 3] = -(e + f) / q2
    A1[6, 3] = (g + k * f) * q
    A1[4, 4] = 1
    A1[5, 4] = -h / q2
    A1[6, 4] = h * k * q
    A1[5, 5] = -1
    A1[6, 6] = -1

    # A2
    A2[4, 1] = -2
    A2[5, 1] = h / q2
    A2[6, 1] = -h * k * q

    # Freeze A_i and build J_i = T^{-1} A_i T.
    A0s = SMatrix{6,6,ComplexF64}(A0)
    A1s = SMatrix{6,6,ComplexF64}(A1)
    A2s = SMatrix{6,6,ComplexF64}(A2)

    J0 = Tinv * A0s * T
    J1 = Tinv * A1s * T
    J2 = Tinv * A2s * T

    return T, Tinv, (J0, J1, J2), λ
end

# -----------------------------------------------------------------------
# Step 2: closed-form Lyapunov solve for the splitting transformation.
#
# Given a 6×6 K, returns (B, P) such that:
#   - B is block-diagonal with B[r1,r1] = K[r1,r1] and B[r2,r2] = K[r2,r2]
#   - P has zero diagonal blocks, P[r1,r2] and P[r2,r1] given by the
#     closed-form expressions below (which exploit the special structure
#     of the lower-right block of J_0).
# -----------------------------------------------------------------------

function _lyap_solve(K::SMatrix{6,6,ComplexF64}, λ::ComplexF64)
    Bm = zeros(ComplexF64, 6, 6)
    Pm = zeros(ComplexF64, 6, 6)

    # B is block-diagonal in the (r1, r2) split.
    Bm[1, 1] = K[1, 1]
    Bm[1, 2] = K[1, 2]
    Bm[2, 1] = K[2, 1]
    Bm[2, 2] = K[2, 2]
    for i in 3:6, j in 3:6
        Bm[i, j] = K[i, j]
    end

    # P[r1, r2] = P[1:2, 3:6] (top-right off-diagonal block)
    @inbounds begin
        Pm[2, 3] = -K[2, 3] / λ
        Pm[2, 4] = -K[2, 4] / λ
        Pm[2, 5] = K[2, 5] / λ
        Pm[2, 6] = K[2, 6] / λ
        Pm[1, 3] = -(K[1, 3] + Pm[2, 3]) / λ
        Pm[1, 4] = -(K[1, 4] + Pm[2, 4]) / λ
        Pm[1, 5] = (K[1, 5] + Pm[2, 5]) / λ
        Pm[1, 6] = (K[1, 6] + Pm[2, 6]) / λ
    end

    # P[r2, r1] = P[3:6, 1:2] (bottom-left off-diagonal block)
    @inbounds begin
        Pm[3, 1] = K[3, 1] / λ
        Pm[4, 1] = K[4, 1] / λ
        Pm[5, 1] = -K[5, 1] / λ
        Pm[6, 1] = -K[6, 1] / λ
        Pm[3, 2] = (K[3, 2] - Pm[3, 1]) / λ
        Pm[4, 2] = (K[4, 2] - Pm[4, 1]) / λ
        Pm[5, 2] = -(K[5, 2] - Pm[5, 1]) / λ
        Pm[6, 2] = -(K[6, 2] - Pm[6, 1]) / λ
    end

    return SMatrix{6,6,ComplexF64}(Bm), SMatrix{6,6,ComplexF64}(Pm)
end

# -----------------------------------------------------------------------
# Step 3: split recurrence.
# Builds B_k, P_k, K6_k for k = 0..kmax+2.
# -----------------------------------------------------------------------

function _split_recurrence(J::NTuple{3,SMatrix{6,6,ComplexF64}}, λ::ComplexF64, kmax::Int)
    N = kmax + 3                          # store k = 0..kmax+2
    P = Vector{SMatrix{6,6,ComplexF64,36}}(undef, N)
    B = Vector{SMatrix{6,6,ComplexF64,36}}(undef, N)
    K6 = Vector{SMatrix{6,6,ComplexF64,36}}(undef, N)

    Z6 = zero(SMatrix{6,6,ComplexF64})
    P[1] = SMatrix{6,6,ComplexF64}(I)     # P_0 = I
    B[1] = J[1]                            # B_0 = J_0
    K6[1] = J[1]

    for k in 1:(kmax+2)
        kk = k + 1                         # 1-based slot for K6/B/P at index k
        # K6_k starts as J_k (only k = 1, 2 contribute), else 0
        Kacc = (k <= 2) ? J[k+1] : Z6
        # +2*(k-1)*P_{k-1}  (k > 1 only)
        if k > 1
            Kacc = Kacc + ComplexF64(2 * (k - 1)) * P[k]
        end
        # convolution: l = 1..k-1
        for l in 1:(k-1)
            kml = k - l
            if kml <= 2
                Kacc = Kacc + J[kml+1] * P[l+1]
            end
            Kacc = Kacc - P[l+1] * B[kml+1]
        end
        K6[kk] = Kacc
        Bk, Pk = _lyap_solve(Kacc, λ)
        B[kk] = Bk
        P[kk] = Pk
    end
    return P, B, K6
end

# -----------------------------------------------------------------------
# Step 4: coefs recurrence.
# Builds K2_k, Q_k, C_k for k = 0..kmax+2 and D_k, E_k, Y_k, Z_k.
# -----------------------------------------------------------------------

function _coefs(B::Vector{SMatrix{6,6,ComplexF64,36}},
    R::SVector{2,Float64}, kmax::Int)

    N = kmax + 3
    Z2 = zero(SMatrix{2,2,ComplexF64})
    I2 = SMatrix{2,2,ComplexF64}(I)

    K2 = Vector{SMatrix{2,2,ComplexF64,4}}(undef, N)
    Qm = Vector{SMatrix{2,2,ComplexF64,4}}(undef, N)
    Cm = Vector{SMatrix{2,2,ComplexF64,4}}(undef, N)

    K2[1] = Z2
    Qm[1] = I2
    # B_0[r1, r1] = first-2x2 block of B_0
    B0_11 = SMatrix{2,2,ComplexF64}(@view B[1][1:2, 1:2])
    Cm[1] = B0_11

    for k in 1:(kmax+2)
        Bk_11 = SMatrix{2,2,ComplexF64}(@view B[k+1][1:2, 1:2])
        K2acc = Bk_11 + ComplexF64(2 * (k - 1)) * Qm[k]
        for l in 1:(k-1)
            Bkml_11 = SMatrix{2,2,ComplexF64}(@view B[k-l+1][1:2, 1:2])
            K2acc = K2acc + Bkml_11 * Qm[l+1] - Qm[l+1] * Cm[k-l+1]
        end
        K2[k+1] = K2acc

        # Q_k = [[0, 0], [-K2[1,1], -K2[1,2]]] (column-major reshape gives this layout)
        Qm[k+1] = @SMatrix ComplexF64[
            0 0
            -K2acc[1, 1] -K2acc[1, 2]
        ]
        # C_k = [[0, 0], [K2[2,1], K2[1,1] + K2[2,2]]]
        Cm[k+1] = @SMatrix ComplexF64[
            0 0
            K2acc[2, 1] K2acc[1, 1]+K2acc[2, 2]
        ]
    end

    # Build D_k for k = 0..kmax.
    D = Vector{SMatrix{2,2,ComplexF64,4}}(undef, kmax + 1)
    # D_0 = [[0, 1], [C_2[2,1], 3]]
    D[1] = @SMatrix ComplexF64[
        0 1
        Cm[3][2, 1] 3
    ]
    for k in 1:kmax
        D[k+1] = @SMatrix ComplexF64[
            0 0
            Cm[k+3][2, 1] Cm[k+2][2, 2]
        ]
    end

    # Lowest-order Y solution and inverse (Eq. 49).
    r1 = ComplexF64(R[1])
    r2 = ComplexF64(R[2])
    Y0 = @SMatrix ComplexF64[
        1 1
        r1 r2
    ]
    Y0inv = (1 / (r1 - r2)) * @SMatrix ComplexF64[
        -r2 1
        r1 -1
    ]

    # E_k, Z_k, Y_k recurrence.
    Y = Vector{SMatrix{2,2,ComplexF64,4}}(undef, kmax + 1)
    Z = Vector{SMatrix{2,2,ComplexF64,4}}(undef, kmax + 1)
    Z[1] = I2
    Y[1] = Y0
    for k in 1:kmax
        # Build Z_k = sum_{l=1..k} E_l * Z_{k-l}, with E_l = Y0inv * D_l * Y0
        Zacc = Z2
        for l in 1:k
            El = Y0inv * D[l+1] * Y0
            Zacc = Zacc + El * Z[k-l+1]
        end
        # Divide each entry by (R[j] - R[i] - 2k)
        Zk = MMatrix{2,2,ComplexF64}(Zacc)
        for i in 1:2, j in 1:2
            Zk[i, j] = Zk[i, j] / (R[j] - R[i] - 2 * k)
        end
        Z[k+1] = SMatrix{2,2,ComplexF64}(Zk)
        Y[k+1] = Y0 * Z[k+1]
    end

    return K2, Qm, Cm, D, Y0, Y0inv, Y, Z
end

# -----------------------------------------------------------------------
# Public builder.
# -----------------------------------------------------------------------

"""
    build_asymptotics(params::GGJParameters, Q::ComplexF64; kmax::Int=8) -> InnerAsymptoticsCache

Construct the `inps` Wasow asymptotic basis for the given GGJ parameters
and dimensionless growth rate `Q`. Truncates each power series at order
`kmax` (default `8`). The returned cache can be evaluated at any `x > 0`
via [`evaluate_asymptotics`](@ref) and queried for an adaptive `X_max`
via [`pick_xmax`](@ref).

Reference: Glasser & Wang, Phys. Plasmas **27**, 012506 (2020), Eqs. 7–53.
"""
function build_asymptotics(params::GGJParameters, Q::ComplexF64; kmax::Int=8)
    p1v = p1(params)
    R = SVector{2,Float64}(p1v + 1.5, -p1v + 1.5)

    T, Tinv, J, λ = _build_tjmat(params, Q)
    P, B, _ = _split_recurrence(J, λ, kmax)
    K2, Qm, Cm, D, Y0, Y0inv, Y, Z = _coefs(B, R, kmax)

    return InnerAsymptoticsCache(
        params, Q, kmax, λ, R, T, Tinv, J,
        P, B, K2, Qm, Cm, D, Y0, Y0inv, Y, Z
    )
end

build_asymptotics(params::GGJParameters, Q::Number; kmax::Int=8) =
    build_asymptotics(params, ComplexF64(Q); kmax=kmax)

# -----------------------------------------------------------------------
# Horner evaluator with optional fractional-power prefactor.
#
# Computes y[i] = (Σ_{k=0..n} c[i,k] * x^k) * x^rvec[i]
# and       dy[i] = d/dx of the above.
#
# `c` is an n_components × (n+1) matrix; the second axis is k = 0..n,
# 1-indexed in Julia.
# -----------------------------------------------------------------------

function _horner(x::Real, c::AbstractMatrix{ComplexF64};
    rvec::Union{Nothing,AbstractVector{<:Real}}=nothing,
    derivative::Bool=false)
    nrows, ncols = size(c)
    n = ncols - 1   # highest power

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

# Pack the cache's Y, Qmat, and the [r2, r1] block of P into the row-major
# reshape order used by `_horner`. Returned matrices are
# n_components × (kmax+1).
function _pack_y_coefs(cache::InnerAsymptoticsCache)
    kmax = cache.kmax
    cc = Matrix{ComplexF64}(undef, 4, kmax + 1)
    @inbounds for k in 0:kmax
        Yk = cache.Y[k+1]
        # column-major reshape of (2,2) → 4 entries:
        # (Y[1,1], Y[2,1], Y[1,2], Y[2,2])
        cc[1, k+1] = Yk[1, 1]
        cc[2, k+1] = Yk[2, 1]
        cc[3, k+1] = Yk[1, 2]
        cc[4, k+1] = Yk[2, 2]
    end
    return cc
end

function _pack_qp_coefs(cache::InnerAsymptoticsCache)
    kmax = cache.kmax
    dd = Matrix{ComplexF64}(undef, 12, kmax + 1)
    @inbounds for k in 0:kmax
        Qk = cache.Qmat[k+1]
        Pk = cache.P[k+1]
        # dd[1:4] from Q (column-major reshape of 2×2 block)
        dd[1, k+1] = Qk[1, 1]
        dd[2, k+1] = Qk[2, 1]
        dd[3, k+1] = Qk[1, 2]
        dd[4, k+1] = Qk[2, 2]
        # dd[5:12] from P[r2, r1] = P[3:6, 1:2] (4×2 → column-major 8 entries)
        dd[5, k+1] = Pk[3, 1]
        dd[6, k+1] = Pk[4, 1]
        dd[7, k+1] = Pk[5, 1]
        dd[8, k+1] = Pk[6, 1]
        dd[9, k+1] = Pk[3, 2]
        dd[10, k+1] = Pk[4, 2]
        dd[11, k+1] = Pk[5, 2]
        dd[12, k+1] = Pk[6, 2]
    end
    return dd
end

# -----------------------------------------------------------------------
# Step 5: evaluator.
# -----------------------------------------------------------------------

"""
    evaluate_asymptotics(cache::InnerAsymptoticsCache, x::Real;
                        derivative::Bool=true, apply_T::Bool=true)
        -> (U, dU)

Evaluate the inps asymptotic basis at `x > 0`. Returns the 6×2 complex
matrix `U` whose two columns are the algebraically-decaying ("small")
asymptotic solutions of the GGJ system, and (if `derivative=true`) the
6×2 matrix `dU` of their derivatives `dU/dx`.

If `apply_T=false`, the result is left in the J-rotated coordinate basis
(used by `asymptotic_residual` for residual checks). The default `apply_T=true`
returns the solutions in the original 6-component first-order-system
basis used by `_physical_uv` and the shooting / Galerkin solvers.
"""
function evaluate_asymptotics(cache::InnerAsymptoticsCache, x::Real;
    derivative::Bool=true, apply_T::Bool=true)
    xfac = 1.0 / (x * x)
    R = cache.R
    rvec = SVector(-R[1] / 2, -R[1] / 2, -R[2] / 2, -R[2] / 2)

    cc = _pack_y_coefs(cache)
    dd = _pack_qp_coefs(cache)

    yy, dyy = _horner(xfac, cc; rvec=rvec, derivative=derivative)
    zz, dzz = _horner(xfac, dd; derivative=derivative)

    # Reshape yy → y (2×2), zz → q (2×2) and p21 (4×2)
    y = SMatrix{2,2,ComplexF64}(yy[1], yy[2], yy[3], yy[4])
    q = SMatrix{2,2,ComplexF64}(zz[1], zz[2], zz[3], zz[4])
    p21 = SMatrix{4,2,ComplexF64}(zz[5], zz[6], zz[7], zz[8],
        zz[9], zz[10], zz[11], zz[12])

    # Splitting matrix pp (6×2): top 2×2 = I, bottom 4×2 = p21.
    pp_m = zeros(ComplexF64, 6, 2)
    pp_m[1, 1] = 1
    pp_m[2, 2] = 1
    @inbounds for i in 1:4, j in 1:2
        pp_m[i+2, j] = p21[i, j]
    end
    pp = SMatrix{6,2,ComplexF64}(pp_m)

    smat = @SMatrix ComplexF64[1 0; 0 xfac]

    qsy = q * smat * y
    U = pp * qsy

    if derivative
        # Apply chain rule: derivatives from horner are wrt xfac, convert to
        # derivatives wrt x via factor (-2*xfac/x) = d(xfac)/dx.
        chain = -2 * xfac / x
        dy_v = chain .* dyy
        dz_v = chain .* dzz

        dy = SMatrix{2,2,ComplexF64}(dy_v[1], dy_v[2], dy_v[3], dy_v[4])
        dq = SMatrix{2,2,ComplexF64}(dz_v[1], dz_v[2], dz_v[3], dz_v[4])
        dp21 = SMatrix{4,2,ComplexF64}(dz_v[5], dz_v[6], dz_v[7], dz_v[8],
            dz_v[9], dz_v[10], dz_v[11], dz_v[12])

        dpp_m = zeros(ComplexF64, 6, 2)
        @inbounds for i in 1:4, j in 1:2
            dpp_m[i+2, j] = dp21[i, j]
        end
        dpp = SMatrix{6,2,ComplexF64}(dpp_m)

        dsmat = @SMatrix ComplexF64[0 0; 0 (-2*xfac/x)]
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
# Step 6: residual `delta` and adaptive xmax.
# -----------------------------------------------------------------------

"""
    asymptotic_residual(cache::InnerAsymptoticsCache, x::Real) -> SVector{2,Float64}

Compute the residual `D₆(x)` of the asymptotic basis at `x` for each of
the two algebraic columns:
returns `‖dU − x·matrix·U‖∞ / max(‖dU‖∞, ‖x·matrix·U‖∞)` per column,
where `matrix = J₀ + xfac·J₁ + xfac²·J₂` is the J-rotated coefficient
matrix.
"""
function asymptotic_residual(cache::InnerAsymptoticsCache, x::Real)
    U, dU = evaluate_asymptotics(cache, x; derivative=true, apply_T=false)

    xfac = 1.0 / (x * x)
    M = cache.J[1]
    if cache.kmax > 0
        M = M + xfac * cache.J[2]
    end
    if cache.kmax > 1
        M = M + xfac * xfac * cache.J[3]
    end

    # matvec(:,:,1) = dU; matvec(:,:,2) = -x*M*U; matvec(:,:,0) = sum.
    # delta(j) = ||matvec(:,j,0)||∞ / max(||matvec(:,j,1)||∞, ||matvec(:,j,2)||∞).
    matvec1 = dU
    matvec2 = -x * (M * U)
    matvec0 = matvec1 + matvec2

    delta = MVector{2,Float64}(0.0, 0.0)
    @inbounds for j in 1:2
        n0 = 0.0
        n1 = 0.0
        n2 = 0.0
        for i in 1:6
            n0 = max(n0, abs(matvec0[i, j]))
            n1 = max(n1, abs(matvec1[i, j]))
            n2 = max(n2, abs(matvec2[i, j]))
        end
        denom = max(n1, n2)
        delta[j] = denom == 0 ? 0.0 : n0 / denom
    end
    return SVector{2,Float64}(delta)
end

"""
    pick_xmax(params::GGJParameters, Q::ComplexF64;
              eps::Float64=1e-7, kmax::Int=8,
              xlogmin::Float64=-1.0, xlogmax::Float64=4.0,
              dxlog::Float64=0.01) -> (Float64, InnerAsymptoticsCache)

Sweep `x` log-uniformly upward from `10^xlogmin` and return the smallest
`x` at which `max(asymptotic_residual(cache, x)) < eps`. Also returns the
`InnerAsymptoticsCache` it built so callers can reuse it.

Throws an `ErrorException` if no `x` in the sweep range achieves the
target tolerance.
"""
function pick_xmax(params::GGJParameters, Q::ComplexF64;
    eps::Float64=1e-7, kmax::Int=8,
    xlogmin::Float64=-1.0, xlogmax::Float64=4.0,
    dxlog::Float64=0.01)
    cache = build_asymptotics(params, Q; kmax=kmax)
    dxfac = 10.0^dxlog
    xlog = xlogmin
    x = 10.0^xlog
    while xlog <= xlogmax
        delta = asymptotic_residual(cache, x)
        if maximum(delta) < eps
            return x, cache
        end
        xlog += dxlog
        x *= dxfac
    end
    error("pick_xmax: failed to reach residual < $eps for x in [10^$xlogmin, 10^$xlogmax]")
end

pick_xmax(params::GGJParameters, Q::Number; kwargs...) =
    pick_xmax(params, ComplexF64(Q); kwargs...)
