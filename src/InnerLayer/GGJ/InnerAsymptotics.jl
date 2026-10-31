# InnerAsymptotics.jl
#
# Wasow asymptotic basis ("inps" basis) for the GGJ inner-layer system.
#
# Reference: A. H. Glasser & Z. R. Wang, Phys. Plasmas 27, 012506 (2020)
# ("GW2020"), Secs. II.A–II.G (Eqs. 1–53) and III (Eqs. 54–55). Notation map (all equation
# numbers are GW2020):
#
#   A0/A1/A2  -> Eqs. 3–5      (A = A₀+x⁻²A₁+x⁻⁴A₂ (Eq. 3); A₀, A₂ Eq. 4; A₁ Eq. 5)
#   T, Tinv   -> Eqs. 6–8      (Jordan basis of A₀; u = Tv, v' = xJv, λ = Q^{-1/2})
#   J0/J1/J2  -> Eqs. 9–10     (Jₖ = T^{-1} Aₖ T)
#   P_k, B_k  -> Eqs. 11–24    (splitting transformation V = PW; Lyapunov solve Eq. 22)
#   Q_k, C_k  -> Eqs. 25–39    (2×2 inner-block diagonalization W = QX)
#   D_k       -> Eqs. 40–43    (shearing transformation X = SY)
#   Y0, R     -> Eqs. 48–49    (lowest-order solution; Mercier Frobenius exponents r±)
#   Y_k, Z_k  -> Eqs. 50–52    (Y-series Y = Y₀Z in shifted-exponent form)
#   U(x)      -> Eq. 53        (final asymptotic solution U = TPQSY at large x)
#   residual  -> Eq. 54        (convergence measure Δ±, used to pick the matching radius)

using LinearAlgebra
using StaticArrays

"""
    InnerAsymptoticsCache

Frozen `inps` Wasow asymptotic-basis construction for one `(GGJParameters, Q)`
pair, stored as `SMatrix`/`SVector` for allocation-free evaluation. Index
convention: `P[k+1]` holds the k-th-order matrix `P_k`, etc.

## Fields

  - `params, Q, kmax` — input parameters and series truncation order.
  - `R = (r₊, r₋)` — Mercier-shifted Frobenius exponents at infinity (Eq. 49).
  - `T` — 6×6 eigenvector basis of A₀ (Eq. 7).
  - `J` — `(J₀, J₁, J₂)`, the J-rotated coefficient matrices (Eqs. 9–10).
  - `P` — splitting matrices, k = 0..kmax+2 (Eqs. 16, 22).
  - `Qmat` — 2×2 inner-block transformation matrices, k = 0..kmax+2 (Eqs. 32–39).
  - `Y` — 2×2 series matrices Y_k = Y₀Z_k, k = 0..kmax (Eqs. 50–52).
"""
struct InnerAsymptoticsCache
    params::GGJParameters
    Q::ComplexF64
    kmax::Int
    R::SVector{2,Float64}
    T::SMatrix{6,6,ComplexF64,36}
    J::NTuple{3,SMatrix{6,6,ComplexF64,36}}
    P::Vector{SMatrix{6,6,ComplexF64,36}}
    Qmat::Vector{SMatrix{2,2,ComplexF64,4}}
    Y::Vector{SMatrix{2,2,ComplexF64,4}}
end

# T, T⁻¹, λ = Q^{-1/2} and J_i = T⁻¹A_iT from the coefficient matrices A₀–A₂ (GW2020 Eqs. 3–10).

function _build_tjmat(p::GGJParameters, Q::ComplexF64)
    e = ComplexF64(p.E)
    f = ComplexF64(p.F)
    h = ComplexF64(p.H)
    g = ComplexF64(p.G)
    k = ComplexF64(p.K)
    q = Q
    q2 = q * q
    λ = 1 / sqrt(q)

    # T (Eq. 7) and T⁻¹ (Eq. 8), listed row by row.
    #! format: off
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
    #! format: on

    # A₀ and A₂: GW2020 Eq. (4); A₁: Eq. (5).
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

    A0s = SMatrix{6,6,ComplexF64}(A0)
    A1s = SMatrix{6,6,ComplexF64}(A1)
    A2s = SMatrix{6,6,ComplexF64}(A2)

    J0 = Tinv * A0s * T
    J1 = Tinv * A1s * T
    J2 = Tinv * A2s * T

    return T, (J0, J1, J2), λ
end

# Closed-form solve of the GW2020 Eq. (22) Lyapunov equations for the off-diagonal blocks of P_k,
# using the structure of J₀'s lower-right block; B_k is the block-diagonal part of K_k (Eq. 21).
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

# Splitting recurrence (GW2020 Eqs. 14–19): P_k, B_k for k = 0..kmax+2, each order by `_lyap_solve` of K_k.
function _split_recurrence(J::NTuple{3,SMatrix{6,6,ComplexF64}}, λ::ComplexF64, kmax::Int)
    N = kmax + 3                          # store k = 0..kmax+2
    P = Vector{SMatrix{6,6,ComplexF64,36}}(undef, N)
    B = Vector{SMatrix{6,6,ComplexF64,36}}(undef, N)

    Z6 = zero(SMatrix{6,6,ComplexF64})
    P[1] = SMatrix{6,6,ComplexF64}(I)     # P_0 = I
    B[1] = J[1]                            # B_0 = J_0

    for k in 1:(kmax+2)
        kk = k + 1                         # 1-based slot for B/P at index k
        # K_k starts as J_k (only k = 1, 2 contribute), else 0
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
        Bk, Pk = _lyap_solve(Kacc, λ)
        B[kk] = Bk
        P[kk] = Pk
    end
    return P, B
end

# Reduced 2×2 construction (GW2020 Eqs. 25–52): Q_k, C_k via K2_k, shearing D_k, lowest-order Y₀
# with exponents r±, and the series Y_k = Y₀Z_k.
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

        Qm[k+1] = @SMatrix ComplexF64[
            0 0
            -K2acc[1, 1] -K2acc[1, 2]
        ]
        Cm[k+1] = @SMatrix ComplexF64[
            0 0
            K2acc[2, 1] K2acc[1, 1]+K2acc[2, 2]
        ]
    end

    # Build D_k for k = 0..kmax — GW2020 Eq. (43); c_2^21 = −(E+F+H+2) is Eq. (39).
    D = Vector{SMatrix{2,2,ComplexF64,4}}(undef, kmax + 1)
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

    # Lowest-order Y solution and inverse — GW2020 Eq. (48), with exponents r± from Eq. (49).
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

    # Z_k and Y_k = Y₀Z_k from the GW2020 Eqs. (50)–(52) recurrence, with E_l = Y₀⁻¹D_lY₀ (Eq. 51).
    Y = Vector{SMatrix{2,2,ComplexF64,4}}(undef, kmax + 1)
    Z = Vector{SMatrix{2,2,ComplexF64,4}}(undef, kmax + 1)
    Z[1] = I2
    Y[1] = Y0
    for k in 1:kmax
        Zacc = Z2
        for l in 1:k
            El = Y0inv * D[l+1] * Y0
            Zacc = Zacc + El * Z[k-l+1]
        end
        Zk = MMatrix{2,2,ComplexF64}(Zacc)
        for i in 1:2, j in 1:2
            Zk[i, j] = Zk[i, j] / (R[j] - R[i] - 2 * k)
        end
        Z[k+1] = SMatrix{2,2,ComplexF64}(Zk)
        Y[k+1] = Y0 * Z[k+1]
    end

    return Qm, Y
end

"""
    build_asymptotics(params::GGJParameters, Q::ComplexF64; kmax::Int=8) -> InnerAsymptoticsCache

Construct the `inps` Wasow asymptotic basis (GW2020 Eqs. 7–53), truncating
each power series at order `kmax`. Evaluate with [`evaluate_asymptotics`](@ref);
pick a matching radius with [`pick_smax`](@ref).
"""
function build_asymptotics(params::GGJParameters, Q::ComplexF64; kmax::Int=8)
    p1v = p1(params)
    # Mercier Frobenius exponents r± = 3/2 ± √(−D_I) — GW2020 Eq. (49).
    R = SVector{2,Float64}(p1v + 1.5, -p1v + 1.5)

    T, J, λ = _build_tjmat(params, Q)
    P, B = _split_recurrence(J, λ, kmax)
    Qm, Y = _coefs(B, R, kmax)
    return InnerAsymptoticsCache(params, Q, kmax, R, T, J, P, Qm, Y)
end

build_asymptotics(params::GGJParameters, Q::Number; kmax::Int=8) =
    build_asymptotics(params, ComplexF64(Q); kmax=kmax)

# Horner sum y[i] = (Σ_{k=0..n} c[i,k+1] zᵏ)·z^rvec[i] and its z-derivative, z = 1/x² (GW2020 Eq. 14 form);
# rvec = −R/2 gives the x^R Mercier prefactor. c is n_components × (n+1); complex z uses the principal branch.
function _horner(x::Number, c::AbstractMatrix{ComplexF64};
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

# Pack Y, and Qmat with P[3:6, 1:2], column-major into `_horner`'s n_components × (kmax+1) layout.
function _pack_y_coefs(cache::InnerAsymptoticsCache)
    kmax = cache.kmax
    cc = Matrix{ComplexF64}(undef, 4, kmax + 1)
    @inbounds for k in 0:kmax
        Yk = cache.Y[k+1]
        # Column-major reshape of (2,2) → 4 entries:
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
        # dd[1:4] from Q (column-major reshape of 2×2)
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

# Evaluator: U = TPQSY of GW2020 Eq. (53).

"""
    evaluate_asymptotics(cache::InnerAsymptoticsCache, x::Number; derivative=true, apply_T=true) -> (U, dU)

Evaluate the inps asymptotic basis at `x`, real (`x > 0`) or complex on a
rotated contour (principal branch of `x^r`; the series is in 1/x², so valid for |arg x| < π/2).
Returns the 6×2 complex matrix `U` whose two columns are the power-like (Mercier) solutions of
the GGJ system (column 1 large, column 2 small), and (if `derivative=true`) the
6×2 matrix `dU` of their derivatives `dU/dx`.

If `apply_T=false`, the result is left in the J-rotated basis (used by `asymptotic_residual`).
The default `apply_T=true` returns it in the GW2020 Eq. (2) state basis, which
[`physical_ua_dua`](@ref) converts to (Ψ, Ξ, Υ).
"""
function evaluate_asymptotics(cache::InnerAsymptoticsCache, x::Number;
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

    # Shearing matrix S = diag(1, 1/x²) — GW2020 Eq. (41); here 1/x² = xfac.
    smat = @SMatrix ComplexF64[1 0; 0 xfac]

    # U = T·P·Q·S·Y, the asymptotic solution of GW2020 Eq. (53) (T applied below).
    qsy = q * smat * y
    U = pp * qsy

    if derivative
        # Horner derivatives are with respect to xfac = 1/x²; d(xfac)/dx = −2·xfac/x.
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

"""
    physical_ua_dua(cache, x::Number) -> (ua, dua)

Convert the inps 6×2 basis at real or complex `x` to physical `(Ψ, Ξ, Υ)`
values and x-derivatives, each 3×2 (column 1 = large power solution, 2 = small).
The inps columns are in the GW2020 Eq. (2) state `u = (xΨ, Ξ, Υ, (xΨ)'/x, Ξ'/x, Υ'/x)`,
hence the /x and ·x factors; same map as the deltac `inpso_get_ua/dua` convention.
"""
function physical_ua_dua(cache::InnerAsymptoticsCache, x::Number)
    U, dU = evaluate_asymptotics(cache, x; derivative=true, apply_T=true)
    ua = zeros(ComplexF64, 3, 2)
    dua = zeros(ComplexF64, 3, 2)
    @inbounds for j in 1:2
        ua[1, j] = U[1, j] / x
        ua[2, j] = U[2, j]
        ua[3, j] = U[3, j]
        dua[1, j] = U[4, j] - U[1, j] / (x * x)
        dua[2, j] = U[5, j] * x
        dua[3, j] = U[6, j] * x
    end
    return ua, dua
end

"""
    asymptotic_residual(cache::InnerAsymptoticsCache, x::Number) -> SVector{2,Float64}

Convergence measure of the asymptotic basis at real or complex `x` for each of the two algebraic
columns, the analogue of GW2020 Eq. 54 in the J-rotated basis. Returns
`‖dU − x·matrix·U‖∞ / max(‖dU‖∞, ‖x·matrix·U‖∞)` per column, where
`matrix = J₀ + xfac·J₁ + xfac²·J₂` is the J-rotated coefficient matrix (the
residual of `v' = xJv`, GW2020 Eq. 6).
"""
function asymptotic_residual(cache::InnerAsymptoticsCache, x::Number)
    U, dU = evaluate_asymptotics(cache, x; derivative=true, apply_T=false)

    xfac = 1.0 / (x * x)
    M = cache.J[1]
    if cache.kmax > 0
        M = M + xfac * cache.J[2]
    end
    if cache.kmax > 1
        M = M + xfac * xfac * cache.J[3]
    end

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
    pick_smax(params, Q; θ=angle(Q)/4, eps=1e-9, kmax=12, cache=nothing,
              slogmin=-1.0, slogmax=6.5, dslog=0.01) -> (S, cache, achieved)

Sweep `s` log-uniformly upward and return the smallest `s` at which the series residual
(GW2020 Eq. 54) along `x = e^{iθ}s` drops below `eps`. If never reached, returns the
residual-minimizing `s` with `achieved = false`.
"""
function pick_smax(params::GGJParameters, Q::ComplexF64;
    θ::Float64=angle(Q) / 4, eps::Float64=1e-9, kmax::Int=12,
    cache::Union{Nothing,InnerAsymptoticsCache}=nothing,
    slogmin::Float64=-1.0, slogmax::Float64=6.5, dslog::Float64=0.01)
    c = cache === nothing ? build_asymptotics(params, Q; kmax=kmax) : cache
    ph = cis(θ)
    best_s = 10.0^slogmin
    best_r = Inf
    slog = slogmin
    while slog <= slogmax
        s = 10.0^slog
        r = maximum(asymptotic_residual(c, ph * s))
        if r < eps
            return s, c, true
        end
        if r < best_r
            best_r = r
            best_s = s
        end
        slog += dslog
    end
    return best_s, c, false
end

pick_smax(params::GGJParameters, Q::Number; kwargs...) =
    pick_smax(params, ComplexF64(Q); kwargs...)
