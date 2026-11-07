# Equations.jl
#
# The GGJ equations as a ToroidalLayer model: the coefficient rows and weak form of GWP2016 Eq. 11, the
# Wasow basis of GW2020 Eqs. 3–10 with its closed-form splitting solve, and the far-field rates the ray
# solver sizes its contour and mesh by.

"""
    layer_rows([CT,] params::GGJParameters, Q, x) -> (Ũ, Ṽ)

GWP2016 Eq. 11 solved for the second derivatives, `u'' = Ũ·u + Ṽ·u'` for u = (Ψ, Ξ, Υ), at real or complex
`x` (used as given). `CT` is the element type (default `ComplexF64`). The Galerkin weak form scales the rows by
diag(1, Q², Q) ([`layer_weak_form`](@ref)); the ray march takes them as the lower half of its companion matrix
([`ode_matrix`](@ref)).
"""
function layer_rows(::Type{CT}, params::GGJParameters, Q::Number, x::Number) where {CT<:Complex}
    e = CT(params.E)
    f = CT(params.F)
    g = CT(params.G)
    h = CT(params.H)
    k = CT(params.K)
    q = CT(Q)
    q2 = q * q
    x2 = x * x
    # Ψ'' = Q Ψ − Q x Ξ + H Υ';  Ξ'' = −(x/Q) Ψ + (x²/Q) Ξ − ((E+F)/Q²) Υ − (H/Q²) Ψ';
    # Υ'' = −(x/Q) Ψ − Q(G−KE) Ξ + (x²/Q + Q(G+KF)) Υ + HKQ Ψ'
    #! format: off
    Ũ = SMatrix{3,3,CT}(
        q,          -x / q,             -x / q,
        -q * x,     x2 / q,             -(g - k * e) * q,
        0,          -(e + f) / q2,      x2 / q + (g + k * f) * q)
    Ṽ = SMatrix{3,3,CT}(
        0,          -h / q2,            h * k * q,
        0,          0,                  0,
        h,          0,                  0)
    #! format: on
    return Ũ, Ṽ
end

"""
    layer_weak_form(params::GGJParameters, Q, x) -> (I, U, V)

3×3 coefficients of `I·u'' − V·u' − U·u = 0` for u = (Ψ, Ξ, Υ) at real or complex `x`, with
(I, V, U) = (A, −B, −C) of GWP2016 Eqs. 12–15: the rows of [`layer_rows`](@ref) scaled by diag(1, Q², Q).
"""
function layer_weak_form(params::GGJParameters, Q::ComplexF64, x::Number)
    Ũ, Ṽ = layer_rows(params, Q, x)
    q, q2 = Q, Q * Q
    Imat = @SMatrix ComplexF64[1 0 0; 0 q2 0; 0 0 q]
    #! format: off
    U = @SMatrix ComplexF64[
        Ũ[1, 1]         Ũ[1, 2]         0
        Ũ[2, 1] * q2    Ũ[2, 2] * q2    Ũ[2, 3] * q2
        Ũ[3, 1] * q     Ũ[3, 2] * q     Ũ[3, 3] * q
    ]
    V = @SMatrix ComplexF64[
        0               0               Ṽ[1, 3]
        Ṽ[2, 1] * q2    0               0
        Ṽ[3, 1] * q     0               0
    ]
    #! format: on
    return Imat, U, V
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

# GW2020's closed-form split: J₀'s exponential block is diag(−λ, −λ, λ, λ) in the basis of Eq. 7.
struct GW2020Split
    λ::ComplexF64
end

wasow_basis(p::GGJParameters, Q::ComplexF64) = ((T, J, λ) = _build_tjmat(p, Q); (T, J, GW2020Split(λ)))
split_block(s::GW2020Split, K::SMatrix{6,6,ComplexF64}) = _lyap_solve(K, s.λ)

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

# The exponent of the outer solutions, ±x²/(2√Q), is real on the ray at θ = arg(Q)/4.
default_ray_angle(::GGJParameters, Q::ComplexF64) = angle(Q) / 4

# Parabolic decay rate Re(e^{2iθ}/√Q) of the double exponential pair along the ray.
parabolic_rate(::GGJParameters, Q::ComplexF64, θ::Real) = real(cis(2θ) / sqrt(Q))

# Fast Ohm pair |√Q| and Υ-family pair √(|Q|·|G + KF|).
ohm_rate(::GGJParameters, Q::ComplexF64) = abs(sqrt(Q))
fast_rates(params::GGJParameters, Q::ComplexF64) = ohm_rate(params, Q) + sqrt(abs(Q) * abs(params.G + params.K * params.F))
