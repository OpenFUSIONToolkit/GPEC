# Equations.jl
#
# The CCC layer as a ToroidalLayer model: Kim, Cho & Choi (1988) Eqs. (1a)–(1c) as coefficient rows and weak form,
# their Wasow basis with a general Sylvester splitting solve, and the rates of the two exponential families of
# their far field (Kim Eq. 2), which the ray solver sizes its contour and mesh by.

"""
    layer_rows([CT,] params::CCCParameters, Q, x) -> (Ũ, Ṽ)

Kim, Cho & Choi (1988) Eqs. (1a)–(1c) solved for the second derivatives, `u'' = Ũ·u + Ṽ·u'` for u = (Ψ, Ξ, Υ):
GWP2016 Eq. 11 with the electron diamagnetic terms iQ*(Ψ − xΥ) in Ohm's law and vorticity, the weight M⊥/M on QΥ''
and the coupling iQ*KQ³Ξ'' in the pressure equation, which enters the Υ'' row through Ξ''.
"""
function layer_rows(::Type{CT}, params::CCCParameters, Q::Number, x::Number) where {CT<:Complex}
    g = params.ggj
    e = CT(g.E)
    f = CT(g.F)
    gg = CT(g.G)
    h = CT(g.H)
    k = CT(g.K)
    iqs = CT(0, params.Qstar)
    m = real(CT)(params.mperp)
    q = CT(Q)
    q2 = q * q
    x2 = x * x
    # Ψ'' = (Q + iQ*) Ψ − Q x Ξ − iQ* x Υ + H Υ';
    # Ξ'' = −(x/Q)(1 + iQ*/Q) Ψ + (x²/Q) Ξ + (iQ* x² − E − F)/Q² Υ − (H/Q²) Ψ';
    # Υ'' = [−x Ψ − Q²(G−KE) Ξ + (x² + Q²(G+KF)) Υ + HKQ² Ψ']/(mQ) − c Ξ'', with c = iQ*KQ²/m
    ξ1 = (-x / q) * (1 + iqs / q)
    ξ2 = x2 / q
    ξ3 = -(e + f) / q2 + iqs * x2 / q2
    ξ4 = -h / q2
    c = iqs * k * q2 / m
    #! format: off
    Ũ = SMatrix{3,3,CT}(
        q + iqs,    ξ1,     -x / (m * q) - c * ξ1,
        -q * x,     ξ2,     -(gg - k * e) * q / m - c * ξ2,
        -iqs * x,   ξ3,     x2 / (m * q) + (gg + k * f) * q / m - c * ξ3)
    Ṽ = SMatrix{3,3,CT}(
        0,          ξ4,     h * k * q / m - c * ξ4,
        0,          0,      0,
        h,          0,      0)
    #! format: on
    return Ũ, Ṽ
end

"""
    layer_weak_form(params::CCCParameters, Q, x) -> (I, U, V)

Kim (1988) Eqs. (1a)–(1c) as `I·u'' − V·u' − U·u = 0`: the rows of [`layer_rows`](@ref) multiplied by the u''
coefficient matrix I = [1 0 0; 0 Q² 0; 0 iQ*KQ³ (M⊥/M)Q], which is lower triangular and independent of x.
"""
function layer_weak_form(params::CCCParameters, Q::ComplexF64, x::Number)
    Ũ, Ṽ = layer_rows(params, Q, x)
    a = ComplexF64(0, params.Qstar) * params.ggj.K * Q^3   # iQ*KQ³ Ξ'' in the Υ row
    mq = params.mperp * Q
    Imat = @SMatrix ComplexF64[1 0 0; 0 Q*Q 0; 0 a mq]
    return Imat, Imat * Ũ, Imat * Ṽ
end

# T, T⁻¹ and J_i = T⁻¹A_iT from A₀–A₂ (GW2020 Eqs. 3–10) with Kim's Q* and m = M⊥/M terms. T's columns 3–6 span the
# exponential subspace of A₀ in a basis regular in Q* and m that equals GW2020 Eq. 7 at Q* = 0, m = 1, so J₀ is
# block-diagonal everywhere and its exponential block is diagonal only at the GGJ point.
function _build_tjmat(p::CCCParameters, Q::ComplexF64)
    g = p.ggj
    e = ComplexF64(g.E)
    f = ComplexF64(g.F)
    h = ComplexF64(g.H)
    gg = ComplexF64(g.G)
    k = ComplexF64(g.K)
    iqs = ComplexF64(0, p.Qstar)
    qs2 = p.Qstar^2
    m = p.mperp
    q = Q
    q2 = q * q
    λ = 1 / sqrt(q)
    hqm = h * q * m
    hkq3 = iqs * k * h * q * q2

    #! format: off
    T = @SMatrix ComplexF64[
        1 0 hqm q2/λ+hkq3 hqm hkq3-q2/λ
        0 0 0 -1/λ 0 1/λ
        0 0 -1/λ 0 1/λ 0
        0 1 -h/λ -q2 h/λ -q2
        0 0 0 1 0 1
        0 0 1 0 1 0
    ]

    Tinv = @SMatrix ComplexF64[
        1 q2 0 0 -hkq3 -hqm
        0 0 -h 1 q2 0
        0 0 -λ/2 0 0 1/2
        0 -λ/2 0 0 1/2 0
        0 0 λ/2 0 0 1/2
        0 λ/2 0 0 1/2 0
    ]
    #! format: on

    # A₀ and A₂: GW2020 Eq. (4); A₁: Eq. (5); Q* and m terms from Kim (1988) Eqs. (1a)–(1c).
    A0 = zeros(ComplexF64, 6, 6)
    A1 = zeros(ComplexF64, 6, 6)
    A2 = zeros(ComplexF64, 6, 6)

    A0[4, 2] = -q
    A0[4, 3] = -iqs
    A0[5, 2] = 1 / q
    A0[5, 3] = iqs / q2
    A0[6, 2] = -iqs * k * q / m
    A0[6, 3] = 1 / (m * q) + k * qs2 / m
    A0[1, 4] = 1
    A0[2, 5] = 1
    A0[3, 6] = 1
    A0[4, 6] = h

    A1[4, 1] = q + iqs
    A1[5, 1] = -1 / q - iqs / q2
    A1[6, 1] = -1 / (m * q) + iqs * k * q / m - k * qs2 / m
    A1[6, 2] = -(gg - k * e) * q / m
    A1[5, 3] = -(e + f) / q2
    A1[6, 3] = (gg + k * f) * q / m + iqs * k * (e + f) / m
    A1[4, 4] = 1
    A1[5, 4] = -h / q2
    A1[6, 4] = h * k * q / m + h * k * iqs / m
    A1[5, 5] = -1
    A1[6, 6] = -1

    A2[4, 1] = -2
    A2[5, 1] = h / q2
    A2[6, 1] = -(h * k * q / m + h * k * iqs / m)

    J0 = Tinv * SMatrix{6,6,ComplexF64}(A0) * T
    J1 = Tinv * SMatrix{6,6,ComplexF64}(A1) * T
    J2 = Tinv * SMatrix{6,6,ComplexF64}(A2) * T
    return T, (J0, J1, J2)
end

# The GW2020 Eq. (22) splitting equations for a general exponential block J22 = J₀[3:6, 3:6], kept as its LU.
struct SylvesterSplit{F}
    J22lu::F
end

function wasow_basis(p::CCCParameters, Q::ComplexF64)
    T, J = _build_tjmat(p, Q)
    return T, J, SylvesterSplit(lu(Matrix(J[1][3:6, 3:6])))
end

# With J011 = [0 1; 0 0]: J22·p21 − p21·J011 + K21 = 0 and J011·p12 − p12·J22 + K12 = 0, solved column by column
# and row by row; B is the block-diagonal part of K (GW2020 Eq. 21).
function split_block(s::SylvesterSplit, K::SMatrix{6,6,ComplexF64})
    Bm = zeros(ComplexF64, 6, 6)
    Pm = zeros(ComplexF64, 6, 6)
    Bm[1:2, 1:2] .= K[1:2, 1:2]
    Bm[3:6, 3:6] .= K[3:6, 3:6]
    p21_1 = -(s.J22lu \ Vector(K[3:6, 1]))
    p21_2 = s.J22lu \ (p21_1 - Vector(K[3:6, 2]))
    Pm[3:6, 1] .= p21_1
    Pm[3:6, 2] .= p21_2
    p12_2 = transpose(s.J22lu) \ Vector(K[2, 3:6])
    p12_1 = transpose(s.J22lu) \ (Vector(K[1, 3:6]) + p12_2)
    Pm[2, 3:6] .= p12_2
    Pm[1, 3:6] .= p12_1
    return SMatrix{6,6,ComplexF64}(Bm), SMatrix{6,6,ComplexF64}(Pm)
end

"""
    parabolic_scales(params::CCCParameters, Q) -> SVector{2,ComplexF64}

Scales σⱼ of the two exponential families exp(±x²/(2σⱼ)) at large x: the roots σ² = z of
z² − (1 + m + KQQ*²)Q z + mQ² = 0, Kim, Cho & Choi (1988) Eq. (2) with m = M⊥/M; (√Q, √m·√Q) at Q* = 0.
"""
function parabolic_scales(p::CCCParameters, Q::ComplexF64)
    m = p.mperp
    b = 1 + m + p.ggj.K * Q * p.Qstar^2
    d = sqrt(b * b - 4 * m)
    return SVector(sqrt(Q * (b + d) / 2), sqrt(Q * (b - d) / 2))
end

# The ray centred between the two families, θ = (arg σ₁ + arg σ₂)/4, so both decay along it.
default_ray_angle(p::CCCParameters, Q::ComplexF64) = ((σ1, σ2) = parabolic_scales(p, Q);
(angle(σ1) + angle(σ2)) / 4)

# Slowest decay rate min_j Re(e^{2iθ}/σⱼ) and fastest local rate max_j |e^{2iθ}/σⱼ| of the two families.
parabolic_rate(p::CCCParameters, Q::ComplexF64, θ::Real) = minimum(real(cis(2θ) / σ) for σ in parabolic_scales(p, Q))
parabolic_resolution_rate(p::CCCParameters, Q::ComplexF64, θ::Real) = maximum(abs(cis(2θ) / σ) for σ in parabolic_scales(p, Q))

# Both families decay at every angle in [0, θ]; the condition is linear in the angle, so the end points decide it.
ray_angle_admissible(p::CCCParameters, Q::ComplexF64, θ::Real) = parabolic_rate(p, Q, 0.0) > 0 && parabolic_rate(p, Q, θ) > 0

# Fast Ohm pair |√(Q + iQ*)| and Υ-family pair √(|Q(G+KF) + iKQ*(E+F)|/m).
ohm_rate(p::CCCParameters, Q::ComplexF64) = abs(sqrt(Q + ComplexF64(0, p.Qstar)))
function fast_rates(p::CCCParameters, Q::ComplexF64)
    g = p.ggj
    return ohm_rate(p, Q) + sqrt(abs(Q * (g.G + g.K * g.F) + ComplexF64(0, p.Qstar) * g.K * (g.E + g.F)) / p.mperp)
end
