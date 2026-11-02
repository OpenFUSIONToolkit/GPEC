"""
    TestKineticMatrices(; scale)

Synthetic X-shaped kinetic matrices for exercising the kinetic code path in tests, with no
physical meaning: `scale` times the ideal matrices' Frobenius norms (Ak, Dk, Hk Hermitian; Bk, Ck,
Ek not). Pass it as `ForceBalanceProblem(eq; kinetic=TestKineticMatrices(; scale))`; a deck cannot
select it.

## Fields

  - `scale::Float64` - Size of the test matrices relative to the ideal matrix norms.
"""
@kwdef struct TestKineticMatrices
    scale::Float64
end

"""
    _build_x_matrix(mpert, mlow, sigma; hermitian=true)

Build an mpert×mpert X-shaped matrix: σ on the diagonal, and on the m ↔ -m anti-diagonal
i·sign(m)·σ (`hermitian=true`, for Ak, Dk, Hk; Logan 2015 Eqs 7.30, 7.33, 7.35) or
sign(m)·σ (`hermitian=false`, for Bk, Ck, Ek; Eqs 7.31, 7.32, 7.34).
"""
function _build_x_matrix(mpert::Int, mlow::Int, sigma::Float64; hermitian::Bool=true)
    W = zeros(ComplexF64, mpert, mpert)
    for i in 1:mpert
        m_i = mlow + i - 1
        W[i, i] = sigma

        # Anti-diagonal partner: m_j = -m_i
        j = -m_i - mlow + 1
        if 1 <= j <= mpert && i != j
            if hermitian
                W[i, j] = im * sign(m_i) * sigma
            else
                W[i, j] = sign(m_i) * sigma
            end
        end
    end
    return W
end

"""
    fixed_kinetic_matrices(mpert, np, mpsi, sigma, mlow, mats, xs)

Build X-shaped test kinetic energy matrices for all 6 components, each scaled by `sigma` times the
Frobenius norm of the corresponding ideal matrix at each ψ, so σ is a dimensionless strength:

| Component | Matrix | Coupling | Hermitian | Relative to |
|:--------- |:------ |:-------- |:--------- |:----------- |
| 1         | Ak     | Wz†·Wz   | Yes       | ‖A(ψ)‖_F    |
| 2         | Bk     | Wz†·Wx   | No        | ‖B(ψ)‖_F    |
| 3         | Ck     | Wz†·Wy   | No        | ‖C(ψ)‖_F    |
| 4         | Dk     | Wx†·Wx   | Yes       | ‖D(ψ)‖_F    |
| 5         | Ek     | Wx†·Wy   | No        | ‖E(ψ)‖_F    |
| 6         | Hk     | Wy†·Wy   | Yes       | ‖H(ψ)‖_F    |

Torque matrices are zero. Returns `(kw_flat, kt_flat)`, each of shape `(mpsi, np^2, 6)`.
"""
function fixed_kinetic_matrices(
    mpert::Int, np::Int, mpsi::Int, sigma::Float64, mlow::Int,
    mats::MatrixSplines, xs::Vector{Float64}
)
    kw_flat = zeros(ComplexF64, mpsi, np^2, 6)
    kt_flat = zeros(ComplexF64, mpsi, np^2, 6)

    ideal_splines = [mats.ideal.A_spline, mats.ideal.B_spline, mats.ideal.C_spline, mats.ideal.D_spline_prim, mats.ideal.E_spline_prim, mats.ideal.H_spline]
    # Ak, Dk, Hk (X†X) are Hermitian; Bk, Ck, Ek (X†Y) are not. Logan 2015 p.169 lists "Ak, Ck, Hk", a typo.
    is_hermitian = [true, false, false, true, false, true]

    X_herm = _build_x_matrix(mpert, mlow, 1.0; hermitian=true)
    X_nonherm = _build_x_matrix(mpert, mlow, 1.0; hermitian=false)

    hint = Ref(1)
    for ipsi in 1:mpsi
        psi = xs[ipsi]
        for ic in 1:6
            ideal_mat = reshape(ideal_splines[ic](psi; hint=hint), np, np)
            norm_ideal = norm(ideal_mat)  # Frobenius norm

            X = is_hermitian[ic] ? X_herm : X_nonherm
            # Tile the mpert×mpert pattern over each n block, scaled by σ‖ideal(ψ)‖_F
            W = zeros(ComplexF64, np, np)
            for jn in 0:(np÷mpert-1)
                offset = jn * mpert
                W[(offset+1):(offset+mpert), (offset+1):(offset+mpert)] .= X
            end
            W .*= sigma * norm_ideal
            kw_flat[ipsi, :, ic] .= vec(W)
        end
    end

    return kw_flat, kt_flat
end
