"""
    FixedKineticMatrices

Test fixtures: synthetic X-shaped kinetic energy matrices keyed off the ideal
matrices' Frobenius norms. Used by `make_kinetic_matrix` when
`ctrl.kinetic_source == "fixed"` to exercise the kinetic-MHD code path before
the calculated NTV pipeline (see `KineticForces.compute_calculated_kinetic_matrices`)
is wired in.
"""

"""
    _build_x_matrix(mpert, mlow, sigma; hermitian=true)

Build an mpert×mpert X-shaped matrix with diagonal σ and anti-diagonal entries.

If `hermitian=true`, anti-diagonal entries are imaginary: W[i,j] = i·sign(m_i)·σ,
which preserves W = W† (Hermiticity). This is appropriate for components Ak, Dk, Hk
which are self-adjoint (X†X form, Logan 2015 Eqs 7.30, 7.33, 7.35).

If `hermitian=false`, anti-diagonal entries are real: W[i,j] = sign(m_i)·σ,
which breaks Hermiticity (W ≠ W†). This is appropriate for cross-term components Bk, Ck, Ek
(Eqs 7.31, 7.32, 7.34 of Logan 2015) which are not self-adjoint in general.
"""
function _build_x_matrix(mpert::Int, mlow::Int, sigma::Float64; hermitian::Bool=true)
    W = zeros(ComplexF64, mpert, mpert)
    for i in 1:mpert
        m_i = mlow + i - 1
        W[i, i] = sigma

        # Anti-diagonal: find j such that m_j = -m_i
        j = -m_i - mlow + 1
        if 1 <= j <= mpert && i != j
            if hermitian
                # Imaginary: W[i,j] = i·sign(m_i)·σ preserves W = W†
                W[i, j] = im * sign(m_i) * sigma
            else
                # Real: W[i,j] = sign(m_i)·σ breaks Hermiticity
                W[i, j] = sign(m_i) * sigma
            end
        end
    end
    return W
end

"""
    fixed_kinetic_matrices(mpert, mpsi, sigma, mlow, ffit, xs)

Build X-shaped fixed kinetic energy matrices for testing all 6 components.

Populates all 6 components of the W (energy) matrices with X-shaped patterns
scaled by `sigma` **relative to the Frobenius norm of the corresponding ideal
matrix** at each ψ. This makes σ a dimensionless perturbation strength that is
portable across equilibria:

| Component | Matrix | Coupling | Hermitian | Relative to |
|:--------- |:------ |:-------- |:--------- |:----------- |
| 1         | Ak     | Wz†·Wz   | Yes       | ‖A(ψ)‖_F    |
| 2         | Bk     | Wz†·Wx   | No        | ‖B(ψ)‖_F    |
| 3         | Ck     | Wz†·Wy   | No        | ‖C(ψ)‖_F    |
| 4         | Dk     | Wx†·Wx   | Yes       | ‖D(ψ)‖_F    |
| 5         | Ek     | Wx†·Wy   | No        | ‖E(ψ)‖_F    |
| 6         | Hk     | Wy†·Wy   | Yes       | ‖H(ψ)‖_F    |

Hermitian components use imaginary anti-diagonal entries (i·sign(m)·σ);
non-Hermitian components use real anti-diagonal entries (sign(m)·σ).

Torque matrices (T) are all zero (torque requires finite rotation frequency).

Returns `(kw_flat, kt_flat)` where each is `(mpsi, mpert^2, 6)`.
"""
function fixed_kinetic_matrices(
    mpert::Int, mpsi::Int, sigma::Float64, mlow::Int,
    ffit::FourFitVars, xs::Vector{Float64}
)
    np = ffit.numpert_total
    kw_flat = zeros(ComplexF64, mpsi, np^2, 6)
    kt_flat = zeros(ComplexF64, mpsi, np^2, 6)

    # Map component index → ideal matrix spline and Hermiticity
    # (component_index, ideal_spline, is_hermitian)
    ideal_splines = [ffit.amats, ffit.bmats, ffit.cmats, ffit.dmats_prim, ffit.emats_prim, ffit.hmats]
    # Ak, Dk, Hk are Hermitian: X†X is trivially self-adjoint.
    # The thesis (Logan 2015 p.169) lists "Ak, Ck, Hk" but this appears to be a typo
    # for "Ak, Dk, Hk" — confirmed by inspecting Fortran PENTRC output where Ck ≠ Ck†.
    # Bk, Ck, Ek are cross terms (X†Y) and not Hermitian in general.
    is_hermitian = [true, false, false, true, false, true]

    # Build unit X-pattern matrices (Hermitian and non-Hermitian variants)
    X_herm = _build_x_matrix(mpert, mlow, 1.0; hermitian=true)
    X_nonherm = _build_x_matrix(mpert, mlow, 1.0; hermitian=false)

    hint = Ref(1)
    for ipsi in 1:mpsi
        psi = xs[ipsi]
        for ic in 1:6
            ideal_mat = reshape(ideal_splines[ic](psi; hint=hint), np, np)
            norm_ideal = norm(ideal_mat)  # Frobenius norm

            X = is_hermitian[ic] ? X_herm : X_nonherm
            # Scale: σ × ‖ideal(ψ)‖_F × unit X-pattern
            # For multi-n, tile the mpert×mpert X-pattern into the np×np block
            W = zeros(ComplexF64, np, np)
            for jn in 0:(ffit.numpert_total÷mpert-1)
                offset = jn * mpert
                W[(offset+1):(offset+mpert), (offset+1):(offset+mpert)] .= X
            end
            W .*= sigma * norm_ideal
            kw_flat[ipsi, :, ic] .= vec(W)
        end
    end

    return kw_flat, kt_flat
end
