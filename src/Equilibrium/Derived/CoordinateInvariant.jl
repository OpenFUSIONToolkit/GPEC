"""
Coordinate-invariant (root-area-weighted field) flux-surface operators.

These building blocks implement the √area weighting of Pharr (2026),
"Coordinate-invariant flux-surface Fourier analysis in tokamaks". They are the
single source of truth for translating a flux-surface field component between the
three field representations (all in tesla), shared by the ForceFreeStates and
PerturbedEquilibrium modules:

  - `b`  bare / normal field         (Fourier weight W = 1)
  - `b̃`  root-area-weighted field    (W = √(J|∇ψ|)) — the coordinate-invariant field
  - `b̄`  area-weighted field         (W = J|∇ψ|)

With `Σ ≡ sqrtamat` (the mpert×mpert √weight convolution) and the scalar surface area
`A ≡ jarea = ∫ J|∇ψ| dθ`, the mode-space vectors are related by

    b̃ = Σ·b,     b̄ = (Σ/√A)·b̃,     Φ = A·b̄ = Σ·√A·b̃     (poloidal flux, weber)

The √-weight (`b̃`) basis is the one in which an operator's singular values / spectra are
independent of the straight-field-line (working) coordinate; `b` and `b̄` are not
coordinate-invariant and are provided only as field/flux recovery views. Poloidal flux is
recovered, whenever needed, as the scalar product `Φ = A·b̄` — it is never stored.
"""

"""
    compute_jac_delpsi(equil, psi, mtheta) -> Vector{Float64}

Compute J·|∇ψ| at `mtheta` equally-spaced θ points on the flux surface at `psi`.
This is the area weight function that maps a bare field component `b(θ)` to its
area-weighted form `J|∇ψ|·b(θ)` in θ-space, the flux per unit-norm cell.
"""
function compute_jac_delpsi(equil::PlasmaEquilibrium, psi::Float64, mtheta::Int)
    jac_delpsi = Vector{Float64}(undef, mtheta)

    hint2d = (Ref(1), Ref(1))
    for itheta in 0:(mtheta-1)
        theta = itheta / mtheta  # normalized to [0, 1)
        m = flux_surface_metric(equil, psi, theta; hint=hint2d)
        jac_delpsi[itheta+1] = abs(m.jac * m.delpsi)
    end

    return jac_delpsi
end

"""
    compute_sqrt_jac_delpsi(equil, psi, mtheta) -> Vector{Float64}

Compute √(J·|∇ψ|) at `mtheta` equally-spaced θ points on the flux surface at `psi`.
This is the √weight function that maps a field component `b(θ)` to its
root-area-weighted form `√(J|∇ψ|)·b(θ)` in θ-space.
"""
compute_sqrt_jac_delpsi(equil::PlasmaEquilibrium, psi::Float64, mtheta::Int) = sqrt.(compute_jac_delpsi(equil, psi, mtheta))

"""
    compute_sqrtamat(equil, psi, ft) -> Matrix{ComplexF64}

Build the √A convolution matrix `sqrtamat`. Produces a Hermitian Toeplitz matrix
with entries sqrtamat[m',k] = ŵ_{m_k − m'} where ŵ_n = (1/N)Σ w_j exp(+inθ_j) and
w(θ) = √(J·|∇ψ|).

For Fourier coefficients `b_fft` of a field b it satisfies `‖sqrtamat·b_fft‖² = ∫ |b|² · J|∇ψ| dθ`
(θ normalized to one period), which is Jacobian-invariant on a given flux surface.
`sqrtamat/√jarea` is the matrix Fortran GPEC writes as `J_surf_2`.
"""
function compute_sqrtamat(
    equil::PlasmaEquilibrium,
    psi::Float64,
    ft::Utilities.FourierTransforms.FourierTransform
)
    mpert = ft.mpert
    mtheta = ft.mtheta

    sqrt_jdp = compute_sqrt_jac_delpsi(equil, psi, mtheta)
    sqrtamat = zeros(ComplexF64, mpert, mpert)

    e_k = zeros(ComplexF64, mpert)
    for k in 1:mpert
        e_k .= 0.0
        e_k[k] = 1.0 + 0.0im

        # Unit mode to θ-space, f(θ_j) = exp(+i m_k θ_j)
        theta_vec = Utilities.FourierTransforms.inverse(ft, e_k)

        theta_vec .*= sqrt_jdp

        # Back to mode space (exp(+imθ) convention)
        sqrtamat[:, k] .= ft(theta_vec)
    end

    return sqrtamat
end

"""
    rootarea_to_area_weight(equil, psi, ft) -> Matrix{ComplexF64}

Build the root-area-weighted → area-weighted field operator `Σ/√A = sqrtamat ./ √jarea` at the
flux surface `psi`, where `jarea = ∫ J|∇ψ| dθ` is the scalar flux-surface area. It maps the
coordinate-invariant root-area-weighted field `b̃` to the area-weighted field `b̄`: `b̄ = (Σ/√A)·b̃`
(both in tesla); `b̄` is not coordinate-invariant and serves only to recover field or flux
(`Φ = A·b̄`). Pharr 2026.
"""
function rootarea_to_area_weight(
    equil::PlasmaEquilibrium,
    psi::Float64,
    ft::Utilities.FourierTransforms.FourierTransform
)
    sqrtamat = compute_sqrtamat(equil, psi, ft)
    jarea = flux_surface_area(equil, psi, ft.mtheta)
    return sqrtamat ./ sqrt(jarea)
end

"""
    area_to_rootarea_weight(equil, psi, ft) -> Matrix{ComplexF64}

Inverse of [`rootarea_to_area_weight`](@ref): the area-weighted → root-area-weighted field operator
`√A·Σ⁻¹ = √jarea · inv(sqrtamat)`, mapping `b̄ → b̃` (`b̃ = √A·Σ⁻¹·b̄`). Pharr 2026.
"""
function area_to_rootarea_weight(
    equil::PlasmaEquilibrium,
    psi::Float64,
    ft::Utilities.FourierTransforms.FourierTransform
)
    sqrtamat = compute_sqrtamat(equil, psi, ft)
    jarea = flux_surface_area(equil, psi, ft.mtheta)
    return sqrt(jarea) .* inv(sqrtamat)
end
