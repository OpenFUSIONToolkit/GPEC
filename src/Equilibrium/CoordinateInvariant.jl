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
recovered, when ever needed, as the scalar product `Φ = A·b̄` — it is never stored.

See `scripts/test_power_norm_invariance.jl` for the numerical invariance proof of
the underlying √weight identity and angle map.
"""

"""
    compute_sqrt_jac_delpsi(equil, psi, mtheta) -> Vector{Float64}

Compute √(J·|∇ψ|) at `mtheta` equally-spaced θ points on the flux surface at `psi`.
This is the √weight function that maps a field component `b(θ)` to its
root-area-weighted form `√(J|∇ψ|)·b(θ)` in θ-space.
"""
function compute_sqrt_jac_delpsi(equil::PlasmaEquilibrium, psi::Float64, mtheta::Int)
    sqrt_jac_delpsi = Vector{Float64}(undef, mtheta)

    hint2d = (Ref(1), Ref(1))
    for itheta in 0:(mtheta-1)
        theta = itheta / mtheta  # normalized to [0, 1)
        m = flux_surface_metric(equil, psi, theta; hint=hint2d)
        sqrt_jac_delpsi[itheta+1] = sqrt(abs(m.jac * m.delpsi))
    end

    return sqrt_jac_delpsi
end

"""
    compute_sqrtamat(equil, psi, ft) -> Matrix{ComplexF64}

Build the √A convolution matrix `sqrtamat`. Produces a Hermitian Toeplitz matrix
with entries sqrtamat[m',k] = ŵ_{m_k − m'} where ŵ_n = (1/N)Σ w_j exp(+inθ_j) and
w(θ) = √(J·|∇ψ|).

Operationally, sqrtamat is the mode-space √weight operator: for a field b with
Fourier coefficients b_fft, it satisfies the identity
  `‖sqrtamat·b_fft‖² = N² · ∫ |b|² · J|∇ψ| dθ`
which is Jacobian-invariant on a given flux surface (see
`scripts/test_power_norm_invariance.jl`).

The backward step uses exp(−imθ)/(1/N) normalization paired with the Julia
forward FT exp(+imθ) so that round-trip = identity and the convolution
structure is correct.
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

        # Standard backward FT: f(θ_j) = (1/N) Σ_m c_m exp(-imθ_j)
        # exp(-imθ) = cos(mθ) - i·sin(mθ), so:
        #   Re(f) = (1/N)(cslth·Re(c) + snlth·Im(c))
        #   Im(f) = (1/N)(cslth·Im(c) - snlth·Re(c))
        real_part = (ft.cslth * real.(e_k) .+ ft.snlth * imag.(e_k)) ./ mtheta
        imag_part = (ft.cslth * imag.(e_k) .- ft.snlth * real.(e_k)) ./ mtheta
        theta_vec = complex.(real_part, imag_part)

        # Multiply pointwise by √(J·|∇ψ|) in theta-space
        theta_vec .*= sqrt_jdp

        # Forward FT: theta-space → mode-space (exp(+imθ), Julia convention)
        sqrtamat[:, k] .= ft(theta_vec)
    end

    return sqrtamat
end

"""
    rootarea_to_area_weight(equil, psi, ft) -> Matrix{ComplexF64}

Build the root-area-weighted → area-weighted field operator `Σ/√A = sqrtamat ./ √jarea` at the
flux surface `psi`, where `jarea = ∫ J|∇ψ| dθ` is the scalar flux-surface area. It maps the
coordinate-invariant root-area-weighted field `b̃` to the area-weighted field `b̄`: `b̄ = (Σ/√A)·b̃`
(both in tesla). Poloidal flux is the scalar product `Φ = A·b̄` (so `Φ = Σ·√A·b̃`), recovered only
when a user supplies/requests flux — it is never stored.

The √-weight (`b̃`) basis is the one in which operator singular values / spectra are independent of
the straight-field-line (working) coordinate — see `field_space_response_matrices`. `b̄` is *not*
coordinate-invariant; it is only a field/flux recovery view. [Pharr 2026]
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
`√A·Σ⁻¹ = √jarea · inv(sqrtamat)`, mapping `b̄ → b̃` (`b̃ = √A·Σ⁻¹·b̄`). [Pharr 2026]
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
