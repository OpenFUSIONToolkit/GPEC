"""
Coordinate-invariant (root-area-weighted field) flux-surface operators.

These building blocks implement the √area weighting of Pharr (2026),
"Coordinate-invariant flux-surface Fourier analysis in tokamaks". They are the
single source of truth for the map between a flux-surface field component and its
root-area-weighted (coordinate-invariant) field representation, shared by the
ForceFreeStates and PerturbedEquilibrium modules.

The central operator is

    rootareafield_to_flux = sqrtamat · √jarea

which maps a root-area-weighted field `b̃` to the coordinate flux harmonics `Φ`:
`Φ = rootareafield_to_flux · b̃` (so `b̃ = rootareafield_to_flux⁻¹ · Φ`). The singular values of any operator
expressed in the `b̃` basis are independent of the straight-field-line
(working) coordinate.

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
    control_surface_rootareafield_to_flux(equil, psi, ft) -> Matrix{ComplexF64}

Build the root-area-weighted field → flux operator `rootareafield_to_flux = sqrtamat · √jarea` at the flux
surface `psi`, where `jarea = ∫ J|∇ψ| dθ` is the scalar flux-surface area.

`rootareafield_to_flux` maps a root-area-weighted field `b̃` to the coordinate flux harmonics `Φ`:
`Φ = rootareafield_to_flux · b̃`. To express a flux-space operator/generator in the coordinate-invariant
`b̃` basis (and back) use:
  - operator   `Ã = rootareafield_to_flux⁻¹ · A · rootareafield_to_flux`     (e.g. permeability `Φ_tot = A·Φ_x`)
  - generator  `G̃ = rootareafield_to_flux⁻¹ · G · rootareafield_to_flux⁻†`   (e.g. inductance, energy = Φ†·G⁻¹·Φ)
  - row map    `C̃ = C · rootareafield_to_flux`              (e.g. coupling, scalar = C·Φ_x)
The singular values / spectrum in the `b̃` basis are coordinate-invariant.
"""
function control_surface_rootareafield_to_flux(
    equil::PlasmaEquilibrium,
    psi::Float64,
    ft::Utilities.FourierTransforms.FourierTransform
)
    sqrtamat = compute_sqrtamat(equil, psi, ft)
    jarea = flux_surface_area(equil, psi, ft.mtheta)
    return sqrtamat .* sqrt(jarea)
end
