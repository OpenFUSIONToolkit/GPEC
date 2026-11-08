# Terms.jl
#
# The terms of the steady modified Rutherford equation F(w) as pure functions of the full island width w in ψ_N
# (Benjamin et al. 2026 Eqs. 7, 10-14 and C.1; Leuthold et al. 2026 Eq. 4.6). Surface quantities are arguments,
# so each term can be checked alone.

"""
    classical_term(delta_prime, D_I, w; model=:mercier_power) -> Float64

Classical drive with the finite-pressure correction, `Δ′(w) = Δ′_mn (w/2)^(−2α₋) (−4D_I)^(1/2)` with
`α₋ = ½ − (−D_I)^(1/2)` (Benjamin Eq. 10); it reduces to `Δ′_mn` at `D_I = −¼`. `model = :constant` returns `Δ′_mn`.
"""
function classical_term(delta_prime::Real, D_I::Real, w::Real; model::Symbol=:mercier_power)
    model === :constant && return Float64(delta_prime)
    alpha_minus = 0.5 - sqrt(-D_I)
    return delta_prime * (w / 2)^(-2 * alpha_minus) * sqrt(-4 * D_I)
end

"""
    bootstrap_term(D_nc, w_d, w; k1=1.7) -> Float64

Bootstrap drive with the transport threshold, `k₁ D_nc w / (w² + (k₁/0.5) w_d²)` (Benjamin Eq. 11): tends to
`k₁ D_nc / w` for `w ≫ w_d` and to `0.5 D_nc w / w_d²` for `w ≪ w_d`.
"""
bootstrap_term(D_nc::Real, w_d::Real, w::Real; k1::Real=1.7) = k1 * D_nc * w / (w^2 + (k1 / 0.5) * w_d^2)

"""
    curvature_term(D_R, H, D_I, w_d, w; k1=1.7) -> Float64

Glasser-Greene-Johnson curvature term, `[k₁ D_R/(α₊ − H)] / [w + k₁ w_d/(0.3(1 + α₊))]` with
`α₊ = ½ + (−D_I)^(1/2)` (Benjamin Eq. 12); stabilizing for `D_R < 0`.
"""
function curvature_term(D_R::Real, H::Real, D_I::Real, w_d::Real, w::Real; k1::Real=1.7)
    alpha_plus = 0.5 + sqrt(-D_I)
    return (k1 * D_R / (alpha_plus - H)) / (w + k1 * w_d / (0.3 * (1 + alpha_plus)))
end

"""
    polarization_term(w; c_pol, eps, rho_theta_psi, beta_p, gradient_ratio, omega_hat, omega_hat_pi, w_ib) -> Float64

Ion-polarization term of Benjamin Eq. C.1,
`−c_pol 3.28 ε^(3/2) I₁ (ρ_iθ,ψ/w)² (β_p/w) G² ω̂(ω̂ − ω̂_pi) w⁴/(w⁴ + w_ib⁴)`, with `I₁ = 0.71`, the poloidal Larmor
radius `ρ_iθ,ψ` in ψ_N (Benjamin's `(ρ_iθ/r_s)/(w/2ψ_s)` with its `dψ/dr = 2ψ_s/r_s` replaced by the layer label's),
`G = (n_i⁻¹ dn_i/dψ)/(q⁻¹ dq/dψ)` and the island frequency `ω̂ = ω/ω_*i` and `ω̂_pi = ω_*pi/ω_*i` normalized to the
ion diamagnetic frequency. Zero at `ω̂ = 0` and at `ω̂ = ω̂_pi`; stabilizing for `ω̂(ω̂ − ω̂_pi) > 0`.
"""
function polarization_term(w::Real; c_pol::Real, eps::Real, rho_theta_psi::Real, beta_p::Real, gradient_ratio::Real, omega_hat::Real,
    omega_hat_pi::Real, w_ib::Real)
    return -c_pol * 3.28 * eps^1.5 * 0.71 * (rho_theta_psi / w)^2 * (beta_p / w) * gradient_ratio^2 *
           omega_hat * (omega_hat - omega_hat_pi) * w^4 / (w^4 + w_ib^4)
end

"""
    reduced_polarization_term(w; c_pol, C_pol, w_ib) -> Float64

Reduced polarization term `−c_pol w_pol²/w³` with `w_pol = C_pol w_ib`; it carries no island frequency.
"""
reduced_polarization_term(w::Real; c_pol::Real, C_pol::Real, w_ib::Real) = -c_pol * (C_pol * w_ib)^2 / w^3

"""
    transport_width(w; c_wd, chi_perp, v_te, R0, q, q1, n, m, Wc_prefac) -> Float64

Transport threshold width `w_d = c_wd [(χ⊥/χ∥(w)) W_c/m²]^(1/4)` in ψ_N (Benjamin Eq. 14 in the Schlutt & Hegna flux
form, `W_c` from [`mre_geometry`](@ref)), with `χ∥(w) = v_Te λ∥(w)` and `λ∥(w) = R₀ q/(n |dq/dψ| w)` (Benjamin p. 4).
"""
function transport_width(w::Real; c_wd::Real, chi_perp::Real, v_te::Real, R0::Real, q::Real, q1::Real, n::Integer, m::Integer, Wc_prefac::Real)
    chi_par = v_te * R0 * abs(q) / (abs(n * q1) * w)
    return c_wd * ((chi_perp / chi_par) * abs(Wc_prefac) / m^2)^0.25
end
