"""
    PerpendicularInertia(; rho)

Perpendicular plasma inertia for the force-balance equations, the ELITE inertia model: the kinetic
energy K = ½∫ρ|ξ_⊥|² dV with the parallel inertia dropped (incompressible displacement). With the
normal displacement Ξ = ξ·∇ψ and the in-surface variable ξ_s = (ξ×B)·e_ψ of the energy principle
[Glasser 2016 Eqs. 8-9], ξ×B = ξ_s∇ψ + Ξu with u = χ′(∇ζ − q∇θ), so |ξ_⊥|² = |ξ×B|²/B² is algebraic
in the two retained variables and K adds the Hermitian bands ⟨μ₀ρ|∇ψ|²/B²⟩ to A, ⟨μ₀ρ u·∇ψ/B²⟩ to
C and ⟨μ₀ρ|u|²/B²⟩ to H of [Glasser 2016 Eq. A6], each scaled by γ² for a mode ∝ e^{γt}
(δW + γ²K is stationary).

Pass it as the `kinetic` source of a [`ForceBalanceProblem`](@ref) with `kinetic_factor = γ²` in
s⁻², or let [`growth_rate`](@ref) find the γ² at which the free-boundary energy vanishes. The F, K,
G reduction then takes the kinetic (non-Q-factorized) path, on which the rational surfaces are
regular points for γ² > 0 and no ideal crossings are needed.

## Fields

  - `rho` - Mass density in kg/m³: a number, or a function of normalized flux ψ_N.
"""
@kwdef struct PerpendicularInertia
    rho::Union{Float64,Function}
end

_rho_at(src::PerpendicularInertia, psi::Real) = src.rho isa Function ? Float64(src.rho(psi)) : src.rho

"""
    inertia_kinetic_matrices(src::PerpendicularInertia, equil, intr, metric) -> (kw_flat, kt_flat)

The perpendicular-inertia bands on the metric's (ψ, θ) grid in the (ψ, mode pair, component) layout
of [`KineticMatrices`](@ref): components A, C and H carry ⟨μ₀ρ|∇ψ|²/B²⟩, ⟨μ₀ρ u·∇ψ/B²⟩ and
⟨μ₀ρ|u|²/B²⟩ per unit γ², the others and the torque matrices are zero. The metric stores
J·g_ij with g_ij = e_i·e_j/J² [Glasser 2016 Eq. A1], so the contravariant products are cofactors:
|∇ψ|² = g22g33 − g23², ∇ψ·∇θ = g23g31 − g12g33, ∇ψ·∇ζ = g12g23 − g22g31, |∇θ|² = g11g33 − g31²,
|∇ζ|² = g11g22 − g12², ∇θ·∇ζ = g31g12 − g11g23 (all in the stored J·g units), and
B² = χ′²(g22 + 2q g23 + q²g33)/J from B = (χ′/J)(e_θ + q e_ζ) [Glasser 2016 Eq. 5].
"""
function inertia_kinetic_matrices(src::PerpendicularInertia, equil::Equilibrium.PlasmaEquilibrium, intr::ForceFreeStatesInternal, metric::MetricData)
    mpsi, mtheta, mpert = metric.mpsi, metric.mtheta, intr.mpert
    chi1 = 2π * equil.psio
    integrands = zeros(Float64, mpsi, mtheta, 3)
    for ipsi in 1:mpsi
        q = equil.profiles.q_spline(metric.xs[ipsi])
        mu0rho = Utilities.MU_0 * _rho_at(src, metric.xs[ipsi])
        for jtheta in 1:mtheta
            g11, g22, g33, g23, g31, g12, jac = @view metric.fs[ipsi, jtheta, 1:7]
            gradpsi2 = g22 * g33 - g23^2
            gradpsi_theta = g23 * g31 - g12 * g33
            gradpsi_zeta = g12 * g23 - g22 * g31
            gradtheta2 = g11 * g33 - g31^2
            gradzeta2 = g11 * g22 - g12^2
            gradtheta_zeta = g31 * g12 - g11 * g23
            b2 = chi1^2 * (g22 + 2q * g23 + q^2 * g33) / jac
            weight = jac * mu0rho / b2
            integrands[ipsi, jtheta, 1] = weight * gradpsi2
            integrands[ipsi, jtheta, 2] = weight * chi1 * (gradpsi_zeta - q * gradpsi_theta)
            integrands[ipsi, jtheta, 3] = weight * chi1^2 * (gradzeta2 - 2q * gradtheta_zeta + q^2 * gradtheta2)
        end
    end
    fc = Utilities.FourierCoefficients(metric.xs, metric.ys, integrands, mpert - 1)

    np = intr.numpert_total
    kw_flat = zeros(ComplexF64, mpsi, np^2, 6)
    kt_flat = zeros(ComplexF64, mpsi, np^2, 6)
    band = zeros(ComplexF64, 2mpert - 1)
    for ipsi in 1:mpsi, (qty, ic) in ((1, 1), (2, 3), (3, 6))
        # Same band layout as the ideal matrices: coefficient m at mpert−m, conjugates fill the other half
        for m in 0:(mpert-1)
            band[mpert-m] = Utilities.get_complex_coeff(fc, ipsi, m, qty)
        end
        for k in 1:(mpert-1)
            band[k+mpert] = conj(band[mpert-k])
        end
        for ipert_n in 1:intr.npert, ipert_m in 1:mpert, jpert_m in 1:mpert
            ipert = ipert_m + (ipert_n - 1) * mpert
            jpert = jpert_m + (ipert_n - 1) * mpert
            kw_flat[ipsi, ipert+(jpert-1)*np, ic] = band[jpert_m-ipert_m+mpert]
        end
    end
    return kw_flat, kt_flat
end

"""
    GrowthRateResult

The outcome of [`growth_rate`](@ref).

## Fields

  - `gamma::Float64` - Growth rate in s⁻¹ of the least-stable free-boundary mode; 0 when the plasma is marginally stable or stable.
  - `gamma_sq::Float64` - γ² at which the least-stable free-boundary energy eigenvalue crosses zero.
  - `marginal::ForceFreeStatesResult` - The ideal (γ² = 0) solve, whose least-stable energy decides stability.
  - `result::Union{Nothing,ForceFreeStatesResult}` - The solve at `gamma_sq`, carrying the eigenmode; `nothing` when stable.
  - `history::Vector` - Every (γ², λ_min, nzero) evaluated, in order, with λ_min the least-stable free-boundary energy at that γ².
"""
struct GrowthRateResult
    gamma::Float64
    gamma_sq::Float64
    marginal::ForceFreeStatesResult
    result::Union{Nothing,ForceFreeStatesResult}
    history::Vector{@NamedTuple{gamma_sq::Float64, lambda::Float64, nzero::Int}}
end

"""
    growth_rate(prob::ForceBalanceProblem, alg::Forward; inertia, gamma_sq_init=1e9, gamma_sq_min=1e4, gamma_sq_max=1e16,
                rtol=1e-3, verbose=false) -> GrowthRateResult

Growth rate of the least-stable free-boundary mode of `prob` with the perpendicular inertia `inertia`
(a [`PerpendicularInertia`](@ref)), ELITE's eigenvalue problem solved by shooting: the ideal solve
gives the marginal least-stable energy λ_min(0); when it is negative, the inertia enters as
`kinetic_factor = γ²` and λ_min(γ²), which rises monotonically with γ² once the interior is
positive, is brought to zero. The root is bracketed by decades from `gamma_sq_init` and closed by
Brent's method to a relative tolerance `rtol` in γ²; every evaluation is one force-balance solve.
`gamma_sq_min` is the floor below which the mode counts as marginal, `gamma_sq_max` the ceiling
beyond which bracketing stops with an error.

The problem's `singfac_min` is set to 0 for the inertial solves unless given: with inertia the
rational surfaces are regular, so the integration runs as a single chunk. A fixed-boundary mode
faster than the free-boundary one shows as `nzero > 0` at the root in `history`; this first version
reports the free-boundary root and leaves the interior count to the caller.
"""
function growth_rate(prob::ForceBalanceProblem, alg::Forward; inertia::PerpendicularInertia,
    gamma_sq_init::Real=1e9, gamma_sq_min::Real=1e4, gamma_sq_max::Real=1e16, rtol::Real=1e-3, verbose::Bool=false)
    history = @NamedTuple{gamma_sq::Float64, lambda::Float64, nzero::Int}[]
    lambda_min(r) = minimum(real, r.free_boundary.et)

    marginal = solve(prob, alg; verbose)
    marginal.free_boundary === nothing && error("growth_rate needs the free-boundary energies: set vac_flag=true")
    push!(history, (gamma_sq=0.0, lambda=lambda_min(marginal), nzero=marginal.diagnostics.nzero))
    verbose && @info @sprintf("growth_rate: marginal λ_min = %.6e (nzero = %d)", history[end].lambda, history[end].nzero)
    history[end].lambda < 0 || return GrowthRateResult(0.0, 0.0, marginal, nothing, history)

    kwargs = copy(prob.ctrl_kwargs)
    haskey(kwargs, :singfac_min) || (kwargs[:singfac_min] = 0.0)
    kwargs[:kinetic] = inertia
    matrices = Ref{Any}(nothing)   # the inertia bands, evaluated once and reused at every γ²
    last = Ref{Any}(nothing)
    function lambda_at(gamma_sq)
        kw = copy(kwargs)
        matrices[] === nothing || (kw[:kinetic] = matrices[])
        kw[:kinetic_factor] = gamma_sq
        inertial = ForceBalanceProblem(prob.equil, prob.wall, kw)
        matrices[] === nothing && (matrices[] = kinetic_matrices(inertial, alg; verbose))
        r = solve(inertial, alg; verbose)
        last[] = r
        push!(history, (gamma_sq=Float64(gamma_sq), lambda=lambda_min(r), nzero=r.diagnostics.nzero))
        verbose && @info @sprintf("growth_rate: γ² = %.4e  λ_min = %.6e  nzero = %d", gamma_sq, history[end].lambda, history[end].nzero)
        return history[end].lambda
    end

    # Bracket by decades: down while already positive, up while still negative
    hi = Float64(gamma_sq_init)
    lam = lambda_at(hi)
    while lam > 0
        hi /= 10
        hi < gamma_sq_min && return GrowthRateResult(sqrt(hi), hi, marginal, last[], history)
        lam = lambda_at(hi)
    end
    lo = hi
    while lam <= 0
        lo = hi
        hi *= 10
        hi > gamma_sq_max && error("growth_rate: λ_min is still negative at γ² = $lo; raise gamma_sq_max or check the inertia density")
        lam = lambda_at(hi)
    end
    gamma_sq = find_zero(lambda_at, (lo, hi), Roots.Brent(); xrtol=rtol)
    return GrowthRateResult(sqrt(gamma_sq), gamma_sq, marginal, last[], history)
end
