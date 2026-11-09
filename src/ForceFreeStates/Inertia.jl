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
    PerpendicularInertia(equil::Equilibrium.PlasmaEquilibrium)

The inertia of the equilibrium's attached kinetic profiles: ρ(ψ_N) = μ m_p n_e(ψ_N), with μ the mean ion mass per
electron of the main ions and the impurity at the local Z_eff, the density the SLAYER layer uses. Errors when no
kinetic profiles are attached (set `kinetic_file` or call `attach_kinetic_profiles!`).
"""
function PerpendicularInertia(equil::Equilibrium.PlasmaEquilibrium)
    data = InnerLayer._layer_kinetic_data(equil, "mass density ρ")
    _require_profiles(data, "the inertia density ρ = μ m_p n_e")
    loaded = InnerLayer.SLAYER._slayer_profiles(data, nothing)
    mu = InnerLayer.SLAYER._layer_mass_per_electron(data, InnerLayer.SLAYER._layer_zeff(loaded, data))
    n_e = loaded.profiles.n_e
    return PerpendicularInertia(; rho=psi -> (mu isa Real ? mu : mu(psi)) * Utilities.M_P * n_e(psi))
end

"""
    ion_diamagnetic_frequency(equil::Equilibrium.PlasmaEquilibrium, n::Integer; z_i=1) -> ψ_N -> ω_*pi

The ion diamagnetic frequency in rad/s for toroidal mode number `n`, ω_*pi = n (dp_i/dΨ)/(Z_i e n_i) with Ψ the
poloidal flux per radian, from the attached kinetic profiles with n_i ∝ n_e and p_i = n_i T_i, the layer
convention (positive for the usual outward-decreasing profiles). The EPED1 criterion counts a mode as unstable
when γ > ω_*pi/2 at the pedestal.
"""
function ion_diamagnetic_frequency(equil::Equilibrium.PlasmaEquilibrium, n::Integer; z_i::Real=1)
    data = InnerLayer._layer_kinetic_data(equil, "ion diamagnetic frequency ω_*pi")
    _require_profiles(data, "ω_*pi")
    p = InnerLayer.SLAYER._slayer_profiles(data, nothing).profiles
    scale = -n / (z_i * equil.psio)
    return psi -> scale * (p.T_i(psi) * p.n_e(psi; deriv=DerivOp(1)) / p.n_e(psi) + p.T_i(psi; deriv=DerivOp(1)))
end

# The shared kinetic-profile loader needs n_e, T_e and T_i even where only some of them enter.
_has_profiles(data) = data !== nothing && data.n_e !== nothing && data.T_e !== nothing && data.T_i !== nothing
function _require_profiles(data, what::AbstractString)
    _has_profiles(data) || throw(ArgumentError("$what needs kinetic data with n_e, T_e and T_i (the shared profile loader reads all three)"))
    return data
end

# Toroidal mode number of the least-stable free-boundary mode of `r` (its lowest n without free-boundary energies).
_least_stable_n(r::ForceFreeStatesResult) = r.free_boundary === nothing ? r.nlow : r.nlow + r.free_boundary.n_tor_idx[1]

# Peak |ω_*pi| over the integration domain of `r` at its least-stable mode's n; NaN without the kinetic profiles.
function _peak_omega_star_pi(r::ForceFreeStatesResult)
    _has_profiles(r.equil.kinetic) || return NaN
    omega = ion_diamagnetic_frequency(r.equil, _least_stable_n(r))
    psis = filter(x -> r.psilow <= x <= r.psilim, r.metric.xs)
    return isempty(psis) ? NaN : maximum(abs ∘ omega, psis)
end

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
        rho = _rho_at(src, metric.xs[ipsi])
        isfinite(rho) && rho > 0 || error("PerpendicularInertia: ρ(ψ_N = $(metric.xs[ipsi])) = $rho; the mass density must be finite and positive")
        mu0rho = Utilities.MU_0 * rho
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
    GrowthRateConfig(; rho=NaN, gamma_sq_init=1e9, gamma_sq_min=1e4, gamma_sq_max=1e16, rtol=1e-3, ftol=1e-6, maxiter=30)

The `[GrowthRate]` deck section: its presence runs [`growth_rate`](@ref) after the force-balance stage, and the
same settings drive the three-argument `growth_rate` method.

## Fields

  - `rho` - Uniform mass density in kg/m³; NaN (the default) derives ρ(ψ_N) from the equilibrium's kinetic profiles.
  - `gamma_sq_init` - First γ² in s⁻² tried when bracketing the root.
  - `gamma_sq_min` - γ² floor in s⁻² below which the mode counts as marginal.
  - `gamma_sq_max` - γ² ceiling in s⁻² beyond which bracketing stops with an error.
  - `rtol` - Relative width of the γ² bracket at which the root iteration stops.
  - `ftol` - Fraction of the marginal |λ_min| below which |λ_min| counts as zero.
  - `maxiter` - Maximum root-iteration steps after bracketing.
"""
@kwdef struct GrowthRateConfig
    rho::Float64 = NaN
    gamma_sq_init::Float64 = 1e9
    gamma_sq_min::Float64 = 1e4
    gamma_sq_max::Float64 = 1e16
    rtol::Float64 = 1e-3
    ftol::Float64 = 1e-6
    maxiter::Int = 30
end

# The inertia a deck section asks for: uniform, or from the equilibrium's kinetic profiles.
inertia_source(config::GrowthRateConfig, equil::Equilibrium.PlasmaEquilibrium) =
    isnan(config.rho) ? PerpendicularInertia(equil) : PerpendicularInertia(; rho=config.rho)

"""
    IdealGrowthRateResult

The outcome of [`growth_rate`](@ref).

## Fields

  - `gamma::Float64` - Growth rate in s⁻¹ of the least-stable free-boundary mode; 0 when the plasma is marginally stable or stable.
  - `gamma_sq::Float64` - γ² at which the least-stable free-boundary energy eigenvalue crosses zero.
  - `converged::Bool` - Whether the root iteration met its `ftol` or `rtol` stop (always true for a stable or marginal plasma); false when
    `maxiter` ran out, in which case `gamma` is the last iterate and `history` brackets the root, or when any solve flagged an ambiguous
    step in its conjugate-point count.
  - `inertia::PerpendicularInertia` - The inertia source the solves used.
  - `config::Union{Nothing,GrowthRateConfig}` - The deck settings when the solve came from a `[GrowthRate]` section or the three-argument method; `nothing` otherwise.
  - `omega_star_pi::Float64` - Peak ion diamagnetic frequency |ω_*pi| in rad/s over the integration domain, at the toroidal n of the
    least-stable free-boundary mode of the solve at `gamma_sq` (of the marginal solve when stable), from the equilibrium's kinetic
    profiles (NaN without n_e, T_e and T_i); the EPED1 criterion counts the mode as unstable when `gamma > omega_star_pi/2`. The
    peak is taken over `[psilow, psilim]`, so a truncated domain can cut off part of a pedestal gradient.
  - `marginal::ForceFreeStatesResult` - The ideal (γ² = 0) solve, whose least-stable energy decides stability.
  - `result::Union{Nothing,ForceFreeStatesResult}` - The solve at `gamma_sq`, carrying the eigenmode; `nothing` when stable.
  - `history::Vector` - Every (γ², λ_min, nzero) evaluated, in order, with λ_min the least-stable free-boundary energy at that γ² and
    `nzero` the exact conjugate-point count of that solve.
"""
struct IdealGrowthRateResult
    gamma::Float64
    gamma_sq::Float64
    converged::Bool
    inertia::PerpendicularInertia
    config::Union{Nothing,GrowthRateConfig}
    omega_star_pi::Float64
    marginal::ForceFreeStatesResult
    result::Union{Nothing,ForceFreeStatesResult}
    history::Vector{@NamedTuple{gamma_sq::Float64, lambda::Float64, nzero::Int}}
end

"""
    growth_rate(prob::ForceBalanceProblem, alg::Forward; inertia, gamma_sq_init=1e9, gamma_sq_min=1e4, gamma_sq_max=1e16,
                rtol=1e-3, ftol=1e-6, maxiter=30, marginal=nothing, config=nothing, verbose=false) -> IdealGrowthRateResult
    growth_rate(prob::ForceBalanceProblem, alg::Forward, config::GrowthRateConfig; marginal=nothing, verbose=false)

Growth rate of the fastest ideal mode of `prob` with the perpendicular inertia `inertia` (a
[`PerpendicularInertia`](@ref)), ELITE's eigenvalue problem solved by shooting. The inertia enters as
`kinetic_factor = γ²`; at each γ² the number of modes growing faster than γ is the number of conjugate
points `nzero` plus one when the least-stable free-boundary energy λ_min is negative, and that count cannot
rise with γ² since the kinetic energy is positive. The ideal solve gives the count at γ = 0; when it is
zero the plasma is stable and `gamma` is 0. Otherwise the count is bracketed by decades from
`gamma_sq_init` (down while nothing is faster, up while something still is). An interior crossing still
present at the lower end is a fixed-boundary eigenvalue inside the bracket, where λ_min has a pole; since
the fastest mode is free-boundary and at least as fast as any fixed-boundary one, the lower end is lifted
above it by bisecting on the count. Then only λ_min changes sign in the bracket and a bracketed secant
(Illinois) iteration brings it to zero, stopping when |λ_min| falls below `ftol` times the reference
|λ_min| (the marginal value, or the bracket's negative end when the marginal energy is positive) or the
bracket is narrower than `rtol` in γ², within `maxiter` steps (a warning and `converged = false`
otherwise). Every evaluation is one force-balance solve. `gamma_sq_min` is the floor below which the mode
counts as marginal (`gamma` = 0), `gamma_sq_max` the ceiling beyond which bracketing stops with an error.
When the count bisection reaches `rtol` with a crossing still present, the fastest mode is a fixed-boundary
one (its edge displacement vanishes) and the bracket's upper end is reported as converged.

The conjugate points are counted exactly, as the eigenphases of the Cayley transform of the solution frame
that wrap through π along ψ (its Maslov index), not by the smallest-eigenvalue detector `diagnostics.nzero`;
a step whose phase match is ambiguous is warned about and marks the result not converged. Starting
`gamma_sq_init` above the fastest growth rate expected is cheapest, since the upward scan needs only the sign of λ_min.

A `marginal` result of the same problem (an earlier ideal `solve`) is reused instead of solving it again;
`config` is recorded on the result and is what the three-argument method passes. The problem must carry no
kinetic source. The inertial solves take every other control of the problem, so a two-pass ψ grid is the
same as the marginal one; with `singfac_min > 0` the kinetic surface finder can still place crossings at
the sharp cond(F̂) peaks that appear near rational surfaces when γ²ρ is far below any physical root, where
only the sign of λ_min matters, and `singfac_min = 0` gives a single chunk (the rational surfaces are
regular with inertia).
"""
function growth_rate(prob::ForceBalanceProblem, alg::Forward; inertia::PerpendicularInertia,
    gamma_sq_init::Real=1e9, gamma_sq_min::Real=1e4, gamma_sq_max::Real=1e16, rtol::Real=1e-3, ftol::Real=1e-6, maxiter::Int=30,
    marginal::Union{Nothing,ForceFreeStatesResult}=nothing, config::Union{Nothing,GrowthRateConfig}=nothing, verbose::Bool=false)
    get(prob.ctrl_kwargs, :kinetic, nothing) === nothing ||
        error("growth_rate needs an ideal marginal problem: remove its kinetic source, the inertia is the only kinetic term")
    history = @NamedTuple{gamma_sq::Float64, lambda::Float64, nzero::Int}[]
    lambda_min(r) = minimum(real, r.free_boundary.et)
    counted_solve(p) = _solve_force_balance(p, alg; verbose, count_conjugate_points=true)
    counted(r) = r.diagnostics !== nothing && r.diagnostics.nzero_conj >= 0
    flagged = Ref(false)
    function check_flags(r, gamma_sq)
        nflag = r.diagnostics.nconj_flags
        nflag > 0 || return
        flagged[] = true
        @warn "growth_rate: the conjugate-point count at γ² = $gamma_sq had $nflag ambiguous step(s); the result is marked not converged"
    end
    faster(h) = h.nzero + (h.lambda < 0)   # modes growing faster than γ; non-increasing in γ²
    solves = Dict{Float64,ForceFreeStatesResult}()   # the latest solve on each side of the root: only a bracket end can be returned
    bracket_end = Dict{Bool,Float64}()   # faster(h) > 0 → the γ² of that side's retained solve
    function finish(gamma_sq, converged)
        r = get(solves, gamma_sq, nothing)
        return IdealGrowthRateResult(sqrt(gamma_sq), gamma_sq, converged && !flagged[], inertia, config, _peak_omega_star_pi(something(r, marginal)), marginal, r, history)
    end

    if marginal !== nothing && !counted(marginal)
        verbose && @info "growth_rate: the supplied marginal solve has no conjugate-point count; solving it again with the count"
        marginal = nothing
    end
    marginal = marginal === nothing ? counted_solve(prob) : marginal
    marginal.free_boundary === nothing && error("growth_rate needs the free-boundary energies: set vac_flag=true")
    check_flags(marginal, 0.0)
    push!(history, (gamma_sq=0.0, lambda=lambda_min(marginal), nzero=marginal.diagnostics.nzero_conj))
    verbose && @info @sprintf("growth_rate: marginal λ_min = %.6e (nzero = %d)", history[end].lambda, history[end].nzero)
    faster(history[1]) == 0 && return finish(0.0, true)

    kwargs = copy(prob.ctrl_kwargs)
    kwargs[:kinetic] = inertia
    kwargs[:kinetic_factor] = Float64(gamma_sq_init)
    kwargs[:kinetic] = kinetic_matrices(ForceBalanceProblem(prob.equil, prob.wall, kwargs), alg; verbose)   # the bands once, reused at every γ²
    seen = Dict{Float64,Int}()   # γ² → history index; each γ² is one solve
    function evaluate(gamma_sq)
        gamma_sq = Float64(gamma_sq)
        haskey(seen, gamma_sq) && return history[seen[gamma_sq]]
        kw = copy(kwargs)
        kw[:kinetic_factor] = gamma_sq
        r = counted_solve(ForceBalanceProblem(prob.equil, prob.wall, kw))
        check_flags(r, gamma_sq)
        push!(history, (gamma_sq=gamma_sq, lambda=lambda_min(r), nzero=r.diagnostics.nzero_conj))
        seen[gamma_sq] = length(history)
        below = faster(history[end]) > 0   # this γ² is below the fastest root
        haskey(bracket_end, below) && delete!(solves, bracket_end[below])
        bracket_end[below] = gamma_sq
        solves[gamma_sq] = r
        verbose && @info @sprintf("growth_rate: γ² = %.4e  λ_min = %.6e  nzero = %d", gamma_sq, history[end].lambda, history[end].nzero)
        return history[end]
    end

    # Bracket the count by decades: down while nothing is faster, up while something still is
    hi = Float64(gamma_sq_init)
    h_hi = evaluate(hi)
    while faster(h_hi) == 0
        hi /= 10
        if hi < gamma_sq_min
            verbose && @info @sprintf("growth_rate: nothing grows faster than γ² = %.2e, below gamma_sq_min: marginal", 10hi)
            return finish(0.0, true)
        end
        h_hi = evaluate(hi)
    end
    lo, h_lo = hi, h_hi
    while faster(h_hi) > 0
        lo, h_lo = hi, h_hi
        hi *= 10
        hi > gamma_sq_max && error("growth_rate: a mode still grows at γ² = $lo; raise gamma_sq_max or check the inertia density")
        h_hi = evaluate(hi)
    end
    # The fastest root lies in (lo, hi]. A crossing still present at lo is a fixed-boundary eigenvalue inside the bracket, where
    # λ_min has a pole; the fastest mode is free-boundary and at least as fast (interlacing), so bisect on the count, which is
    # monotone, until lo is above that eigenvalue and only the free-boundary energy changes sign in the bracket
    while h_lo.nzero > 0 && hi - lo > rtol * hi
        mid = sqrt(lo * hi)
        h_mid = evaluate(mid)
        if faster(h_mid) == 0
            hi, h_hi = mid, h_mid
        else
            lo, h_lo = mid, h_mid
        end
    end
    if h_lo.nzero > 0   # the root sits on the pole to within rtol: the fastest mode has no edge displacement
        verbose && @info @sprintf("growth_rate: the fastest mode is fixed-boundary to within rtol, γ² in [%.6e, %.6e]", lo, hi)
        return finish(hi, true)
    end

    # Illinois iteration on the sign change; the retained end's λ is halved when it is kept twice running
    scale = history[1].lambda < 0 ? abs(history[1].lambda) : abs(h_lo.lambda)
    flo, fhi = h_lo.lambda, h_hi.lambda
    gamma_sq, side, converged = hi, 0, false
    for _ in 1:maxiter
        gamma_sq = (lo * fhi - hi * flo) / (fhi - flo)
        h = evaluate(gamma_sq)
        h.nzero == 0 || error("growth_rate: an interior zero crossing appeared inside the bracket at γ² = $gamma_sq; the free-boundary energy is not monotone there")
        lam = h.lambda
        if abs(lam) <= ftol * scale
            converged = true
            break
        end
        if lam < 0
            lo, flo = gamma_sq, lam
            side == -1 && (fhi /= 2)
            side = -1
        else
            hi, fhi = gamma_sq, lam
            side == 1 && (flo /= 2)
            side = 1
        end
        if hi - lo <= rtol * hi
            converged = true
            break
        end
    end
    converged || @warn "growth_rate: no convergence in $maxiter steps; γ² lies in [$lo, $hi] and the last iterate $gamma_sq is reported"
    return finish(gamma_sq, converged)
end

function growth_rate(prob::ForceBalanceProblem, alg::Forward, config::GrowthRateConfig; marginal::Union{Nothing,ForceFreeStatesResult}=nothing, verbose::Bool=false)
    return growth_rate(prob, alg; inertia=inertia_source(config, prob.equil), config.gamma_sq_init, config.gamma_sq_min, config.gamma_sq_max,
        config.rtol, config.ftol, config.maxiter, marginal, config, verbose)
end
