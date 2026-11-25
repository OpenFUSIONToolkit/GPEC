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
convention (positive for the usual outward-decreasing profiles). Pedestal models compare γ with half of an
effective ω_*pi built from this profile; the conventions in use are in the stability documentation.
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
# The A, C and H integrands at one (ψ, θ) point, weighted by Jμ₀ρ/B², and B² itself; `g` holds the stored J·g_ij and J.
function _inertia_integrands(g11, g22, g33, g23, g31, g12, jac, q, chi1, mu0rho)
    gradpsi2 = g22 * g33 - g23^2
    gradpsi_theta = g23 * g31 - g12 * g33
    gradpsi_zeta = g12 * g23 - g22 * g31
    gradtheta2 = g11 * g33 - g31^2
    gradzeta2 = g11 * g22 - g12^2
    gradtheta_zeta = g31 * g12 - g11 * g23
    b2 = chi1^2 * (g22 + 2q * g23 + q^2 * g33) / jac
    weight = jac * mu0rho / b2
    return weight * gradpsi2, weight * chi1 * (gradpsi_zeta - q * gradpsi_theta), weight * chi1^2 * (gradzeta2 - 2q * gradtheta_zeta + q^2 * gradtheta2), b2
end

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
            a, c, h, _ = _inertia_integrands((@view metric.fs[ipsi, jtheta, 1:7])..., q, chi1, mu0rho)
            integrands[ipsi, jtheta, 1] = a
            integrands[ipsi, jtheta, 2] = c
            integrands[ipsi, jtheta, 3] = h
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

  - `gamma::Float64` - Growth rate in s⁻¹ of the least-stable free-boundary mode with perpendicular inertia only, the exact ideal-MHD
    eigenvalue for adiabatic index Γ = 0 and an upper bound on the Γ = 5/3 rate; 0 when the plasma is stable or the root lies below
    `gamma_sq_min` (see `below_floor`).
  - `gamma_sq::Float64` - γ² at which the least-stable free-boundary energy eigenvalue crosses zero.
  - `converged::Bool` - Whether the root iteration closed its bracket to `rtol` (always true for a stable plasma); false when `maxiter`
    ran out, in which case `gamma` is the last iterate and `bracket` holds the root, when any solve flagged an ambiguous step in its
    conjugate-point count, or when the solve at the root crossed a kinetic singular surface.
  - `below_floor::Bool` - True when the marginal solve is unstable but nothing grows faster than `gamma_sq_min`, so `gamma` is 0 for
    an unstable plasma.
  - `bracket::NTuple{2,Float64}` - Final γ² bracket (lo, hi) of the root; (0, 0) when stable, (0, hi) when `below_floor`.
  - `inertia::PerpendicularInertia` - The inertia source the solves used.
  - `config::Union{Nothing,GrowthRateConfig}` - The deck settings when the solve came from a `[GrowthRate]` section or the three-argument method; `nothing` otherwise.
  - `omega_star_pi::Float64` - Peak ion diamagnetic frequency |ω_*pi| in rad/s over the integration domain, at the toroidal n of the
    least-stable free-boundary mode of the solve at `gamma_sq` (of the marginal solve when stable), from the equilibrium's kinetic
    profiles (NaN without n_e, T_e and T_i). No diamagnetic threshold is applied to `gamma`; with the EPED1 convention (effective
    ω_*pi = half the peak) a mode counts as unstable when `gamma > omega_star_pi/4`, and the stability documentation lists the
    other conventions. The peak is taken over `[psilow, psilim]`, so a truncated domain can cut off part of a pedestal gradient.
  - `marginal::ForceFreeStatesResult` - The ideal (γ² = 0) solve, whose least-stable energy decides stability.
  - `result::Union{Nothing,ForceFreeStatesResult}` - The solve at `gamma_sq`, carrying the eigenmode; `nothing` when stable.
  - `history::Vector` - Every (γ², λ_min, nzero, nneg) evaluated, in order, with λ_min the least-stable free-boundary energy at that
    γ², `nzero` the exact conjugate-point count of that solve and `nneg` its number of negative free-boundary energy eigenvalues;
    `nzero + nneg` is the number of modes growing faster than γ.
"""
struct IdealGrowthRateResult
    gamma::Float64
    gamma_sq::Float64
    converged::Bool
    below_floor::Bool
    bracket::NTuple{2,Float64}
    inertia::PerpendicularInertia
    config::Union{Nothing,GrowthRateConfig}
    omega_star_pi::Float64
    marginal::ForceFreeStatesResult
    result::Union{Nothing,ForceFreeStatesResult}
    history::Vector{@NamedTuple{gamma_sq::Float64, lambda::Float64, nzero::Int, nneg::Int}}
end

"""
    _sign_change_root(lambda_at, lo, hi, flo, fhi; rtol, ftol, scale, maxiter) -> (x, converged, lo, hi)

Root of `lambda_at` in the bracket `(lo, hi)` with `flo < 0 < fhi`, by the Illinois iteration (the retained
end's value is halved when it is kept twice running) safeguarded by geometric bisection. The interpolated
point is replaced by `sqrt(lo * hi)` when it is not strictly inside the bracket, which happens in floating
point when one end value dwarfs the other, or when the same end has been kept three times running.
Stops when the bracket is narrower than `rtol * hi` and returns the end with the smaller |λ|. An iterate
with `|lambda_at(x)| <= ftol * scale` is followed by a probe `rtol/2` across it, so a sign change there
closes the bracket at once; `ftol` alone never stops the iteration, since `scale` can be large next to a pole.
"""
function _sign_change_root(lambda_at, lo::Float64, hi::Float64, flo::Float64, fhi::Float64; rtol::Real, ftol::Real, scale::Real, maxiter::Int)
    best(lo, hi, tlo, thi) = abs(tlo) <= abs(thi) ? lo : hi
    tlo, thi = flo, fhi   # true end values; flo and fhi carry the Illinois halving
    x, side, kept, probe = hi, 0, 0, NaN
    for _ in 1:maxiter
        if isnan(probe)
            x = (lo * fhi - hi * flo) / (fhi - flo)
            (kept >= 3 || !(lo < x < hi)) && (x = sqrt(lo * hi))
        else
            x, probe = probe, NaN
        end
        lam = lambda_at(x)
        newside = lam < 0 ? -1 : 1
        kept = newside == side ? kept + 1 : 1
        if lam < 0
            lo, flo, tlo = x, lam, lam
            side == -1 && (fhi /= 2)
        else
            hi, fhi, thi = x, lam, lam
            side == 1 && (flo /= 2)
        end
        side = newside
        hi - lo <= rtol * hi && return best(lo, hi, tlo, thi), true, lo, hi
        if abs(lam) <= ftol * scale
            p = lam < 0 ? x * (1 + rtol / 2) : x * (1 - rtol / 2)
            lo < p < hi && (probe = p)
        end
    end
    return best(lo, hi, tlo, thi), false, lo, hi
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
(Illinois) iteration, safeguarded by bisection, brings it to zero, stopping when the bracket is narrower than `rtol`
in γ², within `maxiter` steps (a warning and `converged = false` otherwise). Once |λ_min| falls below `ftol` times
the reference |λ_min| (the marginal value, or the bracket's negative end when the marginal energy is positive), a
probe `rtol/2` across the iterate closes the bracket in one more solve. Every evaluation is one force-balance solve.
`gamma_sq_min` is the floor below which a root is not resolved (`gamma` = 0 with `below_floor` set when the marginal
solve is unstable), `gamma_sq_max` the ceiling beyond which bracketing stops with an error.
When the count bisection reaches `rtol` with a crossing still present, the fastest mode is a fixed-boundary
one (its edge displacement vanishes) and the bracket's upper end is reported as converged.

The model keeps the perpendicular inertia only, as ELITE does [Wilson 2002, Eq. 9]. The energy principle has
already been minimized over the parallel displacement, so γ is the exact eigenvalue of ideal MHD with adiabatic
index Γ = 0, and an upper bound on the Γ = 5/3 rate; the two agree only at marginal stability.

The number of modes faster than γ is the Morse index of δW + γ²K with a free end point, the number of conjugate
points plus the number of negative free-boundary energy eigenvalues, which holds because K is positive definite on
(ξ_s, Ξ) [Goedbloed & Sakanaka 1974]. Each `history` entry carries both terms.

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
regular with inertia). A result whose solve at the root crossed such a surface is warned about and marked not
converged.
"""
function growth_rate(prob::ForceBalanceProblem, alg::Forward; inertia::PerpendicularInertia,
    gamma_sq_init::Real=1e9, gamma_sq_min::Real=1e4, gamma_sq_max::Real=1e16, rtol::Real=1e-3, ftol::Real=1e-6, maxiter::Int=30,
    marginal::Union{Nothing,ForceFreeStatesResult}=nothing, config::Union{Nothing,GrowthRateConfig}=nothing, verbose::Bool=false)
    get(prob.ctrl_kwargs, :kinetic, nothing) === nothing ||
        error("growth_rate needs an ideal marginal problem: remove its kinetic source, the inertia is the only kinetic term")
    history = @NamedTuple{gamma_sq::Float64, lambda::Float64, nzero::Int, nneg::Int}[]
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
    entry(r, gamma_sq) = (gamma_sq=gamma_sq, lambda=lambda_min(r), nzero=r.diagnostics.nzero_conj, nneg=count(<(0), real.(r.free_boundary.et)))
    faster(h) = h.nzero + h.nneg   # modes growing faster than γ (Morse index); non-increasing in γ²
    singfac_min = get(prob.ctrl_kwargs, :singfac_min, ForceFreeStatesControl().singfac_min)
    solves = Dict{Float64,ForceFreeStatesResult}()   # the latest solve on each side of the root: only a bracket end can be returned
    bracket_end = Dict{Bool,Float64}()   # faster(h) > 0 → the γ² of that side's retained solve
    function finish(gamma_sq, converged; bracket=(gamma_sq, gamma_sq), below_floor=false)
        r = get(solves, gamma_sq, nothing)
        if r !== nothing && singfac_min > 0 && r.kinetic.kmsing > 0
            @warn "growth_rate: the solve at the root placed $(r.kinetic.kmsing) kinetic singular-surface crossing(s); " *
                  "rerun with singfac_min = 0. The result is marked not converged"
            converged = false
        end
        return IdealGrowthRateResult(sqrt(gamma_sq), gamma_sq, converged && !flagged[], below_floor, bracket, inertia, config,
            _peak_omega_star_pi(something(r, marginal)), marginal, r, history)
    end

    if marginal !== nothing && !counted(marginal)
        verbose && @info "growth_rate: the supplied marginal solve has no conjugate-point count; solving it again with the count"
        marginal = nothing
    end
    marginal = marginal === nothing ? counted_solve(prob) : marginal
    marginal.free_boundary === nothing && error("growth_rate needs the free-boundary energies: set vac_flag=true")
    check_flags(marginal, 0.0)
    push!(history, entry(marginal, 0.0))
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
        push!(history, entry(r, gamma_sq))
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
            verbose && @info @sprintf("growth_rate: unstable, but nothing grows faster than γ² = %.2e, below gamma_sq_min", 10hi)
            return finish(0.0, true; bracket=(0.0, 10hi), below_floor=true)
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
        return finish(hi, true; bracket=(lo, hi))
    end

    scale = history[1].lambda < 0 ? abs(history[1].lambda) : abs(h_lo.lambda)
    function lambda_at(gamma_sq)
        h = evaluate(gamma_sq)
        h.nzero == 0 || error("growth_rate: an interior zero crossing appeared inside the bracket at γ² = $gamma_sq; the free-boundary energy is not monotone there")
        return h.lambda
    end
    gamma_sq, converged, lo, hi = _sign_change_root(lambda_at, lo, hi, h_lo.lambda, h_hi.lambda; rtol, ftol, scale, maxiter)
    converged || @warn "growth_rate: no convergence in $maxiter steps; γ² lies in [$lo, $hi] and the last iterate $gamma_sq is reported"
    return finish(gamma_sq, converged; bracket=(lo, hi))
end

function growth_rate(prob::ForceBalanceProblem, alg::Forward, config::GrowthRateConfig; marginal::Union{Nothing,ForceFreeStatesResult}=nothing, verbose::Bool=false)
    return growth_rate(prob, alg; inertia=inertia_source(config, prob.equil), config.gamma_sq_init, config.gamma_sq_min, config.gamma_sq_max,
        config.rtol, config.ftol, config.maxiter, marginal, config, verbose)
end
