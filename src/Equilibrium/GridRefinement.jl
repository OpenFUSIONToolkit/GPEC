"""
Radial grid refinement utilities for the two-pass auto ψ grid.

Pass 1 forms the equilibrium on a coarse grid; `refined_psi_grid` then derives a knot
density from the measured nodal data (1D profiles, 2D geometry channels, and optional
kinetic profiles) and equidistributes it, pinning mandatory knots (e.g. rational surfaces)
via `merge_mandatory_nodes`. Pass 2 re-forms the equilibrium on the refined grid through
the `override_psi_nodes` keyword of `setup_equilibrium`.

There is ONE sizing rule everywhere — the cubic *derivative* interpolation error model
err(f′) ≈ h³|f''''|/24 ≤ τ·f_scale — applied to three sources of curvature |f''''|:

 1. *Measured* (mid-radius and edge): 5-point divided differences of the pass-1 nodal
    values. Nodal values come from independent field-line integrations, so the estimate
    is immune to inter-knot spline ringing. This source dominates wherever pass-1 data
    supports it — including the edge, where the pass-1 layout is already log-packed
    (on the DIII-D example the measured density supplies ~3× what the edge model demands).
 2. *Separatrix model* (edge floor, ψ ≥ 0.9): q ≈ −A·ln(1−ψ) fed through the same rule
    gives geometric packing with uniform relative q′ error, independent of A. Normally
    inactive; insurance against a pass-1 that under-sampled the divergence.
 3. *Axis model* (core, ψ ≤ 0.03): the equilibrium is a power law in ψ (smooth in
    ρ = √ψ) near the axis, and the same rule on that form gives geometric-in-log(ψ)
    packing. Here the model *replaces* measurement rather than flooring it — nodal data
    from the smallest flux surfaces is dominated by integration and axis-extrapolation
    error, which refinement amplifies rather than resolves.

The stability physics consumes spline derivatives (q′ at rational surfaces, p′ and V′ in
the Euler-Lagrange and ballooning coefficients), whose error at a given spacing is far
larger than the value error — hence the derivative model. Every region's knot count
scales as psi_accuracy^(-1/3), so tightening the tolerance refines edge, pedestal, and
mid proportionally.

Measured curvature is taken with respect to ρ = √ψ_N, in which the equilibrium is regular
at the magnetic axis (R−R₀ ~ √ψ_N makes the geometry channels behave as ψ_N^(k/2) there,
so their ψ-derivatives diverge under refinement but their ρ-derivatives do not); the
resulting ρ-spacing maps back to the ψ density through dψ/dρ = 2√ψ_N, giving natural √ψ
core packing outside the modeled core region as well.
"""

# Cubic spline derivative interpolation error constant: err(f′) ≈ h³|f''''|/24
const CUBIC_DERIV_ERR_CONST = 24.0
# |f''''|/f_scale below this is treated as integrator noise, not curvature. The absolute
# floor covers well-spaced grids; a relative-noise term ε/h⁴ dominates where the sample
# grid is tightly packed, since divided differences amplify per-node noise as h⁻⁴ there
# (a-priori floors carry those regions instead).
const D4_NOISE_FLOOR = 1e3
const NODE_NOISE_REL = 1e-8
# Floor on a channel's value-scale, so an all-zero profile/geometry channel cannot divide by zero
const F_SCALE_FLOOR = 1e-12
# Per-quantity target spacing clamp (ψ_N units)
const H_TARGET_MIN = 1e-4
const H_TARGET_MAX = 0.2
# Model-region splits (ψ_N): below CORE the analytic axis power-law model *replaces* measured
# curvature (smallest-surface nodal data is integration/extrapolation noise); above EDGE the
# separatrix log model *floors* it. Values also referenced in the module and _knot_density docstrings.
const CORE_MODEL_PSI_MAX = 0.03
const EDGE_MODEL_PSI_MIN = 0.9
# θ-lines subsample stride for the 2D geometry channels
const THETA_STRIDE = 8
# --- shared separatrix edge q-law -------------------------------------------------------------
# Minimum knots in the edge band before a fit is attempted.
const EDGE_FIT_MIN_KNOTS = 4
# Weak absolute floor on the diverging fit. The discrimination is done by the r2_log > r2_linear
# comparison below, not by this number: measured EFIT/CAKE reconstructions of diverted DIII-D
# plasmas fit the log law at r2 = 0.929-0.977 (reconstruction noise in the edge q), which overlaps
# the 0.972 of the limited a10 fixed-boundary deck, so no absolute threshold separates the two
# classes. The relative test does separate them cleanly (diverted 0.96 vs 0.75 linear; Solovev
# 0.760 vs 0.9996 linear). This floor only rejects fits that describe nothing at all.
const EDGE_FIT_MIN_R2 = 0.90

# Least-squares slope and coefficient of determination for y = a + b*x.
function _linfit_r2(x::Vector{Float64}, y::Vector{Float64})
    x_bar = sum(x) / length(x)
    y_bar = sum(y) / length(y)
    sxx = sum((x .- x_bar) .^ 2)
    sxx > 0 || return (NaN, NaN, NaN, NaN)
    b = sum((x .- x_bar) .* (y .- y_bar)) / sxx
    ss_res = sum((y .- (y_bar .+ b .* (x .- x_bar))) .^ 2)
    ss_tot = sum((y .- y_bar) .^ 2)
    r2 = ss_tot > 0 ? 1 - ss_res / ss_tot : NaN
    return (b, r2, x_bar, y_bar)
end

"""
    edge_q_law(equil; psi_max, psi_min=EDGE_MODEL_PSI_MIN, min_knots=EDGE_FIT_MIN_KNOTS,
               min_r2=EDGE_FIT_MIN_R2) -> nothing | (; A, q_bar, u_bar, n_knots, r2_log, r2_linear)

Least-squares fit of the separatrix edge law `q = q̄ + A·(ln(1−ψ) − ū)` over the equilibrium's
outer knots — the single shared statement of that model, used both by the grid-refinement edge
density floor and by the resistive-layer overlap scan's out-of-grid surface search.

Returns `nothing` when the diverging model does **not** describe this equilibrium's edge, so a
plasma with finite edge q is never extrapolated as if q blew up:

  - fewer than `min_knots` knots in the band;
  - `A ≥ 0`, i.e. q not rising toward ψ = 1;
  - `r2_log < min_r2` — the log law does not actually fit;
  - `r2_log ≤ r2_linear` — a plain linear-in-ψ fit explains the edge q at least as well, which is
    what a **limited** plasma looks like. This comparison carries no scale and is what separates
    the shipped limited decks (Solovev 0.760 vs 0.9996 linear; LAR 0.904 vs 0.998) from the
    diverted ones (DIII-D 0.996 vs 0.865, 0.999 vs 0.594).

This is a test of the **model**, not a topology classification: it asks whether q diverges
logarithmically here, not whether an x-point exists. Geometric x-point detection is
[`classify_topology`](@ref), which is a separate concern.
"""
function edge_q_law(equil::PlasmaEquilibrium;
    psi_max::Real=Float64(equil.profiles.xs[end]),
    psi_min::Real=EDGE_MODEL_PSI_MIN,
    min_knots::Int=EDGE_FIT_MIN_KNOTS,
    min_r2::Real=EDGE_FIT_MIN_R2)
    xs = collect(Float64, equil.profiles.xs)
    band = findall(x -> x >= psi_min && x < psi_max, xs)
    if length(band) < min_knots
        n_tail = max(min_knots, length(xs) ÷ 10)
        band = filter(i -> xs[i] < psi_max, collect(max(1, length(xs) - n_tail + 1):length(xs)))
    end
    length(band) >= min_knots || return nothing

    q = [Float64(equil.profiles.q_spline(xs[i])) for i in band]
    u = [log(1.0 - xs[i]) for i in band]
    all(isfinite, u) && all(isfinite, q) || return nothing

    A, r2_log, u_bar, q_bar = _linfit_r2(u, q)
    _, r2_linear, _, _ = _linfit_r2([xs[i] for i in band], q)
    (isfinite(A) && A < 0) || return nothing            # q must rise toward the edge
    (isfinite(r2_log) && r2_log >= min_r2) || return nothing
    (isfinite(r2_linear) && r2_log > r2_linear) || return nothing

    return (A=A, q_bar=q_bar, u_bar=u_bar, n_knots=length(band), r2_log=r2_log, r2_linear=r2_linear)
end

"""
ψ at which the edge law reaches `q_target`; closed form, no root-finding needed.
"""
edge_q_law_psi(fit, q_target::Real) = 1.0 - exp(fit.u_bar + (q_target - fit.q_bar) / fit.A)

"""
dq/dψ from the edge law: q = q̄ + A·ln(1−ψ) + const ⇒ dq/dψ = −A/(1−ψ).
"""
edge_q_law_dqdpsi(fit, psi::Real) = -fit.A / (1.0 - psi)
# Rational-surface bracketing (Δ′ robustness). The ideal-MHD Δ′ asymptotic matching samples the
# cubic equilibrium splines' 2nd/3rd derivatives across each rational ψ_s over the matching stencil
# [ψ_s − dpsi, ψ_s + dpsi], dpsi = singfac_min/|n·q′|. A cubic 3rd derivative is piecewise constant
# and jumps at every knot, so a knot inside that stencil makes q‴ inconsistent across the crossing
# and Δ′ non-convergent. We keep the stencil inside ONE clean spline interval by bracketing each
# surface with knots at ψ_s ± w (w = BRACKET_COEF·dpsi) and clearing the zone between them — knots
# stay near the surface (kinetic-layer resolution) but never on it. BRACKET_COEF > 1 leaves margin
# for the pass-1→pass-2 shift of ψ_s.
const BRACKET_COEF = 4.0
# Global minimum knot spacing. No grid — auto refinement, the near-surface bracket, or a fixed
# ldp/pow1 grid packing its edge at high mpsi — may place nodes closer than this. Below it, cubic
# high derivatives are dominated by field-line integration noise rather than curvature (e.g. an
# ldp grid at mpsi=2048 packs the edge to ~6e-7, deep in noise). Chosen well under the spacing of a
# converged uniform reference (~5e-4 at mpsi=2048) so it clips only pathological clustering.
const MIN_KNOT_SPACING = 1.0e-4
# Near-rational resolution patch. The Δ′ extraction samples the equilibrium splines' 3rd
# derivatives at each rational surface, which the smooth-q measured-curvature density starves at
# mid-radius (q looks smooth there, so few knots are placed, yet Δ′ still needs fine local
# resolution). Within RATIONAL_RES_RADIUS of each rational we floor the density at 1/RATIONAL_RES_SPACING
# so equidistribution lays a locally-uniform fine patch there — the resolution Δ′ needs — which
# `bracket_mandatory_nodes` then centers the surface within. The spacing is **fixed** (τ-independent):
# Δ′ is a property of the rational surface, so its accuracy must not depend on the global accuracy
# target τ (that only sizes the pedestal/edge grid) — a τ-dependent patch would leave Δ′ far from
# converged at the default τ. RATIONAL_RES_SPACING is set where the q‴ estimate converges (the ldp
# mpsi ladder converges the q=2 Δ′ by h ≈ 5e-4).
const RATIONAL_RES_SPACING = 5.0e-4
const RATIONAL_RES_RADIUS = 5.0e-3
# Cap on the refined pass-2 interval count (density integral is clamped to this).
const REFINED_N_CAP = 1024
# Pass-1 interval count for the two-pass auto grid: the layout on which the formed equilibrium's
# curvature is measured to provision the refined pass-2 grid. Pass 1 is equilibrium-only (cheap;
# the ODE integration that dominates runtime is on pass 2). Sized to measure the gradient structure
# (pedestal, edge) accurately. NOTE: because the curvature estimate is currently grid-dependent
# (it grows with the measurement grid — see the dynamic-grid issue), a larger pass 1 measures more
# curvature and enlarges pass 2; revisit once the density is made grid-invariant.
const PASS1_INTERVALS = 1024

"""
    wants_two_pass(config::EquilibriumConfig) -> Bool

True when the config requests the automatic two-pass grid: `grid_type = "auto"` (or its
legacy alias `"log_asymptotic"`) with `mpsi = 0` and a positive `psi_accuracy`.
"""
wants_two_pass(config::EquilibriumConfig) =
    config.grid_type in ("auto", "log_asymptotic") && config.mpsi == 0 && config.psi_accuracy > 0

"""
    _validate_psi_nodes(psi_nodes, psilow, psihigh) -> Vector{Float64}

Check that an externally supplied ψ node vector is strictly increasing and spans exactly
[`psilow`, `psihigh`]. Errors loudly on violation (e.g. a psihigh separatrix re-clamp
between grid construction and equilibrium formation).
"""
function _validate_psi_nodes(psi_nodes::Vector{Float64}, psilow::Real, psihigh::Real)
    length(psi_nodes) >= 2 || error("override_psi_nodes must contain at least 2 nodes")
    all(diff(psi_nodes) .> 0) || error("override_psi_nodes must be strictly increasing")
    isapprox(psi_nodes[1], psilow; atol=1e-12) ||
        error("override_psi_nodes[1]=$(psi_nodes[1]) must equal psilow=$psilow")
    isapprox(psi_nodes[end], psihigh; atol=1e-12) ||
        error("override_psi_nodes[end]=$(psi_nodes[end]) must equal psihigh=$psihigh")
    return psi_nodes
end

"""
    _fourth_derivative_nodes(xs, ys) -> Vector{Float64}

Estimate |f''''| at each node of a (possibly nonuniform) grid by 5-point Newton divided
differences: f'''' ≈ 4!·f[x_{i-2},…,x_{i+2}]. The two nodes at each boundary reuse the
nearest interior stencil. Returns zeros when fewer than 5 nodes are available.
"""
function _fourth_derivative_nodes(xs::AbstractVector{<:Real}, ys::AbstractVector{<:Real})
    n = length(xs)
    d4 = zeros(Float64, n)
    n < 5 && return d4
    @inbounds for i in 1:n
        j = clamp(i - 2, 1, n - 4)  # stencil start: centered where possible, one-sided at ends
        # Newton divided-difference table over xs[j:j+4]
        c = Float64[ys[j+k] for k in 0:4]
        for order in 1:4
            for k in 4:-1:order
                c[k+1] = (c[k+1] - c[k]) / (xs[j+k] - xs[j+k-order])
            end
        end
        d4[i] = abs(24.0 * c[5])
    end
    return d4
end

"""
    _density_from_curvature!(rho, d4, f_scale, tau; xs=nothing)

Accumulate (in place, by max) the knot density implied by the h³ derivative error model
for one quantity: ρ = 1/h_target with h_target = (24·τ·f_scale/|f''''|)^(1/3), clamped to
[`H_TARGET_MIN`, `H_TARGET_MAX`]. Normalized curvature below the noise floor is ignored;
when `xs` is given the floor grows as `NODE_NOISE_REL`/h_local⁴ on tightly packed sample
grids, where divided differences amplify nodal noise.
"""
function _density_from_curvature!(rho::Vector{Float64}, d4::Vector{Float64}, f_scale::Float64, tau::Float64;
    xs::Union{Nothing,Vector{Float64}}=nothing)
    n = length(rho)
    @inbounds for i in 1:n
        noise_floor = D4_NOISE_FLOOR
        if xs !== nothing
            h_loc = (xs[min(i + 2, n)] - xs[max(i - 2, 1)]) / 4
            noise_floor = max(noise_floor, NODE_NOISE_REL / h_loc^4)
        end
        d4n = d4[i] / f_scale
        d4n < noise_floor && continue
        h = clamp((CUBIC_DERIV_ERR_CONST * tau / d4n)^(1 / 3), H_TARGET_MIN, H_TARGET_MAX)
        rho[i] = max(rho[i], 1.0 / h)
    end
    return rho
end

"""
    _knot_density(equil::PlasmaEquilibrium; tau, kin=nothing) -> Vector{Float64}

Knot density ρ(ψ) (knots per unit ψ_N) at the pass-1 nodes `equil.profiles.xs`, combining
by max:

  - measured h³-derivative-model curvature of the 1D profiles F, P, dV/dψ, q;
  - curvature of the four rzphi geometry channels along every `THETA_STRIDE`-th θ-line
    (max over θ) — the bicubic ψ-axis shares these knots, so geometry sharpness must
    attract knots just like the 1D profiles;
  - curvature of the kinetic profiles n_i, n_e, T_i, T_e, ω_E when `kin` is given
    (their own grid, rebroadcast by linear interpolation) — steep pedestal gradients in
    the kinetic data pull knots into the pedestal;
  - the model-curvature regions (same h³ rule on analytic |f''''| — see the module
    docstring): the edge floor for ψ ≥ 0.9, geometric-in-log(1−ψ) with dlog = (4τ)^(1/3)
    from uniform relative q′ error on q ≈ −A·ln(1−ψ), independent of A and normally
    inactive; and the core for ψ ≤ 0.03, geometric-in-log(ψ) from the power-law axis
    form, which replaces the (noise-dominated) measurement there. A global minimum
    density applies everywhere.

Rational surfaces are handled by `bracket_mandatory_nodes` after equidistribution, not by
elevating the density here. A running max over ±1 node smooths single-stencil dropouts.
"""
function _knot_density(equil::PlasmaEquilibrium; tau::Float64, kin::Union{Nothing,KineticProfileSplines}=nothing)
    xs = equil.profiles.xs
    n = length(xs)
    # Curvature is measured against ρ = √ψ (regular at the axis); the ρ-space density
    # rho_r converts to the ψ density at the end via dρ/dψ = 1/(2√ψ).
    rs = sqrt.(xs)
    rho_r = zeros(Float64, n)

    # 1D profiles
    for spl in (equil.profiles.F_spline, equil.profiles.P_spline, equil.profiles.dVdpsi_spline, equil.profiles.q_spline)
        vals = spl.y
        f_scale = max(maximum(abs, vals), F_SCALE_FLOOR)
        _density_from_curvature!(rho_r, _fourth_derivative_nodes(rs, vals), f_scale, tau; xs=rs)
    end

    # 2D geometry channels: per-θ-line curvature along ρ, max over θ, plane-wide scale
    for interp in (equil.rzphi_rsquared, equil.rzphi_offset, equil.rzphi_nu, equil.rzphi_jac)
        vals2d = @view interp.nodal_derivs.partials[1, :, :]
        f_scale = max(maximum(abs, vals2d), F_SCALE_FLOOR)
        for itheta in 1:THETA_STRIDE:size(vals2d, 2)
            _density_from_curvature!(rho_r, _fourth_derivative_nodes(rs, @view(vals2d[:, itheta])), f_scale, tau; xs=rs)
        end
    end

    # Kinetic profiles on their own grid, rebroadcast onto xs
    if kin !== nothing
        rs_kin = sqrt.(kin.xs)
        rho_kin = zeros(Float64, length(kin.xs))
        for spl in (kin.ni_spline, kin.ne_spline, kin.Ti_spline, kin.Te_spline, kin.omegaE_spline)
            vals = spl.y
            f_scale = max(maximum(abs, vals), F_SCALE_FLOOR)
            _density_from_curvature!(rho_kin, _fourth_derivative_nodes(rs_kin, vals), f_scale, tau; xs=rs_kin)
        end
        kin_itp = cubic_interp(rs_kin, rho_kin; extrap=ExtendExtrap())
        @inbounds for i in 1:n
            (rs_kin[1] <= rs[i] <= rs_kin[end]) || continue
            rho_r[i] = max(rho_r[i], kin_itp(rs[i]))
        end
    end

    # Convert the ρ-space density to ψ space: ρ_ψ(ψ) = ρ_ρ(ρ)·dρ/dψ = ρ_ρ/(2√ψ)
    rho = [rho_r[i] / (2.0 * rs[i]) for i in 1:n]

    # Running max over ±1 node
    rho_s = copy(rho)
    @inbounds for i in 1:n
        i > 1 && (rho_s[i] = max(rho_s[i], rho[i-1]))
        i < n && (rho_s[i] = max(rho_s[i], rho[i+1]))
    end

    # A-priori regions: geometric edge floor, and a pure a-priori geometric core — the
    # nodal data of the smallest flux surfaces is dominated by integration and axis
    # extrapolation error, so measured curvature is not trusted below the core split.
    dlog = (4.0 * tau)^(1 / 3)
    # The edge floor encodes the DIVERGING edge law q ≈ -A·ln(1-ψ), so it is applied only where
    # that model actually describes the equilibrium. A limited plasma has finite edge q and must
    # not be packed as if q blew up. The density itself stays A-independent (that is the point of
    # the form: uniform relative q′ error regardless of A) -- the fit supplies validity, not slope.
    edge_diverges = edge_q_law(equil) !== nothing
    @inbounds for i in 1:n
        if xs[i] <= CORE_MODEL_PSI_MAX
            rho_s[i] = 1.0 / (dlog * xs[i])
        elseif edge_diverges && xs[i] >= EDGE_MODEL_PSI_MIN
            rho_s[i] = max(rho_s[i], 1.0 / (dlog * (1.0 - xs[i])))
        end
        rho_s[i] = max(rho_s[i], 1.0 / H_TARGET_MAX)
    end

    return rho_s
end

"""
    _equidistribute(xs, rho, N) -> Vector{Float64}

Place N+1 nodes over [xs[1], xs[end]] by equidistributing the density integral
M(ψ) = ∫ρ dψ (trapezoid on the sample nodes, piecewise-linear inversion). Endpoints are
exactly xs[1] and xs[end]; interior nodes are strictly increasing since ρ > 0, which is
a required precondition (a zero density would make M non-invertible).
"""
function _equidistribute(xs::Vector{Float64}, rho::Vector{Float64}, N::Int)
    all(>(0), rho) || error("_equidistribute requires strictly positive density")
    n = length(xs)
    M = zeros(Float64, n)
    @inbounds for i in 2:n
        M[i] = M[i-1] + 0.5 * (rho[i] + rho[i-1]) * (xs[i] - xs[i-1])
    end
    nodes = Vector{Float64}(undef, N + 1)
    nodes[1] = xs[1]
    nodes[end] = xs[end]
    k = 2
    @inbounds for j in 1:(N-1)
        target = M[end] * j / N
        while M[k] < target
            k += 1
        end
        frac = (target - M[k-1]) / (M[k] - M[k-1])
        nodes[j+1] = xs[k-1] + frac * (xs[k] - xs[k-1])
    end
    return nodes
end

"""
    merge_mandatory_nodes(grid, mandatory; delta_frac=0.25, collapse_atol=1e-7) -> Vector{Float64}

Insert mandatory knots (e.g. rational surfaces) into a base grid with a minimum-spacing
guard: for each mandatory node, δ_min = `delta_frac` × (containing base-grid interval), and
any pre-existing non-mandatory node within δ_min is dropped (snap — the mandatory node is
never moved). Endpoints always win: mandatory nodes outside the open span or within δ_min of
an endpoint are discarded. Mandatory nodes within `collapse_atol` of an earlier mandatory
node are collapsed onto it — the same physical surface can be found through several (m, n)
pairs differing only by root-finder noise, and a near-zero interval would ring the
reconstructed splines. Collapse uses an absolute tolerance, not δ_min, so genuinely close but
distinct mandatory nodes (e.g. a kinetic-resonance surface near a rational) survive while
true duplicates still merge.

The function is agnostic to node provenance, so future mandatory sources (e.g. kinetic
resonance surfaces) plug in without change.
"""
function merge_mandatory_nodes(grid::Vector{Float64}, mandatory::Vector{Float64}; delta_frac::Float64=0.25, collapse_atol::Float64=1e-7)
    isempty(mandatory) && return copy(grid)
    lo, hi = grid[1], grid[end]
    mand = sort(unique(mandatory))

    # Local base spacing of the interval containing x (from the original base grid)
    base_h = function (x)
        k = clamp(searchsortedlast(grid, x), 1, length(grid) - 1)
        return grid[k+1] - grid[k]
    end

    kept = Tuple{Float64,Bool}[(g, false) for g in grid]  # (value, is_mandatory)
    last_mand = -Inf
    for m in mand
        (lo < m < hi) || continue
        δ = delta_frac * base_h(m)
        (m - lo < δ || hi - m < δ) && continue
        if m - last_mand < collapse_atol
            @debug "merge_mandatory_nodes: collapsing mandatory node ψ=$m onto ψ=$last_mand (spacing < collapse_atol)"
            continue
        end
        # Snap: drop non-mandatory, non-endpoint nodes within δ of the mandatory node
        filter!(t -> t[2] || t[1] == lo || t[1] == hi || abs(t[1] - m) >= δ, kept)
        push!(kept, (m, true))
        last_mand = m
    end
    sort!(kept; by=first)

    merged = [t[1] for t in kept]
    all(diff(merged) .> 0) || error("merge_mandatory_nodes produced a non-increasing grid")
    return merged
end

"""
    enforce_min_spacing(grid, hmin) -> Vector{Float64}

Drop interior nodes so no two kept nodes are closer than `hmin`, preserving both endpoints.
Guards against noise-level packing from any generator — the auto-grid near-surface bracket, or a
fixed `ldp`/`pow1` grid whose edge spacing collapses (`~(π/2·mpsi)⁻²` for `ldp`) at high mpsi,
below which cubic high derivatives are integration noise, not curvature. A no-op when the grid is
already coarser than `hmin` everywhere (e.g. a converged uniform reference).
"""
function enforce_min_spacing(grid::Vector{Float64}, hmin::Float64)
    n = length(grid)
    (n < 3 || hmin <= 0) && return copy(grid)
    kept = Float64[grid[1]]
    for i in 2:(n-1)
        grid[i] - kept[end] >= hmin && push!(kept, grid[i])
    end
    # Keep the far endpoint; if the last retained interior node crowds it, drop that node instead.
    length(kept) > 1 && grid[n] - kept[end] < hmin && pop!(kept)
    push!(kept, grid[n])
    return kept
end

"""
    bracket_mandatory_nodes(grid, centers, min_half_widths, min_spacing; collapse_atol=1e-7) -> Vector{Float64}

Center each rational `centers[i]` inside a single clean spline interval by replacing the knots
straddling it with a symmetric pair at `centers[i] ± w`. The half-width `w` is the larger of
`min_half_widths[i]` (the Δ′-stencil floor `BRACKET_COEF·dpsi`) and **half the local grid
spacing**, so the bracket blends into the surrounding grid rather than punching a narrow interval
into it: the region stays locally uniform with the center at its midpoint. That local uniformity
is what keeps the cubic 3rd derivative the Δ′ extraction samples both *consistent across* the
surface (no knot-on-surface jump) and *stable across* grid refinement — a narrow bracket amid
wide neighbors would be consistent but noisy. Non-bracket knots within `w` of the center, or
within `min_spacing` of either new bracket knot, are dropped; endpoints always win; bracket knots
outside the open domain are dropped, and knots closer than `collapse_atol` are collapsed to keep
the grid strictly increasing.
"""
function bracket_mandatory_nodes(grid::Vector{Float64}, centers::Vector{Float64}, min_half_widths::Vector{Float64}, min_spacing::Float64; collapse_atol::Float64=1e-7)
    isempty(centers) && return copy(grid)
    length(centers) == length(min_half_widths) || error("bracket_mandatory_nodes: centers and min_half_widths length mismatch")
    lo, hi = grid[1], grid[end]
    order = sortperm(centers)
    kept = Tuple{Float64,Bool}[(g, false) for g in grid]  # (value, is_bracket_knot)
    last_c = -Inf
    for idx in order
        c, hw = centers[idx], min_half_widths[idx]
        (lo < c < hi) || continue
        c - last_c < collapse_atol && continue  # duplicate rational (same q via several (m,n))
        k = clamp(searchsortedlast(grid, c), 1, length(grid) - 1)
        w = max(hw, 0.5 * (grid[k+1] - grid[k]))  # blend to half the local spacing
        left, right = c - w, c + w
        # Drop non-bracket, non-endpoint knots inside the zone or crowding a new bracket knot.
        filter!(kept) do t
            t[2] || t[1] == lo || t[1] == hi ||
                !(left < t[1] < right || abs(t[1] - left) < min_spacing || abs(t[1] - right) < min_spacing)
        end
        left > lo && push!(kept, (left, true))
        right < hi && push!(kept, (right, true))
        last_c = c
    end
    sort!(kept; by=first)
    merged = Float64[]
    for (v, _) in kept
        (isempty(merged) || v - merged[end] > collapse_atol) && push!(merged, v)
    end
    all(diff(merged) .> 0) || error("bracket_mandatory_nodes produced a non-increasing grid")
    return merged
end

"""
    _truncate_density(xs, rho, psihigh) -> (xs_t, rho_t)

Restrict a measured knot density to `[xs[1], psihigh]`, linearly interpolating `rho` at the new
outer endpoint so the density integral stays continuous in `psihigh`. Errors when `psihigh` lies
outside the sampled grid, where the density is unmeasured.
"""
function _truncate_density(xs, rho::Vector{Float64}, psihigh::Float64)
    xs_v = collect(Float64, xs)
    psihigh > xs_v[1] ||
        error("_truncate_density: psihigh=$psihigh must exceed the inner grid bound $(xs_v[1])")
    psihigh <= xs_v[end] + 1e-12 ||
        error(
            "_truncate_density: psihigh=$psihigh exceeds the pass-1 grid end $(xs_v[end]); " *
            "the knot density is unmeasured there — form the enlarged domain first, then refine against it"
        )
    psihigh >= xs_v[end] - 1e-12 && return (xs_v, rho)

    k = searchsortedlast(xs_v, psihigh)
    frac = (psihigh - xs_v[k]) / (xs_v[k+1] - xs_v[k])
    rho_end = rho[k] + frac * (rho[k+1] - rho[k])
    return (vcat(xs_v[1:k], psihigh), vcat(rho[1:k], rho_end))
end

"""
    refined_psi_grid(equil::PlasmaEquilibrium; tau, kin=nothing, mandatory=Float64[],
                     psihigh=nothing, singfac_min=1e-4, n_min=1, bracket_coef=BRACKET_COEF,
                     min_spacing=MIN_KNOT_SPACING, N_cap=1024) -> Vector{Float64}

Build the refined pass-2 ψ grid from a formed pass-1 equilibrium: measured-curvature knot
density (`_knot_density`), equidistribution, a global minimum-spacing floor
(`enforce_min_spacing`), and rational-surface bracketing (`bracket_mandatory_nodes`). `tau` is
the target interpolation accuracy (`psi_accuracy`); `kin` optionally supplies kinetic profiles
whose pedestal gradients attract knots; `mandatory` lists rational-surface ψ values to bracket;
`singfac_min` and `n_min` (smallest |n| in the run) set each surface's matching half-stencil
`dpsi = singfac_min/(n_min·|q′|)`, and the bracket half-width is `bracket_coef·dpsi` (floored at
`min_spacing`). Rational surfaces are bracketed, not pinned: a knot on the surface would make the
Δ′ extraction's cubic 3rd derivative jump mid-stencil (see `BRACKET_COEF`).

`psihigh` builds the grid for a domain *smaller* than the one `equil` was formed on: the measured
density is truncated there and the last node lands exactly on it. This lets a pass-1 equilibrium
supply the density for a re-form on a reduced domain without an extra solve. Passing a `psihigh`
beyond the pass-1 grid is an error — the density out there is unmeasured, so an enlarged domain
must be formed first and refined against that.
"""
function refined_psi_grid(equil::PlasmaEquilibrium;
    tau::Float64,
    kin::Union{Nothing,KineticProfileSplines}=nothing,
    mandatory::Vector{Float64}=Float64[],
    psihigh::Union{Nothing,Real}=nothing,
    singfac_min::Float64=1e-4,
    n_min::Int=1,
    bracket_coef::Float64=BRACKET_COEF,
    min_spacing::Float64=MIN_KNOT_SPACING,
    N_cap::Int=REFINED_N_CAP)
    xs = equil.profiles.xs
    rho = _knot_density(equil; tau, kin)
    if psihigh !== nothing
        xs, rho = _truncate_density(xs, rho, Float64(psihigh))
    end
    # Floor the density to a fixed (τ-independent) locally-uniform fine patch around each rational
    # so Δ′ has the resolution to sample 3rd derivatives there at any accuracy target (see
    # RATIONAL_RES_SPACING).
    if !isempty(mandatory)
        h_rat = max(RATIONAL_RES_SPACING, min_spacing)
        @inbounds for m in mandatory, i in eachindex(xs)
            abs(xs[i] - m) <= RATIONAL_RES_RADIUS && (rho[i] = max(rho[i], 1.0 / h_rat))
        end
    end
    M_total = sum(0.5 * (rho[i] + rho[i-1]) * (xs[i] - xs[i-1]) for i in 2:length(xs))
    N = clamp(ceil(Int, M_total), 32, N_cap)
    N == N_cap && M_total > N_cap &&
        @warn "refined_psi_grid: knot count capped at $N_cap (density integral wants $(ceil(Int, M_total))); psi_accuracy=$tau may not be attainable"
    grid = enforce_min_spacing(_equidistribute(xs, rho, N), min_spacing)
    isempty(mandatory) && return grid
    min_half_widths = [max(bracket_coef * singfac_min / (n_min * abs(equil.profiles.q_deriv(m))), min_spacing) for m in mandatory]
    return bracket_mandatory_nodes(grid, mandatory, min_half_widths, min_spacing)
end

"""
    implied_knot_count(equil::PlasmaEquilibrium; tau, kin=nothing) -> Int

Knot count the measured-curvature density model implies for a formed equilibrium. Used as
a post-refinement consistency diagnostic: if this differs substantially from the actual
grid size, the pass-1 grid under- or over-sampled a feature.
"""
function implied_knot_count(equil::PlasmaEquilibrium; tau::Float64, kin::Union{Nothing,KineticProfileSplines}=nothing)
    xs = equil.profiles.xs
    rho = _knot_density(equil; tau, kin)
    return ceil(Int, sum(0.5 * (rho[i] + rho[i-1]) * (xs[i] - xs[i-1]) for i in 2:length(xs)))
end
