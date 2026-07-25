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
# Local density elevation around mandatory (rational) surfaces: knot spacing h_s = coef·τ^(1/3)
# at the surface, geometric growth away from it, applied within the given radius. This keeps the
# equidistributed base grid from starving the approach to each rational, where the ideal-MHD
# Δ′ extraction is most sensitive to the local equilibrium-spline resolution.
const SING_PACK_COEF = 0.06
const SING_PACK_RADIUS = 0.05

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
    _knot_density(equil::PlasmaEquilibrium; tau, kin=nothing, mandatory=Float64[], sing_pack_coef=SING_PACK_COEF) -> Vector{Float64}

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
    density applies everywhere;
  - local packing around each `mandatory` (rational) surface: spacing `sing_pack_coef`·τ^(1/3)
    at the surface with geometric growth away from it, within `SING_PACK_RADIUS`.

A running max over ±1 node smooths single-stencil dropouts.
"""
function _knot_density(equil::PlasmaEquilibrium; tau::Float64, kin::Union{Nothing,KineticProfileSplines}=nothing,
    mandatory::Vector{Float64}=Float64[], sing_pack_coef::Float64=SING_PACK_COEF)
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
    @inbounds for i in 1:n
        if xs[i] <= CORE_MODEL_PSI_MAX
            rho_s[i] = 1.0 / (dlog * xs[i])
        elseif xs[i] >= EDGE_MODEL_PSI_MIN
            rho_s[i] = max(rho_s[i], 1.0 / (dlog * (1.0 - xs[i])))
        end
        rho_s[i] = max(rho_s[i], 1.0 / H_TARGET_MAX)
    end

    # Local packing around mandatory (rational) surfaces
    h_s = max(sing_pack_coef * tau^(1 / 3), H_TARGET_MIN)
    for psi_m in mandatory
        @inbounds for i in 1:n
            d = abs(xs[i] - psi_m)
            d > SING_PACK_RADIUS && continue
            rho_s[i] = max(rho_s[i], 1.0 / max(dlog * d, h_s))
        end
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
    refined_psi_grid(equil::PlasmaEquilibrium; tau, kin=nothing, mandatory=Float64[],
                     delta_frac=0.25, N_cap=1024, sing_pack_coef=SING_PACK_COEF) -> Vector{Float64}

Build the refined pass-2 ψ grid from a formed pass-1 equilibrium: measured-curvature knot
density (`_knot_density`), equidistribution, and mandatory-knot insertion
(`merge_mandatory_nodes`). `tau` is the target interpolation accuracy (`psi_accuracy`);
`kin` optionally supplies kinetic profiles whose pedestal gradients attract knots;
`mandatory` lists ψ values that must appear as knots (e.g. rational surfaces);
`sing_pack_coef` scales the knot spacing at those surfaces (convergence studies only —
the default is scan-calibrated).
"""
function refined_psi_grid(equil::PlasmaEquilibrium;
    tau::Float64,
    kin::Union{Nothing,KineticProfileSplines}=nothing,
    mandatory::Vector{Float64}=Float64[],
    delta_frac::Float64=0.25,
    N_cap::Int=1024,
    sing_pack_coef::Float64=SING_PACK_COEF)
    xs = equil.profiles.xs
    rho = _knot_density(equil; tau, kin, mandatory, sing_pack_coef)
    M_total = sum(0.5 * (rho[i] + rho[i-1]) * (xs[i] - xs[i-1]) for i in 2:length(xs))
    N = clamp(ceil(Int, M_total), 32, N_cap)
    N == N_cap && M_total > N_cap &&
        @warn "refined_psi_grid: knot count capped at $N_cap (density integral wants $(ceil(Int, M_total))); psi_accuracy=$tau may not be attainable"
    grid = _equidistribute(xs, rho, N)
    return merge_mandatory_nodes(grid, mandatory; delta_frac)
end

"""
    implied_knot_count(equil::PlasmaEquilibrium; tau, kin=nothing, mandatory=Float64[]) -> Int

Knot count the measured-curvature density model implies for a formed equilibrium. Used as
a post-refinement consistency diagnostic: if this differs substantially from the actual
grid size, the pass-1 grid under- or over-sampled a feature.
"""
function implied_knot_count(equil::PlasmaEquilibrium; tau::Float64, kin::Union{Nothing,KineticProfileSplines}=nothing,
    mandatory::Vector{Float64}=Float64[])
    xs = equil.profiles.xs
    rho = _knot_density(equil; tau, kin, mandatory)
    return ceil(Int, sum(0.5 * (rho[i] + rho[i-1]) * (xs[i] - xs[i-1]) for i in 2:length(xs)))
end
