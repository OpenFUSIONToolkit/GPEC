"""
Radial grid refinement utilities for the two-pass auto ψ grid.

Pass 1 forms the equilibrium on a coarse grid; `refined_psi_grid` then derives a knot
density from the measured nodal data (1D profiles, 2D geometry channels, and optional
kinetic profiles) and equidistributes it, pinning mandatory knots (e.g. rational surfaces)
via `merge_mandatory_nodes`. Pass 2 re-forms the equilibrium on the refined grid through
the `override_psi_nodes` keyword of `setup_equilibrium`.

The density uses the cubic-interpolation error model err ≈ h⁴|f''''|/384, so the knot
count of every region scales as psi_accuracy^(-1/4) and tightening the tolerance refines
all regions proportionally. Fourth derivatives are estimated by divided differences on the
pass-1 nodes only — nodal values come from independent field-line integrations and are
immune to inter-knot spline ringing.
"""

# Cubic spline interpolation error constant: err ≈ h⁴|f''''|/384
const CUBIC_ERR_CONST = 384.0
# |f''''|/f_scale below this is treated as integrator noise, not curvature
const D4_NOISE_FLOOR = 1e3
# Per-quantity target spacing clamp (ψ_N units)
const H_TARGET_MIN = 1e-4
const H_TARGET_MAX = 0.2
# θ-lines subsample stride for the 2D geometry channels
const THETA_STRIDE = 8

"""
    wants_two_pass(config::EquilibriumConfig) -> Bool

True when the config requests the automatic two-pass grid: `grid_type = "log_asymptotic"`
with `mpsi = 0` and a positive `psi_accuracy`.
"""
wants_two_pass(config::EquilibriumConfig) =
    config.grid_type == "log_asymptotic" && config.mpsi == 0 && config.psi_accuracy > 0

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
    _density_from_curvature!(rho, xs, d4, f_scale, tau)

Accumulate (in place, by max) the knot density implied by the h⁴ error model for one
quantity: ρ = 1/h_target with h_target = (384·τ·f_scale/|f''''|)^(1/4), clamped to
[`H_TARGET_MIN`, `H_TARGET_MAX`]. Normalized curvature below `D4_NOISE_FLOOR` is ignored.
"""
function _density_from_curvature!(rho::Vector{Float64}, d4::Vector{Float64}, f_scale::Float64, tau::Float64)
    @inbounds for i in eachindex(rho)
        d4n = d4[i] / f_scale
        d4n < D4_NOISE_FLOOR && continue
        h = clamp((CUBIC_ERR_CONST * tau / d4n)^0.25, H_TARGET_MIN, H_TARGET_MAX)
        rho[i] = max(rho[i], 1.0 / h)
    end
    return rho
end

"""
    _measured_edge_log_slope(xs, q_vals; nk=5) -> Float64

Estimate the separatrix log slope A of q ≈ −A·ln(1−ψ) from the outermost pass-1 q nodes.
Returns the fallback A=2.0 when the estimate is non-finite or non-positive (e.g. reversed
edge shear); analytic equilibria with finite edge q give a genuinely small A and hence a
weak edge floor.
"""
function _measured_edge_log_slope(xs::Vector{Float64}, q_vals::AbstractVector{<:Real}; nk::Int=5)
    n = length(xs)
    n < nk + 1 && return 2.0
    j = n - nk
    denom = log((1.0 - xs[j]) / (1.0 - xs[n]))
    A = (q_vals[n] - q_vals[j]) / denom
    return (isfinite(A) && A > 0) ? A : 2.0
end

"""
    _knot_density(equil::PlasmaEquilibrium; tau, kin=nothing) -> Vector{Float64}

Knot density ρ(ψ) (knots per unit ψ_N) at the pass-1 nodes `equil.profiles.xs`, combining
by max:

  - measured h⁴-model curvature of the 1D profiles F, P, dV/dψ, q;
  - curvature of the four rzphi geometry channels along every `THETA_STRIDE`-th θ-line
    (max over θ) — the bicubic ψ-axis shares these knots, so geometry sharpness must
    attract knots just like the 1D profiles;
  - curvature of the kinetic profiles n_i, n_e, T_i, T_e, ω_E when `kin` is given
    (their own grid, rebroadcast by linear interpolation) — steep pedestal gradients in
    the kinetic data pull knots into the pedestal;
  - a-priori floors: geometric-in-log(1−ψ) at the edge with slope A measured from the
    pass-1 q nodes, geometric-in-log(ψ) in the core, and a global minimum density.

A running max over ±1 node smooths single-stencil dropouts.
"""
function _knot_density(equil::PlasmaEquilibrium; tau::Float64, kin::Union{Nothing,KineticProfileSplines}=nothing)
    xs = equil.profiles.xs
    n = length(xs)
    rho = zeros(Float64, n)

    # 1D profiles
    for spl in (equil.profiles.F_spline, equil.profiles.P_spline, equil.profiles.dVdpsi_spline, equil.profiles.q_spline)
        vals = spl.y
        f_scale = max(maximum(abs, vals), 1e-12)
        _density_from_curvature!(rho, _fourth_derivative_nodes(xs, vals), f_scale, tau)
    end

    # 2D geometry channels: per-θ-line curvature along ψ, max over θ, plane-wide scale
    for interp in (equil.rzphi_rsquared, equil.rzphi_offset, equil.rzphi_nu, equil.rzphi_jac)
        vals2d = @view interp.nodal_derivs.partials[1, :, :]
        f_scale = max(maximum(abs, vals2d), 1e-12)
        for itheta in 1:THETA_STRIDE:size(vals2d, 2)
            _density_from_curvature!(rho, _fourth_derivative_nodes(xs, @view(vals2d[:, itheta])), f_scale, tau)
        end
    end

    # Kinetic profiles on their own grid, rebroadcast onto xs
    if kin !== nothing
        rho_kin = zeros(Float64, length(kin.xs))
        for spl in (kin.ni_spline, kin.ne_spline, kin.Ti_spline, kin.Te_spline, kin.omegaE_spline)
            vals = spl.y
            f_scale = max(maximum(abs, vals), 1e-12)
            _density_from_curvature!(rho_kin, _fourth_derivative_nodes(kin.xs, vals), f_scale, tau)
        end
        kin_itp = cubic_interp(kin.xs, rho_kin; extrap=ExtendExtrap())
        @inbounds for i in 1:n
            (kin.xs[1] <= xs[i] <= kin.xs[end]) || continue
            rho[i] = max(rho[i], kin_itp(xs[i]))
        end
    end

    # Running max over ±1 node
    rho_s = copy(rho)
    @inbounds for i in 1:n
        i > 1 && (rho_s[i] = max(rho_s[i], rho[i-1]))
        i < n && (rho_s[i] = max(rho_s[i], rho[i+1]))
    end

    # A-priori floors: edge/core geometric packing and a global minimum
    A = _measured_edge_log_slope(xs, equil.profiles.q_spline.y)
    dlog = (13.0 * tau / A)^0.25
    @inbounds for i in 1:n
        xs[i] >= 0.9 && (rho_s[i] = max(rho_s[i], 1.0 / (dlog * (1.0 - xs[i]))))
        xs[i] <= 0.03 && (rho_s[i] = max(rho_s[i], 1.0 / (dlog * xs[i])))
        rho_s[i] = max(rho_s[i], 1.0 / H_TARGET_MAX)
    end
    return rho_s
end

"""
    _equidistribute(xs, rho, N) -> Vector{Float64}

Place N+1 nodes over [xs[1], xs[end]] by equidistributing the density integral
M(ψ) = ∫ρ dψ (trapezoid on the sample nodes, piecewise-linear inversion). Endpoints are
exactly xs[1] and xs[end]; interior nodes are strictly increasing since ρ > 0.
"""
function _equidistribute(xs::Vector{Float64}, rho::Vector{Float64}, N::Int)
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
    merge_mandatory_nodes(grid, mandatory; delta_frac=0.25) -> Vector{Float64}

Insert mandatory knots (e.g. rational surfaces) into a base grid with a minimum-spacing
guard: for each mandatory node, δ_min = `delta_frac` × (containing base-grid interval),
and any pre-existing non-mandatory node within δ_min is dropped (snap — the mandatory node
is never moved). Endpoints always win: mandatory nodes outside the open span or within
δ_min of an endpoint are discarded. Mandatory nodes within δ_min of an earlier mandatory
node are collapsed onto it — the same physical surface can be found through several (m, n)
pairs differing only by root-finder noise, and a near-zero interval would ring the
reconstructed splines; two genuinely distinct surfaces closer than δ_min cannot be
resolved separately at the local grid resolution anyway.

The function is agnostic to node provenance, so future mandatory sources (e.g. kinetic
resonance surfaces) plug in without change.
"""
function merge_mandatory_nodes(grid::Vector{Float64}, mandatory::Vector{Float64}; delta_frac::Float64=0.25)
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
        if m - last_mand < δ
            @debug "merge_mandatory_nodes: collapsing mandatory node ψ=$m onto ψ=$last_mand (spacing < δ_min)"
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
                     delta_frac=0.25, N_cap=1024) -> Vector{Float64}

Build the refined pass-2 ψ grid from a formed pass-1 equilibrium: measured-curvature knot
density (`_knot_density`), equidistribution, and mandatory-knot insertion
(`merge_mandatory_nodes`). `tau` is the target interpolation accuracy (`psi_accuracy`);
`kin` optionally supplies kinetic profiles whose pedestal gradients attract knots;
`mandatory` lists ψ values that must appear as knots (e.g. rational surfaces).
"""
function refined_psi_grid(equil::PlasmaEquilibrium;
    tau::Float64,
    kin::Union{Nothing,KineticProfileSplines}=nothing,
    mandatory::Vector{Float64}=Float64[],
    delta_frac::Float64=0.25,
    N_cap::Int=1024)
    xs = equil.profiles.xs
    rho = _knot_density(equil; tau, kin)
    M_total = sum(0.5 * (rho[i] + rho[i-1]) * (xs[i] - xs[i-1]) for i in 2:length(xs))
    N = clamp(ceil(Int, M_total), 32, N_cap)
    N == N_cap && M_total > N_cap &&
        @warn "refined_psi_grid: knot count capped at $N_cap (density integral wants $(ceil(Int, M_total))); psi_accuracy=$tau may not be attainable"
    grid = _equidistribute(xs, rho, N)
    return merge_mandatory_nodes(grid, mandatory; delta_frac)
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
