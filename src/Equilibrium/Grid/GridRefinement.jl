"""
Radial grid refinement utilities for the two-pass auto ψ grid.

Pass 1 forms the equilibrium on a coarse grid; `refined_psi_grid` then derives a knot
density from the measured nodal data (1D profiles, 2D geometry channels, and optional
kinetic profiles) and equidistributes it, pinning mandatory knots (e.g. rational surfaces)
via `merge_mandatory_nodes`. Pass 2 re-forms the equilibrium on the refined grid through
the `override_psi_nodes` keyword of `PlasmaEquilibrium`.

Knot spacing follows the cubic spline *derivative* error model err(f′) ≈ h³|f''''|/24 ≤ τ·f_scale
(the stability physics consumes spline derivatives), so knot counts scale as psi_accuracy^(-1/3).
The curvature |f''''| comes from three sources:

 1. *Measured* (mid-radius and edge): 5-point divided differences of the pass-1 nodal values.
 2. *Separatrix model* (edge floor, ψ ≥ 0.9): q ≈ −A·ln(1−ψ) gives geometric packing in log(1−ψ),
    independent of A; applied only when `edge_q_law` finds a diverging edge.
 3. *Axis model* (core, ψ ≤ 0.03): a power law in ψ gives geometric packing in log(ψ). It replaces
    the measurement there, since the smallest surfaces' nodal data is dominated by integration noise.

Measured curvature is taken against ρ = √ψ_N, in which the equilibrium is regular at the axis, and
mapped back to a ψ density through dψ/dρ = 2√ψ_N.
"""

# Cubic spline derivative interpolation error constant: err(f′) ≈ h³|f''''|/24
const CUBIC_DERIV_ERR_CONST = 24.0
# |f''''|/f_scale below this is treated as integrator noise, not curvature; on tightly packed sample
# grids the floor grows as NODE_NOISE_REL/h⁴, since divided differences amplify nodal noise as h⁻⁴.
const D4_NOISE_FLOOR = 1e3
const NODE_NOISE_REL = 1e-8
# Floor on a channel's value-scale, so an all-zero profile/geometry channel cannot divide by zero
const F_SCALE_FLOOR = 1e-12
# Per-quantity target spacing clamp (ψ_N units)
const H_TARGET_MIN = 1e-4
const H_TARGET_MAX = 0.2
# Model-region splits (ψ_N): below CORE the axis model replaces measured curvature; above EDGE the
# separatrix model floors it.
const CORE_MODEL_PSI_MAX = 0.03
const EDGE_MODEL_PSI_MIN = 0.9
# θ-lines subsample stride for the 2D geometry channels
const THETA_STRIDE = 8
# Edge q-law gate.
# Minimum knots in the edge band before a fit is attempted.
const EDGE_FIT_MIN_KNOTS = 4
# Weak backstop against a fit that describes nothing; no absolute r2 separates diverted from limited
# edges, so that discrimination is the r2_log > r2_linear comparison.
const EDGE_FIT_MIN_R2 = 0.5
# Accepted factor between the fitted `A` and the measured local log shear -(1-psi)*q'(psi).
const EDGE_FIT_SHEAR_RATIO = 2.0

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
               min_r2=EDGE_FIT_MIN_R2, shear_ratio=EDGE_FIT_SHEAR_RATIO)
        -> nothing | (; A, q_bar, u_bar, n_knots, r2_log, r2_linear)

Least-squares fit of the separatrix edge law `q = q̄ + A·(ln(1−ψ) − ū)` over the equilibrium's
outer knots. It only gates the grid-refinement edge density floor; placing surfaces beyond the grid
is [`separatrix_q_law`](@ref), and geometric X-point detection is [`classify_topology`](@ref).

Returns `nothing` when the diverging model does not describe the edge:

  - fewer than `min_knots` knots in the band;
  - `A ≥ 0`, i.e. q not rising toward ψ = 1;
  - `r2_log < min_r2`;
  - `r2_log ≤ r2_linear` (a linear-in-ψ fit does at least as well, as for a limited plasma);
  - the fitted `A` and the measured `-(1−ψ)·q′` differ by more than `shear_ratio`.
"""
function edge_q_law(equil::PlasmaEquilibrium;
    psi_max::Real=Float64(equil.profiles.xs[end]),
    psi_min::Real=EDGE_MODEL_PSI_MIN,
    min_knots::Int=EDGE_FIT_MIN_KNOTS,
    min_r2::Real=EDGE_FIT_MIN_R2,
    shear_ratio::Real=EDGE_FIT_SHEAR_RATIO)
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

    # The fitted slope is the local logarithmic shear; it must match the profile's -(1-psi)*q'.
    hint = Ref(1)
    A_meas = median([-(1.0 - xs[i]) * Float64(equil.profiles.q_deriv(xs[i]; hint=hint)) for i in band])
    isfinite(A_meas) && A_meas < 0 || return nothing
    ratio = A / A_meas
    (ratio >= inv(shear_ratio) && ratio <= shear_ratio) || return nothing

    return (A=A, q_bar=q_bar, u_bar=u_bar, n_knots=length(band), r2_log=r2_log, r2_linear=r2_linear)
end


# Rational-surface bracket half-width in units of the Δ′ matching stencil dpsi = singfac_min/|n·q′|.
# A knot inside the stencil makes the cubic 3rd derivative jump mid-stencil and Δ′ non-convergent;
# > 1 leaves margin for the pass-1 → pass-2 shift of ψ_s.
const BRACKET_COEF = 4.0
# Global minimum knot spacing for every grid; below it cubic high derivatives are integration noise.
const MIN_KNOT_SPACING = 1.0e-4
# Fixed (τ-independent) knot spacing within RATIONAL_RES_RADIUS of each rational surface, so Δ′ gets the
# local resolution it needs at any accuracy target.
const RATIONAL_RES_SPACING = 5.0e-4
const RATIONAL_RES_RADIUS = 5.0e-3
# Cap on the refined pass-2 interval count (density integral is clamped to this).
const REFINED_N_CAP = 1024
# Pass-1 interval count for the two-pass auto grid. The curvature estimate grows with this grid, so a
# larger pass 1 also enlarges pass 2.
const PASS1_INTERVALS = 1024

"""
    wants_two_pass(config::EquilibriumConfig) -> Bool

True when the config requests the automatic two-pass grid: `grid_type = "auto"` with `mpsi = 0`
and a positive `psi_accuracy`.
"""
wants_two_pass(config::EquilibriumConfig) =
    config.grid_type == "auto" && config.mpsi == 0 && config.psi_accuracy > 0

"""
    _validate_psi_nodes(psi_nodes, psilow, psihigh) -> Vector{Float64}

Check that an externally supplied ψ node vector is strictly increasing and spans exactly
`psilow` to `psihigh`; errors otherwise.
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

Knot density ρ(ψ) (knots per unit ψ_N) at the pass-1 nodes `equil.profiles.xs`: the max of the
measured curvature density of the 1D profiles (F, P, dV/dψ, q), the four rzphi geometry channels
(every `THETA_STRIDE`-th θ-line), and the kinetic profiles when `kin` is given, smoothed by a running
max over ±1 node, then the core and edge model regions and a global minimum density.
"""
function _knot_density(equil::PlasmaEquilibrium; tau::Float64, kin::Union{Nothing,KineticProfileSplines}=nothing)
    xs = equil.profiles.xs
    n = length(xs)
    # Curvature is measured against ρ = √ψ (regular at the axis) and converted to a ψ density at the end.
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

    # Model regions: geometric core (replaces measurement) and geometric edge floor.
    dlog = (4.0 * tau)^(1 / 3)
    # The edge floor assumes a diverging edge q, so it applies only when the edge law fits.
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

Place N+1 nodes from `xs[1]` to `xs[end]` by equidistributing the density integral M(ψ) = ∫ρ dψ
(trapezoid, piecewise-linear inversion). Requires ρ > 0 everywhere.
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

Insert mandatory knots into a base grid without moving them: non-mandatory nodes within
δ_min = `delta_frac` × (containing base-grid interval) of a mandatory node are dropped. Endpoints
always win, so mandatory nodes outside the open span or within δ_min of an endpoint are discarded.
Mandatory nodes within `collapse_atol` of an earlier one are treated as duplicates and collapsed.
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

Drop interior nodes so no two kept nodes are closer than `hmin`, preserving both endpoints. Guards
against noise-level packing, e.g. the edge of a `core_edge_packed`/`pow1` grid at high mpsi.
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
straddling it with a symmetric pair at `centers[i] ± w`, so the cubic 3rd derivative the Δ′
extraction samples does not jump mid-stencil. `w` is the larger of `min_half_widths[i]` (the
Δ′-stencil floor) and half the local grid spacing, capped at half the distance to neighboring centers
(and, for the stencil part, to the nearer domain edge). Knots within `w` of the center or within
`min_spacing` of a new bracket knot are dropped; endpoints always win. Warns when a cap cuts a bracket
below its stencil floor, since Δ′ there is less accurate.
"""
function bracket_mandatory_nodes(grid::Vector{Float64}, centers::Vector{Float64}, min_half_widths::Vector{Float64}, min_spacing::Float64; collapse_atol::Float64=1e-7)
    isempty(centers) && return copy(grid)
    length(centers) == length(min_half_widths) || error("bracket_mandatory_nodes: centers and min_half_widths length mismatch")
    lo, hi = grid[1], grid[end]
    # In-domain centers in increasing order; centers closer than collapse_atol share one bracket.
    cs, hws = Float64[], Float64[]
    for idx in sortperm(centers)
        c = centers[idx]
        (lo < c < hi) || continue
        (isempty(cs) || c - cs[end] >= collapse_atol) || continue
        push!(cs, c)
        push!(hws, min_half_widths[idx])
    end
    kept = Tuple{Float64,Bool}[(g, false) for g in grid]  # (value, is_bracket_knot)
    capped = Tuple{Float64,Float64,Float64}[]  # (center, capped half-width, stencil floor)
    for (i, c) in enumerate(cs)
        k = clamp(searchsortedlast(grid, c), 1, length(grid) - 1)
        cap_nb = 0.5 * min(i > 1 ? c - cs[i-1] : Inf, i < length(cs) ? cs[i+1] - c : Inf)
        cap_edge = 0.5 * min(c - lo, hi - c)
        # Stencil floor capped by neighbors and edges; blend to half the local spacing capped by neighbors only.
        w = max(min(hws[i], cap_nb, cap_edge), min(0.5 * (grid[k+1] - grid[k]), cap_nb))
        w < hws[i] && push!(capped, (c, w, hws[i]))
        left, right = c - w, c + w
        # Drop non-bracket, non-endpoint knots inside the zone or crowding a new bracket knot.
        filter!(kept) do t
            t[2] || t[1] == lo || t[1] == hi ||
                !(left < t[1] < right || abs(t[1] - left) < min_spacing || abs(t[1] - right) < min_spacing)
        end
        left > lo && push!(kept, (left, true))
        right < hi && push!(kept, (right, true))
    end
    isempty(capped) || @warn "$(length(capped)) rational surface(s) have grid bracket half-widths capped below their Δ′ stencil " *
                             "(neighboring surface or domain edge too close, or a near-tangent surface with q′ ≈ 0); Δ′ there is less accurate. " *
                             "(ψ, half-width, stencil): $(join(capped, ", "))"
    sort!(kept; by=first)
    merged = Float64[]
    for (v, _) in kept
        (isempty(merged) || v - merged[end] > collapse_atol) && push!(merged, v)
    end
    all(diff(merged) .> 0) || error("bracket_mandatory_nodes produced a non-increasing grid")
    return merged
end

"""
    refined_psi_grid(equil::PlasmaEquilibrium; tau, kin=nothing, mandatory=Float64[], pinned=Float64[],
                     singfac_min=1e-4, n_min=1, bracket_coef=BRACKET_COEF,
                     min_spacing=MIN_KNOT_SPACING, N_cap=REFINED_N_CAP) -> Vector{Float64}

Build the refined pass-2 ψ grid from a formed pass-1 equilibrium: knot density, equidistribution,
`enforce_min_spacing`, then `bracket_mandatory_nodes` around rational surfaces.

## Arguments

  - `tau` - target interpolation accuracy (`psi_accuracy`)
  - `kin` - optional kinetic profiles whose gradients attract knots
  - `mandatory` - rational-surface ψ values to bracket (never knotted on)
  - `pinned` - ψ values inserted as plain knots without a cleared zone
  - `singfac_min`, `n_min` - set each surface's half-stencil `dpsi = singfac_min/(n_min·|q′|)`; the
    bracket half-width is `bracket_coef·dpsi`, floored at `min_spacing`
"""
function refined_psi_grid(equil::PlasmaEquilibrium;
    tau::Float64,
    kin::Union{Nothing,KineticProfileSplines}=nothing,
    mandatory::Vector{Float64}=Float64[],
    pinned::Vector{Float64}=Float64[],
    singfac_min::Float64=1e-4,
    n_min::Int=1,
    bracket_coef::Float64=BRACKET_COEF,
    min_spacing::Float64=MIN_KNOT_SPACING,
    N_cap::Int=REFINED_N_CAP)
    xs = equil.profiles.xs
    rho = _knot_density(equil; tau, kin)
    # Fixed fine patch around each rational surface (see RATIONAL_RES_SPACING).
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
    # Pinned knots go in before bracketing so a rational's bracket wins where the two overlap.
    grid = isempty(pinned) ? grid : merge_mandatory_nodes(grid, pinned)
    isempty(mandatory) && return grid
    min_half_widths = [max(bracket_coef * singfac_min / (n_min * abs(equil.profiles.q_deriv(m))), min_spacing) for m in mandatory]
    return bracket_mandatory_nodes(grid, mandatory, min_half_widths, min_spacing)
end

"""
    implied_knot_count(equil::PlasmaEquilibrium; tau, kin=nothing) -> Int

Knot count the density model implies for a formed equilibrium; a post-refinement diagnostic that
flags a pass-1 grid that under- or over-sampled a feature.
"""
function implied_knot_count(equil::PlasmaEquilibrium; tau::Float64, kin::Union{Nothing,KineticProfileSplines}=nothing)
    xs = equil.profiles.xs
    rho = _knot_density(equil; tau, kin)
    return ceil(Int, sum(0.5 * (rho[i] + rho[i-1]) * (xs[i] - xs[i-1]) for i in 2:length(xs)))
end
