# GrowthRateExtraction.jl
#
# Extracts tearing-mode growth rates by contour intersection (Re=0 ∩ Im=0) on a
# complex Q-plane scan: finds intersections of the Re(Δ)=0 and Im(Δ)=0 contours,
# classifies each intersection as a root or pole, and applies the "outside Re=0
# contour, above pole" filter for spurious upper-branch roots.
#
# This PR (5/9) handles the regular-grid path via Contour.jl. PR 6 will add
# a scattered-data path (triangulation) for AMR scans.
#
# Algorithm summary:
#   1. Extract Re(Δ) = re_target and Im(Δ) = im_target contour polylines.
#   2. Find all segment-segment intersections of the two contour families.
#   3. For each intersection, find the closest Im=0 contour and classify as
#      a pole if `max(|Re(Δ)|)` along the local arc exceeds `pole_threshold`.
#   4. For each non-pole intersection, find the closest Re=0 contour. If
#      that contour is approximately closed, take a small +γ step along the
#      Im=0 contour and test whether the step lands inside the Re=0 loop.
#      Roots whose +γ step exits the loop AND that lie above the highest
#      pole are filtered out (spurious upper branches).
#   5. Return the highest-γ surviving root in physical units.

using Contour
using DelaunayTriangulation

# ---------------------------------------------------------------------
# Public result struct + main entry point.
# ---------------------------------------------------------------------

"""
    GrowthRateResult

Output of `find_growth_rates`.

| field                | meaning                                               |
|:-------------------- |:----------------------------------------------------- |
| `Q_root`             | Best (highest-γ surviving) root, normalized           |
| `omega_Hz`           | `Re(Q_root) / tauk` — physical rotation frequency     |
| `gamma_Hz`           | `Im(Q_root) / tauk` — physical growth rate            |
| `Q_root_secondary`   | Second-most-unstable root flagged for ambiguity, or   |
|                      | `NaN+NaNim` if the primary root was unambiguous.      |
| `omega_Hz_secondary` | physical ω of the secondary root, or 0 if none        |
| `gamma_Hz_secondary` | physical γ of the secondary root, or 0 if none        |
| `warning_flags`      | `Vector{Symbol}` of warnings raised on `Q_root`:      |
|                      | `:geom`, `:gap` (root accepted with caveat);          |
|                      | `:spurious` when ≥1 contour near-miss was dropped by  |
|                      | the validity gate (parked in `filtered_roots`); or    |
|                      | `:no_root` when NO usable root was found (`Q_root` is |
|                      | `NaN`; `omega_Hz`/`gamma_Hz` fall back to 0 — check   |
|                      | this flag to tell that apart from a true γ≈0 result). |
|                      | Empty if root is clean.                               |
| `valid_roots`        | All non-pole intersections that survived pole filter  |
| `poles`              | Intersections classified as poles                     |
| `filtered_roots`     | Intersections rejected by the above-pole/outside-Re   |
|                      | filter or the new geom+gap recursion                  |
| `re_contours`        | Extracted Re(Δ)=`re_target` polylines                 |
| `im_contours`        | Extracted Im(Δ)=`im_target` polylines                 |
| `pole_threshold`     | Threshold used for pole classification                |
"""
struct GrowthRateResult
    Q_root::ComplexF64
    omega_Hz::Float64
    gamma_Hz::Float64
    Q_root_secondary::ComplexF64
    omega_Hz_secondary::Float64
    gamma_Hz_secondary::Float64
    warning_flags::Vector{Symbol}
    valid_roots::Vector{ComplexF64}
    poles::Vector{ComplexF64}
    filtered_roots::Vector{ComplexF64}
    re_contours::Vector{Vector{ComplexF64}}
    im_contours::Vector{Vector{ComplexF64}}
    pole_threshold::Float64
end

"""
    find_growth_rates(scan::ScanResult, tauk::Real;
                       re_target=0.0, im_target=0.0,
                       pole_threshold=10.0,
                       filter_above_poles=true,
                       filter_outside_re=true,
                       gap_kHz_threshold=1.0) -> GrowthRateResult

Extract tearing growth-rate eigenvalues from a brute-force `ScanResult` by
contour-intersection analysis. `tauk` is the per-surface time normalization
used to convert `Q` back to physical (Hz) units (`SurfaceCoupling.tauk` for
single-surface scans; `mc.surfaces[mc.ref_idx].tauk` for coupled scans).

# Keyword arguments

  - `re_target`, `im_target` -- contour levels (zero for vanilla dispersion
    root-finding; nonzero values let the caller probe iso-residual contours)
  - `pole_threshold`   -- intersection is classified as a pole when
    `max(|Re(Δ)|)` along the local arc of the nearest Im=0 contour exceeds
    this value
  - `filter_above_poles` -- discard roots whose γ exceeds the highest pole γ
  - `filter_outside_re`  -- restrict the above-pole rejection to roots whose
    +γ step along the Im=0 contour exits the Re=0 contour loop. When `true`,
    roots that are above a pole but geometrically inside the Re=0 contour
    survive (matches the Python default). Note this gate fails when the
    Re=0 contour is OPEN (e.g., exits the Q box edge), letting spurious
    upper-branch roots through; the `:geom` and `:gap` flags below cover
    that case.
  - `gap_kHz_threshold` -- if the highest-γ root is unstable (γ > 0) AND its
    γ exceeds the next root by more than this many kHz, it is flagged as
    a `:gap` warning. Default 1.0 kHz.
  - `residual` -- optional dispersion-residual callable `f(Q::Complex)`. When
    supplied, each contour-intersection root is POLISHED to the true zero of
    `f` by a bounded, neighbour-aware local solve (see `_polish_root`), making
    the extracted root resolution- and thread-independent. `nothing` (default)
    reports the raw marching-squares estimate, which is sensitive to grid and
    floating-point rounding.
  - `polish_maxit` -- max polish iterations per root (default 20). Each costs a
    handful of `f` evaluations; failures fall back to the unpolished point.
  - `validity_rtol` -- with `residual` supplied, a polished root is kept only if
    `|residual| < validity_rtol · median(|Δ|)` (default 1e-3). Intersections
    that don't polish to a true zero (contour near-misses, or surfaces whose
    Δ' BVP failed → `|Δ'|` huge and uncancellable) are dropped from
    `valid_roots`, parked in `filtered_roots`, and flagged `:spurious`. If every
    candidate is spurious, the result carries `:no_root`.

# Spurious-root recursion

After the per-intersection pole / above-pole filters, the remaining roots
are sorted by descending γ. The selection loop walks down this list and at
each candidate evaluates two flags:

  - `:geom` — Re(Δ)=0 contour is locally a downward-concave "hill" at the
    candidate (clean polyline-following quadratic fit).
  - `:gap`  — candidate is unstable AND its γ exceeds the next root's by
    more than `gap_kHz_threshold` kHz.

If BOTH fire, the candidate is discarded as spurious and the next-most-
unstable root is tried. If exactly ONE fires, the candidate is accepted as
primary with that warning recorded, and the next root is exposed as
`Q_root_secondary` so downstream tools can plot or reanalyse it. If
neither fires, the candidate is accepted cleanly.
"""
function find_growth_rates(scan::ScanResult, tauk::Real;
    re_target::Real=0.0, im_target::Real=0.0,
    pole_threshold::Real=10.0,
    filter_above_poles::Bool=true,
    filter_outside_re::Bool=true,
    gap_kHz_threshold::Real=1.0,
    residual=nothing,
    polish_maxit::Integer=20,
    validity_rtol::Real=1e-3)
    return _extract_growth_rates(scan.re_axis, scan.im_axis, scan.Δ,
        Float64(tauk);
        re_target=Float64(re_target),
        im_target=Float64(im_target),
        pole_threshold=Float64(pole_threshold),
        filter_above_poles=filter_above_poles,
        filter_outside_re=filter_outside_re,
        gap_kHz_threshold=Float64(gap_kHz_threshold),
        residual=residual,
        polish_maxit=Int(polish_maxit),
        validity_scale=_residual_scale(scan.Δ),
        validity_rtol=Float64(validity_rtol))
end

"""
    find_growth_rates(amr::AMRResult, tauk::Real;
                       re_target=0.0, im_target=0.0,
                       pole_threshold=10.0,
                       filter_above_poles=true,
                       filter_outside_re=true) -> GrowthRateResult

Extract tearing growth-rate eigenvalues from an AMR `AMRResult` via Delaunay
triangulation + marching triangles on the scattered evaluation points. The
pipeline after contour extraction (segment intersection, pole classification,
outside-Re filter, physical-Hz conversion) is identical to the brute-force
grid path — only the contour extractor changes. Hanging-node issues from the
quadtree's mixed refinement levels are resolved by the triangulation
respecting every evaluated point uniformly.
"""
function find_growth_rates(amr::AMRResult, tauk::Real;
    re_target::Real=0.0, im_target::Real=0.0,
    pole_threshold::Real=10.0,
    filter_above_poles::Bool=true,
    filter_outside_re::Bool=true,
    gap_kHz_threshold::Real=1.0,
    residual=nothing,
    polish_maxit::Integer=20,
    validity_rtol::Real=1e-3)
    return _extract_growth_rates_amr(amr.Q, amr.Δ, Float64(tauk);
        re_target=Float64(re_target),
        im_target=Float64(im_target),
        pole_threshold=Float64(pole_threshold),
        filter_above_poles=filter_above_poles,
        filter_outside_re=filter_outside_re,
        gap_kHz_threshold=Float64(gap_kHz_threshold),
        residual=residual,
        polish_maxit=Int(polish_maxit),
        validity_scale=_residual_scale(amr.Δ),
        validity_rtol=Float64(validity_rtol))
end

# ---------------------------------------------------------------------
# Implementation.
# ---------------------------------------------------------------------

# Bilinear interpolation of `values` on the regular grid `(re_axis, im_axis)`
# at point (qr, qi). Out-of-grid points are clamped to the boundary.
function _bilinear(re_axis::Vector{Float64}, im_axis::Vector{Float64},
    values::Matrix{Float64}, qr::Real, qi::Real)
    nre = length(re_axis)
    nim = length(im_axis)
    i = clamp(searchsortedlast(re_axis, qr), 1, nre - 1)
    j = clamp(searchsortedlast(im_axis, qi), 1, nim - 1)
    tx = (qr - re_axis[i]) / (re_axis[i+1] - re_axis[i])
    ty = (qi - im_axis[j]) / (im_axis[j+1] - im_axis[j])
    tx = clamp(tx, 0.0, 1.0)
    ty = clamp(ty, 0.0, 1.0)
    return (1 - tx) * (1 - ty) * values[i, j] + tx * (1 - ty) * values[i+1, j] +
           (1 - tx) * ty * values[i, j+1] + tx * ty * values[i+1, j+1]
end

# Extract polylines for a single contour level on a regular grid.
# Returns Vector{Vector{ComplexF64}} (one polyline per closed/open curve).
function _extract_contours(re_axis::Vector{Float64}, im_axis::Vector{Float64},
    values::Matrix{Float64}, level::Float64)
    polylines = Vector{Vector{ComplexF64}}()
    for cl in lines(contour(re_axis, im_axis, values, level))
        xs, ys = coordinates(cl)
        path = ComplexF64[xs[i] + ys[i] * im for i in eachindex(xs)]
        length(path) >= 2 && push!(polylines, path)
    end
    return polylines
end

# Segment-segment intersection on the complex plane. Returns the
# intersection point if segments [a,b] and [c,d] cross strictly (parameters
# in (0,1)), else nothing. Endpoint touches return the touch point.
function _segment_intersection(a::ComplexF64, b::ComplexF64,
    c::ComplexF64, d::ComplexF64)
    d1r, d1i = real(b - a), imag(b - a)
    d2r, d2i = real(d - c), imag(d - c)
    denom = d1r * d2i - d1i * d2r
    abs(denom) < 1e-30 && return nothing      # parallel or degenerate
    diffr, diffi = real(c - a), imag(c - a)
    t = (diffr * d2i - diffi * d2r) / denom
    u = (diffr * d1i - diffi * d1r) / denom
    if 0 <= t <= 1 && 0 <= u <= 1
        return a + t * (b - a)
    end
    return nothing
end

# Find all intersections between two families of polylines. Returns
# Vector{ComplexF64}.
function _all_intersections(re_paths::Vector{Vector{ComplexF64}},
    im_paths::Vector{Vector{ComplexF64}})
    out = ComplexF64[]
    for re_path in re_paths
        for i in 1:length(re_path)-1
            a, b = re_path[i], re_path[i+1]
            for im_path in im_paths
                for j in 1:length(im_path)-1
                    c, d = im_path[j], im_path[j+1]
                    pt = _segment_intersection(a, b, c, d)
                    pt !== nothing && push!(out, pt)
                end
            end
        end
    end
    return out
end

# Index of the closest vertex in a polyline to a point.
function _closest_vertex(path::Vector{ComplexF64}, pt::ComplexF64)
    best_i = 0
    best_d = Inf
    for i in eachindex(path)
        d = abs(path[i] - pt)
        if d < best_d
            best_d = d
            best_i = i
        end
    end
    return best_i, best_d
end

# Find the polyline (and vertex within it) whose vertex is closest to pt.
function _closest_polyline_vertex(paths::Vector{Vector{ComplexF64}},
    pt::ComplexF64)
    best_path_idx = 0
    best_vert_idx = 0
    best_d = Inf
    for (pi_, path) in enumerate(paths)
        vi, d = _closest_vertex(path, pt)
        if d < best_d
            best_d = d
            best_path_idx = pi_
            best_vert_idx = vi
        end
    end
    return best_path_idx, best_vert_idx, best_d
end

# Ray-casting point-in-polygon. `polygon` need not be closed (function
# closes it internally).
function _point_in_polygon(pt::ComplexF64, polygon::Vector{ComplexF64})
    n = length(polygon)
    n < 3 && return false
    inside = false
    pr, pi_ = real(pt), imag(pt)
    j = n
    for i in 1:n
        xi, yi = real(polygon[i]), imag(polygon[i])
        xj, yj = real(polygon[j]), imag(polygon[j])
        if ((yi > pi_) != (yj > pi_)) &&
           (pr < (xj - xi) * (pi_ - yi) / (yj - yi) + xi)
            inside = !inside
        end
        j = i
    end
    return inside
end

# ---------------------------------------------------------------------
# Shared analysis: intersections + pole classification + outside-Re filter.
# Both the regular-grid path (_extract_growth_rates) and the AMR
# triangulation path (_extract_growth_rates_amr) funnel through this.
# ---------------------------------------------------------------------
# Geometric "spurious upper-branch" detector — flags candidates where the
# Re(Δ)=0 contour is locally a downward-concave "hill" or "hump" (⌒) at the
# candidate location. Legitimate tearing roots sit at the bottom of upward-
# concave "wells" (∪); spurious upper-branch roots sit at the top of hills.
#
# Algorithm:
#  1. Find the closest Re=0 polyline + closest vertex on it.
#  2. Walk outward along that polyline, collecting consecutive vertices
#     within `max_walk` Q-distance of the candidate. Walking the polyline
#     (rather than averaging over a radius) avoids polluting the fit with
#     vertices from disconnected nearby Re=0 fragments — important on
#     AMR-triangulated meshes where the contour is fragmented.
#  3. Fit γ = a + b·Δω + c·(Δω)² to the collected vertices via least squares.
#     Sign of `c` is the local concavity:
#        c < 0  → contour is concave-DOWN (hill, ⌒) ← SPURIOUS pattern
#        c > 0  → contour is concave-UP (well, ∪)   ← legitimate pattern
#  4. Gate on fit quality: only flag when RMS_residual / γ_spread is below
#     `quality_threshold`. Noisy fits (e.g. multiple overlapping contour
#     fragments) leave the candidate unflagged — letting the gap criterion
#     and downstream review handle ambiguous cases.
#
# Returns `true` when the candidate is on a CLEAN concave-down arc; else
# `false`. The orientation-invariance of the previous 3-point stencil
# version is preserved because we fit γ = f(ω) which has a sign-stable
# second derivative regardless of traversal direction.
function _is_geom_spurious(pt::ComplexF64,
    re_paths::Vector{Vector{ComplexF64}};
    max_walk::Float64=0.5,
    curvature_threshold::Float64=0.05,
    quality_threshold::Float64=0.15)
    re_idx, re_v_idx, _ = _closest_polyline_vertex(re_paths, pt)
    re_idx == 0 && return false
    re_path = re_paths[re_idx]
    n_path = length(re_path)
    n_path < 5 && return false

    # Walk outward from re_v_idx along the polyline, collecting vertices
    # within max_walk Q-distance of pt. Stop in each direction at the first
    # vertex that exceeds the walk radius.
    collected_idx = Int[re_v_idx]
    @inbounds for k in (re_v_idx+1):n_path
        if abs(re_path[k] - pt) < max_walk
            push!(collected_idx, k)
        else
            break
        end
    end
    @inbounds for k in (re_v_idx-1):-1:1
        if abs(re_path[k] - pt) < max_walk
            push!(collected_idx, k)
        else
            break
        end
    end
    n = length(collected_idx)
    n < 5 && return false

    ω₀ = real(pt)
    ωs = Vector{Float64}(undef, n)
    γs = Vector{Float64}(undef, n)
    @inbounds for (i, k) in enumerate(collected_idx)
        ωs[i] = real(re_path[k]) - ω₀
        γs[i] = imag(re_path[k])
    end
    ω_sp = maximum(ωs) - minimum(ωs)
    γ_sp = maximum(γs) - minimum(γs)
    (ω_sp < 1e-6 || γ_sp < 1e-12) && return false

    # Quadratic least-squares fit γ = a + b·ω + c·ω² via the normal equations
    # MᵀM·coeffs = Mᵀγ, where M = [1 ω ω²]. Hand-rolled to avoid an allocation
    # for the n×3 design matrix (we just need the 3×3 normal-equation matrix).
    sx = 0.0
    sx2 = 0.0
    sx3 = 0.0
    sx4 = 0.0
    sy = 0.0
    sxy = 0.0
    sx2y = 0.0
    @inbounds for i in 1:n
        ω = ωs[i]
        γ = γs[i]
        ω2 = ω * ω
        sx += ω
        sx2 += ω2
        sx3 += ω2 * ω
        sx4 += ω2 * ω2
        sy += γ
        sxy += ω * γ
        sx2y += ω2 * γ
    end
    M = [Float64(n) sx sx2;
        sx sx2 sx3;
        sx2 sx3 sx4]
    rhs = [sy, sxy, sx2y]
    coeffs = M \ rhs
    c = coeffs[3]

    # Fit-quality residual norm
    rms_sq = 0.0
    @inbounds for i in 1:n
        pred = coeffs[1] + coeffs[2] * ωs[i] + coeffs[3] * ωs[i]^2
        rms_sq += (γs[i] - pred)^2
    end
    rms = sqrt(rms_sq / n)
    rms_norm = rms / γ_sp

    # Spurious if concave-down AND fit is clean enough to trust
    return c < -curvature_threshold && rms_norm < quality_threshold
end

# γ-gap separation: the candidate at `idx` (in γ-descending order) is unstable
# AND clearly separated above the next-most-unstable candidate by more than
# `gap_kHz_threshold` kHz. Flags an outlier "lone peak" root.
function _is_gap_spurious(sorted_roots::Vector{ComplexF64}, idx::Int,
    tauk::Float64, gap_kHz_threshold::Float64)
    γ_idx = imag(sorted_roots[idx]) / tauk * 1e-3   # kHz
    γ_idx > 0.0 || return false                       # only suspicious if unstable
    idx >= length(sorted_roots) && return false       # nothing below to compare
    γ_next = imag(sorted_roots[idx+1]) / tauk * 1e-3
    return (γ_idx - γ_next) > gap_kHz_threshold
end

# ---------------------------------------------------------------------
# Root polishing (isolate-then-polish).
#
# The contour scan only *locates* roots to the grid/cell resolution: the
# marching-squares intersection is a linear estimate of where Re(f)=0 and
# Im(f)=0 cross, so the residual there is small only relative to the scan's
# huge dynamic range, not zero. Its exact position is sensitive to ULP-level
# jitter in the sampled `f` values (BLAS/thread reduction order), which can
# shift the reported γ between near-degenerate estimates of the SAME root.
#
# `_polish_root` refines one contour intersection to the actual zero of `f`
# by bounded minimization of `|f(Q)|`, so the reported root becomes
# resolution- and thread-independent. It is a no-op-or-better layer: any
# failure (non-finite, no decrease, pole approach) falls back to the
# unpolished contour point.
# ---------------------------------------------------------------------

# Median contour-segment length — a robust proxy for the local grid/cell
# resolution, used to cap how far a polish may move a root.
function _median_segment_length(re_paths::Vector{Vector{ComplexF64}},
    im_paths::Vector{Vector{ComplexF64}})
    lens = Float64[]
    for paths in (re_paths, im_paths), p in paths
        @inbounds for i in 1:length(p)-1
            push!(lens, abs(p[i+1] - p[i]))
        end
    end
    isempty(lens) && return 0.0
    sort!(lens)
    return lens[cld(length(lens), 2)]
end

# Trust-region radius for polishing the intersection at `pts[idx]`. Capped by
# `nn_frac` × distance to the NEAREST other intersection (root OR pole), so a
# polished root can never migrate into a neighbour's basin — `nn_frac < 0.5`
# guarantees, by the triangle inequality, that the result stays strictly
# closer to its own contour point than to any other. Also capped at a few grid
# cells (`k_cell·h`) so refinement only corrects the discretization error.
function _polish_trust_radius(pts::Vector{ComplexF64}, idx::Int, h_grid::Real;
    k_cell::Float64=3.0, nn_frac::Float64=0.45)
    p0 = pts[idx]
    d_nn = Inf
    @inbounds for j in eachindex(pts)
        j == idx && continue
        d = abs(pts[j] - p0)
        d < d_nn && (d_nn = d)
    end
    r_cell = k_cell * Float64(h_grid)
    r = isfinite(d_nn) ? min(r_cell, nn_frac * d_nn) : r_cell
    return r > 0 ? r : r_cell
end

# Bounded Gauss-Newton minimization of |f(Q)| within a disk of radius `R`
# around `Q0`. Returns (Q_best, n_evals, improved). `improved == false` ⇒
# Q0 is returned unchanged (hard fallback). Secant derivative + trust-region
# projection + |f|-decrease backtracking keep it strictly "improve-or-no-op":
# it cannot diverge, jump to another root, or be pulled into a pole.
function _polish_root(f, Q0::ComplexF64, R::Float64; maxit::Int=8,
    tol_step::Float64=1e-12, tol_f::Float64=1e-8)
    (R > 0) || return (Q0, 0, false)
    f0 = ComplexF64(f(Q0))
    nev = 1
    isfinite(f0) || return (Q0, nev, false)
    a0 = abs(f0)
    a0 == 0 && return (Q0, nev, false)

    # Second point for the secant slope: a small real offset scaled to R.
    Qk, fk = Q0, f0
    Qp = Q0 + complex(max(R * 1e-3, eps(Float64)), 0.0)
    fp = ComplexF64(f(Qp))
    nev += 1
    Qbest, abest = Q0, a0

    for _ in 1:maxit
        df = (Qp == Qk) ? ComplexF64(0) : (fp - fk) / (Qp - Qk)
        (isfinite(df) && abs(df) > 0) || break
        step = -fk / df                                   # zero of local linear model
        Qt = Qk + step
        if abs(Qt - Q0) > R                               # project into trust region
            Qt = Q0 + R * (Qt - Q0) / abs(Qt - Q0)
        end
        ft = ComplexF64(f(Qt))
        nev += 1
        bt = 0
        while (!isfinite(ft) || abs(ft) >= abest) && bt < 3      # backtrack on no-decrease
            Qt = Qk + 0.5 * (Qt - Qk)
            (abs(Qt - Q0) <= R) || break
            ft = ComplexF64(f(Qt))
            nev += 1
            bt += 1
        end
        if isfinite(ft) && abs(ft) < abest
            Qp, fp = Qk, fk
            Qk, fk = Qt, ft
            Qbest, abest = Qt, abs(ft)
            (abs(step) < tol_step * (abs(Qk) + tol_step) || abest < tol_f * a0) && break
        else
            break                                          # no further decrease → stop
        end
    end
    return (abest < a0 ? Qbest : Q0, nev, abest < a0)
end

# Median |Δ| over the finite scan values — the "typical residual magnitude"
# reference for the validity gate. A genuine root drives the polished |residual|
# orders of magnitude below this; a near-miss stays near it.
function _residual_scale(Δ)
    a = Float64[]
    for z in Δ
        az = abs(z)
        (isfinite(az) && az < 1e30) && push!(a, az)
    end
    isempty(a) && return 0.0
    sort!(a)
    n = length(a)
    return isodd(n) ? a[(n+1)÷2] : 0.5 * (a[n÷2] + a[n÷2+1])
end

function _run_analysis(re_paths::Vector{Vector{ComplexF64}},
    im_paths::Vector{Vector{ComplexF64}},
    im_re_vals::Vector{Vector{Float64}},
    tauk::Float64;
    pole_threshold::Float64,
    filter_above_poles::Bool,
    filter_outside_re::Bool,
    gap_kHz_threshold::Float64=1.0,
    residual=nothing,
    polish_maxit::Int=20,
    validity_scale::Float64=0.0,
    validity_rtol::Float64=1e-3)
    raw_intersections = _all_intersections(re_paths, im_paths)

    poles = ComplexF64[]
    candidates = Tuple{ComplexF64,Bool}[]    # (pt, on_top_half_re_flag)
    spurious_roots = ComplexF64[]               # polished, but |residual| ≫ 0

    # Isolate-then-polish: when a residual callable is supplied, refine each
    # (non-pole) contour intersection to the true zero of `f`. Trust radii are
    # capped by the spacing to neighbouring intersections so closely-spaced
    # coupled roots stay coherently bound to their own contour crossing.
    do_polish = residual !== nothing
    h_grid = do_polish ? _median_segment_length(re_paths, im_paths) : 0.0
    # Validity gate: a genuine root drives |residual| ≈ 0; a spurious contour
    # near-miss (e.g. a surface whose Δ' BVP failed → |Δ'| huge and
    # uncancellable by the inner layer) leaves |residual| stuck near the typical
    # scan magnitude. Only active when polishing AND a scale is supplied.
    do_gate = do_polish && validity_scale > 0
    polish_evals = 0

    for (i_pt, pt) in enumerate(raw_intersections)
        # --- 1. classify as pole or root via local Re-magnitude on Im contour
        best_im_path_idx, best_im_vert_idx, _ =
            _closest_polyline_vertex(im_paths, pt)
        is_pole = false
        if best_im_path_idx > 0
            re_vals = im_re_vals[best_im_path_idx]
            n = length(re_vals)
            i_prev = max(1, best_im_vert_idx - 1)
            i_next = min(n, best_im_vert_idx + 1)
            local_max = max(abs(re_vals[i_prev]),
                abs(re_vals[i_next]),
                abs(re_vals[best_im_vert_idx]))
            is_pole = local_max > pole_threshold
        end

        if is_pole
            push!(poles, pt)
            continue
        end

        # --- 1b. polish this (non-pole) intersection to the true zero of `f`,
        # bounded to a neighbour-aware trust region (no-op on failure).
        if do_polish
            R = _polish_trust_radius(raw_intersections, i_pt, h_grid)
            pt, nev, _ = _polish_root(residual, pt, R; maxit=polish_maxit)
            polish_evals += nev
            # --- 1c. validity gate: drop intersections that don't polish to a
            # true zero (|residual| stays ≫ 0 relative to the scan scale). These
            # are contour near-misses / failed-BVP surfaces, not real roots.
            if do_gate
                rmag = abs(ComplexF64(residual(pt)))
                polish_evals += 1
                if !(rmag < validity_rtol * validity_scale)   # NaN ⇒ spurious
                    push!(spurious_roots, pt)
                    continue
                end
            end
        end

        # --- 2. "+γ step inside Re contour" flag for spurious-upper-branch filter
        on_top_half_re = false
        best_re_path_idx, _, _ = _closest_polyline_vertex(re_paths, pt)
        if best_im_path_idx > 0 && best_re_path_idx > 0
            re_path = re_paths[best_re_path_idx]
            xs = real.(re_path)
            ys = imag.(re_path)
            contour_extent = max(maximum(xs) - minimum(xs),
                maximum(ys) - minimum(ys))
            closure_gap = abs(re_path[1] - re_path[end])

            if contour_extent > 0 && closure_gap < 0.1 * contour_extent
                # Re=0 contour is approximately closed → containment test applies
                im_path = im_paths[best_im_path_idx]
                n_im = length(im_path)
                im_nearest = best_im_vert_idx
                i_a = min(im_nearest + 1, n_im)
                i_b = max(im_nearest - 1, 1)
                gamma_a = imag(im_path[i_a])
                gamma_b = imag(im_path[i_b])
                gamma_here = imag(im_path[im_nearest])

                tangent = if gamma_a >= gamma_b && gamma_a > gamma_here
                    im_path[i_a] - im_path[im_nearest]
                elseif gamma_b > gamma_here
                    im_path[i_b] - im_path[im_nearest]
                else
                    ComplexF64(0.0, 1.0)        # fall back to straight up
                end

                tlen = abs(tangent)
                if tlen > 0
                    step_size = 0.01 * contour_extent
                    step_pt = pt + (step_size / tlen) * tangent
                    inside = _point_in_polygon(step_pt, re_path)
                    on_top_half_re = !inside
                end
            end
        end

        push!(candidates, (pt, on_top_half_re))
    end

    # --- 3. pole + closed-loop filter (legacy), then geom + gap recursion (new)
    valid_roots = ComplexF64[c[1] for c in candidates]
    filtered_roots = copy(spurious_roots)     # gate-rejected near-misses kept here
    Q_root = ComplexF64(NaN, NaN)
    Q_root_2nd = ComplexF64(NaN, NaN)
    warning_flags = Symbol[]
    isempty(spurious_roots) || push!(warning_flags, :spurious)

    if !isempty(valid_roots)
        order = sortperm(valid_roots; by=q -> -imag(q))
        sorted_pts = valid_roots[order]
        sorted_top = Bool[c[2] for c in candidates][order]

        max_pole_gamma = isempty(poles) ? -Inf : maximum(imag, poles)

        chosen_idx = 0
        for k in 1:length(sorted_pts)
            cand = sorted_pts[k]
            top_re = sorted_top[k]
            # Legacy filter: above-pole + closed-loop outside-Re
            legacy_reject = filter_above_poles && imag(cand) > max_pole_gamma &&
                            (!filter_outside_re || top_re)
            if legacy_reject
                push!(filtered_roots, cand)
                continue
            end
            # New checks: 2 spurious-root flags — :geom and :gap.
            #   :geom — Re=0 contour is locally a downward-concave "hill"
            #           at the candidate (clean polyline-following fit)
            #   :gap  — candidate is unstable AND >1 kHz above next root
            #           (isolated γ peak — spurious outlier signature)
            #
            # Policy: WARN, DO NOT DISCARD. The both-flags-fire criterion
            # is too aggressive in the kink-approach regime where valid
            # roots become sparse — a few-kHz γ separation between the
            # dominant unstable root and the next-stable root can be
            # genuine dispersion structure rather than a "lone peak"
            # artifact, yet :gap fires regardless. So every candidate is
            # accepted with whatever warnings apply, and downstream tools
            # see the same valid_roots regardless of flag combination.
            # filtered_roots is preserved for the legacy above-pole +
            # outside-Re reject branch only.
            geom_flag = _is_geom_spurious(cand, re_paths)
            gap_flag = _is_gap_spurious(sorted_pts, k, tauk,
                gap_kHz_threshold)
            chosen_idx = k
            geom_flag && push!(warning_flags, :geom)
            gap_flag && push!(warning_flags, :gap)
            break
        end

        if chosen_idx > 0
            Q_root = sorted_pts[chosen_idx]
            # When a warning fired, expose the next-down root as secondary so
            # downstream tools can plot/reanalyse. (Indices > chosen_idx in
            # sorted_pts are the next-most-unstable.)
            if !isempty(warning_flags) && chosen_idx < length(sorted_pts)
                Q_root_2nd = sorted_pts[chosen_idx+1]
            end
        end
    end

    # No usable root: either the scan produced no zero-crossing intersections
    # (`valid_roots` empty) or every candidate was rejected by the legacy
    # above-pole filter (`chosen_idx == 0`). Flag this explicitly so callers
    # can distinguish a genuine marginally-stable result (γ ≈ 0) from a failed
    # extraction — both otherwise surface as `gamma_Hz == 0.0` below.
    isnan(real(Q_root)) && push!(warning_flags, :no_root)

    omega_Hz = isnan(real(Q_root)) ? 0.0 : real(Q_root) / tauk
    gamma_Hz = isnan(imag(Q_root)) ? 0.0 : imag(Q_root) / tauk
    omega_Hz_2nd = isnan(real(Q_root_2nd)) ? 0.0 : real(Q_root_2nd) / tauk
    gamma_Hz_2nd = isnan(imag(Q_root_2nd)) ? 0.0 : imag(Q_root_2nd) / tauk

    return GrowthRateResult(Q_root, omega_Hz, gamma_Hz,
        Q_root_2nd, omega_Hz_2nd, gamma_Hz_2nd,
        warning_flags,
        valid_roots, poles, filtered_roots,
        re_paths, im_paths, pole_threshold)
end

# Regular-grid path: extract contours via Contour.jl, compute im_re_vals by
# bilinear interpolation on the grid, then run the shared analysis.
function _extract_growth_rates(re_axis::Vector{Float64},
    im_axis::Vector{Float64},
    Δ_grid::Matrix{ComplexF64},
    tauk::Float64;
    re_target::Float64,
    im_target::Float64,
    pole_threshold::Float64,
    filter_above_poles::Bool,
    filter_outside_re::Bool,
    gap_kHz_threshold::Float64=1.0,
    residual=nothing,
    polish_maxit::Int=20,
    validity_scale::Float64=0.0,
    validity_rtol::Float64=1e-3)
    re_field = real.(Δ_grid)
    im_field = imag.(Δ_grid)

    re_paths = _extract_contours(re_axis, im_axis, re_field, re_target)
    im_paths = _extract_contours(re_axis, im_axis, im_field, im_target)

    im_re_vals = [Float64[_bilinear(re_axis, im_axis, re_field,
        real(v), imag(v))
                          for v in path]
                  for path in im_paths]

    return _run_analysis(re_paths, im_paths, im_re_vals, tauk;
        pole_threshold=pole_threshold,
        filter_above_poles=filter_above_poles,
        filter_outside_re=filter_outside_re,
        gap_kHz_threshold=gap_kHz_threshold,
        residual=residual,
        polish_maxit=polish_maxit,
        validity_scale=validity_scale,
        validity_rtol=validity_rtol)
end

# ---------------------------------------------------------------------
# AMR path: Delaunay triangulation + marching triangles. Hanging nodes
# from the quadtree's mixed refinement levels become first-class vertices
# in the triangulation, so contour segments piece together without gaps.
# ---------------------------------------------------------------------

# Emit a Re=0 and Im=0 segment (if any) from a single triangle. Returns
# `(re_seg, im_seg)` where each may be `nothing`. A segment is a
# `@NamedTuple{p1::ComplexF64, p2::ComplexF64, a1::Float64, a2::Float64}`
# where `a1`, `a2` carry the *complementary* field value at the endpoints
# (Re-value for Im=0 segments, Im-value for Re=0 segments).
function _march_triangle(p1::ComplexF64, p2::ComplexF64, p3::ComplexF64,
    v1::ComplexF64, v2::ComplexF64, v3::ComplexF64,
    re_target::Float64, im_target::Float64)
    return (_march_single(p1, p2, p3, real(v1), real(v2), real(v3),
            imag(v1), imag(v2), imag(v3), re_target),
        _march_single(p1, p2, p3, imag(v1), imag(v2), imag(v3),
            real(v1), real(v2), real(v3), im_target))
end

# Core marching step for one scalar field `f` with complementary field `g`.
# Produces the contour segment at level=L (if any) along with the value of
# `g` linearly interpolated at each endpoint.
@inline function _march_single(p1::ComplexF64, p2::ComplexF64, p3::ComplexF64,
    f1::Float64, f2::Float64, f3::Float64,
    g1::Float64, g2::Float64, g3::Float64,
    L::Float64)
    a1 = f1 >= L
    a2 = f2 >= L
    a3 = f3 >= L
    count = Int(a1) + Int(a2) + Int(a3)
    (count == 0 || count == 3) && return nothing

    # Identify the "odd" vertex and produce crossings on the two edges
    # incident to it.
    if a1 != a2 && a1 != a3
        pt_a, ga = _cross_edge(p1, p2, f1, f2, g1, g2, L)
        pt_b, gb = _cross_edge(p1, p3, f1, f3, g1, g3, L)
    elseif a2 != a1 && a2 != a3
        pt_a, ga = _cross_edge(p2, p1, f2, f1, g2, g1, L)
        pt_b, gb = _cross_edge(p2, p3, f2, f3, g2, g3, L)
    else
        pt_a, ga = _cross_edge(p3, p1, f3, f1, g3, g1, L)
        pt_b, gb = _cross_edge(p3, p2, f3, f2, g3, g2, L)
    end
    return (p1=pt_a, p2=pt_b, a1=ga, a2=gb)
end

# Linear crossing on edge (pa, pb) for field `f` at level `L`, with
# complementary value `g` interpolated at the same parameter.
@inline function _cross_edge(pa::ComplexF64, pb::ComplexF64,
    fa::Float64, fb::Float64,
    ga::Float64, gb::Float64, L::Float64)
    denom = fb - fa
    t = denom == 0 ? 0.0 : (L - fa) / denom
    t = clamp(t, 0.0, 1.0)
    return (pa + t * (pb - pa), ga + t * (gb - ga))
end

# Chain segments into polylines by endpoint matching. Each segment endpoint
# is a `ComplexF64` that is shared bit-exactly with any adjacent triangle's
# crossing (both sides of a triangulation edge compute the same linear
# crossing from identical endpoint values). Returns
# `(paths::Vector{Vector{ComplexF64}}, aux::Vector{Vector{Float64}})`.
function _chain_segments(segs::Vector{<:NamedTuple})
    # Build an endpoint → list-of-segment-indices adjacency map.
    adj = Dict{ComplexF64,Vector{Int}}()
    for (i, s) in enumerate(segs)
        push!(get!(adj, s.p1, Int[]), i)
        push!(get!(adj, s.p2, Int[]), i)
    end

    used = falses(length(segs))
    paths = Vector{Vector{ComplexF64}}()
    aux_vals = Vector{Vector{Float64}}()

    # Walk a polyline starting from segment `start_seg` via endpoint
    # `start_pt`; returns the path and aux values.
    function _walk(start_seg::Int, start_pt::ComplexF64)
        path = ComplexF64[start_pt]
        aux = Float64[]
        # Emit the aux value for start_pt on the first segment
        s0 = segs[start_seg]
        push!(aux, start_pt == s0.p1 ? s0.a1 : s0.a2)

        cur_seg = start_seg
        cur_pt = start_pt
        while true
            used[cur_seg] = true
            s = segs[cur_seg]
            next_pt = cur_pt == s.p1 ? s.p2 : s.p1
            next_aux = cur_pt == s.p1 ? s.a2 : s.a1
            push!(path, next_pt)
            push!(aux, next_aux)

            nbrs = adj[next_pt]
            nxt = 0
            for j in nbrs
                if !used[j] && j != cur_seg
                    nxt = j
                    break
                end
            end
            nxt == 0 && break
            cur_seg = nxt
            cur_pt = next_pt
        end
        return path, aux
    end

    # Open polylines first: start from any endpoint touched by exactly
    # one still-unused segment.
    for (pt, nbrs) in adj
        count = 0
        start_seg = 0
        for j in nbrs
            if !used[j]
                count += 1
                start_seg = j
            end
        end
        if count == 1
            path, aux = _walk(start_seg, pt)
            length(path) >= 2 && (push!(paths, path); push!(aux_vals, aux))
        end
    end

    # Remaining segments form closed loops.
    for i in eachindex(segs)
        used[i] && continue
        path, aux = _walk(i, segs[i].p1)
        length(path) >= 2 && (push!(paths, path); push!(aux_vals, aux))
    end

    return paths, aux_vals
end

# AMR entry point: triangulate the scattered (Q, Δ) points, march triangles
# to extract Re=0 and Im=0 contour segments with complementary-field values
# at endpoints, chain into polylines, then run the shared analysis.
function _extract_growth_rates_amr(Q::Vector{ComplexF64},
    Δ::Vector{ComplexF64},
    tauk::Float64;
    re_target::Float64,
    im_target::Float64,
    pole_threshold::Float64,
    filter_above_poles::Bool,
    filter_outside_re::Bool,
    gap_kHz_threshold::Float64=1.0,
    residual=nothing,
    polish_maxit::Int=20,
    validity_scale::Float64=0.0,
    validity_rtol::Float64=1e-3)
    length(Q) == length(Δ) ||
        throw(ArgumentError("_extract_growth_rates_amr: length(Q) ≠ length(Δ)"))
    length(Q) >= 3 ||
        throw(ArgumentError("_extract_growth_rates_amr: need ≥ 3 points to triangulate"))

    pts = [(real(q), imag(q)) for q in Q]
    tri = triangulate(pts)

    # Segment types (carry complementary-field value at each endpoint)
    re_segs = NamedTuple{(:p1, :p2, :a1, :a2),
        Tuple{ComplexF64,ComplexF64,Float64,Float64}}[]
    im_segs = NamedTuple{(:p1, :p2, :a1, :a2),
        Tuple{ComplexF64,ComplexF64,Float64,Float64}}[]

    for T in each_solid_triangle(tri)
        i1, i2, i3 = T
        p1 = Q[i1]
        p2 = Q[i2]
        p3 = Q[i3]
        v1 = Δ[i1]
        v2 = Δ[i2]
        v3 = Δ[i3]
        re_seg, im_seg = _march_triangle(p1, p2, p3, v1, v2, v3,
            re_target, im_target)
        re_seg !== nothing && push!(re_segs, re_seg)
        im_seg !== nothing && push!(im_segs, im_seg)
    end

    re_paths, _ = _chain_segments(re_segs)
    im_paths, im_re_vals = _chain_segments(im_segs)

    return _run_analysis(re_paths, im_paths, im_re_vals, tauk;
        pole_threshold=pole_threshold,
        filter_above_poles=filter_above_poles,
        filter_outside_re=filter_outside_re,
        gap_kHz_threshold=gap_kHz_threshold,
        residual=residual,
        polish_maxit=polish_maxit,
        validity_scale=validity_scale,
        validity_rtol=validity_rtol)
end
