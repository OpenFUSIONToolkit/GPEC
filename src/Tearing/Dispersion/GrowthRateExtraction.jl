# GrowthRateExtraction.jl
#
# Julia port of CTM-processing/shared/find_growthrates.py: extract tearing
# growth-rate eigenvalues from a 2D Q-plane scan by finding intersections of
# the Re(Δ)=0 and Im(Δ)=0 contours, classifying each intersection as a root
# or pole, and applying the "outside Re=0 contour, above pole" filter for
# spurious upper-branch roots.
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

| field                | meaning                                                |
|----------------------|--------------------------------------------------------|
| `Q_root`             | Best (highest-γ surviving) root, normalized            |
| `omega_Hz`           | `Re(Q_root) / tauk` — physical rotation frequency      |
| `gamma_Hz`           | `Im(Q_root) / tauk` — physical growth rate             |
| `Q_root_secondary`   | Second-most-unstable root flagged for ambiguity, or    |
|                      | `NaN+NaNim` if the primary root was unambiguous.       |
| `omega_Hz_secondary` | physical ω of the secondary root, or 0 if none         |
| `gamma_Hz_secondary` | physical γ of the secondary root, or 0 if none         |
| `warning_flags`      | `Vector{Symbol}` of warnings raised on `Q_root`:       |
|                      | `:geom`, `:gap`. Empty if root is clean.               |
| `valid_roots`        | All non-pole intersections that survived pole filter   |
| `poles`              | Intersections classified as poles                      |
| `filtered_roots`     | Intersections rejected by the above-pole/outside-Re    |
|                      | filter or the new geom+gap recursion                   |
| `re_contours`        | Extracted Re(Δ)=`re_target` polylines                  |
| `im_contours`        | Extracted Im(Δ)=`im_target` polylines                  |
| `pole_threshold`     | Threshold used for pole classification                 |
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
                       gap_kHz_threshold=1.0,
                       angle_threshold_deg=45.0) -> GrowthRateResult

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
    upper-branch roots through. The `angle_threshold_deg` and
    `gap_kHz_threshold` checks below cover that case.
  - `gap_kHz_threshold` -- if the highest-γ root is unstable (γ > 0) AND its
    γ exceeds the next root by more than this many kHz, it is flagged as
    a `:gap` warning. Default 1.0 kHz.
  - `angle_threshold_deg` -- a candidate is flagged with `:geom` warning if
    it sits where the Re(Δ)=0 contour is locally downward-concave AND the
    Im(Δ)=0 tangent makes an angle greater than this (in degrees) with the
    horizontal. Captures the "spurious upper-branch" geometry that the
    `filter_outside_re` gate misses on open contours. Default 45°.

# Spurious-root recursion

After the per-intersection pole / above-pole filters, the remaining roots
are sorted by descending γ. The selection loop walks down this list and at
each candidate evaluates the two new flags `:geom` (concavity + Im exit
angle) and `:gap` (γ-separation from next root). If BOTH flags fire, the
candidate is discarded as spurious and the next root is tried. If exactly
ONE fires, the candidate is accepted as the primary root but a warning is
recorded in `warning_flags`, and the next root is exposed as
`Q_root_secondary` so downstream tools can plot or reanalyse it. If neither
fires, the candidate is accepted cleanly.
"""
function find_growth_rates(scan::ScanResult, tauk::Real;
                           re_target::Real=0.0, im_target::Real=0.0,
                           pole_threshold::Real=10.0,
                           filter_above_poles::Bool=true,
                           filter_outside_re::Bool=true,
                           gap_kHz_threshold::Real=1.0,
                           angle_threshold_deg::Real=45.0,
                           density_radius_Q::Real=0.5,
                           min_neighbors::Integer=2)
    return _extract_growth_rates(scan.re_axis, scan.im_axis, scan.Δ,
                                  Float64(tauk);
                                  re_target=Float64(re_target),
                                  im_target=Float64(im_target),
                                  pole_threshold=Float64(pole_threshold),
                                  filter_above_poles=filter_above_poles,
                                  filter_outside_re=filter_outside_re,
                                  gap_kHz_threshold=Float64(gap_kHz_threshold),
                                  angle_threshold_deg=Float64(angle_threshold_deg),
                                  density_radius_Q=Float64(density_radius_Q),
                                  min_neighbors=Int(min_neighbors))
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
                           angle_threshold_deg::Real=45.0,
                           density_radius_Q::Real=0.5,
                           min_neighbors::Integer=2)
    return _extract_growth_rates_amr(amr.Q, amr.Δ, Float64(tauk);
                                      re_target=Float64(re_target),
                                      im_target=Float64(im_target),
                                      pole_threshold=Float64(pole_threshold),
                                      filter_above_poles=filter_above_poles,
                                      filter_outside_re=filter_outside_re,
                                      gap_kHz_threshold=Float64(gap_kHz_threshold),
                                      angle_threshold_deg=Float64(angle_threshold_deg),
                                      density_radius_Q=Float64(density_radius_Q),
                                      min_neighbors=Int(min_neighbors))
end

# ---------------------------------------------------------------------
# Implementation.
# ---------------------------------------------------------------------

# Bilinear interpolation of `values` on the regular grid `(re_axis, im_axis)`
# at point (qr, qi). Out-of-grid points are clamped to the boundary.
function _bilinear(re_axis::Vector{Float64}, im_axis::Vector{Float64},
                   values::Matrix{Float64}, qr::Real, qi::Real)
    nre = length(re_axis); nim = length(im_axis)
    i = clamp(searchsortedlast(re_axis, qr), 1, nre - 1)
    j = clamp(searchsortedlast(im_axis, qi), 1, nim - 1)
    tx = (qr - re_axis[i]) / (re_axis[i+1] - re_axis[i])
    ty = (qi - im_axis[j]) / (im_axis[j+1] - im_axis[j])
    tx = clamp(tx, 0.0, 1.0); ty = clamp(ty, 0.0, 1.0)
    return (1-tx)*(1-ty)*values[i,j]   + tx*(1-ty)*values[i+1,j] +
           (1-tx)*ty    *values[i,j+1] + tx*ty    *values[i+1,j+1]
end

# Extract polylines for a single contour level on a regular grid.
# Returns Vector{Vector{ComplexF64}} (one polyline per closed/open curve).
function _extract_contours(re_axis::Vector{Float64}, im_axis::Vector{Float64},
                            values::Matrix{Float64}, level::Float64)
    polylines = Vector{Vector{ComplexF64}}()
    for cl in lines(contour(re_axis, im_axis, values, level))
        xs, ys = coordinates(cl)
        path = ComplexF64[xs[i] + ys[i]*im for i in eachindex(xs)]
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
    best_i = 0; best_d = Inf
    for i in eachindex(path)
        d = abs(path[i] - pt)
        if d < best_d
            best_d = d; best_i = i
        end
    end
    return best_i, best_d
end

# Find the polyline (and vertex within it) whose vertex is closest to pt.
function _closest_polyline_vertex(paths::Vector{Vector{ComplexF64}},
                                    pt::ComplexF64)
    best_path_idx = 0; best_vert_idx = 0; best_d = Inf
    for (pi_, path) in enumerate(paths)
        vi, d = _closest_vertex(path, pt)
        if d < best_d
            best_d = d; best_path_idx = pi_; best_vert_idx = vi
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
                            re_paths::Vector{Vector{ComplexF64}},
                            ::Vector{Vector{ComplexF64}},   # im_paths unused
                            ::Float64;                       # angle_threshold_deg unused
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
    @inbounds for k in (re_v_idx + 1):n_path
        if abs(re_path[k] - pt) < max_walk
            push!(collected_idx, k)
        else
            break
        end
    end
    @inbounds for k in (re_v_idx - 1):-1:1
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
    sx  = 0.0; sx2 = 0.0; sx3 = 0.0; sx4 = 0.0
    sy  = 0.0; sxy = 0.0; sx2y = 0.0
    @inbounds for i in 1:n
        ω = ωs[i]; γ = γs[i]
        ω2 = ω * ω
        sx  += ω;       sx2 += ω2
        sx3 += ω2 * ω;  sx4 += ω2 * ω2
        sy  += γ;       sxy += ω * γ
        sx2y += ω2 * γ
    end
    M   = [Float64(n)  sx  sx2;
                 sx  sx2  sx3;
                sx2  sx3  sx4]
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
    γ_next = imag(sorted_roots[idx + 1]) / tauk * 1e-3
    return (γ_idx - γ_next) > gap_kHz_threshold
end

# Local-density check: spurious high-γ outliers are typically isolated in the
# Q plane, while legitimate (coupled) tearing roots cluster densely in the
# resonant region. Counts other valid roots within `density_radius_Q` of the
# candidate; flags when the count is below `min_neighbors`. Distance is in
# normalized Q-units (so the threshold is case-independent up to the natural
# Q-plane scale of the residual).
#
# Disabled for cases with very few total roots (n_roots < `min_total_for_density`,
# default 5): without a meaningful cluster baseline, "isolation" carries no
# signal — uncoupled cases (n_roots = 1-3) would otherwise spuriously fire on
# every candidate.
function _is_density_isolated(sorted_roots::Vector{ComplexF64}, idx::Int,
                              density_radius_Q::Float64, min_neighbors::Int;
                              min_total_for_density::Int=5)
    n_total = length(sorted_roots)
    n_total < min_total_for_density && return false
    n_neighbors = 0
    pt = sorted_roots[idx]
    @inbounds for k in eachindex(sorted_roots)
        k == idx && continue
        if abs(sorted_roots[k] - pt) < density_radius_Q
            n_neighbors += 1
        end
    end
    return n_neighbors < min_neighbors
end

function _run_analysis(re_paths::Vector{Vector{ComplexF64}},
                        im_paths::Vector{Vector{ComplexF64}},
                        im_re_vals::Vector{Vector{Float64}},
                        tauk::Float64;
                        pole_threshold::Float64,
                        filter_above_poles::Bool,
                        filter_outside_re::Bool,
                        gap_kHz_threshold::Float64=1.0,
                        angle_threshold_deg::Float64=45.0,
                        density_radius_Q::Float64=0.5,
                        min_neighbors::Int=2)
    raw_intersections = _all_intersections(re_paths, im_paths)

    poles      = ComplexF64[]
    candidates = Tuple{ComplexF64,Bool}[]    # (pt, on_top_half_re_flag)

    for pt in raw_intersections
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

        # --- 2. "+γ step inside Re contour" flag for spurious-upper-branch filter
        on_top_half_re = false
        best_re_path_idx, _, _ = _closest_polyline_vertex(re_paths, pt)
        if best_im_path_idx > 0 && best_re_path_idx > 0
            re_path = re_paths[best_re_path_idx]
            xs = real.(re_path); ys = imag.(re_path)
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
                    inside  = _point_in_polygon(step_pt, re_path)
                    on_top_half_re = !inside
                end
            end
        end

        push!(candidates, (pt, on_top_half_re))
    end

    # --- 3. pole + closed-loop filter (legacy), then geom + gap recursion (new)
    valid_roots    = ComplexF64[c[1] for c in candidates]
    filtered_roots = ComplexF64[]
    Q_root         = ComplexF64(NaN, NaN)
    Q_root_2nd     = ComplexF64(NaN, NaN)
    warning_flags  = Symbol[]

    if !isempty(valid_roots)
        order = sortperm(valid_roots; by=q -> -imag(q))
        sorted_pts = valid_roots[order]
        sorted_top = Bool[c[2] for c in candidates][order]

        max_pole_gamma = isempty(poles) ? -Inf : maximum(imag, poles)

        chosen_idx = 0
        for k in 1:length(sorted_pts)
            cand   = sorted_pts[k]
            top_re = sorted_top[k]
            # Legacy filter: above-pole + closed-loop outside-Re
            legacy_reject = filter_above_poles && imag(cand) > max_pole_gamma &&
                            (!filter_outside_re || top_re)
            if legacy_reject
                push!(filtered_roots, cand)
                continue
            end
            # New checks: 3 spurious-root flags (any 2+ → discard, 1 → warn)
            #   :geom    — Re=0 contour is locally a downward-concave "hill"
            #              at the candidate (clean polyline-following fit)
            #   :gap     — candidate is unstable AND >1 kHz above next root
            #              (an isolated γ peak — spurious outlier signature)
            #   :density — fewer than `min_neighbors` other roots within
            #              `density_radius_Q` of the candidate. Spurious
            #              high-kHz outliers tend to be isolated in Q-space;
            #              legitimate coupled-tearing roots cluster.
            geom_flag    = _is_geom_spurious(cand, re_paths, im_paths,
                                              angle_threshold_deg)
            gap_flag     = _is_gap_spurious(sorted_pts, k, tauk,
                                             gap_kHz_threshold)
            density_flag = _is_density_isolated(sorted_pts, k,
                                                 density_radius_Q, min_neighbors)
            n_flags = (geom_flag ? 1 : 0) + (gap_flag ? 1 : 0) +
                      (density_flag ? 1 : 0)
            if n_flags >= 2
                # 2+ of {geom, gap, density} → discard, recurse to next
                push!(filtered_roots, cand)
                continue
            end
            # Accept candidate as primary; record any single-flag warning.
            chosen_idx = k
            geom_flag    && push!(warning_flags, :geom)
            gap_flag     && push!(warning_flags, :gap)
            density_flag && push!(warning_flags, :density)
            break
        end

        if chosen_idx > 0
            Q_root = sorted_pts[chosen_idx]
            # When a warning fired, expose the next-down root as secondary so
            # downstream tools can plot/reanalyse. (Indices > chosen_idx in
            # sorted_pts are the next-most-unstable.)
            if !isempty(warning_flags) && chosen_idx < length(sorted_pts)
                Q_root_2nd = sorted_pts[chosen_idx + 1]
            end
        end
    end

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
                                angle_threshold_deg::Float64=45.0,
                                density_radius_Q::Float64=0.5,
                                min_neighbors::Int=2)
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
                          angle_threshold_deg=angle_threshold_deg,
                          density_radius_Q=density_radius_Q,
                          min_neighbors=min_neighbors)
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
    a1 = f1 >= L; a2 = f2 >= L; a3 = f3 >= L
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
    paths    = Vector{Vector{ComplexF64}}()
    aux_vals = Vector{Vector{Float64}}()

    # Walk a polyline starting from segment `start_seg` via endpoint
    # `start_pt`; returns the path and aux values.
    function _walk(start_seg::Int, start_pt::ComplexF64)
        path = ComplexF64[start_pt]
        aux  = Float64[]
        # Emit the aux value for start_pt on the first segment
        s0   = segs[start_seg]
        push!(aux, start_pt == s0.p1 ? s0.a1 : s0.a2)

        cur_seg = start_seg; cur_pt = start_pt
        while true
            used[cur_seg] = true
            s = segs[cur_seg]
            next_pt   = cur_pt == s.p1 ? s.p2 : s.p1
            next_aux  = cur_pt == s.p1 ? s.a2 : s.a1
            push!(path, next_pt)
            push!(aux, next_aux)

            nbrs = adj[next_pt]
            nxt  = 0
            for j in nbrs
                if !used[j] && j != cur_seg
                    nxt = j; break
                end
            end
            nxt == 0 && break
            cur_seg = nxt; cur_pt = next_pt
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
                                     angle_threshold_deg::Float64=45.0,
                                     density_radius_Q::Float64=0.5,
                                     min_neighbors::Int=2)
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
        p1 = Q[i1]; p2 = Q[i2]; p3 = Q[i3]
        v1 = Δ[i1]; v2 = Δ[i2]; v3 = Δ[i3]
        re_seg, im_seg = _march_triangle(p1, p2, p3, v1, v2, v3,
                                          re_target, im_target)
        re_seg !== nothing && push!(re_segs, re_seg)
        im_seg !== nothing && push!(im_segs, im_seg)
    end

    re_paths, _          = _chain_segments(re_segs)
    im_paths, im_re_vals = _chain_segments(im_segs)

    return _run_analysis(re_paths, im_paths, im_re_vals, tauk;
                          pole_threshold=pole_threshold,
                          filter_above_poles=filter_above_poles,
                          filter_outside_re=filter_outside_re,
                          gap_kHz_threshold=gap_kHz_threshold,
                          angle_threshold_deg=angle_threshold_deg,
                          density_radius_Q=density_radius_Q,
                          min_neighbors=min_neighbors)
end
