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

# ---------------------------------------------------------------------
# Public result struct + main entry point.
# ---------------------------------------------------------------------

"""
    GrowthRateResult

Output of `find_growth_rates`.

| field             | meaning                                                |
|-------------------|--------------------------------------------------------|
| `Q_root`          | Best (highest-γ surviving) root, normalized            |
| `omega_Hz`        | `Re(Q_root) / tauk` — physical rotation frequency      |
| `gamma_Hz`        | `Im(Q_root) / tauk` — physical growth rate             |
| `valid_roots`     | All non-pole intersections that survived the filters   |
| `poles`           | Intersections classified as poles                      |
| `filtered_roots`  | Intersections rejected by the above-pole/outside-Re   |
|                   | filter                                                 |
| `re_contours`     | Extracted Re(Δ)=`re_target` polylines                  |
| `im_contours`     | Extracted Im(Δ)=`im_target` polylines                  |
| `pole_threshold`  | Threshold used for pole classification                 |
"""
struct GrowthRateResult
    Q_root::ComplexF64
    omega_Hz::Float64
    gamma_Hz::Float64
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
                       filter_outside_re=true) -> GrowthRateResult

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
    survive (matches the Python default).
"""
function find_growth_rates(scan::ScanResult, tauk::Real;
                           re_target::Real=0.0, im_target::Real=0.0,
                           pole_threshold::Real=10.0,
                           filter_above_poles::Bool=true,
                           filter_outside_re::Bool=true)
    return _extract_growth_rates(scan.re_axis, scan.im_axis, scan.Δ,
                                  Float64(tauk);
                                  re_target=Float64(re_target),
                                  im_target=Float64(im_target),
                                  pole_threshold=Float64(pole_threshold),
                                  filter_above_poles=filter_above_poles,
                                  filter_outside_re=filter_outside_re)
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

# The actual analysis. Mirrors `analyze_amr_data` + `find_growthrates` from
# find_growthrates.py, restricted to the regular-grid input case.
function _extract_growth_rates(re_axis::Vector{Float64},
                                im_axis::Vector{Float64},
                                Δ_grid::Matrix{ComplexF64},
                                tauk::Float64;
                                re_target::Float64,
                                im_target::Float64,
                                pole_threshold::Float64,
                                filter_above_poles::Bool,
                                filter_outside_re::Bool)
    re_field = real.(Δ_grid)
    im_field = imag.(Δ_grid)

    re_paths = _extract_contours(re_axis, im_axis, re_field, re_target)
    im_paths = _extract_contours(re_axis, im_axis, im_field, im_target)

    raw_intersections = _all_intersections(re_paths, im_paths)

    # Pre-compute Re(Δ) values along each Im=0 contour vertex via bilinear
    # interpolation from the grid.
    im_re_vals = [Float64[_bilinear(re_axis, im_axis, re_field,
                                     real(v), imag(v))
                          for v in path]
                  for path in im_paths]

    poles = ComplexF64[]
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

        # --- 2. determine the "+γ step inside Re contour" flag for the
        # spurious-upper-branch filter.
        on_top_half_re = false
        best_re_path_idx, _, _ = _closest_polyline_vertex(re_paths, pt)
        if best_im_path_idx > 0 && best_re_path_idx > 0
            re_path = re_paths[best_re_path_idx]
            xs = real.(re_path); ys = imag.(re_path)
            contour_extent = max(maximum(xs) - minimum(xs),
                                  maximum(ys) - minimum(ys))
            closure_gap = abs(re_path[1] - re_path[end])

            if contour_extent > 0 && closure_gap < 0.1 * contour_extent
                # Re=0 contour is approximately closed → containment test
                # makes sense.
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

    # --- 3. apply pole / outside-Re filtering and pick highest-γ root
    valid_roots    = ComplexF64[c[1] for c in candidates]
    filtered_roots = ComplexF64[]
    Q_root         = ComplexF64(NaN, NaN)

    if !isempty(valid_roots)
        # Sort candidates by descending γ
        order = sortperm(valid_roots; by=q -> -imag(q))
        sorted_pts  = valid_roots[order]
        sorted_top  = Bool[c[2] for c in candidates][order]

        max_pole_gamma = isempty(poles) ? -Inf : maximum(imag, poles)

        chosen_idx = 0
        for k in 1:length(sorted_pts)
            cand   = sorted_pts[k]
            top_re = sorted_top[k]
            reject = filter_above_poles && imag(cand) > max_pole_gamma &&
                     (!filter_outside_re || top_re)
            if reject
                push!(filtered_roots, cand)
            else
                chosen_idx = k
                break
            end
        end

        if chosen_idx > 0
            Q_root = sorted_pts[chosen_idx]
        end
    end

    omega_Hz = isnan(real(Q_root)) ? 0.0 : real(Q_root) / tauk
    gamma_Hz = isnan(imag(Q_root)) ? 0.0 : imag(Q_root) / tauk

    return GrowthRateResult(Q_root, omega_Hz, gamma_Hz,
                             valid_roots, poles, filtered_roots,
                             re_paths, im_paths, pole_threshold)
end
