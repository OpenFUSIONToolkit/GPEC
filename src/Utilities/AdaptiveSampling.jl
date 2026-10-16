"""
    AdaptiveSampling

Sample an expensive function on a one-dimensional grid that refines itself where the function
changes fastest, so that sharp features (a resonance, a zero crossing) are resolved without
paying for a fine uniform grid. Intended for scans whose every point is a full physics
evaluation, e.g. a torque against a rotation shift.
"""
module AdaptiveSampling

export adaptive_sample

"""
    adaptive_sample(f, xs0; max_points=21, rtol=0.05, min_width=0.0, crossing_width=nothing, feature_ratio=10.0) -> (x, y)

Evaluate the vector-valued function `f(x) -> AbstractVector{<:Real}` on the initial grid `xs0`
(sorted, every point evaluated; keep `0` in it when the unshifted point must be reproduced
exactly), then repeatedly bisect the interval whose midpoint is predicted worst until every
interval is resolved or `max_points` evaluations have been spent. The prediction error of an
interval is the distance between the linear and the piecewise-cubic (Catmull-Rom) estimates of
its midpoint, scaled by the interval width relative to the mean width; an interval across which
any component of `f` changes sign is refined first, until it is narrower than `crossing_width`
(default: an eighth of the initial spacing), so zero crossings end up bracketed tightly.
Resolution means every interval's prediction error is below `rtol` times the range of each
component (or is narrower than `min_width`). A second, scale-free trigger catches a narrow
feature riding on a large smooth background, whose tails are far below `rtol` of the range:
an interval whose prediction error exceeds `feature_ratio` times the median error of all
intervals is refined too. Deterministic for given inputs. A feature that leaves no trace at all
at the initial spacing cannot be found by any refinement rule; the initial grid sets that
guarantee.

Returns the sorted sample points `x` and the matrix `y` (`length(x) × ncomponents`).
"""
function adaptive_sample(f, xs0::AbstractVector{<:Real}; max_points::Int=21, rtol::Real=0.05, min_width::Real=0.0, crossing_width=nothing,
    feature_ratio::Real=10.0)
    x = sort(Float64.(collect(xs0)))
    length(x) >= 2 || throw(ArgumentError("adaptive_sample needs at least two initial points"))
    allunique(x) || throw(ArgumentError("adaptive_sample: initial points must be distinct"))
    ys = [Float64.(f(xi)) for xi in x]
    ncomp = length(ys[1])
    cross_w = crossing_width === nothing ? (x[end] - x[1]) / (length(x) - 1) / 8 : Float64(crossing_width)
    while length(x) < max_points
        n = length(x)
        rng = [max(maximum(y[c] for y in ys) - minimum(y[c] for y in ys), eps()) for c in 1:ncomp]
        mean_width = (x[end] - x[1]) / (n - 1)
        # Per-interval prediction error: Catmull-Rom midpoint estimate against the linear one, the
        # worst component relative to its range, weighted by the interval width.
        errs = zeros(n - 1)
        for i in 1:(n-1)
            # Missing neighbours at the ends are extrapolated linearly, so a straight line scores zero there.
            y0 = i > 1 ? ys[i-1] : 2 .* ys[i] .- ys[i+1]
            y3 = i < n - 1 ? ys[i+2] : 2 .* ys[i+1] .- ys[i]
            for c in 1:ncomp
                lin = 0.5 * (ys[i][c] + ys[i+1][c])
                cub = (-y0[c] + 9ys[i][c] + 9ys[i+1][c] - y3[c]) / 16
                errs[i] = max(errs[i], abs(cub - lin) / rng[c] * ((x[i+1] - x[i]) / mean_width))
            end
        end
        med = _median(errs)
        best, best_score = 0, 0.0
        for i in 1:(n-1)
            w = x[i+1] - x[i]
            w > min_width || continue
            # Sign change: refine first, largest interval first, until the crossing is bracketed within cross_w.
            crosses = w > cross_w && any(ys[i][c] * ys[i+1][c] < 0 for c in 1:ncomp)
            # A feature on a smooth background: an error far above the typical one, even if below rtol.
            feature = errs[i] > 1e-9 && errs[i] > feature_ratio * med
            score = crosses ? Inf : (feature ? max(errs[i], rtol) : errs[i])
            if score > best_score || (crosses && !isfinite(best_score) && w > x[best+1] - x[best])
                best, best_score = i, score
            end
        end
        (best == 0 || best_score < rtol) && break
        xm = 0.5 * (x[best] + x[best+1])
        insert!(x, best + 1, xm)
        insert!(ys, best + 1, Float64.(f(xm)))
    end
    y = reduce(vcat, permutedims(v) for v in ys)
    return x, y
end

_median(v) = (s = sort(v);
n = length(s);
isodd(n) ? s[(n+1)÷2] : 0.5 * (s[n÷2] + s[n÷2+1]))

end # module
