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
    adaptive_sample(f, xs0; max_points=21, rtol=0.05, min_width=0.0, crossing_width=nothing) -> (x, y)

Evaluate the vector-valued function `f(x) -> AbstractVector{<:Real}` on the initial grid `xs0`
(sorted, every point evaluated; keep `0` in it when the unshifted point must be reproduced
exactly), then repeatedly bisect the interval whose midpoint is predicted worst until every
interval is resolved or `max_points` evaluations have been spent. The prediction error of an
interval is the distance between the linear and the piecewise-cubic (Catmull-Rom) estimates of
its midpoint, scaled by the interval width relative to the mean width; an interval across which
any component of `f` changes sign is refined first, until it is narrower than `crossing_width`
(default: an eighth of the initial spacing), so zero crossings end up bracketed tightly.
Resolution means every interval's prediction error is below `rtol` times the range of each
component (or is narrower than `min_width`). Deterministic for given inputs.

Returns the sorted sample points `x` and the matrix `y` (`length(x) × ncomponents`).
"""
function adaptive_sample(f, xs0::AbstractVector{<:Real}; max_points::Int=21, rtol::Real=0.05, min_width::Real=0.0, crossing_width=nothing)
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
        best, best_score = 0, 0.0
        for i in 1:(n-1)
            w = x[i+1] - x[i]
            w > min_width || continue
            # Sign change: refine first, largest interval first, until the crossing is bracketed within cross_w.
            crosses = w > cross_w && any(ys[i][c] * ys[i+1][c] < 0 for c in 1:ncomp)
            score = crosses ? Inf : 0.0
            if !crosses
                # Catmull-Rom midpoint estimate against the linear one, per component.
                y0 = i > 1 ? ys[i-1] : ys[i]
                y3 = i < n - 1 ? ys[i+2] : ys[i+1]
                for c in 1:ncomp
                    lin = 0.5 * (ys[i][c] + ys[i+1][c])
                    cub = (-y0[c] + 9ys[i][c] + 9ys[i+1][c] - y3[c]) / 16
                    score = max(score, abs(cub - lin) / rng[c] * (w / mean_width))
                end
            end
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

end # module
