# Is the EL integrand smooth at the knot scale, or has it hit a node-noise floor?
#
# Second divided differences of the splined EL coefficient matrices (matrices/ideal/F,K,G, i.e.
# ffit.fmats_lower/kmats/gmats evaluated at their own knots) approximate f''. For a smooth
# sampled function they vary smoothly knot to knot (lag-1 autocorrelation r1 → +1) and converge
# as the grid refines. For node values sitting on a noise floor they alternate in sign
# (r1 → −2/3, the white-noise value) and their magnitude grows as Δ⁻².
#
# Usage: julia --project=. roughness.jl <rundir> [<rundir> ...]
using HDF5, Printf, Statistics

# f[x_{i-1}, x_i, x_{i+1}] ≈ f''/2 on a non-uniform grid.
function second_divided_difference(x, f)
    n = length(x)
    d = similar(f, n - 2)
    for i in 2:n-1
        left = (f[i] - f[i-1]) / (x[i] - x[i-1])
        right = (f[i+1] - f[i]) / (x[i+1] - x[i])
        d[i-1] = (right - left) / (x[i+1] - x[i-1])
    end
    return d
end

# Lag-1 autocorrelation: +1 for a smooth sequence, −2/3 for the 2nd difference of white noise.
function lag1(d)
    dc = d .- mean(d)
    return sum(dc[1:end-1] .* dc[2:end]) / sum(abs2, dc)
end

# Restrict to a psi window so the axis and edge (where the grid packs) can be looked at separately.
window(x, lo, hi) = findall(p -> lo <= p <= hi, x)

"""
Roughness statistics for one (npsi, np, np) matrix stack over a psi window: the lag-1
autocorrelation of f'' pooled over the largest matrix elements, and the median |f''|.
"""
function roughness(psi, mats, idx; nelem=20)
    x = psi[idx]
    # Rank elements by magnitude so the statistic is not dominated by numerically empty entries.
    amp = dropdims(sum(abs, view(mats, idx, :, :); dims=1); dims=1)
    order = sortperm(vec(amp); rev=true)[1:min(nelem, length(amp))]
    r1s = Float64[]
    curv = Float64[]
    for lin in order
        i, j = Tuple(CartesianIndices(amp)[lin])
        f = view(mats, idx, i, j)
        d = second_divided_difference(x, collect(f))
        scale = maximum(abs, f)
        scale == 0 && continue
        push!(r1s, lag1(real.(d)))
        push!(curv, median(abs.(d)) / scale)   # |f''|/|f|, units of psi^-2
    end
    return median(r1s), median(curv)
end

"""
Node-error amplitude: RMS residual of a local degree-4 least-squares fit over a sliding window of
`w` knots, relative to |f|. Smooth data leaves a residual that falls like Δ^5 (≈32x per mpsi
doubling); data sitting on a noise floor leaves a residual that is constant in Δ.
"""
function fit_residual(x, f; w=9, deg=4)
    n = length(x)
    n < w + 2 && return NaN
    res = Float64[]
    half = w ÷ 2
    for c in (half+1):(n-half)
        idx = (c-half):(c+half)
        # Center and scale so the Vandermonde system stays conditioned on a graded grid.
        t = (x[idx] .- x[c]) ./ (x[c+half] - x[c-half])
        V = reduce(hcat, [t .^ p for p in 0:deg])
        push!(res, abs(f[c] - (V * (V \ collect(f[idx])))[half+1]))
    end
    return sqrt(mean(abs2, res)) / maximum(abs, f)
end

function noise_floor(psi, mats, idx; nelem=20)
    x = psi[idx]
    amp = dropdims(sum(abs, view(mats, idx, :, :); dims=1); dims=1)
    order = sortperm(vec(amp); rev=true)[1:min(nelem, length(amp))]
    vals = Float64[]
    for lin in order
        i, j = Tuple(CartesianIndices(amp)[lin])
        r = fit_residual(x, collect(view(mats, idx, i, j)))
        isnan(r) || push!(vals, r)
    end
    return isempty(vals) ? NaN : median(vals)
end

const REGIONS = [("psi<0.1 (axis pack)", 0.0, 0.1), ("0.3-0.7 (mid)", 0.3, 0.7), ("psi>0.95 (edge pack)", 0.95, 1.0)]

for rundir in ARGS
    h5open(joinpath(rundir, "gpec.h5"), "r") do h5
        psi = read(h5["matrices/psi"])
        qprof = read(h5["splines/profiles/q"])
        @printf("\n%s  (npsi = %d)\n", basename(rundir), length(psi))
        @printf("  %-22s | %-6s | %-30s | %-30s | %-30s\n", "region", "nknot", "F: r1 / |f''|/|f| / resid", "K: r1 / |f''|/|f| / resid", "G: r1 / |f''|/|f| / resid")
        for (name, lo, hi) in REGIONS
            idx = window(psi, lo, hi)
            length(idx) < 8 && continue
            cols = String[]
            for key in ("F", "K", "G")
                mats = read(h5["matrices/ideal"][key])
                r1, curv = roughness(psi, mats, idx)
                push!(cols, @sprintf("%+.3f /%9.2e /%9.2e", r1, curv, noise_floor(psi, mats, idx)))
            end
            # q(psi) is a genuinely smooth control on the same grid.
            dq = second_divided_difference(psi[idx], qprof[idx])
            @printf("  %-22s | %6d | %-30s | %-30s | %-30s | q ctrl r1 %+.3f\n", name, length(idx), cols[1], cols[2], cols[3], lag1(dq))
        end
    end
end
