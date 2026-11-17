# Flattening.jl
#
# The pressure gradient vanishes inside each island and is unchanged outside it; the
# pressure keeps its boundary value at the last profile point and is integrated inward,
# as in the Fortran GPEC saturated-island profile. Islands that overlap flatten their
# union once. With the flattened intervals [a_k, b_k] merged, the integral is exact:
#
#   p_mod(ψ) = p(ψ) + Σ_{k: b_k > ψ} [p(b_k) − p(max(ψ, a_k))],
#
# so every point inward of an interval drops by p(a_k) − p(b_k), and points inside it sit
# at p(b_k). The stored energy removed is ΔW = (3/2) ∫ (p − p_mod) dV.

"""
    FlattenedIsland

One island of the set: mode numbers `m`, `n`, rational surface `psi` (ψ_N), full width `w`
(ψ_N) and the flattened interval `psi_in ≤ ψ ≤ psi_out`, clipped to the profile grid.
`clipped` is true when the island reaches past either end of the grid.
"""
struct FlattenedIsland
    m::Int
    n::Int
    psi::Float64
    w::Float64
    psi_in::Float64
    psi_out::Float64
    clipped::Bool
end

"""
    IslandFlatteningResult

Static flattening of the equilibrium pressure across `islands`.

  - `widths`: the requested full widths keyed by `"m/n"` (ψ_N); every rational surface of each
    `m/n` gets the width
  - `islands`: one `FlattenedIsland` per rational surface found
  - `intervals`: the flattened intervals, merged where islands overlap, inward to outward
  - `delta_p`: pressure drop across each merged interval, p(a_k) − p(b_k) (Pa); for an edge
    interval it is the pedestal-top pressure change
  - `psi`, `p`, `p_mod`: the profile grid with the interval ends added, the equilibrium and
    flattened pressures on it (Pa)
  - `W`, `delta_W`: stored energy (3/2)∫p dV and the part the islands remove (J)
  - `delta_p0`: on-axis pressure drop, the sum of `delta_p` (Pa)
  - `jbs_B`, `jbs_B_mod`, `jbs_fraction_lost`: the given bootstrap profile on `psi`, the same
    with the flattened intervals removed, and the fraction of its volume integral on them;
    empty and NaN when no profile was given. To leading order j_bs ∝ p′, so it vanishes where
    p′ does and is unchanged elsewhere. For ⟨j_bs·B⟩ the fraction is of ∫⟨j_bs·B⟩dV; for the
    toroidal bootstrap current pass j_φ⟨1/R⟩, since dA = ⟨1/R⟩dV/2π.
"""
struct IslandFlatteningResult
    widths::Dict{String,Float64}
    islands::Vector{FlattenedIsland}
    intervals::Vector{Tuple{Float64,Float64}}
    delta_p::Vector{Float64}
    psi::Vector{Float64}
    p::Vector{Float64}
    p_mod::Vector{Float64}
    W::Float64
    delta_W::Float64
    delta_p0::Float64
    jbs_B::Vector{Float64}
    jbs_B_mod::Vector{Float64}
    jbs_fraction_lost::Float64
end

"""
    merged_intervals(islands) -> Vector{Tuple{Float64,Float64}}

The union of the islands' flattened intervals as disjoint intervals, inward to outward.
Intervals that touch or overlap merge.
"""
function merged_intervals(islands::AbstractVector{FlattenedIsland})
    spans = sort!([(i.psi_in, i.psi_out) for i in islands if i.psi_out > i.psi_in])
    merged = Tuple{Float64,Float64}[]
    for (a, b) in spans
        if !isempty(merged) && a <= merged[end][2]
            merged[end] = (merged[end][1], max(merged[end][2], b))
        else
            push!(merged, (a, b))
        end
    end
    return merged
end

"""
    flattened_pressure(p, intervals, psi) -> Float64

The flattened pressure at `psi` for a pressure profile `p(ψ)` and disjoint flattened
`intervals`, anchored at the boundary value outside every interval.
"""
function flattened_pressure(p, intervals, psi::Real)
    pm = p(psi)
    for (a, b) in intervals
        b > psi && (pm += p(b) - p(max(psi, a)))
    end
    return pm
end

"""
    island_flattening(equil; widths, jbs_B=nothing) -> IslandFlatteningResult

Flatten the pressure of `equil` across an island of full width `widths["m/n"]` (ψ_N) at
every rational surface q = m/n (m, n > 0). `jbs_B`, a function of ψ_N giving a bootstrap
profile, adds the part the flattening removes. The pressure is anchored at the last point of
the equilibrium profile grid, so for an island reaching the edge the result depends on where
that grid ends.
"""
function island_flattening(equil::PlasmaEquilibrium; widths::AbstractDict, jbs_B=nothing)
    prof = equil.profiles
    lo, hi = prof.xs[1], prof.xs[end]
    p(psi) = prof.P_spline(psi) / mu0
    dV(psi) = prof.dVdpsi_spline(psi)

    requested = Dict{String,Float64}()
    islands = FlattenedIsland[]
    for (label, w) in widths
        m, n = _mode_numbers(label)
        w >= 0 || throw(ArgumentError("island width for $label must be non-negative, got $w"))
        requested[label] = w
        found = [s.psifac for s in _find_rational_surfaces(equil, n, n) if s.m == m]
        isempty(found) && @warn "No q = $label surface in the equilibrium; island skipped"
        for psi in found
            a, b = psi - w / 2, psi + w / 2
            push!(islands, FlattenedIsland(m, n, psi, w, max(a, lo), min(b, hi), a < lo || b > hi))
        end
    end
    sort!(islands; by=i -> i.psi)
    intervals = merged_intervals(islands)
    delta_p = [p(a) - p(b) for (a, b) in intervals]

    grid = sort!(unique!(vcat(prof.xs, (x for iv in intervals for x in iv)...)))
    p_eq = p.(grid)
    p_mod = [flattened_pressure(p, intervals, x) for x in grid]

    # Exact for the piecewise-cubic integrands: nodes on every knot and interval end.
    x, wq = gausslegendre(4)
    segment(f, a, b) = (b - a) / 2 * sum(wq[j] * f((b - a) / 2 * x[j] + (a + b) / 2) for j in eachindex(x))
    integrate(f) = sum(segment(f, grid[i], grid[i+1]) for i in 1:(length(grid)-1))
    W = 1.5 * integrate(psi -> p(psi) * dV(psi))
    delta_W = 1.5 * integrate(psi -> (p(psi) - flattened_pressure(p, intervals, psi)) * dV(psi))

    inside(psi) = any(a < psi < b for (a, b) in intervals)
    if jbs_B === nothing
        jbs, jbs_mod, lost = Float64[], Float64[], NaN
    else
        jbs = Float64.(jbs_B.(grid))
        jbs_mod = [inside(xi) ? 0.0 : j for (xi, j) in zip(grid, jbs)]
        total = integrate(psi -> jbs_B(psi) * dV(psi))
        lost = sum(integrate_between(psi -> jbs_B(psi) * dV(psi), grid, a, b, segment) for (a, b) in intervals; init=0.0) / total
    end
    return IslandFlatteningResult(requested, islands, intervals, delta_p, grid, p_eq, p_mod, W, delta_W, sum(delta_p; init=0.0),
        jbs, jbs_mod, lost)
end

# Integral of f over [a, b], whose ends are nodes of `grid`.
function integrate_between(f, grid, a, b, segment)
    i0, i1 = searchsortedfirst(grid, a), searchsortedfirst(grid, b)
    return sum(segment(f, grid[i], grid[i+1]) for i in i0:(i1-1); init=0.0)
end

function _mode_numbers(label::AbstractString)
    parts = split(label, "/")
    ok = length(parts) == 2 && all(s -> tryparse(Int, strip(s)) !== nothing, parts)
    ok || throw(ArgumentError("island widths are keyed by \"m/n\", e.g. \"2/1\"; got \"$label\""))
    m, n = parse.(Int, strip.(parts))
    (m > 0 && n > 0) || throw(ArgumentError("island \"$label\" needs positive m and n"))
    return m, n
end
