# GGJGrowthRates.jl
#
# Tearing growth rates from the GGJ inner layer. The dispersion function is the
# full Pletzer-Dewar 4m × 4m determinant (CoupledFullMatch.jl), which keeps the
# interchange channel and with it the Glasser stabilization. It is scanned in the
# same normalized Q plane as the SLAYER residual and its roots are classified
# by winding number (`find_growth_rates(...; classification=:winding)`).
#
# Plane convention. The determinant takes the complex growth rate
# s = γ + iω [s⁻¹]. The SLAYER scans use Q = τ·(−ω + iγ), so s = −i·Q/τ; with
# τ = 1/Q₀ of the reference surface (Q₀ = S^(−1/3)/τ_A, GWP2016 Eq. A16) the
# scanned variable is the GGJ scaled growth rate rotated by a quarter turn,
# Im Q = γ/Q₀ and Re Q = −ω/Q₀, and `GrowthRateResult.gamma_Hz` is γ in s⁻¹.
# s ↦ Q is holomorphic, so a zero of the determinant winds +1 (the SLAYER
# residual, a function of conj(Q), winds −1).

"""
    GGJDispersion(mc::MultiSurfaceCouplingFull)

The GGJ full-matching determinant as a function of the normalized frequency
`Q = (−ω + iγ)/Q₀` used by the SLAYER scans, `g(Q) = mc(−i·Q·Q₀)`, where `Q₀`
is the growth-rate scale `S^(−1/3)/τ_A` of the reference surface
`mc.surfaces[mc.ref_idx]`. Every surface must use a `GGJModel`.
"""
struct GGJDispersion{M<:MultiSurfaceCouplingFull}
    mc::M
    Q0::Float64
end

function GGJDispersion(mc::MultiSurfaceCouplingFull)
    all(sc -> sc.model isa GGJModel, mc.surfaces) ||
        throw(ArgumentError("GGJDispersion: every surface must use a GGJModel"))
    return GGJDispersion(mc, InnerLayer.GGJ.q0(mc.surfaces[mc.ref_idx].params))
end

(g::GGJDispersion)(Q::Number) = g.mc(-im * ComplexF64(Q) * g.Q0)

"""
    ggj_growth_rates(mc::MultiSurfaceCouplingFull;
                     gamma_range, omega_range,
                     nre0=40, nim0=40, passes=4, max_cells=2_000_000,
                     polish=true, winding_rfrac=0.5,
                     parallel=Threads.nthreads() > 1) -> (GrowthRateResult, AMRResult)

Tearing growth rates of the GGJ inner layer matched through the full 4m × 4m
determinant `mc` (coupled for several surfaces, uncoupled for one surface with
its own 2 × 2 block of the raw Δ'). The box `gamma_range` × `omega_range`
(physical: γ in s⁻¹, ω in rad/s) is scanned adaptively in the plane of
[`GGJDispersion`](@ref) and every contour intersection is classified by its
winding number; see `find_growth_rates`. The result is in the units of the
SLAYER extraction: `gamma_Hz` is γ [s⁻¹], `omega_Hz` is −ω [rad/s], and
`Q_root` is the normalized `(−ω + iγ)/Q₀`. The reported root is the least
stable zero in the box. Without drifts the GGJ roots come in ±ω pairs, so
either member of a pair can be reported.

`nre0` × `nim0` is the initial AMR grid and `passes` the number of refinement
passes. Each determinant evaluation solves the inner layer at every surface,
so cost scales with the number of AMR points plus a few hundred evaluations per
intersection for the classification.
"""
function ggj_growth_rates(mc::MultiSurfaceCouplingFull;
    gamma_range::NTuple{2,<:Real}, omega_range::NTuple{2,<:Real},
    nre0::Integer=40, nim0::Integer=40, passes::Integer=4,
    max_cells::Integer=2_000_000,
    polish::Bool=true, winding_rfrac::Real=0.5,
    parallel::Bool=Threads.nthreads() > 1)
    g = GGJDispersion(mc)
    τ = 1 / g.Q0
    re_range = (-Float64(omega_range[2]) * τ, -Float64(omega_range[1]) * τ)
    im_range = (Float64(gamma_range[1]) * τ, Float64(gamma_range[2]) * τ)
    amr = amr_scan(g, re_range, im_range; nre0=nre0, nim0=nim0, passes=passes,
        max_cells=max_cells, parallel=parallel)
    gr = find_growth_rates(amr, τ; residual=g, polish=polish,
        classification=:winding, zero_winding=1, winding_rfrac=winding_rfrac)
    return gr, amr
end
