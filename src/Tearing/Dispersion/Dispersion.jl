# Tearing dispersion relation shared by the GGJ and SLAYER inner-layer models: combines the
# outer-region Δ' with the inner-layer Δ from any `InnerLayerModel` and finds growth-rate roots.
# Roots are isolated by Re=0 ∩ Im=0 contour intersection on Q-plane scans (`find_growth_rates`),
# then optionally polished to the true zero of the residual.
#
# Per-surface residual: r(Q) = Δ'_diag - scale · Δ_inner(Q) - Δ_crit, with `scale` = S^(1/3) for
# SLAYER and 1 for GGJ, and Δ_crit the `dc` offset (zero by default).

module Dispersion

using LinearAlgebra
using StaticArrays

using ..InnerLayer
using ..InnerLayer: InnerLayerModel, layer_response, GGJModel, GGJParameters,
    SLAYERModel, SLAYERParameters

include("SurfaceCoupling.jl")
include("Coupled.jl")
include("CoupledFullMatch.jl")
include("BruteForceScan.jl")
include("ContourSearchAMR.jl")
include("GrowthRateExtraction.jl")
include("GGJGrowthRates.jl")

export SurfaceCoupling, surface_coupling
export MultiSurfaceCoupling, multi_surface_coupling
export MultiSurfaceCouplingFull, multi_surface_coupling_full
export ScanResult, brute_force_scan
export AMRCell, AMRResult, amr_scan
export BoxActivity, MultiBoxAMRResult, multi_box_amr_scan, as_amr_result
export GrowthRateResult, find_growth_rates
export GGJDispersion, ggj_growth_rates

end # module Dispersion
