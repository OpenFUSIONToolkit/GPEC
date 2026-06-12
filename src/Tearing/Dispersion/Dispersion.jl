# Dispersion.jl
#
# Tearing-dispersion-relation solver shared between GGJ and SLAYER inner-layer
# models. Combines the outer-region Δ' from `PerturbedEquilibrium.SingularCoupling`
# with the inner-layer Δ(Q) from any `InnerLayerModel` to find growth-rate
# eigenvalues.
#
# Operating modes (incremental as PRs land):
#   - `SurfaceCoupling`     (this module, PR 3) -- per-surface residual r(Q)
#   - `dispersion_det`      (Coupled.jl, PR 4)  -- multi-surface determinant
#   - `brute_force_scan`    (PR 5)              -- regular 2D Q-plane scan
#   - `find_growth_rates`   (PR 5)              -- contour-intersection root
#                                                  extraction (Re=0 ∩ Im=0)
#   - `amr_scan`            (PR 6)              -- adaptive Q-plane refinement
#
# Roots are ISOLATED by 2D contour intersection on Nyquist-style Q-plane scans
# (`find_growth_rates`, Re=0 ∩ Im=0), then optionally POLISHED to the true zero
# of the residual by a bounded, neighbour-aware local solve (pass the residual
# callable to `find_growth_rates`). Polishing makes the extracted root
# resolution- and thread-independent; without it the reported root is the
# marching-squares estimate, sensitive to grid/rounding. This module provides
# both the residual building blocks and the scan/extraction machinery.
#
# The per-surface residual at one rational surface is
#
#   r(Q) = Δ'_diag - scale · Δ_inner(Q) - Δ_crit
#
# where `scale` is the inner→outer-units conversion factor (S^(1/3) for SLAYER,
# 1 for GGJ since `rescale_delta` is applied internally) and `Δ_crit` is the
# `dc_tmp` chi-parallel offset (zero by default).

module Dispersion

using LinearAlgebra
using StaticArrays

using ..InnerLayer
using ..InnerLayer: InnerLayerModel, solve_inner, GGJModel, GGJParameters,
                    SLAYERModel, SLAYERParameters

include("SurfaceCoupling.jl")
include("Coupled.jl")
include("CoupledFortranMatch.jl")
include("BruteForceScan.jl")
include("ContourSearchAMR.jl")
include("GrowthRateExtraction.jl")

export SurfaceCoupling, surface_coupling
export MultiSurfaceCoupling, multi_surface_coupling
export MultiSurfaceCouplingFortran, multi_surface_coupling_fortran
export ScanResult, brute_force_scan
export AMRCell, AMRResult, amr_scan
export BoxActivity, MultiBoxAMRResult, multi_box_amr_scan, as_amr_result
export GrowthRateResult, find_growth_rates

end # module Dispersion
