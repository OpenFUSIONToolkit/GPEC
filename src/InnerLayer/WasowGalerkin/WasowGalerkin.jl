# WasowGalerkin.jl
#
# Shared, model-agnostic engine for resistive/extended-MHD inner-layer matching:
# a singular-Galerkin Hermite finite-element method with a Wasow asymptotic
# large-x boundary construction. A physics model supplies a `SystemSpec`
# (sizes + coefficient/asymptotic/eigenbasis/exponent/parity callbacks); the
# engine contains no model-specific physics or dimensions.
#
# The single-fluid GGJ system (`InnerLayer.GGJ`) is the first client and the
# regression oracle. The engine is designed to also accept a higher-order
# toroidal two-fluid drift-MHD layer by supplying its own `SystemSpec`,
# including a full-domain (parity-coupled) solve.

module WasowGalerkin

using LinearAlgebra
using StaticArrays

include("ParityBC.jl")
include("SystemSpec.jl")
include("Asymptotics.jl")
include("Galerkin.jl")
include("FullDomain.jl")

export SystemSpec, ParityBC, BCKind, DIRICHLET, NEUMANN, nparities, n_exp
export WasowCache, build_wasow, evaluate_wasow, wasow_residual, pick_xmax, xmax_3level
export solve_galerkin, solve_galerkin_full, FullDomainMatching

end # module WasowGalerkin
