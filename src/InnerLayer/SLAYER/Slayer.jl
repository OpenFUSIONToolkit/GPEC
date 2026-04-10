# # Slayer.jl
# #
# # SLAYER (Slab Layer) drift-MHD two-fluid inner layer model. This is a skeleton. 

# module Slayer

# using LinearAlgebra
# using StaticArrays

# import ..InnerLayerModel, ..solve_inner

# """
#     SlayerModel{S} <: InnerLayerModel

# SLAYER (Slab Layer) drift-MHD two-fluid inner layer model. The type parameter `S`
# selects the solver: (different types of solvers go here)
# """
# struct SlayerModel{S} <: InnerLayerModel end

# SlayerModel(; solver::Symbol=:galerkin) = SlayerModel{solver}()


# export SlayerModel, SlayerParameters

# end # module GGJ
