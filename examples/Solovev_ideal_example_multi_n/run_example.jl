using Pkg;
Pkg.activate(joinpath(@__DIR__, "../.."))
using GeneralizedPerturbedEquilibrium
GeneralizedPerturbedEquilibrium.main([dirname(@__FILE__)])
GeneralizedPerturbedEquilibrium.main([joinpath(@__DIR__, "single_n_1")])
GeneralizedPerturbedEquilibrium.main([joinpath(@__DIR__, "single_n_2")])
