using Pkg;
Pkg.activate(joinpath(@__DIR__, "../.."))
using GeneralizedPerturbedEquilibrium
GeneralizedPerturbedEquilibrium.main([dirname(@__FILE__)])
