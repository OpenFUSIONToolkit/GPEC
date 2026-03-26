using Pkg;
Pkg.activate(joinpath(@__DIR__, "../.."))
using GeneralizedPerturbedEquilibrium, Plots, PlotlyJS
using GeneralizedPerturbedEquilibrium: Analysis
plotlyjs()

h5path = joinpath(@__DIR__, "gpec.h5")

# Summary plots
p_eq = Analysis.Equilibrium.plot_equilibrium_summary(h5path)
p_ffs = Analysis.ForceFreeStates.plot_ffs_summary(h5path)
p_pe  = Analysis.PerturbedEquilibrium.plot_perturbed_equilibrium_summary(h5path)