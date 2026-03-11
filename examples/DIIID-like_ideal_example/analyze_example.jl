using Pkg;
Pkg.activate(joinpath(@__DIR__, "../.."))
using GeneralizedPerturbedEquilibrium, Plots, PlotlyJS
using GeneralizedPerturbedEquilibrium: Analysis
plotlyjs()

h5path = joinpath(@__DIR__, "gpec.h5")
p_modes = Analysis.ForceFreeStates.plot_mode_displacement(h5path; modes=1:5)
p_eigen = Analysis.ForceFreeStates.plot_eigenmode_summary(h5path)
p_stab = Analysis.ForceFreeStates.plot_stability_criterion(h5path)
