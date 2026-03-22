using Pkg;
Pkg.activate(joinpath(@__DIR__, "../.."))
using GeneralizedPerturbedEquilibrium, Plots, PlotlyJS
using GeneralizedPerturbedEquilibrium: Analysis
plotlyjs()

h5path = joinpath(@__DIR__, "gpec.h5")
p_modes = Analysis.ForceFreeStates.plot_mode_displacement(h5path; modes=1:5)
p_eigen = Analysis.ForceFreeStates.plot_eigenmode_summary(h5path)
p_stab = Analysis.ForceFreeStates.plot_stability_criterion(h5path)

# Summary plots
p_dcon_summary = Analysis.ForceFreeStates.plot_dcon_summary(h5path)
p_sing         = Analysis.ForceFreeStates.plot_singular_surfaces(h5path)
p_equil        = Analysis.Equilibrium.plot_equilibrium_summary(h5path)
