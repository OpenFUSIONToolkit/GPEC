using Pkg;
Pkg.activate(joinpath(@__DIR__, "../.."))
using GeneralizedPerturbedEquilibrium, Plots, PlotlyJS
using GeneralizedPerturbedEquilibrium: Analysis
plotlyjs()

# Analyze the multi-n run
h5path = joinpath(@__DIR__, "gpec.h5")
p_modes = Analysis.ForceFreeStates.plot_mode_displacement(h5path; modes=1:5)
p_eigen = Analysis.ForceFreeStates.plot_eigenmode_summary(h5path)
p_stab = Analysis.ForceFreeStates.plot_stability_criterion(h5path)

# Analyze the single-n runs
h5path_n1 = joinpath(@__DIR__, "single_n_1", "euler_n1.h5")
h5path_n2 = joinpath(@__DIR__, "single_n_2", "euler_n2.h5")
p_modes_n1 = Analysis.ForceFreeStates.plot_mode_displacement(h5path_n1; modes=1:5)
p_modes_n2 = Analysis.ForceFreeStates.plot_mode_displacement(h5path_n2; modes=1:5)
