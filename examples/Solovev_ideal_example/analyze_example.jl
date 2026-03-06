using Pkg;
Pkg.activate(joinpath(@__DIR__, "../.."))
using JPEC, Plots, PlotlyJS
plotlyjs()

h5path = joinpath(@__DIR__, "jpec.h5")
p_modes = JPEC.Analysis.ForceFreeStates.plot_mode_displacement(h5path; modes=1:5)
p_eigen = JPEC.Analysis.ForceFreeStates.plot_eigenmode_summary(h5path)
p_stab = JPEC.Analysis.ForceFreeStates.plot_stability_criterion(h5path)
