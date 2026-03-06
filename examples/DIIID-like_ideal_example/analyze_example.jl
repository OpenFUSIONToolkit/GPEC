using Pkg;
Pkg.activate(joinpath(@__DIR__, "../.."))
using JPEC, Plots

h5path = joinpath(@__DIR__, "jpec.h5")
display(JPEC.Analysis.ForceFreeStates.plot_mode_displacement(h5path; modes=1:5))
display(JPEC.Analysis.ForceFreeStates.plot_eigenmode_summary(h5path))
display(JPEC.Analysis.ForceFreeStates.plot_stability_criterion(h5path))
